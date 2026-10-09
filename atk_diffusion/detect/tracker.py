# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Tracks: one object per signal over time (DETECTION_DESIGN §5; ARCHITECTURE
§4.1).

*"Boxes from successive tiles are associated (Hungarian assignment on
time-frequency overlap and family) into tracks: one object per signal over
time, with its first-seen, last-seen, duty cycle, PRI for bursts, and its
frequency drift. A hop pattern is a track whose boxes move. The PTT
scanner's existing notion of a key-up becomes a track of family burst with
the voice decoder as confirmer."*

ASSOCIATION. Detections are taken in time order, in BATCHES of boxes that
START together (within `batch_window_s`, 5 ms — the boxes a tile hands over
for signals that carry on from the last tile all start at its ownership
edge): those compete for tracks in one Hungarian assignment
(`scipy.optimize.linear_sum_assignment`, one detection per track per batch),
while successive boxes — a TDMA slot every 30 ms, sixteen of them in one
tile, or a hop every 20 ms beside a carrier — are assigned one after
another, so a track can take many bursts from one update. The cost of
putting detection d on track T:

    (1 − frequency IoU of d with the best of T's last 4 boxes)  overlap term
  + w_time · max(0, gap) / max_gap_s                       adjacency term
  + w_family when both classes are known and differ        soft class term

and it is infeasible when the gap exceeds `max_gap_s`, the frequency IoU is
below `iou_min`, or (strict) both FAMILIES are known and differ. A box that
duplicates one assigned in the same batch (time-frequency IoU >= 0.5, the
same signal seen twice) joins that box's track instead of starting one, and
so does a FRAGMENT — a box that overlaps a track in frequency but lost the
one-per-batch assignment to another piece of the same signal: frequency
overlap says it is the same channel, and a weak signal broken into pieces
must still be one track.

HOPS. A hopper's boxes do not overlap in frequency, so overlap alone would
start a new track per hop. A detection may also join a track as a HOP when it
starts within `hop_max_gap_s` of the track's last box ending, both boxes are
short dwells (<= `hop_max_dwell_s`), their bandwidths agree within
`hop_bw_ratio`, and families are compatible. A hop link always costs more
than any overlap link, so a steady signal is never stolen by a hopper. A
track whose boxes move by more than their own width `hop_min_jumps` times is
flagged `hop`.

STATISTICS (all classical, all checkable):
  first_seen / last_seen   stream seconds
  center_hz, bw_hz         medians over the recent boxes
  family, cls              majority over the boxes (known values only)
  state                    "confirmed" if any box was confirmed (§5)
  n_bursts, on_time_s      boxes merged into on-intervals (gaps shorter than
                           `burst_gap_s` do not separate bursts)
  duty_cycle               on-time / (last_seen − first_seen)
  pri_s                    median interval between burst starts (>= 3 bursts)
  drift_hz_per_s           least-squares slope of box centre against time
                           (>= 3 boxes over >= `drift_min_span_s`; not for
                           hoppers, whose centre jumping is not drift)

LIMITS. A track sees what its boxes resolve: at the RTL geometry a row is
2.1 ms, so a 2.5 ms TDMA guard is barely a row and two busy DMR slots look
continuous. A track is closed after `max_gap_s` of silence; a station that
keys up an hour later is a new track (the watchlist joins them, not this).
"""

from __future__ import annotations

import math
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from atk_diffusion.detect.boxes import Detection, overlap_tf
from atk_diffusion.detect.classes import UNKNOWN

_BIG = 1e6
#: Recent boxes a track keeps for its medians and drift fit.
HISTORY = 256
#: On-intervals a track keeps for its burst count and PRI (older ones are
#: summarised in the counters, not forgotten).
INTERVALS = 512


def freq_iou(lo1: float, hi1: float, lo2: float, hi2: float) -> float:
    inter = max(0.0, min(hi1, hi2) - max(lo1, lo2))
    union = max(hi1, hi2) - min(lo1, lo2)
    return inter / union if union > 0 else 0.0


def _known(family: str) -> bool:
    return bool(family) and family != "unknown"


@dataclass
class Track:
    """One signal over time. Built and updated only by `Tracker`."""
    id: str
    first_seen: float
    last_seen: float
    f_lo: float                 # the latest box's extent (what association uses)
    f_hi: float
    center_hz: float = 0.0
    bw_hz: float = 0.0
    family: str = "unknown"
    cls: str = ""
    state: str = "proposed"
    confirmed_by: str = ""
    decoded: str = ""
    n_bursts: int = 0
    on_time_s: float = 0.0
    duty_cycle: float = 0.0
    pri_s: float | None = None
    drift_hz_per_s: float | None = None
    hop: bool = False
    n_hops: int = 0
    n_boxes: int = 0
    sources: tuple = ()
    snr_db: float | None = None
    profile: str = ""
    epoch: float | None = None
    closed: bool = False
    families: dict = field(default_factory=dict)
    classes: dict = field(default_factory=dict)
    box_ids: list = field(default_factory=list)
    # working state
    _hist: deque = field(default_factory=lambda: deque(maxlen=HISTORY),
                         repr=False, compare=False)
    _iv: list = field(default_factory=list, repr=False, compare=False)
    _iv_dropped: int = field(default=0, repr=False, compare=False)
    _on_dropped: float = field(default=0.0, repr=False, compare=False)

    @property
    def span_s(self) -> float:
        return float(self.last_seen - self.first_seen)

    @property
    def last_box(self) -> tuple | None:
        return self._hist[-1] if self._hist else None

    def to_json(self) -> dict:
        keys = [k for k in self.__dataclass_fields__ if not k.startswith("_")]
        d = {k: getattr(self, k) for k in keys}
        d["sources"] = list(self.sources)
        d["box_ids"] = list(self.box_ids)
        d["families"] = dict(self.families)
        d["classes"] = dict(self.classes)
        return d

    def words(self) -> str:
        """One line for the AI Detect tab."""
        what = self.cls or self.family or "signal"
        bits = [f"{what} at {self.center_hz / 1e6:.4f} MHz, "
                f"{self.bw_hz / 1e3:.1f} kHz wide"]
        if self.hop:
            bits.append(f"HOPPING ({self.n_hops} hops)")
        bits.append(f"{self.n_bursts} burst{'s' if self.n_bursts != 1 else ''}, "
                    f"duty {100 * self.duty_cycle:.0f}%")
        if self.pri_s:
            bits.append(f"PRI {self.pri_s * 1e3:.1f} ms")
        if self.drift_hz_per_s is not None and abs(self.drift_hz_per_s) >= 0.05:
            bits.append(f"drifting {self.drift_hz_per_s:+.1f} Hz/s")
        bits.append(self.state.upper() + (f" by {self.confirmed_by}"
                                          if self.confirmed_by else ""))
        return "; ".join(bits)


class Tracker:
    """Associates detections into tracks (module docstring).

        tr = Tracker(max_gap_s=10.0)
        touched = tr.update(detections)      # sets det.track_id
        tr.active()                          # live tracks
    """

    def __init__(self, max_gap_s: float = 10.0, iou_min: float = 0.3,
                 family_strict: bool = True, w_time: float = 0.5,
                 w_family: float = 0.5, hop_max_gap_s: float = 0.05,
                 hop_max_dwell_s: float = 0.5, hop_bw_ratio: float = 1.6,
                 hop_min_jumps: int = 3, burst_gap_s: float = 0.0015,
                 drift_min_span_s: float = 0.1, batch_window_s: float = 0.005,
                 recent_boxes: int = 4, max_closed: int = 2000):
        if max_gap_s <= 0:
            raise ValueError("max_gap_s must be positive")
        if not 0.0 < iou_min <= 1.0:
            raise ValueError("iou_min must be in (0, 1]")
        self.max_gap_s = float(max_gap_s)
        self.iou_min = float(iou_min)
        self.family_strict = bool(family_strict)
        self.w_time = float(w_time)
        self.w_family = float(w_family)
        self.hop_max_gap_s = float(hop_max_gap_s)
        self.hop_max_dwell_s = float(hop_max_dwell_s)
        self.hop_bw_ratio = float(hop_bw_ratio)
        self.hop_min_jumps = int(hop_min_jumps)
        self.burst_gap_s = float(burst_gap_s)
        self.drift_min_span_s = float(drift_min_span_s)
        self.batch_window_s = float(batch_window_s)
        self.recent_boxes = max(1, int(recent_boxes))
        self._active: dict[str, Track] = {}
        self.closed: deque = deque(maxlen=int(max_closed))
        self.now = -math.inf

    # -- reading ---------------------------------------------------------------
    def active(self) -> list[Track]:
        return sorted(self._active.values(), key=lambda t: (t.first_seen, t.center_hz))

    def get(self, track_id: str) -> Track | None:
        t = self._active.get(track_id)
        if t is not None:
            return t
        for c in self.closed:
            if c.id == track_id:
                return c
        return None

    # -- lifecycle ---------------------------------------------------------------
    def expire(self, now: float) -> list[Track]:
        """Close tracks silent for longer than max_gap_s before `now`."""
        out = []
        for tid in [t.id for t in self._active.values()
                    if t.last_seen < float(now) - self.max_gap_s]:
            t = self._active.pop(tid)
            t.closed = True
            self.closed.append(t)
            out.append(t)
        return out

    def close_all(self) -> list[Track]:
        """Close every track (a retune: no track spans one)."""
        out = list(self._active.values())
        for t in out:
            t.closed = True
            self.closed.append(t)
        self._active.clear()
        return out

    def clear(self) -> None:
        """Forget every track, open and closed (a new capture)."""
        self._active.clear()
        self.closed.clear()
        self.now = -math.inf

    def note_confirmed(self, det: Detection) -> Track | None:
        """A detection on a track was confirmed after it was tracked."""
        t = self.get(det.track_id) if det.track_id else None
        if t is not None and det.state == "confirmed":
            t.state = "confirmed"
            t.confirmed_by = det.confirmed_by
            t.decoded = det.decoded
            if det.cls:
                t.classes[det.cls] = t.classes.get(det.cls, 0) + 1
                t.cls = det.cls          # the decoder wins the label (§5)
        return t

    # -- association ---------------------------------------------------------------
    def update(self, dets) -> list[Track]:
        """Associate `dets`; set each one's `track_id`. Returns the tracks
        touched, in the order first touched."""
        order = sorted(list(dets), key=lambda d: (d.t0, d.f_lo))
        touched: dict[str, Track] = {}
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and order[j].t0 < order[i].t0 + self.batch_window_s:
                j += 1
            for t in self._assign_batch(order[i:j]):
                touched.setdefault(t.id, t)
            i = j
        return list(touched.values())

    def _f_overlap(self, d: Detection, t: Track) -> float:
        """Frequency IoU of `d` with the best of the track's recent boxes (a
        weak signal's pieces jitter in frequency; a drifting one moves)."""
        best = freq_iou(d.f_lo, d.f_hi, t.f_lo, t.f_hi)
        for h in list(t._hist)[-self.recent_boxes:]:
            best = max(best, freq_iou(d.f_lo, d.f_hi, h[2], h[3]))
        return best

    def _assign_batch(self, batch: list[Detection]) -> list[Track]:
        now = min(d.t0 for d in batch)
        self.now = max(self.now, now)
        self.expire(now)
        tracks = list(self._active.values())
        assigned: dict[int, Track] = {}
        if tracks:
            cost = np.full((len(batch), len(tracks)), _BIG)
            for a, d in enumerate(batch):
                for b, t in enumerate(tracks):
                    cost[a, b] = self._cost(d, t)
            from scipy.optimize import linear_sum_assignment
            rows, cols = linear_sum_assignment(cost)
            for a, b in zip(rows, cols):
                if cost[a, b] < _BIG:
                    assigned[int(a)] = tracks[int(b)]
        out = []
        for a, d in enumerate(batch):
            t = assigned.get(a)
            if t is None:
                # the same signal twice in one batch joins its twin's track
                for a2, t2 in assigned.items():
                    if overlap_tf(d, batch[a2]) >= 0.5:
                        t = t2
                        break
            live = list(self._active.values())
            if t is None and live:
                # a fragment of a signal whose track already took a box in
                # this batch: frequency overlap says it is the same channel
                best = min(((self._cost(d, tr), tr) for tr in live),
                           key=lambda ct: ct[0])
                if best[0] < 1.5:            # an overlap link, never a hop
                    t = best[1]
            if t is None:
                t = self._new_track(d)
                assigned[a] = t
            self._add(t, d)
            out.append(t)
        return out

    def _cost(self, d: Detection, t: Track) -> float:
        last = t.last_box
        gap = d.t0 - t.last_seen
        if gap > self.max_gap_s:
            return _BIG
        if self.family_strict and _known(d.family) and _known(t.family) \
                and d.family != t.family:
            return _BIG
        soft = self.w_family if (d.cls and t.cls and d.cls != t.cls
                                 and UNKNOWN not in (d.cls, t.cls)) else 0.0
        f_ov = self._f_overlap(d, t)
        if f_ov >= self.iou_min:
            return (1.0 - f_ov) + self.w_time * max(0.0, gap) / self.max_gap_s + soft
        # a hop: sequential, short dwells, similar bandwidth
        if last is None:
            return _BIG
        l_t0, l_t1, l_lo, l_hi = last[0], last[1], last[2], last[3]
        tol = 0.25 * min(d.duration_s, l_t1 - l_t0)
        if not (-tol <= d.t0 - l_t1 <= self.hop_max_gap_s):
            return _BIG
        if d.duration_s > self.hop_max_dwell_s or (l_t1 - l_t0) > self.hop_max_dwell_s:
            return _BIG
        bw1, bw2 = max(d.bw_hz, 1e-9), max(l_hi - l_lo, 1e-9)
        if max(bw1, bw2) / min(bw1, bw2) > self.hop_bw_ratio:
            return _BIG
        return 1.5 + 0.5 * max(0.0, d.t0 - l_t1) / max(self.hop_max_gap_s, 1e-9) + soft

    def _new_track(self, d: Detection) -> Track:
        t = Track(id="trk" + uuid.uuid4().hex[:10], first_seen=d.t0,
                  last_seen=d.t1, f_lo=d.f_lo, f_hi=d.f_hi,
                  center_hz=d.center_hz, bw_hz=d.bw_hz, profile=d.profile,
                  epoch=d.epoch)
        self._active[t.id] = t
        return t

    def _add(self, t: Track, d: Detection) -> None:
        d.track_id = t.id
        last = t.last_box
        if last is not None:
            moved = abs(d.center_hz - last[4])
            if freq_iou(d.f_lo, d.f_hi, last[2], last[3]) == 0.0 and \
                    moved > max(d.bw_hz, last[3] - last[2]):
                t.n_hops += 1
        t._hist.append((d.t0, d.t1, d.f_lo, d.f_hi, d.center_hz, d.bw_hz))
        t.n_boxes += 1
        t.box_ids.append(d.id)
        if len(t.box_ids) > HISTORY:
            del t.box_ids[:len(t.box_ids) - HISTORY]
        t.first_seen = min(t.first_seen, d.t0)
        t.last_seen = max(t.last_seen, d.t1)
        t.f_lo, t.f_hi = d.f_lo, d.f_hi
        cs = np.array([h[4] for h in t._hist])
        bws = np.array([h[5] for h in t._hist])
        t.center_hz = float(np.median(cs))
        t.bw_hz = float(np.median(bws))
        t.families[d.family] = t.families.get(d.family, 0) + 1
        known = {k: v for k, v in t.families.items() if _known(k)}
        t.family = Counter(known).most_common(1)[0][0] if known else "unknown"
        if d.cls:
            t.classes[d.cls] = t.classes.get(d.cls, 0) + 1
            if t.state != "confirmed":
                t.cls = Counter(t.classes).most_common(1)[0][0]
        if d.state == "confirmed":
            t.state = "confirmed"
            t.confirmed_by = d.confirmed_by
            t.decoded = d.decoded
            if d.cls:
                t.cls = d.cls
        t.sources = tuple(p for p in ("energy", "cyclic", "learned")
                          if p in set(t.sources) | set(d.sources))
        if d.snr_db is not None and (t.snr_db is None or d.snr_db > t.snr_db):
            t.snr_db = float(d.snr_db)
        if t.profile == "" and d.profile:
            t.profile = d.profile
        self._add_interval(t, d.t0, d.t1)
        t.hop = t.n_hops >= self.hop_min_jumps
        t.drift_hz_per_s = None if t.hop else self._drift(t)

    def _add_interval(self, t: Track, a: float, b: float) -> None:
        iv = t._iv
        g = self.burst_gap_s
        if iv and a >= iv[-1][0]:
            if a <= iv[-1][1] + g:
                iv[-1][1] = max(iv[-1][1], b)
            else:
                iv.append([a, b])
        else:
            iv.append([a, b])
            iv.sort(key=lambda x: x[0])
            merged = [list(iv[0])]
            for s, e in iv[1:]:
                if s <= merged[-1][1] + g:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            iv[:] = merged
        if len(iv) > INTERVALS:
            drop = len(iv) - INTERVALS
            t._iv_dropped += drop
            t._on_dropped += sum(e - s for s, e in iv[:drop])
            del iv[:drop]
        t.n_bursts = t._iv_dropped + len(iv)
        t.on_time_s = t._on_dropped + sum(e - s for s, e in iv)
        span = t.last_seen - t.first_seen
        t.duty_cycle = float(min(1.0, t.on_time_s / span)) if span > 0 else 1.0
        if len(iv) >= 3:
            starts = np.array([s for s, _e in iv])
            t.pri_s = float(np.median(np.diff(starts)))
        else:
            t.pri_s = None

    def _drift(self, t: Track) -> float | None:
        if len(t._hist) < 3:
            return None
        h = np.array([((a + b) / 2.0, c) for a, b, _lo, _hi, c, _bw in t._hist])
        if h[:, 0].max() - h[:, 0].min() < self.drift_min_span_s:
            return None
        tt = h[:, 0] - h[:, 0].mean()
        denom = float(np.sum(tt * tt))
        if denom <= 0:
            return None
        return float(np.sum(tt * (h[:, 1] - h[:, 1].mean())) / denom)
