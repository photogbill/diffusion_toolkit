# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Low-SNR escalation — the IQ buffer and when to search it
(DETECTION_DESIGN §4.3; ARCHITECTURE §4.2 `cyclo.escalate`).

Bill, 2026-10-08: *"if a different detector is running, and the SNR is
below a certain threshold, it uses a buffer to run cyclostationary
detection."*

    IqBuffer(seconds, fs)          the last N seconds of raw IQ at the
                                   profile's rate (a dial, 2–10 s) — the
                                   PTT scanner's pre-roll ring, generalised
    EscalationPolicy               when to look: a proposer unsure (its
                                   confidence in an ambiguous band); a
                                   region's SNR under the profile's line
                                   (`escalate_snr_db`); the hunter dwelling
                                   in a quiet band; the analyst asking —
                                   each answered (yes/no, reason in words)
    escalate(buffer, region, …)    the cyclic proposer over the buffered
                                   seconds; every detection flagged
                                   "escalated", integration_s = the seconds
                                   used, so its box carries "α 8.0 s"
    Escalator(profile, seconds)    the detector pipeline's escalation hook:
                                   fed every block, it holds the ring, asks
                                   the policy, escalates, and logs why

WHY A BUFFER. The cyclic detector is the one proposer whose depth grows with
observation time: its statistic's mean grows in proportion to the time
integrated (the SCF estimate's variance falls as 1/(T·Δf)), while a tile is
one second. Seconds of buffer find what no tile can — the tests show a
cell 12 dB under the floor that one second misses and the buffer finds.

THE TRIGGER A WEAK SIGNAL CAN PULL. A signal under the CFAR line makes no
energy box, so no box-driven trigger can ever fire for it; that is what the
HUNTER DWELLING trigger is for (`Escalator.dwell(region)`): while the hunter
(plan B6) or the analyst parks on a quiet band, the pipeline hands the
Escalator every tile, and the band is searched over the buffer as often as
the cooldown allows — the one way a signal below the floor becomes a box.

THE CLOCK IS THE STREAM'S. The cooldown (one look per region per
`cooldown_s`) is counted in stream seconds, not wall seconds: its purpose is
not to search the same buffered seconds twice, and a recorded capture
processed faster than real time would otherwise get one look in its whole
length (the first version used the wall clock).

MEMORY, stated: complex64 is 8 bytes a sample — 19 MB a second at
2.4 MS/s (10 s = 192 MB), 160 MB a second at 20 MS/s. The buffer refuses,
in words, to allocate beyond `max_bytes` (2 GiB by default) rather than
push the machine into swap under the cognitive core.

CONTIGUITY. A ring holds ONE stretch of time at ONE tuning. A block that
does not follow the last one (a dropped USB transfer, a gap, a retune) is
not glued on — integrating across a gap would mix two different stretches
of signal — so the buffer restarts and says why (`notes`).
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from atk_diffusion import profiles as _profiles

#: An escalated region narrower than this is widened, about its centre, to
#: it: a weak signal's energy box shows only its strongest part, and the
#: cyclic look needs the whole channel (12.5 kHz: the narrowband voice
#: channel, which holds P25, DMR, NXDN and the pagers).
MIN_REGION_HZ = 12_500.0

#: The part of the span a region may reach (receivers roll off at the edges;
#: the cyclic proposer's own limit).
USABLE = 0.45


class IqBuffer:
    """A bounded ring of complex64 samples with stream times.

    Times are STREAM times in seconds (the same clock the detections use);
    `push(x, t_end)` takes a block whose last sample ENDS at t_end, i.e.
    the block covers [t_end − n/fs, t_end). `epoch` (optional) is the wall
    clock of stream time 0, passed through to the detections."""

    def __init__(self, seconds: float, fs: float, channels: int = 1,
                 max_bytes: int = 2 << 30, epoch: float | None = None):
        try:
            fs = float(fs)
            seconds = float(seconds)
            channels = int(channels)
        except (TypeError, ValueError):
            raise ValueError("a buffer is a number of seconds at a sample rate "
                             "for a number of channels") from None
        if not (fs > 0 and seconds > 0 and math.isfinite(fs)
                and math.isfinite(seconds)):
            raise ValueError("a buffer needs a positive length and sample rate")
        if channels < 1:
            raise ValueError("a buffer holds at least one channel")
        n = int(round(seconds * fs))
        if n < 1:
            raise ValueError(f"{seconds:g} s at {fs:,.0f} S/s is less than one "
                             "sample")
        need = n * 8 * channels
        if need > int(max_bytes):
            raise ValueError(
                f"{seconds:g} s at {fs:,.0f} S/s × {channels} channel(s) is "
                f"{need / 1e9:,.2f} GB of complex64 — over the "
                f"{max_bytes / 1e9:,.2f} GB limit. Shorten the buffer (the "
                "dial is 2–10 s) or raise the limit deliberately.")
        self.fs = fs
        self.capacity = n
        self.channels = channels
        self._buf = np.zeros((self.channels, n), dtype=np.complex64)
        self._write = 0
        self._held = 0
        self._t_end: float | None = None
        self.center_hz: float | None = None
        self.epoch = epoch
        self.notes: list[str] = []

    # -- state ---------------------------------------------------------------
    @property
    def seconds(self) -> float:
        """Seconds of contiguous IQ held now (a property, not a method)."""
        return self._held / self.fs

    @property
    def capacity_seconds(self) -> float:
        return self.capacity / self.fs

    @property
    def nbytes(self) -> int:
        return int(self._buf.nbytes)

    def span(self) -> tuple:
        """(t0, t1) of the held stretch in stream time, or (None, None)."""
        if self._t_end is None or self._held == 0:
            return None, None
        return self._t_end - self._held / self.fs, self._t_end

    def clear(self, why: str = "") -> None:
        self._write = 0
        self._held = 0
        self._t_end = None
        if why:
            self.notes.append(why)

    # -- writing -------------------------------------------------------------
    def push(self, x, t_end: float, center_hz: float | None = None) -> None:
        """Append a block ending at stream time `t_end` (seconds). A block
        that does not continue the held stretch, or arrives at another
        tuning, restarts the buffer (and `notes` says why)."""
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[0] != self.channels:
            raise ValueError(f"this buffer holds {self.channels} channel(s); "
                             f"the block has shape {x.shape}")
        n = x.shape[1]
        if n == 0:
            return
        t_end = float(t_end)
        start = t_end - n / self.fs
        if center_hz is not None:
            if self.center_hz is not None and abs(float(center_hz)
                                                  - self.center_hz) > 0.5:
                self.clear(f"retuned from {self.center_hz:,.0f} Hz to "
                           f"{float(center_hz):,.0f} Hz — the buffer "
                           "restarted (a ring holds one tuning)")
            self.center_hz = float(center_hz)
        if self._t_end is not None:
            jump = start - self._t_end
            if abs(jump) > 0.5 / self.fs + 1e-6:
                what = "a gap" if jump > 0 else "an overlap"
                self.clear(f"{what} of {abs(jump) * 1e3:,.3f} ms in the stream "
                           f"at t = {start:.6f} s — the buffer restarted "
                           "(integrating across it would mix two stretches "
                           "of time)")
        if n >= self.capacity:
            self._buf[:] = x[:, n - self.capacity:]
            self._write = 0
            self._held = self.capacity
        else:
            end = self._write + n
            if end <= self.capacity:
                self._buf[:, self._write:end] = x
            else:
                k = self.capacity - self._write
                self._buf[:, self._write:] = x[:, :k]
                self._buf[:, : n - k] = x[:, k:]
            self._write = end % self.capacity
            self._held = min(self.capacity, self._held + n)
        self._t_end = t_end

    # -- reading -------------------------------------------------------------
    def window(self, t0: float | None = None, t1: float | None = None
               ) -> tuple:
        """(samples, t_start): the held samples in [t0, t1), oldest first,
        clipped to what is held; t_start is the stream time of the first
        returned sample. 1-D for a one-channel buffer, else [channels, n]."""
        a, b = self.span()
        if a is None:
            empty = np.zeros((self.channels, 0), np.complex64)
            return (empty[0] if self.channels == 1 else empty), None
        lo = a if t0 is None else max(a, float(t0))
        hi = b if t1 is None else min(b, float(t1))
        if hi <= lo:
            empty = np.zeros((self.channels, 0), np.complex64)
            return (empty[0] if self.channels == 1 else empty), lo
        i0 = int(round((lo - a) * self.fs))
        i1 = min(self._held, int(round((hi - a) * self.fs)))
        oldest = (self._write - self._held) % self.capacity
        idx = (oldest + np.arange(i0, i1)) % self.capacity
        out = self._buf[:, idx]
        return (out[0] if self.channels == 1 else out), a + i0 / self.fs

    def get(self, t0: float | None = None, t1: float | None = None
            ) -> np.ndarray:
        """The held samples in [t0, t1) (see `window` for their start time)."""
        return self.window(t0, t1)[0]


def _region_of(region):
    """(lo, hi) from a Detection, a pair, or None."""
    if region is None:
        return None
    if hasattr(region, "f_lo"):
        return float(region.f_lo), float(region.f_hi)
    try:
        lo, hi = region
        lo, hi = sorted((float(lo), float(hi)))
    except (TypeError, ValueError):
        raise ValueError("a region is an (f_lo_hz, f_hi_hz) pair, a "
                         "Detection, or None for the whole span") from None
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        raise ValueError("a region needs two different, finite frequencies")
    return lo, hi


@dataclass
class EscalationPolicy:
    """When the cyclic detector should look over the buffer, in words.

    Triggers (DETECTION_DESIGN §4.3): a proposer's confidence inside the
    `ambiguous` band (the detector pipeline's own band, 0.35–0.65); a
    region's SNR under `escalate_snr_db` (the profile's line); the hunter
    dwelling in a quiet band; the analyst asking. Guards: the buffer must
    hold `min_seconds`; a region — or one overlapping half of it — is not
    re-escalated within `cooldown_s` (the analyst's request overrides the
    cooldown; the whole span counts as one region). `now` is whatever clock the caller keeps — the Escalator
    passes stream time."""

    escalate_snr_db: float = 6.0
    ambiguous: tuple = (0.35, 0.65)
    min_seconds: float = 2.0
    cooldown_s: float = 10.0
    _last: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_profile(cls, profile, **kw) -> "EscalationPolicy":
        prof = (profile if isinstance(profile, _profiles.ReceiverProfile)
                else _profiles.new_profile(str(profile)))
        return cls(escalate_snr_db=float(prof.escalate_snr_db), **kw)

    @staticmethod
    def _key(region) -> tuple:
        r = _region_of(region)
        if r is None:
            return ("span",)
        return (round(r[0], -2), round(r[1], -2))

    def _recent(self, region, now: float) -> float | None:
        """Seconds since this region — or one overlapping half of it — was
        last searched, or None. Overlap, not identity: a weak signal's
        energy box moves by a few hundred hertz from tile to tile, and an
        exact key let the same signal be searched on every tile."""
        key = self._key(region)
        if key == ("span",):
            t = self._last.get(key)
            return None if t is None else now - t
        lo, hi = key
        best = None
        for k, t in self._last.items():
            if k == ("span",):
                continue
            ov = min(hi, k[1]) - max(lo, k[0])
            if ov > 0 and ov >= 0.5 * min(hi - lo, k[1] - k[0]):
                ago = now - t
                best = ago if best is None else min(best, ago)
        return best

    def check(self, detection=None, snr_db: float | None = None,
              confidence: float | None = None, hunter_dwelling: bool = False,
              analyst: bool = False, region=None,
              buffer_seconds: float | None = None,
              now: float | None = None) -> tuple:
        """(escalate?, reason in words). `detection` supplies confidence,
        SNR, source and region when given; explicit arguments override."""
        now = time.monotonic() if now is None else float(now)
        src = "a"
        if detection is not None:
            confidence = detection.confidence if confidence is None \
                else confidence
            snr_db = detection.snr_db if snr_db is None else snr_db
            region = region if region is not None else (detection.f_lo,
                                                        detection.f_hi)
            src = "the " + " and ".join(detection.sources) if \
                detection.sources else "a"
        if buffer_seconds is not None and buffer_seconds < self.min_seconds:
            return False, (f"the buffer holds only {buffer_seconds:.1f} s; a "
                           f"cyclic look needs at least {self.min_seconds:.1f} "
                           "s to be worth its cost")
        key = self._key(region)
        reason = None
        if analyst:
            reason = "the analyst asked for a cyclic look at this region"
        else:
            ago = self._recent(region, now)
            if ago is not None:
                if ago < self.cooldown_s:
                    return False, (f"this region was searched {ago:.1f} s ago; "
                                   f"the next look in "
                                   f"{self.cooldown_s - ago:.1f} s (the "
                                   "analyst can ask sooner)")
            lo, hi = self.ambiguous
            if hunter_dwelling:
                reason = ("the hunter is dwelling in a quiet band — the buffer "
                          "is searched for signals under the floor")
            elif confidence is not None and lo <= confidence <= hi:
                reason = (f"{src} proposer is unsure (confidence "
                          f"{confidence:.2f}, inside the ambiguous band "
                          f"{lo:.2f}–{hi:.2f}) — seconds of buffer can settle"
                          " it")
            elif snr_db is not None and snr_db < self.escalate_snr_db:
                reason = (f"this region is only {snr_db:.1f} dB above the "
                          "floor, under the profile's escalation line of "
                          f"{self.escalate_snr_db:.1f} dB")
        if reason is None:
            why = []
            if confidence is not None:
                why.append(f"confidence {confidence:.2f} is outside the "
                           f"ambiguous band {self.ambiguous[0]:.2f}–"
                           f"{self.ambiguous[1]:.2f}")
            if snr_db is not None:
                why.append(f"SNR {snr_db:.1f} dB is above the "
                           f"{self.escalate_snr_db:.1f} dB line")
            return False, "no trigger: " + ("; ".join(why) if why else
                                            "nothing asked for a cyclic look")
        self._last[key] = now
        return True, reason


def _widen(region, center_hz: float, fs: float, min_width_hz: float):
    """A region narrower than `min_width_hz` widened about its centre,
    inside the usable span; (region, words or '')."""
    if region is None:
        return None, ""
    lo, hi = region
    w = hi - lo
    edge_lo = float(center_hz) - USABLE * fs
    edge_hi = float(center_hz) + USABLE * fs
    words = ""
    if min_width_hz and w < float(min_width_hz):
        c = 0.5 * (lo + hi)
        lo, hi = c - 0.5 * float(min_width_hz), c + 0.5 * float(min_width_hz)
        words = (f"the region {w:,.0f} Hz wide was widened to "
                 f"{float(min_width_hz):,.0f} Hz about its centre — a weak "
                 "signal's energy box shows only its strongest part")
    lo, hi = max(lo, edge_lo), min(hi, edge_hi)
    if hi <= lo:
        raise ValueError("the region lies outside the receiver's usable span")
    return (lo, hi), words


def escalate(buffer: IqBuffer, region, center_hz: float, profile,
             classes=None, pfa: float = 1e-3, report: dict | None = None,
             min_width_hz: float = MIN_REGION_HZ) -> list:
    """Run the cyclic proposer over everything the buffer holds, for one
    region — an (f_lo_hz, f_hi_hz) pair, a Detection (its band), or None
    for the whole span on the channel grid. Detections come back flagged
    "escalated" with integration_s = the seconds actually used.

    The buffer must be at `center_hz` (when it knows its tuning): a region
    is absolute frequency, and a ring recorded at another tuning holds
    another band. A region narrower than `min_width_hz` is widened about
    its centre (and the report says so)."""
    from atk_diffusion.cyclo.proposer import cyclic_proposer
    if buffer.center_hz is not None and abs(buffer.center_hz
                                            - float(center_hz)) > 0.5:
        raise ValueError(f"the buffer was recorded at {buffer.center_hz:,.0f} "
                         f"Hz, not {float(center_hz):,.0f} Hz — it holds "
                         "another band")
    rep = report if isinstance(report, dict) else {}
    x, t0 = buffer.window()
    if x.shape[-1] == 0:
        rep.update({"buffer_seconds": 0.0, "detections": 0,
                    "words": "the buffer is empty — nothing to search"})
        return []
    reg, widened = _widen(_region_of(region), center_hz, buffer.fs,
                          min_width_hz)
    dets = cyclic_proposer(x, buffer.fs, center_hz, profile, classes=classes,
                           regions=[reg] if reg is not None else None,
                           t0=t0, epoch=buffer.epoch, pfa=pfa, report=rep)
    used = x.shape[-1] / buffer.fs
    for d in dets:
        d.with_flag("escalated")
        d.integration_s = used
        d.measurements = dict(d.measurements)
        d.measurements["buffer_seconds"] = used
        if widened:
            d.measurements["region_note"] = widened
    rep["buffer_seconds"] = used
    rep["region"] = list(reg) if reg is not None else None
    if widened:
        rep["region_note"] = widened
    return dets


class Escalator:
    """The detector pipeline's escalation hook, over a live IQ ring
    (DETECTION_DESIGN §4.3) — the glue between `DetectorPipeline`'s callable
    contract and this module's buffer, policy and `escalate`.

        esc = Escalator(profile, seconds=5.0)
        pipe = DetectorPipeline(profile, escalation=esc)
        pipe.feed(block, center_hz, t_start)     # the pipeline feeds the ring
        esc.dwell((f_lo, f_hi))                  # the hunter parks on a band

    The pipeline pushes EVERY block into the ring (`feed`, called from
    `DetectorPipeline.feed`), so when a tile raises a trigger the cyclic
    detector integrates over the last N seconds, not the one-second tile —
    the whole point: the SCF estimate's variance falls as 1/(T·Δf). Each
    triggered detection is checked against the policy (ambiguous confidence,
    SNR under the profile's line, cooldown per region, a minimum of buffered
    seconds); each band the hunter dwells on is searched whenever the
    cooldown allows, with or without a detection there (`wants_every_tile`
    tells the pipeline to call even on an empty tile). The reason, the
    refusal and what was found are kept in `log`, in words, with the stream
    time."""

    def __init__(self, profile, seconds: float = 5.0, policy=None,
                 classes=None, pfa: float = 1e-3, channels: int = 1,
                 log_len: int = 50):
        prof = (profile if isinstance(profile, _profiles.ReceiverProfile)
                else _profiles.new_profile(str(profile)))
        self.profile = prof
        self.buffer = IqBuffer(float(seconds), float(prof.sample_rate),
                               channels=channels)
        self.policy = policy or EscalationPolicy.from_profile(prof)
        self.classes = classes
        self.pfa = float(pfa)
        self.log: deque = deque(maxlen=int(log_len))
        self.last_report: dict = {}
        self.dwelling: dict = {}           # policy key -> region (None = span)
        self.escalations = 0
        self.found = 0

    # -- the hunter / the analyst --------------------------------------------
    def dwell(self, region=None, on: bool = True) -> None:
        """The hunter (plan B6) — or the analyst — is dwelling in a quiet
        band: search `region` ((f_lo_hz, f_hi_hz), or None for the whole
        span) over the buffer as often as the cooldown allows, whether or
        not any proposer sees anything there. `on=False` stops it."""
        r = _region_of(region)
        key = EscalationPolicy._key(r)
        if on:
            self.dwelling[key] = r
            self._say(None, f"dwelling on {self._where(r)}: it will be "
                            "searched over the buffer every "
                            f"{self.policy.cooldown_s:g} s of stream")
        else:
            self.dwelling.pop(key, None)
            self._say(None, f"stopped dwelling on {self._where(r)}")

    @property
    def wants_every_tile(self) -> bool:
        """True while dwelling: the pipeline calls the hook on every tile,
        even one with no detection to escalate."""
        return bool(self.dwelling)

    # -- feeding and asking ----------------------------------------------------
    def feed(self, x, t_end: float, center_hz: float | None = None) -> None:
        """Every block, in stream order (cheap: a ring copy)."""
        before = len(self.buffer.notes)
        self.buffer.push(x, t_end, center_hz)
        for note in self.buffer.notes[before:]:
            self._say(t_end, note)

    def ask(self, region, center_hz: float, hunter_dwelling: bool = False
            ) -> list:
        """The analyst (or the hunter) asks for a cyclic look at a region now:
        the analyst's request ignores the cooldown."""
        r = _region_of(region)
        ok, why = self.policy.check(region=r, analyst=not hunter_dwelling,
                                    hunter_dwelling=hunter_dwelling,
                                    buffer_seconds=self.buffer.seconds,
                                    now=self._now(None))
        return self._look(r, center_hz, ok, why, quiet=False)

    def __call__(self, detections, iq, fs, center_hz, t0, profile) -> list:
        if abs(float(fs) - self.buffer.fs) > 0.5:
            self._say(t0, f"escalation skipped: the stream is at {fs:g} S/s, "
                          f"the ring at {self.buffer.fs:g}")
            return []
        if self.buffer.seconds == 0 and np.asarray(iq).size:
            # a host that never fed the ring: at least the tile is there
            x = np.asarray(iq)
            self.feed(x, float(t0) + x.shape[-1] / float(fs), center_hz)
        now = self._now(t0)
        out = []
        for r in list(self.dwelling.values()):
            ok, why = self.policy.check(region=r, hunter_dwelling=True,
                                        buffer_seconds=self.buffer.seconds,
                                        now=now)
            out.extend(self._look(r, center_hz, ok, why, quiet=True))
        for d in detections or []:
            ok, why = self.policy.check(detection=d,
                                        buffer_seconds=self.buffer.seconds,
                                        now=now)
            out.extend(self._look(d, center_hz, ok, why, quiet=False))
        return out

    # -- reading -------------------------------------------------------------
    def status(self) -> dict:
        a, b = self.buffer.span()
        return {"buffer_seconds": self.buffer.seconds,
                "capacity_seconds": self.buffer.capacity_seconds,
                "held": [a, b], "dwelling": [self._where(r) for r in
                                             self.dwelling.values()],
                "escalations": self.escalations, "found": self.found,
                "log": list(self.log)[-5:],
                "words": (f"escalation buffer {self.buffer.capacity_seconds:g} "
                          f"s ({self.buffer.seconds:.1f} s held); "
                          f"{self.escalations} looks, {self.found} found"
                          + (f"; dwelling on " + ", ".join(
                              self._where(r) for r in self.dwelling.values())
                             if self.dwelling else ""))}

    # -- inside --------------------------------------------------------------
    def _now(self, t0) -> float:
        _a, b = self.buffer.span()
        if b is not None:
            return float(b)
        return float(t0) if t0 is not None else 0.0

    @staticmethod
    def _where(r) -> str:
        if r is None:
            return "the whole span"
        lo, hi = (r.f_lo, r.f_hi) if hasattr(r, "f_lo") else r
        return f"{lo:,.0f}–{hi:,.0f} Hz"

    def _say(self, t, words: str) -> None:
        stamp = f"t = {float(t):.2f} s: " if t is not None else ""
        self.log.append(stamp + str(words))

    def _look(self, region, center_hz, ok: bool, why: str, quiet: bool
              ) -> list:
        t = self._now(None)
        where = self._where(region)
        if not ok:
            # a dwelling band's cooldown refusals would flood the log
            if not (quiet and "searched" in why):
                self._say(t, f"{where}: {why}")
            return []
        rep: dict = {}
        try:
            found = escalate(self.buffer, region, center_hz, self.profile,
                             classes=self.classes, pfa=self.pfa, report=rep)
        except ValueError as e:
            self._say(t, f"{where}: {why} — but the escalation was refused: "
                         f"{e}")
            return []
        self.escalations += 1
        self.found += len(found)
        self.last_report = rep
        secs = rep.get("buffer_seconds", self.buffer.seconds)
        if found:
            what = "; ".join(
                f"{d.measurements.get('class_words', d.family)} at "
                f"{d.f_lo:,.0f}–{d.f_hi:,.0f} Hz"
                for d in found)
            self._say(t, f"{where}: {why} → {len(found)} found over "
                         f"{secs:.1f} s of buffer: {what}")
        else:
            self._say(t, f"{where}: {why} → nothing found over {secs:.1f} s "
                         "of buffer (no listed cycle frequency cleared its "
                         "derived threshold)")
        return found
