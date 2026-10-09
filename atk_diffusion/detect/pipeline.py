# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The detector, end to end (DETECTION_DESIGN §1–§5, §7, §12.1; ARCHITECTURE
§4.1).

    IQ at the profile's rate
      -> FRONT END    the profile's STFT, tiles in dB above the measured floor
      -> PROPOSERS    energy (CFAR, always on) · cyclic (switchable) ·
                      learned 2D (switchable)    -> merged, disagreement kept
      -> [LOW-SNR]    optional: the tile through the denoiser; boxes only it
                      finds are added BESIDE the raw ones, flagged "denoised"
      -> CUTOUT +     the box shifted, filtered, integer-decimated to its
         CLASSIFIER   canonical rate -> Classifier1D -> PrototypeBank ->
                      class or UNKNOWN
      -> [ESCALATION] optional host hook for low-SNR / ambiguous boxes
      -> TRACKER      one object per signal over time
    -> Proposed detections (a decoder's confirmation is detect.confirm's job)

    pipe = DetectorPipeline("rtlsdr_2400000_cu8", rf=rf)
    for block, t in stream:
        for det in pipe.feed(block, center_hz=162.4e6, t_start=t): ...
    pipe.finish(); pipe.status()

RULES IT KEEPS.
* *Energy is always on* (ARCHITECTURE §4.1); cyclic and learned are each a
  switch. A model refused for this profile costs ONE feature, never the
  pipeline (plan §2.7): learned boxes go off, the refusal line goes in
  `status()["refusal"]` in the card's own words, energy and cyclic go on.
* *No box spans a retune.* A new centre frequency (or a gap in the stream
  times) ends the current stream: its last partial tile is processed, tiles
  and floor tracking start again (a retune also re-measures the floor and
  closes every track).
* *Each burst once.* Tiles overlap; every tile owns a time window
  (dsp.stft), a box complete inside a tile is kept by the tile owning its
  midpoint, a box cut by a tile edge is clipped to the ownership window and
  trimmed against what the previous tile already emitted — so a burst on a
  boundary appears once and a carrier that lasts a minute is a chain of
  abutting boxes on one track.
* *Where proposers disagree, all are shown* (§1): merged boxes keep every
  source; class disagreements are flagged and counted.
* *Denoised boxes are beside, never instead* (§3): a box the denoised path
  also found on the raw tile is dropped from the denoised side; one only it
  found is flagged "denoised", and its SNR is measured on the RAW tile.
* *Measured.* Wall and CPU time per tile, per stage, in `status()`.

THE CYCLIC PROPOSER is `atk_diffusion.cyclo.proposer.cyclic_proposer(x, fs,
center_hz, profile, classes=None, regions=None, t0=0.0, epoch=None, ...,
report=None)`, written beside this module; it is imported only when switched
on and called per tile with the tile's IQ. `cyclic_mode="regions"` (the
default) passes the energy boxes as `regions` [(f_lo, f_hi) absolute Hz]:
the cyclic detector characterises what energy found, and a tile with no
energy box costs nothing. `cyclic_mode="span"` passes None and lets it search
the class table's cycle frequencies across the whole span — the
below-the-floor search, slow on a live tile (DETECTION_DESIGN §4.1 offers it
on cuts and over the escalation buffer, §4.3). If it cannot be imported, or
raises, the status says so in words and the rest runs.

THE ESCALATION HOOK (DETECTION_DESIGN §4.3) is a host callable

    escalation(detections, iq, fs, center_hz, t0, profile) -> list[Detection]

given each tile's low-SNR (below `profile.escalate_snr_db`) or ambiguous
(confidence 0.35–0.65) detections and the tile's raw IQ; what it returns is
flagged "escalated" and merged in. Its own IQ ring buffer and trigger policy
are its business (cyclo.escalate); an exception is reported, not raised. A
hook with a `feed(block, t_end, center_hz)` method is given every block (the
ring); one whose `wants_every_tile` is true (the Escalator while the hunter
dwells on a quiet band) is called on every tile, even one with no detection
— a signal under the CFAR line makes no box to trigger on.

`run_on_capture(path)` reads a SigMF capture in chunks, refuses one of
another profile in the plan's words, and can write the detections back as
SigMF annotations (`atk:source = "proposed"`, replacing only earlier proposed
ones — taught and confirmed labels are never touched).

LIMITS. Classification needs the box's IQ, kept for one tile span (about
21 MB at 2.4 MS/s, 170 MB at 20 MS/s) only when a classifier, the cyclic
proposer or escalation is on. Boxes shorter than the classifier's window at
the canonical rate are not classified (said in their measurements).
"""

from __future__ import annotations

import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from atk_diffusion import profiles as _profiles
from atk_diffusion.detect import boxes as _boxes
from atk_diffusion.detect import classes as _classes
from atk_diffusion.detect.boxes import Detection
from atk_diffusion.detect.tracker import Tracker, freq_iou
from atk_diffusion.dsp import cfar as _cfar
from atk_diffusion.dsp import stft as _stft
from atk_diffusion.dsp.floor import NoiseFloor

#: Proposer switches; energy cannot be switched off (it is the baseline).
DEFAULT_PROPOSERS = {"energy": True, "cyclic": False, "learned": False}

#: Classifier windows taken from one box (evenly spaced, results averaged).
MAX_WINDOWS = 4

#: Confidence band that counts as ambiguous for the escalation hook (§4.3).
AMBIGUOUS = (0.35, 0.65)


class _IqHistory:
    """The raw IQ of the current stream, kept just long enough to cut boxes
    out of the tile being processed."""

    def __init__(self, fs: float):
        self.fs = float(fs)
        self._blocks: deque = deque()          # (first sample index, array)
        self.t_start = 0.0
        self._next = 0

    def reset(self, t_start: float) -> None:
        self._blocks.clear()
        self.t_start = float(t_start)
        self._next = 0

    def append(self, x: np.ndarray) -> None:
        if x.size:
            self._blocks.append((self._next, x))
            self._next += x.size

    def slice(self, t0: float, t1: float) -> tuple[np.ndarray, float]:
        """Samples covering [t0, t1) that are still held, and the stream
        time of the first one returned."""
        i0 = max(0, int(math.floor((t0 - self.t_start) * self.fs)))
        i1 = max(i0, int(math.ceil((t1 - self.t_start) * self.fs)))
        parts, first = [], None
        for s, arr in self._blocks:
            lo, hi = max(i0, s), min(i1, s + arr.size)
            if lo < hi:
                if first is None:
                    first = lo
                parts.append(arr[lo - s:hi - s])
        if not parts:
            return np.zeros(0, dtype=np.complex64), float(t0)
        x = parts[0] if len(parts) == 1 else np.concatenate(parts)
        return x, self.t_start + first / self.fs

    def prune(self, t_keep: float) -> None:
        k = int(math.floor((t_keep - self.t_start) * self.fs))
        while self._blocks and self._blocks[0][0] + self._blocks[0][1].size <= k:
            self._blocks.popleft()


class _Stats:
    def __init__(self):
        self.wall_ms: deque = deque(maxlen=500)
        self.cpu_ms: deque = deque(maxlen=500)
        self.stage: dict[str, deque] = {}
        self.tiles = 0
        self.by_source = {"energy": 0, "cyclic": 0, "learned": 0,
                          "denoised": 0, "escalated": 0}
        self.agree = 0
        self.only = {"energy": 0, "cyclic": 0, "learned": 0}
        self.class_disagreements = 0
        self.unknown = 0
        self.classified = 0
        self.detections = 0

    def add_stage(self, name: str, ms: float) -> None:
        self.stage.setdefault(name, deque(maxlen=500)).append(float(ms))


class DetectorPipeline:
    """The detector for one receiver profile (module docstring)."""

    def __init__(self, profile, rf=None, proposers: dict | None = None,
                 proposer2d=None, classifier=None, prototypes=None,
                 denoiser=None, escalation=None, progress=None,
                 tracker: Tracker | None = None, cfar: dict | None = None,
                 threads: int = 0, merge_iou: float = 0.3,
                 cyclic_mode: str = "regions"):
        self.rf = rf
        self.progress = progress
        self.notes: deque = deque(maxlen=50)
        self.refusals: list[str] = []
        if isinstance(profile, _profiles.ReceiverProfile):
            self.profile = profile
        else:
            pid = str(profile).strip().lower()
            if rf is not None:
                self.profile = _profiles.load_profile(rf, pid)
            else:
                self.profile = _profiles.new_profile(pid)
                self.profile.notes.append("defaults — no rf_data given, so the "
                                          "profile file was not read")
        self.fs = float(self.profile.sample_rate)
        self.layout = _stft.tile_layout(self.fs, self.profile.stft)
        self.proposers = dict(DEFAULT_PROPOSERS)
        self.proposers.update(dict(proposers or {}))
        if not self.proposers.get("energy", True):
            self._note("the energy proposer cannot be switched off: it is the "
                       "classical baseline every other proposer is judged against")
        self.proposers["energy"] = True
        self.cfar_kw = {"pfa": float(self.profile.cfar_pfa)}
        self.cfar_kw.update(dict(cfar or {}))
        self.threads = int(threads)
        self.merge_iou = float(merge_iou)
        if cyclic_mode not in ("regions", "span"):
            raise ValueError("cyclic_mode is 'regions' (characterise what the "
                             "energy proposer found) or 'span' (search the whole "
                             "span, below the floor too — slow)")
        self.cyclic_mode = cyclic_mode
        self._cyclic_report: dict = {}
        self.denoiser = denoiser
        self.escalation = escalation
        self.tracker = tracker or Tracker()
        self.proposer2d = None
        if proposer2d is not None:
            self.proposer2d = self._accept_model(proposer2d, "proposer2d")
        self.classifiers: dict = {}
        for clf in (classifier.values() if isinstance(classifier, dict)
                    else ([classifier] if classifier is not None else [])):
            m = self._accept_model(clf, "classifier1d")
            if m is not None:
                self.classifiers[m.canonical_class] = m
        self.banks: dict = {}
        for bank in (prototypes.values() if isinstance(prototypes, dict)
                     else ([prototypes] if prototypes is not None else [])):
            if str(bank.profile) != self.profile.id:
                self.refusals.append(f"prototype bank refused: it was made for "
                                     f"{bank.profile}, this detector is "
                                     f"{self.profile.id}")
                continue
            self.banks[bank.canonical_class] = bank
        self.initial_floor = None
        try:
            self.initial_floor = NoiseFloor.from_profile(self.profile)
        except ValueError as e:
            self._note(f"the profile's measured floor was not used: {e}; the "
                       "floor is estimated from live data instead")
        self._cyclic_fn = None
        self._cyclic_unavailable = ""
        self._cyclic_checked = False
        self._cyclic_error = ""
        self.stats = _Stats()
        self.retunes = 0
        self.discontinuities = 0
        self._builder: _stft.TileBuilder | None = None
        self._center: float | None = None
        self._expect_t: float | None = None
        self._last_floor = None
        self._recent: list[Detection] = []
        self._recent_dn: list[Detection] = []
        self._iq = _IqHistory(self.fs)
        self._last_tile_info: dict = {}

    # -- configuration helpers -----------------------------------------------------
    def _note(self, words: str) -> None:
        self.notes.append(str(words))
        if self.progress:
            try:
                self.progress(str(words))
            except Exception:                          # noqa: BLE001
                pass

    def _accept_model(self, m, kind: str):
        """Load (a model folder) or check (a loaded wrapper) a model for this
        profile; a refusal becomes the status line, never an exception."""
        from atk_diffusion import cards as _cards
        from atk_diffusion.detect import onnx_models as _om
        what = "learned proposer" if kind == "proposer2d" else "classifier"
        try:
            if isinstance(m, (str, Path)):
                m = (_om.Proposer2D if kind == "proposer2d" else _om.Classifier1D)(
                    m, profile=self.profile.id, threads=self.threads)
            else:
                card = getattr(m, "card", None)
                if card is None or card.kind != kind:
                    raise _cards.CardRefusal(f"the {what} given is not a {kind} "
                                             "model with a card")
                _profiles.check_match(card.profile, self.profile.id,
                                      what=f"this {kind} model ({card.name})")
            if kind == "proposer2d":
                st = dict((m.card.input or {}).get("stft") or {})
                lay = self.layout
                want = {"fft_size": lay.fft_size, "hop": lay.hop,
                        "window": lay.window, "tile_rows": lay.rows}
                bad = [k for k, v in want.items()
                       if k in st and str(st[k]).lower() != str(v).lower()]
                if bad:
                    raise _cards.CardRefusal(
                        f"{m.card.name} was trained on another spectrogram "
                        f"geometry ({', '.join(bad)} differ from this profile's)")
            else:
                can = m.canonical_class
                rates = {c.cls: c.rate for c in _profiles.canonical_rates(self.fs)}
                if can not in rates or abs(rates[can] - m.canonical_rate) > 0.5:
                    raise _cards.CardRefusal(
                        f"{m.card.name} expects {can or 'an unstated class of'} "
                        f"cuts at {m.canonical_rate:g} S/s; this profile's "
                        f"canonical rate for that class is "
                        f"{rates.get(can, 'none')}")
            return m
        except (_profiles.ProfileMismatch, _cards.CardRefusal, ValueError,
                RuntimeError, OSError) as e:
            self.refusals.append(f"{what} refused: {e}")
            return None

    @property
    def refusal(self) -> str:
        return "; ".join(self.refusals)

    def _cyclic(self):
        if not self._cyclic_checked:
            self._cyclic_checked = True
            try:
                from atk_diffusion.cyclo.proposer import cyclic_proposer
                self._cyclic_fn = cyclic_proposer
            except Exception as e:                     # noqa: BLE001
                self._cyclic_unavailable = (
                    "cyclic proposer unavailable: atk_diffusion.cyclo.proposer "
                    f"could not be imported ({type(e).__name__}: {e}); the "
                    "energy and learned proposers continue")
                self._note(self._cyclic_unavailable)
        return self._cyclic_fn

    @property
    def _need_iq(self) -> bool:
        return bool(self.classifiers or self.escalation is not None
                    or self.proposers.get("cyclic"))

    # -- the stream ---------------------------------------------------------------
    def _start(self, center_hz: float, t_start: float, epoch, floor) -> None:
        self._builder = _stft.TileBuilder(
            self.fs, center_hz, self.profile.stft, floor=floor, t_start=t_start,
            profile=self.profile.id, epoch=epoch,
            floor_margin_db=float(self.cfar_kw.get("floor_margin_db", 1.0)))
        self._center = float(center_hz)
        self._expect_t = float(t_start)
        self._recent, self._recent_dn = [], []
        self._iq.reset(t_start)

    def _end_stream(self) -> list[Detection]:
        out = []
        if self._builder is not None:
            for tile in self._builder.flush():
                out += self._process(tile)
            self._last_floor = self._builder.floor
        self._builder = None
        self._recent, self._recent_dn = [], []
        return out

    def feed(self, iq_block, center_hz: float, t_start: float,
             epoch: float | None = None) -> list[Detection]:
        """Process a block of IQ at the profile's rate, tuned to `center_hz`,
        whose first sample is at stream time `t_start` (seconds; `epoch` is
        the wall-clock time of stream time 0 when known). Returns the NEW
        detections the block completed, each with its track id."""
        x = np.asarray(iq_block, dtype=np.complex64).ravel()
        out: list[Detection] = []
        center_hz = float(center_hz)
        if self._builder is None or self._center is None \
                or abs(center_hz - self._center) > 0.5:
            if self._builder is not None:
                out += self._end_stream()
                self.tracker.close_all()
                self.retunes += 1
                self._note(f"retuned to {center_hz / 1e6:.6f} MHz: tiles and "
                           "the floor start again; tracks closed")
            self._start(center_hz, t_start, epoch, self._initial_floor())
        elif abs(float(t_start) - self._expect_t) > 1.5 / self.fs:
            carried = self._builder.floor
            out += self._end_stream()
            self.discontinuities += 1
            self._note(f"the stream jumped from {self._expect_t:.6f} s to "
                       f"{float(t_start):.6f} s: tiles start again (the floor "
                       "is kept)")
            self._start(center_hz, t_start, epoch,
                        carried.copy() if carried is not None else self._initial_floor())
        if x.size == 0:
            return out
        feed = getattr(self.escalation, "feed", None)
        if callable(feed):
            # the escalation ring sees EVERY block, so a trigger integrates
            # over seconds, not one tile (cyclo.escalate.Escalator)
            try:
                feed(x, float(t_start) + x.size / self.fs, center_hz)
            except Exception as e:                     # noqa: BLE001
                self._note(f"the escalation ring refused a block: {e}")
        if self._need_iq:
            self._iq.append(x)
        w0, c0 = time.perf_counter(), time.process_time()
        tiles = self._builder.push(x)
        front_wall = (time.perf_counter() - w0) * 1e3
        front_cpu = (time.process_time() - c0) * 1e3
        self._expect_t = float(t_start) + x.size / self.fs
        n = max(1, len(tiles))
        for tile in tiles:
            out += self._process(tile, front_wall / n, front_cpu / n)
        if tiles and self._need_iq:
            self._iq.prune(self._builder.next_tile_t0 - 0.25)
        return out

    def finish(self) -> list[Detection]:
        """End of the stream: process the last partial tile."""
        return self._end_stream()

    def reset(self, counters: bool = False) -> None:
        """Forget the stream (and, with `counters`, the statistics and every
        track — a new capture starts clean)."""
        self._builder = None
        self._center = None
        self._expect_t = None
        self._recent, self._recent_dn = [], []
        if counters:
            self.tracker.clear()
            self.stats = _Stats()
            self.retunes = self.discontinuities = 0
        else:
            self.tracker.close_all()

    def _initial_floor(self):
        return self.initial_floor.copy() if self.initial_floor is not None else None

    # -- one tile ---------------------------------------------------------------
    def _timed(self, name, fn, *a, **kw):
        t0 = time.perf_counter()
        r = fn(*a, **kw)
        self.stats.add_stage(name, (time.perf_counter() - t0) * 1e3)
        return r

    def _process(self, tile, front_wall: float = 0.0, front_cpu: float = 0.0
                 ) -> list[Detection]:
        w0, c0 = time.perf_counter(), time.process_time()
        info: dict = {}
        energy = self._timed("energy", _cfar.energy_proposer, tile, info=info,
                             **self.cfar_kw)
        raw = list(energy)
        if self.proposers.get("cyclic"):
            raw += self._timed("cyclic", self._run_cyclic, tile, energy)
        if self.proposers.get("learned") and self.proposer2d is not None:
            raw += self._timed("learned", self._run_learned, tile)
        merged = _boxes.merge(raw, iou=self.merge_iou)
        kept = self._own(merged, tile, self._recent)
        if self.denoiser is not None:
            extra = self._timed("denoised", self._run_denoised, tile, merged)
            kept += extra
        if self.classifiers and kept:
            self._timed("classify", self._classify, kept, tile)
        if self.escalation is not None and (kept or self._escalation_dwelling):
            kept = self._timed("escalation", self._escalate, kept, tile)
        self._timed("track", self.tracker.update, kept)
        self._recent = [d for d in kept if "denoised" not in d.flags]
        self._recent_dn = [d for d in kept if "denoised" in d.flags]
        self._count(kept)
        self._last_tile_info = info
        if info.get("capped"):
            self._note(f"tile {tile.index}: more than {self.cfar_kw.get('max_boxes', 1000)} "
                       "energy boxes — the threshold is in the noise; only the "
                       "strongest were kept")
        st = self.stats
        st.tiles += 1
        st.wall_ms.append((time.perf_counter() - w0) * 1e3 + front_wall)
        st.cpu_ms.append((time.process_time() - c0) * 1e3 + front_cpu)
        st.add_stage("front_end", front_wall)
        return kept

    def _own(self, dets, tile, recent) -> list[Detection]:
        """The ownership rule (module docstring)."""
        rp = tile.row_period
        edge = 0.5 * rp
        tol = 2.0 * rp
        out = []
        for d in dets:
            tl = (not tile.first) and d.t0 <= tile.t0 + edge
            tr = (not tile.final) and d.t1 >= tile.t1 - edge
            if not tl and not tr:
                mid = 0.5 * (d.t0 + d.t1)
                lo = tile.own_t0 - (0.0 if tile.first else tol)
                hi = math.inf if tile.final else tile.own_t1
                if not lo <= mid < hi:
                    continue
                if any(_boxes.overlap_tf(d, r) >= 0.3 for r in recent):
                    continue                    # the previous tile emitted it
                out.append(d)
                continue
            a, b = max(d.t0, tile.own_t0), min(d.t1, tile.own_t1)
            if tl:
                ends = [r.t1 for r in recent
                        if freq_iou(d.f_lo, d.f_hi, r.f_lo, r.f_hi) >= 0.3
                        and r.t1 > a]
                if ends:
                    a = max(a, max(ends))
            if b - a <= 1e-9:
                continue
            d.t0, d.t1 = a, b
            out.append(d)
        return out

    def _run_learned(self, tile) -> list[Detection]:
        try:
            return self.proposer2d.run(tile)
        except Exception as e:                         # noqa: BLE001
            self.refusals.append(f"learned proposer stopped: {e}")
            self.proposer2d = None
            return []

    def _run_cyclic(self, tile, energy) -> list[Detection]:
        fn = self._cyclic()
        if fn is None:
            return []
        lay = tile.layout
        x, t_x0 = self._iq.slice(tile.t0, tile.data_t1 + (lay.fft_size - lay.hop) / self.fs)
        if x.size == 0:
            return []
        if self.cyclic_mode == "regions":
            regions = [(d.f_lo, d.f_hi) for d in energy]
            if not regions:
                return []               # nothing found by energy to characterise
        else:
            regions = None              # the whole span (the grid of the class table)
        rep: dict = {}
        try:
            dets = fn(x, self.fs, tile.center_hz, self.profile, classes=None,
                      regions=regions, t0=t_x0, epoch=tile.epoch, report=rep)
            self._cyclic_report = rep
        except Exception as e:                         # noqa: BLE001
            msg = f"cyclic proposer failed on tile {tile.index}: {type(e).__name__}: {e}"
            if msg != self._cyclic_error:
                self._cyclic_error = msg
                self._note(msg)
            return []
        out = []
        for d in dets or []:
            if "cyclic" not in d.sources:
                d.sources = tuple(d.sources) + ("cyclic",)
            if not d.profile:
                d.profile = self.profile.id
            if d.epoch is None:
                d.epoch = tile.epoch
            out.append(d)
        return out

    def _run_denoised(self, tile, raw_merged) -> list[Detection]:
        try:
            spec = self.denoiser(tile.spec)
            t2 = tile.with_spec(spec)
        except Exception as e:                         # noqa: BLE001
            self._note(f"the low-SNR denoised path failed on tile {tile.index}: "
                       f"{type(e).__name__}: {e}")
            return []
        t2.floor_ok = False          # a floor test means nothing on a reconstruction
        kw = dict(self.cfar_kw)
        kw["floor_test"] = False
        dets = _cfar.energy_proposer(t2, **kw)
        if self.proposers.get("learned") and self.proposer2d is not None:
            dets += self._run_learned(t2)
        dets = _boxes.merge(dets, iou=self.merge_iou)
        only = []
        for d in dets:
            if any(_boxes.overlap_tf(d, r) >= 0.3 or _contains(r, d)
                   for r in raw_merged):
                continue
            d.with_flag("denoised")
            only.append(d)
        return self._own(only, tile, self._recent_dn)

    def _classify(self, dets, tile) -> None:
        from atk_diffusion.detect.prototypes import l2_normalize
        from atk_diffusion.dsp import resample as _rs
        groups: dict[str, list] = {}
        for d in dets:
            bw = max(d.bw_hz, tile.bin_hz)
            can = _profiles.canonical_for(self.fs, bw)
            clf = self.classifiers.get(can.cls)
            if clf is None:
                continue
            x, _t = self._iq.slice(d.t0, d.t1)
            if x.size == 0:
                d.measurements["classify"] = "no IQ held for this box"
                continue
            try:
                y, fs_out, _info = _rs.cut_to_canonical(x, self.fs,
                                                        d.center_hz - tile.center_hz, bw)
            except Exception as e:                     # noqa: BLE001
                d.measurements["classify"] = f"the cut failed: {e}"
                continue
            L = clf.iq_len
            trim = min(16, y.size // 10)
            usable = y[trim:y.size - trim] if y.size - 2 * trim >= L else y
            if usable.size < L:
                d.measurements["classify"] = (f"too short to classify: "
                                              f"{usable.size} samples at "
                                              f"{fs_out:g} S/s, the classifier "
                                              f"needs {L}")
                continue
            k = min(MAX_WINDOWS, usable.size // L)
            starts = np.linspace(0, usable.size - L, k).astype(int)
            groups.setdefault(can.cls, []).append(
                (d, np.stack([usable[s:s + L] for s in starts])))
        for cc, items in groups.items():
            clf = self.classifiers[cc]
            bank = self.banks.get(cc)
            try:
                res = clf.run(np.concatenate([w for _d, w in items]))
            except Exception as e:                     # noqa: BLE001
                self._note(f"the classifier {clf.name} failed: {type(e).__name__}: {e}")
                continue
            pos = 0
            for d, w in items:
                n = w.shape[0]
                probs = res.probs[pos:pos + n].mean(axis=0)
                emb = l2_normalize(res.embeddings[pos:pos + n].mean(axis=0))[0]
                cyc = np.median(res.cycle_hz[pos:pos + n], axis=0)
                pos += n
                ti = int(np.argmax(probs))
                top, p_top = res.classes[ti], float(probs[ti])
                m = d.measurements
                if d.confidence is not None and "learned" in d.sources:
                    m["proposer_score"] = d.confidence
                m.update({"classifier_top": top, "classifier_p": round(p_top, 4),
                          "symbol_rate_model_hz": round(float(cyc[0]), 3),
                          "carrier_offset_model_hz": round(float(cyc[1]), 3)})
                self.stats.classified += 1
                if bank is not None:
                    match = bank.classify(emb)
                    m["prototype_distance"] = round(float(match.distance), 4)
                    m["prototype_threshold"] = round(float(match.threshold), 4)
                    if match.note:
                        m["class_note"] = match.note
                    if match.unknown:
                        d.cls = _classes.UNKNOWN
                        d.confidence = None
                        m["nearest_class"] = match.nearest
                        self.stats.unknown += 1
                    else:
                        d.cls = match.cls
                        d.confidence = (round(float(probs[res.classes.index(match.cls)]), 4)
                                        if match.cls in res.classes else None)
                        if bank.is_taught(match.cls):
                            d.with_flag("taught")
                        if match.cls in res.classes and match.cls != top:
                            m.setdefault("other_classes", []).append(top)
                            d.with_flag("disagreement")
                else:
                    d.cls = top
                    d.confidence = round(p_top, 4)
                    m["open_set"] = ("no prototype bank: the classifier's argmax "
                                     "is forced (no UNKNOWN)")
                c = _classes.get(d.cls)
                if c is not None and d.family == "unknown":
                    d.family = c.family

    @property
    def _escalation_dwelling(self) -> bool:
        """The hook asks to see every tile (cyclo.escalate.Escalator while
        the hunter dwells on a quiet band): a signal under the CFAR line
        makes no box, so only this lets the buffer find it."""
        try:
            return bool(getattr(self.escalation, "wants_every_tile", False))
        except Exception:                              # noqa: BLE001
            return False

    def _escalate(self, kept, tile) -> list[Detection]:
        lo, hi = AMBIGUOUS
        trig = [d for d in kept
                if (d.snr_db is not None and d.snr_db < self.profile.escalate_snr_db)
                or (d.confidence is not None and lo <= d.confidence <= hi)]
        if not trig and not self._escalation_dwelling:
            return kept
        x, t_x0 = self._iq.slice(tile.t0, tile.data_t1)
        try:
            extra = self.escalation(trig, x, self.fs, tile.center_hz, t_x0,
                                    self.profile)
        except Exception as e:                         # noqa: BLE001
            self._note(f"escalation failed on tile {tile.index}: "
                       f"{type(e).__name__}: {e}")
            return kept
        extra = list(extra or [])
        if not extra:
            return kept
        for d in extra:
            d.with_flag("escalated")
            if not d.profile:
                d.profile = self.profile.id
            if d.epoch is None:
                d.epoch = tile.epoch      # the ring does not know the wall clock
        return _boxes.merge(kept + extra, iou=self.merge_iou)

    def _count(self, kept) -> None:
        st = self.stats
        for d in kept:
            st.detections += 1
            srcs = set(d.sources)
            for s in srcs:
                if s in st.by_source:
                    st.by_source[s] += 1
            if "denoised" in d.flags:
                st.by_source["denoised"] += 1
            if "escalated" in d.flags:
                st.by_source["escalated"] += 1
            if len(srcs) > 1:
                st.agree += 1
            elif len(srcs) == 1:
                s = next(iter(srcs))
                if s in st.only:
                    st.only[s] += 1
            if "disagreement" in d.flags:
                st.class_disagreements += 1

    # -- reading ---------------------------------------------------------------
    def tracks(self):
        return self.tracker.active()

    @property
    def floor(self):
        if self._builder is not None and self._builder.floor is not None:
            return self._builder.floor
        return self._last_floor

    def status(self) -> dict:
        """Everything the AI Detect tab shows, as numbers and as lines."""
        st = self.stats

        def summ(dq):
            if not dq:
                return {"tiles": 0}
            a = np.array(dq)
            return {"tiles": int(st.tiles), "mean": float(a.mean()),
                    "peak": float(a.max()), "last": float(a[-1])}

        lat = summ(st.wall_ms)
        cpu = summ(st.cpu_ms)
        stage = {k: float(np.mean(v)) for k, v in st.stage.items() if v}
        cyclic_on = bool(self.proposers.get("cyclic"))
        if cyclic_on and not self._cyclic_checked:
            self._cyclic()
        cyc_line = self._cyclic_unavailable if cyclic_on else ""
        learned_on = bool(self.proposers.get("learned")) and self.proposer2d is not None
        fl = self.floor
        lay = self.layout
        lines = [f"Detector for {_profiles.describe(self.profile.id)} "
                 f"({self.profile.id}): {lay.words()}.",
                 f"Energy proposer: ON — CA-CFAR at Pfa {self.cfar_kw['pfa']:g} "
                 "per cell (threshold derived, not tuned), plus the "
                 "floor-referenced test for wide signals."]
        if not cyclic_on:
            lines.append("Cyclic proposer: off.")
        elif cyc_line:
            lines.append(cyc_line[0].upper() + cyc_line[1:] + ".")
        else:
            how = ("characterising the energy boxes" if self.cyclic_mode == "regions"
                   else "searching the whole span (slow)")
            lines.append(f"Cyclic proposer: ON — {how}."
                         + (f" Last problem: {self._cyclic_error}." if self._cyclic_error else ""))
        if learned_on:
            lines.append(f"Learned proposer: ON — {self.proposer2d.name}.")
        elif self.proposers.get("learned"):
            lines.append("Learned proposer: switched on but no model is loaded"
                         + (" (refused — see below)." if self.refusals else "."))
        else:
            lines.append("Learned proposer: off.")
        if self.classifiers:
            for cc, clf in self.classifiers.items():
                bank = self.banks.get(cc)
                lines.append(f"Classifier for {cc} cuts: {clf.name}"
                             + (f", open set with {len(bank.classes)} prototype "
                                "classes" if bank is not None else
                                " — NO prototype bank, so it cannot say UNKNOWN"))
        if self.denoiser is not None:
            lines.append("Low-SNR denoised path: ON — boxes only it finds are "
                         "flagged 'denoised' and shown beside the raw ones.")
        esc_status: dict = {}
        if self.escalation is not None:
            try:
                esc_status = dict(self.escalation.status()) \
                    if callable(getattr(self.escalation, "status", None)) else {}
            except Exception as e:                     # noqa: BLE001
                esc_status = {"words": f"its status failed: {e}"}
            lines.append("Low-SNR escalation: ON"
                         + (f" — {esc_status['words']}." if esc_status.get("words")
                            else "."))
        if self.refusals:
            lines.append("REFUSED: " + self.refusal)
        lines.append(fl.describe() if fl is not None else
                     "Noise floor: not yet measured (the first tile measures it).")
        if lat.get("tiles"):
            lines.append(f"Latency per tile: mean {lat['mean']:.0f} ms, peak "
                         f"{lat['peak']:.0f} ms over {lat['tiles']} tiles "
                         f"({lay.step_rows * lay.row_period:.3g} s of new data "
                         f"each; CPU {cpu['mean']:.0f} ms a tile).")
        bs = st.by_source
        lines.append(f"Detections: {st.detections} ({bs['energy']} energy, "
                     f"{bs['cyclic']} cyclic, {bs['learned']} learned, "
                     f"{bs['denoised']} denoised-only, {bs['escalated']} "
                     f"escalated); {st.agree} found by more than one proposer; "
                     f"{st.class_disagreements} class disagreements; "
                     f"{len(self.tracker.active())} active tracks.")
        return {
            "profile": self.profile.id,
            "proposers": {"energy": True,
                          "cyclic": cyclic_on and not cyc_line,
                          "learned": learned_on},
            "latency_ms": lat,
            "cpu_ms": cpu,
            "stage_ms": stage,
            "counts_by_source": dict(bs),
            "agreements": int(st.agree),
            "disagreements": {"class": int(st.class_disagreements),
                              "only_energy": int(st.only["energy"]),
                              "only_cyclic": int(st.only["cyclic"]),
                              "only_learned": int(st.only["learned"])},
            "classified": int(st.classified),
            "unknown": int(st.unknown),
            "refusal": self.refusal,
            "cyclic": cyc_line,
            "cyclic_report": {k: self._cyclic_report[k] for k in
                              ("mode", "probe_calls", "pfa_per_call", "integration_s")
                              if k in self._cyclic_report},
            "floor": fl.describe() if fl is not None else "",
            "escalation": esc_status,
            "tracks_active": len(self.tracker.active()),
            "tracks_closed": len(self.tracker.closed),
            "retunes": int(self.retunes),
            "discontinuities": int(self.discontinuities),
            "tile_layout": lay.words(),
            "last_tile": dict(self._last_tile_info),
            "notes": list(self.notes),
            "lines": lines,
        }

    # -- recorded captures ---------------------------------------------------------
    def run_on_capture(self, path, write_annotations: bool = False,
                       chunk_seconds: float = 1.0, channel: int = 0,
                       max_seconds: float | None = None, progress=None) -> dict:
        """Detect over a SigMF capture of this profile, in chunks. Returns
        {detections, tracks, status, annotations_written, seconds,
        cpu_seconds}. With `write_annotations`, the detections go into the
        capture's .sigmf-meta (atk:source "proposed"; earlier proposed
        annotations are replaced, nothing else is touched)."""
        from atk_diffusion import sigmf as _sigmf
        say = progress or self.progress
        meta = _sigmf.read_meta(path)
        cap = _profiles.profile_from_meta(meta)
        if cap != self.profile.id:
            try:
                a, b = _profiles.describe(self.profile.id), _profiles.describe(cap)
            except ValueError:
                a, b = self.profile.id, cap
            raise _profiles.ProfileMismatch(
                f"this detector is set up for {a} (its STFT geometry, floor and "
                f"models are that receiver's); this capture is {b}. Profiles "
                "never mix: run the detector for that receiver, or resample the "
                "capture deliberately (a logged step).")
        fs = _sigmf.sample_rate_of(meta)
        if abs(fs - self.fs) > 0.5:
            raise _profiles.ProfileMismatch(
                f"the capture says {cap} but its sample rate is {fs:g} S/s, not "
                f"{self.fs:g}; its metadata is inconsistent")
        n = _sigmf.num_samples(path, meta)
        if max_seconds is not None:
            n = min(n, int(max_seconds * fs))
        multi = _sigmf.channels_of(meta) > 1
        bounds = sorted({int(c.get("core:sample_start", 0))
                         for c in meta.get("captures", []) or []} | {n})
        epoch = _epoch_of(meta)
        self.reset(counters=True)
        chunk = max(1, int(chunk_seconds * fs))
        dets: list[Detection] = []
        w0, c0 = time.perf_counter(), time.process_time()
        pos = 0
        last_say = -1
        while pos < n:
            seg_end = next((b for b in bounds if b > pos), n)
            count = min(chunk, seg_end - pos, n - pos)
            x = _sigmf.load(path, pos, count, channel=channel if multi else None,
                            meta=meta)
            dets += self.feed(x, _sigmf.center_of(meta, pos), t_start=pos / fs,
                              epoch=epoch)
            pos += count
            if say and int(10 * pos / n) != last_say:
                last_say = int(10 * pos / n)
                try:
                    say(f"detecting: {pos / fs:.1f} of {n / fs:.1f} s, "
                        f"{len(dets)} detections so far")
                except Exception:                      # noqa: BLE001
                    pass
        dets += self.finish()
        wall, cpu = time.perf_counter() - w0, time.process_time() - c0
        written = 0
        if write_annotations:
            written = self.write_annotations(path, dets, fs, n)
        tracks = self.tracker.active() + list(self.tracker.closed)
        return {"detections": dets, "tracks": tracks, "status": self.status(),
                "annotations_written": written, "seconds": n / fs,
                "wall_seconds": wall, "cpu_seconds": cpu,
                "multichannel_note": (f"channel {channel} of a multi-channel "
                                      "capture" if multi else "")}

    def write_annotations(self, path, dets, fs: float, n_samples: int) -> int:
        """Detections -> SigMF annotations in the capture's own meta."""
        from atk_diffusion import sigmf as _sigmf
        anns = []
        for d in dets:
            s0 = max(0, int(round(d.t0 * fs)))
            cnt = max(1, min(int(round((d.t1 - d.t0) * fs)), n_samples - s0))
            extra = {"atk:source": "proposed", "atk:tier": d.state,
                     "atk:proposer": "+".join(d.sources), "atk:family": d.family}
            if d.confidence is not None:
                extra["atk:confidence"] = round(float(d.confidence), 4)
            if d.snr_db is not None:
                extra["atk:snr_db"] = round(float(d.snr_db), 2)
            if d.track_id:
                extra["atk:track_id"] = d.track_id
            if d.flags:
                extra["atk:flags"] = list(d.flags)
            if d.alpha_hz:
                extra["atk:alpha_hz"] = float(d.alpha_hz)
            anns.append(_sigmf.Annotation(s0, cnt, float(d.f_lo), float(d.f_hi),
                                          d.cls or d.family, extra=extra))
        total = _sigmf.add_annotations(path, anns, replace_source="proposed")
        if self.rf is not None:
            try:
                self.rf.record(_sigmf.meta_path(path), "annotations",
                               f"{len(anns)} proposed detections written by the "
                               "detector pipeline")
            except Exception as e:                     # noqa: BLE001
                self._note(f"the write log could not record the annotations: {e}")
        self._note(f"wrote {len(anns)} proposed annotations ({total} in the file)")
        return len(anns)


def _contains(outer: Detection, inner: Detection) -> bool:
    return (outer.t0 <= inner.t0 and inner.t1 <= outer.t1
            and outer.f_lo <= inner.f_lo and inner.f_hi <= outer.f_hi)


def _epoch_of(meta: dict) -> float | None:
    caps = sorted(meta.get("captures", []) or [],
                  key=lambda c: int(c.get("core:sample_start", 0)))
    for c in caps:
        s = c.get("core:datetime")
        if s:
            try:
                return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
            except ValueError:
                return None
    return None
