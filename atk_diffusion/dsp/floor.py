# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The noise floor every tile is measured against (DETECTION_DESIGN §2;
plan §3.4; decision D5).

*"Inputs are dB above the measured noise floor, not absolute dB. The floor
comes from the profile's impairment measurement (plan §3.4) and is tracked
slowly at run time."* That one choice takes gain settings, LNA state and the
receiver's sensitivity out of what a model has to learn. So the floor is a
measurement, made carefully, and never silently pulled up by the very signals
it is the reference for.

WHERE A FLOOR COMES FROM, best first:

1. `from_terminated` — a capture with the antenna replaced by a terminator:
   no signals, so the mean power per bin over time IS the floor (the maximum
   likelihood estimate for exponential bin powers). The receiver's own DC
   spike and spurs are part of it on purpose: with no antenna they are the
   receiver, and folding them into the floor stops them being proposed
   forever. A "terminated" capture whose bins are not stationary (bursts)
   is named in `notes` — was the antenna still connected?
2. `from_profile` — the same shape stored in the profile's `impairments`
   (`floor_db_per_bin`, written by `to_impairments`). A stored shape is
   LEVELLED once against live data (`align`), because the measurement was
   made at some gain and today's may differ.
3. `estimate` — from live data, robustly: a low per-bin quantile across
   frames, bias-corrected for exponential noise (the q-quantile of Exp(μ) is
   −μ·ln(1−q)); a morphological opening across bins (removes any signal
   narrower than a quarter of the span, keeps the monotone roll-off at the
   band edges exactly); then, twice, every cell above 4× that prior (a
   noise cell does so 1.8 % of the time) is excluded — and the frequency
   neighbours of strong cells with it (window skirts) — each bin's mean is
   re-estimated from what is left with the
   TRUNCATED-exponential maximum likelihood (so excluding the top of the
   noise does not bias it low), bins that lost more than a quarter of their
   cells (occupied: what is left there is a signal's lower tail) are filled
   from their neighbours, and a median filter across bins takes out what is
   left.
   This is what keeps the floor within 1 dB with 30 % of the cells occupied
   (the test) — a plain low percentile is biased up by −10·log10(1−occupancy)
   dB (1.5 dB at 30 %), which is why it is only the starting point here.

TRACKING (`update`, once per tile from that tile's new frames):
* a GLOBAL offset — the median over bins of (this block's quantile estimate
  − the floor) — is followed with time constant `tau_level_s` (5 s), and
  snapped to at once when it exceeds `jump_db` (1 dB in one block is never
  thermal drift: it is a gain change), settling fully on the next block too,
  since the block that held the step measured only part of it; the tile it
  happened in is flagged (`Tile.floor_ok`);
* each bin's residual is followed slowly (`tau_s`, 30 s), clipped to
  `max_step_db`, and GATED: a bin that rose more than `gate_db` above the
  floor is a signal that arrived, not the floor moving, and is not followed.
  The exclusion threshold in tracking is set from the CURRENT floor, so a
  measured DC spike is kept and never eroded.

UNITS: dB of |FFT(x·w)|²/(Σw)² per bin (dsp.stft), shifted low -> high, at
the profile's STFT geometry; `floor_db` has one value per FFT bin.

LIMITS, stated: a signal occupying more than half of the bins at once, or a
bin occupied all the time by a signal less than 3 dB above the floor (measured:
at 1–2 dB the floor absorbs it, rising ~3.5 dB; from 3 dB up it does not),
cannot be told from the floor by any blind estimator — that is what the
terminated measurement is for. Without one, the receiver's own DC spike and spurs are
not in the floor and will be proposed (the class table's `dc_spike` and
`spur` negatives exist for exactly that).
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

#: Cells above this many times the prior floor are excluded as "probably
#: signal". A noise-only cell exceeds 4x its mean 1.8 % of the time (Exp(1)
#: tail) and the truncated-exponential estimate corrects for exactly that, so
#: the line can sit low enough to catch weak channels: measured, a line at
#: 6.9x let 8 dB channels busy two-thirds of the time lift the floor by 5.9 dB;
#: with this line (and the occupied-bin and skirt rules below) the worst bin
#: of the 32 %-occupied test scene is 0.29–0.54 dB off over five seeds
#: (0.46–0.62 dB from the 512-frame subset a tile uses), with no bias on noise
#: alone (tests/test_detect_frontend.py).
EXCLUDE_FACTOR = 4.0

#: Only cells this far above the prior (14 dB; a noise cell: e^-25) also
#: exclude their frequency neighbours as window skirt. Not every exceedance:
#: the window correlates neighbouring bins (Hann: 0.44), so dropping the
#: neighbours of mere noise exceedances drops above-average NOISE cells and
#: biased the floor 0.27 dB low on real STFT data — measured, then fixed.
SKIRT_FACTOR = 25.0

#: Bins need at least this many surviving cells for their own estimate;
#: fewer and they are filled from their neighbours.
MIN_CELLS = 16

#: ...and at least this fraction of their cells must survive. A noise-only
#: bin keeps ~98 %; a bin that lost more than a quarter of its cells is
#: OCCUPIED, and what survives there is the lower tail of a signal, not the
#: floor — measured: without this rule a 200 kHz signal 10 dB up, on all
#: the time, was absorbed into the floor and never detected.
MIN_KEPT_FRACTION = 0.75

_TINY = 1e-30


def _db(x) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=np.float64), _TINY))


def _lin(db) -> np.ndarray:
    return np.power(10.0, np.asarray(db, dtype=np.float64) / 10.0)


def _odd(n: int) -> int:
    n = max(1, int(n))
    return n if n % 2 else n + 1


def quantile_floor(frames, q: float = 0.25) -> np.ndarray:
    """Per-bin q-quantile across frames, bias-corrected for exponential bin
    power: an unbiased floor where a bin is empty, biased up by
    −10·log10(1−occupancy) dB where it is not."""
    f = np.asarray(frames)
    v = np.quantile(f, q, axis=0)
    return np.asarray(v, dtype=np.float64) / (-math.log1p(-q))


def opening_db(db, width: int) -> np.ndarray:
    """Morphological opening across bins: removes peaks narrower than
    `width`, keeps valleys and monotone ramps (the band-edge roll-off)."""
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    w = _odd(width)
    return maximum_filter1d(minimum_filter1d(np.asarray(db, np.float64), w,
                                             mode="nearest"), w, mode="nearest")


#: Bins either side of an occupied run that are not trusted as the ends of
#: the fill across it: the window's leakage lifts them (Hann: ±2 bins).
#: Measured: filling from the adjacent bins put a 200 kHz signal's floor
#: 2.5 dB high; reaching past the skirt puts it on the floor.
SKIRT_BINS = 3


def _fill_gaps(db: np.ndarray, good: np.ndarray, skirt: int = SKIRT_BINS) -> np.ndarray:
    """Interpolate the floor (linearly in dB) across bins that are not
    `good`, from good bins at least `skirt` bins away from them — not from
    the signal's skirts. (Tried and measured worse: filling the residual
    against the opening; where occupancy is broad the opening is lifted too.)"""
    bad = ~np.asarray(good, dtype=bool)
    if not bad.any():
        return db
    if skirt > 0:
        from scipy.ndimage import binary_dilation
        wide = binary_dilation(bad, iterations=int(skirt))
        if (~wide).any():
            bad = wide
    if bad.all():
        return db
    idx = np.arange(db.size)
    out = np.asarray(db, dtype=np.float64).copy()
    out[bad] = np.interp(idx[bad], idx[~bad], out[~bad])
    return out


def truncated_mle(frames, prior_lin, factor: float = EXCLUDE_FACTOR,
                  min_cells: int = MIN_CELLS, iters: int = 8):
    """Per-bin floor from the cells below factor×prior (and not next to a
    cell above it), corrected for the truncation:

        E[X | X < c] = μ − c·e^(−c/μ) / (1 − e^(−c/μ))

    solved for μ by fixed point (a contraction: its slope at c/μ = 4 is 0.3).
    Returns (mu, good, kept_counts)."""
    f = np.asarray(frames, dtype=np.float32)
    prior = np.asarray(prior_lin, dtype=np.float64)
    c = (factor * prior).astype(np.float32)
    exc = f > c[None, :]
    # a cell beside a STRONG one is signal skirt, not floor (window leakage)
    strong = f > (SKIRT_FACTOR * prior).astype(np.float32)[None, :]
    side = np.zeros_like(exc)
    side[:, 1:] |= strong[:, :-1]
    side[:, :-1] |= strong[:, 1:]
    keep = ~(exc | side)
    cnt = keep.sum(axis=0)
    s = np.where(keep, f, np.float32(0)).sum(axis=0, dtype=np.float64)
    m = s / np.maximum(cnt, 1)
    cc = c.astype(np.float64)
    mu = prior.copy()
    for _ in range(iters):
        r = cc / np.maximum(mu, _TINY)
        mu = m + cc * np.exp(-r) / np.maximum(-np.expm1(-r), _TINY)
    good = (cnt >= min_cells) & (cnt >= MIN_KEPT_FRACTION * f.shape[0]) & (m > 0)
    return mu, good, cnt


class NoiseFloor:
    """The per-bin floor a tile is measured against (module docstring).

    `floor_db` float32 [fft_size], dB per bin, shifted low -> high.
    `above(spec_db)` -> dB above the floor (broadcast over the last axis).
    """

    def __init__(self, floor_db, source: str = "", frames: int = 0,
                 needs_alignment: bool = False, tau_s: float = 30.0,
                 tau_level_s: float = 5.0, gate_db: float = 3.0,
                 max_step_db: float = 1.0, jump_db: float = 1.0,
                 q: float = 0.25, notes: list | None = None):
        fl = np.asarray(floor_db, dtype=np.float32).ravel()
        if fl.size < 2 or not np.all(np.isfinite(fl)):
            raise ValueError("a noise floor needs a finite value for every bin")
        self.floor_db = fl.copy()
        self.source = str(source)
        self.frames = int(frames)
        self.needs_alignment = bool(needs_alignment)
        self.tau_s = float(tau_s)
        self.tau_level_s = float(tau_level_s)
        self.gate_db = float(gate_db)
        self.max_step_db = float(max_step_db)
        self.jump_db = float(jump_db)
        self.q = float(q)
        self.notes: list[str] = list(notes or [])
        self.updates = 0
        self.last: dict = {}
        self._settle = 0

    # -- reading ---------------------------------------------------------------
    @property
    def bins(self) -> int:
        return int(self.floor_db.size)

    @property
    def floor_lin(self) -> np.ndarray:
        return _lin(self.floor_db)

    def level_db(self) -> float:
        """The floor's median level, dB per bin."""
        return float(np.median(self.floor_db))

    def above(self, spec_db) -> np.ndarray:
        """dB above the floor. The last axis must be the bins."""
        s = np.asarray(spec_db, dtype=np.float32)
        if s.shape[-1] != self.bins:
            raise ValueError(f"this floor has {self.bins} bins; the spectrum "
                             f"has {s.shape[-1]} — a floor belongs to one STFT "
                             "geometry")
        return (s - self.floor_db).astype(np.float32)

    def copy(self) -> "NoiseFloor":
        out = NoiseFloor(self.floor_db, self.source, self.frames,
                         self.needs_alignment, self.tau_s, self.tau_level_s,
                         self.gate_db, self.max_step_db, self.jump_db, self.q,
                         list(self.notes))
        out.updates = self.updates
        out._settle = self._settle
        return out

    def describe(self) -> str:
        lo, hi = float(np.min(self.floor_db)), float(np.max(self.floor_db))
        words = (f"noise floor {self.source or 'of unknown origin'}: median "
                 f"{self.level_db():.1f} dB per bin (bins span {lo:.1f} … "
                 f"{hi:.1f} dB)")
        if self.updates:
            words += f", tracked over {self.updates} updates"
        if self.needs_alignment:
            words += ", not yet levelled against live data"
        for n in self.notes:
            words += f"; {n}"
        return words

    # -- construction ------------------------------------------------------------
    @classmethod
    def estimate(cls, frames_lin, q: float = 0.25, open_bins: int | None = None,
                 smooth_bins: int | None = None, refine: int = 2,
                 source: str = "estimated from live data", **kw) -> "NoiseFloor":
        """Robust floor from single-frame bin powers [frames, bins] (linear,
        as dsp.stft.frame_power makes them). `open_bins` (default a quarter
        of the span) bounds the widest signal the first pass removes;
        `smooth_bins` (default 1/32 of the span; 0 = none) is the final
        median filter."""
        f = np.asarray(frames_lin, dtype=np.float32)
        if f.ndim != 2 or f.shape[0] < 4:
            raise ValueError("at least four frames are needed to estimate a "
                             "noise floor")
        n, b = f.shape
        ow = _odd(open_bins if open_bins else max(3, b // 4))
        sw = _odd(smooth_bins) if smooth_bins else (
            0 if smooth_bins == 0 else _odd(max(3, b // 32)))
        prior_db = opening_db(_db(quantile_floor(f, q)), ow)
        for _ in range(max(1, int(refine))):
            mu, good, _cnt = truncated_mle(f, _lin(prior_db))
            db = _fill_gaps(_db(mu), good)
            if sw > 1:
                from scipy.ndimage import median_filter
                db = median_filter(db, size=sw, mode="nearest")
            prior_db = db
        return cls(prior_db, source=source, frames=n, q=q, **kw)

    @classmethod
    def from_samples(cls, x, fs: float, geom, max_frames: int = 4096,
                     **kw) -> "NoiseFloor":
        """`estimate` from IQ samples through the profile's STFT."""
        frames = _frames_lin(x, geom, max_frames)
        return cls.estimate(frames, **kw)

    @classmethod
    def from_terminated(cls, x_or_path, fs: float | None = None, geom=None,
                        max_seconds: float = 10.0, channel: int = 0,
                        profile=None, **kw) -> "NoiseFloor":
        """The floor MEASURED from a terminated capture (a SigMF path, or IQ
        samples with `fs`). With `profile` (a ReceiverProfile), the capture's
        profile must match it and the geometry is the profile's."""
        from atk_diffusion import profiles as _profiles
        from atk_diffusion import sigmf as _sigmf
        name = "samples"
        if isinstance(x_or_path, (str, Path)):
            meta = _sigmf.read_meta(x_or_path)
            fs = _sigmf.sample_rate_of(meta)
            pid = _profiles.profile_from_meta(meta)
            if profile is not None:
                _profiles.check_match(profile.id, pid,
                                      what=f"the noise floor of {profile.id}")
            n = _sigmf.num_samples(x_or_path, meta)
            count = min(n, int(max_seconds * fs))
            ch = channel if _sigmf.channels_of(meta) > 1 else None
            x = _sigmf.load(x_or_path, 0, count, channel=ch, meta=meta)
            name = _sigmf.base_of(x_or_path).name
        else:
            x = x_or_path
            if fs is None:
                raise ValueError("samples need their sample rate (fs)")
        if geom is None:
            geom = profile.stft if profile is not None else \
                _profiles.default_stft(float(fs))
        frames = _frames_lin(x, geom, max_frames=int(max_seconds * float(fs)
                                                      / geom.hop) + 1)
        if frames.shape[0] < 16:
            raise ValueError(f"the terminated capture holds {frames.shape[0]} "
                             "frames; measure at least a second")
        mu = frames.mean(axis=0, dtype=np.float64)
        notes = []
        # stationarity: against a level a burst cannot lift (median / ln 2 is
        # the exponential mean), a noise cell exceeds 20x with probability
        # e^-20 — a handful of such cells means bursts were received. A spur
        # is stationary and passes (it is the receiver's).
        robust = np.median(frames, axis=0) / math.log(2.0)
        hot = int(np.sum(frames > 20.0 * robust[None, :]))
        if hot >= 3:
            notes.append(f"{hot} of its cells are bursts more than 13 dB above "
                         "their bin's usual level — was the antenna still "
                         "connected? A terminated floor should hold only the "
                         "receiver")
        return cls(_db(mu), source=f"measured from terminated capture {name}",
                   frames=int(frames.shape[0]), notes=notes, **kw)

    @classmethod
    def from_profile(cls, profile, **kw) -> "NoiseFloor | None":
        """The measured shape in `profile.impairments['floor_db_per_bin']`,
        to be levelled against live data on first use; None if the profile
        has none. A shape for another FFT size is refused in words."""
        imp = getattr(profile, "impairments", None) or {}
        v = imp.get("floor_db_per_bin")
        if v is None:
            return None
        arr = np.asarray(v, dtype=np.float32).ravel()
        n = int(profile.stft.fft_size)
        if arr.size != n:
            raise ValueError(f"the profile's measured floor has {arr.size} "
                             f"bins but its STFT geometry has {n}; re-measure "
                             "the floor terminated (plan §3.4) so the shape "
                             "matches the geometry")
        when = imp.get("floor_measured", "") or "date not recorded"
        return cls(arr, source=f"measured terminated ({when}), from the profile",
                   needs_alignment=True, **kw)

    def to_impairments(self) -> dict:
        """The entries `profiles\\<p>.json` `impairments` stores (plan §3.4)."""
        return {"floor_db_per_bin": [round(float(v), 3) for v in self.floor_db],
                "floor_fft_size": self.bins,
                "floor_source": self.source,
                "floor_measured": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                time.gmtime()),
                "floor_units": "dB of |FFT(x*w)|^2/(sum w)^2 per bin, shifted "
                               "low to high, at the profile's STFT geometry"}

    # -- tracking ----------------------------------------------------------------
    def align(self, frames_lin) -> float:
        """Level a measured shape against live data once: shift it by the
        median over bins of (live quantile estimate − shape). Returns the
        shift in dB."""
        f = np.asarray(frames_lin, dtype=np.float32)
        self._check_bins(f)
        g = float(np.median(_db(quantile_floor(f, self.q)) - self.floor_db))
        self.floor_db = (self.floor_db + g).astype(np.float32)
        self.needs_alignment = False
        self.last = {"aligned_db": g}
        return g

    def update(self, frames_lin, duration_s: float) -> dict:
        """Track slowly from new single-frame powers [frames, bins] covering
        `duration_s` seconds. Returns what happened, in numbers:
        offset_db (the global offset seen), snapped (a gain change was
        followed at once), gated_bins (bins that rose like a signal and were
        not followed), unmeasured_bins, moved_db (median change applied)."""
        f = np.asarray(frames_lin, dtype=np.float32)
        self._check_bins(f)
        if f.shape[0] < 4:
            return {"offset_db": 0.0, "skipped": "too few frames"}
        old = self.floor_db.astype(np.float64)
        g = float(np.median(_db(quantile_floor(f, self.q)) - old))
        base = old + g
        mu, good, _cnt = truncated_mle(f, _lin(base))
        resid = _db(mu) - base
        accept = good & (resid <= self.gate_db)
        gated = int(np.sum(good & (resid > self.gate_db)))
        w_bin = min(1.0, max(0.0, float(duration_s)) / self.tau_s)
        snapped = abs(g) > self.jump_db
        # a step usually lands mid-block, so the block that saw it measured
        # only part of it: the NEXT update settles fully too
        settling = self._settle > 0 and not snapped
        if snapped:
            self._settle = 1
        elif self._settle > 0:
            self._settle -= 1
        w_lvl = 1.0 if (snapped or settling) else \
            min(1.0, max(0.0, float(duration_s)) / self.tau_level_s)
        step = np.where(accept, np.clip(resid, -self.max_step_db,
                                        self.max_step_db), 0.0) * w_bin
        new = old + w_lvl * g + step
        self.floor_db = new.astype(np.float32)
        self.updates += 1
        self.last = {"offset_db": g, "snapped": bool(snapped),
                     "settling": bool(settling), "gated_bins": gated,
                     "unmeasured_bins": int(np.sum(~good)),
                     "moved_db": float(np.median(new - old))}
        return dict(self.last)

    def _check_bins(self, f: np.ndarray) -> None:
        if f.ndim != 2 or f.shape[1] != self.bins:
            raise ValueError(f"this floor has {self.bins} bins; the frames "
                             f"have {f.shape[-1] if f.ndim else 0} — a floor "
                             "belongs to one STFT geometry")


def _frames_lin(x, geom, max_frames: int) -> np.ndarray:
    """Single-frame linear powers of up to `max_frames` frames of `x`."""
    from atk_diffusion.dsp import stft as _stft
    x = np.asarray(x, dtype=np.complex64).ravel()
    n, hop = int(geom.fft_size), int(geom.hop)
    need = (max(1, int(max_frames)) - 1) * hop + n
    x = x[:need]
    view = _stft._frames_view(x, n, hop)
    w = _stft.window_array(geom.window, n)
    out = np.empty((view.shape[0], n), dtype=np.float32)
    for i in range(0, view.shape[0], _stft.FFT_BATCH):
        out[i:i + _stft.FFT_BATCH] = _stft.frame_power(view[i:i + _stft.FFT_BATCH], w)
    return out
