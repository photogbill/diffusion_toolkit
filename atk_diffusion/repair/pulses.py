# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Pulse-train completion and deinterleaving for the Signals bench — plan
§4.D3.

    *"Dropped pulses from fading wreck PRI and stagger analysis; inpaint the
    train and flag inferred pulses."*  — plan §4.D3

THE INPUT IS ATK'S OWN PDW. `atk/core/pulse.py` describes every pulse as a
dict — `start`, `end` (sample indices), `toa_s`, `width_s`, `amplitude`,
`amplitude_db`, `freq_hz`, `chirp_hz` — and this module reads exactly that
shape (only `toa_s` is required). It ADDS, on copies: `emitter` (an id, −1
unassigned), `inferred` (False for a received pulse), `tier` ("measured"
for a received PDW — a classical measurement of the record — and "inferred"
for a completed one), and on inferred pulses `sigma_toa_s`, `method`
("pri_fill"), `reason` and `masked_by`.

AN INFERRED PULSE CANNOT BE MISTAKEN FOR A RECEIVED ONE. It has
`inferred: True`, `tier: "inferred"`, NO samples (`start = end = −1`, so any
code that slices the I/Q by start/end — ATK's `intra_pulse_summary` does —
skips it) and NO amplitude (`amplitude`, `amplitude_db` are NaN: a pulse
that was not received has no measured level, and a fabricated one would
pass for a measurement). Its width, carrier and chirp are the emitter's
medians, because that is what "the same emitter's missing pulse" means.

THE METHOD, step by step — every rule below is there because a measured
failure needed it (`experiments.pulse_eval`):

  1. **Carrier and width** cluster the pulses first (as ATK's de-interleaver
     does): emitters on different carriers never meet in the PRI stage.
  2. **SDIF** — Milojević & Popović (1992), "Improvement of a deinterleaving
     algorithm based on SDIF": the histogram of differences of ONE level c
     at a time, each against the threshold T(τ) = x·(E − c)·e^(−τ/(k·τmax)),
     E the pulses still unassigned. **CDIF** — Mardia (1989) — accumulates
     the levels and asks that both τ and 2τ clear the threshold; it is
     computed for the analysis view. Counting is in a ±`rel_tol` window on a
     geometric grid (a PRI at 10 µs and one at 10 ms get the same relative
     resolution), and each peak is refined to the mean of the differences
     in its window — the first maximum of a flat-topped window sits up to
     `rel_tol` low, enough to derail step 3.
  3. **Sequence search** for each candidate PRI, smallest first, with up to
     `max_missing` consecutive pulses allowed missing — the fading the plan
     is about — on a grid refitted by least squares as pulses are found
     (jitter is measured against the grid, never against the last pulse). A
     sequence found is removed and the histograms rebuilt (M&P's loop). Two
     PHASES: the tight tolerance to exhaustion first, so constant trains and
     staggers claim their pulses, then the jitter tolerance on the rest.
  4. **What a sequence must pass**, each against a failure seen:
       * a PATTERN of misses is refused — a 1/2 ms stagger is exactly a
         1 ms train missing every third pulse; fading misses at random, a
         stagger at the same place every frame (`periodic_misses`);
       * SIGNIFICANCE — the binomial chance that random pulses at the
         available density fill that many windows, times the searches made,
         must be under 1 % (five noise pulses can look periodic);
       * TIGHTNESS — residual σ under 0.4 × the window: coincidences fill a
         window uniformly (σ ≈ 0.58 × it), a real train does not;
       * NOT A HARMONIC — if every intermediate position b·k/n holds a pulse
         (clearly more often than chance) scattered more widely than it is
         offset, the sequence is every n-th pulse of a jittered train; a
         stagger's frame passes, its other positions sit in a tight cluster
         at a constant offset (`is_harmonic`);
       * in the tight phase, NOT JITTERED BEYOND THE WINDOW — a sequence
         whose missed slots have a pulse just outside the window is left to
         the jitter phase (`near_misses`).
  5. **Merging**, by period family (shortest first, so a multiple meets its
     fundamental) and in time order (so a train broken by fades grows left
     to right). A sequence on an existing sub-train's grid continues it.
     Same period, overlapping in time and phase-locked: another POSITION of
     a stagger (levels from the phases; positions closer than the train's
     own jitter are one position; levels that repeat are reduced to the
     smallest repeating unit — 2/1/2/1 at 6 ms is 2/1 at 3 ms). Same period,
     overlapping, NOT locked: two emitters ("also for overlapping emitters":
     the same carrier and width, separated by PRI). Same period one after
     the other: one emitter's separate stretches, never filled between.
     Same-PRI emitters whose pulses sit on one grid are joined.
  6. **Absorption, then completion.** Before anything is inferred, an
     UNASSIGNED received pulse on an emitter's grid with the emitter's
     carrier and width is taken into it — a real pulse always beats an
     inferred one. Then each gap of k PRIs (2 ≤ k ≤ `max_missing` + 1, and
     only if it is a whole number of PRIs within tolerance) gets k − 1
     inferred pulses, spaced between the received ones either side. One
     that lands on another emitter's pulse is marked `reason: "collision"`
     with `masked_by`: the two arrived together and the detector saw one.

JITTER IS MEASURED ON RECEIVED PULSES ONLY. An inferred pulse sits on the
grid, so a completed train looks more regular than the emitter is — measured:
a first-difference classifier calls a ±6 % jittered train "constant" once it
is completed. The emitter's `jitter_pct` never includes inferred pulses.

MEASURED (`experiments.pulse_eval`, six emitters — constant, jittered ±6 %,
two- and three-level staggers, two on one carrier and width, one scanning —
Gilbert–Elliott fading at 15 % mean drop, 150 noise pulses/s, 12 scenes):
99.7 % of received pulses to the right emitter; 0 of 2,310 inferred pulses
where nothing was transmitted; 80 % of dropped pulses recovered; the
mean-of-first-differences PRI error 15.7 % on received pulses and 0.7 % on
completed trains. At 30 % mean drop: 98 % and 0.9 % false pulses — it
degrades, and says so.

THE ONE THING THIS DOES NOT DO: separate two pulses that overlap IN TIME.
The Fraunhofer time-domain blind source separation paper (plan §9,
2509.15603) works on the I/Q itself; this works on PDWs, where a collision is
already one PDW. Two identical radars (same carrier, width and PRI) whose
clocks stay locked for the whole capture are indistinguishable here — a
bearing tells them apart, not a PDW.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from atk_diffusion import provenance as _prov

#: The fields ATK's pulse.py writes (only toa_s is required here).
PDW_FIELDS = ("start", "end", "toa_s", "width_s", "amplitude", "amplitude_db",
              "freq_hz", "chirp_hz")

METHOD = "pri_fill"                      # provenance: inferred
_prov.METHOD_TIERS.setdefault(METHOD, "inferred")

DEFAULTS = {
    "rel_tol": 0.03,         # window and search tolerance, fraction of the PRI
    "jitter_tol": 0.15,      # the second pass, for jittered trains
    "abs_tol_s": 1e-6,       # never tighter than the TOA measurement
    "max_missing": 4,        # most consecutive pulses filled
    "min_pulses": 5,         # a sequence shorter than this is not an emitter
    "max_level": 8,          # SDIF levels examined
    "x": 0.10, "k": 0.30,    # Milojević & Popović threshold constants
    "freq_tol_hz": 20_000.0, # carrier clustering (ATK's default)
    "width_tol": 0.35,       # width clustering, relative (ATK's default)
    "jitter_pct": 1.0,       # σ/PRI above this is called jittered
    "max_sigma_ratio": 0.4,  # a sequence's residual σ must be under this × tol
    "defer_jittered": True,  # tight-pass sequences whose misses have pulses just
                             # outside the window are left to the jitter pass
}


# ---------------------------------------------------------------------------
# PDWs in ATK's shape
# ---------------------------------------------------------------------------
def as_pdws(rows) -> list[dict]:
    """Copies of PDW dicts (ATK's shape), or an array of TOAs in seconds,
    sorted by time. Missing optional fields are NaN; nothing else is
    changed."""
    out = []
    for r in rows:
        if isinstance(r, dict):
            d = dict(r)
        else:
            d = {"toa_s": float(r)}
        if "toa_s" not in d:
            raise ValueError("a PDW needs toa_s (seconds)")
        d["toa_s"] = float(d["toa_s"])
        for f in ("width_s", "freq_hz", "chirp_hz", "amplitude", "amplitude_db"):
            d.setdefault(f, math.nan)
        d.setdefault("start", -1)
        d.setdefault("end", -1)
        out.append(d)
    out.sort(key=lambda d: d["toa_s"])
    return out


def toas(pdws) -> np.ndarray:
    if isinstance(pdws, np.ndarray):
        return np.sort(pdws.astype(np.float64))
    return np.array([float(p["toa_s"]) if isinstance(p, dict) else float(p)
                     for p in pdws], dtype=np.float64)


def received_only(train) -> list[dict]:
    return [p for p in train if not p.get("inferred")]


# ---------------------------------------------------------------------------
# Difference histograms, the M&P threshold, the PRI transform
# ---------------------------------------------------------------------------
@dataclass
class HistLevel:
    """One level of SDIF (or CDIF up to that level): counts on a geometric
    τ grid, the threshold, and the peaks that clear it (ascending τ)."""
    level: int
    taus: np.ndarray
    counts: np.ndarray
    threshold: np.ndarray
    peaks: list
    cumulative: bool = False

    def to_json(self) -> dict:
        return {"level": self.level, "cumulative": self.cumulative,
                "peaks_s": [float(p) for p in self.peaks],
                "taus_s": self.taus.tolist(), "counts": self.counts.tolist(),
                "threshold": self.threshold.tolist()}


def tau_grid(t, rel_tol: float, tau_max: float | None = None,
             tau_min: float | None = None) -> np.ndarray:
    t = np.sort(np.asarray(t, dtype=np.float64))
    if t.size < 3:
        return np.zeros(0)
    d1 = np.diff(t)
    pos = d1[d1 > 0]
    lo = float(tau_min or (np.min(pos) * 0.5 if pos.size else 1e-9))
    span = float(t[-1] - t[0])
    hi = float(tau_max or span / 3.0)
    if hi <= lo:
        return np.zeros(0)
    step = 1.0 + max(rel_tol, 1e-4) / 2.0
    n = int(np.ceil(np.log(hi / lo) / np.log(step))) + 1
    return lo * step ** np.arange(min(n, 20000))


def _window_counts(d: np.ndarray, taus: np.ndarray, rel_tol: float) -> np.ndarray:
    d = np.sort(d)
    hi = np.searchsorted(d, taus * (1 + rel_tol), side="right")
    lo = np.searchsorted(d, taus * (1 - rel_tol), side="left")
    return (hi - lo).astype(np.float64)


def mp_threshold(taus, n_pulses: int, level: int, tau_max: float,
                 x: float = DEFAULTS["x"], k: float = DEFAULTS["k"]) -> np.ndarray:
    """Milojević & Popović's T(τ) = x·(E − c)·exp(−τ/(k·τmax))."""
    return x * max(n_pulses - level, 0) * np.exp(-np.asarray(taus) / (k * tau_max))


def _peaks(taus, counts, thr, rel_tol, d=None) -> list:
    """Peaks above the threshold, strongest first claimed, each REFINED to
    the mean of the differences inside its window: a ±rel_tol window makes
    a flat-topped plateau, and taking its first maximum would bias every
    PRI low by up to rel_tol — enough to derail the sequence search."""
    above = counts > np.maximum(thr, 1.0)
    if not above.any():
        return []
    c = counts.copy()
    c[~above] = -1
    order = np.argsort(-c, kind="stable")
    ds = np.sort(d) if d is not None else None
    taken: list[float] = []
    for i in order:
        if c[i] < 0:
            break
        tau = float(taus[i])
        if ds is not None:
            lo = np.searchsorted(ds, tau * (1 - rel_tol), "left")
            hi = np.searchsorted(ds, tau * (1 + rel_tol), "right")
            if hi > lo:
                tau = float(np.mean(ds[lo:hi]))
        if any(abs(tau - u) <= 2 * rel_tol * u for u in taken):
            continue
        taken.append(tau)
    return sorted(taken)


def sdif(t, max_level: int = DEFAULTS["max_level"],
         rel_tol: float = DEFAULTS["rel_tol"], x: float = DEFAULTS["x"],
         k: float = DEFAULTS["k"], tau_max: float | None = None) -> list[HistLevel]:
    """SDIF levels 1..max_level on TOAs (or PDWs)."""
    t = toas(t) if not isinstance(t, np.ndarray) else np.sort(t)
    taus = tau_grid(t, rel_tol, tau_max)
    out = []
    if taus.size == 0:
        return out
    tmax = float(taus[-1])
    for c in range(1, min(int(max_level), t.size - 1) + 1):
        d = t[c:] - t[:-c]
        counts = _window_counts(d, taus, rel_tol)
        thr = mp_threshold(taus, t.size, c, tmax, x, k)
        out.append(HistLevel(c, taus, counts, thr,
                             _peaks(taus, counts, thr, rel_tol, d)))
    return out


def cdif(t, max_level: int = DEFAULTS["max_level"],
         rel_tol: float = DEFAULTS["rel_tol"], x: float = DEFAULTS["x"],
         k: float = DEFAULTS["k"], tau_max: float | None = None) -> list[HistLevel]:
    """CDIF (Mardia 1989): the histogram accumulated over levels 1..c; a peak
    is a candidate only if twice its τ also clears the threshold (Mardia's
    check against noise peaks)."""
    t = toas(t) if not isinstance(t, np.ndarray) else np.sort(t)
    taus = tau_grid(t, rel_tol, tau_max)
    out = []
    if taus.size == 0:
        return out
    tmax = float(taus[-1])
    acc = np.zeros(taus.size)
    dall = []
    for c in range(1, min(int(max_level), t.size - 1) + 1):
        d = t[c:] - t[:-c]
        dall.append(d)
        acc = acc + _window_counts(d, taus, rel_tol)
        thr = mp_threshold(taus, t.size, c, tmax, x, k)
        cand = _peaks(taus, acc, thr, rel_tol, np.concatenate(dall))
        keep = []
        for tau in cand:
            j = np.searchsorted(taus, 2 * tau)
            if j < taus.size and acc[j - 1:j + 2].max(initial=0) > thr[min(j, taus.size - 1)]:
                keep.append(tau)
        out.append(HistLevel(c, taus, acc.copy(), thr, keep, cumulative=True))
    return out


def pri_transform(t, taus=None, max_lag: int = 12,
                  rel_tol: float = DEFAULTS["rel_tol"]) -> tuple[np.ndarray, np.ndarray]:
    """The difference-phasor PRI transform, as in ATK's own
    `pulse.pri_transform` (after Nelson 1993): each difference d adds
    exp(j·2π·d/τ), weighted down with distance; at the true PRI the phasors
    add, at twice it they alternate and cancel. Returns (taus, |D|/Σw)."""
    t = toas(t) if not isinstance(t, np.ndarray) else np.sort(t)
    if taus is None:
        taus = tau_grid(t, rel_tol)
    taus = np.asarray(taus, dtype=np.float64)
    if taus.size == 0 or t.size < 4:
        return taus, np.zeros(taus.size)
    d = np.concatenate([t[c:] - t[:-c] for c in range(1, min(max_lag, t.size - 1) + 1)])
    d = d[d > 0]
    mag = np.zeros(taus.size)
    for i0 in range(0, taus.size, 256):
        tt = taus[i0:i0 + 256, None]
        w = 1.0 / (1.0 + (d[None, :] / tt) / max_lag)
        acc = np.sum(np.exp(2j * np.pi * d[None, :] / tt) * w, axis=1)
        mag[i0:i0 + 256] = np.abs(acc) / np.maximum(np.sum(w, axis=1), 1e-12)
    return taus, mag


# ---------------------------------------------------------------------------
# Sequence search on a refitted grid, with the periodic-miss guard
# ---------------------------------------------------------------------------
@dataclass
class _Seq:
    idx: np.ndarray          # indices into the sorted pulse list
    slots: np.ndarray        # integer slot of each pulse on the grid
    a: float                 # grid: t ≈ a + b·slot
    b: float
    sigma: float             # residual standard deviation (s)
    group: int = -1          # parameter cluster (-1 = PRI-only pass)

    @property
    def t0(self) -> float:
        return float(self.a + self.b * self.slots[0])

    @property
    def t1(self) -> float:
        return float(self.a + self.b * self.slots[-1])


def _fit(slots, tt):
    k = np.asarray(slots, dtype=np.float64)
    if k.size < 2 or np.ptp(k) == 0:
        return float(tt[0] - 0.0), 0.0, 0.0
    b, a = np.polyfit(k, tt, 1)
    res = tt - (a + b * k)
    sig = float(np.std(res, ddof=min(2, max(0, k.size - 1)))) if k.size > 2 else 0.0
    return float(a), float(b), sig


def periodic_misses(slots, max_m: int = 8, low: float = 0.15,
                    high: float = 0.6, min_hit_rate: float = 0.5) -> tuple[bool, str]:
    """(True, why) when a sequence's hits and misses form a PATTERN — some
    position modulo m (almost) always empty while another is (almost)
    always full — or when it hits fewer than half its slots. Fading misses
    at random positions; a stagger misses at the same ones every frame."""
    s = np.asarray(slots, dtype=np.int64)
    s = s - s[0]
    n = int(s[-1]) + 1
    occ = np.zeros(n, dtype=bool)
    occ[s] = True
    if occ.mean() < min_hit_rate:
        return True, (f"only {occ.mean():.0%} of the slots hold a pulse — too "
                      "sparse to be one train with fading")
    for m in range(2, min(max_m, n // 3) + 1):
        rates = [occ[r::m].mean() for r in range(m) if occ[r::m].size >= 3]
        if len(rates) == m and min(rates) <= low and max(rates) >= high:
            r = int(np.argmin(rates))
            return True, (f"position {r} of every {m} is empty in "
                          f"{1 - min(rates):.0%} of frames — a pattern, not "
                          "fading")
    return False, ""


def chance_pvalue(hits: int, slots: int, p_slot: float) -> float:
    """Probability that `hits` of `slots` windows catch a pulse by chance
    when each does so with probability `p_slot` (the first, the starting
    pulse, is not counted): the binomial upper tail, as the regularised
    incomplete beta function (scipy.special — scipy.stats costs seconds to
    import for one number)."""
    from scipy.special import betainc
    h, n = int(hits) - 1, int(slots) - 1
    if n <= 0 or h <= 0:
        return 1.0
    p = min(max(float(p_slot), 1e-12), 0.999)
    return float(betainc(h, n - h + 1, p))


def is_harmonic(t, avail, idx, b: float, rel_tol: float, jitter_tol: float,
                abs_tol: float, max_n: int = 4) -> int:
    """n (2…max_n) when a sequence of period b is every n-th pulse of a
    denser train, else 0. Every intermediate position t + k·b/n (k = 1…n−1)
    must hold a still-available pulse for at least half the sequence's
    pulses — and clearly more often than the pulse density alone would put
    one there — and those pulses must scatter about the position more
    widely than they are offset from it. That last test tells a jittered
    train seen at a multiple of its PRI (a wide scatter) from a stagger's
    frame (its other positions sit in a tight cluster at a CONSISTENT
    offset from b·k/n). The sequence-level form of the
    subharmonic check of CDIF/SDIF."""
    t = np.asarray(t, dtype=np.float64)
    own = np.asarray(idx, dtype=np.int64)
    tt = t[own]
    if tt.size < 3:
        return 0
    free = np.array(avail, dtype=bool, copy=True)
    free[own] = False
    ft = t[free]
    if ft.size < 2:
        return 0
    span = float(ft[-1] - ft[0]) or 1e-30
    dens = ft.size / span
    for n in range(2, int(max_n) + 1):
        tol = max(rel_tol * b, jitter_tol * b / n, abs_tol)
        p_chance = 1.0 - math.exp(-dens * 2.0 * tol)
        ok = True
        for k in range(1, n):
            pos = tt + k * b / n
            j = np.clip(np.searchsorted(ft, pos), 1, ft.size - 1)
            d0 = ft[j] - pos
            d1 = ft[j - 1] - pos
            off = np.where(np.abs(d0) < np.abs(d1), d0, d1)
            hit = np.abs(off) <= tol
            if np.mean(hit) < max(0.5, p_chance + 0.3):
                ok = False
                break
            o = off[hit]
            # a stagger's other positions sit in a TIGHT cluster away from
            # b·k/n (|mean| far beyond their spread); a jittered train's
            # scatter is as wide as any offset it shows
            if abs(float(np.mean(o))) > max(2.0 * float(np.std(o)),
                                            0.25 * rel_tol * b, abs_tol):
                ok = False
                break
        if ok:
            return n
    return 0


def near_misses(t, avail, idx, slots, a: float, b: float, tol: float,
                wide: float) -> tuple[float, int]:
    """(fraction, count) of a sequence's MISSED slots that have an available
    pulse just outside the window (tol < |offset| ≤ wide). A train jittered
    beyond the window misses slots that are not empty."""
    s = np.asarray(slots, dtype=np.int64)
    full = np.arange(s[0], s[-1] + 1)
    missed = np.setdiff1d(full, s)
    if missed.size == 0:
        return 0.0, 0
    ft = np.asarray(t, dtype=np.float64)[avail]
    if ft.size == 0:
        return 0.0, int(missed.size)
    pos = a + b * missed
    j = np.clip(np.searchsorted(ft, pos), 1, max(ft.size - 1, 1))
    off = np.minimum(np.abs(ft[j] - pos), np.abs(ft[j - 1] - pos))
    return float(np.mean((off > tol) & (off <= wide))), int(missed.size)


def sequence_search(t, avail, tau: float, tol: float,
                    max_missing: int = DEFAULTS["max_missing"],
                    min_pulses: int = DEFAULTS["min_pulses"],
                    trials: float = 1.0, alpha: float = 0.01,
                    max_sigma_ratio: float = DEFAULTS["max_sigma_ratio"],
                    harmonic_check: tuple | None = None,
                    defer_wide: float = 0.0) -> list[_Seq]:
    """Extract every sequence of period ≈ `tau` from the available pulses
    (marks them unavailable). See the module docstring, steps 3–4. A
    sequence must also be SIGNIFICANT: the chance that random pulses at the
    available density fill that many of its windows, times the number of
    searches made (`trials`), must be below `alpha` — a run of five noise
    pulses that happens to look periodic is not an emitter. And its pulses
    must sit TIGHTER than its window (residual σ ≤ `max_sigma_ratio`·tol):
    pulses caught by coincidence, or lucky stretches of a train jittered
    beyond the window, spread uniformly across it (σ ≈ 0.58·tol); a real
    train at this tolerance does not."""
    t = np.asarray(t, dtype=np.float64)
    found: list[_Seq] = []
    n = t.size
    if n == 0 or tau <= 0:
        return found
    live_t = t[avail]
    p_slot = _chance(live_t, tol) if live_t.size >= 2 else 0.0
    p_wide = (_chance(live_t, defer_wide) - p_slot) if defer_wide > tol else 0.0
    n_trials = max(1.0, float(trials) * max(1, live_t.size))
    lo_b, hi_b = tau * 0.95, tau * 1.05
    for i in np.flatnonzero(avail):
        if not avail[i]:
            continue
        lo = np.searchsorted(t, t[i] + tau - tol, "left")
        hi = np.searchsorted(t, t[i] + tau + tol, "right")
        if not avail[lo:hi].any():
            continue
        idx, slots = [int(i)], [0]
        a, b = float(t[i]), float(tau)
        s1 = s2 = st = skt = 0.0
        cnt = 1
        st = float(t[i])
        slot, misses = 0, 0
        last_t = t[-1] + tol
        while True:
            slot += 1
            e = a + b * slot
            if e > last_t:
                break
            lo = np.searchsorted(t, e - tol, "left")
            hi = np.searchsorted(t, e + tol, "right")
            j = -1
            if hi > lo:
                cand = np.arange(lo, hi)
                cand = cand[avail[cand]]
                if cand.size:
                    j = int(cand[np.argmin(np.abs(t[cand] - e))])
            if j >= 0:
                idx.append(j)
                slots.append(slot)
                misses = 0
                cnt += 1
                s1 += slot
                s2 += slot * slot
                st += t[j]
                skt += slot * t[j]
                if cnt >= 3:
                    den = cnt * s2 - s1 * s1
                    if den > 0:
                        b = min(hi_b, max(lo_b, (cnt * skt - s1 * st) / den))
                else:
                    # the first observed interval corrects a candidate PRI
                    # that came from a coarse histogram
                    b = min(hi_b, max(lo_b, (float(t[j]) - float(t[i])) / slot))
                # the grid's phase is the MEAN over the pulses found, never
                # the last one (a jittered pulse would drag the next window)
                a = (st - b * s1) / cnt
            else:
                misses += 1
                if misses > max_missing:
                    break
        if len(idx) < min_pulses:
            continue
        bad, _why = periodic_misses(slots)
        if bad:
            continue
        if chance_pvalue(len(idx), slots[-1] + 1, p_slot) * n_trials > alpha:
            continue
        ii = np.array(idx, dtype=np.int64)
        ss = np.array(slots, dtype=np.int64)
        fa, fb, sig = _fit(ss, t[ii])
        if sig > max_sigma_ratio * tol:
            continue
        if harmonic_check and is_harmonic(t, avail, ii, fb, *harmonic_check):
            continue
        if defer_wide > tol:
            frac, n_miss = near_misses(t, avail, ii, ss, fa, fb, tol, defer_wide)
            if n_miss >= 2 and frac >= max(0.5, p_wide + 0.3):
                continue          # jittered beyond this window: the wide pass owns it
        found.append(_Seq(ii, ss, fa, fb, sig))
        avail[ii] = False
    return found


#: A search window that a random pulse falls into this often (window width
#: × pulse density) cannot tell a train from a coincidence.
MAX_CHANCE = 0.5


def _chance(t_live: np.ndarray, tol: float) -> float:
    """Probability that SOME available pulse falls in a ±tol window by
    chance: the live pulse density times the window."""
    if t_live.size < 2:
        return 0.0
    span = float(t_live[-1] - t_live[0]) or 1e-30
    return t_live.size / span * 2.0 * tol


def _extract(t, pool: np.ndarray, avail_all: np.ndarray, cfg: dict,
             group: int) -> list[_Seq]:
    """M&P's loop on the pulses `pool` (indices into t), in two phases: the
    tight tolerance first, to exhaustion — constant and staggered trains
    claim their pulses before anything wider is tried — then the jitter
    tolerance on what is left. In each: SDIF candidates at that window,
    smallest τ first; a sequence found is removed and the histograms are
    rebuilt from level 1. A candidate whose window would catch a random
    pulse more than MAX_CHANCE of the time is not searched (after denser
    trains are removed it may be, which is why the loop rebuilds)."""
    seqs: list[_Seq] = []
    for rel in (cfg["rel_tol"], cfg["jitter_tol"]):
        while True:
            live = pool[avail_all[pool]]
            if live.size < cfg["min_pulses"]:
                break
            tl = t[live]
            levels = sdif(tl, cfg["max_level"], rel, cfg["x"], cfg["k"])
            n_cands = max(1, sum(len(L.peaks) for L in levels))
            got = False
            for L in levels:
                for tau in L.peaks:
                    tol = max(rel * tau, cfg["abs_tol_s"])
                    if tol >= 0.45 * tau or _chance(tl, tol) > MAX_CHANCE:
                        continue
                    sub_avail = np.zeros(t.size, dtype=bool)
                    sub_avail[live] = avail_all[live]
                    new = sequence_search(t, sub_avail, tau, tol,
                                          cfg["max_missing"], cfg["min_pulses"],
                                          trials=2 * n_cands,
                                          max_sigma_ratio=cfg["max_sigma_ratio"],
                                          harmonic_check=(cfg["rel_tol"],
                                                          cfg["jitter_tol"],
                                                          cfg["abs_tol_s"]),
                                          defer_wide=(cfg["jitter_tol"] * tau
                                                      if (rel < cfg["jitter_tol"] and cfg["defer_jittered"]) else 0.0))
                    if new:
                        for q in new:
                            q.group = group
                            avail_all[q.idx] = False
                        seqs += new
                        got = True
                        break
                if got:
                    break
            if not got:
                break
    return seqs


# ---------------------------------------------------------------------------
# Parameter clustering (carrier and width), as ATK's de-interleaver does
# ---------------------------------------------------------------------------
def _clusters(freq, width, cfg) -> np.ndarray:
    n = freq.size
    lab = np.full(n, -1, dtype=np.int64)
    if not np.isfinite(freq).any() and not np.isfinite(width).any():
        return np.zeros(n, dtype=np.int64)
    cf: list[float] = []
    cw: list[float] = []
    cn: list[int] = []
    for i in range(n):
        f, w = freq[i], width[i]
        best, bd = -1, np.inf
        for c in range(len(cf)):
            df = abs(f - cf[c]) if np.isfinite(f) and np.isfinite(cf[c]) else 0.0
            dw = (abs(w - cw[c]) / max(cw[c], 1e-12)
                  if np.isfinite(w) and np.isfinite(cw[c]) else 0.0)
            if df <= cfg["freq_tol_hz"] and dw <= cfg["width_tol"]:
                d = df / cfg["freq_tol_hz"] + dw / cfg["width_tol"]
                if d < bd:
                    best, bd = c, d
        if best < 0:
            cf.append(f)
            cw.append(w)
            cn.append(1)
            lab[i] = len(cf) - 1
        else:
            lab[i] = best
            cn[best] += 1
            if np.isfinite(f):
                cf[best] += (f - cf[best]) / cn[best]
            if np.isfinite(w):
                cw[best] += (w - cw[best]) / cn[best]
    return lab


# ---------------------------------------------------------------------------
# Emitters: merging sequences into sub-trains, bursts, staggers
# ---------------------------------------------------------------------------
@dataclass
class Emitter:
    """One emitter's train. `pri_s` is the PRI (constant or jittered) or the
    FRAME (staggered, with `levels_s` summing to it). `subtrains` are index
    lists into the deinterleaved PDW list — one per stagger position — each
    with its grid (a, b) and residual σ; `bursts` are (t_first, t_last)."""
    id: int
    kind: str
    pri_s: float
    levels_s: list = field(default_factory=list)
    jitter_pct: float = 0.0
    freq_hz: float = math.nan
    width_s: float = math.nan
    chirp_hz: float = math.nan
    received: list = field(default_factory=list)
    subtrains: list = field(default_factory=list)   # [{idx, a, b, sigma, burst}]
    bursts: list = field(default_factory=list)
    inferred: list = field(default_factory=list)    # inferred PDW dicts
    notes: list = field(default_factory=list)

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("inferred", None)
        d["n_received"] = len(self.received)
        d["n_inferred"] = len(self.inferred)
        return d

    def words(self) -> str:
        if self.kind.startswith("staggered"):
            lv = " / ".join(f"{v * 1e6:,.1f}" for v in self.levels_s)
            head = f"{self.kind}, frame {self.pri_s * 1e6:,.1f} µs ({lv} µs)"
        else:
            head = f"{self.kind} PRI {self.pri_s * 1e6:,.2f} µs"
            if self.kind == "jittered":
                head += f" (±{self.jitter_pct:.1f}% σ)"
        return (f"emitter {self.id}: {head}; {len(self.received)} received, "
                f"{len(self.inferred)} INFERRED pulse(s)"
                + (f", {len(self.bursts)} bursts" if len(self.bursts) > 1 else ""))


@dataclass
class Deinterleaved:
    pdws: list            # sorted copies, `emitter` set (−1 unassigned)
    emitters: list
    unassigned: list      # indices into pdws
    notes: list = field(default_factory=list)
    config: dict = field(default_factory=dict)

    def train(self, emitter_id: int, completed: bool = True) -> list[dict]:
        e = self.emitters[emitter_id]
        out = [self.pdws[i] for i in e.received]
        if completed:
            out = out + list(e.inferred)
        return sorted(out, key=lambda p: p["toa_s"])

    def completed(self) -> list[dict]:
        """Every pulse, received and inferred, in time order."""
        out = list(self.pdws)
        for e in self.emitters:
            out += e.inferred
        return sorted(out, key=lambda p: p["toa_s"])

    def lines(self) -> list[str]:
        out = [e.words() for e in self.emitters]
        out.append(f"{len(self.unassigned)} pulse(s) not assigned to any emitter.")
        n_inf = sum(len(e.inferred) for e in self.emitters)
        if n_inf:
            out.append(f"{n_inf} pulse(s) are INFERRED — "
                       + _prov.TIER_WORDS["inferred"]
                       + " They carry no samples and no amplitude.")
        return out + list(self.notes)


def _fit_sub(idx, a0: float, b0: float, t) -> dict:
    """A sub-train: its pulses on one grid, one pulse per slot (the one
    nearest the grid), slots renumbered from 0, the grid refitted twice."""
    idx = np.unique(np.asarray(idx, dtype=np.int64))
    a, b = float(a0), float(b0)
    for _ in range(2):
        slots = np.round((t[idx] - a) / b).astype(np.int64)
        order = np.argsort(np.abs(t[idx] - (a + b * slots)))
        _, first = np.unique(slots[order], return_index=True)
        keep = np.sort(order[first])
        idx, slots = idx[keep], slots[keep]
        o = np.argsort(slots)
        idx, slots = idx[o], slots[o]
        if idx.size >= 2:
            a, b, sig = _fit(slots, t[idx])
    slots = np.round((t[idx] - a) / b).astype(np.int64)
    a = a + b * slots[0]
    slots = slots - slots[0]
    res = t[idx] - (a + b * slots)
    sig = float(np.std(res, ddof=min(2, max(0, idx.size - 1)))) if idx.size > 2 else 0.0
    return {"idx": idx, "slots": slots, "a": float(a), "b": float(b),
            "sigma": sig, "t0": float(t[idx[0]]), "t1": float(t[idx[-1]])}


def _phase_on(tt: np.ndarray, a: float, b: float) -> tuple[float, float]:
    """(mean phase in [0, 1), resultant length) of times on a grid."""
    if tt.size == 0:
        return 0.0, 0.0
    z = np.mean(np.exp(2j * np.pi * (((tt - a) / b) % 1.0)))
    return float((np.angle(z) / (2 * np.pi)) % 1.0), float(abs(z))


def _on_grid(tt: np.ndarray, q: dict, reach: float, tol: float) -> bool:
    """Do these times continue sub-train q — inside its extent widened by
    `reach`, at least 80 % of them within `tol` of q's grid?"""
    near = tt[(tt >= q["t0"] - reach) & (tt <= q["t1"] + reach)]
    if near.size < max(2, int(0.8 * tt.size)):
        return False
    res = near - (q["a"] + q["b"] * np.round((near - q["a"]) / q["b"]))
    return bool(np.mean(np.abs(res) <= tol) >= 0.8)


def _params_ok(fs, ws, g, cfg) -> bool:
    if np.isfinite(fs) and np.isfinite(g["f"]) and abs(fs - g["f"]) > cfg["freq_tol_hz"]:
        return False
    if np.isfinite(ws) and np.isfinite(g["w"]) and \
            abs(ws - g["w"]) > cfg["width_tol"] * max(g["w"], 1e-12):
        return False
    return True


def _continues(tt, g, cfg) -> dict | None:
    """The sub-train of group g that times `tt` continue, if any."""
    M = cfg["max_missing"]
    for q in g["subs"]:
        tol_q = max(cfg["rel_tol"] * q["b"], 4 * q["sigma"], cfg["abs_tol_s"])
        if _on_grid(tt, q, (M + 1.5) * q["b"], tol_q):
            return q
    return None


def _merge_into_groups(seqs: list[_Seq], t, freq, width, cfg) -> list[list[dict]]:
    """Burst groups, built so that fragments join up.

    Sequences are taken by PERIOD FAMILY, shortest period first (so a train
    found at a multiple of its PRI meets the fundamental's group and joins
    it), and within a family IN TIME ORDER (so a train broken by fades grows
    left to right). A sequence whose pulses sit on an existing sub-train's
    grid — overlapping it or within `max_missing` PRIs of its ends, within
    the jitter — CONTINUES it. Otherwise, one of the same period that
    overlaps a group in time and is phase-locked to it is a new POSITION of
    a stagger. Same period, overlapping, not locked: another emitter. A last
    pass joins consecutive groups that continue one another."""
    rel = cfg["rel_tol"]
    M = cfg["max_missing"]
    fams: list[list[_Seq]] = []
    for s in sorted(seqs, key=lambda q: q.b):
        if fams and s.b <= fams[-1][-1].b * (1 + 2 * rel):
            fams[-1].append(s)
        else:
            fams.append([s])
    groups: list[dict] = []
    for fi, fam in enumerate(fams):
        for s in sorted(fam, key=lambda q: q.t0):
            fs = float(np.nanmedian(freq[s.idx])) if np.isfinite(freq[s.idx]).any() else math.nan
            ws = float(np.nanmedian(width[s.idx])) if np.isfinite(width[s.idx]).any() else math.nan
            tt = t[s.idx]
            placed = False
            for g in reversed(groups):            # the most recent first
                b = g["subs"][0]["b"]
                ratio = s.b / b
                harmonic = round(ratio)
                same = abs(ratio - 1.0) <= rel
                multiple = (2 <= harmonic <= M + 1
                            and abs(ratio - harmonic) <= rel * harmonic)
                if not (same or multiple) or not _params_ok(fs, ws, g, cfg):
                    continue
                q = _continues(tt, g, cfg)
                if q is not None:
                    q.update(_fit_sub(np.concatenate([q["idx"], s.idx]),
                                      q["a"], q["b"], t))
                    placed = True
                    break
                if not same:
                    continue
                g_t0 = min(x["t0"] for x in g["subs"])
                g_t1 = max(x["t1"] for x in g["subs"])
                lo, hi = max(s.t0, g_t0), min(s.t1, g_t1)
                if hi - lo < 2 * b:
                    continue
                inside = tt[(tt >= lo - b) & (tt <= hi + b)]
                mu, R = _phase_on(inside, g["subs"][0]["a"], b)
                if inside.size >= 3 and R >= 0.9:
                    # at the phase of an existing sub-train (within the
                    # jitter): the same position, not a new one
                    home = None
                    for x in g["subs"]:
                        mx, _ = _phase_on(t[x["idx"]], g["subs"][0]["a"], b)
                        tol_ph = max(rel, 4 * max(x["sigma"], s.sigma) / b)
                        if abs(((mu - mx) + 0.5) % 1.0 - 0.5) <= tol_ph:
                            home = x
                            break
                    if home is not None:
                        home.update(_fit_sub(np.concatenate([home["idx"], s.idx]),
                                             home["a"], home["b"], t))
                    else:
                        g["subs"].append(_fit_sub(s.idx, s.a, s.b, t))
                    placed = True
                    break
            if not placed:
                groups.append({"subs": [_fit_sub(s.idx, s.a, s.b, t)],
                               "f": fs, "w": ws, "fam": fi})
    # join consecutive groups of one family that continue one another
    groups.sort(key=lambda g: min(x["t0"] for x in g["subs"]))
    joined: list[dict] = []
    for g in groups:
        home = None
        for h in reversed(joined):
            if h["fam"] != g["fam"] or len(h["subs"]) != len(g["subs"]):
                continue
            if not _params_ok(g["f"], g["w"], h, cfg):
                continue
            pairs = []
            for x in g["subs"]:
                q = _continues(t[x["idx"]], h, cfg)
                if q is None or any(q is p for p, _ in pairs):
                    break
                pairs.append((q, x))
            if len(pairs) == len(g["subs"]):
                home = (h, pairs)
                break
        if home is None:
            joined.append(g)
        else:
            for q, x in home[1]:
                q.update(_fit_sub(np.concatenate([q["idx"], x["idx"]]),
                                  q["a"], q["b"], t))
    return [g["subs"] for g in joined]


def _describe_group(subs: list[dict], t, cfg) -> dict:
    """Period, kind and levels of one burst group, and — when the levels
    repeat — the sub-trains to combine. The search can find a train at a
    multiple of its true frame: a 1/2 ms stagger seen at 6 ms reads
    2/1/2/1, and equal levels mean a constant PRI seen at m times itself.
    The smallest repeating unit is the truth; `merge_sets` says which
    sub-trains become one at the reduced period."""
    F = float(np.median([s["b"] for s in subs]))
    m = len(subs)
    if m == 1:
        s = subs[0]
        jit = 100.0 * s["sigma"] / F if F > 0 else 0.0
        kind = "jittered" if jit > cfg["jitter_pct"] else "constant"
        return {"kind": kind, "pri": F, "levels": [], "jitter": jit,
                "merge_sets": None}
    ref = subs[0]
    ph = np.array([_phase_on(t[s["idx"]], ref["a"], F)[0] for s in subs])
    order = np.argsort(ph)
    ps = ph[order]
    levels = np.diff(np.concatenate([ps, [ps[0] + 1.0]])) * F
    sig_med = float(np.median([s["sigma"] for s in subs]))
    jit = 100.0 * sig_med / (F / m)
    # levels that differ by less than the train's own jitter are equal
    tol = max(cfg["rel_tol"] * F, 2.0 * sig_med)
    # two "positions" closer than the train's own jitter are one position
    min_gap = max(cfg["rel_tol"] * F, 4.0 * sig_med)
    close = levels < min_gap
    if close.any() and not close.all():
        start = int(np.flatnonzero(~close)[0]) + 1     # a position after a real gap
        sets, cur = [], []
        for k in range(m):
            i = (start + k) % m
            cur.append(int(order[i]))
            if not close[i]:
                sets.append(cur)
                cur = []
        if cur:
            sets.append(cur)
        return {"kind": "", "pri": F, "levels": [], "jitter": jit,
                "merge_sets": sets}
    for unit in range(1, m):
        if m % unit:
            continue
        rows = levels.reshape(m // unit, unit)
        if np.all(np.abs(rows - rows[0]) <= tol):
            sets = [[int(order[i + k * unit]) for k in range(m // unit)]
                    for i in range(unit)]
            return {"kind": "", "pri": F * unit / m, "levels": [],
                    "jitter": jit, "merge_sets": sets}
    # the levels in time order, starting at the position of the first pulse
    first = int(np.argmin([t[s["idx"][0]] for s in subs]))
    levels = np.roll(levels, -int(np.flatnonzero(order == first)[0]))
    return {"kind": f"staggered ({m}-level)", "pri": F,
            "levels": [float(v) for v in levels], "jitter": jit,
            "merge_sets": None}


def deinterleave(pdws, **kw) -> Deinterleaved:
    """Split PDWs (ATK's shape) into per-emitter trains and complete each.

    Keywords override `DEFAULTS` (rel_tol, jitter_tol, abs_tol_s,
    max_missing, min_pulses, max_level, x, k, freq_tol_hz, width_tol,
    jitter_pct). `complete=False` stops before inferring anything."""
    cfg = dict(DEFAULTS)
    do_complete = bool(kw.pop("complete", True))
    unknown = set(kw) - set(cfg)
    if unknown:
        raise ValueError(f"unknown setting(s): {', '.join(sorted(unknown))}")
    cfg.update(kw)
    P = as_pdws(pdws)
    for p in P:
        p["emitter"] = -1
        p["inferred"] = False
        p["tier"] = "measured"
    t = toas(P)
    n = t.size
    freq = np.array([p["freq_hz"] for p in P], dtype=np.float64)
    width = np.array([p["width_s"] for p in P], dtype=np.float64)
    notes: list[str] = []
    if n < cfg["min_pulses"]:
        return Deinterleaved(P, [], list(range(n)),
                             [f"{n} pulse(s) — too few for an emitter "
                              f"(at least {cfg['min_pulses']})."], cfg)
    lab = _clusters(freq, width, cfg)
    avail = np.ones(n, dtype=bool)
    seqs: list[_Seq] = []
    for g in np.unique(lab):
        pool = np.flatnonzero(lab == g)
        if pool.size >= cfg["min_pulses"]:
            seqs += _extract(t, pool, avail, cfg, int(g))
    pool = np.flatnonzero(avail)
    if pool.size >= cfg["min_pulses"]:
        extra = _extract(t, pool, avail, cfg, -1)
        if extra:
            notes.append(f"{len(extra)} sequence(s) found by PRI alone — their "
                         "carrier or width varies from pulse to pulse.")
        seqs += extra
    groups = _merge_into_groups(seqs, t, freq, width, cfg)

    # burst groups -> descriptions; then link bursts of one emitter
    descr = []
    for st in groups:
        d = _describe_group(st, t, cfg)
        for _ in range(4):
            if not d["merge_sets"]:
                break
            # positions that coincide, or levels that repeat: combine to the
            # smallest set of distinct positions and re-describe
            st = [_fit_sub(np.concatenate([st[j]["idx"] for j in ms]),
                           st[ms[0]]["a"], d["pri"], t) for ms in d["merge_sets"]]
            d = _describe_group(st, t, cfg)
        idx_all = np.unique(np.concatenate([s["idx"] for s in st]))
        descr.append({"subs": st, "d": d, "t0": float(t[idx_all[0]]),
                      "t1": float(t[idx_all[-1]]), "idx": idx_all,
                      "f": float(np.nanmedian(freq[idx_all])) if np.isfinite(freq[idx_all]).any() else math.nan,
                      "w": float(np.nanmedian(width[idx_all])) if np.isfinite(width[idx_all]).any() else math.nan})
    descr.sort(key=lambda g: g["t0"])
    emit_groups: list[list[dict]] = []
    for g in descr:
        home = None
        for eg in emit_groups:
            last = eg[-1]
            same_kind = (g["d"]["kind"].split(" ")[0] == last["d"]["kind"].split(" ")[0])
            if not same_kind or abs(g["d"]["pri"] - last["d"]["pri"]) > cfg["rel_tol"] * last["d"]["pri"]:
                continue
            if g["t0"] <= last["t1"]:
                continue                      # overlapping in time: another emitter
            if np.isfinite(g["f"]) and np.isfinite(last["f"]) and abs(g["f"] - last["f"]) > cfg["freq_tol_hz"]:
                continue
            if np.isfinite(g["w"]) and np.isfinite(last["w"]) and \
                    abs(g["w"] - last["w"]) > cfg["width_tol"] * max(last["w"], 1e-12):
                continue
            home = eg
            break
        if home is None:
            emit_groups.append([g])
        else:
            home.append(g)

    emitters: list[Emitter] = []
    for eid, eg in enumerate(emit_groups):
        idx_all = np.unique(np.concatenate([g["idx"] for g in eg]))
        d0 = eg[0]["d"]
        pri = float(np.median([g["d"]["pri"] for g in eg]))
        levels = d0["levels"]
        e = Emitter(eid, d0["kind"], pri, list(levels),
                    float(np.median([g["d"]["jitter"] for g in eg])),
                    float(np.nanmedian(freq[idx_all])) if np.isfinite(freq[idx_all]).any() else math.nan,
                    float(np.nanmedian(width[idx_all])) if np.isfinite(width[idx_all]).any() else math.nan,
                    float(np.nanmedian([P[i]["chirp_hz"] for i in idx_all]))
                    if np.isfinite([P[i]["chirp_hz"] for i in idx_all]).any() else math.nan)
        for bi, g in enumerate(eg):
            e.bursts.append((g["t0"], g["t1"]))
            for s in g["subs"]:
                e.subtrains.append({"idx": s["idx"], "slots": s["slots"],
                                    "a": s["a"], "b": s["b"],
                                    "sigma": s["sigma"], "burst": bi})
        e.received = sorted(int(i) for i in idx_all)
        for i in e.received:
            P[i]["emitter"] = eid
        if len(eg) > 1:
            e.notes.append(f"heard in {len(eg)} separate stretches; nothing is "
                           "inferred across the gaps between them — each is "
                           f"longer than {cfg['max_missing']} PRIs (a scanning "
                           "beam, a pause, or a long fade).")
        emitters.append(e)

    emitters = _join_emitters(emitters, P, t, cfg)
    for e in emitters:
        for i in e.received:
            P[i]["emitter"] = e.id
    out = Deinterleaved(P, emitters, [], notes, cfg)
    if do_complete:
        _absorb_and_complete(out, t, cfg)
    out.unassigned = [i for i, p in enumerate(P) if p["emitter"] < 0]
    return out


def _rebuild(e: Emitter, idx, t, cfg) -> None:
    """A non-staggered emitter's sub-trains from scratch: its pulses split
    into stretches at gaps longer than `max_missing` PRIs, one grid each."""
    idx = np.unique(np.asarray(idx, dtype=np.int64))
    tt = t[idx]
    cut = np.flatnonzero(np.diff(tt) > (cfg["max_missing"] + 1.5) * e.pri_s) + 1
    e.subtrains, e.bursts = [], []
    for bi, piece in enumerate(np.split(idx, cut)):
        if piece.size == 0:
            continue
        st = _fit_sub(piece, float(t[piece[0]]), e.pri_s, t)
        st["burst"] = bi
        e.subtrains.append(st)
        e.bursts.append((float(t[piece[0]]), float(t[piece[-1]])))
    e.received = sorted(int(i) for i in idx)
    sig = [x["sigma"] for x in e.subtrains if x["idx"].size > 2]
    e.jitter_pct = 100.0 * float(np.median(sig)) / e.pri_s if sig else 0.0
    e.kind = "jittered" if e.jitter_pct > cfg["jitter_pct"] else "constant"


def _join_emitters(emitters: list, P, t, cfg) -> list:
    """Join non-staggered emitters of the same PRI whose pulses sit on one
    grid: fragments of one jittered train that overlap in time, which the
    burst grouping cannot see as one."""
    em = sorted(emitters, key=lambda e: -len(e.received))
    alive = [True] * len(em)
    for i, A in enumerate(em):
        if not alive[i] or A.kind.startswith("staggered"):
            continue
        for j in range(i + 1, len(em)):
            B = em[j]
            if not alive[j] or B.kind.startswith("staggered"):
                continue
            if abs(A.pri_s - B.pri_s) > cfg["rel_tol"] * A.pri_s:
                continue
            if np.isfinite(A.freq_hz) and np.isfinite(B.freq_hz) and \
                    abs(A.freq_hz - B.freq_hz) > cfg["freq_tol_hz"]:
                continue
            if np.isfinite(A.width_s) and np.isfinite(B.width_s) and \
                    abs(A.width_s - B.width_s) > cfg["width_tol"] * max(A.width_s, 1e-12):
                continue
            ia = np.array(A.received, dtype=np.int64)
            k = np.round((t[ia] - t[ia[0]]) / A.pri_s)
            a, b, sig = _fit(k, t[ia])
            if b <= 0:
                continue
            tb = t[np.array(B.received, dtype=np.int64)]
            res = tb - (a + b * np.round((tb - a) / b))
            tol = max(cfg["rel_tol"] * b, 4 * sig, cfg["abs_tol_s"])
            if np.mean(np.abs(res) <= tol) >= 0.8:
                _rebuild(A, np.concatenate([ia, np.array(B.received)]), t, cfg)
                A.notes.append("joined with a fragment of the same train found "
                               "separately (same PRI, same grid).")
                alive[j] = False
    out = [e for e, ok in zip(em, alive) if ok]
    out.sort(key=lambda e: t[e.received[0]] if e.received else 0.0)
    for nid, e in enumerate(out):
        e.id = nid
        if len(e.bursts) > 1 and not any("separate stretches" in n for n in e.notes):
            e.notes.append(f"heard in {len(e.bursts)} separate stretches; nothing "
                           "is inferred across the gaps between them — each is "
                           f"longer than {cfg['max_missing']} PRIs (a scanning "
                           "beam, a pause, or a long fade).")
    return out


def _absorb_and_complete(D: Deinterleaved, t, cfg) -> None:
    P = D.pdws
    owner = np.array([p["emitter"] for p in P], dtype=np.int64)
    widths = np.array([p["width_s"] if np.isfinite(p["width_s"]) else 0.0 for p in P])
    fq = np.array([p["freq_hz"] for p in P], dtype=np.float64)
    wd = np.array([p["width_s"] for p in P], dtype=np.float64)

    def fits(j, e) -> bool:
        """An unassigned pulse may join an emitter only if its carrier and
        width agree with the emitter's (a noise pulse on the grid does not)."""
        if np.isfinite(fq[j]) and np.isfinite(e.freq_hz) and \
                abs(fq[j] - e.freq_hz) > cfg["freq_tol_hz"]:
            return False
        if np.isfinite(wd[j]) and np.isfinite(e.width_s) and \
                abs(wd[j] - e.width_s) > cfg["width_tol"] * max(e.width_s, 1e-12):
            return False
        return True

    for e in D.emitters:
        new_inferred = []
        absorbed = 0
        for st in e.subtrains:
            idx = list(st["idx"])
            slots = list(st["slots"])
            a, b, sig = st["a"], st["b"], st["sigma"]
            tol = max(cfg["rel_tol"] * b, cfg["abs_tol_s"], 3 * sig)
            out_idx, out_slots = [idx[0]], [slots[0]]
            for (i0, s0), (i1, s1) in zip(zip(idx[:-1], slots[:-1]), zip(idx[1:], slots[1:])):
                k = int(s1 - s0)
                if k >= 2:
                    t0, t1 = t[i0], t[i1]
                    if k - 1 > cfg["max_missing"]:
                        e.notes.append(f"a gap of {k - 1} PRIs at "
                                       f"{t0:.6f} s is longer than "
                                       f"{cfg['max_missing']} — not filled.")
                    elif abs((t1 - t0) - k * b) > tol * math.sqrt(k):
                        e.notes.append(f"the gap at {t0:.6f} s is not a whole "
                                       "number of PRIs — not filled.")
                    else:
                        for m in range(1, k):
                            f = m / k
                            tm = t0 + f * (t1 - t0)
                            # a real, unassigned pulse on the grid beats an inferred one
                            lo = np.searchsorted(t, tm - tol, "left")
                            hi = np.searchsorted(t, tm + tol, "right")
                            cands = [j for j in range(lo, hi)
                                     if owner[j] < 0 and fits(j, e)]
                            if cands:
                                j = min(cands, key=lambda q: abs(t[q] - tm))
                                owner[j] = e.id
                                P[j]["emitter"] = e.id
                                out_idx.append(j)
                                out_slots.append(s0 + m)
                                absorbed += 1
                                continue
                            sigma = max(sig, cfg["abs_tol_s"] / 3.0) * math.sqrt(1.0 + f * f + (1 - f) ** 2)
                            masked = -1
                            w_e = e.width_s if np.isfinite(e.width_s) else 0.0
                            lo2 = np.searchsorted(t, tm - max(widths.max(initial=0.0), w_e) - tol, "left")
                            hi2 = np.searchsorted(t, tm + w_e + tol, "right")
                            for j in range(lo2, hi2):
                                if owner[j] != e.id and t[j] - tol <= tm <= t[j] + widths[j] + tol:
                                    masked = j
                                    break
                            new_inferred.append(_inferred_pdw(e, tm, sigma, masked, P))
                out_idx.append(i1)
                out_slots.append(s1)
            o = np.argsort(out_slots)
            st["idx"] = np.array(out_idx, dtype=np.int64)[o]
            st["slots"] = np.array(out_slots, dtype=np.int64)[o]
        e.received = sorted(set(e.received) | {int(i) for st in e.subtrains for i in st["idx"]})
        e.inferred = sorted(new_inferred, key=lambda p: p["toa_s"])
        if absorbed:
            e.notes.append(f"{absorbed} unassigned received pulse(s) sat on "
                           "this emitter's grid and were taken into it before "
                           "anything was inferred.")
        if e.inferred:
            n_col = sum(1 for p in e.inferred if p["reason"] == "collision")
            e.notes.append(f"{len(e.inferred)} pulse(s) INFERRED (k·PRI gaps)"
                           + (f", {n_col} of them under another emitter's "
                              "pulse (a collision)" if n_col else "") + ".")


def _inferred_pdw(e: Emitter, toa: float, sigma: float, masked: int, P) -> dict:
    reason = "collision" if masked >= 0 else "dropout"
    words = ("hidden under a pulse of another emitter that arrived at the same "
             "moment" if masked >= 0 else
             "missing where this emitter's PRI says a pulse was due (fading or "
             "a dropout)")
    return {"start": -1, "end": -1, "toa_s": float(toa),
            "width_s": e.width_s, "amplitude": math.nan, "amplitude_db": math.nan,
            "freq_hz": e.freq_hz, "chirp_hz": e.chirp_hz,
            "emitter": e.id, "inferred": True, "tier": "inferred",
            "method": METHOD, "sigma_toa_s": float(sigma), "reason": reason,
            "masked_by": int(masked),
            "masked_by_emitter": int(P[masked]["emitter"]) if masked >= 0 else -1,
            "note": "INFERRED pulse — " + words + "; not received."}


# ---------------------------------------------------------------------------
# Analysis of one train
# ---------------------------------------------------------------------------
def first_difference_pri(train) -> dict:
    """The naive reading a dropout wrecks: mean and median of first
    differences of the received TOAs. Reported beside the completed one so
    the effect of completion is a number."""
    t = toas(train)
    if t.size < 2:
        return {"mean_s": math.nan, "median_s": math.nan}
    d = np.diff(t)
    return {"mean_s": float(np.mean(d)), "median_s": float(np.median(d)),
            "std_s": float(np.std(d))}


def analyse_train(train, **kw) -> dict:
    """PRI analysis of ONE emitter's train (received, or completed): kind,
    PRI or frame and levels, jitter, the SDIF / CDIF / PRI-transform best
    candidates and the naive first-difference numbers — for received only
    and, when inferred pulses are present, for the completed train. Read
    the jitter from the RECEIVED view: inferred pulses sit on the grid, so
    the completed view understates it."""
    P = as_pdws(train)
    rec = [p for p in P if not p.get("inferred")]
    out = {"n_received": len(rec), "n_inferred": len(P) - len(rec)}

    def one(pp):
        t = toas(pp)
        D = deinterleave(pp, complete=False, freq_tol_hz=float("inf"),
                         width_tol=float("inf"), **kw)
        e = max(D.emitters, key=lambda q: len(q.received)) if D.emitters else None
        sd = sdif(t)
        best_sd = next((L.peaks[0] for L in sd if L.peaks), math.nan)
        cd = cdif(t)
        best_cd = next((L.peaks[0] for L in cd if L.peaks), math.nan)
        taus, mag = pri_transform(t)
        best_pt = float(taus[int(np.argmax(mag))]) if mag.size else math.nan
        fd = first_difference_pri(pp)
        return {"kind": e.kind if e else "no emitter found",
                "pri_s": e.pri_s if e else math.nan,
                "levels_s": e.levels_s if e else [],
                "jitter_pct": e.jitter_pct if e else math.nan,
                "sdif_first_peak_s": best_sd, "cdif_first_peak_s": best_cd,
                "pri_transform_peak_s": best_pt,
                "first_difference": fd}
    out["received"] = one(rec)
    if len(rec) != len(P):
        out["completed"] = one(P)
    return out
