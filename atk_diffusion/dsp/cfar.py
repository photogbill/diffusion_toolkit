# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The energy proposer: cell-averaging CFAR on the tile, boxes from what
clears it (DETECTION_DESIGN §3, build step 2; ARCHITECTURE §4.1).

*"Cell-averaging CFAR on the per-frame PSD, in dB above floor, with a
per-profile false-alarm rate. Cheap, classical, explains itself."* Always on,
needs no training, and the baseline every learned proposer is scored against
(plan §7: classical baselines first).

THE THRESHOLD IS DERIVED, NOT TUNED — ported from Bill's own
`atk/core/siga/detect.py` (`cfar`): for N training cells of exponentially
distributed power, α = N·(Pfa^(−1/N) − 1). *"That is a derivation, not a
knob, and it is the reason a CFAR result can be compared against [another
detector] at a MATCHED false-alarm rate."* Two facts of THIS front end change
the arithmetic, and both are handled exactly rather than hoped away:

1. A tile row is the MAX of `pool` frames (dsp.stft), not one frame. The
   max of P exponentials is a sum of independent exponentials with rates
   1..P (Rényi), so the false-alarm probability of "cell > α·mean(training)"
   has a closed form:

       Pfa(α) = Σ_{k=1..P} (−1)^(k+1)·C(P,k)·Π_{i=1..P} (i / (i + kα/N))^N

   which for P = 1 is (1 + α/N)^(−N) — the formula above. α is solved from
   it numerically.
2. The window makes neighbouring bins correlated (Hann: power correlation
   0.44 at one bin, 0.03 at two), so N training cells carry less than N
   cells' worth of evidence. N is replaced by the EQUIVALENT number of
   independent cells, N·(1 + (2/N)·Σ_pairs ρ²)^(−1), computed from the
   actual window's spectrum for the actual training offsets.

Measured per cell on noise through the finished front end (tiles in dB above
the estimated floor, 2.4 MS/s, Hann 1024, 5 frames a row; the test repeats
it): Pfa stated 1e-3 -> 0.92e-3 (2D: 0.92e-3); 1e-4 -> 0.87e-4 (2D:
0.95e-4). With N used naively instead of N_eff, measured 1.2e-3 and 1.4e-4.

WIDE SIGNALS. Bill's caveat, from the same file: *"CFAR compares each bin
against its own NEIGHBOURS, so a signal wider than the training window is
invisible to it: the training cells sit inside the signal and the threshold
rises with it."* The tile is already in units of the measured floor, so a
second, FLOOR-REFERENCED test runs beside the CFAR: a cell is also detected
when it exceeds the floor by the max-of-P threshold for the same Pfa plus a
margin (default 1 dB) for the floor's own uncertainty. With the margin it
adds about 1/16 of the CFAR's false alarms; it stands down for any tile
whose floor just moved (`Tile.floor_ok`, a gain change) rather than flood
the waterfall.

WEAK SIGNALS THAT STAY PUT. A continuous signal a few dB above the floor
clears a per-row threshold only now and then, and its box shatters into
dozens (measured: 81 boxes in 3 s for a 4 dB carrier; 23 at 6 dB). The
INTEGRATED layer averages the mean-pooled power over blocks of
`integrate_rows` rows (n = rows x frames per cell) and tests the blocks the
same two ways — CA-CFAR with α from the F distribution (a mean of n
exponentials over a mean of N·n is F(2n, 2nN)) and the floor test with the
Gamma quantile — at `pfa x integrate_pfa_factor` (1e-4: every block false
alarm marks a whole block and becomes a box, so it must be far rarer; at
1e-3 it cost ~29 false boxes an hour at the RTL geometry, at 1e-4 about 3).
Measured on 1.9 million noise blocks the F-derived α holds its stated rate
down to 1e-6 (1.1e-6 measured). Each detected component is then trimmed to
the rows its own power occupies — its row profile across its columns must
clear a level a noise row reaches once in a thousand — so a strong burst
keeps its own edges, not a block's. The same 4 dB carrier then reads as one
box per tile.

BOXES. Cells above threshold are grouped into connected components over
(time x frequency) with scipy.ndimage, after bridging gaps of `gap_rows`
rows and `gap_bins` bins (an FSK burst dips between its tones; a fade costs a
row) — the bridge decides what is connected, the box is drawn tight around
the cells that actually cleared. Components smaller than `min_cells` (4),
shorter than `min_duration_s` or narrower than `min_bins` are dropped:
measured on noise at the RTL geometry, single and paired false cells are
~2.8 a tile, and three scattered ones bridged into one box about 70 times an
hour — at 4 cells that is gone, while a real signal of 2 rows x 2 bins
already has 4. Each box becomes a `Detection`:
sources ("energy",), family "unknown", absolute t/f from the tile, `snr_db`
from the MEAN-pooled power in the box (dB above the floor), the peak
max-pooled level in `measurements`. More than `max_boxes` in one tile means
the threshold is in the noise, not on signals: the strongest are kept and the
`info` dict says so (Bill's pulse.py lesson — the 2026-09-29 freeze).

LIMITS. A CA-CFAR's threshold rises beside a strong signal (its training
cells see the skirt): a weak signal within `guard + train` bins of a strong
one is harder to see — the 2D option and the floor-referenced test both
help, neither removes it. Bins at the span's edges have fewer training cells
and so a higher α (honest, not hidden). With hop < fft_size consecutive frames
overlap and the effective P is interpolated — approximate, and said so.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from atk_diffusion.detect.boxes import Detection

#: Correlations below this are treated as zero.
_RHO_EPS = 1e-6


# ---------------------------------------------------------------------------
# The threshold, derived
# ---------------------------------------------------------------------------
def pfa_of_alpha(alpha: float, n_eff: float, pool: int = 1) -> float:
    """P(cell > α·mean of N training cells) for cells that are the max of
    `pool` independent exponentials, N = `n_eff` (real-valued allowed)."""
    p = max(1, int(pool))
    a = float(alpha) / float(n_eff)
    i = np.arange(1, p + 1, dtype=np.float64)
    tot = 0.0
    for k in range(1, p + 1):
        logprod = float(n_eff) * float(np.sum(np.log(i / (i + k * a))))
        tot += (-1.0) ** (k + 1) * math.comb(p, k) * math.exp(logprod)
    return float(tot)


@lru_cache(maxsize=4096)
def _alpha_int(pfa: float, n_eff: float, pool: int) -> float:
    if pool == 1:
        return float(n_eff * (pfa ** (-1.0 / n_eff) - 1.0))
    from scipy.optimize import brentq
    lo, hi = 1e-6, 1.0
    while pfa_of_alpha(hi, n_eff, pool) > pfa:
        hi *= 2.0
        if hi > 1e9:
            raise ValueError("no CFAR threshold reaches that false-alarm rate")
    return float(brentq(lambda a: math.log(max(pfa_of_alpha(a, n_eff, pool),
                                                  1e-300)) - math.log(pfa),
                        lo, hi, xtol=1e-10, rtol=1e-12))


def cfar_alpha(pfa: float, n_eff: float, pool: float = 1.0) -> float:
    """The multiplier α on the training mean for false-alarm rate `pfa`.
    A non-integer `pool` (overlapping frames) interpolates log α between
    the neighbouring integers."""
    pfa = float(pfa)
    if not 0.0 < pfa < 1.0:
        raise ValueError(f"a false-alarm rate is between 0 and 1, got {pfa!r}")
    if n_eff <= 0:
        raise ValueError("a CFAR needs at least one training cell")
    n_eff = round(float(n_eff), 6)
    p = max(1.0, float(pool))
    lo = int(math.floor(p))
    if p - lo < 1e-9:
        return _alpha_int(pfa, n_eff, lo)
    a0, a1 = _alpha_int(pfa, n_eff, lo), _alpha_int(pfa, n_eff, lo + 1)
    w = p - lo
    return float(math.exp((1 - w) * math.log(a0) + w * math.log(a1)))


def floor_threshold(pfa: float, pool: float = 1.0) -> float:
    """Linear threshold, in floor units, that a max-of-`pool` noise cell
    exceeds with probability `pfa` when the floor is known exactly:
    −ln(1 − (1 − Pfa)^(1/P))."""
    p = max(1.0, float(pool))
    return float(-math.log(-math.expm1(math.log1p(-float(pfa)) / p)))


@lru_cache(maxsize=4096)
def mean_cell_alpha(pfa: float, n_eff: float, n_avg: float) -> float:
    """α for cells that are the MEAN of `n_avg` independent exponentials,
    trained on `n_eff` such cells: CUT / training mean ~ F(2n, 2n·N), so
    α = F⁻¹(1 − Pfa; 2n, 2n·N) — exact for independent cells."""
    from scipy.stats import f as _f
    n = max(1.0, float(n_avg))
    return float(_f.isf(float(pfa), 2.0 * n, 2.0 * n * max(1e-6, float(n_eff))))


def mean_floor_threshold(pfa: float, n_avg: float) -> float:
    """Known-floor threshold for a mean of `n_avg` exponentials (in floor
    units): the Gamma(n, 1/n) upper quantile."""
    from scipy.special import gammainccinv
    n = max(1.0, float(n_avg))
    return float(gammainccinv(n, float(pfa)) / n)


# ---------------------------------------------------------------------------
# What the window does to independence
# ---------------------------------------------------------------------------
@lru_cache(maxsize=64)
def window_stats(window: str, fft_size: int, hop: int, pool: int) -> tuple:
    """(bin_rho2, frame_rho2, pool_eff) for a window:

    bin_rho2[m-1]   power correlation of bins m apart in one frame, white
                    noise: |Σ w²·e^(−j2πmn/N)|² / (Σw²)²
    frame_rho2[l-1] power correlation of one bin in frames l apart:
                    (Σ w[n]·w[n+l·hop] / Σw²)²  (zero when hop >= fft)
    pool_eff        the equivalent number of independent frames in a row
    """
    from atk_diffusion.dsp.stft import window_array
    w = window_array(window, int(fft_size)).astype(np.float64)
    n = w.size
    w2 = w * w
    s2 = float(w2.sum())
    idx = np.arange(n)
    bin_r = []
    for m in range(1, 33):
        r = abs(complex(np.sum(w2 * np.exp(-2j * np.pi * m * idx / n)))) / s2
        r2 = r * r
        if r2 < _RHO_EPS:
            break
        bin_r.append(r2)
    frame_r = []
    for lag in range(1, max(1, int(pool))):
        d = lag * int(hop)
        if d >= n:
            break
        r = float(np.sum(w[:n - d] * w[d:])) / s2
        if r * r < _RHO_EPS:
            break
        frame_r.append(r * r)
    p = max(1, int(pool))
    s = sum((p - lag) * r for lag, r in enumerate(frame_r, start=1))
    pool_eff = p / (1.0 + 2.0 * s / p) if p > 1 else 1.0
    return tuple(bin_r), tuple(frame_r), float(pool_eff)


def _pairs_sum(offsets: np.ndarray, bin_rho2: tuple) -> float:
    """Σ over pairs of training cells in one row of the power correlation
    at their separation."""
    if not len(offsets) or not bin_rho2:
        return 0.0
    occ = np.zeros(int(offsets.max() - offsets.min()) + 1, dtype=bool)
    occ[offsets - offsets.min()] = True
    tot = 0.0
    for m, r in enumerate(bin_rho2, start=1):
        if m >= occ.size:
            break
        tot += r * float(np.sum(occ[:-m] & occ[m:]))
    return tot


def effective_cells(n: int, pairs: float) -> float:
    """N_eff = N / (1 + (2/N)·Σ_pairs ρ²): independent cells with the same
    variance of the mean as N correlated ones."""
    if n <= 0:
        return 0.0
    return float(n / (1.0 + 2.0 * pairs / n))


@lru_cache(maxsize=4096)
def _neff_1d(nl: int, nr: int, guard: int, bin_rho2: tuple) -> float:
    offs = np.r_[np.arange(-guard - nl, -guard), np.arange(guard + 1, guard + nr + 1)]
    return effective_cells(int(offs.size), _pairs_sum(offs, bin_rho2))


@lru_cache(maxsize=256)
def _neff_2d_interior(guard: int, train: int, guard_rows: int, train_rows: int,
                      bin_rho2: tuple) -> tuple[int, float]:
    full = np.arange(-guard - train, guard + train + 1)
    split = np.r_[np.arange(-guard - train, -guard), np.arange(guard + 1, guard + train + 1)]
    n_full_rows = 2 * train_rows
    n_split_rows = 2 * guard_rows + 1
    n = n_full_rows * full.size + n_split_rows * split.size
    pairs = n_full_rows * _pairs_sum(full, bin_rho2) + \
        n_split_rows * _pairs_sum(split, bin_rho2)
    return n, effective_cells(n, pairs)


# ---------------------------------------------------------------------------
# The CFAR
# ---------------------------------------------------------------------------
def ca_cfar(spec_db_above_floor, pfa: float, guard: int = 2, train: int = 16,
            pool: int = 1, window: str = "hann", fft_size: int | None = None,
            hop: int | None = None, two_d: bool = False, guard_rows: int = 1,
            train_rows: int = 4, return_threshold: bool = False,
            cell: str = "max", n_avg: float = 1.0, linear: bool = False):
    """Cell-averaging CFAR. `spec_db_above_floor` is [rows, bins] (a tile's
    spec, or a single PSD as [1, bins]); each row is the max of `pool`
    frames of a `fft_size`-point `window` STFT with `hop` (these set the
    derived α — see the module docstring). Along frequency per row by
    default; `two_d=True` trains on a (time x frequency) annulus instead.
    `cell="mean"` declares cells that are instead the MEAN of `n_avg`
    frames (the integrated layer; α from the F distribution); `linear`
    says the input is linear power in floor units rather than dB.

    Returns the boolean detection mask, or (mask, threshold_db) with
    `return_threshold` (threshold in the same dB-above-floor units)."""
    s = np.asarray(spec_db_above_floor, dtype=np.float64 if linear else np.float32)
    if s.ndim == 1:
        s = s[None, :]
    if s.ndim != 2:
        raise ValueError("CFAR runs on a [rows, bins] array")
    if cell not in ("max", "mean"):
        raise ValueError("cell is 'max' (max-pooled rows) or 'mean' (averaged cells)")
    rows, bins = s.shape
    guard, train = max(0, int(guard)), max(1, int(train))
    n_fft = int(fft_size) if fft_size else bins
    hop_ = int(hop) if hop else n_fft
    bin_r2, _fr, pool_eff = window_stats(str(window), n_fft, hop_, max(1, int(pool)))
    lin = s if linear else np.power(10.0, s.astype(np.float64) / 10.0)

    def alpha_of(n_eff: float) -> float:
        if cell == "max":
            return cfar_alpha(pfa, n_eff, pool_eff)
        return mean_cell_alpha(float(pfa), round(float(n_eff), 6), float(n_avg))
    if not two_d:
        c = np.zeros((rows, bins + 1), dtype=np.float64)
        np.cumsum(lin, axis=1, out=c[:, 1:])
        idx = np.arange(bins)
        l_lo = np.clip(idx - guard - train, 0, bins)
        l_hi = np.clip(idx - guard, 0, bins)
        r_lo = np.clip(idx + guard + 1, 0, bins)
        r_hi = np.clip(idx + guard + train + 1, 0, bins)
        nl, nr = l_hi - l_lo, r_hi - r_lo
        n = nl + nr
        sums = (c[:, l_hi] - c[:, l_lo]) + (c[:, r_hi] - c[:, r_lo])
        mean = sums / np.maximum(n, 1)[None, :]
        alpha = np.zeros(bins, dtype=np.float64)
        for a, b in set(zip(nl.tolist(), nr.tolist())):
            if a + b == 0:
                continue
            sel = (nl == a) & (nr == b)
            alpha[sel] = alpha_of(_neff_1d(a, b, guard, bin_r2))
        thr = mean * alpha[None, :]
        valid = (n > 0)[None, :]
    else:
        gr, tr = max(0, int(guard_rows)), max(0, int(train_rows))
        sat = np.zeros((rows + 1, bins + 1), dtype=np.float64)
        sat[1:, 1:] = lin.cumsum(axis=0).cumsum(axis=1)
        ri, bi = np.arange(rows), np.arange(bins)

        def rect(r0, r1, b0, b1):
            r0 = np.clip(r0, 0, rows)[:, None]
            r1 = np.clip(r1, 0, rows)[:, None]
            b0 = np.clip(b0, 0, bins)[None, :]
            b1 = np.clip(b1, 0, bins)[None, :]
            area = np.maximum(r1 - r0, 0) * np.maximum(b1 - b0, 0)
            ssum = sat[r1, b1] - sat[r0, b1] - sat[r1, b0] + sat[r0, b0]
            return ssum, area

        so, ao = rect(ri - gr - tr, ri + gr + tr + 1, bi - guard - train,
                      bi + guard + train + 1)
        sg, ag = rect(ri - gr, ri + gr + 1, bi - guard, bi + guard + 1)
        n = ao - ag
        mean = (so - sg) / np.maximum(n, 1)
        n_int, neff_int = _neff_2d_interior(guard, train, gr, tr, bin_r2)
        ratio = neff_int / n_int
        alpha = np.zeros_like(mean)
        for nn in np.unique(n):
            if nn <= 0:
                continue
            alpha[n == nn] = alpha_of(max(1.0, nn * ratio))
        thr = mean * alpha
        valid = n > 0
    mask = (lin > thr) & valid
    if return_threshold:
        thr_db = (10.0 * np.log10(np.maximum(thr, 1e-30))).astype(np.float32)
        return mask, thr_db
    return mask


def floor_mask(spec_db_above_floor, pfa: float, pool: float = 1.0,
               margin_db: float = 1.0) -> np.ndarray:
    """The floor-referenced test: cells above the known-floor threshold for
    `pfa` plus `margin_db` (module docstring, WIDE SIGNALS)."""
    t = floor_threshold(pfa, pool) * 10.0 ** (float(margin_db) / 10.0)
    t_db = 10.0 * math.log10(t)
    return np.asarray(spec_db_above_floor, dtype=np.float32) > t_db


def trim_components(mask: np.ndarray, mean_above: np.ndarray, pool_eff: float,
                    bin_rho2: tuple, gate_pfa: float = 1e-3) -> np.ndarray:
    """Trim each connected component of `mask` (rows x bins, the integrated
    layer's blocks spread back over their rows) to the rows its own power
    occupies: the component's ROW PROFILE — the mean power over its columns,
    row by row — must clear the level a noise row reaches with probability
    `gate_pfa` (a mean of pool x columns exponentials, the columns counted
    as independent only as far as the window allows). Rows from the first
    to the last that clear stay (a weak signal stays whole); the block's
    leftover rows are cut (a strong burst keeps its own edges). Averaging
    across the columns is what makes this clean: tested cell by cell, one
    noise cell anywhere across a wide burst's 44 columns stretched its box by
    up to a block (measured, before this)."""
    from scipy import ndimage as ndi
    m = np.asarray(mask, dtype=bool)
    out = np.zeros_like(m)
    if not m.any():
        return out
    lab, _n = ndi.label(m, structure=np.ones((3, 3), dtype=bool))
    infl = 1.0 + 2.0 * float(sum(bin_rho2))
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        cm = lab[sl] == i
        cols = cm.any(axis=0)
        prof = np.asarray(mean_above[sl], dtype=np.float64)[:, cols].mean(axis=1)
        n_eff = float(pool_eff) * max(1.0, float(cols.sum()) / infl)
        rows = np.flatnonzero(prof > mean_floor_threshold(gate_pfa, n_eff))
        if rows.size == 0:
            continue
        keep = cm.copy()
        keep[:rows[0]] = False
        keep[rows[-1] + 1:] = False
        out[sl] |= keep
    return out


def integrated_mask(tile, pfa: float, rows_per_block: int = 8,
                    guard: int = 2, train: int = 16, floor_test: bool = True,
                    floor_margin_db: float = 1.0, row_gate_pfa: float = 1e-3,
                    info: dict | None = None) -> np.ndarray:
    """The INTEGRATED layer: weak signals that stay put.

    A continuous signal 4–6 dB above the floor clears the per-row threshold
    only now and then, and its box shatters (measured: 81 boxes over 3 s for
    a 4 dB carrier). Averaging the MEAN-pooled power over `rows_per_block`
    rows integrates n = rows x frames-per-row frames per cell; those cells
    are CA-CFAR'd with α from the F distribution and floor-tested with the
    Gamma quantile — both derived for `pfa`, the same way as the per-row
    layer — and each detected component is trimmed to the rows its own power
    occupies (`trim_components`, at `row_gate_pfa`). Returns a [rows_valid,
    bins] mask (rows past the last whole block are False)."""
    lay = tile.layout
    rv = int(tile.rows_valid) if tile.rows_valid else tile.rows
    k = max(1, int(rows_per_block))
    nb = rv // k
    out = np.zeros((rv, tile.bins), dtype=bool)
    if nb < 1:
        return out
    m = tile.mean_above[:nb * k].astype(np.float64)
    blocks = m.reshape(nb, k, tile.bins).mean(axis=1)
    bin_r2, _f, pool_eff = window_stats(lay.window, lay.fft_size, lay.hop, lay.pool)
    n = k * pool_eff
    bm = ca_cfar(blocks, pfa, guard=guard, train=train, pool=lay.pool,
                 window=lay.window, fft_size=lay.fft_size, hop=lay.hop,
                 cell="mean", n_avg=n, linear=True)
    if floor_test and tile.floor_ok:
        t = mean_floor_threshold(pfa, n) * 10.0 ** (float(floor_margin_db) / 10.0)
        bm |= blocks > t
    if bm.any():
        out[:nb * k] = trim_components(np.repeat(bm, k, axis=0), m, pool_eff,
                                       bin_r2, row_gate_pfa)
    if info is not None:
        info["blocks_detected"] = int(bm.sum())
        info["cells_integrated"] = int(out.sum())
    return out


# ---------------------------------------------------------------------------
# Boxes
# ---------------------------------------------------------------------------
def components(mask: np.ndarray, gap_rows: int = 1, gap_bins: int = 2):
    """Connected components of `mask` after bridging gaps; each is
    (row0, bin0, row1, bin1, cell_mask) with exclusive ends and the
    component's own cleared cells within its box."""
    from scipy import ndimage as ndi
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    gr, gb = max(0, int(gap_rows)), max(0, int(gap_bins))
    bridged = m
    if gr or gb:
        st = np.ones((2 * gr + 1, 2 * gb + 1), dtype=bool)
        bridged = ndi.binary_dilation(m, structure=st)
    lab, nlab = ndi.label(bridged, structure=np.ones((3, 3), dtype=bool))
    own = np.where(m, lab, 0)
    out = []
    for i, sl in enumerate(ndi.find_objects(own), start=1):
        if sl is None:
            continue
        cells = own[sl] == i
        out.append((sl[0].start, sl[1].start, sl[0].stop, sl[1].stop, cells))
    return out


def energy_proposer(tile, pfa: float = 1e-4, guard: int = 2, train: int = 16,
                    two_d: bool = False, guard_rows: int = 1, train_rows: int = 4,
                    floor_test: bool = True, floor_margin_db: float = 1.0,
                    integrate_rows: int = 8, integrate_pfa_factor: float = 1e-4,
                    row_gate_pfa: float = 1e-3,
                    min_cells: int = 4, min_duration_s: float = 0.0,
                    min_bins: int = 1, gap_rows: int = 1, gap_bins: int = 2,
                    max_boxes: int = 1000, profile: str | None = None,
                    epoch: float | None = None,
                    info: dict | None = None) -> list[Detection]:
    """CFAR (+ the floor-referenced test, + the integrated layer) on one tile
    -> Proposed energy boxes in absolute time and frequency (module
    docstring). The integrated layer runs at `pfa x integrate_pfa_factor`
    (each of its false alarms covers a whole block, so it must be rarer);
    `integrate_rows=0` turns it off. Pass a dict as `info` to receive the
    numbers: cells_cfar, cells_floor, cells_integrated, components, kept,
    capped, floor_test (used or why not)."""
    lay = tile.layout
    rv = int(tile.rows_valid) if tile.rows_valid else tile.rows
    spec = tile.spec[:rv]
    stats = info if info is not None else {}
    stats.update({"cells_cfar": 0, "cells_floor": 0, "cells_integrated": 0,
                  "components": 0, "kept": 0, "capped": False, "pfa": float(pfa)})
    if rv == 0:
        stats["floor_test"] = "no data rows"
        return []
    mask = ca_cfar(spec, pfa, guard=guard, train=train, pool=lay.pool,
                   window=lay.window, fft_size=lay.fft_size, hop=lay.hop,
                   two_d=two_d, guard_rows=guard_rows, train_rows=train_rows)
    stats["cells_cfar"] = int(mask.sum())
    if floor_test and tile.floor_ok:
        _b, _f, pool_eff = window_stats(lay.window, lay.fft_size, lay.hop, lay.pool)
        fm = floor_mask(spec, pfa, pool_eff, floor_margin_db)
        stats["cells_floor"] = int((fm & ~mask).sum())
        mask = mask | fm
        stats["floor_test"] = "used"
    else:
        stats["floor_test"] = ("off" if not floor_test else
                               f"stood down: the floor moved "
                               f"{tile.floor_offset_db:+.1f} dB at this tile")
    if integrate_rows and int(integrate_rows) > 1:
        im = integrated_mask(tile, float(pfa) * float(integrate_pfa_factor),
                             rows_per_block=int(integrate_rows), guard=guard,
                             train=train, floor_test=floor_test,
                             floor_margin_db=floor_margin_db,
                             row_gate_pfa=row_gate_pfa, info=stats)
        mask = mask | im
    comps = components(mask, gap_rows=gap_rows, gap_bins=gap_bins)
    stats["components"] = len(comps)
    pid = tile.profile if profile is None else str(profile)
    ep = tile.epoch if epoch is None else epoch
    dets = []
    for r0, b0, r1, b1, cells in comps:
        n_cells = int(cells.sum())
        if n_cells < int(min_cells):
            continue
        if (r1 - r0) * tile.row_period < float(min_duration_s) - 1e-12:
            continue
        if (b1 - b0) < int(min_bins):
            continue
        t0, t1, f_lo, f_hi = tile.pixels_to_tf(r0, b0, r1, b1)
        box_spec = spec[r0:r1, b0:b1]
        peak = float(np.max(box_spec[cells]))
        snr = tile.box_snr_db(r0, b0, r1, b1, mask=cells)
        dets.append(Detection(
            t0=t0, t1=t1, f_lo=f_lo, f_hi=f_hi, sources=("energy",),
            family="unknown", snr_db=None if snr is None else round(snr, 2),
            profile=pid, epoch=ep,
            measurements={"peak_db": round(peak, 2), "cells": n_cells,
                          "rows": int(r1 - r0), "bins": int(b1 - b0),
                          "pfa": float(pfa)}))
    if len(dets) > int(max_boxes):
        dets.sort(key=lambda d: d.measurements["peak_db"], reverse=True)
        dets = dets[:int(max_boxes)]
        stats["capped"] = True
    stats["kept"] = len(dets)
    return dets
