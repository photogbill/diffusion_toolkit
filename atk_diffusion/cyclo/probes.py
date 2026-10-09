# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Targeted cyclic probes — "is there a feature HERE", for a few multiplies a
sample (DETECTION_DESIGN §3 cyclic proposer, §4.1; ARCHITECTURE §4.2
`cyclo.probes`; ATK FUTURE_PLANS 2026-09-29, *Cyclostationary probes on the
RF tab*).

Bill, 2026-09-29: *"Is it possible to incorporate cyclostationary especially
for cell towers which are as low to the noise floor as possible? I've always
wanted to pull one from below the noise floor."*

A full SCF (`cyclo.scf`) asks "what features exist anywhere" and costs a
surface. These ask a yes/no question at a SHORT LIST of cycle frequencies or
lags keyed to technologies, at about a thousandth of the cost:

  * `symbol_rate_line(x, fs, rates)` — digital by its symbol rate. Multiply
    the signal by a delayed copy of itself (lag d: x[n]·x*[n−d]; d = 0 is
    the squared envelope |x|²). A digital signal leaves a SPECTRAL LINE at
    its symbol rate in that product; analog FM and noise leave none. The
    line is measured with a single-bin DFT (what a Goertzel filter computes)
    at each listed rate, searched over the rate's clock tolerance, and
    compared with the median of 32 neighbouring bins. Several lags (½ to 2
    symbols) are summed, because a constant-envelope FSK puts its line at
    lags near a symbol, a PSK at lag 0, and nobody knows in advance which.
  * `cp_probe(x, fs, lag_s)` — OFDM by its cyclic prefix. Every CP-OFDM
    symbol ends with a copy of its own start, so the signal correlates with
    itself at a lag of exactly the useful-symbol length (66.7 µs for LTE /
    NR at 15 kHz spacing). The integrated, normalised correlation at that
    lag is compared with the same correlation at 32 nearby lags where OFDM
    has none. The same lag product, Fourier-analysed, gives the symbol-
    timing peak train (its period is the OFDM symbol with its prefix) and
    its phase gives the van de Beek fractional carrier offset.
  * `carrier_conj(x, fs)` — the conjugate (x²) line at twice the carrier of
    BPSK / AM / MSK-class signals; its ABSENCE is a finding too (proper
    QPSK, QAM, OFDM and FSK have none).

EVERY THRESHOLD IS DERIVED, NEVER TUNED (the bench's rule, and FUTURE_PLANS:
*"the threshold must be derived from the noise-only distribution at that
integration"*). For noise alone, each DFT bin of a lag product is complex
Gaussian (central-limit theorem), so its power is exponential. The statistic
is that power over the k-th smallest of K neighbouring bins — an ORDER-
STATISTIC CFAR, whose false-alarm probability is exact for exponential cells
(Rohling 1983):

        P(stat > T) = Π_{i=0}^{k−1} (K − i) / (K − i + T)

so the threshold for a stated Pfa is solved, not chosen. Summing L lags, the
null of the sum is the L-fold convolution of that distribution, computed
numerically. Searching M cycle frequencies (rates × clock tolerance) costs
the Šidák correction 1 − (1 − Pfa)^{1/M}. Coloured noise (a cut is
low-passed) and an unknown noise level cost nothing: both are absorbed by
the neighbours, which is why the statistic is a ratio. The tests check the
empirical false-alarm rate on noise against the stated Pfa.

THE HONEST LEVER (FUTURE_PLANS): processing gain grows with integration
time; the statistic's mean grows linearly with T (the SCF variance falls as
1/(T·Δf)). Measured here, with the derived thresholds:

  * an LTE-like 15 kHz OFDM signal at −10 dB in-band SNR (its spectrum a
    tenth of the floor's) is detected by `cp_probe` in 1 s at 1.92 MS/s
    (statistic 23–57 against a threshold of 12.8), about 20 ms of CPU;
  * a QPSK (root-raised-cosine, 0.35) at −10 dB in-band SNR is found in 5 s
    at a 48 kHz cut (Pd 0.9), from its squared-envelope line;
  * a C4FM-like (raised-cosine) 4800 sym/s 4FSK — P25 / DMR / NXDN96 — has a
    WEAK symbol-rate line (3.6 % of its power at the best lag, against 9.5 %
    for unfiltered 4FSK and far more for PSK at lag 0), and because 4800 Hz
    sits close to its own 8 kHz bandwidth its lag products are correlated
    over ~14 samples, so only ONE lag is independent (summing correlated
    lags looked like a 3 dB gain and was an inflated null — measured, and
    removed). It is found at 0 dB in-band SNR (its spectrum level WITH the
    floor) in 3–5 s (Pd 0.9) and at −3 dB in about 15 s — below what a
    per-frame CFAR sees, not 10 dB under the floor. That is a property of
    constant-envelope FSK, not of the code.

WHAT THIS IS NOT. Cyclostationary detection does not beat a matched filter
on a KNOWN sequence: for a known LTE cell, ATK's PSS/SSS search remains the
strongest detector. These probes answer what a correlator cannot — is there
a modulated signal here at all, of what family, with what timing — before
any sequence is known. And a probe finds only the rates it is given (the
class table); `cyclo.scf` is the blind instrument.
"""

from __future__ import annotations

import functools
import math

import numpy as np

#: Reference (neighbour) cells per test, and which order statistic is used
#: (the median: robust to a second line among the neighbours).
DEFAULT_K = 32
#: Bins between the searched span and the first reference bin each side.
DEFAULT_GUARD = 2


# ---------------------------------------------------------------------------
# The order-statistic CFAR null — derived thresholds
# ---------------------------------------------------------------------------
def os_ccdf(t, K: int = DEFAULT_K, k: int | None = None) -> np.ndarray:
    """P(X > t·Y_(k)) for X and K reference cells i.i.d. exponential:
    Π_{i<k} (K−i)/(K−i+t). Vectorised over t."""
    k = int(k or K // 2)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    i = np.arange(k, dtype=float)
    num = (K - i)[None, :]
    return np.exp(np.sum(np.log(num / (num + np.maximum(t, 0.0)[:, None])),
                         axis=1))


@functools.lru_cache(maxsize=64)
def _sum_ccdf_table(K: int, k: int, L: int) -> tuple:
    """The CCDF of a sum of L i.i.d. OS-CFAR ratios on a grid (numerical
    convolution of the exact single-ratio distribution)."""
    n = 1 << 16
    tmax = 160.0 * L + 400.0
    dt = tmax / n
    edges = np.arange(n + 1) * dt
    G = os_ccdf(edges, K, k)
    mass = -np.diff(G)                       # probability per cell
    tail = float(G[-1])                      # beyond the grid
    m = 1 << int(math.ceil(math.log2(n * L + 1)))
    F = np.fft.rfft(mass, m)
    conv = np.maximum(np.fft.irfft(F ** L, m)[: n * L], 0.0)
    # CCDF from the tail end (accurate at small probabilities)
    ccdf = np.cumsum(conv[::-1])[::-1]
    ccdf = np.concatenate([ccdf[1:], [0.0]]) + L * tail
    grid = (np.arange(n * L) + 1) * dt       # upper edge of each cell
    return grid, ccdf


def os_threshold(pfa: float, K: int = DEFAULT_K, k: int | None = None,
                 L: int = 1) -> float:
    """The statistic a test must exceed for false-alarm probability `pfa`:
    one OS-CFAR ratio (L = 1, solved exactly) or a sum of L of them
    (numerical convolution of the exact distribution)."""
    k = int(k or K // 2)
    p = float(min(0.5, max(1e-14, pfa)))
    if int(L) <= 1:
        lo, hi = 1e-9, 1e9
        for _ in range(200):                 # bisection on log P (monotone)
            mid = math.sqrt(lo * hi)
            if os_ccdf(mid, K, k)[0] > p:
                lo = mid
            else:
                hi = mid
            if hi / lo < 1 + 1e-10:
                break
        return float(hi)
    grid, ccdf = _sum_ccdf_table(int(K), k, int(L))
    idx = int(np.searchsorted(-ccdf, -p))
    return float(grid[min(idx, grid.size - 1)])


def os_pvalue(stat: float, K: int = DEFAULT_K, k: int | None = None,
              L: int = 1) -> float:
    """P(noise alone gives a statistic ≥ `stat`) for one test."""
    k = int(k or K // 2)
    if int(L) <= 1:
        return float(os_ccdf(stat, K, k)[0])
    grid, ccdf = _sum_ccdf_table(int(K), k, int(L))
    if stat <= grid[0]:
        return 1.0
    return float(max(np.interp(stat, grid, ccdf), 1e-300))


def sidak(pfa: float, trials: int) -> float:
    """Per-test probability that keeps the family-wise rate at `pfa`."""
    m = max(1, int(trials))
    return float(-math.expm1(math.log1p(-min(0.999999, float(pfa))) / m))


def family_p(p_test: float, trials: int) -> float:
    """Family-wise p-value of the best of `trials` tests."""
    m = max(1, int(trials))
    return float(-math.expm1(m * math.log1p(-min(0.999999999, p_test))))


def confidence_from_p(p_family: float, pfa: float) -> float:
    """A monotone map of the measured significance to 0..1 — NOT a
    calibrated probability (calibration needs cabled data, DETECTION_DESIGN
    §6.4). 0.45 = just cleared the threshold; 0.95 = noise alone would do
    this a million times less often than the false-alarm rate. Below the
    threshold it falls toward 0, so a near miss reads as ambiguous."""
    s = -math.log10(max(1e-300, float(p_family)))
    s0 = -math.log10(max(1e-300, float(pfa)))
    if p_family <= pfa:
        return float(min(0.95, 0.45 + 0.5 * (s - s0) / 6.0))
    return float(max(0.0, min(0.449, 0.45 * s / max(s0, 1e-9))))


# ---------------------------------------------------------------------------
# The single-bin DFT bank ("Goertzel at the listed rates")
# ---------------------------------------------------------------------------
def zoom_dft(y, fs: float, alpha: float, offsets_hz) -> np.ndarray:
    """The DFT of y at α + each offset (Hz), for y 1-D or [C, N].

    What a Goertzel filter computes, done the numpy way: demodulate by α,
    sum blocks of b samples (a boxcar decimator whose nulls fall exactly on
    the frequencies that would alias onto the bins), then a small exact DFT,
    with the boxcar's droop divided out; the last partial block is added
    exactly. For a SPECTRAL LINE anywhere in the zoomed span the result is
    the single-bin DFT itself (a test holds it to that). For the broadband
    noise around the line it is the DFT of a boxcar-decimated copy — a
    statistically identical value (exponential power, independent bins),
    not a numerically identical one, and the neighbouring reference bins
    are made the same way, which is all a ratio statistic needs. About two
    operations a sample instead of one complex exponential per sample per
    bin."""
    y2 = np.atleast_2d(np.asarray(y))
    C, N = y2.shape
    offs = np.asarray(offsets_hz, dtype=float)
    span = float(np.max(np.abs(offs))) if offs.size else 0.0
    b = int(max(1, min(max(1, N // 16),
                       math.floor(0.05 * fs / max(span, 1e-9)))))
    nb = N // b
    m = nb * b
    n = np.arange(m, dtype=np.float64)
    ph = np.exp(-2j * np.pi * np.mod(float(alpha) / fs * n, 1.0)
                ).astype(np.complex64)
    z = (y2[:, :m].astype(np.complex64, copy=False) * ph[None, :]
         ).reshape(C, nb, b).sum(axis=2, dtype=np.complex128)
    tj = (np.arange(nb) * b + (b - 1) / 2.0) / fs
    E = np.exp(-2j * np.pi * np.outer(tj, offs))
    num = np.sin(np.pi * offs * b / fs)
    den = b * np.sin(np.pi * offs / fs)
    droop = np.where(np.abs(den) > 1e-15, num / np.where(den == 0, 1, den), 1.0)
    out = (z @ E) / droop[None, :]
    if m < N:                       # the last partial block, exactly
        nt = np.arange(m, N, dtype=np.float64)
        Et = np.exp(-2j * np.pi * np.outer(nt, float(alpha) + offs) / fs)
        out = out + y2[:, m:].astype(np.complex128) @ Et
    return out[0] if np.asarray(y).ndim == 1 else out


# ---------------------------------------------------------------------------
# Helpers shared by the probes
# ---------------------------------------------------------------------------
def correlation_length(x, max_lag: int = 32, level: float = 0.1) -> int:
    """Samples over which the record is correlated with itself (≥ 1): the
    first lag at which the normalised autocorrelation falls below `level`.
    White noise gives 1; a cut low-passed to a two-sided band B gives about
    fs/B (the first zero of band-limited noise's autocorrelation, where
    two lag products become uncorrelated). Reference cells and summed lags
    are spaced at least this far apart, as the derived null assumes."""
    x = np.asarray(x)
    if x.ndim > 1:
        x = x[0]
    n = min(x.size, 1 << 18)
    if n < 4 * max_lag:
        return 1
    xs = x[:n].astype(np.complex128) - np.mean(x[:n])
    r0 = float(np.vdot(xs, xs).real) or 1.0
    for k in range(1, int(max_lag) + 1):
        if abs(np.vdot(xs[:-k], xs[k:])) / r0 < level:
            return int(k)
    return int(max_lag)


def bandlimit(x, fs: float, half_bw_hz: float, transition: float = 0.25):
    """Low-pass x (1-D or [C, N]) to ±half_bw_hz: the class-bandwidth
    pre-filter. A quadratic detector's noise grows with the square of the
    noise it is fed, so removing the noise outside the class's band before
    the lag product is worth ~3 dB of sensitivity on a 12.5 kHz channel
    holding an 8 kHz signal (measured)."""
    from scipy.signal import firwin, oaconvolve
    fs = float(fs)
    cut = float(half_bw_hz)
    if cut <= 0 or cut >= 0.49 * fs:
        return np.asarray(x)
    tw = max(transition * cut, fs / 2000.0)
    ntaps = int(min(4097, max(31, 4.0 * fs / tw))) | 1
    taps = firwin(ntaps, cut + 0.5 * tw, fs=fs).astype(np.float32)
    x = np.asarray(x)
    if x.ndim == 1:
        return oaconvolve(x, taps, mode="same").astype(np.complex64)
    return oaconvolve(x, taps[None, :], mode="same", axes=1
                      ).astype(np.complex64)


def autocorrelation(x, max_lag: int) -> np.ndarray:
    """r[k] = ⟨x[n+k]·x*[n]⟩ / ⟨|x|²⟩ for k = 0..max_lag (FFT-based, from
    up to 2^17 samples)."""
    import scipy.fft as sfft
    x = np.asarray(x)
    if x.ndim > 1:
        x = x[0]
    n = int(min(x.size, 1 << 17))
    xs = x[:n].astype(np.complex128) - np.mean(x[:n])
    m = int(sfft.next_fast_len(2 * n))
    X = sfft.fft(xs, m)
    r = sfft.ifft(np.abs(X) ** 2)[: int(max_lag) + 1]
    return r / (r[0].real or 1.0)


class _LagCorr:
    """Σ_k r(k)·r*(k+Δ)·e^{−j2παk/fs} / Σ_k |r(k)|²·e^{−j2παk/fs}, with the
    two-sided r and the phase computed once (they are reused for every
    candidate lag)."""

    def __init__(self, r: np.ndarray, fs: float, alpha: float):
        K = r.size - 1
        self.K = K
        self.full = np.concatenate([np.conj(r[:0:-1]), r])     # k = −K..K
        k = np.arange(-K, K + 1)
        self.ph = np.exp(-2j * np.pi * float(alpha) * k / float(fs))
        self.den = abs(float(np.sum(np.abs(self.full) ** 2 * self.ph).real))

    def __call__(self, delta: int) -> float:
        d = abs(int(delta))
        if d >= self.full.size or self.den <= 0:
            return 0.0
        a = self.full[: self.full.size - d]
        b = self.full[d:]
        num = np.sum(a * np.conj(b) * self.ph[: self.full.size - d])
        return float(abs(num) / max(self.den, 1e-300))


def lag_product_correlation(r: np.ndarray, delta: int, fs: float,
                            alpha: float) -> float:
    """|corr| between the DFTs at α of two lag products whose lags differ
    by `delta`, for Gaussian noise with normalised autocorrelation r[k]
    (k ≥ 0; r[−k] = r*[k]):

        Cov(V_d1(α), V_d2(α)) ∝ Σ_k r(k)·r*(k + Δ)·e^{−j2παk/fs}

    (the Gaussian fourth-moment theorem). The summed statistic's derived
    null assumes independent lags; lags are chosen so this stays small. It
    is NOT a matter of spacing alone: when α approaches the band's width,
    the band overlaps its α-shifted self over only (B − α) Hz, and the lag
    products stay correlated over ~fs/(B − α) samples (measured: 0.58
    between lags 8 and 14 for 4800 sym/s in an 8.3 kHz band)."""
    return _LagCorr(np.asarray(r), fs, alpha)(delta)


def independent_lags(fs: float, rate: float, r: np.ndarray, family: str = "",
                     max_lags: int = 6, rho_max: float = 0.25) -> list[int]:
    """The lags whose products carry the symbol-rate line, chosen so that
    every pair is statistically independent on this noise (|ρ| below
    `rho_max`, computed from the measured autocorrelation r).

    Constant-envelope FSK (family "fsk": P25, DMR, NXDN, POCSAG, FLEX, BLE)
    has NO line at lag 0 — its envelope is flat — and puts it at lags of ¾
    to 2 symbols, strongest near 1–1.5 symbols for raised-cosine shaping
    (measured: 3.6 % of the signal power for C4FM-like shaping, 9.5 % at one
    symbol unfiltered). Linear modulations (PSK, QAM, VSB) put their
    strongest line at lag 0, the squared envelope. An unknown family gets
    lag 0 plus the FSK range."""
    sps = float(fs) / float(rate)
    fam = str(family or "").lower()
    lo = max(1, int(round((0.75 if fam == "fsk" else 0.5) * sps)))
    hi = max(lo, int(round(2.0 * sps)))
    chosen: list[int] = [] if fam == "fsk" else [0]
    # candidates fanned out from the middle of the useful range, so a
    # small budget lands where FSK's line is strongest
    mid = int(round(1.25 * sps))
    cands = sorted(range(lo, hi + 1), key=lambda d: (abs(d - mid), d))
    corr = _LagCorr(np.asarray(r), fs, rate)
    for d in cands:
        if len(chosen) >= max_lags:
            break
        if all(corr(d - c) < rho_max for c in chosen):
            chosen.append(d)
    return sorted(chosen)


def _lag_product(x2: np.ndarray, d: int) -> np.ndarray:
    """x[n]·x*[n−d] per row. Only the squared envelope (d = 0) has its
    mean removed: for d > 0 the mean is tiny and its leakage to a cycle
    frequency α ≫ 1/T is bounded by fs/(πα) samples' worth — far below the
    noise in the bin — so the pass is not spent."""
    if d == 0:
        y = (x2.real.astype(np.float32) ** 2 + x2.imag.astype(np.float32) ** 2)
        return (y - y.mean(axis=1, keepdims=True)).astype(np.complex64)
    return (x2[:, d:] * np.conj(x2[:, :-d])).astype(np.complex64)


# ---------------------------------------------------------------------------
# symbol_rate_line — digital by its symbol rate
# ---------------------------------------------------------------------------
def _line_core(x2: np.ndarray, fs: float, rate: float, lags: list[int],
               K: int, guard: int, tol_ppm: float, oversample: int = 2,
               chunk: int = 32) -> dict:
    """Statistic per channel for one rate: max over the searched offsets of
    Σ_lags |V_d|² / (k-th smallest reference |V_d|²)."""
    C, N = x2.shape
    dmax = max(lags) if lags else 0
    T = (N - dmax) / fs
    half = max(1, int(math.ceil(tol_ppm * 1e-6 * rate * T)))
    search = np.arange(-half * oversample, half * oversample + 1) / oversample
    j = np.arange(1, K // 2 + 1)
    refs = np.concatenate([-(half + guard + j), half + guard + j]).astype(float)
    offs = np.concatenate([search, refs]) / T
    k = K // 2
    S = search.size
    total = np.zeros((C, S))
    for c0 in range(0, C, chunk):
        xc = x2[c0:c0 + chunk]
        for d in lags:
            y = _lag_product(xc, d)
            v = zoom_dft(y, fs, rate, offs)
            p = np.abs(np.atleast_2d(v)) ** 2
            med = np.sort(p[:, S:], axis=1)[:, k - 1]
            total[c0:c0 + chunk] += p[:, :S] / np.maximum(med, 1e-300)[:, None]
    best = np.argmax(total, axis=1)
    stat = total[np.arange(C), best]
    return {"stat": stat, "alpha": rate + search[best] / T, "points": S,
            "half_bins": half, "T": T}


def symbol_rate_line(x, fs: float, rates, pfa: float = 1e-3, lags=None,
                     tol_ppm: float = 100.0, K: int = DEFAULT_K,
                     guard: int = DEFAULT_GUARD, bandwidth_hz=None,
                     family="", max_lags: int = 3):
    """Is there a spectral line at any of `rates` (Hz) in x's lag products?

    x: 1-D complex (one channel) -> dict; [C, N] -> list of dicts (the
    proposer's channel grid, vectorised). `pfa` is the probability that
    THIS CALL reports a line on noise alone (over every rate and every
    searched offset). `tol_ppm` is the clock tolerance searched around each
    rate (transmitter plus receiver; an uncorrected RTL can be 50 ppm off).
    `bandwidth_hz`, when given, band-limits x to ±bandwidth/2 first (the
    class-bandwidth pre-filter). `lags=None` uses `independent_lags` per rate,
    chosen by `family` ("fsk", "psk_qam", … — one string, or a dict
    rate → family) because FSK and PSK put their lines at different lags,
    and spaced so their noise is independent (`independent_lags`).

    Each result: {probe, detected, statistic, threshold, integration_s, pfa,
    pfa_per_test, trials, best_rate_hz, alpha_hz, p_value, per_rate[...],
    method, words}. The statistic is in units of "times the median
    neighbouring bin", summed over the lags."""
    xa = np.asarray(x)
    one = xa.ndim == 1
    x2 = np.atleast_2d(xa).astype(np.complex64, copy=False)
    fs = float(fs)
    if bandwidth_hz:
        x2 = np.atleast_2d(bandlimit(x2, fs, 0.5 * float(bandwidth_hz)))
    C, N = x2.shape
    rates = [float(r) for r in np.atleast_1d(rates) if 0 < float(r) < 0.5 * fs]
    # the noise's own autocorrelation decides which lags are independent
    max_sps = max([fs / r for r in rates], default=1.0)
    rr = autocorrelation(x2[0], int(min(4 * max_sps + 64, 4096)))
    per = []
    searched = 0
    for r in rates:
        fam = (family.get(r, "") if isinstance(family, dict) else family)
        lg = (list(lags) if lags is not None
              else independent_lags(fs, r, rr, family=fam,
                                    max_lags=max_lags))
        lg = [d for d in lg if d < N // 4]
        if not lg or N / fs * r < 32:        # fewer than 32 symbols: refuse
            continue
        core = _line_core(x2, fs, r, lg, K, guard, tol_ppm)
        searched += core["points"]
        per.append((r, lg, core))
    trials = max(1, searched)
    p_test = sidak(pfa, trials)
    out = []
    for c in range(C):
        rows = []
        for r, lg, core in per:
            thr = os_threshold(p_test, K, K // 2, len(lg))
            st = float(core["stat"][c])
            pv = os_pvalue(st, K, K // 2, len(lg))
            rows.append({"rate_hz": r, "statistic": st, "threshold": thr,
                         "detected": bool(st > thr),
                         "alpha_hz": float(core["alpha"][c]), "p_value": pv,
                         "lags": lg, "searched_bins": core["points"]})
        out.append(_line_result(rows, N / fs, pfa, p_test, trials, rates,
                                bool(bandwidth_hz), bandwidth_hz))
    return out[0] if one else out


def _line_result(rows, T, pfa, p_test, trials, rates, prefiltered, bw):
    best = max(rows, key=lambda r: r["statistic"] / r["threshold"],
               default=None)
    det = any(r["detected"] for r in rows)
    if best is None:
        why = ("no listed rate can be tested: each needs at least 32 symbols "
               f"in the record ({T:.3f} s here) and must be below half the "
               "sample rate")
        return {"probe": "symbol_rate_line", "detected": False,
                "statistic": 0.0, "threshold": float("inf"),
                "integration_s": T, "pfa": pfa, "pfa_per_test": p_test,
                "trials": trials, "best_rate_hz": None, "alpha_hz": None,
                "p_value": 1.0, "per_rate": [], "method": _LINE_METHOD,
                "words": why}
    pf = family_p(best["p_value"], trials)
    if det:
        hits = [r for r in rows if r["detected"]]
        words = ("a spectral line at " + ", ".join(
            f"{r['alpha_hz']:,.2f} Hz ({r['statistic'] / r['threshold']:.1f}"
            "× the threshold)" for r in hits)
            + f" after {T:.2f} s of integration — a digital signal with that "
            "symbol rate (or a harmonic of one) is present. A hint for "
            "routing, never an identification: the decoder says what it is.")
    else:
        words = (f"no line at the listed rate(s) after {T:.2f} s (best "
                 f"{best['statistic']:.1f} against a threshold of "
                 f"{best['threshold']:.1f} at {best['rate_hz']:,.0f} Hz). "
                 "Either no digital signal with these rates is here, or it is "
                 "too weak for this integration time — the statistic grows "
                 "in proportion to the time integrated.")
    return {"probe": "symbol_rate_line", "detected": det,
            "statistic": float(best["statistic"]),
            "threshold": float(best["threshold"]), "integration_s": T,
            "pfa": pfa, "pfa_per_test": p_test, "trials": trials,
            "best_rate_hz": best["rate_hz"], "alpha_hz": best["alpha_hz"],
            "p_value": best["p_value"], "p_family": pf,
            "confidence": confidence_from_p(pf, pfa),
            "prefiltered_hz": bw if prefiltered else None,
            "per_rate": rows, "method": _LINE_METHOD, "words": words}


_LINE_METHOD = ("lag products x[n]·x*[n−d] (d = 0 is |x|²), single-bin DFT "
                "at each listed rate over its clock tolerance; statistic = "
                "line power over the median of 32 neighbouring bins, summed "
                "over the lags; threshold from the exact order-statistic CFAR "
                "null at the stated Pfa (Šidák over rates × offsets)")


# ---------------------------------------------------------------------------
# cp_probe — OFDM by its cyclic prefix (the cell-tower probe)
# ---------------------------------------------------------------------------
def _corr_at_lags(x: np.ndarray, lags: np.ndarray, chunk: int = 1 << 18):
    """acc[l] = Σ_n x[n]·conj(x[n+lag_l]) over a common n range, plus the
    energies of x[n] and x[n+D] over it (chunked, constant memory)."""
    lmax = int(lags.max())
    N = x.size - lmax
    acc = np.zeros(lags.size, dtype=np.complex128)
    e0 = 0.0
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        a = x[s:e]
        for i, d in enumerate(lags):
            acc[i] += np.vdot(x[s + d:e + d], a)
        e0 += float(np.vdot(a, a).real)
    return acc, e0, N


def cp_probe(x, fs: float, lag_s: float, pfa: float = 1e-3,
             K: int = DEFAULT_K, period_pfa: float | None = None,
             period_hint_hz: float | None = None) -> dict:
    """Is there CP-OFDM with useful-symbol length `lag_s` in x?

    statistic = |Σ x[n]·x*[n+D]|² at the useful-symbol lag D, over the
    k-th smallest of the same at K reference lags near D (where OFDM has no
    correlation) — the integrated normalised correlation compared with its
    own neighbourhood, so a DC offset, a spur or coloured noise, which
    correlate at every lag, cannot fake it. Threshold: the exact
    order-statistic CFAR null at `pfa` (one test).

    Estimates (in `estimates` and at top level):
      rho            |normalised correlation| at D (0..1): the strength, a
                     ratio to the window energy, so it does not move with
                     gain — the FUTURE_PLANS caption number
      cfo_hz         van de Beek fractional carrier offset −arg(ρ)·fs/(2πD),
                     unambiguous within ±1/(2·lag) (±7.5 kHz for LTE)
      symbol_period_s, symbol_rate_hz   the period of the CP correlation
                     train (useful symbol + prefix); for LTE the slot-average
                     71.43 µs = 1/14 kHz
      cp_fraction, cp_samples           prefix length from period − lag
      timing_offset_s                   first symbol start, mod the period
      snr_db         the OFDM's SNR within x's band from ρ = β·s/(1+s)
                     (β = cp_fraction), when the period was found

    `period_hint_hz` (the class table's symbol rate, 14 kHz for LTE)
    narrows the period search to ±0.5 % of it: fewer cells searched, a lower
    derived threshold, so the timing is found at a lower SNR. Without it the
    search covers every prefix fraction from 0 to 35 %.
    """
    x = np.asarray(x)
    if x.ndim > 1:
        x = x[0]
    fs = float(fs)
    D = int(round(float(lag_s) * fs))
    T = x.size / fs
    base = {"probe": "cp_probe", "detected": False, "statistic": 0.0,
            "threshold": float("inf"), "integration_s": T, "pfa": pfa,
            "lag_s": float(lag_s), "lag_samples": D,
            "lag_error_samples": float(lag_s) * fs - D,
            "method": _CP_METHOD}
    if D < 4:
        base["words"] = (f"the lag {lag_s * 1e6:.2f} µs is only {D} samples at "
                         f"{fs:,.0f} S/s — too short to probe; use a wider cut")
        base["estimates"] = {}
        return base
    xs = x.astype(np.complex64, copy=False)
    xs = xs - np.mean(xs)
    c = correlation_length(xs)
    step = max(2, c)
    g = max(4, 2 * c)
    below = [D - g - step * j for j in range(K // 2) if D - g - step * j > 2 * c]
    above_n = K - len(below)
    above = [D + g + step * j for j in range(above_n)]
    refs = np.array(below + above, dtype=np.int64)
    if x.size < int(refs.max()) + 8 * D + 256:
        base["words"] = ("the record is too short for this lag — it needs at "
                         "least a few dozen OFDM symbols")
        base["estimates"] = {}
        return base
    lags = np.concatenate([[D], refs])
    acc, e0, n_used = _corr_at_lags(xs, lags)
    eD = float(np.vdot(xs[D:D + n_used], xs[D:D + n_used]).real)
    p = np.abs(acc) ** 2
    k = K // 2
    med = float(np.sort(p[1:])[k - 1])
    stat = float(p[0] / max(med, 1e-300))
    thr = os_threshold(pfa, K, k, 1)
    pv = os_pvalue(stat, K, k, 1)
    det = stat > thr
    rho = float(abs(acc[0]) / math.sqrt(max(e0 * eD, 1e-300)))
    cfo = float(-np.angle(acc[0]) * fs / (2 * np.pi * D))
    # phase noise of the correlation: its SNR is the statistic on the mean
    # scale (the median of an exponential is ln 2 of its mean)
    snr_corr = max(stat * math.log(2.0) - 1.0, 1e-3)
    est = {"rho": rho, "cfo_hz": cfo, "cfo_fraction": cfo * D / fs,
           "cfo_se_hz": float(fs / (2 * np.pi * D) / math.sqrt(2 * snr_corr)),
           "cfo_range_hz": 0.5 * fs / D, "reference_lags": [int(refs.min()),
                                                             int(refs.max())],
           "reference_step": int(step)}
    est.update(_cp_period(xs, fs, D, period_pfa if period_pfa else pfa,
                          hint=period_hint_hz))
    beta = est.get("cp_fraction")
    if est.get("period_detected") and beta and 0 < rho < 0.95 * beta:
        s = rho / (beta - rho)
        est["snr_db"] = float(10 * np.log10(s))
        est["snr_method"] = ("from the CP correlation strength ρ = β·s/(1+s), "
                             "β the measured prefix fraction — the OFDM's SNR "
                             "within the probed band")
    else:
        est["snr_db"] = None
    words = _cp_words(det, stat, thr, T, lag_s, est)
    base.update({"detected": bool(det), "statistic": stat, "threshold": thr,
                 "p_value": pv, "confidence": confidence_from_p(pv, pfa),
                 "estimates": est, "words": words, **est})
    return base


def _cp_period(xs: np.ndarray, fs: float, D: int, pfa: float,
               K: int = DEFAULT_K, hint: float | None = None) -> dict:
    """The CP correlation train: p[n] = x[n]·x*[n+D] is large once per OFDM
    symbol (inside the prefix) — its period is the symbol with its prefix,
    its phase the symbol timing. Searched over prefix fractions 0..35 %."""
    n = xs.size - D
    b = max(1, D // 16)
    m = (n // b) * b
    if m // b < 512:
        return {"period_detected": False}
    pb = np.zeros(m // b, dtype=np.complex128)
    chunk = (1 << 18) // b * b
    for s in range(0, m, chunk):
        e = min(m, s + chunk)
        prod = xs[s:e] * np.conj(xs[s + D:e + D])
        pb[s // b:e // b] = prod.reshape(-1, b).sum(axis=1)
    # every prefix pulse carries the same phase, −2π·CFO·D/fs (the van de
    # Beek phase): taken out here, or the line's phase reads it as timing —
    # 9 samples at a 1 kHz offset, half a symbol at ±7.5 kHz (measured)
    total = complex(pb.sum())
    if abs(total) > 0:
        pb = pb * (np.conj(total) / abs(total))
    pb = pb - pb.mean()
    import scipy.fft as sfft
    nf = int(sfft.next_fast_len(pb.size))
    F = sfft.fft(pb, nf)
    fr = np.fft.fftfreq(nf, b / fs)
    P = np.abs(F) ** 2
    lo, hi = fs / (1.35 * D), fs / (1.0005 * D)
    if hint and lo <= float(hint) <= hi:
        lo, hi = 0.995 * float(hint), 1.005 * float(hint)
    band = np.flatnonzero((fr >= lo) & (fr <= hi))
    if band.size < 4:
        return {"period_detected": False}
    kpk = int(band[np.argmax(P[band])])
    j = np.arange(1, K // 2 + 1)
    ref = np.concatenate([P[(kpk - DEFAULT_GUARD - j) % nf],
                          P[(kpk + DEFAULT_GUARD + j) % nf]])
    stat = float(P[kpk] / max(np.sort(ref)[K // 2 - 1], 1e-300))
    cells = max(1, int(round(band.size * pb.size / nf)))
    thr = os_threshold(sidak(pfa, cells), K, K // 2, 1)
    # the peak frequency to a small fraction of a bin: a 1/16-bin zoom DFT
    # around the peak, then a parabola. (A parabola through the POWER of
    # three FFT bins — the first version — is pinned to the bin centre for
    # an unwindowed line: 0.21 bins off read as 0.01, i.e. a receiver clock
    # 30 ppm off read as 1 ppm, and a second-long fold smeared by dozens
    # of samples.)
    fs_b = fs / b
    df = fs_b / nf
    offs = np.linspace(-1.0, 1.0, 33) * df
    zv = np.abs(zoom_dft(pb, fs_b, float(fr[kpk]), offs)) ** 2
    j = int(np.argmax(zv))
    f_s = float(fr[kpk] + offs[j])
    if 0 < j < zv.size - 1:
        den = zv[j - 1] - 2 * zv[j] + zv[j + 1]
        if abs(den) > 0:
            f_s += float(0.5 * (zv[j - 1] - zv[j + 1]) / den) * (offs[1] - offs[0])
    period = 1.0 / f_s
    cp_samples = period * fs - D
    beta = max(0.0, cp_samples / (period * fs))
    # timing: the train's line referenced to x[0] — its phase AT the refined
    # frequency (at the bin's centre it is off by up to a quarter period for
    # a line half a bin away); blocks are centred at sample (b-1)/2, the
    # prefix pulse at its middle
    ang = float(np.angle(zoom_dft(pb, fs_b, f_s, [0.0])[0]))
    t_c = (-ang / (2 * np.pi * f_s) + (b - 1) / (2 * fs)) % period
    start = (t_c - 0.5 * cp_samples / fs) % period
    return {"period_detected": bool(stat > thr), "period_statistic": stat,
            "period_threshold": thr, "symbol_period_s": period,
            "symbol_rate_hz": f_s, "cp_samples": float(cp_samples),
            "cp_fraction": float(beta), "timing_offset_s": float(start)}


def _cp_words(det, stat, thr, T, lag_s, est) -> str:
    scs = 1.0 / float(lag_s)
    if det:
        bits = [f"OFDM with {scs / 1e3:,.3g} kHz subcarrier spacing: the "
                f"signal correlates with itself at {lag_s * 1e6:,.2f} µs "
                f"({stat / thr:.1f}× the threshold after {T:.2f} s)",
                f"strength ρ = {est['rho']:.4f}",
                f"carrier offset {est['cfo_hz']:+,.0f} Hz (cyclic prefix, "
                f"±{est['cfo_range_hz']:,.0f} Hz unambiguous)"]
        if est.get("period_detected"):
            bits.append(f"symbol period {est['symbol_period_s'] * 1e6:,.2f} µs"
                        f" (prefix ≈ {est['cp_fraction'] * 100:.1f} %)")
        if est.get("snr_db") is not None:
            bits.append(f"about {est['snr_db']:+.1f} dB SNR in this band")
        return "; ".join(bits) + (". Point the raster search here — the PSS/SSS"
                                  " correlators confirm a cell.")
    return (f"no cyclic-prefix correlation at {lag_s * 1e6:,.2f} µs after "
            f"{T:.2f} s (statistic {stat:.1f}, threshold {thr:.1f}). The "
            "statistic grows with integration time: a cell 10 dB under the "
            "floor wants about a second, 20 dB under wants a hundred.")


_CP_METHOD = ("normalised correlation at the useful-symbol lag, integrated "
              "over the record, against the same at 32 reference lags; exact "
              "order-statistic CFAR threshold at the stated Pfa; van de Beek "
              "CFO from its phase; symbol timing from the period of the lag "
              "product")


# ---------------------------------------------------------------------------
# cp_timing_phases — how many cells share the channel (the cell-tower survey)
# ---------------------------------------------------------------------------
#: The honest limit, said in every result.
CP_CELLS_LIMIT = (
    "Transmitters that are time-synchronised — TDD LTE, 5G NR, single-"
    "frequency broadcast networks — put their prefixes at the same instant "
    "and are counted as ONE; FDD LTE cells are usually not time-aligned, "
    "which is why this works there. Two transmitters whose prefixes start "
    "less than one prefix length apart are also counted as one.")


def _fold_lags(xs: np.ndarray, lags, n_use: int, P: float, R: int,
               energy_lag: int | None = None, chunk: int = 1 << 20):
    """F[l, r] = Σ_{n < n_use, phase(n) in bin r} x[n]·x*[n + lags[l]] —
    the lag products folded over the period P (samples, real) into R phase
    bins — and, with `energy_lag` D, E[r] = Σ ½(|x[n]|² + |x[n+D]|²)."""
    lags = [int(d) for d in lags]
    F = np.zeros((len(lags), R), dtype=np.complex128)
    E = np.zeros(R) if energy_lag is not None else None
    for s in range(0, n_use, chunk):
        e = min(n_use, s + chunk)
        n = np.arange(s, e, dtype=np.float64)
        idx = (np.floor(np.mod(n, P) * (R / P)).astype(np.int64)) % R
        a = xs[s:e]
        for i, d in enumerate(lags):
            p = a * np.conj(xs[s + d:e + d])
            F[i] += (np.bincount(idx, weights=p.real, minlength=R)
                     + 1j * np.bincount(idx, weights=p.imag, minlength=R))
        if E is not None:
            D = int(energy_lag)
            w = 0.5 * (np.abs(a) ** 2 + np.abs(xs[s + D:e + D]) ** 2)
            E += np.bincount(idx, weights=w, minlength=R)
    return F, E


def _circ_sum(v: np.ndarray, starts: np.ndarray, L: int) -> np.ndarray:
    """Σ_{i<L} v[..., (s + i) mod R] for each start s (the last axis is
    circular)."""
    R = v.shape[-1]
    ext = np.concatenate([v, v[..., :L]], axis=-1)
    c = np.concatenate([np.zeros(v.shape[:-1] + (1,), dtype=v.dtype),
                        np.cumsum(ext, axis=-1)], axis=-1)
    s = np.asarray(starts) % R
    return c[..., s + L] - c[..., s]


def cp_timing_phases(x, fs: float, lag_s: float, period_s: float | None = None,
                     pfa: float = 1e-3, max_cells: int = 4,
                     K: int = DEFAULT_K) -> dict:
    """How many DISTINCT, non-time-aligned CP-OFDM transmitters share this
    channel — the cell-tower survey (Bill, 2026-10-09).

    Every OFDM symbol's prefix copies its own end, so x[n]·x*[n + D] (D the
    useful-symbol lag) has a mean only inside each transmitter's prefixes.
    Folded over the OFDM symbol period P and summed over one prefix length
    L, that is the van de Beek timing metric γ(τ); each transmitter whose
    symbols start at its own instant gives its own peak, with its own phase
    (its fractional carrier offset −arg γ·fs/(2πD)) and its own strength
    ρ = |γ|/energy (≈ its share of the power, S_c/(S_total + N)).

    THE THRESHOLD IS DERIVED (the module's rule). The same fold at K
    reference lags, where no prefix correlates, gives at every phase K
    noise-only values; the statistic at lag D is the order-statistic CFAR
    ratio over them — exact (os_ccdf) for exponential cells, whatever the
    signal and noise levels. The period is split into windows of one prefix
    length on two interleaved grids (so a prefix never straddles a boundary
    by more than a quarter), 2·⌊P/L⌋ tests, Šidák over them — so `pfa` is
    the probability that noise alone reports ANY transmitter. A peak is
    placed to the sample by the fine metric; peaks closer than one prefix
    length are one transmitter (the minimum separation).

    period_s   the OFDM symbol period with its prefix (1/14 kHz for LTE).
               None: measured from the record (the prefix train's line,
               `_cp_period`). Given: the period actually received (clock
               error included — an RTL can be 50 ppm off, which smears a
               second-long fold by dozens of samples) is measured around it
               when the record shows it, else the given one is used. A
               measured period is then refined by the fold's own sharpness.

    Returns {probe, n_cells, cells: [{timing_offset_s, timing_offset_samples,
    rho, relative_db, cfo_hz, statistic, threshold, p_value}], more_above,
    threshold, pfa, tests, period_s, period_source, prefix_samples,
    min_separation_s, integration_s, lag_s, method, limit, words}.

    LIMIT (in every result's words): time-synchronised transmitters (TDD
    LTE, NR, SFN) coincide and count as one; FDD LTE cells are usually not
    time-aligned, which is why this works there."""
    x = np.asarray(x)
    if x.ndim > 1:
        x = x[0]
    fs = float(fs)
    D = int(round(float(lag_s) * fs))
    T = x.size / fs
    max_cells = max(1, int(max_cells))
    base = {"probe": "cp_timing_phases", "n_cells": 0, "cells": [],
            "more_above": 0, "threshold": float("inf"), "pfa": float(pfa),
            "tests": 0, "period_s": None, "period_source": "",
            "prefix_samples": None, "min_separation_s": None,
            "integration_s": T, "lag_s": float(lag_s), "lag_samples": D,
            "method": _CP_CELLS_METHOD, "limit": CP_CELLS_LIMIT}
    if not 0.0 < float(pfa) < 1.0:
        raise ValueError(f"pfa is a probability between 0 and 1, not {pfa!r}")
    if D < 4:
        base["words"] = (f"the lag {lag_s * 1e6:.2f} µs is only {D} samples at "
                         f"{fs:,.0f} S/s — too short to fold; use a wider cut. "
                         + CP_CELLS_LIMIT)
        return base
    xs = x.astype(np.complex64, copy=False)
    xs = (xs - np.mean(xs)).astype(np.complex64)
    if not np.all(np.isfinite(xs)):
        raise ValueError("the IQ holds samples that are not finite (NaN or "
                         "infinity) — a damaged file or a failed read")
    # -- the period ------------------------------------------------------------
    p_period = min(float(pfa), 1e-3)
    hint = (1.0 / float(period_s)) if period_s else None
    est = _cp_period(xs, fs, D, p_period, K, hint=hint)
    if est.get("period_detected"):
        P = float(est["symbol_period_s"]) * fs
        source = ("measured from the record" + (" around the period given"
                                                if hint else ""))
        measured = True
    elif period_s:
        P = float(period_s) * fs
        source = "given (the record does not show the period by itself)"
        measured = False
    else:
        base["words"] = ("no OFDM symbol period at this lag shows in the record "
                         f"after {T:.2f} s — no cell to count (give period_s "
                         "to fold anyway). " + CP_CELLS_LIMIT)
        base["period_source"] = "not found"
        return base
    L = int(round(P - D))
    if L < 1:
        raise ValueError(f"the symbol period ({P / fs * 1e6:.2f} µs) must be "
                         f"longer than the lag ({lag_s * 1e6:.2f} µs): the "
                         "difference is the prefix")
    # -- reference lags (cp_probe's spacing: clear of the signal's own
    #    correlation length) ----------------------------------------------------
    c = correlation_length(xs)
    step = max(2, c)
    g = max(4, 2 * c)
    below = [D - g - step * j for j in range(K // 2) if D - g - step * j > 2 * c]
    above = [D + g + step * j for j in range(K - len(below))]
    refs = below + above
    n_use = xs.size - max(refs + [D]) - 1
    R = max(8, int(round(P)))
    if n_use < 64 * P:
        base["words"] = ("the record is too short to fold: it needs at least "
                         f"64 OFDM symbols ({64 * P / fs * 1e3:.1f} ms) beyond "
                         "the longest reference lag. " + CP_CELLS_LIMIT)
        return base
    if measured:
        P = _refine_period(xs, D, P, L, n_use, R)
        R = max(8, int(round(P)))
    F, E = _fold_lags(xs, [D] + refs, n_use, P, R, energy_lag=D)
    # -- the decision: windows of one prefix on two interleaved grids ---------
    Lb = max(1, int(round(L * R / P)))                 # prefix length in bins
    m1 = max(1, R // Lb)
    starts = np.concatenate([np.floor(np.arange(m1) * R / m1),
                             np.floor(np.arange(m1) * R / m1) + Lb // 2]
                            ).astype(np.int64) % R
    G = _circ_sum(F, starts, Lb)                       # [1 + K, windows]
    pw = np.abs(G) ** 2
    k = K // 2
    ref_k = np.sort(pw[1:], axis=0)[k - 1]
    stat = pw[0] / np.maximum(ref_k, 1e-300)
    tests = int(starts.size)
    thr = os_threshold(sidak(float(pfa), tests), K, k, 1)
    # -- placing the peaks: the fine metric over every phase bin --------------
    allr = np.arange(R)
    gam = _circ_sum(F[0], allr, Lb)
    gE = _circ_sum(E, allr, Lb)
    order = np.argsort(stat)[::-1]
    cells, extras = [], []
    for wi in order:
        if stat[wi] <= thr:
            break
        s0 = int(starts[wi])
        win = (s0 + np.arange(-Lb, Lb + 1)) % R
        r = int(win[np.argmax(np.abs(gam[win]))])
        # sub-bin refinement on |γ|² (parabola through the peak)
        a_, b_, c_ = (np.abs(gam[(r - 1) % R]) ** 2, np.abs(gam[r]) ** 2,
                      np.abs(gam[(r + 1) % R]) ** 2)
        den = a_ - 2 * b_ + c_
        frac = 0.5 * (a_ - c_) / den if abs(den) > 0 else 0.0
        frac = float(min(0.5, max(-0.5, frac)))
        tau = (r + frac) % R                               # bins
        # one transmitter per prefix length: the two interleaved windows of
        # one peak, and its neighbours, are the same transmitter — for the
        # count beyond max_cells as much as for the cells kept
        if any(min(abs(tau - q), R - abs(tau - q)) < Lb
               for q in [c["_bin"] for c in cells] + extras):
            continue
        if len(cells) >= max_cells:
            extras.append(tau)
            continue
        rho = float(abs(gam[r]) / max(gE[r], 1e-300))
        cfo = float(-np.angle(gam[r]) * fs / (2 * np.pi * D))
        cells.append({"_bin": tau, "timing_offset_samples": tau * P / R,
                      "timing_offset_s": tau * P / R / fs, "rho": rho,
                      "cfo_hz": cfo, "statistic": float(stat[wi]),
                      "threshold": float(thr),
                      "p_value": float(os_pvalue(float(stat[wi]), K, k, 1))})
    cells.sort(key=lambda q: q["rho"], reverse=True)
    top = cells[0]["rho"] if cells else 0.0
    for q in cells:
        q.pop("_bin")
        q["relative_db"] = (10 * math.log10(q["rho"] / top)
                            if top > 0 and q["rho"] > 0 else None)
    base.update({"n_cells": len(cells), "cells": cells, "more_above": len(extras),
                 "threshold": float(thr), "tests": tests,
                 "period_s": P / fs, "period_source": source,
                 "prefix_samples": L, "min_separation_s": L / fs,
                 "reference_lags": [int(min(refs)), int(max(refs))]})
    base["words"] = _cells_words(base, T, lag_s)
    return base


def _refine_period(xs: np.ndarray, D: int, P0: float, L: int, n_use: int,
                   R: int, steps: int = 6) -> float:
    """The measured period, refined by the fold's own sharpness: the fold of
    the lag product at D is evaluated at P0 + j·ΔP (|j| ≤ steps), ΔP = L/(4·
    symbols in the record) — the period error that smears the fold by a
    quarter prefix over the record — and the one with the tallest timing
    peak kept. (The prefix-train line already gives the period to well
    inside that tolerance at usable SNR — measured; this is insurance.)"""
    nsym = n_use / P0
    dP = L / (4.0 * nsym)
    Lb = max(1, int(round(L * R / P0)))
    best, bestP = -1.0, P0
    for j in range(-steps, steps + 1):
        Pj = P0 + j * dP
        F, _E = _fold_lags(xs, [D], n_use, Pj, R)
        v = float(np.max(np.abs(_circ_sum(F[0], np.arange(R), Lb)) ** 2))
        if v > best:
            best, bestP = v, Pj
    return bestP


def _cells_words(r: dict, T: float, lag_s: float) -> str:
    n = r["n_cells"]
    head = (f"{n} distinct CP-OFDM transmitter{'s' if n != 1 else ''} at this "
            f"lag ({lag_s * 1e6:,.2f} µs), the timing metric folded over the "
            f"{r['period_s'] * 1e6:,.2f} µs symbol period ({r['period_source']})"
            f" for {T:.2f} s")
    if n == 0:
        return (head + ": no timing peak cleared the derived threshold "
                f"({r['threshold']:.1f}). " + CP_CELLS_LIMIT)
    bits = []
    for i, q in enumerate(r["cells"], start=1):
        rel = (f", {q['relative_db']:+.1f} dB" if q["relative_db"] is not None
               and i > 1 else "")
        bits.append(f"#{i} at {q['timing_offset_s'] * 1e6:,.1f} µs, ρ "
                    f"{q['rho']:.4f}{rel}, CFO {q['cfo_hz']:+,.0f} Hz")
    more = (f" (and {r['more_above']} more peak{'s' if r['more_above'] != 1 else ''}"
            " above the threshold — raise max_cells)" if r["more_above"] else "")
    return head + ": " + "; ".join(bits) + more + ". " + CP_CELLS_LIMIT


_CP_CELLS_METHOD = (
    "van de Beek timing metric: the lag product x[n]·x*[n+D] folded over the "
    "OFDM symbol period and summed over one prefix length; each phase window "
    "tested against the same fold at 32 reference lags (exact order-"
    "statistic CFAR null), Šidák over the 2·⌊P/L⌋ windows; peaks placed to "
    "the sample on the fine metric, at least one prefix apart; per peak, ρ = "
    "|γ|/energy and the CFO from arg γ")


# ---------------------------------------------------------------------------
# carrier_conj — the x² (conjugate) line
# ---------------------------------------------------------------------------
def carrier_conj(x, fs: float, pfa: float = 1e-3, power: int = 2,
                 band_hz: tuple | None = None, K: int = DEFAULT_K,
                 candidates: int = 16) -> dict:
    """A spectral line in x^power at power × the carrier offset.

    power = 2: BPSK, AM, ASK, MSK-class (conjugate cyclostationary) put a
    line at 2·f_c; proper QPSK, QAM, OFDM and FSK put none — so an ABSENCE,
    with a long enough record, is a positive statement about the family.
    power = 4 finds QPSK's carrier at 4·f_c (ambiguous within ±fs/8).
    `band_hz` = (lo, hi) limits the carrier search (fewer trials, a lower
    threshold). Threshold: exact OS-CFAR null, Šidák over the bins searched.
    """
    x = np.asarray(x)
    if x.ndim > 1:
        x = x[0]
    fs = float(fs)
    T = x.size / fs
    pw = int(power)
    xs = x.astype(np.complex128) - np.mean(x)
    pwr = float(np.mean(np.abs(xs) ** 2)) or 1.0
    y = (xs / math.sqrt(pwr)) ** pw
    import scipy.fft as sfft
    nf = int(sfft.next_fast_len(min(y.size, 1 << 21)))
    Y = sfft.fft(y[:nf], nf)
    P = np.abs(Y) ** 2
    fr = np.fft.fftfreq(nf, 1.0 / fs)
    if band_hz:
        lo, hi = sorted(float(v) * pw for v in band_hz)
        band = np.flatnonzero((fr >= lo) & (fr <= hi))
    else:
        band = np.arange(nf)
    if band.size < 4 or x.size < 256:
        return {"probe": "carrier_conj", "detected": False, "statistic": 0.0,
                "threshold": float("inf"), "integration_s": T, "pfa": pfa,
                "power": pw, "estimates": {},
                "words": "record too short for a carrier line"}
    # candidate bins by an approximate local floor, then the exact OS ratio
    span = max(4 * K, nf // 512)
    approx = P[band] / np.maximum(_block_median(P, span)[band], 1e-300)
    top = band[np.argsort(approx)[::-1][: int(candidates)]]
    j = np.arange(1, K // 2 + 1)
    scored = []
    for kk in top:
        ref = np.concatenate([P[(kk - DEFAULT_GUARD - j) % nf],
                              P[(kk + DEFAULT_GUARD + j) % nf]])
        scored.append((float(P[kk] / max(np.sort(ref)[K // 2 - 1], 1e-300)),
                       int(kk)))
    scored.sort(reverse=True)
    stat, kpk = scored[0] if scored else (0.0, 0)
    sep = DEFAULT_GUARD + K // 2
    second = next(((s, k2) for s, k2 in scored[1:]
                   if min(abs(k2 - kpk), nf - abs(k2 - kpk)) > sep), (0.0, kpk))
    cells = max(1, int(round(band.size * min(1.0, x.size / nf))))
    p_test = sidak(pfa, cells)
    thr = os_threshold(p_test, K, K // 2, 1)
    pv = os_pvalue(stat, K, K // 2, 1)
    pf = family_p(pv, cells)
    a, c0, c1 = P[(kpk - 1) % nf], P[kpk], P[(kpk + 1) % nf]
    den = a - 2 * c0 + c1
    delta = 0.5 * (a - c1) / den if abs(den) > 0 else 0.0
    alpha = float(fr[kpk] + delta * fs / nf)
    carrier = alpha / pw
    det = stat > thr
    est = {"alpha_hz": alpha, "carrier_offset_hz": carrier,
           "unambiguous_range_hz": [-fs / (2 * pw), fs / (2 * pw)],
           "line_fraction": float(abs(Y[kpk]) / max(1, min(y.size, nf))),
           "second_statistic": float(second[0]),
           "second_alpha_hz": float(fr[second[1]])}
    if det:
        words = (f"a conjugate line at {alpha:+,.1f} Hz in x^{pw}: carrier "
                 f"offset {carrier:+,.1f} Hz ({stat / thr:.1f}× the threshold "
                 f"after {T:.2f} s). "
                 + ("The constellation is NOT symmetric under conjugation — "
                    "BPSK, AM, ASK or MSK-class, not proper QPSK or QAM."
                    if pw == 2 else "A QPSK-class carrier (4th-power line)."))
    else:
        words = (f"no line in x^{pw} after {T:.2f} s (best {stat:.1f}, "
                 f"threshold {thr:.1f}). "
                 + ("That is a FINDING, not a failure: BPSK, AM, ASK and MSK "
                    "all make one at twice the carrier, so its absence points "
                    "at the proper-QPSK / 8PSK / QAM / OFDM / FSK family — "
                    "provided the record holds a few hundred symbols."
                    if pw == 2 else ""))
    return {"probe": "carrier_conj", "detected": bool(det), "statistic": stat,
            "threshold": thr, "integration_s": T, "pfa": pfa,
            "pfa_per_test": p_test, "trials": cells, "p_value": pv,
            "p_family": pf, "confidence": confidence_from_p(pf, pfa),
            "power": pw, "estimates": est, "words": words,
            "method": (f"x^{pw} spectrum; line over the median of 32 "
                       "neighbouring bins; exact order-statistic CFAR "
                       "threshold, Šidák over the bins searched"), **est}


def _block_median(P: np.ndarray, span: int) -> np.ndarray:
    """A local median floor, coarse (block medians, interpolated) — only to
    RANK candidate lines; the decision uses the exact OS ratio."""
    n = P.size
    step = max(1, span // 4)
    centres = np.arange(0, n, step)
    floors = np.array([np.median(P[max(0, c - span): min(n, c + span + 1)])
                       for c in centres])
    return np.interp(np.arange(n), centres, floors)
