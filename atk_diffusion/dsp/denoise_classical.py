# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The classical denoisers every learned one is scored against, and the
measuring stick they are all scored with (plan §7, §4.B3; ARCHITECTURE §2
rule 8).

Plan §7: *"Every denoiser is scored against Wiener, median and wavelet …
A learned tool that does not beat the classical one is not shipped."* These
are those three, written so they can be trusted as the baseline:

* **Wiener** — the spectral Wiener gain G = ξ/(1 + ξ) on a spectrogram tile,
  with the a-priori SNR ξ estimated from the tile and a NOISE PSD ESTIMATE
  (measured from noise, or a low percentile over time when no reference is
  given); and the same gain on the complex STFT of IQ (1D), resynthesised by
  overlap-add. |G| ≤ 1: it can only attenuate, so it cannot put a signal
  where there was none.
* **Median** — a 2D median filter on the tile in dB.
* **Wavelet** — a 2D discrete wavelet transform (Haar or Daubechies-4,
  periodised, orthonormal) with BayesShrink soft thresholding (Chang, Yu &
  Vetterli 2000), the noise level from the finest diagonal band by MAD.
  Written here in numpy: PyWavelets is not installed in ATK's core
  environment and is not required.

All three work on the tile in dB ABOVE THE FLOOR — the log domain, where the
spectral estimation noise is additive and of constant variance (the log is
the variance-stabilising transform for a scaled-gamma periodogram), which is
the assumption each of them makes. Their outputs carry the tier `cleaned`
(provenance): a classical process applied to the record, nothing added, but
not the record.

THE MEASURING STICK. To score a denoiser you need the representation it
works in and a detector that does not care whose output it is looking at:

* `stft_power` / `power_tile` — the power spectrogram (fft-shifted, scaled
  so white noise of power σ² reads σ² in every bin), pooled in time by mean
  or max (the pipeline's tiles are max-pooled, ARCHITECTURE §4.1).
* `estimate_floor` — the noise floor per bin, "a low percentile over time"
  (ARCHITECTURE §4.1), corrected to the mean by the known distribution.
* `ca_cfar` — cell-averaging CFAR along frequency per frame
  (DETECTION_DESIGN §3), with the threshold multiplier DERIVED from the
  false-alarm rate, never tuned: exactly F(2K, 2NK) for mean-pooled cells
  (the classic α = N(Pfa^(−1/N) − 1) at K = 1, as in ATK's own
  `siga/detect.py`), and for max-pooled cells the same question solved
  numerically. This is the internal fallback for `dsp.cfar` (another
  engineer's module); the weak-burst experiment uses that one when present.

HONEST LIMITS. Hann windows correlate adjacent bins, so CFAR training cells
are not quite independent and the realised false-alarm rate runs somewhat
above nominal (the tests measure it). The Wiener gain here is the
power-spectral form, not the decision-directed speech-enhancement one. The
wavelet uses periodic boundaries, so a signal at one edge of a tile can leak
a little into the other edge's coefficients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from atk_diffusion import provenance

DB = 10.0 / math.log(10.0)       # dB per neper of power

#: The three classical methods, by the name `provenance.METHOD_TIERS` knows.
METHODS = ("wiener", "median", "wavelet")


# ---------------------------------------------------------------------------
# The representation: power spectrogram, pooled tiles, dB above the floor
# ---------------------------------------------------------------------------
def window(name: str, n: int) -> np.ndarray:
    name = str(name or "hann").lower()
    if name in ("hann", "hanning"):
        return np.hanning(n + 1)[:-1].astype(np.float64)   # periodic Hann
    if name in ("rect", "rectangular", "boxcar", "none"):
        return np.ones(n)
    if name == "hamming":
        return np.hamming(n + 1)[:-1]
    if name == "blackman":
        return np.blackman(n + 1)[:-1]
    raise ValueError(f"unknown window {name!r}")


def stft_power(x, fft_size: int, hop: int | None = None,
               window_name: str = "hann") -> np.ndarray:
    """|STFT|² as [frames, bins], bins fft-shifted (−fs/2 … +fs/2), scaled
    so complex white noise of power σ² per sample reads σ² in every bin."""
    x = np.asarray(x, dtype=np.complex64)
    n = int(fft_size)
    hop = int(hop or n)
    if x.size < n:
        raise ValueError(f"{x.size} samples is shorter than one {n}-point frame")
    w = window(window_name, n)
    frames = 1 + (x.size - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(frames)[:, None]
    seg = x[idx] * (w / math.sqrt(np.sum(w ** 2)))[None, :].astype(np.float32)
    X = np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1)
    return (X.real.astype(np.float64) ** 2 + X.imag.astype(np.float64) ** 2)


def pool_rows(P: np.ndarray, factor: int, mode: str = "mean") -> np.ndarray:
    """Pool consecutive frames in groups of `factor` (a remainder is
    dropped). mode 'mean' integrates; 'max' keeps a short burst whole."""
    f = int(factor)
    if f <= 1:
        return np.asarray(P, dtype=np.float64)
    rows = P.shape[0] // f
    if rows < 1:
        raise ValueError(f"{P.shape[0]} frames cannot be pooled by {f}")
    g = P[: rows * f].reshape(rows, f, P.shape[1])
    if mode == "mean":
        return g.mean(axis=1)
    if mode == "max":
        return g.max(axis=1)
    raise ValueError("pooling is 'mean' or 'max'")


def power_tile(x, fft_size: int, hop: int | None = None, pool: int = 1,
               mode: str = "mean", window_name: str = "hann") -> np.ndarray:
    return pool_rows(stft_power(x, fft_size, hop, window_name), pool, mode)


def samples_for_rows(rows: int, fft_size: int, hop: int | None = None,
                     pool: int = 1) -> int:
    """How many samples make exactly `rows` pooled rows."""
    hop = int(hop or fft_size)
    return int(fft_size) + hop * (int(rows) * int(pool) - 1)


def db_above(P: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """10·log10(P / floor), floor per bin (broadcast over rows)."""
    return DB * np.log(np.maximum(P, 1e-30) / np.maximum(np.asarray(floor), 1e-30))


def gamma_quantile(q: float, k: int = 1) -> float:
    """The q-quantile of the mean of k unit-mean exponentials."""
    from scipy.stats import gamma
    return float(gamma.ppf(q, k, scale=1.0 / k))


def estimate_floor(P_frames: np.ndarray, q: float = 0.25, k: int = 1) -> np.ndarray:
    """The noise floor per bin from frames that may hold signals: the
    q-quantile over time, divided by the q-quantile of the noise-only
    distribution (mean of k exponentials) so it estimates the noise MEAN.
    A signal present in fewer than (1 − q) of the frames barely moves it."""
    Pq = np.quantile(np.asarray(P_frames, dtype=np.float64), q, axis=0)
    return Pq / gamma_quantile(q, k)


def logpower_noise_stats(k: int = 1, mode: str = "mean") -> tuple[float, float]:
    """(mean, std) in dB of 10·log10 of the pooled power of noise alone,
    in floor units — what a noise-only tile cell reads. Mean pooling:
    exact (digamma/trigamma); max pooling: quadrature of the max of k
    exponentials. K = 1: −2.51 dB and 5.57 dB."""
    k = int(k)
    if mode == "mean":
        from scipy.special import digamma, polygamma
        return (DB * float(digamma(k) - math.log(k)),
                DB * math.sqrt(float(polygamma(1, k))))
    if mode == "max":
        u = (np.arange(200000) + 0.5) / 200000.0
        m = -np.log1p(-u ** (1.0 / k))          # inverse CDF of max of k Exp(1)
        v = DB * np.log(m)
        return float(v.mean()), float(v.std())
    raise ValueError("pooling is 'mean' or 'max'")


# ---------------------------------------------------------------------------
# CA-CFAR — the classical detector every denoiser is scored with
# ---------------------------------------------------------------------------
def cfar_alpha(pfa: float, n_train: int, k: int = 1, mode: str = "mean") -> float:
    """The CA-CFAR threshold multiplier on the training mean, DERIVED from
    the per-cell false-alarm rate (never tuned).

    mean pooling: cell X and training mean Ȳ of N cells are means of k and
    N·k exponentials, so X/Ȳ ~ F(2k, 2Nk) exactly; α is its upper-Pfa point
    (= N(Pfa^(−1/N) − 1) at k = 1).
    max pooling: the cell is a max of k exponentials (CDF (1 − e^(−x))^k);
    the training mean is taken as normal by the central limit theorem
    (N ≥ 8 cells) and P(X > αȲ) = Pfa solved by quadrature and bisection."""
    pfa = float(pfa)
    if not 0.0 < pfa < 1.0:
        raise ValueError("a false-alarm rate is between 0 and 1")
    n, k = int(n_train), max(1, int(k))
    if n < 1:
        raise ValueError("CFAR needs training cells")
    if mode == "mean":
        from scipy.stats import f as fdist
        return float(fdist.isf(pfa, 2 * k, 2 * k * n))
    if mode != "max":
        raise ValueError("pooling is 'mean' or 'max'")
    mu = sum(1.0 / i for i in range(1, k + 1))
    sd = math.sqrt(sum(1.0 / i ** 2 for i in range(1, k + 1)) / n)
    z, w = np.polynomial.hermite_e.hermegauss(60)
    ybar = np.maximum(mu + sd * z, 1e-6)
    w = w / math.sqrt(2.0 * math.pi)

    def pfa_of(a):
        return float(np.sum(w * (1.0 - (1.0 - np.exp(-a * ybar)) ** k)))
    lo, hi = 1e-3, 1e4
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        if pfa_of(mid) > pfa:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def _training_kernel(guard: int, train: int) -> np.ndarray:
    g, t = int(guard), int(train)
    ker = np.ones(2 * (g + t) + 1)
    ker[t: t + 2 * g + 1] = 0.0
    return ker / (2.0 * t)


def cfar_ratio(P: np.ndarray, guard: int = 2, train: int = 8) -> np.ndarray:
    """Each cell's power over the mean of its training cells (along
    frequency, the same frame). Scale-free: units cancel."""
    from scipy.ndimage import convolve1d
    P = np.asarray(P, dtype=np.float64)
    local = convolve1d(P, _training_kernel(guard, train), axis=-1, mode="reflect")
    return P / np.maximum(local, 1e-30)


def ca_cfar(P: np.ndarray, pfa: float, guard: int = 2, train: int = 8,
            k: int = 1, mode: str = "mean") -> np.ndarray:
    """Boolean mask of cells above α(Pfa) × their training mean."""
    return cfar_ratio(P, guard, train) > cfar_alpha(pfa, 2 * int(train), k, mode)


def run_statistic(ratio: np.ndarray, min_bins: int = 1) -> float:
    """max over the tile of the smallest ratio in any run of `min_bins`
    contiguous bins: a run of r bins clears α exactly when this exceeds α.
    The one scalar per tile the experiments calibrate thresholds on."""
    from scipy.ndimage import minimum_filter1d
    r = max(1, int(min_bins))
    if r == 1:
        return float(np.max(ratio))
    # windows hanging off the edge see the constant 0 and never win the max
    m = minimum_filter1d(ratio, size=r, axis=-1, mode="constant", cval=0.0)
    return float(np.max(m))


@dataclass
class CfarBox:
    row0: int
    row1: int            # exclusive
    bin0: int
    bin1: int            # exclusive
    cells: int
    peak_ratio_db: float


def cfar_boxes(mask: np.ndarray, ratio: np.ndarray | None = None,
               min_bins: int = 1, min_cells: int = 1) -> list[CfarBox]:
    """Connected regions of a CFAR mask as boxes (DETECTION_DESIGN §3:
    "its boxes are time runs of contiguous bins above threshold")."""
    from scipy.ndimage import find_objects, label
    lab, n = label(mask)
    out = []
    for i, sl in enumerate(find_objects(lab), start=1):
        if sl is None:
            continue
        cells = int(np.sum(lab[sl] == i))
        width = sl[1].stop - sl[1].start
        if width < min_bins or cells < min_cells:
            continue
        peak = float(DB * math.log(max(float(np.max(ratio[sl][lab[sl] == i])), 1e-30))) \
            if ratio is not None else float("nan")
        out.append(CfarBox(sl[0].start, sl[0].stop, sl[1].start, sl[1].stop,
                           cells, peak))
    return out


def cfar_detect_db(tile_db: np.ndarray, pfa: float, guard: int = 2,
                   train: int = 8, k: int = 1, mode: str = "mean",
                   min_bins: int = 1) -> tuple[bool, list[CfarBox]]:
    """CA-CFAR on a tile in dB above the floor: (anything found, boxes)."""
    P = np.power(10.0, np.asarray(tile_db, dtype=np.float64) / 10.0)
    ratio = cfar_ratio(P, guard, train)
    mask = ratio > cfar_alpha(pfa, 2 * int(train), k, mode)
    boxes = cfar_boxes(mask, ratio, min_bins=min_bins)
    return bool(boxes), boxes


# ---------------------------------------------------------------------------
# Wiener
# ---------------------------------------------------------------------------
def wiener_gain(P: np.ndarray, noise: np.ndarray | float,
                size=(3, 3)) -> np.ndarray:
    """G = ξ/(1 + ξ), ξ = max(⟨P/N⟩ − 1, 0) averaged over a `size`
    time-frequency neighbourhood (averaging the a-posteriori SNR first, then
    subtracting 1, keeps the noise-only bias of ξ small)."""
    from scipy.ndimage import uniform_filter
    gamma = np.asarray(P, dtype=np.float64) / np.maximum(np.asarray(noise, dtype=np.float64), 1e-30)
    xi = np.maximum(uniform_filter(gamma, size=size, mode="reflect") - 1.0, 0.0)
    return xi / (1.0 + xi)


def wiener_tile(tile_db: np.ndarray, noise_lin=None, size=(3, 3),
                q: float = 0.25) -> np.ndarray:
    """The spectral Wiener filter on a tile in dB above the floor.

    `noise_lin` is the noise level in the tile's own linear units (per bin
    or scalar) — the mean of 10^(dB/10) over noise-only tiles, which is 1
    for mean-pooled tiles and the harmonic number H_k for max-pooled ones.
    When not given it is estimated per bin from the tile (a low percentile
    over rows). Output: dB above the floor of (noise + |Ŝ|²), |Ŝ|² = G²·P —
    noise-only cells come out near 0 dB and smooth, signal cells keep their
    level."""
    P = np.power(10.0, np.asarray(tile_db, dtype=np.float64) / 10.0)
    if noise_lin is None:
        N = estimate_floor(P, q=q, k=1)[None, :]
    else:
        N = np.asarray(noise_lin, dtype=np.float64)
    G = wiener_gain(P, N, size)
    S = G * G * P
    return DB * np.log1p(S / np.maximum(N, 1e-30))


def wiener_iq(x, noise_power: float | None = None, nfft: int = 64,
              size=(3, 3), q: float = 0.25) -> np.ndarray:
    """The Wiener gain on the complex STFT of IQ (Hann, 50 % overlap),
    resynthesised by overlap-add: the 1D Wiener filter. `noise_power` is
    the noise power per complex sample; estimated from the STFT (a low
    percentile over time) when not given. Output length = input length."""
    from scipy.signal import istft, stft
    x = np.asarray(x, dtype=np.complex128)
    n = int(nfft)
    f, t, Z = stft(x, nperseg=n, noverlap=n // 2, window="hann",
                   return_onesided=False, boundary="even", padded=True)
    P = np.abs(Z) ** 2                         # [bins, frames]
    w = np.hanning(n + 1)[:-1]
    if noise_power is None:
        N = estimate_floor(P.T, q=q, k=1)[:, None]
    else:
        N = float(noise_power) * np.sum(w ** 2) / np.sum(w) ** 2
    G = wiener_gain(P, N, size=(size[1], size[0]))
    _, y = istft(G * Z, nperseg=n, noverlap=n // 2, window="hann",
                 input_onesided=False, boundary=True)
    y = np.asarray(y)[: x.size]
    if y.size < x.size:
        y = np.concatenate([y, np.zeros(x.size - y.size)])
    return y.astype(np.complex64)


# ---------------------------------------------------------------------------
# Median
# ---------------------------------------------------------------------------
def median_tile(tile_db: np.ndarray, size=(3, 3)) -> np.ndarray:
    """A 2D median filter on the tile in dB (impulsive noise and isolated
    hot cells go; an edge survives better than under a mean)."""
    from scipy.ndimage import median_filter
    return median_filter(np.asarray(tile_db, dtype=np.float64), size=size,
                         mode="reflect")


# ---------------------------------------------------------------------------
# Wavelets — a numpy DWT (Haar, Daubechies-4) and BayesShrink
# ---------------------------------------------------------------------------
_SQ3 = math.sqrt(3.0)
WAVELETS = {
    "haar": np.array([1.0, 1.0]) / math.sqrt(2.0),
    # Daubechies-4 (two vanishing moments; PyWavelets calls it 'db2')
    "db2": np.array([1 + _SQ3, 3 + _SQ3, 3 - _SQ3, 1 - _SQ3]) / (4.0 * math.sqrt(2.0)),
}
WAVELETS["d4"] = WAVELETS["db2"]


def _filters(name: str):
    try:
        h = WAVELETS[str(name).lower()]
    except KeyError:
        raise ValueError(f"unknown wavelet {name!r} — one of "
                         f"{', '.join(sorted(WAVELETS))}") from None
    L = h.size
    g = np.array([(-1) ** n * h[L - 1 - n] for n in range(L)])
    return h, g


def dwt1(x: np.ndarray, wavelet: str = "db2", axis: int = -1):
    """One level of the periodised orthonormal DWT along `axis`:
    a[k] = Σ h[n] x[(2k+n) mod N], d[k] = Σ g[n] x[(2k+n) mod N]."""
    h, g = _filters(wavelet)
    x = np.moveaxis(np.asarray(x, dtype=np.float64), axis, -1)
    N = x.shape[-1]
    if N % 2:
        raise ValueError("a DWT level needs an even length (pad first)")
    idx = (2 * np.arange(N // 2)[:, None] + np.arange(h.size)[None, :]) % N
    seg = x[..., idx]                            # [..., N/2, L]
    a = seg @ h
    d = seg @ g
    return np.moveaxis(a, -1, axis), np.moveaxis(d, -1, axis)


def idwt1(a: np.ndarray, d: np.ndarray, wavelet: str = "db2", axis: int = -1):
    """The inverse of `dwt1` (exact: the transform is orthonormal)."""
    h, g = _filters(wavelet)
    a = np.moveaxis(np.asarray(a, dtype=np.float64), axis, -1)
    d = np.moveaxis(np.asarray(d, dtype=np.float64), axis, -1)
    N = a.shape[-1] * 2
    au = np.zeros(a.shape[:-1] + (N,))
    du = np.zeros_like(au)
    au[..., 0::2] = a
    du[..., 0::2] = d
    x = np.zeros_like(au)
    for n in range(h.size):
        x += h[n] * np.roll(au, n, axis=-1) + g[n] * np.roll(du, n, axis=-1)
    return np.moveaxis(x, -1, axis)


def dwt2(x: np.ndarray, wavelet: str = "db2", levels: int = 3):
    """Multi-level 2D DWT. Returns (approximation, [(LH, HL, HH) per level,
    finest first]). Both dimensions must be divisible by 2**levels."""
    a = np.asarray(x, dtype=np.float64)
    details = []
    for _ in range(int(levels)):
        lo, hi = dwt1(a, wavelet, axis=1)
        ll, lh = dwt1(lo, wavelet, axis=0)
        hl, hh = dwt1(hi, wavelet, axis=0)
        details.append((lh, hl, hh))
        a = ll
    return a, details


def idwt2(a: np.ndarray, details, wavelet: str = "db2") -> np.ndarray:
    for lh, hl, hh in reversed(details):
        lo = idwt1(a, lh, wavelet, axis=0)
        hi = idwt1(hl, hh, wavelet, axis=0)
        a = idwt1(lo, hi, wavelet, axis=1)
    return a


def soft(x: np.ndarray, thr: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - thr, 0.0)


def bayes_shrink(details, sigma: float | None = None):
    """BayesShrink: per subband, T = σ²/σ_X with σ_X² = max(⟨c²⟩ − σ², 0);
    σ from the finest diagonal band by MAD (median |HH1| / 0.6745) unless
    given. A subband with no signal variance above the noise is zeroed.
    Returns (shrunk details, σ used)."""
    if sigma is None:
        sigma = float(np.median(np.abs(details[0][2]))) / 0.6745
    out = []
    for band in details:
        nb = []
        for c in band:
            sx2 = max(float(np.mean(c ** 2)) - sigma ** 2, 0.0)
            if sx2 <= 0.0:
                nb.append(np.zeros_like(c))
            else:
                nb.append(soft(c, sigma ** 2 / math.sqrt(sx2)))
        out.append(tuple(nb))
    return out, sigma


def wavelet_tile(tile_db: np.ndarray, wavelet: str = "db2", levels: int = 3,
                 sigma_db: float | None = None) -> np.ndarray:
    """BayesShrink wavelet denoising of a tile in dB. The tile is
    reflect-padded to a multiple of 2**levels and cropped back. `sigma_db`
    is the noise standard deviation in dB if it was measured (else MAD)."""
    x = np.asarray(tile_db, dtype=np.float64)
    m = 2 ** int(levels)
    pr, pc = (-x.shape[0]) % m, (-x.shape[1]) % m
    xp = np.pad(x, ((0, pr), (0, pc)), mode="reflect") if (pr or pc) else x
    a, det = dwt2(xp, wavelet, levels)
    det, _sig = bayes_shrink(det, sigma_db)
    return idwt2(a, det, wavelet)[: x.shape[0], : x.shape[1]]


# ---------------------------------------------------------------------------
# One entry point with the tier on the result
# ---------------------------------------------------------------------------
@dataclass
class Cleaned:
    out: np.ndarray
    method: str
    tier: str
    params: dict = field(default_factory=dict)

    @property
    def words(self) -> str:
        return provenance.TIER_WORDS[self.tier]


def clean_tile(tile_db: np.ndarray, method: str, noise_lin=None,
               sigma_db: float | None = None, **params) -> Cleaned:
    """Run one classical method on a tile in dB above the floor. The result
    carries `tier` = cleaned (provenance.tier_for(method))."""
    m = str(method).lower()
    if m not in METHODS:
        raise ValueError(f"unknown classical method {method!r} — one of "
                         f"{', '.join(METHODS)}")
    tier = provenance.tier_for(m)
    if m == "wiener":
        size = tuple(params.get("size", (3, 3)))
        out = wiener_tile(tile_db, noise_lin=noise_lin, size=size)
        used = {"size": list(size), "noise": "measured" if noise_lin is not None
                else "estimated (low percentile over rows)"}
    elif m == "median":
        size = tuple(params.get("size", (3, 3)))
        out = median_tile(tile_db, size=size)
        used = {"size": list(size)}
    elif m == "wavelet":
        wv = params.get("wavelet", "db2")
        lv = int(params.get("levels", 3))
        out = wavelet_tile(tile_db, wavelet=wv, levels=lv, sigma_db=sigma_db)
        used = {"wavelet": wv, "levels": lv, "threshold": "BayesShrink",
                "sigma": "measured" if sigma_db is not None else "MAD"}
    else:
        raise ValueError(f"unknown classical method {method!r} — one of "
                         f"{', '.join(METHODS)}")
    return Cleaned(out=out, method=m, tier=tier, params=used)
