# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The spectral correlation function (SCF): FAM and SSCA, the cyclic domain
profile, and the classifier's fixed-geometry SCF image
(DETECTION_DESIGN §4.1; ARCHITECTURE §4.2 `cyclo.scf`).

Bill, 2026-10-08: *"incorporate cyclostationary detection and processing
into some of the training data, and give access to run signal cuts of RF
through cyclostationary / AI trained on it."*

CYCLOSTATIONARITY IN ONE PARAGRAPH, FOR AN RF ANALYST WHO HAS NOT USED IT.
Noise looks the same at every instant. A modulated signal does not: its
statistics repeat once per symbol (and, for some modulations, at twice the
carrier). Multiply a signal by a delayed copy of itself — exactly what a
delay-and-multiply clock-recovery circuit does — and the product contains a
SINE WAVE at the symbol rate; a spectral line. Noise multiplied by itself
makes no line. The frequency of such a line is called a CYCLE FREQUENCY, α.
The spectral correlation function S(f, α) says how strongly the signal's
spectrum at f + α/2 moves in lock-step with its spectrum at f − α/2. At
α = 0 it is the ordinary power spectrum; at any other α it is zero for noise
and non-zero only where a signal has a clock at that α. Collapse S over f
(take the maximum over f for each α) and you get the CYCLIC DOMAIN PROFILE:
one curve over α whose peaks ARE the symbol rate, the chip rate, the hop
rate, and — in the CONJUGATE version, built from x·x instead of x·x* — twice
the carrier offset of BPSK / AM / MSK-class signals (proper QPSK and QAM have
no conjugate feature at all, which is itself a measurement).

WHAT IS HERE, ported from Bill's own tested bench (`atk/core/siga/csp`,
caf.py, fam.py, ssca.py, summary.py), with numpy only:

  * `fam`   — the FFT Accumulation Method (Roberts, Brown & Loomis). A bank of
              Np channels (Δf = fs/Np), every pair of channels multiplied and
              Fourier-transformed along time (Δα = fs/(hop·P)). Coarse in α,
              cheap.
  * `ssca`  — the Strip Spectral Correlation Analyzer: Δα = fs/N, as fine as
              the record allows, at N × N′ points of memory. Refuses rather
              than exhausting memory.
  * `cyclic_profile` — max over f of |S(f, α)|, from FAM or SSCA,
              non-conjugate or conjugate. This is the CURVE the cut viewer
              plots.
  * `lag_profile`, `local_ratio`, `detection_threshold` — Bill's lag-domain
              α-profile: one FFT per lag, with a KNOWN null distribution
              (exponential, mean 1), so a threshold comes from a false-alarm
              rate instead of from a constant somebody typed. This is the
              DETECTOR the measurements use to say which peaks are real.
  * `scf_image` — the classifier's second input (DETECTION_DESIGN §4): a
              coherent FAM max-pooled onto a FIXED grid (default 64
              frequency rows × 128 cycle-frequency columns) from the
              profile's FAM geometry, so a model never meets an SCF of
              another geometry.
  * `scf_at`, `spectral_coherence`, `scf_scan` — the SCF at one α by
              frequency smoothing; the slow, obviously-correct reference the
              fast methods are tested against.

THREE CHANGES FROM THE BENCH, each a correction, not a preference:

  1. **The conjugate mapping.** For the conjugate SCF the two spectral
     components are X(α/2 + f) and X(α/2 − f) — the second one REVERSED in
     frequency. The bench's `fam(conj=True)` mapped channel pairs with the
     non-conjugate rule (α = f₁ − f₂) and its `scf`/`spectral_coherence`
     (conj=True) multiplied X(f + α/2)·X(f − α/2). Measured on a BPSK with a
     600 Hz carrier offset: the bench's conjugate coherence at the true
     α = 1200 Hz was 0.446 against 0.455 at an arbitrary 3333 Hz — no
     feature; with the reversed spectrum it is 0.962 against 0.437. Here the
     conjugate FAM uses α = f₁ + f₂, f = (f₁ − f₂)/2. (The bench's lag-domain
     `carrier_probe`, which is what ATK actually uses to find carriers, is
     unaffected — it was right all along.)
  2. **The whole record.** The bench's lag profile FFTs at most 65,536
     samples and silently drops the rest. A cyclic estimate improves with
     every sample (its variance falls as 1/(T·Δf)), so `lag_profile` uses
     the whole cut, up to 2²⁰ samples.
  3. **FAM keeps the central quarter of each channel pair**, as the method
     is defined (|α_q| ≤ Δf/2), and tapers along time. Keeping every α_q of
     every pair, as the bench did, adds edge-of-strip estimates that are
     attenuated by the channel filter and only raise the floor.

LIMITS, stated. The SCF needs a few hundred symbols to resolve a cycle
frequency, so a very short burst gives nothing. An SCF is an analysis, not a
filter — it cleans nothing (the filters that USE what it finds are in
`cyclo.filters`). FAM's α resolution is set by the record LENGTH, not by
the SNR: a short cut cannot separate two close symbol rates however clean
it is (`ScfSurface.valid` and `resolution_note` say when that bites).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Refuse an SSCA above this many complex strip points (≈ 256 MB at 16
#: bytes each). Lower than the bench's 64 M because the toolkit shares the
#: machine with the cognitive core; the refusal says what to change.
SSCA_MAX_POINTS = 16_000_000

#: The lag profile's longest FFT: 2^20 samples (≈ 22 s at a 48 kHz cut).
LAG_PROFILE_MAX = 1 << 20


# ---------------------------------------------------------------------------
# The surface record
# ---------------------------------------------------------------------------
@dataclass
class ScfSurface:
    """A spectral-correlation surface with what is needed not to over-read it.

    `magnitude[i, j]` is |S| (or the coherence, 0..1, when `normalized`) at
    cycle frequency `alphas[i]` and frequency `freqs[j]`. `valid` records
    whether the reliability condition — Δf comfortably coarser than Δα —
    held. A surface that fails it looks exactly like one that passes, which
    is why it is recorded rather than left to the eye.
    """

    alphas: np.ndarray           # Hz, bin centres
    freqs: np.ndarray            # Hz, bin centres
    magnitude: np.ndarray        # [alpha, freq]
    conj: bool
    method: str
    df_hz: float
    dalpha_hz: float
    normalized: bool = False
    note: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return bool(self.magnitude.size) and self.df_hz >= 4.0 * self.dalpha_hz

    def resolution_note(self) -> str:
        bits = [f"Δf ≈ {self.df_hz:,.1f} Hz", f"Δα ≈ {self.dalpha_hz:,.3f} Hz"]
        if self.magnitude.size and not self.valid:
            bits.append("Δf is NOT comfortably coarser than Δα — this surface "
                        "is unreliable; lengthen the record or coarsen the "
                        "frequency resolution")
        return " · ".join(bits)


def _empty(conj: bool, method: str, df: float, da: float, note: str
           ) -> ScfSurface:
    return ScfSurface(np.zeros(0), np.zeros(0), np.zeros((0, 0)), bool(conj),
                      method, float(df), float(da), note=note)


# ---------------------------------------------------------------------------
# Small shared pieces
# ---------------------------------------------------------------------------
def window(kind: str, n: int) -> np.ndarray:
    """hann · hamming · blackmanharris · rectangular, length n, float64."""
    n = int(n)
    k = str(kind or "hann").lower()
    if k == "hann":
        return np.hanning(n)
    if k == "hamming":
        return np.hamming(n)
    if k in ("rectangular", "rect", "boxcar"):
        return np.ones(n)
    if k == "blackmanharris":
        a = (0.35875, 0.48829, 0.14128, 0.01168)
        idx = np.arange(n)
        w = np.zeros(n)
        for i, ai in enumerate(a):
            w += ((-1) ** i) * ai * np.cos(2 * np.pi * i * idx / max(1, n - 1))
        return w
    raise ValueError(f"unknown window {kind!r} — hann, hamming, "
                     "blackmanharris or rectangular")


def _as_complex(x) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("the SCF is computed on one channel; pass x[c] for a "
                         "multi-channel cut")
    return x.astype(np.complex128, copy=False)


def _reduce_max(out_flat: np.ndarray, keys: np.ndarray, vals: np.ndarray
                ) -> None:
    """out_flat[key] = max(out_flat[key], vals) for every element.

    `keys` must hold each value in ONE contiguous run (true for every
    channel pair of FAM and every strip of SSCA, where α moves
    monotonically along the transform axis); keys < 0 are dropped. A
    segmented max (reduceat) collapses the runs first, so the slow
    `np.maximum.at` touches a few hundred values instead of millions.
    """
    if keys.size == 0:
        return
    brk = np.flatnonzero(keys[1:] != keys[:-1]) + 1
    starts = np.concatenate([[0], brk])
    seg = np.maximum.reduceat(vals, starts)
    k = keys[starts]
    ok = k >= 0
    if ok.any():
        np.maximum.at(out_flat, k[ok], seg[ok])


def _grid_index(v: np.ndarray, lo: float, hi: float, n: int) -> np.ndarray:
    """Bin index of v on [lo, hi) with n bins; −1 outside."""
    i = np.floor((v - lo) / (hi - lo) * n).astype(np.int64)
    return np.where((i >= 0) & (i < n), i, -1)


def _centres(lo: float, hi: float, n: int) -> np.ndarray:
    return lo + (np.arange(n) + 0.5) * (hi - lo) / n


# ---------------------------------------------------------------------------
# FAM — the FFT Accumulation Method
# ---------------------------------------------------------------------------
def _channelise(x: np.ndarray, np_: int, starts: np.ndarray, win: np.ndarray
                ) -> np.ndarray:
    """X[p, k]: the windowed Np-point FFT of the block at starts[p],
    converted to a COMPLEX DEMODULATE referenced to absolute time (the
    e^{−j2πk·s_p/Np} factor). Skip that factor and the surface smears along
    α by a channel-dependent tilt."""
    idx = starts[:, None] + np.arange(np_)[None, :]
    X = np.fft.fft(x[idx] * win[None, :], axis=1)
    kk = np.arange(np_)
    X *= np.exp(-2j * np.pi * np.outer(starts % np_, kk) / np_)
    return X


def fam_plan(n: int, fs: float, channel_fft: int = 64, hop: int | None = None,
             max_blocks: int = 8192) -> dict:
    """Channelizer size, hop, block count and the resolutions they give."""
    np_ = int(channel_fft)
    hop = int(hop or max(1, np_ // 4))
    blocks = max(0, (int(n) - np_) // hop + 1)
    p = int(min(blocks, int(max_blocks)))
    return {"np": np_, "hop": hop, "p": p, "available_blocks": blocks,
            "df_hz": float(fs) / np_,
            "dalpha_hz": float(fs) / (hop * max(1, p)),
            "points": float(np_) * np_ * max(1, p),
            "seconds_used": (np_ + hop * max(0, p - 1)) / float(fs)}


def fam_cost(n: int, fs: float, channel_fft: int = 64, hop: int | None = None,
             max_blocks: int = 8192) -> str:
    """The sentence shown BEFORE the run (a wait you were told about is a
    cost; a wait you were not is a fault report — the bench's idiom)."""
    pl = fam_plan(n, fs, channel_fft, hop, max_blocks)
    pts = pl["points"]
    secs = pts / 4e7        # ~40 M correlation points a second, one core, numpy
    note = (f"FAM surface: about {pts:,.0f} correlation points "
            f"(~{secs:,.1f} s on one core).")
    if pl["available_blocks"] > pl["p"]:
        note += (f" The record holds {pl['available_blocks']:,} blocks; "
                 f"{pl['p']:,} are used ({pl['seconds_used']:.3f} s).")
    return (f"{note} Δf ≈ {pl['df_hz']:,.1f} Hz, "
            f"Δα ≈ {pl['dalpha_hz']:,.3f} Hz.")


def _fam_pairs(x: np.ndarray, fs: float, np_: int, hop: int, win_kind: str,
               conj: bool, seg_blocks: int, n_segments: int):
    """The FAM core. Yields, for each first channel k1:
        (f [Np], alpha [Q, Np], mag [Q, Np], chan_power [Np])
    with |S| averaged (incoherently) over `n_segments` segments of
    `seg_blocks` blocks spread evenly across x. One segment = the textbook
    coherent FAM; several = the same surface with its variance reduced,
    at a coarser α resolution (used by `scf_image`)."""
    seg_len = np_ + hop * (seg_blocks - 1)
    avail = x.size - seg_len
    if avail < 0:
        return
    n_seg = int(max(1, min(n_segments, avail // seg_len + 1)))
    seg_starts = (np.linspace(0, avail, n_seg).astype(np.int64)
                  if n_seg > 1 else np.zeros(1, np.int64))
    win = window(win_kind, np_)
    starts = (seg_starts[:, None]
              + hop * np.arange(seg_blocks)[None, :]).ravel()
    X = _channelise(x, np_, starts, win).reshape(n_seg, seg_blocks, np_)
    g = np.hamming(seg_blocks) if seg_blocks > 2 else np.ones(seg_blocks)
    g = g / g.sum()
    chan_power = np.mean(np.einsum("p,spk->sk", g, np.abs(X) ** 2), axis=0)
    pfft = int(seg_blocks)
    aq = np.fft.fftshift(np.fft.fftfreq(pfft, hop / float(fs)))
    df = float(fs) / np_
    keep = np.abs(aq) <= 0.5 * df * (1.0 + 1e-9)
    aq_k = aq[keep]
    fk = np.fft.fftfreq(np_, 1.0 / float(fs))
    Xc = X if conj else np.conj(X)
    for k1 in range(np_):
        prod = X[:, :, k1][:, :, None] * Xc            # [S, P, Np]
        prod *= g[None, :, None]
        F = np.fft.fftshift(np.fft.fft(prod, axis=1), axes=1)[:, keep, :]
        mag = np.abs(F).mean(axis=0) if n_seg > 1 else np.abs(F[0])
        if conj:
            alpha = (fk[k1] + fk)[None, :] + aq_k[:, None]
            f = 0.5 * (fk[k1] - fk)
        else:
            alpha = (fk[k1] - fk)[None, :] + aq_k[:, None]
            f = 0.5 * (fk[k1] + fk)
        yield k1, f, alpha, mag, chan_power


def fam(x, fs: float, channel_fft: int = 64, hop: int | None = None,
        win: str = "hamming", conj: bool = False,
        alpha_max: float | None = None, normalized: bool = False,
        max_blocks: int = 8192, n_alpha: int | None = None) -> ScfSurface:
    """The blind surface S(f, α) by FAM. See `fam_cost` before calling it on
    a long record. `normalized=True` returns the spectral COHERENCE (0..1,
    gain-free); False returns |S|.

    Resolution: Δf = fs/Np from the channelizer, Δα = fs/(hop·P) from the
    record; the α grid is at the natural Δα unless that would exceed 4096
    rows, in which case rows are max-pooled (a line keeps its height)."""
    x = _as_complex(x)
    fs = float(fs)
    pl = fam_plan(x.size, fs, channel_fft, hop, max_blocks)
    np_, hop_, p = pl["np"], pl["hop"], pl["p"]
    method = "FAM (FFT accumulation method)" + (", conjugate" if conj else "")
    if p < 8:
        return _empty(conj, method, pl["df_hz"], pl["dalpha_hz"],
                      "record too short for a FAM surface (fewer than 8 "
                      "blocks)")
    x = x - np.mean(x)
    a_max = float(alpha_max) if alpha_max else fs
    n_a = int(n_alpha or min(4096, max(64, int(np.ceil(2 * a_max
                                                         / pl["dalpha_hz"])))))
    n_f = np_
    out = np.zeros((n_a, n_f))
    flat = out.reshape(-1)
    for k1, f, alpha, mag, cp in _fam_pairs(x, fs, np_, hop_, win, conj,
                                            p, 1):
        if normalized:
            mag = mag / np.sqrt(cp[k1] * cp + 1e-300)[None, :]
        ai = _grid_index(alpha, -a_max, a_max, n_a)
        fi = _grid_index(f, -fs / 2, fs / 2, n_f)
        keys = np.where(ai >= 0, ai * n_f + np.clip(fi, 0, n_f - 1)[None, :],
                        -1)
        _reduce_max(flat, keys.T.ravel(), mag.T.ravel())
    return ScfSurface(
        alphas=_centres(-a_max, a_max, n_a), freqs=_centres(-fs / 2, fs / 2, n_f),
        magnitude=out, conj=bool(conj), method=method, df_hz=pl["df_hz"],
        dalpha_hz=pl["dalpha_hz"], normalized=bool(normalized),
        note=(f"Np={np_} channels, hop={hop_}, P={p} blocks "
              f"({pl['seconds_used']:.3f} s) — {pl['points']:,.0f} "
              "correlation points"),
        extra={"np": np_, "hop": hop_, "p": p, "window": win})


# ---------------------------------------------------------------------------
# SSCA — the Strip Spectral Correlation Analyzer
# ---------------------------------------------------------------------------
def ssca_plan(n: int, fs: float, strips: int = 32,
              max_samples: int = 1 << 16) -> dict:
    s = int(strips)
    m = int(min(int(n), int(max_samples)))
    return {"strips": s, "n": m, "df_hz": float(fs) / s,
            "dalpha_hz": float(fs) / max(1, m), "points": float(s) * m}


def ssca_cost(n: int, fs: float, strips: int = 32,
              max_samples: int = 1 << 16) -> str:
    pl = ssca_plan(n, fs, strips, max_samples)
    mem = pl["points"] * 16 / 1e6
    return (f"SSCA surface: about {pl['points']:,.0f} strip points, about "
            f"{mem:,.0f} MB of working memory. Δf ≈ {pl['df_hz']:,.1f} Hz, "
            f"Δα ≈ {pl['dalpha_hz']:,.4f} Hz.")


def _ssca_strips(x: np.ndarray, fs: float, strips: int, conj: bool,
                 win_kind: str):
    """Yields (fk, alpha [n], f [n], |S| [n], strip_power) per strip, with q
    in ascending α order."""
    n = x.size
    w = window(win_kind, strips)
    pad = np.concatenate([np.zeros(strips // 2, complex), x,
                          np.zeros(strips, complex)])
    frames = np.lib.stride_tricks.sliding_window_view(pad, strips)[:n]
    X = np.fft.fft(frames * w[None, :], axis=1)            # [n, strips]
    ref = x if conj else np.conj(x)
    fk = np.fft.fftfreq(strips, 1.0 / fs)
    fq = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / fs))
    nn = np.arange(n)
    p_strip = np.mean(np.abs(X) ** 2, axis=0) + 1e-300
    for k in range(strips):
        col = X[:, k] * ref * np.exp(-2j * np.pi * k * (nn % strips) / strips)
        S = np.fft.fftshift(np.fft.fft(col)) / n
        yield (fk[k], fk[k] + fq, 0.5 * (fk[k] - fq), np.abs(S), p_strip[k])


def ssca(x, fs: float, strips: int = 32, conj: bool = False,
         max_samples: int = 1 << 16, alpha_max: float | None = None,
         normalized: bool = False, win: str = "hann",
         n_alpha: int | None = None) -> ScfSurface:
    """The fine-α blind surface. Refuses rather than exhausting memory.

    FAM and SSCA trade the opposite way: FAM is coarse in α and cheap; SSCA
    resolves α to fs/N — two emitters at 9600 and 9615 Bd are one smear to
    FAM and two lines to SSCA. The mapping is α = f_k + f_q, f = (f_k − f_q)/2
    for both the non-conjugate and the conjugate surface."""
    x = _as_complex(x)
    fs = float(fs)
    pl = ssca_plan(x.size, fs, strips, max_samples)
    method = "SSCA (strip spectral correlation analyzer)" + (
        ", conjugate" if conj else "")
    if pl["points"] > SSCA_MAX_POINTS:
        return _empty(conj, method, pl["df_hz"], pl["dalpha_hz"],
                      f"refused: {pl['points']:,.0f} strip points would need "
                      f"about {pl['points'] * 16 / 1e9:,.2f} GB. Shorten the "
                      "record (max_samples) or use fewer strips.")
    x = x[: pl["n"]]
    if x.size < 4 * int(strips):
        return _empty(conj, method, pl["df_hz"], pl["dalpha_hz"],
                      "record too short for an SSCA surface")
    x = x - np.mean(x)
    p_ref = float(np.mean(np.abs(x) ** 2)) + 1e-300
    a_max = float(alpha_max) if alpha_max else fs
    n_a = int(n_alpha or min(4096, max(128, int(2 * a_max / pl["dalpha_hz"]
                                                / 8))))
    n_f = int(strips)
    out = np.zeros((n_a, n_f))
    flat = out.reshape(-1)
    for _fk, alpha, f, mag, ps in _ssca_strips(x, fs, int(strips), conj, win):
        if normalized:
            mag = mag / np.sqrt(ps * p_ref)
        ai = _grid_index(alpha, -a_max, a_max, n_a)
        fi = np.clip(_grid_index(f, -fs / 2, fs / 2, n_f), 0, n_f - 1)
        keys = np.where(ai >= 0, ai * n_f + fi, -1)
        _reduce_max(flat, keys, mag)
    return ScfSurface(
        alphas=_centres(-a_max, a_max, n_a), freqs=_centres(-fs / 2, fs / 2, n_f),
        magnitude=out, conj=bool(conj), method=method, df_hz=pl["df_hz"],
        dalpha_hz=pl["dalpha_hz"], normalized=bool(normalized),
        note=f"{strips} strips × {x.size:,} samples",
        extra={"strips": int(strips), "samples": int(x.size), "window": win})


# ---------------------------------------------------------------------------
# The cyclic domain profile — max over f of |S(f, α)|
# ---------------------------------------------------------------------------
def alpha_profile(surface: ScfSurface) -> tuple:
    """(alphas, profile) — a surface collapsed over frequency by its MAXIMUM.

    Maximum, not mean: a cyclic feature is confined to the band the signal
    occupies, so averaging it against a wide empty span buries it in
    proportion to how narrow the signal is — exactly backwards, since a
    narrow signal in a wide capture is the case that most needs help."""
    if surface.magnitude.size == 0:
        return np.zeros(0), np.zeros(0)
    return surface.alphas, np.max(surface.magnitude, axis=1)


def cyclic_profile(x, fs: float, method: str = "fam", conj: bool = False,
                   alpha_max: float | None = None, channel_fft: int = 64,
                   hop: int | None = None, win: str = "hamming",
                   max_blocks: int = 8192, normalized: bool = False,
                   n_alpha: int | None = None, strips: int = 32,
                   max_samples: int = 1 << 16) -> tuple:
    """(alpha_axis_hz, profile): the cyclic domain profile, max over f of
    |S(f, α)| (or of the coherence when `normalized`), from FAM (default) or
    SSCA, non-conjugate or conjugate.

    The α axis runs over [−alpha_max, alpha_max] (default ±fs/2) at the
    method's natural resolution (FAM: fs/(hop·P); SSCA: fs/N), up to 2¹⁸
    points. The non-conjugate profile is symmetric in α (S^{−α} = conj S^α),
    so its α ≥ 0 half holds everything; the conjugate one is not, because
    the carrier can sit on either side of the cut's centre.

    Which peaks are REAL is a separate question with a derived answer: see
    `lag_profile` + `detection_threshold`, which `dsp.measure.cyclic_peaks`
    uses to label this curve."""
    x = _as_complex(x)
    fs = float(fs)
    a_max = float(alpha_max) if alpha_max else fs / 2.0
    if str(method).lower() == "ssca":
        pl = ssca_plan(x.size, fs, strips, max_samples)
        if pl["points"] > SSCA_MAX_POINTS or pl["n"] < 4 * strips:
            return np.zeros(0), np.zeros(0)
        xs = x[: pl["n"]]
        xs = xs - np.mean(xs)
        n_a = int(n_alpha or min(1 << 18, int(np.ceil(2 * a_max
                                                      / pl["dalpha_hz"]))))
        out = np.zeros(n_a)
        p_ref = float(np.mean(np.abs(xs) ** 2)) + 1e-300
        for _fk, alpha, _f, mag, ps in _ssca_strips(xs, fs, int(strips), conj,
                                                    "hann"):
            if normalized:
                mag = mag / np.sqrt(ps * p_ref)
            _reduce_max(out, _grid_index(alpha, -a_max, a_max, n_a), mag)
        return _centres(-a_max, a_max, n_a), out
    pl = fam_plan(x.size, fs, channel_fft, hop, max_blocks)
    if pl["p"] < 8:
        return np.zeros(0), np.zeros(0)
    xs = x - np.mean(x)
    n_a = int(n_alpha or min(1 << 18, int(np.ceil(2 * a_max
                                                  / pl["dalpha_hz"]))))
    out = np.zeros(n_a)
    for k1, _f, alpha, mag, cp in _fam_pairs(xs, fs, pl["np"], pl["hop"], win,
                                             conj, pl["p"], 1):
        if normalized:
            mag = mag / np.sqrt(cp[k1] * cp + 1e-300)[None, :]
        ai = _grid_index(alpha, -a_max, a_max, n_a)
        _reduce_max(out, ai.T.ravel(), mag.T.ravel())
    return _centres(-a_max, a_max, n_a), out


# ---------------------------------------------------------------------------
# The classifier's fixed-geometry SCF image (DETECTION_DESIGN §4)
# ---------------------------------------------------------------------------
def scf_image(x, fs: float, fam_geom, out_shape: tuple = (64, 128),
              conj: bool = False, max_blocks: int = 8192) -> tuple:
    """(img, f_axis_hz, alpha_axis_hz) — the SCF on a FIXED grid, the
    classifier's second input.

        img      float32 [H, W], normalised to 0..1 (the largest |S| is 1)
        rows     H frequencies over [−fs/2, fs/2)  (f_axis_hz: bin centres)
        columns  W cycle frequencies over [0, fs/2) non-conjugate, or
                 [−fs/2, fs/2) conjugate          (alpha_axis_hz: centres)

    THE GEOMETRY IS THE PROFILE'S (`fam_geom` = profiles.FamGeometry):
    channel_fft, hop, window and max_seconds come from the receiver profile
    and are written into every model card, so a classifier never meets an
    SCF of another geometry (DETECTION_DESIGN §4.1, "the sample-rate law
    seen from the detector's side"). Change the geometry and the model must
    be retrained — that is the point of fixing it.

    How the grid is filled: ONE coherent FAM over the first `max_seconds`
    of the cut (at most `max_blocks` blocks — the whole 2 s of a voice-class
    cut; the first ~55 ms of a 2.4 MS/s one), its fine (f, α) points
    MAX-pooled into the grid: a narrow cyclic line keeps its height, where
    a mean would dilute it by the pooling ratio. Coherent, because averaging
    the magnitudes of many short FAM segments reduces only their variance,
    not their bias — measured: 64-block segments left a self-noise floor at
    15 % of the spectral peak across the signal's band, as tall as a BPSK's
    symbol-rate feature, where the coherent surface's floor is 3 %. The
    α = 0 column is the power spectrum, so the image carries the PSD and the
    cyclic features on one scale, gain-free after normalisation. Cost:
    about Np² × max_blocks correlation points (≈ 34 M for Np = 64)."""
    x = _as_complex(x)
    fs = float(fs)
    H, W = int(out_shape[0]), int(out_shape[1])
    np_ = int(getattr(fam_geom, "channel_fft", 64))
    hop = int(getattr(fam_geom, "hop", max(1, np_ // 4)))
    win = str(getattr(fam_geom, "window", "hamming"))
    max_s = float(getattr(fam_geom, "max_seconds", 2.0))
    f_axis = _centres(-fs / 2, fs / 2, H)
    a_lo, a_hi = ((-fs / 2, fs / 2) if conj else (0.0, fs / 2))
    a_axis = _centres(a_lo, a_hi, W)
    img = np.zeros((H, W))
    xs = x[: max(np_, int(round(max_s * fs)))]
    pl = fam_plan(xs.size, fs, np_, hop, max_blocks)
    if pl["p"] < 8:
        return img.astype(np.float32), f_axis, a_axis
    xs = xs - np.mean(xs)
    flat = img.reshape(-1)          # [H, W] → key = fi * W + ai
    for _k1, f, alpha, mag, _cp in _fam_pairs(xs, fs, np_, hop, win, conj,
                                              pl["p"], 1):
        ai = _grid_index(alpha, a_lo, a_hi, W)
        fi = np.clip(_grid_index(f, -fs / 2, fs / 2, H), 0, H - 1)
        # key = fi*W + ai; within a column (k2) fi is constant and ai rises
        keys = np.where(ai >= 0, fi[None, :] * W + ai, -1)
        _reduce_max(flat, keys.T.ravel(), mag.T.ravel())
    top = float(img.max())
    if top > 0:
        img = img / top
    return img.astype(np.float32), f_axis, a_axis


# ---------------------------------------------------------------------------
# Lag domain — Bill's α-profile with a known null distribution
# ---------------------------------------------------------------------------
def caf(x, fs: float, alpha: float, max_lag: int = 64,
        conj: bool = False) -> np.ndarray:
    """The cyclic autocorrelation at ONE cycle frequency, lags 0..max_lag:

        non-conjugate:  R(α, τ) = ⟨ x[n+τ] · conj(x[n]) · e^{−j2παn/fs} ⟩
        conjugate:      R(α, τ) = ⟨ x[n+τ] · x[n]       · e^{−j2παn/fs} ⟩

    The targeted, cheap direction: when α is already suspected — a known
    baud, twice a measured carrier — this answers in O(N·L) with no
    surface at all."""
    x = _as_complex(x)
    n = x.size
    lags = int(max(1, max_lag))
    if n < 2 * lags + 8:
        return np.zeros(lags + 1, dtype=np.complex128)
    ph = np.exp(-2j * np.pi * float(alpha) * np.arange(n) / float(fs))
    out = np.zeros(lags + 1, dtype=np.complex128)
    for t in range(lags + 1):
        a = x[t:]
        b = x[: n - t] if conj else np.conj(x[: n - t])
        out[t] = np.mean(a * b * ph[: n - t])
    return out


def lag_profile(x, fs: float, lags=None, conj: bool = False,
                alpha_max: float | None = None) -> tuple:
    """(alphas_hz, statistic) — the α-profile, one FFT per lag.

    The statistic is N·|R(α,τ)|²/R(0,0)², maximised over the lags, which for
    stationary noise is the maximum of exponential(1) variates — so a value
    of 12 is not "quite big", it is e⁻¹² unlikely per bin, and
    `detection_threshold` turns a false-alarm rate and a bin count into the
    number to compare against.

    Windowed (Blackman-Harris) before the α transform, as on the bench: the
    α = 0 term is the signal's total power, four to five orders above
    everything else, and with a rectangular window its 1/α sidelobes
    manufactured a 146 Hz "symbol rate" on a 600 Bd QPSK. Uses the whole
    record up to 2²⁰ samples (the bench truncated at 2¹⁶)."""
    x = _as_complex(x)
    n = int(min(x.size, LAG_PROFILE_MAX))
    if n < 256:
        return np.zeros(0), np.zeros(0)
    x = x[:n] - np.mean(x[:n])
    power = float(np.mean(np.abs(x) ** 2)) or 1.0
    if lags is None:
        top = max(4, min(n // 8, 256))
        lags = np.concatenate([[0], np.unique(np.geomspace(1, top, 12)
                                              .astype(int))])
    lags = np.asarray(sorted(set(int(t) for t in lags)), dtype=int)
    import scipy.fft as sfft
    m = int(sfft.next_fast_len(n))
    alphas = np.fft.fftshift(np.fft.fftfreq(m, 1.0 / float(fs)))
    best = np.zeros(m)
    for t in lags:
        if t >= n - 16:
            continue
        a = x[t:]
        b = x[: n - t] if conj else np.conj(x[: n - t])
        prod = a * b
        if not conj and t == 0:
            prod = prod - np.mean(prod)
        prod = prod * window("blackmanharris", prod.size)
        spec = np.abs(np.fft.fftshift(sfft.fft(prod, m))) ** 2
        best = np.maximum(best, spec * (float(n) / (float(prod.size) ** 2
                                                    * power ** 2)))
    if alpha_max:
        keep = np.abs(alphas) <= float(alpha_max)
        return alphas[keep], best[keep]
    return alphas, best


#: Median of the maximum of ~13 independent Exp(1) variates: `local_ratio`
#: divides by a local median, so this puts that ratio back on the
#: exponential scale the threshold is derived on (Bill's constant).
_MAX_L_MEDIAN = 2.96


def local_ratio(stat, span: int | None = None) -> np.ndarray:
    """Each α bin over the MEDIAN of its own neighbourhood — CFAR in α.

    The global floor is the wrong floor: the self-noise of an oversampled
    signal is not white in cycle frequency (it piles up inside the signal's
    own band), so a peak compared with the median of the WHOLE axis is
    compared with the empty part. On the bench a proper QPSK's conjugate
    self-noise measured 126 against the global floor and 7.6 against its own
    neighbourhood; a real BPSK carrier measured 851."""
    s = np.asarray(stat, dtype=float)
    n = s.size
    if n < 32:
        return np.ones(n)
    span = int(span) if span else max(16, n // 32)
    step = max(1, span // 4)
    centres = np.arange(0, n, step)
    floors = np.empty(centres.size)
    for i, c in enumerate(centres):
        seg = s[max(0, c - span): min(n, c + span + 1)]
        floors[i] = float(np.median(seg)) if seg.size else 1.0
    interp = np.interp(np.arange(n), centres, floors)
    return s / np.maximum(interp, 1e-12)


def detection_threshold(pfa: float, n_bins: int, n_lags: int = 13) -> float:
    """The local-ratio value a feature must clear for a false-alarm rate.

    Under a stationary null the statistic at one α and one lag is Exp(1),
    so P(stat > t) = e^{−t}; maximising over `n_lags` lags and searching
    `n_bins` cycle frequencies needs t = ln(bins·lags/pfa), divided by the
    same constant `local_ratio` divided by. Change `pfa` and the threshold
    moves the way the algebra says — the sensitivity claim is checkable."""
    trials = max(2.0, float(n_bins) * max(1, int(n_lags)))
    p = min(0.5, max(1e-15, float(pfa)))
    return float(np.log(trials / p) / _MAX_L_MEDIAN)


# ---------------------------------------------------------------------------
# Frequency smoothing — the SCF at one α; the reference implementation
# ---------------------------------------------------------------------------
def _spectrum(x: np.ndarray, nfft: int, win_kind: str = "hann") -> np.ndarray:
    seg = x[: min(x.size, nfft)]
    return np.fft.fft(seg * window(win_kind, seg.size), nfft)


def _shifted_pair(X: np.ndarray, alpha: float, fs: float, conj: bool):
    n = X.size
    shift = int(round(float(alpha) / (fs / n) / 2.0))
    up = np.roll(X, -shift)                      # X(f + α/2)
    if conj:
        rev = X[(-np.arange(n)) % n]             # X(−f)
        return up, np.roll(rev, shift)           # X(α/2 − f)
    return up, np.conj(np.roll(X, shift))        # X*(f − α/2)


def scf_at(x, fs: float, alpha: float, df: float | None = None,
           conj: bool = False, nfft: int = 4096) -> np.ndarray:
    """S^α(f) by frequency smoothing, fftshifted: X(f+α/2)·X*(f−α/2) (or the
    conjugate X(f+α/2)·X(α/2−f)) smoothed over `df` Hz. Smoothing is not
    cosmetic: without it this is a raw periodogram product whose variance
    never falls. At α = 0 non-conjugate it is the power spectrum."""
    x = _as_complex(x)
    fs = float(fs)
    n = int(nfft)
    X = _spectrum(x, n)
    up, dn = _shifted_pair(X, alpha, fs, conj)
    prod = up * dn
    width = max(1, int(round(float(df or fs / 64) / (fs / n))))
    if width > 1:
        prod = np.convolve(prod, np.ones(width) / width, mode="same")
    return np.fft.fftshift(prod) / n


def spectral_coherence(x, fs: float, alpha: float, df: float | None = None,
                       conj: bool = False, nfft: int = 4096) -> np.ndarray:
    """|S^α(f)| / sqrt(S⁰(f+α/2)·S⁰(f∓α/2)), 0..1 — the detection view,
    not the picture: the raw SCF is largest where the signal is loudest, the
    coherence says how locked the two spectral components are, on the same
    scale whatever the level."""
    x = _as_complex(x)
    fs = float(fs)
    n = int(nfft)
    X = _spectrum(x, n)
    up, dn = _shifted_pair(X, alpha, fs, conj)
    width = max(1, int(round(float(df or fs / 64) / (fs / n))))
    k = np.ones(width) / width

    def smooth(v):
        return np.convolve(v, k, mode="same") if width > 1 else v

    num = np.abs(smooth(up * dn))
    den = np.sqrt(np.abs(smooth(np.abs(up) ** 2))
                  * np.abs(smooth(np.abs(dn) ** 2))) + 1e-30
    return np.fft.fftshift(np.clip(num / den, 0.0, 1.0))


def scf_scan(x, fs: float, alphas, df: float | None = None,
             conj: bool = False, nfft: int = 4096,
             normalized: bool = True) -> ScfSurface:
    """A surface built by scanning α explicitly — slow and obviously correct.
    The fast methods are tested against it: a speed-up nobody checked
    against a definition is a speed-up nobody should trust (the bench's
    rule)."""
    x = _as_complex(x)
    fs = float(fs)
    alphas = np.asarray(list(alphas), dtype=float)
    fn = spectral_coherence if normalized else scf_at
    rows = [np.abs(fn(x, fs, float(a), df=df, conj=conj, nfft=nfft))
            for a in alphas]
    freqs = np.fft.fftshift(np.fft.fftfreq(int(nfft), 1.0 / fs))
    return ScfSurface(
        alphas=alphas, freqs=freqs,
        magnitude=(np.asarray(rows) if rows else np.zeros((0, int(nfft)))),
        conj=bool(conj), method="scanned SCF (frequency smoothing)",
        df_hz=float(df or fs / 64),
        dalpha_hz=(float(np.min(np.diff(alphas))) if alphas.size > 1
                   else fs / max(1, x.size)),
        normalized=bool(normalized),
        note=f"{alphas.size} cycle frequencies × {nfft} spectral bins")


# ---------------------------------------------------------------------------
# Peaks and harmonics (the bench's summary layer)
# ---------------------------------------------------------------------------
def profile_peaks(alphas, profile, threshold: float,
                  min_sep_hz: float = 0.0, count: int = 6) -> list:
    """Distinct peaks above `threshold`, tallest first: [(alpha, value)]."""
    alphas = np.asarray(alphas, dtype=float)
    work = np.asarray(profile, dtype=float).copy()
    out = []
    if alphas.size == 0:
        return out
    for _ in range(int(count)):
        k = int(np.argmax(work))
        if not np.isfinite(work[k]) or work[k] <= threshold:
            break
        out.append((float(alphas[k]), float(work[k])))
        if min_sep_hz > 0:
            work[np.abs(alphas - alphas[k]) <= min_sep_hz] = -np.inf
        else:
            work[k] = -np.inf
    return out


def fold_harmonics(peaks: list, tol: float = 0.03) -> tuple:
    """(fundamental, harmonic_orders) from a set of α peaks. A symbol clock
    puts lines at the rate AND its harmonics — 9 600, 19 200, 28 800 are one
    signal, not three; the fundamental is the smallest peak that explains
    the others as integer multiples."""
    vals = sorted(abs(a) for a, _v in peaks if abs(a) > 0)
    if not vals:
        return 0.0, []
    for base in vals:
        orders = []
        for v in vals:
            k = round(v / base)
            if k >= 1 and abs(v - k * base) <= tol * max(v, k * base):
                orders.append(int(k))
        if len(orders) >= 2 and max(orders) > 1:
            return float(base), sorted(set(orders))
    return float(vals[0]), [1]
