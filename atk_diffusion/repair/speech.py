# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Classical speech enhancement before Whisper — plan §4.D2, the classical
half.

    *"A clean before transcribe step on the DSD and PTT audio paths. The
    largest practical win in this plan and the cheapest; a first-wins
    item."*  — plan §4.D2

Three enhancers that have been the baselines of the field for forty years,
because a learned enhancer that does not beat them is not shipped (plan §7):

  * `spectral_subtraction` — Boll (1979), "Suppression of acoustic noise in
    speech using spectral subtraction", with Berouti, Schwartz & Makhoul's
    (1979) over-subtraction factor that falls as the frame's SNR rises
    (α = 4 − 0.15·SNR, clamped to 1…5) and a spectral floor (β = 0.02 of the
    noise) that keeps the residual noise from turning into "musical" tones.
  * `wiener` — the Wiener gain ξ/(1+ξ) with the DECISION-DIRECTED a-priori
    SNR of Ephraim & Malah (1984): ξ = 0.98·|Â(prev)|²/λ + 0.02·max(γ−1, 0),
    floored at −25 dB (Cappé 1994: the floor is what removes musical noise).
  * `mmse_lsa` — Ephraim & Malah (1985), "Speech enhancement using a
    minimum mean-square error log-spectral amplitude estimator":
    G = ξ/(1+ξ)·exp(½·E1(v)), v = ξγ/(1+ξ), with the same decision-directed ξ.

All three need the noise power in every frequency bin while somebody is
talking. That comes from MCRA — Cohen & Berdugo (2002), "Noise estimation by
minima controlled recursive averaging": the minimum of the smoothed power
over a sliding window says where speech is absent, and the noise estimate is
averaged only there. It needs no noise-only lead-in (a DSD clip begins when
the voice begins), adapts within about a second, and can be primed from a
noise-only clip when one exists.

TIER: CLEANED. Each enhancer is a gain between 0 and 1 applied per bin to the
recording's own spectrum: it removes, it never adds a component that was not
there (`provenance` — "nothing added, but this is not the record"). The plan
called D2's output Invented; that word belongs to the learned enhancer in
`speech_learned`, which can and does make speech-shaped sound from noise.

WHAT THIS CANNOT DO, plainly: help a transcript that noise has already
destroyed (a gain cannot restore a masked syllable), or treat non-stationary
interference (a second voice, a squelch tail) as noise — MCRA tracks the
stationary floor. Aggressive suppression can lower Whisper's accuracy even
while it raises the SNR; that is why `experiments.wer_eval` measures word
error rate, not SNR. Nothing here identifies a person or "enhances a voice"
for identification (plan §2.4): it reduces noise for transcription.

WAV I/O is the standard library's `wave` plus numpy: PCM 8/16/24/32-bit in,
16-bit PCM mono out (DSD writes 8 kHz; Whisper accepts anything).
"""

from __future__ import annotations

import json
import time
import wave
from pathlib import Path

import numpy as np

from atk_diffusion import provenance as _prov

METHODS = ("spectral_subtraction", "wiener", "mmse_lsa")

#: Plain names an operator sees.
METHOD_WORDS = {
    "spectral_subtraction": "spectral subtraction (Boll 1979, Berouti 1979)",
    "wiener": "Wiener filter, decision-directed SNR (Ephraim & Malah 1984)",
    "mmse_lsa": "MMSE log-spectral amplitude (Ephraim & Malah 1985)",
}

DD_ALPHA = 0.98          # decision-directed smoothing (Ephraim & Malah 1984)
XI_MIN_DB = -25.0        # a-priori SNR floor (Cappé 1994)
GAIN_FLOOR_DB = -20.0    # most a gain may attenuate one bin


# ---------------------------------------------------------------------------
# WAV in and out — standard library only
# ---------------------------------------------------------------------------
def read_wav(path) -> tuple[np.ndarray, int, dict]:
    """PCM WAV -> (mono float32 in [-1, 1), rate, info). Several channels are
    averaged to one (Whisper hears one); `info` says how many there were."""
    p = Path(path)
    try:
        with wave.open(str(p), "rb") as w:
            nch, sw, fs, nfr = (w.getnchannels(), w.getsampwidth(),
                                w.getframerate(), w.getnframes())
            raw = w.readframes(nfr)
    except wave.Error as e:
        raise ValueError(f"{p.name} is not a PCM WAV this reader understands "
                         f"({e}). Convert it to 16-bit PCM WAV first (ATK's "
                         "FFmpeg does this).") from None
    except FileNotFoundError:
        raise FileNotFoundError(f"audio file not found: {p}") from None
    if sw == 1:
        a = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 2:
        a = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        a = v.astype(np.float32) / float(1 << 23)
    elif sw == 4:
        a = np.frombuffer(raw, "<i4").astype(np.float64) / float(1 << 31)
        a = a.astype(np.float32)
    else:
        raise ValueError(f"{p.name}: {8 * sw}-bit samples are not supported")
    if nch > 1:
        a = a[: (a.size // nch) * nch].reshape(-1, nch).mean(axis=1)
    return a.astype(np.float32), int(fs), {"channels": nch, "bits": 8 * sw,
                                           "frames": nfr, "rate": int(fs)}


def write_wav(path, x, fs: int) -> dict:
    """mono float -> 16-bit PCM WAV. Returns {path, samples, clipped}: a
    clipped sample is counted and said, never hidden."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=np.float64).ravel()
    clipped = int(np.sum(np.abs(x) > 32767.0 / 32768.0))
    v = np.clip(np.round(x * 32768.0), -32768, 32767).astype("<i2")
    tmp = p.with_name(p.name + ".tmp")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(fs))
        w.writeframes(v.tobytes())
    tmp.replace(p)
    return {"path": str(p), "samples": int(x.size), "clipped": clipped}


# ---------------------------------------------------------------------------
# STFT with perfect reconstruction (square-root periodic Hann, 50 % overlap)
# ---------------------------------------------------------------------------
def frame_size(fs: float) -> int:
    """32 ms, rounded to a power of two: 256 at 8 kHz, 512 at 16 kHz."""
    return int(2 ** int(round(np.log2(0.032 * float(fs)))))


def _window(n: int) -> np.ndarray:
    return np.sqrt(0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n))


def stft(x, n_fft: int, hop: int | None = None) -> np.ndarray:
    """Frames padded by REFLECTION at both ends (zero padding would put
    all-zero frames first, and a noise tracker primed on a silent frame
    believes the noise is zero). `istft` trims the padding exactly."""
    hop = int(hop or n_fft // 2)
    x = np.asarray(x, dtype=np.float64)
    if x.size > n_fft + hop:
        pad = np.pad(x, (n_fft, n_fft + hop), mode="reflect")
    else:
        pad = np.pad(x, (n_fft, n_fft + hop), mode="wrap" if x.size else "constant")
    nfr = 1 + (pad.size - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(nfr)[:, None]
    return np.fft.rfft(pad[idx] * _window(n_fft)[None, :], axis=1)


def istft(X, n_fft: int, length: int, hop: int | None = None) -> np.ndarray:
    hop = int(hop or n_fft // 2)
    frames = np.fft.irfft(X, n=n_fft, axis=1) * _window(n_fft)[None, :]
    out = np.zeros(hop * (frames.shape[0] - 1) + n_fft)
    for i in range(frames.shape[0]):
        out[i * hop:i * hop + n_fft] += frames[i]
    return out[n_fft:n_fft + int(length)]


# ---------------------------------------------------------------------------
# Noise power per bin: MCRA (Cohen & Berdugo 2002)
# ---------------------------------------------------------------------------
def mcra(P, hop_s: float, alpha_s: float = 0.8, alpha_d: float = 0.95,
         alpha_p: float = 0.2, delta: float = 5.0, window_s: float = 1.0,
         init=None) -> tuple[np.ndarray, np.ndarray]:
    """Noise PSD per frame and bin from the power spectrogram `P` [frames,
    bins], and the speech-presence probability. The estimate used for frame
    l is built from frames before it. The paper's constants; the minimum is
    searched over `window_s` (1 s — shorter than the paper's so a five-second
    DSD clip still gets an estimate).

    Two departures, both for DSD's output: the estimate starts from the
    median of the first window's frames (÷ ln 2, the median of an
    exponential) rather than from one frame, and frames of exact digital
    silence — DSD writes zeros between overs — are skipped, because a
    minimum taken over silence is zero and would freeze the tracker."""
    P = np.asarray(P, dtype=np.float64)
    nfr, nb = P.shape
    eps = 1e-20
    Sf = P.copy()
    if nb >= 3:
        Sf[:, 1:-1] = 0.25 * P[:, :-2] + 0.5 * P[:, 1:-1] + 0.25 * P[:, 2:]
    L = max(8, int(round(float(window_s) / max(float(hop_s), 1e-6))))
    silent = P.sum(axis=1) <= 1e-30
    live = np.flatnonzero(~silent)
    if init is not None:
        lam = np.array(init, dtype=np.float64)
    elif live.size:
        lam = np.median(P[live[:L]], axis=0) / np.log(2.0)
    else:
        lam = np.full(nb, eps)
    lam = np.maximum(lam, eps)
    S = Sf[live[0]].copy() if live.size else np.full(nb, eps)
    Smin = S.copy()
    Stmp = S.copy()
    p = np.zeros(nb)
    out = np.empty_like(P)
    pres = np.empty_like(P)
    for l in range(nfr):
        out[l] = lam
        if silent[l]:
            pres[l] = p
            continue
        S = alpha_s * S + (1.0 - alpha_s) * Sf[l]
        if l > 0 and l % L == 0:
            Smin = np.minimum(Stmp, S)
            Stmp = S.copy()
        else:
            Smin = np.minimum(Smin, S)
            Stmp = np.minimum(Stmp, S)
        ind = (S / np.maximum(Smin, eps)) > delta
        p = alpha_p * p + (1.0 - alpha_p) * ind
        pres[l] = p
        ad = alpha_d + (1.0 - alpha_d) * p
        lam = np.maximum(ad * lam + (1.0 - ad) * P[l], eps)
    return out, pres


# ---------------------------------------------------------------------------
# The three enhancers
# ---------------------------------------------------------------------------
def _gains(P, lam, method: str, xi_min: float, g_min: float) -> np.ndarray:
    from scipy.special import exp1
    nfr = P.shape[0]
    G = np.empty_like(P)
    if method == "spectral_subtraction":
        snr = 10.0 * np.log10(np.maximum(P.sum(axis=1), 1e-20)
                              / np.maximum(lam.sum(axis=1), 1e-20))
        alpha = np.clip(4.0 - 0.15 * snr, 1.0, 5.0)[:, None]
        S2 = np.maximum(P - alpha * lam, 0.02 * lam)
        G = np.sqrt(S2 / np.maximum(P, 1e-20))
        return np.clip(G, g_min, 1.0)
    a2_prev = None
    lam_prev = None
    for l in range(nfr):
        gamma = np.minimum(P[l] / lam[l], 1e4)
        ml = np.maximum(gamma - 1.0, 0.0)
        if a2_prev is None:
            xi = np.maximum(ml, xi_min)
        else:
            xi = np.maximum(DD_ALPHA * a2_prev / lam_prev
                            + (1.0 - DD_ALPHA) * ml, xi_min)
        if method == "wiener":
            g = xi / (1.0 + xi)
        else:                                   # mmse_lsa
            v = np.maximum(xi * gamma / (1.0 + xi), 1e-10)
            g = xi / (1.0 + xi) * np.exp(0.5 * exp1(v))
        g = np.clip(g, g_min, 1.0)
        G[l] = g
        a2_prev = g * g * P[l]
        lam_prev = lam[l]
    return G


def enhance(x, fs: float, method: str = "mmse_lsa", noise=None,
            gain_floor_db: float = GAIN_FLOOR_DB,
            xi_min_db: float = XI_MIN_DB) -> tuple[np.ndarray, dict]:
    """Enhance mono audio. `noise` (optional) is a noise-only clip at the
    same rate that primes the noise estimate. Returns (y, info) with
    `info["tier"] == "cleaned"` and the numbers that say what was done."""
    if method not in METHODS:
        raise ValueError(f"unknown enhancer {method!r} — one of "
                         f"{', '.join(METHODS)}")
    x = np.asarray(x, dtype=np.float64).ravel()
    n_fft = frame_size(fs)
    hop = n_fft // 2
    X = stft(x, n_fft, hop)
    P = np.abs(X) ** 2
    init = None
    if noise is not None and np.size(noise) >= n_fft:
        init = np.mean(np.abs(stft(noise, n_fft, hop)) ** 2, axis=0)
    lam, pres = mcra(P, hop / float(fs), init=init)
    G = _gains(P, lam, method, 10 ** (xi_min_db / 10.0),
               10 ** (gain_floor_db / 20.0))
    y = istft(X * G, n_fft, x.size, hop)
    absent = pres < 0.2          # cells where MCRA says nobody is talking
    att = None
    if absent.any() and np.sum(P[absent]) > 0:
        att = float(10 * np.log10(max(np.sum(P[absent] * G[absent] ** 2), 1e-30)
                                  / np.sum(P[absent])))
    snr_in = float(10 * np.log10(max(np.sum(P), 1e-30) / max(np.sum(lam), 1e-30)))
    info = {"method": method, "words": METHOD_WORDS[method],
            "tier": _prov.tier_for(method), "rate": float(fs),
            "n_fft": n_fft, "hop": hop,
            "noise_tracker": "MCRA (Cohen & Berdugo 2002)"
                             + (", primed from a noise-only clip" if init is not None else ""),
            "params": {"gain_floor_db": gain_floor_db, "xi_min_db": xi_min_db,
                       "dd_alpha": DD_ALPHA},
            "speech_absent_fraction": float(np.mean(absent)),
            "attenuation_in_pauses_db": att,
            "posterior_snr_in_db": snr_in}
    return y.astype(np.float32), info


def enhance_wav(in_path, out_path, method: str = "mmse_lsa", noise_path=None,
                **kw) -> dict:
    """WAV -> enhanced WAV plus `<out>.json` beside it (tier, method, the
    source's hash). Writes nothing else."""
    x, fs, winfo = read_wav(in_path)
    noise = None
    if noise_path:
        noise, nfs, _ = read_wav(noise_path)
        if nfs != fs:
            raise ValueError(f"the noise clip is {nfs} Hz and the audio "
                             f"{fs} Hz; they must match")
    y, info = enhance(x, fs, method, noise=noise, **kw)
    w = write_wav(out_path, y, fs)
    side = {"source": str(in_path), "source_sha256": _prov.sha256_path(in_path),
            "output": str(out_path), "source_wav": winfo,
            "clipped_samples": w["clipped"],
            "provenance": _prov.stamp("repair.speech", method=method),
            "tier_words": _prov.TIER_WORDS[info["tier"]], **info}
    sp = Path(str(out_path) + ".json")
    sp.write_text(json.dumps(side, indent=2, default=str), encoding="utf-8")
    side["sidecar"] = str(sp)
    return side


def classical_enhancers(methods=METHODS) -> dict:
    """{name: callable(in_wav, out_wav) -> dict} for experiments.wer_eval."""
    def make(m):
        return lambda src, dst: enhance_wav(src, dst, m)
    return {m: make(m) for m in methods}


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------
def segmental_snr(clean, test, fs: float, frame_ms: float = 20.0,
                  lo: float = -10.0, hi: float = 35.0,
                  active_only: bool = True) -> float:
    """Segmental SNR (dB): the mean over frames of the per-frame SNR, each
    clamped to [lo, hi] (the usual convention). `active_only` scores only
    frames where the clean signal is within 40 dB of its loudest frame."""
    c = np.asarray(clean, dtype=np.float64).ravel()
    t = np.asarray(test, dtype=np.float64).ravel()
    n = min(c.size, t.size)
    L = max(1, int(round(frame_ms * 1e-3 * fs)))
    k = n // L
    if k == 0:
        return float("nan")
    C = c[:k * L].reshape(k, L)
    E = (c[:k * L] - t[:k * L]).reshape(k, L)
    pc = np.sum(C ** 2, axis=1)
    pe = np.sum(E ** 2, axis=1)
    snr = 10 * np.log10(np.maximum(pc, 1e-20) / np.maximum(pe, 1e-20))
    snr = np.clip(snr, lo, hi)
    if active_only:
        keep = pc > np.max(pc) * 1e-4
        snr = snr[keep] if keep.any() else snr
    return float(np.mean(snr))


def speech_like(fs: float, seconds: float, rng, f0=(110.0, 170.0)) -> np.ndarray:
    """A VOICED-SPEECH-LIKE test signal — not speech, and no person's voice:
    glottal pulse trains at a wandering pitch, through three formant
    resonators, in syllables of 120–260 ms separated by pauses of 80–300 ms.
    Used only to measure the enhancers here and in the experiments."""
    from scipy.signal import lfilter
    n = int(round(seconds * fs))
    out = np.zeros(n)
    t = 0
    while t < n:
        pause = int(rng.uniform(0.08, 0.30) * fs)
        syl = int(rng.uniform(0.12, 0.26) * fs)
        s0, s1 = t + pause, min(n, t + pause + syl)
        if s1 - s0 < 16:
            break
        m = s1 - s0
        pitch = rng.uniform(*f0) * (1 + 0.1 * np.sin(np.linspace(0, np.pi, m)))
        phase = np.cumsum(pitch / fs)
        pulses = (np.diff(np.floor(phase), prepend=0) > 0).astype(np.float64)
        src = lfilter([1.0], [1.0, -0.95], pulses)          # glottal tilt
        y = src
        for fc, bw in ((rng.uniform(450, 800), 90), (rng.uniform(1100, 1900), 110),
                       (rng.uniform(2300, min(3000, 0.45 * fs)), 160)):
            r = np.exp(-np.pi * bw / fs)
            th = 2 * np.pi * fc / fs
            y = lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], y)
        env = np.sin(np.linspace(0, np.pi, m)) ** 0.6
        seg = y * env
        out[s0:s1] = seg / (np.max(np.abs(seg)) + 1e-12)
        t = s1
    return (0.3 * out).astype(np.float32)


def add_noise(x, snr_db: float, rng, active_only: bool = True) -> np.ndarray:
    """White Gaussian noise at `snr_db` relative to the ACTIVE signal power."""
    x = np.asarray(x, dtype=np.float64)
    act = x[np.abs(x) > 1e-6] if active_only else x
    ps = float(np.mean(act ** 2)) if act.size else 0.0
    sigma = np.sqrt(ps / 10 ** (snr_db / 10.0)) if ps > 0 else 0.0
    return (x + sigma * rng.normal(size=x.size)).astype(np.float32)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
