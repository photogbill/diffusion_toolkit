# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Breathing rate, heart rate and breathing pauses from ESP32 Wi-Fi CSI —
the classical pipeline (plan §4.I1).

    RESEARCH-GRADE MEASUREMENT — NOT A MEDICAL DEVICE, NOT A DIAGNOSIS.

Every output of this module carries that sentence (`RESEARCH_LABEL`), in its
`label` field and in its lines. The plan is explicit: *"labeled plainly as a
measurement with research-grade accuracy, never a diagnosis and not a
medical device; it can sit beside Medical Support without being part of
it — a casualty's breathing, watched from across the room."* Plan principle
2.5: never medical facts.

THE PIPELINE, AS PULSEFI PUBLISHED IT (arXiv 2510.24744): amplitude-only
CSI → DC removed → band-pass 0.1–0.5 Hz for breathing and 0.8–2.17 Hz for the
heart → Savitzky–Golay smoothing → windows → (their LSTM; ours is
`learn.vitals`, beside this classical estimator). The choices the paper's
outline leaves open are made here and stated, so they can be checked:

* on the BREATHING path, each frame is first divided by its mean amplitude
  over the live subcarriers: the ESP32's AGC wanders at breathing
  frequencies and moves every subcarrier together, a breath does not (an
  addition to the published outline, switchable). Not on the heart path:
  the AGC's wander is mostly below 0.8 Hz, and part of a heartbeat's tiny
  movement is common to all subcarriers — dividing it out loses the beat;
* zero-phase 4th-order Butterworth band-passes (second-order sections — a
  0.1 Hz edge at tens of Hz is numerically fragile otherwise);
* Savitzky–Golay, cubic, over 1.0 s (breathing) and 0.25 s (heart);
* 30 s windows every 5 s — three breaths at the slowest rate in the band;
* SUBCARRIER SELECTION: subcarriers are ranked by how sharp their in-band
  spectral peak is, and the top 10 are combined by their first principal
  component (different subcarriers see the same chest motion with different
  signs — averaging them can cancel it, PCA does not);
* THE RATE is the spectral peak (Hann window, zero-padded, interpolated),
  CROSS-CHECKED by the autocorrelation's peak in the band's lag range. A
  window reports a rate only when the peak is a narrow line (not the hump
  band-passed noise makes), the rhythm repeats (autocorrelation high
  enough) and the two estimates agree; otherwise it reports NO rate and
  says why — the pipeline's "I don't know";
* the heart band is cleared around harmonics of the breathing rate before
  its peak is taken (an amplitude response to a 5 mm chest movement is not
  linear, and its harmonics land in the heart band);
* BREATHING PAUSES ("apnea events" in the research literature): the
  breathing signal's RMS envelope below 30 % of its running median for at
  least 10 s (flickers under 2 s bridged). A pause here is a measurement of
  a quiet signal — it is not a finding about a person.

PLACEMENT (the 2019 paper, arXiv 1908.05108, Fresnel-zone model). The pair's
Fresnel zones are ellipses with Tx and Rx at the foci; the n-th boundary is
where the reflected path is nλ/2 longer than the direct one. A small chest
movement changes the CSI amplitude most in the MIDDLE of a zone and least on
a BOUNDARY (where the reflected path is in or out of phase with the direct
one). `placement` says which zone the chest is in, how sensitive that spot
is, and how far to move it.

LIMITS, stated. Heart rate from amplitude-only CSI is far weaker than
breathing (a heartbeat moves the chest wall about a tenth as far); motion of
the person or anyone else in the room swamps both; the ESP32's AGC can step
inside a window. The synthetic experiment (`experiments.vitals_eval`)
measures the error under stated conditions; Bill's first experiment is his
own breathing against a count, three postures, one ESP32 pair.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from atk_diffusion import provenance

provenance.METHOD_TIERS.setdefault("csi_vitals", "measured")

METHOD = "csi_vitals"
TIER = provenance.tier_for(METHOD)
RESEARCH_LABEL = ("RESEARCH-GRADE MEASUREMENT from Wi-Fi CSI — not a medical "
                  "device and not a diagnosis.")
BREATH_BAND = (0.1, 0.5)      # Hz (PulseFi)
HEART_BAND = (0.8, 2.17)      # Hz (PulseFi)
SAVGOL_S = {"breath": 1.0, "heart": 0.25}
WINDOW_S = 30.0
HOP_S = 5.0
TOP_K = 10
#: A RHYTHM, not filtered noise. Band-passed noise has a hump of its own (the
#: filter's shape on a falling spectrum) and an autocorrelation that seems to
#: agree with it — the pipeline's way of hallucinating a pulse in an empty
#: room. So a window reports a rate only when (1) its spectral peak stands
#: MIN_PROMINENCE times above its NEIGHBOURHOOD in the band (a narrow line,
#: not a hump), (2) the autocorrelation at that period is at least MIN_R, and
#: (3) the two estimates agree. The numbers are DERIVED from the NOISE-ONLY
#: distribution of the synthetic experiment's empty-room windows, not tuned
#: for accuracy: the autocorrelation thresholds at its 99th percentile
#: (0.64 breathing, 0.51 heart), the prominence thresholds near its 95th
#: (36 and 24); the conjunction with the agreement test is stricter than
#: either. They are re-measured on Bill's own empty room —
#: `experiments.vitals_eval` reports the empty-room report rate.
MIN_PROMINENCE = {"breath": 30.0, "heart": 25.0}
MIN_R = {"breath": 0.65, "heart": 0.5}
AGREE_BPM = {"breath": 2.0, "heart": 6.0}
C = 299_792_458.0


# ---------------------------------------------------------------------------
# The steps
# ---------------------------------------------------------------------------
def amplitude(H) -> np.ndarray:
    """|CSI|, float32. Accepts complex CSI or amplitudes already."""
    X = np.asarray(H)
    return (np.abs(X) if np.iscomplexobj(X) else X).astype(np.float32)


def normalise_agc(A) -> np.ndarray:
    """Divide each frame by its mean amplitude over the live subcarriers. The
    ESP32's AGC moves every subcarrier together; a breath moves them with
    different signs and sizes, so this removes the gain and keeps the
    breath. (An addition to the published pipeline, stated: without it a
    wandering gain fills the breathing band of an empty room.)"""
    A = np.asarray(A, dtype=np.float64)
    prof = A.mean(axis=0)
    live = prof > 0.05 * (prof.max() if prof.size else 0.0)
    if not np.any(live):
        return A
    g = A[:, live].mean(axis=1, keepdims=True)
    return A / np.where(g > 0, g, 1.0)


def remove_dc(A, axis: int = 0) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    return A - A.mean(axis=axis, keepdims=True)


def bandpass(x, fs: float, band: tuple, order: int = 4, axis: int = 0) -> np.ndarray:
    """Zero-phase Butterworth band-pass in second-order sections."""
    from scipy.signal import butter, sosfiltfilt
    lo, hi = float(band[0]), float(band[1])
    nyq = 0.5 * float(fs)
    if hi >= nyq:
        raise ValueError(f"a {hi:g} Hz band edge needs CSI faster than "
                         f"{2 * hi:g} Hz; this is {fs:g} Hz")
    sos = butter(order, [lo, hi], btype="bandpass", fs=float(fs), output="sos")
    x = np.asarray(x, dtype=np.float64)
    padlen = min(x.shape[axis] - 1, 3 * (2 * order + 1) * 10)
    return sosfiltfilt(sos, x, axis=axis, padlen=max(0, padlen))


def savgol(x, fs: float, window_s: float, order: int = 3, axis: int = 0) -> np.ndarray:
    from scipy.signal import savgol_filter
    n = int(round(float(window_s) * float(fs)))
    n = max(order + 2, n | 1)              # odd, longer than the order
    x = np.asarray(x, dtype=np.float64)
    if x.shape[axis] <= n:
        return x
    return savgol_filter(x, n, order, axis=axis)


def _spectrum(x, fs: float, pad: int = 8):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    nfft = 1 << int(math.ceil(math.log2(max(16, pad * n))))
    P = np.abs(np.fft.rfft(x * np.hanning(n), nfft)) ** 2
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    return f, P


def spectral_rate(x, fs: float, band: tuple, reject: tuple = (),
                  reject_hz: float = 0.04, min_prominence: float = 0.0
                  ) -> tuple[float | None, float]:
    """(rate per minute, LOCAL prominence). The band's spectral peak (Hann,
    zero-padded, interpolated); `reject` frequencies (± reject_hz) are
    cleared first. The prominence is the peak over the median of its
    neighbourhood in the band, outside the window's main lobe — a narrow line
    scores high, a filter-shaped hump of noise does not. None when the
    prominence is under `min_prominence`."""
    x = np.asarray(x, dtype=np.float64)
    f, P = _spectrum(x, fs)
    sel = (f >= band[0]) & (f <= band[1])
    if not np.any(sel):
        return None, 0.0
    Pb = P.copy()
    for r in reject:
        Pb[np.abs(f - r) <= reject_hz] = 0.0
    idx = np.flatnonzero(sel)
    i = idx[int(np.argmax(Pb[idx]))]
    T = x.size / float(fs)
    lobe = 1.5 * 2.0 / T                       # Hann main lobe, with margin
    reach = 0.3 * (band[1] - band[0]) + lobe
    near = sel & (np.abs(f - f[i]) > lobe) & (np.abs(f - f[i]) <= reach) & (Pb > 0)
    ref = float(np.median(P[near])) if np.any(near) else float(np.median(P[idx]))
    prom = float(Pb[i] / ref) if ref > 0 else 0.0
    frac = 0.0
    if 0 < i < P.size - 1:
        a, b, c = Pb[i - 1], Pb[i], Pb[i + 1]
        den = a - 2 * b + c
        if den != 0:
            frac = 0.5 * (a - c) / den
    fpk = (i + max(-0.5, min(0.5, frac))) * (f[1] - f[0])
    if prom < min_prominence or Pb[i] <= 0:
        return None, prom
    return 60.0 * float(fpk), prom


def autocorr_rate(x, fs: float, band: tuple, min_r: float = 0.2
                  ) -> tuple[float | None, float]:
    """(rate per minute, r): the earliest strong autocorrelation peak in the
    band's lag range, and its normalised height. None when r is under
    `min_r`."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    if n < 8 or not np.any(x):
        return None, 0.0
    nfft = 1 << int(math.ceil(math.log2(2 * n)))
    r = np.fft.irfft(np.abs(np.fft.rfft(x, nfft)) ** 2)[:n]
    r = r / r[0]
    lo = max(1, int(math.floor(fs / band[1])))
    hi = min(n - 2, int(math.ceil(fs / band[0])))
    if hi <= lo:
        return None, 0.0
    seg = r[lo:hi + 1]
    # the EARLIEST local peak within 90% of the best: a rhythm also peaks at
    # two and three periods, and taking the highest picks a sub-harmonic
    peaks = [j for j in range(1, seg.size - 1)
             if seg[j] >= seg[j - 1] and seg[j] >= seg[j + 1]]
    if not peaks:
        peaks = [int(np.argmax(seg))]
    best = float(max(seg[j] for j in peaks))
    if best < min_r or best <= 0:
        return None, best
    i = lo + next(j for j in peaks if seg[j] >= 0.9 * best)
    ri = float(r[i])
    if ri < min_r:
        return None, ri
    frac = 0.0
    if 0 < i < n - 1:
        a, b, c = r[i - 1], r[i], r[i + 1]
        den = a - 2 * b + c
        if den != 0:
            frac = 0.5 * (a - c) / den
    lag = (i + max(-0.5, min(0.5, frac))) / fs
    return (60.0 / lag if lag > 0 else None), ri


def select_subcarriers(X, fs: float, band: tuple, k: int = TOP_K) -> np.ndarray:
    """Indices of the `k` subcarriers with the sharpest in-band peak (peak over
    the band's median power). Dead and null subcarriers fall out: no
    variance, no peak."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        return np.array([0])
    Z = X - X.mean(axis=0)
    live = Z.std(axis=0) > 0
    n = Z.shape[0]
    nfft = 1 << int(math.ceil(math.log2(max(16, 2 * n))))
    P = np.abs(np.fft.rfft(Z * np.hanning(n)[:, None], nfft, axis=0)) ** 2
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    sel = (f >= band[0]) & (f <= band[1])
    score = np.zeros(X.shape[1])
    if np.any(sel):
        Pb = P[sel]
        med = np.median(Pb, axis=0)
        ok = live & (med > 0)
        score[ok] = Pb.max(axis=0)[ok] / med[ok]
    order = np.argsort(-score)
    n_ok = int(np.sum(score > 0))
    return order[:max(1, min(int(k), n_ok or 1))]


def combine(X) -> np.ndarray:
    """The first principal component of the selected subcarriers (from the
    k x k covariance — cheap at any recording length)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1 or X.shape[1] == 1:
        return X.ravel()
    Z = X - X.mean(axis=0)
    sd = Z.std(axis=0)
    Z = Z / np.where(sd > 0, sd, 1.0)
    w, V = np.linalg.eigh(Z.T @ Z)
    return Z @ V[:, -1]


def windows(n: int, fs: float, win_s: float = WINDOW_S,
            hop_s: float = HOP_S) -> list[tuple[int, int]]:
    w = int(round(win_s * fs))
    h = max(1, int(round(hop_s * fs)))
    if n < w:
        return [(0, n)] if n >= int(10 * fs) else []
    return [(a, a + w) for a in range(0, n - w + 1, h)]


def apnea_events(breath, fs: float, *, min_pause_s: float = 10.0,
                 drop: float = 0.3, env_s: float = 4.0,
                 baseline_s: float = 60.0, bridge_s: float = 2.0) -> list[dict]:
    """Breathing pauses: the breathing signal's RMS envelope below `drop` x
    its running median for at least `min_pause_s`. Each event carries the
    research label."""
    from scipy.ndimage import median_filter, uniform_filter1d
    x = np.asarray(breath, dtype=np.float64)
    if x.size < int(2 * min_pause_s * fs):
        return []
    env = np.sqrt(uniform_filter1d(x * x, max(1, int(env_s * fs))))
    base = median_filter(env, size=max(3, int(baseline_s * fs)), mode="nearest")
    low = env < drop * base
    # bridge flickers shorter than `bridge_s`: one pause, not three
    gap = int(bridge_s * fs)
    if gap > 0 and low.any():
        idx = np.flatnonzero(low)
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= gap:
                low[a:b] = True
    out = []
    i = 0
    n = low.size
    while i < n:
        if not low[i]:
            i += 1
            continue
        j = i
        while j < n and low[j]:
            j += 1
        dur = (j - i) / fs
        if dur >= min_pause_s:
            depth = float(np.mean(env[i:j]) / max(np.mean(base[i:j]), 1e-12))
            out.append({"t0_s": i / fs, "t1_s": j / fs, "duration_s": dur,
                        "relative_level": depth,
                        "words": (f"breathing signal quiet for {dur:.0f} s from "
                                  f"{i / fs:.0f} s ({100 * depth:.0f}% of its "
                                  "usual level)"),
                        "label": RESEARCH_LABEL, "tier": TIER})
        i = j
    return out


# ---------------------------------------------------------------------------
# The estimate
# ---------------------------------------------------------------------------
@dataclass
class VitalsReport:
    fs: float
    windows: list = field(default_factory=list)
    breath_bpm: float | None = None
    heart_bpm: float | None = None
    apnea: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    label: str = RESEARCH_LABEL
    tier: str = TIER
    method: str = METHOD

    def lines(self) -> list[str]:
        out = [self.label]
        usable = [w for w in self.windows if w.get("breath_bpm") is not None]
        out.append(f"breathing: {self.breath_bpm:.1f} per minute (median of "
                   f"{len(usable)} windows)" if self.breath_bpm is not None
                   else "breathing: no clear rhythm — no rate is reported")
        heart = [w for w in self.windows if w.get("heart_bpm") is not None]
        out.append(f"heart: {self.heart_bpm:.0f} per minute (median of "
                   f"{len(heart)} windows; weaker than breathing — read with care)"
                   if self.heart_bpm is not None
                   else "heart: no clear rhythm — no rate is reported")
        out.append(f"breathing pauses of 10 s or more: {len(self.apnea)}")
        out += self.notes
        return out

    def to_json(self) -> dict:
        d = asdict(self)
        d["lines"] = self.lines()
        return d


def _rate(x, fs, kind, reject=()):
    """-> (rate or None, the spectral estimate, the autocorrelation
    estimate, agree, prominence, r, why-not). A rate is reported only when
    both estimates pass their tests and agree."""
    band = BREATH_BAND if kind == "breath" else HEART_BAND
    word = "breathing" if kind == "breath" else "heart"
    spec, prom = spectral_rate(x, fs, band, reject=reject,
                               min_prominence=MIN_PROMINENCE[kind])
    ac, r = autocorr_rate(x, fs, band, min_r=MIN_R[kind])
    agree = None
    if spec is not None and ac is not None:
        agree = abs(spec - ac) <= max(AGREE_BPM[kind], 0.1 * spec)
    if spec is None and ac is None:
        why = f"no clear {word} rhythm"
    elif spec is None:
        why = f"no clear {word} line in the spectrum"
    elif ac is None:
        why = f"the {word} rhythm does not repeat (autocorrelation {r:.2f})"
    elif not agree:
        why = (f"{word}: the spectrum ({spec:.1f}) and the autocorrelation "
               f"({ac:.1f}) disagree")
    else:
        why = ""
    return (spec if agree else None), spec, ac, agree, prom, r, why


def estimate(H, fs: float, *, win_s: float = WINDOW_S, hop_s: float = HOP_S,
             k: int = TOP_K, gaps: list | None = None,
             apnea: bool = True, agc_normalise: bool = True) -> VitalsReport:
    """The classical pipeline over a recording on a uniform grid (`csi.
    resample_uniform` makes one). `gaps` (seconds, from that function) mark
    windows that cannot be trusted. -> VitalsReport (every window labelled)."""
    fs = float(fs)
    A = amplitude(H)
    if A.ndim != 2 or A.shape[0] < int(10 * fs):
        raise ValueError("vital signs need at least 10 s of CSI as "
                         "(time, subcarrier)")
    rep = VitalsReport(fs=fs)
    # the AGC is divided out of the BREATHING path only: its wander lives at
    # breathing frequencies, while the heart's small movement is partly
    # common to all subcarriers and dividing by their mean would remove it
    Ab = remove_dc(normalise_agc(A) if agc_normalise else A)
    xb = savgol(bandpass(Ab, fs, BREATH_BAND), fs, SAVGOL_S["breath"])
    xh = savgol(bandpass(remove_dc(A), fs, HEART_BAND), fs, SAVGOL_S["heart"])
    gaps = gaps or []
    for a, b in windows(A.shape[0], fs, win_s, hop_s):
        t0, t1 = a / fs, b / fs
        w = {"t0_s": t0, "t1_s": t1, "label": RESEARCH_LABEL, "tier": TIER,
             "breath_bpm": None, "heart_bpm": None, "why": ""}
        bad = [g for g in gaps if g[1] > t0 and g[0] < t1]
        if bad:
            w["why"] = (f"a gap of {max(g[1] - g[0] for g in bad):.1f} s in the "
                        "CSI inside this window — not measured")
            rep.windows.append(w)
            continue
        sb = select_subcarriers(xb[a:b], fs, BREATH_BAND, k)
        br = combine(xb[a:b, sb])
        rate, spec, ac, agree, prom, r, why_b = _rate(br, fs, "breath")
        w.update(breath_bpm=rate, breath_spectral_bpm=spec,
                 breath_autocorr_bpm=ac, breath_agree=agree,
                 breath_prominence=prom, breath_r=r)
        reject = ()
        if spec is not None:
            fb = spec / 60.0
            reject = tuple(m * fb for m in range(2, 9))
        sh = select_subcarriers(xh[a:b], fs, HEART_BAND, k)
        hr = combine(xh[a:b, sh])
        hrate, hs, hac, hagree, hprom, hr_r, why_h = _rate(hr, fs, "heart", reject)
        w.update(heart_bpm=hrate, heart_spectral_bpm=hs, heart_autocorr_bpm=hac,
                 heart_agree=hagree, heart_prominence=hprom, heart_r=hr_r)
        w["why"] = "; ".join(x for x in (why_b, why_h) if x)
        rep.windows.append(w)
    bs = [w["breath_bpm"] for w in rep.windows if w.get("breath_bpm") is not None]
    hs_ = [w["heart_bpm"] for w in rep.windows if w.get("heart_bpm") is not None]
    rep.breath_bpm = float(np.median(bs)) if bs else None
    rep.heart_bpm = float(np.median(hs_)) if hs_ else None
    if apnea:
        sb = select_subcarriers(xb, fs, BREATH_BAND, k)
        rep.apnea = apnea_events(combine(xb[:, sb]), fs)
    if not rep.windows:
        rep.notes.append("the recording is shorter than one window")
    return rep


# ---------------------------------------------------------------------------
# Placement — the Fresnel-zone model
# ---------------------------------------------------------------------------
def _vec(p) -> np.ndarray:
    v = np.asarray(p, dtype=np.float64).ravel()
    if v.size == 2:
        v = np.r_[v, 0.0]
    if v.size != 3:
        raise ValueError("positions are (x, y) or (x, y, z) in metres")
    return v


def fresnel_zone(tx, rx, point, freq_hz: float = 2.437e9) -> dict:
    """Which Fresnel zone of the Tx–Rx pair `point` is in, and where in it.
    -> {zone (1 = the first), fraction (0..1 across the zone), excess_m,
    wavelength_m, r1_m (first-zone radius at the point's place along the
    line)}."""
    t, r, p = _vec(tx), _vec(rx), _vec(point)
    lam = C / float(freq_hz)
    d = float(np.linalg.norm(r - t))
    if d <= 0:
        raise ValueError("the transmitter and receiver are in the same place")
    d1, d2 = float(np.linalg.norm(p - t)), float(np.linalg.norm(r - p))
    excess = d1 + d2 - d
    half = lam / 2.0
    zone = int(excess // half) + 1
    frac = (excess % half) / half
    u = float(np.clip(np.dot(p - t, (r - t) / d), 0.0, d))
    r1 = math.sqrt(lam * u * (d - u) / d)
    return {"zone": zone, "fraction": frac, "excess_m": excess,
            "wavelength_m": lam, "r1_m": r1, "link_m": d}


def sensitivity(excess_m: float, wavelength_m: float,
                reflection_phase: float = math.pi) -> float:
    """|sin θ|, θ the reflected path's phase against the direct one: 1 in the
    middle of a zone, 0 on a boundary (small-movement, direct-path-dominant
    model)."""
    th = 2 * math.pi * excess_m / wavelength_m + reflection_phase
    return abs(math.sin(th))


def placement(tx, rx, chest, freq_hz: float = 2.437e9,
              reflection_phase: float = math.pi, search_m: float = 0.08,
              step_m: float = 0.001) -> dict:
    """Where the chest sits in the pair's Fresnel zones, how sensitive that
    spot is, and the smallest move (perpendicular to the Tx–Rx line) that
    reaches the most sensitive spot. Every result carries the research
    label."""
    t, r, p = _vec(tx), _vec(rx), _vec(chest)
    z = fresnel_zone(t, r, p, freq_hz)
    s_here = sensitivity(z["excess_m"], z["wavelength_m"], reflection_phase)
    los = (r - t) / np.linalg.norm(r - t)
    off = (p - t) - np.dot(p - t, los) * los
    if np.linalg.norm(off) < 1e-9:
        up = np.array([0.0, 0.0, 1.0]) if abs(los[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        off = np.cross(los, up)
    nvec = off / np.linalg.norm(off)
    best = (s_here, 0.0)
    for k in np.arange(step_m, search_m + step_m / 2, step_m):
        for sgn in (1.0, -1.0):
            q = p + sgn * k * nvec
            zz = fresnel_zone(t, r, q, freq_hz)
            s = sensitivity(zz["excess_m"], zz["wavelength_m"], reflection_phase)
            if s > best[0] + 1e-6:
                best = (s, sgn * k)
        if best[0] >= 0.98:
            break
    move = best[1]
    if abs(move) < 1e-9:
        words = (f"the chest is in Fresnel zone {z['zone']}, "
                 f"{100 * z['fraction']:.0f}% of the way across it; its "
                 f"sensitivity is {s_here:.2f} of the best — a good spot")
    else:
        way = "away from" if move > 0 else "toward"
        words = (f"the chest is in Fresnel zone {z['zone']}, "
                 f"{100 * z['fraction']:.0f}% of the way across it; its "
                 f"sensitivity is {s_here:.2f} of the best. Moving it "
                 f"{100 * abs(move):.1f} cm {way} the line between the pair "
                 f"raises that to {best[0]:.2f}.")
    return {**z, "sensitivity": s_here, "best_sensitivity": best[0],
            "move_m": move, "words": words, "label": RESEARCH_LABEL,
            "model": "Fresnel zones, direct path dominant, reflection phase "
                     f"{reflection_phase:.2f} rad (arXiv 1908.05108's premise)"}
