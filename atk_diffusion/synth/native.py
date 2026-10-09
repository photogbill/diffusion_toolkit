# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Native synthetic signals: numpy generators for every class in the class
table with a `native` key, at ANY sample rate, with exact labels (plan §3.3,
§4.A; DETECTION_DESIGN §6 step 2, §9; ARCHITECTURE §4.3).

WHY A NATIVE GENERATOR EXISTS BESIDE TORCHSIG. Bill, 2026-10-08, on TorchSig:
*"it only works with a specific version of python … it was a pain the last
time."* So nothing in the toolkit DEPENDS on TorchSig installing: this module
makes every class on the v1 list with numpy and scipy alone, and the dataset
builder, the scene composer and the tests run on it. TorchSig 2.2.0
(`synth.torchsig_backend`) is the second generator, compared on the domain gap,
never a prerequisite.

THE SAMPLE-RATE LAW. Bill: *"OmniSIG only works if the sample rate is
identical in training … as in the field."* Every waveform is generated at a
convenient NATIVE rate (an integer number of samples per symbol, the OFDM FFT
rate, …) and brought to the profile's exact rate by an exact rational
polyphase resampler; when a ratio has to be approximated (huge factors only),
the realised symbol rate is reported, not the requested one (ATK's own
`siga/synth.py` rule: *"tests compare against truth, never against the
request"*).

THE SNR, DEFINED ONCE (`SNR_DEFINITION`): signal power while the signal is ON
÷ noise power inside the signal's 99 % occupied bandwidth (ITU-R SM.328: the
band outside which 0.5 % of the power lies on each side), both at the antenna
port, before the receiver's own impairments. Bursty signals (DMR TDMA, ADS-B
squitters, Wi-Fi frames) are measured over their on-time; a tone's (and the DC
spike's) occupied bandwidth is the resolution of its own duration, 1/T, so its
SNR is the classical E/N0. Bill's `siga/synth.py` uses the SAMPLED bandwidth
instead and says so; here the occupied band is used because a detector's box
and a cyclic proposer both see a signal in its own band, and the same SNR
means the same difficulty for a 7.6 kHz DMR burst and a 9 MHz LTE carrier.

THE LABEL (dict):
    cls, family, bandwidth_hz (99 % occupied, measured on the clean
    realisation), symbol_rate_hz (0 = none: analog, noise, tone), carrier_
    offset_hz, snr_db (None from `waveform`, NaN for the noise class),
    sample_start, sample_count (first on-sample, span to the last), f_lo_hz,
    f_hi_hz (the 99 % edges, relative to the capture centre), bursts [[start,
    count], …], native, fs, generator, params (every realised parameter),
    clipped (True when a wider signal is seen through the receiver's span)

A SIGNAL WIDER THAN THE RATE CAN HOLD IS REFUSED IN WORDS (`SynthRefusal`):
*"LTE downlink at 50 resource blocks is 9.0 MHz wide; 2.4 MS/s holds at most
2.4 MHz …"* — unless `clip=True`, which shows the slice a receiver tuned there
would see (the scene composer uses it for a 10 MHz carrier in an RTL's span).

FIDELITY, STATED HONESTLY. Physical-layer structure is real where it is cheap
and checkable: P25/DMR/NXDN deviations, symbol rates, pulse shapes and sync
words; DMR's 27.5 ms bursts on a 30 ms TDMA grid; POCSAG's 576-bit preamble,
sync codeword and BCH(31,21) codewords; ADS-B DF17/DF11 squitters with a
valid CRC-24 (a decoder can decode them); LoRa's preamble, sync word and SFD
with continuous-phase chirps; LTE's 15 kHz numerology, normal CP, PSS and cell
reference-signal grid; NR's 30 kHz numerology and CP; 802.11a/g's STF, LTF and
pilots; BLE's preamble and access address; ATSC's 8 levels, pilot and segment
sync. Not modelled: voice codecs and FEC payloads (random bits), FLEX's 1600
baud header (examples are cut from the data blocks), LTE's SSS and PBCH, NR's
SSB, BLE whitening/CRC, Wi-Fi scrambling/coding, a DJI-exact drone link (the
drone class is a bursty OFDM stand-in), CTCSS on NOAA, broadcast pre-emphasis
and audio processing beyond a limiter. Synthetic data is a PRIOR — the domain
gap against cabled captures is the number that says how good it is (plan §7).

Tier: synthetic output is INVENTED (`provenance`): it was never received, and
the label says so wherever the data goes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from atk_diffusion import provenance
from atk_diffusion.detect import classes as _classes

provenance.METHOD_TIERS.setdefault("synthetic_native", "invented")

GENERATOR = "native"
METHOD = "synthetic_native"
OBW_FRACTION = 0.99
DEFAULT_NOISE_DBFS = -30.0
SNR_DEFINITION = ("signal power while on / noise power inside the signal's "
                  "99 % occupied bandwidth (ITU-R SM.328), at the antenna port, "
                  "before the receiver's own impairments; a tone's occupied "
                  "bandwidth is 1/T")

_RATE_LIMIT = 10_000          # largest up/down factor of the polyphase resampler


class SynthRefusal(ValueError):
    """The generator will not make this; the message says why, in words."""


def _rate_words(fs: float) -> str:
    fs = float(fs)
    if fs >= 1e6:
        return f"{fs / 1e6:g} MS/s"
    if fs >= 1e3:
        return f"{fs / 1e3:g} kS/s"
    return f"{fs:g} S/s"


def _hz_words(f: float) -> str:
    f = abs(float(f))
    if f >= 1e6:
        return f"{f / 1e6:.3g} MHz"
    if f >= 1e3:
        return f"{f / 1e3:.3g} kHz"
    return f"{f:.3g} Hz"


# ---------------------------------------------------------------------------
# rates and resampling
# ---------------------------------------------------------------------------
def _frac(v) -> Fraction:
    if isinstance(v, Fraction):
        return v
    if isinstance(v, (int, np.integer)):
        return Fraction(int(v))
    fv = float(v)
    if fv.is_integer():
        return Fraction(int(fv))
    return Fraction(fv).limit_denominator(1_000_000)


def _ratio(fs_from, fs_to) -> tuple[int, int, float]:
    """(up, down, λ): resample by up/down; λ = declared ÷ effective rate,
    1.0 exactly when the ratio is representable within the factor limit."""
    r = _frac(fs_to) / _frac(fs_from)
    if r.numerator <= _RATE_LIMIT and r.denominator <= _RATE_LIMIT:
        return r.numerator, r.denominator, 1.0
    if r >= 1:
        a = r.limit_denominator(max(1, int(_RATE_LIMIT / r)))
    else:
        a = 1 / (1 / r).limit_denominator(max(1, int(_RATE_LIMIT * r)))
    return a.numerator, a.denominator, float(r / a)


def _resample(x: np.ndarray, up: int, down: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128)
    if up == down:
        return x
    from scipy.signal import resample_poly
    return resample_poly(x, up, down)


def _guard_s(fs_n, up: int, down: int) -> float:
    """Seconds of native signal the polyphase filter needs on each side."""
    return (10.0 * max(up, down) / up + 4.0) / float(fs_n)


def _render(xn: np.ndarray, fs_n, fs: float, n: int, t0_s: float,
            up: int, down: int) -> np.ndarray:
    """Resample native samples to the declared rate and return `n` samples
    starting `t0_s` seconds into `xn`."""
    y = _resample(xn, up, down)
    fs_eff = float(fs_n) * up / down
    k0 = int(round(t0_s * fs_eff))
    out = y[k0:k0 + n]
    if out.size < n:
        out = np.concatenate([out, np.zeros(n - out.size, complex)])
    return out


def _shift(x: np.ndarray, f_hz: float, fs: float) -> np.ndarray:
    if not f_hz:
        return x
    n = np.arange(x.size, dtype=np.float64)
    return x * np.exp(2j * np.pi * float(f_hz) / float(fs) * n)


# ---------------------------------------------------------------------------
# pulses (ATK's siga/synth.py shapes, ported)
# ---------------------------------------------------------------------------
def rrc_taps(beta: float, sps: int, span: int = 12) -> np.ndarray:
    """Root-raised-cosine taps, unit energy; singularities by their limits."""
    sps = max(1, int(sps))
    beta = float(beta)
    n = np.arange(-span * sps / 2.0, span * sps / 2.0 + 1)
    t = n / float(sps)
    h = np.empty_like(t)
    if beta <= 0:
        h = np.sinc(t)
    else:
        sing = np.isclose(np.abs(t), 1.0 / (4.0 * beta), atol=1e-9)
        zero = np.isclose(t, 0.0, atol=1e-12)
        ok = ~(sing | zero)
        tt = t[ok]
        num = (np.sin(np.pi * tt * (1 - beta))
               + 4 * beta * tt * np.cos(np.pi * tt * (1 + beta)))
        den = np.pi * tt * (1 - (4 * beta * tt) ** 2)
        h[ok] = num / den
        h[zero] = 1.0 - beta + 4.0 * beta / np.pi
        if sing.any():
            h[sing] = (beta / np.sqrt(2.0)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
    e = np.sqrt(np.sum(h ** 2))
    return h / e if e > 0 else h


def rc_taps(beta: float, sps: int, span: int = 12) -> np.ndarray:
    """Raised-cosine (Nyquist) taps, peak 1 at t = 0."""
    sps = max(1, int(sps))
    t = np.arange(-span * sps / 2.0, span * sps / 2.0 + 1) / float(sps)
    b = float(beta)
    den = 1.0 - (2.0 * b * t) ** 2
    sing = np.isclose(den, 0.0, atol=1e-10)
    h = np.empty_like(t)
    h[~sing] = np.sinc(t[~sing]) * np.cos(np.pi * b * t[~sing]) / den[~sing]
    if sing.any():
        h[sing] = (np.pi / 4.0) * np.sinc(1.0 / (2.0 * b))
    return h


def gauss_freq_pulse(bt: float, sps: int, span: int = 4) -> np.ndarray:
    """GFSK/GMSK frequency pulse: a symbol-long rectangle through a Gaussian
    filter of bandwidth-time product `bt`."""
    sps = max(1, int(sps))
    t = np.arange(-span * sps / 2.0, span * sps / 2.0 + 1) / float(sps)
    alpha = np.sqrt(np.log(2.0) / 2.0) / max(1e-6, float(bt))
    g = np.exp(-(np.pi ** 2) * (t ** 2) / (alpha ** 2))
    g /= np.sum(g)
    return np.convolve(g, np.ones(sps))


def _freq_pulse(kind: str, sps: int, beta: float = 0.2, bt: float = 0.5
                ) -> np.ndarray:
    if kind == "rect":
        p = np.ones(sps)
    elif kind == "rc":
        p = rc_taps(beta, sps, 12)
    elif kind == "rrc":
        p = rrc_taps(beta, sps, 12)
    elif kind == "gauss":
        p = gauss_freq_pulse(bt, sps, 4)
    else:
        raise ValueError(f"unknown frequency pulse {kind!r}")
    return p * (sps / np.sum(p))       # a held level L gives exactly dev·L


# ---------------------------------------------------------------------------
# the occupied band (ITU-R SM.328)
# ---------------------------------------------------------------------------
def occupied_band(x, fs: float, frac: float = OBW_FRACTION,
                  max_nfft: int = 1 << 18, max_segments: int = 32
                  ) -> tuple[float, float]:
    """(f_lo, f_hi), Hz relative to the samples' own centre: the band outside
    which (1 − frac)/2 of the power lies on each side. Short records use one
    Hann-windowed periodogram (zero-padded); long ones a Welch average."""
    x = np.asarray(x, dtype=np.complex64).ravel()
    n = x.size
    if n < 2 or not np.any(x):
        return 0.0, 0.0
    fs = float(fs)
    if n <= max_nfft:
        nfft = max(4096, 1 << int(math.ceil(math.log2(n))))
        w = np.hanning(n + 2)[1:-1]
        p = np.abs(np.fft.fft(x * w, nfft)) ** 2
    else:
        nfft = max_nfft
        w = np.hanning(nfft + 2)[1:-1].astype(np.float32)
        starts = np.arange(0, n - nfft + 1, nfft // 2)
        if starts.size > max_segments:
            starts = starts[np.linspace(0, starts.size - 1, max_segments)
                            .round().astype(int)]
        p = np.zeros(nfft)
        for s in starts:
            p += np.abs(np.fft.fft(x[s:s + nfft] * w)) ** 2
    p = np.fft.fftshift(p)
    df = fs / nfft
    edges = (np.arange(nfft) - nfft // 2) * df + df / 2.0
    cum = np.cumsum(p)
    cum /= cum[-1]
    q = (1.0 - float(frac)) / 2.0
    lo = float(np.interp(q, cum, edges))
    hi = float(np.interp(1.0 - q, cum, edges))
    return lo, max(hi, lo + df)


# ---------------------------------------------------------------------------
# shared building blocks
# ---------------------------------------------------------------------------
_DIBIT_LEVEL = np.array([1.0 / 3.0, 1.0, -1.0 / 3.0, -1.0])   # 00 01 10 11


def _bits(hexstr: str) -> np.ndarray:
    nbits = 4 * len(hexstr)
    v = int(hexstr, 16)
    return np.array([(v >> (nbits - 1 - i)) & 1 for i in range(nbits)],
                    dtype=np.int8)


def _dibit_levels(bits: np.ndarray) -> np.ndarray:
    b = np.asarray(bits, dtype=int)
    b = b[: (b.size // 2) * 2].reshape(-1, 2)
    return _DIBIT_LEVEL[b[:, 0] * 2 + b[:, 1]]


def _pick_sps(rs, fs: float, sps_min: int) -> int:
    """The native samples-per-symbol (≥ sps_min, ≤ 4·sps_min) whose native
    rate makes the smallest EXACT resampling ratio to fs; sps_min when none
    is exact (the realised rate is then reported through λ)."""
    rs = _frac(rs)
    best, score = sps_min, None
    for sps in range(int(sps_min), 4 * int(sps_min) + 1):
        r = _frac(fs) / (rs * sps)
        m = max(r.numerator, r.denominator)
        if m <= _RATE_LIMIT and (score is None or m < score):
            best, score = sps, m
    return best


def _cpm(levels: np.ndarray, rs, dev_hz: float, pulse: str, fs: float, n: int,
         lead_syms: int, beta: float = 0.2, bt: float = 0.5,
         sps_min: int = 8) -> tuple[np.ndarray, float]:
    """Continuous-phase FSK: instantaneous frequency = dev·Σ L_k g(t − kT).
    `levels` covers the window plus `lead_syms` on each side; the window
    starts at symbol `lead_syms`. Returns (baseband at fs, λ)."""
    rs = _frac(rs)
    need = max(sps_min, int(math.ceil(4.0 * (abs(dev_hz) + float(rs)) / float(rs))))
    sps = _pick_sps(rs, fs, need)
    fs_n = rs * sps
    up, down, lam = _ratio(fs_n, fs)
    p = _freq_pulse(pulse, sps, beta, bt)
    from scipy.signal import upfirdn
    f = upfirdn(p, np.asarray(levels, dtype=np.float64), up=sps) * float(dev_hz)
    delay = (p.size - 1) / 2.0
    ph = 2.0 * np.pi * np.cumsum(f) / float(fs_n)
    x = np.exp(1j * ph)
    t0 = (lead_syms * sps + delay) / float(fs_n)
    return _render(x, fs_n, fs, n, t0, up, down), lam


def _linear(symbols: np.ndarray, rs, beta: float, fs: float, n: int,
            lead_syms: int, sps: int = 8) -> tuple[np.ndarray, float]:
    """Linear modulation with RRC shaping (roll-off `beta`)."""
    rs = _frac(rs)
    sps = _pick_sps(rs, fs, sps)
    fs_n = rs * sps
    up, down, lam = _ratio(fs_n, fs)
    taps = rrc_taps(beta, sps, 12)
    from scipy.signal import upfirdn
    x = upfirdn(taps, np.asarray(symbols, dtype=np.complex128), up=sps)
    delay = (taps.size - 1) / 2.0
    t0 = (lead_syms * sps + delay) / float(fs_n)
    return _render(x, fs_n, fs, n, t0, up, down), lam


def _lead(rs, fs: float, extra: int = 8) -> int:
    """Symbols of lead-in each side: the pulse span plus the resampler's."""
    rs = float(rs)
    return int(math.ceil(rs * 40.0 / max(1.0, min(fs, 8 * rs)))) + extra + 12


def _alphabet(kind: str) -> np.ndarray:
    if kind == "bpsk":
        return np.array([1.0, -1.0], dtype=complex)
    if kind == "qpsk":
        return np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    if kind == "8psk":
        return np.exp(2j * np.pi * np.arange(8) / 8.0)
    if kind in ("16qam", "64qam", "256qam"):
        side = {"16qam": 4, "64qam": 8, "256qam": 16}[kind]
        lv = np.arange(-(side - 1), side, 2, dtype=float)
        grid = (lv[:, None] + 1j * lv[None, :]).ravel()
        return grid / np.sqrt(np.mean(np.abs(grid) ** 2))
    if kind == "ask4":
        lv = np.array([-3.0, -1.0, 1.0, 3.0])
        return (lv / np.sqrt(np.mean(lv ** 2))).astype(complex)
    raise ValueError(f"no alphabet for {kind!r}")


def _lowpass_noise(n: int, rate_hz: float, fs: float, rng) -> np.ndarray:
    """Smooth random process, bandwidth ~rate_hz, n samples at fs."""
    m = max(4, int(math.ceil(n * rate_hz / fs)) + 4)
    knots = rng.standard_normal(m)
    k = np.convolve(knots, np.hanning(5), mode="same") / 2.0
    t = np.linspace(0, m - 1, n)
    return np.interp(t, np.arange(m), k)


def _voice(n: int, fs: float, rng, band=(300.0, 3000.0),
           pauses: bool = True) -> np.ndarray:
    """Speech-like message: band-limited noise with a syllabic (≈4 Hz)
    envelope and pauses, limited to ±1 (a deviation limiter)."""
    from scipy.signal import butter, sosfilt
    w = rng.standard_normal(n + 2048)
    hi = min(band[1], 0.45 * fs)
    sos = butter(4, [band[0], hi], btype="band", fs=fs, output="sos")
    v = sosfilt(sos, w)[2048:]
    env = _lowpass_noise(n, 4.0, fs, rng)
    env = np.clip(0.8 * env + (0.35 if pauses else 1.2), 0.0, None)
    v = v * env
    s = float(np.std(v))
    if s > 0:
        v = v / (3.0 * s)
    return np.clip(v, -1.0, 1.0)


def _gate(n: int, fs: float, period_s: float, on_s: float, phase_s: float,
          ramp_s: float) -> tuple[np.ndarray, list]:
    """A burst envelope: on for `on_s` every `period_s`, raised-cosine ramps
    of `ramp_s` OUTSIDE the on-time. Returns (envelope, [[start, count]])."""
    t = np.arange(n) / fs + float(phase_s)
    u = np.mod(t, period_s)
    env = np.zeros(n)
    on = u < on_s
    env[on] = 1.0
    if ramp_s > 0:
        up = (u >= period_s - ramp_s)
        env[up] = 0.5 - 0.5 * np.cos(np.pi * (u[up] - (period_s - ramp_s)) / ramp_s)
        dn = (u >= on_s) & (u < on_s + ramp_s)
        env[dn] = 0.5 + 0.5 * np.cos(np.pi * (u[dn] - on_s) / ramp_s)
    bursts = _runs(on)
    return env, bursts


def _runs(mask: np.ndarray) -> list:
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    starts = np.nonzero(d == 1)[0]
    ends = np.nonzero(d == -1)[0]
    return [[int(s), int(e - s)] for s, e in zip(starts, ends)]


def _place(n: int, bursts_x: list[np.ndarray], rng, min_gap: int = 0,
           span: tuple[int, int] | None = None) -> tuple[np.ndarray, list]:
    """Put burst waveforms into an n-sample window at random non-overlapping
    positions. A burst longer than the window is sliced at a random point."""
    out = np.zeros(n, dtype=complex)
    taken: list[list[int]] = []
    for b in bursts_x:
        b = np.asarray(b, dtype=complex)
        if b.size >= n:
            s = int(rng.integers(0, b.size - n + 1))
            out[:] += b[s:s + n]
            taken.append([0, n])
            continue
        for _ in range(64):
            s = int(rng.integers(0, n - b.size + 1))
            if all(s + b.size + min_gap <= t0 or s >= t0 + c + min_gap
                   for t0, c in taken):
                out[s:s + b.size] += b
                taken.append([s, int(b.size)])
                break
    taken.sort()
    return out, taken


# ---------------------------------------------------------------------------
# context passed to each kind
# ---------------------------------------------------------------------------
@dataclass
class _Ctx:
    cls: str
    fs: float
    n: int
    rng: np.random.Generator
    p: dict


@dataclass
class _Wave:
    x: np.ndarray                    # baseband, n samples at ctx.fs
    on: list                         # [[start, count]] in samples
    symbol_rate: float = 0.0
    params: dict = field(default_factory=dict)


# Per-kind parameters a caller may pass (anything else is refused by name).
_ACCEPT: dict[str, set] = {}
# Per-class overrides of a kind's defaults.
_CLASS_PRESETS: dict[str, dict] = {
    "nfm_voice": {"deviation_hz": 2500.0, "ctcss": None},
    "noaa_wx": {"deviation_hz": 5000.0, "ctcss": False},
    "p25": {"symbol_rate": 4800.0, "deviation_hz": 1800.0, "pulse": "rc",
            "rolloff": 0.2, "sync": "5575F5FF77FF", "frame_symbols": 864},
    "nxdn96": {"symbol_rate": 4800.0, "deviation_hz": 2400.0, "pulse": "rrc",
               "rolloff": 0.2, "sync": "CDF59", "frame_symbols": 192},
    "nxdn48": {"symbol_rate": 2400.0, "deviation_hz": 1050.0, "pulse": "rrc",
               "rolloff": 0.2, "sync": "CDF59", "frame_symbols": 192},
}


# ---------------------------------------------------------------------------
# the kinds
# ---------------------------------------------------------------------------
def _resolve_common_rate(p: dict, fs: float, rng, bw_per_rs: float,
                         lo: float = 1 / 40, hi: float = 1 / 4) -> float:
    """A reference class's symbol rate: given, from a target occupied
    bandwidth, or drawn log-uniformly so the signal fills 2.5–25 % of fs."""
    if p.get("symbol_rate"):
        return float(p["symbol_rate"])
    if p.get("bandwidth_hz"):
        return float(p["bandwidth_hz"]) / bw_per_rs
    bw = math.exp(rng.uniform(math.log(lo * fs), math.log(hi * fs)))
    return bw / bw_per_rs


# -- analog FM voice -------------------------------------------------------
_CTCSS = (67.0, 69.3, 71.9, 74.4, 77.0, 79.7, 82.5, 85.4, 88.5, 91.5, 94.8,
          97.4, 100.0, 103.5, 107.2, 110.9, 114.8, 118.8, 123.0, 127.3, 131.8,
          136.5, 141.3, 146.2, 151.4, 156.7, 162.2, 167.9, 173.8, 179.9,
          186.2, 192.8, 203.5, 210.7, 218.1, 225.7, 233.6, 241.8, 250.3)


def _res_nfm(cls, fs, rng, p):
    dev = float(p.get("deviation_hz", 2500.0))
    ctcss = p.get("ctcss")
    if ctcss is None:
        ctcss = bool(rng.uniform() < 0.5)
    tone = float(p.get("ctcss_hz") or rng.choice(_CTCSS)) if ctcss else 0.0
    p.update(deviation_hz=dev, ctcss=bool(ctcss), ctcss_hz=tone,
             audio_hz=[300.0, 3000.0])
    return 2.0 * (dev + 3000.0), 0.0


def _syn_nfm(c: _Ctx) -> _Wave:
    dev = c.p["deviation_hz"]
    fs_n = 48000 * max(1, int(math.ceil(4.0 * (dev + 3000.0) / 48000.0)))
    up, down, _ = _ratio(fs_n, c.fs)
    g = _guard_s(fs_n, up, down) + 0.01
    nn = int(math.ceil((c.n / c.fs + 2 * g) * fs_n))
    m = _voice(nn, fs_n, c.rng)
    if c.p["ctcss"]:
        t = np.arange(nn) / fs_n
        m = 0.85 * m + 0.15 * np.sin(2 * np.pi * c.p["ctcss_hz"] * t
                                     + c.rng.uniform(0, 2 * np.pi))
    ph = 2 * np.pi * dev * np.cumsum(m) / fs_n
    x = np.exp(1j * ph)
    return _Wave(_render(x, fs_n, c.fs, c.n, g, up, down), [[0, c.n]])


_ACCEPT["nfm"] = {"deviation_hz", "ctcss", "ctcss_hz"}


# -- AM (reference DSB with carrier) --------------------------------------
def _res_am(cls, fs, rng, p):
    m = float(p.get("mod_index", rng.uniform(0.3, 0.9)))
    p.update(mod_index=m, audio_hz=[300.0, 3000.0])
    return 6000.0, 0.0


def _syn_am(c: _Ctx) -> _Wave:
    fs_n = 48000
    up, down, _ = _ratio(fs_n, c.fs)
    g = _guard_s(fs_n, up, down) + 0.01
    nn = int(math.ceil((c.n / c.fs + 2 * g) * fs_n))
    x = 1.0 + c.p["mod_index"] * _voice(nn, fs_n, c.rng)
    x = x * np.exp(1j * c.rng.uniform(0, 2 * np.pi))
    return _Wave(_render(x, fs_n, c.fs, c.n, g, up, down), [[0, c.n]])


_ACCEPT["am"] = {"mod_index"}


# -- broadcast FM (stereo, pilot, RDS) ---------------------------------------
def _res_wfm(cls, fs, rng, p):
    p.setdefault("deviation_hz", 75000.0)
    p.setdefault("stereo", True)
    p.setdefault("rds", True)
    return 2.0 * (float(p["deviation_hz"]) + 53000.0), 0.0


def _syn_wfm(c: _Ctx) -> _Wave:
    fs_n = 960_000
    up, down, _ = _ratio(fs_n, c.fs)
    g = _guard_s(fs_n, up, down) + 0.002
    nn = int(math.ceil((c.n / c.fs + 2 * g) * fs_n))
    rng = c.rng
    from scipy.signal import butter, sosfilt
    sos = butter(6, 15000.0, btype="low", fs=fs_n, output="sos")
    common = sosfilt(sos, rng.standard_normal(nn + 4096))[4096:]
    side = sosfilt(sos, rng.standard_normal(nn + 4096))[4096:]
    env = np.clip(0.8 + 0.2 * _lowpass_noise(nn, 2.0, fs_n, rng), 0.3, None)
    left = (common + 0.4 * side) * env
    right = (common - 0.4 * side) * env
    # broadcast audio is heavily processed: compressed so it sits near full
    # modulation most of the time (a soft limiter on each channel)
    for a in (left, right):
        a /= (np.std(a) + 1e-12)
        np.tanh(3.0 * a, out=a)
    t = np.arange(nn) / fs_n
    mono = 0.5 * (left + right)
    comp = mono.copy()
    if c.p["stereo"]:
        comp = comp + 0.5 * (left - right) * np.cos(2 * np.pi * 38000.0 * t)
    comp = np.clip(0.9 * comp, -0.9, 0.9)            # the broadcast limiter
    if c.p["stereo"]:
        comp = comp + 0.09 * np.cos(2 * np.pi * 19000.0 * t)
    if c.p["rds"]:
        nb = int(math.ceil(nn / fs_n * 1187.5)) + 2
        bits = rng.integers(0, 2, nb) * 2 - 1
        chips = np.repeat(np.stack([bits, -bits], 1).ravel(), 1)
        k = np.floor(t * 2375.0).astype(int)
        rds = chips[np.minimum(k, chips.size - 1)].astype(float)
        rds = np.convolve(rds, np.hanning(int(fs_n / 2375.0)), mode="same")
        rds /= (np.max(np.abs(rds)) + 1e-12)
        comp = comp + 0.04 * rds * np.cos(2 * np.pi * 57000.0 * t)
    ph = 2 * np.pi * c.p["deviation_hz"] * np.cumsum(comp) / fs_n
    x = np.exp(1j * ph)
    return _Wave(_render(x, fs_n, c.fs, c.n, g, up, down), [[0, c.n]],
                 params={"pilot_hz": 19000.0 if c.p["stereo"] else 0.0})


_ACCEPT["wfm"] = {"deviation_hz", "stereo", "rds"}


# -- 4-level FSK: P25 C4FM, NXDN, and DMR ---------------------------------
def _res_c4fm(cls, fs, rng, p):
    p.setdefault("symbol_rate", 4800.0)
    p.setdefault("deviation_hz", 1800.0)
    p.setdefault("pulse", "rrc")
    p.setdefault("rolloff", 0.2)
    p.setdefault("sync", "")
    p.setdefault("frame_symbols", 0)
    rs = float(p["symbol_rate"])
    return 2.0 * float(p["deviation_hz"]) + rs, rs


def _c4fm_levels(c: _Ctx, nsym: int) -> np.ndarray:
    lv = _DIBIT_LEVEL[c.rng.integers(0, 4, nsym)]
    sync, frame = c.p.get("sync"), int(c.p.get("frame_symbols") or 0)
    if sync and frame:
        sl = _dibit_levels(_bits(sync))
        ph = int(c.rng.integers(0, frame))
        for s in range(-ph, nsym, frame):
            a, b = max(0, s), min(nsym, s + sl.size)
            if b > a:
                lv[a:b] = sl[a - s:b - s]
    return lv


def _syn_c4fm(c: _Ctx) -> _Wave:
    rs = float(c.p["symbol_rate"])
    lead = _lead(rs, c.fs)
    nsym = int(math.ceil(c.n / c.fs * rs)) + 2 * lead
    x, lam = _cpm(_c4fm_levels(c, nsym), rs, c.p["deviation_hz"],
                  c.p["pulse"], c.fs, c.n, lead, beta=c.p["rolloff"])
    return _Wave(x, [[0, c.n]], rs * lam)


_ACCEPT["c4fm_4800"] = _ACCEPT["c4fm_2400"] = {
    "symbol_rate", "deviation_hz", "pulse", "rolloff", "sync", "frame_symbols"}

DMR_SYNC_BS_VOICE = "755FD7DF75F7"


def _res_dmr(cls, fs, rng, p):
    p.setdefault("symbol_rate", 4800.0)
    p.setdefault("deviation_hz", 1944.0)
    p.setdefault("pulse", "rrc")
    p.setdefault("rolloff", 0.2)
    p.setdefault("slots", 2)
    if int(p["slots"]) not in (1, 2):
        raise SynthRefusal("DMR 'slots' is 1 (one talker: a 27.5 ms burst every "
                           "60 ms) or 2 (both slots: every 30 ms)")
    rs = float(p["symbol_rate"])
    p.update(burst_s=132 / rs, period_s=144 * int(3 - int(p["slots"])) / rs)
    return 2.0 * float(p["deviation_hz"]) + rs, rs


def _syn_dmr(c: _Ctx) -> _Wave:
    """Two-slot TDMA (ETSI TS 102 361): 132-symbol bursts (27.5 ms) of 54
    payload + 24 sync + 54 payload symbols on a 30 ms grid (60 ms with one
    slot); 2.5 ms between bursts (the CACH, not transmitted here)."""
    rs = float(c.p["symbol_rate"])
    lead = _lead(rs, c.fs)
    nsym = int(math.ceil(c.n / c.fs * rs)) + 2 * lead
    per = int(round(c.p["period_s"] * rs))            # 144 or 288 symbols
    ph = int(c.rng.integers(0, per))
    # bursts start ph symbols before the window's first symbol (index
    # `lead`), then every `per`: start indices s ≡ lead − ph (mod per)
    lv = _DIBIT_LEVEL[c.rng.integers(0, 4, nsym)]
    sl = _dibit_levels(_bits(DMR_SYNC_BS_VOICE))
    for s in range((lead - ph) % per - per, nsym, per):
        a = s + 54
        lo, hi = max(0, a), min(nsym, a + sl.size)
        if hi > lo:
            lv[lo:hi] = sl[lo - a:hi - a]
    x, lam = _cpm(lv, rs, c.p["deviation_hz"], c.p["pulse"], c.fs, c.n, lead,
                  beta=c.p["rolloff"])
    # in window time a burst starts at −(ph + ½)/rs + k·period (a symbol's
    # leading edge is half a symbol before its centre)
    env, bursts = _gate(c.n, c.fs, per / rs / lam, 132 / rs / lam,
                        (ph + 0.5) / rs / lam, 0.5 / rs / lam)
    return _Wave(x * env, bursts, rs * lam,
                 {"burst_ms": 1000 * 132 / rs / lam,
                  "period_ms": 1000 * per / rs / lam})


_ACCEPT["dmr_4800"] = {"symbol_rate", "deviation_hz", "pulse", "rolloff",
                       "slots"}


# -- pagers ----------------------------------------------------------------
POCSAG_SYNC = 0x7CD215D8
POCSAG_IDLE = 0x7A89C197


def _bch3121(data21: int) -> int:
    """POCSAG codeword: 21 data bits, BCH(31,21) check (g = 0x769), even
    parity — a valid codeword a pager decoder accepts."""
    g = 0x769
    v = (data21 & 0x1FFFFF) << 10
    r = v
    for i in range(30, 9, -1):
        if r & (1 << i):
            r ^= g << (i - 10)
    cw = v | (r & 0x3FF)
    par = bin(cw).count("1") & 1
    return (cw << 1) | par


def _pocsag_bits(rng, nbatches: int) -> np.ndarray:
    out = [np.array([1, 0] * 288, dtype=np.int8)]          # 576-bit preamble
    for _ in range(nbatches):
        words = [POCSAG_SYNC]
        for _ in range(16):
            r = rng.uniform()
            if r < 0.3:
                words.append(POCSAG_IDLE)
            elif r < 0.5:
                words.append(_bch3121(int(rng.integers(0, 1 << 20))))   # address
            else:
                words.append(_bch3121((1 << 20) | int(rng.integers(0, 1 << 20))))
        for w in words:
            out.append(np.array([(w >> (31 - i)) & 1 for i in range(32)],
                                dtype=np.int8))
    return np.concatenate(out)


def _res_pocsag(cls, fs, rng, p):
    baud = float(p.get("baud") or rng.choice([512.0, 1200.0, 2400.0]))
    if baud not in (512.0, 1200.0, 2400.0):
        raise SynthRefusal("POCSAG runs at 512, 1200 or 2400 baud")
    p.update(baud=baud, deviation_hz=float(p.get("deviation_hz", 4500.0)))
    return 2.0 * p["deviation_hz"] + baud, baud


def _syn_pocsag(c: _Ctx) -> _Wave:
    """A POCSAG transmission (576-bit preamble, then batches of a sync
    codeword and 16 BCH codewords); the window is a random stretch of it.
    Bit 1 is the low tone (−4.5 kHz), the common convention."""
    rs = c.p["baud"]
    lead = _lead(rs, c.fs)
    need = int(math.ceil(c.n / c.fs * rs)) + 2 * lead
    nb = max(1, int(math.ceil((need - 576) / 544.0)) + 1)
    bits = _pocsag_bits(c.rng, nb)
    start = int(c.rng.integers(0, max(1, bits.size - need + 1)))
    seg = bits[start:start + need]
    if seg.size < need:
        seg = np.concatenate([seg, bits[:need - seg.size]])
    lv = np.where(seg == 1, -1.0, 1.0)
    x, lam = _cpm(lv, rs, c.p["deviation_hz"], "gauss", c.fs, c.n, lead, bt=1.0)
    pre = max(0, min(576 - start - lead, need - 2 * lead))
    return _Wave(x, [[0, c.n]], rs * lam,
                 {"preamble_fraction": round(pre / max(1, need - 2 * lead), 3)})


_ACCEPT["pocsag"] = {"baud", "deviation_hz"}


def _res_flex(cls, fs, rng, p):
    rs = float(p.get("symbol_rate") or rng.choice([1600.0, 3200.0]))
    if rs not in (1600.0, 3200.0):
        raise SynthRefusal("FLEX data runs at 1600 or 3200 symbols/s")
    lv = int(p.get("levels") or rng.choice([2, 4]))
    if lv not in (2, 4):
        raise SynthRefusal("FLEX uses 2- or 4-level FSK")
    p.update(symbol_rate=rs, levels=lv,
             deviation_hz=float(p.get("deviation_hz", 4800.0)),
             mode=f"{int(rs * (1 if lv == 2 else 2))}/{lv}")
    return 2.0 * p["deviation_hz"] + rs, rs


def _syn_flex(c: _Ctx) -> _Wave:
    """FLEX data blocks at the mode's rate (±4.8 kHz outer, ±1.6 kHz inner).
    The 1600-baud sync header is not in the examples (stated limit)."""
    rs = c.p["symbol_rate"]
    lead = _lead(rs, c.fs)
    nsym = int(math.ceil(c.n / c.fs * rs)) + 2 * lead
    if c.p["levels"] == 2:
        lv = np.where(c.rng.integers(0, 2, nsym) == 1, 1.0, -1.0)
    else:
        lv = _DIBIT_LEVEL[c.rng.integers(0, 4, nsym)]
    x, lam = _cpm(lv, rs, c.p["deviation_hz"], "gauss", c.fs, c.n, lead, bt=1.0)
    return _Wave(x, [[0, c.n]], rs * lam)


_ACCEPT["flex"] = {"symbol_rate", "levels", "deviation_hz"}


# -- ADS-B -------------------------------------------------------------------
def crc24(bits) -> int:
    """Mode S CRC-24 (generator 0x1FFF409) over the message bits."""
    poly = 0x1FFF409
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    v <<= 24
    nb = len(bits) + 24
    for i in range(nb - 1, 23, -1):
        if v & (1 << i):
            v ^= poly << (i - 24)
    return v & 0xFFFFFF


def adsb_bits(rng, frame: str = "long") -> np.ndarray:
    """A DF17 extended squitter (112 bits) or DF11 all-call reply (56 bits)
    with random ICAO address and payload and a VALID parity."""
    if frame == "long":
        head = [1, 0, 0, 0, 1] + list(rng.integers(0, 2, 3))      # DF17, CA
        body = list(rng.integers(0, 2, 24 + 56))                    # AA, ME
    else:
        head = [0, 1, 0, 1, 1] + list(rng.integers(0, 2, 3))      # DF11
        body = list(rng.integers(0, 2, 24))
    msg = head + body
    crc = crc24(msg)
    return np.array(msg + [(crc >> (23 - i)) & 1 for i in range(24)],
                    dtype=np.int8)


def _res_adsb(cls, fs, rng, p):
    fr = p.get("frame") or ("long" if rng.uniform() < 0.8 else "short")
    if fr not in ("long", "short"):
        raise SynthRefusal("ADS-B 'frame' is 'long' (DF17, 112 bits) or "
                           "'short' (DF11, 56 bits)")
    p.update(frame=fr, squitters=int(p.get("squitters", 1)),
             rise_us=float(p.get("rise_us", 0.1)))
    return 2.0e6, 1.0e6


def _adsb_burst(bits: np.ndarray, rise_us: float, phase: float) -> np.ndarray:
    fs_n = 20_000_000                                  # 0.05 µs resolution
    chip = 10                                          # 0.5 µs
    pre = np.zeros(16, dtype=np.int8)
    pre[[0, 2, 7, 9]] = 1                              # 0, 1.0, 3.5, 4.5 µs
    data = np.empty(bits.size * 2, dtype=np.int8)
    data[0::2] = bits
    data[1::2] = 1 - bits
    chips = np.concatenate([pre, data]).astype(float)
    env = np.repeat(chips, chip)
    r = max(1, int(round(rise_us * 1e-6 * fs_n)))
    w = np.hanning(r + 2)[1:-1]
    env = np.convolve(env, w / w.sum(), mode="full")
    return env * np.exp(1j * phase), fs_n


def _syn_adsb(c: _Ctx) -> _Wave:
    bursts = []
    for _ in range(max(1, c.p["squitters"])):
        bits = adsb_bits(c.rng, c.p["frame"])
        env, fs_n = _adsb_burst(bits, c.p["rise_us"], c.rng.uniform(0, 2 * np.pi))
        up, down, _ = _ratio(fs_n, c.fs)
        pad = int(math.ceil(_guard_s(fs_n, up, down) * fs_n)) + 8
        y = _resample(np.concatenate([np.zeros(pad), env, np.zeros(pad)]),
                      up, down)
        k0 = int(round(pad * up / down))
        k1 = int(round((pad + env.size) * up / down))
        bursts.append(y[k0:k1])
    gap = int(round(c.fs * 2e-6))
    x, on = _place(c.n, bursts, c.rng, min_gap=gap)
    return _Wave(x, on, 1.0e6, {"bits": int(112 if c.p["frame"] == "long" else 56)})


_ACCEPT["adsb"] = {"frame", "squitters", "rise_us"}


# -- LoRa CSS ----------------------------------------------------------------
def _res_lora(cls, fs, rng, p):
    allowed = [b for b in (125e3, 250e3, 500e3) if b <= 0.9 * fs]
    bw = p.get("bw")
    if bw is None:
        if not allowed:
            raise SynthRefusal(f"LoRa's narrowest channel (125 kHz) does not "
                               f"fit {_rate_words(fs)}")
        bw = float(rng.choice(allowed))
    bw = float(bw)
    sf = int(p.get("sf") or rng.integers(7, 13))
    if not 6 <= sf <= 12:
        raise SynthRefusal("LoRa's spreading factor is 6 to 12")
    p.update(bw=bw, sf=sf, payload_symbols=int(p.get("payload_symbols")
                                                or rng.integers(8, 49)))
    return bw, bw / 2 ** sf


def _syn_lora(c: _Ctx) -> _Wave:
    """Preamble of 8 up-chirps, the LoRaWAN public sync word (0x34), 2.25
    down-chirps, then random payload symbols — continuous phase, generated
    directly at the profile rate (no resampling)."""
    bw, sf = c.p["bw"], c.p["sf"]
    m = 2 ** sf
    ts = m / bw
    fs = c.fs
    syms = ([("up", 0)] * 8 + [("up", (3 * 8) % m), ("up", (4 * 8) % m)]
            + [("down", 0)] * 2 + [("qdown", 0)]
            + [("up", int(s)) for s in c.rng.integers(0, m, c.p["payload_symbols"])])
    freqs = []
    for kind, s in syms:
        if kind == "qdown":
            ns = int(round(ts * fs / 4))
            t = np.arange(ns) / fs
            freqs.append(bw / 2 - bw * np.mod(t / ts, 1.0))
            continue
        ns = int(round(ts * fs))
        t = np.arange(ns) / fs
        u = np.mod(t / ts + s / m, 1.0)
        freqs.append((-bw / 2 + bw * u) if kind == "up" else (bw / 2 - bw * u))
    f = np.concatenate(freqs)
    ph = 2 * np.pi * np.cumsum(f) / fs + c.rng.uniform(0, 2 * np.pi)
    pkt = np.exp(1j * ph)
    x, on = _place(c.n, [pkt], c.rng)
    return _Wave(x, on, bw / m, {"packet_ms": 1000 * pkt.size / fs})


_ACCEPT["lora"] = {"bw", "sf", "payload_symbols"}


# -- the chirp jammer (GNSS L1 "personal privacy device") -------------------
def _res_jammer(cls, fs, rng, p):
    sw = float(p.get("sweep_hz", 2.0e6))
    per = float(p.get("sweep_period_s") or rng.uniform(8e-6, 30e-6))
    p.update(sweep_hz=sw, sweep_period_s=per)
    return sw, 1.0 / per


def _syn_jammer(c: _Ctx) -> _Wave:
    sw, per = c.p["sweep_hz"], c.p["sweep_period_s"]
    t = np.arange(c.n) / c.fs + c.rng.uniform(0, per)
    f = -sw / 2 + sw * np.mod(t / per, 1.0)
    ph = 2 * np.pi * np.cumsum(f) / c.fs + c.rng.uniform(0, 2 * np.pi)
    return _Wave(np.exp(1j * ph), [[0, c.n]], 1.0 / per)


_ACCEPT["chirp_jammer"] = {"sweep_hz", "sweep_period_s"}


# -- linear references -----------------------------------------------------
def _res_linear(cls, fs, rng, p):
    beta = float(p.get("rolloff") or rng.uniform(0.2, 0.5))
    p["rolloff"] = beta
    rs = _resolve_common_rate(p, fs, rng, 1.0 + beta)
    p["symbol_rate"] = rs
    return (1.0 + beta) * rs, rs


def _syn_linear(c: _Ctx) -> _Wave:
    kind = _classes.get(c.cls).native
    rs = c.p["symbol_rate"]
    lead = _lead(rs, c.fs)
    nsym = int(math.ceil(c.n / c.fs * rs)) + 2 * lead
    a = _alphabet(kind)
    sy = a[c.rng.integers(0, a.size, nsym)]
    x, lam = _linear(sy * np.exp(1j * c.rng.uniform(0, 2 * np.pi)), rs,
                     c.p["rolloff"], c.fs, c.n, lead)
    return _Wave(x, [[0, c.n]], rs * lam)


for _k in ("bpsk", "qpsk", "8psk", "16qam", "64qam", "ask4"):
    _ACCEPT[_k] = {"symbol_rate", "bandwidth_hz", "rolloff"}


# -- FSK references -----------------------------------------------------------
def _res_2fsk(cls, fs, rng, p):
    h = float(p.get("h") or rng.uniform(0.6, 1.4))
    p["h"] = h
    rs = _resolve_common_rate(p, fs, rng, h + 1.0)
    p["symbol_rate"] = rs
    return (h + 1.0) * rs, rs


def _syn_2fsk(c: _Ctx) -> _Wave:
    rs, h = c.p["symbol_rate"], c.p["h"]
    lead = _lead(rs, c.fs)
    nsym = int(math.ceil(c.n / c.fs * rs)) + 2 * lead
    lv = np.where(c.rng.integers(0, 2, nsym) == 1, 1.0, -1.0)
    x, lam = _cpm(lv, rs, h * rs / 2.0, "rect", c.fs, c.n, lead)
    return _Wave(x, [[0, c.n]], rs * lam)


_ACCEPT["2fsk"] = {"symbol_rate", "bandwidth_hz", "h"}


def _res_gfsk(cls, fs, rng, p):
    h = float(p.get("h") or 0.5)
    bt = float(p.get("bt") or rng.uniform(0.3, 0.5))
    p.update(h=h, bt=bt)
    rs = _resolve_common_rate(p, fs, rng, h + 0.5)
    p["symbol_rate"] = rs
    return (h + 0.5) * rs, rs


def _syn_gfsk(c: _Ctx) -> _Wave:
    rs, h = c.p["symbol_rate"], c.p["h"]
    lead = _lead(rs, c.fs)
    nsym = int(math.ceil(c.n / c.fs * rs)) + 2 * lead
    lv = np.where(c.rng.integers(0, 2, nsym) == 1, 1.0, -1.0)
    x, lam = _cpm(lv, rs, h * rs / 2.0, "gauss", c.fs, c.n, lead, bt=c.p["bt"])
    return _Wave(x, [[0, c.n]], rs * lam)


_ACCEPT["gfsk"] = {"symbol_rate", "bandwidth_hz", "h", "bt"}


# -- Bluetooth LE ------------------------------------------------------------
BLE_ADV_AA = 0x8E89BED6


def _ble_bits(rng, pdu_bytes: int) -> np.ndarray:
    def lsb_first(v: int, nbits: int):
        return [(v >> i) & 1 for i in range(nbits)]
    aa = lsb_first(BLE_ADV_AA, 32)
    pre = [0, 1] * 4 if aa[0] == 0 else [1, 0] * 4
    header = list(rng.integers(0, 2, 16))
    payload = list(rng.integers(0, 2, 8 * pdu_bytes))
    crc = list(rng.integers(0, 2, 24))
    return np.array(pre + aa + header + payload + crc, dtype=np.int8)


def _res_ble(cls, fs, rng, p):
    p.update(packets=int(p.get("packets", 1)),
             pdu_bytes=int(p.get("pdu_bytes") or rng.integers(6, 38)),
             bt=0.5, h=0.5)
    return 1.1e6, 1.0e6


def _syn_ble(c: _Ctx) -> _Wave:
    """LE 1M advertising packets: GFSK BT 0.5, h 0.5 (±250 kHz), 1 Msym/s;
    preamble, access address 0x8E89BED6 (LSB first), random PDU and CRC."""
    rs = 1.0e6
    bursts = []
    lam = 1.0
    for _ in range(max(1, c.p["packets"])):
        bits = _ble_bits(c.rng, c.p["pdu_bytes"])
        lv = np.where(bits == 1, 1.0, -1.0)
        lead = 16
        lvp = np.concatenate([np.zeros(lead), lv, np.zeros(lead)])
        n_pkt = int(round(bits.size / rs * c.fs))
        x, lam = _cpm(lvp, rs, 250e3, "gauss", c.fs, n_pkt, lead, bt=0.5)
        x = x * np.exp(1j * c.rng.uniform(0, 2 * np.pi))
        bursts.append(x)
    gap = int(round(150e-6 * c.fs))
    xo, on = _place(c.n, bursts, c.rng, min_gap=gap)
    return _Wave(xo, on, rs * lam)


_ACCEPT["ble"] = {"packets", "pdu_bytes"}


# -- OFDM ----------------------------------------------------------------------
def _qam_symbols(rng, mod: str, shape) -> np.ndarray:
    a = _alphabet(mod)
    return a[rng.integers(0, a.size, shape)]


def _ofdm_stream(n_fft: int, used: np.ndarray, cps: list, scs: float, fs: float,
                 n: int, rng, mod: str, special=None, start_sym: int = 0
                 ) -> tuple[np.ndarray, float, float]:
    """An OFDM stream at fs_n = n_fft·scs, CP lengths cycling through `cps`
    (samples at the native rate). `special(sym_index, X_row)` may overwrite
    resource elements (sync, reference signals). Returns (baseband at fs, λ,
    mean OFDM symbol rate)."""
    fs_n = _frac(scs) * n_fft
    up, down, lam = _ratio(fs_n, fs)
    g = _guard_s(fs_n, up, down)
    mean_len = n_fft + float(np.mean(cps))
    nsym = int(math.ceil((n / fs + 2 * g) * float(fs_n) / mean_len)) + 2
    used = np.asarray(used) % n_fft
    X = np.zeros((nsym, n_fft), dtype=complex)
    X[:, used] = _qam_symbols(rng, mod, (nsym, used.size))
    if special is not None:
        for i in range(nsym):
            special(start_sym + i, X[i])
    td = np.fft.ifft(X, axis=1) * (n_fft / math.sqrt(max(1, used.size)))
    parts = []
    for i in range(nsym):
        cp = int(cps[(start_sym + i) % len(cps)])
        parts.append(td[i, n_fft - cp:] if cp else td[i, :0])
        parts.append(td[i])
    x = np.concatenate(parts)
    rate = float(fs_n) / mean_len
    return _render(x, fs_n, fs, n, g, up, down), lam, rate


LTE_FFT = {6: 128, 15: 256, 25: 512, 50: 1024, 75: 1536, 100: 2048}


def _lte_pss(u: int) -> np.ndarray:
    n = np.arange(62)
    d = np.empty(62, dtype=complex)
    a = n[:31]
    d[:31] = np.exp(-1j * np.pi * u * a * (a + 1) / 63.0)
    b = n[31:]
    d[31:] = np.exp(-1j * np.pi * u * (b + 1) * (b + 2) / 63.0)
    return d


def _fits_list(options, width_of, fs, off):
    return [o for o in options
            if width_of(o) <= 0.95 * fs and abs(off) + width_of(o) / 2 <= 0.5 * fs]


def _res_lte(cls, fs, rng, p, off=0.0):
    nrb = p.get("n_rb")
    if nrb is None:
        ok = _fits_list(sorted(LTE_FFT, reverse=True),
                        lambda r: r * 180e3 + 15e3, fs, off)
        nrb = ok[0] if ok else 6
    nrb = int(nrb)
    if nrb not in LTE_FFT:
        raise SynthRefusal(f"LTE has {sorted(LTE_FFT)} resource blocks "
                           "(1.4/3/5/10/15/20 MHz)")
    p.update(n_rb=nrb, n_fft=LTE_FFT[nrb], scs_hz=15000.0,
             qam=p.get("qam") or str(rng.choice(["qpsk", "16qam", "64qam"])),
             cell_id=int(p.get("cell_id", rng.integers(0, 504))))
    return nrb * 180e3 + 15e3, 14000.0


def _syn_lte(c: _Ctx) -> _Wave:
    """LTE downlink, normal CP (160/144 of 2048), PSS (Zadoff-Chu, root by
    the cell's N_ID2) in the last symbol of slots 0 and 10, a BPSK stand-in
    for the SSS before it, cell reference signals (port 0) every 6th
    subcarrier in symbols 0 and 4 of every slot."""
    nrb, nf = c.p["n_rb"], c.p["n_fft"]
    k = np.arange(1, 6 * nrb + 1)
    used = np.concatenate([-k[::-1], k])
    cps = [int(v * nf / 2048) for v in (160, 144, 144, 144, 144, 144, 144)]
    nid2 = c.p["cell_id"] % 3
    pss = _lte_pss((25, 29, 34)[nid2])
    pss_bins = np.concatenate([np.arange(-31, 0), np.arange(1, 32)]) % nf
    vshift = c.p["cell_id"] % 6
    rng = c.rng
    qp = _alphabet("qpsk")

    def special(i, row):
        s = i % 140
        if s in (6, 76):
            row[pss_bins] = pss
        elif s in (5, 75):
            row[pss_bins] = rng.choice([-1.0, 1.0], 62)
        if s % 7 in (0, 4):
            off = vshift if s % 7 == 0 else (vshift + 3) % 6
            rs_bins = used[(np.arange(used.size) % 6) == off]
            row[rs_bins % nf] = qp[rng.integers(0, 4, rs_bins.size)]

    x, lam, rate = _ofdm_stream(nf, used, cps, 15000.0, c.fs, c.n, rng,
                                c.p["qam"], special, int(rng.integers(0, 140)))
    return _Wave(x, [[0, c.n]], 14000.0 * lam,
                 {"cp_lag_s": 1.0 / 15000.0 / lam, "ofdm_rate_hz": rate * lam})


_ACCEPT["ofdm_lte"] = {"n_rb", "qam", "cell_id"}

NR30_RB = (273, 106, 78, 51, 38, 24, 11)


def _res_nr(cls, fs, rng, p, off=0.0):
    nrb = p.get("n_rb")
    if nrb is None:
        ok = _fits_list(NR30_RB, lambda r: r * 360e3, fs, off)
        nrb = ok[0] if ok else 11
    nrb = int(nrb)
    nsc = 12 * nrb
    nf = 1 << int(math.ceil(math.log2(nsc / 0.85)))
    p.update(n_rb=nrb, n_fft=nf, scs_hz=30000.0,
             qam=p.get("qam") or str(rng.choice(["qpsk", "16qam", "64qam", "256qam"])))
    return nrb * 360e3, 28000.0


def _syn_nr(c: _Ctx) -> _Wave:
    """NR downlink at 30 kHz SCS (µ = 1), normal CP: 144/2048 of the FFT,
    +16 on the first symbol of every 0.5 ms. No SSB (stated limit)."""
    nrb, nf = c.p["n_rb"], c.p["n_fft"]
    half = 6 * nrb
    used = np.concatenate([np.arange(-half, 0), np.arange(0, half)])
    cps = [int(160 * nf / 2048)] + [int(144 * nf / 2048)] * 13
    x, lam, rate = _ofdm_stream(nf, used, cps, 30000.0, c.fs, c.n, c.rng,
                                c.p["qam"], None, int(c.rng.integers(0, 14)))
    return _Wave(x, [[0, c.n]], 28000.0 * lam,
                 {"cp_lag_s": 1.0 / 30000.0 / lam, "ofdm_rate_hz": rate * lam})


_ACCEPT["ofdm_nr30"] = {"n_rb", "qam"}

_WIFI_S = {-24: 1 + 1j, -20: -1 - 1j, -16: 1 + 1j, -12: -1 - 1j, -8: -1 - 1j,
           -4: 1 + 1j, 4: -1 - 1j, 8: -1 - 1j, 12: 1 + 1j, 16: 1 + 1j,
           20: 1 + 1j, 24: 1 + 1j}
_WIFI_L = (1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1,
           -1, 1, 1, 1, 1, 0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1,
           1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1)
_WIFI_PILOTS = {-21: 1.0, -7: 1.0, 7: 1.0, 21: -1.0}


def _wifi_frame(rng, n_data: int, mod: str) -> np.ndarray:
    """One 802.11a/g frame at 20 MHz: STF (8 µs), LTF (8 µs), SIGNAL (4 µs),
    `n_data` DATA symbols (4 µs each, 0.8 µs guard)."""
    def spec(d):
        X = np.zeros(64, dtype=complex)
        for k, v in d.items():
            X[k % 64] = v
        return X
    stf = np.fft.ifft(spec({k: v * math.sqrt(13 / 6) for k, v in _WIFI_S.items()}))
    stf = np.tile(stf[:16], 10)
    lt = np.fft.ifft(spec({k: v for k, v in zip(range(-26, 27), _WIFI_L)}))
    ltf = np.concatenate([lt[-32:], lt, lt])
    data_k = [k for k in range(-26, 27) if k != 0 and k not in _WIFI_PILOTS]
    syms = []
    for i in range(n_data + 1):
        m = "bpsk" if i == 0 else mod
        d = dict(zip(data_k, _qam_symbols(rng, m, len(data_k))))
        pol = 1.0 if i == 0 else float(rng.choice([-1.0, 1.0]))
        d.update({k: v * pol for k, v in _WIFI_PILOTS.items()})
        td = np.fft.ifft(spec(d))
        syms.append(np.concatenate([td[-16:], td]))
    fr = np.concatenate([stf, ltf] + syms)
    return fr / math.sqrt(np.mean(np.abs(fr) ** 2))


def _res_wifi(cls, fs, rng, p):
    p.update(frames=int(p.get("frames", 1)),
             data_symbols=int(p.get("data_symbols") or rng.integers(10, 120)),
             qam=p.get("qam") or str(rng.choice(["bpsk", "qpsk", "16qam", "64qam"])))
    return 52 * 312.5e3 + 312.5e3, 250e3


def _syn_wifi(c: _Ctx) -> _Wave:
    fs_n = 20_000_000
    up, down, lam = _ratio(fs_n, c.fs)
    bursts = []
    for _ in range(max(1, c.p["frames"])):
        fr = _wifi_frame(c.rng, c.p["data_symbols"], c.p["qam"])
        fr = fr * np.exp(1j * c.rng.uniform(0, 2 * np.pi))
        pad = int(math.ceil(_guard_s(fs_n, up, down) * fs_n)) + 8
        y = _resample(np.concatenate([np.zeros(pad), fr, np.zeros(pad)]), up, down)
        k0 = int(round(pad * up / down))
        k1 = int(round((pad + fr.size) * up / down))
        bursts.append(y[k0:k1])
    x, on = _place(c.n, bursts, c.rng, min_gap=int(round(16e-6 * c.fs)))
    return _Wave(x, on, 250e3 * lam, {"cp_lag_s": 3.2e-6 / lam})


_ACCEPT["ofdm_wifi"] = {"frames", "data_symbols", "qam"}


def _res_drone(cls, fs, rng, p):
    p.update(on_ms=float(p.get("on_ms") or rng.uniform(1.0, 2.0)),
             off_ms=float(p.get("off_ms") or rng.uniform(0.5, 1.0)),
             qam=p.get("qam") or str(rng.choice(["qpsk", "16qam", "64qam"])))
    return 600 * 15e3 + 15e3, 15000.0 * 1024 / 1152


def _syn_drone(c: _Ctx) -> _Wave:
    """A bursty OFDM link (1024-point FFT, 600 subcarriers at 15 kHz, CP
    1/8) on a TDD-like on/off pattern — a structural stand-in for digital
    drone video links, not a protocol-exact one."""
    k = np.arange(1, 301)
    used = np.concatenate([-k[::-1], k])
    x, lam, rate = _ofdm_stream(1024, used, [128], 15000.0, c.fs, c.n, c.rng,
                                c.p["qam"])
    per = (c.p["on_ms"] + c.p["off_ms"]) / 1000.0
    env, on = _gate(c.n, c.fs, per, c.p["on_ms"] / 1000.0,
                    c.rng.uniform(0, per), 20e-6)
    return _Wave(x * env, on, rate * lam, {"cp_lag_s": 1.0 / 15000.0 / lam})


_ACCEPT["ofdm_drone"] = {"on_ms", "off_ms", "qam"}


def _res_ofdm(cls, fs, rng, p):
    nf = int(p.get("n_fft") or rng.choice([64, 128, 256]))
    frac_used = float(p.get("used_fraction") or rng.uniform(0.7, 0.85))
    cpf = float(p.get("cp_fraction") or rng.choice([0.25, 0.125, 0.0625]))
    nused = 2 * int(frac_used * nf / 2)
    p.update(n_fft=nf, used=nused, cp_fraction=cpf,
             qam=p.get("qam") or str(rng.choice(["qpsk", "16qam", "64qam"])))
    if p.get("bandwidth_hz"):
        bw = float(p["bandwidth_hz"])
    elif p.get("scs_hz"):
        bw = float(p["scs_hz"]) * nused
    else:
        bw = math.exp(rng.uniform(math.log(fs / 40), math.log(fs / 4)))
    p["scs_hz"] = bw / nused
    cp = int(round(cpf * nf))
    return bw, p["scs_hz"] * nf / (nf + cp)


def _syn_ofdm(c: _Ctx) -> _Wave:
    nf, nu = c.p["n_fft"], c.p["used"]
    k = np.arange(1, nu // 2 + 1)
    used = np.concatenate([-k[::-1], k])
    cp = int(round(c.p["cp_fraction"] * nf))
    x, lam, rate = _ofdm_stream(nf, used, [cp], c.p["scs_hz"], c.fs, c.n,
                                c.rng, c.p["qam"])
    return _Wave(x, [[0, c.n]], rate * lam,
                 {"cp_lag_s": 1.0 / c.p["scs_hz"] / lam})


_ACCEPT["ofdm"] = {"n_fft", "used_fraction", "cp_fraction", "qam",
                   "bandwidth_hz", "scs_hz"}


# -- ATSC 8-VSB ----------------------------------------------------------------
ATSC_RS = Fraction(4_500_000 * 684, 286)          # 10.762 237 76 Msym/s


def _res_vsb(cls, fs, rng, p):
    return 6.0e6, float(ATSC_RS)


def _syn_vsb(c: _Ctx) -> _Wave:
    """8 real levels plus the pilot (+1.25), segment sync (+5 −5 −5 +5)
    every 832 symbols, vestigial-sideband shaping: an RRC of Nyquist
    frequency Rs/4 (roll-off 0.1152, the 0.31 MHz transition) centred Rs/4
    above the carrier, so the pilot sits 0.31 MHz above the lower edge."""
    rs = ATSC_RS
    fs_n = 2 * rs
    up, down, lam = _ratio(fs_n, c.fs)
    g = _guard_s(fs_n, up, down)
    nsym = int(math.ceil((c.n / c.fs + 2 * g) * float(rs))) + 64
    lv = np.array([-7.0, -5, -3, -1, 1, 3, 5, 7])[c.rng.integers(0, 8, nsym)]
    ph = int(c.rng.integers(0, 832))
    for s in range(-ph, nsym, 832):
        for j, v in enumerate((5.0, -5.0, -5.0, 5.0)):
            if 0 <= s + j < nsym:
                lv[s + j] = v
    lv = lv + 1.25
    p = rrc_taps(0.1152, 4, 64)                      # T' = 2/Rs = 4 samples
    nn = np.arange(p.size) - (p.size - 1) / 2.0
    hc = p * np.exp(2j * np.pi * nn / 8.0)          # +Rs/4 at fs_n = 2Rs
    from scipy.signal import upfirdn
    y = upfirdn(hc, lv.astype(complex), up=2)
    y = y * np.exp(-2j * np.pi * np.arange(y.size) / 8.0)
    t0 = g + ((p.size - 1) / 2.0) / float(fs_n)
    x = _render(y, fs_n, c.fs, c.n, t0, up, down)
    return _Wave(x * np.exp(1j * c.rng.uniform(0, 2 * np.pi)), [[0, c.n]],
                 float(rs) * lam, {"pilot_offset_hz": -float(rs) / 4 / lam})


_ACCEPT["vsb8"] = set()


# -- analog FPV video (5.8 GHz drones) ---------------------------------------
NTSC_LINE_S = 1.0 / (4.5e6 / 286)                    # 63.556 µs
NTSC_FSC = 4.5e6 * 455 / 572                         # 3.579545 MHz


def _res_fpv(cls, fs, rng, p):
    # representative values, not a measured transmitter: 8 MHz of peak
    # deviation (≈17 MHz occupied), a 6.5 MHz FM audio subcarrier
    p.update(deviation_hz=float(p.get("deviation_hz", 8.0e6)),
             audio_subcarrier_hz=float(p.get("audio_subcarrier_hz", 6.5e6)))
    return 18.0e6, 0.0


def _syn_fpv(c: _Ctx) -> _Wave:
    """NTSC-like composite video (H-sync, colour burst, random smooth luma
    and chroma, vertical blanking) plus an FM audio subcarrier, frequency
    modulated onto the carrier."""
    fs_n = 40_000_000
    up, down, _ = _ratio(fs_n, c.fs)
    g = _guard_s(fs_n, up, down)
    nn = int(math.ceil((c.n / c.fs + 2 * g) * fs_n))
    rng = c.rng
    t = np.arange(nn) / fs_n + rng.uniform(0, 262.5 * NTSC_LINE_S)
    u = np.mod(t, NTSC_LINE_S) * 1e6                  # µs into the line
    line = np.floor(t / NTSC_LINE_S)
    vblank = np.mod(line, 262.5) < 20
    luma = 7.5 + 92.5 * np.clip(0.5 + 0.3 * _lowpass_noise(nn, 1.0e6, fs_n, rng),
                                0, 1)
    chroma = 20 * np.cos(2 * np.pi * NTSC_FSC * t
                         + 2 * np.pi * _lowpass_noise(nn, 2e4, fs_n, rng))
    ire = np.where((u >= 10.9) & (u < 63.5556 - 1.5) & ~vblank, luma + chroma, 0.0)
    burst = (u >= 5.3) & (u < 7.8)
    ire = np.where(burst, 20 * np.sin(2 * np.pi * NTSC_FSC * t), ire)
    ire = np.where(u < 4.7, -40.0, ire)
    v = (ire - 30.0) / 70.0
    aud = np.clip(_lowpass_noise(nn, 3000.0, fs_n, rng), -1.0, 1.0)
    sc = np.cos(2 * np.pi * c.p["audio_subcarrier_hz"] * t
                + 2 * np.pi * 50e3 * np.cumsum(aud) / fs_n)
    f = c.p["deviation_hz"] * (v + 0.1 * sc)
    x = np.exp(1j * 2 * np.pi * np.cumsum(f) / fs_n)
    return _Wave(_render(x, fs_n, c.fs, c.n, g, up, down), [[0, c.n]], 0.0,
                 {"line_rate_hz": 1.0 / NTSC_LINE_S})


_ACCEPT["fpv_analog"] = {"deviation_hz", "audio_subcarrier_hz"}


# -- negatives ------------------------------------------------------------------
def _res_zero(cls, fs, rng, p):
    return 0.0, 0.0


def _syn_noise(c: _Ctx) -> _Wave:
    return _Wave(np.zeros(c.n, complex), [])


def _syn_tone(c: _Ctx) -> _Wave:
    return _Wave(np.full(c.n, np.exp(1j * c.rng.uniform(0, 2 * np.pi))),
                 [[0, c.n]])


def _syn_dc(c: _Ctx) -> _Wave:
    """The DC spike: a constant with a slow wander (LO leakage drifts with
    temperature), always at 0 Hz."""
    w = 1.0 + 0.05 * _lowpass_noise(c.n, 5.0, c.fs, c.rng)
    return _Wave(w * np.exp(1j * c.rng.uniform(0, 2 * np.pi)), [[0, c.n]])


for _k in ("noise", "tone", "dc"):
    _ACCEPT[_k] = set()

# kind -> (resolve, synthesise)
KINDS = {
    "nfm": (_res_nfm, _syn_nfm), "am": (_res_am, _syn_am),
    "wfm": (_res_wfm, _syn_wfm),
    "c4fm_4800": (_res_c4fm, _syn_c4fm), "c4fm_2400": (_res_c4fm, _syn_c4fm),
    "dmr_4800": (_res_dmr, _syn_dmr),
    "pocsag": (_res_pocsag, _syn_pocsag), "flex": (_res_flex, _syn_flex),
    "adsb": (_res_adsb, _syn_adsb), "lora": (_res_lora, _syn_lora),
    "chirp_jammer": (_res_jammer, _syn_jammer),
    "bpsk": (_res_linear, _syn_linear), "qpsk": (_res_linear, _syn_linear),
    "8psk": (_res_linear, _syn_linear), "16qam": (_res_linear, _syn_linear),
    "64qam": (_res_linear, _syn_linear), "ask4": (_res_linear, _syn_linear),
    "2fsk": (_res_2fsk, _syn_2fsk), "gfsk": (_res_gfsk, _syn_gfsk),
    "ble": (_res_ble, _syn_ble),
    "ofdm_lte": (_res_lte, _syn_lte), "ofdm_nr30": (_res_nr, _syn_nr),
    "ofdm_wifi": (_res_wifi, _syn_wifi), "ofdm_drone": (_res_drone, _syn_drone),
    "ofdm": (_res_ofdm, _syn_ofdm), "vsb8": (_res_vsb, _syn_vsb),
    "fpv_analog": (_res_fpv, _syn_fpv),
    "noise": (_res_zero, _syn_noise), "tone": (_res_zero, _syn_tone),
    "dc": (_res_zero, _syn_dc),
}
_OFFSET_AWARE = {"ofdm_lte", "ofdm_nr30"}
_FIXED_AT_DC = {"dc"}
_NARROW = {"tone", "dc"}


# ---------------------------------------------------------------------------
# the public API
# ---------------------------------------------------------------------------
def supported() -> list[str]:
    """Every class the native generator makes (all with a `native` key)."""
    return [c.name for c in _classes.CLASSES if c.native]


def _class(cls: str):
    c = _classes.get(cls)
    if c is None:
        raise SynthRefusal(f"{cls!r} is not in the class table "
                           f"(detect.classes); known: {', '.join(supported())}")
    if not c.native or c.native not in KINDS:
        raise SynthRefusal(f"{c.label} has no native generator")
    return c


def _resolve(cls: str, fs: float, rng, params: dict | None, off: float):
    c = _class(cls)
    kind = c.native
    p = dict(params or {})
    bad = sorted(set(p) - _ACCEPT[kind])
    if bad:
        raise SynthRefusal(
            f"{c.label} does not take {', '.join(map(repr, bad))}; it takes "
            + (", ".join(sorted(_ACCEPT[kind])) or "no parameters"))
    for k, v in _CLASS_PRESETS.get(cls, {}).items():
        p.setdefault(k, v)
    res, syn = KINDS[kind]
    if kind in _OFFSET_AWARE:
        bw, rs = res(cls, fs, rng, p, off)
    else:
        bw, rs = res(cls, fs, rng, p)
    return c, kind, p, float(bw), float(rs), syn


def _fit_words(c, p, bw, fs, off) -> str:
    what = c.label
    if c.native == "ofdm_lte":
        what = f"LTE downlink at {p['n_rb']} resource blocks"
    elif c.native == "ofdm_nr30":
        what = f"NR downlink at {p['n_rb']} resource blocks (30 kHz)"
    if bw > fs:
        return (f"{what} is {_hz_words(bw)} wide; {_rate_words(fs)} holds at "
                f"most {_hz_words(fs)}. It cannot be generated at this "
                "profile's rate — use a narrower configuration where the "
                "class has one, a receiver profile with a higher rate, or "
                "clip=True to see the slice a receiver here would hear.")
    return (f"{what} ({_hz_words(bw)} wide) at {off / 1e3:+.1f} kHz from the "
            f"centre would reach past the edge of the {_rate_words(fs)} band "
            f"(±{_hz_words(fs / 2)}) and wrap around. Move it inward or use "
            "clip=True.")


def can_generate(cls: str, fs: float, params: dict | None = None,
                 carrier_offset_hz: float = 0.0) -> tuple[bool, str]:
    """(ok, why) without generating anything. Random defaults are drawn from
    a fixed seed, so a class whose configuration is random (LoRa's channel
    width) is judged on one draw that fits when any fits."""
    try:
        c, kind, p, bw, rs, _ = _resolve(cls, float(fs), np.random.default_rng(0),
                                         params, float(carrier_offset_hz))
    except SynthRefusal as e:
        return False, str(e)
    off = 0.0 if kind in _FIXED_AT_DC else float(carrier_offset_hz)
    if bw > fs or abs(off) + bw / 2.0 > fs / 2.0:
        return False, _fit_words(c, p, bw, float(fs), off)
    return True, ""


def nominal_bandwidth(cls: str, fs: float, params: dict | None = None) -> float:
    """The class's design bandwidth at this rate (before generating)."""
    _, _, _, bw, _, _ = _resolve(cls, float(fs), np.random.default_rng(0),
                                 params, 0.0)
    return bw


def _jsonable(v):
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(a) for a in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(a) for k, a in v.items()}
    return v


def waveform(cls: str, fs: float, n_samples: int, rng: np.random.Generator,
             *, carrier_offset_hz: float = 0.0, params: dict | None = None,
             clip: bool = False) -> tuple[np.ndarray, dict]:
    """The clean signal of one class: `n_samples` complex64 at `fs`, unit
    mean power over its on-samples, centred `carrier_offset_hz` from the
    capture centre. Returns (s, label) with label["snr_db"] None.

    Refuses (SynthRefusal, in words) a signal wider than `fs` can hold or
    one that would wrap past the band edge — unless `clip=True`, which
    renders it at a higher working rate and passes it through the receiver's
    anti-alias filter, so what comes back is the slice a receiver tuned here
    would see (label["clipped"] True, the label's edges the visible part)."""
    fs = float(fs)
    n = int(n_samples)
    if fs <= 0 or n <= 0:
        raise SynthRefusal("a sample rate and a length must both be positive")
    off_req = float(carrier_offset_hz)
    c, kind, p, bw, rs_nom, syn = _resolve(cls, fs, rng, params, off_req)
    off = 0.0 if kind in _FIXED_AT_DC else off_req
    if kind in _FIXED_AT_DC and off_req:
        p["note"] = "the DC spike is at 0 Hz by definition; the offset was ignored"
    fits = bw <= fs and abs(off) + bw / 2.0 <= fs / 2.0
    if not fits and not clip:
        raise SynthRefusal(_fit_words(c, p, bw, fs, off))
    m = 1
    if not fits:
        m = max(2, int(math.ceil(2.0 * (abs(off) + bw / 2.0) / fs * 1.1)))
    margin = 64 if m > 1 else 0
    fs_w = fs * m
    n_w = (n + 2 * margin) * m
    wave = syn(_Ctx(c.name, fs_w, n_w, rng, p))
    x = np.asarray(wave.x, dtype=np.complex128)
    if wave.on:
        idx = np.concatenate([np.arange(s, s + k) for s, k in wave.on])
        pw = float(np.mean(np.abs(x[idx]) ** 2)) if idx.size else 0.0
        if pw > 0:
            x = x / math.sqrt(pw)
    x = _shift(x, off, fs_w)
    on = wave.on
    visible = 1.0
    if m > 1:
        from atk_diffusion.dsp import resample as _rs
        full = float(np.mean(np.abs(x) ** 2)) if x.size else 0.0
        y, _ = _rs.decimate(x, m, fs_w)
        x = y[margin:margin + n]
        on = []
        for s, k in wave.on:
            a = max(0, int(math.floor(s / m)) - margin)
            b = min(n, int(math.ceil((s + k) / m)) - margin)
            if b > a:
                on.append([a, b - a])
        vis = float(np.mean(np.abs(x) ** 2)) if x.size else 0.0
        visible = vis / full if full > 0 else 0.0
        if visible < 1e-6 or not on:
            raise SynthRefusal(f"none of {c.label} at {off / 1e6:+.3f} MHz "
                               f"falls inside the {_rate_words(fs)} span")
        idx = np.concatenate([np.arange(s, s + k) for s, k in on])
        pw = float(np.mean(np.abs(x[idx]) ** 2))
        x = x / math.sqrt(pw)
    x = x.astype(np.complex64)
    if on:
        idx = np.concatenate([np.arange(s, s + k) for s, k in on])
        seg = x[idx]
        if kind in _NARROW:
            t_on = idx.size / fs
            f_lo, f_hi = off - 0.5 / t_on, off + 0.5 / t_on
        else:
            f_lo, f_hi = occupied_band(seg, fs)
        start = int(on[0][0])
        count = int(on[-1][0] + on[-1][1] - start)
    else:
        f_lo = f_hi = off
        start, count = 0, n
    rs_real = float(wave.symbol_rate or 0.0)
    params_out = {k: v for k, v in p.items()}
    params_out.update(wave.params or {})
    if m > 1:
        params_out["visible_power_fraction"] = round(visible, 4)
    label = {
        "cls": c.name, "family": c.family,
        "bandwidth_hz": float(f_hi - f_lo) if on else 0.0,
        "symbol_rate_hz": rs_real,
        "carrier_offset_hz": off,
        "snr_db": None,
        "sample_start": start, "sample_count": count,
        "f_lo_hz": float(f_lo), "f_hi_hz": float(f_hi),
        "bursts": [[int(s), int(k)] for s, k in on],
        "native": kind, "fs": fs, "generator": GENERATOR,
        "params": _jsonable(params_out), "clipped": bool(m > 1),
        "nominal_bandwidth_hz": bw,
    }
    return x, label


def noise(n: int, power: float, rng: np.random.Generator) -> np.ndarray:
    """Complex white Gaussian noise of total power `power`."""
    return (math.sqrt(power / 2.0) * (rng.standard_normal(n)
                                      + 1j * rng.standard_normal(n))
            ).astype(np.complex64)


def signal_amplitude(snr_db: float, bandwidth_hz: float, noise_power: float,
                     fs: float) -> float:
    """The amplitude a unit-power signal needs for `snr_db` by
    SNR_DEFINITION: P_s = SNR · N0 · B with N0 = noise_power / fs."""
    b = max(float(bandwidth_hz), float(fs) / 1e9)
    return math.sqrt(10.0 ** (float(snr_db) / 10.0) * noise_power / float(fs) * b)


def generate(cls: str, fs: float, n_samples: int, snr_db: float,
             rng: np.random.Generator, *, carrier_offset_hz: float = 0.0,
             noise_dbfs: float = DEFAULT_NOISE_DBFS,
             params: dict | None = None, clip: bool = False
             ) -> tuple[np.ndarray, dict]:
    """One labelled example: the class's signal at `snr_db` (SNR_DEFINITION)
    plus complex white noise of total power `noise_dbfs` over the whole
    band. The noise class returns noise alone with snr_db NaN."""
    fs = float(fs)
    n = int(n_samples)
    pn = 10.0 ** (float(noise_dbfs) / 10.0)
    if cls == "noise":
        _class(cls)
        lab = {"cls": "noise", "family": _classes.get("noise").family,
               "bandwidth_hz": 0.0, "symbol_rate_hz": 0.0,
               "carrier_offset_hz": 0.0, "snr_db": float("nan"),
               "sample_start": 0, "sample_count": n, "f_lo_hz": 0.0,
               "f_hi_hz": 0.0, "bursts": [], "native": "noise", "fs": fs,
               "generator": GENERATOR, "params": {}, "clipped": False,
               "nominal_bandwidth_hz": 0.0, "noise_dbfs": float(noise_dbfs)}
        if params:
            raise SynthRefusal("noise only takes no parameters")
        return noise(n, pn, rng), lab
    s, lab = waveform(cls, fs, n, rng, carrier_offset_hz=carrier_offset_hz,
                      params=params, clip=clip)
    a = signal_amplitude(snr_db, lab["bandwidth_hz"], pn, fs)
    x = (a * s + noise(n, pn, rng)).astype(np.complex64)
    lab["snr_db"] = float(snr_db)
    lab["noise_dbfs"] = float(noise_dbfs)
    return x, lab
