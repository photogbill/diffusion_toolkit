# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The receiver impairment model: measure a receiver once, terminated, and
make synthetic data sound like THAT receiver (plan §3.4, track A).

Bill, 2026-10-08: *"the bladeRF files won't work for the RTL-SDR or KrakenSDR
or HackRF files."* What makes them not work is mostly here: the noise floor
and its spectral shape (the front end's filters), the DC spike (LO leakage),
I/Q gain and phase imbalance (the image), spurs the receiver makes itself, and
the converter's resolution. A detector trained on a textbook receiver has
never seen any of it. So, per profile, from a capture with NO ANTENNA (a 50 Ω
terminator on the port):

    measure_impairments(x, fs, datatype)  -> the dict below
    store(rf, profile_id, impairments)    -> profiles\\<profile>.json
    apply_impairments(x, fs, impairments, rng, datatype=…)
                                          -> synthetic data through this receiver

THE DICT (every key is a measurement of the record; tier "measured")

    floor_db_per_bin        256 values, dBFS: the noise POWER falling in each
                            of 256 equal bins from -fs/2 to +fs/2 (fftshift
                            order), so the bins sum (in linear power) to the
                            floor's total power. Spurs and DC are removed first.
    floor_mean_dbfs         that total noise power, dBFS (a full-scale complex
                            tone is 0 dBFS) — what iq.power_dbfs reads on the
                            terminated capture once DC and spurs are out
    dc_offset_i, _q         the DC offset itself (full-scale units)
    dc_spike_db             how far the DC spike stands above the floor in the
                            DC bin of the 256-bin floor: 10·log10(|dc|² / P_bin)
    iq_gain_imbalance_db    20·log10(1+ε), ε = Q's amplitude over I's minus one
    iq_phase_imbalance_deg  departure from quadrature, degrees
    image_rejection_db      what those two imply (ATK's own formula)
    enob_est                effective bits the receiver delivers AT THIS GAIN:
                            (0 dBFS − floor_mean_dbfs − 1.76) / 6.02, capped at
                            the converter's bits. A terminated capture cannot
                            show distortion, so this is an upper bound on what a
                            full-scale signal would get.
    spurs                   [{offset_hz, level_db, level_above_floor_db}] —
                            level_db is the tone's power in dBFS
    measured_at, n_samples, fs, datatype, adc_bits, method, tier, nfft

I/Q IMBALANCE IS BILL'S MODEL, PORTED. ATK's `iq_balance.py` / `iq_correction`
(Bill's code, tested against his HackRF's 17.6 dB image) define it:

    Q_out = g · (Q·cos φ + I·sin φ),   g = 1 + ε,  I untouched
    ε = sqrt(E[Q²]/E[I²]) − 1,   φ = asin(E[I·Q] / sqrt(E[I²]·E[Q²]))

Thermal noise is circular, so on a terminated capture these statistics see
only the receiver. `apply_impairments` uses the same equation forwards, so
measure(apply(x)) returns what was applied (the tests prove it).

ORDER, chosen so the two directions invert each other:
    apply:    floor shape (filters signal AND noise, as the front end does)
              → spurs → I/Q imbalance → DC → quantisation (the file's own
              datatype, through `dsp.iq`, so clipping and levels are exact)
    measure:  DC (mean) → I/Q (Bill's estimator) → correct I/Q → spurs (fitted
              and subtracted) → floor → ENOB

SPURS ARE FOUND WITH A THRESHOLD DERIVED, NOT TUNED: a bin of an average of K
periodograms of noise is Gamma(K, 1/K) distributed about the local median, so
the threshold is set where the chance that ANY of the nfft bins exceeds it is
1e-3. Each candidate is then fitted by projection (frequency refined to a
fraction of a bin), which gives its level to a tenth of a dB.

WHERE IT IS FILED. `profiles\\<p>.json` `impairments` is shared with
`dsp.floor`, whose `NoiseFloor.to_impairments` owns the key
`floor_db_per_bin` in ITS units (fft_size bins of |FFT(x·w)|²/(Σw)² at the
profile's STFT geometry) and whose `NoiseFloor.from_profile` refuses an array
of another length. So `store` files this module's 256-bin dBFS shape as
`floor_shape_db_256` and never overwrites `floor_db_per_bin`;
`apply_impairments` uses `floor_shape_db_256`, or dsp.floor's shape when that
is all a profile has (only the SHAPE is used — the level is
`floor_mean_dbfs`). `measure_impairments` itself returns `floor_db_per_bin`
(256 points) exactly as specified.

LIMITS, stated: the floor shape is assumed to be the front end's power
response and is applied to signals as well as noise (true of the IF/anti-alias
filters, not of noise added after them); impairments that depend on signal
level (compression, intermodulation, AGC) cannot be measured terminated and are
not modelled; the measurement is valid for the gain, frequency and firmware it
was taken at — re-measure when any of them changes (the file carries the date
and the device serial).
"""

from __future__ import annotations


import math
import time

import numpy as np

from atk_diffusion import provenance
from atk_diffusion.dsp import iq as _iq

provenance.METHOD_TIERS.setdefault("measure_impairments", "measured")
# Impairments applied to anything produce data no receiver recorded.
provenance.METHOD_TIERS.setdefault("apply_impairments", "invented")

FLOOR_BINS = 256
SPUR_NFFT = 8192
SPUR_FALSE_ALARM = 1e-3        # chance that ANY bin of a noise-only PSD fires
MAX_SPURS = 32
DC_GUARD_BINS = 3              # spur search ignores |f| < this many bins
MIN_SAMPLES = 16 * FLOOR_BINS


class ImpairmentError(ValueError):
    """A measurement could not be made; the message says why."""


# ---------------------------------------------------------------------------
# small numerics
# ---------------------------------------------------------------------------
def _avg_periodogram(x: np.ndarray, nfft: int, overlap: float = 0.5,
                     max_segments: int = 4096) -> tuple[np.ndarray, int]:
    """Hann-windowed averaged periodogram, POWER PER BIN (sum = mean power),
    fftshift order. Returns (p, segments_used)."""
    x = np.asarray(x, dtype=np.complex64)
    n = x.size
    if n < nfft:
        raise ImpairmentError(f"{n} samples is fewer than one {nfft}-point "
                              "segment")
    hop = max(1, int(nfft * (1.0 - overlap)))
    starts = np.arange(0, n - nfft + 1, hop)
    if starts.size > max_segments:
        idx = np.linspace(0, starts.size - 1, max_segments).round().astype(int)
        starts = starts[idx]
    w = np.hanning(nfft + 2)[1:-1].astype(np.float32)
    acc = np.zeros(nfft, dtype=np.float64)
    step = 256
    for i in range(0, starts.size, step):
        seg = np.stack([x[s:s + nfft] for s in starts[i:i + step]])
        f = np.fft.fft(seg * w, axis=1)
        acc += np.sum(np.abs(f) ** 2, axis=0)
    p = acc / starts.size / float(np.sum(w ** 2)) / nfft
    return np.fft.fftshift(p), int(starts.size)


def _freqs(nfft: int, fs: float) -> np.ndarray:
    return (np.arange(nfft) - nfft // 2) * (float(fs) / nfft)


def _db(p) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(p, dtype=np.float64), 1e-30))


def image_rejection_db(gain_err: float, phase_deg: float) -> float:
    """Image rejection for amplitude ratio 1+ε and phase error φ, capped at
    90 dB — ATK's `iq_correction.image_rejection_db`, ported."""
    g = 1.0 + float(gain_err)
    c = math.cos(math.radians(float(phase_deg)))
    num = 1.0 + 2.0 * g * c + g * g
    den = 1.0 - 2.0 * g * c + g * g
    if den <= num * 1e-9:
        return 90.0
    return min(90.0, 10.0 * math.log10(num / den))


def _iq_stats(x: np.ndarray) -> tuple[float, float]:
    """Bill's blind estimator on DC-free samples -> (ε, φ degrees)."""
    i = x.real.astype(np.float64)
    q = x.imag.astype(np.float64)
    n = max(1, x.size)
    p_i = float(np.dot(i, i) / n)
    p_q = float(np.dot(q, q) / n)
    if p_i <= 0 or p_q <= 0:
        return 0.0, 0.0
    s = float(np.dot(i, q) / n) / math.sqrt(p_i * p_q)
    s = max(-1.0, min(1.0, s))
    return math.sqrt(p_q / p_i) - 1.0, math.degrees(math.asin(s))


def correct_iq(x: np.ndarray, gain_err: float, phase_deg: float) -> np.ndarray:
    """Undo Bill's model: x_Q = (Q / g − I·sin φ) / cos φ (ATK iq_balance)."""
    g = 1.0 + float(gain_err)
    phi = math.radians(float(phase_deg))
    i = x.real.astype(np.float64)
    q = x.imag.astype(np.float64)
    q2 = (q / g - i * math.sin(phi)) / math.cos(phi)
    return (i + 1j * q2).astype(np.complex64)


def apply_iq(x: np.ndarray, gain_db: float, phase_deg: float) -> np.ndarray:
    """Bill's model forwards: Q_out = g·(Q·cos φ + I·sin φ)."""
    g = 10.0 ** (float(gain_db) / 20.0)
    phi = math.radians(float(phase_deg))
    i = x.real.astype(np.float64)
    q = x.imag.astype(np.float64)
    q2 = g * (q * math.cos(phi) + i * math.sin(phi))
    return (i + 1j * q2).astype(np.complex64)


def _gamma_threshold(k: int, n_bins: int, p_any: float) -> float:
    """Ratio over the local mean that a K-average noise bin exceeds with
    probability p_any / n_bins (Gamma(K, 1/K))."""
    from scipy.special import gammainccinv
    k = max(1, int(k))
    p_bin = max(1e-300, float(p_any) / max(1, int(n_bins)))
    return float(gammainccinv(k, p_bin) / k)


def _golden_max(fun, a: float, b: float, tol: float) -> float:
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = fun(c), fun(d)
    for _ in range(80):
        if (b - a) <= tol:
            break
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = fun(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = fun(d)
    return 0.5 * (a + b)


def _fit_tone(x: np.ndarray, fs: float, f0: float, bin_hz: float,
              n_fit: int) -> tuple[float, complex]:
    """Refine a tone found in a `bin_hz`-resolution spectrum, coarse to fine:
    maximise the projection |mean(x·e^{-j2πfn/fs})| over a fit length that
    grows ×4 per stage, each search kept inside the previous stage's main
    lobe (a long projection is a few Hz wide; searching a whole bin with it
    lands on a sidelobe). The searching runs on block sums of the signal
    demodulated at f0 (offsets stay under 1/32 of a block's bandwidth, so the
    block sum costs < 0.02 dB); the amplitude is then one exact projection
    over all `n_fit` samples. Returns (f, complex amplitude)."""
    seg = np.asarray(x[:n_fit], dtype=np.complex128)
    nn = np.arange(seg.size, dtype=np.float64)
    y = seg * np.exp(-2j * np.pi * f0 / fs * nn)
    blk = max(1, int(round(fs / bin_hz)) // 32)
    nb = y.size // blk
    if nb < 4:
        blk, nb = 1, y.size
    b = y[:nb * blk].reshape(nb, blk).sum(axis=1)
    tb = (np.arange(nb, dtype=np.float64) * blk + 0.5 * (blk - 1)) / fs

    def make(nuse):
        bb = b[:nuse]
        tt = tb[:nuse]
        return lambda d: float(abs(np.sum(bb * np.exp(-2j * np.pi * d * tt))))

    delta = 0.0
    width = bin_hz                      # main-lobe half-width so far
    nuse = max(4, int(round(fs / bin_hz)) // blk)
    while True:
        nuse = min(nuse, nb)
        tspan = nuse * blk / fs
        half = min(width, 1.0 / tspan)
        delta = _golden_max(make(nuse), delta - 0.9 * half, delta + 0.9 * half,
                            tol=1e-3 / tspan)
        width = 1.0 / tspan
        if nuse >= nb:
            break
        nuse *= 4
    f = f0 + delta
    a = np.mean(seg * np.exp(-2j * np.pi * f / fs * nn))
    return float(f), complex(a)


# ---------------------------------------------------------------------------
# measure
# ---------------------------------------------------------------------------
def default_adc_bits(datatype: str | None) -> int | None:
    if not datatype:
        return None
    d = _iq.norm_dt(datatype)
    return {"cu8": 8, "ci8": 8, "ci16": 16, "ci16q11": 12, "cf32": None}[d]


def measure_impairments(x_terminated, fs: float, datatype: str,
                        adc_bits: int | None = None) -> dict:
    """Measure a receiver from a TERMINATED capture (no antenna). `x` is
    complex, unit full scale (as `dsp.iq.to_complex` / `sigmf.load` return
    it). See the module docstring for every key. Raises ImpairmentError in
    words when the capture cannot support a measurement."""
    x = np.asarray(x_terminated, dtype=np.complex64).ravel()
    fs = float(fs)
    if fs <= 0:
        raise ImpairmentError("a sample rate must be positive")
    if x.size < MIN_SAMPLES:
        raise ImpairmentError(
            f"{x.size} samples is too short to measure a receiver; record at "
            f"least {MIN_SAMPLES} (a second or more is better) with the "
            "antenna port terminated.")
    dt = _iq.norm_dt(datatype) if datatype else "cf32"
    bits = int(adc_bits) if adc_bits else default_adc_bits(dt)
    clip = _iq.clipped_fraction(x, dt)
    if clip > 1e-3:
        raise ImpairmentError(
            f"{100 * clip:.2f} % of the samples sit at the converter's rails. "
            "A terminated capture should be nowhere near full scale — lower "
            "the gain, check that the antenna port really is terminated, and "
            "record again.")
    # 1. DC
    dc = complex(np.mean(x.astype(np.complex128)))
    x1 = (x - np.complex64(dc)).astype(np.complex64)
    # 2. I/Q imbalance (Bill's estimator), 3. correct it
    eps, phi = _iq_stats(x1)
    x2 = correct_iq(x1, eps, phi)
    # 4. spurs: averaged periodogram, threshold from the noise statistics
    nfft = SPUR_NFFT
    while nfft > FLOOR_BINS and x2.size < 8 * nfft:
        nfft //= 2
    p, k = _avg_periodogram(x2, nfft, overlap=0.0)
    from scipy.ndimage import median_filter
    local = median_filter(p, size=65, mode="wrap")
    # the median of a Gamma(K,1/K) bin sits slightly below its mean
    local = local / max(1e-6, 1.0 - 1.0 / (3.0 * k))
    thr = _gamma_threshold(k, nfft, SPUR_FALSE_ALARM)
    ratio = p / np.maximum(local, 1e-30)
    f = _freqs(nfft, fs)
    bin_hz = fs / nfft
    cand = np.nonzero(ratio > thr)[0]
    cand = cand[np.abs(f[cand]) > DC_GUARD_BINS * bin_hz]
    peaks = []
    for c in cand:
        lo, hi = max(0, c - 1), min(nfft - 1, c + 1)
        if ratio[c] >= ratio[lo] and ratio[c] >= ratio[hi]:
            peaks.append(int(c))
    peaks.sort(key=lambda c: -ratio[c])
    peaks = peaks[:MAX_SPURS]
    n_fit = int(min(x2.size, 1 << 21))
    x3 = x2.astype(np.complex128)
    nn = np.arange(x3.size, dtype=np.float64)
    fitted = []
    for c in peaks:
        f0, a = _fit_tone(x3, fs, float(f[c]), bin_hz, n_fit)
        if any(abs(f0 - s["offset_hz"]) < 1.5 * bin_hz for s in fitted):
            continue
        x3 -= a * np.exp(2j * np.pi * f0 / fs * nn)
        fitted.append({"offset_hz": f0, "amplitude": a})
    x3 = x3.astype(np.complex64)
    # 5. floor: 256 bins, power per bin
    pf, _ = _avg_periodogram(x3, FLOOR_BINS)
    total = float(np.sum(pf))
    if total <= 0:
        raise ImpairmentError("the capture holds no noise at all — it is "
                              "silent or constant; it cannot describe a "
                              "receiver.")
    floor_mean = 10.0 * math.log10(total)
    floor_db = _db(pf)
    dc_bin = 10.0 ** (floor_db[FLOOR_BINS // 2] / 10.0)
    dc_pow = abs(dc) ** 2
    spurs = []
    ff = _freqs(FLOOR_BINS, fs)
    for s in fitted:
        level = 10.0 * math.log10(max(abs(s["amplitude"]) ** 2, 1e-30))
        kb = int(np.argmin(np.abs(ff - s["offset_hz"])))
        spurs.append({"offset_hz": round(float(s["offset_hz"]), 3),
                      "level_db": round(level, 2),
                      "level_above_floor_db": round(level - float(floor_db[kb]), 2)})
    spurs.sort(key=lambda s: s["offset_hz"])
    # 6. ENOB from the floor: a full-scale complex tone is 0 dBFS
    enob = (-floor_mean - 1.76) / 6.02
    if bits:
        enob = min(enob, float(bits))
    gain_db = 20.0 * math.log10(1.0 + eps)
    return {
        "floor_db_per_bin": [round(float(v), 3) for v in floor_db],
        "floor_bins": FLOOR_BINS,
        "floor_units": "dBFS: noise power per bin, 256 equal bins from -fs/2 "
                       "(fftshift order); the bins sum to floor_mean_dbfs",
        "floor_mean_dbfs": round(floor_mean, 3),
        "dc_offset_i": float(dc.real), "dc_offset_q": float(dc.imag),
        "dc_spike_db": round(10.0 * math.log10(max(dc_pow, 1e-30) / dc_bin), 2),
        "iq_gain_imbalance_db": round(gain_db, 4),
        "iq_phase_imbalance_deg": round(phi, 4),
        "image_rejection_db": round(image_rejection_db(eps, phi), 2),
        "enob_est": round(float(enob), 2),
        "spurs": spurs,
        "spur_threshold_db": round(10.0 * math.log10(thr), 2),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_samples": int(x.size),
        "fs": fs, "datatype": dt, "adc_bits": bits,
        "clipped_fraction": float(clip),
        "nfft": {"floor": FLOOR_BINS, "spurs": int(nfft)},
        "method": "measure_impairments",
        "tier": provenance.tier_for("measure_impairments"),
    }


def measure_sigmf(path, max_samples: int = 8_000_000,
                  skip_seconds: float = 0.1) -> dict:
    """Measure a terminated SigMF capture. The first `skip_seconds` are
    skipped (tuner settling); at most `max_samples` are used."""
    from atk_diffusion import profiles as _profiles
    from atk_diffusion import sigmf as _sigmf
    meta = _sigmf.read_meta(path)
    fs = _sigmf.sample_rate_of(meta)
    if fs <= 0:
        raise ImpairmentError(f"{_sigmf.base_of(path).name} has no sample "
                              "rate in its metadata.")
    start = int(round(skip_seconds * fs))
    x = _sigmf.load(path, start=start, count=int(max_samples), meta=meta)
    if x.ndim > 1:
        raise ImpairmentError("measure one channel at a time (a Kraken "
                              "capture is five receivers; split it first).")
    out = measure_impairments(x, fs, _sigmf.datatype_of(meta))
    out["source_capture"] = _sigmf.base_of(path).name
    try:
        out["profile"] = _profiles.profile_from_meta(meta)
    except ValueError:
        pass
    return out


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------
def shape_filter(floor_db_per_bin, ntaps: int = 1025) -> np.ndarray:
    """A complex FIR whose power response follows the measured floor shape
    (normalised: sum |h|² = 1, so white noise keeps its power). Frequency
    sampling on a dense grid, Hann-windowed."""
    s = 10.0 ** (np.asarray(floor_db_per_bin, dtype=np.float64) / 10.0)
    if s.size < 2 or not np.all(np.isfinite(s)):
        raise ImpairmentError("the floor shape is not a list of numbers")
    nb = s.size
    grid = 4096
    # bin centres in fftshift order: -fs/2 + (k)fs/nb ... ; periodic interp
    fb = (np.arange(nb) - nb // 2) / nb
    fg = (np.arange(grid) - grid // 2) / grid
    xp = np.concatenate([fb - 1.0, fb, fb + 1.0])
    yp = np.concatenate([s, s, s])
    sg = np.interp(fg, xp, yp)
    amp = np.sqrt(sg / np.mean(sg))
    h = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(amp)))
    ntaps = int(ntaps) | 1
    c = grid // 2
    h = h[c - ntaps // 2: c + ntaps // 2 + 1] * np.hanning(ntaps + 2)[1:-1]
    h = h / math.sqrt(float(np.sum(np.abs(h) ** 2)))
    return h.astype(np.complex64)


def quantise(x: np.ndarray, datatype: str, adc_bits: int | None = None
             ) -> np.ndarray:
    """Through the file format itself: exactly the levels and the clipping a
    receiver writing `datatype` produces. A 12-bit converter in a ci16 file
    (the bladeRF) is quantised at its Q11 scale."""
    dt = _iq.norm_dt(datatype)
    if dt == "cf32":
        return np.asarray(x, dtype=np.complex64)
    if dt == "ci16" and adc_bits and int(adc_bits) <= 12:
        dt = "ci16q11"
    return _iq.to_complex(_iq.from_complex(x, dt), dt)


def apply_impairments(x, fs: float, impairments: dict,
                      rng: np.random.Generator,
                      adc_bits: int | None = None,
                      datatype: str | None = None) -> np.ndarray:
    """Pass `x` (signals plus white noise at the receiver's floor level,
    unit full scale) through a measured receiver: floor shape, spurs, I/Q
    imbalance, DC, quantisation. An empty `impairments` applies nothing but
    the quantisation (when a datatype is given). Spur phases come from `rng`.

    The caller sets the levels: for the floor to come out where it was
    measured, the white noise in `x` must have total power
    10^(floor_mean_dbfs/10) — `noise_power(impairments)` returns it."""
    y = np.asarray(x, dtype=np.complex64).copy()
    imp = impairments or {}
    fs = float(fs)
    n = y.size
    shape = floor_shape(imp)
    if shape:
        from scipy.signal import oaconvolve
        h = shape_filter(shape)
        y = oaconvolve(y, h, mode="same").astype(np.complex64)
    spurs = imp.get("spurs") or []
    if spurs and n:
        nn = np.arange(n, dtype=np.float64)
        acc = np.zeros(n, dtype=np.complex128)
        for s in spurs:
            f = float(s.get("offset_hz", 0.0))
            if abs(f) >= fs / 2:
                continue
            lvl = s.get("level_db")
            if lvl is None:
                continue
            a = 10.0 ** (float(lvl) / 20.0)
            ph = float(rng.uniform(0, 2 * np.pi))
            acc += a * np.exp(1j * (2 * np.pi * f / fs * nn + ph))
        y = (y + acc).astype(np.complex64)
    gdb = float(imp.get("iq_gain_imbalance_db", 0.0) or 0.0)
    pdeg = float(imp.get("iq_phase_imbalance_deg", 0.0) or 0.0)
    if gdb or pdeg:
        y = apply_iq(y, gdb, pdeg)
    dci = imp.get("dc_offset_i")
    dcq = imp.get("dc_offset_q")
    if dci is not None or dcq is not None:
        y = y + np.complex64(complex(float(dci or 0.0), float(dcq or 0.0)))
    elif imp.get("dc_spike_db") is not None and shape:
        p_bin = 10.0 ** (float(shape[len(shape) // 2]) / 10.0)
        a = math.sqrt(p_bin * 10.0 ** (float(imp["dc_spike_db"]) / 10.0))
        y = y + np.complex64(a * np.exp(1j * rng.uniform(0, 2 * np.pi)))
    dt = datatype or imp.get("datatype")
    if dt:
        bits = adc_bits if adc_bits is not None else imp.get("adc_bits")
        y = quantise(y, dt, bits)
    return y.astype(np.complex64)


def noise_power(impairments: dict | None, default_dbfs: float = -30.0
                ) -> float:
    """The white-noise power (linear, full-scale units) a generator should
    add so the floor lands where this receiver's was measured."""
    imp = impairments or {}
    v = imp.get("floor_mean_dbfs")
    return 10.0 ** ((float(v) if v is not None else float(default_dbfs)) / 10.0)


def floor_shape(impairments: dict | None) -> list | None:
    """The floor's spectral shape a synthetic receiver should have: this
    module's 256-bin shape as filed by `store` (`floor_shape_db_256`), a
    fresh measurement dict's `floor_db_per_bin`, or — when that is all a
    profile holds — dsp.floor's fft_size-bin shape (only the shape is used)."""
    imp = impairments or {}
    for key in ("floor_shape_db_256", "floor_db_per_bin"):
        v = imp.get(key)
        if v is not None and len(v) >= 2:
            return list(v)
    return None


def is_measured(impairments: dict | None) -> bool:
    imp = impairments or {}
    return floor_shape(imp) is not None and imp.get("floor_mean_dbfs") is not None


def describe(impairments: dict | None) -> str:
    """One line for status displays and manifests."""
    imp = impairments or {}
    if not is_measured(imp):
        return ("receiver impairments not measured — synthetic data uses a "
                "textbook receiver (record a terminated capture and run "
                "dsp.impair.measure_sigmf)")
    sp = imp.get("spurs") or []
    return (f"measured {imp.get('measured_at', '?')}: floor "
            f"{imp['floor_mean_dbfs']:.1f} dBFS, DC +{imp.get('dc_spike_db', 0):.1f} dB, "
            f"image rejection {imp.get('image_rejection_db', 0):.1f} dB, "
            f"ENOB ≈ {imp.get('enob_est', 0):.1f}, {len(sp)} spur"
            f"{'' if len(sp) == 1 else 's'}")


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------
def store(rf, profile_id: str, impairments: dict, device_serial: str = "",
          firmware: str = ""):
    """Write the measurement into `profiles\\<profile>.json` (through
    profiles.load_profile / save_profile, which also records the file in the
    write log). Returns the path. The profile's own note says when and on
    which device it was measured — re-measure when either changes."""
    from atk_diffusion import profiles as _profiles
    if not isinstance(impairments, dict) or not is_measured(impairments):
        raise ImpairmentError("that is not a measurement (no floor) — run "
                              "measure_impairments on a terminated capture "
                              "first.")
    pid = str(profile_id).strip().lower()
    _profiles.parse_profile_id(pid)
    fs = impairments.get("fs")
    pr = _profiles.parse_profile_id(pid)
    if fs and abs(float(fs) - pr.sample_rate) > 0.5:
        raise _profiles.ProfileMismatch(
            f"this measurement was made at {float(fs):g} S/s; "
            f"{_profiles.describe(pid)} runs at {pr.sample_rate} S/s. A "
            "receiver's impairments are measured at its profile's own rate.")
    prof = _profiles.load_profile(rf, pid)
    imp = dict(prof.impairments or {})         # keep what other tools filed
    meas = dict(impairments)
    # dsp.floor owns 'floor_db_per_bin' / 'floor_units' (its geometry, its
    # units): this measurement's shape is filed beside them, never over them
    for mine, filed in (("floor_db_per_bin", "floor_shape_db_256"),
                        ("floor_units", "floor_shape_units"),
                        ("floor_bins", "floor_shape_bins")):
        if mine in meas:
            meas[filed] = meas.pop(mine)
    imp.update(meas)
    if device_serial:
        imp["device_serial"] = str(device_serial)
        prof.device_serial = str(device_serial)
    if firmware:
        imp["firmware"] = str(firmware)
        prof.firmware = str(firmware)
    prof.impairments = imp
    prof.notes = [n for n in prof.notes if not str(n).startswith("defaults")]
    prof.notes.append(
        f"impairments measured {imp.get('measured_at', '?')} from "
        f"{imp.get('n_samples', '?')} terminated samples"
        + (f" on serial {device_serial}" if device_serial else "")
        + (f", firmware {firmware}" if firmware else "")
        + " — re-measure when the device, firmware or gain preset changes")
    return _profiles.save_profile(rf, prof)
