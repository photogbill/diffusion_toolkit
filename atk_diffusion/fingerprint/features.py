# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Classical RF fingerprints — specific emitter identification by hardware
imperfections, measured (plan C1, the classical half).

*"Is this our radio? Is this the beacon we heard yesterday?"* Two radios of
the same model differ in their oscillators, their I/Q modulators, their
power amplifiers and the way they key up. Each difference is a number a
classical measurement can read off a burst, and an analyst can check:

    cfo_ppm             carrier offset from the channel, in ppm of the carrier
                        (the transmitter's crystal, less the receiver's own
                        offset when the profile knows it)
    tx_irr_db           image-rejection ratio of the transmitter's I/Q
                        modulator — the circularity of the de-rotated signal.
                        A receiver never sees the transmitter's own I/Q axes,
                        so only this rotation-invariant part of the gain /
                        phase imbalance is identifiable; the gain and phase
                        split is reported for the RECEIVER's axes (`rx_*`)
    tx_dc_db            carrier leakage (LO feed-through): the de-rotated mean
                        relative to the signal, dB
    rise_time_ms        turn-on transient: 30 % -> 90 % of the settled
                        amplitude (noise power subtracted first; measured at
                        10 dB SNR and above)
    overshoot_pct       how far the envelope overshoots on key-up
    keyup_offset_hz     the synthesiser's frequency error in the first 3 ms
                        after key-up, against its settled frequency
    phase_noise_rms_hz  phase-noise proxy: RMS of the instantaneous frequency
                        about its smoothed track, thermal part removed in
                        quadrature — measured only on an UNMODULATED burst
                        (a key-up with no speech); otherwise not reported
    acpr_upper_db,      spectral regrowth: adjacent-channel power relative to
    acpr_lower_db       the channel, noise floor subtracted (the PA's
                        nonlinearity)
    symbol_clock_ppm    symbol-rate offset from nominal (digital modes, given
                        the nominal rate; informative only on long bursts)

THE RECEIVER'S OWN FINGERPRINT is in every capture (plan §3.1: "the
mirror of §4.C"): its LO offset, its I/Q imbalance, its DC spike. It is
HELD CONSTANT by the receiver profile — fingerprints are compared only
within one profile (`library`), the receiver's measured ppm is removed when
the profile carries it, and its I/Q imbalance and DC are reported
separately (`rx_*`) as a check that the receiver has not changed, never as
features of the transmitter.

THE CHANNEL CONFOUND, NOTED. Multipath reshapes the turn-on envelope and
the spectrum; Doppler adds v/c of the carrier to the offset (0.1 ppm at
30 m/s); low SNR inflates the phase-noise proxy and buries regrowth. Each
feature's sensitivity is stated beside it (`CONFOUND`), the SNR travels
with every fingerprint, and the learned model (`learn.fingerprint`) is
trained with random channels precisely to be resilient to this.

Every value is MEASURED tier: a classical number from the record.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from atk_diffusion import provenance as _prov

_prov.METHOD_TIERS.setdefault("fingerprint_features", "measured")

#: Transmitter features, in vector order, with units.
FEATURES = (("cfo_ppm", "ppm"), ("tx_irr_db", "dB"), ("tx_dc_db", "dB"),
            ("rise_time_ms", "ms"), ("overshoot_pct", "%"),
            ("keyup_offset_hz", "Hz"), ("phase_noise_rms_hz", "Hz"),
            ("acpr_upper_db", "dBc"), ("acpr_lower_db", "dBc"),
            ("symbol_clock_ppm", "ppm"))
NAMES = tuple(n for n, _ in FEATURES)

#: How the channel and the receiver can move each feature.
CONFOUND = {
    "cfo_ppm": "Doppler adds v/c (0.1 ppm at 30 m/s); the receiver's own LO "
               "offset is removed only when the profile carries it",
    "tx_irr_db": "robust to flat fading; frequency-selective multipath adds a "
                 "small improper part; low SNR biases it toward circular",
    "tx_dc_db": "robust to the channel; the receiver's DC spike is at the "
                "receiver's centre, not the transmitter's, unless CFO is ~0",
    "rise_time_ms": "multipath and AGC reshape the envelope",
    "overshoot_pct": "multipath and AGC reshape the envelope",
    "keyup_offset_hz": "robust to the channel; needs the key-up in the "
                       "capture; modulation adds scatter, so it is averaged",
    "phase_noise_rms_hz": "inflated by noise (corrected in quadrature); only "
                          "measured on an unmodulated burst (a key-up with no "
                          "speech), else not reported",
    "acpr_upper_db": "noise-limited at low SNR; adjacent signals contaminate it",
    "acpr_lower_db": "noise-limited at low SNR; adjacent signals contaminate it",
    "symbol_clock_ppm": "needs a long burst: resolution ~1/duration",
}


@dataclass
class Fingerprint:
    values: dict
    diagnostics: dict = field(default_factory=dict)
    snr_db: float | None = None
    profile: str = ""
    notes: list = field(default_factory=list)
    tier: str = "measured"

    def vector(self, names=NAMES) -> np.ndarray:
        return np.array([self.values.get(n, np.nan) if self.values.get(n) is not None
                         else np.nan for n in names], dtype=np.float64)

    def to_json(self) -> dict:
        return {"values": self.values, "diagnostics": self.diagnostics,
                "snr_db": self.snr_db, "profile": self.profile,
                "notes": self.notes, "tier": self.tier}

    @classmethod
    def from_json(cls, d: dict) -> "Fingerprint":
        return cls(dict(d.get("values", {})), dict(d.get("diagnostics", {})),
                   d.get("snr_db"), d.get("profile", ""), list(d.get("notes", [])),
                   d.get("tier", "measured"))


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------
def _movavg(x: np.ndarray, n: int) -> np.ndarray:
    n = max(1, int(n))
    if n == 1:
        return np.asarray(x, dtype=np.float64)
    k = np.ones(n) / n
    return np.convolve(np.asarray(x, dtype=np.float64), k, mode="same")


def burst_bounds(x, fs: float, threshold_db: float | None = None,
                 smooth_ms: float = 0.5) -> tuple[int, int, float, float]:
    """(start, end, floor power, burst power). The floor is the quietest
    fiftieth of the smoothed power — so the capture must hold some noise
    before the key-up (a few percent of it), as a cut with margins does. The threshold sits
    halfway (in dB) between the floor and the burst, never under 3 dB, so
    a 5 dB burst is found and a 30 dB burst is cut at its shoulders."""
    p = _movavg(np.abs(np.asarray(x)) ** 2, smooth_ms * 1e-3 * fs)
    floor = max(float(np.percentile(p, 2)), 1e-30)
    top = float(np.percentile(p, 90))
    if threshold_db is None:
        threshold_db = max(3.0, 0.5 * 10 * math.log10(max(top / floor, 1.0)))
    above = np.nonzero(p > floor * 10 ** (threshold_db / 10))[0]
    # smoothed noise alone spans ~3 dB between its 5th and 90th percentiles
    if above.size == 0 or top < 2.5 * floor:
        raise ValueError("no burst above the noise floor in this capture")
    s, e = int(above[0]), int(above[-1]) + 1
    burst = float(np.median(p[s:e]))
    if burst < 2.0 * floor:
        raise ValueError("no burst above the noise floor in this capture")
    return s, e, floor, burst


def carrier_offset_hz(x, fs: float, method: str = "mean_freq", order: int = 4) -> float:
    """Carrier offset of a burst. "mean_freq": the power-weighted mean
    instantaneous frequency (any modulation symmetric about its carrier:
    FM, FSK, C4FM). "power_law": the M-th power line (PSK, order M)."""
    x = np.asarray(x, dtype=np.complex128)
    if method == "mean_freq":
        return float(np.angle(np.sum(x[1:] * np.conj(x[:-1]))) * fs / (2 * math.pi))
    if method == "power_law":
        y = x ** int(order)
        n = 1 << int(math.ceil(math.log2(max(len(y), 16)) + 2))
        S = np.abs(np.fft.fft(y, n))
        i = int(np.argmax(S))
        a, b, c = S[i - 1], S[(i + 1) % n], S[i]
        den = a - 2 * c + b
        off = 0.5 * (a - b) / den if den < 0 else 0.0
        f = (i + off) * fs / n
        f = (f + fs / 2) % fs - fs / 2
        return float(f / int(order))
    raise ValueError("method is 'mean_freq' or 'power_law'")


def circularity(y) -> float:
    """|E[y^2]| / E[|y|^2] of a zero-mean signal: 0 for a proper signal."""
    y = np.asarray(y, dtype=np.complex128)
    y = y - y.mean()
    return float(abs(np.mean(y * y)) / max(np.mean(np.abs(y) ** 2), 1e-30))


def irr_db_from_circularity(rho: float) -> float:
    """Image-rejection ratio |nu/mu|^2 (dB, negative) for an I/Q imbalance
    y = mu s + nu s* of a proper s: rho = 2|mu nu| / (|mu|^2 + |nu|^2)."""
    rho = min(max(float(rho), 0.0), 1.0)
    r = rho / (1.0 + math.sqrt(1.0 - rho * rho))      # = (1 - sqrt(1-rho^2)) / rho
    return 20.0 * math.log10(max(r, 1e-15))


def iq_imbalance(x) -> dict:
    """Gain (dB) and phase (deg) imbalance in the capture's own I/Q axes,
    and the image-rejection ratio. On a raw capture of a signal with a
    carrier offset these are the RECEIVER's (the transmitter's improper
    part spins at twice the offset and averages away)."""
    x = np.asarray(x, dtype=np.complex128)
    x = x - x.mean()
    i, q = x.real, x.imag
    pi, pq = float(np.mean(i * i)), float(np.mean(q * q))
    piq = float(np.mean(i * q))
    gain = 10.0 * math.log10(max(pi, 1e-30) / max(pq, 1e-30))
    phase = math.degrees(math.asin(max(-1.0, min(1.0, piq / math.sqrt(max(pi * pq, 1e-60))))))
    return {"gain_db": gain, "phase_deg": phase,
            "irr_db": irr_db_from_circularity(circularity(x))}


def turn_on(x, fs: float, start: int, end: int, floor_p: float = 0.0,
            smooth_ms: float = 0.25, keyup_ms: float = 3.0) -> dict:
    """Rise time (30 -> 90 % of the settled amplitude, the noise power
    subtracted first: a 10 % mark sits inside the noise below ~25 dB SNR
    and would measure the noise, not the radio), overshoot, and the
    frequency error of the first `keyup_ms` after key-up against the
    settled frequency."""
    x = np.asarray(x, dtype=np.complex128)
    pw = _movavg(np.abs(x) ** 2, smooth_ms * 1e-3 * fs) - float(floor_p)
    env = np.sqrt(np.maximum(pw, 0.0))
    mid = slice(start + (end - start) // 4, end - (end - start) // 4)
    steady = float(np.median(env[mid])) if end > start + 8 else float(np.max(env))
    base = max(0, start - int(2e-3 * fs))
    look = env[base:min(end, start + int(10e-3 * fs))]
    try:
        i30 = base + int(np.nonzero(look >= 0.3 * steady)[0][0])
        i90 = base + int(np.nonzero(look >= 0.9 * steady)[0][0])
        rise = (i90 - i30) / fs * 1e3
    except IndexError:
        rise = float("nan")
    seg = env[start:min(end, start + int(10e-3 * fs))]
    peak = float(np.max(_movavg(seg, 0.25e-3 * fs))) if seg.size else steady
    overshoot = 100.0 * max(peak - steady, 0.0) / max(steady, 1e-30)
    inst = np.angle(x[1:] * np.conj(x[:-1])) * fs / (2 * math.pi)
    w = np.abs(x[1:]) ** 2
    k0 = start + int(0.2e-3 * fs)
    k1 = start + int(keyup_ms * 1e-3 * fs)
    early = float(np.sum(inst[k0:k1] * w[k0:k1]) / max(np.sum(w[k0:k1]), 1e-30))
    late = float(np.sum(inst[mid] * w[mid]) / max(np.sum(w[mid]), 1e-30))
    return {"rise_time_ms": rise, "overshoot_pct": overshoot,
            "keyup_offset_hz": early - late}


def unmodulated(y, fs: float, width_hz: float = 100.0) -> bool:
    """True when most of a de-rotated burst's power is within +/- width of
    its carrier — a key-up with no speech or data."""
    Y = np.abs(np.fft.fft(np.asarray(y, dtype=np.complex128))) ** 2
    f = np.fft.fftfreq(Y.size, 1.0 / fs)
    return float(Y[np.abs(f) <= width_hz].sum() / max(Y.sum(), 1e-30)) > 0.5


def phase_noise_rms_hz(y, fs: float, snr_lin: float, smooth_ms: float = 5.0) -> float:
    """RMS of the instantaneous frequency about its smoothed track (Hz),
    with the thermal-noise part (fs^2 / (4 pi^2 SNR)) removed in
    quadrature."""
    y = np.asarray(y, dtype=np.complex128)
    f = np.angle(y[1:] * np.conj(y[:-1])) * fs / (2 * math.pi)
    r = f - _movavg(f, smooth_ms * 1e-3 * fs)
    edge = int(smooth_ms * 1e-3 * fs)
    if r.size > 4 * edge:
        r = r[edge:-edge]
    raw = float(np.mean(r ** 2))
    thermal = fs * fs / (4 * math.pi ** 2 * max(snr_lin, 1e-3))
    return math.sqrt(max(raw - thermal, 0.0))


def acpr(y, fs: float, channel_bw: float, spacing: float,
         noise_psd: float | None = None) -> tuple:
    """(upper, lower) adjacent-channel power ratio, dBc, Welch PSD of the
    de-rotated burst, with the noise floor's share of each band removed.
    A band whose power is not clearly above its noise share is NOT
    measurable at this SNR and comes back None."""
    from scipy.signal import welch
    if spacing + channel_bw / 2 > fs / 2:
        return None, None
    f, p = welch(np.asarray(y, dtype=np.complex128), fs=fs,
                 nperseg=min(1024, len(y)), return_onesided=False,
                 scaling="density")
    df = float(np.abs(f[1] - f[0])) if f.size > 1 else fs

    def band(lo, hi):
        sel = (f >= lo) & (f < hi)
        raw = float(np.sum(p[sel]) * df)
        nz = (noise_psd or 0.0) * float(np.count_nonzero(sel)) * df
        return raw - nz, raw, nz
    main, _, _ = band(-channel_bw / 2, channel_bw / 2)
    out = []
    for lo_f in (spacing, -spacing):
        pw, raw, nz = band(lo_f - channel_bw / 2, lo_f + channel_bw / 2)
        if main <= 0 or raw < 2.0 * nz or pw <= 0:
            out.append(None)
        else:
            out.append(10 * math.log10(pw / main))
    return out[0], out[1]


def symbol_rate_offset(y, fs: float, nominal: float, span_ppm: float = 2000.0
                       ) -> tuple[float, float, float]:
    """(measured rate Hz, offset ppm, line SNR dB): the spectral line at the
    symbol rate of the squared derivative of the (half-symbol smoothed)
    instantaneous frequency — the FSK symbol clock — found in a zero-padded
    FFT within +/- span_ppm of nominal and refined by a parabola. The line
    SNR is against the spectrum around it."""
    y = np.asarray(y, dtype=np.complex128)
    f = np.angle(y[1:] * np.conj(y[:-1]))
    f = _movavg(f, max(1.0, 0.5 * fs / nominal))
    d = np.diff(f)
    s = d * d
    s = (s - s.mean()) * np.hanning(s.size)
    n = 1 << int(math.ceil(math.log2(s.size * 8)))
    S = np.abs(np.fft.rfft(s, n))
    fr = np.arange(S.size) * fs / n
    lo, hi = nominal * (1 - span_ppm * 1e-6), nominal * (1 + span_ppm * 1e-6)
    sel = np.nonzero((fr >= lo) & (fr <= hi))[0]
    if sel.size < 3:
        return float("nan"), float("nan"), 0.0
    i = int(sel[np.argmax(S[sel])])
    a, b, c = S[i - 1], S[i + 1], S[i]
    den = a - 2 * c + b
    off = 0.5 * (a - b) / den if den < 0 else 0.0
    rate = float((i + off) * fs / n)
    around = S[(fr > nominal * 0.8) & (fr < nominal * 1.2)]
    noise = float(np.median(around)) + 1e-30
    return rate, (rate - nominal) / nominal * 1e6, 20 * math.log10(float(S[i]) / noise)


# ---------------------------------------------------------------------------
def extract(x, fs: float, center_hz: float | None = None, *,
            symbol_rate: float | None = None, channel_bw: float = 11_000.0,
            channel_spacing: float = 12_500.0, cfo_method: str = "mean_freq",
            rx_ppm: float | None = None, profile: str = "",
            threshold_db: float | None = None) -> Fingerprint:
    """The fingerprint of one burst. `x` is a cut at complex baseband
    (the signal near 0 Hz, its key-up inside the capture), `fs` its rate,
    `center_hz` the absolute frequency of 0 Hz (for ppm). `rx_ppm` is the
    receiver's measured LO offset from its profile, removed when given."""
    x = np.asarray(x, dtype=np.complex128)
    if x.size < int(0.005 * fs):
        raise ValueError("a burst shorter than 5 ms cannot be fingerprinted")
    s, e, floor_p, burst_p = burst_bounds(x, fs, threshold_db)
    notes = []
    pre = x[:max(0, s - int(1e-3 * fs))]
    if pre.size >= int(2e-3 * fs):
        floor_p = float(np.mean(np.abs(pre) ** 2))        # the noise, measured
    else:
        notes.append("less than 2 ms of noise before the key-up: the floor is "
                     "estimated from the quietest part of the capture")
    snr_lin = max(burst_p / max(floor_p, 1e-30) - 1.0, 1e-3)
    settle = int(0.010 * fs)
    core = x[min(s + settle, e - 1):e] if e - s > 2 * settle else x[s:e]
    cfo = carrier_offset_hz(core, fs, cfo_method)
    vals: dict = {}
    if center_hz:
        ppm = cfo / float(center_hz) * 1e6
        if rx_ppm is not None:
            ppm -= float(rx_ppm)
        else:
            notes.append("the receiver's own ppm offset is not in its profile; "
                         "cfo_ppm includes it (same for every emitter seen by "
                         "this receiver)")
        vals["cfo_ppm"] = ppm
    else:
        vals["cfo_ppm"] = None
        notes.append("no centre frequency: carrier offset reported in Hz only")
    t = np.arange(core.size) / fs
    y = core * np.exp(-2j * math.pi * cfo * t)
    mean = complex(np.mean(y))
    sig_p = float(np.mean(np.abs(y - mean) ** 2))
    vals["tx_dc_db"] = 10 * math.log10(max(abs(mean) ** 2, 1e-30) / max(sig_p, 1e-30))
    vals["tx_irr_db"] = irr_db_from_circularity(circularity(y))
    vals.update(turn_on(x, fs, s, e, floor_p))
    if snr_lin < 10.0:
        vals["rise_time_ms"] = vals["overshoot_pct"] = None
        notes.append("below 10 dB SNR the key-up envelope is noise: rise time "
                     "and overshoot not reported")
    if unmodulated(y - mean, fs):
        vals["phase_noise_rms_hz"] = phase_noise_rms_hz(y, fs, snr_lin)
    else:
        vals["phase_noise_rms_hz"] = None
        notes.append("the burst is modulated: the phase-noise proxy needs an "
                     "unmodulated key-up and is not reported")
    noise_psd = floor_p / fs
    up, lo = acpr(y, fs, channel_bw, channel_spacing, noise_psd)
    vals["acpr_upper_db"], vals["acpr_lower_db"] = up, lo
    if up is None or lo is None:
        notes.append("adjacent-channel power is under the noise floor at this "
                     "SNR: spectral regrowth not measurable")
    if symbol_rate:
        rate, ppm_clk, line = symbol_rate_offset(y, fs, symbol_rate)
        vals["symbol_clock_ppm"] = ppm_clk if line > 10.0 else None
        if line <= 10.0:
            notes.append(f"no symbol-rate line above 10 dB at {symbol_rate:g} Hz "
                         "(analog, or too short a burst)")
    else:
        vals["symbol_clock_ppm"] = None
    rx = iq_imbalance(x[s:e])
    diag = {"cfo_hz": cfo, "rx_gain_db": rx["gain_db"], "rx_phase_deg": rx["phase_deg"],
            "rx_irr_db": rx["irr_db"],
            "rx_dc_db": 10 * math.log10(max(abs(complex(np.mean(x[s:e]))) ** 2, 1e-30)
                                        / max(float(np.mean(np.abs(x[s:e]) ** 2)), 1e-30)),
            "burst_start_s": s / fs, "burst_s": (e - s) / fs}
    for k, v in list(vals.items()):
        if isinstance(v, float) and not math.isfinite(v):
            vals[k] = None
    return Fingerprint(vals, diag, 10 * math.log10(snr_lin), profile, notes)
