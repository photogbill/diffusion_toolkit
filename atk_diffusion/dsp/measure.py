# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Classical, checkable measurements of a cut (DETECTION_DESIGN §4,
*Measurement, classical*; §4.1; ARCHITECTURE §4.2 `dsp.measure`).

*"These are numbers the bench can show and an analyst can check; they are
not the model's opinion. Where the classifier's regressed cycle parameters
disagree with the classical ones, the classical ones are shown and the
disagreement is logged."* Every result here carries `tier = "measured"` and a
`method` sentence saying exactly how it was computed, so it can be redone by
hand.

    psd                Welch power spectral density, power per Hz
    noise_floor        the floor per Hz: median of the PSD bins, corrected
                       for the Welch average's own median bias
    occupied_bandwidth the band holding 99 % of the power above the floor
    snr_above_floor    in-band power over the floor in that band, in dB,
                       with its standard error and whether it is measurable
    cyclic_peaks       peaks of Bill's lag-domain cyclic profile that clear a
                       threshold DERIVED from a false-alarm rate
    symbol_rate        blind symbol rate: the fundamental of the strongest
                       harmonic comb in the cyclic profile, refined
    carrier_offset     conjugate x² line ÷ 2 (BPSK/AM-class), x⁴ ÷ 4
                       (QPSK-class), else the centre of the occupied band
    envelope_bursts    burst start/stop times from the smoothed envelope
    burst_stats        burst length, duty cycle, PRI (constant, staggered,
                       jittered) from burst start times

THE FLOOR IS THE HARD PART, and it is said so. The median of a PSD is the
floor only when at least half the span is noise. A cut is drawn around a
signal, so inside a cut the signal may fill most of the band: the cut
records the floor measured on the SOURCE span at cut time
(`atk:floor_per_hz`), and every function here takes `floor_per_hz` to use
it. An energy measurement below the floor's own uncertainty is reported as
"not measurable" with its standard error, never as a confident number —
that is the energy detector's SNR wall, and it is why the cyclic
measurements exist.
"""

from __future__ import annotations

import math

import numpy as np

TIER = "measured"


def _x1(x) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim > 1:
        x = x[0]
    return x.astype(np.complex128, copy=False)


# ---------------------------------------------------------------------------
# Spectra and the floor
# ---------------------------------------------------------------------------
def psd(x, fs: float, nfft: int = 1024, overlap: float = 0.5,
        max_frames: int = 4096) -> dict:
    """Welch PSD (Hann window), power per Hz, fftshifted, so that
    Σ psd·rbw ≈ mean power. Returns {freqs_hz, psd, rbw_hz, enbw_hz,
    frames, method}."""
    x = _x1(x)
    fs = float(fs)
    n = int(min(nfft, max(16, x.size)))
    hop = max(1, int(n * (1.0 - overlap)))
    frames = max(1, (x.size - n) // hop + 1)
    step = max(1, frames // max_frames)
    starts = np.arange(0, frames, step)[:max_frames] * hop
    w = np.hanning(n)
    idx = starts[:, None] + np.arange(n)[None, :]
    X = np.fft.fft(x[idx] * w[None, :], axis=1)
    p = np.mean(np.abs(X) ** 2, axis=0) / (fs * np.sum(w ** 2))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / fs))
    return {"freqs_hz": freqs, "psd": np.fft.fftshift(p),
            "rbw_hz": fs / n, "enbw_hz": fs * np.sum(w ** 2) / np.sum(w) ** 2,
            "frames": int(starts.size),
            "method": f"Welch, Hann, {n}-point FFT, {int(starts.size)} frames"}


def _median_bias(frames: int) -> float:
    """Median of the mean of K i.i.d. Exp(1): the Welch average's median
    sits this far below its mean (ln 2 for one frame, → 1 for many)."""
    from scipy.special import gammaincinv
    k = max(1, int(frames))
    return float(gammaincinv(k, 0.5) / k)


def noise_floor(x, fs: float, nfft: int = 1024, spec: dict | None = None
                ) -> dict:
    """The noise floor per Hz: the median PSD bin, divided by the Welch
    average's median bias. Valid when at least half the span is noise; the
    spread across bins (dB) is returned so a lumpy floor is visible."""
    sp = spec or psd(x, fs, nfft)
    p = sp["psd"]
    # the PASSBAND only: a canonical cut is low-passed, and its stopband
    # bins (tens of dB down) would drag a plain median to nothing
    top = float(np.percentile(p, 90)) if p.size else 0.0
    keep = p > 1e-3 * top
    if not keep.any() or top <= 0:
        return {"floor_per_hz": 0.0, "floor_db_per_hz": -np.inf,
                "spread_db": 0.0, "tier": TIER, "method": "no power"}
    med = float(np.median(p[keep])) / _median_bias(sp["frames"])
    lp = 10 * np.log10(p[keep])
    spread = float(1.4826 * np.median(np.abs(lp - np.median(lp))))
    return {"floor_per_hz": med, "floor_db_per_hz": 10 * math.log10(med),
            "spread_db": spread, "tier": TIER,
            "passband_fraction": float(keep.mean()),
            "method": ("median PSD bin of the passband (bins within 30 dB of "
                       "the 90th percentile), corrected for the Welch "
                       f"average's median bias ({sp['method']}); valid when "
                       "at least half the passband is noise")}


def occupied_bandwidth(x, fs: float, fraction: float = 0.99,
                       nfft: int = 4096, floor_per_hz: float | None = None
                       ) -> dict:
    """The band holding `fraction` of the power ABOVE the floor (the energy
    rule, not x-dB-down: a rounded shoulder and a brick wall can share a
    −20 dB width and occupy wildly different bands). Edges are offsets from
    the cut's centre, in Hz."""
    x = _x1(x)
    nfft = int(min(nfft, max(64, 2 ** int(np.log2(max(64, x.size // 16))))))
    sp = psd(x, fs, nfft)
    p = sp["psd"]
    floor = (float(floor_per_hz) if floor_per_hz
             else noise_floor(x, fs, spec=sp)["floor_per_hz"])
    # only bins SIGNIFICANTLY above the floor count: subtracting the floor
    # from thousands of noise bins leaves their positive halves, which add
    # up and stretch a 6 kHz signal across the whole cut (measured)
    sig = p > floor * (1.0 + 3.0 / math.sqrt(max(1, sp["frames"])))
    lin = np.where(sig, np.maximum(p - floor, 0.0), 0.0)
    total = float(lin.sum())
    out = {"tier": TIER, "resolution_hz": sp["rbw_hz"],
           "method": (f"{fraction:.0%} of the power above the floor, counting "
                      "bins 3 standard errors above it "
                      f"({sp['method']})"), "caveats": []}
    if total <= 0:
        out.update({"value_hz": None, "lower_hz": None, "upper_hz": None,
                    "centre_hz": None, "peak_over_floor_db": 0.0})
        out["caveats"].append("no power above the floor")
        return out
    c = np.cumsum(lin) / total
    lo_i = int(np.searchsorted(c, (1 - fraction) / 2.0))
    hi_i = int(np.searchsorted(c, 1 - (1 - fraction) / 2.0))
    f = sp["freqs_hz"]
    lo, hi = float(f[min(lo_i, f.size - 1)]), float(f[min(hi_i, f.size - 1)])
    peak = float(10 * np.log10(max(p.max(), 1e-300) / max(floor, 1e-300)))
    out.update({"value_hz": max(0.0, hi - lo + sp["rbw_hz"]),
                "lower_hz": lo, "upper_hz": hi, "centre_hz": 0.5 * (lo + hi),
                "peak_over_floor_db": peak})
    if out["value_hz"] < 3 * sp["rbw_hz"]:
        out["caveats"].append("only a few FFT bins wide — measure with a "
                              "longer FFT before trusting it")
    if hi_i >= f.size - 2 or lo_i <= 1:
        out["caveats"].append("the power reaches the edge of the cut — the "
                              "true bandwidth may be larger than the cut")
    if peak < 3.0:
        out["caveats"].append("the signal is within 3 dB of the floor: the "
                              "edges are uncertain")
    return out


def snr_above_floor(x, fs: float, band: tuple | None = None,
                    floor_per_hz: float | None = None, nfft: int = 1024,
                    floor_spread_db: float | None = None) -> dict:
    """SNR in dB above the floor inside `band` = (lo_hz, hi_hz), offsets from
    the cut's centre (default: the 99 % occupied band).

        snr = (mean PSD in band − floor) / floor

    with a standard error from the Welch average's own variance and the
    floor's spread across bins (receiver ripple). `measurable` is False when
    the excess is under three standard errors: then the energy cannot be
    told from the floor's own uncertainty — the SNR wall — and `snr_db` is
    the best estimate, reported with that warning, not as a reading."""
    x = _x1(x)
    sp = psd(x, fs, nfft)
    p, f = sp["psd"], sp["freqs_hz"]
    nf = noise_floor(x, fs, spec=sp)
    floor = float(floor_per_hz) if floor_per_hz else nf["floor_per_hz"]
    spread = nf["spread_db"] if floor_spread_db is None else floor_spread_db
    if band is None:
        ob = occupied_bandwidth(x, fs, floor_per_hz=floor)
        if ob["value_hz"] is None:
            band = (-0.25 * fs, 0.25 * fs)
        else:
            band = (ob["lower_hz"], ob["upper_hz"])
    lo, hi = sorted(float(v) for v in band)
    sel = (f >= lo) & (f <= hi)
    if not sel.any():
        sel[np.argmin(np.abs(f - 0.5 * (lo + hi)))] = True
    bins = int(sel.sum())
    mean_in = float(np.mean(p[sel]))
    excess = mean_in / max(floor, 1e-300) - 1.0
    # standard errors: the band's own average, and the floor's ripple when it
    # was estimated from these bins (relative, 1-sigma)
    se_band = 1.0 / math.sqrt(max(1, bins * sp["frames"]))
    se_floor = (10 ** (spread / 10.0) - 1.0) / math.sqrt(max(1, bins)) \
        if floor_per_hz is None else 0.0
    se = math.sqrt(se_band ** 2 + se_floor ** 2) * (1.0 + max(excess, 0.0))
    measurable = excess > 3.0 * se
    snr_db = 10 * math.log10(excess) if excess > 0 else None
    se_db = (10 * math.log10(1 + se / excess) if excess > 0 else None)
    return {"snr_db": snr_db, "se_db": se_db, "measurable": bool(measurable),
            "signal_power": max(excess, 0.0) * floor * (hi - lo),
            "noise_power": floor * (hi - lo), "band_hz": [lo, hi],
            "floor_per_hz": floor, "tier": TIER,
            "method": ("mean PSD in the band over the floor per Hz, minus 1 "
                       f"({sp['method']}); floor "
                       + ("given (measured on the source span)"
                          if floor_per_hz else "the median PSD bin")),
            "words": (f"{snr_db:+.1f} dB above the floor in "
                      f"{hi - lo:,.0f} Hz" if (measurable and snr_db is not None)
                      else "below what energy can measure here: the excess is "
                           "within three standard errors of the floor's own "
                           "uncertainty")}


# ---------------------------------------------------------------------------
# Cyclic measurements (Bill's lag-domain profile, derived thresholds)
# ---------------------------------------------------------------------------
def cyclic_peaks(x, fs: float, conj: bool = False, pfa: float = 1e-6,
                 count: int = 6, alpha_min_hz: float | None = None,
                 alpha_max_hz: float | None = None) -> dict:
    """Peaks of the lag-domain cyclic profile that clear the threshold
    derived from `pfa` (scf.detection_threshold): {alphas_hz, ratio,
    threshold, peaks: [{alpha_hz, statistic, words}], resolution_hz}."""
    from atk_diffusion.cyclo import scf as _scf
    x = _x1(x)
    fs = float(fs)
    alphas, stat = _scf.lag_profile(x, fs, conj=conj)
    if alphas.size == 0:
        return {"alphas_hz": [], "ratio": [], "threshold": None, "peaks": [],
                "resolution_hz": None, "tier": TIER,
                "method": "record too short for a cyclic profile"}
    ratio = _scf.local_ratio(stat)
    n = min(x.size, _scf.LAG_PROFILE_MAX)
    res = fs / n
    lo = float(alpha_min_hz) if alpha_min_hz is not None else (
        0.0 if conj else max(8.0 * res, fs / 5000.0))
    hi = float(alpha_max_hz) if alpha_max_hz else 0.5 * fs
    band = (np.abs(alphas) >= lo) & (np.abs(alphas) <= hi)
    if not conj:
        band &= alphas > 0                 # symmetric: one side holds it all
    cells = min(int(band.sum()), max(8, int(n * band.sum() / alphas.size)))
    thr = _scf.detection_threshold(pfa, cells)
    work = np.where(band, ratio, 0.0)
    guard = max(3, int(round(4 * res / max(alphas[1] - alphas[0], 1e-12))))
    peaks = []
    for _ in range(int(count)):
        k = int(np.argmax(work))
        if work[k] <= thr:
            break
        a = float(alphas[k])
        peaks.append({"alpha_hz": a, "statistic": float(work[k]),
                      "words": (f"twice a carrier at {a / 2:+,.1f} Hz "
                                "(BPSK/AM/MSK-class)" if conj
                                else "a symbol rate, chip rate or one of "
                                     "their harmonics")})
        work[max(0, k - guard):k + guard + 1] = 0.0
        if not conj:
            m = int(np.argmin(np.abs(alphas + a)))
            work[max(0, m - guard):m + guard + 1] = 0.0
    return {"alphas_hz": alphas, "ratio": ratio, "threshold": thr,
            "peaks": peaks, "resolution_hz": res, "cells": cells, "pfa": pfa,
            "conj": bool(conj), "tier": TIER,
            "method": ("lag-domain cyclic profile (Bill's bench): N·|R(α,τ)|²"
                       "/R(0,0)² maximised over 13 lags, over its local median;"
                       f" threshold for Pfa {pfa:g} over {cells:,} cells")}


def _fit_rate(x: np.ndarray, fs: float, floor_hz: float = 0.0,
              target_ratio: float = 6.0) -> tuple:
    """Decimate a heavily oversampled cut before profiling (the bench's
    `_fit_rate_to_bandwidth`): NOT an optimisation — the profile's own
    self-noise grows with samples per symbol, and on the bench a 600 Bd
    QPSK at 80 samples a symbol reported 170 Bd until this was added."""
    try:
        ob = occupied_bandwidth(x, fs)
        occ = float(ob["value_hz"] or 0.0)
    except Exception:                                      # noqa: BLE001
        occ = 0.0
    from_bw = occ * target_ratio if 0 < occ < fs / max(2.0, target_ratio) \
        else 0.0
    target = max(from_bw, float(floor_hz))
    if target <= 0 or target >= 0.5 * fs:
        return x, fs, 1
    q = int(fs // target)
    if q < 2:
        return x, fs, 1
    from scipy.signal import resample_poly
    y = resample_poly(x, 1, q)
    if y.size < 2048:
        return x, fs, 1
    return y, fs / q, q


def _comb_fundamental(peaks: list, tol: float = 0.04) -> dict | None:
    """The base of the strongest harmonic comb among detected peaks (a
    4-FSK's tallest α sat at its tone spacing; a 600 Bd QPSK's at 146 Hz —
    the fundamental of the comb carrying the most statistic is the rate).
    Ties go to the LOWEST base. The base must itself be a detected peak."""
    vals = [(abs(p["alpha_hz"]), p["statistic"]) for p in peaks]
    best = None
    for base, _v in vals:
        if base <= 0:
            continue
        comb = [(a, v) for a, v in vals
                if 1 <= round(a / base) <= 6 and abs(a - round(a / base) * base)
                <= tol * a]
        total = sum(v for _a, v in comb)
        if best is None or total > best[0] + 1e-9 or (
                abs(total - best[0]) <= 1e-9 and base < best[1]):
            best = (total, base, comb)
    if best is None:
        return None
    return {"alpha_hz": best[1], "comb": [(round(a, 3), round(v, 2))
                                          for a, v in best[2]],
            "comb_total": round(best[0], 2)}


def _line_power(y: np.ndarray, fs: float, alpha: float) -> complex:
    n = np.arange(y.size, dtype=np.float64)
    return complex(np.sum(y * np.exp(-2j * np.pi * np.mod(alpha / fs * n, 1.0))))


def refine_rate(x, fs: float, alpha: float, lags=(0,), span_bins: float = 2.0
                ) -> dict:
    """Refine a cycle frequency to a fraction of a bin: maximise the line
    power of the best lag product over ±span_bins/T, then fit a parabola.
    Returns {alpha_hz, lag, line (complex coefficient), step_hz}."""
    x = _x1(x)
    fs = float(fs)
    best = None
    for d in lags:
        d = int(d)
        y = (np.abs(x) ** 2 if d == 0 else x[d:] * np.conj(x[:-d]))
        y = y - np.mean(y)
        mag = abs(_line_power(y, fs, alpha))
        if best is None or mag > best[0]:
            best = (mag, d, y)
    _m, d, y = best
    T = y.size / fs
    grid = alpha + np.linspace(-span_bins, span_bins, 41) / T
    from atk_diffusion.cyclo.probes import zoom_dft
    v = np.abs(zoom_dft(y, fs, alpha, grid - alpha)) ** 2
    k = int(np.argmax(v))
    step = grid[1] - grid[0]
    a_hat = float(grid[k])
    if 0 < k < v.size - 1:
        den = v[k - 1] - 2 * v[k] + v[k + 1]
        if abs(den) > 0:
            a_hat += float(0.5 * (v[k - 1] - v[k + 1]) / den) * step
    return {"alpha_hz": a_hat, "lag": d, "line": _line_power(y, fs, a_hat),
            "step_hz": float(step), "T": T}


def symbol_rate(x, fs: float, hint_hz: float | None = None,
                pfa: float = 1e-6) -> dict:
    """Blind symbol rate from the non-conjugate cyclic profile.

    The bench's estimator, ported: decimate a heavily oversampled cut,
    take the lag-domain profile, keep peaks above the derived threshold,
    report the FUNDAMENTAL of the strongest harmonic comb (not the tallest
    α), then refine it to a fraction of a bin. `hint_hz` narrows the
    search to 0.6–1.6× the hint (fewer cells, a lower threshold — worth a
    few dB), which is recorded."""
    x = _x1(x)
    fs = float(fs)
    work, fs_w, q = _fit_rate(x, fs, floor_hz=(6.0 * float(hint_hz)
                                               if hint_hz else 0.0))
    if hint_hz:
        lo, hi = 0.6 * float(hint_hz), 1.6 * float(hint_hz)
    else:
        lo, hi = None, 0.5 * fs_w
    cp = cyclic_peaks(work, fs_w, conj=False, pfa=pfa, count=8,
                      alpha_min_hz=lo, alpha_max_hz=hi)
    out = {"tier": TIER, "decimated_by": q, "profiled_at_hz": fs_w,
           "resolution_hz": cp["resolution_hz"], "threshold": cp["threshold"],
           "hint_hz": hint_hz, "caveats": [],
           "method": ("fundamental of the strongest harmonic comb in the "
                      "non-conjugate cyclic profile, refined by a fine "
                      "single-bin DFT search")}
    if not cp["peaks"]:
        out.update({"value_hz": None, "known": False, "confidence": 0.0,
                    "statistic": None})
        out["caveats"].append(
            "no cyclic feature clears the derived threshold — no symbol "
            "clock, or the record is too short (cyclic resolution is set by "
            "record LENGTH, not by SNR)")
        return out
    fund = (_comb_fundamental(cp["peaks"]) if not hint_hz else
            {"alpha_hz": abs(cp["peaks"][0]["alpha_hz"]), "comb": None,
             "comb_total": None})
    a0 = fund["alpha_hz"]
    stat = max(p["statistic"] for p in cp["peaks"]
               if abs(abs(p["alpha_hz"]) - a0) <= 0.05 * a0) \
        if any(abs(abs(p["alpha_hz"]) - a0) <= 0.05 * a0 for p in cp["peaks"]) \
        else cp["peaks"][0]["statistic"]
    sps = fs / a0
    lags = sorted({0, max(1, int(round(0.5 * sps))), max(1, int(round(sps))),
                   max(1, int(round(1.25 * sps)))})
    lags = [d for d in lags if d < x.size // 4]
    ref = refine_rate(x, fs, a0, lags=lags)
    margin = stat - cp["threshold"]
    out.update({"value_hz": ref["alpha_hz"], "known": True,
                "coarse_hz": a0, "statistic": float(stat),
                "comb": fund.get("comb"), "line_lag": ref["lag"],
                "confidence": float(min(0.95, 0.45 + 0.5 * min(
                    1.0, margin / 20.0)))})
    if q > 1:
        out["caveats"].append(f"profiled after decimating {q}× — the cut was "
                              "oversampled for its own bandwidth (see the "
                              "bench note: this changes the answer)")
    if ref["alpha_hz"] < 4 * cp["resolution_hz"]:
        out["caveats"].append("the rate is within four α bins of zero — the "
                              "record holds too few symbols to trust it")
    return out


def carrier_offset(x, fs: float, pfa: float = 1e-6,
                   band_hz: tuple | None = None) -> dict:
    """Carrier offset from the cut's centre: the conjugate x² line ÷ 2 when
    there is one (BPSK/AM/ASK/MSK-class); else the x⁴ line ÷ 4 (QPSK-class,
    ambiguous within ±fs/8); else the centre of the occupied band. The
    method used is in `method`; `conjugate_feature` records the x² result,
    whose absence is itself a finding about the family."""
    from atk_diffusion.cyclo import probes as _probes
    x = _x1(x)
    fs = float(fs)
    c2 = _probes.carrier_conj(x, fs, pfa=pfa, power=2, band_hz=band_hz)
    out = {"tier": TIER, "conjugate_feature": ("present" if c2["detected"]
                                               else "absent"),
           "x2": {k: c2.get(k) for k in ("statistic", "threshold", "detected",
                                         "carrier_offset_hz")}}
    if c2["detected"]:
        out.update({"value_hz": c2["carrier_offset_hz"],
                    "confidence": c2["confidence"],
                    "method": "conjugate (x²) cyclic line ÷ 2"})
        return out
    c4 = _probes.carrier_conj(x, fs, pfa=pfa, power=4, band_hz=band_hz)
    out["x4"] = {k: c4.get(k) for k in ("statistic", "threshold", "detected",
                                        "carrier_offset_hz",
                                        "second_statistic")}
    # a QPSK carrier makes ONE x⁴ line; an FSK whose 4× modulation index is
    # an integer (C4FM: tones 1200 Hz apart at 4800 Bd) makes one per tone —
    # measured: the outer tone came out as "the carrier". Several comparable
    # lines mean tones, not a carrier.
    single = c4.get("second_statistic", 0.0) < 0.25 * c4.get("statistic", 0.0)
    if c4["detected"] and not single:
        out["x4_lines"] = ("several x⁴ lines of comparable strength — FSK tones,"
                           " not a QPSK carrier")
    if c4["detected"] and single:
        out.update({"value_hz": c4["carrier_offset_hz"],
                    "confidence": c4["confidence"] * 0.9,
                    "method": "fourth-power (x⁴) line ÷ 4 — ambiguous within "
                              f"±{fs / 8:,.0f} Hz"})
        return out
    ob = occupied_bandwidth(x, fs)
    out.update({"value_hz": ob["centre_hz"], "confidence": 0.3 if ob[
        "centre_hz"] is not None else 0.0,
        "method": "centre of the 99 % occupied band (no x² or x⁴ line: "
                  "FSK, OFDM, 8PSK/QAM or analog — the centroid is only as "
                  "good as the band edges)"})
    return out


# ---------------------------------------------------------------------------
# Bursts, duty, PRI
# ---------------------------------------------------------------------------
def envelope_bursts(x, fs: float, threshold_db: float = 6.0,
                    smooth_s: float | None = None,
                    min_len_s: float | None = None,
                    max_bursts: int = 20000) -> dict:
    """Bursts from the smoothed envelope: runs above the median envelope
    (the floor — the mean would be pulled up by the bursts being measured)
    by `threshold_db`, gaps shorter than the smoothing merged, runs shorter
    than `min_len_s` dropped. Returns {bursts: [{t0_s, t1_s, length_s,
    peak_db}], floor, threshold_db, method}; refuses (in words) when the
    threshold is in the noise."""
    x = _x1(x)
    fs = float(fs)
    w = max(1, int(round((smooth_s or max(4.0 / fs, 1e-3)) * fs)))
    w = min(w, max(1, x.size // 8))
    p = np.abs(x) ** 2
    env = np.convolve(p, np.ones(w) / w, mode="same")
    floor = float(np.median(env))
    out = {"tier": TIER, "threshold_db": float(threshold_db),
           "smooth_s": w / fs, "floor": floor, "bursts": [],
           "method": (f"envelope |x|² smoothed over {w} samples; runs "
                      f"{threshold_db:g} dB above its median")}
    if floor <= 0:
        return out
    above = env > floor * 10 ** (threshold_db / 10.0)
    if not above.any() or above.all():
        if above.all():
            out["words"] = ("the envelope is above the threshold everywhere — "
                            "a continuous signal, not bursts")
        return out
    d = np.diff(above.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1) + 1)
    if above[0]:
        starts.insert(0, 0)
    if above[-1]:
        ends.append(x.size)
    s = np.asarray(starts)
    e = np.asarray(ends)
    if s.size > 1:                         # merge gaps shorter than w
        gap = s[1:] - e[:-1]
        keep = np.concatenate([[True], gap >= w])
        s = s[keep]
        e = np.concatenate([e[:-1][keep[1:]], [e[-1]]])
    min_len = int(round((min_len_s or 2 * w / fs) * fs))
    ok = (e - s) >= max(1, min_len)
    s, e = s[ok], e[ok]
    if s.size > max_bursts:
        out["words"] = (f"{s.size:,} candidate bursts at +{threshold_db:g} dB "
                        "— the threshold is in the noise; raise it")
        return out
    out["bursts"] = [{"t0_s": float(a / fs), "t1_s": float(b / fs),
                      "length_s": float((b - a) / fs),
                      "peak_db": float(10 * np.log10(env[a:b].max() / floor))}
                     for a, b in zip(s.tolist(), e.tolist())]
    return out


def burst_stats(bursts, total_s: float | None = None,
                jitter_tol: float = 0.05) -> dict:
    """Burst length, duty cycle and PRI from burst records or start times.

    PRI kind, after the bench (`atk/core/pulse.classify_pri`): stagger is
    tested FIRST (with a two-level stagger the median lands on one level
    and the spread collapses); then constant (MAD within `jitter_tol` of
    the median — dropped pulses are ignored, not called jitter), jittered,
    or irregular."""
    if bursts and isinstance(bursts[0], dict):
        t0 = np.array([b["t0_s"] for b in bursts], dtype=float)
        lens = np.array([b["length_s"] for b in bursts], dtype=float)
    else:
        t0 = np.asarray(bursts or [], dtype=float)
        lens = np.zeros(t0.size)
    out = {"tier": TIER, "count": int(t0.size),
           "method": "burst start times: median interval, MAD spread, "
                     "stagger clusters"}
    if lens.any():
        out["length_s"] = float(np.median(lens))
        if total_s:
            out["duty"] = float(np.sum(lens) / float(total_s))
    if t0.size < 3:
        out.update({"pri_s": None, "pri_kind": "too few bursts"})
        return out
    d = np.diff(np.sort(t0))
    d = d[d > 0]
    if d.size < 2:
        out.update({"pri_s": None, "pri_kind": "too few bursts"})
        return out
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med))) * 1.4826
    spread = mad / med if med > 0 else 1.0
    levels = _stagger_levels(d, jitter_tol)
    if 1 < len(levels) <= 4:
        out.update({"pri_s": float(np.mean(d)), "pri_kind":
                    f"staggered ({len(levels)}-level)", "levels_s": levels,
                    "jitter_pct": spread * 100})
    elif spread <= jitter_tol:
        gaps = int(np.sum(np.abs(d - med) > 0.5 * med))
        out.update({"pri_s": med, "pri_kind": "constant",
                    "jitter_pct": spread * 100, "gaps_ignored": gaps})
    elif spread < 0.5:
        out.update({"pri_s": float(np.mean(d)), "pri_kind": "jittered",
                    "jitter_pct": spread * 100})
    else:
        out.update({"pri_s": float(np.mean(d)),
                    "pri_kind": "irregular / several emitters",
                    "jitter_pct": spread * 100})
    out["prf_hz"] = 1.0 / out["pri_s"] if out.get("pri_s") else None
    return out


def _stagger_levels(d: np.ndarray, tol: float, min_support: float = 0.15
                    ) -> list:
    """Distinct interval levels, each holding ≥ min_support of the
    intervals (relative tolerance tol). Multiples of a level (dropped
    pulses) do not count as levels."""
    vals = np.sort(d)
    levels: list = []
    used = np.zeros(vals.size, bool)
    for i, v in enumerate(vals):
        if used[i]:
            continue
        grp = np.abs(vals - v) <= tol * v
        used |= grp
        if grp.mean() >= min_support:
            levels.append(float(np.median(vals[grp])))
    base = min(levels) if levels else 0
    return [lv for lv in levels
            if lv == base or abs(lv / base - round(lv / base)) > 2 * tol]


# ---------------------------------------------------------------------------
# Everything at once (the cut's Analyze step)
# ---------------------------------------------------------------------------
def measure_all(x, fs: float, floor_per_hz: float | None = None,
                pfa: float = 1e-6) -> dict:
    """The classical measurements of a cut, each with tier and method."""
    x = _x1(x)
    fs = float(fs)
    T = x.size / fs
    ob = occupied_bandwidth(x, fs, floor_per_hz=floor_per_hz)
    band = ((ob["lower_hz"], ob["upper_hz"]) if ob["value_hz"] else None)
    snr = snr_above_floor(x, fs, band=band, floor_per_hz=floor_per_hz)
    rate = symbol_rate(x, fs, pfa=pfa)
    carrier = carrier_offset(x, fs, pfa=pfa)
    eb = envelope_bursts(x, fs)
    bs = burst_stats(eb["bursts"], total_s=T)
    return {"duration_s": T, "sample_rate_hz": fs,
            "occupied_bandwidth": ob, "snr": snr, "symbol_rate": rate,
            "carrier_offset": carrier, "bursts": {**bs, "list": eb["bursts"][
                :200], "detector": eb["method"]}}
