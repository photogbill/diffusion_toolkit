# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Filtering with what the cyclic detector found (DETECTION_DESIGN §4.3,
D11–D12; ARCHITECTURE §4.2 `cyclo.filters`).

Bill, 2026-10-08: *"if a different detector is running, and the SNR is below
a certain threshold, it uses a buffer to run cyclostationary detection. BUT,
can it use that information to improve SNR by filtering out noise as
well?"* Yes — three classical mechanisms, in the order of what they give,
each reporting the MEASURED before/after SNR and an honest size:

1. `matched_parameters` + `matched_filter` — the cyclic profile gives the
   symbol rate, the carrier offset and the symbol timing precisely: the
   settings a matched filter and a timing loop need. The matched filter is
   the maximum-SNR linear receiver per symbol, so this is the unglamorous
   answer that delivers most of the gain for most signals: cyclostationary
   → parameters → the right demodulator settings. The parameters are a
   measurement (tier MEASURED) handed to the demodulator; `matched_filter`
   also applies them — carrier removed, root-raised-cosine at the measured
   rate — and writes that output (tier CLEANED) with the SNR at the symbol
   instants. A constant-envelope FSK has no RRC matched filter (its
   receiver is a discriminator): then only the parameters are produced, and
   the reason is said.

2. `fresh_clean` / `fresh_separate` — FRESH filtering, the cyclic Wiener
   filter (Gardner). A cyclostationary signal's spectrum at f moves in
   lock-step with its spectrum at f ± α (its spectral REDUNDANCY); noise in
   different bands is independent. So the signal at f can be estimated
   from BOTH its own band and the frequency-shifted copies x·e^{∓j2παt}
   (and, for BPSK/AM-class signals, the conjugate copy x*·e^{j2πβt}), and
   the noise in the copies is fresh noise that averages down. Here it is
   built per frequency bin of a short-time Fourier transform: for each bin
   the minimum-mean-square-error combination of the bin and its shifted
   copies, whose statistics are all MEASURED from the cut — the
   cross-spectra between the bin and its copies can only come from the
   signal, and the signal's own power is the bin's power minus the noise
   floor. Blind: it needs only α (which the detector just produced) and the
   floor (which the cut recorded). The weights are fixed over time, so the
   result is a linear periodically time-varying filter — a FRESH filter.

   `fresh_separate` does the same for two (or more) signals on top of each
   other, each with its own cycle frequencies. The one number the bin
   cannot give is each signal's own power there, because the bin also holds
   the others. It is IDENTIFIED, not assumed: from pairs of a signal's
   copies that are both coherent with the bin (a one-factor model,
   P_k = R_0i·R_j0/R_ji), using only pairs whose correlations clear a
   derived significance test (a pair of copies that is noise in that bin
   gives a ratio of two noises, which is what the first version of this
   function averaged — and reported +16 dB for a signal it had made 7 dB
   worse); gaps inside a signal's identified band are interpolated, nothing
   is extrapolated beyond it; a signal with no identifiable bins at all
   (a proper QPSK with a small roll-off has no two coherent copies anywhere)
   gets the REMAINDER of the bin's power once the identified signals are
   taken out. Nothing time-invariant can separate two signals that overlap
   in frequency; this can, to the extent that each has spectral redundancy
   the other lacks.

3. `score` — SCORE beamforming (Agee, Schell & Gardner, cross-SCORE) on the
   Kraken's five coherent channels: the array weights that make the output
   most self-coherent at α — no direction of arrival, no calibration, no
   training sequence, only the cycle frequency. The array steers itself
   onto the signal with that α and nulls what lacks it.

Plus the two non-cyclic cleans the cut also offers: `wiener_clean` (the
time-invariant Wiener filter — FRESH's baseline, beside it as plan §7
requires) and `rfi_mask_interp` (mask strong narrowband / impulsive
interference in the time-frequency plane and interpolate across it — tier
INFERRED, because the filled cells are a guess).

HONEST SIZING, measured on synthetic cuts with known truth (the tests hold
these, against ground truth, not against the reports):
  * Wiener: removes the noise outside the signal's band — the band ratio:
    +9.7 dB for a 4.8 kBd QPSK (β 0.1) alone in a 48 kHz cut; INSIDE the
    band it gains nothing, by definition.
  * FRESH over the best time-invariant filter, white noise: +3.1 dB for a
    BPSK (its conjugate copy is a complete second look), +0.4 dB for a
    QPSK with a 0.35 roll-off (redundant only in the roll-off), +4 dB for
    rectangular-pulse BPSK.
  * FRESH separate, a BPSK 4800 Bd and a QPSK 6000 Bd of equal power fully
    on top of each other: each gains ~4.8 dB of SIR (−0.7 → +4.1 dB; the
    BPSK's conjugate copy is a second look at it, and lets the QPSK's
    estimate cancel much of the BPSK); two QPSK (4800 / 6000 Bd) only
    ~2 dB — the roll-off copies are all a proper signal offers. Real,
    measured, and nothing like "clean separation".
  * SCORE, five channels: up to 10·log10(5) ≈ 7 dB of array gain on white
    noise, plus nulling of an interferer that does not share α: one 10 dB
    above the signal is put 45–51 dB down, and the SINR goes from −10.4 dB
    in one element to +6.7 dB out of the beamformer.
  * Matched filter: about 10·log10(1+β) dB at the symbol instants over the
    in-band SNR (+1.3 dB measured for β = 0.35). Its real value is the
    demodulator working at the right rate, carrier and timing.

THE BLIND SNR METHODS, written into every report (on real cuts these are
estimates; in the tests they are checked against ground truth):
  * Wiener and FRESH — from the measured spectra: per bin the output's
    signal component |wᴴp|²/P_s over the rest of its power wᴴRw, summed
    (for a time-invariant gain g that is Σg²P_s/Σg²N₀, the output's signal
    over its noise); right when the floor is right and the noise is white
    within the cut. Wiener's "before" is the whole cut's SNR (signal over
    ALL the noise in its passband; a low-passed cut's stopband counts as
    the little it holds, not as a full floor; a bin's signal counts only
    when the bin is significantly above the floor).
  * FRESH separate — each signal's identified power per bin; SINR before =
    that power over everything else in the bin, after = the same output
    split as above, with the other signals counted as interference.
  * SCORE — the signal's array response is found blind from the SCORE
    weights themselves (SCORE converges on the maximum-SINR beamformer
    w ∝ R⁻¹a, so a ∝ R·w), its power by Capon's estimator (which nulls the
    interferers) less the white-noise leak; SINR before = that power over
    everything else in the best single channel, after = the same for the
    beamformer output. Assumes the channels have equal gain (the Kraken's
    are calibrated together). The eigen-floor numbers, which count an
    interferer as signal, are kept beside them for comparison (the first
    version reported only those: a 17 dB SINR improvement read as −3.7 dB).
    Measured: within ~0.5 dB at 0 dB SNR a channel; a few dB optimistic at
    −10 dB a channel, where the signal is a small part of every channel.
  * Matched filter — the floor through the filter's known noise gain: SNR
    at the symbol instants = (mean |y(t_k)|² − N₀·fs·Σh²)/(N₀·fs·Σh²).
"""

from __future__ import annotations

import math

import numpy as np

from atk_diffusion import provenance as _prov

#: Matched-filter parameters are a measurement handed to the demodulator,
#: not a filtered signal. (The filtered output is "matched_filter", which
#: provenance already lists as CLEANED.)
_prov.METHOD_TIERS.setdefault("matched_parameters", "measured")

#: The fewest samples the STFT filters work on: their statistics are
#: estimated per bin over time, which needs a few hundred frames.
MIN_SAMPLES = 1024

#: False-alarm probability of the significance gate that decides whether a
#: measured cross-spectrum is a real coherence or noise — for the WHOLE
#: spectrum of one pair of copies (Bonferroni over the bins).
GATE_P = 1e-3


# ---------------------------------------------------------------------------
# Inputs, refused in words
# ---------------------------------------------------------------------------
def _rate(fs) -> float:
    try:
        f = float(fs)
    except (TypeError, ValueError):
        raise ValueError(f"the sample rate {fs!r} is not a number") from None
    if not (f > 0 and math.isfinite(f)):
        raise ValueError(f"the sample rate must be a positive number of Hz, "
                         f"not {fs!r}")
    return f


def _iq_1d(x, what: str, min_samples: int = MIN_SAMPLES) -> tuple:
    """(x as 1-D complex128, note). A Kraken cut gives channel 0, and the
    note says so; empty, too short or damaged input is refused in words."""
    a = np.asarray(x)
    note = ""
    if a.ndim == 2:
        if a.shape[0] > 1:
            note = (f"channel 0 of {a.shape[0]} — only SCORE uses the array; "
                    "this filter works on one channel")
        a = a[0]
    elif a.ndim != 1:
        raise ValueError(f"{what} takes IQ as (n,) or (channels, n); this has "
                         f"shape {a.shape}")
    if a.size == 0:
        raise ValueError(f"{what}: there are no samples to filter")
    try:
        a = a.astype(np.complex128, copy=False)
    except (TypeError, ValueError):
        raise ValueError(f"{what}: the samples are not numbers") from None
    if a.size < int(min_samples):
        raise ValueError(f"{what} needs at least {int(min_samples):,} samples "
                         f"to estimate its statistics; this has {a.size:,}. "
                         "Cut a longer stretch.")
    bad = int(np.count_nonzero(~np.isfinite(a)))
    if bad:
        raise ValueError(f"{what}: {bad:,} of the samples are not finite (NaN "
                         "or infinity) — a damaged file or a failed read. "
                         "Nothing was filtered.")
    return a, note


def _alphas(vals, fs: float, what: str, drop_zero: bool) -> list:
    out = []
    for v in (vals or []):
        try:
            a = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{what}: {v!r} is not a cycle frequency in "
                             "Hz") from None
        if not math.isfinite(a):
            raise ValueError(f"{what}: a cycle frequency must be finite, "
                             f"not {v!r}")
        if abs(a) >= fs:
            raise ValueError(f"{what}: α = {a:,.1f} Hz is beyond what a cut at "
                             f"{fs:,.0f} S/s can show (|α| < {fs:,.0f} Hz)")
        if drop_zero and a == 0.0:
            continue
        out.append(a)
    return out


# ---------------------------------------------------------------------------
# Matched-filter parameters (§4.3, 1)
# ---------------------------------------------------------------------------
def _prefilter(x: np.ndarray, fs: float) -> tuple:
    """x band-limited to its measured occupied band (25 % wider), when that
    band is measurable and clearly narrower than the cut — else x as it is.

    WHY: the rate line lives in lag products (quadratic in the noise) and
    the carrier in x² / x⁴ (quadratic / quartic), so noise outside the
    signal's band costs dB for nothing: measured, a QPSK at +5 dB in-band
    SNR alone in a 48 kHz cut had its x⁴ carrier line buried until the cut
    was band-limited to the signal. A cut is already low-passed to its box;
    this is for boxes drawn generously."""
    from atk_diffusion.cyclo.probes import bandlimit
    from atk_diffusion.dsp import measure as _m
    try:
        ob = _m.occupied_bandwidth(x, fs)
    except Exception:                                      # noqa: BLE001
        return x, None
    w = ob.get("value_hz")
    caveats = " ".join(ob.get("caveats") or [])
    if (not w or ob.get("peak_over_floor_db", 0.0) < 3.0
            or "edge" in caveats or 1.25 * float(w) >= 0.6 * fs):
        return x, None
    c = float(ob["centre_hz"])
    half = 0.5 * 1.25 * float(w)
    n = np.arange(x.size, dtype=np.float64)
    rot = np.exp(-2j * np.pi * np.mod(c / fs * n, 1.0))
    y = np.asarray(bandlimit((x * rot).astype(np.complex64), fs, half),
                   dtype=np.complex128) * np.conj(rot)
    return y, {"centre_hz": c, "half_width_hz": half,
               "occupied_bandwidth_hz": float(w)}


def matched_parameters(x, fs: float, rate_hint_hz: float | None = None,
                       pfa: float = 1e-6) -> dict:
    """{symbol_rate_hz, carrier_offset_hz, timing_offset_s, confidence,
    method, samples_per_symbol, tier, rate, carrier, timing, prefilter,
    words}.

    symbol rate   the fundamental of the strongest harmonic comb in the
                  non-conjugate cyclic profile, refined to a fraction of a
                  bin (dsp.measure.symbol_rate)
    carrier       the conjugate x² line ÷ 2 (BPSK/AM-class), else the x⁴
                  line ÷ 4 (QPSK-class), else the band centre
    timing        the phase of the symbol-rate line in the squared envelope:
                  the instant within each symbol where the envelope peaks —
                  the pulse centres, where a matched-filter output should be
                  sampled. Offered only when that line is significant; a
                  constant-envelope FSK has none, and its timing is left to
                  the demodulator's own discriminator (said, not guessed).

    The cut is first band-limited to its measured occupied band when the
    box was drawn wider than the signal (`prefilter` says so)."""
    from atk_diffusion.cyclo import probes as _probes
    from atk_diffusion.dsp import measure as _m
    fs = _rate(fs)
    x, note = _iq_1d(x, "matched parameters", min_samples=256)
    xf, pre = _prefilter(x, fs)
    sr = _m.symbol_rate(xf, fs, hint_hz=rate_hint_hz, pfa=pfa)
    co = _m.carrier_offset(xf, fs, pfa=pfa)
    out = {"tier": "measured", "rate": sr, "carrier": co,
           "symbol_rate_hz": sr.get("value_hz"),
           "carrier_offset_hz": co.get("value_hz"),
           "timing_offset_s": None, "samples_per_symbol": None,
           "confidence": float(sr.get("confidence") or 0.0),
           "prefilter": pre}
    if note:
        out["note"] = note
    timing = {"method": "phase of the symbol-rate line in |x|²"}
    if sr.get("known"):
        a = float(sr["value_hz"])
        out["samples_per_symbol"] = fs / a
        # the clock line in the envelope of the cut AS GIVEN: band-limiting
        # an FSK cuts its spectral tails, and the FM-to-AM conversion makes
        # an envelope clock line that is not there (measured: statistic 104
        # against 1.4 on the same 4FSK) — that line is what tells a linear
        # modulation, which has a matched RRC, from an FSK, which has not
        y = np.abs(x) ** 2
        y = y - y.mean()
        T = y.size / fs
        j = np.arange(1, 17)
        offs = np.concatenate([[0.0], -(2 + j), 2 + j]) / T
        v = _probes.zoom_dft(y, fs, a, offs)
        p = np.abs(v) ** 2
        st = float(p[0] / max(np.sort(p[1:])[15], 1e-300))
        thr = _probes.os_threshold(1e-3, 32, 16, 1)
        timing.update({"statistic": st, "threshold": thr})
        if st > thr:
            t0 = (-np.angle(v[0]) / (2 * np.pi * a)) % (1.0 / a)
            out["timing_offset_s"] = float(t0)
            timing["words"] = (f"symbol centres at {t0 * 1e6:,.1f} µs after the "
                               f"cut's first sample, every {1e6 / a:,.1f} µs")
        else:
            timing["words"] = ("the squared envelope carries no clear clock "
                               "line (constant-envelope FSK, or too weak): "
                               "timing is left to the demodulator")
    out["timing"] = timing
    bits = []
    if sr.get("known"):
        bits.append(f"symbol rate {sr['value_hz']:,.2f} Bd "
                    f"({out['samples_per_symbol']:.2f} samples a symbol)")
    else:
        bits.append("no symbol rate found")
    if co.get("value_hz") is not None:
        bits.append(f"carrier {co['value_hz']:+,.1f} Hz ({co['method']})")
    if out["timing_offset_s"] is not None:
        bits.append(timing["words"])
    out["method"] = ("cyclic: rate from the non-conjugate profile (comb "
                     "fundamental, refined); carrier from the conjugate / "
                     "fourth-power line or the band centre; timing from the "
                     "phase of the rate line in |x|²"
                     + ("; after band-limiting the cut to its occupied band"
                        if pre else ""))
    out["words"] = "; ".join(bits) + (". Hand these to the demodulator: "
                                      "matched filter at this rate, centred on "
                                      "this carrier, sampled at these instants.")
    return out


def rrc_taps(beta: float, sps: float, span: int = 8) -> np.ndarray:
    """Root-raised-cosine, unit energy (Σh² = 1), odd length, centred on its
    middle tap; `sps` (samples a symbol) need not be an integer."""
    b = float(min(max(beta, 0.0), 1.0))
    k = int(math.ceil(int(span) * float(sps)))
    t = np.arange(-k, k + 1, dtype=np.float64) / float(sps)
    h = np.empty(t.size)
    for i, tt in enumerate(t):
        if abs(tt) < 1e-12:
            h[i] = 1.0 - b + 4.0 * b / math.pi
        elif b > 0 and abs(abs(4.0 * b * tt) - 1.0) < 1e-9:
            h[i] = b / math.sqrt(2.0) * ((1 + 2 / math.pi) * math.sin(math.pi / (4 * b))
                                         + (1 - 2 / math.pi) * math.cos(math.pi / (4 * b)))
        else:
            h[i] = ((math.sin(math.pi * tt * (1 - b))
                     + 4 * b * tt * math.cos(math.pi * tt * (1 + b)))
                    / (math.pi * tt * (1 - (4 * b * tt) ** 2)))
    return h / math.sqrt(float(np.sum(h ** 2)))


def rolloff_from_obw(obw_hz: float, rate_hz: float,
                     fraction: float = 0.99) -> float:
    """The RRC roll-off β whose raised-cosine spectrum holds `fraction` of
    its power in `obw_hz` — NOT obw/rate − 1: the outer part of the cosine
    taper holds little power, so the 99 % band of β = 0.35 is only
    R·(1 + 0.48β) (measured first as β ≈ 0.16 by the naive formula).
    Solved exactly: the power beyond u₀ of the taper (u ∈ [0, 1] across it)
    on both sides is β·[(1 − u₀) − sin(πu₀)/π]·… = (1 − fraction), and the
    band is R·(1 − β + 2u₀β). Clipped to 0.05–1."""
    r = float(obw_hz) / float(rate_hz) - 1.0
    out = 1.0 - float(fraction)

    def v_of(beta: float) -> float:              # 1 − u₀ for this β
        target = 0.5 * out / beta
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            g = 0.5 * mid - math.sin(math.pi * mid) / (2 * math.pi)
            lo, hi = (mid, hi) if g < target else (lo, mid)
        return 0.5 * (lo + hi)

    def width(beta: float) -> float:             # band/R − 1
        return beta * (1.0 - 2.0 * v_of(beta))

    lo, hi = 0.05, 1.0
    if r <= width(lo):
        return lo
    if r >= width(hi):
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if width(mid) < r else (lo, mid)
    return 0.5 * (lo + hi)


def symbol_instants(n: int, fs: float, rate: float, timing_s: float,
                    guard_symbols: int = 8) -> np.ndarray:
    """Fractional sample positions of the symbol centres t_k = timing + k/R
    inside [guard, n − guard] (the filter's edges excluded)."""
    sps = fs / float(rate)
    first = float(timing_s) * fs
    g = guard_symbols * sps
    k0 = int(math.ceil((g - first) / sps))
    k1 = int(math.floor((n - 1 - g - first) / sps))
    if k1 < k0:
        return np.zeros(0)
    return first + sps * np.arange(k0, k1 + 1, dtype=np.float64)


def sample_at(y: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """y at fractional sample positions (linear interpolation; y is the
    smooth output of a filter much narrower than the sample rate)."""
    idx = np.arange(y.size, dtype=np.float64)
    return np.interp(pos, idx, y.real) + 1j * np.interp(pos, idx, y.imag)


def matched_filter(x, fs: float, params: dict | None = None,
                   floor_per_hz: float | None = None,
                   rolloff: float | None = None, span: int = 8) -> tuple:
    """The matched filter itself (§4.3, 1): the measured carrier removed, a
    root-raised-cosine at the measured symbol rate (roll-off from the
    measured occupied bandwidth unless given), at the cut's own rate.
    (y, report). The report carries the parameters, the roll-off and how it
    was obtained, the symbol instants (first, spacing, count) and the
    MEASURED SNR: before = in-band, after = at the symbol instants, both
    against the floor (`floor_per_hz`; measured from the cut when absent).

    Refused, in words (ValueError), when there is no symbol rate or no clock
    line in the envelope — a constant-envelope FSK's receiver is a
    discriminator, not an RRC; its parameters still go to the demodulator.
    `params`: a `matched_parameters` result (measured here when None)."""
    from atk_diffusion.dsp import measure as _m
    from scipy.signal import fftconvolve
    fs = _rate(fs)
    x, note = _iq_1d(x, "the matched filter", min_samples=256)
    mp = params if params is not None else matched_parameters(x, fs)
    R = mp.get("symbol_rate_hz")
    if not R:
        raise ValueError("the matched filter needs a symbol rate and the "
                         "cyclic analysis found none — run it on a longer or "
                         "stronger cut, or pass the rate")
    R = float(R)
    timing = mp.get("timing_offset_s")
    if timing is None:
        raise ValueError("no clock line in the squared envelope: this looks "
                         "like constant-envelope FSK (or is too weak), whose "
                         "receiver is a discriminator, not a root-raised-"
                         "cosine — the parameters go to the demodulator, no "
                         "filtered file is made")
    fc = float(mp.get("carrier_offset_hz") or 0.0)
    n = np.arange(x.size, dtype=np.float64)
    xd = x * np.exp(-2j * np.pi * np.mod(fc / fs * n, 1.0))
    n0 = float(floor_per_hz) if floor_per_hz else float(
        _m.noise_floor(x, fs)["floor_per_hz"])
    if rolloff is None:
        ob = _m.occupied_bandwidth(xd, fs, floor_per_hz=n0)
        w = ob.get("value_hz")
        beta = rolloff_from_obw(w, R) if w else 0.35
        how = (f"from the measured 99 % occupied bandwidth {w:,.0f} Hz (the "
               "raised-cosine β holding 99 % of its power there)" if w else
               "0.35 assumed — the occupied bandwidth was not measurable")
    else:
        beta = float(min(1.0, max(0.0, float(rolloff))))
        how = "given"
    sps = fs / R
    h = rrc_taps(beta, sps, span)
    y = fftconvolve(xd, h, mode="same")
    pos = symbol_instants(x.size, fs, R, float(timing), guard_symbols=span)
    noise_out = n0 * fs * float(np.sum(h ** 2))
    p_sym = float(np.mean(np.abs(sample_at(y, pos)) ** 2)) if pos.size else 0.0
    after = (10 * math.log10((p_sym - noise_out) / noise_out)
             if pos.size and p_sym > noise_out else None)
    B = (1.0 + beta) * R
    sb = _m.snr_above_floor(x, fs, band=(fc - 0.5 * B, fc + 0.5 * B),
                            floor_per_hz=n0)
    before = sb.get("snr_db")
    rep = {"method": "matched_filter", "tier": _prov.tier_for("matched_filter"),
           "symbol_rate_hz": R, "carrier_offset_hz": fc,
           "timing_offset_s": float(timing), "rolloff": beta,
           "rolloff_method": how, "samples_per_symbol": sps,
           "taps": int(h.size), "floor_per_hz": n0,
           "instants": {"first_s": float(pos[0] / fs) if pos.size else None,
                        "spacing_s": 1.0 / R, "count": int(pos.size)},
           "snr_before_db": before, "snr_after_db": after,
           "gain_db": (after - before if after is not None and before is not None
                       else None),
           "snr_method": ("blind, against the floor: before = the in-band SNR "
                          f"over (1+β)·R = {B:,.0f} Hz around the carrier; "
                          "after = the SNR at the symbol instants, "
                          "(mean |y(t_k)|² − N₀·fs·Σh²)/(N₀·fs·Σh²)"),
           "sizing": ("the matched filter is the maximum-SNR linear receiver "
                      "per symbol: about 10·log10(1+β) dB at the symbol "
                      "instants over the in-band SNR (+1.3 dB at β 0.35); its "
                      "real value is the demodulator working at the right "
                      "rate, carrier and timing"),
           "words": (f"matched filter: carrier {fc:+,.1f} Hz removed, "
                     f"root-raised-cosine (β {beta:.2f}) at {R:,.2f} Bd; "
                     + (f"SNR {before:+.1f} dB in-band → {after:+.1f} dB at "
                        "the symbol instants, measured against the floor."
                        if before is not None and after is not None else
                        "the SNR was not measurable against the floor."))}
    if note:
        rep["note"] = note
    return y.astype(np.complex64), rep


# ---------------------------------------------------------------------------
# The short-time Fourier transform the FRESH and Wiener filters work in
# ---------------------------------------------------------------------------
def _win(nper: int) -> np.ndarray:
    """sqrt of the periodic Hann: analysis × synthesis at 50 % overlap sums
    to exactly 1, so an unmodified STFT reconstructs the input exactly."""
    return np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nper) / nper))


def stft(x, nper: int) -> np.ndarray:
    """[F, T] complex: frames of `nper` at hop nper/2 (padded both ends)."""
    x = np.asarray(x, dtype=np.complex128)
    hop = nper // 2
    w = _win(nper)
    pad = np.concatenate([np.zeros(nper, complex), x,
                          np.zeros(nper + hop, complex)])
    nfr = (pad.size - nper) // hop + 1
    idx = np.arange(nper)[None, :] + hop * np.arange(nfr)[:, None]
    return np.fft.fft(pad[idx] * w[None, :], axis=1).T


def istft(Z: np.ndarray, nper: int, n: int) -> np.ndarray:
    hop = nper // 2
    w = _win(nper)
    fr = np.fft.ifft(Z.T, axis=1) * w[None, :]
    out = np.zeros((fr.shape[0] - 1) * hop + nper, dtype=np.complex128)
    for i in range(fr.shape[0]):
        out[i * hop:i * hop + nper] += fr[i]
    return out[nper:nper + n]


def _nperseg(fs: float, alphas, n: int) -> int:
    """Bins fine enough to follow the spectral correlation (≥ 16 per α), and
    frames enough (≥ 200) to estimate each bin's statistics."""
    a = [abs(float(v)) for v in alphas if abs(float(v)) > 0]
    want = 16.0 * fs / min(a) if a else 256.0
    nper = int(2 ** round(math.log2(min(max(want, 32.0), 4096.0))))
    while nper > 32 and 2 * n / nper < 200:
        nper //= 2
    return nper


def _check_nper(nperseg, n: int) -> int | None:
    if nperseg is None:
        return None
    try:
        v = int(nperseg)
    except (TypeError, ValueError):
        raise ValueError(f"nperseg must be a whole number of samples, not "
                         f"{nperseg!r}") from None
    if v < 8 or v % 2 or v > n:
        raise ValueError(f"nperseg {nperseg!r} must be even, at least 8 and no "
                         f"longer than the cut ({n:,} samples)")
    return v


def _shifted(x: np.ndarray, fs: float, s: float, conj: bool = False):
    n = np.arange(x.size, dtype=np.float64)
    if conj:
        return np.conj(x) * np.exp(2j * np.pi * np.mod(s / fs * n, 1.0))
    return x * np.exp(-2j * np.pi * np.mod(s / fs * n, 1.0))


def _branch_signals(x: np.ndarray, fs: float, alphas, conj_alphas,
                    both_signs: bool = True) -> tuple:
    """[x, x·e^{∓j2παt} …, x*·e^{j2πβt} …] and their labels. The copy
    x·e^{−j2παt} carries the spectrum of x at f + α into bin f."""
    sigs, labels = [x], ["x (as received)"]
    for a in alphas:
        a = float(a)
        for s in ((a, -a) if both_signs else (a,)):
            sigs.append(_shifted(x, fs, s))
            labels.append(f"x shifted by {s:+,.1f} Hz")
    for b in conj_alphas:
        b = float(b)
        sigs.append(_shifted(x, fs, b, conj=True))
        labels.append(f"conj(x) shifted by {b:+,.1f} Hz")
    return sigs, labels


def _frame_independence(nper: int) -> float:
    """1/(1 + 2ρ₁²): the fraction of the STFT's frames that are independent
    evidence, ρ₁ = Σw[n]w[n+hop]/Σw² the correlation of one bin in two
    half-overlapped frames (1/π for the √Hann window at half overlap, so 1/1.20)."""
    w = _win(nper)
    hop = nper // 2
    rho = float(np.sum(w[hop:] * w[:nper - hop]) / np.sum(w ** 2))
    return 1.0 / (1.0 + 2.0 * rho ** 2)


def _stack(sigs, nper: int) -> np.ndarray:
    return np.stack([stft(s, nper) for s in sigs], axis=1)     # [F, B, T]


def _floor_per_bin(P0: np.ndarray, floor_per_hz, fs: float, nper: int
                   ) -> tuple:
    """The noise power of one STFT bin, and how it was obtained."""
    if floor_per_hz:
        n0 = float(floor_per_hz) * fs * float(np.sum(_win(nper) ** 2))
        return n0, "the floor given (measured on the source span at cut time)"
    pb = P0 >= 1e-3 * float(P0.max())
    n0 = float(np.percentile(P0[pb], 25)) if pb.any() else float(np.median(P0))
    return n0, ("blind: the 25th percentile of the passband's bin powers — "
                "too high when the signal fills the passband, which makes "
                "the filter more cautious, not less")


def _mmse(V: np.ndarray, p: np.ndarray, Ps: np.ndarray) -> tuple:
    """Per-bin MMSE: w = R⁻¹p, out = wᴴv, mmse = Ps − Re(pᴴw); and the
    output's power per bin split into its signal component, |wᴴp|²/Ps (the
    part of the estimate correlated with the signal), and the whole,
    wᴴRw — what `_out_db` turns into an SNR."""
    F, B, T = V.shape
    R = np.einsum("fbt,fct->fbc", V, np.conj(V)) / T
    load = 1e-9 * np.real(np.trace(R, axis1=1, axis2=2)) / B + 1e-300
    R = R + load[:, None, None] * np.eye(B)[None, :, :]
    w = np.linalg.solve(R, p[:, :, None])[:, :, 0]
    out = np.einsum("fb,fbt->ft", np.conj(w), V)
    wp = np.einsum("fb,fb->f", np.conj(w), p)
    mmse = np.maximum(Ps - np.real(wp), 1e-300)
    sig = np.where(Ps > 0, np.abs(wp) ** 2 / np.maximum(Ps, 1e-300), 0.0)
    tot = np.real(np.einsum("fb,fbc,fc->f", np.conj(w), R, w))
    return out, mmse, w, sig, tot


def _out_db(sig: np.ndarray, tot: np.ndarray) -> float:
    """The output's SNR: its signal component over everything else in it,
    summed over the bins. For a time-invariant gain g this is
    Σg²P_s/Σg²N₀ — exactly the output's signal over its noise. (The first
    version used Σ(P_s − MMSE)/Σ MMSE, which charges noise-only bins their
    own fluctuation as 'error' and read a Wiener output 1 dB low.)"""
    num = float(np.sum(sig))
    den = float(np.sum(np.maximum(tot - sig, 0.0)))
    return 10 * math.log10(num / den) if num > 0 and den > 0 else -math.inf


def _lti_out(P0: np.ndarray, n0: float) -> tuple:
    """(gain, signal part, whole output) per bin of the time-invariant
    Wiener filter g = P_s/P0."""
    Ps = np.maximum(P0 - n0, 0.0)
    g = Ps / np.maximum(P0, 1e-300)
    return g, g ** 2 * Ps, g ** 2 * P0


def _full_band_db(P0: np.ndarray, n0: float, frames: int) -> float:
    """The whole cut's SNR: signal power over ALL the noise in it. A bin's
    noise is the floor, or what the bin holds when that is less (the
    stopband of a low-passed cut holds almost nothing — counting a full
    floor there understated a canonical cut's SNR by its stopband share).
    A bin's signal counts only when the bin is significantly above the
    floor (3 standard errors of its own average): the positive halves of a
    thousand noise bins' fluctuations are not signal."""
    gate = n0 * (1.0 + 3.0 / math.sqrt(max(1, int(frames))))
    Ps = np.where(P0 > gate, P0 - n0, 0.0)
    noise = float(np.sum(np.minimum(P0, n0)))
    return (10 * math.log10(max(float(Ps.sum()), 1e-300) / noise)
            if noise > 0 else math.inf)


# ---------------------------------------------------------------------------
# Wiener (time-invariant) — FRESH's baseline
# ---------------------------------------------------------------------------
def wiener_clean(x, fs: float, floor_per_hz: float | None = None,
                 nperseg: int | None = None) -> tuple:
    """The time-invariant Wiener filter: each STFT bin scaled by
    P_s/(P_s + N0). It removes the noise OUTSIDE the signal's band — the
    measured gain — and cannot raise the SNR inside it, which the report
    says (`in_band_gain_db` = 0). (y, report): snr_before_db = the whole
    cut's SNR, snr_after_db = the output's (the in-band SNR)."""
    fs = _rate(fs)
    x, note = _iq_1d(x, "the Wiener filter")
    nper = int(_check_nper(nperseg, x.size) or _nperseg(fs, [], x.size))
    Z = stft(x, nper)
    P0 = np.mean(np.abs(Z) ** 2, axis=1)
    n0, how = _floor_per_bin(P0, floor_per_hz, fs, nper)
    g, sig, tot = _lti_out(P0, n0)
    y = istft(Z * g[:, None], nper, x.size)
    inband = _out_db(sig, tot)
    full = _full_band_db(P0, n0, Z.shape[1])
    occ = float(np.mean(g > 0.5))
    rep = {"method": "wiener", "tier": _prov.tier_for("wiener"),
           "nperseg": nper, "floor_per_bin": n0, "floor_method": how,
           "snr_before_db": full, "snr_after_db": inband,
           "gain_db": inband - full, "in_band_gain_db": 0.0,
           "full_band_snr_before_db": full, "in_band_snr_db": inband,
           "bins_kept": occ, "snr_method": _SNR_METHOD,
           "sizing": ("a time-invariant filter removes the noise outside the "
                      "signal's band (the gain stated); inside the band the "
                      "SNR is unchanged — 0 dB, by definition"),
           "words": (f"Wiener filter: the cut's SNR {full:+.1f} dB → "
                     f"{inband:+.1f} dB, the noise outside the signal's band "
                     f"removed ({inband - full:+.1f} dB); inside the band the "
                     "SNR is unchanged.")}
    if note:
        rep["note"] = note
    return y.astype(np.complex64), rep


_SNR_METHOD = ("blind, from the measured spectra and spectral correlations: "
               "per STFT bin the output's signal component |wᴴp|²/P_s over "
               "the rest of its power wᴴRw, summed over the bins (for a "
               "time-invariant gain g, Σg²P_s/Σg²N₀ — the in-band SNR); right "
               "when the floor is right and the noise is white within the "
               "cut")


# ---------------------------------------------------------------------------
# FRESH clean (§4.3, 2)
# ---------------------------------------------------------------------------
def fresh_clean(x, fs: float, alphas, conj_alphas=(),
                floor_per_hz: float | None = None,
                nperseg: int | None = None, both_signs: bool = True) -> tuple:
    """Blind cyclic Wiener (FRESH) filter from the cycle frequencies the
    detector found. (y, report).

    alphas        non-conjugate cycle frequencies (symbol rate, chip rate…);
                  each gives copies shifted by +α and −α
    conj_alphas   conjugate cycle frequencies (2·carrier for BPSK/AM, 2·fc ±
                  the rate…); each gives a conjugated shifted copy
    floor_per_hz  the noise floor (power per Hz); the cut records the floor
                  measured on the source span — pass it. Without it the
                  floor is estimated blind (and said so).

    The report: snr_before_db (in-band, what the best time-invariant filter
    leaves), snr_after_db, gain_db (over that baseline — FRESH's own,
    honest number), full_band_snr_before_db (the cut as it was), the floor
    and how it was obtained, branches, the fraction of bins where the
    copies contributed, the blind SNR method, and the honest sizing."""
    fs = _rate(fs)
    x, note = _iq_1d(x, "FRESH")
    alphas = _alphas(alphas, fs, "FRESH", drop_zero=True)
    conj_alphas = _alphas(conj_alphas, fs, "FRESH", drop_zero=False)
    if not alphas and not conj_alphas:
        raise ValueError("FRESH needs at least one cycle frequency — run the "
                         "cyclic analysis first (its peaks are the α to use)")
    nper = int(_check_nper(nperseg, x.size)
               or _nperseg(fs, alphas + [b for b in conj_alphas if b], x.size))
    sigs, labels = _branch_signals(x, fs, alphas, conj_alphas, both_signs)
    V = _stack(sigs, nper)
    P0 = np.mean(np.abs(V[:, 0]) ** 2, axis=1)
    n0, how = _floor_per_bin(P0, floor_per_hz, fs, nper)
    Ps = np.maximum(P0 - n0, 0.0)
    T = V.shape[2]
    p = np.einsum("fbt,ft->fb", V, np.conj(V[:, 0])) / T     # E[v X0*]
    p[:, 0] = Ps
    out, _mmse_f, w, sig, tot = _mmse(V, p, Ps)
    y = istft(out, nper, x.size)
    _g, s_lti, t_lti = _lti_out(P0, n0)
    before = _out_db(s_lti, t_lti)
    after = _out_db(sig, tot)
    used = float(np.mean(np.sum(np.abs(w[:, 1:]) ** 2, axis=1)
                         > 0.05 * np.abs(w[:, 0]) ** 2))
    rep = {"method": "fresh", "tier": _prov.tier_for("fresh"),
           "alphas_hz": alphas, "conj_alphas_hz": conj_alphas,
           "branches": labels, "nperseg": nper, "floor_per_bin": n0,
           "floor_method": how, "snr_before_db": before,
           "snr_after_db": after, "gain_db": after - before,
           "full_band_snr_before_db": _full_band_db(P0, n0, T),
           "bins_using_copies": used, "snr_method": _SNR_METHOD,
           "sizing": _FRESH_SIZING,
           "words": (f"FRESH (cyclic Wiener) with {len(labels) - 1} shifted "
                     f"copies: in-band SNR {before:+.1f} → {after:+.1f} dB "
                     f"({after - before:+.1f} dB over the best time-invariant "
                     "filter), measured blind from this cut's spectra.")}
    if note:
        rep["note"] = note
    return y.astype(np.complex64), rep


_FRESH_SIZING = ("against white noise the gain over a time-invariant filter "
                 "is bounded by the signal's spectral redundancy: ~+3 dB "
                 "BPSK, ~+0.4 dB QPSK at 0.35 roll-off, ~+4 dB rectangular "
                 "BPSK; against co-channel interference ~+4.8 dB each for a "
                 "BPSK and a QPSK of equal power on top of each other, ~2 dB "
                 "for two QPSK, more with more excess bandwidth")


# ---------------------------------------------------------------------------
# FRESH separate (§4.3, 2 — co-channel)
# ---------------------------------------------------------------------------
def _normalise_set(s, fs: float) -> tuple:
    if isinstance(s, dict):
        return (_alphas(s.get("alphas", []), fs, "FRESH separate", True),
                _alphas(s.get("conj", s.get("conj_alphas", [])), fs,
                        "FRESH separate", False))
    if isinstance(s, (int, float)):
        s = [s]
    return _alphas(s, fs, "FRESH separate", True), []


def _fill_inside(est: np.ndarray) -> np.ndarray:
    """Per-bin estimates (NaN where none) in FFT order -> gaps INSIDE the
    identified band interpolated, zero outside it. Done in frequency order
    (fftshifted): in FFT order a baseband signal's band is split across the
    two ends of the array and its 'interior' would be the empty middle."""
    F = est.size
    s = np.fft.fftshift(est)
    ok = np.isfinite(s)
    out = np.zeros(F)
    if ok.any():
        idx = np.flatnonzero(ok)
        lo, hi = int(idx[0]), int(idx[-1])
        inside = np.arange(lo, hi + 1)
        out[lo:hi + 1] = np.interp(inside, idx, s[ok])
    return np.fft.ifftshift(out)


def _rc_psd(f: np.ndarray, rate: float, beta: float) -> np.ndarray:
    """The raised-cosine spectrum (|RRC|², peak 1) at offsets f from its
    centre, for symbol rate `rate` and roll-off `beta`."""
    a = np.abs(f)
    lo, hi = (1 - beta) * rate / 2, (1 + beta) * rate / 2
    return np.where(a <= lo, 1.0, np.where(
        a >= hi, 0.0, 0.5 * (1 + np.cos(np.pi / max(beta * rate, 1e-9)
                                         * (a - lo)))))


def _rc_from_copies(R, mine, labels, coherent, fr, fs, F, alphas):
    """(centre, rate, β) of a proper signal's raised-cosine spectrum, from
    where its ±α copies are coherent with the bin. The +α copy (it carries
    f + α) is coherent across the LOWER roll-off overlap [f_lo, f_lo + βR],
    with |R_0i|² a sin² bump; the −α copy across the upper one. The bumps'
    centroids sit at f_lo + βR/2 and f_hi − βR/2 — their mean is the centre
    — and a sin² bump over a width βR has an RMS width 0.181·βR. None when
    the copies do not show both bumps."""
    if not alphas:
        return None
    rate = float(alphas[0])
    df = fs / F
    cents, widths = [], []
    for sign in (+1, -1):
        tag = f"x shifted by {sign * rate:+,.1f} Hz"
        idx = next((i for i in mine if labels[i].endswith(tag)), None)
        if idx is None:
            return None
        ok = coherent(0, idx)
        if not ok.any():
            return None
        p = np.abs(R[:, 0, idx]) ** 2
        p = p - float(np.median(p))
        # the bump's extent: its coherent bins, two bins more each side
        sh = np.fft.fftshift(ok)
        k = np.flatnonzero(sh)
        lo, hi = max(0, int(k[0]) - 2), min(F - 1, int(k[-1]) + 2)
        sel = np.zeros(F, dtype=bool)
        sel[lo:hi + 1] = True
        sel = np.fft.ifftshift(sel)
        w = np.where(sel, np.maximum(p, 0.0), 0.0)
        if w.sum() <= 0:
            return None
        c = float(np.sum(w * fr) / w.sum())
        s = math.sqrt(max(float(np.sum(w * (fr - c) ** 2) / w.sum())
                          - df ** 2 / 12.0, 0.0))
        cents.append(c)
        widths.append(s)
    beta = float(min(1.0, max(0.05, np.mean(widths) / (0.181 * rate))))
    return float(np.mean(cents)), rate, beta


def fresh_separate(x, fs: float, alpha_sets, floor_per_hz: float | None = None,
                   nperseg: int | None = None) -> tuple:
    """Separate signals that overlap in frequency, each by its own cycle
    frequencies. alpha_sets: one entry per signal — a list of non-conjugate
    α, or {"alphas": [...], "conj": [...]}. Returns (list of y, report).

    For signal k, each bin's MMSE estimate uses ALL the copies (so the other
    signals' copies help cancel them). Signal k's own power in the bin —
    which the bin cannot give, because it also holds the others — is
    IDENTIFIED (module docstring): from significant pairs of k's coherent
    copies, P_k = R_0i·R_j0/R_ji; interpolated inside k's identified band,
    zero outside it; a signal with no identifiable bin takes the remainder
    of the bin once the identified signals are out (shared equally, and said
    so, when two or more cannot be identified). Each signal's report says
    how its power was obtained."""
    fs = _rate(fs)
    x, note = _iq_1d(x, "FRESH separate")
    if not isinstance(alpha_sets, (list, tuple)):
        raise ValueError("alpha_sets is a list: one set of cycle frequencies "
                         "per signal")
    sets = [_normalise_set(s, fs) for s in alpha_sets]
    if len(sets) < 2:
        raise ValueError("separation needs two or more sets of cycle "
                         "frequencies, one per signal")
    for k, (al, cj) in enumerate(sets):
        if not al and not cj:
            raise ValueError(f"signal {k + 1} has no cycle frequency — each "
                             "signal is separated by its own α")
    all_a = [a for s in sets for a in s[0]] + [b for s in sets for b in s[1]
                                              if b]
    nper = int(_check_nper(nperseg, x.size) or _nperseg(fs, all_a, x.size))
    labels = ["x (as received)"]
    sigs = [x]
    owner = [None]
    for k, (al, cj) in enumerate(sets):
        for a in al:
            for s in (a, -a):
                sigs.append(_shifted(x, fs, s))
                labels.append(f"signal {k + 1}: x shifted by {s:+,.1f} Hz")
                owner.append(k)
        for b in cj:
            sigs.append(_shifted(x, fs, b, conj=True))
            labels.append(f"signal {k + 1}: conj(x) shifted by {b:+,.1f} Hz")
            owner.append(k)
    V = _stack(sigs, nper)
    F, B, T = V.shape
    R = np.einsum("fbt,fct->fbc", V, np.conj(V)) / T
    diag = np.real(np.einsum("fbb->fb", R))
    P0 = diag[:, 0]
    n0, how = _floor_per_bin(P0, floor_per_hz, fs, nper)
    avail = np.maximum(P0 - n0, 0.0)
    # a sample cross-correlation of two independent series has
    # |R̂_ij|²·T_eff/(R_ii·R_jj) ~ Exp(1), T_eff the frames' worth of
    # independent evidence (half-overlapped frames are correlated: T/1.20
    # for the √Hann window). Coherent means clearing ln(F/p): `GATE_P` is the
    # chance of ANY spurious coherent bin across the spectrum — a lone
    # noise bin that passed a per-bin test identified a whole signal in the
    # first version of this gate.
    T_eff = T * _frame_independence(nper)
    z = math.log(F / GATE_P)

    def coherent(i: int, j: int) -> np.ndarray:
        return (np.abs(R[:, i, j]) ** 2 * T_eff
                > z * np.maximum(diag[:, i] * diag[:, j], 1e-300))

    P = np.zeros((len(sets), F))
    ident = [False] * len(sets)
    frac = [0.0] * len(sets)
    conj_of = {i: labels[i].split(": ", 1)[-1].startswith("conj")
               for i in range(1, B)}
    for k in range(len(sets)):
        mine = [i for i in range(1, B) if owner[i] == k]
        num = np.zeros(F)
        den = np.zeros(F)
        for ai in range(len(mine)):
            for bi in range(ai + 1, len(mine)):
                i, j = mine[ai], mine[bi]
                ok = coherent(0, i) & coherent(0, j) & coherent(i, j)
                if not ok.any():
                    continue
                rji = R[:, j, i]
                wgt = np.where(ok, np.abs(rji), 0.0)
                val = np.real(R[:, 0, i] * R[:, j, 0]
                              / np.where(ok, rji, 1.0))
                num += wgt * np.where(ok, val, 0.0)
                den += wgt
        est = np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)
        # a conjugate copy of a BPSK/AM-class signal (real symbols) mirrors
        # the spectrum about the carrier, |S(f)| = |S(2f_c − f)|, so where
        # it is coherent its cross-spectrum IS the signal's power: |R_0c|.
        # It covers the whole band, where pairs exist only in the roll-off
        # (interpolating across the flat top from the roll-off bins read a
        # BPSK's middle 20 % low, and the QPSK beside it 20 % high).
        single = np.full(F, np.nan)
        for i in mine:
            if conj_of[i]:
                ok = coherent(0, i)
                single = np.where(ok & ~np.isfinite(single),
                                  np.abs(R[:, 0, i]), single)
        est = np.where(np.isfinite(est), est, single)
        if np.isfinite(est).any():
            ident[k] = True
            frac[k] = float(np.mean(np.isfinite(est)))
            P[k] = np.clip(_fill_inside(est), 0.0, avail)
    unid = [k for k in range(len(sets)) if not ident[k]]
    rest = np.maximum(avail - P[[k for k in range(len(sets)) if ident[k]]]
                      .sum(axis=0), 0.0) if any(ident) else avail.copy()
    # an unidentified signal still shows WHERE it is: its own copies are
    # coherent with the bin only inside its band (in its roll-off, at the
    # band's edges), so their hull is its band; and HOW strong: at the
    # middle of a root-Nyquist roll-off the coherence is A/2, A its flat
    # level. The remainder of each bin is shared in proportion to those.
    weight = np.zeros((len(sets), F))
    shape_known = {}
    fr = np.fft.fftfreq(F, 1.0 / fs)
    for k in unid:
        mine = [i for i in range(1, B) if owner[i] == k]
        coh = np.zeros(F, dtype=bool)
        mag = np.zeros(F)
        for i in mine:
            ok = coherent(0, i)
            coh |= ok
            mag = np.maximum(mag, np.where(ok, np.abs(R[:, 0, i]), 0.0))
        if not coh.any():
            weight[k] = 1.0
            shape_known[k] = False
            continue
        shape_known[k] = True
        A = 2.0 * float(np.max(mag))
        model = _rc_from_copies(R, mine, labels, coherent, fr, fs, F,
                                sets[k][0])
        if model is None:
            inside = _fill_inside(np.where(coh, 1.0, np.nan)) > 0.5
            weight[k] = np.where(inside, A, 0.0)
        else:
            fc, rate, beta = model
            weight[k] = A * _rc_psd(fr - fc, rate, beta)
    wsum = weight[unid].sum(axis=0) if unid else np.zeros(F)
    for k in unid:
        P[k] = np.where(wsum > 0, rest * weight[k] / np.maximum(wsum, 1e-300),
                        0.0)
    tot = P.sum(axis=0)
    over = tot > avail
    if over.any():                       # identified powers cannot exceed the bin
        P[:, over] *= (avail[over] / np.maximum(tot[over], 1e-300))[None, :]
    outs, per = [], []
    for k in range(len(sets)):
        mine = [i for i in range(1, B) if owner[i] == k]
        Pk = P[k]
        p = np.zeros((F, B), dtype=np.complex128)
        p[:, 0] = Pk
        for i in mine:
            p[:, i] = R[:, i, 0]
        out, _mmse_k, _w, sig, tot = _mmse(V, p, Pk)
        y = istft(out, nper, x.size).astype(np.complex64)
        others = float(np.sum(np.maximum(P0 - Pk, 0.0)))
        before = (10 * math.log10(max(float(Pk.sum()), 1e-300) / others)
                  if others > 0 else math.inf)
        after = _out_db(sig, tot)
        if ident[k]:
            power_how = (f"identified from pairs of its own coherent copies in "
                         f"{frac[k] * 100:.0f} % of the bins, interpolated "
                         "inside its band")
        elif len(unid) == 1:
            power_how = ("no pair of its copies is coherent anywhere (proper "
                         "modulation, little excess bandwidth): its power is "
                         "the remainder of each bin once the identified "
                         "signals are taken out"
                         + (", inside its own band (the hull of the bins "
                            "where its copies are coherent)"
                            if shape_known.get(k) else ""))
        else:
            power_how = (f"{len(unid)} signals could not be identified: they "
                         "share the remainder of each bin in proportion to "
                         "their bands and levels as their own copies show "
                         "them (a root-Nyquist roll-off assumed: coherence "
                         "A/2 at its middle) — an ASSUMPTION, so this "
                         "signal's SINR is an estimate under it"
                         + ("" if shape_known.get(k) else
                            "; its copies were nowhere coherent, so its share "
                            "is EQUAL"))
        outs.append(y)
        per.append({"signal": k + 1, "alphas_hz": sets[k][0],
                    "conj_alphas_hz": sets[k][1],
                    "sinr_before_db": before, "sinr_after_db": after,
                    "gain_db": after - before, "identified": bool(ident[k]),
                    "bins_identified": frac[k], "power_method": power_how})
    rep = {"method": "fresh_separate", "tier": _prov.tier_for("fresh_separate"),
           "signals": per, "branches": labels, "nperseg": nper,
           "floor_per_bin": n0, "floor_method": how,
           "snr_before_db": per[0]["sinr_before_db"],
           "snr_after_db": per[0]["sinr_after_db"],
           "gain_db": per[0]["gain_db"],
           "gate": (f"a cross-spectrum counts as coherent when |R̂|²·T_eff/"
                    f"(R_ii·R_jj) > ln({F}/{GATE_P:g}) — the noise-only "
                    "distribution of a sample cross-correlation is "
                    "exponential; T_eff the independent frames, and the "
                    f"chance of any spurious coherent bin of the {F} is "
                    f"{GATE_P:g}"),
           "snr_method": ("blind: each signal's power per bin identified as "
                          "power_method says; SINR before = that power over "
                          "everything else in the bin; after = the output's "
                          "signal component over everything else in it, bin "
                          "by bin (a frequency-dependent gain, which an "
                          "equalizer undoes, is not counted as interference). "
                          "Measured against ground truth: within 0.5 dB for "
                          "signals of equal power; ~1.5 dB optimistic for a "
                          "proper signal 6 dB under its neighbour"),
           "sizing": _FRESH_SIZING,
           "words": "; ".join(f"signal {d['signal']}: "
                              f"{d['sinr_before_db']:+.1f} → "
                              f"{d['sinr_after_db']:+.1f} dB SINR"
                              for d in per)}
    if note:
        rep["note"] = note
    return outs, rep


# ---------------------------------------------------------------------------
# SCORE (§4.3, 3) — the Kraken's five channels
# ---------------------------------------------------------------------------
def _cyclic_ref(X: np.ndarray, fs: float, alpha: float, conj: bool, lag: int
                ) -> tuple:
    M, N = X.shape
    n = np.arange(lag, N, dtype=np.float64)
    ph = np.exp(2j * np.pi * np.mod(alpha / fs * n, 1.0))
    x = X[:, lag:]
    src = X[:, :N - lag]
    u = (np.conj(src) if conj else src) * ph[None, :]
    return x, u


def _best_lag(X: np.ndarray, fs: float, alpha: float, conj: bool) -> int:
    sps = fs / max(abs(alpha), 1e-9)
    cands = sorted({0, 1, 2, 3, 4, int(round(0.25 * sps)), int(round(0.5 * sps))})
    cands = [c for c in cands if 0 <= c < X.shape[1] // 4]
    best, lag = -1.0, 0
    p0 = float(np.mean(np.abs(X) ** 2)) or 1.0
    for c in cands:
        x, u = _cyclic_ref(X, fs, alpha, conj, c)
        v = float(np.sum(np.abs(np.mean(x * np.conj(u), axis=1)))) / p0
        if v > best:
            best, lag = v, c
    return lag


def score(X, fs: float, alpha: float, conj: bool = False,
          lag: int | None = None) -> tuple:
    """Cross-SCORE blind beamforming. X: [channels, n] coherent samples (the
    Kraken cut keeps all five). Returns (y, weights, report).

    The weights w maximise the correlation between the output y = wᴴx and
    a reference built from the array data at the cycle frequency,
    u(t) = x(t−τ)·e^{j2παt} (or x*(t−τ)·e^{j2παt} for a conjugate α): the
    principal eigenvector of R_xx⁻¹·R_xu·R_uu⁻¹·R_ux, whose eigenvalue is the
    squared self-coherence achieved. Only the signal of interest has a
    feature at α, so maximising it steers onto that signal and nulls what
    lacks it — no direction of arrival, no array calibration, no training
    sequence. `lag` τ: None picks the lag where the array's cyclic
    correlation at α is strongest (0 for a conjugate carrier feature).
    `w` is unit-norm, so the output's noise power equals one channel's.

    The report's SINR is blind and counts an interferer as interference
    (module docstring); `snr_before_eigen_db` / `snr_after_eigen_db` are the
    eigen-floor numbers, which count it as signal."""
    fs = _rate(fs)
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("SCORE needs two or more coherent channels — a "
                         "KrakenSDR cut keeps all five; a single-channel cut "
                         "has no array to steer")
    if X.shape[1] < 256:
        raise ValueError(f"SCORE needs at least 256 samples a channel to "
                         f"estimate the array's statistics; this has "
                         f"{X.shape[1]:,}")
    X = X.astype(np.complex128, copy=False)
    bad = int(np.count_nonzero(~np.isfinite(X)))
    if bad:
        raise ValueError(f"SCORE: {bad:,} samples are not finite (NaN or "
                         "infinity) — a damaged file. Nothing was filtered.")
    try:
        alpha = float(alpha)
    except (TypeError, ValueError):
        raise ValueError(f"SCORE: {alpha!r} is not a cycle frequency in "
                         "Hz") from None
    if not math.isfinite(alpha) or abs(alpha) >= fs:
        raise ValueError(f"SCORE: α = {alpha!r} Hz is not a cycle frequency a "
                         f"cut at {fs:,.0f} S/s can show")
    M, N = X.shape
    tau = int(_best_lag(X, fs, alpha, conj) if lag is None else lag)
    if not 0 <= tau < N // 2:
        raise ValueError(f"SCORE: the lag {lag!r} must be between 0 and half "
                         "the cut")
    x, u = _cyclic_ref(X, fs, alpha, conj, tau)
    L = x.shape[1]
    Rxx = x @ x.conj().T / L
    Ruu = u @ u.conj().T / L
    Rxu = x @ u.conj().T / L
    eps = 1e-12 * np.real(np.trace(Rxx)) / M
    A = np.linalg.solve(Rxx + eps * np.eye(M), Rxu) @ np.linalg.solve(
        Ruu + eps * np.eye(M), Rxu.conj().T)
    ev, vecs = np.linalg.eig(A)
    k = int(np.argmax(ev.real))
    w = vecs[:, k]
    w = w / np.linalg.norm(w)
    y = (w.conj() @ X).astype(np.complex64)
    rho = float(math.sqrt(max(ev.real[k], 0.0)))
    # -- blind SINR that counts an interferer as interference -----------------
    lam = np.sort(np.linalg.eigvalsh(Rxx))
    sigma2 = float(np.mean(lam[: max(1, M - 2)]))
    ch_pow = np.real(np.diag(Rxx))
    # the signal's array response: SCORE converges on the maximum-SINR
    # beamformer w ∝ R⁻¹a, so a ∝ R·w (measured: within ~0.5 dB at 0 dB SNR
    # a channel; the principal singular vector of R_xu, tried first, picks
    # up the interferer when the feature is weak and was 2.7 dB off)
    a = Rxx @ w
    a = a * (math.sqrt(M) / max(float(np.linalg.norm(a)), 1e-300))
    Ri = np.linalg.inv(Rxx + eps * np.eye(M))
    capon = 1.0 / max(float(np.real(a.conj() @ Ri @ a)), 1e-300)
    p_s = max(capon - sigma2 / M, 1e-12 * float(ch_pow.mean()))
    soi_ch = p_s * np.abs(a) ** 2
    sinr_ch = soi_ch / np.maximum(ch_pow - soi_ch, 1e-300)
    best_ch = int(np.argmax(sinr_ch))
    before = 10 * math.log10(max(float(sinr_ch[best_ch]), 1e-12))
    p_out = float(np.real(w.conj() @ Rxx @ w))
    soi_out = p_s * abs(complex(w.conj() @ a)) ** 2
    after = 10 * math.log10(max(soi_out / max(p_out - soi_out, 1e-300), 1e-12))
    # the eigen-floor numbers (interferer counted as signal), kept beside
    snr_ch_e = (ch_pow - sigma2) / sigma2
    before_e = 10 * math.log10(max(float(snr_ch_e.max()), 1e-12))
    after_e = 10 * math.log10(max((p_out - sigma2) / sigma2, 1e-12))
    rep = {"method": "score", "tier": _prov.tier_for("score"),
           "alpha_hz": float(alpha), "conj": bool(conj), "lag": tau,
           "weights": [[float(c.real), float(c.imag)] for c in w],
           "self_coherence": rho, "channels": M,
           "eigenvalues": [float(v) for v in lam], "noise_per_channel": sigma2,
           "signal_response": [[float(c.real), float(c.imag)] for c in a],
           "signal_power_per_channel": p_s,
           "best_channel": best_ch, "snr_before_db": before,
           "snr_after_db": after, "gain_db": after - before,
           "snr_before_eigen_db": before_e, "snr_after_eigen_db": after_e,
           "max_white_noise_gain_db": 10 * math.log10(M),
           "snr_method": ("blind: the signal's array response from the "
                          "SCORE weights (a ∝ R·w — SCORE converges on the "
                          "maximum-SINR beamformer for the one signal with "
                          "α), its power by Capon less the white-noise leak "
                          f"(noise per channel = mean of the {max(1, M - 2)} "
                          "smallest eigenvalues); SINR before = that power "
                          "over everything else in the best single channel, "
                          "after = the same for the beamformer output. "
                          "Assumes equal-gain channels"),
           "sizing": (f"up to 10·log10({M}) = {10 * math.log10(M):.1f} dB of "
                      "array gain on white noise, plus nulling of interferers "
                      "that do not share α (tens of dB)"),
           "words": (f"SCORE on {M} channels at α = {alpha:,.1f} Hz"
                     f"{' (conjugate)' if conj else ''}: best single channel "
                     f"{before:+.1f} dB SINR → beamformed {after:+.1f} dB; "
                     f"output self-coherence {rho:.3f}.")}
    if conj:
        # a conjugate carrier feature at lag 0 has coherence 1 for any BPSK
        # pulse shape, so the measured coherence gives the SINR directly
        rep["sinr_from_coherence_db"] = (
            10 * math.log10(rho / (1 - rho)) if 0 < rho < 1 else None)
        rep["sinr_from_coherence_method"] = (
            "ρ = SINR/(1+SINR) for a BPSK-class conjugate feature (clean "
            "coherence 1); counts interference correctly")
    return y, w, rep


# ---------------------------------------------------------------------------
# RFI mask and interpolation (DETECTION_DESIGN §4.2 Clean)
# ---------------------------------------------------------------------------
def rfi_mask_interp(x, fs: float, pfa: float = 1e-4,
                    nperseg: int | None = None,
                    impulse_factor: float = 4.0) -> tuple:
    """Mask strong narrowband and impulsive interference in the
    time-frequency plane and fill the masked cells by interpolating each
    frequency row across time (real and imaginary parts). Tier INFERRED: the
    filled cells are a guess.

    A cell is masked when its power exceeds what the row's own noise could
    produce at `pfa` per cell: |Z|² > μ·ln(1/pfa), μ the row's typical
    power (median/ln 2 — the median, because the interference must not set
    its own threshold). A frame is masked whole when its median cell is
    `impulse_factor`× its rows' typical power (a broadband impulse). The
    mask is widened by one frame each side (not wrapping round the ends).

    LIMIT, plainly: it cannot tell a burst of the signal from a burst of
    interference. Use it where the interference is the intermittent part
    and the signal is continuous (a continuous signal raises its own rows'
    medians and is left alone)."""
    fs = _rate(fs)
    x, note = _iq_1d(x, "the RFI mask")
    try:
        pfa = float(pfa)
        impulse_factor = float(impulse_factor)
    except (TypeError, ValueError):
        raise ValueError("pfa and impulse_factor are numbers") from None
    if not 0.0 < pfa < 1.0:
        raise ValueError(f"pfa is a probability between 0 and 1, not {pfa!r}")
    if not impulse_factor > 1.0:
        raise ValueError("impulse_factor must be above 1 (a frame that many "
                         "times its rows' typical power is an impulse)")
    nper = int(_check_nper(nperseg, x.size) or 256)
    Z = stft(x, nper)
    P = np.abs(Z) ** 2
    mu = np.median(P, axis=1) / math.log(2.0) + 1e-300
    cell = P > mu[:, None] * math.log(1.0 / pfa)
    frame = np.median(P / mu[:, None], axis=0) > impulse_factor
    mask = cell | frame[None, :]
    grown = mask.copy()
    grown[:, 1:] |= mask[:, :-1]
    grown[:, :-1] |= mask[:, 1:]
    mask = grown
    Zf = Z.copy()
    t = np.arange(Z.shape[1])
    for f in range(Z.shape[0]):
        m = mask[f]
        if not m.any():
            continue
        good = ~m
        if good.sum() < 2:
            Zf[f, m] = 0.0
            continue
        Zf[f, m] = (np.interp(t[m], t[good], Z[f, good].real)
                    + 1j * np.interp(t[m], t[good], Z[f, good].imag))
    y = istft(Zf, nper, x.size)
    removed = float(np.sum(np.abs(Z) ** 2 - np.abs(Zf) ** 2))
    total = float(np.sum(P)) or 1.0
    p_in = float(np.mean(np.abs(x) ** 2))
    p_out = float(np.mean(np.abs(y) ** 2))
    rep = {"method": "rfi_mask_interp", "tier": _prov.tier_for("rfi_mask_interp"),
           "masked_fraction": float(mask.mean()),
           "impulsive_frames": int(frame.sum()), "nperseg": nper,
           "pfa_per_cell": pfa, "impulse_factor": impulse_factor,
           "power_removed_db": (10 * math.log10(max(removed, 1e-300) / total)
                                if removed > 0 else None),
           "power_ratio_db": (10 * math.log10(p_out / p_in)
                              if p_in > 0 and p_out > 0 else None),
           "words": (f"masked {mask.mean() * 100:.2f} % of the time-frequency "
                     f"cells ({int(frame.sum())} impulsive frames) and filled "
                     "them by interpolation — INFERRED samples, a best guess"),
           "sizing": ("removes intermittent interference; does nothing for "
                      "white noise and cannot tell a signal burst from an "
                      "interference burst")}
    if note:
        rep["note"] = note
    return y.astype(np.complex64), rep
