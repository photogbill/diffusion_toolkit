# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The three classical filters of DETECTION_DESIGN §4.3 (matched, FRESH,
SCORE) and the cut's two non-cyclic cleans (Wiener, RFI mask), checked
against GROUND TRUTH: every test builds the signal and the noise
separately, so the SNR a filter reports blind can be compared with the SNR
its output really has. The numbers asserted here are the ones the module
docstring quotes."""

from __future__ import annotations

import math

import numpy as np
import pytest

import helpers_cyclo as H
from atk_diffusion import provenance
from atk_diffusion.cyclo import filters as F

FS = 48_000.0


def _lowpass(x, cutoff=7_000.0, taps=129):
    from scipy.signal import firwin, lfilter
    return lfilter(firwin(taps, cutoff, fs=FS), 1, x)


def _psk(kind, snr_db, rng, n, rate=4800.0, beta=0.35, fc=0.0, delay=0,
         rect=False):
    """(signal, noise): unit-floor white noise, the signal at `snr_db`
    in-band (its spectral level against the floor over (1+β)·R)."""
    s = H.linmod(kind, FS, rate, n, rng, beta=beta, fc=fc, delay=delay,
                 rect=rect)
    s = s * H.inband_amplitude(snr_db, rate * (1 + beta), FS)
    return s, H.noise(n, rng)


# ---------------------------------------------------------------------------
# the STFT the FRESH and Wiener filters work in
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nper", [32, 256])
def test_the_stft_reconstructs_exactly(rng, nper):
    x = H.noise(5_000, rng)
    y = F.istft(F.stft(x, nper), nper, x.size)
    assert np.max(np.abs(y - x)) < 1e-10


# ---------------------------------------------------------------------------
# 1. matched-filter parameters and the matched filter
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind,fc,snr,delay", [("bpsk", 700.0, 5.0, 3),
                                               ("qpsk", -900.0, 8.0, 7)])
def test_matched_parameters_recover_rate_carrier_and_timing(kind, fc, snr,
                                                            delay):
    rng = np.random.default_rng(21)
    s, nz = _psk(kind, snr, rng, int(FS), fc=fc, delay=delay)
    mp = F.matched_parameters(s + nz, FS)
    assert mp["tier"] == "measured"
    assert mp["symbol_rate_hz"] == pytest.approx(4800.0, abs=0.5)
    assert mp["carrier_offset_hz"] == pytest.approx(fc, abs=3.0)
    # the symbol centres (truth: sample `delay` + k·10), circularly
    t = mp["timing_offset_s"] * FS
    sps = FS / 4800.0
    assert min(abs(t - delay), sps - abs(t - delay)) < 0.5
    assert mp["samples_per_symbol"] == pytest.approx(10.0, rel=1e-3)
    assert "Bd" in mp["words"] and "demodulator" in mp["words"]
    # the cut was not band-limited to the signal: the parameters say they did
    assert mp["prefilter"] and mp["prefilter"]["occupied_bandwidth_hz"] < 8000


def test_matched_parameters_leave_fsk_timing_to_the_discriminator(rng):
    n = int(FS)
    x = H.fsk4(FS, 4800.0, n, rng) * H.inband_amplitude(15.0, 8000.0, FS) \
        + H.noise(n, rng)
    mp = F.matched_parameters(_lowpass(x), FS)
    assert mp["timing_offset_s"] is None
    assert "left to the demodulator" in mp["timing"]["words"]
    with pytest.raises(ValueError, match="discriminator"):
        F.matched_filter(_lowpass(x), FS, params=mp, floor_per_hz=1.0 / FS)


def test_the_matched_filter_snr_at_the_symbol_instants_is_measured_right():
    """Blind SNR before (in-band) and after (at the symbol instants)
    against the truth: the same filter applied to the signal alone and to
    the noise alone. Gain ≈ 10·log10(1+β) = +1.3 dB at β 0.35."""
    rng = np.random.default_rng(31)
    n = int(FS)
    s, nz = _psk("bpsk", 5.0, rng, n, fc=700.0, delay=3)
    s, nz = _lowpass(s), _lowpass(nz)              # a cut is low-passed
    y, rep = F.matched_filter(s + nz, FS, floor_per_hz=1.0 / FS)
    assert rep["tier"] == "cleaned" == provenance.tier_for("matched_filter")
    assert y.dtype == np.complex64 and y.shape == (n,)
    assert rep["rolloff"] == pytest.approx(0.35, abs=0.05)
    mp = F.matched_parameters(s + nz, FS)
    ys, _ = F.matched_filter(s, FS, params=mp, floor_per_hz=1.0 / FS,
                             rolloff=rep["rolloff"])
    yn, _ = F.matched_filter(nz, FS, params=mp, floor_per_hz=1.0 / FS,
                             rolloff=rep["rolloff"])
    pos = F.symbol_instants(n, FS, rep["symbol_rate_hz"],
                            rep["timing_offset_s"], 8)
    true_after = 10 * np.log10(
        np.mean(np.abs(F.sample_at(ys.astype(complex), pos)) ** 2)
        / np.mean(np.abs(F.sample_at(yn.astype(complex), pos)) ** 2))
    assert rep["snr_before_db"] == pytest.approx(5.0, abs=0.5)
    assert rep["snr_after_db"] == pytest.approx(true_after, abs=0.3)
    assert rep["gain_db"] == pytest.approx(10 * math.log10(1.35), abs=0.4)
    assert rep["instants"]["count"] == pos.size > 4000
    assert "measured against the floor" in rep["words"]


@pytest.mark.parametrize("beta", [0.2, 0.35, 0.5, 0.8])
def test_the_rolloff_is_inverted_from_the_99_percent_bandwidth(beta):
    """The 99 % band of a raised-cosine spectrum, computed numerically,
    gives back its β (the naive width/rate − 1 gave 0.16 for 0.35)."""
    R = 4800.0
    f = np.linspace(-R, R, 200_001)
    a = np.abs(f)
    lo, hi = (1 - beta) * R / 2, (1 + beta) * R / 2
    P = np.where(a <= lo, 1.0, np.where(
        a >= hi, 0.0, 0.5 * (1 + np.cos(np.pi / (beta * R) * (a - lo)))))
    c = np.cumsum(P) / P.sum()
    obw = f[np.searchsorted(c, 0.995)] - f[np.searchsorted(c, 0.005)]
    assert F.rolloff_from_obw(obw, R) == pytest.approx(beta, abs=0.01)


# ---------------------------------------------------------------------------
# Wiener (the time-invariant baseline)
# ---------------------------------------------------------------------------
def test_wiener_gains_the_out_of_band_noise_and_nothing_in_band():
    """A 4.8 kBd QPSK alone in a 48 kHz cut of white noise: the measured
    gain is the band ratio, ~8.7 dB; in-band nothing."""
    rng = np.random.default_rng(41)
    n = int(2 * FS)
    s, nz = _psk("qpsk", 0.0, rng, n, beta=0.1)
    y, rep = F.wiener_clean(s + nz, FS, floor_per_hz=1.0 / FS)
    true_before = 10 * np.log10(np.mean(np.abs(s) ** 2) / np.mean(np.abs(nz) ** 2))
    true_after = H.sinad(y, s)
    assert rep["tier"] == "cleaned"
    assert rep["snr_before_db"] == pytest.approx(true_before, abs=0.3)
    assert rep["snr_after_db"] == pytest.approx(true_after, abs=0.3)
    assert rep["gain_db"] == pytest.approx(true_after - true_before, abs=0.4)
    assert rep["in_band_gain_db"] == 0.0
    assert true_after - true_before > 9.0          # the band ratio, 9.6 dB


def test_wiener_full_band_snr_counts_a_low_passed_cut_correctly():
    """A canonical cut is low-passed: its stopband holds almost no noise.
    Counting a full floor there understated the cut's SNR by its stopband
    share (here 6 dB)."""
    rng = np.random.default_rng(42)
    n = int(2 * FS)
    s, nz = _psk("qpsk", 6.0, rng, n)
    s, nz = _lowpass(s, 6_000.0, 257), _lowpass(nz, 6_000.0, 257)
    _y, rep = F.wiener_clean(s + nz, FS, floor_per_hz=1.0 / FS)
    true_full = 10 * np.log10(np.mean(np.abs(s) ** 2) / np.mean(np.abs(nz) ** 2))
    assert rep["full_band_snr_before_db"] == pytest.approx(true_full, abs=0.6)


# ---------------------------------------------------------------------------
# 2. FRESH clean — gain over the time-invariant baseline, bounded
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind,conj,lo,hi", [
    ("bpsk", [0.0, 4800.0, -4800.0], 2.0, 4.5),    # the conjugate copy: ~3 dB
    ("qpsk", [], 0.1, 1.5),                        # the roll-off only
])
def test_fresh_gain_is_measured_positive_and_bounded_by_redundancy(kind, conj,
                                                                   lo, hi):
    rng = np.random.default_rng(5)
    n = int(2 * FS)
    s, nz = _psk(kind, 0.0, rng, n)
    x = s + nz
    yw, _rw = F.wiener_clean(x, FS, floor_per_hz=1.0 / FS)
    yf, rf = F.fresh_clean(x, FS, [4800.0], conj, floor_per_hz=1.0 / FS)
    true_gain = H.sinad(yf, s) - H.sinad(yw, s)
    assert rf["tier"] == "cleaned"
    assert lo < true_gain < hi, true_gain
    assert rf["gain_db"] > 0
    assert rf["gain_db"] == pytest.approx(true_gain, abs=0.8)
    assert rf["snr_before_db"] == pytest.approx(H.sinad(yw, s), abs=1.0)
    assert len(rf["branches"]) == 1 + 2 + len(conj)
    assert rf["full_band_snr_before_db"] < rf["snr_before_db"]
    assert "spectral redundancy" in rf["sizing"]


def test_fresh_blind_floor_is_said_when_no_floor_is_given(rng):
    s, nz = _psk("bpsk", 3.0, rng, int(FS))
    _y, rep = F.fresh_clean(s + nz, FS, [4800.0], [0.0])
    assert rep["floor_method"].startswith("blind")


# ---------------------------------------------------------------------------
# 2b. FRESH separate — two co-channel signals with different baud rates
# ---------------------------------------------------------------------------
def _sinad_per_bin(y, s, nper=128, trim=3000):
    """The truth for an MMSE output: its component correlated with the true
    signal, projected bin by bin (a frequency-dependent gain — which an
    equalizer undoes — counts as signal), over everything else in it.
    `H.sinad` fits ONE scalar and counts that shaping as error."""
    Y = F.stft(np.asarray(y)[trim:-trim], nper)
    S = F.stft(np.asarray(s)[trim:-trim], nper)
    c = np.sum(Y * np.conj(S), axis=1) / np.maximum(
        np.sum(np.abs(S) ** 2, axis=1), 1e-300)
    sig = np.abs(c) ** 2 * np.sum(np.abs(S) ** 2, axis=1)
    tot = np.sum(np.abs(Y) ** 2, axis=1)
    return 10 * np.log10(sig.sum() / max((tot - sig).sum(), 1e-300))


def test_fresh_separate_two_cochannel_signals_with_different_bauds():
    """BPSK 4800 Bd and QPSK 6000 Bd of equal power fully on top of each
    other (both 15 dB above the noise): each comes out ~4.8 dB better in
    SIR (−0.7 → +4.1 dB), and the blind report agrees with the truth (the
    first version reported +11 and +16 dB for gains of +0.8 and −6.8)."""
    rng = np.random.default_rng(7)
    n = int(2 * FS)
    a = H.inband_amplitude(15.0, 6000 * 1.35, FS)
    s1 = a * H.linmod("bpsk", FS, 4800.0, n, rng, beta=0.35)
    s2 = a * H.linmod("qpsk", FS, 6000.0, n, rng, beta=0.35)
    x = s1 + s2 + H.noise(n, rng)
    outs, rep = F.fresh_separate(
        x, FS, [{"alphas": [4800.0], "conj": [0.0, 4800.0, -4800.0]},
                {"alphas": [6000.0]}], floor_per_hz=1.0 / FS)
    assert rep["tier"] == "cleaned" and len(outs) == 2
    for k, s in enumerate((s1, s2)):
        before, after = H.sinad(x, s), H.sinad(outs[k], s)
        got = rep["signals"][k]
        assert after - before > 4.0, (k, before, after)
        assert got["sinr_before_db"] == pytest.approx(before, abs=0.5)
        assert got["sinr_after_db"] == pytest.approx(_sinad_per_bin(outs[k], s),
                                                     abs=0.6)
    assert rep["signals"][0]["identified"]
    assert not rep["signals"][1]["identified"]
    assert "remainder" in rep["signals"][1]["power_method"]


def test_fresh_separate_two_proper_signals_says_its_assumption():
    """Two QPSK (no conjugate feature, 0.35 roll-off): neither power can be
    identified from pairs; the remainder is shared by each one's raised-
    cosine model at its own α (band and level from where and how strongly
    its copies are coherent), the report says it is an ASSUMPTION — and
    the small real gain (~2 dB) is still a gain."""
    rng = np.random.default_rng(8)
    n = int(2 * FS)
    a = H.inband_amplitude(15.0, 6000 * 1.35, FS)
    s1 = a * H.linmod("qpsk", FS, 4800.0, n, rng, beta=0.35)
    s2 = a * H.linmod("qpsk", FS, 6000.0, n, rng, beta=0.35)
    x = s1 + s2 + H.noise(n, rng)
    outs, rep = F.fresh_separate(x, FS, [[4800.0], [6000.0]],
                                 floor_per_hz=1.0 / FS)
    for k, s in enumerate((s1, s2)):
        gain = H.sinad(outs[k], s) - H.sinad(x, s)
        assert gain > 1.0
        assert "ASSUMPTION" in rep["signals"][k]["power_method"]
        assert rep["signals"][k]["sinr_after_db"] == pytest.approx(
            _sinad_per_bin(outs[k], s), abs=1.0)


# ---------------------------------------------------------------------------
# 3. SCORE on a simulated five-element array
# ---------------------------------------------------------------------------
def _array(rng, n, snr_db=0.0, inr_db=10.0, M=5):
    s = H.linmod("bpsk", FS, 4800.0, n, rng, beta=0.35)
    i = H.linmod("qpsk", FS, 6000.0, n, rng, beta=0.35)
    S = math.sqrt(10 ** (snr_db / 10)) * np.outer(H.uca(M, 0.3), s)
    I = math.sqrt(10 ** (inr_db / 10)) * np.outer(H.uca(M, 2.1), i)
    N = np.stack([H.noise(n, rng) for _ in range(M)])
    return S, I, N


@pytest.mark.parametrize("conj,alpha", [(False, 4800.0), (True, 0.0)])
def test_score_steers_onto_the_signal_and_nulls_the_interferer(conj, alpha):
    """Desired BPSK (4800 Bd) from one direction, a QPSK interferer (6000 Bd)
    10 dB stronger from another, noise: one element sees −10.4 dB SINR; the
    beamformer gives ~+6.7 dB (within the 7 dB white-noise array gain) and
    puts the interferer ~45 dB down. The blind report counts the
    interferer as interference (the first version said −3.7 dB 'gain')."""
    rng = np.random.default_rng(11)
    n = int(FS)
    S, I, N = _array(rng, n)
    y, w, rep = F.score(S + I + N, FS, alpha, conj=conj)
    ys, yi, yn = w.conj() @ S, w.conj() @ I, w.conj() @ N
    out = 10 * np.log10(np.mean(np.abs(ys) ** 2)
                        / (np.mean(np.abs(yi) ** 2) + np.mean(np.abs(yn) ** 2)))
    single = 10 * np.log10(1.0 / (10.0 + 1.0))
    null = 10 * np.log10((np.mean(np.abs(yi) ** 2) / np.mean(np.abs(ys) ** 2))
                         / 10.0)
    assert out - single > 15.0
    assert out < 10 * math.log10(5) + 0.3          # the array cannot beat M
    assert null < -30.0
    assert rep["tier"] == "cleaned" and rep["channels"] == 5
    assert np.linalg.norm(w) == pytest.approx(1.0)
    assert np.allclose(y, w.conj() @ (S + I + N), atol=1e-4)
    assert rep["snr_before_db"] == pytest.approx(single, abs=1.5)
    assert rep["snr_after_db"] == pytest.approx(out, abs=1.5)
    assert rep["gain_db"] > 12.0
    assert rep["snr_before_eigen_db"] > rep["snr_before_db"] + 15
    if conj:
        assert rep["sinr_from_coherence_db"] == pytest.approx(out, abs=1.0)


def test_score_white_noise_gain_is_bounded_by_the_array():
    rng = np.random.default_rng(12)
    n = int(FS)
    S, _I, N = _array(rng, n, snr_db=-5.0, inr_db=-200.0)
    _y, w, rep = F.score(S + N, FS, 4800.0)
    ys, yn = w.conj() @ S, w.conj() @ N
    out = 10 * np.log10(np.mean(np.abs(ys) ** 2) / np.mean(np.abs(yn) ** 2))
    assert out == pytest.approx(-5.0 + 10 * math.log10(5), abs=0.6)
    assert rep["max_white_noise_gain_db"] == pytest.approx(6.99, abs=0.01)


# ---------------------------------------------------------------------------
# RFI mask and interpolation — INFERRED
# ---------------------------------------------------------------------------
def _rfi(n, rng):
    t = np.arange(n) / FS
    r = np.zeros(n, complex)
    for k in range(int(n / FS * 10)):               # 20 ms on, every 100 ms
        a0 = int((0.1 * k + 0.03) * FS)
        a1 = a0 + int(0.02 * FS)
        r[a0:a1] += 10.0 * np.exp(2j * np.pi * 2000.0 * t[a0:a1])
    for _ in range(15):                             # impulses
        p = int(rng.integers(0, n - 10))
        r[p:p + 3] += 30.0 * np.exp(1j * rng.uniform(0, 2 * np.pi))
    return r


def test_rfi_mask_removes_tone_bursts_and_impulses_and_is_inferred():
    rng = np.random.default_rng(3)
    n = int(FS)
    s, nz = _psk("qpsk", 10.0, rng, n)
    clean_ref = H.sinad(s + nz, s)
    x = s + nz + _rfi(n, rng)
    y, rep = F.rfi_mask_interp(x, FS)
    assert rep["tier"] == "inferred" == provenance.tier_for("rfi_mask_interp")
    before, after = H.sinad(x, s), H.sinad(y, s)
    assert after > before + 8.0
    assert after > clean_ref - 3.5
    assert 0.0 < rep["masked_fraction"] < 0.25 and rep["impulsive_frames"] > 0
    assert rep["power_ratio_db"] < -5.0           # most of the power was RFI
    assert "INFERRED" in rep["words"]
    # a continuous signal alone is left alone
    y2, rep2 = F.rfi_mask_interp(s + nz, FS)
    assert rep2["masked_fraction"] < 0.01
    assert abs(H.sinad(y2, s) - clean_ref) < 0.2


def test_rfi_mask_does_not_wrap_from_the_first_frame_to_the_last(rng):
    n = 20_000
    x = H.noise(n, rng)
    x[10:20] += 400.0                               # an impulse at the start
    y, rep = F.rfi_mask_interp(x, FS)
    assert rep["impulsive_frames"] >= 1
    # the end of the record is untouched (np.roll used to mask it too)
    assert np.allclose(y[-600:], x[-600:], atol=1e-5)


# ---------------------------------------------------------------------------
# every output carries its tier and measured numbers; bad inputs in words
# ---------------------------------------------------------------------------
def test_every_output_carries_its_tier_and_measured_numbers():
    rng = np.random.default_rng(13)
    n = int(FS)
    s, nz = _psk("bpsk", 6.0, rng, n, fc=500.0, delay=2)
    x = _lowpass(s + nz)
    runs = {
        "matched_filter": F.matched_filter(x, FS, floor_per_hz=1.0 / FS)[1],
        "wiener": F.wiener_clean(x, FS, floor_per_hz=1.0 / FS)[1],
        "fresh": F.fresh_clean(x, FS, [4800.0], [1000.0],
                               floor_per_hz=1.0 / FS)[1],
        "fresh_separate": F.fresh_separate(
            x, FS, [{"alphas": [4800.0], "conj": [1000.0]}, [6000.0]],
            floor_per_hz=1.0 / FS)[1],
        "score": F.score(np.stack([x, x * 1j, -x, x, x]) + np.stack(
            [H.noise(n, rng) for _ in range(5)]), FS, 4800.0)[2],
        "rfi_mask_interp": F.rfi_mask_interp(x, FS)[1],
    }
    for method, rep in runs.items():
        assert rep["tier"] == provenance.tier_for(method), method
        assert rep["tier"] in provenance.TIERS
        assert rep["words"] and rep["sizing"], method
        if method != "rfi_mask_interp":
            assert math.isfinite(rep["snr_before_db"]), method
            assert math.isfinite(rep["snr_after_db"]), method
            assert rep["snr_method"], method
    assert F.matched_parameters(x, FS)["tier"] == provenance.tier_for(
        "matched_parameters") == "measured"


def test_a_kraken_cut_given_to_a_one_channel_filter_says_channel_0(rng):
    X = np.stack([H.noise(4096, rng) for _ in range(5)])
    _y, rep = F.wiener_clean(X, FS)
    assert "channel 0 of 5" in rep["note"]


@pytest.mark.parametrize("call,words", [
    (lambda: F.wiener_clean(np.zeros(0, complex), FS), "no samples"),
    (lambda: F.fresh_clean(np.ones(100, complex), FS, [4800.0]),
     "at least 1,024 samples"),
    (lambda: F.fresh_clean(np.full(4096, np.nan, complex), FS, [4800.0]),
     "not finite"),
    (lambda: F.fresh_clean(np.ones(4096, complex), FS, ["fast"]),
     "not a cycle frequency"),
    (lambda: F.fresh_clean(np.ones(4096, complex), FS, [60_000.0]),
     "beyond what a cut"),
    (lambda: F.fresh_clean(np.ones(4096, complex), FS, []),
     "at least one cycle frequency"),
    (lambda: F.fresh_separate(np.ones(4096, complex), FS, [[4800.0]]),
     "two or more sets"),
    (lambda: F.fresh_separate(np.ones(4096, complex), FS, [[4800.0], []]),
     "signal 2 has no cycle frequency"),
    (lambda: F.score(np.ones((1, 4096), complex), FS, 4800.0),
     "two or more coherent channels"),
    (lambda: F.score(np.ones((5, 100), complex), FS, 4800.0), "256 samples"),
    (lambda: F.score(np.full((5, 4096), np.nan, complex), FS, 4800.0),
     "not finite"),
    (lambda: F.score(np.ones((5, 4096), complex), FS, float("nan")),
     "not a cycle frequency"),
    (lambda: F.wiener_clean(np.ones(4096, complex), FS, nperseg=33),
     "must be even"),
    (lambda: F.rfi_mask_interp(np.ones(4096, complex), FS, pfa=2.0),
     "probability"),
    (lambda: F.wiener_clean(np.ones(4096, complex), -5.0), "positive number"),
    (lambda: F.wiener_clean(np.ones((2, 2, 4096), complex), FS), "shape"),
    (lambda: F.matched_parameters(np.ones(10, complex), FS), "at least 256"),
    (lambda: F.matched_filter(np.ones(4096, complex), FS,
                              params={"symbol_rate_hz": None}),
     "needs a symbol rate"),
])
def test_bad_inputs_are_refused_in_words(call, words):
    with pytest.raises(ValueError, match=words):
        call()
