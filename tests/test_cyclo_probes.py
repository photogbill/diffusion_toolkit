# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The targeted probes (DETECTION_DESIGN §3, §4.1; ATK FUTURE_PLANS
2026-09-29). Two kinds of test: the threshold is DERIVED, so the empirical
false-alarm rate on noise must match the stated Pfa; and the probes find
what they are for — including Bill's cell-tower case, an LTE-like OFDM
signal whose spectrum is below the noise floor, found by its cyclic prefix
in about a second."""

from __future__ import annotations

import math

import numpy as np
import pytest

import helpers_cyclo as H
from atk_diffusion.cyclo import probes as P

FS = 48_000.0


def _binomial_ok(hits: int, trials: int, p: float, z: float = 3.5) -> bool:
    """hits is within z standard deviations of trials·p (one-sided above,
    and not absurdly low)."""
    mu = trials * p
    sd = math.sqrt(trials * p * (1 - p))
    return (hits <= mu + z * sd) and (hits >= max(0.0, mu - z * sd) - 1)


def _coloured(n, rng, cutoff=7800.0):
    from scipy.signal import firwin, lfilter
    return lfilter(firwin(161, cutoff, fs=FS), 1, H.noise(n, rng)).astype(
        np.complex64)


# ---------------------------------------------------------------------------
# the derived null
# ---------------------------------------------------------------------------
def test_os_cfar_threshold_matches_simulation():
    rng = np.random.default_rng(0)
    K, k, trials = 32, 16, 120_000
    for L in (1, 3):
        T = P.os_threshold(1e-2, K, k, L)
        X = rng.exponential(size=(trials, L))
        Y = rng.exponential(size=(trials, L, K))
        med = np.sort(Y, axis=2)[:, :, k - 1]
        emp = float(np.mean((X / med).sum(axis=1) > T))
        assert emp == pytest.approx(1e-2, rel=0.15), (L, emp)
        assert P.os_pvalue(T, K, k, L) == pytest.approx(1e-2, rel=0.05)


def test_sidak_and_family_p_are_inverse():
    p = P.sidak(1e-3, 50)
    assert P.family_p(p, 50) == pytest.approx(1e-3, rel=1e-6)


def test_zoom_dft_is_exact_for_a_spectral_line(rng):
    n = 30_011
    t = np.arange(n) / FS
    y = 0.7 * np.exp(2j * np.pi * 4801.3 * t) + H.noise(n, rng, 1e-6)
    offs = np.array([-2.0, 0.0, 1.3, 2.6])
    z = P.zoom_dft(y, FS, 4800.0, offs)
    direct = np.array([np.sum(y * np.exp(-2j * np.pi * (4800.0 + o) * t))
                       for o in offs])
    assert np.allclose(z, direct, rtol=1e-3, atol=1e-2 * np.abs(direct).max())


# ---------------------------------------------------------------------------
# empirical false-alarm rates on noise (the claim the thresholds make)
# ---------------------------------------------------------------------------
def test_symbol_rate_line_false_alarm_rate_on_coloured_noise():
    hits, trials = 0, 200
    for seed in range(trials):
        x = _coloured(12_000, np.random.default_rng(1000 + seed))
        r = P.symbol_rate_line(x, FS, [4800.0, 2400.0], pfa=0.05,
                               bandwidth_hz=8300.0, family="fsk")
        hits += r["detected"]
    assert _binomial_ok(hits, trials, 0.05), hits


def test_symbol_rate_line_false_alarm_rate_psk_lags():
    hits, trials = 0, 200
    for seed in range(trials):
        x = _coloured(12_000, np.random.default_rng(5000 + seed))
        r = P.symbol_rate_line(x, FS, [4800.0], pfa=0.05,
                               bandwidth_hz=6480.0, family="psk_qam")
        hits += r["detected"]
    assert _binomial_ok(hits, trials, 0.05), hits


def test_cp_probe_false_alarm_rate():
    hits, trials = 0, 200
    for seed in range(trials):
        x = H.noise(20_000, np.random.default_rng(9000 + seed))
        hits += P.cp_probe(x, 1.92e6, 1 / 15_000, pfa=0.05)["detected"]
    assert _binomial_ok(hits, trials, 0.05), hits


def test_carrier_conj_false_alarm_rate():
    hits, trials = 0, 150
    for seed in range(trials):
        x = _coloured(8192, np.random.default_rng(12_000 + seed))
        hits += P.carrier_conj(x, FS, pfa=0.05)["detected"]
    assert _binomial_ok(hits, trials, 0.05), hits


def test_independent_lags_are_independent_on_this_noise(rng):
    x = P.bandlimit(_coloured(48_000, rng), FS, 4150.0)
    r = P.autocorrelation(x, 200)
    lags = P.independent_lags(FS, 4800.0, r, family="psk_qam")
    assert lags[0] == 0
    for i, a in enumerate(lags):
        for b in lags[i + 1:]:
            assert P.lag_product_correlation(r, b - a, FS, 4800.0) < 0.25


# ---------------------------------------------------------------------------
# what the probes are for
# ---------------------------------------------------------------------------
def test_cp_probe_finds_an_lte_like_cell_below_the_floor():
    """Bill's cell-tower case. 15 kHz-SCS OFDM (72 QPSK subcarriers, normal
    CP) whose spectrum is a TENTH of the noise floor's (−10 dB in-band),
    one second at 1.92 MS/s: the cyclic prefix finds it, and the carrier
    offset and symbol period come with it."""
    fs = 1.92e6
    rng = np.random.default_rng(10)
    n = int(fs)
    s = H.lte_like(fs, n, rng, cfo=1500.0)
    x = (H.inband_amplitude(-10.0, 1.08e6, fs) * s + H.noise(n, rng)
         ).astype(np.complex64)
    # its energy is below the floor in every bin it occupies
    spec = np.abs(np.fft.fft(x[:65536])) ** 2
    assert np.median(spec) > 0           # (the floor dominates every bin)
    r = P.cp_probe(x, fs, 1 / 15_000, period_hint_hz=14_000.0)
    assert r["detected"], r["words"]
    assert r["statistic"] > r["threshold"]
    assert r["integration_s"] == pytest.approx(1.0)
    assert abs(r["cfo_hz"] - 1500.0) < 4 * r["cfo_se_hz"] + 100.0
    assert r["symbol_period_s"] == pytest.approx(1 / 14_000, rel=0.005)
    assert "OFDM" in r["words"]


def test_cp_probe_timing_prefix_and_snr_at_good_snr():
    fs = 1.92e6
    rng = np.random.default_rng(3)
    n = int(0.5 * fs)
    s = np.roll(H.lte_like(fs, n, rng), 300)
    x = (H.inband_amplitude(3.0, 1.08e6, fs) * s + H.noise(n, rng))
    r = P.cp_probe(x, fs, 1 / 15_000)
    assert r["detected"] and r["period_detected"]
    assert r["cp_fraction"] == pytest.approx(64 / 960, rel=0.1)
    assert r["snr_db"] is not None
    true_snr = 10 * np.log10(10 ** 0.3 * 1.08e6 / fs)
    assert r["snr_db"] == pytest.approx(true_snr, abs=2.0)
    # the slot's first prefix starts at sample 300 (mod the symbol period)
    period = r["symbol_period_s"] * fs
    t = r["timing_offset_s"] * fs
    d = min(abs(t - 300 % period), period - abs(t - 300 % period))
    assert d < 12, (t, period)


def test_cp_probe_refuses_a_lag_too_short_for_the_rate():
    r = P.cp_probe(H.noise(48_000, np.random.default_rng(0)), FS, 1 / 15_000)
    assert not r["detected"] and "too short" in r["words"]


def test_symbol_rate_line_finds_qpsk_below_the_floor():
    rng = np.random.default_rng(601)
    n = int(4 * FS)
    s = H.linmod("qpsk", FS, 4800.0, n, rng, beta=0.35)
    x = _coloured_sum(s, -8.0, 6480.0, rng)
    r = P.symbol_rate_line(x, FS, [4800.0, 2400.0], bandwidth_hz=6480.0,
                           family="psk_qam")
    assert r["detected"] and r["best_rate_hz"] == 4800.0
    assert r["alpha_hz"] == pytest.approx(4800.0, abs=0.5)


def _coloured_sum(s, snr_db, occ, rng):
    from scipy.signal import firwin, lfilter
    x = H.inband_amplitude(snr_db, occ, FS) * s + H.noise(s.size, rng)
    return lfilter(firwin(161, 7800, fs=FS), 1, x).astype(np.complex64)


def test_a_c4fm_like_4fsk_is_found_at_the_floor_in_seconds():
    """P25/DMR-class 4FSK has a weak line (stated in the module): 0 dB
    in-band — its spectrum level with the floor — in five seconds."""
    rng = np.random.default_rng(705)
    n = int(5 * FS)
    x = _coloured_sum(H.fsk4(FS, 4800.0, n, rng), 0.0, 7000.0, rng)
    r = P.symbol_rate_line(x, FS, [4800.0], bandwidth_hz=8300.0,
                           family="fsk")
    assert r["detected"], r["words"]
    assert r["per_rate"][0]["lags"] and 0 not in r["per_rate"][0]["lags"]


def test_the_statistic_grows_with_integration_time():
    """The honest lever: the mean statistic grows in proportion to T."""
    means = []
    for T in (1.0, 4.0):
        st = []
        for seed in range(4):
            rng = np.random.default_rng(800 + seed)
            n = int(T * FS)
            x = _coloured_sum(H.linmod("qpsk", FS, 4800.0, n, rng), -10.0,
                              6480.0, rng)
            st.append(P.symbol_rate_line(x, FS, [4800.0], bandwidth_hz=6480.0,
                                         family="psk_qam")["statistic"])
        means.append(np.mean(st))
    assert means[1] > 2.0 * means[0], means


def test_carrier_conj_present_for_bpsk_and_absent_as_a_finding_for_qpsk(rng):
    n = 48_000
    b = H.linmod("bpsk", FS, 2400.0, n, rng, fc=600.0) + H.noise(n, rng)
    r = P.carrier_conj(b, FS)
    assert r["detected"] and r["carrier_offset_hz"] == pytest.approx(600.0,
                                                                     abs=1.0)
    q = H.linmod("qpsk", FS, 2400.0, n, rng, fc=600.0) + H.noise(n, rng)
    r = P.carrier_conj(q, FS)
    assert not r["detected"] and "FINDING" in r["words"]
    r4 = P.carrier_conj(q, FS, power=4)
    assert r4["detected"] and r4["carrier_offset_hz"] == pytest.approx(
        600.0, abs=1.0)


def test_every_probe_returns_the_contract_keys(rng):
    x = H.noise(48_000, rng)
    for r in (P.symbol_rate_line(x, FS, [4800.0]),
              P.cp_probe(H.noise(60_000, rng), 1.92e6, 1 / 15_000),
              P.carrier_conj(x, FS)):
        for k in ("statistic", "threshold", "detected", "integration_s",
                  "pfa", "words"):
            assert k in r, (r.get("probe"), k)
        assert "estimates" in r or "per_rate" in r


def test_multichannel_input_is_vectorised(rng):
    X = np.stack([H.noise(24_000, rng) for _ in range(4)])
    out = P.symbol_rate_line(X, FS, [4800.0])
    assert isinstance(out, list) and len(out) == 4
