# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The classical baselines (Wiener, median, wavelet) and the measuring stick
(power tiles, floor, CA-CFAR) — dsp.denoise_classical. Pure numpy/scipy:
these run in ATK's core environment."""

from __future__ import annotations

import math

import numpy as np
import pytest

from atk_diffusion.dsp import denoise_classical as C


def _noise(rng, n, power=1.0):
    return (math.sqrt(power / 2) * (rng.standard_normal(n)
                                    + 1j * rng.standard_normal(n))).astype(np.complex64)


# -- the representation --------------------------------------------------------
def test_stft_power_reads_the_noise_power_in_every_bin(rng):
    P = C.stft_power(_noise(rng, 64 * 400, power=0.25), 64)
    assert P.shape == (400, 64)
    assert abs(P.mean() - 0.25) < 0.01
    assert np.all(np.abs(P.mean(axis=0) / 0.25 - 1) < 0.25)


@pytest.mark.parametrize("k,mode", [(1, "mean"), (4, "mean"), (4, "max")])
def test_noise_tile_statistics_match_theory(rng, k, mode):
    x = _noise(rng, 64 * 4 * 600)
    D = C.db_above(C.power_tile(x, 64, pool=k, mode=mode), 1.0)
    mu, sd = C.logpower_noise_stats(k, mode)
    assert abs(D.mean() - mu) < 0.15 and abs(D.std() - sd) < 0.15
    if k == 1:
        assert abs(mu + 2.507) < 0.01 and abs(sd - 5.570) < 0.01


def test_floor_estimate_ignores_a_part_time_signal(rng):
    x = _noise(rng, 64 * 512)
    n = np.arange(x.size)
    burst = (n % (64 * 10)) < 64 * 2                      # on 20 % of frames
    x = x + (burst * 10.0 * np.exp(2j * np.pi * 0.25 * n)).astype(np.complex64)
    P = C.stft_power(x, 64)
    F = C.estimate_floor(P, q=0.25)
    sig_bin = 32 + 16
    assert abs(10 * math.log10(F[sig_bin])) < 1.5        # still the noise
    assert P[:, sig_bin].mean() > 50 * F[sig_bin]


# -- CFAR ----------------------------------------------------------------------
def test_cfar_alpha_is_the_classic_formula_at_k1():
    for n in (8, 16, 32):
        for pfa in (1e-2, 1e-4):
            assert math.isclose(C.cfar_alpha(pfa, n), n * (pfa ** (-1 / n) - 1),
                                rel_tol=1e-6)
    with pytest.raises(ValueError):
        C.cfar_alpha(0.0, 8)


@pytest.mark.parametrize("k,mode", [(1, "mean"), (4, "mean"), (4, "max")])
def test_cfar_realised_false_alarm_rate_is_nominal(rng, k, mode):
    """On independent cells the derived α gives the asked-for Pfa."""
    pfa = 1e-2
    E = rng.exponential(size=(2000, 128, k))
    P = E.mean(axis=2) if mode == "mean" else E.max(axis=2)
    got = C.ca_cfar(P, pfa, guard=2, train=8, k=k, mode=mode)[:, 16:-16].mean()
    assert 0.8 * pfa < got < 1.25 * pfa, got


def test_cfar_on_a_real_spectrogram_runs_a_little_hot(rng):
    """Hann correlates neighbouring bins, so training cells are not quite
    independent: measured, and bounded (the docstring says so)."""
    pfa = 1e-2
    P = C.stft_power(_noise(rng, 64 * 3000), 64)
    got = C.ca_cfar(P, pfa, guard=2, train=8)[:, 12:-12].mean()
    assert 0.7 * pfa < got < 2.0 * pfa, got


def test_cfar_boxes_find_a_rectangle(rng):
    """The guard must cover the signal's width, or its own bins land in the
    training cells and it masks itself — the CA-CFAR trap Bill's siga code
    warns about."""
    D = C.db_above(rng.exponential(size=(40, 64)), 1.0)
    D[10:20, 30:34] += 20.0
    found, boxes = C.cfar_detect_db(D, 1e-4, guard=3, train=8, min_bins=2)
    assert found
    big = max(boxes, key=lambda b: b.cells)
    assert (big.row0, big.row1) == (10, 20) and big.bin0 >= 29 and big.bin1 <= 35
    assert big.peak_ratio_db > 10


def test_run_statistic_requires_contiguous_bins():
    r = np.ones((3, 20))
    r[1, 5] = 50.0                 # one hot cell
    r[2, 10:13] = 8.0              # a run of three
    assert C.run_statistic(r, 1) == 50.0
    assert C.run_statistic(r, 3) == 8.0
    assert C.run_statistic(r, 4) == 1.0


# -- the denoisers ---------------------------------------------------------------
def _tile_with_rect(rng, rows=48, bins=64, level_db=6.0):
    clean = np.zeros((rows, bins))
    clean[16:32, 20:30] = 10 ** (level_db / 10)
    P = (1.0 + clean) * rng.exponential(size=(rows, bins))
    return C.db_above(P, 1.0), clean


def test_wiener_tile_smooths_noise_and_keeps_the_signal(rng):
    D, clean = _tile_with_rect(rng)
    out = C.wiener_tile(D, noise_lin=1.0)
    noise_region = (slice(0, 12), slice(40, 64))
    assert out[noise_region].std() < 0.3 * D[noise_region].std()
    assert out[16:32, 20:30].mean() > out[noise_region].mean() + 3.0


def test_median_and_wavelet_reduce_error_against_the_clean_tile(rng):
    D, clean = _tile_with_rect(rng, level_db=10.0)
    truth = 10 * np.log10(1.0 + clean)
    mu, _sd = C.logpower_noise_stats(1, "mean")
    raw_err = np.mean((D - mu - truth) ** 2)
    for name in ("median", "wavelet"):
        res = C.clean_tile(D, name)
        assert res.tier == "cleaned" and "CLEANED" in res.words
        out = res.out
        bias = np.median(out[:10, 40:]) - 0.0
        err = np.mean((out - bias - truth) ** 2)
        assert err < 0.6 * raw_err, (name, err, raw_err)


@pytest.mark.parametrize("wavelet", ["haar", "db2"])
def test_dwt_is_orthonormal_and_reconstructs_exactly(rng, wavelet):
    x = rng.standard_normal((32, 48))
    a, det = C.dwt2(x, wavelet, levels=3)
    energy = np.sum(a ** 2) + sum(np.sum(c ** 2) for band in det for c in band)
    assert math.isclose(energy, np.sum(x ** 2), rel_tol=1e-10)
    assert np.allclose(C.idwt2(a, det, wavelet), x, atol=1e-10)
    lo, hi = C.dwt1(rng.standard_normal(16), wavelet)
    assert lo.shape == (8,) and hi.shape == (8,)
    with pytest.raises(ValueError, match="even length"):
        C.dwt1(np.ones(7), wavelet)
    with pytest.raises(ValueError, match="unknown wavelet"):
        C.dwt1(np.ones(8), "sym8")


def test_db2_has_two_vanishing_moments(rng):
    """A linear ramp has no detail coefficients away from the wrap-around."""
    _a, d = C.dwt1(np.arange(64, dtype=float), "db2")
    assert np.max(np.abs(d[:-1])) < 1e-9


def test_bayes_shrink_kills_pure_noise_bands(rng):
    x = rng.standard_normal((64, 64))
    a, det = C.dwt2(x, "db2", 2)
    shrunk, sigma = C.bayes_shrink(det)
    assert abs(sigma - 1.0) < 0.15
    kept = sum(np.count_nonzero(c) for band in shrunk for c in band)
    total = sum(c.size for band in det for c in band)
    assert kept < 0.2 * total


def test_wavelet_tile_handles_odd_sizes(rng):
    D = rng.standard_normal((37, 45))
    assert C.wavelet_tile(D, levels=3).shape == (37, 45)


def test_wiener_iq_improves_snr_and_never_adds_energy(rng):
    n = 8192
    t = np.arange(n)
    s = (0.5 * np.exp(2j * np.pi * 0.05 * t)).astype(np.complex64)
    noise = _noise(rng, n, power=1.0)
    y = s + noise
    out = C.wiener_iq(y, noise_power=1.0, nfft=64)
    assert out.shape == y.shape and out.dtype == np.complex64
    snr_in = 10 * np.log10(np.sum(np.abs(s) ** 2) / np.sum(np.abs(y - s) ** 2))
    snr_out = 10 * np.log10(np.sum(np.abs(s) ** 2) / np.sum(np.abs(out - s) ** 2))
    assert snr_out > snr_in + 6.0
    assert np.sum(np.abs(out) ** 2) <= np.sum(np.abs(y) ** 2) * 1.01
    # with no noise estimate given, it measures one
    out2 = C.wiener_iq(y, nfft=64)
    assert np.isfinite(out2).all()


def test_unknown_method_is_refused_in_words(rng):
    with pytest.raises(ValueError, match="unknown classical method"):
        C.clean_tile(np.zeros((8, 8)), "bilateral")
