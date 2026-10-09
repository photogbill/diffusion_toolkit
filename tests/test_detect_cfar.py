# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The energy proposer: CA-CFAR with a derived threshold, measured on noise,
and boxes with the right edges (DETECTION_DESIGN §3; plan §7)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from atk_diffusion.dsp import cfar, stft
from atk_diffusion.dsp.stft import Tile
from atk_diffusion.profiles import StftGeometry

FS = 2_400_000.0
NOISE_P = 2e-4
GEOM = StftGeometry()


def _noise(rng, n, power=NOISE_P):
    return ((rng.standard_normal(n) + 1j * rng.standard_normal(n))
            * math.sqrt(power / 2)).astype(np.complex64)


def _add(x, rng, t0, t1, f, bw, snr_db):
    """A band-limited Gaussian signal, flat over exactly |f - f0| < bw/2,
    with in-band SNR `snr_db` against the noise (the per-bin SNR)."""
    n0, n1 = int(round(t0 * FS)), int(round(t1 * FS))
    n = n1 - n0
    spec = np.fft.fft(rng.standard_normal(n) + 1j * rng.standard_normal(n))
    spec[np.abs(np.fft.fftfreq(n, 1 / FS)) > bw / 2] = 0
    s = np.fft.ifft(spec)
    s *= math.sqrt(10 ** (snr_db / 10) * NOISE_P * bw / FS / np.mean(np.abs(s) ** 2))
    x[n0:n1] += (s * np.exp(2j * np.pi * f * np.arange(n0, n1) / FS)).astype(np.complex64)


def _tiles(x, center=0.0):
    return list(stft.tiles(x, FS, center, GEOM))


# ---------------------------------------------------------------------------
# the threshold, derived
# ---------------------------------------------------------------------------
def test_alpha_reduces_to_the_textbook_formula_for_one_frame():
    for pfa, n in ((1e-3, 32), (1e-4, 17.5), (1e-6, 8)):
        assert cfar.cfar_alpha(pfa, n, 1) == pytest.approx(n * (pfa ** (-1 / n) - 1))


def test_max_of_p_formula_matches_simulation(rng):
    """Pfa(α) for max-of-P cells over N training cells, against a direct
    Monte Carlo of independent exponentials."""
    p, n, pfa = 5, 16, 2e-3
    a = cfar.cfar_alpha(pfa, n, p)
    assert cfar.pfa_of_alpha(a, n, p) == pytest.approx(pfa, rel=1e-6)
    trials = 150_000
    cut = rng.exponential(size=(trials, p)).max(axis=1)
    train = rng.exponential(size=(trials, n, p)).max(axis=2).mean(axis=1)
    measured = float(np.mean(cut > a * train))
    assert measured == pytest.approx(pfa, rel=0.3)


def test_the_window_correlates_neighbouring_bins():
    bin_r2, frame_r2, pool_eff = cfar.window_stats("hann", 1024, 1024, 5)
    assert bin_r2[0] == pytest.approx(4 / 9, abs=1e-3)       # (2/3)² for Hann
    assert bin_r2[1] == pytest.approx(1 / 36, abs=1e-3)
    assert frame_r2 == () and pool_eff == 5.0                # hop = fft: frames independent
    _b, fr2, pe = cfar.window_stats("hann", 1024, 512, 5)
    assert fr2 and pe < 5.0                                   # 50 % overlap: fewer
    assert cfar.window_stats("boxcar", 64, 64, 1)[0] == ()    # rectangular: independent


@pytest.mark.parametrize("pfa", [1e-3, 1e-4])
def test_measured_false_alarm_rate_is_the_stated_one(rng, pfa):
    """On noise through the real STFT and pooling, per cell: CA-CFAR along
    frequency, 2D, and the floor-referenced test (no margin) each within a
    factor of 2 of the stated Pfa."""
    ts = list(stft.tiles(_noise(rng, int(1.1 * FS)), FS, 0.0, GEOM, final=False))
    lay = ts[0].layout
    kw = dict(pool=lay.pool, window=lay.window, fft_size=lay.fft_size, hop=lay.hop)
    cells = hits1 = hits2 = hitsf = 0
    for t in ts:
        m1 = cfar.ca_cfar(t.spec, pfa, **kw)
        m2 = cfar.ca_cfar(t.spec, pfa, two_d=True, **kw)
        _b, _f, pe = cfar.window_stats(lay.window, lay.fft_size, lay.hop, lay.pool)
        mf = cfar.floor_mask(t.spec, pfa, pe, margin_db=0.0)
        inner = (slice(6, -6), slice(18, -18))
        cells += t.spec[inner].size
        hits1 += int(m1[inner].sum())
        hits2 += int(m2[inner].sum())
        hitsf += int(mf[inner].sum())
    for hits in (hits1, hits2, hitsf):
        assert 0.5 * pfa < hits / cells < 2.0 * pfa, (hits, cells)


def test_the_integrated_layer_holds_its_rate_too(rng):
    ts = list(stft.tiles(_noise(rng, int(1.95 * FS)), FS, 0.0, GEOM, final=False))
    lay = ts[0].layout
    _b, _f, pe = cfar.window_stats(lay.window, lay.fft_size, lay.hop, lay.pool)
    pfa, hits, fl, cells = 1e-3, 0, 0, 0
    for t in ts:
        blocks = t.mean_above.astype(np.float64).reshape(64, 8, -1).mean(axis=1)
        m = cfar.ca_cfar(blocks, pfa, pool=lay.pool, window=lay.window,
                         fft_size=lay.fft_size, hop=lay.hop, cell="mean",
                         n_avg=8 * pe, linear=True)
        hits += int(m[:, 18:-18].sum())
        fl += int((blocks[:, 18:-18] > cfar.mean_floor_threshold(pfa, 8 * pe)).sum())
        cells += blocks[:, 18:-18].size
    assert 0.5 * pfa < hits / cells < 2.0 * pfa
    assert 0.5 * pfa < fl / cells < 2.0 * pfa


def test_noise_alone_gives_no_boxes(rng):
    info = {}
    n = sum(len(cfar.energy_proposer(t, pfa=1e-4, info=info))
            for t in _tiles(_noise(rng, int(2.0 * FS))))
    assert n == 0, info


# ---------------------------------------------------------------------------
# boxes
# ---------------------------------------------------------------------------
BURSTS = [  # t0, t1, offset Hz, bandwidth Hz, in-band SNR dB
    (0.20, 0.25, 200e3, 25e3, 10.0),
    (0.40, 0.46, -300e3, 12.5e3, 10.0),
    (0.62, 0.64, 600e3, 100e3, 12.0),
    (0.70, 0.90, -700e3, 50e3, 8.0),
]


def test_bursts_at_moderate_snr_are_boxed_with_correct_edges(rng):
    x = _noise(rng, int(1.0 * FS))
    for b in BURSTS:
        _add(x, rng, *b)
    tile = _tiles(x, center=162.4e6)[0]
    dets = cfar.energy_proposer(tile, pfa=1e-4)
    assert len(dets) == len(BURSTS), [(d.t0, d.center_hz) for d in dets]
    rp, bh = tile.row_period, tile.bin_hz
    for t0, t1, f, bw, snr in BURSTS:
        d = min(dets, key=lambda d: abs(d.center_hz - (162.4e6 + f)))
        assert d.sources == ("energy",) and d.family == "unknown"
        assert d.state == "proposed" and d.profile == tile.profile
        assert abs(d.t0 - t0) <= rp + 1e-9 and abs(d.t1 - t1) <= rp + 1e-9
        assert abs(d.f_lo - (162.4e6 + f - bw / 2)) <= 1.5 * bh
        assert abs(d.f_hi - (162.4e6 + f + bw / 2)) <= 1.5 * bh
        assert d.snr_db == pytest.approx(snr, abs=2.0)
        assert d.measurements["peak_db"] > d.snr_db


def test_a_wide_signal_is_found_inside_as_well_as_at_its_edges(rng):
    """Wider than the CFAR's training window (Bill's caveat): the floor-
    referenced test sees its interior."""
    x = _noise(rng, int(1.0 * FS))
    _add(x, rng, 0.05, 0.95, -400e3, 300e3, 8.0)
    tile = _tiles(x)[0]
    dets = cfar.energy_proposer(tile, pfa=1e-4)
    big = max(dets, key=lambda d: d.bw_hz * d.duration_s)
    assert big.bw_hz > 0.9 * 300e3 and big.duration_s > 0.8
    # CFAR alone: its training cells sit inside the signal (it does not even
    # reliably see the edges at 8 dB)
    no_floor = cfar.energy_proposer(tile, pfa=1e-4, floor_test=False, integrate_rows=0)
    assert max((d.bw_hz * d.duration_s for d in no_floor), default=0.0) < 0.5 * 300e3 * 0.9


def test_a_weak_continuous_signal_is_one_box_not_shards(rng):
    x = _noise(rng, int(1.0 * FS))
    _add(x, rng, 0.0, 1.0, 300e3, 12.5e3, 4.0)
    tile = _tiles(x)[0]
    dets = [d for d in cfar.energy_proposer(tile, pfa=1e-4)
            if abs(d.center_hz - 300e3) < 20e3]
    assert len(dets) == 1 and dets[0].duration_s > 0.9
    shards = [d for d in cfar.energy_proposer(tile, pfa=1e-4, integrate_rows=0)
              if abs(d.center_hz - 300e3) < 20e3]
    assert len(shards) > 3                     # what the integrated layer prevents


def test_the_floor_test_stands_down_when_the_floor_moved(rng):
    tile = _tiles(_noise(rng, int(1.2 * FS)))[0]
    tile.floor_ok, tile.floor_offset_db = False, 4.2
    info = {}
    cfar.energy_proposer(tile, info=info)
    assert info["floor_test"].startswith("stood down") and "+4.2 dB" in info["floor_test"]


def test_too_many_boxes_means_the_threshold_is_in_the_noise(rng):
    spec = np.zeros((GEOM.tile_rows, GEOM.fft_size), np.float32)
    spec[::6, ::8] = 30.0
    spec[1::6, ::8] = 30.0
    spec[2::6, ::8] = 30.0
    tile = Tile.from_spec(spec, FS, 0.0, GEOM)
    info = {}
    dets = cfar.energy_proposer(tile, max_boxes=50, info=info)
    assert len(dets) == 50 and info["capped"] and info["components"] > 50


def test_min_duration_and_size_filters(rng):
    spec = np.zeros((GEOM.tile_rows, GEOM.fft_size), np.float32)
    spec[100:102, 300] = 40.0          # 2 cells: below min_cells
    spec[200:240, 500:504] = 30.0       # 40 rows: 85 ms
    tile = Tile.from_spec(spec, FS, 0.0, GEOM, profile="rtlsdr_2400000_cu8")
    dets = cfar.energy_proposer(tile)
    assert len(dets) == 1 and dets[0].profile == "rtlsdr_2400000_cu8"
    assert cfar.energy_proposer(tile, min_duration_s=0.1) == []
    assert cfar.energy_proposer(tile, min_bins=5) == []
