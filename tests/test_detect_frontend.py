# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The front end: STFT, tiles in dB above the floor, the floor itself
(DETECTION_DESIGN §2; decisions D5, D7)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from atk_diffusion import profiles, sigmf
from atk_diffusion.dsp import floor as FL
from atk_diffusion.dsp import stft
from atk_diffusion.profiles import StftGeometry

FS = 2_400_000.0
NOISE_P = 2e-4          # total noise power (unit full scale)


def _noise(rng, n, power=NOISE_P):
    return ((rng.standard_normal(n) + 1j * rng.standard_normal(n))
            * math.sqrt(power / 2)).astype(np.complex64)


def _true_floor_db(power=NOISE_P, n=1024):
    """Per-bin floor of white noise for a Hann window: σ²·Σw²/(Σw)²."""
    w = stft.window_array("hann", n).astype(np.float64)
    return 10 * math.log10(power * np.sum(w * w) / np.sum(w) ** 2)


# ---------------------------------------------------------------------------
# geometry and the spectrogram
# ---------------------------------------------------------------------------
def test_rtl_tile_layout_is_measured_not_rounded_away():
    lay = stft.tile_layout(FS, StftGeometry())
    assert lay.pool == 5 and lay.frames == 2560 and lay.rows == 512
    assert lay.bin_hz == pytest.approx(2343.75)
    assert lay.row_period == pytest.approx(5 * 1024 / FS)
    assert lay.seconds == pytest.approx(1.0923, abs=1e-4)
    assert lay.overlap_rows == 128 and lay.step_rows == 384
    words = lay.words()
    assert "1.092 s per tile" in words and "asks 1 s" in words


def test_bad_geometry_is_refused_in_words():
    with pytest.raises(ValueError, match="window"):
        stft.tile_layout(FS, StftGeometry(window="not-a-window"))
    with pytest.raises(ValueError, match="tile_overlap"):
        stft.tile_layout(FS, StftGeometry(tile_overlap=1.0))


def test_a_tone_reads_its_power_in_the_right_bin():
    n, geom = 1024, StftGeometry()
    k = 37                                         # bins above the centre
    f = k * FS / n
    amp = 0.1
    x = (amp * np.exp(2j * np.pi * f * np.arange(n * 20) / FS)).astype(np.complex64)
    s_db, times = stft.spectrogram(x, FS, geom, t_start=5.0)
    assert s_db.shape == (20, n)
    peak = int(np.argmax(s_db[3]))
    assert peak == n // 2 + k                      # low -> high, centre at n//2
    assert s_db[3, peak] == pytest.approx(20 * math.log10(amp), abs=0.05)
    assert times[0] == 5.0 and times[1] == pytest.approx(5.0 + n / FS)
    freqs = stft.frame_freqs(FS, n, center_hz=100e6)
    assert freqs[peak] == pytest.approx(100e6 + f)


def test_pixels_and_time_frequency_are_exact_inverses(rng):
    x = _noise(rng, int(1.2 * FS))
    tile = next(iter(stft.tiles(x, FS, 162.4e6, StftGeometry(), t_start=10.0)))
    assert tile.f1 - tile.f0 == pytest.approx(FS)
    assert tile.freqs[tile.bins // 2] == pytest.approx(162.4e6)
    box = (12.25, 100.5, 40.0, 131.0)
    t0, t1, f_lo, f_hi = tile.pixels_to_tf(*box)
    assert t0 == pytest.approx(10.0 + 12.25 * tile.row_period)
    back = tile.tf_to_pixels(t0, t1, f_lo, f_hi)
    assert np.allclose(back, box)
    # clipping to the tile
    r0, b0, r1, b1 = tile.tf_to_pixels(0.0, 1e9, 0.0, 1e12)
    assert (r0, b0, r1, b1) == (0.0, 0.0, tile.rows, tile.bins)


def test_tile_builder_is_chunk_invariant(rng):
    """One big block or hundreds of small ones: identical tiles, bit for bit."""
    x = _noise(rng, int(2.6 * FS))
    x[600_000:700_000] += 0.05                       # a carrier burst at DC
    whole = list(stft.tiles(x, FS, 100e6, StftGeometry(), t_start=3.0))
    b = stft.TileBuilder(FS, 100e6, StftGeometry(), t_start=3.0)
    parts, pos = [], 0
    while pos < x.size:
        n = int(rng.integers(1, 60_000))
        parts += b.push(x[pos:pos + n])
        pos += n
    parts += b.flush()
    assert len(parts) == len(whole) >= 3
    for a, c in zip(whole, parts):
        assert np.array_equal(a.spec, c.spec)
        assert np.array_equal(a.abs_db, c.abs_db)
        assert np.array_equal(a.mean_above, c.mean_above)
        assert (a.t0, a.own_t0, a.own_t1, a.rows_valid, a.final) == \
            (c.t0, c.own_t0, c.own_t1, c.rows_valid, c.final)


def test_tiles_overlap_and_ownership_windows_tile_the_stream(rng):
    x = _noise(rng, int(3.3 * FS))
    ts = list(stft.tiles(x, FS, 0.0, StftGeometry()))
    lay = ts[0].layout
    assert ts[0].first and ts[-1].final and not ts[0].final
    assert ts[0].own_t0 == ts[0].t0 == 0.0
    for a, b in zip(ts, ts[1:]):
        assert b.t0 - a.t0 == pytest.approx(lay.step_rows * lay.row_period)
        assert a.t1 - b.t0 == pytest.approx(lay.overlap_seconds)     # they overlap
        assert b.own_t0 == pytest.approx(a.own_t1)                     # and abut
    last = ts[-1]
    assert last.own_t1 == pytest.approx(last.data_t1)
    assert last.data_t1 == pytest.approx(x.size / FS, abs=lay.row_period)
    assert np.all(last.spec[last.rows_valid:] == 0.0)                 # padded at the floor


def test_a_burst_on_a_tile_boundary_is_whole_in_one_tile(rng):
    x = _noise(rng, int(2.0 * FS))
    lay = stft.tile_layout(FS, StftGeometry())
    t_b = 1 * lay.step_rows * lay.row_period + 0.05     # inside the shared region
    n0 = int(t_b * FS)
    x[n0:n0 + int(0.1 * FS)] += 0.03 * np.exp(2j * np.pi * 300e3 * np.arange(int(0.1 * FS)) / FS)
    ts = list(stft.tiles(x, FS, 0.0, StftGeometry()))
    col = int(ts[0].f_to_bin(300e3))
    whole = []
    for t in ts:
        r0, r1 = int(t.t_to_row(t_b)) + 2, int(t.t_to_row(t_b + 0.1)) - 2
        if 0 <= r0 and r1 < t.rows:
            whole.append(t.index)
            assert np.all(t.spec[r0:r1, col] > 10)
    assert len(whole) >= 1


def test_with_spec_never_replaces_the_raw_measurements(rng):
    tile = next(iter(stft.tiles(_noise(rng, int(1.2 * FS)), FS, 0.0, StftGeometry())))
    t2 = tile.with_spec(np.zeros_like(tile.spec))
    assert np.all(t2.spec == 0) and t2.abs_db is tile.abs_db
    assert t2.mean_above is tile.mean_above
    with pytest.raises(ValueError, match="replacement"):
        tile.with_spec(np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# the floor
# ---------------------------------------------------------------------------
def _receiver_shape(nb=1024):
    """A receiver-like floor: flat with a gentle ripple and smooth (raised-
    cosine) 6 dB roll-offs at the band edges."""
    k = np.arange(nb)
    db = -70.0 + 0.4 * np.sin(k / 37.0)
    edge = 60
    tap = 0.5 * (1 - np.cos(np.pi * np.arange(edge) / edge))
    db[:edge] -= 6 * (1 - tap)
    db[-edge:] -= 6 * (1 - tap[::-1])
    return db


def _occupied_frames(rng, nf=2000, nb=1024):
    """Single-frame bin powers over the receiver shape, 30 % of the cells
    occupied: continuous channels (one wider than broadcast FM), bursty
    channels, wideband bursts, and weak busy channels, 8-25 dB up."""
    shape = _receiver_shape(nb)
    mu = 10 ** (shape / 10)
    f = (rng.exponential(1.0, (nf, nb)) * mu).astype(np.float32)
    clean = f.copy()
    occ = np.zeros((nf, nb), bool)

    def add(rows, cols, snr_db):
        s = 10 ** (snr_db / 10)
        sub = np.ix_(rows, cols)
        f[sub] = (mu[cols] * (s * rng.uniform(0.6, 1.4, (len(rows), len(cols)))
                              + rng.exponential(1, (len(rows), len(cols))))).astype(np.float32)
        occ[sub] = True

    allr = np.arange(nf)
    add(allr, np.arange(100, 104), 20)
    add(allr, np.arange(300, 320), 15)
    add(allr, np.arange(600, 690), 12)
    for c0 in range(150, 900, 75):
        on = np.flatnonzero((np.arange(nf) // 100 + c0) % 2 == 0)
        add(on, np.arange(c0, c0 + 12), rng.uniform(10, 25))
    for r0 in range(0, nf, 400):
        add(np.arange(r0, r0 + 100), np.arange(400, 560), 10)
    for c0 in range(700, 1000, 40):
        on = np.flatnonzero((np.arange(nf) // 150 + c0) % 3 != 0)
        add(on, np.arange(c0, c0 + 20), rng.uniform(8, 20))
    add(np.arange(0, nf, 3), np.arange(200, 300), 9)
    return f, clean, occ, shape


def test_floor_is_recovered_within_1_db_at_30_percent_occupancy(rng):
    f, clean, occ, shape = _occupied_frames(rng)
    assert 0.29 < occ.mean() < 0.35
    fl = FL.NoiseFloor.estimate(f)
    err = fl.floor_db - shape
    assert np.max(np.abs(err)) < 1.0, (np.argmax(np.abs(err)), np.max(np.abs(err)))
    assert np.median(np.abs(err)) < 0.15
    # a plain low percentile is what this replaces: it is badly lifted
    naive = FL._db(FL.quantile_floor(f, 0.1)) - shape
    assert np.max(naive) > 5.0
    # and on noise alone the estimate is unbiased
    noise_only = FL.NoiseFloor.estimate(clean).floor_db - shape
    assert abs(float(np.mean(noise_only))) < 0.05


def test_floor_from_a_terminated_capture_and_from_the_profile(tmp_path, rng):
    prof = profiles.new_profile("rtlsdr_2400000_cu8")
    x = _noise(rng, int(1.5 * FS))
    x[::2] += 0.004                                  # the receiver's own DC spike
    sigmf.write_pair(tmp_path / "term", x, FS, 100e6, datatype="cf32", hw="RTL-SDR",
                     extra_global={"atk:receiver_profile": prof.id})
    fl = FL.NoiseFloor.from_terminated(tmp_path / "term", profile=prof)
    assert "terminated" in fl.source and not fl.notes
    assert fl.level_db() == pytest.approx(_true_floor_db(), abs=0.15)
    dc = fl.floor_db[512] - np.median(fl.floor_db)
    assert dc > 3.0                                   # the spike is part of the floor
    # stored in the profile, levelled once against live data at +10 dB gain
    prof.impairments.update(fl.to_impairments())
    shape = FL.NoiseFloor.from_profile(prof)
    assert shape.needs_alignment
    live = FL._frames_lin(_noise(rng, int(0.5 * FS), NOISE_P * 10) + 0.004 * np.sqrt(10) * (np.arange(int(0.5 * FS)) % 2 == 0),
                          prof.stft, 512)
    moved = shape.align(live)
    assert moved == pytest.approx(10.0, abs=0.3) and not shape.needs_alignment
    assert shape.floor_db[512] - np.median(shape.floor_db) == pytest.approx(dc, abs=0.5)
    # a shape for another geometry is refused, in words
    prof.stft = StftGeometry(fft_size=2048, hop=2048)
    with pytest.raises(ValueError, match="re-measure the floor"):
        FL.NoiseFloor.from_profile(prof)
    assert FL.NoiseFloor.from_profile(profiles.new_profile("hackrf_2000000_ci8")) is None


def test_a_terminated_capture_with_bursts_is_named(rng):
    x = _noise(rng, int(1.2 * FS))
    for k in range(5):
        x[k * 400_000:k * 400_000 + 20_000] += 0.2
    fl = FL.NoiseFloor.from_terminated(x, fs=FS, geom=StftGeometry())
    assert fl.notes and "antenna" in fl.notes[0]


def test_capture_of_another_profile_is_refused(tmp_path, rng):
    sigmf.write_pair(tmp_path / "b", _noise(rng, 100_000), 4e6, 100e6,
                     datatype="ci16", hw="bladeRF x115")
    with pytest.raises(profiles.ProfileMismatch, match="bladeRF"):
        FL.NoiseFloor.from_terminated(tmp_path / "b",
                                      profile=profiles.new_profile("rtlsdr_2400000_cu8"))


def test_tracking_snaps_to_a_gain_step_and_gates_a_new_signal(rng):
    geom = StftGeometry()
    base = FL._frames_lin(_noise(rng, int(1.0 * FS)), geom, 2000)
    fl = FL.NoiseFloor.estimate(base)
    lvl = fl.level_db()
    # a 6 dB gain step: followed at once, and said so
    up = FL._frames_lin(_noise(rng, int(0.5 * FS), NOISE_P * 4), geom, 512)
    info = fl.update(up, duration_s=0.5)
    assert info["snapped"] and info["offset_db"] == pytest.approx(6.0, abs=0.3)
    assert fl.level_db() == pytest.approx(lvl + 6.0, abs=0.3)
    # a carrier that appears is a signal, not the floor rising: gated
    x = _noise(rng, int(0.5 * FS), NOISE_P * 4)
    x += (0.02 * np.exp(2j * np.pi * 300e3 * np.arange(x.size) / FS)).astype(np.complex64)
    col = 512 + int(round(300e3 / (FS / 1024)))
    before = fl.floor_db[col]
    for _ in range(20):
        info = fl.update(FL._frames_lin(x, geom, 512), duration_s=0.8)
    assert abs(fl.floor_db[col] - before) < 0.5
    assert not info["snapped"]


def test_tracking_keeps_a_measured_dc_spike(rng):
    geom = StftGeometry()
    x = _noise(rng, int(1.0 * FS))
    x[::2] += 0.004
    fl = FL.NoiseFloor.from_terminated(x, fs=FS, geom=geom)
    spike = fl.floor_db[512] - np.median(fl.floor_db)
    for _ in range(30):
        y = _noise(rng, int(0.3 * FS))
        y[::2] += 0.004
        fl.update(FL._frames_lin(y, geom, 512), duration_s=1.0)
    assert fl.floor_db[512] - np.median(fl.floor_db) == pytest.approx(spike, abs=0.6)


def test_a_tile_after_a_gain_step_says_the_floor_moved(rng):
    x = np.concatenate([_noise(rng, int(2.0 * FS)), _noise(rng, int(2.0 * FS), NOISE_P * 4)])
    ts = list(stft.tiles(x, FS, 0.0, StftGeometry()))
    flags = [(round(t.t0, 2), t.floor_ok, round(t.floor_offset_db, 1)) for t in ts]
    assert all(ok for t0, ok, _o in flags if t0 < 1.0), flags
    moved = [o for _t0, ok, o in flags if not ok]
    # the tile holding the step measures most of it (its new frames straddle it)
    assert moved and moved[0] > 4.0, flags
    # and the floor has settled on the new level by the last tile
    last = ts[-1]
    assert last.floor_ok
    assert np.median(last.abs_db - last.spec) == pytest.approx(_true_floor_db() + 6.0, abs=0.3)


def test_floor_checks_its_geometry(rng):
    fl = FL.NoiseFloor(np.zeros(1024))
    with pytest.raises(ValueError, match="1024 bins"):
        fl.above(np.zeros((4, 512)))
    with pytest.raises(ValueError, match="four frames"):
        FL.NoiseFloor.estimate(np.ones((2, 64)))
