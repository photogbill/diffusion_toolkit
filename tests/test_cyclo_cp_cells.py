# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""How many cells share the channel — `probes.cp_timing_phases`, the
cell-tower survey (Bill, 2026-10-09): distinct, non-time-aligned CP-OFDM
transmitters as separate peaks of the prefix timing metric folded over the
symbol period, each with its own timing, strength and carrier offset; a
derived threshold whose false-alarm rate is checked on noise; and the
honest limit — time-synchronised cells count as one — in the words."""

from __future__ import annotations

import math

import numpy as np
import pytest

import helpers_cyclo as H
from atk_diffusion.cyclo import probes as P

FS = 1.92e6
LAG = 1 / 15_000
PERIOD = 1 / 14_000


def _cell(n, rng, cfo, shift=0):
    """An LTE-like cell (1.4 MHz class: 72 subcarriers, normal prefix)
    whose slots start `shift` samples late."""
    return np.roll(H.lte_like(FS, n + 2_000, rng, cfo=cfo), shift)[:n]


def _scene(seed, cells, seconds=0.5, snr_db=3.0):
    """cells: [(relative dB, timing shift in samples, cfo Hz)] — the first
    at `snr_db` in-band, the others relative to it."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    a = H.inband_amplitude(snr_db, 1.08e6, FS)
    x = H.noise(n, rng)
    for rel, shift, cfo in cells:
        x = x + a * 10 ** (rel / 20) * _cell(n, rng, cfo, shift)
    return x.astype(np.complex64)


def _circ_us(a_s, b_s, period=PERIOD):
    d = (a_s - b_s) % period
    return min(d, period - d) * 1e6


def test_two_cells_twenty_microseconds_apart_are_two():
    """Two FDD cells on one channel, their symbols 38 samples (19.8 µs)
    apart, the second 6 dB weaker, each with its own carrier offset."""
    x = _scene(1, [(0.0, 0, 1_200.0), (-6.0, 38, -2_500.0)])
    r = P.cp_timing_phases(x, FS, LAG)
    assert r["probe"] == "cp_timing_phases" and r["n_cells"] == 2
    a, b = r["cells"]
    assert _circ_us(a["timing_offset_s"], 0.0) < 1.0
    assert _circ_us(b["timing_offset_s"], 38 / FS) < 1.0
    assert b["relative_db"] == pytest.approx(-6.0, abs=1.0)
    assert a["relative_db"] == 0.0 and a["rho"] > b["rho"]
    assert a["cfo_hz"] == pytest.approx(1_200.0, abs=150.0)
    assert b["cfo_hz"] == pytest.approx(-2_500.0, abs=150.0)
    for c in r["cells"]:
        assert c["statistic"] > c["threshold"] == r["threshold"]
        assert c["p_value"] < r["pfa"]
    assert r["period_s"] == pytest.approx(PERIOD, rel=2e-6)
    assert r["period_source"].startswith("measured")
    assert r["prefix_samples"] == 9                     # 137.14 − 128
    assert r["min_separation_s"] == pytest.approx(9 / FS)
    assert r["words"].startswith("2 distinct CP-OFDM transmitters")
    assert "time-synchronised" in r["words"] and r["limit"] in r["words"]


def test_one_cell_is_one():
    x = _scene(2, [(0.0, 300, 500.0)])
    r = P.cp_timing_phases(x, FS, LAG, period_s=PERIOD)
    assert r["n_cells"] == 1
    assert _circ_us(r["cells"][0]["timing_offset_s"], 300 / FS) < 1.0
    assert r["cells"][0]["relative_db"] == 0.0


def test_time_aligned_cells_count_as_one_and_the_words_say_why():
    """TDD LTE and NR cells are time-synchronised: their prefixes coincide.
    The count is then one, and the limit is in the result's words."""
    x = _scene(3, [(0.0, 0, 1_000.0), (-6.0, 0, -2_000.0)])
    r = P.cp_timing_phases(x, FS, LAG)
    assert r["n_cells"] == 1
    assert "time-synchronised" in r["words"] and "counted as ONE" in r["words"]
    assert "FDD LTE cells are usually not time-aligned" in r["words"]


def test_cells_closer_than_one_prefix_are_one():
    x = _scene(4, [(0.0, 0, 1_000.0), (-3.0, 4, -2_000.0)])
    assert P.cp_timing_phases(x, FS, LAG)["n_cells"] == 1


def test_more_cells_than_asked_for_are_counted_not_hidden():
    x = _scene(5, [(0.0, 0, 800.0), (-3.0, 40, -1_500.0), (-4.0, 85, 2_500.0)],
               snr_db=6.0)
    r = P.cp_timing_phases(x, FS, LAG, max_cells=2)
    assert r["n_cells"] == 2 and r["more_above"] == 1
    assert "1 more peak above the threshold" in r["words"]
    full = P.cp_timing_phases(x, FS, LAG, max_cells=4)
    assert full["n_cells"] == 3 and full["more_above"] == 0
    got = [c["timing_offset_s"] for c in full["cells"]]
    for want in (0, 40, 85):
        assert min(_circ_us(g, want / FS) for g in got) < 1.0


def test_a_receiver_clock_30_ppm_off_is_measured_not_assumed():
    """The nominal period given, the receiver's clock 30 ppm fast: folded
    at the nominal period half a second would smear by ~29 samples; the
    period actually received is measured around the one given."""
    from scipy.signal import resample
    x = _scene(6, [(0.0, 0, 1_000.0), (-6.0, 38, -2_000.0)], seconds=0.5)
    y = resample(x, int(round(x.size * (1 + 30e-6)))).astype(np.complex64)
    r = P.cp_timing_phases(y, FS, LAG, period_s=PERIOD)
    assert r["period_source"].startswith("measured from the record around")
    assert r["period_s"] == pytest.approx(PERIOD * (1 + 30e-6), rel=3e-6)
    assert r["n_cells"] == 2
    dt = (r["cells"][1]["timing_offset_s"] - r["cells"][0]["timing_offset_s"])
    assert _circ_us(dt, 38 / FS * (1 + 30e-6)) < 1.0


def test_the_prefix_train_period_and_timing_are_read_off_the_bin_centre():
    """`_cp_period` (behind cp_probe) once refined the period with a
    parabola through the POWER of three FFT bins, which pins an unwindowed
    line to the bin centre: a clock 30 ppm off (0.2 bins in half a second)
    read as 1 ppm, and the timing took the line's phase at the bin centre
    with the cell's carrier-offset phase still in it (+8 samples at 1 kHz).
    Now: the period to a few ppm, the prefix start to two samples."""
    from scipy.signal import resample
    x = _scene(8, [(0.0, 300, 1_000.0)], seconds=0.5, snr_db=5.0)
    y = resample(x, int(round(x.size * (1 + 30e-6)))).astype(np.complex64)
    r = P.cp_probe(y, FS, LAG, period_hint_hz=14_000.0)
    assert r["period_detected"]
    assert r["symbol_period_s"] == pytest.approx(PERIOD * (1 + 30e-6), rel=4e-6)
    period = r["symbol_period_s"] * FS
    t = r["timing_offset_s"] * FS
    want = 300 * (1 + 30e-6) % period
    assert min(abs(t - want), period - abs(t - want)) < 3


def test_noise_alone_reports_no_cell_at_the_stated_false_alarm_rate():
    """The threshold is derived (order-statistic CFAR against the fold at
    reference lags), so the rate of 'one or more cells' on noise must be
    the stated pfa — checked here with the fold forced (period given)."""
    fs, pfa, trials = 0.96e6, 0.15, 100
    hits = 0
    for seed in range(trials):
        x = H.noise(int(0.05 * fs), np.random.default_rng(7_000 + seed))
        r = P.cp_timing_phases(x.astype(np.complex64), fs, LAG,
                               period_s=PERIOD, pfa=pfa)
        hits += r["n_cells"] > 0
    mu, sd = trials * pfa, math.sqrt(trials * pfa * (1 - pfa))
    assert mu - 3.5 * sd <= hits <= mu + 3.5 * sd, hits
    # without a period, noise shows none, and the words say so
    r = P.cp_timing_phases(H.noise(int(0.3 * FS), np.random.default_rng(1)),
                           FS, LAG)
    assert r["n_cells"] == 0 and r["period_source"] == "not found"
    assert "no OFDM symbol period" in r["words"]


def test_inputs_are_refused_in_words():
    rng = np.random.default_rng(0)
    x = H.noise(48_000, rng).astype(np.complex64)
    r = P.cp_timing_phases(x, 48_000.0, LAG)
    assert r["n_cells"] == 0 and "too short to fold" in r["words"]
    r = P.cp_timing_phases(x[:5_000], FS, LAG, period_s=PERIOD)
    assert r["n_cells"] == 0 and "too short to fold" in r["words"]
    with pytest.raises(ValueError, match="longer than the lag"):
        P.cp_timing_phases(H.noise(400_000, rng), FS, LAG, period_s=LAG / 2)
    with pytest.raises(ValueError, match="probability"):
        P.cp_timing_phases(x, FS, LAG, pfa=1.5)
    bad = H.noise(400_000, rng)
    bad[5] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        P.cp_timing_phases(bad, FS, LAG)
