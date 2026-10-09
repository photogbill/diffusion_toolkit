# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The receiver impairment model (plan §3.4): measure a synthetic receiver
whose impairments we set, recover them; apply them, re-measure, get them back;
store them in the profile with the write log."""

from __future__ import annotations

import math

import numpy as np
import pytest

from atk_diffusion import profiles
from atk_diffusion.dsp import impair

FS = 2_400_000.0


def _receiver(rng, n=1_048_576, floor_dbfs=-30.0, gain_db=0.5, phase_deg=3.0,
              dc=(0.012, -0.007), spur=(300_000.0, -52.0), datatype="cu8"):
    """A terminated RTL-like receiver with known impairments: a front-end
    roll-off (a gentle low-pass), one spur, I/Q imbalance, DC, 8-bit
    quantisation."""
    w = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    from scipy.signal import firwin, lfilter
    h = firwin(63, 0.8)                   # roll-off near the band edges
    w = lfilter(h, 1.0, w)
    w *= math.sqrt(10 ** (floor_dbfs / 10) / np.mean(np.abs(w) ** 2))
    nn = np.arange(n)
    a = 10 ** (spur[1] / 20)
    w = w + a * np.exp(1j * (2 * np.pi * spur[0] / FS * nn + 0.4))
    w = impair.apply_iq(w.astype(np.complex64), gain_db, phase_deg)
    w = w + np.complex64(complex(*dc))
    return impair.quantise(w, datatype)


def test_measure_recovers_known_imbalance_dc_spur_and_floor(rng):
    x = _receiver(rng)
    m = impair.measure_impairments(x, FS, "cu8")
    assert m["iq_gain_imbalance_db"] == pytest.approx(0.5, abs=0.03)
    assert m["iq_phase_imbalance_deg"] == pytest.approx(3.0, abs=0.15)
    assert m["dc_offset_i"] == pytest.approx(0.012, abs=1e-3)
    assert m["dc_offset_q"] == pytest.approx(-0.007, abs=1e-3)
    assert m["dc_spike_db"] > 10           # it stands well above one bin
    assert len(m["floor_db_per_bin"]) == 256
    # floor total is the noise we made (the spur and DC are taken out first)
    assert m["floor_mean_dbfs"] == pytest.approx(-30.0, abs=0.3)
    # the spur: one, where we put it, at the level we put it
    near = [s for s in m["spurs"] if abs(s["offset_hz"] - 300_000) < 100]
    assert len(near) == 1, m["spurs"]
    assert near[0]["level_db"] == pytest.approx(-52.0, abs=0.5)
    # its image (at -300 kHz, from the imbalance) is NOT reported as a spur
    assert not [s for s in m["spurs"] if abs(s["offset_hz"] + 300_000) < 2000]
    # the roll-off shows in the floor shape: edges below the middle
    fl = np.array(m["floor_db_per_bin"])
    assert fl[:8].mean() < fl[100:156].mean() - 3
    assert m["image_rejection_db"] == pytest.approx(
        impair.image_rejection_db(10 ** (0.5 / 20) - 1, 3.0), abs=0.5)
    assert 0 < m["enob_est"] <= 8
    assert m["tier"] == "measured" and m["n_samples"] == x.size


def test_noise_alone_reports_no_spurs(rng):
    """The spur threshold is derived from the noise statistics: pure noise
    should almost never fire (false-alarm chance 1e-3 per capture)."""
    n = 524_288
    w = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 0.01
    m = impair.measure_impairments(w.astype(np.complex64), FS, "cf32")
    assert m["spurs"] == []
    assert abs(m["iq_gain_imbalance_db"]) < 0.05
    assert abs(m["iq_phase_imbalance_deg"]) < 0.2


def test_apply_then_measure_gives_the_same_receiver(rng):
    m1 = impair.measure_impairments(_receiver(rng), FS, "cu8")
    n = 1_048_576
    w = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    w *= math.sqrt(impair.noise_power(m1))
    y = impair.apply_impairments(w.astype(np.complex64), FS, m1, rng,
                                 datatype="cu8")
    m2 = impair.measure_impairments(y, FS, "cu8")
    assert m2["floor_mean_dbfs"] == pytest.approx(m1["floor_mean_dbfs"], abs=0.3)
    diff = np.array(m2["floor_db_per_bin"]) - np.array(m1["floor_db_per_bin"])
    assert np.median(np.abs(diff)) < 0.5        # the shape survives
    assert m2["iq_gain_imbalance_db"] == pytest.approx(
        m1["iq_gain_imbalance_db"], abs=0.03)
    assert m2["iq_phase_imbalance_deg"] == pytest.approx(
        m1["iq_phase_imbalance_deg"], abs=0.15)
    assert m2["dc_offset_i"] == pytest.approx(m1["dc_offset_i"], abs=1e-3)
    assert m2["dc_offset_q"] == pytest.approx(m1["dc_offset_q"], abs=1e-3)
    s1 = [s for s in m1["spurs"] if abs(s["offset_hz"] - 300_000) < 100][0]
    s2 = [s for s in m2["spurs"] if abs(s["offset_hz"] - 300_000) < 100][0]
    assert s2["level_db"] == pytest.approx(s1["level_db"], abs=0.5)


def test_quantisation_goes_through_the_file_format():
    x = np.array([0.5 + 0.25j, 2.0 - 2.0j], dtype=np.complex64)
    y = impair.quantise(x, "ci8")
    assert y[0] == pytest.approx(0.5 + 0.25j, abs=1 / 128)
    assert y[1].real == pytest.approx(127 / 128) and y[1].imag == -1.0  # clipped
    # a 12-bit bladeRF in a ci16 file is quantised at Q11
    z = impair.quantise(np.array([1e-4 + 0j], np.complex64), "ci16", 12)
    assert z[0] == 0                              # below half an LSB of 2048


def test_refusals_are_sentences(rng):
    with pytest.raises(impair.ImpairmentError, match="too short"):
        impair.measure_impairments(np.zeros(100, np.complex64), FS, "cu8")
    with pytest.raises(impair.ImpairmentError, match="rails"):
        impair.measure_impairments(np.full(8192, 1 + 1j, np.complex64), FS, "cu8")


def test_store_writes_the_profile_and_the_log(rf, rng):
    m = impair.measure_impairments(_receiver(rng, n=262_144), FS, "cu8")
    path = impair.store(rf, "rtlsdr_2400000_cu8", m, device_serial="00000001",
                        firmware="r820t2")
    prof = profiles.load_profile(rf, "rtlsdr_2400000_cu8")
    assert prof.impairments["floor_mean_dbfs"] == m["floor_mean_dbfs"]
    # filed beside dsp.floor's key, not over it (see impair's docstring)
    assert prof.impairments["floor_shape_db_256"] == m["floor_db_per_bin"]
    assert "floor_db_per_bin" not in prof.impairments
    assert impair.is_measured(prof.impairments)
    assert prof.device_serial == "00000001" and prof.firmware == "r820t2"
    assert any("re-measure" in n for n in prof.notes)
    assert rf.verify(path)[0]
    assert "measured" in impair.describe(prof.impairments)
    # a measurement at another rate is refused in words
    with pytest.raises(profiles.ProfileMismatch, match="own rate"):
        impair.store(rf, "rtlsdr_2048000_cu8", m)


def test_measure_sigmf_reads_a_terminated_capture(tmp_path, rng):
    from atk_diffusion import sigmf
    x = _receiver(rng, n=300_000)
    sigmf.write_pair(tmp_path / "term", x, FS, 100e6, datatype="cu8",
                     hw="RTL-SDR v4")
    m = impair.measure_sigmf(tmp_path / "term", skip_seconds=0.0)
    assert m["profile"] == "rtlsdr_2400000_cu8"
    assert m["source_capture"] == "term"
    assert m["iq_phase_imbalance_deg"] == pytest.approx(3.0, abs=0.3)


def test_store_keeps_dsp_floors_own_floor_and_noisefloor_still_loads(rf, rng):
    """Integration with dsp.floor (another engineer's module): its
    NoiseFloor owns profiles' `floor_db_per_bin` at the STFT geometry; a
    measurement filed by impair.store must neither overwrite it nor make
    NoiseFloor.from_profile refuse the profile."""
    try:
        from atk_diffusion.dsp.floor import NoiseFloor
    except ImportError as e:
        pytest.skip(f"dsp.floor is not importable yet ({e})")
    pid = "rtlsdr_2400000_cu8"
    prof = profiles.load_profile(rf, pid)
    x = _receiver(rng, n=262_144)
    nf = NoiseFloor.from_terminated(x, FS, prof.stft)
    prof.impairments = dict(nf.to_impairments())
    profiles.save_profile(rf, prof)
    m = impair.measure_impairments(x, FS, "cu8")
    impair.store(rf, pid, m)
    prof = profiles.load_profile(rf, pid)
    assert len(prof.impairments["floor_db_per_bin"]) == prof.stft.fft_size
    assert len(prof.impairments["floor_shape_db_256"]) == 256
    back = NoiseFloor.from_profile(prof)
    assert back is not None and back.bins == prof.stft.fft_size
    # apply uses this module's 256-bin shape, not dsp.floor's
    assert impair.floor_shape(prof.impairments) == prof.impairments["floor_shape_db_256"]
