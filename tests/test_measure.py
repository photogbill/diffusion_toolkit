# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Classical, checkable measurements (DETECTION_DESIGN §4) — each against a
synthetic signal whose truth is known, each tier "measured" with a method
sentence."""

from __future__ import annotations

import numpy as np
import pytest

import helpers_cyclo as H
from atk_diffusion.dsp import measure as M

FS = 48_000.0


def _qpsk(rng, snr_db=10.0, fc=1000.0, n=96_000):
    s = H.linmod("qpsk", FS, 4800.0, n, rng, beta=0.35, fc=fc)
    return H.inband_amplitude(snr_db, 6480.0, FS) * s + H.noise(n, rng)


def test_occupied_bandwidth_and_centre(rng):
    ob = M.occupied_bandwidth(_qpsk(rng), FS)
    # RRC 0.35 at 4800 Bd: 99 % of the power within ~(1 + 0.35)·4800
    assert 4500 < ob["value_hz"] < 6600
    assert ob["centre_hz"] == pytest.approx(1000.0, abs=150.0)
    assert ob["tier"] == "measured" and "99%" in ob["method"]


def test_snr_above_floor_is_right_and_knows_its_wall(rng):
    s = M.snr_above_floor(_qpsk(rng, snr_db=10.0), FS)
    assert s["measurable"] and s["snr_db"] == pytest.approx(10.0, abs=1.2)
    # far below the floor the energy cannot be told from the floor's own
    # uncertainty: reported as not measurable, never as a reading
    weak = M.snr_above_floor(_qpsk(rng, snr_db=-25.0, n=24_000), FS,
                             band=(0.0, 2000.0))
    assert not weak["measurable"]
    assert "below what energy can measure" in weak["words"]


def test_noise_floor_ignores_a_cuts_stopband(rng):
    from scipy.signal import firwin, lfilter
    w = lfilter(firwin(161, 7800, fs=FS), 1, H.noise(96_000, rng))
    nf = M.noise_floor(w, FS)
    # white noise of unit power has 1/fs per Hz in the passband
    assert nf["floor_per_hz"] == pytest.approx(1.0 / FS, rel=0.15)
    assert nf["passband_fraction"] < 0.5


def test_blind_symbol_rate_is_the_comb_fundamental(rng):
    r = M.symbol_rate(_qpsk(rng), FS)
    assert r["known"] and r["value_hz"] == pytest.approx(4800.0, abs=0.5)
    b = H.linmod("bpsk", FS, 2400.0, 96_000, rng) * 3 + H.noise(96_000, rng)
    assert M.symbol_rate(b, FS)["value_hz"] == pytest.approx(2400.0, abs=0.5)
    f = H.fsk4(FS, 4800.0, 96_000, rng) * 3 + H.noise(96_000, rng)
    rf = M.symbol_rate(f, FS)
    assert rf["known"] and rf["value_hz"] == pytest.approx(4800.0, abs=1.0)


def test_an_oversampled_cut_is_decimated_and_says_so(rng):
    s = H.linmod("qpsk", FS, 600.0, 96_000, rng)
    r = M.symbol_rate(s * 3 + H.noise(96_000, rng), FS)
    assert r["known"] and r["value_hz"] == pytest.approx(600.0, abs=0.5)
    assert r["decimated_by"] > 1
    assert any("decimating" in c for c in r["caveats"])


def test_noise_has_no_symbol_rate(rng):
    r = M.symbol_rate(H.noise(48_000, rng), FS)
    assert not r["known"] and r["value_hz"] is None


def test_carrier_offset_by_the_right_method(rng):
    b = H.linmod("bpsk", FS, 2400.0, 48_000, rng, fc=-700.0) * 3 + H.noise(
        48_000, rng)
    c = M.carrier_offset(b, FS)
    assert c["value_hz"] == pytest.approx(-700.0, abs=1.0)
    assert "x²" in c["method"] and c["conjugate_feature"] == "present"
    c = M.carrier_offset(_qpsk(rng, fc=1000.0), FS)
    assert c["value_hz"] == pytest.approx(1000.0, abs=1.0)
    assert "x⁴" in c["method"] and c["conjugate_feature"] == "absent"
    # C4FM's x⁴ shows a line per tone — not taken as a carrier
    f = H.fsk4(FS, 4800.0, 96_000, rng) * 3 + H.noise(96_000, rng)
    c = M.carrier_offset(f, FS)
    assert "occupied band" in c["method"]
    assert abs(c["value_hz"]) < 300.0


def test_bursts_duty_and_constant_pri(rng):
    n = 96_000
    y = H.noise(n, rng, 0.01)
    for k in range(40):
        a = int((0.013 + 0.05 * k) * FS)
        y[a:a + 480] += 1.0
    eb = M.envelope_bursts(y, FS)
    bs = M.burst_stats(eb["bursts"], total_s=n / FS)
    assert bs["count"] == 40
    assert bs["length_s"] == pytest.approx(0.010, abs=0.002)
    assert bs["pri_s"] == pytest.approx(0.05, rel=0.01)
    assert bs["pri_kind"] == "constant"
    assert bs["duty"] == pytest.approx(0.2, abs=0.03)


def test_a_staggered_pri_is_told_from_jitter():
    t = np.cumsum(np.tile([0.010, 0.013], 30))
    bs = M.burst_stats(list(t))
    assert bs["pri_kind"].startswith("staggered")
    assert sorted(round(v, 3) for v in bs["levels_s"]) == [0.010, 0.013]


def test_measure_all_carries_tier_and_method_everywhere(rng):
    out = M.measure_all(_qpsk(rng, n=48_000), FS)
    for key in ("occupied_bandwidth", "snr", "symbol_rate", "carrier_offset",
                "bursts"):
        assert out[key]["tier"] == "measured", key
        assert out[key]["method"], key


def test_cyclic_peaks_label_symbol_rate_and_carrier(rng):
    b = H.linmod("bpsk", FS, 2400.0, 48_000, rng, fc=600.0) * 2 + H.noise(
        48_000, rng)
    nc = M.cyclic_peaks(b, FS)
    assert any(abs(p["alpha_hz"] - 2400.0) < 2 for p in nc["peaks"])
    cj = M.cyclic_peaks(b, FS, conj=True)
    assert abs(cj["peaks"][0]["alpha_hz"] - 1200.0) < 2
    assert "carrier" in cj["peaks"][0]["words"]
