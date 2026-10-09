# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Classical RF fingerprints (plan C1) and the emitter library with its
honest UNKNOWN, per receiver profile."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from atk_diffusion import profiles
from atk_diffusion.experiments import fingerprint_eval as FE
from atk_diffusion.fingerprint import features as F
from atk_diffusion.fingerprint import library as L
from atk_diffusion.geo import products as P

FS = 48_000.0
FC = 446.0e6
PROFILE = "hackrf_8000000_ci8"


def _fp(x, **kw):
    return F.extract(x, FS, FC, symbol_rate=4800.0, profile=PROFILE, rx_ppm=0.0, **kw)


# -- the measurements -----------------------------------------------------------
def test_carrier_offset_estimators():
    rng = np.random.default_rng(0)
    t = np.arange(20000) / FS
    tone = np.exp(2j * math.pi * 1234.5 * t) + 0.05 * (rng.normal(size=t.size)
                                                        + 1j * rng.normal(size=t.size))
    assert abs(F.carrier_offset_hz(tone, FS) - 1234.5) < 0.5
    sym = np.exp(1j * (math.pi / 4 + math.pi / 2 * rng.integers(0, 4, 2000)))
    qpsk = np.repeat(sym, 10) * np.exp(2j * math.pi * 321.0 * np.arange(20000) / FS)
    assert abs(F.carrier_offset_hz(qpsk, FS, "power_law", 4) - 321.0) < 2.0
    with pytest.raises(ValueError):
        F.carrier_offset_hz(tone, FS, "magic")


def test_image_rejection_and_receiver_axes():
    rng = np.random.default_rng(1)
    s = (rng.normal(size=200_000) + 1j * rng.normal(size=200_000)) / math.sqrt(2)
    g, ph = 10 ** (0.5 / 20), math.radians(3.0)
    k1 = (1 + g * np.exp(-1j * ph)) / 2
    k2 = (1 - g * np.exp(1j * ph)) / 2
    y = k1 * s + k2 * np.conj(s)
    want = 20 * math.log10(abs(k2) / abs(k1))
    assert abs(F.irr_db_from_circularity(F.circularity(y)) - want) < 0.5
    # a receiver's own imbalance, in its own axes
    i = (1.0 + 0.03) * s.real
    q = (1.0 - 0.03) * (s.imag * math.cos(math.radians(2)) + s.real * math.sin(math.radians(2)))
    rx = F.iq_imbalance(i + 1j * q)
    assert abs(rx["gain_db"] - 20 * math.log10(1.03 / 0.97)) < 0.05
    assert abs(rx["phase_deg"] - 2.0) < 0.2
    assert F.irr_db_from_circularity(0.0) < -150                 # perfectly proper
    assert F.irr_db_from_circularity(0.01) < F.irr_db_from_circularity(0.1) < 0


def test_extract_reads_the_radio_not_the_noise():
    radios = FE.same_model_pair(0, 2)
    rng = np.random.default_rng(2)
    fps = [_fp(FE.simulate_burst(radios[0], rng, snr_db=25.0)) for _ in range(4)]
    cfo = np.mean([f.values["cfo_ppm"] for f in fps])
    assert abs(cfo - radios[0].cfo_ppm) < 0.3
    assert all(f.tier == "measured" and f.profile == PROFILE for f in fps)
    assert all(abs(f.snr_db - 25.0) < 2.0 for f in fps)
    assert fps[0].values["phase_noise_rms_hz"] is None          # modulated
    assert any("modulated" in n for n in fps[0].notes)
    fast = FE.RadioModel("fast", ramp_ms=0.5, ramp_damping=0.8)
    slow = FE.RadioModel("slow", ramp_ms=1.5, ramp_damping=0.8)
    r_fast = _fp(FE.simulate_burst(fast, rng, snr_db=30.0)).values["rise_time_ms"]
    r_slow = _fp(FE.simulate_burst(slow, rng, snr_db=30.0)).values["rise_time_ms"]
    assert r_slow > 2 * r_fast
    quiet = FE.RadioModel("quiet", linewidth_hz=5.0)
    noisy = FE.RadioModel("noisy", linewidth_hz=40.0)
    pn = [_fp(FE.simulate_burst(r, rng, snr_db=25.0, modulation="carrier"))
          .values["phase_noise_rms_hz"] for r in (quiet, noisy)]
    assert pn[1] > pn[0] > 0
    low = _fp(FE.simulate_burst(radios[0], rng, snr_db=6.0))
    assert low.values["rise_time_ms"] is None
    assert any("below 10 dB" in n for n in low.notes)
    # the receiver's own ppm is removed only when the profile carries it
    x = FE.simulate_burst(radios[0], rng, snr_db=25.0)
    a = F.extract(x, FS, FC, profile=PROFILE).values["cfo_ppm"]
    b = F.extract(x, FS, FC, profile=PROFILE, rx_ppm=0.4).values["cfo_ppm"]
    assert math.isclose(a - b, 0.4, abs_tol=1e-9)
    assert any("not in its profile" in n for n in F.extract(x, FS, FC).notes)
    with pytest.raises(ValueError, match="no burst"):
        F.extract((rng.normal(size=9000) + 1j * rng.normal(size=9000)), FS, FC)
    with pytest.raises(ValueError, match="5 ms"):
        F.extract(np.ones(100, complex), FS, FC)


def test_symbol_clock_on_a_long_clean_burst():
    r = FE.RadioModel("clk", cfo_ppm=0.0, clock_ppm=300.0, pll_offset_hz=0.0,
                      linewidth_hz=1.0)
    x = FE.simulate_burst(r, np.random.default_rng(3), snr_db=35.0, duration_s=1.0,
                          multipath=0.0, doppler_hz=0.0, lead_s=0.06)
    fp = _fp(x)
    assert abs(fp.values["symbol_clock_ppm"] - 300.0) < 40.0


# -- the library ------------------------------------------------------------------
@pytest.fixture(scope="module")
def three_radios():
    radios = FE.same_model_pair(0, 3)
    # a radio that is plainly not one of them (another make): crystal 2 ppm
    # away, a slow ramp
    other = FE.RadioModel("other-make", cfo_ppm=-1.6, ramp_ms=2.2,
                          ramp_damping=0.9, pll_offset_hz=-100.0)
    enrol = FE.bursts(radios[:2], 10, snrs=(15.0, 20.0, 30.0), seed=11)
    test = FE.bursts(radios[:2], 8, snrs=(15.0, 20.0, 30.0), seed=12)
    stranger = FE.bursts([other], 8, snrs=(15.0, 20.0, 30.0), seed=13)
    same_model = FE.bursts(radios[2:], 8, snrs=(15.0, 20.0, 30.0), seed=14)
    return radios, [(_fp(x), i) for x, i, _ in enrol], \
        [(_fp(x), i) for x, i, _ in test], [_fp(x) for x, _, _ in stranger], \
        [_fp(x) for x, _, _ in same_model]


def test_enrol_match_unknown_and_threshold(rf, three_radios):
    radios, enrol, test, stranger, same_model = three_radios
    lib = L.EmitterLibrary(rf)
    ids = {}
    for fp, i in enrol:
        ids[i] = lib.add(fp, ids.get(i), name=radios[i].name)
    assert len(lib.emitters) == 2
    right = sum(lib.match(fp).emitter_id == ids[i] for fp, i in test)
    assert right / len(test) >= 0.85
    m = lib.match(test[0][0])
    assert m.tier == "proposed" and m.candidates and "proposed" in m.words
    cal = lib.calibrate_threshold([(fp, ids[i]) for fp, i in test[::2]], stranger)
    assert cal["known_accepted"] >= 0.95
    assert cal["unknown_rejected"] >= 0.75
    unknown = [lib.match(fp) for fp in stranger]
    assert sum(m.emitter_id == L.UNKNOWN for m in unknown) / len(unknown) >= 0.75
    assert "UNKNOWN" in [m for m in unknown if m.emitter_id == L.UNKNOWN][0].words
    # a third unit of the SAME model is the hard case: measured, not assumed
    rate = np.mean([lib.match(fp).emitter_id == L.UNKNOWN for fp in same_model])
    assert 0.0 <= rate <= 1.0


def test_profiles_never_mix(rf, three_radios):
    radios, enrol, _, _, _ = three_radios
    lib = L.EmitterLibrary(rf)
    eid = lib.add(enrol[0][0], name="A")
    other = F.Fingerprint.from_json({**enrol[1][0].to_json(),
                                     "profile": "rtlsdr_2400000_cu8"})
    with pytest.raises(L.ProfileRefused, match="compared only within one receiver"):
        lib.match(other)
    with pytest.raises(profiles.ProfileMismatch):
        lib.add(other, eid)
    with pytest.raises(ValueError, match="without a receiver profile"):
        lib.add(F.Fingerprint({"cfo_ppm": 1.0}))


def test_the_library_is_a_product_that_round_trips(rf, three_radios):
    radios, enrol, test, _, _ = three_radios
    lib = L.EmitterLibrary(rf, threshold=4.0)
    ids = {}
    for fp, i in enrol:
        ids[i] = lib.add(fp, ids.get(i), name=radios[i].name)
    m = lib.match(test[0][0])
    lib.add_sighting(m.emitter_id, "2026-10-08T14:00:00Z", 38.70, -77.50, 446.1e6,
                     distance=m.distance, decoder_id="DMR:3110001", profile=PROFILE)
    lib.add_sighting(m.emitter_id, "2026-10-08T15:00:00Z", None, None, 446.1e6)
    d = lib.save()
    em_csv = (rf.products("emitters") / "emitters.csv").read_bytes()
    assert em_csv.startswith(b"\xef\xbb\xbf") and b"\r\n" in em_csv
    rows = em_csv.decode("utf-8-sig").strip().split("\r\n")
    assert rows[0] == "emitter_id,name,profile,feature,mean,std,n"
    assert all(len(r.split(",")) == 7 for r in rows[1:])
    sg = P.read_geojson(rf.products("emitters") / "sightings.geojson")
    assert len(sg["features"]) == 1                          # one had no position
    assert sg["features"][0]["properties"]["atk:tier"] == "proposed"
    assert P.verify_run(d) == (True, [])
    again = L.EmitterLibrary(rf)
    assert set(again.emitters) == set(lib.emitters) and again.threshold == 4.0
    em = again.emitters[m.emitter_id]
    assert "DMR:3110001" in em.decoder_ids
    assert np.allclose(em.mean, lib.emitters[m.emitter_id].mean)
    assert again.match(test[0][0]).emitter_id == m.emitter_id
    assert len(again.sightings) == 2
