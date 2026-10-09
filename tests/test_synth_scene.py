# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Environment profiles and the scene composer (plan §3.6, D8): the prior is
well-formed and sourced, its cellular rows are ATK's own, a composed scene has
the right kinds of signal in the right bands with boxes that hold their
energy, the terrain's channel has unit mean gain and the right Doppler, and a
measured receiver is applied."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from atk_diffusion import profiles
from atk_diffusion.detect import classes
from atk_diffusion.synth import environments as E
from atk_diffusion.synth import scene as S

RTL_P = "rtlsdr_2400000_cu8"
HACK_P = "hackrf_20000000_ci8"
SNAPSHOT = Path("/home/claude/atk_snapshot/atk/core")


@pytest.fixture(scope="module")
def env():
    return E.builtin("us-va-nokesville")


# -- the environment profile ---------------------------------------------------------
def test_the_builtin_prior_is_valid_sourced_and_says_it_is_a_prior(env):
    assert E.validate(env) == []
    assert env.itu_region == 2 and env.country == "US"
    assert env.terrain_class == "rural" and env.channel()["k_factor_db"] == 6.0
    assert "prior, not the place" in env.caveat and "prior" in env.describe()
    for a in env.allocations:
        assert a.source, a.service                  # every allocation is sourced
    services = " ".join(a.service for a in env.allocations)
    for want in ("FM broadcast", "NOAA", "VHF land mobile", "UHF land mobile",
                 "700 MHz", "800 MHz", "ISM", "ADS-B", "GNSS L1", "2.4 GHz"):
        assert want in services, want
    fm = [a for a in env.allocations if a.service == "FM broadcast"][0]
    assert (fm.f_lo_hz, fm.f_hi_hz, fm.channel_hz) == (88e6, 108e6, 200e3)
    noaa = [a for a in env.allocations if "NOAA" in a.service][0]
    centres = noaa.f_lo_hz + noaa.channel_hz / 2 + noaa.channel_hz * np.arange(7)
    assert np.allclose(centres, 162.400e6 + 25e3 * np.arange(7))
    gnss = [a for a in env.allocations if "GNSS" in a.service][0]
    assert gnss.f_lo_hz < 1575.42e6 < gnss.f_hi_hz


def test_no_operator_station_or_tower_is_asserted(env):
    text = json.dumps(env.to_json()).lower()
    for word in ("verizon", "at&t", "t-mobile", "sprint", "dish", "wtop",
                 "wamu", "tower at", "latitude", "longitude", "earfcn"):
        assert word not in text, word


def test_cellular_rows_are_atks_own_tables(env):
    """The US downlink rows equal ATK's lte_bands.py / nr_bands.py (read
    from ATK's files when present — they are not a run-time dependency)."""
    lte_f, nr_f = SNAPSHOT / "lte_bands.py", SNAPSHOT / "nr_bands.py"
    if not (lte_f.exists() and nr_f.exists()):
        pytest.skip("ATK's source is not beside the toolkit here")

    def load(p):
        spec = importlib.util.spec_from_file_location(f"_atk_{p.stem}", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    lte, nr = load(lte_f), load(nr_f)
    want_lte = {(b, lte.BAND_BY_NUM[b].dl_low, lte.BAND_BY_NUM[b].dl_high)
                for b in lte.REGIONS["US"]}
    got_lte = {(int(c.band[1:]), c.dl_lo_hz, c.dl_hi_hz)
               for c in env.cellular_bands if c.technology == "lte"}
    assert got_lte == want_lte
    want_nr = {(b, nr.BAND_BY_NUM[b].dl_low, nr.BAND_BY_NUM[b].dl_high)
               for b in nr.REGIONS["US"]}
    got_nr = {(int(c.band[1:]), c.dl_lo_hz, c.dl_hi_hz)
              for c in env.cellular_bands if c.technology == "nr"}
    assert got_nr == want_nr
    modes = {int(c.band[1:]): c.duplex for c in env.cellular_bands
             if c.technology == "lte"}
    assert all(modes[b] == lte.BAND_BY_NUM[b].mode for b in modes)


def test_save_load_and_refusals(rf, env):
    p = E.save(rf, env)
    assert p == rf.environments_dir() / "us-va-nokesville.json"
    assert rf.verify(p)[0]
    back = E.load(rf, "us-va-nokesville")
    assert back.to_json()["allocations"] == env.to_json()["allocations"]
    assert E.list_environments(rf) == ["us-va-nokesville"]
    with pytest.raises(E.EnvironmentError, match="no environment profile 'mars'"):
        E.load(rf, "mars")
    bad = E.builtin("us-va-nokesville")
    bad.allocations[0].classes.append("smoke_signals")
    with pytest.raises(E.EnvironmentError, match="not in the class table"):
        E.save(rf, bad)
    d = env.to_json()
    d["terrain_class"] = "swamp"
    with pytest.raises(E.EnvironmentError, match="unknown terrain"):
        E.EnvironmentProfile.from_json(d)


# -- the channel ---------------------------------------------------------------------
@pytest.mark.parametrize("terrain", ["rural", "urban", "mountain"])
def test_the_terrain_channel_has_unit_mean_gain(terrain):
    model = E.TERRAIN_CHANNELS[terrain]
    rng = np.random.default_rng(2)
    fs = 20e6
    s = np.ones(4000, np.complex64)
    gains = [np.mean(np.abs(S.apply_channel(s, fs, model, 50.0, rng)[0][2000:]) ** 2)
             for _ in range(300)]
    assert np.mean(gains) == pytest.approx(1.0, abs=0.12)


def test_rayleigh_fading_has_the_right_doppler_and_statistics():
    rng = np.random.default_rng(3)
    fs, f_d = 10e3, 100.0
    y, los = S.apply_channel(np.ones(200_000, np.complex64), fs,
                             {"k_factor_db": None, "rms_delay_s": 0.0}, f_d, rng)
    assert los == 0.0
    p = np.abs(y) ** 2
    assert np.mean(p) == pytest.approx(1.0, rel=0.25)
    assert np.mean(p < 0.1) == pytest.approx(1 - math.exp(-0.1), abs=0.05)  # Rayleigh
    spec = np.abs(np.fft.fftshift(np.fft.fft(y))) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(y.size, 1 / fs))
    inside = spec[np.abs(f) <= 1.05 * f_d].sum() / spec.sum()
    assert inside > 0.97                         # Clarke: nothing beyond f_D


# -- scenes ----------------------------------------------------------------------------
def _psd(x, fs):
    n = x.size
    X = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2 / n / fs
    return np.fft.fftshift(np.fft.fftfreq(n, 1 / fs)), X


def test_an_fm_band_scene_has_stations_on_the_raster_and_holds_its_energy(env):
    fc, fs = 98.0e6, 2.4e6
    r = S.compose_scene_detailed(env, RTL_P, fc, 0.05, np.random.default_rng(3))
    assert r.x.dtype == np.complex64 and r.x.size == int(0.05 * fs)
    fm = [a for a in r.annotations if a.label == "fm_broadcast"]
    assert fm, "no stations"
    for a in r.annotations:
        d = a.to_sigmf()
        assert d["atk:environment"] == "us-va-nokesville"
        assert d["atk:source"] == "synthetic" and "atk:snr_db" in d
        assert fc - fs / 2 - 1 <= d["core:freq_lower_edge"] < d["core:freq_upper_edge"] \
            <= fc + fs / 2 + 1
    for a in fm:                                 # odd tenths: 88.1 … 107.9 MHz
        c = (a.extra["atk:carrier_offset_hz"] - a.extra["atk:doppler_hz"] + fc) / 1e5
        assert abs(c - round(c)) < 1e-6 and round(c) % 2 == 1
    assert r.info["agc_gain_db"] <= 0.0 and not r.info["receiver_impairments_applied"]
    assert "not measured" in r.info["receiver"]
    # in the composite: every unclipped box stands above the floor
    f, X = _psd(r.x.astype(np.complex128), fs)
    n0 = 10 ** (r.info["noise_dbfs"] / 10) / fs * 10 ** (r.info["agc_gain_db"] / 10)
    for lab in r.labels:
        if lab["clipped"]:
            continue
        m = (f >= lab["f_lo_hz"]) & (f <= lab["f_hi_hz"])
        assert 10 * math.log10(np.mean(X[m]) / n0) > 3.0, lab["cls"]


def test_a_land_mobile_scene_has_only_land_mobile_classes(env):
    r = S.compose_scene_detailed(env, RTL_P, 460e6, 0.2, np.random.default_rng(3))
    allowed = {"nfm_voice", "dmr", "p25", "nxdn96", "nxdn48", "spur"}
    assert r.annotations and {a.label for a in r.annotations} <= allowed
    mob = [e for e in r.info["emitters"] if e["cls"] in allowed - {"spur"}]
    assert all(e["mobility"] == "mobile" for e in mob)
    dmr = [a for a in r.annotations if a.label == "dmr"]
    if dmr:                                      # TDMA: a box per burst
        assert all(a.sample_count <= 0.0276 * 2.4e6 + 2 for a in dmr)


def test_a_cellular_scene_shows_the_slice_of_a_wider_carrier(env):
    r = S.compose_scene_detailed(env, RTL_P, 740e6, 0.02, np.random.default_rng(3))
    lte = [l for l in r.labels if l["cls"] == "lte_dl"]
    assert lte and all(l["clipped"] for l in lte)
    assert all(l["bandwidth_hz"] <= 2.4e6 for l in lte)


def test_a_hackrf_ism_scene(env):
    r = S.compose_scene_detailed(env, HACK_P, 2437e6, 0.002, np.random.default_rng(3))
    assert {a.label for a in r.annotations} <= {"wifi_24", "ble", "drone_digital", "spur"}
    assert r.x.size == 40_000


def test_a_measured_receiver_is_applied(rf, env):
    from atk_diffusion.dsp import impair
    rng = np.random.default_rng(1)
    n = 262_144
    w = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * math.sqrt(1e-3 / 2)
    w = impair.apply_iq(w.astype(np.complex64), 0.4, 2.0) + np.complex64(0.01 - 0.004j)
    m = impair.measure_impairments(impair.quantise(w, "cu8"), 2.4e6, "cu8")
    impair.store(rf, RTL_P, m, device_serial="T1")
    prof = profiles.load_profile(rf, RTL_P)
    r = S.compose_scene_detailed(env, prof, 162.45e6, 0.05, np.random.default_rng(4))
    assert r.info["receiver_impairments_applied"]
    lv = (r.x.real * 127.5 + 127.5)              # on the cu8 grid
    assert np.allclose(lv, np.round(lv), atol=1e-3)
    assert abs(np.mean(r.x) - (0.01 - 0.004j)) < 2e-3


def test_choose_center_stays_in_the_receivers_tuning_range(env):
    rng = np.random.default_rng(0)
    for _ in range(50):
        c = S.choose_center(env, RTL_P, rng)
        assert 24e6 <= c <= 1766e6
    c = S.choose_center(env, HACK_P, rng, classes=["wifi_24"])
    assert 2400e6 <= c <= 2483.5e6
    with pytest.raises(E.EnvironmentError, match="tuning range"):
        S.choose_center(env, "kiwisdr_12000_ci16", rng)


def test_compose_scene_with_torchsig_falls_back_in_words(env):
    pytest.importorskip("torch", reason="PyTorch is only in the training environment")
    pytest.importorskip("torchsig", reason="TorchSig is only in the training environment")
    import torch
    torch.set_num_threads(1)
    x, anns = S.compose_scene(env, RTL_P, 460e6, 0.1, np.random.default_rng(3),
                              generator="torchsig")
    gens = {a.extra["atk:generator"] for a in anns}
    assert "torchsig 2.2.0" in gens
    r = S.compose_scene_detailed(env, RTL_P, 740e6, 0.01, np.random.default_rng(3),
                                 generator="torchsig")
    lte = [a for a in r.annotations if a.label == "lte_dl"]
    assert lte and all(a.extra["atk:generator"].startswith("native (TorchSig could not")
                       for a in lte)
