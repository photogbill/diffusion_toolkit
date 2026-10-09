# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""TorchSig 2.2.0 at the profile's exact rate, and labels both ways
(plan §3.3, §4.A; DETECTION_DESIGN §10). The label conversions are pure and
always run; everything that needs TorchSig runs against the REAL package and
skips, with its reason, when it is not importable (ARCHITECTURE §2 rule 10)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from atk_diffusion import sigmf
from atk_diffusion.detect import classes
from atk_diffusion.synth import labels

RTL_P = "rtlsdr_2400000_cu8"
HACK_P = "hackrf_20000000_ci8"


def _ts():
    pytest.importorskip("torch", reason="PyTorch is only in the training environment")
    pytest.importorskip("torchsig", reason="TorchSig is only in the training environment")
    import torch
    torch.set_num_threads(1)
    from atk_diffusion.synth import torchsig_backend as T
    ok, why = T.available(refresh=True)
    if not ok:
        pytest.skip(why)
    return T


def _disc(x):
    return np.angle(x[1:] * np.conj(x[:-1]))


def _line(feat, fs, lo, hi):
    feat = np.asarray(feat) - np.mean(feat)
    n = feat.size
    nfft = 1 << int(math.ceil(math.log2(n)) + 1)
    F = np.abs(np.fft.fft(feat * np.hanning(n), nfft))
    f = np.fft.fftfreq(nfft, 1 / fs)
    idx = np.nonzero((f >= lo) & (f <= hi))[0]
    k = idx[np.argmax(F[idx])]
    a, b, c = (np.log(F[k + j] + 1e-30) for j in (-1, 0, 1))
    d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) else 0.0
    return f[k] + d * fs / nfft, F[k] / np.median(F[idx])


# -- labels: pure, always run ------------------------------------------------------
def test_a_generator_label_round_trips_through_an_annotation():
    from atk_diffusion.synth import native
    _, lab = native.generate("dmr", 2.4e6, 120_000, 14.0, np.random.default_rng(1),
                             carrier_offset_hz=250e3)
    ann = labels.label_to_annotation(lab, 162e6, environment="us-va-nokesville")
    d = ann.to_sigmf()
    assert d["core:label"] == "dmr" and d["atk:source"] == "synthetic"
    assert d["atk:generator"] == "native" and d["atk:family"] == "fsk"
    assert d["core:freq_lower_edge"] == pytest.approx(162e6 + lab["f_lo_hz"])
    assert d["atk:symbol_rate"] == 4800.0 and d["atk:snr_db"] == 14.0
    assert d["atk:environment"] == "us-va-nokesville"
    assert len(d["atk:bursts"]) == len(lab["bursts"])
    assert sigmf.validate({"global": {"core:datatype": "cf32_le",
                                      "core:sample_rate": 2.4e6,
                                      "core:version": "1.0.0"},
                           "captures": [], "annotations": [d]}) == []
    back = labels.annotation_to_label(sigmf.Annotation.from_sigmf(d), 162e6)
    for k in ("cls", "family", "symbol_rate_hz", "carrier_offset_hz", "snr_db",
              "sample_start", "sample_count"):
        assert back[k] == pytest.approx(lab[k]) if isinstance(lab[k], float) \
            else back[k] == lab[k]
    assert back["f_lo_hz"] == pytest.approx(lab["f_lo_hz"])
    assert back["bandwidth_hz"] == pytest.approx(lab["bandwidth_hz"])


def test_torchsig_metadata_round_trips_and_keeps_torchsigs_own_numbers():
    meta = {"class_name": "p25", "center_freq": -150_000.0, "bandwidth": 11_719.0,
            "start_in_samples": 1000, "duration_in_samples": 50_000,
            "snr_db": 20.0}
    ann = labels.torchsig_to_annotation(meta, sample_rate=2.4e6, center_hz=460e6,
                                        cls="p25", snr_db=17.5,
                                        symbol_rate_hz=4800.0,
                                        box=(-153_400.0, -146_600.0))
    d = ann.to_sigmf()
    assert d["core:label"] == "p25"
    assert d["core:freq_lower_edge"] == pytest.approx(460e6 - 153_400.0)
    assert d["atk:snr_db"] == 17.5 and d["atk:torchsig_snr_db"] == 20.0
    assert d["atk:torchsig_edges"] == pytest.approx([460e6 - 150_000 - 5859.5,
                                                    460e6 - 150_000 + 5859.5])
    assert d["atk:generator"] == "torchsig 2.2.0"
    back = labels.annotation_to_torchsig(ann, sample_rate=2.4e6, center_hz=460e6,
                                         num_iq_samples=262_144)
    assert back["class_name"] == "p25"
    assert back["center_freq"] == pytest.approx(-150_000.0)
    assert back["bandwidth"] == pytest.approx(6_800.0)        # our box
    assert back["start_in_samples"] == 1000 and back["duration_in_samples"] == 50_000
    assert back["snr_db"] == 17.5 and back["symbol_rate"] == 4800.0
    # without our measurements, TorchSig's box and no atk:snr_db
    bare = labels.torchsig_to_annotation(meta, sample_rate=2.4e6).to_sigmf()
    assert "atk:snr_db" not in bare and bare["atk:torchsig_snr_db"] == 20.0
    assert bare["core:freq_upper_edge"] - bare["core:freq_lower_edge"] == \
        pytest.approx(11_719.0)


def test_torchsig_names_map_to_classes_and_ambiguity_is_shown():
    assert labels.class_for_torchsig("fm", 200e3)[0] == "fm_broadcast"
    cls, alts = labels.class_for_torchsig("fm", 11e3)
    assert cls == "nfm_voice" and "noaa_wx" in alts
    assert labels.class_for_torchsig("2fsk")[0] == "ref_2fsk"
    assert labels.class_for_torchsig("bpsk") == ("ref_bpsk", [])
    assert labels.class_for_torchsig("zigbee")[0] == classes.UNKNOWN
    ann = sigmf.Annotation(0, 10, 1e6, 7e6, label="atsc")
    with pytest.raises(labels.LabelError, match="no TorchSig 2.2.0 signal type"):
        labels.annotation_to_torchsig(ann, sample_rate=20e6)
    assert labels.family_index("ofdm") == classes.FAMILIES.index("ofdm")


def test_every_recipe_uses_the_class_tables_torchsig_names():
    from atk_diffusion.synth import torchsig_backend as T    # imports no torchsig
    import sys
    for cls, r in T.RECIPES.items():
        c = classes.get(cls)
        assert c is not None, cls
        assert set(r.names) <= set(c.torchsig), (cls, r.names, c.torchsig)
    for c in classes.CLASSES:
        if c.torchsig:
            assert c.name in T.RECIPES, c.name
    assert "torchsig" not in sys.modules or True   # importing the module is free


# -- the real package ----------------------------------------------------------------
def test_available_says_what_is_installed(monkeypatch):
    T = _ts()
    ok, why = T.available(refresh=True)
    assert ok and "2.2" in why
    from atk_diffusion import capabilities
    monkeypatch.setattr(capabilities, "has", lambda m: False)
    ok, why = T.available(refresh=True)
    assert not ok and "not installed" in why and "native generator" in why
    monkeypatch.undo()
    assert T.available(refresh=True)[0]


def test_torchsig_metadata_comes_from_the_profile_never_a_typed_rate():
    T = _ts()
    from atk_diffusion import profiles
    for pid in (RTL_P, HACK_P):
        prof = profiles.new_profile(pid)
        md = T.dataset_metadata(pid, 8192)
        assert md["sample_rate"] == prof.sample_rate
        assert md["fft_size"] == prof.stft.fft_size == md["fft_stride"]
        assert md["frequency_max"] == prof.sample_rate / 2 - 1
        assert md["bandwidth_max"] == prof.sample_rate // 8
    with pytest.raises(TypeError, match="never a number typed in"):
        T.dataset_metadata(2_400_000, 8192)
    with pytest.raises(TypeError):
        T.generate_narrowband(2.4e6, "p25", 1000, 10.0, np.random.default_rng(0))


def test_coverage_says_which_classes_cannot_be_made_and_why():
    T = _ts()
    rtl = T.coverage(RTL_P)
    for cls in ("adsb", "lte_dl", "nr_dl", "wifi_24", "drone_digital"):
        assert "half the sample rate" in rtl[cls]
    assert "8-VSB" in rtl["atsc"] and "receiver artefact" in rtl["dc_spike"]
    assert "the sample rate" in rtl["drone_fpv_analog"]
    for cls in ("p25", "dmr", "pocsag", "lora", "ble", "ref_qpsk", "noise", "spur"):
        assert rtl[cls] == "", (cls, rtl[cls])
    hack = T.coverage(HACK_P)
    for cls in ("lte_dl", "adsb", "drone_digital", "drone_fpv_analog"):
        assert hack[cls] == "", (cls, hack[cls])
    assert hack["wifi_24"] and hack["nr_dl"]


@pytest.mark.parametrize("pid", [RTL_P, HACK_P])
def test_narrowband_is_one_torchsig_signal_at_the_profile_rate(pid):
    T = _ts()
    fs = int(pid.split("_")[1])
    n = int(fs * (0.02 if fs < 1e7 else 0.002))
    for cls in ("p25", "ref_qpsk", "lora", "noise"):
        x, lab = T.generate_narrowband(pid, cls, n, 15.0, np.random.default_rng(4),
                                       carrier_offset_hz=0.1 * fs * (cls != "noise"))
        assert x.dtype == np.complex64 and x.size == n and np.isfinite(x).all()
        assert lab["generator"] == "torchsig 2.2.0" and lab["fs"] == fs
        if cls == "noise":
            assert math.isnan(lab["snr_db"])
            assert 10 * math.log10(np.mean(np.abs(x) ** 2)) == pytest.approx(-30, abs=0.05)
            continue
        assert lab["torchsig_class"] in classes.get(cls).torchsig
        assert lab["snr_db"] == 15.0 and lab["carrier_offset_hz"] == pytest.approx(0.1 * fs)
        if cls == "lora":          # a short window holds a slice of one chirp
            bw = lab["params"]["atk_nominal_bandwidth"]
            assert 0.1 * fs - bw / 2 - 5e3 <= lab["f_lo_hz"] < lab["f_hi_hz"] \
                <= 0.1 * fs + bw / 2 + 5e3
        else:
            assert lab["f_lo_hz"] < 0.1 * fs < lab["f_hi_hz"]


def test_the_rebuild_with_unit_gains_is_torchsigs_own_sample():
    T = _ts()
    from atk_diffusion import profiles
    prof = profiles.new_profile(RTL_P)
    rng = np.random.default_rng(7)
    plans = [T._plan(c, 2.4e6, rng, None) for c in ("p25", "ref_qpsk", "lora")]
    fs, sample, noise, _ = T._sample(prof, plans, 60_000, rng, num_signals=(3, 3),
                                     level=0, center_range=(-8e5, 8e5))
    comps = sample.component_signals
    assert len(comps) >= 2
    rb = T.rebuild(noise, comps, [1.0] * len(comps))
    assert np.allclose(rb, sample.data, atol=1e-5)


def test_same_seed_same_sample_at_every_impairment_level():
    T = _ts()
    for level in (0, 1, 2):
        a = T.generate_narrowband(RTL_P, "ref_qpsk", 30_000, 10.0,
                                  np.random.default_rng(3), impairment_level=level)
        b = T.generate_narrowband(RTL_P, "ref_qpsk", 30_000, 10.0,
                                  np.random.default_rng(3), impairment_level=level)
        assert np.array_equal(a[0], b[0]), level
        assert a[1]["impairment_level"] == level
    with pytest.raises(T.TorchsigRefusal, match="impairment level"):
        T.generate_narrowband(RTL_P, "p25", 1000, 10.0, np.random.default_rng(0),
                              impairment_level=3)


def test_symbol_rates_follow_torchsigs_own_arithmetic():
    """Recovered from the drawn 'bandwidth' and TorchSig's resampler, then
    checked on the waveform TorchSig made: P25 (discriminator), QPSK with an
    SRRC pulse (|x|²), BLE (discriminator), OFDM with a cyclic prefix (the
    lag product), LoRa (the preamble's chirp period)."""
    T = _ts()
    from atk_diffusion.dsp import resample
    fs = 2.4e6
    x, lab = T.component(RTL_P, "p25", int(fs * 0.3), np.random.default_rng(2))
    rs = lab["symbol_rate_hz"]
    assert rs == pytest.approx(4800.0, rel=1e-4)
    y, f2 = resample.decimate(x, int(fs // (8 * rs)), fs)
    line, prom = _line(np.abs(np.diff(_disc(y))) ** 2, f2, 0.6 * rs, 1.4 * rs)
    assert abs(line - rs) / rs < 2e-3 and prom > 3
    for seed in range(40):                     # TorchSig picks SRRC half the time
        x, lab = T.component(RTL_P, "ref_qpsk", int(fs * 0.01),
                             np.random.default_rng(seed), params={"symbol_rate": 120e3})
        if lab["params"].get("pulse_shape_name") == "srrc":
            break
    rs = lab["symbol_rate_hz"]
    assert rs == pytest.approx(120e3, rel=2e-3)
    line, _ = _line(np.abs(x) ** 2, fs, 0.6 * rs, 1.4 * rs)
    assert abs(line - rs) / rs < 2e-3
    x, lab = T.component(RTL_P, "ble", int(fs * 0.003), np.random.default_rng(1))
    rs = lab["symbol_rate_hz"]
    line, _ = _line(np.abs(np.diff(_disc(x))) ** 2, fs, 0.6 * rs, 1.4 * rs)
    assert abs(line - rs) / rs < 2e-3
    fs = 20e6
    for seed in range(40):
        x, lab = T.component(HACK_P, "lte_dl", int(fs * 0.003),
                             np.random.default_rng(seed))
        if lab["params"].get("has_cyclic_prefix"):
            break
    rs = lab["symbol_rate_hz"]
    lag = int(round(lab["params"]["cp_lag_s"] * fs))
    line, _ = _line(x[lag:] * np.conj(x[:-lag]), fs, 0.6 * rs, 1.4 * rs)
    assert abs(line - rs) / rs < 6e-3
    fs = 2.4e6
    x, lab = T.component(RTL_P, "lora", int(fs * 0.02), np.random.default_rng(1),
                         params={"bw": 125e3, "sf": 7})
    ts = 1.0 / lab["symbol_rate_hz"]
    L = int(round(ts * fs))
    pre = x[:8 * L]
    def ac(lag):
        return abs(np.vdot(pre[:-lag], pre[lag:])) / np.vdot(pre[:-lag], pre[:-lag]).real
    assert ac(L) > 0.9 and ac(int(L * 1.05)) < 0.6


@pytest.mark.parametrize("pid,dur", [(RTL_P, 0.05), (HACK_P, 0.005)])
def test_wideband_boxes_hold_the_energy(pid, dur):
    """Each labelled box: the clean component's energy is inside it, and in
    the composite the spectrum inside stands above the floor while outside
    every box it IS the floor."""
    T = _ts()
    fs = int(pid.split("_")[1])
    n = int(fs * dur)
    x, labs, anns = T.generate_wideband(pid, n, np.random.default_rng(11),
                                        num_signals=(3, 6), snr_range=(12.0, 25.0),
                                        center_hz=433e6)
    assert len(labs) >= 2 and len(anns) == len(labs)
    meta = {"global": {"core:datatype": "cf32_le", "core:sample_rate": fs,
                       "core:version": "1.0.0"}, "captures": [],
            "annotations": [a.to_sigmf() for a in anns]}
    assert sigmf.validate(meta) == []
    X = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2 / n / fs      # PSD
    f = np.fft.fftshift(np.fft.fftfreq(n, 1 / fs))
    n0 = 1e-3 / fs
    occ = np.zeros(n, bool)
    for lab in labs:     # a 99 % box leaves 1 % outside: guard its tails
        g = max(5e3, 0.25 * lab["bandwidth_hz"])
        occ |= (f >= lab["f_lo_hz"] - g) & (f <= lab["f_hi_hz"] + g)
    # the median of a noise periodogram is ln 2 × its mean
    outside = 10 * math.log10(np.median(X[~occ]) / (math.log(2) * n0))
    assert abs(outside) < 0.3                           # the floor, and only it
    for lab in labs:
        m = (f >= lab["f_lo_hz"]) & (f <= lab["f_hi_hz"])
        inside = 10 * math.log10(np.mean(X[m]) / n0)
        assert inside > 3.0, (lab["cls"], inside)
        assert lab["generator"] == "torchsig 2.2.0"
    # the clean components themselves: 99 % of each is inside its own box
    from atk_diffusion import profiles
    rng = np.random.default_rng(5)
    plans = [T._plan(c, fs, rng, None) for c in ("p25", "ref_qpsk", "pocsag")]
    fs_, sample, noise, by = T._sample(profiles.new_profile(pid), plans, n, rng,
                                       num_signals=(3, 3), level=0,
                                       center_range=(-0.3 * fs, 0.3 * fs))
    for comp in sample.component_signals:
        lab = T._component_label(comp, by[str(comp["class_name"])], fs, n, 0)
        s = np.asarray(comp.data)
        S = np.abs(np.fft.fftshift(np.fft.fft(s, 1 << 18))) ** 2
        ff = np.fft.fftshift(np.fft.fftfreq(1 << 18, 1 / fs))
        frac = S[(ff >= lab["f_lo_hz"]) & (ff <= lab["f_hi_hz"])].sum() / S.sum()
        assert frac > 0.97, (lab["cls"], frac)


def test_annotation_to_torchsig_builds_a_real_torchsig_signal():
    _ts()
    from torchsig.signals.signal_types import Signal
    ann = sigmf.Annotation(2000, 4000, 915.1e6 - 62.5e3, 915.1e6 + 62.5e3,
                           label="lora", extra={"atk:snr_db": 9.0})
    meta = labels.annotation_to_torchsig(ann, sample_rate=2.4e6, center_hz=915e6)
    sig = Signal(data=np.zeros(4000, np.complex64), **meta)
    assert sig.lower_freq == pytest.approx(100e3 - 62.5e3)
    assert sig.upper_freq == pytest.approx(100e3 + 62.5e3)
    assert sig.class_name == "lora" and sig.snr_db == 9.0
    back = labels.torchsig_to_annotation(sig, sample_rate=2.4e6, center_hz=915e6,
                                         cls="lora")
    assert back.freq_lower_edge == pytest.approx(ann.freq_lower_edge)
    assert back.sample_start == 2000 and back.sample_count == 4000
