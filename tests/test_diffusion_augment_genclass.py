# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""B4 (learn.augment: the synthetic baseline, the local receiver model,
TFD-lite; experiments.augment_eval) and generative classification
(learn.genclass, plan §4.R) with its discriminative yardstick."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from atk_diffusion import cards, profiles
from atk_diffusion.learn import augment as A

PID = "rtlsdr_2400000_cu8"


# -- the synthetic baseline -----------------------------------------------------
@pytest.mark.parametrize("fs", [48000.0, 256000.0, 2.4e6])
def test_every_local_kind_is_n_samples_at_unit_power(fs, rng):
    for kind in A.LOCAL_KINDS:
        for n in (16, 97, 1000):
            try:
                s, truth = A.signal(kind, fs, n, rng, prefer_native=False)
            except ValueError as e:      # a rate too low for the waveform, said so
                assert (kind, fs < 2e6) in (("adsb", True), ("lora", fs < 125e3))
                assert "needs at least" in str(e)
                continue
            assert s.shape == (n,) and s.dtype == np.complex64
            s0, c0 = truth["active"]
            assert np.mean(np.abs(s[s0:s0 + c0]) ** 2) == pytest.approx(1.0, rel=1e-3)
            assert truth["generator"] == "local"


def test_bandwidths_are_measured_not_assumed(rng):
    fs = 2.4e6
    lora, _ = A.signal("lora", fs, 40000, rng, prefer_native=False)
    assert 110e3 < A.occupied_bandwidth(lora, fs) < 160e3
    adsb, _ = A.signal("adsb", fs, 288, rng, prefer_native=False)
    assert A.occupied_bandwidth(adsb, fs) > 1e6        # PPM gaps ARE its spectrum
    x, truth = A.burst("pocsag", 256000.0, 4096, rng)
    assert truth["sample_start"] + truth["sample_count"] <= 4096
    assert abs(truth["offset_hz"]) + truth["bw_hz"] / 2 <= 0.45 * 256000.0 + 1
    assert np.count_nonzero(x) == truth["sample_count"]


def test_class_names_go_to_synth_native_when_it_is_installed(rng):
    native = pytest.importorskip("atk_diffusion.synth.native",
                                 reason="synth.native is another engineer's module")
    assert native is not None
    s, truth = A.signal("pocsag", 256000.0, 4096, rng)
    assert truth["generator"] == "synth.native"
    s0, c0 = truth["active"]
    assert np.mean(np.abs(s[s0:s0 + c0]) ** 2) == pytest.approx(1.0, rel=1e-3)
    b, tb = A.burst_samples("adsb", 2.4e6, 288, rng)
    assert b.size == 288 and tb["generator"] == "synth.native"


def test_the_local_receiver_quantises_images_and_shapes(rng):
    rx = A.LocalReceiver(adc_bits=8, signal_dbfs=0.0, noise_dbfs=-30.0,
                         iq_gain_db=1.0, iq_phase_deg=5.0, edge_droop_db=6.0)
    z = rx.noise_only(1 << 15, 2.4e6, rng)
    assert len(np.unique(z.real)) < 60                 # 8-bit levels
    g = np.sqrt(np.mean(z.imag ** 2) / np.mean(z.real ** 2))
    assert g == pytest.approx(10 ** (1.0 / 20), rel=0.05)     # Bill's model: Q scaled
    from atk_diffusion.dsp import denoise_classical as C
    P = C.stft_power(z, 64).mean(axis=0)
    assert 10 * np.log10(P[32] / P[2]) > 3.0           # droop toward the band edges
    d = A.receiver_for("hackrf_8000000_ci8")
    assert d.adc_bits == 8 and any("not measurements" in n for n in d.notes)


def test_impair_follows_the_measured_profile_or_says_textbook(rng):
    w, words = A.receiver_noise(4096, 2.4e6, PID, rng)
    assert "not measured" in words and w.shape == (4096,)
    prof = profiles.new_profile(PID)
    prof.impairments = {"dc_offset_i": 0.05, "dc_offset_q": 0.0,
                        "iq_gain_imbalance_db": 0.5, "iq_phase_imbalance_deg": 2.0,
                        "floor_mean_dbfs": -30.0, "floor_db_per_bin": [-54.0] * 256,
                        "datatype": "cu8"}
    y, words = A.impair(np.zeros(4096, np.complex64), 2.4e6, prof, rng)
    assert abs(np.mean(y).real - 0.05) < 0.01


def test_make_classification_set_shapes(rng):
    X, y = A.make_classification_set(["bpsk", "tone"], 5, 64, 48000.0, rng)
    assert X.shape == (10, 64) and list(y) == [0] * 5 + [1] * 5
    Z = A.iq_to_channels(X)
    assert Z.shape == (10, 2, 64) and np.allclose(A.channels_to_iq(Z), X)


# -- TFD-lite ----------------------------------------------------------------------
def test_the_blur_grows_with_t_and_is_invertible():
    m = A.blur_envelope([0, 25, 49], 50, 16, blur_max=0.3)
    assert np.allclose(m[0], 1.0)
    assert m[2].min() == pytest.approx(0.3, rel=1e-6) and np.all(m > 0)
    assert np.all(m[1] >= m[2]) and np.all(m[1] <= m[0])


@pytest.fixture(scope="module")
def tiny_tfd(tmp_path_factory):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from atk_diffusion.learn import unet as U
    from atk_diffusion.paths import RfData
    rng = np.random.default_rng(0)
    X, y = A.make_classification_set(["bpsk", "ofdm"], 6, 64, 2.4e6, rng)
    rf = RfData(tmp_path_factory.mktemp("b4") / "rf_data", create=True)
    d = A.train_augmenter(rf, PID, X, y, ["bpsk", "ofdm"], steps=12, batch=6,
                          unet=U.TINY_1D, cfg=A.TFDConfig(T=40), name="tiny_tfd")
    return rf, d, X, y


def test_tfd_lite_trains_a_card_and_augments_with_invented_flags(tiny_tfd):
    import torch
    from atk_diffusion.learn import diffusion as D
    rf, d, X, y = tiny_tfd
    card = cards.load(d, expect_kind="augmenter", for_profile=PID)
    assert card.tier == "invented" and card.input["tfd"]["blur"] is True
    assert "no RF-Diffusion code" in card.license
    aug = A.Augmenter.load(d, for_profile=PID)
    x0 = torch.as_tensor(A.iq_to_channels(A.unit_power(X[:2])[0]))
    same = aug.tfd.degrade(x0, np.zeros(2, dtype=np.int64), torch.zeros_like(x0))
    assert torch.allclose(same, x0 * math.sqrt(D.abar(aug.tfd.ac, 0)), atol=1e-5)
    Xo, yo, info = A.augment_dataset(X, y, aug, strength=0.3, steps=3, n_per=2)
    assert Xo.shape == (3 * len(X), 64) and list(yo[:len(y)]) == list(y)
    assert info["tier"] == "invented" and info["invented"].sum() == 2 * len(X)
    rms_in = np.sqrt(np.mean(np.abs(X) ** 2, axis=1))
    rms_out = np.sqrt(np.mean(np.abs(Xo[len(X):2 * len(X)]) ** 2, axis=1))
    assert np.all(np.isfinite(rms_out)) and rms_out.shape == rms_in.shape
    with pytest.raises(ValueError, match="64-sample windows"):
        aug.augment(np.zeros((1, 32), np.complex64))


def test_the_augmentation_experiment_judges_by_the_domain_gap(tiny_tfd, tmp_path):
    from atk_diffusion.experiments import augment_eval as AE
    _rf, d, _X, _y = tiny_tfd
    res = AE.run(None, PID, augmenter=d, kinds=("bpsk", "ofdm"), window=64,
                 n_synth_per=10, n_real_per=3, n_test_per=8, sample_steps=2,
                 classifier_steps=15, out_dir=tmp_path / "b4")
    acc = res["accuracy_on_cabled_like"]
    assert set(acc) == {"synthetic", "synthetic + augmented", "little real only",
                        "synthetic + little real", "cabled-like (upper line)"}
    assert all(0.0 <= v <= 1.0 for v in acc.values())
    assert set(res["domain_gap"]) == set(acc) - {"cabled-like (upper line)"}
    assert isinstance(res["kept"], bool) and 0.0 <= res["label_flip_rate"] <= 1.0
    assert res["tier_augmented"] == "invented"
    js = json.loads((tmp_path / "b4" / "augment_eval.json").read_text(encoding="utf-8"))
    assert js["result"]["kept"] == res["kept"]


# -- generative classification -------------------------------------------------------
def test_the_discriminative_yardstick_learns_an_easy_task(rng):
    pytest.importorskip("torch")
    import torch
    torch.set_num_threads(1)
    from atk_diffusion.learn import genclass as G
    X, y = A.make_classification_set(["tone", "noise"], 24, 64, 48000.0, rng, snr_db=(15, 15))
    m = G.train_discriminative(X, y, 2, steps=60, seed=0)
    Xt, yt = A.make_classification_set(["tone", "noise"], 16, 64, 48000.0, rng, snr_db=(15, 15))
    assert np.mean(G.predict_discriminative(m, Xt) == yt) > 0.8


def test_generative_classification_runs_and_proposes(tmp_path, rng):
    pytest.importorskip("torch")
    import torch
    torch.set_num_threads(1)
    from atk_diffusion.learn import genclass as G
    from atk_diffusion.learn import unet as U
    X, y = A.make_classification_set(["tone", "chirp"], 8, 64, 48000.0, rng)
    d = G.train_genclass(None, PID, X, y, ["tone", "chirp"], steps=15, batch=8,
                         unet=U.TINY_1D, T=50, out_dir=tmp_path / "gc")
    card = cards.load(d, expect_kind="genclass", for_profile=PID)
    assert card.tier == "proposed" and card.class_names() == ["tone", "chirp"]
    gc = G.GenerativeClassifier.load(d, for_profile=PID)
    idx, names, err, margin = gc.classify(X[:5], n_t=4, seed=3)
    assert idx.shape == (5,) and err.shape == (5, 2) and np.all(margin >= 0)
    assert all(n in ("tone", "chirp") for n in names)
    again = gc.errors(X[:5], n_t=4, seed=3)
    assert np.allclose(again, err)                     # the draws are paired and fixed
    with pytest.raises(ValueError, match="somewhere to put it"):
        G.compare(None, PID, kinds=("tone",), snrs_db=(0.0,))


def test_the_comparison_reports_both_curves(tmp_path):
    pytest.importorskip("torch")
    import torch
    torch.set_num_threads(1)
    from atk_diffusion.learn import genclass as G
    from atk_diffusion.learn import unet as U
    res = G.compare(None, PID, kinds=("tone", "chirp"), snrs_db=(-3.0, 6.0),
                    n_train_per=6, n_test_per=4, window=64, fs=48000.0, gen_steps=10,
                    disc_steps=10, n_t=3, unet=U.TINY_1D, T=50, out_dir=tmp_path / "cmp")
    for k in ("generative", "discriminative"):
        assert len(res["accuracy"][k]) == 2
        assert all(0.0 <= v <= 1.0 for v in res["accuracy"][k])
    assert res["tier"] == "proposed" and "lowest SNR" in res["verdict"]
    assert (tmp_path / "cmp" / "genclass_compare_detail.md").exists()
