# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""A6: the receiver-to-receiver translator (learn.translator) and its
experiment (experiments.translator_eval). The pairing and the classical
comparator are checked exactly; the learned path with a tiny model."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from atk_diffusion import cards, profiles, sigmf
from atk_diffusion.learn import translator as TR

A_PID = "bladerf1_2400000_ci16"
B_PID = "rtlsdr_2400000_cu8"


def _cn(rng, n, p=1.0):
    return math.sqrt(p / 2) * (rng.normal(size=n) + 1j * rng.normal(size=n))


def test_align_pair_finds_lag_offset_and_phase_and_leaves_b_alone(rng):
    a = _cn(rng, 4096)
    lag, f, ph = 37, 0.0123, 1.1
    b = np.concatenate([_cn(rng, lag, 1e-4), a])[:4096]
    n = np.arange(b.size)
    b = b * np.exp(1j * (2 * np.pi * f * n + ph)) + _cn(rng, b.size, 1e-3)
    b_before = b.copy()
    aa, bb, info = TR.align_pair(a, b)
    assert info["lag"] == lag
    assert abs(info["offset_cycles_per_sample"] - f) < 1.0 / 4096
    assert TR.nmse_db(aa, bb) < -15
    assert np.array_equal(b, b_before)


def test_the_widely_linear_comparator_learns_gain_filter_image_and_dc(rng):
    A = np.stack([_cn(rng, 128) for _ in range(40)])
    h = np.array([0.1, 0.9 * np.exp(0.3j), 0.2])
    B = np.stack([np.convolve(a, h, mode="same") + 0.15 * np.conj(a) + (0.05 - 0.02j)
                  for a in A]) + _cn(rng, 40 * 128, 1e-6).reshape(40, 128)
    lin = TR.fit_linear(A[:30], B[:30], taps=5)
    assert abs(lin["c"] - (0.05 - 0.02j)) < 1e-3
    assert TR.nmse_db(TR.apply_linear(lin, A[30:]), B[30:]) < -40
    assert TR.nmse_db(A[30:], B[30:]) > -10


def test_fingerprints_see_quantisation_dc_and_the_image(rng):
    from atk_diffusion.learn import augment as AUG
    z = _cn(rng, 1 << 14, 1e-3)
    flat = TR.stats(z)
    rtl = AUG.LocalReceiver(adc_bits=8, signal_dbfs=0.0, noise_dbfs=-30.0, dc_dbfs=-30.0,
                            iq_gain_db=1.0, iq_phase_deg=6.0)
    rx = TR.stats(rtl.noise_only(1 << 14, 2.4e6, rng))
    assert flat["improperness"] < 0.03 and rx["improperness"] > 0.05
    assert rx["dc_db"] > flat["dc_db"] + 20
    assert rx["levels_per_rms"] < 50 < flat["levels_per_rms"]


def test_different_rates_need_the_logged_resampler(tmp_path, rng):
    pytest.importorskip("torch")
    from atk_diffusion.paths import RfData
    A = np.stack([_cn(rng, 64) for _ in range(4)]).astype(np.complex64)
    with pytest.raises(profiles.ProfileMismatch, match="resample"):
        TR.train_translator(RfData(tmp_path / "rf", create=True), "bladerf1_4000000_ci16",
                            B_PID, A, A, steps=1)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    torch.set_num_threads(1)
    from atk_diffusion.learn import augment as AUG
    from atk_diffusion.learn import unet as U
    from atk_diffusion.paths import RfData
    from atk_diffusion.experiments import translator_eval as TE
    rng = np.random.default_rng(0)
    ra, rb = TE.stand_in_receivers()
    S, _y = AUG.make_classification_set(["bpsk", "ofdm"], 12, 64, 2.4e6, rng, snr_db=(60, 60))
    PA = np.stack([ra.apply(s, 2.4e6, rng) for s in S])
    PB = np.stack([rb.apply(s, 2.4e6, rng) for s in S])
    rf = RfData(tmp_path_factory.mktemp("a6") / "rf_data", create=True)
    d = TR.train_translator(rf, A_PID, B_PID, PA, PB, steps=10, batch=8, T=100,
                            unet=U.TINY_1D, sample_steps=3, linear_taps=5, name="tiny_a6")
    return rf, d, PA


def test_training_writes_an_invented_card_under_the_target_profile(tiny):
    rf, d, _PA = tiny
    assert d.parent == rf.models(B_PID)
    card = cards.load(d, expect_kind="translator")
    assert card.tier == "invented" and card.weights["format"] == "onnx"
    assert card.input["from_profile"] == A_PID and card.input["to_profile"] == B_PID
    assert card.sample_rate == 2.4e6 and card.input["window"] == 64
    m = card.metrics
    assert set(m["val_psd_distance_db"]) == {"diffusion", "linear", "identity"}
    assert isinstance(m["beats_classical"], bool)
    with pytest.raises(profiles.ProfileMismatch, match="Profiles never mix"):
        TR.Translator.load(d, from_profile="hackrf_2400000_ci8")


def test_translate_a_stream_and_a_capture_labelled_translated_from(tiny, tmp_path):
    rf, d, PA = tiny
    tr = TR.Translator.load(d, from_profile=A_PID)
    assert tr.backend == "onnxruntime"
    x = PA[:3].reshape(-1)
    y, info = tr.translate(x, steps=2)
    assert y.shape == x.shape and np.all(np.isfinite(y))
    assert info["tier"] == "invented" and info["translated_from"] == A_PID
    base = tmp_path / "blade"
    sigmf.write_pair(base, x, 2.4e6, 433.9e6, datatype="ci16q11",
                     extra_global={"atk:receiver_profile": A_PID})
    out = TR.translate_capture(str(base) + ".sigmf-data", d, tmp_path / "as_rtl",
                               rf=rf, steps=2)
    g = sigmf.read_meta(out["meta"])["global"]
    assert g["atk:translated_from"] == A_PID and g["atk:receiver_profile"] == B_PID
    assert g["atk:tier"] == "invented" and g["atk:method"] == "diffusion_translate"
    assert g["atk:model_sha256"] == cards.load(d).weights["sha256"]
    assert sigmf.load(out["data"]).size == x.size
    other = tmp_path / "rtl"
    sigmf.write_pair(other, x, 2.4e6, extra_global={"atk:receiver_profile": B_PID})
    with pytest.raises(profiles.ProfileMismatch):
        TR.translate_capture(str(other) + ".sigmf-data", d, tmp_path / "nope")


def test_the_experiment_reports_statistics_downstream_gap_and_hallucination(tiny, tmp_path):
    from atk_diffusion.experiments import translator_eval as TE
    _rf, d, _PA = tiny
    res = TE.run(None, translator=d, kinds=("bpsk", "ofdm"), window=64, n_pairs=24,
                 n_test=8, n_downstream_per=8, sample_steps=2, classifier_steps=10,
                 hallucination_trials=3, seed=1, out_dir=tmp_path / "a6")
    assert res["tier_translated"] == "invented"
    assert set(res["signal_statistics_vs_real_b"]) == {"raw A", "linear", "diffusion",
                                                       "synthetic + impairments"}
    acc = res["downstream_accuracy_on_real_b"]
    assert all(0.0 <= v <= 1.0 for v in acc.values()) and "real B (upper line)" in acc
    assert set(res["domain_gap"]) == set(acc) - {"real B (upper line)"}
    assert res["noise_fingerprints"]["real B"]["psd_distance_db"] == 0.0
    assert res["hallucination"]["eligible"] <= 3
    js = json.loads((tmp_path / "a6" / "translator_eval.json").read_text(encoding="utf-8"))
    assert js["result"]["to_profile"] == B_PID
    assert "domain gap" in (tmp_path / "a6" / "translator_eval_detail.md").read_text(encoding="utf-8")
