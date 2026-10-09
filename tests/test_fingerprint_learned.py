# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The learned fingerprint (plan C1/C2): a channel-resilient CNN on raw IQ
with an honest UNKNOWN, denoise-first measured, and the first experiment
— two same-model radios through random channels — end to end."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from atk_diffusion import cards, profiles
from atk_diffusion.experiments import fingerprint_eval as FE

PROFILE = "hackrf_8000000_ci8"
FS = 48_000.0


def test_windows_and_the_random_channel():
    from atk_diffusion.learn import fingerprint as LF
    radio = FE.same_model_pair(0, 1)[0]
    x = FE.simulate_burst(radio, np.random.default_rng(0), snr_db=25.0)
    w = LF.windows(x, 256, None, 6, FS)
    assert w.shape == (6, 2, 256) and w.dtype == np.float32
    assert np.allclose(np.mean(w[:, 0] ** 2 + w[:, 1] ** 2, axis=1), 1.0, atol=1e-4)
    c = LF.random_channel(w, np.random.default_rng(1), fs=FS)
    assert c.shape == w.shape
    assert np.allclose(np.mean(c[:, 0] ** 2 + c[:, 1] ** 2, axis=1), 1.0, atol=1e-4)
    assert not np.allclose(c, w)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from atk_diffusion.learn import fingerprint as LF
    radios = FE.same_model_pair(0, 3)
    kw = {"duration_s": 0.12}
    train = FE.bursts(radios[:2], 24, snrs=(15.0, 20.0, 30.0), seed=21, **kw)
    held = FE.bursts(radios[:2], 8, snrs=(15.0, 20.0, 30.0), seed=22, **kw)
    stranger = FE.bursts([FE.RadioModel("other-make", cfo_ppm=-1.6, ramp_ms=2.2,
                                        ramp_damping=0.9)], 8,
                         snrs=(15.0, 20.0, 30.0), seed=23, **kw)
    d = LF.train([x for x, _, _ in train], [i for _, i, _ in train],
                 tmp_path_factory.mktemp("fp") / "cnn", profile=PROFILE,
                 class_names=["radio-A", "radio-B"], steps=150, batch=48,
                 width=16, max_windows=6,
                 heldout=([x for x, _, _ in held], [i for _, i, _ in held]),
                 unknown=[x for x, _, _ in stranger])
    test = FE.bursts(radios[:2], 8, snrs=(15.0, 30.0), seed=24, **kw)
    return LF, d, test, stranger


def test_the_card_the_profile_and_the_open_set(trained):
    LF, d, test, stranger = trained
    card = cards.load(d, expect_kind="fingerprint", for_profile=PROFILE)
    assert card.tier == "proposed" and card.profile == PROFILE
    assert card.class_names() == ["radio-A", "radio-B"]
    assert card.metrics["train_loss_last"] < card.metrics["train_loss_first"]
    assert "unknown_rejection" in card.metrics
    with pytest.raises(profiles.ProfileMismatch, match="trained for the HackRF"):
        LF.load(d, "rtlsdr_2400000_cu8")
    model = LF.load(d, PROFILE)
    ids = model.identify([x for x, _, _ in test])
    assert all(r["tier"] == "proposed" for r in ids)
    acc = np.mean([r["class"] == i for r, (_, i, _) in zip(ids, test)])
    assert acc >= 0.75
    ev = LF.evaluate(model, [x for x, _, _ in test], [i for _, i, _ in test],
                     [s for _, _, s in test], unknown=[x for x, _, _ in stranger],
                     unknown_snr=[s for _, _, s in stranger])
    assert set(ev["per_snr"]) == {"15", "30"}
    assert 0.0 <= ev["unknown_rejection"] <= 1.0
    assert ev["per_snr"]["30"]["unknown_rejected"] is not None


def test_denoise_first_is_measured_with_its_hallucination_rate(trained):
    LF, d, test, _ = trained
    model = LF.load(d, PROFILE)

    def moving_average(x, fs):
        k = np.ones(5) / 5
        return np.convolve(x, k, mode="same").astype(np.complex64)

    def invents_a_burst(x, fs):
        y = np.array(x, copy=True)
        y[len(y) // 3: 2 * len(y) // 3] += 10.0          # "restores" a signal
        return y
    raw = LF.evaluate(model, [x for x, _, _ in test], [i for _, i, _ in test],
                      [s for _, _, s in test])
    den = LF.evaluate(model, [x for x, _, _ in test], [i for _, i, _ in test],
                      [s for _, _, s in test], denoiser=moving_average)
    assert den["denoiser"] == "moving_average"
    assert den["denoiser_hallucination_rate"] == 0.0
    assert set(den["per_snr"]) == set(raw["per_snr"])
    bad = LF.evaluate(model, [x for x, _, _ in test[:2]], [i for _, i, _ in test[:2]],
                      [s for _, _, s in test[:2]], denoiser=invents_a_burst)
    assert bad["denoiser_hallucination_rate"] == 1.0


def test_onnx_export_matches_torch(trained):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx", reason="ONNX export needs the onnx package")
    ort = pytest.importorskip("onnxruntime", reason="inference needs onnxruntime")
    LF, d, test, _ = trained
    path = LF.export_onnx(d, PROFILE)
    model = LF.load(d, PROFILE)
    w = LF.windows(test[0][0], model.window, None, model.max_windows, FS)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    lg_o, e_o = sess.run(None, {"iq": w})
    with torch.no_grad():
        lg_t, e_t = model.net(torch.from_numpy(w))
    assert np.allclose(lg_o, lg_t.numpy(), atol=1e-4)
    assert np.allclose(e_o, e_t.numpy(), atol=1e-4)
    card = cards.load(d, expect_kind="fingerprint")
    assert card.input["onnx"]["file"] == "fingerprint.onnx"


def test_the_first_experiment_end_to_end(rf):
    res = FE.run(rf, enrol_per_radio=6, test_per_radio=6, learned=False)
    cl = res["classical"]["per_snr"]
    assert set(cl) == {"5", "10", "15", "20", "30"}
    assert all(0.0 <= v["accuracy"] <= 1.0 for v in cl.values())
    assert cl["30"]["accuracy"] >= 0.5
    rep = Path(res["report"]["markdown"])
    text = rep.read_text()
    assert "can the system tell them apart" in text and "| 30 |" in text
    js = json.loads(Path(res["report"]["json"]).read_text())
    assert js["experiment"].startswith("C1/C2")
    assert rf.verify(rep)[0]
    assert (rf.products("emitters") / "library.json").exists()
