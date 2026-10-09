# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Calibration (DETECTION_DESIGN §6.4, §4 open set) and self-supervised
pretraining (§6.1): the fits recover known answers, the open set rejects
what it never saw, and pretrained backbones run, load and are refused in
words when they do not fit."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from atk_diffusion import cards, profiles
from atk_diffusion.learn import calibrate as CAL
from atk_diffusion.learn import common as K

import helpers_learn as H


# ---------------------------------------------------------------------------
# Calibration — pure numpy/scipy
# ---------------------------------------------------------------------------
def test_ece_is_small_when_calibrated_and_large_when_overconfident():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.0, 1.0, 20_000)
    y = rng.uniform(size=p.size) < p                 # right exactly p of the time
    assert CAL.ece(p, y) < 0.02
    assert CAL.ece(np.clip(p + 0.3, 0, 1), y) > 0.2   # says more than it knows


def test_temperature_recovers_a_known_temperature():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 4.0, size=(6000, 4))
    p = K.softmax(z / 2.5)
    y = np.array([rng.choice(4, p=row) for row in p])
    r = CAL.fit_temperature(z, y)
    assert r["temperature"] == pytest.approx(2.5, rel=0.08)
    assert r["nll_after"] < r["nll_before"]
    assert r["ece_after"] < r["ece_before"]
    assert np.argmax(z / r["temperature"], 1).tolist() == np.argmax(z, 1).tolist()


def test_temperature_refuses_unknown_labels():
    with pytest.raises(ValueError, match="outside the logits"):
        CAL.fit_temperature(np.zeros((3, 2)), np.array([0, 1, -1]))


def test_platt_fixes_low_running_detector_scores():
    """FCOS-style scores run low: right 95 % of the time at 0.4. Temperature
    alone cannot lift 0.4 above 0.5; Platt's bias can."""
    rng = np.random.default_rng(2)
    s = rng.uniform(0.05, 0.6, 8000)
    true_p = 1 / (1 + np.exp(-(2.0 * np.log(s / (1 - s)) + 2.5)))
    y = rng.uniform(size=s.size) < true_p
    r = CAL.fit_platt(s, y)
    assert r["a"] == pytest.approx(2.0, rel=0.15)
    assert r["b"] == pytest.approx(2.5, rel=0.15)
    assert r["ece_after"] < 0.03 < r["ece_before"]
    assert CAL.apply_platt(np.array([0.4]), r["a"], r["b"])[0] > 0.8


def _clusters(rng, centres, n, spread=0.15):
    out = {}
    for name, c in centres.items():
        e = c[None] + spread * rng.normal(size=(n, len(c)))
        out[name] = e / np.linalg.norm(e, axis=1, keepdims=True)
    return out


@pytest.mark.parametrize("prefer", ["builtin", "bank"])
def test_open_set_rejects_what_it_never_saw(prefer, tmp_path):
    if prefer == "bank":
        try:
            importlib.import_module("atk_diffusion.detect.prototypes")
        except ImportError as e:
            pytest.skip(f"detect.prototypes is not importable yet: {e}")
    rng = np.random.default_rng(3)
    eye = np.eye(16)
    c = {"dmr": eye[0], "p25": eye[1]}
    train = _clusters(rng, c, 60)
    held = _clusters(rng, c, 40)
    test = _clusters(rng, c, 40)
    unknown = _clusters(rng, {"x": eye[2]}, 50)["x"]
    r = CAL.open_set(H.PROFILE, "voice", train, held, unknown_emb=unknown,
                     known_test=test, out_dir=tmp_path, prefer=prefer)
    assert r["unknown_rejection"] > 0.95
    assert r["false_unknown_rate"] < 0.2
    assert r["accuracy_on_accepted"] == 1.0
    assert set(r["thresholds"]) == {"dmr", "p25"}
    assert (tmp_path / ("prototypes.json" if prefer == "bank"
                        else "prototypes_builtin.json")).exists()


# ---------------------------------------------------------------------------
# Self-supervised pieces
# ---------------------------------------------------------------------------
def test_augmentations_keep_shape_power_and_nt_xent_prefers_agreement():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from atk_diffusion.learn import ssl as S
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 2, 128)
    v = S.augment_iq(x, 48_000.0, 480.0, 16, (0, 20), g)
    assert v.shape == x.shape
    assert torch.allclose(v.pow(2).sum(1).mean(1), torch.ones(8), atol=1e-4)
    z = torch.nn.functional.normalize(torch.randn(8, 16, generator=g), dim=1)
    other = torch.nn.functional.normalize(torch.randn(8, 16, generator=g), dim=1)
    assert float(S.nt_xent(z, z)) < float(S.nt_xent(z, other))


def test_patch_mask_covers_the_asked_fraction():
    torch = pytest.importorskip("torch")
    from atk_diffusion.learn import ssl as S
    m = S.patch_mask(4, (64, 128), 8, 0.5, torch.Generator().manual_seed(1))
    assert m.shape == (4, 1, 64, 128)
    assert float(m.mean()) == pytest.approx(0.5, abs=0.01)


def test_pretrain_2d_runs_measures_and_loads_into_a_proposer(rf):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from atk_diffusion.learn import proposer2d as P2
    from atk_diffusion.learn import ssl as S
    d = H.make_wideband(rf, splits={"train": 12, "val": 4})
    md = S.pretrain_2d(rf, H.PROFILE, "msm_v1", dataset_dir=d, epochs=2,
                       batch_size=4, width=8, threads=1)
    card = cards.load(md, expect_kind="ssl_backbone", for_profile=H.PROFILE)
    assert card.input["branch"] == "2d" and card.tier == "measured"
    assert card.input["normalize"]["applied_by"] == "host"
    m = card.metrics
    assert m["masked_mse"] > 0 and m["trivial_mse"] > 0 and m["classical_fill_mse"] > 0
    model = P2.build_model(tile_shape=(64, 128), width=8, fpn_channels=32,
                           head_convs=1)
    rep = S.load_backbone_into(model, md, profile=H.PROFILE)
    assert rep["target"] == "ThinResNet2d" and rep["loaded"] > 10
    state = torch.load(md / "backbone.pt", weights_only=True)
    for k, v in model.backbone.body.state_dict().items():
        assert torch.equal(v, state[k]), k
    with pytest.raises(cards.CardRefusal, match="architectures must match"):
        S.load_backbone_into(P2.build_model(tile_shape=(64, 128), width=16), md)
    with pytest.raises(profiles.ProfileMismatch):
        S.load_backbone_into(model, md, profile="hackrf_8000000_ci8")
    from atk_diffusion.learn import classifier1d as C1
    with pytest.raises(cards.CardRefusal, match="no 2d backbone"):
        S.load_backbone_into(C1.TwoBranchClassifier(3, iq_width=8), md)


def test_a_proposer_trains_from_the_pretrained_backbone(rf):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from atk_diffusion.learn import proposer2d as P2
    from atk_diffusion.learn import ssl as S
    d = H.make_wideband(rf, splits={"train": 8, "val": 4, "test": 4})
    sd = S.pretrain_2d(rf, H.PROFILE, "msm_for_p", dataset_dir=d, epochs=1,
                       batch_size=4, width=8, threads=1)
    lines = []
    md = P2.train(rf, H.PROFILE, d, "p_from_ssl", epochs=1, batch_size=4,
                  width=16, fpn_channels=32, head_convs=1, pretrained=sd,
                  threads=1, latency_repeats=2, latency_threads=1,
                  progress=lines.append)
    card = cards.load(md)
    assert card.input["arch"]["width"] == 8          # adopted from the backbone
    ssl_norm = cards.load(sd).input["normalize"]
    assert card.input["graph_normalize"]["mean_db"] == ssl_norm["mean_db"]
    assert any("pretrained self-supervised" in n for n in card.notes)
    assert any("adopted from the pretrained backbone" in x for x in lines)


def test_pretrain_1d_from_a_dataset_probes_and_loads_into_a_classifier(rf):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from atk_diffusion.learn import classifier1d as C1
    from atk_diffusion.learn import ssl as S
    d = H.make_narrowband(rf, per_class={"train": 16, "val": 8}, L=256,
                          with_scf=False)
    cfg = {"name": "k5", "kernel": 5, "dilations": [1, 1, 1, 1]}
    md = S.pretrain_1d(rf, H.PROFILE, "clr_v1", dataset_dir=d, rf_config=cfg,
                       width=8, epochs=2, batch_size=16, threads=1)
    card = cards.load(md, expect_kind="ssl_backbone")
    assert card.input["branch"] == "1d" and card.input["canonical"]["rate"] == 48_000.0
    m = card.metrics
    assert m["knn_probe_accuracy"] is not None and m["knn_probe_untrained"] is not None
    assert -1.0 <= m["negative_cosine"] <= m["positive_cosine"] <= 1.0
    model = C1.TwoBranchClassifier(3, cfg, iq_width=8, use_scf=False)
    rep = S.load_backbone_into(model, md, canonical_rate=48_000.0)
    assert rep["target"] == "ResNet1d"
    with pytest.raises(cards.CardRefusal, match="pretrained on cuts at 48000"):
        S.load_backbone_into(model, md, canonical_rate=480_000.0)
    # the classifier adopts the backbone's architecture and trains from it
    lines = []
    cd = C1.train(rf, H.PROFILE, d, "clf_from_ssl", epochs=1, pretrained=md,
                  iq_width=32, use_scf=False, batch_size=16, threads=1,
                  latency_repeats=2, latency_threads=1, progress=lines.append)
    arch = cards.load(cd).input["arch"]
    assert arch["iq_width"] == 8 and arch["rf_config"]["kernel"] == 5
    assert any("no receptive-field search" in x for x in lines)


def test_pretrain_1d_from_a_raw_capture_cuts_at_the_canonical_rate(rf):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from atk_diffusion.learn import ssl as S
    cap = H.make_capture(rf, seconds=0.06)
    md = S.pretrain_1d(rf, H.PROFILE, "clr_cap", captures=[cap], window=128,
                       cuts_per_capture=24, width=4, epochs=1, batch_size=12,
                       threads=1)
    card = cards.load(md)
    assert card.input["canonical"] == {"class": "voice", "cls": "voice",
                                       "rate": 48_000.0, "decimation": 50}
    assert card.datasets[0]["kind"] == "capture" and card.metrics["cuts"] == 24
    other = H.make_capture(rf, profile="hackrf_8000000_ci8", seconds=0.002,
                           name="hack")
    with pytest.raises(profiles.ProfileMismatch):
        S.pretrain_1d(rf, H.PROFILE, "clr_bad", captures=[other], window=64,
                      cuts_per_capture=2, epochs=1, threads=1)


def test_pretrain_2d_from_raw_captures_through_the_front_end(rf):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    try:
        importlib.import_module("atk_diffusion.dsp.stft")
    except ImportError as e:
        pytest.skip(f"dsp.stft (the front end) is not importable yet: {e}")
    from atk_diffusion.learn import ssl as S
    cap = H.make_capture(rf, seconds=0.05)
    geom = profiles.StftGeometry(fft_size=128, hop=128, window="hann",
                                 tile_seconds=0.004, tile_rows=32,
                                 tile_overlap=0.25)
    md = S.pretrain_2d(rf, H.PROFILE, "msm_cap", captures=[cap], stft=geom,
                       epochs=1, batch_size=4, width=8, threads=1)
    card = cards.load(md)
    assert card.input["tile_shape"] == [32, 128]
    assert card.input["stft"]["fft_size"] == 128
    assert card.metrics["heldout_tiles"] >= 1
