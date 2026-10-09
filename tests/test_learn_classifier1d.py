# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The 1D classifier (DETECTION_DESIGN §4, §4.1, §6, §7): the dataset, the
receptive-field search, the cumulant baseline, training, the ONNX contract
(ARCHITECTURE §5), the card — and that ATK's own inference wrappers accept
what training writes."""

from __future__ import annotations

import importlib
import json

import numpy as np
import pytest

from atk_diffusion import cards
from atk_diffusion.learn import common as K
from atk_diffusion.paths import RfData

import helpers_learn as H


def _rect_psk(kind: str, n: int, rng) -> np.ndarray:
    if kind == "bpsk":
        return rng.choice([-1.0, 1.0], n).astype(np.complex128)
    return (rng.choice([-1, 1], n) + 1j * rng.choice([-1, 1], n)) / np.sqrt(2)


# ---------------------------------------------------------------------------
# Pure
# ---------------------------------------------------------------------------
def test_cumulants_match_the_textbook_and_bills_table():
    pytest.importorskip("torch")
    from atk_diffusion.learn import classifier1d as C1
    rng = np.random.default_rng(0)
    f = C1.cumulant_features(np.stack([_rect_psk("bpsk", 4096, rng),
                                       _rect_psk("qpsk", 4096, rng)]))
    # |C40|: BPSK 2, QPSK 1 (ATK csp/cumulants.py CYCLIC_TABLE); C42: -2, -1
    assert f[0, 1] == pytest.approx(2.0, abs=0.05)
    assert f[1, 1] == pytest.approx(1.0, abs=0.05)
    assert f[0, 3] == pytest.approx(-2.0, abs=0.05)
    assert f[1, 3] == pytest.approx(-1.0, abs=0.05)


def test_the_cumulant_baseline_separates_clean_psk():
    pytest.importorskip("torch")
    from atk_diffusion.learn import classifier1d as C1
    rng = np.random.default_rng(1)
    tr = np.stack([_rect_psk(k, 512, rng) for k in ["bpsk", "qpsk"] * 10])
    te = np.stack([_rect_psk(k, 512, rng) for k in ["bpsk", "qpsk"] * 5])
    y_tr = np.array([0, 1] * 10)
    y_te = np.array([0, 1] * 5)
    r = C1.cumulant_baseline(tr, y_tr, te, y_te, 2)
    assert r["accuracy"] == 1.0 and "cumulants" in r["method"]


def test_the_receptive_field_arithmetic_is_measured_not_asserted():
    """The analytic receptive field equals the span of input samples that
    move one output position (gradient support)."""
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from atk_diffusion.learn import classifier1d as C1
    for k, dil in ((3, (1, 1, 1, 1)), (5, (1, 2, 2, 4))):
        net = C1.ResNet1d(2, 4, k, dil).eval()
        # positive weights and a positive input keep every ReLU on, so the
        # gradient reaches every sample the wiring connects (with random
        # weights a switched-off ReLU can hide the outermost ones)
        with torch.no_grad():
            for mod in net.modules():
                if isinstance(mod, torch.nn.Conv1d):
                    mod.weight.fill_(1.0 / mod.weight[0].numel())
        x = torch.ones(1, 2, 2048, requires_grad=True)
        y = net.stages(net.stem(x))
        y[0, :, y.shape[2] // 2].sum().backward()
        nz = torch.nonzero(x.grad.abs().sum(dim=1)[0] > 0).flatten()
        span = int(nz.max() - nz.min() + 1)
        assert span == C1.receptive_field(k, dil), (k, dil)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def test_narrowband_items_labels_cycle_targets_and_unknowns(rf):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from atk_diffusion.learn import data as D
    d = H.make_narrowband(rf, per_class={"train": 6}, L=256)
    ds = D.NarrowbandDataset(d, "train", H.PROFILE,
                             classes=["ref_bpsk", "ref_qpsk"], window=128,
                             train=True, seed=2)
    it = ds[0]
    assert it["iq"].shape == (2, 128) and it["scf"].shape == (1, 8, 32)
    assert float(it["iq"].pow(2).sum(0).mean()) == pytest.approx(1.0, rel=1e-4)
    assert float(it["scf"].abs().max()) == pytest.approx(1.0)       # scf_norm max
    labs = ds.model_labels()
    assert set(labs.tolist()) == {-1, 0, 1}                          # 2fsk unknown
    i = int(np.argmax(ds.field("symbol_rate_hz") > 0))
    rec = ds[i]
    assert float(rec["cycle"][0]) == pytest.approx(
        ds.field("symbol_rate_hz")[i] / 48_000.0, rel=1e-5)
    assert rec["cycle_mask"].tolist() == [1.0, 1.0]
    known = D.NarrowbandDataset(d, "train", H.PROFILE,
                                classes=["ref_bpsk", "ref_qpsk"], known_only=True)
    assert len(known) == 12 and (known.model_labels() >= 0).all()
    assert sorted(known.order(np.random.default_rng(0))) == list(range(12))


def test_a_dataset_without_scf_says_what_to_do(rf):
    pytest.importorskip("torch")
    from atk_diffusion.learn import data as D
    d = H.make_narrowband(rf, per_class={"train": 2}, with_scf=False)
    with pytest.raises(K.DatasetRefused, match="use_scf=False"):
        D.NarrowbandDataset(d, "train", H.PROFILE)
    ds = D.NarrowbandDataset(d, "train", H.PROFILE, use_scf=False)
    assert ds[0]["scf"].shape == (1, 1, 1)


def test_iq_only_model_keeps_the_scf_input_in_its_graph(tmp_path):
    torch = pytest.importorskip("torch")
    ort = pytest.importorskip("onnxruntime")
    torch.set_num_threads(1)
    from atk_diffusion.learn import classifier1d as C1
    from atk_diffusion.learn import export as X
    m = C1.TwoBranchClassifier(3, iq_width=4, use_scf=False).eval()
    p = X.export(m, (torch.randn(2, 2, 64), torch.zeros(2, 1, 1, 1)),
                 tmp_path / "m.onnx", ("iq", "scf"), ("logits", "embedding", "cycle"),
                 dynamic_axes={"iq": {0: "b"}, "scf": {0: "b"}, "logits": {0: "b"},
                               "embedding": {0: "b"}, "cycle": {0: "b"}})
    s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    assert [i.name for i in s.get_inputs()] == ["iq", "scf"]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def test_receptive_field_search_ranks_configs(rf):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from atk_diffusion.learn import classifier1d as C1
    d = H.make_narrowband(rf, per_class={"train": 8, "val": 4}, L=128,
                          with_scf=False)
    cfgs = [{"name": "small", "kernel": 3, "dilations": [1, 1, 1, 1]},
            {"name": "wide", "kernel": 5, "dilations": [1, 2, 4, 8]}]
    lines = []
    r = C1.search_receptive_fields(rf, H.PROFILE, d, cfgs, epochs=1, iq_width=4,
                                   batch_size=8, device="cpu", threads=1,
                                   progress=lines.append)
    assert {x["name"] for x in r["results"]} == {"small", "wide"}
    assert r["best"]["name"] in ("small", "wide")
    assert all(0.0 <= x["val_accuracy"] <= 1.0 for x in r["results"])
    wide = next(x for x in r["results"] if x["name"] == "wide")
    assert wide["receptive_field"] == C1.receptive_field(5, [1, 2, 4, 8])
    assert len(lines) == 2 and "receptive-field search" in lines[0]


# ---------------------------------------------------------------------------
# One trained model, many checks
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    torch.set_num_threads(1)
    from atk_diffusion.learn import classifier1d as C1
    rf = RfData(tmp_path_factory.mktemp("c1d") / "rf_data", create=True)
    d = H.make_narrowband(rf, classes=("ref_bpsk", "ref_qpsk", "ref_2fsk", "ref_am"),
                          per_class={"train": 32, "val": 12, "test": 12}, seed=4)
    lines = []
    md = C1.train(rf, H.PROFILE, d, "amc_v1", epochs=16,
                  rf_config={"name": "k5", "kernel": 5, "dilations": [1, 1, 1, 1]},
                  held_out_classes=("ref_am",), iq_width=8, scf_width=8,
                  batch_size=16, threads=1, latency_repeats=3, latency_threads=1,
                  progress=lines.append)
    return rf, d, md, lines


def test_the_card_carries_what_inference_needs(trained):
    _rf, d, md, _lines = trained
    card = cards.load(md, expect_kind="classifier1d", for_profile=H.PROFILE)
    inp = card.input
    assert inp["iq_len"] == 256 and inp["iq_norm"] == "rms"
    assert inp["scf_norm"] == "max" and inp["scf_shape"] == [8, 32]
    assert inp["canonical"]["cls"] == inp["canonical"]["class"] == "voice"
    assert inp["canonical"]["rate"] == 48_000.0 and inp["cycle_scale"] == [48_000.0] * 2
    assert inp["receptive_field"]["samples"] > 0
    assert card.class_names() == ["ref_bpsk", "ref_qpsk", "ref_2fsk"]
    assert isinstance(card.calibration["temperature"], float)
    fit = card.calibration["temperature_fit"]
    # temperature scaling minimises NLL (T = 1 is a candidate, so NLL cannot
    # rise); ECE is reported, not guaranteed — on a small set it can rise
    assert fit["nll_after"] <= fit["nll_before"] + 1e-9
    assert {"ece_before", "ece_after", "ece_heldout_before",
            "ece_heldout_after"} <= set(fit)
    m = card.metrics
    assert 0.0 <= m["accuracy"] <= 1.0 and m["accuracy_vs_snr"]
    assert m["classical_baseline"]["n"] == 36
    assert m["unknown_rejection"] is not None
    assert card.calibration["open_set"]["n_unknown"] == 24       # ref_am, val + test
    assert m["onnx_check"]["ok"] and m["latency_budget_ms"] == 20.0
    assert card.datasets[0]["sha256"] == K.sha256_path(d / "manifest.json")


def test_it_learned_something(trained):
    _rf, _d, md, _lines = trained
    m = cards.load(md).metrics
    print("held-out accuracy", m["accuracy"], "cumulant baseline",
          m["classical_baseline"]["accuracy"])
    assert m["accuracy"] >= 0.6                     # chance is 1/3


def test_onnx_contract_dynamic_batch_and_unit_embeddings(trained):
    ort = pytest.importorskip("onnxruntime")
    _rf, _d, md, _lines = trained
    s = ort.InferenceSession(str(md / "model.onnx"), providers=["CPUExecutionProvider"])
    assert [i.name for i in s.get_inputs()] == ["iq", "scf"]
    assert [o.name for o in s.get_outputs()] == ["logits", "embedding", "cycle"]
    for b in (1, 5):
        rng = np.random.default_rng(b)
        lo, emb, cyc = s.run(None, {"iq": rng.normal(size=(b, 2, 256)).astype(np.float32),
                                    "scf": rng.random((b, 1, 8, 32)).astype(np.float32)})
        assert lo.shape == (b, 3) and emb.shape == (b, 64) and cyc.shape == (b, 2)
        assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)


def test_the_run_log_and_the_refusals(trained):
    pytest.importorskip("torch")
    from atk_diffusion.learn import classifier1d as C1
    rf, d, _md, lines = trained
    text = "\n".join(lines)
    assert "held out for unknown rejection: ref_am" in text
    assert "temperature" in text and "cumulant baseline" in text
    with pytest.raises(K.DatasetRefused, match="not in the dataset"):
        C1.train(rf, H.PROFILE, d, "x", held_out_classes=("lora",))
    with pytest.raises(FileExistsError):
        C1.train(rf, H.PROFILE, d, "amc_v1", epochs=1)


def test_atks_classifier_wrapper_loads_and_runs_it(trained):
    """Integration: detect.onnx_models.Classifier1D (another engineer's
    module) reads this card and runs this graph."""
    try:
        om = importlib.import_module("atk_diffusion.detect.onnx_models")
    except ImportError as e:
        pytest.skip(f"detect.onnx_models is not importable yet: {e}")
    _rf, d, md, _lines = trained
    clf = om.Classifier1D(md, H.PROFILE, threads=1)
    sh = K.ShardSet(d, "test", K.load_manifest(d))
    win = np.stack([sh.iq(i) for i in range(4)])
    scf = np.stack([sh.scf(i) for i in range(4)])
    out = clf.run(win, scf)
    assert out.probs.shape == (4, 3) and len(out.top) == 4
    assert out.cycle_hz.shape == (4, 2)
    assert not any("no temperature" in n for n in out.notes)


def test_the_bank_beside_the_card_loads_in_atk(trained):
    try:
        pm = importlib.import_module("atk_diffusion.detect.prototypes")
    except ImportError as e:
        pytest.skip(f"detect.prototypes is not importable yet: {e}")
    _rf, _d, md, _lines = trained
    bank = pm.PrototypeBank.load(md, H.PROFILE, "voice")
    assert bank.classes == ["ref_2fsk", "ref_bpsk", "ref_qpsk"]
    assert all(bank.is_calibrated(c) for c in bank.classes)
    info = json.loads((md / "card.json").read_text())["calibration"]["open_set"]
    assert info["implementation"].endswith("PrototypeBank")


def test_a_stopped_run_resumes_from_its_checkpoint(rf):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from atk_diffusion.learn import classifier1d as C1
    from atk_diffusion.learn import data as D
    d = H.make_narrowband(rf, per_class={"train": 4}, with_scf=False, L=128)
    ds = D.NarrowbandDataset(d, "train", H.PROFILE, use_scf=False)

    class PowerCut(torch.utils.data.Subset):
        served = 0

        def __getitem__(self, i):
            PowerCut.served += 1
            if PowerCut.served > 16:                 # partway into epoch 2
                raise RuntimeError("power cut")
            return super().__getitem__(i)

        def __getitems__(self, idx):
            return [self.__getitem__(i) for i in idx]

    run = K.Run.start(rf, H.PROFILE, "resume_c1d")
    new = lambda: C1.TwoBranchClassifier(3, iq_width=4, use_scf=False)  # noqa: E731
    with pytest.raises(RuntimeError, match="power cut"):
        C1.fit(new(), PowerCut(ds, range(len(ds))), 3, batch_size=4, run=run)
    ck = run.dir / "checkpoints" / "last.pt"
    assert torch.load(ck, weights_only=True)["epoch"] == 1
    hist = C1.fit(new(), ds, 3, batch_size=4, run=run, resume_from=ck)
    assert [h["epoch"] for h in hist] == [1, 2, 3]
    with pytest.raises(ValueError, match="not a training checkpoint"):
        bad = run.dir / "bad.pt"
        torch.save({"model": {}}, bad)
        C1.fit(new(), ds, 1, resume_from=bad)
