# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The 2D AI proposer (DETECTION_DESIGN §3, §6, §7; D3): datasets, the
thin-box rescue, training, the ONNX contract (ARCHITECTURE §5), the card."""

from __future__ import annotations

import json

import numpy as np
import pytest

from atk_diffusion import cards, profiles
from atk_diffusion.learn import common as K
from atk_diffusion.paths import RfData

import helpers_learn as H


# ---------------------------------------------------------------------------
# Pure (no PyTorch)
# ---------------------------------------------------------------------------
def test_box_policy_widens_thin_boxes_and_drops_negatives(rf):
    d = H.make_wideband(rf, splits={"train": 2}, with_negative=True)
    m = K.open_dataset(d, H.PROFILE, "wideband")
    pol = K.BoxPolicy(min_box_px=10)
    fam = K.family_map(m, pol.families)
    neg = K.negative_classes(m)
    boxes = np.array([[5, 50, 60, 53],       # 3 bins wide: widened to 10
                      [0, 10, 4, 40],         # 4 rows tall at the top edge
                      [0, 70, 64, 71]], np.float32)   # the spur (negative)
    famidx = np.array([m["families"].index("fm"), m["families"].index("burst"),
                       m["families"].index("unknown")])
    clsidx = np.array([m["classes"].index("nfm_voice"), m["classes"].index("adsb"),
                       m["classes"].index("spur")])
    b, lab, keep = K.prepare_boxes(boxes, famidx, clsidx, (64, 128), pol, fam, neg)
    assert keep.tolist() == [True, True, False]
    assert b[0, 3] - b[0, 1] == pytest.approx(10) and b[0, 1] == pytest.approx(46.5)
    assert b[1, 2] - b[1, 0] == pytest.approx(10) and b[1, 0] == 0.0  # kept inside
    assert [pol.families[i] for i in lab] == ["fm", "burst"]
    b2, lab2, keep2 = K.prepare_boxes(boxes, famidx, clsidx, (64, 128),
                                      K.BoxPolicy(min_box_px=10, negatives="unknown"),
                                      fam, neg)
    assert keep2.all() and pol.families[lab2[2]] == "unknown"


def test_a_dataset_of_another_profile_is_refused_in_words(rf):
    d = H.make_wideband(rf, splits={"train": 2})
    with pytest.raises(profiles.ProfileMismatch,
                       match="built for the RTL-SDR at 2.4 MS/s; this run is "
                             "for the HackRF One at 8 MS/s"):
        K.open_dataset(d, "hackrf_8000000_ci8", "wideband")
    with pytest.raises(K.DatasetRefused, match="needs a narrowband one"):
        K.open_dataset(d, H.PROFILE, "narrowband")
    (d / "train" / "tiles_000.npz").unlink()
    with pytest.raises(K.DatasetRefused, match="missing"):
        K.open_dataset(d, H.PROFILE, "wideband")


def test_a_changed_file_is_named_not_used(rf):
    d = H.make_wideband(rf, splits={"train": 2})
    f = d / "train" / "tiles_000.npz"
    rf.record(f, "dataset")
    K.open_dataset(d, H.PROFILE, "wideband", rf=rf)          # recorded, unchanged
    f.write_bytes(f.read_bytes() + b"\0")
    with pytest.raises(K.DatasetRefused, match="changed after it was recorded"):
        K.open_dataset(d, H.PROFILE, "wideband", rf=rf)


# ---------------------------------------------------------------------------
# PyTorch pieces
# ---------------------------------------------------------------------------
def test_tiles_dataset_items_flips_and_collate(rf):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from atk_diffusion.learn import data as D
    d = H.make_wideband(rf, splits={"train": 4})
    ds = D.WidebandTiles(d, "train", H.PROFILE, train=True, seed=3)
    img, t = ds[0]
    assert img.shape == (1, 64, 128) and img.dtype == torch.float32
    assert float(img.max()) <= ds.norm.clip_hi_db
    assert torch.equal(t["boxes"], t["boxes_rc"][:, [1, 0, 3, 2]])
    for i in range(len(ds)):
        _img, t = ds[i]
        rc = t["boxes_rc"].numpy()
        assert (rc[:, 0] >= 0).all() and (rc[:, 2] <= 64).all()
        assert (rc[:, 1] >= 0).all() and (rc[:, 3] <= 128).all()
        assert (rc[:, 2] - rc[:, 0] >= 10 - 1e-4).all()
    imgs, tgts = D.collate_tiles([ds[0], ds[1]])
    assert len(imgs) == 2 and len(tgts) == 2


def test_the_rescue_gives_a_thin_box_points_without_stealing():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from atk_diffusion.learn import proposer2d as P2
    # two levels: stride 8 (anchor size 8) and stride 16, on a 32×64 image
    a8 = [[x - 4, y - 4, x + 4, y + 4] for y in range(0, 32, 8) for x in range(0, 64, 8)]
    a16 = [[x - 8, y - 8, x + 8, y + 8] for y in range(0, 32, 16) for x in range(0, 64, 16)]
    anchors = torch.tensor(a8 + a16, dtype=torch.float32)
    gt = torch.tensor([[13.0, 2.0, 19.0, 30.0],      # thin and tall (x0, y0, x1, y1)
                       [30.0, 2.0, 50.0, 30.0]])     # already assigned below
    matched = torch.full((len(anchors),), -1, dtype=torch.int64)
    matched[len(a8)] = 1                             # a coarse point owns box 1
    out, n = P2.rescue_unassigned(gt, anchors, matched)
    assert n == 1
    got = torch.nonzero(out == 0).flatten().tolist()
    assert got and all(i < len(a8) for i in got)      # finest level only
    for i in got:
        cx = 0.5 * (anchors[i, 0] + anchors[i, 2])
        assert 13 < float(cx) < 19
    assert int(out[len(a8)]) == 1                     # not stolen


def test_no_resize_transform_refuses_another_geometry():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from atk_diffusion.learn import proposer2d as P2
    model = P2.build_model(tile_shape=(64, 128), width=8, fpn_channels=32,
                           head_convs=1).eval()
    with torch.no_grad():
        model([torch.zeros(1, 64, 128)])
        with pytest.raises(ValueError, match="trained on tiles of 64×128"):
            model([torch.zeros(1, 64, 96)])


def test_a_tall_thin_box_that_fcos_cannot_assign_is_rescued_in_the_loss():
    """200 rows tall -> FCOS sends it to P4 (stride 16); 8 bins wide between
    16 and 32 -> no P4 point inside it. Without the rescue it never trains."""
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from atk_diffusion.learn import proposer2d as P2
    model = P2.build_model(tile_shape=(256, 128), width=8, fpn_channels=32,
                           head_convs=1).train()
    tgt = [{"boxes": torch.tensor([[20.0, 0.0, 28.0, 200.0]]),
            "labels": torch.tensor([0])}]
    losses = model([torch.zeros(1, 256, 128)], tgt)
    assert model.head.rescued_boxes == 1
    vals = {k: float(v.detach()) for k, v in losses.items()}
    assert all(np.isfinite(v) for v in vals.values())
    assert vals["bbox_regression"] > 0               # it has positives now


# ---------------------------------------------------------------------------
# One trained model, many checks
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("onnxruntime")
    torch.set_num_threads(1)
    from atk_diffusion.learn import proposer2d as P2
    rf = RfData(tmp_path_factory.mktemp("p2d") / "rf_data", create=True)
    d = H.make_wideband(rf, splits={"train": 24, "val": 8, "test": 8}, seed=1)
    lines = []
    md = P2.train(rf, H.PROFILE, d, "ai_v1", epochs=20, batch_size=4, width=8,
                  fpn_channels=32, head_convs=1, threads=1, latency_repeats=3,
                  latency_threads=1, progress=lines.append)
    return rf, d, md, lines


def test_the_card_says_what_the_model_is_for(trained):
    rf, d, md, _lines = trained
    card = cards.load(md, expect_kind="proposer2d", for_profile=H.PROFILE)
    inp = card.input
    assert inp["tile"]["shape"] == [1, 1, 64, 128]
    assert inp["normalize"] == "none"                  # the host feeds raw dB
    assert inp["graph_normalize"]["applied_by"] == "graph"
    assert inp["families"][:3] == ["fm", "am", "fsk"]
    assert inp["boxes"]["order"] == ["row0", "bin0", "row1", "bin1"]
    assert inp["boxes"]["min_box_px"] == 10.0          # finest stride 8 + 2
    assert inp["stft"]["tile_rows"] == 64
    m = card.metrics
    for k in ("map50", "map50_95", "ap50_any", "latency_ms", "map_synthetic"):
        assert m[k] is not None, k
    assert m["onnx_check"]["ok"] is True
    assert m["latency"]["threads"] == 1 and "logical CPUs" in m["latency"]["machine"]
    assert card.datasets[0]["sha256"] == K.sha256_path(d / "manifest.json")
    assert card.weights["file"] == "model.onnx"
    assert card.weights["train_state"]["file"] == "model.pt"
    assert "a" in card.calibration["platt"] or card.calibration["platt"]["skipped"]
    assert 0.0 < card.calibration["operating_threshold"]["threshold"] < 1.0
    assert card.calibration["min_score"] == \
        card.calibration["operating_threshold"]["threshold"]
    assert card.tier == "proposed"
    assert {c["name"] for c in card.classes} == {"fm", "burst", "ofdm"}
    assert any("widened" in n for n in card.notes)


def test_it_learned_something(trained):
    _rf, _d, md, _lines = trained
    card = cards.load(md)
    print("held-out AP@0.5 (families aside):", card.metrics["ap50_any"],
          "mAP@0.5:", card.metrics["map50"])
    assert card.metrics["ap50_any"] >= 0.25, card.metrics


def test_onnx_io_is_the_contract_and_matches_pytorch(trained):
    torch = pytest.importorskip("torch")
    import onnxruntime as ort
    from atk_diffusion.learn import proposer2d as P2
    _rf, _d, md, _lines = trained
    sess = ort.InferenceSession(str(md / "model.onnx"),
                                providers=["CPUExecutionProvider"])
    ins = sess.get_inputs()
    assert [i.name for i in ins] == ["tile"] and ins[0].shape == [1, 1, 64, 128]
    assert [o.name for o in sess.get_outputs()] == ["boxes", "scores", "labels"]
    model, card = P2.load_torch(md, H.PROFILE)
    wrapper = P2.ProposerExport(model,
                                K.TileNorm.from_json(card.input["graph_normalize"]))
    rng = np.random.default_rng(7)
    lin = rng.exponential(1.0, (64, 128))
    lin[10:50, 40:90] += 30.0
    x = (10 * np.log10(lin)).astype(np.float32)[None, None]
    boxes, scores, labels = sess.run(None, {"tile": x})
    assert boxes.dtype == np.float32 and labels.dtype == np.int64
    assert boxes.shape[1] == 4 and len(scores) == len(labels) == len(boxes)
    with torch.no_grad():
        rb, rs, rl = (v.numpy() for v in wrapper(torch.from_numpy(x)))
    keep = rs >= 0.051
    assert np.allclose(rb[keep], boxes[scores >= 0.051], atol=1e-2)
    assert np.array_equal(rl[keep], labels[scores >= 0.051])
    if len(boxes):
        assert (boxes[:, 2] >= boxes[:, 0]).all() and (boxes[:, 2] <= 64).all()


def test_refusals(trained):
    pytest.importorskip("torch")
    from atk_diffusion.learn import export as X
    from atk_diffusion.learn import proposer2d as P2
    rf, d, md, _lines = trained
    with pytest.raises(profiles.ProfileMismatch, match="trained for the RTL-SDR"):
        X.OnnxRunner(md, "proposer2d", for_profile="bladerf1_4000000_ci16")
    with pytest.raises(FileExistsError, match="already exists"):
        P2.train(rf, H.PROFILE, d, "ai_v1", epochs=1)


def test_the_run_folder_tells_the_story(trained):
    rf, _d, _md, lines = trained
    runs = sorted(rf.runs(H.PROFILE).glob("*_proposer2d_ai_v1"))
    assert runs
    log = (runs[-1] / "run.log").read_text(encoding="utf-8")
    assert "training on the CPU" in log and "ONNX export verified" in log
    recs = [json.loads(x) for x in
            (runs[-1] / "metrics.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in recs if r.get("split") == "train"] == list(range(1, 21))
    assert (runs[-1] / "checkpoints" / "last.pt").exists()
    assert any("CPU latency" in x for x in lines)
    ok, why = rf.verify(_md / "model.onnx")
    assert ok, why


def test_atks_proposer_wrapper_loads_and_runs_it(trained):
    """Integration: detect.onnx_models.Proposer2D (another engineer's module)
    reads this card, checks the tile geometry against it, and runs it."""
    import importlib
    try:
        om = importlib.import_module("atk_diffusion.detect.onnx_models")
        st = importlib.import_module("atk_diffusion.dsp.stft")
    except ImportError as e:
        pytest.skip(f"detect.onnx_models / dsp.stft are not importable yet: {e}")
    _rf, d, md, _lines = trained
    prop = om.Proposer2D(md, H.PROFILE, threads=1)
    card = cards.load(md)
    geom = profiles.StftGeometry(**{k: v for k, v in card.input["stft"].items()
                                    if k in profiles.StftGeometry.__dataclass_fields__})
    ts = K.TileSet(d, "test", K.load_manifest(d))
    tile = st.Tile.from_spec(ts.spec(0), 2.4e6, 100e6, geom, profile=H.PROFILE)
    dets = prop.run(tile, min_score=0.05)
    assert all(x.sources == ("learned",) for x in dets)
    assert all(x.family in ("fm", "am", "fsk", "psk_qam", "ofdm", "burst",
                            "spread", "unknown") for x in dets)
    assert prop.min_score == card.calibration["min_score"]


def test_a_stopped_run_resumes_from_its_checkpoint(rf):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from atk_diffusion.learn import data as D
    from atk_diffusion.learn import proposer2d as P2
    d = H.make_wideband(rf, splits={"train": 4})
    ds = D.WidebandTiles(d, "train", H.PROFILE)

    class PowerCut(torch.utils.data.Subset):
        served = 0

        def __getitem__(self, i):
            PowerCut.served += 1
            if PowerCut.served > 6:                  # partway into epoch 2
                raise RuntimeError("power cut")
            return super().__getitem__(i)

        def __getitems__(self, idx):
            return [self.__getitem__(i) for i in idx]

    run = K.Run.start(rf, H.PROFILE, "resume_p2d")
    kw = dict(tile_shape=ds.shape, width=8, fpn_channels=32, head_convs=1)
    with pytest.raises(RuntimeError, match="power cut"):
        P2.fit(P2.build_model(**kw), PowerCut(ds, range(4)), 3, batch_size=2,
               run=run)
    ck = run.dir / "checkpoints" / "last.pt"
    assert torch.load(ck, weights_only=True)["epoch"] == 1
    hist = P2.fit(P2.build_model(**kw), ds, 3, batch_size=2, run=run,
                  resume_from=ck)
    assert [h["epoch"] for h in hist] == [1, 2, 3]
    assert "resumed from last.pt: epoch 1 done" in run.log_path.read_text()
