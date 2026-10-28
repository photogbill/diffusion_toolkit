# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""From the training library to models (RFMLS Training): validation and
test hold REAL windows only, copies train only when their parent trains, a
noise class is added, the verdict never calls a weak grade ready, and
retiring a model takes it out of use without deleting it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from atk_diffusion import cards
from atk_diffusion.library import build, dirty, store, train
from atk_diffusion.synth import datasets as D

from tests.test_library_store import PID, _box, _capture


def _two_types(rf, seconds=1.6):
    lib = store.TrainingLibrary(rf, PID)
    lib.new_type("fire", link="nfm_voice")
    lib.new_type("pagers", link="pocsag")
    for i in range(2):
        b, lab = _capture(rf, name=f"nfm{i}", seed=20 + i, seconds=seconds)
        lib.add_original_from_capture(b, _box(lab), "fire")
        b, lab = _capture(rf, name=f"poc{i}", cls="pocsag", offset=-120e3,
                          seed=40 + i, burst=(0.2, 1.0), seconds=seconds)
        lib.add_original_from_capture(b, _box(lab, burst=(0.2, 1.0)), "pagers")
    for t in ("fire", "pagers"):
        dirty.make_dirty(lib, t, 4, dirty.recipe("typical"), seed=1)
    return lib


def test_the_plan_says_what_is_wrong_before_anything_runs(rf):
    lib = store.TrainingLibrary(rf, PID)
    lib.new_type("fire", link="nfm_voice")
    b, lab = _capture(rf)
    lib.add_original_from_capture(b, _box(lab), "fire")
    p = build.plan(lib, "voice")
    assert p["usable"] == 1
    assert any("weak test" in w for w in p["warnings"])
    assert any("no copies" in w for w in p["warnings"])
    assert "nothing to train" in build.plan(lib, "spread")["warnings"][0]


def test_held_out_splits_are_real_only_and_copies_follow_their_parent(rf):
    lib = _two_types(rf)
    man = build.build_recognizer_dataset(rf, PID, "voice", "lib1", window=256,
                                         compute_scf=False)
    assert man["classes"] == ["fire", "pagers", "noise"]
    d = Path(man["path"])
    tiers = {}
    for split in D.SPLITS:
        for f in sorted((d / split).glob("shard_*.npz")):
            with np.load(f) as z:
                tiers.setdefault(split, set()).update(int(t) for t in z["tier"])
    assert tiers["train"] >= {0, 1, 4}
    assert tiers["test"] <= {0, 4} and tiers["val"] <= {0, 4}
    lib_block = man["library"]
    assert lib_block["test_sources"] == {"fire": 1, "pagers": 1}
    # a role flipped to test takes its copies out of training
    tr = [e for e in lib.examples("fire", "originals") if e.role == "train"][0]
    lib.set_role("fire", tr.id, "test")
    man2 = build.build_recognizer_dataset(rf, PID, "voice", "lib2", window=256,
                                          compute_scf=False, types=["pagers", "fire"])
    fire = man2["library"]["counts"].get("fire")
    assert fire is None          # fire now has no train original: left out
    assert any("left out" in w for w in man2["library"]["plan_warnings"])


def test_a_single_type_still_trains_against_noise_and_without_noise_refuses(rf):
    lib = store.TrainingLibrary(rf, PID)
    lib.new_type("fire", link="nfm_voice")
    b, lab = _capture(rf)
    lib.add_original_from_capture(b, _box(lab), "fire")
    man = build.build_recognizer_dataset(rf, PID, "voice", "one", window=256,
                                         compute_scf=False)
    assert man["classes"] == ["fire", "noise"]
    with pytest.raises(store.LibraryError, match="two answers"):
        build.build_recognizer_dataset(rf, PID, "voice", "one_b", window=256,
                                       compute_scf=False, noise_class=False)


def _fake_card(acc, per, beats=True, lat=3.0):
    c = cards.new_card("m", "classifier1d", PID)
    c.classes = [{"name": n, "source": "trained", "examples": 1}
                 for n in list(per) + ["noise"]]
    c.metrics = {"accuracy": acc, "per_class_accuracy": dict(per),
                 "beats_classical": beats, "latency_ms": lat,
                 "latency_budget_ms": 20.0}
    return c


def test_the_verdict_never_calls_a_weak_grade_ready():
    man = {"library": {"counts": {"a": {"test": {"real": 30}}, "b": {"test": {"real": 30}}},
                       "test_sources": {"a": 1, "b": 0}}}
    v = train.recognizer_verdict(_fake_card(0.97, {"a": 0.97, "b": 0.96}), man)
    assert v["level"] == "usable"
    assert any("weak test" in r for r in v["reasons"])
    man["library"]["test_sources"]["b"] = 2
    v = train.recognizer_verdict(_fake_card(0.97, {"a": 0.97, "b": 0.96}), man)
    assert v["level"] == "ready" and v["headline"].startswith("Ready")
    v = train.recognizer_verdict(_fake_card(0.97, {"a": 0.97, "b": 0.96},
                                            beats=False), man)
    assert v["level"] != "ready"
    v = train.recognizer_verdict(_fake_card(0.6, {"a": 0.9, "b": 0.3}), man)
    assert v["level"] == "not_ready" and any("Weakest: b" in r for r in v["reasons"])
    v = train.recognizer_verdict(_fake_card(None, {}), {})
    assert v["level"] == "unmeasured"


def test_train_recognizer_end_to_end_tiny_then_retire_and_bring_back(rf):
    torch = pytest.importorskip("torch", reason="PyTorch is only in the "
                                "training environment")
    torch.set_num_threads(1)
    _two_types(rf)
    lines = []
    res = train.train_recognizer(rf, PID, "voice", preset="quick", epochs=1,
                                 window=256, device="cpu", threads=1,
                                 train_cap_per_type=40, progress=lines.append)
    assert Path(res["model_dir"]).is_dir()
    assert res["verdict"]["level"] in ("ready", "usable", "not_ready")
    assert Path(res["report"]).exists()
    assert any("[100%]" in l for l in lines)
    ov = train.models_overview(rf, PID)
    rec = [m for m in ov if m["what"] == "recognizer"]
    assert len(rec) == 1 and rec[0]["in_use"] and rec[0]["canonical"] == "voice"
    assert rec[0]["verdict"]["per_type"]
    train.retire(rf, PID, res["model"])
    ov = train.models_overview(rf, PID)
    assert [m["retired"] for m in ov if m["what"] == "recognizer"] == [True]
    assert cards.find(rf, PID, "classifier1d") == []
    train.bring_back(rf, PID, res["model"])
    assert len(cards.find(rf, PID, "classifier1d")) == 1


def test_finder_scenes_hold_the_librarys_signals(rf):
    lib = _two_types(rf)
    man = build.build_finder_dataset(rf, PID, "find1", 3, scene_seconds=0.3,
                                     per_scene=(2, 2), seed=1)
    assert man["library"]["used"] and man["library"]["boxes_from_library"] > 0
    assert "fire" in man["classes"] and "pagers" in man["classes"]
    files = list(Path(man["path"]).rglob("tiles_*.npz"))
    assert files and not list(Path(man["path"]).rglob("*.sigmf-data"))
    seen = set()
    for f in files:
        with np.load(f) as z:
            seen.update(int(c) for c in z["box_class"])
    names = [man["classes"][i] for i in seen if i >= 0]
    assert {"fire", "pagers"} & set(names)


def test_train_finder_end_to_end_tiny(rf):
    torch = pytest.importorskip("torch", reason="PyTorch is only in the "
                                "training environment")
    pytest.importorskip("torchvision", reason="the finder is torchvision FCOS")
    torch.set_num_threads(1)
    _two_types(rf)
    res = train.train_finder(rf, PID, preset="quick", scenes=4, epochs=1,
                             device="cpu", threads=1, max_tiles=4)
    assert Path(res["model_dir"]).is_dir()
    assert res["verdict"]["level"] in ("ready", "usable", "not_ready", "unmeasured")
    assert "synthetic scenes" in res["verdict"]["reasons"][0]
    ov = train.models_overview(rf, PID)
    assert [m["what"] for m in ov] == ["finder"] and ov[0]["in_use"]
