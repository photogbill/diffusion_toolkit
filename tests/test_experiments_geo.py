# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The map tracks' first experiments (experiments.geo_eval: E1, E3, E4, E5),
each run small: the coverage test must catch its two planted lies, the
locators report coverage with exact intervals, the aperture beats nothing it
should not, the residual is measured before and after, and the report is
written under the profile's runs folder and the write log."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from atk_diffusion.experiments import geo_eval as G


def test_clopper_pearson_is_exact_at_the_edges():
    assert G.clopper_pearson(0, 0) == (0.0, 1.0)
    lo, hi = G.clopper_pearson(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.35          # 1 - 0.025**(1/10) = 0.308
    lo, hi = G.clopper_pearson(10, 10)
    assert hi == 1.0 and 0.65 < lo < 0.75
    lo, hi = G.clopper_pearson(45, 50)
    assert lo < 0.9 < hi


def test_the_coverage_test_catches_both_planted_lies(rf):
    said = []
    r = G.df_coverage(50, failure_trials=12, max_cells=4000, seed=1,
                      progress=said.append)
    rows = {row["label"]: row for row in r["rows"]}
    assert len(rows) == 4 and r["passed"] is True
    honest = rows["stated sigma = true sigma"]
    assert honest["trials"] == 50 and not honest["expect_failure"]
    assert honest["intervals"]["0.9"][1] >= 0.9           # not caught lying
    for lie in ("sigma understated 3x", "shared 3 deg bias, unmodelled"):
        assert rows[lie]["expect_failure"] and rows[lie]["passed"]
        assert rows[lie]["intervals"]["0.9"][1] < 0.9      # the test saw it
    assert rows["shared 3 deg bias, modelled"]["passed"]
    assert said == ["coverage 'stated sigma = true sigma': 50/50"]
    assert r["words"].startswith("the stated regions are honest")
    rep = G.write_report(rf, r, name="e1")
    md = Path(rep["markdown"]).read_text("utf-8")
    assert "| sigma understated 3x | 12 |" in md and "(meant to fail)" in md
    assert json.loads(Path(rep["json"]).read_text("utf-8"))["passed"] is True
    assert Path(rep["dir"]).parent == rf.runs(G.DEFAULT_PROFILE) / "geo_eval"
    for k in ("json", "markdown"):
        assert rf.verify(rep[k]) == (True, "")
    again = G.write_report(rf, r, name="e1")          # never the same folder
    assert again["dir"] != rep["dir"]


def test_a_row_that_fails_is_said(rf):
    row = G._row("too small", {0.5: 1, 0.9: 2, 0.95: 2}, 20, [10.0], [1.0])
    assert row["passed"] is False and row["intervals"]["0.9"][1] < 0.9
    planted = G._row("planted", {0.5: 10, 0.9: 19, 0.95: 20}, 20, [], [],
                     expect_failure=True)
    assert planted["passed"] is False and planted["median_error_m"] is None


def test_the_coverage_test_refuses_no_trials():
    with pytest.raises(ValueError, match="at least one trial"):
        G.df_coverage(0)


def _world():
    from atk_diffusion.geo import whereami as W
    w = W.DriveWorld(size_m=2000, n_cells=4, n_fm=2, seed=3)
    return w.drive(160, seed=1) + w.drive(160, seed=2), w.drive(60, seed=5), \
        w.drive(60, seed=6)


def test_where_am_i_classical_locators_report_coverage(monkeypatch):
    db, cal, test = _world()
    said = []
    r = G.whereami_experiment(db, cal, test, learned=False, progress=said.append)
    assert r["synthetic"] is False and r["database_points"] == len(db)
    assert r["test_points"] == 60 and r["calibration_points"] == 60
    for m in ("kernel_uncalibrated", "kernel", "map_inversion_uncalibrated",
              "map_inversion"):
        res = r["methods"][m]
        assert res["n"] == 60 and np.isfinite(res["median_error_m"])
        for lv, (lo, hi) in res["coverage_intervals"].items():
            assert 0.0 <= lo <= res["coverage"][lv] <= hi <= 1.0
    assert "calibration" in r["methods"]["kernel"]
    assert said[0] == "where-am-I: database-point kernel"
    assert "mdn" not in r["methods"]


def test_where_am_i_synthetic_world_and_the_learned_mdn(monkeypatch, rf):
    torch = pytest.importorskip("torch", reason="PyTorch is only in the "
                                "training environment")
    torch.set_num_threads(1)
    from atk_diffusion import cards
    from atk_diffusion.geo import whereami as W
    real = W.DriveWorld.drive
    monkeypatch.setattr(W.DriveWorld, "drive",
                        lambda self, n, **kw: real(self, min(int(n), 50), **kw))
    r = G.whereami_experiment(seed=0, mdn_epochs=3, rf=rf)
    assert r["synthetic"] is True and r["database_points"] == 250
    mdn = r["methods"]["mdn"]
    assert mdn["epochs"] == 3 and mdn["n"] == 50
    assert set(mdn["coverage_intervals"]) == set(mdn["coverage"])
    # the trained model is kept beside the reports, never in a temp folder
    home = Path(mdn["model"])
    assert (rf.runs(G.DEFAULT_PROFILE) / "geo_eval") in home.parents
    assert cards.load(home).kind == "position"
    r2 = G.whereami_experiment(*_world(), seed=0, mdn_epochs=2)
    assert r2["methods"]["mdn"]["model"].startswith("a temporary folder")


def test_without_pytorch_the_learned_parts_say_skipped(monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "torch" else real(name, *a))
    db, cal, test = _world()
    r = G.whereami_experiment(db[:120], cal, test)
    assert r["methods"]["mdn"] == {"skipped": "PyTorch is not in this environment"}
    e5 = G.reach_residual_experiment(seed=1)
    assert e5["learned"] == "skipped: PyTorch is not in this environment"


def test_the_synthetic_aperture_against_the_parked_bearing():
    r = G.aperture_experiment(seed=0)
    assert r["synthetic"] is True and r["snapshots"] > 10
    assert r["dpd_self_calibrated_error_m"] < 2000
    assert r["parked_range"].startswith("none")
    assert 0.0 <= r["dpd_self_calibrated_truth_level"] <= 1.0
    assert r["words"]
    with pytest.raises(ValueError, match="known position"):
        G.aperture_experiment(snapshots=[object()], tower=None)


def test_predicted_reach_residual_before_and_after(rf):
    r = G.reach_residual_experiment(seed=0, learned=False, rf=rf)
    assert r["residual_after_kriging_rmse_db"] < r["residual_before_rmse_db"]
    run = Path(r["run_dir"])
    assert run.parent == rf.products("coverage") and (run / "manifest.json").exists()
    man = json.loads((run / "manifest.json").read_text("utf-8"))
    assert man["kind"] == "coverage" and "measured" in man["tiers"]
    rep = G.write_report(rf, r, name="e5")
    md = Path(rep["markdown"]).read_text("utf-8")
    assert "- **residual_before_rmse_db**" in md


def test_the_learned_residual_is_measured_with_its_budget(rf):
    torch = pytest.importorskip("torch", reason="PyTorch is only in the "
                                "training environment")
    torch.set_num_threads(1)
    r = G.reach_residual_experiment(seed=0, rf=rf, learned_steps=4,
                                    learned_fields=8, learned_size=16)
    assert r["learned_budget"] == {"steps": 4, "fields": 8, "size": 16}
    assert (rf.runs(G.DEFAULT_PROFILE) / "geo_eval") in Path(r["learned_model"]).parents
    assert np.isfinite(r["residual_after_learned_rmse_db"])
    assert 0.0 <= r["learned_hallucination_rate"] <= 1.0
    rep = G.write_report(rf, r, name="e5_learned")
    md = Path(rep["markdown"]).read_text("utf-8")
    assert "## learned_budget" in md and "- **steps**: 4" in md
