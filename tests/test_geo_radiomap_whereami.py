# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Radio maps from sparse samples (E2), the learned residual (E2/E5), and
where-am-I from the spectrum (E3) — classical and learned, each calibrated
and measured on held-out data."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from atk_diffusion import cards, provenance
from atk_diffusion.geo import products as P
from atk_diffusion.geo import radiomap as RM
from atk_diffusion.geo import whereami as W
from atk_diffusion.geo.products import GeoGrid
from atk_diffusion.geo.terrain import enu_m


def _field_world(seed=0, n=150):
    """A smooth field (correlated shadowing on a trend) sampled at n points."""
    rng = np.random.default_rng(seed)
    g = GeoGrid.around(38.7, -77.5, 2500, 50)
    shadow = W._Field(rng, 6.0, 400.0)

    def truth(lat, lon):
        e, nn = enu_m(lat, lon, 38.7, -77.5)
        return -70.0 - 0.004 * e + shadow(e, nn)
    lat = 38.7 + rng.uniform(-0.02, 0.02, n)
    lon = -77.5 + rng.uniform(-0.026, 0.026, n)
    vals = truth(lat, lon) + rng.normal(0, 1.0, n)
    return g, truth, RM.Measurements(lat, lon, vals, "rx_dbm")


# -- E2 classical ---------------------------------------------------------------
def test_idw_and_kriging_interpolate_and_kriging_knows_where_it_is_blind():
    g, truth, meas = _field_world()
    est, near = RM.idw(meas, g)
    assert est.shape == g.shape and np.all(np.isfinite(est))
    one = RM.Measurements([38.7], [-77.5], [-55.0])
    assert np.allclose(RM.idw_points(one, [38.71], [-77.49]), -55.0)
    k = RM.ordinary_kriging(meas, g)
    lat, lon = g.mesh()
    err_k = np.sqrt(np.mean((k.estimate - truth(lat, lon)) ** 2))
    err_i = np.sqrt(np.mean((est - truth(lat, lon)) ** 2))
    assert err_k < err_i + 0.5
    # kriging sigma: small beside a sample, large in an empty corner
    r, c = g.rowcol(meas.lat[0], meas.lon[0])
    assert k.sigma[int(round(r)), int(round(c))] < np.max(k.sigma) / 2
    assert k.variogram.range_m > 0 and k.variogram.sill > 0
    cv_k = RM.cross_validate(meas, "kriging")
    cv_i = RM.cross_validate(meas, "idw")
    assert cv_k["n"] == len(meas) and cv_k["rmse_db"] < cv_i["rmse_db"] + 0.5


def test_nested_variogram_finds_both_scales():
    rng = np.random.default_rng(2)
    h = rng.uniform(1, 4000, 4000)
    true = RM.Variogram("nested", 0.5, 3.0, 300.0, psill2=6.0, range2_m=3000.0)
    g = true(h) * rng.exponential(1.0, h.size)          # noisy semivariances
    lags, gam, cnt = RM.bin_pairs(h, g, 4000.0, 16, log=True)
    vg = RM.fit_variogram_curve(lags, gam, cnt, 9.5, "nested")
    assert 100 < vg.range_m < 900 and vg.range2_m > vg.range_m
    assert abs(vg.sill - 9.5) < 2.5


def test_radiomap_product_with_a_physics_layer(rf):
    g, truth, meas = _field_world(seed=3, n=80)
    lat, lon = g.mesh()
    physics = truth(lat, lon) + 4.0                     # physics 4 dB optimistic
    out = RM.radiomap_product(rf, meas, g, physics=(g, physics), run="drive-1")
    run = Path(out["run_dir"])
    man = json.loads((run / "manifest.json").read_text())
    assert man["kind"] == "radiomaps"
    tiers = {l["file"]: l["tier"] for l in man["layers"]}
    assert tiers["samples.geojson"] == "measured"
    assert tiers["measured_kriging.tif"] == "inferred"
    assert tiers["residual_kriging.tif"] == "inferred"
    assert "idw" in man["extra"]["summary"]["cross_validation"]
    assert -5.5 < out["residual"]["mean_db"] < -2.5
    sig = P.read_geotiff(run / "measured_kriging_sigma.tif")
    assert sig.tags["atk:units"] == "dB" and sig.tier == "inferred"


# -- E3 classical ---------------------------------------------------------------
@pytest.fixture(scope="module")
def small_world():
    world = W.DriveWorld(size_m=3000, n_cells=5, n_fm=2, seed=4)
    db = sum((world.drive(250, seed=s) for s in range(100, 105)), [])
    return world, db, world.drive(100, seed=200), world.drive(100, seed=300)


def test_drive_records_round_trip_through_csv(tmp_path, small_world):
    world, db_recs, _, _ = small_world
    keys = sorted({k for r in db_recs[:50] for k in r.features})
    p = tmp_path / "drive.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", "lat", "lon"] + keys)
        for r in db_recs[:50]:
            w.writerow([r.t, r.lat, r.lon] + [r.features.get(k, "") for k in keys])
    back = W.records_from_csv(p)
    assert len(back) == 50 and back[3].features == db_recs[3].features
    assert back[3].lat == pytest.approx(db_recs[3].lat)


def test_kernel_locator_finds_a_driven_place_and_calibration_helps(small_world):
    world, db_recs, cal, test = small_world
    db = W.FingerprintDB.from_records(db_recs)
    assert len(db) == len(db_recs) and all(np.isfinite(db.floors))
    loc = W.KernelLocator(db)
    r = db_recs[123]
    again = world.features(r.lat, r.lon, np.random.default_rng(9))[0]
    post = loc.locate(again)
    assert post.method == "fingerprint_kernel" and post.tier == "inferred"
    from atk_diffusion.geo.terrain import haversine_m
    assert float(haversine_m(*post.map_point(), r.lat, r.lon)) < 300.0
    before = np.mean([loc.log_density_at(x.features, x.lat, x.lon) for x in cal])
    loc.calibrate(cal)
    after = np.mean([loc.log_density_at(x.features, x.lat, x.lon) for x in cal])
    assert after >= before
    ev = loc.evaluate(test)
    assert ev["n"] == len(test) and set(ev["coverage"]) == {"0.5", "0.9", "0.95"}
    assert sum(ev["credible_level_histogram"]) == len(test)
    assert "held the truth" in ev["words"]


def test_map_locator_inverts_the_radio_map_honestly(rf, small_world):
    world, db_recs, cal, test = small_world
    db = W.FingerprintDB.from_records(db_recs)
    ml = W.MapLocator(db, max_points=150, max_cells=1600)
    ml.calibrate(cal)
    ev = ml.evaluate(test)
    assert ev["median_error_m"] < 400.0
    # honest: the 90 % region holds the truth at roughly its stated rate
    assert 0.7 <= ev["coverage"]["0.9"] <= 1.0
    post = ml.locate(test[0].features)
    run = Path(post.to_products(rf, kind="position", run="wai-test",
                                params={"calibration": ml.calibration}))
    man = json.loads((run / "manifest.json").read_text())
    assert man["kind"] == "position" and man["tier"] == "inferred"
    assert man["params"]["calibration"]["n"] == len(cal)


# -- E2/E5 learned residual -----------------------------------------------------
def test_learned_radiomap_card_baselines_hallucination_and_onnx(tmp_path):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from atk_diffusion.learn import radiomap as LR
    train = LR.synthetic_fields(48, 16, seed=1)
    held = LR.synthetic_fields(3, 16, seed=2)
    d = LR.train(train, tmp_path / "rm", steps=60, batch=8, channels=16,
                 heldout=held, seed=0)
    card = cards.load(d, expect_kind="radiomap")
    assert card.tier == "invented" and provenance.tier_for(LR.METHOD) == "invented"
    for k in ("rmse_model_db", "rmse_kriging_db", "rmse_physics_only_db",
              "hallucination_rate", "beats_kriging"):
        assert k in card.metrics
    assert 0.0 <= card.metrics["hallucination_rate"] <= 1.0
    model = LR.load(d)
    mean, std, draws = LR.sample(model, held["terrain"][0], held["physics"][0],
                                 held["mask"][0], held["samples"][0],
                                 n_samples=3, steps=8)
    assert mean.shape == (16, 16) and draws.shape == (3, 16, 16)
    assert np.all(std >= 0)
    corr, sig = LR.correct_grid(model, np.random.default_rng(0).normal(-90, 5, (23, 31)),
                                np.full((23, 31), 120.0), np.zeros((23, 31)),
                                np.zeros((23, 31)), n_samples=2, steps=6)
    assert corr.shape == (23, 31) and sig.shape == (23, 31)
    with pytest.raises(ValueError, match="16 x 16"):
        LR.sample(model, np.zeros((8, 8)), np.zeros((8, 8)), np.zeros((8, 8)),
                  np.zeros((8, 8)))
    # inference without PyTorch: the same sampler over onnxruntime
    pytest.importorskip("onnx", reason="ONNX export needs the onnx package")
    pytest.importorskip("onnxruntime", reason="inference needs onnxruntime")
    LR.export_onnx(d)
    m1, _, _ = LR.sample(LR.load(d), held["terrain"][1], held["physics"][1],
                         held["mask"][1], held["samples"][1], n_samples=2,
                         steps=6, seed=5)
    m2, _, _ = LR.sample_onnx(d, held["terrain"][1], held["physics"][1],
                              held["mask"][1], held["samples"][1], n_samples=2,
                              steps=6, seed=5)
    assert np.max(np.abs(m1 - m2)) < 0.05


# -- E3 learned -------------------------------------------------------------------
def test_learned_position_mdn_is_calibrated_and_carded(tmp_path, small_world):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from atk_diffusion.learn import position as LP
    world, db_recs, cal, test = small_world
    d = LP.train(db_recs, tmp_path / "pos", calib_records=cal, epochs=40,
                 components=3, hidden=32)
    card = cards.load(d, expect_kind="position")
    assert card.tier == "invented" and card.calibration["sigma_scale"] > 0
    assert card.metrics["train_nll_last"] < card.metrics["train_nll_first"]
    model = LP.load(d)
    post = model.locate(test[0].features)
    assert post.method == "learned_position" and post.tier == "invented"
    ev = model.evaluate(test[:40])
    assert ev["n"] == 40 and 0.0 <= ev["coverage"]["0.9"] <= 1.0
    assert np.isfinite(ev["median_error_m"])
    with pytest.raises(cards.CardRefusal):
        cards.load(d, expect_kind="radiomap")
