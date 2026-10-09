# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Products: GeoTIFF/GeoJSON in open formats, manifests, the write log, and
import into a new install (plan §3.7, D11)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from atk_diffusion.geo import products as P


def _grid():
    return P.GeoGrid(-77.60, 38.60, -77.40, 38.75, 40, 30)


def _array(grid, dtype=np.float32):
    lat, lon = grid.mesh()
    return ((lat - 38.6) * 100 + (lon + 77.6) * 10).astype(dtype)


# -- the grid -----------------------------------------------------------------
def test_grid_centres_sizes_and_bilinear_sampling():
    g = _grid()
    assert g.lats()[0] > g.lats()[-1]                       # row 0 is north
    assert np.isclose(g.lons()[0], -77.60 + 0.5 * g.dlon)
    dx, dy = g.cell_size_m()
    assert 400 < dx < 500 and 500 < dy < 600
    a = _array(g, np.float64)
    # a linear field is reproduced exactly by bilinear interpolation
    lat, lon = np.array([38.66, 38.70]), np.array([-77.51, -77.45])
    v = g.sample(a, lat, lon)
    assert np.allclose(v, (lat - 38.6) * 100 + (lon + 77.6) * 10, atol=1e-9)
    assert np.isnan(g.sample(a, 10.0, 10.0)[0])
    total = g.cell_area_m2().sum() * g.width
    lat0 = math.radians(38.675)
    expect = (0.15 * math.pi / 180 * P.EARTH_RADIUS_M) * \
        (0.20 * math.pi / 180 * P.EARTH_RADIUS_M * math.cos(lat0))
    assert abs(total / expect - 1) < 1e-3


def test_around_refuses_a_grid_that_would_exhaust_the_machine():
    g = P.GeoGrid.around(38.7, -77.5, 5000, 100)
    assert g.shape == (100, 100)
    with pytest.raises(P.ProductError, match="above the"):
        P.GeoGrid.around(38.7, -77.5, 200_000, 10)


# -- GeoTIFF ------------------------------------------------------------------
@pytest.mark.parametrize("engine", ["tifffile", "builtin"])
def test_geotiff_round_trip_and_geo_tags(tmp_path, engine):
    if engine == "tifffile":
        pytest.importorskip("tifffile")
    g = _grid()
    a = _array(g)
    a[3, 4] = -9999.0
    card = {"name": "m1", "kind": "radiomap", "metrics": {"rmse_db": 3.2}}
    p = P.write_geotiff(tmp_path / f"x_{engine}.tif", a, g.bounds,
                        nodata=-9999.0, cog=True, tile=16, engine=engine,
                        tags={"atk:tier": "inferred", "atk:card": card,
                              "atk:params": {"freq_hz": 146.52e6}})
    for read_engine in ("builtin", "tifffile"):
        if read_engine == "tifffile":
            pytest.importorskip("tifffile")
        r = P.read_geotiff(p, engine=read_engine)
        assert r.array.dtype == np.float32
        assert np.array_equal(r.array, a)
        assert np.allclose(r.bounds, g.bounds, atol=1e-12)
        assert r.nodata == -9999.0
        assert r.tags["atk:tier"] == "inferred"
        assert r.tags["atk:card"] == card                 # exact, via JSON
        assert r.tags["atk:params"]["freq_hz"] == 146.52e6
        assert r.overviews >= 1 and r.tiled
        assert np.isnan(r.masked()[3, 4])
    # the geo tags themselves, numerically, read with tifffile's own parser
    tifffile = pytest.importorskip("tifffile")
    with tifffile.TiffFile(str(p)) as tf:
        page = tf.pages[0]
        scale = page.tags[33550].value
        tie = page.tags[33922].value
        keys = page.tags[34735].value
        assert np.allclose(scale[:2], (g.dlon, g.dlat), rtol=0, atol=1e-15)
        assert np.allclose(tie, (0, 0, 0, -77.60, 38.75, 0), atol=1e-12)
        kd = {keys[4 + 4 * i]: keys[7 + 4 * i] for i in range(keys[3])}
        assert kd[1024] == 2 and kd[1025] == 1 and kd[2048] == 4326
        assert kd[2054] == 9102
        assert page.tags[42113].value.strip("\x00") == "-9999"
        meta = P.parse_gdal_metadata(page.tags[42112].value)
        assert meta["atk:tier"] == "inferred"
        assert json.loads(meta["atk:card"]) == card
        geo = tf.pages[0].geotiff_tags
        assert int(geo["GeographicTypeGeoKey"]) == 4326
        assert all(int(pg.subfiletype) & 1 for pg in tf.pages[1:])


def test_builtin_writer_puts_every_ifd_before_the_data(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    g = P.GeoGrid(0.0, 0.0, 1.0, 1.0, 64, 48)
    a = np.arange(64 * 48, dtype=np.uint16).reshape(48, 64)
    p = P.write_geotiff(tmp_path / "cog.tif", a, g, tags={"atk:tier": "measured"},
                        tile=16, engine="builtin")
    with tifffile.TiffFile(str(p)) as tf:
        ifds = [pg.offset for pg in tf.pages]
        first_data = min(min(pg.dataoffsets) for pg in tf.pages)
        assert len(ifds) >= 2 and max(ifds) < first_data
        # smallest overview's data first, full resolution's last (COG order)
        starts = [min(pg.dataoffsets) for pg in tf.pages]
        assert starts == sorted(starts, reverse=True)
        assert np.array_equal(tf.pages[0].asarray(), a)


def test_integer_overviews_are_not_averaged_and_floats_skip_nodata(tmp_path):
    g = P.GeoGrid(0.0, 0.0, 1.0, 1.0, 32, 32)
    m = np.zeros((32, 32), np.uint8)
    m[::2, ::2] = 1
    f = np.ones((32, 32), np.float32)
    f[0:2, 0:2] = [[-1, -1], [-1, 5]]
    lv = P._overview_levels(m, 16, 255)
    assert set(np.unique(lv[0])) <= {0, 1}
    lf = P._overview_levels(f, 16, -1.0)
    assert lf[0][0, 0] == 5.0                     # the one valid pixel


def test_a_product_without_a_tier_is_refused(tmp_path):
    g = _grid()
    with pytest.raises(P.ProductError, match="carries its tier"):
        P.write_geotiff(tmp_path / "x.tif", _array(g), g, tags={})
    with pytest.raises(ValueError):
        P.write_geotiff(tmp_path / "x.tif", _array(g), g,
                        tags={"atk:tier": "probably"})
    with pytest.raises(P.ProductError, match="one band"):
        P.write_geotiff(tmp_path / "x.tif", np.zeros((2, 3, 4)), g,
                        tags={"atk:tier": "measured"})


# -- GeoJSON ------------------------------------------------------------------
def test_geojson_tier_winding_and_longitude_first(tmp_path):
    sq = np.array([[-77.5, 38.6], [-77.5, 38.7], [-77.4, 38.7], [-77.4, 38.6]])
    feats = [P.point_feature(38.65, -77.45, {"name": "tower"}),
             P.multipolygon_feature([(sq, [])], {"level": 0.9,
                                                 "atk:tier": "inferred"})]
    p = P.write_geojson(tmp_path / "a.geojson", feats, tier="measured",
                        meta={"method": "test"})
    d = P.read_geojson(p)
    assert d["features"][0]["geometry"]["coordinates"] == [-77.45, 38.65]
    assert d["features"][0]["properties"]["atk:tier"] == "measured"
    assert d["features"][1]["properties"]["atk:tier"] == "inferred"
    ring = np.array(d["features"][1]["geometry"]["coordinates"][0][0])
    assert P._ring_area(ring) > 0                    # exterior counter-clockwise
    assert d["atk"]["method"] == "test"
    with pytest.raises(P.ProductError, match="no atk:tier"):
        P.write_geojson(tmp_path / "b.geojson",
                        [P.point_feature(0, 0, {"x": 1})])


# -- runs, manifests, the write log, import -----------------------------------
def test_product_run_manifest_write_log_and_import(rf, tmp_path):
    from atk_diffusion.paths import RfData
    g = _grid()
    run = P.ProductRun(rf, "coverage", label="test reach", tier="inferred",
                       params={"freq_hz": 146.52e6, "model": "fspl"},
                       method="free_space")
    run.add_geotiff("received_power_dbm.tif", _array(g), g, tier="inferred",
                    nodata=-9999.0, layer={"name": "physics", "role": "physics"})
    run.add_geojson("tx.geojson", [P.point_feature(38.7, -77.5, {})],
                    tier="measured")
    man_path = run.finish()
    man = json.loads(man_path.read_text())
    assert man["kind"] == "coverage" and man["tier"] == "inferred"
    assert man["params"]["model"] == "fspl"
    assert set(man["files"]) == {"received_power_dbm.tif", "tx.geojson"}
    assert man["layers"][0]["name"] == "physics"
    logged = rf.log.entries()
    for rel in list(man["files"]) + ["manifest.json"]:
        ok, why = rf.verify(run.dir / rel)
        assert ok, why
    assert any(k.endswith("received_power_dbm.tif") for k in logged)
    assert P.verify_run(run.dir) == (True, [])
    assert P.list_runs(rf, "coverage")[0][0] == run.dir

    # a new install: import from the old rf_data, verified
    rf2 = RfData(tmp_path / "new_install_rf", create=True)
    dest, words = P.import_run(run.dir, rf2)
    assert "imported 2 files" in words
    assert P.verify_run(dest) == (True, [])
    assert rf2.verify(dest / "tx.geojson")[0]
    again, words2 = P.import_run(run.dir, rf2)
    assert again == dest and "identical" in words2

    # tampering is named, and a tampered product is not imported
    (run.dir / "tx.geojson").write_text("{}")
    ok, problems = P.verify_run(run.dir)
    assert not ok and "changed after it was written" in problems[0]
    with pytest.raises(P.ProductError, match="will not be imported"):
        P.import_run(run.dir, RfData(tmp_path / "third", create=True))


def test_reopening_a_run_adds_layers(rf):
    g = _grid()
    run = P.ProductRun(rf, "coverage", run="r1", tier="inferred")
    run.add_geotiff("physics.tif", _array(g), g, tier="inferred",
                    layer={"name": "physics"})
    run.finish()
    again = P.ProductRun.open(rf, "coverage", "r1")
    again.add_geotiff("measured.tif", _array(g), g, tier="measured",
                      layer={"name": "measurement"})
    man = json.loads(again.finish().read_text())
    assert [l["name"] for l in man["layers"]] == ["physics", "measurement"]
    assert man["tiers"] == ["inferred", "measured"]


def test_kinds_and_run_names(rf):
    with pytest.raises(P.ProductError, match="not a product kind"):
        P.product_dir(rf, "maps", "x")
    with pytest.raises(P.ProductError, match="give it a run name"):
        P.product_dir(rf, "tracks")
    assert P.product_dir(rf, "emitters") == rf.root / "products" / "emitters"
    name = P.run_name("Reach from Bull Run!", when=0)
    assert name == "19700101T000000Z_Reach-from-Bull-Run"
