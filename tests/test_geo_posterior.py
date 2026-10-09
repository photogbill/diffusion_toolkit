# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Geolocation as a cloud: the bearing posterior, its lobes, its HPD
regions, and the shared-bias lesson from ATK's own coverage audit (E1)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from atk_diffusion.geo import contours as C
from atk_diffusion.geo import posterior as PO
from atk_diffusion.geo import products as P
from atk_diffusion.geo import terrain as T
from atk_diffusion.geo.products import GeoGrid

TRUTH = (38.70, -77.50)


def _from(lat, lon, sigma=1.0, bias=0.0, rng=None, **kw):
    b = float(T.initial_bearing_deg(lat, lon, *TRUTH)) + bias
    if rng is not None:
        b += rng.normal(0, sigma)
    return PO.Bearing(lat, lon, b % 360.0, sigma, **kw)


# -- contours -------------------------------------------------------------------
def test_hpd_threshold_area_and_polygon_of_a_gaussian():
    g = GeoGrid.around(38.7, -77.5, 3000, 20)
    lat, lon = g.mesh()
    e, n = T.enu_m(lat, lon, 38.7, -77.5)
    s = 400.0
    dens = np.exp(-0.5 * (e ** 2 + n ** 2) / s ** 2)
    prob = dens * g.cell_area_m2()
    prob /= prob.sum()
    regs = C.hpd_regions(g, dens, prob, (0.5, 0.9))
    for r in regs:
        want = math.pi * s * s * (-2 * math.log(1 - r.level)) / 1e6
        assert abs(r.area_km2 / want - 1) < 0.03
        assert r.lobes == 1
        assert C.point_in_polygons(38.7, -77.5, r.polygons)
        traced = C.ring_area_m2(r.polygons[0][0]) / 1e6
        assert abs(traced / want - 1) < 0.05
    assert C.credible_level(dens, prob, float(dens.max())) < 0.01
    with pytest.raises(ValueError):
        C.hpd_threshold(dens, prob, 0.0)


def test_point_in_polygons_respects_holes():
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], float)
    hole = np.array([[4, 4], [6, 4], [6, 6], [4, 6], [4, 4]], float)
    polys = [(sq, [hole])]
    assert C.point_in_polygons(2.0, 2.0, polys)
    assert not C.point_in_polygons(5.0, 5.0, polys)
    assert not C.point_in_polygons(5.0, 12.0, polys)


# -- the posterior --------------------------------------------------------------
def test_two_crossing_bearings_give_the_textbook_region():
    a = T.destination(*TRUTH, 180.0, 8000.0)
    b = T.destination(*TRUTH, 270.0, 6000.0)
    sig = 1.5
    post = PO.locate([_from(*a, sigma=sig), _from(*b, sigma=sig)],
                     search_radius_m=20_000)
    la, lo = post.map_point()
    assert T.haversine_m(la, lo, *TRUTH) < 25.0
    r90 = [r for r in post.regions() if r.level == 0.9][0]
    s1, s2 = 8000 * math.radians(sig), 6000 * math.radians(sig)
    want = math.pi * s1 * s2 * (-2 * math.log(0.1)) / 1e6
    assert abs(r90.area_km2 / want - 1) < 0.15
    assert r90.lobes == 1 and not post.unbounded
    assert post.credible_level(*TRUTH) < 0.2
    cov = post.covariance_m()
    assert abs(math.sqrt(cov[0, 0]) / s1 - 1) < 0.15      # east spread: from a
    slat, slon = post.samples(4000, np.random.default_rng(0))
    assert T.haversine_m(slat.mean(), slon.mean(), *post.mean_point()) < 30


def test_a_bearing_and_a_soft_range_make_two_lobes():
    a = T.destination(*TRUTH, 200.0, 7000.0)
    station = PO.Bearing(*a, 20.0, 1.0)            # passes through TRUTH
    # a receiver east of the bearing line heard the emitter at a strength that
    # says ~4 km: the ring crosses the bearing line twice, 4 km apart
    rx = T.destination(*TRUTH, 80.0, 4000.0)
    rng = PO.SoftRange(*rx, median_m=4000.0, sigma_log10=0.01)
    post = PO.locate([station], [rng], search_radius_m=15_000)
    r90 = [r for r in post.regions() if r.level == 0.9][0]
    assert r90.lobes == 2
    assert post.contains(*TRUTH, 0.9)
    assert "2 pieces" in post.summary()["words"]


def test_sense_ambiguous_bearing_shows_both_directions():
    st = T.destination(*TRUTH, 90.0, 5000.0)
    b = _from(*st, sigma=2.0, ambiguous_180=True)
    rg = PO.SoftRange(*st, ranges_m=(4000, 5000, 6000), pdf=(0.0, 1.0, 0.0))
    post = PO.locate([b], [rg], search_radius_m=12_000)
    r90 = [r for r in post.regions() if r.level == 0.9][0]
    assert r90.lobes == 2
    twin = T.destination(*st, 90.0, 5000.0)                 # the mirror point
    assert post.contains(*TRUTH, 0.95) and post.contains(*twin, 0.95)


def test_shared_bias_is_integrated_out_not_ignored():
    """ATK's coverage audit: many bearings from one mis-aligned array.
    Treated as independent they shrink the region around the wrong place;
    with the shared bias modelled the truth is inside."""
    rng = np.random.default_rng(7)
    track = [T.destination(*TRUTH, 200.0 - 4 * i, 9000.0) for i in range(12)]
    obs = [_from(la, lo, sigma=1.0, bias=3.0, rng=rng, station="kraken-1")
           for la, lo in track]
    naive = PO.locate(obs, search_radius_m=25_000)
    honest = PO.locate(obs, search_radius_m=25_000, bias_sigma_deg=3.0)
    assert naive.credible_level(*TRUTH) > 0.95
    assert honest.credible_level(*TRUTH) < 0.9
    a_naive = [r.area_km2 for r in naive.regions() if r.level == 0.9][0]
    a_honest = [r.area_km2 for r in honest.regions() if r.level == 0.9][0]
    assert a_honest > 2 * a_naive


def test_an_outlier_bearing_with_and_without_the_outlier_model():
    rng = np.random.default_rng(11)
    sites = [T.destination(*TRUTH, az, 6000.0) for az in (0, 70, 150, 230)]
    obs = [_from(la, lo, sigma=1.5, rng=rng) for la, lo in sites]
    wild = T.destination(*TRUTH, 300.0, 5000.0)
    obs.append(PO.Bearing(*wild, (float(T.initial_bearing_deg(*wild, *TRUTH)) + 70) % 360,
                          1.5))
    plain = PO.locate(obs, search_radius_m=15_000)
    robust = PO.locate(obs, search_radius_m=15_000, outlier_prob=0.05)
    assert robust.credible_level(*TRUTH) < 0.9
    assert T.haversine_m(*robust.map_point(), *TRUTH) < \
        T.haversine_m(*plain.map_point(), *TRUTH)


def test_a_land_prior_and_an_unbounded_single_bearing():
    water_east = T.FunctionTerrain(lambda la, lo: np.where(lo > -77.5, -5.0, 20.0))
    st = T.destination(*TRUTH, 0.0, 6000.0)          # north of the truth
    b = PO.Bearing(*st, 180.0, 2.0)
    post = PO.locate([b], prior=PO.land_prior(water_east), search_radius_m=10_000)
    lat, lon = post.grid.mesh()
    assert post.prob[lon > -77.499].sum() < 0.01
    assert post.unbounded
    assert "edge of the search area" in post.summary()["words"]
    with pytest.raises(ValueError, match="at least one bearing"):
        PO.locate([])


def test_the_posterior_as_a_product(rf):
    a = T.destination(*TRUTH, 180.0, 8000.0)
    b = T.destination(*TRUTH, 270.0, 6000.0)
    obs = [_from(*a, sigma=2.0, time="2026-10-08T12:00:00Z", station="A"),
           _from(*b, sigma=2.0, time="2026-10-08T12:00:05Z", station="B")]
    post = PO.locate(obs, search_radius_m=15_000)
    run = Path(post.to_products(
        rf, run="df-test", extra_features=[("bearings.geojson",
                                            PO.bearing_features(obs, 15_000),
                                            "measured")]))
    man = json.loads((run / "manifest.json").read_text())
    assert man["kind"] == "tracks" and man["tier"] == "inferred"
    assert man["method"] == "bearing_posterior"
    assert len(man["params"]["bearings"]) == 2
    regs = P.read_geojson(run / "regions.geojson")
    levels = [f["properties"]["credible_level"] for f in regs["features"]]
    assert levels == [0.95, 0.9, 0.5]
    assert all(f["properties"]["atk:tier"] == "inferred" for f in regs["features"])
    bear = P.read_geojson(run / "bearings.geojson")
    assert bear["features"][0]["properties"]["atk:tier"] == "measured"
    dens = P.read_geotiff(run / "posterior_density.tif")
    area = dens.grid.cell_area_m2() / 1e6
    assert abs(float((dens.array * area).sum()) - 1.0) < 1e-3
    samples = P.read_geojson(run / "samples.geojson")
    assert len(samples["features"]) == 400
    assert P.verify_run(run) == (True, [])
