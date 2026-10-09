# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The Kraken as a moving synthetic aperture (plan E4): a simulated tower at
a known position, a driven loop, direct position determination with array
self-calibration, against the stationary single-position bearing."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from atk_diffusion.geo import aperture as AP
from atk_diffusion.geo import products as P
from atk_diffusion.geo import terrain as T

CENTER = (38.70, -77.50)
FREQ = 98.1e6


def _tower(dist=8000.0, az=0.0):
    la, lo = T.destination(*CENTER, az, dist)
    return float(la), float(lo)


def test_array_geometry_and_steering():
    g = AP.ArrayGeometry.uca(1.0)
    xy = g.body_xy()
    assert np.allclose(np.hypot(xy[:, 0], xy[:, 1]), 1.0)
    assert np.allclose(xy[0], [1.0, 0.0])                  # element 0 forward
    assert math.isclose(g.neighbour_spacing_m(), 2 * math.sin(math.pi / 5))
    # heading east: the forward element points east
    assert np.allclose(g.enu(90.0)[0], [1.0, 0.0], atol=1e-12)
    assert not g.ambiguous(FREQ) and AP.ArrayGeometry.ula(0.5).ambiguous(FREQ)
    assert AP.ArrayGeometry.uca(3.0).ambiguous(FREQ)       # too wide: twins
    # a plane wave from the east: the east element leads by k r
    a = AP.steering(g, 90.0, 90.0, FREQ)[:, 0]
    k = 2 * math.pi * FREQ / AP.C_LIGHT
    assert np.isclose(np.angle(a[0]), (k * 1.0 + math.pi) % (2 * math.pi) - math.pi)


def test_single_snapshot_bearing_is_right_and_has_no_range():
    tower = _tower()
    g = AP.ArrayGeometry.uca(1.0)
    snaps = AP.simulate_drive(*tower, [CENTER[0]], [CENTER[1]], g, FREQ,
                              phase_sigma_deg=0.0, gain_sigma_db=0.0,
                              heading_sigma_deg=0.0, gps_sigma_m=0.0,
                              rng=np.random.default_rng(1))
    b, post = AP.stationary_fix(snaps[0], g, FREQ, sigma_deg=2.0)
    assert abs(T.wrap180(b.bearing_deg - 0.0)) < 1.0
    assert post.unbounded                                  # a bearing, no range
    assert "edge of the search area" in post.summary()["words"]


def test_driven_loop_finds_the_tower_and_beats_the_parked_bearing(rf):
    tower = _tower()
    g = AP.ArrayGeometry.uca(1.0)
    rlat, rlon = AP.loop_route(*CENTER, 1500.0, 24)
    snaps = AP.simulate_drive(*tower, rlat, rlon, g, FREQ, snr_db=10.0,
                              n_samples=128, phase_sigma_deg=5.0,
                              rng=np.random.default_rng(5))
    res = AP.locate_moving(snaps, g, FREQ, sigma_deg=3.0, max_cells=60_000)
    err = float(T.haversine_m(*res.estimate, *tower))
    assert err < 600.0
    assert res.posterior.contains(*tower, 0.95)
    assert res.meta["relative_angle_spread_deg"] > 300
    assert "self-calibrated" in " ".join(res.posterior.notes)
    # the parked comparator: one bearing, cross-range error, no range at all
    b, park = AP.stationary_fix(snaps[0], g, FREQ, sigma_deg=3.0)
    assert park.unbounded and not res.posterior.unbounded
    # without self-calibration the fixed channel errors bias the range
    raw = AP.locate_moving(snaps, g, FREQ, sigma_deg=3.0, calibrate=False,
                           max_cells=60_000)
    assert float(T.haversine_m(*raw.estimate, *tower)) > err
    two = AP.two_step(res)
    assert float(T.haversine_m(*two.map_point(), *tower)) < 1500
    # as a product, with the route, the per-snapshot bearings and the truth
    run = Path(res.to_products(rf, run="e4-test", truth=tower,
                               truth_label="broadcast tower (simulated)"))
    man = json.loads((run / "manifest.json").read_text())
    assert man["kind"] == "tracks" and man["method"] == "dpd"
    assert man["tier"] == "inferred"
    assert "channel_phase_deg" in man["params"]
    names = {l["file"] for l in man["layers"]}
    assert {"route.geojson", "snapshot_bearings.geojson", "truth.geojson",
            "regions.geojson", "posterior_density.tif"} <= names
    route = P.read_geojson(run / "route.geojson")
    assert route["features"][0]["properties"]["atk:tier"] == "measured"


def test_a_short_straight_drive_is_not_self_calibrated():
    tower = _tower(6000.0, 45.0)
    g = AP.ArrayGeometry.uca(1.0)
    lat = np.linspace(CENTER[0], CENTER[0] + 0.004, 6)
    lon = np.full(6, CENTER[1])
    snaps = AP.simulate_drive(*tower, lat, lon, g, FREQ, n_samples=64,
                              rng=np.random.default_rng(2))
    res = AP.locate_moving(snaps, g, FREQ, max_cells=20_000)
    assert res.gains is None
    assert any("not self-calibrated" in n for n in res.posterior.notes)


def test_a_linear_array_keeps_both_lobes():
    tower = _tower(5000.0, 90.0)
    g = AP.ArrayGeometry.ula(1.2, axis_deg=0.0)            # along the vehicle
    lat = np.linspace(CENTER[0] - 0.01, CENTER[0] + 0.01, 8)
    lon = np.full(8, CENTER[1])                            # driving north
    snaps = AP.simulate_drive(*tower, lat, lon, g, FREQ, n_samples=64,
                              phase_sigma_deg=0.0, gain_sigma_db=0.0,
                              heading_sigma_deg=0.0, rng=np.random.default_rng(3))
    res = AP.locate_moving(snaps, g, FREQ, calibrate=False, sigma_deg=2.0,
                           search_radius_m=10_000, max_cells=60_000)
    mirror = T.destination(*CENTER, 270.0, 5000.0)
    assert res.posterior.contains(*tower, 0.95)
    assert res.posterior.contains(float(mirror[0]), float(mirror[1]), 0.95)
    r95 = [r for r in res.posterior.regions() if r.level == 0.95][0]
    assert r95.lobes >= 2
