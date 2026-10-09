# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Track inpainting, road-conditioned fill, reappearance (plan §4.D4)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from atk_diffusion import provenance
from atk_diffusion.repair import tracks as T

T0 = 1.76e9
CENTER = (38.70, -77.55)          # somewhere in Prince William County


def _track(xy_fn, dur, dt, sig, rng, gap=None, plane=None):
    plane = plane or T.Plane(*CENTER)
    ts = np.arange(0, dur, dt)
    X, Y = xy_fn(ts)
    la, lo = plane.inv(X + rng.normal(0, sig, ts.size), Y + rng.normal(0, sig, ts.size))
    fx = [T.Fix(float(t) + T0, float(a), float(b), sigma_m=sig)
          for t, a, b in zip(ts, la, lo)]
    if gap:
        fx = [f for f in fx if not (gap[0] <= f.t - T0 <= gap[1])]
    return fx, plane


def _err(fixes, xy_fn, plane):
    inf = [f for f in fixes if f.tier == "inferred"]
    X, Y = xy_fn(np.array([f.t - T0 for f in inf]))
    la, lo = plane.inv(X, Y)
    e = T.haversine_m(np.array([f.lat for f in inf]), np.array([f.lon for f in inf]), la, lo)
    return e, np.array([f.sigma_m for f in inf])


def test_plane_is_exact_and_great_circles_are_straight(rng):
    p = T.Plane(52.0, -20.0)
    lat, lon = rng.uniform(40, 60, 50), rng.uniform(-40, 0, 50)
    x, y = p.fwd(lat, lon)
    la, lo = p.inv(x, y)
    assert np.max(np.abs(la - lat)) < 1e-9 and np.max(np.abs(lo - lon)) < 1e-9
    assert np.max(np.abs(np.hypot(x, y) - T.haversine_m(52.0, -20.0, lat, lon))) < 1e-3
    # a great circle through the centre is a straight line in the plane
    la, lo = T.slerp(50.0, -30.0, 54.0, -10.0, np.linspace(0, 1, 9))
    m = T.midpoint(50.0, -30.0, 54.0, -10.0)
    x, y = T.Plane(*m).fwd(la, lo)
    resid = np.polyfit(x, y, 1, full=True)[1]
    assert float(resid[0]) < 1.0


def test_fixes_from_atk_rows():
    rows = [{"utc_iso": "x", "epoch": "100.5", "lat": "38.7", "lon": "-77.5",
             "altitude_m": "90"}, {"epoch": "", "lat": "1", "lon": "2"},
            {"t": 99.0, "lat": 38.69, "lon": -77.49, "alt": 1000}]
    fx = T.fixes_from_rows(rows)
    assert [f.t for f in fx] == [99.0, 100.5] and fx[1].alt_m == 90.0
    ads = T.fixes_from_rows([{"t": 1, "lat": 1, "lon": 2, "alt": 1000}],
                            alt_key="alt", alt_scale=0.3048)
    assert abs(ads[0].alt_m - 304.8) < 1e-9


def test_every_filled_point_is_inferred_with_sigma(rng):
    fx, plane = _track(lambda t: (20 * t, 0 * t), 200, 1.0, 5.0, rng, gap=(80, 120))
    res = T.fill_gaps(fx, "kalman", kind="vehicle")
    rec = [f for f in res.fixes if f.tier == "record"]
    assert len(rec) == len(fx) and all(a is b for a, b in zip(rec, fx))
    inf = res.inferred()
    assert len(inf) == res.gaps[0].points > 30
    for f in inf:
        assert f.tier == "inferred" and f.method == "kalman_fill"
        assert provenance.tier_for(f.method) == "inferred" and f.sigma_m > 0
    s = np.array([f.sigma_m for f in inf])
    assert s[len(s) // 2] > 2 * s[0]                 # grows into the gap
    assert s[-1] < s[len(s) // 2]                    # and shrinks toward the fix
    assert "floor" in res.lines[0]                   # a straight record: the floor
    e, s = _err(res.fixes, lambda t: (20 * t, 0 * t), plane)
    assert e.mean() < 5 and np.mean(e <= 2 * s) > 0.9


@pytest.mark.parametrize("method", ["great_circle", "turn", "kalman"])
def test_methods_on_a_turn_and_honest_sigma(method, rng):
    circ = lambda t: (200 * np.sin(15 * t / 200), 200 * (1 - np.cos(15 * t / 200)))
    fx, plane = _track(circ, 200, 1.0, 5.0, rng, gap=(100, 120))
    res = T.fill_gaps(fx, method, kind="vehicle", gate=None)
    e, s = _err(res.fixes, circ, plane)
    assert np.mean(e <= 2 * s) >= 0.9               # sigma never oversells
    if method == "great_circle":
        assert e.mean() > 25                         # it cuts the corner
    else:
        assert e.mean() < 12


def test_great_circle_across_an_ocean_hole(rng):
    """An airliner across a 10-minute ADS-B hole flies a great circle."""
    D = float(T.haversine_m(52.0, -30.0, 53.5, -10.0))
    dur = D / 240.0
    ts = np.arange(0, dur, 10.0)
    la, lo = T.slerp(52.0, -30.0, 53.5, -10.0, ts / dur)
    fx = [T.Fix(float(t) + T0, float(a), float(b), sigma_m=30.0) for t, a, b in zip(ts, la, lo)
          if not 3000 <= t <= 3600]
    for m in ("great_circle", "kalman"):
        res = T.fill_gaps(fx, m, kind="aircraft")
        inf = res.inferred()
        tla, tlo = T.slerp(52.0, -30.0, 53.5, -10.0, np.array([f.t - T0 for f in inf]) / dur)
        err = T.haversine_m(np.array([f.lat for f in inf]), np.array([f.lon for f in inf]), tla, tlo)
        assert err.max() < 60, m


def test_stitched_tracks_are_refused_not_filled(rng):
    east = lambda t: (20 * t, 0 * t)
    a, plane = _track(east, 100, 1.0, 5.0, rng)
    # (1) a second object 5 km away 30 s later: impossible speed
    far = [T.Fix(f.t + 130, *plane.inv(5000 + 20 * (f.t - T0), 3000.0), sigma_m=5.0)
           for f in a[:60]]
    far = [T.Fix(f.t, float(f.lat), float(f.lon), sigma_m=5.0) for f in far]
    res = T.fill_gaps(a + far, "kalman", kind="vehicle")
    assert not res.gaps[0].filled and "m/s" in res.gaps[0].why
    assert res.inferred() == []
    # (2) a plausible speed, but the fix after the gap is off the motion
    north = []
    for k in range(60):
        la, lo = plane.inv(2000.0, 900.0 + 20.0 * k)
        north.append(T.Fix(a[-1].t + 60 + k, float(la), float(lo), sigma_m=5.0))
    res = T.fill_gaps(a + north, "kalman", kind="vehicle")
    assert not res.gaps[0].filled and "σ" in res.gaps[0].why
    assert res.gaps[0].gate_d2 is not None and res.inferred() == []


def _grid_graph(n=6, spacing=100.0):
    plane = T.Plane(*CENTER)
    nodes, edges = {}, []
    for i in range(n):
        for j in range(n):
            la, lo = plane.inv(i * spacing, j * spacing)
            nodes[(i, j)] = (float(la), float(lo))
    for i in range(n):
        for j in range(n):
            if i + 1 < n:
                edges.append(((i, j), (i + 1, j)))
            if j + 1 < n:
                edges.append(((i, j), (i, j + 1)))
    return T.RoadGraph(nodes, edges), plane


def _corner(t):
    """10 m/s east along y=0 to x=400, then north along x=400."""
    d = 10.0 * t
    x = np.where(d <= 400, d, 400.0)
    y = np.where(d <= 400, 0.0, d - 400)
    return x, y


def test_road_fill_turns_the_corner(rng):
    g, plane = _grid_graph()
    fx, _ = _track(_corner, 80, 1.0, 3.0, rng, gap=(30, 50), plane=plane)
    straight = T.fill_gaps(fx, "kalman", kind="vehicle", gate=None)
    road = T.fill_gaps(fx, "road", kind="vehicle", graph=g)
    e_k, _ = _err(straight.fixes, _corner, plane)
    e_r, s_r = _err(road.fixes, _corner, plane)
    assert road.gaps[0].filled and road.inferred()[0].method == "road_fill"
    assert e_r.mean() < 0.5 * e_k.mean() and e_r.max() < 30
    assert np.mean(e_r <= 2 * s_r) > 0.9
    poly, L = g.route(g.snap(*plane.inv(250.0, 0.0)), g.snap(*plane.inv(400.0, 150.0)))
    assert abs(L - 300.0) < 1.0                       # along the streets
    with pytest.raises(ValueError, match="road graph"):
        T.fill_gaps(fx, "road", kind="vehicle")


def test_road_fill_refusals(rng):
    g, plane = _grid_graph()
    off = lambda t: (10.0 * t, 50.0 + 0 * t)          # between the streets
    fx, _ = _track(off, 60, 1.0, 1.0, rng, gap=(20, 35), plane=plane)
    res = T.fill_gaps(fx, "road", kind="vehicle", graph=g, max_snap_m=20)
    assert not res.gaps[0].filled and "from the nearest road" in res.gaps[0].why
    res = T.fill_gaps(fx, "road", kind="pedestrian", graph=g, max_snap_m=100, gate=None)
    assert not res.gaps[0].filled


def test_reappear_contains_the_truth(rng):
    hits = 0
    for k in range(20):
        r = np.random.default_rng(100 + k)
        fx, plane = _track(lambda t: (15 * t, 5 * t), 60, 1.0, 5.0, r)
        reg = T.reappear(fx, T0 + 90.0, prob=0.95, kind="vehicle")
        tx, ty = 15 * 90.0, 5 * 90.0
        la, lo = plane.inv(tx, ty)
        P = T.Plane(reg.lat, reg.lon)
        rx, ry = P.fwd([p[0] for p in reg.ring], [p[1] for p in reg.ring])
        px, py = P.fwd(la, lo)
        from matplotlib.path import Path as MPath
        hits += MPath(np.stack([rx, ry], axis=1)).contains_point((float(px), float(py)))
    assert hits >= 16                                   # ~95 % nominal
    feat = reg.to_feature()
    lon0, lat0 = feat["geometry"]["coordinates"][0][0]
    assert abs(lat0 - reg.lat) < 0.1 and abs(lon0 - reg.lon) < 0.1   # [lon, lat]
    assert feat["properties"]["tier"] == "inferred"
    with pytest.raises(ValueError):
        T.reappear(fx, fx[-1].t - 1)


def test_reappear_on_roads(rng):
    g, plane = _grid_graph()
    fx, _ = _track(lambda t: (10.0 * t, 0 * t), 15, 1.0, 1.0, rng, plane=plane)
    feat = T.reappear_on_roads(fx, fx[-1].t + 20.0, g)
    pieces = feat["geometry"]["coordinates"]
    assert pieces and feat["properties"]["tier"] == "inferred"
    tla, tlo = plane.inv(140.0 + 200.0, 0.0)            # where it would be
    pts = np.array([q for p in pieces for q in p])
    d = T.haversine_m(pts[:, 1], pts[:, 0], tla, tlo)
    assert d.min() < 15.0


def test_bearings_wrap_through_north(rng):
    t = np.arange(0, 120.0, 1.0)
    true = (350.0 + 0.2 * t) % 360.0                    # crosses 360 -> 0
    obs = (true + rng.normal(0, 1.0, t.size)) % 360.0
    keep = (t < 40) | (t > 70)
    out = T.fill_bearings(t[keep], obs[keep], sigma_deg=1.0)
    assert out["gaps"] and out["inferred"]
    for p in out["inferred"]:
        tr = (350.0 + 0.2 * p["t"]) % 360.0
        diff = abs((p["bearing_deg"] - tr + 180) % 360 - 180)
        assert diff < 3 * p["sigma_deg"] + 0.5 and p["tier"] == "inferred"


def test_geojson_is_longitude_first_and_says_inferred(tmp_path, rng):
    fx, plane = _track(lambda t: (20 * t, 0 * t), 100, 1.0, 5.0, rng, gap=(40, 60))
    res = T.fill_gaps(fx, "kalman", kind="vehicle")
    gj = res.to_geojson(name="test", ellipses_every=5)
    line = gj["features"][0]
    assert line["geometry"]["type"] == "LineString"
    lon, lat = line["geometry"]["coordinates"][0]
    assert abs(lat - fx[0].lat) < 1e-12 and abs(lon - fx[0].lon) < 1e-12
    pts = [f for f in gj["features"] if f["geometry"]["type"] == "Point"]
    assert pts and all(f["properties"]["tier"] == "inferred"
                       and f["properties"]["sigma_m"] > 0 for f in pts)
    assert any(f["geometry"]["type"] == "Polygon" for f in gj["features"])
    p = T.write_geojson(tmp_path / "tracks" / "t.geojson", gj)
    assert json.loads(p.read_text())["type"] == "FeatureCollection"
    assert gj["properties"]["gaps"][0]["filled"] is True


def test_graph_from_geojson_and_route():
    plane = T.Plane(*CENTER)
    def ll(x, y):
        la, lo = plane.inv(x, y)
        return [float(lo), float(la)]
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {},
         "geometry": {"type": "LineString", "coordinates": [ll(0, 0), ll(100, 0), ll(200, 0)]}},
        {"type": "Feature", "properties": {},
         "geometry": {"type": "LineString", "coordinates": [ll(200, 0), ll(200, 100)]}}]}
    g = T.RoadGraph.from_geojson(gj)
    assert len(g.nodes) == 4 and len(g.edges) == 3
    a = g.snap(*plane.inv(50.0, 3.0))
    b = g.snap(*plane.inv(197.0, 60.0))
    poly, L = g.route(a, b)
    assert abs(L - 210.0) < 1.0 and abs(a["dist_m"] - 3.0) < 0.1
