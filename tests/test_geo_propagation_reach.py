# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Propagation models, antenna patterns, and the reach map with its three
layers (plan E5)."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from atk_diffusion.geo import antenna as A
from atk_diffusion.geo import products as P
from atk_diffusion.geo import propagation as PR
from atk_diffusion.geo import reach as R
from atk_diffusion.geo import terrain as T
from atk_diffusion.geo.products import GeoGrid


# -- free space, two-ray, knife edges ------------------------------------------
def test_free_space_and_two_ray():
    assert math.isclose(float(PR.fspl_db(1000.0, 100e6)), 72.44, abs_tol=0.01)
    # beyond the breakpoint two-ray falls 40 dB/decade: 12 dB per doubling
    for pol in ("vertical", "horizontal"):
        l5 = float(PR.two_ray_db(20e3, 150e6, 30.0, 2.0, polarization=pol))
        l10 = float(PR.two_ray_db(40e3, 150e6, 30.0, 2.0, polarization=pol))
        assert 11.0 < l10 - l5 < 12.6
    # and approaches the textbook 40 log d - 20 log(ht hr) far out (horizontal)
    d = 40e3
    asym = 40 * math.log10(d) - 20 * math.log10(30.0 * 2.0)
    assert abs(float(PR.two_ray_db(d, 150e6, 30.0, 2.0,
                                   polarization="horizontal")) - asym) < 1.0
    with pytest.raises(ValueError, match="polarization"):
        PR.two_ray_db(1e3, 1e8, 1, 1, polarization="circular")


def test_knife_edge_values():
    assert math.isclose(float(PR.knife_edge_loss_db(0.0)), 6.03, abs_tol=0.01)
    assert float(PR.knife_edge_loss_db(-0.8)) == 0.0
    assert math.isclose(float(PR.knife_edge_loss_db(2.4)), 20.54, abs_tol=0.02)


def _ridge_profile(D=10e3, n=201, peaks=((0.5, 100.0),)):
    d = np.linspace(0, D, n)
    z = np.zeros(n)
    for frac, h in peaks:
        z[int(round(frac * (n - 1)))] = h
    return T.Profile(d, None, None, z)


def test_single_edge_bullington_and_deygout_agree_with_the_formula():
    prof = _ridge_profile()
    lam = PR.C_LIGHT / 150e6
    h = 100.0 - 40.0                        # above the 40 m - 40 m line
    v = h * math.sqrt(2 * 10e3 / (lam * 5e3 * 5e3))
    want = float(PR.knife_edge_loss_db(v))
    # antennas high enough that the flat ground is clear of every sub-path
    b = PR.bullington(prof, 40.0, 40.0, 150e6, k=1e12)
    g = PR.deygout(prof, 40.0, 40.0, 150e6, k=1e12)
    assert math.isclose(b.loss_db, want, abs_tol=0.01)
    assert math.isclose(g.loss_db, want, abs_tol=0.01)
    assert not b.line_of_sight and len(g.edges) == 1
    two = _ridge_profile(peaks=((0.3, 100.0), (0.7, 95.0)))
    g2 = PR.deygout(two, 40.0, 40.0, 150e6, k=1e12)
    assert len(g2.edges) == 2 and g2.loss_db > g.loss_db
    assert "edges at" in g2.words()
    clear = PR.deygout(_ridge_profile(peaks=()), 40.0, 40.0, 150e6, k=1e12)
    # low antennas over flat ground: the ground itself sits in the Fresnel
    # zone and the knife-edge methods charge for it (pessimistic on smooth
    # earth — ITM's smooth-earth blend is the model for that case)
    low = PR.deygout(_ridge_profile(peaks=()), 2.0, 2.0, 150e6)
    assert low.loss_db > 3.0
    assert clear.loss_db == 0.0 and "clear" in clear.words()


# -- ITM, validated against the NTIA QKPFL test 1 ------------------------------
#: ITS QKPFL test 1, "path 2200": Crystal Palace to Mursley, England.
#: 156 intervals of 499 m; 41.5 MHz; antennas 143.9 m and 8.5 m;
#: horizontal polarisation; eps 15, sigma 0.005; N0 314; climate 5;
#: zsys 0. Published basic transmission loss (dB), rows = reliability
#: 1/10/50/90/99 %, columns = confidence 50/90/10 %.
CRYSTAL_PALACE = [
    96, 84, 65, 46, 46, 46, 61, 41, 33, 27, 23, 19, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 17, 19, 21, 23, 25, 27,
    29, 35, 46, 41, 35, 30, 33, 35, 37, 40, 35, 30, 51, 62, 76,
    46, 46, 46, 46, 46, 46, 50, 56, 67, 106, 83, 95, 112, 137, 137,
    76, 103, 122, 122, 83, 71, 61, 64, 67, 71, 74, 77, 79, 86, 91,
    83, 76, 68, 63, 76, 107, 107, 107, 119, 127, 133, 135, 137, 142, 148,
    152, 152, 107, 137, 104, 91, 99, 120, 152, 152, 137, 168, 168, 122, 137,
    137, 170, 183, 183, 187, 194, 201, 192, 152, 152, 166, 177, 198, 156, 127,
    116, 107, 104, 101, 98, 95, 103, 91, 97, 102, 107, 107, 107, 103, 98,
    94, 91, 105, 122, 122, 122, 122, 122, 137, 137, 137, 137, 137, 137, 137,
    137, 140, 144, 147, 150, 152, 159]
PUBLISHED = {1: (128.6, 137.6, 119.6), 10: (132.2, 140.8, 123.5),
             50: (135.8, 144.3, 127.2), 90: (138.0, 146.5, 129.4),
             99: (139.7, 148.4, 131.0)}


def test_itm_reproduces_the_published_qkpfl_table():
    pytest.importorskip("itmlogic", reason="ITM needs itmlogic")
    z = np.array(CRYSTAL_PALACE, dtype=float)
    assert z.size == 157
    prof = T.Profile(np.arange(157) * 499.0, None, None, z)
    pairs = [(r, c) for r in PUBLISHED for c in (50, 90, 10)]
    res = PR.itm_p2p(prof, 41.5e6, 143.9, 8.5, polarization="horizontal",
                     eps_r=15.0, sigma_s_m=0.005, n0=314.0, climate=5,
                     zsys=0.0, quantiles=pairs)
    assert res.kwx == 0 and res.warning == ""
    assert math.isclose(res.fspl_db, 102.6, abs_tol=0.05)
    assert res.mode == "double horizon, diffraction dominant"
    assert math.isclose(res.he_m[0], 240.6, abs_tol=0.2)
    assert math.isclose(res.he_m[1], 18.4, abs_tol=0.2)
    assert math.isclose(res.dh_m, 89.0, abs_tol=0.5)
    for r, row in PUBLISHED.items():
        for c, want in zip((50, 90, 10), row):
            assert abs(res.quantiles[(float(r), float(c))] - want) <= 0.15, (r, c)


def test_itm_refuses_in_words_when_itmlogic_is_absent(monkeypatch):
    monkeypatch.setattr(PR, "itm_available",
                        lambda: (False, "The Longley-Rice model (ITM) needs "
                                        "the itmlogic package"))
    prof = _ridge_profile()
    with pytest.raises(PR.ItmUnavailable, match="needs the itmlogic package"):
        PR.itm_p2p(prof, 150e6, 10, 2)
    tx = R.Transmitter(38.7, -77.5, 150e6)
    with pytest.raises(PR.ItmUnavailable):
        R.predicted_reach(tx, R.Receiver(), T.FlatTerrain(), 1.0, model="itm",
                          grid_m=500, write=False)
    with pytest.raises(ValueError, match="20 MHz to 20 GHz"):
        pytest.importorskip("itmlogic")
        monkeypatch.undo()
        PR.itm_p2p(prof, 5e6, 10, 2)


# -- antennas -----------------------------------------------------------------
def test_antenna_patterns():
    v = A.Dipole("vertical")
    assert math.isclose(float(v.gain_dbi(123.0, 0.0)), 2.15, abs_tol=1e-9)
    want60 = 2.15 + 20 * math.log10(math.cos(math.pi / 2 * math.sin(math.radians(60)))
                                    / math.cos(math.radians(60)))
    assert math.isclose(float(v.gain_dbi(0.0, 60.0)), want60, abs_tol=1e-9)
    h = A.Dipole("horizontal")
    assert math.isclose(float(h.gain_dbi(0.0, 0.0)), 2.15, abs_tol=1e-9)
    assert float(h.gain_dbi(90.0, 0.0)) <= 2.15 - 37.0
    d = A.Directional(10.0, 60.0, 40.0, front_to_back_db=18.0)
    assert math.isclose(float(d.gain_dbi(30.0, 0.0)), 7.0)       # 3 dB at bw/2
    assert math.isclose(float(d.gain_dbi(0.0, 20.0)), 7.0)
    assert math.isclose(float(d.gain_dbi(180.0, 0.0)), -8.0)
    t = A.TablePattern(6.0, az_deg=[0, 90, 180, 270], az_db=[0, -10, -20, -10],
                       el_deg=[-90, 0, 90], el_db=[-20, 0, -20])
    assert math.isclose(float(t.gain_dbi(45.0, 0.0)), 1.0)
    assert math.isclose(float(t.gain_dbi(315.0, 45.0)), 6.0 - 5.0 - 10.0)
    full = A.TablePattern(az_deg=[0, 180], el_deg=[0, 10],
                          table_db=[[8.0, -2.0], [4.0, -6.0]])
    assert math.isclose(float(full.gain_dbi(90.0, 5.0)), 1.0)
    for p in (v, d, t, full):
        q = A.from_json(json.loads(json.dumps(p.to_json())))
        assert np.allclose(q.gain_dbi([0, 45, 200], [0, 10, -5]),
                           p.gain_dbi([0, 45, 200], [0, 10, -5]))


def _atk_result(label, parts=(), value=None, refusal=""):
    return SimpleNamespace(label=label, parts=tuple(parts), value=value,
                           refusal=refusal)


def test_from_atk_reads_the_designers_results_by_their_fields():
    db = lambda x: SimpleNamespace(db=x)                          # noqa: E731
    yagi = A.from_atk(_atk_result("Yagi STARTING dimensions (5 elements) for 146 MHz",
                                  [("Estimated gain", db(10.5), "dBi for an OPTIMISED design")],
                                  value=db(10.5)))
    assert isinstance(yagi, A.Directional) and yagi.gain == 10.5
    assert "Kraus" in yagi.note
    col = A.from_atk(_atk_result("coaxial collinear, 4 sections, for 446 MHz",
                                 [("Estimated gain", db(5.0), "dBd, before feed losses")]))
    assert isinstance(col, A.Dipole) and math.isclose(col.gain, 7.15)
    dish = A.from_atk(_atk_result("parabolic dish, 1.2 m, at 1.42 GHz",
                                  [("Gain", db(22.0), "dBi at 55% efficiency"),
                                   ("Half-power beamwidth", "12.31°", "")]))
    assert dish.az_bw == 12.31 and dish.fb == 25.0
    gp = A.from_atk(_atk_result("quarter-wave ground plane for 146 MHz"))
    assert gp.orientation == "vertical" and gp.gain == 2.15
    with pytest.raises(ValueError, match="refused"):
        A.from_atk(_atk_result("cantenna at 2.4 GHz", refusal="the can is too small"))
    with pytest.raises(ValueError, match="states no gain"):
        A.from_atk(_atk_result("microstrip patch for 2.4 GHz"))
    assert A.from_atk(_atk_result("microstrip patch for 2.4 GHz"),
                      gain_dbi=6.5).gain == 6.5
    with pytest.raises(ValueError, match="no pattern model"):
        A.from_atk(_atk_result("magnetic loop, 1 m, at 7 MHz"))


def test_from_atk_with_the_real_antenna_designer():
    """Integration: ATK's own designer, when its source is reachable
    (ATK_HOME, or the read-only snapshot on the build machine)."""
    cands = [os.environ.get("ATK_HOME", ""), "/home/claude/atk_snapshot"]
    home = next((c for c in cands if c and (Path(c) / "atk" / "core" / "rf" /
                                            "antenna.py").exists()), None)
    if home is None:
        pytest.skip("ATK's antenna designer is not reachable (set ATK_HOME)")
    sys.path.insert(0, home)
    try:
        from atk.core.rf import antenna as atk_ant
        from atk.core.rf.units import Frequency
    finally:
        sys.path.remove(home)
    f = Frequency.mhz(146.0)
    y = A.from_atk(atk_ant.yagi(f, elements=5))
    assert isinstance(y, A.Directional) and 8 < y.gain < 14
    m = A.from_atk(atk_ant.moxon(f))
    assert m.gain == 6.0 and m.fb == 20.0
    j = A.from_atk(atk_ant.jpole(f))
    assert isinstance(j, A.Dipole)
    dish = A.from_atk(atk_ant.dish(Frequency.ghz(1.42), 2.0))
    assert dish.az_bw > 0
    with pytest.raises(ValueError, match="refused"):
        A.from_atk(atk_ant.cantenna(Frequency.ghz(2.4), 0.02))


# -- the reach map --------------------------------------------------------------
def _ridge_terrain():
    g = GeoGrid(-77.60, 38.64, -77.40, 38.76, 160, 100)
    lat, lon = g.mesh()
    z = 80.0 + 150.0 * np.exp(-((lon + 77.47) / 0.004) ** 2)   # N-S ridge east of tx
    return T.GridTerrain(g, z, label="synthetic ridge")


def test_reach_free_space_numbers_and_the_mask():
    tx = R.Transmitter(38.70, -77.50, 146.52e6, power_dbm=37.0, height_agl_m=2.0,
                       antenna=A.Isotropic())
    rx = R.Receiver(height_agl_m=2.0, antenna=A.Isotropic(), sensitivity_dbm=-90.0)
    res = R.predicted_reach(tx, rx, T.FlatTerrain(), 3.0, model="fspl",
                            grid_m=300, write=False)
    lat, lon = res.grid.mesh()
    d = T.haversine_m(38.70, -77.50, lat, lon)
    sel = np.isfinite(res.power_dbm)
    assert np.allclose(res.power_dbm[sel],
                       37.0 - PR.fspl_db(d[sel], 146.52e6), atol=1e-6)
    assert np.array_equal(res.mask[sel] == 1, res.power_dbm[sel] >= -90.0)
    assert np.all(res.mask[~sel] == R.MASK_NODATA)
    assert "A planning prediction" in res.words()


def test_terrain_shadow_and_a_pointed_antenna():
    terr = _ridge_terrain()
    tx = R.Transmitter(38.70, -77.50, 146.52e6, height_agl_m=2.0)
    rx = R.Receiver(sensitivity_dbm=-100.0)
    free = R.predicted_reach(tx, rx, terr, 4.0, model="fspl", grid_m=400, write=False)
    dey = R.predicted_reach(tx, rx, terr, 4.0, model="deygout", grid_m=400,
                            write=False)
    lat, lon = dey.grid.mesh()
    behind = (lon > -77.46) & np.isfinite(dey.power_dbm)
    front = (lon < -77.52) & np.isfinite(dey.power_dbm)
    shadow = (free.power_dbm - dey.power_dbm)
    assert np.nanmean(shadow[behind]) > np.nanmean(shadow[front]) + 10.0
    yagi = R.Transmitter(38.70, -77.50, 146.52e6, antenna=A.Directional(10, 50),
                         azimuth_deg=270.0)
    pointed = R.predicted_reach(yagi, rx, T.FlatTerrain(), 3.0, model="fspl",
                                grid_m=400, write=False)
    lat, lon = pointed.grid.mesh()
    west = (lon < -77.52) & np.isfinite(pointed.power_dbm)
    east = (lon > -77.48) & np.isfinite(pointed.power_dbm)
    assert np.nanmean(pointed.power_dbm[west]) > np.nanmean(pointed.power_dbm[east]) + 15


def test_itm_reach_writes_the_physics_then_measurement_then_correction(rf):
    pytest.importorskip("itmlogic", reason="ITM needs itmlogic")
    terr = _ridge_terrain()
    tx = R.Transmitter(38.70, -77.50, 146.52e6, power_dbm=37.0, height_agl_m=10.0,
                       name="relief net")
    rx = R.Receiver(sensitivity_dbm=-105.0)
    res = R.predicted_reach(tx, rx, terr, 3.0, model="itm", grid_m=300, rf=rf,
                            run="itm-test")
    run_dir = Path(res.run_dir)
    man = json.loads((run_dir / "manifest.json").read_text())
    assert man["tier"] == "inferred" and man["method"] == "itm"
    assert man["params"]["transmitter"]["power_w"] == pytest.approx(5.0, rel=0.01)
    roles = {l["role"] for l in man["layers"]}
    assert {"physics", "physics-mask", "diagnostic"} <= roles
    assert any("inside 1 km" in n for n in man["extra"]["notes"])
    phys = P.read_geotiff(run_dir / "received_power_dbm.tif")
    assert phys.tier == "inferred" and phys.tags["atk:model"] == "itm"
    assert np.allclose(phys.masked(), res.power_dbm, equal_nan=True, atol=1e-4)

    # the drive: what physics missed is a 6 dB clutter loss west of -77.5
    rng = np.random.default_rng(3)
    lat = 38.70 + rng.uniform(-0.02, 0.02, 80)
    lon = -77.50 + rng.uniform(-0.03, 0.03, 80)
    truth = phys.sample(lat, lon) - np.where(lon < -77.5, 6.0, 0.0)
    feats = [P.point_feature(a, o, {"rx_dbm": float(v)})
             for a, o, v in zip(lat, lon, truth + rng.normal(0, 1.0, 80))
             if np.isfinite(v)]
    stats = R.add_measurement_layer(rf, "itm-test",
                                    P.feature_collection(feats), interpolate="kriging")
    assert stats["n"] == len(feats)
    assert -5.0 < stats["residual_mean_db"] < -1.0
    corr = np.where(phys.grid.mesh()[1] < -77.5, -6.0, 0.0)
    cstats = R.add_correction_layer(rf, "itm-test", corr, sigma_db=np.full(corr.shape, 2.0),
                                    card={"name": "radiomap-test", "kind": "radiomap"},
                                    hallucination_rate=0.01)
    assert cstats["max_abs_correction_db"] == 6.0
    man = json.loads((run_dir / "manifest.json").read_text())
    roles = {l["role"] for l in man["layers"]}
    assert {"physics", "measurement", "residual", "correction"} <= roles
    tiers = {l["file"]: l["tier"] for l in man["layers"]}
    assert tiers["received_power_dbm.tif"] == "inferred"
    assert tiers["measured_points.geojson"] == "measured"
    assert tiers["correction_db.tif"] == "invented"
    assert P.verify_run(run_dir) == (True, [])
    corrected = P.read_geotiff(run_dir / "corrected_power_dbm.tif")
    assert corrected.tier == "invented"
    assert corrected.tags["atk:hallucination_rate"] == 0.01
