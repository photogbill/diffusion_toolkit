# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""DTED read per MIL-PRF-89020B from bytes this test lays down itself, the
mosaic, and the terrain arithmetic (plan E5)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from atk_diffusion.geo import dted as D
from atk_diffusion.geo import terrain as T
from atk_diffusion.geo.products import EARTH_RADIUS_M, GeoGrid


# -- an independent DTED byte builder, field by field from the standard -------
def _sm(v: int) -> bytes:
    """signed magnitude, big-endian"""
    return ((0x8000 | -v) if v < 0 else v).to_bytes(2, "big")


def _tile_bytes(lat_o: str, lon_o: str, dsi_lat: str, dsi_lon: str,
                interval_tenths: int, elev, raw_override=None) -> bytes:
    n_lat, n_lon = elev.shape
    uhl = (b"UHL1" + lon_o.encode() + lat_o.encode()
           + b"%04d" % interval_tenths + b"%04d" % interval_tenths
           + b"NA  " + b"U  " + b" " * 12 + b"%04d" % n_lon + b"%04d" % n_lat
           + b"0" + b" " * 24)
    assert len(uhl) == 80
    dsi = bytearray(b" " * 648)
    dsi[0:3] = b"DSI"
    dsi[3:4] = b"U"
    dsi[59:64] = b"DTED1"
    dsi[141:144] = b"MSL"
    dsi[144:149] = b"WGS84"
    dsi[185:194] = dsi_lat.encode()
    dsi[194:204] = dsi_lon.encode()
    dsi[273:277] = b"%04d" % interval_tenths
    dsi[277:281] = b"%04d" % interval_tenths
    dsi[281:285] = b"%04d" % n_lat
    dsi[285:289] = b"%04d" % n_lon
    acc = bytearray(b" " * 2700)
    acc[0:3] = b"ACC"
    out = bytearray(uhl + bytes(dsi) + bytes(acc))
    for col in range(n_lon):
        rec = bytearray([0xAA, 0, 0, col, 0, col, 0, 0])
        for row in range(n_lat):
            if raw_override and (row, col) in raw_override:
                rec += raw_override[(row, col)]
            else:
                rec += _sm(int(elev[row, col]))
        rec += sum(rec).to_bytes(4, "big")
        out += rec
    return bytes(out)


ELEV = np.array([[100, -28, D.VOID, 10],      # south row
                 [110, 0, 50, 20],
                 [120, 1234, 60, 30],
                 [130, 300, 70, 40],
                 [140, 200, 80, 50]], dtype=np.int16)   # north row


def test_reads_a_byte_exact_tile_with_negative_heights_and_a_void(tmp_path):
    p = tmp_path / "n38.dt1"
    p.write_bytes(_tile_bytes("0380000N", "0770000W", "380000.0N",
                              "0770000.0W", 9000, ELEV))
    t = D.read_dted(p)
    assert (t.lat0, t.lon0) == (38.0, -77.0)
    assert (t.dlat_s, t.dlon_s) == (900.0, 900.0)
    assert (t.n_lat, t.n_lon) == (5, 4) and t.level == 1
    assert np.array_equal(t.elev, ELEV)
    assert t.elev[0, 1] == -28 and t.voids == 1
    assert t.problems == [] and t.checksum_errors == []
    assert t.vertical_datum == "MSL" and t.horizontal_datum == "WGS84"
    assert math.isclose(t.lat1, 39.0) and math.isclose(t.lon1, -76.25)
    # bilinear: on a post exactly, between posts linearly, NaN by a void
    assert t.elevation(38.25, -77.0)[0] == 110
    assert math.isclose(t.elevation(38.125, -77.0)[0], 105.0)
    assert np.isnan(t.elevation(38.1, -76.6)[0])          # touches the void
    assert np.isfinite(t.elevation(38.1, -76.6, void="skip")[0])
    assert np.isnan(t.elevation(40.0, -77.0)[0])          # outside
    assert "1 void posts" in t.describe()


def test_checksum_failure_is_named_not_hidden(tmp_path):
    b = bytearray(_tile_bytes("0380000N", "0770000W", "380000.0N",
                              "0770000.0W", 9000, ELEV))
    rec = 12 + 2 * 5
    b[D.HEADER_LEN + 2 * rec + 9] ^= 0x01          # one height bit in column 2
    p = tmp_path / "bad.dt1"
    p.write_bytes(bytes(b))
    t = D.read_dted(p)
    assert t.checksum_errors == [2]
    assert "fail their checksum" in t.problems[0]
    with pytest.raises(D.DtedError, match="checksum"):
        D.read_dted(p, strict=True)


def test_twos_complement_negatives_from_a_nonconformant_producer(tmp_path):
    raw = {(1, 0): (-5 & 0xFFFF).to_bytes(2, "big")}     # 0xFFFB
    p = tmp_path / "tc.dt1"
    p.write_bytes(_tile_bytes("0380000N", "0770000W", "380000.0N",
                              "0770000.0W", 9000, ELEV, raw_override=raw))
    t = D.read_dted(p)
    assert t.elev[1, 0] == -5 and t.twos_complement_values == 1
    assert any("two's complement" in s for s in t.problems)
    assert t.elev[0, 2] == D.VOID                       # the void is not touched


def test_not_dted_and_truncated_are_refused_in_words(tmp_path):
    p = tmp_path / "x.dt1"
    p.write_bytes(b"hello" * 1000)
    with pytest.raises(D.DtedError, match="UHL"):
        D.read_dted(p)
    good = _tile_bytes("0380000N", "0770000W", "380000.0N", "0770000.0W",
                       9000, ELEV)
    p.write_bytes(good[:D.HEADER_LEN + 30])
    t = D.read_dted(p)
    assert "truncated" in t.problems[0]


def test_module_writer_matches_the_standard_and_round_trips(tmp_path):
    p = D.write_dted(tmp_path / "w078" / "n38.dt1", ELEV, 38.0, -78.0, 900, 900)
    raw = p.read_bytes()
    assert len(raw) == D.HEADER_LEN + 4 * (12 + 2 * 5)
    assert raw[0:4] == b"UHL1" and raw[4:12] == b"0780000W"
    assert raw[12:20] == b"0380000N" and raw[47:55] == b"00040005"
    assert raw[80:83] == b"DSI" and raw[80 + 59:80 + 64] == b"DTED1"
    assert raw[80 + 185:80 + 194] == b"380000.0N"
    assert raw[80 + 194:80 + 204] == b"0780000.0W"
    assert raw[728:731] == b"ACC"
    first = raw[D.HEADER_LEN:D.HEADER_LEN + 22]
    assert first[0] == 0xAA and first[8:10] == _sm(100)
    assert int.from_bytes(first[-4:], "big") == sum(first[:-4])
    t = D.read_dted(p)
    assert np.array_equal(t.elev, ELEV) and t.lon0 == -78.0


def test_mosaic_is_continuous_across_tiles_and_edges(tmp_path):
    rows, cols = np.mgrid[0:5, 0:5]

    def field(col_offset):
        return (100 + 10 * rows + 3 * (cols + col_offset)).astype(np.int16)
    D.write_dted(tmp_path / "w078" / "n38.dt1", field(0), 38.0, -78.0, 900, 900)
    D.write_dted(tmp_path / "w077" / "n38.dt1", field(4), 38.0, -77.0, 900, 900)
    (tmp_path / "notes.dt1").write_text("not a tile")
    m = D.DtedMosaic.from_folder(tmp_path)
    assert len(m.tiles) == 2 and len(m.skipped) == 1
    assert "2 DTED tiles" in m.describe()

    def truth(lat, lon):
        return 100 + 10 * (lat - 38.0) * 4 + 3 * (lon + 78.0) * 4
    lat = np.array([38.1, 38.6, 38.5, 39.0, 38.3])
    lon = np.array([-77.9, -76.2, -77.0, -77.5, -76.0])
    assert np.allclose(m.elevation(lat, lon), truth(lat, lon))
    assert np.isnan(m.elevation(37.5, -77.5))[0]
    terr = T.profile(m, 38.2, -77.8, 38.8, -76.3, n=50)
    assert terr.voids == 0
    assert np.allclose(terr.ground_m, truth(terr.lat, terr.lon), atol=1e-6)


# -- terrain ------------------------------------------------------------------
def test_geodesy():
    assert math.isclose(T.haversine_m(38, -77, 39, -77),
                        math.pi / 180 * EARTH_RADIUS_M, rel_tol=1e-12)
    assert math.isclose(T.initial_bearing_deg(0, 0, 0, 1), 90.0)
    assert math.isclose(T.initial_bearing_deg(38, -77, 39, -77), 0.0, abs_tol=1e-9)
    lat, lon = T.destination(38.7, -77.5, 47.0, 12_345.0)
    assert math.isclose(T.haversine_m(38.7, -77.5, lat, lon), 12_345.0, rel_tol=1e-9)
    assert math.isclose(T.initial_bearing_deg(38.7, -77.5, lat, lon), 47.0,
                        abs_tol=1e-6)
    la, lo = T.batch_points(38.7, -77.5, [38.8, 38.6], [-77.4, -77.6], 5)
    assert la.shape == (2, 5) and math.isclose(la[0, -1], 38.8)
    e, n = T.enu_m(38.71, -77.49, 38.7, -77.5)
    lat2, lon2 = T.from_enu(e, n, 38.7, -77.5)
    assert math.isclose(lat2, 38.71) and math.isclose(lon2, -77.49)


def test_bulge_horizon_and_fresnel_numbers():
    # 50 km path, k = 4/3: the midpoint bulge is d1*d2/(2kR)
    assert math.isclose(float(T.earth_bulge_m(25e3, 50e3)),
                        25e3 ** 2 / (2 * 4 / 3 * EARTH_RADIUS_M))
    assert math.isclose(T.radio_horizon_m(10.0) / 1000, 4.12 * math.sqrt(10),
                        rel_tol=0.01)
    lam = T.C_LIGHT / 150e6
    assert math.isclose(float(T.fresnel_radius_m(5e3, 5e3, 150e6)),
                        math.sqrt(lam * 2500.0), rel_tol=1e-12)


def test_line_of_sight_and_fresnel_clearance_over_a_ridge():
    g = GeoGrid(-77.6, 38.6, -77.3, 38.8, 120, 80)
    lat, lon = g.mesh()
    ridge = 200.0 * np.exp(-((lon + 77.45) / 0.01) ** 2)   # a ridge at -77.45
    terr = T.GridTerrain(g, ridge)
    prof = T.profile(terr, 38.7, -77.55, 38.7, -77.35, n=200)
    low = T.clearance(prof, 10.0, 10.0, 150e6)
    assert not low.line_of_sight and low.min_clearance_m < -150
    assert abs(low.at_m - prof.length_m / 2) < 500
    assert "no line of sight" in low.words()
    high = T.clearance(prof, 400.0, 400.0, 150e6)
    assert high.line_of_sight and high.fresnel_clear
    # a grazing path: line of sight, Fresnel zone partly blocked
    graze = T.clearance(prof, 210.0, 210.0, 150e6)
    assert graze.line_of_sight and 0 < graze.min_fresnel_ratio < 0.6
    assert "Fresnel" in graze.words()
    flat = T.profile(T.FlatTerrain(), 38.7, -77.55, 38.7, -77.35, n=50)
    assert T.line_of_sight(flat, 30.0, 30.0)
    # voids along a profile are counted and filled, never silently zero
    holes = T.FunctionTerrain(lambda la, lo: np.where(lo > -77.4, np.nan, 50.0))
    pv = T.profile(holes, 38.7, -77.55, 38.7, -77.35, n=40)
    assert pv.voids > 0 and np.all(np.isfinite(pv.ground_m))
    assert "interpolated" in pv.notes[0]
