# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Highest-posterior-density regions as polygons (plan E1, E3, E4).

A posterior on a grid becomes the shapes an analyst reads on the map: the
SMALLEST region holding 50 / 90 / 95 % of the probability. For a lopsided
or multi-lobed posterior — two bearings that cross twice, a sense-ambiguous
bearing, a fingerprint that matches two streets — that region is several
pieces, and every piece is drawn. One ellipse around the mean would put the
truth in the gap between the lobes and call it 95 %.

The threshold is exact on the grid: cells sorted by density, the densest
taken until their mass reaches the level. The polygon is that threshold's
iso-line, traced by `contourpy` (it ships with matplotlib; marching squares
with linear interpolation between cell centres), with the grid padded by a
zero border lying exactly on the grid's edge so every polygon closes inside
the bounds. Membership questions ("is the truth inside the 90 % region?")
are answered on the grid itself, not by the traced line — the line is for
the eye.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from atk_diffusion.geo import products as _products
from atk_diffusion.geo.products import EARTH_RADIUS_M, GeoGrid

LEVELS = (0.5, 0.9, 0.95)


def hpd_threshold(density: np.ndarray, prob: np.ndarray, level: float) -> float:
    """The density t such that cells with density >= t hold `level` of the
    mass (the smallest such region)."""
    if not 0.0 < level <= 1.0:
        raise ValueError("a credible level is in (0, 1]")
    dens = np.asarray(density, dtype=np.float64).ravel()
    p = np.asarray(prob, dtype=np.float64).ravel()
    order = np.argsort(-dens, kind="stable")
    csum = np.cumsum(p[order])
    i = int(np.searchsorted(csum, level * csum[-1] - 1e-15))
    i = min(i, order.size - 1)
    return float(dens[order[i]])


def credible_level(density: np.ndarray, prob: np.ndarray, value: float,
                   own_mass: float | None = None) -> float:
    """The smallest HPD level whose region contains a point of density
    `value`: the mass of every cell denser than it, plus half of the mass
    at exactly that density (a uniform position inside its cell)."""
    dens = np.asarray(density, dtype=np.float64)
    p = np.asarray(prob, dtype=np.float64)
    tot = float(p.sum())
    above = float(p[dens > value].sum())
    tie = float(p[dens == value].sum()) if own_mass is None else float(own_mass)
    return min(1.0, (above + 0.5 * tie) / tot)


def _padded(grid: GeoGrid, density: np.ndarray):
    z = np.asarray(density, dtype=np.float64)[::-1]          # south -> north
    lats = grid.lats()[::-1]
    lons = grid.lons()
    x = np.concatenate([[grid.west], lons, [grid.east]])
    y = np.concatenate([[grid.south], lats, [grid.north]])
    zp = np.zeros((z.shape[0] + 2, z.shape[1] + 2))
    zp[1:-1, 1:-1] = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return x, y, zp


def polygons(grid: GeoGrid, density: np.ndarray, threshold: float) -> list:
    """[(exterior (n, 2) lon/lat, [holes])] where density >= threshold."""
    import contourpy
    x, y, z = _padded(grid, density)
    zmax = float(z.max())
    if not np.isfinite(threshold) or zmax <= 0 or threshold > zmax:
        return []
    t = max(float(threshold), 1e-300)
    cg = contourpy.contour_generator(x, y, z,
                                     fill_type=contourpy.FillType.OuterOffset)
    pts_list, offs_list = cg.filled(t, zmax * 2.0 + 1.0)
    out = []
    for pts, offs in zip(pts_list, offs_list):
        rings = [pts[offs[i]:offs[i + 1]] for i in range(len(offs) - 1)]
        rings = [r for r in rings if r.shape[0] >= 3]
        if rings:
            out.append((rings[0], rings[1:]))
    return out


def ring_area_m2(ring_lonlat: np.ndarray) -> float:
    """Area of a ring (lon/lat) on a local plane — a few metres' error per
    km² at the scales here."""
    r = np.asarray(ring_lonlat, dtype=np.float64)
    lat0 = float(np.mean(r[:, 1]))
    k = math.pi / 180.0 * EARTH_RADIUS_M
    x = (r[:, 0] - r[0, 0]) * k * math.cos(math.radians(lat0))
    y = (r[:, 1] - r[0, 1]) * k
    return abs(0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def _in_ring(lon: float, lat: float, ring: np.ndarray) -> bool:
    x, y = ring[:, 0], ring[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cross = ((y > lat) != (y2 > lat))
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = x + (lat - y) * (x2 - x) / (y2 - y)
    return bool(np.count_nonzero(cross & (lon < xi)) % 2)


def point_in_polygons(lat: float, lon: float, polys: list) -> bool:
    for ext, holes in polys:
        if _in_ring(lon, lat, ext) and not any(_in_ring(lon, lat, h) for h in holes):
            return True
    return False


@dataclass
class Region:
    level: float
    threshold: float
    polygons: list = field(default_factory=list)
    area_km2: float = 0.0           # exact on the grid (cells above threshold)
    lobes: int = 0


def hpd_regions(grid: GeoGrid, density: np.ndarray, prob: np.ndarray,
                levels=LEVELS) -> list[Region]:
    area = grid.cell_area_m2() * np.ones(grid.shape)
    out = []
    for lv in levels:
        t = hpd_threshold(density, prob, lv)
        polys = polygons(grid, density, t)
        out.append(Region(float(lv), t, polys,
                          float(area[np.asarray(density) >= t].sum() / 1e6),
                          len(polys)))
    return out


def region_features(regions: list[Region], props: dict | None = None) -> list:
    """One MultiPolygon feature per credible level (largest level first, so
    the map draws the 95 % region under the 50 %)."""
    feats = []
    for r in sorted(regions, key=lambda r: -r.level):
        if not r.polygons:
            continue
        p = {"credible_level": r.level, "area_km2": round(r.area_km2, 4),
             "lobes": r.lobes, "label": f"{int(round(r.level * 100))} % region",
             **(props or {})}
        feats.append(_products.multipolygon_feature(r.polygons, p))
    return feats
