# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Terrain along a path: great-circle profiles, earth curvature, line of
sight, and Fresnel-zone clearance (plan E5; the ground under E1-E4).

Every propagation model past free space asks the same question of the
ground between two antennas, so it is asked once, here:

* **The geodesy** — great-circle distance, initial bearing, destination and
  intermediate points on a sphere of the mean earth radius. Spherical, not
  ellipsoidal, on purpose: the bearing error against WGS-84 is under 0.1 deg
  and the distance error under 0.5 %, both far inside every DF sigma and
  propagation uncertainty these tools carry — and the formulas vectorise
  over a whole grid of cells at once.
* **The profile** — the ground sampled at even steps along the great circle
  from a DTED mosaic (`dted.DtedMosaic`), a gridded raster (`GridTerrain`),
  a function, or flat ground. Void posts are counted and filled from their
  neighbours along the path; the count travels with the profile.
* **Curvature with k = 4/3** — standard refraction bends radio paths so the
  earth looks flatter: a point at d1 from one end and d2 from the other sits
  d1*d2 / (2 k R) above the chord. Added to the terrain, it turns the
  question into straight lines over an "effective" profile.
* **Line of sight and Fresnel clearance** — the clearance of the straight
  antenna-to-antenna line over the effective profile, and the same
  clearance in units of the first Fresnel radius sqrt(lambda d1 d2 / D).
  The usual rule: 0.6 of the first zone clear is "free space"; less is
  partial obstruction; a negative clearance is diffraction, and the knife-
  edge parameter v = -sqrt(2) * clearance / F1 is what `propagation` feeds
  the diffraction models.

LIMITS. k = 4/3 is the standard atmosphere; ducting and sub-refraction (k
from below 1 to infinite) are weather, not terrain, and are not modelled.
The profile is bare earth: DTED has no buildings or trees (clutter is what
the learned residual layer of E2/E5 is for).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from atk_diffusion.geo.products import EARTH_RADIUS_M, GeoGrid

K_STANDARD = 4.0 / 3.0
C_LIGHT = 299_792_458.0


# ---------------------------------------------------------------------------
# Geodesy on the sphere (vectorised)
# ---------------------------------------------------------------------------
def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2, dtype=np.float64) - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def initial_bearing_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    """True bearing from point 1 toward point 2, degrees clockwise from
    north, in [0, 360)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2, dtype=np.float64) - lon1)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.mod(np.degrees(np.arctan2(y, x)), 360.0)


def destination(lat, lon, bearing_deg, dist_m) -> tuple[np.ndarray, np.ndarray]:
    p1 = np.radians(lat)
    l1 = np.radians(lon)
    b = np.radians(bearing_deg)
    d = np.asarray(dist_m, dtype=np.float64) / EARTH_RADIUS_M
    p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(b))
    l2 = l1 + np.arctan2(np.sin(b) * np.sin(d) * np.cos(p1),
                         np.cos(d) - np.sin(p1) * np.sin(p2))
    return np.degrees(p2), (np.degrees(l2) + 540.0) % 360.0 - 180.0


def great_circle_points(lat1, lon1, lat2, lon2, fractions) -> tuple[np.ndarray, np.ndarray]:
    """Points at `fractions` (0..1) of the way along the great circle."""
    f = np.asarray(fractions, dtype=np.float64)
    p1, l1, p2, l2 = map(math.radians, (lat1, lon1, lat2, lon2))
    v1 = np.array([math.cos(p1) * math.cos(l1), math.cos(p1) * math.sin(l1),
                   math.sin(p1)])
    v2 = np.array([math.cos(p2) * math.cos(l2), math.cos(p2) * math.sin(l2),
                   math.sin(p2)])
    om = math.acos(max(-1.0, min(1.0, float(v1 @ v2))))
    if om < 1e-12:
        return np.full(f.shape, float(lat1)), np.full(f.shape, float(lon1))
    s = math.sin(om)
    a = np.sin((1 - f) * om) / s
    b = np.sin(f * om) / s
    v = a[:, None] * v1 + b[:, None] * v2
    lat = np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1)))
    lon = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
    return lat, lon


def batch_points(lat0, lon0, lats, lons, n: int) -> tuple[np.ndarray, np.ndarray]:
    """n points from one origin to each of many ends, along great circles:
    (M, n) latitude and longitude arrays. Vectorised slerp."""
    lats = np.asarray(lats, dtype=np.float64).ravel()
    lons = np.asarray(lons, dtype=np.float64).ravel()
    p1, l1 = math.radians(lat0), math.radians(lon0)
    v1 = np.array([math.cos(p1) * math.cos(l1), math.cos(p1) * math.sin(l1),
                   math.sin(p1)])
    p2, l2 = np.radians(lats), np.radians(lons)
    v2 = np.stack([np.cos(p2) * np.cos(l2), np.cos(p2) * np.sin(l2),
                   np.sin(p2)], axis=1)
    om = np.arccos(np.clip(v2 @ v1, -1.0, 1.0))
    f = np.linspace(0.0, 1.0, int(n))
    s = np.sin(om)
    small = s < 1e-12
    s = np.where(small, 1.0, s)
    a = np.sin((1 - f)[None, :] * om[:, None]) / s[:, None]
    b = np.sin(f[None, :] * om[:, None]) / s[:, None]
    a = np.where(small[:, None], 1.0 - f[None, :], a)
    b = np.where(small[:, None], f[None, :], b)
    v = a[..., None] * v1[None, None, :] + b[..., None] * v2[:, None, :]
    lat = np.degrees(np.arcsin(np.clip(v[..., 2], -1, 1)))
    lon = np.degrees(np.arctan2(v[..., 1], v[..., 0]))
    return lat, lon


def enu_m(lat, lon, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """(east, north) metres in a local tangent plane at (lat0, lon0). For
    distances of tens of km; beyond that use the great-circle functions."""
    k = math.pi / 180.0 * EARTH_RADIUS_M
    e = (np.asarray(lon, dtype=np.float64) - lon0) * k * math.cos(math.radians(lat0))
    n = (np.asarray(lat, dtype=np.float64) - lat0) * k
    return e, n


def from_enu(east, north, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    k = math.pi / 180.0 * EARTH_RADIUS_M
    lat = lat0 + np.asarray(north, dtype=np.float64) / k
    lon = lon0 + np.asarray(east, dtype=np.float64) / (k * math.cos(math.radians(lat0)))
    return lat, lon


def wrap180(deg) -> np.ndarray:
    return (np.asarray(deg, dtype=np.float64) + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# Terrain sources (anything with .elevation(lat, lon))
# ---------------------------------------------------------------------------
class FlatTerrain:
    """Ground at a constant height (default sea level) — the sanity line."""

    def __init__(self, height_m: float = 0.0):
        self.height_m = float(height_m)

    def elevation(self, lat, lon) -> np.ndarray:
        return np.full(np.broadcast(np.asarray(lat), np.asarray(lon)).shape,
                       self.height_m, dtype=np.float64)

    def describe(self) -> str:
        return f"flat ground at {self.height_m:g} m (no terrain loaded)"


class GridTerrain:
    """Heights on a `GeoGrid` (row 0 north), bilinear — synthetic terrain,
    or a DEM GeoTIFF read with `products.read_geotiff`."""

    def __init__(self, grid: GeoGrid, heights, label: str = "gridded terrain"):
        self.grid = grid
        self.heights = np.asarray(heights, dtype=np.float64)
        if self.heights.shape != grid.shape:
            raise ValueError(f"heights {self.heights.shape} do not fit the "
                             f"grid {grid.shape}")
        self.label = label

    @classmethod
    def from_raster(cls, raster) -> "GridTerrain":
        return cls(raster.grid, raster.masked(), label=f"raster {raster.path}")

    def elevation(self, lat, lon) -> np.ndarray:
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        shape = np.broadcast(lat, lon).shape
        v = self.grid.sample(self.heights, np.broadcast_to(lat, shape).ravel(),
                             np.broadcast_to(lon, shape).ravel())
        return v.reshape(shape)

    def describe(self) -> str:
        return f"{self.label} ({self.grid.width} x {self.grid.height})"


class FunctionTerrain:
    def __init__(self, fn, label: str = "terrain function"):
        self.fn = fn
        self.label = label

    def elevation(self, lat, lon) -> np.ndarray:
        return np.asarray(self.fn(np.asarray(lat, dtype=np.float64),
                                  np.asarray(lon, dtype=np.float64)),
                          dtype=np.float64)

    def describe(self) -> str:
        return self.label


def describe_terrain(terrain) -> str:
    d = getattr(terrain, "describe", None)
    return d() if callable(d) else type(terrain).__name__


def _fill_voids(z: np.ndarray) -> tuple[np.ndarray, int]:
    """Fill NaN samples along a profile (last axis) by linear interpolation
    between valid neighbours; a profile with no valid sample is sea level.
    Returns (filled, how many were filled)."""
    z = np.array(z, dtype=np.float64, copy=True)
    bad = ~np.isfinite(z)
    n = int(bad.sum())
    if not n:
        return z, 0
    flat = z.reshape(-1, z.shape[-1])
    bflat = bad.reshape(-1, z.shape[-1])
    x = np.arange(z.shape[-1])
    for i in np.nonzero(bflat.any(axis=1))[0]:
        good = ~bflat[i]
        if good.any():
            flat[i, bflat[i]] = np.interp(x[bflat[i]], x[good], flat[i, good])
        else:
            flat[i, :] = 0.0
    return flat.reshape(z.shape), n


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------
@dataclass
class Profile:
    """Ground along a great circle, end to end. `ground_m[0]` is under the
    first antenna. `voids` samples had no height and were filled."""
    distance_m: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    ground_m: np.ndarray
    voids: int = 0
    notes: list = field(default_factory=list)

    @property
    def length_m(self) -> float:
        return float(self.distance_m[-1])

    @property
    def spacing_m(self) -> float:
        return float(self.distance_m[1] - self.distance_m[0]) \
            if self.distance_m.size > 1 else 0.0

    def itm_pfl(self) -> list[float]:
        """The profile as ITM wants it: [intervals, spacing_m, z0..zN]."""
        return [int(self.ground_m.size - 1), self.spacing_m] + \
            [float(v) for v in self.ground_m]


def profile(terrain, lat1, lon1, lat2, lon2, step_m: float | None = None,
            n: int | None = None, max_points: int = 4000) -> Profile:
    """Sample the ground from (lat1, lon1) to (lat2, lon2) at even steps
    (`step_m`, default ~ 30 m, or exactly `n` points)."""
    d = float(haversine_m(lat1, lon1, lat2, lon2))
    if n is None:
        step = float(step_m) if step_m else 30.0
        n = int(min(max_points, max(3, math.ceil(d / step) + 1)))
    n = max(3, int(n))
    f = np.linspace(0.0, 1.0, n)
    lat, lon = great_circle_points(lat1, lon1, lat2, lon2, f)
    z = np.asarray(terrain.elevation(lat, lon), dtype=np.float64).reshape(-1)
    z, voids = _fill_voids(z)
    notes = []
    if voids:
        notes.append(f"{voids} of {n} profile samples had no terrain height "
                     "(DTED void or outside the loaded tiles) and were "
                     "interpolated from their neighbours")
    return Profile(distance_m=f * d, lat=lat, lon=lon, ground_m=z,
                   voids=voids, notes=notes)


def batch_profiles(terrain, lat0, lon0, lats, lons, n: int):
    """Profiles from one transmitter to many cells with n samples each:
    (distances (M,), ground (M, n), voids filled (int)). Vectorised: one
    call to the terrain for all M*n points."""
    plat, plon = batch_points(lat0, lon0, lats, lons, n)
    z = np.asarray(terrain.elevation(plat.ravel(), plon.ravel()),
                   dtype=np.float64).reshape(plat.shape)
    z, voids = _fill_voids(z)
    d = haversine_m(lat0, lon0, np.asarray(lats).ravel(), np.asarray(lons).ravel())
    return d, z, voids


# ---------------------------------------------------------------------------
# Curvature, line of sight, Fresnel
# ---------------------------------------------------------------------------
def earth_bulge_m(d_m, total_m, k: float = K_STANDARD) -> np.ndarray:
    """Height of the earth's surface above the chord at d from one end of a
    path of length `total_m`, for effective radius k R."""
    d = np.asarray(d_m, dtype=np.float64)
    return d * (np.asarray(total_m, dtype=np.float64) - d) / (2.0 * k * EARTH_RADIUS_M)


def effective_profile(prof: Profile, k: float = K_STANDARD) -> np.ndarray:
    return prof.ground_m + earth_bulge_m(prof.distance_m, prof.length_m, k)


def fresnel_radius_m(d1_m, d2_m, freq_hz: float, zone: int = 1) -> np.ndarray:
    lam = C_LIGHT / float(freq_hz)
    d1 = np.asarray(d1_m, dtype=np.float64)
    d2 = np.asarray(d2_m, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(zone * lam * d1 * d2 / np.maximum(d1 + d2, 1e-9))


@dataclass
class Clearance:
    """The antenna-to-antenna line over the effective profile."""
    line_of_sight: bool
    min_clearance_m: float          # worst clearance (negative: obstructed)
    at_m: float                     # where along the path
    min_fresnel_ratio: float        # worst clearance / first Fresnel radius
    fresnel_clear: bool             # >= 0.6 of the first zone everywhere
    clearance_m: np.ndarray         # per interior sample
    v: np.ndarray                   # knife-edge parameter per interior sample

    def words(self) -> str:
        if not self.line_of_sight:
            return (f"no line of sight: the terrain at {self.at_m / 1000:.2f} km "
                    f"stands {-self.min_clearance_m:.0f} m into the path "
                    "(k = 4/3 earth)")
        if not self.fresnel_clear:
            return (f"line of sight, but only {self.min_fresnel_ratio:.2f} of "
                    f"the first Fresnel zone is clear at "
                    f"{self.at_m / 1000:.2f} km (0.6 is the free-space rule)")
        return "line of sight with the first Fresnel zone 60 % clear"


def clearance(prof: Profile, h_tx_agl: float, h_rx_agl: float,
              freq_hz: float | None = None, k: float = K_STANDARD) -> Clearance:
    """Line of sight and Fresnel clearance between antennas `h_tx_agl` and
    `h_rx_agl` metres above the ground at the two ends."""
    D = prof.length_m
    if D <= 0:
        raise ValueError("a path needs two different ends")
    ze = effective_profile(prof, k)
    a = prof.ground_m[0] + float(h_tx_agl)
    b = prof.ground_m[-1] + float(h_rx_agl)
    d = prof.distance_m
    line = a + (b - a) * d / D
    inner = slice(1, d.size - 1)
    c = (line - ze)[inner]
    di = d[inner]
    if c.size == 0:
        c = np.array([np.inf])
        di = np.array([D / 2])
    i = int(np.argmin(c))
    j = i
    if freq_hz:
        f1 = fresnel_radius_m(di, D - di, freq_hz)
        ratio = c / np.maximum(f1, 1e-9)
        j = int(np.argmin(ratio))
        v = -math.sqrt(2.0) * ratio
        min_ratio = float(ratio[j])
    else:
        v = np.full(c.shape, np.nan)
        min_ratio = float("nan")
    los = bool(np.all(c > 0))
    # obstructed: where the terrain stands deepest into the path; clear:
    # where the Fresnel zone is tightest
    at = float(di[i]) if not los else float(di[j])
    return Clearance(line_of_sight=los, min_clearance_m=float(c[i]), at_m=at,
                     min_fresnel_ratio=min_ratio,
                     fresnel_clear=bool(freq_hz) and bool(min_ratio >= 0.6),
                     clearance_m=c, v=v)


def line_of_sight(prof: Profile, h_tx_agl: float, h_rx_agl: float,
                  k: float = K_STANDARD) -> bool:
    return clearance(prof, h_tx_agl, h_rx_agl, None, k).line_of_sight


def radio_horizon_m(h_m: float, k: float = K_STANDARD) -> float:
    """Distance to the smooth-earth radio horizon from height h."""
    return math.sqrt(2.0 * k * EARTH_RADIUS_M * max(float(h_m), 0.0))
