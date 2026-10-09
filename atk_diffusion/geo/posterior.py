# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Geolocation as a cloud — the posterior of an emitter's position given
bearings, soft ranges and a prior (plan E1).

*"Sample plausible emitter positions given the bearings and terrain, with
the lobes, instead of one dot with an ellipse."* A fix is a probability
distribution over the map, so this module computes that distribution —
exactly, on a grid — and hands back what an analyst needs from it: the
most probable point, the posterior mean, a cloud of samples, and the
highest-posterior-density regions at 50 / 90 / 95 % (`contours`), which are
as many pieces as the evidence makes them.

THE MODEL, STATED SO IT CAN BE DISAGREED WITH.
* A bearing is the true great-circle bearing from the receiver to the
  emitter plus Gaussian error of the stated sigma (wrapped, in degrees).
  `ambiguous_180=True` is a sense-ambiguous bearing (two-element
  interferometer, Watson-Watt without sense): the likelihood is the even
  mixture of theta and theta+180, and the posterior shows both lobes.
* `bias_sigma_deg` is the error ATK's own coverage audit found decisive
  (`atk/core/siga/coverage.py`: sixteen bearings with an unmodelled 3 deg
  alignment error put the truth outside the stated 95 % ellipse in none of
  forty trials). Bearings that share a `station` share one unknown bias of
  that sigma; it is integrated out exactly (rank-one update), so many
  bearings from one mis-aligned array do not shrink the cloud as if they
  were independent.
* `outlier_prob` mixes each bearing with a uniform: a multipath bearing
  pointing nowhere costs a little everywhere instead of everything
  somewhere.
* Soft range (arXiv 2305.13911 — "all probable values, not one"): a range
  likelihood around a receiver, as a discrete pdf over range, a log-normal,
  or a received-strength path-loss model with log-normal shadowing.
* The prior is uniform over the SEARCH AREA (`search_radius_m` around the
  receivers), times an optional prior raster or function — a land mask, a
  terrain-derived weight. A posterior that reaches the edge of the search
  area is unbounded in range, and the summary says so: the contour is then
  the prior's edge, not the evidence's.

THE GRID. A coarse pass over the whole search area finds where the
posterior lives; a fine grid (up to `max_cells`) is laid over that box and
the posterior recomputed there, so a 50 m fix in a 60 km search area is
resolved. Probabilities include the cell areas (equal-angle cells are not
equal-area). Sampling from the grid is exact for the gridded density.

HONESTY IS MEASURED, NOT CLAIMED. `experiments.geo_eval` runs hundreds of
simulated DF geometries and counts how often the truth falls inside the
stated 90 % region. The region is exactly as honest as the sigma it is
given — which is why that experiment also runs a deliberately understated
sigma and must catch it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from atk_diffusion import provenance as _prov
from atk_diffusion.geo import contours as _contours
from atk_diffusion.geo import products as _products
from atk_diffusion.geo import terrain as _terrain
from atk_diffusion.geo.products import EARTH_RADIUS_M, GeoGrid

_prov.METHOD_TIERS.setdefault("bearing_posterior", "inferred")
_prov.METHOD_TIERS.setdefault("df_bearing", "measured")


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------
@dataclass
class Bearing:
    lat: float
    lon: float
    bearing_deg: float               # true north, clockwise
    sigma_deg: float = 3.0
    time: float | str | None = None
    station: str = ""                # bearings from one array share its bias
    ambiguous_180: bool = False

    def to_json(self) -> dict:
        return {"lat": self.lat, "lon": self.lon, "bearing_deg": self.bearing_deg,
                "sigma_deg": self.sigma_deg, "time": self.time,
                "station": self.station, "ambiguous_180": self.ambiguous_180}


@dataclass
class SoftRange:
    """A range likelihood around (lat, lon). Give ONE of:
    `ranges_m` + `pdf` (a discrete distribution over range), `median_m` +
    `sigma_log10` (log-normal), or `rss_dbm` with a path-loss model
    (`ref_dbm` at `ref_m`, `exponent`, `shadow_db`)."""
    lat: float
    lon: float
    ranges_m: tuple | None = None
    pdf: tuple | None = None
    median_m: float | None = None
    sigma_log10: float | None = None
    rss_dbm: float | None = None
    ref_dbm: float = -40.0
    ref_m: float = 1.0
    exponent: float = 2.0
    shadow_db: float = 6.0
    label: str = ""

    def loglik(self, d_m) -> np.ndarray:
        d = np.maximum(np.asarray(d_m, dtype=np.float64), 1e-3)
        if self.ranges_m is not None and self.pdf is not None:
            r = np.asarray(self.ranges_m, dtype=np.float64)
            p = np.asarray(self.pdf, dtype=np.float64)
            o = np.argsort(r)
            v = np.interp(d, r[o], p[o], left=0.0, right=0.0)
            return np.log(np.maximum(v, 1e-300))
        if self.median_m is not None and self.sigma_log10:
            z = (np.log10(d) - math.log10(self.median_m)) / self.sigma_log10
            return -0.5 * z ** 2 - np.log(d)          # density in d, not log d
        if self.rss_dbm is not None:
            pred = self.ref_dbm - 10.0 * self.exponent * np.log10(d / self.ref_m)
            return -0.5 * ((self.rss_dbm - pred) / self.shadow_db) ** 2
        raise ValueError("a SoftRange needs ranges_m+pdf, median_m+sigma_log10, "
                         "or rss_dbm with a path-loss model")

    def to_json(self) -> dict:
        return {k: (list(v) if isinstance(v, (tuple, np.ndarray)) else v)
                for k, v in self.__dict__.items()}


def land_prior(terrain, sea_level_m: float = 0.0, water_weight: float = 1e-3):
    """A prior that an emitter is on land: weight 1 where the terrain is
    above `sea_level_m`, `water_weight` (not zero: coastlines in DTED are
    approximate and a boat is possible) elsewhere."""
    def fn(lat, lon):
        z = np.asarray(terrain.elevation(lat, lon), dtype=np.float64)
        return np.where(np.isfinite(z) & (z > sea_level_m), 1.0, water_weight)
    return fn


def _prior_weights(prior, lat, lon) -> np.ndarray:
    if prior is None:
        return np.ones(lat.shape)
    if callable(prior):
        w = np.asarray(prior(lat, lon), dtype=np.float64)
    elif hasattr(prior, "sample"):                 # a GeoRaster
        w = np.asarray(prior.sample(lat.ravel(), lon.ravel())).reshape(lat.shape)
        w = np.where(np.isfinite(w), w, 1.0)
    else:
        g, a = prior
        w = g.sample(a, lat.ravel(), lon.ravel()).reshape(lat.shape)
        w = np.where(np.isfinite(w), w, 1.0)
    if np.any(w < 0):
        raise ValueError("prior weights must be >= 0")
    return w


# ---------------------------------------------------------------------------
# Likelihoods on a grid
# ---------------------------------------------------------------------------
def _logsumexp2(a, b):
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def bearing_loglik(lat, lon, bearings, bias_sigma_deg: float = 0.0,
                   outlier_prob: float = 0.0) -> np.ndarray:
    """Sum of bearing log-likelihoods at points (lat, lon arrays)."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    total = np.zeros(lat.shape)
    eps = float(outlier_prob)
    sb = float(bias_sigma_deg)
    groups: dict = {}
    for b in bearings:
        if b.sigma_deg <= 0:
            raise ValueError("a bearing sigma must be positive")
        beta = _terrain.initial_bearing_deg(b.lat, b.lon, lat, lon)
        r = _terrain.wrap180(b.bearing_deg - beta)
        s = float(b.sigma_deg)
        if sb > 0 and not b.ambiguous_180:
            groups.setdefault(b.station or f"_{id(b)}", []).append((r, s))
            continue
        ll = -0.5 * (r / s) ** 2 - math.log(s)
        if b.ambiguous_180:
            r2 = _terrain.wrap180(b.bearing_deg + 180.0 - beta)
            ll = _logsumexp2(ll, -0.5 * (r2 / s) ** 2 - math.log(s)) - math.log(2.0)
        if eps > 0:
            # density per degree: Gaussian 1/(sqrt(2 pi) s), uniform 1/360
            g = ll - 0.5 * math.log(2 * math.pi)
            ll = _logsumexp2(math.log(1 - eps) + g,
                             np.full(g.shape, math.log(eps / 360.0)))
        total += ll
    for members in groups.values():
        inv = np.array([1.0 / s ** 2 for _, s in members])
        a = sum(r ** 2 * w for (r, _), w in zip(members, inv))
        c = sum(r * w for (r, _), w in zip(members, inv))
        q = a - sb ** 2 * c ** 2 / (1.0 + sb ** 2 * inv.sum())
        total += -0.5 * q
    return total


def _evaluate(grid: GeoGrid, bearings, ranges, prior, bias, outliers):
    lat, lon = grid.mesh()
    lp = bearing_loglik(lat, lon, bearings, bias, outliers) if bearings else \
        np.zeros(lat.shape)
    for rg in ranges or ():
        lp = lp + rg.loglik(_terrain.haversine_m(rg.lat, rg.lon, lat, lon))
    w = _prior_weights(prior, lat, lon)
    with np.errstate(divide="ignore"):
        lp = lp + np.log(w)
    return lp


# ---------------------------------------------------------------------------
# The posterior
# ---------------------------------------------------------------------------
class GridPosterior:
    """A posterior over position on a GeoGrid. `logp` is the log of the
    density per unit AREA (likelihood x prior); cell areas are applied
    here. Shared by E1 (bearings), E3 (where-am-I) and E4 (aperture)."""

    def __init__(self, grid: GeoGrid, logp: np.ndarray, *, method: str,
                 notes: list | None = None, meta: dict | None = None,
                 unbounded: bool = False):
        self.grid = grid
        self.method = method
        self.tier = _prov.tier_for(method)
        lp = np.asarray(logp, dtype=np.float64)
        if lp.shape != grid.shape:
            raise ValueError("logp must be on the grid")
        finite = np.isfinite(lp)
        if not finite.any():
            raise ValueError("the evidence rules out every place in the search "
                             "area (the observations contradict each other, or "
                             "the area is too small)")
        area = grid.cell_area_m2() * np.ones(grid.shape)
        m = float(lp[finite].max())
        p = np.where(finite, np.exp(lp - m), 0.0) * area
        self.prob = p / p.sum()
        self.density = self.prob / (area / 1e6)          # per km^2
        self.notes = list(notes or [])
        self.meta = dict(meta or {})
        self.unbounded = bool(unbounded)
        self._regions: dict = {}

    # -- points -----------------------------------------------------------------
    def map_point(self) -> tuple[float, float]:
        """The densest point, refined inside its cell by a parabola through
        its neighbours' log density."""
        r, c = np.unravel_index(int(np.argmax(self.density)), self.grid.shape)
        ld = np.log(np.maximum(self.density, 1e-300))

        def off(a, b, cc):
            den = a - 2 * cc + b
            return 0.0 if den >= 0 else float(np.clip(0.5 * (a - b) / den, -0.5, 0.5))
        dr = off(ld[r - 1, c], ld[r + 1, c], ld[r, c]) \
            if 0 < r < self.grid.height - 1 else 0.0
        dc = off(ld[r, c - 1], ld[r, c + 1], ld[r, c]) \
            if 0 < c < self.grid.width - 1 else 0.0
        return (float(self.grid.lats()[r] - dr * self.grid.dlat),
                float(self.grid.lons()[c] + dc * self.grid.dlon))

    def mean_point(self) -> tuple[float, float]:
        lat, lon = self.grid.mesh()
        return float(np.sum(self.prob * lat)), float(np.sum(self.prob * lon))

    def covariance_m(self) -> np.ndarray:
        """2x2 covariance (east, north) in metres about the mean."""
        lat0, lon0 = self.mean_point()
        lat, lon = self.grid.mesh()
        e, n = _terrain.enu_m(lat, lon, lat0, lon0)
        return np.array([[np.sum(self.prob * e * e), np.sum(self.prob * e * n)],
                         [np.sum(self.prob * e * n), np.sum(self.prob * n * n)]])

    def samples(self, n: int, rng=None) -> tuple[np.ndarray, np.ndarray]:
        rng = rng if rng is not None else np.random.default_rng()
        idx = rng.choice(self.prob.size, size=int(n), p=self.prob.ravel())
        r, c = np.unravel_index(idx, self.grid.shape)
        lat = self.grid.lats()[r] + (rng.random(n) - 0.5) * self.grid.dlat
        lon = self.grid.lons()[c] + (rng.random(n) - 0.5) * self.grid.dlon
        return lat, lon

    # -- regions ---------------------------------------------------------------
    def threshold(self, level: float) -> float:
        return _contours.hpd_threshold(self.density, self.prob, level)

    def credible_level(self, lat: float, lon: float) -> float:
        """The smallest credible level whose HPD region holds the point
        (1.0 outside the search area)."""
        if not bool(self.grid.contains(lat, lon)):
            return 1.0
        r, c = self.grid.rowcol(lat, lon)
        r = int(np.clip(np.round(r), 0, self.grid.height - 1))
        c = int(np.clip(np.round(c), 0, self.grid.width - 1))
        return _contours.credible_level(self.density, self.prob,
                                        float(self.density[r, c]),
                                        float(self.prob[r, c]))

    def contains(self, lat: float, lon: float, level: float) -> bool:
        return self.credible_level(lat, lon) <= level

    def regions(self, levels=_contours.LEVELS) -> list:
        key = tuple(levels)
        if key not in self._regions:
            self._regions[key] = _contours.hpd_regions(self.grid, self.density,
                                                       self.prob, levels)
        return self._regions[key]

    def edge_mass(self) -> float:
        p = self.prob
        return float(p[0, :].sum() + p[-1, :].sum() + p[1:-1, 0].sum()
                     + p[1:-1, -1].sum())

    def summary(self, levels=_contours.LEVELS) -> dict:
        la, lo = self.map_point()
        regs = self.regions(levels)
        words = [f"most probable point {abs(la):.5f} {'N' if la >= 0 else 'S'} "
                 f"{abs(lo):.5f} {'E' if lo >= 0 else 'W'}"]
        for r in regs:
            words.append(f"{int(round(r.level * 100))} % region {r.area_km2:,.3g} km2"
                         f" in {r.lobes} piece{'s' if r.lobes != 1 else ''}")
        if self.unbounded:
            words.append("the probability runs to the edge of the search area — "
                         "the evidence does not bound the range, and the region "
                         "is clipped by the search area, not by the evidence")
        return {"map": [la, lo], "mean": list(self.mean_point()),
                "regions": [{"level": r.level, "area_km2": r.area_km2,
                             "lobes": r.lobes} for r in regs],
                "unbounded": self.unbounded, "tier": self.tier,
                "words": "; ".join(words) + "."}

    # -- products ----------------------------------------------------------------
    def to_products(self, rf, *, kind: str = "tracks", run: str | None = None,
                    label: str = "", n_samples: int = 400, seed: int = 0,
                    extra_features: list | None = None, params: dict | None = None,
                    card=None) -> str:
        """Write the posterior as a product: density GeoTIFF (per km2), the
        HPD regions, the MAP and mean, a cloud of samples; plus any extra
        features (bearings, the route, the truth in a test)."""
        summ = self.summary()
        pr = _products.ProductRun(rf, kind, run, label=label or self.method,
                                  tier=self.tier, method=self.method,
                                  params={**(params or {}), **self.meta},
                                  card=card, description=summ["words"])
        pr.add_geotiff("posterior_density.tif", self.density.astype(np.float32),
                       self.grid, tier=self.tier,
                       tags={"atk:units": "probability per km2"},
                       layer={"name": "posterior density", "role": "posterior",
                              "units": "1/km2"})
        pr.add_geojson("regions.geojson",
                       _contours.region_features(self.regions(),
                                                 {"method": self.method}),
                       tier=self.tier, layer={"name": "credible regions",
                                              "role": "regions"})
        la, lo = self.map_point()
        ma, mo = self.mean_point()
        pr.add_geojson("estimate.geojson",
                       [_products.point_feature(la, lo, {"kind": "most probable point"}),
                        _products.point_feature(ma, mo, {"kind": "posterior mean"})],
                       tier=self.tier, layer={"name": "estimate", "role": "estimate"})
        slat, slon = self.samples(n_samples, np.random.default_rng(seed))
        pr.add_geojson("samples.geojson",
                       [_products.point_feature(a, o, {"kind": "posterior sample"})
                        for a, o in zip(slat, slon)],
                       tier=self.tier, layer={"name": "posterior samples (the cloud)",
                                              "role": "samples"})
        for name, feats, tier in extra_features or ():
            pr.add_geojson(name, feats, tier=tier,
                           layer={"name": name.rsplit(".", 1)[0], "role": "evidence"})
        pr.finish(summary=summ, notes=self.notes)
        return str(pr.dir)


# ---------------------------------------------------------------------------
# E1: from bearings
# ---------------------------------------------------------------------------
def bearing_features(bearings, length_m: float) -> list:
    out = []
    for b in bearings:
        for k, th in enumerate([b.bearing_deg] + ([b.bearing_deg + 180.0]
                                                  if b.ambiguous_180 else [])):
            la, lo = _terrain.destination(b.lat, b.lon, th, length_m)
            out.append(_products.line_feature(
                [b.lat, float(la)], [b.lon, float(lo)],
                {"bearing_deg": float(th % 360.0), "sigma_deg": b.sigma_deg,
                 "station": b.station, "time": b.time,
                 "ambiguous_twin": bool(k)}))
    return out


def adaptive_posterior(evaluate, sites_lat, sites_lon, *, method: str,
                       search_radius_m: float = 30_000.0,
                       cell_m: float | None = None, max_cells: int = 250_000,
                       grid: GeoGrid | None = None, meta: dict | None = None,
                       notes: list | None = None) -> GridPosterior:
    """Coarse pass over the whole search area (`search_radius_m` around the
    sites), then a fine grid over where the posterior lives. `evaluate(grid)`
    returns the log density per unit area on that grid."""
    notes = list(notes or [])
    unbounded = False
    if grid is None:
        coarse = GeoGrid.covering(sites_lat, sites_lon, search_radius_m,
                                  max(search_radius_m / 80.0, 10.0))
        lp = evaluate(coarse)
        area = coarse.cell_area_m2() * np.ones(coarse.shape)
        la = lp + np.log(area)
        if not np.isfinite(la).any():
            raise ValueError("the evidence rules out every place in the search "
                             "area (the observations contradict each other, "
                             "or the area is too small)")
        keep =np.isfinite(la) & (la >= np.nanmax(la[np.isfinite(la)]) - 25.0)
        rows = np.nonzero(keep.any(axis=1))[0]
        cols = np.nonzero(keep.any(axis=0))[0]
        r0, r1 = max(rows[0] - 2, 0), min(rows[-1] + 2, coarse.height - 1)
        c0, c1 = max(cols[0] - 2, 0), min(cols[-1] + 2, coarse.width - 1)
        # mass on the coarse grid's border: the range is not bounded
        pc = np.where(np.isfinite(la), np.exp(la - np.nanmax(la)), 0.0)
        pc /= pc.sum()
        border = pc[0].sum() + pc[-1].sum() + pc[1:-1, 0].sum() + pc[1:-1, -1].sum()
        if border > 0.01:
            unbounded = True
        lats, lons = coarse.lats(), coarse.lons()
        north = lats[r0] + 0.5 * coarse.dlat
        south = lats[r1] - 0.5 * coarse.dlat
        west = lons[c0] - 0.5 * coarse.dlon
        east = lons[c1] + 0.5 * coarse.dlon
        k = math.pi / 180.0 * EARTH_RADIUS_M
        h_m = (north - south) * k
        w_m = (east - west) * k * math.cos(math.radians(0.5 * (north + south)))
        cell = cell_m or max(math.sqrt(h_m * w_m / max_cells), 2.0)
        nh = int(min(max(8, math.ceil(h_m / cell)), max_cells // 8))
        nw = int(min(max(8, math.ceil(w_m / cell)), max_cells // nh))
        grid = GeoGrid(west, south, east, north, nw, nh)
    lp = evaluate(grid)
    post = GridPosterior(grid, lp, method=method, notes=notes, meta=meta,
                         unbounded=unbounded)
    if not unbounded and post.edge_mass() > 0.01:
        post.unbounded = True
    return post


def locate(bearings=(), ranges=(), prior=None, *, grid: GeoGrid | None = None,
           search_radius_m: float = 30_000.0, cell_m: float | None = None,
           max_cells: int = 250_000, bias_sigma_deg: float = 0.0,
           outlier_prob: float = 0.0, method: str = "bearing_posterior") -> GridPosterior:
    """The posterior of an emitter's position (see the module docstring)."""
    bearings = list(bearings or [])
    ranges = list(ranges or [])
    if not bearings and not ranges:
        raise ValueError("a fix needs at least one bearing or range")
    if not 0.0 <= outlier_prob < 1.0:
        raise ValueError("outlier_prob is a probability below 1")
    sites_lat = [b.lat for b in bearings] + [r.lat for r in ranges]
    sites_lon = [b.lon for b in bearings] + [r.lon for r in ranges]
    meta = {"bearings": [b.to_json() for b in bearings],
            "ranges": [r.to_json() for r in ranges],
            "bias_sigma_deg": bias_sigma_deg, "outlier_prob": outlier_prob,
            "search_radius_m": search_radius_m,
            "prior": None if prior is None else getattr(prior, "__name__",
                                                        type(prior).__name__)}

    def evaluate(g):
        return _evaluate(g, bearings, ranges, prior, bias_sigma_deg, outlier_prob)
    return adaptive_posterior(evaluate, sites_lat, sites_lon, method=method,
                              search_radius_m=search_radius_m, cell_m=cell_m,
                              max_cells=max_cells, grid=grid, meta=meta)
