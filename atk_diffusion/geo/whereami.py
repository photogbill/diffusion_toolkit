# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Where am I, from the spectrum — the radio map inverted (plan E3).

Bill, 2026-10-08: *"I love the map inversion."* A relief team under GPS
jamming or indoors still hears the cell towers and the broadcasters ATK
already speaks. A database of what was heard WHERE (drive data with GPS as
truth) turns what is heard NOW into a position posterior — contours on the
map, not a dot — with no GPS at all.

THE CLASSICAL METHOD (this module), beside the learned one
(`learn.position`, a mixture-density network):

* **The database** (`FingerprintDB`) — every drive record: time, latitude,
  longitude, and features by name: per-cell RSRP ("lte:310-260-1201"), FM
  station RSSI ("fm:88.5"), anything in dB. A feature the receiver did not
  hear is MISSING, and missing means "below the receiver's floor", which is
  information: a strong tower that is absent rules a place out. The floor
  per feature is taken from the data (3 dB under the weakest it was ever
  heard) unless given.
* **The likelihood** — for each database point, the product over features
  of a Student-t on the dB difference (scale `sigma_db`, `nu` degrees of
  freedom): fading and body shadowing make the occasional feature 20 dB
  off, and a Gaussian would let one such feature veto the right place.
* **The posterior** — those likelihoods as weights on the database points,
  spread by a Gaussian kernel of `bandwidth_m` into a density on a grid
  (`posterior.GridPosterior`): MAP, mean, samples, HPD regions.
* **Honest calibration on a held-out route** — features are not
  independent (one tower's fading moves all its sectors), so the raw
  product is over-confident. `calibrate()` chooses the temperature and the
  kernel widths that maximise the log density at the TRUE positions of a
  route not in the database (a proper scoring rule); `evaluate()` then
  reports, on another held-out route, the error distribution AND how often
  the truth falls inside the stated 50 / 90 / 95 % regions.
* **The radio map run backwards** (`MapLocator`) — the database-point
  kernel can only point at places it has driven. `MapLocator` krigs every
  feature into a continuous map with its kriging variance and inverts
  that, so a position between the driven roads is found — and, far from
  every drive, the variance makes the posterior wide by itself. In the
  synthetic drive world it is the better-calibrated of the two; the
  experiment reports both, with the measured coverage, never the hoped
  one.

LIMITS, STATED. The database knows only where it has been: a position off
the driven roads is shown as the nearest place the spectrum resembles, and
the calibrated region is what says how far to trust that. A network that
re-plans (a cell switched off, a new sector) breaks the fingerprint until
the database is re-driven; `evaluate` on a fresh drive is the check.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from atk_diffusion import provenance as _prov
from atk_diffusion.geo import posterior as _post
from atk_diffusion.geo.products import EARTH_RADIUS_M, GeoGrid
from atk_diffusion.geo.terrain import enu_m, from_enu, haversine_m

_prov.METHOD_TIERS.setdefault("fingerprint_kernel", "inferred")

LEVELS = (0.5, 0.9, 0.95)


@dataclass
class DriveRecord:
    t: float | str
    lat: float
    lon: float
    features: dict


def records_from_csv(path, time_col: str = "time", lat_col: str = "lat",
                     lon_col: str = "lon") -> list[DriveRecord]:
    """A drive log: one row per instant; every other numeric column is a
    feature in dB; an empty cell is "not heard"."""
    out = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                lat, lon = float(row[lat_col]), float(row[lon_col])
            except (KeyError, TypeError, ValueError):
                continue
            feats = {}
            for k, v in row.items():
                if k in (time_col, lat_col, lon_col) or v in (None, ""):
                    continue
                try:
                    feats[k] = float(v)
                except ValueError:
                    continue
            out.append(DriveRecord(row.get(time_col, ""), lat, lon, feats))
    return out


class FingerprintDB:
    """Drive records as a matrix: X[i, j] = feature j at point i, NaN when
    not heard."""

    def __init__(self, lat, lon, X, keys, floors=None, t=None):
        self.lat = np.asarray(lat, dtype=np.float64)
        self.lon = np.asarray(lon, dtype=np.float64)
        self.X = np.asarray(X, dtype=np.float64)
        self.keys = list(keys)
        self.t = t
        if self.X.shape != (self.lat.size, len(self.keys)):
            raise ValueError("X must be (points, features)")
        f = np.full(len(self.keys), np.nan)
        for j in range(len(self.keys)):
            col = self.X[:, j]
            col = col[np.isfinite(col)]
            f[j] = (col.min() - 3.0) if col.size else -130.0
        if floors:
            for j, k in enumerate(self.keys):
                if k in floors:
                    f[j] = float(floors[k])
        self.floors = f

    def __len__(self) -> int:
        return int(self.lat.size)

    @classmethod
    def from_records(cls, records, keys=None, min_count: int = 3,
                     floors: dict | None = None) -> "FingerprintDB":
        records = list(records)
        if not records:
            raise ValueError("an empty drive cannot be a database")
        if keys is None:
            count: dict = {}
            for r in records:
                for k in r.features:
                    count[k] = count.get(k, 0) + 1
            keys = sorted(k for k, c in count.items() if c >= min_count)
        if not keys:
            raise ValueError("no feature was heard often enough to be a "
                             "fingerprint")
        X = np.full((len(records), len(keys)), np.nan)
        index = {k: j for j, k in enumerate(keys)}
        for i, r in enumerate(records):
            for k, v in r.features.items():
                j = index.get(k)
                if j is not None and v is not None and math.isfinite(float(v)):
                    X[i, j] = float(v)
        return cls([r.lat for r in records], [r.lon for r in records], X, keys,
                   floors, [r.t for r in records])

    def vector(self, features: dict) -> np.ndarray:
        v = np.full(len(self.keys), np.nan)
        for j, k in enumerate(self.keys):
            x = features.get(k)
            if x is not None and math.isfinite(float(x)):
                v[j] = float(x)
        return v


def kde_on_grid(grid: GeoGrid, lat, lon, w, bandwidth_m: float) -> np.ndarray:
    """Weighted Gaussian kernel density (per m2) on a grid — separable, so
    it costs (points x (rows + cols)) memory, not points x cells."""
    lat0, lon0 = grid.center
    pe, pn = enu_m(lat, lon, lat0, lon0)
    ge, _ = enu_m(np.full(grid.width, lat0), grid.lons(), lat0, lon0)
    _, gn = enu_m(grid.lats(), np.full(grid.height, lon0), lat0, lon0)
    h2 = 2.0 * bandwidth_m ** 2
    gx = np.exp(-(ge[None, :] - pe[:, None]) ** 2 / h2)
    gy = np.exp(-(gn[None, :] - pn[:, None]) ** 2 / h2)
    dens = (gy * np.asarray(w)[:, None]).T @ gx
    return dens / (math.pi * h2)


class KernelLocator:
    """The classical where-am-I (see the module docstring).

    The position density given a fingerprint is a two-scale kernel mixture
    over the database points, weighted by their likelihoods:

        (1 - p_far) * N(x_i, h) + p_far * N(x_i, H)

    `h` is how far a good match is from the truth; `H` (kilometres) with
    weight `p_far` is the honest admission that some fingerprints match the
    wrong street — measured on a held-out route, never assumed zero. All
    four numbers (temperature, h, H, p_far) are chosen by `calibrate`."""

    def __init__(self, db: FingerprintDB, sigma_db: float = 6.0, nu: float = 4.0,
                 temperature: float = 1.0, bandwidth_m: float = 150.0,
                 far_bandwidth_m: float = 2000.0, p_far: float = 0.0):
        self.db = db
        self.sigma_db = float(sigma_db)
        self.nu = float(nu)
        self.temperature = float(temperature)
        self.bandwidth_m = float(bandwidth_m)
        self.far_bandwidth_m = float(far_bandwidth_m)
        self.p_far = float(p_far)
        self.calibration: dict = {}
        self.lat0, self.lon0 = float(np.mean(db.lat)), float(np.mean(db.lon))
        self._e, self._n = enu_m(db.lat, db.lon, self.lat0, self.lon0)

    def params(self) -> dict:
        return {"sigma_db": self.sigma_db, "nu": self.nu,
                "temperature": self.temperature, "bandwidth_m": self.bandwidth_m,
                "far_bandwidth_m": self.far_bandwidth_m, "p_far": self.p_far,
                "database_points": len(self.db), "features": len(self.db.keys)}

    # -- likelihood ------------------------------------------------------------
    def loglik(self, x: np.ndarray) -> np.ndarray:
        X = self.db.X
        obs = np.where(np.isfinite(x), x, self.db.floors)
        ref = np.where(np.isfinite(X), X, self.db.floors[None, :])
        use = np.isfinite(X) | np.isfinite(x)[None, :]
        d = (ref - obs[None, :]) / self.sigma_db
        ll = -0.5 * (self.nu + 1.0) * np.log1p(d ** 2 / self.nu)
        return np.sum(np.where(use, ll, 0.0), axis=1)

    def _vec(self, features) -> np.ndarray:
        return features if isinstance(features, np.ndarray) else \
            self.db.vector(features)

    def weights(self, features) -> np.ndarray:
        ll = self.loglik(self._vec(features)) / self.temperature
        w = np.exp(ll - ll.max())
        return w / w.sum()

    @staticmethod
    def _active(w, mass: float = 1 - 1e-6):
        o = np.argsort(-w)
        k = int(np.searchsorted(np.cumsum(w[o]), mass)) + 1
        return o[:max(k, 1)]

    # -- the density ------------------------------------------------------------
    def _mix(self, d2) -> np.ndarray:
        h2 = 2.0 * self.bandwidth_m ** 2
        out = (1.0 - self.p_far) * np.exp(-d2 / h2) / (math.pi * h2)
        if self.p_far > 0:
            H2 = 2.0 * self.far_bandwidth_m ** 2
            out = out + self.p_far * np.exp(-d2 / H2) / (math.pi * H2)
        return out

    def density_at(self, w, lat, lon) -> np.ndarray:
        """Posterior density (per m2) at points."""
        act = self._active(w)
        e, n = enu_m(np.atleast_1d(lat), np.atleast_1d(lon), self.lat0, self.lon0)
        d2 = (e[:, None] - self._e[act][None, :]) ** 2 + \
            (n[:, None] - self._n[act][None, :]) ** 2
        return self._mix(d2) @ w[act]

    def sample(self, w, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
        i = rng.choice(w.size, size=int(n), p=w)
        far = rng.random(n) < self.p_far
        s = np.where(far, self.far_bandwidth_m, self.bandwidth_m)
        e = self._e[i] + rng.normal(0, 1, n) * s
        nn = self._n[i] + rng.normal(0, 1, n) * s
        return from_enu(e, nn, self.lat0, self.lon0)

    def credible_level(self, features, lat: float, lon: float, n: int = 3000,
                       rng=None) -> float:
        """The HPD level at which the region first holds (lat, lon), by
        Monte Carlo: the share of posterior samples denser than the point."""
        rng = rng if rng is not None else np.random.default_rng(0)
        w = self.weights(features)
        sl, so = self.sample(w, n, rng)
        ds = self.density_at(w, sl, so)
        dt = float(self.density_at(w, lat, lon)[0])
        return float(np.mean(ds > dt))

    def log_density_at(self, features, lat: float, lon: float) -> float:
        """log posterior density (per m2) at a point — the proper score."""
        v = float(self.density_at(self.weights(features), lat, lon)[0])
        return math.log(max(v, 1e-300))

    def locate(self, features, cell_m: float | None = None,
               max_side: int = 300) -> _post.GridPosterior:
        w = self.weights(features)
        act = self._active(w)
        lat, lon = self.db.lat[act], self.db.lon[act]
        reach = 4.0 * max(self.bandwidth_m, self.far_bandwidth_m
                          if self.p_far > 0 else 0.0)
        span_lat = (lat.max() - lat.min()) * math.pi / 180 * EARTH_RADIUS_M
        span_lon = (lon.max() - lon.min()) * math.pi / 180 * EARTH_RADIUS_M * \
            math.cos(math.radians(float(lat.mean())))
        span = max(span_lat, span_lon) + 2 * reach
        cell = max(cell_m or self.bandwidth_m / 3.0, span / max_side, 2.0)
        grid = GeoGrid.covering(lat, lon, reach, cell)
        ww = w[act] / w[act].sum()
        dens = (1.0 - self.p_far) * kde_on_grid(grid, lat, lon, ww, self.bandwidth_m)
        if self.p_far > 0:
            dens = dens + self.p_far * kde_on_grid(grid, lat, lon, ww,
                                                   self.far_bandwidth_m)
        with np.errstate(divide="ignore"):
            logp = np.log(dens)
        return _post.GridPosterior(grid, logp, method="fingerprint_kernel",
                                   meta=self.params())

    # -- calibration on a held-out route ----------------------------------------
    def calibrate(self, records, temperatures=None, bandwidths=None,
                  far_bandwidths=None, far_probs=None) -> dict:
        """Choose temperature, h, H and p_far that maximise the mean log
        density at the TRUE positions of a held-out route (a proper scoring
        rule: the honest forecast scores best)."""
        recs = list(records)
        if len(recs) < 5:
            raise ValueError("calibration needs a held-out route of at least "
                             "five records")
        temps = np.asarray(temperatures if temperatures is not None
                           else np.geomspace(0.1, 30.0, 16))
        hs = np.asarray(bandwidths if bandwidths is not None
                        else (10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0))
        Hs = np.asarray(far_bandwidths if far_bandwidths is not None
                        else (300.0, 700.0, 1500.0, 3000.0))
        ps = np.asarray(far_probs if far_probs is not None
                        else (0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5))
        LL = np.stack([self.loglik(self.db.vector(r.features)) for r in recs])
        te, tn = enu_m(np.array([r.lat for r in recs]),
                       np.array([r.lon for r in recs]), self.lat0, self.lon0)
        D2 = (te[:, None] - self._e[None, :]) ** 2 + (tn[:, None] - self._n[None, :]) ** 2
        best = None
        for t in temps:
            z = LL / float(t)
            W = np.exp(z - z.max(axis=1, keepdims=True))
            W /= W.sum(axis=1, keepdims=True)
            near = {h: np.sum(W * np.exp(-D2 / (2 * h * h)), axis=1) / (2 * math.pi * h * h)
                    for h in hs}
            far = {H: np.sum(W * np.exp(-D2 / (2 * H * H)), axis=1) / (2 * math.pi * H * H)
                   for H in Hs}
            for h in hs:
                for H in Hs:
                    for p in ps:
                        dens = (1 - p) * near[h] + p * far[H]
                        score = float(np.mean(np.log(np.maximum(dens, 1e-300))))
                        if best is None or score > best[4]:
                            best = (float(t), float(h), float(H), float(p), score)
        self.temperature, self.bandwidth_m, self.far_bandwidth_m, self.p_far = best[:4]
        self.calibration = {"temperature": best[0], "bandwidth_m": best[1],
                            "far_bandwidth_m": best[2], "p_far": best[3],
                            "mean_log_density": best[4], "n": len(recs)}
        return self.calibration

    def evaluate(self, records, levels=LEVELS, use_grid: bool = False,
                 seed: int = 0) -> dict:
        """Errors and the coverage of the stated regions on a held-out route.
        Credible levels by Monte Carlo (default) or on the posterior grid."""
        rng = np.random.default_rng(seed)
        e_map, e_mean, cls_ = [], [], []
        for r in records:
            x = self.db.vector(r.features)
            w = self.weights(x)
            if use_grid:
                post = self.locate(x)
                la, lo = post.map_point()
                cls_.append(post.credible_level(r.lat, r.lon))
            else:
                # the MAP among the best-weighted database points; the
                # credible level of the truth from 1000 posterior samples
                top = np.argsort(-w)[:200]
                i = int(top[np.argmax(self.density_at(w, self.db.lat[top],
                                                      self.db.lon[top]))])
                la, lo = float(self.db.lat[i]), float(self.db.lon[i])
                sl, so = self.sample(w, 1000, rng)
                ds = self.density_at(w, sl, so)
                cls_.append(float(np.mean(ds > self.density_at(w, r.lat, r.lon)[0])))
            e_map.append(float(haversine_m(la, lo, r.lat, r.lon)))
            act = self._active(w)
            ma = float(np.sum(w[act] * self.db.lat[act]) / w[act].sum())
            mo = float(np.sum(w[act] * self.db.lon[act]) / w[act].sum())
            e_mean.append(float(haversine_m(ma, mo, r.lat, r.lon)))
        return summarize(np.array(e_map), np.array(e_mean), np.array(cls_), levels)


class MapLocator:
    """The radio map run backwards (plan E3: "the same machinery as E2 run
    backwards"). Every feature is kriged from the drive into a continuous
    map WITH its kriging standard deviation, on a grid over the driven area;
    a fingerprint's likelihood at each cell is then, per feature,

        heard at x:      Student-t density of x around the map value, scale
                         sqrt(kriging variance + extra^2)
        not heard:       the probability the map value is below the floor

    Far from every drive the kriging variance grows and the likelihood
    flattens — the posterior says "I don't know this place" by itself,
    which the database-point kernel cannot. Each feature gets its own
    nested variogram (short-range shadowing on a long-range trend, fitted
    on log-spaced lags) and its own kriging solve. `calibrate()` fits the
    temperature and the extra spread on a held-out route by the log
    score."""

    def __init__(self, db: FingerprintDB, cell_m: float = 100.0,
                 margin_m: float = 800.0, nu: float = 4.0,
                 temperature: float = 1.0, extra_db: float = 0.0,
                 thin_m: float = 60.0, max_points: int = 250,
                 max_cells: int = 3000, model: str = "nested"):
        from atk_diffusion.geo import radiomap as _rm
        self.db = db
        self.nu = float(nu)
        self.temperature = float(temperature)
        self.extra_db = float(extra_db)
        self.calibration: dict = {}
        k = math.pi / 180.0 * EARTH_RADIUS_M
        span = max(np.ptp(db.lat) * k, np.ptp(db.lon) * k *
                   math.cos(math.radians(float(np.mean(db.lat))))) + 2 * margin_m
        cell = max(float(cell_m), span / math.sqrt(max_cells))
        self.grid = GeoGrid.covering(db.lat, db.lon, margin_m, cell)
        lat0, lon0 = self.grid.center
        # thin the drive into cells (positions averaged; features averaged
        # over the samples that heard them)
        e, n = enu_m(db.lat, db.lon, lat0, lon0)
        tm = float(thin_m)
        while True:
            key = np.round(e / tm).astype(np.int64) * 1_000_003 + \
                np.round(n / tm).astype(np.int64)
            uniq, inv = np.unique(key, return_inverse=True)
            if uniq.size <= max_points:
                break
            tm *= 1.25
        cnt = np.bincount(inv)
        pe = np.bincount(inv, e) / cnt
        pn = np.bincount(inv, n) / cnt
        F = len(db.keys)
        V = np.empty((uniq.size, F))
        for j in range(F):
            col = db.X[:, j]
            ok = np.isfinite(col)
            s = np.bincount(inv[ok], col[ok], minlength=uniq.size)
            c = np.bincount(inv[ok], minlength=uniq.size)
            V[:, j] = np.where(c > 0, s / np.maximum(c, 1), db.floors[j])
        self.thin_m = tm
        self.n_points = int(uniq.size)
        # one variogram and one kriging solve per feature: the share of a
        # feature's spread that is short-range shadowing differs by feature
        # (a near tower is mostly trend, a far broadcaster mostly shadowing)
        from scipy.linalg import lu_factor, lu_solve
        npt = uniq.size
        i, jj = np.triu_indices(npt, k=1)
        hpair = np.hypot(pe[i] - pe[jj], pn[i] - pn[jj])
        hmat = np.hypot(pe[:, None] - pe[None, :], pn[:, None] - pn[None, :])
        LAT, LON = self.grid.mesh()
        ge, gn = enu_m(LAT.ravel(), LON.ravel(), lat0, lon0)
        mu = np.empty((F, ge.size))
        sdm = np.empty((F, ge.size))
        self.variograms = []
        for f in range(F):
            v = V[:, f]
            lags, gam, cts = _rm.bin_pairs(hpair, 0.5 * (v[i] - v[jj]) ** 2,
                                           0.5 * float(hpair.max()), 16, log=True)
            vg = _rm.fit_variogram_curve(lags, gam, cts, float(np.var(v)) or 1.0,
                                         model)
            self.variograms.append(vg)
            A = np.ones((npt + 1, npt + 1))
            A[:npt, :npt] = vg(hmat)
            A[npt, npt] = 0.0
            lu = lu_factor(A)
            for s0 in range(0, ge.size, 4000):
                sl = slice(s0, s0 + 4000)
                b = np.ones((npt + 1, ge[sl].size))
                b[:npt] = vg(np.hypot(pe[:, None] - ge[None, sl],
                                      pn[:, None] - gn[None, sl]))
                sol = lu_solve(lu, b)
                lam = sol[:npt]
                mu[f, sl] = v @ lam
                sdm[f, sl] = np.sqrt(np.maximum(np.sum(lam * b[:npt], axis=0)
                                                + sol[npt], 0.0))
        self.mu = mu                                      # (F, cells), dB
        self.sd_map = sdm                                 # kriging sigma, dB
        self.variogram = self.variograms[0]
        from scipy.special import gammaln
        self._tconst = (gammaln((self.nu + 1) / 2) - gammaln(self.nu / 2)
                        - 0.5 * math.log(self.nu * math.pi))

    def params(self) -> dict:
        return {"method": "radio map inversion (kriging)", "nu": self.nu,
                "temperature": self.temperature, "extra_db": self.extra_db,
                "cell_m": self.grid.cell_size_m()[1], "thin_m": self.thin_m,
                "kriging_points": self.n_points, "features": len(self.db.keys),
                "variograms": {k: v.to_json() for k, v in
                               zip(self.db.keys, self.variograms)}}

    def loglik_cells(self, x: np.ndarray, extra_db: float | None = None) -> np.ndarray:
        from scipy.special import log_ndtr
        ex = self.extra_db if extra_db is None else float(extra_db)
        t = np.sqrt(self.sd_map ** 2 + ex ** 2 + 1e-6)
        heard = np.isfinite(x)
        ll = np.zeros(self.mu.shape[1])
        if heard.any():
            z = (x[heard, None] - self.mu[heard]) / t[heard]
            ll += np.sum(self._tconst - np.log(t[heard])
                         - 0.5 * (self.nu + 1) * np.log1p(z ** 2 / self.nu), axis=0)
        if (~heard).any():
            z = (self.db.floors[~heard, None] - self.mu[~heard]) / t[~heard]
            ll += np.sum(log_ndtr(z), axis=0)
        return ll

    def locate(self, features) -> _post.GridPosterior:
        x = features if isinstance(features, np.ndarray) else self.db.vector(features)
        ll = self.loglik_cells(x) / self.temperature
        return _post.GridPosterior(self.grid, ll.reshape(self.grid.shape),
                                   method="fingerprint_kernel",
                                   meta=self.params())

    def _truth_cells(self, recs):
        r, c = self.grid.rowcol(np.array([x.lat for x in recs]),
                                np.array([x.lon for x in recs]))
        r = np.clip(np.round(r).astype(int), 0, self.grid.height - 1)
        c = np.clip(np.round(c).astype(int), 0, self.grid.width - 1)
        return r * self.grid.width + c

    def calibrate(self, records, temperatures=None, extras=None) -> dict:
        recs = list(records)
        if len(recs) < 5:
            raise ValueError("calibration needs a held-out route of at least "
                             "five records")
        temps = np.asarray(temperatures if temperatures is not None
                           else np.geomspace(0.5, 20.0, 14))
        exs = np.asarray(extras if extras is not None else (0.0, 2.0, 4.0, 8.0))
        area = (self.grid.cell_area_m2() * np.ones(self.grid.shape)).ravel()
        tc = self._truth_cells(recs)
        vecs = [self.db.vector(r.features) for r in recs]
        best = None
        for ex in exs:
            LL = np.stack([self.loglik_cells(v, ex) for v in vecs])
            for t in temps:
                z = LL / float(t) + np.log(area)[None, :]
                zmax = z.max(axis=1, keepdims=True)
                lse = zmax[:, 0] + np.log(np.exp(z - zmax).sum(axis=1))
                logp = z[np.arange(len(recs)), tc] - lse - np.log(area[tc])
                score = float(np.mean(logp))
                if best is None or score > best[2]:
                    best = (float(t), float(ex), score)
        self.temperature, self.extra_db = best[0], best[1]
        self.calibration = {"temperature": best[0], "extra_db": best[1],
                            "mean_log_density": best[2], "n": len(recs)}
        return self.calibration

    def evaluate(self, records, levels=LEVELS) -> dict:
        e_map, e_mean, cls_ = [], [], []
        for r in records:
            post = self.locate(r.features)
            la, lo = post.map_point()
            ma, mo = post.mean_point()
            e_map.append(float(haversine_m(la, lo, r.lat, r.lon)))
            e_mean.append(float(haversine_m(ma, mo, r.lat, r.lon)))
            cls_.append(post.credible_level(r.lat, r.lon))
        return summarize(np.array(e_map), np.array(e_mean), np.array(cls_), levels)


def summarize(e_map, e_mean, cls_, levels=LEVELS) -> dict:
    """Error distribution and coverage of stated regions, in numbers and
    words — shared by the classical and the learned locators."""
    coverage = {f"{lv:g}": float(np.mean(cls_ <= lv)) for lv in levels}
    hist, _ = np.histogram(cls_, bins=10, range=(0.0, 1.0))
    return {"n": int(e_map.size), "median_error_m": float(np.median(e_map)),
            "p90_error_m": float(np.percentile(e_map, 90)),
            "median_error_mean_m": float(np.median(e_mean)),
            "coverage": coverage,
            "credible_level_histogram": hist.tolist(),
            "words": (f"median error {np.median(e_map):.0f} m (90 % of fixes "
                      f"within {np.percentile(e_map, 90):.0f} m); the stated "
                      f"90 % region held the truth "
                      f"{100 * coverage.get('0.9', float('nan')):.0f} % of the "
                      f"time, the 50 % region "
                      f"{100 * coverage.get('0.5', float('nan')):.0f} %, over "
                      f"{e_map.size} held-out points")}


# ---------------------------------------------------------------------------
# A synthetic drive world: the experiment's stand-in for Bill's drive data
# ---------------------------------------------------------------------------
class _Field:
    """A smooth Gaussian random field (random Fourier features): spatially
    correlated shadowing with correlation length `length_m`."""

    def __init__(self, rng, sigma_db: float, length_m: float, m: int = 64):
        self.w = rng.normal(0.0, 1.0 / length_m, (m, 2))
        self.b = rng.uniform(0, 2 * math.pi, m)
        self.a = sigma_db * math.sqrt(2.0 / m)

    def __call__(self, e, n) -> np.ndarray:
        ph = np.multiply.outer(np.asarray(e), self.w[:, 0]) + \
            np.multiply.outer(np.asarray(n), self.w[:, 1]) + self.b
        return self.a * np.cos(ph).sum(axis=-1)


class DriveWorld:
    """Cell towers and FM broadcasters around a town, with correlated
    shadowing, fast fading and receiver floors; roads on a grid; drives as
    random walks on the roads. Truth is known everywhere."""

    def __init__(self, lat0: float = 38.70, lon0: float = -77.50,
                 size_m: float = 6000.0, n_cells: int = 9, n_fm: int = 4,
                 road_spacing_m: float = 500.0, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.lat0, self.lon0, self.size = lat0, lon0, float(size_m)
        self.spacing = float(road_spacing_m)
        self.rng = rng
        self.tx = []
        for i in range(n_cells):
            e, n = rng.uniform(-1.2, 1.2, 2) * size_m / 2
            self.tx.append({"key": f"lte:310-260-{1200 + i}", "e": e, "n": n,
                            "p": rng.uniform(12, 20), "kind": "lte",
                            "shadow": _Field(rng, 7.0, 150.0), "floor": -124.0})
        for i in range(n_fm):
            ang = rng.uniform(0, 2 * math.pi)
            d = rng.uniform(8e3, 30e3)
            self.tx.append({"key": f"fm:{88.1 + 2.2 * i:.1f}", "e": d * math.cos(ang),
                            "n": d * math.sin(ang), "p": rng.uniform(40, 55),
                            "kind": "fm", "shadow": _Field(rng, 5.0, 400.0),
                            "floor": -100.0})

    def features(self, lat, lon, rng=None, fading_db: float = 2.0) -> list[dict]:
        rng = rng if rng is not None else self.rng
        e, n = enu_m(np.atleast_1d(lat), np.atleast_1d(lon), self.lat0, self.lon0)
        out = [dict() for _ in range(e.size)]
        for t in self.tx:
            d = np.maximum(np.hypot(e - t["e"], n - t["n"]), 10.0)
            if t["kind"] == "lte":
                rx = t["p"] - (128.1 + 37.6 * np.log10(d / 1000.0))
            else:
                rx = t["p"] - (20 * np.log10(d) + 20 * np.log10(98e6) - 147.55) \
                    - 25.0 * np.log10(np.maximum(d / 5000.0, 1.0))
            rx = rx + t["shadow"](e, n) + rng.normal(0, fading_db, e.size)
            for i in np.nonzero(rx >= t["floor"])[0]:
                out[i][t["key"]] = float(round(rx[i], 1))
        return out

    def drive(self, n_points: int, step_m: float = 30.0, seed: int | None = None,
              t0: float = 0.0) -> list[DriveRecord]:
        """A random walk on the road grid, a sample every `step_m`."""
        rng = np.random.default_rng(seed)
        half = int(self.size / 2 // self.spacing)
        lim = half * self.spacing
        n_sub = max(1, int(round(self.spacing / step_m)))
        step = self.spacing / n_sub
        x, y = (float(v) for v in rng.integers(-half, half + 1, 2) * self.spacing)
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        def options(back):
            ok = [d for d in dirs if abs(x + d[0] * self.spacing) <= lim + 1e-6
                  and abs(y + d[1] * self.spacing) <= lim + 1e-6]
            fwd = [d for d in ok if d != back]
            return fwd or ok
        dx, dy = options(None)[rng.integers(len(options(None)))]
        pts = []
        k = 0
        while len(pts) < n_points:
            pts.append((x, y))
            x, y = x + dx * step, y + dy * step
            k += 1
            if k == n_sub:                      # an intersection: turn or not
                k = 0
                ch = options((-dx, -dy))
                dx, dy = ch[rng.integers(len(ch))]
        e = np.array([p[0] for p in pts])
        n = np.array([p[1] for p in pts])
        lat, lon = from_enu(e, n, self.lat0, self.lon0)
        feats = self.features(lat, lon, rng)
        return [DriveRecord(t0 + i, float(lat[i]), float(lon[i]), feats[i])
                for i in range(len(pts))]
