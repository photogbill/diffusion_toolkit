# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Radio maps from sparse measurements — the classical half of E2 (plan
§4.E2, E5's measured layer).

*"Where can we reach?"* asked of the measurements themselves: field-
strength samples from ATK's own receivers (GeoJSON points, one value each)
become a gridded map by two classical interpolators, and — when a physics
prediction exists (`reach.predicted_reach`) — the RESIDUAL between them,
which is the only thing the learned model (`learn.radiomap`) is allowed to
learn.

* **IDW** — inverse-distance weighting over the k nearest samples. No
  model, no uncertainty; the baseline everything else must beat.
* **Ordinary kriging** — the best linear unbiased predictor under a fitted
  variogram: the empirical semivariogram binned by lag, a spherical /
  exponential / Gaussian model fitted by Cressie's weighted least squares,
  then one linear system solved for every grid cell. Kriging gives a
  STANDARD DEVIATION beside every estimate — the honest "how far from a
  sample is this" — and that map is written too.
* **Cross-validation** — k-fold RMSE / bias / MAE for each method on the
  samples themselves, written into the product's manifest: the classical
  numbers a learned radio map must beat (plan §7) to be shipped.

Values are in dB (dBm, dBuV/m, dB of residual): interpolating in dB is the
usual practice and keeps log-normal shadowing Gaussian. Distances are in a
local tangent plane at the grid centre (fine for the tens of km a drive
covers). Samples closer than a metre are merged (averaged) — kriging's
system is singular with two samples on one spot.

TIERS. Sample points are MEASURED. Interpolated maps and residual maps are
INFERRED (a statistical model fills between samples).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from atk_diffusion import provenance as _prov
from atk_diffusion.geo import products as _products
from atk_diffusion.geo.products import GeoGrid
from atk_diffusion.geo.terrain import enu_m

_prov.METHOD_TIERS.setdefault("field_strength", "measured")


# ---------------------------------------------------------------------------
@dataclass
class Measurements:
    """Sparse samples: position and one value in dB each."""
    lat: np.ndarray
    lon: np.ndarray
    value: np.ndarray
    key: str = "rx_dbm"
    time: np.ndarray | None = None
    props: list = field(default_factory=list)

    def __post_init__(self):
        self.lat = np.asarray(self.lat, dtype=np.float64).ravel()
        self.lon = np.asarray(self.lon, dtype=np.float64).ravel()
        self.value = np.asarray(self.value, dtype=np.float64).ravel()
        if not (self.lat.size == self.lon.size == self.value.size):
            raise ValueError("lat, lon and value must be the same length")
        ok = np.isfinite(self.lat) & np.isfinite(self.lon) & np.isfinite(self.value)
        if not ok.all():
            self.lat, self.lon, self.value = self.lat[ok], self.lon[ok], self.value[ok]
            if self.time is not None:
                self.time = np.asarray(self.time)[ok]
            if self.props:
                self.props = [p for p, k in zip(self.props, ok) if k]

    def __len__(self) -> int:
        return int(self.value.size)

    @classmethod
    def from_geojson(cls, src, value_key: str = "rx_dbm") -> "Measurements":
        d = src if isinstance(src, dict) else _products.read_geojson(src)
        lat, lon, val, props, t = [], [], [], [], []
        for f in d.get("features", []):
            g = f.get("geometry") or {}
            p = f.get("properties") or {}
            if g.get("type") != "Point" or value_key not in p:
                continue
            try:
                v = float(p[value_key])
            except (TypeError, ValueError):
                continue
            lon.append(float(g["coordinates"][0]))
            lat.append(float(g["coordinates"][1]))
            val.append(v)
            props.append(p)
            t.append(p.get("time"))
        if not val:
            raise ValueError(f"no Point features carry {value_key!r}")
        return cls(np.array(lat), np.array(lon), np.array(val), value_key,
                   np.array(t, dtype=object), props)

    def subset(self, mask) -> "Measurements":
        m = np.asarray(mask)
        return Measurements(self.lat[m], self.lon[m], self.value[m], self.key,
                            None if self.time is None else np.asarray(self.time)[m],
                            [p for p, k in zip(self.props, np.atleast_1d(m)) if k]
                            if self.props and m.dtype == bool else [])

    def features(self, tier: str = "measured", extra: dict | None = None) -> list:
        out = []
        for i in range(len(self)):
            props = {self.key: float(self.value[i]), "atk:tier": tier}
            if self.time is not None and self.time[i] is not None:
                props["time"] = self.time[i] if isinstance(self.time[i], str) \
                    else float(self.time[i])
            for k, arr in (extra or {}).items():
                v = arr[i]
                props[k] = None if v is None or (isinstance(v, float)
                                                 and not math.isfinite(v)) \
                    else float(v)
            out.append(_products.point_feature(self.lat[i], self.lon[i], props))
        return out

    def merged(self, tol_m: float = 1.0) -> "Measurements":
        """Samples within `tol_m` of each other averaged into one."""
        if len(self) < 2:
            return self
        lat0, lon0 = float(np.mean(self.lat)), float(np.mean(self.lon))
        e, n = enu_m(self.lat, self.lon, lat0, lon0)
        key = np.round(e / tol_m).astype(np.int64) * 1_000_003 + \
            np.round(n / tol_m).astype(np.int64)
        uniq, inv = np.unique(key, return_inverse=True)
        if uniq.size == len(self):
            return self
        cnt = np.bincount(inv)
        return Measurements(np.bincount(inv, self.lat) / cnt,
                            np.bincount(inv, self.lon) / cnt,
                            np.bincount(inv, self.value) / cnt, self.key)


def _xy(meas: Measurements, lat0: float, lon0: float):
    e, n = enu_m(meas.lat, meas.lon, lat0, lon0)
    return np.column_stack([e, n])


def _grid_xy(grid: GeoGrid, lat0: float, lon0: float):
    lat, lon = grid.mesh()
    e, n = enu_m(lat.ravel(), lon.ravel(), lat0, lon0)
    return np.column_stack([e, n])


# ---------------------------------------------------------------------------
# IDW
# ---------------------------------------------------------------------------
def idw(meas: Measurements, grid: GeoGrid, power: float = 2.0, k: int = 12,
        radius_m: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-distance weighting: (estimate (grid shape), distance to the
    nearest sample in metres). Cells farther than `radius_m` from every
    sample are NaN."""
    from scipy.spatial import cKDTree
    if len(meas) == 0:
        raise ValueError("IDW needs at least one sample")
    lat0, lon0 = grid.center
    src = _xy(meas, lat0, lon0)
    dst = _grid_xy(grid, lat0, lon0)
    kk = int(min(max(1, k), len(meas)))
    tree = cKDTree(src)
    d, i = tree.query(dst, k=kk)
    if kk == 1:
        d, i = d[:, None], i[:, None]
    exact = d[:, 0] < 1e-6
    w = 1.0 / np.maximum(d, 1e-6) ** float(power)
    est = np.sum(w * meas.value[i], axis=1) / np.sum(w, axis=1)
    est = np.where(exact, meas.value[i[:, 0]], est)
    if radius_m is not None:
        est = np.where(d[:, 0] <= radius_m, est, np.nan)
    return est.reshape(grid.shape), d[:, 0].reshape(grid.shape)


def idw_points(meas: Measurements, lat, lon, power: float = 2.0,
               k: int = 12) -> np.ndarray:
    from scipy.spatial import cKDTree
    lat0, lon0 = float(np.mean(meas.lat)), float(np.mean(meas.lon))
    src = _xy(meas, lat0, lon0)
    e, n = enu_m(np.asarray(lat), np.asarray(lon), lat0, lon0)
    kk = int(min(max(1, k), len(meas)))
    d, i = cKDTree(src).query(np.column_stack([np.ravel(e), np.ravel(n)]), k=kk)
    if kk == 1:
        d, i = d[:, None], i[:, None]
    w = 1.0 / np.maximum(d, 1e-6) ** float(power)
    return np.sum(w * meas.value[i], axis=1) / np.sum(w, axis=1)


# ---------------------------------------------------------------------------
# Variograms
# ---------------------------------------------------------------------------
VARIOGRAM_MODELS = ("spherical", "exponential", "gaussian", "nested")


def _core(model: str, h, a):
    a = max(float(a), 1e-9)
    if model == "spherical":
        r = np.minimum(h / a, 1.0)
        return 1.5 * r - 0.5 * r ** 3
    if model in ("exponential", "nested"):
        return 1.0 - np.exp(-3.0 * h / a)
    if model == "gaussian":
        return 1.0 - np.exp(-3.0 * (h / a) ** 2)
    raise ValueError(f"unknown variogram model {model!r}")


@dataclass
class Variogram:
    """nugget + psill * structure(h / range). "nested" adds a second
    exponential structure (psill2, range2): short-range shadowing on top of
    a long-range trend, as a drive's radio map usually shows."""
    model: str
    nugget: float
    psill: float               # partial sill (sill = nugget + psill + psill2)
    range_m: float             # practical range
    lags_m: list = field(default_factory=list)
    gamma: list = field(default_factory=list)
    counts: list = field(default_factory=list)
    note: str = ""
    psill2: float = 0.0
    range2_m: float = 0.0

    @property
    def sill(self) -> float:
        return self.nugget + self.psill + self.psill2

    def __call__(self, h) -> np.ndarray:
        h = np.asarray(h, dtype=np.float64)
        v = self.nugget + self.psill * _core(self.model, h, self.range_m)
        if self.psill2 > 0:
            v = v + self.psill2 * _core("exponential", h, self.range2_m)
        return np.where(h > 0, v, 0.0)

    def to_json(self) -> dict:
        return {"model": self.model, "nugget": self.nugget, "psill": self.psill,
                "range_m": self.range_m, "psill2": self.psill2,
                "range2_m": self.range2_m, "lags_m": list(self.lags_m),
                "gamma": list(self.gamma), "counts": list(self.counts),
                "note": self.note}


def lag_edges(max_lag_m: float, n_lags: int = 12, first_m: float | None = None,
              log: bool = False) -> np.ndarray:
    """Lag-bin edges: even, or (log=True) geometric from `first_m`, so the
    short-range structure is not hidden inside one wide first bin."""
    if not log:
        return np.linspace(0.0, max_lag_m, int(n_lags) + 1)
    f = first_m or max_lag_m / 200.0
    return np.concatenate([[0.0], np.geomspace(f, max_lag_m, int(n_lags))])


def empirical_variogram(meas: Measurements, n_lags: int = 12,
                        max_lag_m: float | None = None, log_lags: bool = False):
    """(lag centres m, semivariance, pair counts) from every pair of samples
    (half the maximum distance by default)."""
    lat0, lon0 = float(np.mean(meas.lat)), float(np.mean(meas.lon))
    xy = _xy(meas, lat0, lon0)
    n = len(meas)
    if n < 3:
        raise ValueError("a variogram needs at least three samples")
    if n > 3000:                       # pairs grow as n^2: subsample
        idx = np.random.default_rng(0).choice(n, 3000, replace=False)
        xy, val = xy[idx], meas.value[idx]
    else:
        val = meas.value
    i, j = np.triu_indices(xy.shape[0], k=1)
    h = np.hypot(*(xy[i] - xy[j]).T)
    g = 0.5 * (val[i] - val[j]) ** 2
    return bin_pairs(h, g, max_lag_m or 0.5 * float(h.max()), n_lags, log_lags)


def bin_pairs(h, g, max_lag_m: float, n_lags: int = 12, log: bool = False):
    """Bin pair distances and half squared differences into a variogram."""
    edges = lag_edges(float(max_lag_m), n_lags, log=log)
    which = np.digitize(h, edges) - 1
    lags, gam, cnt = [], [], []
    for b in range(edges.size - 1):
        sel = which == b
        c = int(sel.sum())
        if c >= 3:
            lags.append(float(h[sel].mean()))
            gam.append(float(g[sel].mean()))
            cnt.append(c)
    return np.array(lags), np.array(gam), np.array(cnt)


def fit_variogram(meas: Measurements, model: str = "exponential",
                  n_lags: int = 12, max_lag_m: float | None = None) -> Variogram:
    """Fit a variogram model by Cressie's weighted least squares (weights
    N_h / gamma_model(h)^2). A fit that fails becomes a pure nugget — no
    spatial correlation, i.e. kriging degrades to the mean — and says so."""
    if model not in VARIOGRAM_MODELS:
        raise ValueError(f"variogram model is one of {', '.join(VARIOGRAM_MODELS)}")
    lags, gam, cnt = empirical_variogram(meas, n_lags, max_lag_m)
    var = float(np.var(meas.value)) or 1e-6
    return fit_variogram_curve(lags, gam, cnt, var, model)


def fit_variogram_curve(lags, gam, cnt, var: float,
                        model: str = "exponential") -> Variogram:
    """Fit a model to an empirical variogram (lags, semivariances, pair
    counts) by Cressie's weighted least squares; `var` bounds the sill."""
    from scipy.optimize import least_squares
    lags, gam, cnt = (np.asarray(v, dtype=np.float64) for v in (lags, gam, cnt))
    var = float(var) or 1e-6
    nested = model == "nested"
    if lags.size < (5 if nested else 3):
        return Variogram(model, var, 0.0, 1.0, lags.tolist(), gam.tolist(),
                         cnt.tolist(), "too few lag bins for a fit: pure nugget "
                                       "(no spatial correlation assumed)")
    hmax = float(lags.max())
    hmin = max(float(lags.min()), 1.0)

    def make(p):
        if nested:
            return Variogram(model, p[0], p[1], p[2], psill2=p[3], range2_m=p[4])
        return Variogram(model, p[0], p[1], p[2])

    def resid(p):
        m = np.maximum(make(p)(lags), 1e-9)
        return np.sqrt(cnt) * (gam - m) / m

    g0 = min(gam[0], var)
    if nested:
        x0 = [0.3 * g0, 0.7 * g0 + 1e-6, 4 * hmin, max(var - g0, 1e-6), 0.5 * hmax]
        lo = [0.0, 0.0, hmin * 0.25, 0.0, 2 * hmin]
        hi = [2 * var + 1e-6, 3 * var + 1e-6, 0.5 * hmax, 3 * var + 1e-6, 4 * hmax]
    else:
        x0 = [0.5 * g0, max(var - 0.5 * g0, 1e-6), 0.5 * hmax]
        lo = [0.0, 0.0, max(float(lags.min()) * 0.25, 1.0)]
        hi = [2.0 * var + 1e-6, 3.0 * var + 1e-6, 4.0 * hmax]
    x0 = [min(max(v, l + 1e-9), h - 1e-9) for v, l, h in zip(x0, lo, hi)]
    try:
        r = least_squares(resid, x0, bounds=(lo, hi))
        vg = make([float(v) for v in r.x])
        if vg.psill + vg.psill2 < 1e-6 * var:
            vg.note = "the fit found no spatial structure: kriging is the mean"
    except Exception as e:                                 # noqa: BLE001
        vg = Variogram(model, var, 0.0, 1.0)
        vg.note = f"variogram fit failed ({e}); pure nugget used"
    vg.lags_m, vg.gamma, vg.counts = lags.tolist(), gam.tolist(), cnt.tolist()
    return vg


# ---------------------------------------------------------------------------
# Ordinary kriging
# ---------------------------------------------------------------------------
@dataclass
class Kriged:
    estimate: np.ndarray
    sigma: np.ndarray            # kriging standard deviation (same units)
    variogram: Variogram
    n_samples: int
    notes: list = field(default_factory=list)


def _thin(meas: Measurements, max_points: int, notes: list) -> Measurements:
    if len(meas) <= max_points:
        return meas
    lat0, lon0 = float(np.mean(meas.lat)), float(np.mean(meas.lon))
    e, n = enu_m(meas.lat, meas.lon, lat0, lon0)
    span = max(np.ptp(e), np.ptp(n), 1.0)
    cell = span / math.sqrt(max_points) * 1.2
    while True:
        thinned = meas.merged(cell)
        if len(thinned) <= max_points:
            notes.append(f"{len(meas)} samples averaged into {len(thinned)} "
                         f"cells of {cell:.0f} m for kriging")
            return thinned
        cell *= 1.3


def _ok_system(xy: np.ndarray, vg: Variogram):
    from scipy.linalg import lu_factor
    n = xy.shape[0]
    h = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    A = np.ones((n + 1, n + 1))
    A[:n, :n] = vg(h)
    A[n, n] = 0.0
    return lu_factor(A)


def kriging_points(meas: Measurements, xy_q: np.ndarray, vg: Variogram,
                   lat0: float, lon0: float, chunk: int = 20000):
    """Ordinary kriging at query points (local metres): (estimate, sigma)."""
    from scipy.linalg import lu_solve
    xy = _xy(meas, lat0, lon0)
    lu = _ok_system(xy, vg)
    n = xy.shape[0]
    est = np.empty(xy_q.shape[0])
    var = np.empty(xy_q.shape[0])
    for s in range(0, xy_q.shape[0], chunk):
        q = xy_q[s:s + chunk]
        h = np.hypot(xy[:, None, 0] - q[None, :, 0], xy[:, None, 1] - q[None, :, 1])
        b = np.ones((n + 1, q.shape[0]))
        b[:n] = vg(h)
        sol = lu_solve(lu, b)
        lam, mu = sol[:n], sol[n]
        est[s:s + chunk] = lam.T @ meas.value
        var[s:s + chunk] = np.sum(lam * b[:n], axis=0) + mu
    return est, np.sqrt(np.maximum(var, 0.0))


def ordinary_kriging(meas: Measurements, grid: GeoGrid,
                     variogram: Variogram | None = None,
                     model: str = "exponential",
                     max_points: int = 1500) -> Kriged:
    notes: list = []
    m = _thin(meas.merged(1.0), max_points, notes)
    if len(m) < 3:
        raise ValueError("kriging needs at least three distinct samples")
    vg = variogram or fit_variogram(m, model)
    if vg.note:
        notes.append(vg.note)
    lat0, lon0 = grid.center
    est, sig = kriging_points(m, _grid_xy(grid, lat0, lon0), vg, lat0, lon0)
    return Kriged(est.reshape(grid.shape), sig.reshape(grid.shape), vg, len(m),
                  notes)


# ---------------------------------------------------------------------------
# Cross-validation, residuals
# ---------------------------------------------------------------------------
def cross_validate(meas: Measurements, method: str = "kriging", folds: int = 10,
                   seed: int = 0, model: str = "exponential",
                   variogram: Variogram | None = None) -> dict:
    """k-fold prediction error at held-out samples: rmse, mae, bias (dB)."""
    m = meas.merged(1.0)
    n = len(m)
    if n < 6:
        raise ValueError("cross-validation needs at least six samples")
    k = int(min(max(2, folds), n))
    order = np.random.default_rng(seed).permutation(n)
    pred = np.full(n, np.nan)
    lat0, lon0 = float(np.mean(m.lat)), float(np.mean(m.lon))
    vg = variogram
    if method == "kriging" and vg is None:
        vg = fit_variogram(m, model)
    for f in range(k):
        test = order[f::k]
        train = np.setdiff1d(order, test)
        tr = m.subset(np.isin(np.arange(n), train))
        te_lat, te_lon = m.lat[test], m.lon[test]
        if method == "idw":
            pred[test] = idw_points(tr, te_lat, te_lon)
        elif method == "kriging":
            e, nn = enu_m(te_lat, te_lon, lat0, lon0)
            pred[test], _ = kriging_points(tr, np.column_stack([e, nn]), vg,
                                           lat0, lon0)
        else:
            raise ValueError("method is 'idw' or 'kriging'")
    err = pred - m.value
    return {"method": method, "folds": k, "n": n,
            "rmse_db": float(np.sqrt(np.mean(err ** 2))),
            "mae_db": float(np.mean(np.abs(err))),
            "bias_db": float(np.mean(err))}


def physics_at(meas: Measurements, physics) -> np.ndarray:
    """The physics prediction at each sample. `physics` is a GeoRaster or
    (GeoGrid, array)."""
    if hasattr(physics, "sample"):
        return np.asarray(physics.sample(meas.lat, meas.lon))
    grid, arr = physics
    return grid.sample(arr, meas.lat, meas.lon)


def residuals(meas: Measurements, physics) -> Measurements:
    """measured - physics at each sample, as a new Measurements ('residual_db').
    Samples outside the physics raster are dropped."""
    p = physics_at(meas, physics)
    r = meas.value - p
    keep = np.isfinite(r)
    return Measurements(meas.lat[keep], meas.lon[keep], r[keep], "residual_db")


# ---------------------------------------------------------------------------
# The product
# ---------------------------------------------------------------------------
def radiomap_product(rf, meas: Measurements, grid: GeoGrid, *, physics=None,
                     method: str = "kriging", run: str | None = None,
                     label: str = "", model: str = "exponential",
                     params: dict | None = None) -> dict:
    """Write `products\\radiomaps\\<run>\\`: the samples (MEASURED), the
    interpolated map and — for kriging — its standard deviation (INFERRED),
    and with a physics layer the residual points and residual map. The
    cross-validation numbers go in the manifest. Returns a summary dict."""
    if method not in ("kriging", "idw"):
        raise ValueError("method is 'kriging' or 'idw'")
    pr = _products.ProductRun(rf, "radiomaps", run, label=label or method,
                              tier="inferred", method=method,
                              params={"value_key": meas.key, "method": method,
                                      "variogram_model": model,
                                      "grid": grid.to_json(), **(params or {})})
    out = {"run_dir": str(pr.dir), "n_samples": len(meas)}
    pr.add_geojson("samples.geojson", meas.features("measured"),
                   tier="measured", layer={"name": "samples", "role":
                                           "measurement", "units": "dB"})

    def interp(m: Measurements, stem: str, role: str):
        if method == "kriging":
            k = ordinary_kriging(m, grid, model=model)
            pr.add_geotiff(f"{stem}_kriging.tif", k.estimate.astype(np.float32),
                           grid, tier="inferred", nodata=-9999.0,
                           tags={"atk:units": "dB", "atk:variogram": k.variogram.to_json()},
                           layer={"name": f"{stem} (kriging)", "role": role})
            pr.add_geotiff(f"{stem}_kriging_sigma.tif",
                           k.sigma.astype(np.float32), grid, tier="inferred",
                           nodata=-9999.0, tags={"atk:units": "dB"},
                           layer={"name": f"{stem} uncertainty (1 sigma)",
                                  "role": "uncertainty"})
            return k.estimate, k.variogram, k.notes
        est, near = idw(m, grid)
        pr.add_geotiff(f"{stem}_idw.tif", est.astype(np.float32), grid,
                       tier="inferred", nodata=-9999.0, tags={"atk:units": "dB"},
                       layer={"name": f"{stem} (IDW)", "role": role})
        return est, None, []
    est, vg, notes = interp(meas, "measured", "measurement")
    cv = {"idw": cross_validate(meas, "idw")}
    try:
        cv["kriging"] = cross_validate(meas, "kriging", model=model)
    except (ValueError, np.linalg.LinAlgError) as e:
        notes.append(f"kriging cross-validation not run: {e}")
    out["cross_validation"] = cv
    out["notes"] = notes
    if vg is not None:
        out["variogram"] = vg.to_json()
    if physics is not None:
        res = residuals(meas, physics)
        pred = physics_at(meas, physics)
        pr.add_geojson("residual_points.geojson",
                       meas.features("inferred", {"physics_dbm": pred,
                                                  "residual_db": meas.value - pred}),
                       tier="inferred", layer={"name": "residual at samples",
                                               "role": "residual"})
        if len(res) >= 3:
            interp(res, "residual", "residual")
        out["residual"] = {"n": len(res), "mean_db": float(np.mean(res.value)),
                           "std_db": float(np.std(res.value)),
                           "rmse_db": float(np.sqrt(np.mean(res.value ** 2)))}
    pr.finish(summary=out)
    return out
