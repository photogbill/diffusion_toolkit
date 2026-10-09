# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Predicted reach on the map — terrain propagation with DTED level 1
(plan E5; the AURA planning question *"where will this radio reach from
here?"*).

Bill, 2026-10-08: *"if I have the DTED level 1 loaded, we can use space loss
and other propagation models to show on the map the likely reach given the
radio, power levels, antenna patterns."*

    predicted_reach(tx, rx, terrain, radius_km, model="itm", grid_m=250)

For every cell of a grid around the transmitter: the great-circle terrain
profile (`terrain.batch_profiles`, from a `dted.DtedMosaic` or any terrain),
the path loss by the chosen model (`propagation`), the transmitter
antenna's gain toward the cell (bearing and take-off angle in the antenna's
frame, `antenna`), the receiver's antenna, line losses, and the received
power. The reach mask is received power at or above the receiver's
sensitivity plus a fade margin.

THREE LAYERS, KEPT APART (the plan's words: "the map shows physics,
measurement and learned correction as three layers an analyst can toggle").
`predicted_reach` writes the PHYSICS layer (INFERRED tier) to
`products\\coverage\\<run>\\`. `add_measurement_layer` adds what a drive
actually measured (MEASURED points; an INFERRED kriged surface between
them) and the residual against the physics. `add_correction_layer` adds the
learned correction from `learn.radiomap` (INVENTED tier) and the corrected
map, with its uncertainty. Each is its own GeoTIFF/GeoJSON listed in the
manifest's `layers` with a role ATK's map can toggle on: physics,
measurement, correction.

HONEST EDGES. ITM is undefined inside 1 km: those cells use free space +
Deygout diffraction and the manifest counts them. ITM's error flags are
kept per cell (`itm_warnings.tif`). The receiver's azimuth is unknown (a
radio in a hand or a car), so only its antenna's ELEVATION pattern is used.
Bare earth: buildings and trees are not in DTED — the measured layer and
the learned residual are where clutter shows up. A reach map is a
prediction for planning, never a promise of contact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from atk_diffusion import provenance as _prov
from atk_diffusion.geo import antenna as _ant
from atk_diffusion.geo import products as _products
from atk_diffusion.geo import propagation as _prop
from atk_diffusion.geo import terrain as _terrain
from atk_diffusion.geo.products import EARTH_RADIUS_M, GeoGrid

NODATA = -9999.0
MASK_NODATA = 255


@dataclass
class Transmitter:
    lat: float
    lon: float
    freq_hz: float
    power_dbm: float = 37.0                 # 5 W
    height_agl_m: float = 2.0
    antenna: _ant.AntennaPattern = field(default_factory=lambda: _ant.Dipole("vertical"))
    azimuth_deg: float = 0.0                # boresight, true
    tilt_deg: float = 0.0                   # beam elevation (negative = down)
    line_loss_db: float = 0.0
    polarization: str = "vertical"
    name: str = ""

    @classmethod
    def watts(cls, lat, lon, freq_hz, power_w: float, **kw) -> "Transmitter":
        if power_w <= 0:
            raise ValueError("transmit power must be positive")
        return cls(lat, lon, freq_hz, 10.0 * math.log10(power_w * 1000.0), **kw)

    def to_json(self) -> dict:
        return {"lat": self.lat, "lon": self.lon, "freq_hz": self.freq_hz,
                "power_dbm": self.power_dbm,
                "power_w": 10 ** (self.power_dbm / 10) / 1000.0,
                "height_agl_m": self.height_agl_m,
                "antenna": self.antenna.to_json(),
                "antenna_words": self.antenna.describe(),
                "azimuth_deg": self.azimuth_deg, "tilt_deg": self.tilt_deg,
                "line_loss_db": self.line_loss_db,
                "polarization": self.polarization, "name": self.name}


@dataclass
class Receiver:
    height_agl_m: float = 1.5
    antenna: _ant.AntennaPattern = field(default_factory=lambda: _ant.Dipole("vertical", 0.0))
    sensitivity_dbm: float = -110.0
    fade_margin_db: float = 0.0
    line_loss_db: float = 0.0
    name: str = ""

    def to_json(self) -> dict:
        return {"height_agl_m": self.height_agl_m,
                "antenna": self.antenna.to_json(),
                "antenna_words": self.antenna.describe(),
                "sensitivity_dbm": self.sensitivity_dbm,
                "fade_margin_db": self.fade_margin_db,
                "line_loss_db": self.line_loss_db, "name": self.name}


@dataclass
class ReachResult:
    grid: GeoGrid
    power_dbm: np.ndarray            # NaN outside the radius
    loss_db: np.ndarray
    mask: np.ndarray                 # uint8: 1 reach, 0 not, 255 outside
    model: str
    method: str
    tier: str
    summary: dict
    notes: list
    run_dir: str = ""
    kwx: np.ndarray | None = None

    def words(self) -> str:
        return self.summary.get("words", "")


def _antenna_angles(d, dz, k):
    """Take-off elevation (deg) toward a point d metres away and dz metres
    higher, on an earth of effective radius kR."""
    return np.degrees(np.arctan2(dz - d ** 2 / (2.0 * k * EARTH_RADIUS_M),
                                 np.maximum(d, 1.0)))


def predicted_reach(tx: Transmitter, rx: Receiver, terrain, radius_km: float,
                    model: str = "itm", grid_m: float = 250.0, *, rf=None,
                    run: str | None = None, label: str = "",
                    reliability: float = 50.0, confidence: float = 50.0,
                    k: float = _terrain.K_STANDARD, profile_step_m: float | None = None,
                    max_profile_points: int = 256, ground: str = "average",
                    itm_kw: dict | None = None, progress=None,
                    write: bool = True) -> ReachResult:
    """The physics layer. `terrain` has `.elevation(lat, lon)` (a DTED
    mosaic, a GridTerrain, FlatTerrain). With `rf`, writes
    products\\coverage\\<run>\\ and returns its folder in `run_dir`."""
    if model not in _prop.MODELS:
        raise ValueError(f"unknown model {model!r} — one of "
                         f"{', '.join(_prop.MODELS)}")
    if model == "itm":
        ok, why = _prop.itm_available()
        if not ok:
            raise _prop.ItmUnavailable(why)
    method = _prop.MODELS[model]
    tier = _prov.tier_for(method)
    say = progress or (lambda s: None)
    radius_m = float(radius_km) * 1000.0
    grid = GeoGrid.around(tx.lat, tx.lon, radius_m, float(grid_m))
    LAT, LON = grid.mesh()
    d_all = _terrain.haversine_m(tx.lat, tx.lon, LAT, LON)
    az_all = _terrain.initial_bearing_deg(tx.lat, tx.lon, LAT, LON)
    inside = d_all <= radius_m
    z_tx = float(np.nan_to_num(np.asarray(terrain.elevation(tx.lat, tx.lon)).ravel()[0]))
    z_cell = np.asarray(terrain.elevation(LAT, LON), dtype=np.float64)
    void_cells = int(np.count_nonzero(~np.isfinite(z_cell) & inside))
    z_cell = np.where(np.isfinite(z_cell), z_cell, z_tx)
    eps, sig = _prop.GROUNDS.get(ground, _prop.GROUNDS["average"])
    loss = np.full(grid.shape, np.nan)
    kwx = np.zeros(grid.shape, dtype=np.uint8) if model == "itm" else None
    notes: list = []
    idx = np.argwhere(inside)
    n_cells = idx.shape[0]
    say(f"reach: {n_cells:,} cells by {_prop.MODEL_WORDS[model]}")
    if model == "fspl":
        loss[inside] = _prop.fspl_db(d_all[inside], tx.freq_hz)
    elif model == "two_ray":
        loss[inside] = _prop.two_ray_db(d_all[inside], tx.freq_hz,
                                        tx.height_agl_m, rx.height_agl_m, eps,
                                        sig, tx.polarization)
    else:
        step = profile_step_m or max(30.0, min(float(grid_m) / 2.0, 90.0))
        n_pts = int(min(max_profile_points, max(16, math.ceil(radius_m / step) + 1)))
        voids = 0
        short = 0
        chunk = max(64, int(2_000_000 // n_pts))
        done = 0
        for s in range(0, n_cells, chunk):
            rows = idx[s:s + chunk]
            la, lo = LAT[rows[:, 0], rows[:, 1]], LON[rows[:, 0], rows[:, 1]]
            dist, z, nv = _terrain.batch_profiles(terrain, tx.lat, tx.lon, la, lo,
                                                  n_pts)
            voids += nv
            for m in range(rows.shape[0]):
                r, c = rows[m]
                D = float(dist[m])
                if D < 1.0:
                    loss[r, c] = float(_prop.fspl_db(1.0, tx.freq_hz))
                    continue
                prof = _terrain.Profile(np.linspace(0.0, D, n_pts), None, None,
                                        z[m])
                if model == "itm" and D >= 1000.0:
                    res = _prop.itm_p2p(prof, tx.freq_hz, tx.height_agl_m,
                                        rx.height_agl_m,
                                        polarization=tx.polarization,
                                        eps_r=eps, sigma_s_m=sig,
                                        reliability=reliability,
                                        confidence=confidence, **(itm_kw or {}))
                    loss[r, c] = res.loss_db
                    kwx[r, c] = res.kwx
                else:
                    if model == "itm":
                        short += 1
                    fn = _prop.bullington if model == "bullington" else _prop.deygout
                    dif = fn(prof, tx.height_agl_m, rx.height_agl_m, tx.freq_hz, k)
                    loss[r, c] = float(_prop.fspl_db(D, tx.freq_hz)) + dif.loss_db
            done += rows.shape[0]
            say(f"reach: {done:,}/{n_cells:,} cells")
        if voids:
            notes.append(f"{voids:,} profile samples had no terrain height "
                         "(void or outside the loaded DTED) and were "
                         "interpolated along the path")
        if short:
            notes.append(f"{short} cells inside 1 km of the transmitter use free "
                         "space + Deygout diffraction (ITM is undefined below "
                         "1 km)")
        if model == "itm":
            bad = int(np.count_nonzero(kwx >= 3))
            if bad:
                notes.append(f"{bad} cells carry an ITM out-of-range warning "
                             "(itm_warnings.tif); their numbers are probably "
                             "invalid")
    if void_cells:
        notes.append(f"{void_cells} cells have no terrain height of their own")
    # antennas
    el = _antenna_angles(d_all, (z_cell + rx.height_agl_m)
                         - (z_tx + tx.height_agl_m), k)
    g_tx = tx.antenna.gain_dbi(_terrain.wrap180(az_all - tx.azimuth_deg),
                               el - tx.tilt_deg)
    g_rx = rx.antenna.gain_dbi(np.zeros_like(el), -el)
    power = (tx.power_dbm - tx.line_loss_db + g_tx - loss + g_rx
             - rx.line_loss_db)
    power = np.where(inside, power, np.nan)
    thresh = rx.sensitivity_dbm + rx.fade_margin_db
    mask = np.where(inside, (power >= thresh).astype(np.uint8),
                    MASK_NODATA).astype(np.uint8)
    # summary
    area = grid.cell_area_m2() * np.ones(grid.shape)
    reach_km2 = float(area[mask == 1].sum() / 1e6)
    circle_km2 = float(area[inside].sum() / 1e6)
    sectors = {}
    for s0 in range(0, 360, 30):
        sel = (mask == 1) & (az_all >= s0) & (az_all < s0 + 30)
        sectors[f"{s0:03d}-{s0 + 30:03d}"] = round(float(d_all[sel].max()) / 1000, 2) \
            if sel.any() else 0.0
    pw = 10 ** (tx.power_dbm / 10) / 1000.0
    words = (f"At {tx.freq_hz / 1e6:.4g} MHz, {pw:.3g} W from "
             f"{tx.height_agl_m:g} m ({tx.antenna.describe()}) reaches "
             f"{reach_km2:,.1f} km2 of the {radius_km:g} km circle "
             f"({100 * reach_km2 / max(circle_km2, 1e-9):.0f} %) at "
             f"{thresh:g} dBm or better, by {_prop.MODEL_WORDS[model]}"
             + (f" at {reliability:g} % reliability" if model == "itm" else "")
             + ". A planning prediction, not a promise of contact.")
    summary = {"words": words, "reach_km2": reach_km2, "circle_km2": circle_km2,
               "max_reach_km_by_sector": sectors, "cells": int(n_cells),
               "threshold_dbm": thresh,
               "median_power_dbm": float(np.nanmedian(power))}
    result = ReachResult(grid, power, np.where(inside, loss, np.nan), mask,
                         model, method, tier, summary, notes, kwx=kwx)
    if write and rf is not None:
        params = {"transmitter": tx.to_json(), "receiver": rx.to_json(),
                  "model": model, "model_words": _prop.MODEL_WORDS[model],
                  "radius_km": radius_km, "grid_m": grid_m, "k_factor": k,
                  "reliability_pct": reliability, "confidence_pct": confidence,
                  "ground": ground, "terrain": _terrain.describe_terrain(terrain),
                  "grid": grid.to_json()}
        pr = _products.ProductRun(rf, "coverage", run, label=label or
                                  f"{tx.name or 'tx'}-{model}", tier=tier,
                                  params=params, method=method,
                                  description=words)
        style = {"colormap": "viridis", "min": thresh - 30, "max": thresh + 40}
        pr.add_geotiff("received_power_dbm.tif",
                       np.where(np.isfinite(power), power, NODATA).astype(np.float32),
                       grid, tier=tier, nodata=NODATA,
                       tags={"atk:units": "dBm", "atk:model": model},
                       layer={"name": "physics: received power", "role": "physics",
                              "units": "dBm", "style": style})
        pr.add_geotiff("reach_mask.tif", mask, grid, tier=tier,
                       nodata=MASK_NODATA,
                       tags={"atk:threshold_dbm": thresh},
                       layer={"name": "physics: reach", "role": "physics-mask",
                              "values": {"1": "reached", "0": "not reached"}})
        pr.add_geotiff("path_loss_db.tif",
                       np.where(np.isfinite(result.loss_db), result.loss_db,
                                NODATA).astype(np.float32), grid, tier=tier,
                       nodata=NODATA, tags={"atk:units": "dB"},
                       layer={"name": "path loss", "role": "diagnostic"})
        if kwx is not None:
            pr.add_geotiff("itm_warnings.tif", kwx, grid, tier=tier,
                           tags={"atk:values": _prop.KWX_WORDS},
                           layer={"name": "ITM warnings", "role": "diagnostic"})
        pr.add_geojson("transmitter.geojson",
                       [_products.point_feature(tx.lat, tx.lon,
                                                {"role": "transmitter",
                                                 **tx.to_json()})],
                       tier=tier)
        pr.finish(summary=summary, notes=notes)
        result.run_dir = str(pr.dir)
    return result


# ---------------------------------------------------------------------------
# The measured and learned layers beside the physics
# ---------------------------------------------------------------------------
def add_measurement_layer(rf, run: str, measurements, *,
                          value_key: str = "rx_dbm",
                          interpolate: str | None = "kriging") -> dict:
    """Add what was MEASURED to a coverage run: the points (with the physics
    prediction and the residual at each), and — `interpolate` "kriging" or
    "idw" — a surface between them and the residual surface (INFERRED)."""
    from atk_diffusion.geo import radiomap as _rm
    pr = _products.ProductRun.open(rf, "coverage", run)
    phys = _products.read_geotiff(pr.path("received_power_dbm.tif"))
    meas = measurements if isinstance(measurements, _rm.Measurements) else \
        _rm.Measurements.from_geojson(measurements, value_key)
    pred = phys.sample(meas.lat, meas.lon)
    resid = meas.value - pred
    pr.add_geojson("measured_points.geojson",
                   meas.features("measured", {"physics_dbm": pred,
                                              "residual_db": resid}),
                   tier="measured",
                   layer={"name": "measurement: samples", "role": "measurement",
                          "units": "dBm"})
    ok = np.isfinite(resid)
    stats = {"n": int(ok.sum()),
             "residual_mean_db": float(np.mean(resid[ok])) if ok.any() else None,
             "residual_std_db": float(np.std(resid[ok])) if ok.any() else None,
             "residual_rmse_db": float(np.sqrt(np.mean(resid[ok] ** 2)))
             if ok.any() else None}
    grid = phys.grid
    if interpolate and ok.sum() >= 6:
        res = _rm.Measurements(meas.lat[ok], meas.lon[ok], resid[ok], "residual_db")
        if interpolate == "kriging":
            surf = _rm.ordinary_kriging(meas, grid).estimate
            rsurf = _rm.ordinary_kriging(res, grid)
            rs, rsig = rsurf.estimate, rsurf.sigma
        else:
            surf, _ = _rm.idw(meas, grid)
            rs, _ = _rm.idw(res, grid)
            rsig = None
        method = interpolate
        tier = _prov.tier_for(method)
        valid = np.isfinite(phys.masked())
        pr.add_geotiff(f"measured_dbm_{method}.tif",
                       np.where(valid, surf, NODATA).astype(np.float32), grid,
                       tier=tier, nodata=NODATA, tags={"atk:units": "dBm",
                                                       "atk:method": method},
                       layer={"name": f"measurement: {method} surface",
                              "role": "measurement", "units": "dBm"})
        pr.add_geotiff(f"residual_db_{method}.tif",
                       np.where(valid, rs, NODATA).astype(np.float32), grid,
                       tier=tier, nodata=NODATA, tags={"atk:units": "dB",
                                                       "atk:method": method},
                       layer={"name": "measured minus physics", "role": "residual",
                              "units": "dB"})
        if rsig is not None:
            pr.add_geotiff("residual_sigma_db_kriging.tif",
                           np.where(valid, rsig, NODATA).astype(np.float32),
                           grid, tier=tier, nodata=NODATA,
                           tags={"atk:units": "dB"},
                           layer={"name": "residual uncertainty (1 sigma)",
                                  "role": "uncertainty"})
    pr.finish(measurement=stats)
    return stats


def add_correction_layer(rf, run: str, correction_db, *, sigma_db=None,
                         card=None, method: str = "diffusion_radiomap",
                         hallucination_rate: float | None = None) -> dict:
    """Add a learned correction (INVENTED tier) and the corrected map to a
    coverage run. `correction_db` is on the physics layer's grid."""
    tier = _prov.tier_for(method)
    pr = _products.ProductRun.open(rf, "coverage", run)
    phys = _products.read_geotiff(pr.path("received_power_dbm.tif"))
    corr = np.asarray(correction_db, dtype=np.float64)
    if corr.shape != phys.grid.shape:
        raise ValueError(f"the correction is {corr.shape}; the physics layer "
                         f"is {phys.grid.shape}")
    p = phys.masked()
    valid = np.isfinite(p) & np.isfinite(corr)
    cardj = card.to_json() if hasattr(card, "to_json") else card
    tags = {"atk:units": "dB", "atk:method": method}
    if cardj:
        tags["atk:card"] = cardj
    if hallucination_rate is not None:
        tags["atk:hallucination_rate"] = float(hallucination_rate)
    pr.add_geotiff("correction_db.tif",
                   np.where(valid, corr, NODATA).astype(np.float32), phys.grid,
                   tier=tier, nodata=NODATA, tags=tags,
                   layer={"name": "learned correction", "role": "correction",
                          "units": "dB"})
    pr.add_geotiff("corrected_power_dbm.tif",
                   np.where(valid, p + corr, NODATA).astype(np.float32),
                   phys.grid, tier=tier, nodata=NODATA,
                   tags={**tags, "atk:units": "dBm"},
                   layer={"name": "physics + learned correction",
                          "role": "correction", "units": "dBm"})
    if sigma_db is not None:
        s = np.asarray(sigma_db, dtype=np.float64)
        pr.add_geotiff("correction_sigma_db.tif",
                       np.where(valid, s, NODATA).astype(np.float32), phys.grid,
                       tier=tier, nodata=NODATA, tags=tags,
                       layer={"name": "correction uncertainty (1 sigma)",
                              "role": "uncertainty"})
    stats = {"mean_correction_db": float(np.mean(corr[valid])) if valid.any() else None,
             "max_abs_correction_db": float(np.max(np.abs(corr[valid])))
             if valid.any() else None, "hallucination_rate": hallucination_rate}
    pr.finish(correction=stats)
    return stats
