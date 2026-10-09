# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The map tracks' first experiments (plan E1, E3, E4, E5), measured.

Each is a function that returns a result dict and writes a report
(markdown + JSON) under the profile's `runs\\geo_eval\\`. Synthetic
stand-ins make every one runnable here; on Bill's machine the same function
is pointed at his drive data, his Kraken snapshots, his DTED.

* `df_coverage` (E1) — THE COVERAGE TEST. Over many simulated DF
  geometries (2-4 receivers, 3-15 km, sigmas 1.5-5 deg), how often does the
  true emitter fall inside the stated 50 / 90 / 95 % regions? A region is a
  claim about long-run frequency, so the claim is counted, not trusted.
  Ported from ATK's own audit (`atk/core/siga/coverage.py`, Bill's code):
  every rate carries an exact Clopper-Pearson interval, and a row FAILS
  only when the interval rules out the stated level from below (the
  dangerous direction — regions too small). Two rows are SUPPOSED to fail:
  bearings whose sigma is understated three-fold, and a shared 3 deg array
  bias left unmodelled — they prove the test can see a lie. The bias row is
  run again with the bias modelled, and must pass.
* `whereami_experiment` (E3) — drive data with GPS as truth; a held-out
  calibration route and a held-out test route; the classical locators
  (database kernel, radio-map inversion) and, with PyTorch, the learned MDN;
  error distribution and coverage of the stated regions for each.
* `aperture_experiment` (E4) — a known tower, a driven loop: direct
  position determination with array self-calibration, against the parked
  single-position bearing and the two-step bearings-then-fusion.
* `reach_residual_experiment` (E5) — a broadcaster's predicted coverage
  from its published parameters, a drive that measures it, and the
  residual before and after a correction (kriging always; the learned
  diffusion residual with PyTorch), measured at held-out drive points.
"""

from __future__ import annotations

import json
import math
import time
import zlib
from pathlib import Path

import numpy as np

from atk_diffusion.geo import posterior as _post
from atk_diffusion.geo import terrain as _terrain

LEVELS = (0.5, 0.9, 0.95)
TARGET = 0.90
INTERVAL_P = 0.95
DEFAULT_PROFILE = "krakensdr_2400000_cu8"


def clopper_pearson(hits: int, trials: int, p: float = INTERVAL_P) -> tuple:
    """Exact binomial interval (never under-covers) — the beta quantiles."""
    from scipy.stats import beta
    n, k = int(trials), int(hits)
    if n <= 0:
        return 0.0, 1.0
    a = (1.0 - p) / 2.0
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def _row(label, levels_hit, trials, errors, areas, expect_failure=False, note=""):
    rates, ints = {}, {}
    for lv, h in levels_hit.items():
        rates[f"{lv:g}"] = h / trials if trials else float("nan")
        ints[f"{lv:g}"] = clopper_pearson(h, trials)
    hi90 = ints[f"{TARGET:g}"][1]
    detected = hi90 < TARGET
    passed = detected if expect_failure else not detected
    return {"label": label, "trials": trials, "rates": rates, "intervals": ints,
            "median_error_m": float(np.median(errors)) if errors else None,
            "median_area90_km2": float(np.median(areas)) if areas else None,
            "expect_failure": expect_failure, "passed": bool(passed), "note": note}


def _geometry(rng, center=(38.70, -77.50)):
    tl, to = _terrain.destination(*center, rng.uniform(0, 360), rng.uniform(0, 4000))
    k = int(rng.integers(2, 5))
    az = np.sort(rng.uniform(0, 360, k))
    if k == 2 and abs(((az[1] - az[0] + 180) % 360) - 180) < 25:
        az[1] = (az[0] + rng.uniform(40, 140)) % 360   # not a degenerate pair
    sites = [_terrain.destination(float(tl), float(to), a, rng.uniform(3e3, 15e3))
             for a in az]
    return (float(tl), float(to)), [(float(a), float(b)) for a, b in sites]


def df_coverage(n_trials: int = 300, *, seed: int = 0, levels=LEVELS,
                sigmas=(1.5, 3.0, 5.0), failure_trials: int | None = None,
                max_cells: int = 40_000, progress=None) -> dict:
    """The coverage test (see the module docstring). Returns rows."""
    say = progress or (lambda s: None)
    if int(n_trials) < 1 or (failure_trials is not None and int(failure_trials) < 1):
        raise ValueError("the coverage test needs at least one trial per row "
                         "(a rate of nothing is not a rate)")
    ft = int(failure_trials or max(40, n_trials // 3))

    def run(label, trials, *, understate=1.0, bias=0.0, model_bias=0.0,
            expect_failure=False, note=""):
        rng = np.random.default_rng(seed + zlib.crc32(label.encode()) % 10_000)
        hits = {lv: 0 for lv in levels}
        errors, areas = [], []
        for t in range(trials):
            truth, sites = _geometry(rng)
            obs = []
            for st in sites:
                sig = float(rng.choice(sigmas))
                true_b = float(_terrain.initial_bearing_deg(*st, *truth))
                b = true_b + bias + rng.normal(0, sig * understate)
                obs.append(_post.Bearing(*st, b % 360.0, sig, station="array"))
            post = _post.locate(obs, search_radius_m=25_000, max_cells=max_cells,
                                bias_sigma_deg=model_bias)
            cl = post.credible_level(*truth)
            for lv in levels:
                hits[lv] += int(cl <= lv)
            errors.append(float(_terrain.haversine_m(*post.map_point(), *truth)))
            areas.append(next(r.area_km2 for r in post.regions((TARGET,))))
            if (t + 1) % 50 == 0:
                say(f"coverage '{label}': {t + 1}/{trials}")
        return _row(label, hits, trials, errors, areas, expect_failure, note)
    rows = [
        run("stated sigma = true sigma", n_trials,
            note="the model matches the errors: the regions must hold the "
                 "truth at their stated rate"),
        run("sigma understated 3x", ft, understate=3.0, expect_failure=True,
            note="bearings three times noisier than stated: must be caught"),
        run("shared 3 deg bias, unmodelled", ft, bias=3.0, expect_failure=True,
            note="every bearing from one mis-aligned array: must be caught"),
        run("shared 3 deg bias, modelled", ft, bias=3.0, model_bias=3.0,
            note="the same bias integrated out (bias_sigma_deg=3)"),
    ]
    ok = all(r["passed"] for r in rows)
    return {"experiment": "E1 DF coverage", "levels": list(levels),
            "target": TARGET, "rows": rows, "passed": ok,
            "words": ("the stated regions are honest: every row passed"
                      if ok else "FAILED: at least one row's interval contradicts "
                                 "its claim")}


# ---------------------------------------------------------------------------
def _model_home(rf, profile: str, tag: str):
    """Where an experiment's own trained model goes: beside its report under
    the profile's runs folder (kept, with its card) when there is an rf_data
    root; a temporary folder otherwise (tests). -> (context manager, words)."""
    import contextlib
    import tempfile
    if rf is not None:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        d = Path(rf.runs(profile)) / "geo_eval" / f"{stamp}_{tag}"
        k = 2
        while d.exists():
            d = Path(rf.runs(profile)) / "geo_eval" / f"{stamp}_{tag}_{k}"
            k += 1
        d.mkdir(parents=True)
        return contextlib.nullcontext(str(d)), str(d)
    return tempfile.TemporaryDirectory(), "a temporary folder (no rf_data given)"


def whereami_experiment(db_records=None, calib_records=None, test_records=None,
                        *, seed: int = 0, learned: bool = True,
                        progress=None, mdn_epochs: int = 120, rf=None,
                        profile: str = DEFAULT_PROFILE) -> dict:
    """E3: classical locators (and the MDN when PyTorch is here) on held-out
    routes. Without records, the synthetic drive world stands in.
    `mdn_epochs` is the learned model's training budget; with `rf` the MDN
    is kept under the profile's runs\\geo_eval\\ (never a temp folder in the
    user profile)."""
    from atk_diffusion.geo import whereami as W
    say = progress or (lambda s: None)
    synthetic = db_records is None
    if synthetic:
        world = W.DriveWorld(size_m=4000, n_cells=8, n_fm=3, seed=seed)
        db_records = sum((world.drive(400, seed=seed + s) for s in range(1, 6)), [])
        calib_records = world.drive(150, seed=seed + 50)
        test_records = world.drive(200, seed=seed + 60)
    db = W.FingerprintDB.from_records(db_records)
    out = {"experiment": "E3 where-am-I", "synthetic": synthetic,
           "database_points": len(db), "features": len(db.keys),
           "calibration_points": len(calib_records), "test_points": len(test_records),
           "methods": {}}
    say("where-am-I: database-point kernel")
    kl = W.KernelLocator(db)
    out["methods"]["kernel_uncalibrated"] = kl.evaluate(test_records)
    cal = kl.calibrate(calib_records)
    out["methods"]["kernel"] = {**kl.evaluate(test_records), "calibration": cal}
    say("where-am-I: radio-map inversion")
    ml = W.MapLocator(db)
    out["methods"]["map_inversion_uncalibrated"] = ml.evaluate(test_records)
    cal = ml.calibrate(calib_records)
    out["methods"]["map_inversion"] = {**ml.evaluate(test_records),
                                       "calibration": cal}
    if learned:
        try:
            import importlib.util
            if importlib.util.find_spec("torch") is None:
                raise ImportError("PyTorch is not in this environment")
            from atk_diffusion.learn import position as LP
            say("where-am-I: mixture-density network")
            home, where = _model_home(rf, profile, "e3_mdn")
            with home as td:
                d = LP.train(db_records, Path(td) / "mdn", calib_records=calib_records,
                             epochs=int(mdn_epochs), seed=seed)
                out["methods"]["mdn"] = LP.load(d).evaluate(test_records)
                out["methods"]["mdn"]["epochs"] = int(mdn_epochs)
                out["methods"]["mdn"]["model"] = str(d) if rf is not None else where
        except ImportError as e:
            out["methods"]["mdn"] = {"skipped": str(e)}
    for m, r in out["methods"].items():
        if "coverage" in r:
            n = r["n"]
            r["coverage_intervals"] = {lv: clopper_pearson(round(v * n), n)
                                       for lv, v in r["coverage"].items()}
    return out


def aperture_experiment(snapshots=None, tower=None, *, freq_hz: float = 98.1e6,
                        radius_m: float = 1.0, seed: int = 0,
                        sigma_deg: float = 3.0) -> dict:
    """E4: localize a known tower from a driven loop; versus the parked
    bearing and the two-step method. Without snapshots, simulated: a tower
    8 km north of a 1.5 km loop, 5-channel circle of `radius_m`."""
    from atk_diffusion.geo import aperture as AP
    geom = AP.ArrayGeometry.uca(radius_m)
    synthetic = snapshots is None
    if synthetic:
        center = (38.70, -77.50)
        t = _terrain.destination(*center, 0.0, 8000.0)
        tower = (float(t[0]), float(t[1]))
        rl, ro = AP.loop_route(*center, 1500.0, 36)
        snapshots = AP.simulate_drive(*tower, rl, ro, geom, freq_hz,
                                      rng=np.random.default_rng(seed))
    if tower is None:
        raise ValueError("the experiment needs the tower's known position")
    res = AP.locate_moving(snapshots, geom, freq_hz, sigma_deg=sigma_deg)
    raw = AP.locate_moving(snapshots, geom, freq_hz, sigma_deg=sigma_deg,
                           calibrate=False)
    park_b, park = AP.stationary_fix(snapshots[0], geom, freq_hz, sigma_deg=sigma_deg)
    two = AP.two_step(res)
    true_b = float(_terrain.initial_bearing_deg(park_b.lat, park_b.lon, *tower))
    rng_m = float(_terrain.haversine_m(park_b.lat, park_b.lon, *tower))
    berr = float(_terrain.wrap180(park_b.bearing_deg - true_b))
    err = lambda p: float(_terrain.haversine_m(*p, *tower))   # noqa: E731
    return {"experiment": "E4 synthetic aperture", "synthetic": synthetic,
            "snapshots": len(snapshots),
            "dpd_self_calibrated_error_m": err(res.estimate),
            "dpd_self_calibrated_truth_level": res.posterior.credible_level(*tower),
            "dpd_uncalibrated_error_m": err(raw.estimate),
            "two_step_error_m": err(two.map_point()),
            "parked_bearing_error_deg": berr,
            "parked_cross_range_error_m": abs(math.radians(berr)) * rng_m,
            "parked_range": "none — one position gives a bearing, not a fix",
            "channel_phase_deg": res.meta.get("channel_phase_deg"),
            "words": res.words}


def reach_residual_experiment(*, seed: int = 0, learned: bool = True,
                              rf=None, model: str = "deygout",
                              learned_steps: int = 500, learned_fields: int = 160,
                              learned_size: int = 32,
                              profile: str = DEFAULT_PROFILE) -> dict:
    """E5: predicted reach from published parameters; a drive; the residual
    before and after correction, at held-out drive points. Synthetic: a
    ridge-and-valley terrain whose built-up valleys add a clutter loss the
    physics does not model. `learned_steps` / `learned_fields` /
    `learned_size` are the learned residual's training budget (the default
    is a CPU-sized smoke run; on a GPU give it tens of thousands of steps)
    — the result says which budget it measured. With `rf` the learned model
    is kept under `profile`'s runs\\geo_eval\\ (never a temp folder in the
    user profile)."""
    from atk_diffusion.geo import radiomap as RM
    from atk_diffusion.geo import reach as R
    from atk_diffusion.geo.products import GeoGrid
    rng = np.random.default_rng(seed)
    g = GeoGrid.around(38.70, -77.50, 9000, 150)
    lat, lon = g.mesh()
    e, n = _terrain.enu_m(lat, lon, 38.70, -77.50)
    z = 120 + 60 * np.sin(e / 1500.0) * np.cos(n / 2100.0)
    terr = _terrain.GridTerrain(g, z, label="synthetic ridges and valleys")
    tx = R.Transmitter(38.70, -77.50, 98.1e6, power_dbm=60.0, height_agl_m=60.0,
                       name="broadcaster (published parameters)")
    res = R.predicted_reach(tx, R.Receiver(sensitivity_dbm=-90.0), terr, 6.0,
                            model=model, grid_m=300, rf=rf,
                            run=f"e5-{seed}" if rf else None, write=rf is not None)
    pg = res.grid
    plat, plon = pg.mesh()
    zz = terr.elevation(plat, plon)
    clutter = np.where(zz < 120, -8.0, 0.0)                # the unmodelled part
    truth = res.power_dbm + clutter
    ok = np.isfinite(truth)
    idx = np.argwhere(ok)
    pick = idx[rng.choice(len(idx), size=min(300, len(idx)), replace=False)]
    m_lat = plat[pick[:, 0], pick[:, 1]]
    m_lon = plon[pick[:, 0], pick[:, 1]]
    m_val = truth[pick[:, 0], pick[:, 1]] + rng.normal(0, 2.0, len(pick))
    train = np.arange(len(pick)) % 3 != 0                  # a third held out
    meas = RM.Measurements(m_lat[train], m_lon[train], m_val[train], "rx_dbm")
    held_lat, held_lon, held_val = m_lat[~train], m_lon[~train], m_val[~train]
    phys_held = pg.sample(res.power_dbm, held_lat, held_lon)
    out = {"experiment": "E5 predicted reach, residual", "model": model,
           "residual_before_rmse_db": float(np.sqrt(np.nanmean((held_val - phys_held) ** 2)))}
    resid = RM.residuals(meas, (pg, res.power_dbm))
    kr = RM.ordinary_kriging(resid, pg)
    kh = pg.sample(kr.estimate, held_lat, held_lon)
    out["residual_after_kriging_rmse_db"] = float(
        np.sqrt(np.nanmean((held_val - phys_held - kh) ** 2)))
    if rf is not None:
        R.add_measurement_layer(rf, f"e5-{seed}", meas)
        out["run_dir"] = res.run_dir
    if learned:
        try:
            import importlib.util
            if importlib.util.find_spec("torch") is None:
                raise ImportError("PyTorch is not in this environment")
            from atk_diffusion.learn import radiomap as LR
            fields = LR.synthetic_fields(int(learned_fields), int(learned_size),
                                         seed=seed + 7)
            out["learned_budget"] = {"steps": int(learned_steps),
                                     "fields": int(learned_fields),
                                     "size": int(learned_size)}
            home, where = _model_home(rf, profile, "e5_radiomap")
            with home as td:
                d = LR.train(fields, Path(td) / "rm", steps=int(learned_steps),
                             seed=seed)
                out["learned_model"] = str(d) if rf is not None else where
                mdl = LR.load(d)
                mask = np.zeros(pg.shape)
                rr, cc = pg.rowcol(resid.lat, resid.lon)
                rr = np.clip(np.round(rr).astype(int), 0, pg.height - 1)
                cc = np.clip(np.round(cc).astype(int), 0, pg.width - 1)
                rvals = np.zeros(pg.shape)
                mask[rr, cc] = 1.0
                rvals[rr, cc] = resid.value
                corr, sig = LR.correct_grid(mdl, np.nan_to_num(res.power_dbm, nan=-140),
                                            zz, mask, rvals, seed=seed)
                lh = pg.sample(corr, held_lat, held_lon)
                out["residual_after_learned_rmse_db"] = float(
                    np.sqrt(np.nanmean((held_val - phys_held - lh) ** 2)))
                out["learned_hallucination_rate"] = LR.hallucination_rate(mdl)
                if rf is not None:
                    R.add_correction_layer(rf, f"e5-{seed}", corr, sigma_db=sig,
                                           card=mdl[1],
                                           hallucination_rate=out["learned_hallucination_rate"])
        except ImportError as e:
            out["learned"] = f"skipped: {e}"
    return out


# ---------------------------------------------------------------------------
def write_report(rf, result: dict, *, profile: str = DEFAULT_PROFILE,
                 name: str = "geo_eval") -> dict:
    """result.json + report.md under <rf_data>\\<profile>\\runs\\geo_eval\\."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    base = Path(rf.runs(profile)) / "geo_eval"
    d = base / f"{stamp}_{name}"
    k = 2
    while d.exists():                     # never reuse a run's folder
        d = base / f"{stamp}_{name}_{k}"
        k += 1
    d.mkdir(parents=True)
    jp = d / "result.json"
    jp.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    lines = [f"# {result.get('experiment', name)}", "",
             f"Run {stamp} (UTC). Generated from result.json; every number "
             "below is in it.", ""]
    if "rows" in result:
        lines += ["| row | trials | 50 % | 90 % [95 % CI] | 95 % | median error | "
                  "median 90 % area | verdict |", "|---|---|---|---|---|---|---|---|"]
        for r in result["rows"]:
            lo, hi = r["intervals"]["0.9"]
            verdict = ("ok" if r["passed"] else "FAIL") + \
                (" (meant to fail)" if r["expect_failure"] else "")
            lines.append(f"| {r['label']} | {r['trials']} | {r['rates']['0.5']:.0%} | "
                         f"{r['rates']['0.9']:.0%} [{lo:.0%}, {hi:.0%}] | "
                         f"{r['rates']['0.95']:.0%} | {r['median_error_m']:.0f} m | "
                         f"{r['median_area90_km2']:.3g} km2 | {verdict} |")
        lines += ["", result.get("words", "")]
    else:
        for k, v in result.items():
            if isinstance(v, dict):
                lines.append(f"## {k}")
                for k2, v2 in v.items():
                    lines.append(f"- **{k2}**: {v2}")
            else:
                lines.append(f"- **{k}**: {v}")
    mp = d / "report.md"
    mp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for p in (jp, mp):
        rf.record(p, "experiment", name)
    return {"dir": str(d), "json": str(jp), "markdown": str(mp)}
