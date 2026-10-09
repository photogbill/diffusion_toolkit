# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""J's first experiment: measured openings versus the prediction (plan §4.J).

*"One evening, Virginia to three chosen Kiwis; measured openings versus the
prediction."* `run()` takes the evening's WSPR spots (from `hf.wspr` —
ALL_WSPR.TXT files or wsprd decodes of each Kiwi's audio), builds the
band x hour openness matrix for each Kiwi from transmitters near HERE, asks
VOACAP (`hf.openness.Voacap`) for the same circuit, band and hour, and
writes the table side by side: measured spots / median SNR / best distance /
open, beside predicted REL / SNR / open, and where both exist, whether they
agree. Products (matrix JSON + CSV, reach arcs GeoJSON) go to
`rf.products("hf")`, the report under the Kiwi profile's runs folder.

Without a decoder or a network here, `synthetic_evening()` makes a stated,
simple evening of spots (bands opening and closing with the hour, SNR
falling with distance) so the whole path runs; a synthetic table is labelled
synthetic. Without voacapl, every predicted cell says "prediction
unavailable" in words — the measured half still stands on its own.

HERE defaults to the 4-character square FM18 (northern Virginia); the three
default Kiwi regions are illustrative grid squares, to be replaced by the
Kiwis Bill chooses.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from atk_diffusion import profiles as _profiles
from atk_diffusion.hf import maidenhead as _mh
from atk_diffusion.hf import openness as _op
from atk_diffusion.hf import wspr as _wspr

HERE_GRID = "FM18"
DEFAULT_KIWIS = ({"name": "England", "grid": "IO91"},
                 {"name": "Colorado", "grid": "DM79"},
                 {"name": "Puerto Rico", "grid": "FK68"})
DEFAULT_BANDS = ("80m", "40m", "30m", "20m")
DEFAULT_HOURS = (20, 21, 22, 23, 0, 1, 2, 3)
KIWI_PROFILE = _profiles.make_profile_id("kiwisdr", 12000, "ci16")


def _open_model(band: str, hour: int, km: float) -> float:
    """The synthetic evening's rule: probability that a path is open. A
    stated toy — 20 m fades after dark on long paths, 40/30 m open through
    the evening, 80 m only after dark and short to medium paths."""
    dark = hour >= 23 or hour <= 10
    if band == "20m":
        return 0.85 if (hour <= 22 and hour >= 12) else (0.35 if km > 3000 else 0.1)
    if band == "40m":
        return 0.8 if hour >= 21 or hour <= 6 else 0.3
    if band == "30m":
        return 0.7 if hour >= 20 or hour <= 4 else 0.4
    if band == "80m":
        return (0.75 if km < 3000 else 0.25) if dark else 0.05
    return 0.0


def synthetic_evening(here_grid: str = HERE_GRID, kiwis=DEFAULT_KIWIS, *,
                      date: str = "2026-10-08", hours=DEFAULT_HOURS,
                      bands=DEFAULT_BANDS, stations: int = 6, rng=None) -> list:
    """Spots from `stations` transmitters around HERE to each Kiwi, every
    WSPR slot of each hour, under `_open_model`. SYNTHETIC — exercising the
    code path, not a propagation claim."""
    rng = rng or np.random.default_rng(0)
    la0, lo0 = _mh.to_latlon(here_grid)
    tx = []
    for i in range(stations):
        la = la0 + float(rng.uniform(-1.5, 1.5))
        lo = lo0 + float(rng.uniform(-2.0, 2.0))
        tx.append((f"K4T{chr(65 + i)}{chr(66 + i)}", _mh.from_latlon(la, lo, 4)))
    spots = []
    base = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    for k in kiwis:
        rx_grid = _mh.normalize(k["grid"])
        for h in hours:
            day = base if h >= hours[0] else base + timedelta(days=1)
            for band in bands:
                for call, g in tx:
                    la1, lo1 = _mh.to_latlon(g)
                    la2, lo2 = _mh.to_latlon(rx_grid)
                    km = _wspr.distance_km(la1, lo1, la2, lo2)
                    p = _open_model(band, h, km)
                    for slot in range(0, 60, 10):            # a few slots/hour
                        if rng.random() < p:
                            snr = int(round(-8 - 9 * np.log10(max(km, 100) / 1000)
                                            + rng.normal(0, 4)))
                            snr = max(-32, min(10, snr))
                            f = (_wspr.WSPR_DIAL_MHZ[band] * 1e6 + 1500
                                 + float(rng.uniform(-100, 100)))
                            s = _wspr.Spot(
                                time_utc=f"{day:%Y-%m-%d}T{h:02d}:{slot:02d}:00Z",
                                snr_db=snr, dt_s=float(rng.normal(0, 0.5)),
                                freq_hz=f, call=call, grid=g, power_dbm=23,
                                drift_hz=0, band=band, rx_call=k["name"],
                                rx_grid=rx_grid, source="synthetic evening")
                            spots.append(_wspr.place(s))
    return spots


def run(rf=None, spots=None, *, here_grid: str = HERE_GRID,
        here_radius_km: float = 400.0, kiwis=DEFAULT_KIWIS,
        bands=DEFAULT_BANDS, hours=DEFAULT_HOURS, year: int = 2026,
        month: int = 10, ssn: float = 100.0, voacap: _op.Voacap | None = None,
        out_dir=None, progress: Callable[[str], None] | None = None) -> dict:
    """Measured versus predicted, per Kiwi, band and hour. -> result dict."""
    synthetic = spots is None
    if synthetic:
        spots = synthetic_evening(here_grid, kiwis, hours=hours, bands=bands)
    la0, lo0 = _mh.to_latlon(here_grid)
    here = {"lat": la0, "lon": lo0, "radius_km": float(here_radius_km)}
    voacap = voacap or _op.Voacap()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    per_kiwi = []
    rows = []
    for k in kiwis:
        rx_grid = _mh.normalize(k["grid"])
        m = _op.matrix(spots, bands=bands, hours=hours,
                       receivers={"grids": [rx_grid]}, here=here,
                       label=f"{here_grid} → {k['name']} ({rx_grid})")
        la1, lo1 = _mh.to_latlon(rx_grid)
        pred = voacap.predict((la0, lo0), (la1, lo1), year=year, month=month,
                              ssn=ssn, hours=[h if h else 24 for h in hours],
                              freqs_mhz=[_op.band_freq_mhz(b) for b in bands])
        agree_n = both_n = 0
        for b in bands:
            fb = _op.band_freq_mhz(b)
            for h in hours:
                c = m["cells"][b][h]
                row = {"kiwi": k["name"], "band": b, "hour_utc": h,
                       "measured_spots": c["spots"], "median_snr_db": c["median_snr_db"],
                       "best_km": c["best_km"], "measured_open": c["open"],
                       "predicted_rel": None, "predicted_snr_db": None,
                       "predicted_open": None, "agree": None,
                       "prediction": "" if pred.get("available")
                       else pred.get("why", "prediction unavailable")}
                if pred.get("available"):
                    cell = pred["table"]["hours"].get(h % 24, {}).get(fb, {})
                    if "rel" in cell:
                        row["predicted_rel"] = cell["rel"]
                        row["predicted_snr_db"] = cell.get("snr")
                        row["predicted_open"] = cell["rel"] >= 0.5
                        row["agree"] = row["predicted_open"] == row["measured_open"]
                        both_n += 1
                        agree_n += int(row["agree"])
                    else:
                        row["prediction"] = "no prediction for this band and hour"
                rows.append(row)
        files = {}
        if rf is not None:
            files = _op.write_products(rf, m, spots, run=f"hf_eval_{stamp}_"
                                       f"{rx_grid}", prediction=pred)
        per_kiwi.append({"kiwi": k["name"], "grid": rx_grid,
                         "spots_used": m["spots_used"],
                         "prediction_available": bool(pred.get("available")),
                         "prediction_why": pred.get("why", ""),
                         "agreement": (agree_n / both_n) if both_n else None,
                         "compared_cells": both_n, "products": files,
                         "matrix_lines": _op.matrix_lines(m)})
        if progress:
            progress(f"{k['name']}: {m['spots_used']} spots")
    result = {"experiment": "hf_eval (plan §4.J) — measured versus predicted",
              "here": here_grid, "kiwis": per_kiwi, "rows": rows,
              "synthetic": synthetic, "tier": "measured",
              "prediction_tier": "inferred",
              "what": ("SYNTHETIC spots — exercising the code path, not a "
                       "propagation claim" if synthetic else
                       "WSPR decodes heard by the Kiwis"),
              "created": stamp}
    result["report_md"] = report_md(result)
    out = Path(out_dir) if out_dir is not None else (
        Path(rf.runs(KIWI_PROFILE)) / f"hf_eval_{stamp}" if rf is not None else None)
    files = []
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        jp, mp = out / "result.json", out / "report.md"
        jp.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        mp.write_text(result["report_md"], encoding="utf-8")
        files = [str(jp), str(mp)]
        if rf is not None:
            for f in (jp, mp):
                try:
                    rf.record(f, "experiment", "hf_eval")
                except Exception:                          # noqa: BLE001
                    pass
    result["files"] = files
    return result


def report_md(r: dict) -> str:
    lines = ["# HF propagation now — measured versus predicted (plan §4.J)", "",
             f"From **{r['here']}**. *{r['what']}.* Measured cells are WSPR "
             "decodes (MEASURED); predicted cells are VOACAP (a model, "
             "INFERRED).", ""]
    for k in r["kiwis"]:
        lines.append(f"## {k['kiwi']} ({k['grid']}) — {k['spots_used']} spots")
        if not k["prediction_available"]:
            lines.append(f"*{k['prediction_why']}.*")
        elif k["agreement"] is not None:
            lines.append(f"Measured and predicted agree on open/closed in "
                         f"{100 * k['agreement']:.0f}% of {k['compared_cells']}"
                         " cells.")
        lines += ["", "```"] + k["matrix_lines"] + ["```", ""]
        lines += ["| band | hour UTC | spots | median SNR | best km | open | "
                  "predicted REL | predicted open |", "|---|---|---|---|---|---|---|---|"]
        for row in r["rows"]:
            if row["kiwi"] != k["kiwi"]:
                continue
            pr = ("—" if row["predicted_rel"] is None
                  else f"{row['predicted_rel']:.2f}")
            po = ("prediction unavailable" if row["predicted_open"] is None
                  else ("open" if row["predicted_open"] else "closed"))
            snr = "—" if row["median_snr_db"] is None else f"{row['median_snr_db']:.0f} dB"
            km = "—" if row["best_km"] is None else f"{row['best_km']:.0f}"
            lines.append(f"| {row['band']} | {row['hour_utc']:02d} | "
                         f"{row['measured_spots']} | {snr} | {km} | "
                         f"{'yes' if row['measured_open'] else 'nothing heard'} | "
                         f"{pr} | {po} |")
        lines.append("")
    return "\n".join(lines) + "\n"
