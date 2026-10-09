# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""HF propagation NOW: which bands are open, hour by hour, between here and
there — measured from WSPR spots heard by Kiwis, beside a prediction
(plan §4.J; products in rf_data\\products\\hf\\, §3.7).

*"Which band do I use to reach the team in the next valley"* — answered from
measurement. A WSPR decode is a measured path; many of them, binned by band
and UTC hour, are a BAND-OPENNESS MATRIX:

    band x hour -> spots, distinct stations, median SNR, best distance, open?

`matrix()` builds it from spots received by the Kiwis in chosen REGIONS
(around a point, or by grid prefix), optionally only from transmitters near
HERE — that is "between here and there". `arcs_geojson()` paints each spot
as a great-circle line from transmitter to receiver, coloured by SNR in
ATK's own waterfall ramp (blue weak → yellow strong), for the map.
`write_products()` puts the matrix (JSON and CSV) and the arcs (GeoJSON)
under `rf.products("hf")` — open formats, outside the install, read by ATK's
Leaflet map or anything else. (The plan names Parquet for the matrix; that
needs pyarrow, which the core environment does not carry, so the table is
CSV — open, and readable by everything.)

THE PREDICTION, BESIDE IT. `Voacap` drives voacapl (the Linux/Wine port of
VOACAP, the ITS HF prediction program) when it is installed: it writes a
point-to-point Method 30 input deck, runs the program as its own process,
and reads REL (circuit reliability) and SNR per hour and frequency out of
its output. When voacapl is not installed — or runs and prints nothing
readable — the prediction is "unavailable" IN WORDS, never a number from
somewhere else. THE DECK WRITER AND THE OUTPUT READER ARE WRITTEN FROM THE
PROGRAM'S DOCUMENTED CARD FORMAT AND ARE NOT VERIFIED IN THIS SANDBOX (no
voacapl here); a test runs them against the real program and is skipped,
with that reason, until it is installed.

LIMITS. WSPR measures what WSPR stations transmitted: an empty cell means
"nothing was heard", not "closed" — the matrix says how many stations it
rests on. Grid centres carry ±55–110 km. A Kiwi's antenna and noise floor
shape what it hears; spots from different Kiwis are kept apart by `rx_grid`.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np

from atk_diffusion import provenance
from atk_diffusion.hf import maidenhead as _mh
from atk_diffusion.hf import wspr as _wspr

provenance.METHOD_TIERS.setdefault("wspr_openness", "measured")
provenance.METHOD_TIERS.setdefault("voacap", "inferred")

SNR_RANGE = (-30.0, 10.0)      # WSPR's useful range, for the arc colours


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------
def _latlon_of_grid(g: str):
    return _mh.to_latlon(g) if g and _mh.is_grid(g) else None


def in_region(grid: str, region: dict | None) -> bool:
    """Is a locator inside a region? A region is {lat, lon, radius_km} or
    {grids: ["FM18", "FN"]} (prefixes); None is everywhere."""
    if region is None:
        return True
    if "grids" in region:
        g = str(grid or "").upper()
        return any(g.startswith(str(p).upper()) for p in region["grids"])
    ll = _latlon_of_grid(grid)
    if ll is None:
        return False
    return _wspr.distance_km(ll[0], ll[1], float(region["lat"]),
                             float(region["lon"])) <= float(region["radius_km"])


def select(spots, receivers: dict | None = None, here: dict | None = None
           ) -> list:
    """Spots heard by Kiwis in `receivers` (a region), from transmitters in
    `here` (a region) when given — "between here and there"."""
    return [s for s in spots if in_region(s.rx_grid, receivers)
            and in_region(s.grid, here)]


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------
def matrix(spots, *, bands=None, hours=range(24), min_spots: int = 1,
           receivers: dict | None = None, here: dict | None = None,
           label: str = "") -> dict:
    """The band x hour openness matrix. -> {bands, hours, cells{band{hour}},
    spots_used, tier, what}."""
    used = select(spots, receivers, here)
    bands = list(bands) if bands else [b for b in _wspr.BAND_ORDER
                                       if any(s.band == b for s in used)]
    hours = [int(h) for h in hours]
    cells: dict = {b: {} for b in bands}
    for b in bands:
        for h in hours:
            ss = [s for s in used if s.band == b and s.hour == h]
            dist = [s.distance_km for s in ss if s.distance_km is not None]
            cells[b][h] = {
                "spots": len(ss),
                "stations": len({s.call for s in ss}),
                "median_snr_db": float(np.median([s.snr_db for s in ss])) if ss else None,
                "best_km": float(max(dist)) if dist else None,
                "open": len(ss) >= int(min_spots)}
    return {"label": label, "bands": bands, "hours": hours, "cells": cells,
            "spots_used": len(used), "min_spots": int(min_spots),
            "receivers": receivers, "here": here,
            "tier": provenance.tier_for("wspr_openness"),
            "what": ("measured: WSPR decodes binned by band and UTC hour. An "
                     "empty cell means nothing was heard, not that the band "
                     "was closed.")}


def matrix_lines(m: dict) -> list[str]:
    """A plain-text table: one row per band, '·' nothing heard, else spots."""
    hs = m["hours"]
    out = ["band  " + " ".join(f"{h:02d}" for h in hs)]
    for b in m["bands"]:
        row = []
        for h in hs:
            c = m["cells"][b][h]
            row.append(" ·" if not c["spots"] else f"{min(c['spots'], 99):2d}")
        out.append(f"{b:5s} " + " ".join(row))
    out.append("(spots per hour; · = nothing heard — not proof of a closed band)")
    return out


def to_csv(m: dict, path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["band", "hour_utc", "spots", "stations", "median_snr_db",
                    "best_km", "open", "tier"])
        for b in m["bands"]:
            for h in m["hours"]:
                c = m["cells"][b][h]
                w.writerow([b, h, c["spots"], c["stations"], c["median_snr_db"],
                            c["best_km"], c["open"], m["tier"]])
    return p


# ---------------------------------------------------------------------------
# Arcs for the map
# ---------------------------------------------------------------------------
def great_circle(lat1, lon1, lat2, lon2, n: int = 32) -> list[list[float]]:
    """[lon, lat] points along the great circle (GeoJSON order). Longitudes
    are kept continuous across ±180° so the line is drawn the short way."""
    p1, l1, p2, l2 = map(math.radians, (lat1, lon1, lat2, lon2))
    a = np.array([math.cos(p1) * math.cos(l1), math.cos(p1) * math.sin(l1), math.sin(p1)])
    b = np.array([math.cos(p2) * math.cos(l2), math.cos(p2) * math.sin(l2), math.sin(p2)])
    om = math.acos(max(-1.0, min(1.0, float(np.dot(a, b)))))
    pts = []
    prev = None
    for t in np.linspace(0.0, 1.0, max(2, int(n))):
        if om < 1e-12:
            v = a
        else:
            v = (math.sin((1 - t) * om) * a + math.sin(t * om) * b) / math.sin(om)
        lat = math.degrees(math.atan2(v[2], math.hypot(v[0], v[1])))
        lon = math.degrees(math.atan2(v[1], v[0]))
        if prev is not None:
            while lon - prev > 180:
                lon -= 360
            while lon - prev < -180:
                lon += 360
        prev = lon
        pts.append([round(lon, 5), round(lat, 5)])
    return pts


def snr_colour(snr_db: float) -> str:
    from atk_diffusion.ask.point import ramp_rgb
    r, g, b = (int(v) for v in ramp_rgb(float(snr_db), *SNR_RANGE))
    return f"#{r:02x}{g:02x}{b:02x}"


def arcs_geojson(spots, n_points: int = 32) -> dict:
    """One LineString per spot with both grids known, transmitter to
    receiver, coloured by SNR in ATK's waterfall ramp."""
    feats = []
    for s in spots:
        a, b = _latlon_of_grid(s.grid), _latlon_of_grid(s.rx_grid)
        if a is None or b is None:
            continue
        feats.append({"type": "Feature",
                      "geometry": {"type": "LineString",
                                   "coordinates": great_circle(a[0], a[1], b[0],
                                                               b[1], n_points)},
                      "properties": {
                          "name": f"{s.call} → {s.rx_call or s.rx_grid}",
                          "band": s.band, "snr_db": s.snr_db,
                          "distance_km": s.distance_km, "time_utc": s.time_utc,
                          "hour_utc": s.hour, "power_dbm": s.power_dbm,
                          "colour": snr_colour(s.snr_db),
                          "stroke": snr_colour(s.snr_db), "stroke-width": 2,
                          "tier": "measured",
                          "description": (f"{s.band}, {s.snr_db} dB SNR, "
                                          f"{s.distance_km} km at {s.time_utc}"
                                          " — a WSPR decode (measured)")}})
    return {"type": "FeatureCollection", "features": feats,
            "properties": {"what": "WSPR reach arcs, transmitter → Kiwi, "
                                   "coloured by SNR", "tier": "measured",
                           "snr_range_db": list(SNR_RANGE)}}


def write_products(rf, m: dict, spots, run: str = "", prediction: dict | None = None
                   ) -> dict:
    """matrix JSON + CSV and the arcs GeoJSON under rf.products('hf')/<run>/,
    each recorded in the write log. -> {name: path}."""
    run = run or time.strftime("%Y%m%dT%H%MZ", time.gmtime())
    d = Path(rf.products("hf")) / run
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    jp = d / "openness.json"
    body = dict(m)
    body["prediction"] = prediction or {"available": False,
                                        "why": "no prediction was asked for"}
    body["provenance"] = provenance.stamp("hf.openness", spots=len(spots))
    jp.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    out["matrix_json"] = jp
    out["matrix_csv"] = to_csv(m, d / "openness.csv")
    gp = d / "arcs.geojson"
    gp.write_text(json.dumps(arcs_geojson(select(spots, m.get("receivers"),
                                                 m.get("here")))),
                  encoding="utf-8")
    out["arcs"] = gp
    for k, p in out.items():
        try:
            rf.record(p, f"hf-{k}", run)
        except Exception:                                  # noqa: BLE001
            pass
    return {k: str(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# The prediction — VOACAP through voacapl, at arm's length
# ---------------------------------------------------------------------------
_LABELS = ("FREQ", "MODE", "TANGLE", "DELAY", "V HITE", "MUFday", "LOSS",
           "DBU", "S DBW", "N DBW", "SNR", "RPWRG", "REL", "MPROB", "S PRB",
           "SIG LW", "SIG UP", "SNR LW", "SNR UP", "TGAIN", "RGAIN", "SNRxx",
           "DBM")


def _ll(v: float, pos: str, neg: str, width: int) -> str:
    return f"{abs(v):{width}.2f}{pos if v >= 0 else neg}"


def voacap_deck(tx: tuple, rx: tuple, *, year: int, month: int, ssn: float,
                freqs_mhz, hours=range(1, 25), power_kw: float = 0.1,
                min_takeoff_deg: float = 3.0, required_rel: float = 90.0,
                required_snr: float = 24.0, noise_dbw: float = 145.0,
                tx_label: str = "TX", rx_label: str = "RX") -> str:
    """A point-to-point Method 30 deck (VOACAP's card format: a 10-character
    card name, then fixed-width fields). Isotropic antennas; `power_kw` on
    the transmit antenna card. UNVERIFIED HERE — see the module docstring."""
    fq = [float(f) for f in list(freqs_mhz)[:11]]
    fq += [0.0] * (11 - len(fq))
    hs = list(hours)
    h0, h1 = min(hs), max(hs)
    lines = [
        "LINEMAX      55       number of lines-per-page",
        "COEFFS    CCIR",
        f"TIME      {h0:4d}{h1:5d}    1    1",
        f"MONTH     {int(year):5d}{float(month):5.2f}",
        f"SUNSPOT   {float(ssn):5.0f}.",
        f"LABEL     {tx_label[:20]:<20}{rx_label[:20]}",
        "CIRCUIT   " + _ll(tx[0], "N", "S", 5) + "   " + _ll(tx[1], "E", "W", 7)
        + "   " + _ll(rx[0], "N", "S", 5) + "   " + _ll(rx[1], "E", "W", 7)
        + "  S     0",
        f"SYSTEM       1. {noise_dbw:4.0f}. {min_takeoff_deg:4.2f} "
        f"{required_rel:3.0f}. {required_snr:4.1f} 3.00 0.10",
        "FPROB      1.00 1.00 1.00 0.00",
        f"ANTENNA       1    1    2   30     0.000[default/isotrope     ]  0.0"
        f"{power_kw:10.4f}",
        "ANTENNA       2    2    2   30     0.000[default/isotrope     ]  0.0"
        "    0.0000",
        "FREQUENCY " + "".join(f"{f:5.2f}" for f in fq),
        "METHOD       30    0",
        "EXECUTE",
        "QUIT"]
    return "\n".join(lines) + "\n"


def parse_voacap_output(text: str, freqs_mhz) -> dict:
    """{hour: {freq_mhz: {"rel", "snr"}}, "muf": {hour: mhz}} from a Method 30
    output, read by the labels at the right of each row (FREQ, REL, SNR …).
    Empty when nothing readable is there."""
    fq = [float(f) for f in freqs_mhz if float(f) > 0]
    out: dict = {"hours": {}, "muf": {}}
    hour = None
    num = re.compile(r"[-+]?\d+(?:\.\d+)?")
    for raw in str(text or "").splitlines():
        line = raw.rstrip()
        lab = next((L for L in _LABELS if line.endswith(L)), None)
        if lab is None:
            continue
        vals = [float(v) for v in num.findall(line[:-len(lab)])]
        if lab == "FREQ":
            if len(vals) < 2:
                continue
            hour = int(round(vals[0])) % 24
            out["muf"][hour] = vals[1]
            out["hours"].setdefault(hour, {f: {} for f in fq})
            continue
        if hour is None or lab not in ("REL", "SNR"):
            continue
        tail = vals[-len(fq):] if len(vals) >= len(fq) else []
        for f, v in zip(fq, tail):
            out["hours"][hour].setdefault(f, {})[lab.lower()] = v
    return out


class Voacap:
    """voacapl as a separate program. `available()` says whether it is
    there; `predict()` returns {"available": True, "table": …} or
    {"available": False, "why": "prediction unavailable — …"}."""

    def __init__(self, exe: str | None = None, itshfbc: str | None = None,
                 runner: Callable[[list, str], tuple] | None = None):
        self.exe = exe or shutil.which("voacapl") or ""
        self.itshfbc = itshfbc or ""
        self.runner = runner

    def available(self) -> tuple[bool, str]:
        if self.runner is not None:
            return True, ""
        if not self.exe:
            return False, ("prediction unavailable — voacapl is not installed "
                           "(the ITS VOACAP program, run as its own process; "
                           "install it and its itshfbc data folder, then set "
                           "both paths)")
        if not self.itshfbc or not Path(self.itshfbc).is_dir():
            return False, ("prediction unavailable — voacapl is installed but "
                           "its itshfbc data folder is not set or not found")
        return True, ""

    def predict(self, tx: tuple, rx: tuple, *, year: int, month: int,
                ssn: float, freqs_mhz, hours=range(1, 25), **deck_kw) -> dict:
        ok, why = self.available()
        if not ok:
            return {"available": False, "why": why}
        deck = voacap_deck(tx, rx, year=year, month=month, ssn=ssn,
                           freqs_mhz=freqs_mhz, hours=hours, **deck_kw)
        run_dir = Path(tempfile.mkdtemp(prefix="voacap_"))
        (run_dir / "voacapx.dat").write_text(deck, encoding="ascii")
        args = [self.exe, f"--run-dir={run_dir}", self.itshfbc,
                "voacapx.dat", "voacapx.out"]
        try:
            if self.runner is not None:
                rc, text = self.runner(args, str(run_dir))
            else:
                pr = subprocess.run(args, capture_output=True, text=True,
                                    timeout=120, cwd=str(run_dir))
                rc, text = pr.returncode, (pr.stdout or "") + (pr.stderr or "")
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"available": False, "why": f"prediction unavailable — "
                                               f"voacapl could not run: {exc}"}
        outp = run_dir / "voacapx.out"
        body = outp.read_text(errors="replace") if outp.exists() else str(text or "")
        table = parse_voacap_output(body, freqs_mhz)
        if not table["hours"]:
            tail = str(text or "")[-300:]
            return {"available": False,
                    "why": ("prediction unavailable — voacapl ran (code "
                            f"{rc}) but its output could not be read"
                            + (f": …{tail!r}" if tail else ""))}
        return {"available": True, "table": table, "deck": deck,
                "tier": provenance.tier_for("voacap"),
                "what": "VOACAP prediction (a model): REL is the fraction of "
                        "days of the month the circuit meets the required SNR"}


def band_freq_mhz(band: str) -> float:
    """The WSPR dial frequency of a band (what is predicted for it)."""
    return float(_wspr.WSPR_DIAL_MHZ.get(band, 0.0))
