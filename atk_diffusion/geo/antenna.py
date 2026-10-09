# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Antenna patterns for the reach map (plan E5: "the antenna pattern from
ATK's own antenna designer").

A pattern answers one question: the gain, in dBi, toward a direction given
as azimuth and elevation in the ANTENNA's frame — azimuth 0 is boresight
(where it is pointed), elevation 0 the horizon. The reach map rotates each
cell's bearing into that frame with the antenna's pointing.

* `Isotropic` — the reference.
* `Dipole` — the half-wave dipole's exact far-field pattern
  cos(pi/2 cos t) / sin t, vertical (omni in azimuth) or horizontal
  (broadside at azimuth 0, nulls off the ends), 2.15 dBi peak.
* `Directional` — a single-lobe beam from its peak gain and 3 dB
  beamwidths, the 3GPP TR 36.814 sector model: 12 (angle/beamwidth)^2
  dB down, limited by the front-to-back ratio. Good for Yagis, panels,
  dishes; it does not draw sidelobes.
* `TablePattern` — measured or modelled cuts (azimuth and elevation, dB
  relative to the peak) or a full azimuth x elevation table.

ATK'S DESIGNER. `from_atk(result)` reads a `Result` from ATK's
`atk/core/rf/antenna.py` (Bill's antenna calculator) by its fields — label,
value, parts, refusal — without importing ATK: the gain from the "Gain" /
"Expected gain" / "Estimated gain" part (dBd is converted to dBi), the
beamwidth from "Half-power beamwidth" when the design states one. A design
the calculator REFUSED is refused here with the calculator's own sentence.
Where the designer states no gain (a patch, a cantenna) the caller must
supply one — this module does not invent a number the designer would not
give. The pattern's shape is a model of the antenna TYPE, not a simulation
of the cut list: the note on every converted pattern says so.
"""

from __future__ import annotations

import math
import re

import numpy as np

DIPOLE_GAIN_DBI = 2.15
FLOOR_DB = -40.0                     # deepest null a pattern reports


def _wrap(az) -> np.ndarray:
    return (np.asarray(az, dtype=np.float64) + 180.0) % 360.0 - 180.0


class AntennaPattern:
    kind = "pattern"
    note = ""

    def gain_dbi(self, az_deg, el_deg) -> np.ndarray:
        raise NotImplementedError

    @property
    def peak_gain_dbi(self) -> float:
        az = np.linspace(-180, 180, 361)
        el = np.linspace(-90, 90, 181)
        A, E = np.meshgrid(az, el)
        return float(np.max(self.gain_dbi(A, E)))

    def describe(self) -> str:
        return self.kind

    def to_json(self) -> dict:
        raise NotImplementedError


class Isotropic(AntennaPattern):
    kind = "isotropic"

    def __init__(self, gain_dbi: float = 0.0):
        self.gain = float(gain_dbi)

    def gain_dbi(self, az_deg, el_deg) -> np.ndarray:
        return np.full(np.broadcast(np.asarray(az_deg), np.asarray(el_deg)).shape,
                       self.gain)

    @property
    def peak_gain_dbi(self) -> float:
        return self.gain

    def describe(self) -> str:
        return f"isotropic, {self.gain:+.1f} dBi"

    def to_json(self) -> dict:
        return {"type": "isotropic", "gain_dbi": self.gain}


class Dipole(AntennaPattern):
    """Half-wave dipole. `gain_dbi` sets the peak (2.15 for a real one; a
    ground plane or J-pole is modelled with this shape and its own gain)."""
    kind = "dipole"

    def __init__(self, orientation: str = "vertical",
                 gain_dbi: float = DIPOLE_GAIN_DBI, note: str = ""):
        o = orientation.lower()
        if o not in ("vertical", "horizontal"):
            raise ValueError("a dipole is 'vertical' or 'horizontal'")
        self.orientation = o
        self.gain = float(gain_dbi)
        self.note = note

    def gain_dbi(self, az_deg, el_deg) -> np.ndarray:
        az = np.radians(np.asarray(az_deg, dtype=np.float64))
        el = np.radians(np.asarray(el_deg, dtype=np.float64))
        if self.orientation == "vertical":
            cos_t = np.sin(el)                      # angle from the vertical axis
        else:                                       # axis across boresight
            cos_t = np.sin(az) * np.cos(el)
        sin_t = np.sqrt(np.clip(1.0 - cos_t ** 2, 0.0, 1.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.abs(np.cos(0.5 * math.pi * cos_t)) / sin_t
            g = 20.0 * np.log10(np.where(sin_t > 1e-9, f, 0.0))
        return np.maximum(self.gain + np.nan_to_num(g, nan=FLOOR_DB,
                                                    neginf=FLOOR_DB),
                          self.gain + FLOOR_DB)

    @property
    def peak_gain_dbi(self) -> float:
        return self.gain

    def describe(self) -> str:
        return f"{self.orientation} half-wave dipole pattern, {self.gain:+.2f} dBi"

    def to_json(self) -> dict:
        return {"type": "dipole", "orientation": self.orientation,
                "gain_dbi": self.gain, "note": self.note}


class Directional(AntennaPattern):
    """One main lobe: G = peak - min(12 (az/az_bw)^2 + 12 (el/el_bw)^2,
    front_to_back) dB (3GPP TR 36.814), beamwidths full 3 dB widths."""
    kind = "directional"

    def __init__(self, gain_dbi: float, az_beamwidth_deg: float,
                 el_beamwidth_deg: float | None = None,
                 front_to_back_db: float = 20.0, note: str = ""):
        if az_beamwidth_deg <= 0:
            raise ValueError("a beamwidth must be positive")
        self.gain = float(gain_dbi)
        self.az_bw = float(az_beamwidth_deg)
        self.el_bw = float(el_beamwidth_deg or az_beamwidth_deg)
        self.fb = abs(float(front_to_back_db))
        self.note = note

    def gain_dbi(self, az_deg, el_deg) -> np.ndarray:
        az = _wrap(az_deg)
        el = np.asarray(el_deg, dtype=np.float64)
        a_h = np.minimum(12.0 * (az / self.az_bw) ** 2, self.fb)
        a_v = np.minimum(12.0 * (el / self.el_bw) ** 2, self.fb)
        return self.gain - np.minimum(a_h + a_v, self.fb)

    @property
    def peak_gain_dbi(self) -> float:
        return self.gain

    def describe(self) -> str:
        return (f"directional, {self.gain:+.1f} dBi, {self.az_bw:.0f}° x "
                f"{self.el_bw:.0f}° beam, {self.fb:.0f} dB front-to-back")

    def to_json(self) -> dict:
        return {"type": "directional", "gain_dbi": self.gain,
                "az_beamwidth_deg": self.az_bw, "el_beamwidth_deg": self.el_bw,
                "front_to_back_db": self.fb, "note": self.note}


class TablePattern(AntennaPattern):
    """From tables. Either two cuts — `az_deg`/`az_db` and `el_deg`/`el_db`,
    each in dB RELATIVE to the peak (0 at the peak, negative elsewhere),
    combined as peak + H(az) + V(el) — or a full table `table_db[el, az]`
    of absolute gains (dBi) on the `el_deg` x `az_deg` axes."""
    kind = "table"

    def __init__(self, peak_gain_dbi: float = 0.0, az_deg=None, az_db=None,
                 el_deg=None, el_db=None, table_db=None, note: str = ""):
        self.gain = float(peak_gain_dbi)
        self.note = note
        self.table = None
        if table_db is not None:
            self.az = np.asarray(az_deg, dtype=np.float64)
            self.el = np.asarray(el_deg, dtype=np.float64)
            self.table = np.asarray(table_db, dtype=np.float64)
            if self.table.shape != (self.el.size, self.az.size):
                raise ValueError("table_db must be (len(el_deg), len(az_deg))")
            self.gain = float(self.table.max())
        else:
            self.az = np.asarray(az_deg if az_deg is not None else [0.0],
                                 dtype=np.float64)
            self.az_db = np.asarray(az_db if az_db is not None else [0.0],
                                    dtype=np.float64)
            self.el = np.asarray(el_deg if el_deg is not None else [0.0],
                                 dtype=np.float64)
            self.el_db = np.asarray(el_db if el_db is not None else [0.0],
                                    dtype=np.float64)
            if self.az.size != self.az_db.size or self.el.size != self.el_db.size:
                raise ValueError("each cut needs as many gains as angles")
            if np.max(self.az_db) > 1e-6 or np.max(self.el_db) > 1e-6:
                # cuts given as absolute gains: make them relative
                self.az_db = self.az_db - np.max(self.az_db)
                self.el_db = self.el_db - np.max(self.el_db)

    @staticmethod
    def _interp_az(az_q, az, vals):
        order = np.argsort(np.mod(az, 360.0))
        a = np.mod(az, 360.0)[order]
        v = vals[order]
        a = np.concatenate([a - 360.0, a, a + 360.0])
        v = np.concatenate([v, v, v])
        return np.interp(np.mod(np.asarray(az_q, dtype=np.float64), 360.0), a, v)

    def gain_dbi(self, az_deg, el_deg) -> np.ndarray:
        az = np.asarray(az_deg, dtype=np.float64)
        el = np.asarray(el_deg, dtype=np.float64)
        shape = np.broadcast(az, el).shape
        az = np.broadcast_to(az, shape).ravel()
        el = np.broadcast_to(el, shape).ravel()
        if self.table is None:
            h = self._interp_az(az, self.az, self.az_db) if self.az.size > 1 \
                else np.zeros_like(az)
            if self.el.size > 1:
                o = np.argsort(self.el)
                v = np.interp(el, self.el[o], self.el_db[o])
            else:
                v = np.zeros_like(el)
            return (self.gain + np.maximum(h + v, FLOOR_DB)).reshape(shape)
        # bilinear on (el, az), azimuth periodic
        rows = np.empty((self.el.size, az.size))
        for i in range(self.el.size):
            rows[i] = self._interp_az(az, self.az, self.table[i])
        o = np.argsort(self.el)
        out = np.empty(az.size)
        els = self.el[o]
        rows = rows[o]
        for j in range(az.size):
            out[j] = np.interp(el[j], els, rows[:, j])
        return out.reshape(shape)

    def describe(self) -> str:
        return f"pattern table, {self.gain:+.1f} dBi peak"

    def to_json(self) -> dict:
        d = {"type": "table", "peak_gain_dbi": self.gain, "note": self.note,
             "az_deg": self.az.tolist(), "el_deg": self.el.tolist()}
        if self.table is not None:
            d["table_db"] = self.table.tolist()
        else:
            d["az_db"] = self.az_db.tolist()
            d["el_db"] = self.el_db.tolist()
        return d


def from_json(d: dict) -> AntennaPattern:
    t = d.get("type")
    if t == "isotropic":
        return Isotropic(d.get("gain_dbi", 0.0))
    if t == "dipole":
        return Dipole(d.get("orientation", "vertical"),
                      d.get("gain_dbi", DIPOLE_GAIN_DBI), d.get("note", ""))
    if t == "directional":
        return Directional(d["gain_dbi"], d["az_beamwidth_deg"],
                           d.get("el_beamwidth_deg"),
                           d.get("front_to_back_db", 20.0), d.get("note", ""))
    if t == "table":
        if "table_db" in d:
            return TablePattern(d.get("peak_gain_dbi", 0.0), d["az_deg"], None,
                                d["el_deg"], None, d["table_db"], d.get("note", ""))
        return TablePattern(d.get("peak_gain_dbi", 0.0), d["az_deg"], d["az_db"],
                            d["el_deg"], d["el_db"], note=d.get("note", ""))
    raise ValueError(f"unknown antenna pattern type {t!r}")


def beamwidth_from_gain(gain_dbi: float, efficiency: float = 0.7) -> float:
    """Kraus: G ~ 41253 eta / (bw_az bw_el); equal beamwidths assumed."""
    g = 10.0 ** (float(gain_dbi) / 10.0)
    return float(min(360.0, math.sqrt(41253.0 * efficiency / max(g, 1e-6))))


# ---------------------------------------------------------------------------
# ATK's antenna designer
# ---------------------------------------------------------------------------
_OMNI = ("dipole", "inverted v", "ground plane", "j-pole", "slim jim",
         "collinear", "end-fed", "discone")
_DIRECTIONAL = {"yagi": 15.0, "moxon": 20.0, "cubical quad": 15.0,
                "helix": 15.0, "biquad": 20.0, "parabolic dish": 25.0,
                "corner reflector": 20.0, "patch": 15.0, "cantenna": 15.0}


def _db_of(value) -> float | None:
    if value is None:
        return None
    if hasattr(value, "db"):
        return float(value.db)
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"[-+]?\d+(\.\d+)?", str(value))
    return float(m.group(0)) if m else None


def from_atk(result, *, orientation: str = "vertical",
             gain_dbi: float | None = None,
             beamwidth_deg: float | None = None) -> AntennaPattern:
    """A pattern from a Result of ATK's antenna designer. `gain_dbi` and
    `beamwidth_deg` override or supply what the design does not state."""
    refusal = getattr(result, "refusal", "") or ""
    label = str(getattr(result, "label", "") or "")
    if refusal:
        raise ValueError(f"ATK's antenna designer refused {label!r}: {refusal}")
    low = label.lower()
    parts = list(getattr(result, "parts", ()) or ())
    stated = None
    beam = None
    for p in parts:
        try:
            name, value, note = p
        except (TypeError, ValueError):
            continue
        n = str(name).lower()
        if n in ("gain", "expected gain", "estimated gain") and stated is None:
            g = _db_of(value)
            if g is not None:
                if "dbd" in str(note).lower():
                    g += DIPOLE_GAIN_DBI
                stated = g
        elif "half-power beamwidth" in n:
            beam = _db_of(value)
    if stated is None and any(k in low for k in ("yagi", "helix")):
        stated = _db_of(getattr(result, "value", None))
    g = gain_dbi if gain_dbi is not None else stated
    bw = beamwidth_deg if beamwidth_deg is not None else beam
    note = (f"from ATK's antenna designer: {label}. The shape is a model of "
            "the antenna type, not a simulation of this cut list")
    if any(k in low for k in _OMNI):
        if "dipole" in low or "inverted v" in low:
            return Dipole(orientation, DIPOLE_GAIN_DBI if g is None else g,
                          note=note + f"; mounted {orientation}")
        return Dipole("vertical", DIPOLE_GAIN_DBI if g is None else g,
                      note=note + "; a vertical half-wave shape is assumed")
    for key, fb in _DIRECTIONAL.items():
        if key in low:
            if g is None:
                raise ValueError(f"ATK's designer states no gain for {label!r}; "
                                 "give gain_dbi (measured or from the data "
                                 "sheet) — a number is not invented here")
            if bw is None:
                bw = beamwidth_from_gain(g)
                note += f"; beamwidth {bw:.0f}° estimated from the gain (Kraus)"
            return Directional(g, bw, bw, fb, note=note)
    raise ValueError(f"no pattern model for {label!r} — describe it as a "
                     "TablePattern or a Directional pattern")
