# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Maidenhead grid locators <-> latitude and longitude (plan §4.J).

WSPR spots carry the transmitter's location as a Maidenhead locator (FN42,
FM18lr); KiwiSDRs publish theirs the same way. The HF-now map needs both as
coordinates, and the reach arcs need them as points.

    pair 1  field       A–R   20° of longitude x 10° of latitude
    pair 2  square      0–9    2° x 1°
    pair 3  subsquare   a–x    5' x 2.5'
    pair 4  extended    0–9    30" x 15"
    pair 5  ext. sub    a–x    1.25" x 0.625"

A locator names a RECTANGLE, not a point: `to_latlon` returns its centre by
default (or its south-west corner), `bounds` the rectangle, and the
uncertainty is half its size — a 4-character locator is about ±110 km by
±55 km, which is why a WSPR distance is quoted to the kilometre and trusted
to the tens. Case-insensitive on the way in; written in the conventional
case (FM18lr) on the way out. Pure standard library.
"""

from __future__ import annotations

import math
import re

#: (longitude size, latitude size, alphabet) per character pair.
_LEVELS = (
    (20.0, 10.0, "ABCDEFGHIJKLMNOPQR"),
    (2.0, 1.0, "0123456789"),
    (2.0 / 24.0, 1.0 / 24.0, "abcdefghijklmnopqrstuvwx"),
    (2.0 / 240.0, 1.0 / 240.0, "0123456789"),
    (2.0 / 5760.0, 1.0 / 5760.0, "abcdefghijklmnopqrstuvwx"),
)

_GRID = re.compile(r"^[A-Ra-r]{2}(?:[0-9]{2}(?:[A-Xa-x]{2}(?:[0-9]{2}(?:[A-Xa-x]{2})?)?)?)?$")


def is_grid(s: str) -> bool:
    """A valid 2, 4, 6, 8 or 10 character locator."""
    return bool(_GRID.match(str(s or "").strip()))


def normalize(grid: str) -> str:
    """Conventional case — field upper, subsquares lower: 'fm18LR' ->
    'FM18lr'. Refuses what is not a locator, in words."""
    g = str(grid or "").strip()
    if not is_grid(g):
        raise ValueError(f"{grid!r} is not a Maidenhead locator (like FN42 or "
                         "FM18lr)")
    return "".join(g[i:i + 2].upper() if i == 0 else g[i:i + 2].lower()
                   for i in range(0, len(g), 2))


def bounds(grid: str) -> tuple[float, float, float, float]:
    """(lat_min, lon_min, lat_max, lon_max) of the locator's rectangle."""
    g = normalize(grid)
    lon, lat = -180.0, -90.0
    w = h = 0.0
    for k in range(len(g) // 2):
        w, h, alpha = _LEVELS[k]
        lon += alpha.index(g[2 * k]) * w
        lat += alpha.index(g[2 * k + 1]) * h
    return lat, lon, lat + h, lon + w


def to_latlon(grid: str, center: bool = True) -> tuple[float, float]:
    """The locator's centre (or south-west corner) as (lat, lon)."""
    la0, lo0, la1, lo1 = bounds(grid)
    if center:
        return 0.5 * (la0 + la1), 0.5 * (lo0 + lo1)
    return la0, lo0


def from_latlon(lat: float, lon: float, precision: int = 6) -> str:
    """(lat, lon) -> a locator of `precision` characters (2, 4, 6, 8, 10)."""
    p = int(precision)
    if p not in (2, 4, 6, 8, 10):
        raise ValueError("a locator has 2, 4, 6, 8 or 10 characters")
    la, lo = float(lat), float(lon)
    if not (-90.0 <= la <= 90.0) or not (-180.0 <= lo <= 180.0):
        raise ValueError(f"({lat}, {lon}) is not on the Earth")
    x = lo + 180.0
    y = la + 90.0
    out = []
    for k in range(p // 2):
        w, h, alpha = _LEVELS[k]
        i = min(int(math.floor(x / w + 1e-12)), len(alpha) - 1)
        j = min(int(math.floor(y / h + 1e-12)), len(alpha) - 1)
        out.append(alpha[i] + alpha[j])
        x -= i * w
        y -= j * h
    return "".join(out)


def uncertainty_km(grid: str) -> float:
    """Half the diagonal of the locator's rectangle, in km — how far the
    centre can be from where the station actually is."""
    la0, lo0, la1, lo1 = bounds(grid)
    lat_c = math.radians(0.5 * (la0 + la1))
    dy = (la1 - la0) * 111.32
    dx = (lo1 - lo0) * 111.32 * math.cos(lat_c)
    return 0.5 * math.hypot(dx, dy)
