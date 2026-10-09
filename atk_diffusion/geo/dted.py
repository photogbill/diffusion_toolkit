# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""DTED levels 0, 1 and 2, read natively per MIL-PRF-89020B (plan E5).

Bill, 2026-10-08: *"if I have the DTED level 1 loaded, we can use space loss
and other propagation models to show on the map the likely reach given the
radio, power levels, antenna patterns."* This is the "DTED level 1 loaded"
half: a reader for the files themselves, a mosaic of tiles, and a bilinear
`elevation(lat, lon)` that the terrain profiles and the reach map stand on.

WHY NATIVE AND NOT GDAL. The plan said "DTED1 read through GDAL". GDAL is
not in ATK's core environment and is a large binary dependency for a format
that is a fixed 3428-byte header and fixed-length records. So the file is
read here, field by field, from the specification:

    UHL   80 bytes   'UHL1', origin (SW post) longitude and latitude
                     DDDMMSSH, intervals in tenths of arc seconds, counts
    DSI   648 bytes  product level 'DTED0/1/2', datums, corners, counts
    ACC   2700 bytes accuracy (absolute / relative, horizontal / vertical)
    data  one record per LONGITUDE line, west to east:
            0xAA sentinel, 3-byte block count, 2-byte longitude count,
            2-byte latitude count, then n_lat elevations SOUTH TO NORTH,
            each 2 bytes big-endian SIGNED MAGNITUDE (bit 15 is the sign,
            bits 0-14 the metres), then a 4-byte checksum: the sum of every
            byte of the record before it.
            Voids are -32767 (0xFFFF).

A GDAL-readable product is still the output (`products.write_geotiff`).

WHAT IT CHECKS AND SAYS. Every record's sentinel and checksum is verified;
a bad record is NAMED (its longitude column) rather than silently used or
silently dropped — the tile still loads and `problems` says what is wrong.
Some producers wrote negative heights in two's complement instead of signed
magnitude; like GDAL, a "negative" value below -16000 that is not the void
code is re-read as two's complement and counted, so a Dead Sea tile is not
read as a 32 km trench and the count says the file is non-conformant.

LIMITS. Heights are metres above mean sea level (EGM96 for DTED1/2 — the
DSI's vertical datum is reported). Bilinear interpolation between posts:
at level 1 the posts are 3 arc seconds (~90 m) apart, so terrain smaller
than that is not in the data at all, whatever the interpolation does. A
point whose interpolation stencil touches a void is NaN (or, with
`void="skip"`, the valid neighbours renormalised), and the profile code
counts it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

UHL_LEN = 80
DSI_LEN = 648
ACC_LEN = 2700
HEADER_LEN = UHL_LEN + DSI_LEN + ACC_LEN        # 3428
SENTINEL = 0xAA
VOID = -32767
SUFFIXES = (".dt0", ".dt1", ".dt2")


class DtedError(ValueError):
    """The file is not a DTED tile, or not one this reader can trust."""


# ---------------------------------------------------------------------------
# Coordinates in the header
# ---------------------------------------------------------------------------
def _dms(s: str, deg_digits: int) -> float:
    """'DDDMMSSH' / 'DDMMSSH' / 'DDDMMSS.SH' -> signed degrees."""
    s = s.strip()
    if not s:
        raise DtedError("an empty coordinate field")
    hemi = s[-1].upper()
    body = s[:-1]
    try:
        d = int(body[:deg_digits])
        m = int(body[deg_digits:deg_digits + 2])
        sec = float(body[deg_digits + 2:])
    except ValueError:
        raise DtedError(f"{s!r} is not a DTED coordinate") from None
    v = d + m / 60.0 + sec / 3600.0
    if hemi in ("S", "W"):
        v = -v
    elif hemi not in ("N", "E"):
        raise DtedError(f"{s!r} has no hemisphere letter")
    return v


def _fmt_dms(v: float, deg_digits: int, hemis: str, decimals: int = 0) -> str:
    h = hemis[0] if v >= 0 else hemis[1]
    a = abs(v)
    total = round(a * 3600.0 * 10 ** decimals)
    whole, frac = divmod(total, 10 ** decimals)
    d, rem = divmod(int(whole), 3600)
    m, s = divmod(rem, 60)
    out = f"{d:0{deg_digits}d}{m:02d}{s:02d}"
    if decimals:
        out += f".{int(frac):0{decimals}d}"
    return out + h


def _int_field(b: bytes, what: str) -> int:
    try:
        return int(b.decode("ascii").strip())
    except ValueError:
        raise DtedError(f"the {what} field {b!r} is not a number") from None


# ---------------------------------------------------------------------------
@dataclass
class DtedTile:
    """One DTED cell. `elev` is (n_lat, n_lon) int16 metres with row 0 the
    SOUTH edge and column 0 the WEST edge (the file's own order, transposed
    into rows); voids are `VOID`."""
    path: str
    level: int | None
    lat0: float                 # latitude of the south-west post (origin)
    lon0: float
    dlat_s: float               # post spacing, arc seconds
    dlon_s: float
    n_lat: int                  # posts per longitude line (south -> north)
    n_lon: int                  # longitude lines (west -> east)
    elev: np.ndarray
    vertical_datum: str = ""
    horizontal_datum: str = ""
    abs_vertical_accuracy_m: float | None = None
    security: str = ""
    problems: list = field(default_factory=list)
    checksum_errors: list = field(default_factory=list)
    twos_complement_values: int = 0

    @property
    def lat1(self) -> float:
        return self.lat0 + (self.n_lat - 1) * self.dlat_s / 3600.0

    @property
    def lon1(self) -> float:
        return self.lon0 + (self.n_lon - 1) * self.dlon_s / 3600.0

    @property
    def voids(self) -> int:
        return int(np.count_nonzero(self.elev == VOID))

    def describe(self) -> str:
        lvl = f"DTED level {self.level}" if self.level is not None else "DTED"
        valid = self.elev[self.elev != VOID]
        rng = (f"{int(valid.min())}..{int(valid.max())} m" if valid.size
               else "no valid heights")
        s = (f"{lvl} {self.lat0:+.0f} {self.lon0:+.0f}: {self.n_lon} x "
             f"{self.n_lat} posts at {self.dlon_s:g}\" x {self.dlat_s:g}\", "
             f"{rng}")
        if self.voids:
            s += f", {self.voids} void posts"
        if self.problems:
            s += "; " + "; ".join(self.problems)
        return s

    def elevation(self, lat, lon, void: str = "nan") -> np.ndarray:
        """Bilinear height at points inside this tile (NaN outside)."""
        lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
        lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))
        r = (lat - self.lat0) * 3600.0 / self.dlat_s
        c = (lon - self.lon0) * 3600.0 / self.dlon_s
        eps = 1e-9
        inside = ((r >= -eps) & (r <= self.n_lat - 1 + eps)
                  & (c >= -eps) & (c <= self.n_lon - 1 + eps))
        r = np.clip(r, 0.0, self.n_lat - 1.0)
        c = np.clip(c, 0.0, self.n_lon - 1.0)
        r0 = np.clip(np.floor(r).astype(np.int64), 0, max(self.n_lat - 2, 0))
        c0 = np.clip(np.floor(c).astype(np.int64), 0, max(self.n_lon - 2, 0))
        r1 = np.minimum(r0 + 1, self.n_lat - 1)
        c1 = np.minimum(c0 + 1, self.n_lon - 1)
        fr = np.clip(r - r0, 0.0, 1.0)
        fc = np.clip(c - c0, 0.0, 1.0)
        corners = [(r0, c0, (1 - fr) * (1 - fc)), (r0, c1, (1 - fr) * fc),
                   (r1, c0, fr * (1 - fc)), (r1, c1, fr * fc)]
        num = np.zeros_like(lat)
        wsum = np.zeros_like(lat)
        anyvoid = np.zeros(lat.shape, dtype=bool)
        for rr, cc, w in corners:
            v = self.elev[rr, cc]
            ok = v != VOID
            # a stencil weight of zero does not make the point void
            anyvoid |= (~ok) & (w > 1e-12)
            num += np.where(ok, v * w, 0.0)
            wsum += np.where(ok, w, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = num / wsum
        if void == "nan":
            out = np.where(anyvoid, np.nan, out)
        elif void != "skip":
            raise ValueError("void must be 'nan' or 'skip'")
        out = np.where(wsum > 0, out, np.nan)
        return np.where(inside, out, np.nan)


# ---------------------------------------------------------------------------
def _decode_signed_magnitude(raw: np.ndarray) -> tuple[np.ndarray, int]:
    """uint16 big-endian words -> int16 heights, and how many values were
    found in two's complement (non-conformant files, as GDAL handles them)."""
    raw = raw.astype(np.int32)
    mag = raw & 0x7FFF
    neg = (raw & 0x8000) != 0
    v = np.where(neg, -mag, mag)
    tc = neg & (v < -16000) & (v != VOID)
    if np.any(tc):
        v = np.where(tc, raw - 65536, v)
    return v.astype(np.int16), int(np.count_nonzero(tc))


def encode_signed_magnitude(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.int32)
    if np.any(np.abs(v) > 32767):
        raise DtedError("a height outside +/-32767 m cannot be written")
    return np.where(v < 0, 0x8000 | (-v), v).astype(">u2")


def read_dted(path, strict: bool = False) -> DtedTile:
    """Read one DTED file. `strict=True` refuses a file with any bad record;
    otherwise the tile loads and `problems` names what is wrong."""
    p = Path(path)
    data = p.read_bytes()
    if len(data) < HEADER_LEN:
        raise DtedError(f"{p.name} is {len(data)} bytes — shorter than a "
                        f"DTED header ({HEADER_LEN})")
    uhl, dsi, acc = data[:80], data[80:728], data[728:HEADER_LEN]
    if uhl[:3] != b"UHL":
        raise DtedError(f"{p.name} does not start with a UHL record — not DTED")
    if dsi[:3] != b"DSI":
        raise DtedError(f"{p.name} has no DSI record where the standard puts it")
    if acc[:3] != b"ACC":
        raise DtedError(f"{p.name} has no ACC record where the standard puts it")
    lon0 = _dms(uhl[4:12].decode("ascii"), 3)
    lat0 = _dms(uhl[12:20].decode("ascii"), 3)
    dlon_s = _int_field(uhl[20:24], "longitude interval") / 10.0
    dlat_s = _int_field(uhl[24:28], "latitude interval") / 10.0
    acc_txt = uhl[28:32].decode("ascii", "replace").strip()
    try:
        abs_v = float(acc_txt)
    except ValueError:
        abs_v = None
    security = uhl[32:35].decode("ascii", "replace").strip()
    n_lon = _int_field(uhl[47:51], "number of longitude lines")
    n_lat = _int_field(uhl[51:55], "number of latitude points")
    if dlon_s <= 0 or dlat_s <= 0 or n_lon < 2 or n_lat < 2:
        raise DtedError(f"{p.name}: impossible spacing or counts in the UHL")
    lvl_txt = dsi[59:64].decode("ascii", "replace")
    level = int(lvl_txt[4]) if lvl_txt.startswith("DTED") and \
        lvl_txt[4:5].isdigit() else None
    vdatum = dsi[141:144].decode("ascii", "replace").strip()
    hdatum = dsi[144:149].decode("ascii", "replace").strip()
    problems: list[str] = []
    try:
        dsi_nlat = _int_field(dsi[281:285], "DSI latitude lines")
        dsi_nlon = _int_field(dsi[285:289], "DSI longitude lines")
        if (dsi_nlat, dsi_nlon) != (n_lat, n_lon):
            problems.append(f"the DSI says {dsi_nlon} x {dsi_nlat} posts, the "
                            f"UHL {n_lon} x {n_lat}; the UHL is used")
    except DtedError:
        problems.append("the DSI post counts are unreadable; the UHL is used")
    rec_len = 12 + 2 * n_lat
    body = np.frombuffer(data, dtype=np.uint8, offset=HEADER_LEN)
    have = body.size // rec_len
    if have < n_lon:
        msg = (f"{p.name} holds {have} of its {n_lon} longitude records — the "
               "file is truncated")
        if strict or have == 0:
            raise DtedError(msg)
        problems.append(msg)
    recs = body[:have * rec_len].reshape(have, rec_len)
    elev = np.full((n_lat, n_lon), VOID, dtype=np.int16)
    bad_sentinel = np.nonzero(recs[:, 0] != SENTINEL)[0]
    if bad_sentinel.size:
        msg = (f"{bad_sentinel.size} records lack the 0xAA sentinel "
               f"(columns {bad_sentinel[:5].tolist()}…)")
        if strict:
            raise DtedError(msg)
        problems.append(msg)
    lon_count = (recs[:, 4].astype(np.int64) << 8) | recs[:, 5]
    words = recs[:, 8:8 + 2 * n_lat].reshape(have, n_lat, 2)
    raw = (words[:, :, 0].astype(np.uint16) << 8) | words[:, :, 1]
    heights, n_tc = _decode_signed_magnitude(raw)
    ck_stored = ((recs[:, -4].astype(np.int64) << 24)
                 | (recs[:, -3].astype(np.int64) << 16)
                 | (recs[:, -2].astype(np.int64) << 8)
                 | recs[:, -1].astype(np.int64))
    ck_calc = recs[:, :-4].astype(np.int64).sum(axis=1)
    bad_ck = np.nonzero(ck_stored != ck_calc)[0]
    if bad_ck.size:
        msg = (f"{bad_ck.size} longitude records fail their checksum "
               f"(columns {bad_ck[:8].tolist()}) — those heights may be wrong")
        if strict:
            raise DtedError(msg)
        problems.append(msg)
    for i in range(have):
        if recs[i, 0] != SENTINEL:
            continue
        col = int(lon_count[i])
        if not 0 <= col < n_lon:
            problems.append(f"record {i} names longitude line {col}, outside "
                            "the tile; skipped")
            continue
        elev[:, col] = heights[i]
    if n_tc:
        problems.append(f"{n_tc} heights were written in two's complement, "
                        "not signed magnitude (a non-conformant producer); "
                        "read as two's complement")
    return DtedTile(path=str(p), level=level, lat0=lat0, lon0=lon0,
                    dlat_s=dlat_s, dlon_s=dlon_s, n_lat=n_lat, n_lon=n_lon,
                    elev=elev, vertical_datum=vdatum, horizontal_datum=hdatum,
                    abs_vertical_accuracy_m=abs_v, security=security,
                    problems=problems, checksum_errors=bad_ck.tolist(),
                    twos_complement_values=n_tc)


# ---------------------------------------------------------------------------
# Writing (synthetic terrain for experiments and tests)
# ---------------------------------------------------------------------------
def _pad(s: str, n: int) -> bytes:
    b = s.encode("ascii")
    if len(b) > n:
        raise DtedError(f"{s!r} does not fit a {n}-byte field")
    return b.ljust(n, b" ")


def header_bytes(lat0: float, lon0: float, dlat_s: float, dlon_s: float,
                 n_lat: int, n_lon: int, level: int = 1,
                 abs_vertical_accuracy_m: int | None = None) -> bytes:
    """UHL + DSI + ACC exactly as MIL-PRF-89020B lays them out."""
    lon_o = _fmt_dms(lon0, 3, "EW")
    lat_o = _fmt_dms(lat0, 3, "NS")
    acc = "NA" if abs_vertical_accuracy_m is None else f"{int(abs_vertical_accuracy_m):04d}"
    uhl = (b"UHL" + b"1" + _pad(lon_o, 8) + _pad(lat_o, 8)
           + _pad(f"{int(round(dlon_s * 10)):04d}", 4)
           + _pad(f"{int(round(dlat_s * 10)):04d}", 4)
           + _pad(acc, 4) + _pad("U", 3) + _pad("", 12)
           + _pad(f"{n_lon:04d}", 4) + _pad(f"{n_lat:04d}", 4)
           + b"0" + _pad("", 24))
    lat1 = lat0 + (n_lat - 1) * dlat_s / 3600.0
    lon1 = lon0 + (n_lon - 1) * dlon_s / 3600.0
    dsi = (b"DSI" + b"U" + _pad("", 2) + _pad("", 27) + _pad("", 26)
           + _pad(f"DTED{int(level)}", 5) + _pad("", 15) + _pad("", 8)
           + _pad("01", 2) + b"A" + _pad("0000", 4) + _pad("0000", 4)
           + _pad("0000", 4) + _pad("", 8) + _pad("", 16)
           + _pad("PRF89020B", 9) + _pad("00", 2) + _pad("0005", 4)
           + _pad("MSL", 3) + _pad("WGS84", 5) + _pad("", 10)
           + _pad("0000", 4) + _pad("", 22)
           + _pad(_fmt_dms(lat0, 2, "NS", 1), 9)
           + _pad(_fmt_dms(lon0, 3, "EW", 1), 10)
           + _pad(_fmt_dms(lat0, 2, "NS"), 7) + _pad(_fmt_dms(lon0, 3, "EW"), 8)
           + _pad(_fmt_dms(lat1, 2, "NS"), 7) + _pad(_fmt_dms(lon0, 3, "EW"), 8)
           + _pad(_fmt_dms(lat1, 2, "NS"), 7) + _pad(_fmt_dms(lon1, 3, "EW"), 8)
           + _pad(_fmt_dms(lat0, 2, "NS"), 7) + _pad(_fmt_dms(lon1, 3, "EW"), 8)
           + _pad("0000000.0", 9)
           + _pad(f"{int(round(dlat_s * 10)):04d}", 4)
           + _pad(f"{int(round(dlon_s * 10)):04d}", 4)
           + _pad(f"{n_lat:04d}", 4) + _pad(f"{n_lon:04d}", 4)
           + _pad("00", 2) + _pad("", 101) + _pad("", 100) + _pad("", 156))
    accr = (b"ACC" + _pad(acc, 4) + _pad(acc, 4) + _pad(acc, 4) + _pad(acc, 4)
            + _pad("", 4) + b" " + _pad("", 31) + _pad("00", 2)
            + _pad("", 9 * 284) + _pad("", 18) + _pad("", 69))
    assert len(uhl) == UHL_LEN and len(dsi) == DSI_LEN and len(accr) == ACC_LEN
    return uhl + dsi + accr


def record_bytes(column: int, heights) -> bytes:
    """One longitude record: sentinel, counts, heights south->north,
    checksum."""
    h = encode_signed_magnitude(heights)
    head = bytes([SENTINEL, (column >> 16) & 0xFF, (column >> 8) & 0xFF,
                  column & 0xFF, (column >> 8) & 0xFF, column & 0xFF, 0, 0])
    body = head + h.tobytes()
    ck = sum(body)
    return body + ck.to_bytes(4, "big")


def write_dted(path, elev, lat0: float, lon0: float, dlat_s: float,
               dlon_s: float, level: int = 1) -> Path:
    """Write a DTED file from `elev` (n_lat, n_lon), row 0 SOUTH, column 0
    WEST, metres, VOID for voids. For synthetic terrain in experiments and
    tests — real DTED comes from NGA."""
    e = np.asarray(elev)
    n_lat, n_lon = e.shape
    out = bytearray(header_bytes(lat0, lon0, dlat_s, dlon_s, n_lat, n_lon, level))
    for col in range(n_lon):
        out += record_bytes(col, e[:, col])
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(bytes(out))
    return p


# ---------------------------------------------------------------------------
# A mosaic of tiles
# ---------------------------------------------------------------------------
class DtedMosaic:
    """Tiles keyed by their south-west corner; `elevation(lat, lon)` finds
    the tile for each point. Shared edges (a tile's north row is the next
    tile's south row) resolve to whichever tile holds the point."""

    def __init__(self, tiles):
        self.tiles: dict[tuple[int, int], DtedTile] = {}
        self.skipped: list[str] = []
        for t in tiles:
            key = (int(math.floor(t.lat0 + 1e-9)), int(math.floor(t.lon0 + 1e-9)))
            self.tiles[key] = t

    @classmethod
    def from_folder(cls, folder, progress=None) -> "DtedMosaic":
        """Every .dt0/.dt1/.dt2 under `folder` (the usual w078/n38.dt1
        tree). Files that are not DTED are listed in `skipped`, not fatal."""
        tiles, skipped = [], []
        files = sorted(f for f in Path(folder).rglob("*")
                       if f.suffix.lower() in SUFFIXES and f.is_file())
        for i, f in enumerate(files):
            try:
                tiles.append(read_dted(f))
            except DtedError as e:
                skipped.append(f"{f.name}: {e}")
            if progress:
                progress(f"read {i + 1}/{len(files)} DTED tiles")
        if not tiles:
            raise DtedError(f"no readable DTED tiles under {folder}"
                            + (f" ({len(skipped)} unreadable)" if skipped else ""))
        m = cls(tiles)
        m.skipped = skipped
        return m

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        ts = list(self.tiles.values())
        return (min(t.lon0 for t in ts), min(t.lat0 for t in ts),
                max(t.lon1 for t in ts), max(t.lat1 for t in ts))

    def describe(self) -> str:
        lv = sorted({t.level for t in self.tiles.values() if t.level is not None})
        voids = sum(t.voids for t in self.tiles.values())
        probs = sum(len(t.problems) for t in self.tiles.values())
        s = (f"{len(self.tiles)} DTED tiles (level "
             f"{'/'.join(map(str, lv)) or '?'}) covering "
             f"{self.bounds[1]:.0f}..{self.bounds[3]:.0f} N, "
             f"{self.bounds[0]:.0f}..{self.bounds[2]:.0f} E")
        if voids:
            s += f"; {voids:,} void posts"
        if probs:
            s += f"; {probs} tiles report problems"
        return s

    def _key(self, lat: float, lon: float):
        ky, kx = int(math.floor(lat)), int(math.floor(lon))
        for dy in (0, -1):
            for dx in (0, -1):
                k = (ky + dy, kx + dx)
                t = self.tiles.get(k)
                if t is not None and t.lat0 - 1e-9 <= lat <= t.lat1 + 1e-9 \
                        and t.lon0 - 1e-9 <= lon <= t.lon1 + 1e-9:
                    return k
        return None

    def elevation(self, lat, lon, void: str = "nan") -> np.ndarray:
        """Bilinear metres above MSL; NaN where no tile covers the point or
        the stencil touches a void (`void="skip"` renormalises instead)."""
        lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
        lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))
        shape = np.broadcast(lat, lon).shape
        lat = np.broadcast_to(lat, shape).ravel()
        lon = np.broadcast_to(lon, shape).ravel()
        out = np.full(lat.shape, np.nan)
        ky = np.floor(lat).astype(np.int64)
        kx = np.floor(lon).astype(np.int64)
        # the common case: the point's floor key has a tile
        for key in set(zip(ky.tolist(), kx.tolist())):
            t = self.tiles.get(key)
            sel = (ky == key[0]) & (kx == key[1])
            if t is not None:
                out[sel] = t.elevation(lat[sel], lon[sel], void=void)
        # points on a north/east edge with no tile beyond it
        edge = (lat == np.floor(lat)) | (lon == np.floor(lon))
        miss = np.nonzero(np.isnan(out) & edge)[0]
        for i in miss:
            k = self._key(lat[i], lon[i])
            if k is not None and k != (ky[i], kx[i]):
                out[i] = self.tiles[k].elevation(lat[i], lon[i], void=void)[0]
        return out.reshape(shape)

    __call__ = elevation
