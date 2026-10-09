# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Products: results live outside the install, in open formats (plan §3.7, D11).

Bill, 2026-10-08: *"as long as we can save results outside of the install
folder and import them into new installs."* Everything the map tracks make —
coverage rasters, radio maps, emitter tracks, position posteriors, the
fingerprint library — is a PRODUCT under `rf_data\\products\\<kind>\\`, never
in ATK or this repo, so a new install that points at `rf_data\\` has them.

    products\\coverage\\<run>\\    predicted reach (E5): GeoTIFF + parameters
    products\\radiomaps\\<run>\\   measured / learned radio maps (E2)
    products\\emitters\\           fingerprint library and the RF social graph (C)
    products\\tracks\\<run>\\      DF posteriors, synthetic-aperture solutions (E1, E4)
    products\\position\\<run>\\    where-am-I posteriors (E3)
    products\\hf\\                 HF band-openness (J)

Every run folder carries `manifest.json` — the parameters, the tier, the
model card when a learned model made it, and the SHA-256 of every file — and
every file is recorded in the rf_data write log the moment it is written, so
a product that changed after it was made is NAMED, not trusted.

WHY THESE FORMATS. Rasters are GeoTIFF in EPSG:4326 with the real GeoTIFF
keys (GeoKeyDirectory, ModelPixelScale, ModelTiepoint) so GDAL, QGIS,
GeoServer and ATK's Leaflet map read them without this code; the tags travel
twice — as GDAL_METADATA XML (what GDAL and GeoServer show) and as a JSON
ImageDescription (an exact round trip for this toolkit). Vectors are GeoJSON
(RFC 7946: longitude first) whose features all carry `atk:tier`.

THE COG CLAIM, STATED EXACTLY. With `cog=True` a raster is tiled, deflate-
compressed and carries internal overviews (reduced-resolution IFDs, the way
GDAL stores them), which is what makes a GeoTIFF fast to serve. tifffile —
the default writer — puts each overview's IFD beside its own data; a strict
Cloud-Optimized GeoTIFF wants every IFD ahead of the data. ATK's map and
GeoServer read either. The built-in writer (`engine="builtin"`, also the
fallback when tifffile is missing — `capabilities` promises one) writes the
strict layout. Neither engine writes GDAL's optional "ghost" header;
`gdal_translate -of COG` adds it if some server insists.

LIMITS. One band per file (a product with several layers writes several
files — which is what the map's toggles want anyway). EPSG:4326 only; a
raster in another CRS is refused on reading, in words, rather than mislaid.
No antimeridian crossing (west < east). Distances use a spherical earth
(R = 6 371 008.8 m): under 0.5 % in distance, far below every propagation
and DF uncertainty these products carry.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import struct
import time
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from atk_diffusion import provenance as _prov

CRS = "EPSG:4326"
EARTH_RADIUS_M = 6_371_008.8
FORMAT_VERSION = 1
MANIFEST = "manifest.json"

#: Every product kind (plan §3.7) and the ones that are one folder per run.
PRODUCT_KINDS = ("coverage", "radiomaps", "emitters", "tracks", "position", "hf")
RUN_KINDS = ("coverage", "radiomaps", "tracks", "position")

# GeoTIFF / GDAL tag codes
_TAG_DESCRIPTION = 270
_TAG_PIXEL_SCALE = 33550
_TAG_TIEPOINT = 33922
_TAG_GEOKEYS = 34735
_TAG_GEO_DOUBLE = 34736
_TAG_GEO_ASCII = 34737
_TAG_GDAL_METADATA = 42112
_TAG_GDAL_NODATA = 42113

#: GeoKeyDirectory for EPSG:4326, pixel-is-area, as GDAL itself writes it.
_GEO_ASCII = "WGS 84|"
_GEOKEYS_4326 = (1, 1, 0, 5,
                 1024, 0, 1, 2,                     # GTModelType = geographic
                 1025, 0, 1, 1,                     # GTRasterType = pixel is area
                 2048, 0, 1, 4326,                  # GeographicType = WGS 84
                 2049, _TAG_GEO_ASCII, len(_GEO_ASCII), 0,   # citation
                 2054, 0, 1, 9102)                  # angular unit = degree


class ProductError(ValueError):
    """A product could not be written or read. The message says why."""


# ---------------------------------------------------------------------------
# The grid every raster product sits on
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GeoGrid:
    """A north-up raster in EPSG:4326. Row 0 is the NORTH edge (the way a
    GeoTIFF stores it); `lats()` are pixel centres, north to south; `lons()`
    west to east. Pixels are areas: the bounds are the outer edges."""

    west: float
    south: float
    east: float
    north: float
    width: int
    height: int

    def __post_init__(self):
        if not (self.west < self.east and self.south < self.north):
            raise ProductError(f"bounds {self.bounds} are not west<east, "
                               "south<north (antimeridian crossing is not "
                               "supported)")
        if not (-90.0 <= self.south and self.north <= 90.0):
            raise ProductError(f"latitudes {self.south}..{self.north} are off "
                               "the earth")
        if int(self.width) < 1 or int(self.height) < 1:
            raise ProductError("a grid needs at least one pixel")

    # -- geometry --------------------------------------------------------------
    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (float(self.west), float(self.south), float(self.east),
                float(self.north))

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.height), int(self.width))

    @property
    def dlon(self) -> float:
        return (self.east - self.west) / self.width

    @property
    def dlat(self) -> float:
        return (self.north - self.south) / self.height

    def lons(self) -> np.ndarray:
        return self.west + (np.arange(self.width) + 0.5) * self.dlon

    def lats(self) -> np.ndarray:
        return self.north - (np.arange(self.height) + 0.5) * self.dlat

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """(LAT, LON), each (height, width)."""
        lon, lat = np.meshgrid(self.lons(), self.lats())
        return lat, lon

    @property
    def center(self) -> tuple[float, float]:
        return (0.5 * (self.south + self.north), 0.5 * (self.west + self.east))

    def cell_size_m(self) -> tuple[float, float]:
        """(east-west, north-south) size of a pixel in metres at the centre."""
        k = math.pi / 180.0 * EARTH_RADIUS_M
        return (self.dlon * k * math.cos(math.radians(self.center[0])),
                self.dlat * k)

    def cell_area_m2(self) -> np.ndarray:
        """Area of each pixel, (height, 1) — exact on the sphere, so a
        probability per pixel can be turned into a density per km²."""
        edges = self.north - np.arange(self.height + 1) * self.dlat
        s = np.sin(np.radians(edges))
        band = (s[:-1] - s[1:]) * EARTH_RADIUS_M ** 2 * math.radians(self.dlon)
        return np.abs(band)[:, None]

    def rowcol(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        """Fractional (row, col) of pixel CENTRES: (0, 0) is the NW pixel."""
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        return ((self.north - lat) / self.dlat - 0.5,
                (lon - self.west) / self.dlon - 0.5)

    def contains(self, lat, lon) -> np.ndarray:
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        return ((lat >= self.south) & (lat <= self.north)
                & (lon >= self.west) & (lon <= self.east))

    def sample(self, array, lat, lon, nodata=None) -> np.ndarray:
        """Bilinear sample of `array` (this grid's shape) at points. NaN
        outside the grid and wherever the stencil touches nodata/NaN."""
        a = np.asarray(array, dtype=np.float64)
        if a.shape != self.shape:
            raise ProductError(f"array {a.shape} is not on this grid {self.shape}")
        if nodata is not None and not (isinstance(nodata, float)
                                       and math.isnan(nodata)):
            a = np.where(a == nodata, np.nan, a)
        lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
        lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))
        r, c = self.rowcol(lat, lon)
        inside = self.contains(lat, lon)
        r = np.clip(r, 0.0, self.height - 1.0)
        c = np.clip(c, 0.0, self.width - 1.0)
        r0 = np.clip(np.floor(r).astype(int), 0, max(self.height - 2, 0))
        c0 = np.clip(np.floor(c).astype(int), 0, max(self.width - 2, 0))
        r1 = np.minimum(r0 + 1, self.height - 1)
        c1 = np.minimum(c0 + 1, self.width - 1)
        fr = np.clip(r - r0, 0.0, 1.0)
        fc = np.clip(c - c0, 0.0, 1.0)
        v = (a[r0, c0] * (1 - fr) * (1 - fc) + a[r0, c1] * (1 - fr) * fc
             + a[r1, c0] * fr * (1 - fc) + a[r1, c1] * fr * fc)
        return np.where(inside, v, np.nan)

    # -- construction ------------------------------------------------------------
    @classmethod
    def around(cls, lat: float, lon: float, radius_m: float,
               cell_m: float, max_cells: int = 4_000_000) -> "GeoGrid":
        """A square-ish grid of `cell_m` pixels covering `radius_m` around a
        point. Refuses grids above `max_cells` pixels (say why) rather than
        exhausting the machine."""
        if radius_m <= 0 or cell_m <= 0:
            raise ProductError("radius and cell size must be positive")
        k = math.pi / 180.0 * EARTH_RADIUS_M
        coslat = max(math.cos(math.radians(lat)), 1e-6)
        n = max(1, int(math.ceil(2.0 * radius_m / cell_m)))
        if n * n > max_cells:
            raise ProductError(
                f"{2 * radius_m / 1000:.1f} km at {cell_m:g} m pixels is "
                f"{n} x {n} = {n * n:,} pixels, above the {max_cells:,} limit. "
                "Use a coarser grid or a smaller radius.")
        half_lat = 0.5 * n * cell_m / k
        half_lon = 0.5 * n * cell_m / (k * coslat)
        return cls(lon - half_lon, lat - half_lat, lon + half_lon,
                   lat + half_lat, n, n)

    @classmethod
    def covering(cls, lats, lons, margin_m: float, cell_m: float,
                 max_cells: int = 4_000_000) -> "GeoGrid":
        """The smallest grid of `cell_m` pixels holding every point plus a
        margin. Pixels are square in metres at the centre latitude."""
        lats = np.asarray(lats, dtype=np.float64).ravel()
        lons = np.asarray(lons, dtype=np.float64).ravel()
        k = math.pi / 180.0 * EARTH_RADIUS_M
        lat_c = 0.5 * (lats.min() + lats.max())
        coslat = max(math.cos(math.radians(lat_c)), 1e-6)
        s = lats.min() - margin_m / k
        n = lats.max() + margin_m / k
        w = lons.min() - margin_m / (k * coslat)
        e = lons.max() + margin_m / (k * coslat)
        h = max(1, int(math.ceil((n - s) * k / cell_m)))
        wd = max(1, int(math.ceil((e - w) * k * coslat / cell_m)))
        if h * wd > max_cells:
            raise ProductError(f"{wd} x {h} pixels is above the {max_cells:,} "
                               "limit; use a coarser grid")
        # widen to whole pixels, centred
        dlat = cell_m / k
        dlon = cell_m / (k * coslat)
        cy, cx = 0.5 * (s + n), 0.5 * (w + e)
        return cls(cx - 0.5 * wd * dlon, max(-90.0, cy - 0.5 * h * dlat),
                   cx + 0.5 * wd * dlon, min(90.0, cy + 0.5 * h * dlat), wd, h)

    def to_json(self) -> dict:
        return {"west": self.west, "south": self.south, "east": self.east,
                "north": self.north, "width": int(self.width),
                "height": int(self.height), "crs": CRS}

    @classmethod
    def from_json(cls, d: dict) -> "GeoGrid":
        return cls(float(d["west"]), float(d["south"]), float(d["east"]),
                   float(d["north"]), int(d["width"]), int(d["height"]))

    @classmethod
    def from_bounds(cls, bounds, shape) -> "GeoGrid":
        w, s, e, n = (float(v) for v in bounds)
        h, wd = int(shape[0]), int(shape[1])
        return cls(w, s, e, n, wd, h)


# ---------------------------------------------------------------------------
# Tags: GDAL_METADATA XML + JSON ImageDescription
# ---------------------------------------------------------------------------
def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "to_json"):
        return o.to_json()
    return str(o)


def _dumps(obj) -> str:
    return json.dumps(obj, default=_json_default, ensure_ascii=False,
                      allow_nan=True)


def _check_tags(tags: dict | None) -> dict:
    t = dict(tags or {})
    tier = t.get("atk:tier")
    if not tier:
        raise ProductError("every product carries its tier: put 'atk:tier' "
                           "in the tags (provenance.tier_for(method))")
    t["atk:tier"] = _prov.check_tier(tier)
    return t


def gdal_metadata_xml(tags: dict) -> str:
    """Tags as GDAL's metadata XML (dataset domain). Non-string values are
    JSON, so a model card or a parameter set reads back exactly."""
    root = ET.Element("GDALMetadata")
    for k, v in tags.items():
        item = ET.SubElement(root, "Item", name=str(k))
        item.text = v if isinstance(v, str) else _dumps(v)
    return ET.tostring(root, encoding="unicode")


def parse_gdal_metadata(xml_text: str) -> dict:
    """GDAL metadata XML -> {name: text}. Band-level items keep a
    `band<N>:` prefix. Values are left as text (GDAL's meaning)."""
    out: dict = {}
    if not xml_text:
        return out
    try:
        root = ET.fromstring(xml_text.strip().rstrip("\x00"))
    except ET.ParseError:
        return out
    for item in root.iter("Item"):
        name = item.get("name", "")
        if not name:
            continue
        sample = item.get("sample")
        key = f"band{int(sample) + 1}:{name}" if sample is not None else name
        out[key] = item.text or ""
    return out


def _nodata_text(nodata) -> str | None:
    if nodata is None:
        return None
    v = float(nodata)
    if math.isnan(v):
        return "nan"
    return repr(int(v)) if v.is_integer() else repr(v)


# ---------------------------------------------------------------------------
# Overviews
# ---------------------------------------------------------------------------
def _overview(a: np.ndarray, nodata) -> np.ndarray:
    """Halve the resolution. Floats: the mean of the valid pixels in each
    2x2 block (nodata where none is valid). Integers (masks, classes): the
    north-west pixel of the block — a mask is never averaged into 0.5."""
    h, w = a.shape
    if np.issubdtype(a.dtype, np.floating):
        hh, ww = (h + 1) // 2, (w + 1) // 2
        pad = np.full((hh * 2, ww * 2), np.nan, dtype=np.float64)
        src = a.astype(np.float64)
        if nodata is not None and not math.isnan(float(nodata)):
            src = np.where(src == float(nodata), np.nan, src)
        pad[:h, :w] = src
        blocks = pad.reshape(hh, 2, ww, 2)
        valid = np.isfinite(blocks)
        n = valid.sum(axis=(1, 3))
        s = np.where(valid, blocks, 0.0).sum(axis=(1, 3))
        with np.errstate(invalid="ignore", divide="ignore"):
            out = s / n
        fill = np.nan if nodata is None else float(nodata)
        out = np.where(n > 0, out, fill)
        return out.astype(a.dtype)
    return a[::2, ::2].copy()


def _overview_levels(a: np.ndarray, tile: int, nodata) -> list[np.ndarray]:
    out = []
    cur = a
    while max(cur.shape) > tile:
        cur = _overview(cur, nodata)
        out.append(cur)
    return out


# ---------------------------------------------------------------------------
# GeoTIFF writing
# ---------------------------------------------------------------------------
def _has_tifffile() -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec("tifffile") is not None
    except (ImportError, ValueError):
        return False


_DTYPES = {np.dtype("uint8"), np.dtype("int8"), np.dtype("uint16"),
           np.dtype("int16"), np.dtype("uint32"), np.dtype("int32"),
           np.dtype("float32"), np.dtype("float64")}


def write_geotiff(path, array, bounds, nodata=None, tags: dict | None = None,
                  cog: bool = True, *, tile: int = 256, compress: bool = True,
                  engine: str = "auto", rf=None,
                  record_kind: str = "product") -> Path:
    """Write one band as a GeoTIFF in EPSG:4326.

    `bounds` is (west, south, east, north) in degrees — the outer pixel
    edges — or a `GeoGrid`. `array[0]` is the NORTH row. `tags` must hold
    `atk:tier`; everything in it is written as GDAL metadata and as the
    JSON ImageDescription. `cog=True`: tiled, deflate, internal overviews
    (see the module docstring for exactly what that means per engine).
    `engine` is "tifffile", "builtin" or "auto" (tifffile when installed).
    With `rf` (an RfData), the file is recorded in the write log."""
    a = np.asarray(array)
    if a.ndim != 2:
        raise ProductError(f"one band per GeoTIFF: got an array of shape "
                           f"{a.shape}")
    if a.dtype == np.bool_:
        a = a.astype(np.uint8)
    if a.dtype not in _DTYPES:
        a = a.astype(np.float32)
    grid = bounds if isinstance(bounds, GeoGrid) else \
        GeoGrid.from_bounds(bounds, a.shape)
    if grid.shape != a.shape:
        raise ProductError(f"the array is {a.shape} but the grid is {grid.shape}")
    t = _check_tags(tags)
    t.setdefault("atk:crs", CRS)
    t.setdefault("atk:written", time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                              time.gmtime()))
    tile = max(16, int(tile) // 16 * 16)
    # TIFF ASCII tags are 7-bit: JSON escapes, XML character references
    desc = json.dumps({"atk_product": FORMAT_VERSION, "crs": CRS,
                       "bounds": list(grid.bounds), "shape": list(a.shape),
                       "dtype": str(a.dtype), "nodata": nodata, "tags": t},
                      default=_json_default, ensure_ascii=True)
    nd = _nodata_text(nodata)
    xml = gdal_metadata_xml(t).encode("ascii", "xmlcharrefreplace").decode("ascii")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    eng = engine
    if eng == "auto":
        eng = "tifffile" if _has_tifffile() else "builtin"
    if eng == "tifffile":
        _write_tifffile(tmp, a, grid, nd, xml, desc, cog, tile, compress, nodata)
    elif eng == "builtin":
        _write_builtin(tmp, a, grid, nd, xml, desc, cog, tile, compress, nodata)
    else:
        raise ProductError(f"unknown GeoTIFF engine {engine!r}")
    tmp.replace(p)
    if rf is not None:
        rf.record(p, record_kind, f"GeoTIFF {a.shape[1]}x{a.shape[0]} "
                                  f"{t['atk:tier']}")
    return p


def _geo_values(grid: GeoGrid):
    scale = (grid.dlon, grid.dlat, 0.0)
    tie = (0.0, 0.0, 0.0, grid.west, grid.north, 0.0)
    return scale, tie


def _write_tifffile(path, a, grid, nd, xml, desc, cog, tile, compress, nodata):
    import tifffile
    scale, tie = _geo_values(grid)
    extratags = [(_TAG_PIXEL_SCALE, "d", 3, scale, True),
                 (_TAG_TIEPOINT, "d", 6, tie, True),
                 (_TAG_GEOKEYS, "H", len(_GEOKEYS_4326), _GEOKEYS_4326, True),
                 (_TAG_GEO_ASCII, "s", 0, _GEO_ASCII, True),
                 (_TAG_GDAL_METADATA, "s", 0, xml, True)]
    if nd is not None:
        extratags.append((_TAG_GDAL_NODATA, "s", 0, nd, True))
    opts = {"metadata": None, "software": False, "photometric": "minisblack"}
    if compress:
        opts["compression"] = "zlib"
    with tifffile.TiffWriter(str(path)) as tw:
        if cog:
            tw.write(a, tile=(tile, tile), description=desc,
                     extratags=extratags, **opts)
            for ov in _overview_levels(a, tile, nodata):
                tw.write(ov, tile=(tile, tile), subfiletype=1, **opts)
        else:
            tw.write(a, description=desc, extratags=extratags,
                     rowsperstrip=max(1, min(a.shape[0], 65536 // max(
                         1, a.shape[1] * a.dtype.itemsize))), **opts)


# -- the built-in writer: classic little-endian TIFF, strict COG layout -------
_TIFF_TYPES = {"B": (1, 1), "s": (2, 1), "H": (3, 2), "I": (4, 4), "d": (12, 8)}


def _sample_format(dt: np.dtype) -> int:
    if np.issubdtype(dt, np.floating):
        return 3
    if np.issubdtype(dt, np.signedinteger):
        return 2
    return 1


def _encode_entries(entries: list[tuple[int, str, object]]):
    """[(tag, fmt, value)] -> (tag, type, count, packed bytes)."""
    out = []
    for tag, fmt, val in sorted(entries, key=lambda e: e[0]):
        typ, size = _TIFF_TYPES[fmt]
        if fmt == "s":
            raw = (val if isinstance(val, bytes) else str(val).encode("utf-8")) + b"\x00"
            count = len(raw)
        else:
            vals = list(val) if isinstance(val, (list, tuple)) else [val]
            count = len(vals)
            raw = struct.pack("<" + fmt * count, *vals)
        out.append((tag, typ, count, raw))
    return out


def _ifd_size(encoded) -> int:
    extra = 0
    for _tag, _typ, _count, raw in encoded:
        if len(raw) > 4:
            extra += len(raw) + (len(raw) & 1)
    return 2 + 12 * len(encoded) + 4 + extra


def _tiles(a: np.ndarray, tile: int, compress: bool, fill) -> list[bytes]:
    h, w = a.shape
    nty, ntx = -(-h // tile), -(-w // tile)
    le = a.astype(a.dtype.newbyteorder("<"), copy=False)
    out = []
    for ty in range(nty):
        for tx in range(ntx):
            blk = np.full((tile, tile), fill, dtype=le.dtype)
            part = le[ty * tile:(ty + 1) * tile, tx * tile:(tx + 1) * tile]
            blk[:part.shape[0], :part.shape[1]] = part
            raw = blk.tobytes()
            out.append(zlib.compress(raw, 6) if compress else raw)
    return out


def _write_builtin(path, a, grid, nd, xml, desc, cog, tile, compress, nodata):
    fill = 0
    if nodata is not None:
        try:
            fill = np.array(nodata).astype(a.dtype).item()
        except (TypeError, ValueError, OverflowError):
            fill = 0
    if cog:
        levels = [a] + _overview_levels(a, tile, nodata)
        tsize = tile
    else:
        levels = [a]
        tsize = None
    scale, tie = _geo_values(grid)
    pages = []
    for i, lev in enumerate(levels):
        if tsize:
            blocks = _tiles(lev, tsize, compress, fill)
        else:   # one strip per block of rows
            rps = max(1, min(lev.shape[0], 65536 // max(1, lev.shape[1]
                                                         * lev.dtype.itemsize)))
            le = lev.astype(lev.dtype.newbyteorder("<"), copy=False)
            blocks = []
            for r0 in range(0, lev.shape[0], rps):
                raw = le[r0:r0 + rps].tobytes()
                blocks.append(zlib.compress(raw, 6) if compress else raw)
        ent = [(254, "I", 1 if i else 0), (256, "I", lev.shape[1]),
               (257, "I", lev.shape[0]), (258, "H", lev.dtype.itemsize * 8),
               (259, "H", 8 if compress else 1), (262, "H", 1),
               (277, "H", 1), (284, "H", 1),
               (339, "H", _sample_format(lev.dtype))]
        if tsize:
            ent += [(322, "I", tsize), (323, "I", tsize),
                    (324, "I", [0] * len(blocks)),
                    (325, "I", [len(b) for b in blocks])]
        else:
            ent += [(273, "I", [0] * len(blocks)), (278, "I", rps),
                    (279, "I", [len(b) for b in blocks])]
        if i == 0:
            ent += [(_TAG_DESCRIPTION, "s", desc),
                    (_TAG_PIXEL_SCALE, "d", list(scale)),
                    (_TAG_TIEPOINT, "d", list(tie)),
                    (_TAG_GEOKEYS, "H", list(_GEOKEYS_4326)),
                    (_TAG_GEO_ASCII, "s", _GEO_ASCII),
                    (_TAG_GDAL_METADATA, "s", xml)]
            if nd is not None:
                ent.append((_TAG_GDAL_NODATA, "s", nd))
        pages.append({"entries": ent, "blocks": blocks})
    # layout: header, every IFD (full resolution first), then the data from
    # the smallest overview to the full resolution (the COG order)
    pos = 8
    for pg in pages:
        pg["ifd_at"] = pos
        pos += _ifd_size(_encode_entries(pg["entries"]))
        pos += pos & 1
    for pg in reversed(pages):
        offs = []
        for b in pg["blocks"]:
            offs.append(pos)
            pos += len(b)
        pg["offsets"] = offs
    if pos >= 2 ** 32:
        raise ProductError("this raster is over 4 GB; the built-in writer "
                           "writes classic TIFF only — install tifffile")
    buf = bytearray(b"II" + struct.pack("<HI", 42, 8))
    for k, pg in enumerate(pages):
        key = 324 if tsize else 273
        ent = [(t, f, (pg["offsets"] if t == key else v))
               for t, f, v in pg["entries"]]
        enc = _encode_entries(ent)
        start = pg["ifd_at"]
        assert len(buf) == start, "IFD layout drifted"
        nxt = pages[k + 1]["ifd_at"] if k + 1 < len(pages) else 0
        extra_at = start + 2 + 12 * len(enc) + 4
        head = bytearray(struct.pack("<H", len(enc)))
        extra = bytearray()
        for tag, typ, count, raw in enc:
            if len(raw) <= 4:
                head += struct.pack("<HHI", tag, typ, count) + raw.ljust(4, b"\x00")
            else:
                head += struct.pack("<HHII", tag, typ, count,
                                    extra_at + len(extra))
                extra += raw
                if len(raw) & 1:
                    extra += b"\x00"
        head += struct.pack("<I", nxt)
        buf += head + extra
        if len(buf) & 1:
            buf += b"\x00"
    for pg in reversed(pages):
        for off, b in zip(pg["offsets"], pg["blocks"]):
            assert len(buf) == off, "data layout drifted"
            buf += b
    Path(path).write_bytes(bytes(buf))


# ---------------------------------------------------------------------------
# GeoTIFF reading
# ---------------------------------------------------------------------------
@dataclass
class GeoRaster:
    """A GeoTIFF read back: the full-resolution band, where it sits, its
    nodata value, its tags (the JSON ImageDescription when this toolkit
    wrote it, else GDAL's metadata), and how many overviews it carries."""
    array: np.ndarray
    grid: GeoGrid
    nodata: float | None = None
    tags: dict = field(default_factory=dict)
    crs: str = CRS
    overviews: int = 0
    path: str = ""
    tiled: bool = False

    @property
    def bounds(self):
        return self.grid.bounds

    @property
    def tier(self) -> str:
        return str(self.tags.get("atk:tier", ""))

    def masked(self) -> np.ndarray:
        """float64 with NaN where the raster has no data."""
        a = self.array.astype(np.float64)
        if self.nodata is not None and not math.isnan(float(self.nodata)):
            a = np.where(self.array == self.nodata, np.nan, a)
        return a

    def sample(self, lat, lon) -> np.ndarray:
        return self.grid.sample(self.masked(), lat, lon)


def _geo_from_tags(scale, tie, keys, shape) -> GeoGrid:
    if scale is None or tie is None:
        raise ProductError("this TIFF has no georeferencing (no "
                           "ModelPixelScale / ModelTiepoint)")
    keyd = {}
    if keys is not None:
        k = list(keys)
        n = int(k[3]) if len(k) >= 4 else 0
        for i in range(n):
            kid, loc, _cnt, val = k[4 + 4 * i: 8 + 4 * i]
            if int(loc) == 0:
                keyd[int(kid)] = int(val)
    model = keyd.get(1024)
    epsg = keyd.get(2048)
    if model is not None and model != 2:
        raise ProductError("this GeoTIFF is in a projected CRS; products are "
                           "EPSG:4326 — reproject it first (gdalwarp -t_srs "
                           "EPSG:4326)")
    if epsg is not None and epsg != 4326:
        raise ProductError(f"this GeoTIFF is on geographic CRS EPSG:{epsg}, "
                           "not WGS 84 (EPSG:4326)")
    sx, sy = float(scale[0]), float(scale[1])
    i, j, x, y = float(tie[0]), float(tie[1]), float(tie[3]), float(tie[4])
    west = x - i * sx
    north = y + j * sy
    if keyd.get(1025, 1) == 2:            # pixel-is-point: tie at the centre
        west -= 0.5 * sx
        north += 0.5 * sy
    h, w = shape
    return GeoGrid(west, north - h * sy, west + w * sx, north, w, h)


def _parse_nodata(text):
    if text is None:
        return None
    s = (text.decode("ascii", "ignore") if isinstance(text, bytes)
         else str(text)).strip().rstrip("\x00")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _tags_from(desc, xml) -> dict:
    if desc:
        try:
            d = json.loads(desc)
            if isinstance(d, dict) and "tags" in d and d.get("atk_product"):
                return dict(d["tags"])
        except (json.JSONDecodeError, TypeError):
            pass
    return parse_gdal_metadata(xml or "")


def read_geotiff(path, engine: str = "auto") -> GeoRaster:
    """Read a single-band GeoTIFF in EPSG:4326 (ours or anybody's)."""
    p = Path(path)
    if not p.exists():
        raise ProductError(f"{p} does not exist")
    eng = engine
    if eng == "auto":
        eng = "tifffile" if _has_tifffile() else "builtin"
    if eng == "tifffile":
        import tifffile
        with tifffile.TiffFile(str(p)) as tf:
            page = tf.pages[0]
            tags = page.tags

            def val(code):
                t = tags.get(code)
                return None if t is None else t.value
            arr = page.asarray()
            if arr.ndim != 2:
                raise ProductError(f"{p.name} has {arr.ndim} dimensions; "
                                   "products are one band")
            grid = _geo_from_tags(val(_TAG_PIXEL_SCALE), val(_TAG_TIEPOINT),
                                  val(_TAG_GEOKEYS), arr.shape)
            desc = val(_TAG_DESCRIPTION)
            xml = val(_TAG_GDAL_METADATA)
            nodata = _parse_nodata(val(_TAG_GDAL_NODATA))
            n_ov = sum(1 for pg in tf.pages[1:]
                       if int(getattr(pg, "subfiletype", 0) or 0) & 1)
            tiled = bool(page.is_tiled)
    elif eng == "builtin":
        info = _read_builtin(p)
        arr = info["array"]
        grid = _geo_from_tags(info["tags"].get(_TAG_PIXEL_SCALE),
                              info["tags"].get(_TAG_TIEPOINT),
                              info["tags"].get(_TAG_GEOKEYS), arr.shape)
        desc = info["tags"].get(_TAG_DESCRIPTION)
        xml = info["tags"].get(_TAG_GDAL_METADATA)
        nodata = _parse_nodata(info["tags"].get(_TAG_GDAL_NODATA))
        n_ov = info["overviews"]
        tiled = info["tiled"]
    else:
        raise ProductError(f"unknown GeoTIFF engine {engine!r}")
    if isinstance(desc, bytes):
        desc = desc.decode("utf-8", "replace")
    if isinstance(xml, bytes):
        xml = xml.decode("utf-8", "replace")
    return GeoRaster(array=np.asarray(arr), grid=grid, nodata=nodata,
                     tags=_tags_from(desc, xml), overviews=n_ov, path=str(p),
                     tiled=tiled)


_TYPE_FMT = {1: ("B", 1), 2: ("s", 1), 3: ("H", 2), 4: ("I", 4), 6: ("b", 1),
             8: ("h", 2), 9: ("i", 4), 11: ("f", 4), 12: ("d", 8),
             16: ("Q", 8)}


def _read_builtin(p: Path) -> dict:
    """A small TIFF reader: classic TIFF, either byte order, strips or tiles,
    no compression or deflate, no predictor, one sample per pixel — enough
    for every product either engine writes."""
    data = p.read_bytes()
    if data[:2] == b"II":
        bo = "<"
    elif data[:2] == b"MM":
        bo = ">"
    else:
        raise ProductError(f"{p.name} is not a TIFF")
    magic = struct.unpack(bo + "H", data[2:4])[0]
    if magic == 43:
        raise ProductError(f"{p.name} is a BigTIFF; the built-in reader "
                           "reads classic TIFF — install tifffile")
    if magic != 42:
        raise ProductError(f"{p.name} is not a TIFF")
    off = struct.unpack(bo + "I", data[4:8])[0]
    ifds = []
    seen = set()
    while off and off not in seen and len(ifds) < 64:
        seen.add(off)
        n = struct.unpack(bo + "H", data[off:off + 2])[0]
        tags = {}
        for k in range(n):
            e = off + 2 + 12 * k
            tag, typ, count = struct.unpack(bo + "HHI", data[e:e + 8])
            if typ not in _TYPE_FMT:
                continue
            fmt, size = _TYPE_FMT[typ]
            nbytes = size * count
            if nbytes <= 4:
                raw = data[e + 8:e + 8 + nbytes]
            else:
                vo = struct.unpack(bo + "I", data[e + 8:e + 12])[0]
                raw = data[vo:vo + nbytes]
            if fmt == "s":
                tags[tag] = raw.rstrip(b"\x00").decode("utf-8", "replace")
            else:
                v = struct.unpack(bo + fmt * count, raw)
                tags[tag] = v[0] if count == 1 else v
        ifds.append(tags)
        off = struct.unpack(bo + "I", data[off + 2 + 12 * n: off + 6 + 12 * n])[0]
    if not ifds:
        raise ProductError(f"{p.name} has no image")
    t = ifds[0]

    def one(v):
        return v[0] if isinstance(v, tuple) else v
    w, h = int(one(t[256])), int(one(t[257]))
    bits = int(one(t.get(258, 8)))
    comp = int(one(t.get(259, 1)))
    spp = int(one(t.get(277, 1)))
    pred = int(one(t.get(317, 1)))
    sfmt = int(one(t.get(339, 1)))
    if spp != 1:
        raise ProductError(f"{p.name} has {spp} samples per pixel; products "
                           "are one band")
    if comp not in (1, 8, 32946):
        raise ProductError(f"{p.name} uses TIFF compression {comp}; the "
                           "built-in reader knows none and deflate — install "
                           "tifffile")
    if pred != 1:
        raise ProductError(f"{p.name} uses a predictor; install tifffile")
    kind = {1: "u", 2: "i", 3: "f"}.get(sfmt, "u")
    dt = np.dtype(f"{bo}{kind}{bits // 8}")

    def chunk(o, c):
        raw = data[o:o + c]
        return zlib.decompress(raw) if comp in (8, 32946) else raw

    def as_list(v):
        return list(v) if isinstance(v, tuple) else [v]
    out = np.empty((h, w), dtype=dt.newbyteorder("="))
    tiled = 322 in t
    if tiled:
        tw_, th_ = int(one(t[322])), int(one(t[323]))
        offs, cnts = as_list(t[324]), as_list(t[325])
        ntx = -(-w // tw_)
        for i, (o, c) in enumerate(zip(offs, cnts)):
            blk = np.frombuffer(chunk(o, c), dtype=dt)[:tw_ * th_]
            blk = blk.reshape(th_, tw_)
            ty, tx = divmod(i, ntx)
            r0, c0 = ty * th_, tx * tw_
            rr, cc = min(th_, h - r0), min(tw_, w - c0)
            if rr > 0 and cc > 0:
                out[r0:r0 + rr, c0:c0 + cc] = blk[:rr, :cc]
    else:
        rps = int(one(t.get(278, h)))
        offs, cnts = as_list(t[273]), as_list(t[279])
        for i, (o, c) in enumerate(zip(offs, cnts)):
            r0 = i * rps
            rows = min(rps, h - r0)
            blk = np.frombuffer(chunk(o, c), dtype=dt)[:rows * w]
            out[r0:r0 + rows] = blk.reshape(rows, w)
    n_ov = sum(1 for d in ifds[1:] if int(one(d.get(254, 0))) & 1)
    return {"array": out, "tags": t, "overviews": n_ov, "tiled": tiled}


# ---------------------------------------------------------------------------
# GeoJSON
# ---------------------------------------------------------------------------
def _ll(lat, lon) -> list[float]:
    lat, lon = float(lat), float(lon)
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise ProductError("a coordinate is not a number")
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        raise ProductError(f"({lat}, {lon}) is not a latitude/longitude")
    return [round(lon, 7), round(lat, 7)]


def point_feature(lat, lon, properties: dict | None = None) -> dict:
    return {"type": "Feature", "geometry": {"type": "Point",
                                            "coordinates": _ll(lat, lon)},
            "properties": dict(properties or {})}


def line_feature(lats, lons, properties: dict | None = None) -> dict:
    coords = [_ll(a, o) for a, o in zip(lats, lons)]
    if len(coords) < 2:
        raise ProductError("a line needs two points")
    return {"type": "Feature", "geometry": {"type": "LineString",
                                            "coordinates": coords},
            "properties": dict(properties or {})}


def _ring_area(ring) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def _close(ring) -> np.ndarray:
    r = np.asarray(ring, dtype=np.float64)
    if r.shape[0] < 3:
        raise ProductError("a polygon ring needs three points")
    if not np.allclose(r[0], r[-1]):
        r = np.vstack([r, r[:1]])
    return r


def polygon_coords(exterior, holes=()) -> list:
    """Rings given as (n, 2) arrays of (lon, lat) -> GeoJSON polygon
    coordinates with the RFC 7946 winding (exterior counter-clockwise,
    holes clockwise)."""
    out = []
    ext = _close(exterior)
    if _ring_area(ext) < 0:
        ext = ext[::-1]
    out.append([_ll(lat, lon) for lon, lat in ext])
    for hole in holes:
        hr = _close(hole)
        if _ring_area(hr) > 0:
            hr = hr[::-1]
        out.append([_ll(lat, lon) for lon, lat in hr])
    return out


def multipolygon_feature(polygons, properties: dict | None = None) -> dict:
    """`polygons`: [(exterior, [holes...]), ...], rings as (lon, lat)."""
    coords = [polygon_coords(ext, holes) for ext, holes in polygons]
    return {"type": "Feature", "geometry": {"type": "MultiPolygon",
                                            "coordinates": coords},
            "properties": dict(properties or {})}


def feature_collection(features, **atk) -> dict:
    fc = {"type": "FeatureCollection", "features": list(features)}
    if atk:
        fc["atk"] = atk
    return fc


def write_geojson(path, features, *, tier: str | None = None, rf=None,
                  meta: dict | None = None, record_kind: str = "product") -> Path:
    """Write a FeatureCollection. Every feature carries `atk:tier` in its
    properties: `tier` fills it where missing; a feature without one when no
    `tier` is given is refused. `meta` goes in the collection's `atk`
    member (a foreign member, allowed by RFC 7946)."""
    if isinstance(features, dict) and features.get("type") == "FeatureCollection":
        meta = {**(features.get("atk") or {}), **(meta or {})}
        features = features.get("features", [])
    feats = []
    for i, f in enumerate(features):
        if not isinstance(f, dict) or f.get("type") != "Feature":
            raise ProductError(f"item {i} is not a GeoJSON Feature")
        props = dict(f.get("properties") or {})
        t = props.get("atk:tier") or tier
        if not t:
            raise ProductError(f"feature {i} has no atk:tier — every product "
                               "carries its tier")
        props["atk:tier"] = _prov.check_tier(t)
        g = f.get("geometry")
        if not isinstance(g, dict) or "type" not in g:
            raise ProductError(f"feature {i} has no geometry")
        feats.append({"type": "Feature", "geometry": g, "properties": props})
    fc = feature_collection(feats, **(meta or {}))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(fc, default=_json_default, ensure_ascii=False,
                              indent=1), encoding="utf-8")
    tmp.replace(p)
    if rf is not None:
        rf.record(p, record_kind, f"GeoJSON {len(feats)} features")
    return p


def read_geojson(path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("type") != "FeatureCollection":
        raise ProductError(f"{Path(path).name} is not a GeoJSON "
                           "FeatureCollection")
    return d


# ---------------------------------------------------------------------------
# Product folders, manifests, the write log, import into a new install
# ---------------------------------------------------------------------------
def _safe(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name)).strip("-.")
    if not s:
        raise ProductError("an empty name cannot be a folder")
    return s[:96]


def run_name(label: str = "", when: float | None = None) -> str:
    """'<UTC stamp>_<label>' — sortable, unique per second, readable."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(when))
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(label)).strip("-.")[:48]
    return f"{stamp}_{slug}" if slug else stamp


def product_dir(rf, kind: str, run: str | None = None) -> Path:
    if kind not in PRODUCT_KINDS:
        raise ProductError(f"{kind!r} is not a product kind — one of "
                           f"{', '.join(PRODUCT_KINDS)}")
    base = Path(rf.products(kind))
    if kind in RUN_KINDS and not run:
        raise ProductError(f"a {kind} product is one run: give it a run name")
    return base / _safe(run) if run else base


def _sha(path) -> str:
    return _prov.sha256_path(path)


def load_manifest(run_dir) -> dict:
    p = Path(run_dir) / MANIFEST
    if not p.exists():
        raise ProductError(f"{Path(run_dir).name} has no {MANIFEST}")
    return json.loads(p.read_text(encoding="utf-8"))


class ProductRun:
    """One product folder: write files through it and they are hashed into
    the manifest and recorded in the rf_data write log; `finish()` writes
    `manifest.json`. Re-opening an existing run (`ProductRun.open`) adds
    layers to it — the measured and learned layers beside the physics."""

    def __init__(self, rf, kind: str, run: str | None = None, *,
                 label: str = "", params: dict | None = None,
                 tier: str | None = None, card=None, description: str = "",
                 method: str = ""):
        if kind in RUN_KINDS and not run:
            run = run_name(label or kind)
        self.rf = rf
        self.kind = kind
        self.run = run or ""
        self.dir = product_dir(rf, kind, run)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.params = dict(params or {})
        self.tier = _prov.check_tier(tier) if tier else ""
        self.card = card.to_json() if hasattr(card, "to_json") else card
        self.description = description
        self.method = method
        self.files: dict = {}
        self.layers: list = []
        self.created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.extra: dict = {}

    @classmethod
    def open(cls, rf, kind: str, run: str | None = None) -> "ProductRun":
        d = product_dir(rf, kind, run)
        m = load_manifest(d)
        pr = cls(rf, kind, run, params=m.get("params"), tier=m.get("tier") or None,
                 card=m.get("card"), description=m.get("description", ""),
                 method=m.get("method", ""))
        pr.files = dict(m.get("files", {}))
        pr.layers = list(m.get("layers", []))
        pr.created = m.get("created", pr.created)
        pr.extra = dict(m.get("extra", {}))
        return pr

    # -- writing ---------------------------------------------------------------
    def path(self, name: str) -> Path:
        return self.dir / _safe(name)

    def _note(self, p: Path, tier: str, layer: dict | None, role: str):
        rel = p.relative_to(self.dir).as_posix()
        self.files[rel] = {"sha256": _sha(p), "bytes": p.stat().st_size,
                           "tier": tier, "role": role}
        self.rf.record(p, f"product:{self.kind}", f"{rel} ({tier})")
        if layer is not None:
            entry = {"file": rel, "tier": tier, **layer}
            self.layers = [l for l in self.layers if l.get("file") != rel]
            self.layers.append(entry)

    def add_geotiff(self, name: str, array, grid, *, tier: str, nodata=None,
                    tags: dict | None = None, layer: dict | None = None,
                    cog: bool = True, role: str = "raster", **kw) -> Path:
        t = {"atk:tier": tier, "atk:product_kind": self.kind,
             "atk:run": self.run, **(tags or {})}
        if self.method and "atk:method" not in t:
            t["atk:method"] = self.method
        if self.card and "atk:card" not in t:
            t["atk:card"] = self.card
        p = write_geotiff(self.path(name), array, grid, nodata=nodata,
                          tags=t, cog=cog, **kw)
        self._note(p, _prov.check_tier(tier), layer, role)
        return p

    def add_geojson(self, name: str, features, *, tier: str,
                    layer: dict | None = None, meta: dict | None = None,
                    role: str = "vector") -> Path:
        m = {"product_kind": self.kind, "run": self.run, **(meta or {})}
        p = write_geojson(self.path(name), features, tier=tier, meta=m)
        self._note(p, _prov.check_tier(tier), layer, role)
        return p

    def add_json(self, name: str, obj, *, tier: str = "measured",
                 role: str = "table") -> Path:
        p = self.path(name)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(obj, default=_json_default, indent=2,
                                  ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)
        self._note(p, _prov.check_tier(tier), None, role)
        return p

    def add_text(self, name: str, text: str, *, tier: str = "measured",
                 role: str = "report", encoding: str = "utf-8") -> Path:
        p = self.path(name)
        tmp = p.with_name(p.name + ".tmp")
        with open(tmp, "w", encoding=encoding, newline="") as f:
            f.write(text)
        tmp.replace(p)
        self._note(p, _prov.check_tier(tier), None, role)
        return p

    def add_file(self, path, *, tier: str, role: str = "file",
                 layer: dict | None = None) -> Path:
        """Record a file something else wrote inside this run's folder."""
        p = Path(path)
        if self.dir.resolve() not in p.resolve().parents:
            raise ProductError(f"{p} is not inside {self.dir}")
        self._note(p, _prov.check_tier(tier), layer, role)
        return p

    def finish(self, **extra) -> Path:
        self.extra.update(extra)
        tiers = sorted({v.get("tier") for v in self.files.values()
                        if v.get("tier")})
        from atk_diffusion import __version__
        man = {"atk_product": FORMAT_VERSION, "kind": self.kind,
               "run": self.run, "created": self.created,
               "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "tier": self.tier or (tiers[0] if len(tiers) == 1 else ""),
               "tiers": tiers,
               "tier_words": {t: _prov.TIER_WORDS[t] for t in tiers},
               "method": self.method, "params": self.params, "card": self.card,
               "description": self.description, "crs": CRS,
               "layers": self.layers, "files": self.files,
               "toolkit_version": __version__, "extra": self.extra}
        p = self.dir / MANIFEST
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(man, default=_json_default, indent=2,
                                  ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)
        self.rf.record(p, f"product:{self.kind}", "manifest")
        return p


def list_runs(rf, kind: str) -> list[tuple[Path, dict]]:
    """Every run of a kind that has a manifest, newest first."""
    if kind not in PRODUCT_KINDS:
        raise ProductError(f"{kind!r} is not a product kind")
    base = Path(rf.products(kind))
    out = []
    if base.is_dir():
        for d in base.iterdir():
            if (d / MANIFEST).exists():
                try:
                    out.append((d, load_manifest(d)))
                except (ProductError, json.JSONDecodeError):
                    continue
    out.sort(key=lambda t: t[1].get("created", ""), reverse=True)
    return out


def verify_run(run_dir) -> tuple[bool, list[str]]:
    """(ok, problems): every file the manifest lists is present and has the
    hash it was written with."""
    d = Path(run_dir)
    try:
        man = load_manifest(d)
    except (ProductError, json.JSONDecodeError) as e:
        return False, [str(e)]
    problems = []
    for rel, info in (man.get("files") or {}).items():
        p = d / rel
        if not p.exists():
            problems.append(f"{rel} is listed in the manifest but missing")
        elif info.get("sha256") and _sha(p) != info["sha256"]:
            problems.append(f"{rel} changed after it was written (its hash no "
                            "longer matches the manifest) — named, not used")
    return not problems, problems


def import_run(src_dir, rf, kind: str | None = None) -> tuple[Path, str]:
    """Bring a product folder from another rf_data (a USB stick, an old
    install) into this one: verified against its manifest first, copied,
    verified again, recorded in this write log. Returns (path, words)."""
    src = Path(src_dir)
    ok, problems = verify_run(src)
    if not ok:
        raise ProductError(f"{src.name} will not be imported: "
                           + "; ".join(problems))
    man = load_manifest(src)
    k = kind or man.get("kind")
    if k not in PRODUCT_KINDS:
        raise ProductError(f"{src.name}'s manifest names no product kind")
    run = man.get("run") or ""
    if k in RUN_KINDS:
        dest = product_dir(rf, k, run or src.name)
    else:
        dest = product_dir(rf, k, run or None)
    if dest.exists() and (dest / MANIFEST).exists():
        same, _ = verify_run(dest)
        theirs = load_manifest(dest).get("files", {})
        if same and theirs == man.get("files", {}):
            return dest, f"{dest.name} is already here, identical — nothing copied"
        # a different product of the same name: kept apart, never merged
        # silently (a library is merged deliberately, by its own tool)
        base = (dest.with_name(dest.name + "_imported") if k in RUN_KINDS
                else dest / f"imported_{_safe(src.name)}")
        cand, i = base, 2
        while cand.exists():
            cand = base.with_name(f"{base.name}{i}")
            i += 1
        dest = cand
    dest.mkdir(parents=True, exist_ok=True)
    for rel in list((man.get("files") or {}).keys()) + [MANIFEST]:
        s = src / rel
        t = dest / rel
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, t)
        rf.record(t, f"product:{k}", f"imported from {src}")
    ok2, problems2 = verify_run(dest)
    if not ok2:
        raise ProductError("the copy does not match its manifest: "
                           + "; ".join(problems2))
    return dest, (f"imported {len(man.get('files', {}))} files into "
                  f"{dest.relative_to(Path(rf.root)).as_posix()}")
