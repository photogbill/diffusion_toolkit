# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""WSPR spots: parse them, place them, and decode them at arm's length
(plan §4.J — HF propagation now, from the Kiwi network).

Bill: *"I also like the HF propagation now concept."* ATK already talks to
KiwiSDRs; a Kiwi tuned to a WSPR frequency hears two-minute beacon
transmissions from stations all over the world, each carrying its call, its
grid and its power. Decoded LOCALLY from the Kiwi's audio — nothing is sent
anywhere — every decode is a measured path: this band was open, from there
to here, at this hour, at this SNR.

WHAT IS PARSED. Two line formats, by their tokens rather than by columns
(column widths drift between versions):

* WSJT-X's `ALL_WSPR.TXT` (and wsprd's `wspr_spots.txt`):
      YYMMDD HHMM [sync] SNR DT FREQ_MHZ  CALL [GRID] DBM  DRIFT …
      210314 1830   3 -21  0.18  14.0970957  K1ABC FN42 37   0   1  0 …
* wsprd's own console output:
      HHMM SNR DT FREQ_MHZ DRIFT  CALL [GRID] DBM
      1830 -21  0.2  14.097096  0  K1ABC FN42 37

The message is a WSPR type 1 (CALL GRID4 DBM), type 2 (PFX/CALL DBM, no
grid) or type 3 (<CALL> GRID6 DBM, a hashed call). A line that is not a
spot is counted and skipped, with the reason kept.

THE DECODER IS GPL, SO IT RUNS AS ITS OWN PROGRAM. `wsprd` (part of WSJT-X,
GPL-3) is never vendored, linked or imported: `run_wsprd` starts it as a
separate process from a path the analyst configures, on a 12 kHz mono WAV
(`prepare_wav` makes one from the Kiwi's audio), and parses what it prints.
This repository is all rights reserved (decision D1), and that is the only
way GPL code may touch it.

PLACE. Distance and bearings are great-circle (mean Earth radius
6371.0088 km) between grid centres, so they carry the locators' own
uncertainty (±55–110 km for four characters).
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import wave
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from atk_diffusion.hf import maidenhead as _mh

EARTH_KM = 6371.0088

#: Amateur bands (MHz), as wide as the IARU regions allow — a spot's band is
#: the one its frequency falls in.
BANDS = (("2200m", 0.1357, 0.1378), ("630m", 0.472, 0.479),
         ("160m", 1.8, 2.0), ("80m", 3.5, 4.0), ("60m", 5.25, 5.45),
         ("40m", 7.0, 7.3), ("30m", 10.1, 10.15), ("20m", 14.0, 14.35),
         ("17m", 18.068, 18.168), ("15m", 21.0, 21.45), ("12m", 24.89, 24.99),
         ("10m", 28.0, 29.7), ("6m", 50.0, 54.0), ("4m", 70.0, 70.5),
         ("2m", 144.0, 148.0))
BAND_ORDER = tuple(b[0] for b in BANDS)

#: WSPR dial frequencies (USB), MHz — what a Kiwi is tuned to for each band.
WSPR_DIAL_MHZ = {"2200m": 0.136, "630m": 0.4742, "160m": 1.8366,
                 "80m": 3.5686, "60m": 5.2872, "40m": 7.0386, "30m": 10.1387,
                 "20m": 14.0956, "17m": 18.1046, "15m": 21.0946,
                 "12m": 24.9246, "10m": 28.1246, "6m": 50.293}

_CALL = re.compile(r"^(?:<[^<>\s]*>|(?:[A-Z0-9]{1,4}/)?(?=[A-Z0-9]*[0-9])"
                   r"(?=[A-Z0-9]*[A-Z])[A-Z0-9]{3,7}(?:/[A-Z0-9]{1,4})?)$")
_INT = re.compile(r"^[+-]?\d+$")
_FLOAT = re.compile(r"^[+-]?\d+\.\d+$")


def band_of(freq_hz: float) -> str:
    f = float(freq_hz) / 1e6
    for name, lo, hi in BANDS:
        if lo <= f <= hi:
            return name
    return ""


def distance_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance (haversine)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_KM * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    """Initial great-circle bearing from point 1 to point 2, 0–360°."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


@dataclass
class Spot:
    """One WSPR decode: a measured path from `call` (at `grid`) to the
    receiver (`rx_call` at `rx_grid`)."""
    time_utc: str                 # ISO 8601, minute resolution
    snr_db: int
    dt_s: float
    freq_hz: float
    call: str
    grid: str = ""
    power_dbm: int | None = None
    drift_hz: int | None = None
    band: str = ""
    rx_call: str = ""
    rx_grid: str = ""
    distance_km: float | None = None
    bearing_tx_to_rx: float | None = None
    bearing_rx_to_tx: float | None = None
    source: str = ""
    tier: str = "measured"

    @property
    def hour(self) -> int:
        return int(self.time_utc[11:13])

    def to_json(self) -> dict:
        return asdict(self)


def place(spot: Spot) -> Spot:
    """Fill distance and bearings when both grids are known."""
    if spot.grid and spot.rx_grid and _mh.is_grid(spot.grid) \
            and _mh.is_grid(spot.rx_grid):
        la1, lo1 = _mh.to_latlon(spot.grid)
        la2, lo2 = _mh.to_latlon(spot.rx_grid)
        spot.distance_km = round(distance_km(la1, lo1, la2, lo2), 1)
        spot.bearing_tx_to_rx = round(bearing_deg(la1, lo1, la2, lo2), 1)
        spot.bearing_rx_to_tx = round(bearing_deg(la2, lo2, la1, lo1), 1)
    return spot


class SpotParser:
    """Line parser for both formats; counts what it skipped and why."""

    def __init__(self, rx_call: str = "", rx_grid: str = "",
                 date: str | None = None, source: str = ""):
        self.rx_call = rx_call
        self.rx_grid = _mh.normalize(rx_grid) if rx_grid else ""
        self.date = date              # 'YYYY-MM-DD' for wsprd's dateless lines
        self.source = source
        self.good = 0
        self.skipped = 0
        self.last_error = ""

    def _skip(self, why: str) -> None:
        self.skipped += 1
        self.last_error = why

    def status(self) -> str:
        s = f"{self.good} spots read, {self.skipped} lines skipped"
        return s + (f"; the last: {self.last_error}" if self.last_error else "")

    def parse(self, line: str) -> Spot | None:
        tok = str(line or "").strip().split()
        if len(tok) < 6:
            if tok:
                self._skip("too few fields for a spot")
            return None
        i = 0
        date = self.date
        if re.fullmatch(r"\d{6}", tok[0]) and re.fullmatch(r"\d{4}|\d{6}", tok[1]):
            d = tok[0]
            date = f"20{d[0:2]}-{d[2:4]}-{d[4:6]}"
            hhmm = tok[1][:4]
            i = 2
        elif re.fullmatch(r"\d{4}", tok[0]):
            hhmm = tok[0]
            i = 1
        else:
            self._skip("no time at the start of the line")
            return None
        fi = None
        for j in range(i, min(len(tok), i + 6)):
            if _FLOAT.match(tok[j]) and len(tok[j].split(".")[1]) >= 4 \
                    and 0.05 < float(tok[j]) < 1000.0:
                fi = j
                break
        if fi is None:
            self._skip("no frequency (MHz with four or more decimals)")
            return None
        nums = tok[i:fi]
        if len(nums) not in (2, 3) or not all(_FLOAT.match(n) or _INT.match(n)
                                              for n in nums):
            self._skip("the SNR and time offset are not where they belong")
            return None
        snr, dt = int(round(float(nums[-2]))), float(nums[-1])
        if not (-60 <= snr <= 60) or abs(dt) > 10:
            self._skip(f"an SNR of {snr} or a DT of {dt} is not a WSPR decode")
            return None
        freq_hz = float(tok[fi]) * 1e6
        rest = tok[fi + 1:]
        drift = None
        if rest and _INT.match(rest[0]) and len(rest) > 1 and _CALL.match(rest[1].upper()):
            drift = int(rest[0])            # wsprd console: drift, then message
            rest = rest[1:]
        if not rest or not _CALL.match(rest[0].upper()):
            self._skip("no callsign where the message starts")
            return None
        call = rest[0].upper()
        grid, power = "", None
        k = 1
        if k < len(rest) and _mh.is_grid(rest[k]) and len(rest[k]) in (4, 6):
            grid = _mh.normalize(rest[k])
            k += 1
        if k < len(rest) and _INT.match(rest[k]) and 0 <= int(rest[k]) <= 60:
            power = int(rest[k])
            k += 1
        else:
            self._skip("no power (dBm) after the call")
            return None
        if drift is None and k < len(rest) and _INT.match(rest[k]) \
                and abs(int(rest[k])) <= 9:
            drift = int(rest[k])
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        t = f"{date}T{hhmm[:2]}:{hhmm[2:4]}:00Z"
        sp = Spot(time_utc=t, snr_db=snr, dt_s=dt, freq_hz=freq_hz, call=call,
                  grid=grid, power_dbm=power, drift_hz=drift,
                  band=band_of(freq_hz), rx_call=self.rx_call,
                  rx_grid=self.rx_grid, source=self.source)
        self.good += 1
        return place(sp)


def parse_lines(lines, rx_call: str = "", rx_grid: str = "",
                date: str | None = None, source: str = "") -> list[Spot]:
    p = SpotParser(rx_call, rx_grid, date, source)
    return [s for s in (p.parse(x) for x in lines) if s is not None]


def parse_file(path, rx_call: str = "", rx_grid: str = "",
               date: str | None = None) -> tuple[list[Spot], str]:
    """Spots from ALL_WSPR.TXT or a saved wsprd output. -> (spots, status)."""
    p = SpotParser(rx_call, rx_grid, date, source=Path(path).name)
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = p.parse(line)
            if s is not None:
                out.append(s)
    return out, p.status()


# ---------------------------------------------------------------------------
# The decoder, at arm's length
# ---------------------------------------------------------------------------
def prepare_wav(audio, fs: float, path, seconds: float = 120.0) -> Path:
    """The Kiwi's audio as wsprd wants it: 12 000 Hz, mono, 16-bit, up to two
    minutes (one WSPR cycle)."""
    from scipy.signal import resample_poly
    from fractions import Fraction
    if np.iscomplexobj(np.asarray(audio)):
        raise ValueError("wsprd takes real audio (the Kiwi's USB audio), not IQ")
    x = np.asarray(audio, dtype=np.float64).ravel()
    fr = Fraction(12000, int(round(float(fs)))).limit_denominator(1000)
    y = resample_poly(x, fr.numerator, fr.denominator) if fr != 1 else x
    y = y[:int(12000 * seconds)]
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    y = (y / peak * 0.5 * 32767) if peak > 0 else y
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(12000)
        w.writeframes(np.round(y).astype("<i2").tobytes())
    return p


def wsprd_args(wav, *, wsprd_path: str, dial_mhz: float, out_dir,
               deep: bool = False, cycles: int | None = None) -> list[str]:
    """wsprd's command line: -a (where it writes its files), -f (the dial
    frequency in MHz), optionally -d (deeper search) and -C (cycles)."""
    args = [str(wsprd_path), "-a", str(out_dir), "-f", f"{float(dial_mhz):.6f}"]
    if deep:
        args.append("-d")
    if cycles:
        args += ["-C", str(int(cycles))]
    return args + [str(wav)]


def run_wsprd(wav, *, wsprd_path: str | None, dial_mhz: float, out_dir,
              rx_call: str = "", rx_grid: str = "", date: str | None = None,
              runner: Callable[[list], tuple] | None = None,
              timeout_s: float = 180.0) -> tuple[list[Spot], str]:
    """Decode one two-minute WAV with wsprd as a SEPARATE PROCESS. -> (spots,
    words). A missing decoder is a sentence, not a crash."""
    exe = None
    if wsprd_path:
        p = Path(wsprd_path)
        exe = str(p) if p.exists() else shutil.which(str(wsprd_path))
    if not exe and runner is None:
        return [], ("wsprd is not configured or not found — set its path (it "
                    "comes with WSJT-X; it is GPL and runs as its own program, "
                    "the toolkit never includes it). Nothing was decoded.")
    args = wsprd_args(wav, wsprd_path=exe or str(wsprd_path), dial_mhz=dial_mhz,
                      out_dir=out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    try:
        if runner is not None:
            rc, text = runner(args)
        else:
            pr = subprocess.run(args, capture_output=True, text=True,
                                timeout=timeout_s, cwd=str(out_dir))
            rc, text = pr.returncode, (pr.stdout or "") + (pr.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], f"wsprd could not run: {exc}"
    p = SpotParser(rx_call, rx_grid, date, source=f"wsprd {Path(str(wav)).name}")
    spots = [s for s in (p.parse(x) for x in str(text).splitlines()) if s]
    words = (f"wsprd decoded {len(spots)} spot{'s' if len(spots) != 1 else ''}"
             + ("" if rc == 0 else f" (it ended with code {rc})"))
    return spots, words
