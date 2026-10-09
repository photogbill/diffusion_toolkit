# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""ESP32 Wi-Fi CSI as a sensor: parse it, read it from USB serial, replay it
(plan §4.I — I1 vital signs, I2 presence and motion).

The ATK path for passive sensing is an ESP32 pair — a transmitter and a
receiver, a few dollars each — streaming Channel State Information over USB
serial as a SENSOR: no SDR, and no laptop Wi-Fi card (Windows cannot extract
CSI from one). PulseFi (2025, arXiv 2510.24744) does vital signs with
amplitude-only CSI from exactly this hardware.

THE LINES, AS IMPLEMENTED. The receiving ESP32 prints one line per received
frame. Two firmwares are common, and both are handled; a header line, when
the firmware prints one, wins over everything below.

1. Espressif's esp-csi examples (`csi_recv`, `csi_recv_router`), on the
   ESP32 / S2 / S3 / C3 — 24 fields, then the data, quoted, comma-separated:

       type,id,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,
       aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,
       secondary_channel,local_timestamp,ant,sig_len,rx_state,len,first_word,
       data
       CSI_DATA,0,aa:bb:cc:dd:ee:ff,-38,11,1,7,0,1,1,1,0,0,0,-96,0,6,0,
       1234567,0,47,0,128,0,"[0,0,3,4,…]"

   (Newer esp-csi builds for the C5/C6 print a different column set; they
   print their header first, and the header is used.)

2. The ESP32-CSI-Tool (S. M. Hernandez) — a ROLE in the second field and the
   host clock in two extra fields, the data space-separated:

       type,role,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,
       not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,
       channel,secondary_channel,local_timestamp,ant,sig_len,rx_state,
       real_time_set,real_timestamp,len,CSI_DATA
       CSI_DATA,STA,aa:bb:cc:dd:ee:ff,-38,…,1234567,0,47,0,1,1696000000,128,
       [0 0 3 4 …]

THE DATA. Per Espressif's ESP-IDF documentation (Wi-Fi Channel State
Information): each subcarrier is TWO SIGNED BYTES, the IMAGINARY part
followed by the REAL part — so the pair [3, 4] is 4 + 3j. Amplitude does not
care about the order; phase does, so it is kept right. When `first_word`
(first_word_invalid) is 1, the first four bytes are invalid by a hardware
limitation and the first two subcarriers are zeroed. The subcarriers are as
the ESP32 reports them (LLTF, then HT-LTF and STBC-HT-LTF when present);
guard and null subcarriers read zero and fall out at subcarrier selection.

DEFENSIVE BY DESIGN. A serial console tears lines, interleaves boot logs and
drops bytes. A line whose `len` field disagrees with its data, whose bracket
never closes, or that is not CSI at all is SKIPPED and counted, with the last
reason kept in words (`CsiParser.status`). Timestamps are the ESP32's own
32-bit microsecond clock, unwrapped.

THE READER is a plain iterator over pyserial (imported lazily — a missing
pyserial is a sentence, not a crash), never a GUI thread: ATK runs it in its
own worker. `replay` reads a saved console log the same way.

LIMITS. The ESP32's AGC moves the amplitude; the vitals pipeline removes the
mean per window but cannot undo a gain step inside one. CSI arrives at the
rate frames arrive, so `to_matrix` resamples to a uniform grid and reports
the gaps instead of hiding them.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

#: esp-csi csi_recv field order (before the data).
ESPRESSIF_FIELDS = ("type", "id", "mac", "rssi", "rate", "sig_mode", "mcs",
                    "bandwidth", "smoothing", "not_sounding", "aggregation",
                    "stbc", "fec_coding", "sgi", "noise_floor", "ampdu_cnt",
                    "channel", "secondary_channel", "local_timestamp", "ant",
                    "sig_len", "rx_state", "len", "first_word")
#: ESP32-CSI-Tool field order (before the data).
CSI_TOOL_FIELDS = ("type", "role", "mac", "rssi", "rate", "sig_mode", "mcs",
                   "bandwidth", "smoothing", "not_sounding", "aggregation",
                   "stbc", "fec_coding", "sgi", "noise_floor", "ampdu_cnt",
                   "channel", "secondary_channel", "local_timestamp", "ant",
                   "sig_len", "rx_state", "real_time_set", "real_timestamp",
                   "len")
ROLES = ("AP", "STA", "PASSIVE", "STATION", "SOFTAP")
DATA_NAMES = ("data", "csi_data", "csi")

_MAC = re.compile(r"^[0-9a-fA-F]{2}(:[0-9a-fA-F]{2}){5}$")
#: The ESP32's rx_ctrl timestamp is 32-bit microseconds.
TS_WRAP = 1 << 32
#: The esp-csi examples' console rate; match your sdkconfig.
DEFAULT_BAUD = 921_600


@dataclass
class CsiFrame:
    csi: np.ndarray                 # complex64, one per subcarrier
    mac: str = ""
    rssi: int | None = None
    channel: int | None = None
    noise_floor: int | None = None
    t_us: int | None = None         # the ESP32's local clock (µs, 32-bit)
    host_t: float | None = None     # when the host read the line
    variant: str = ""
    fields: dict = field(default_factory=dict)

    @property
    def amplitude(self) -> np.ndarray:
        return np.abs(self.csi).astype(np.float32)

    @property
    def n_subcarriers(self) -> int:
        return int(self.csi.size)


def _int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def iq_pairs_to_complex(values, order: str = "imag_real") -> np.ndarray:
    """Signed bytes in pairs -> complex64. Espressif's order is imaginary,
    then real (ESP-IDF, Wi-Fi Channel State Information)."""
    v = np.asarray(values, dtype=np.float32)
    if v.size % 2:
        raise ValueError("CSI data has an odd number of bytes")
    a, b = v[0::2], v[1::2]
    if order == "imag_real":
        return (b + 1j * a).astype(np.complex64)
    if order == "real_imag":
        return (a + 1j * b).astype(np.complex64)
    raise ValueError(f"unknown CSI byte order {order!r}")


class CsiParser:
    """Stateful line parser: remembers a header when the firmware prints one,
    and counts what it skipped and why."""

    def __init__(self, order: str = "imag_real"):
        self.order = order
        self.header: tuple | None = None
        self.good = 0
        self.skipped = 0
        self.bad = 0
        self.last_error = ""
        self.reasons: Counter = Counter()

    def _fail(self, why: str) -> None:
        self.bad += 1
        self.last_error = why
        self.reasons[why.split(" (")[0]] += 1

    def status(self) -> str:
        s = (f"{self.good} CSI frames read, {self.bad} damaged lines skipped, "
             f"{self.skipped} other lines ignored")
        if self.last_error:
            s += f"; the last damaged line: {self.last_error}"
        return s

    def parse(self, line, host_t: float | None = None) -> CsiFrame | None:
        if isinstance(line, (bytes, bytearray)):
            line = bytes(line).decode("utf-8", errors="replace")
        s = str(line).strip()
        if not s:
            return None
        low = s.lower()
        m = re.search(r"(^|\s)type,", low)
        if m and "CSI_DATA" not in s and "[" not in s:
            names = tuple(p.strip() for p in s[m.end() - len("type,"):].split(","))
            self.header = tuple(n for n in names if n.lower() not in DATA_NAMES)
            self.skipped += 1
            return None
        k = s.find("CSI_DATA")
        if k < 0:
            self.skipped += 1          # boot log, a prompt, anything else
            return None
        s = s[k:]
        a = s.rfind("[")
        if a < 0:
            self._fail("no CSI data bracket (a torn line)")
            return None
        b = s.find("]", a)
        if b < 0:
            self._fail("the CSI data never closes (a torn line)")
            return None
        try:
            vals = [int(t) for t in re.split(r"[,\s]+", s[a + 1:b].strip()) if t]
        except ValueError:
            self._fail("the CSI data holds something that is not a number")
            return None
        if not vals:
            self._fail("the CSI data is empty")
            return None
        if any(v < -128 or v > 127 for v in vals):
            self._fail("a CSI byte is outside -128..127")
            return None
        meta = [p.strip().strip('"') for p in s[:a].rstrip().rstrip('"').rstrip(",").split(",")]
        if self.header and len(self.header) == len(meta):
            names, variant = self.header, "header"
        elif len(meta) > 1 and meta[1].upper() in ROLES:
            names, variant = CSI_TOOL_FIELDS, "esp32-csi-tool"
        elif len(meta) == len(ESPRESSIF_FIELDS):
            names, variant = ESPRESSIF_FIELDS, "esp-csi"
        else:
            names, variant = (), "unknown"
        fields = dict(zip(names, meta)) if names else {}
        if not fields:
            mac = next((m for m in meta if _MAC.match(m)), "")
            fields = {"mac": mac}
            if mac:
                i = meta.index(mac)
                if i + 1 < len(meta):
                    fields["rssi"] = meta[i + 1]
            if len(meta) >= 2 and _int(meta[-1]) is not None:
                if _int(meta[-1]) == len(vals):
                    fields["len"] = meta[-1]
                elif _int(meta[-2]) == len(vals):
                    fields["len"] = meta[-2]
                    fields["first_word"] = meta[-1]
        n_len = _int(fields.get("len"))
        if n_len is not None and n_len != len(vals):
            self._fail(f"the length field says {n_len} but the data has "
                       f"{len(vals)} values (a torn line)")
            return None
        if len(vals) % 2:
            self._fail("an odd number of CSI bytes (a torn line)")
            return None
        csi = iq_pairs_to_complex(vals, self.order)
        fw = _int(fields.get("first_word", fields.get("first_word_invalid")))
        if fw == 1 and csi.size >= 2:
            csi[:2] = 0
        ts = _int(fields.get("local_timestamp", fields.get("timestamp")))
        self.good += 1
        return CsiFrame(csi=csi, mac=str(fields.get("mac", "")).lower(),
                        rssi=_int(fields.get("rssi")),
                        channel=_int(fields.get("channel")),
                        noise_floor=_int(fields.get("noise_floor")),
                        t_us=ts, host_t=host_t, variant=variant, fields=fields)


def parse_line(line, header: tuple | None = None) -> CsiFrame | None:
    """One line, no state (a header can be passed in)."""
    p = CsiParser()
    p.header = header
    return p.parse(line)


def replay(path, parser: CsiParser | None = None) -> Iterator[CsiFrame]:
    """Frames from a saved console log, in order."""
    p = parser or CsiParser()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            fr = p.parse(line)
            if fr is not None:
                yield fr


def serial_frames(port: str = "", baud: int = DEFAULT_BAUD, *,
                  timeout: float = 1.0, max_frames: int | None = None,
                  max_seconds: float | None = None,
                  stop: Callable[[], bool] | None = None,
                  opener: Callable[[], object] | None = None,
                  parser: CsiParser | None = None,
                  log_to=None) -> Iterator[CsiFrame]:
    """CSI frames from the ESP32 on a USB serial port — a plain iterator for
    the host's worker (never a GUI thread). `opener()` may return any object
    with `readline()` and `close()` (tests, or a host-managed port).
    `log_to` (a path) keeps every raw line, so a session can be replayed.
    It ends at `max_frames`, at `max_seconds`, when `stop()` says so — and,
    when neither `stop` nor `max_seconds` is given, at the first read that
    times out with nothing received (a live session passes `stop`)."""
    if opener is None:
        try:
            import serial                      # pyserial
        except ImportError:
            raise RuntimeError(
                "pyserial is not installed in this environment, so the ESP32 "
                "CSI sensor cannot be read. It belongs in ATK's core "
                "environment (get_diffusion.bat installs it). Nothing was "
                "read.") from None
        if not port:
            raise ValueError("name the ESP32's serial port (for example COM5)")
        ser = serial.Serial(port, int(baud), timeout=float(timeout))
    else:
        ser = opener()
    p = parser or CsiParser()
    n = 0
    t0 = time.monotonic()
    fh = open(log_to, "a", encoding="utf-8") if log_to else None
    try:
        while True:
            if stop is not None and stop():
                break
            if max_seconds is not None and time.monotonic() - t0 >= max_seconds:
                break
            raw = ser.readline()
            if raw is None:
                break
            if isinstance(raw, (bytes, bytearray)):
                if not raw:
                    if max_seconds is None and stop is None:
                        break           # a timeout with nothing to wait for
                    continue
                text = bytes(raw).decode("utf-8", errors="replace")
            else:
                text = str(raw)
                if not text:
                    break
            if fh is not None:
                fh.write(text if text.endswith("\n") else text + "\n")
            fr = p.parse(text, host_t=time.time())
            if fr is None:
                continue
            yield fr
            n += 1
            if max_frames is not None and n >= max_frames:
                break
    finally:
        if fh is not None:
            fh.close()
        try:
            ser.close()
        except Exception:                                  # noqa: BLE001
            pass


def unwrap_us(ts) -> np.ndarray:
    """The ESP32's 32-bit microsecond clock, made monotonic."""
    t = np.asarray(ts, dtype=np.int64)
    if t.size == 0:
        return t.astype(np.float64)
    d = np.diff(t)
    wraps = np.cumsum(np.r_[0, (d < -(TS_WRAP // 2)).astype(np.int64)])
    return (t + wraps * TS_WRAP).astype(np.float64)


def to_matrix(frames, mac: str | None = None, n_sc: int | None = None
              ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Frames -> (t seconds from the first frame, H (frames, subcarriers)
    complex64, info). Only one link (`mac`, default the most common sender)
    and one subcarrier count (default the most common) are kept — frames of
    another format are counted in `info`, not mixed in."""
    fr = list(frames)
    if mac:
        fr = [f for f in fr if f.mac == mac.lower()]
    elif fr:
        mac = Counter(f.mac for f in fr).most_common(1)[0][0]
        fr = [f for f in fr if f.mac == mac]
    if not fr:
        raise ValueError("no CSI frames from that sender")
    want = n_sc or Counter(f.n_subcarriers for f in fr).most_common(1)[0][0]
    keep = [f for f in fr if f.n_subcarriers == want]
    info = {"mac": mac, "subcarriers": int(want), "frames": len(keep),
            "dropped_other_format": len(fr) - len(keep)}
    if all(f.t_us is not None for f in keep):
        t = unwrap_us([f.t_us for f in keep]) / 1e6
        info["clock"] = "esp32"
    elif all(f.host_t is not None for f in keep):
        t = np.array([f.host_t for f in keep], dtype=np.float64)
        info["clock"] = "host"
    else:
        raise ValueError("the frames carry no timestamps")
    t = t - t[0]
    H = np.stack([f.csi for f in keep]).astype(np.complex64)
    order = np.argsort(t, kind="stable")
    return t[order], H[order], info


def resample_uniform(t, X, fs: float, max_gap_s: float = 0.5
                     ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Linear interpolation of each column of X (amplitude, or complex) onto
    a uniform grid at `fs`. -> (t_grid, X_grid, info with the gaps longer
    than `max_gap_s` and the fraction of time they cover). Gaps are filled
    by interpolation AND reported — a flat line across a gap is not a
    measurement, and the vitals pipeline drops windows that hold one."""
    t = np.asarray(t, dtype=np.float64)
    X = np.asarray(X)
    if t.size < 2:
        raise ValueError("at least two frames are needed")
    grid = np.arange(t[0], t[-1], 1.0 / float(fs))
    if np.iscomplexobj(X):
        out = np.empty((grid.size, X.shape[1]), dtype=np.complex64)
        for k in range(X.shape[1]):
            out[:, k] = (np.interp(grid, t, X[:, k].real)
                         + 1j * np.interp(grid, t, X[:, k].imag))
    else:
        out = np.empty((grid.size, X.shape[1]), dtype=np.float32)
        for k in range(X.shape[1]):
            out[:, k] = np.interp(grid, t, X[:, k])
    dt = np.diff(t)
    gaps = [(float(t[i]), float(t[i + 1])) for i in np.flatnonzero(dt > max_gap_s)]
    covered = sum(b - a for a, b in gaps)
    span = float(t[-1] - t[0])
    return grid - grid[0], out, {"fs": float(fs), "gaps": gaps,
                                 "gap_fraction": covered / span if span > 0 else 0.0,
                                 "mean_rate_hz": (t.size - 1) / span if span > 0 else 0.0}


def save_log(frames, path) -> Path:
    """Write frames back as esp-csi lines (with a header) — for building a
    replayable test or a share-able session from parsed frames."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(",".join(ESPRESSIF_FIELDS) + ",data\n")
        for i, fr in enumerate(frames):
            c = fr.csi
            pairs = np.empty(c.size * 2, dtype=np.int64)
            pairs[0::2] = np.clip(np.round(c.imag), -128, 127)
            pairs[1::2] = np.clip(np.round(c.real), -128, 127)
            vals = {"type": "CSI_DATA", "id": i, "mac": fr.mac or "00:00:00:00:00:00",
                    "rssi": fr.rssi if fr.rssi is not None else 0,
                    "channel": fr.channel if fr.channel is not None else 0,
                    "noise_floor": fr.noise_floor if fr.noise_floor is not None else 0,
                    "local_timestamp": fr.t_us if fr.t_us is not None else 0,
                    "len": pairs.size, "first_word": 0}
            row = [str(vals.get(n, 0)) for n in ESPRESSIF_FIELDS]
            f.write(",".join(row) + ',"[' + ",".join(str(int(v)) for v in pairs)
                    + ']"\n')
    return p
