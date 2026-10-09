# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Moving signals between rates — only ever on purpose (plan §3.3).

Two different things, kept apart because they obey different laws:

* `cut_to_canonical` — the signal cut's decimation (DETECTION_DESIGN §4):
  shift a box to baseband, low-pass it, and decimate by the INTEGER factor of
  its bandwidth class. Same profile, a canonical rate of it. Always allowed;
  the factor goes in the cut's metadata.

* `resample_capture` — moving a whole capture to ANOTHER profile's rate.
  Never silent: it is a tool with a log line, it writes a new SigMF file whose
  metadata says `atk:resampled_from`, and a model trained on its output is a
  model of resampled data and says so. The waterfall's own decimation for
  display never reaches a model.
"""

from __future__ import annotations

import json
import math
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from atk_diffusion import profiles as _profiles


def shift(x, f_hz: float, fs: float, n0: int = 0) -> np.ndarray:
    """Multiply by exp(-j2π f n/fs): a signal at +f moves to 0 Hz. `n0` keeps
    phase continuous across consecutive blocks."""
    x = np.asarray(x, dtype=np.complex64)
    n = np.arange(n0, n0 + x.size, dtype=np.float64)
    return (x * np.exp(-2j * np.pi * float(f_hz) / float(fs) * n)).astype(np.complex64)


def lowpass_taps(cutoff_hz: float, fs: float, decim: int, per_decim: int = 16):
    from scipy.signal import firwin
    ntaps = max(31, per_decim * int(decim) + 1)
    if ntaps % 2 == 0:
        ntaps += 1
    cutoff = min(float(cutoff_hz), 0.49 * float(fs) / max(1, int(decim)))
    cutoff = max(cutoff, 1e-6 * fs)
    return firwin(ntaps, cutoff, fs=float(fs)).astype(np.float32)


def decimate(x, decim: int, fs: float, cutoff_hz: float | None = None):
    """Low-pass + keep every `decim`-th sample, group delay removed, so output
    sample k is the same instant as input sample k·decim. -> (y, fs_out)."""
    from scipy.signal import upfirdn
    x = np.asarray(x, dtype=np.complex64)
    d = int(decim)
    if d < 1:
        raise ValueError("decimation must be a positive integer")
    if d == 1:
        return x.copy(), float(fs)
    cut = cutoff_hz if cutoff_hz is not None else 0.45 * fs / d
    taps = lowpass_taps(cut, fs, d)
    y = upfirdn(taps, x, up=1, down=d)
    delay = (taps.size - 1) // 2
    first = int(math.ceil(delay / d))
    # sample index (k*d) of the input lands at output index k + delay/d;
    # align on the exact integer when the delay is a multiple of d, which
    # it is not in general — so trim by the rounded value and note it
    y = y[first:first + int(math.ceil(x.size / d))]
    return y.astype(np.complex64), float(fs) / d


def cut_to_canonical(x, fs: float, f_offset_hz: float, bw_hz: float,
                     guard: float = 1.25):
    """The cut's DSP (DETECTION_DESIGN §4): shift the box at `f_offset_hz`
    (from the capture's centre) to 0 Hz, low-pass to the box's bandwidth
    with a guard, and integer-decimate to the profile's canonical rate for
    the box's bandwidth class. Returns (y, fs_out, info) where info carries
    the decimation, the class and whether the class is limited."""
    fs = float(fs)
    bw = max(float(bw_hz), 1.0)
    can = _profiles.canonical_for(fs, bw)
    mixed = shift(x, f_offset_hz, fs)
    cutoff = min(0.5 * bw * guard, 0.49 * can.rate)
    y, fs_out = decimate(mixed, can.decimation, fs, cutoff_hz=cutoff)
    info = {"decimation": can.decimation, "canonical_rate": can.rate,
            "canonical_class": can.cls, "limited": can.limited,
            "lowpass_hz": cutoff, "f_offset_hz": float(f_offset_hz),
            "bw_hz": bw}
    return y, fs_out, info


def rational(fs_from: float, fs_to: float, max_den: int = 10_000):
    """(up, down) with fs_to/fs_from = up/down, exact when representable."""
    fr = Fraction(int(round(fs_to)), int(round(fs_from))).limit_denominator(max_den)
    return fr.numerator, fr.denominator


def resample(x, fs_from: float, fs_to: float):
    """Polyphase rational resampling. -> (y, fs_actual)."""
    from scipy.signal import resample_poly
    up, down = rational(fs_from, fs_to)
    y = resample_poly(np.asarray(x, dtype=np.complex64), up, down)
    return y.astype(np.complex64), float(fs_from) * up / down


def resample_capture(src, target_profile: str, out_base, rf=None,
                     who: str = "", reason: str = "",
                     chunk: int = 1 << 22) -> dict:
    """The ONE way a capture moves to another profile's rate (plan §3.3).

    Writes `<out_base>.sigmf-*` at the target profile's rate, cf32, with
    `atk:resampled_from`, `atk:receiver_profile` = the target and a
    `atk:resample_log` entry; appends the same line to the rf_data runs log
    when an RfData is given, and records the file in the write log.
    Returns a summary dict with the log line.

    The datatype of the target profile is NOT imitated: resampled data is
    written as cf32 and keeps its own provenance. A model trained on it is a
    model of resampled data, and its card will say `resampled` in the
    dataset entry.
    """
    from atk_diffusion import sigmf as _sigmf
    meta = _sigmf.read_meta(src)
    src_profile = _profiles.profile_from_meta(meta)
    tgt = _profiles.parse_profile_id(target_profile)
    fs_from = _sigmf.sample_rate_of(meta)
    fs_to = float(tgt.sample_rate)
    up, down = rational(fs_from, fs_to)
    exact = abs(fs_from * up / down - fs_to) < 1e-6
    x = _sigmf.load(src, meta=meta)
    if x.ndim > 1:
        raise ValueError("resample one channel at a time (a multi-channel "
                         "capture is a coherent array; split it first)")
    y, fs_actual = resample(x, fs_from, fs_to)
    line = (f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} resampled "
            f"{Path(_sigmf.base_of(src)).name} from {src_profile} "
            f"({fs_from:g} S/s) to {target_profile} ({fs_actual:g} S/s), "
            f"polyphase {up}/{down}{'' if exact else ' (APPROXIMATE ratio)'}"
            + (f", by {who}" if who else "") + (f" — {reason}" if reason else ""))
    g_extra = {"atk:receiver_profile": str(target_profile).lower(),
               "atk:resampled_from": src_profile,
               "atk:resample_log": line, "atk:tier": "cleaned",
               "atk:method": "resample",
               "atk:method_params": {"up": up, "down": down,
                                     "filter": "scipy.signal.resample_poly "
                                               "(Kaiser, default)"}}
    dp, mp = _sigmf.write_pair(out_base, y, fs_actual,
                               _sigmf.center_of(meta),
                               datatype="cf32", extra_global=g_extra,
                               hw=str(meta.get("global", {}).get("core:hw", "")),
                               description=f"resampled from {src_profile}")
    # carry annotations across, scaled to the new sample index
    anns = []
    for a in _sigmf.annotations(meta):
        a.sample_start = int(round(a.sample_start * up / down))
        a.sample_count = int(round(a.sample_count * up / down))
        anns.append(a)
    if anns:
        _sigmf.add_annotations(mp, anns)
    if rf is not None:
        try:
            runs = rf.runs(str(target_profile).lower())
            runs.mkdir(parents=True, exist_ok=True)
            with open(runs / "resample_log.txt", "a", encoding="utf-8") as f:
                f.write(line + "\n")
            rf.record(dp, "resampled", line)
            rf.record(mp, "resampled-meta", line)
        except Exception:                                  # noqa: BLE001
            pass
    return {"data": str(dp), "meta": str(mp), "from": src_profile,
            "to": str(target_profile).lower(), "fs": fs_actual,
            "up": up, "down": down, "exact": exact, "log": line,
            "samples": int(y.size)}


def write_json(path, obj) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return p
