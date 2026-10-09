# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Raw I/Q bytes <-> complex samples, for every datatype ATK records.

    cu8      RTL-SDR, KrakenSDR — unsigned offset-binary bytes, 127.5 is zero
    ci8      HackRF — signed bytes
    ci16     bladeRF (SC16 Q11: full scale 2048), SpyServer, KiwiSDR (32768)
    cf32     complex64 — cuts, synthetic data, anything already processed

Full scale maps to 1.0, so a level in dBFS means the same thing for every
receiver — except where the hardware itself differs, which is the point of
receiver profiles. The Q11 scaling is the one trap: the bytes of a bladeRF
file are identical to any ci16_le file, and only `atk:datatype = ci16q11`
says the full scale is 2048, not 32768. Read it with the wrong scale and
every level is 24 dB low.
"""

from __future__ import annotations

import numpy as np

BYTES_PER_SAMPLE = {"cu8": 2, "ci8": 2, "ci16": 4, "ci16q11": 4, "cf32": 8}

_ALIASES = {"cu8_le": "cu8", "cs8": "ci8", "ci8_le": "ci8",
            "ci16_le": "ci16", "cs16_le": "ci16", "sc16": "ci16",
            "sc16q11": "ci16q11", "cf32_le": "cf32", "cfloat32": "cf32"}


def norm_dt(datatype: str) -> str:
    d = str(datatype or "").strip().lower()
    d = _ALIASES.get(d, d)
    if d not in BYTES_PER_SAMPLE:
        raise ValueError(f"unsupported I/Q datatype {datatype!r}")
    return d


def bytes_per_sample(datatype: str) -> int:
    return BYTES_PER_SAMPLE[norm_dt(datatype)]


def full_scale(datatype: str) -> float:
    d = norm_dt(datatype)
    return {"cu8": 127.5, "ci8": 128.0, "ci16": 32768.0, "ci16q11": 2048.0,
            "cf32": 1.0}[d]


def to_complex(raw, datatype: str) -> np.ndarray:
    """bytes / bytearray / memoryview / ndarray -> complex64, unit full scale."""
    d = norm_dt(datatype)
    buf = raw if isinstance(raw, np.ndarray) else np.frombuffer(raw, np.uint8)
    if d == "cf32":
        a = np.ascontiguousarray(buf).view(np.complex64) \
            if buf.dtype != np.complex64 else buf
        return a.astype(np.complex64, copy=False)
    if d == "cu8":
        u = np.ascontiguousarray(buf).view(np.uint8)
        u = u[: (u.size // 2) * 2].astype(np.float32)
        f = (u - 127.5) / 127.5
    elif d == "ci8":
        s = np.ascontiguousarray(buf).view(np.int8)
        s = s[: (s.size // 2) * 2].astype(np.float32)
        f = s / 128.0
    else:   # ci16 / ci16q11, little-endian
        s = np.ascontiguousarray(buf).view("<i2")
        s = s[: (s.size // 2) * 2].astype(np.float32)
        f = s / full_scale(d)
    out = np.empty(f.size // 2, dtype=np.complex64)
    out.real = f[0::2]
    out.imag = f[1::2]
    return out


def from_complex(x, datatype: str) -> bytes:
    """complex -> raw bytes of `datatype`, clipped to full scale. Used for the
    cabled loop's transmit files and for tests that need a 'recorded' file."""
    d = norm_dt(datatype)
    x = np.asarray(x, dtype=np.complex64)
    if d == "cf32":
        return x.tobytes()
    inter = np.empty(x.size * 2, dtype=np.float32)
    inter[0::2] = x.real
    inter[1::2] = x.imag
    fs = full_scale(d)
    if d == "cu8":
        v = np.clip(np.round(inter * fs + 127.5), 0, 255).astype(np.uint8)
    elif d == "ci8":
        v = np.clip(np.round(inter * fs), -128, 127).astype(np.int8)
    else:
        lim = 2047 if d == "ci16q11" else 32767
        v = np.clip(np.round(inter * fs), -lim - 1, lim).astype("<i2")
    return v.tobytes()


def deinterleave(x: np.ndarray, channels: int) -> np.ndarray:
    """Sample-interleaved multi-channel complex -> (channels, n)."""
    c = int(channels)
    if c <= 1:
        return np.asarray(x)[None, :]
    n = (x.size // c) * c
    return np.asarray(x[:n]).reshape(-1, c).T.copy()


def power_dbfs(x) -> float:
    """Mean power in dB relative to full scale (a full-scale tone is 0)."""
    x = np.asarray(x)
    p = float(np.mean(np.abs(x) ** 2)) if x.size else 0.0
    return 10.0 * np.log10(p) if p > 0 else -np.inf


def clipped_fraction(x, datatype: str) -> float:
    """Fraction of samples at the converter's rails — the ADC-overload check
    the cabled loop runs before raising transmit gain."""
    d = norm_dt(datatype)
    x = np.asarray(x)
    if d == "cf32" or x.size == 0:
        return 0.0
    lim = {"cu8": 127.5 / 127.5, "ci8": 127.0 / 128.0, "ci16": 32767 / 32768,
           "ci16q11": 2047 / 2048}[d] * 0.995
    hit = (np.abs(x.real) >= lim) | (np.abs(x.imag) >= lim)
    return float(np.mean(hit))
