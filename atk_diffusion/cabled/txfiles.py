# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Transmit files for the cabled calibration loop (plan §3.5, §4.A6).

The loop plays a KNOWN signal set through a cable and attenuators into the
receiver under test, so the recording is real-receiver-impaired data with
perfect labels. This module makes the file the transmitter plays and the
ground-truth manifest the labels come from.

WHAT IS IN A FILE, in time order:

    [start marker] gap [signal 1] gap [signal 2] … gap [end marker]

* The MARKER is how a recording is lined up with the file: a known waveform
  found by cross-correlation (`cabled.loop.align_labels`). Two kinds:
  `chirp` — an up-chirp then a down-chirp; a frequency offset between the
  radios shifts the two correlation peaks in opposite directions, so their
  mean is the true start and their difference measures the offset — robust
  to the ±20 ppm a HackRF's oscillator allows (the default); `pn` — a
  maximal-length sequence as BPSK chips, correlated in 32 segments and
  combined non-coherently, which tolerates an offset of roughly a quarter
  cycle per segment (about ±5 kHz at the defaults) — use it when the radios
  are locked or calibrated. The marker is defined by PARAMETERS in the
  manifest and regenerated at
  the receiver's rate when aligning — nothing is resampled. The END marker
  measures the clock drift between the two radios.
* The SIGNALS come from the toolkit's native generator
  (`atk_diffusion.synth.native.waveform`, another part of the toolkit,
  imported lazily: the clean signal of a class at the TRANSMITTER's rate,
  with its measured 99 % bandwidth and exact symbol rate); when it is not
  available, or does not make that class, a minimal local generator (PSK
  with root-raised-cosine shaping, FSK/C4FM, AM, FM, OFDM, tone, noise) is
  used, and the manifest says which made each signal and why.
* SCALING: the whole file is scaled so its largest I or Q sample is
  `backoff` of full scale (0.7 by default, 3 dB of headroom) — the DAC never
  clips; the manifest records the scale, the peak and the PAPR.

FORMATS — exactly what the tools read:

    hackrf    cs8: signed 8-bit I,Q interleaved     hackrf_transfer -t <file>
    bladerf*  SC16 Q11: int16 LE I,Q, ±2048          bladeRF-cli: tx config
                                                     file=<file> format=bin

THE MANIFEST (`<name>.manifest.json`) is the ground truth: transmitter, rate,
format, the file's SHA-256, the marker's parameters and positions, and each
signal's class, start, length, frequency offset from the TX centre,
bandwidth and symbol rate. `cabled.loop.align_labels` turns it into SigMF
annotations on the recording.

LIMITS. The local fallback generators are textbook waveforms, not
protocol-exact ones (a C4FM here is 4FSK at 4800 symbols/s with P25's
deviations, not a P25 frame). Nothing here transmits — `cabled.loop` plans
and runs a transmission, and only under `cabled.safety`.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from atk_diffusion import profiles as _profiles
from atk_diffusion.dsp import iq as _iq

#: transmitter family -> (file format, datatype for dsp.iq, extension)
FORMATS = {"hackrf": ("cs8", "ci8", ".cs8"),
           "bladerf1": ("sc16q11", "ci16q11", ".bin"),
           "bladerf2": ("sc16q11", "ci16q11", ".bin")}

#: TX sample-rate limits by family (hackrf_transfer -h: 2–20 MHz; the
#: bladeRF boards from ATK's own radio table, atk/core/radios.py).
TX_RATES = {"hackrf": (2_000_000.0, 20_000_000.0),
            "bladerf1": (160_000.0, 40_000_000.0),
            "bladerf2": (520_834.0, 61_440_000.0)}

MARKERS = ("chirp", "pn")

#: Maximal-length LFSR feedback taps (Fibonacci form), degree -> taps. The
#: period of each is checked to be 2**degree - 1 by the tests.
MSEQ_TAPS = {5: (5, 3), 6: (6, 5), 7: (7, 6), 8: (8, 6, 5, 4), 9: (9, 5),
             10: (10, 7), 11: (11, 9), 12: (12, 6, 4, 1)}


@dataclass
class TxFile:
    path: Path
    manifest_path: Path
    manifest: dict

    def lines(self) -> list[str]:
        m = self.manifest
        return [f"{self.path.name}: {m['n_samples']} samples at "
                f"{m['tx_rate']:g} S/s ({m['duration_s']:.3f} s) for the "
                f"{m['transmitter']}, {m['format']}",
                f"{len(m['signals'])} signals between a {m['marker']['kind']} "
                f"start marker and " + ("an end marker" if m.get("end_marker")
                                        else "no end marker"),
                f"peak {m['peak']:.2f} of full scale, PAPR "
                f"{m['papr_db']:.1f} dB — the DAC does not clip"]


# ---------------------------------------------------------------------------
# Markers (parametric, regenerated at any rate)
# ---------------------------------------------------------------------------
def mseq(degree: int) -> np.ndarray:
    """A maximal-length sequence of ±1, length 2**degree − 1."""
    taps = MSEQ_TAPS.get(int(degree))
    if taps is None:
        raise ValueError(f"no m-sequence taps for degree {degree} "
                         f"(have {sorted(MSEQ_TAPS)})")
    n = int(degree)
    state = [1] * n
    out = np.empty(2 ** n - 1, dtype=np.int8)
    for i in range(out.size):
        bit = state[-1]
        out[i] = bit
        fb = 0
        for t in taps:
            fb ^= state[t - 1]
        state = [fb] + state[:-1]
    return (1 - 2 * out.astype(np.int8)).astype(np.float32)


def default_marker_params(kind: str, tx_rate: float,
                          rx_rate: float | None = None) -> dict:
    """A marker that fits inside both radios' bandwidths with room to spare."""
    fs = min(float(tx_rate), float(rx_rate or tx_rate))
    if kind == "chirp":
        return {"bandwidth_hz": 0.5 * fs, "duration_s": 0.005, "f_offset_hz": 0.0}
    if kind == "pn":
        return {"chip_rate_hz": 0.25 * fs, "degree": 10, "f_offset_hz": 0.0,
                "segments": 32}
    raise ValueError(f"unknown marker {kind!r} (one of {', '.join(MARKERS)})")


def chirp_parts(fs: float, params: dict, f_shift_hz: float = 0.0
                ) -> tuple[np.ndarray, np.ndarray]:
    """(up-chirp, down-chirp), each `duration_s` long, at `fs`."""
    B = float(params["bandwidth_hz"])
    T = float(params["duration_s"])
    f0 = float(params.get("f_offset_hz", 0.0)) + float(f_shift_hz)
    n = max(8, int(round(T * fs)))
    t = np.arange(n) / float(fs)
    up = np.exp(2j * np.pi * ((f0 - B / 2) * t + B * t * t / (2 * T)))
    dn = np.exp(2j * np.pi * ((f0 + B / 2) * t - B * t * t / (2 * T)))
    return up.astype(np.complex64), dn.astype(np.complex64)


def pn_waveform(fs: float, params: dict, f_shift_hz: float = 0.0) -> np.ndarray:
    chips = mseq(int(params.get("degree", 10)))
    rate = float(params["chip_rate_hz"])
    n = int(math.ceil(chips.size * float(fs) / rate))
    t = np.arange(n) / float(fs)
    idx = np.minimum((t * rate).astype(np.int64), chips.size - 1)
    f0 = float(params.get("f_offset_hz", 0.0)) + float(f_shift_hz)
    return (chips[idx] * np.exp(2j * np.pi * f0 * t)).astype(np.complex64)


def marker_waveform(kind: str, fs: float, params: dict,
                    f_shift_hz: float = 0.0) -> np.ndarray:
    if kind == "chirp":
        up, dn = chirp_parts(fs, params, f_shift_hz)
        return np.concatenate([up, dn])
    if kind == "pn":
        return pn_waveform(fs, params, f_shift_hz)
    raise ValueError(f"unknown marker {kind!r}")


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
def rrc(beta: float, sps: int, span: int = 8) -> np.ndarray:
    """Root-raised-cosine taps, unit energy."""
    n = np.arange(-span * sps, span * sps + 1, dtype=np.float64) / sps
    h = np.empty_like(n)
    for i, t in enumerate(n):
        if abs(t) < 1e-12:
            h[i] = 1.0 - beta + 4 * beta / np.pi
        elif beta > 0 and abs(abs(4 * beta * t) - 1.0) < 1e-9:
            h[i] = (beta / np.sqrt(2)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                                          + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = (np.sin(np.pi * t * (1 - beta)) + 4 * beta * t
                    * np.cos(np.pi * t * (1 + beta))) / (np.pi * t * (1 - (4 * beta * t) ** 2))
    return (h / np.sqrt(np.sum(h * h))).astype(np.float64)


_PSK = {"bpsk": 2, "qpsk": 4, "8psk": 8}


def _unit(x: np.ndarray) -> np.ndarray:
    p = float(np.mean(np.abs(x) ** 2)) if x.size else 0.0
    return (x / np.sqrt(p)).astype(np.complex64) if p > 0 else x.astype(np.complex64)


def local_signal(key: str, fs: float, n: int, rng, symbol_rate: float | None = None,
                 bandwidth_hz: float | None = None) -> tuple[np.ndarray, dict]:
    """The minimal local generator (see the module docstring).
    -> (x unit-power complex64 at `fs`, facts {symbol_rate_hz, bandwidth_hz})."""
    k = str(key).lower()
    fs = float(fs)
    if k in _PSK or k in ("16qam", "64qam", "ask4"):
        sr = float(symbol_rate or min(fs / 8.0, 25_000.0))
        sps = max(2, int(round(fs / sr)))
        sr = fs / sps
        nsym = int(math.ceil(n / sps)) + 16
        if k in _PSK:
            m = _PSK[k]
            sym = np.exp(2j * np.pi * rng.integers(0, m, nsym) / m)
        elif k == "ask4":
            sym = (2 * rng.integers(0, 4, nsym) - 3).astype(np.complex128)
        else:
            m = 4 if k == "16qam" else 8
            lv = 2 * np.arange(m) - (m - 1)
            sym = rng.choice(lv, nsym) + 1j * rng.choice(lv, nsym)
        up = np.zeros(nsym * sps, dtype=np.complex128)
        up[::sps] = sym
        x = np.convolve(up, rrc(0.35, sps), mode="same")[:n]
        return _unit(x), {"symbol_rate_hz": sr, "bandwidth_hz": 1.35 * sr}
    if k in ("2fsk", "gfsk", "c4fm_4800", "c4fm_2400", "dmr_4800", "4fsk"):
        if k.startswith("c4fm") or k in ("dmr_4800", "4fsk"):
            sr = float(symbol_rate or (2400.0 if k == "c4fm_2400" else 4800.0))
            levels = np.array([-1800.0, -600.0, 600.0, 1800.0])
        else:
            sr = float(symbol_rate or 9600.0)
            levels = np.array([-0.5 * sr, 0.5 * sr])
        sps = max(2, int(round(fs / sr)))
        nsym = int(math.ceil(n / sps)) + 1
        f = np.repeat(rng.choice(levels, nsym), sps)[:n]
        if k == "gfsk":
            g = np.exp(-0.5 * (np.arange(-2 * sps, 2 * sps + 1) / (0.5 * sps)) ** 2)
            f = np.convolve(f, g / g.sum(), mode="same")
        x = np.exp(2j * np.pi * np.cumsum(f) / fs)
        dev = float(np.max(np.abs(levels)))
        return _unit(x), {"symbol_rate_hz": fs / sps,
                          "bandwidth_hz": 2 * (dev + sr / 2)}
    t = np.arange(n) / fs
    if k in ("am",):
        x = 1.0 + 0.5 * np.sin(2 * np.pi * 1_000.0 * t)
        return _unit(x.astype(np.complex128)), {"symbol_rate_hz": None,
                                                "bandwidth_hz": 2_000.0}
    if k in ("nfm", "fm", "wfm"):
        dev = 2_500.0 if k == "nfm" else 75_000.0
        dev = min(dev, 0.2 * fs)
        x = np.exp(1j * (dev / 1_000.0) * np.sin(2 * np.pi * 1_000.0 * t))
        return _unit(x), {"symbol_rate_hz": None, "bandwidth_hz": 2 * (dev + 1_000.0)}
    if k == "tone":
        return np.ones(n, dtype=np.complex64), {"symbol_rate_hz": None,
                                                "bandwidth_hz": 0.0}
    if k == "noise":
        bw = float(bandwidth_hz or 0.2 * fs)
        w = rng.standard_normal(n + 512) + 1j * rng.standard_normal(n + 512)
        from scipy.signal import firwin, lfilter
        taps = firwin(129, min(0.49 * fs, bw / 2), fs=fs)
        x = lfilter(taps, 1.0, w)[512:512 + n]
        return _unit(x), {"symbol_rate_hz": None, "bandwidth_hz": bw}
    if k in ("ofdm", "ofdm_lte"):
        nfft, ncp, used = 64, 16, 48
        nsym = int(math.ceil(n / (nfft + ncp))) + 1
        out = []
        for _ in range(nsym):
            X = np.zeros(nfft, dtype=np.complex128)
            idx = np.r_[1:used // 2 + 1, nfft - used // 2:nfft]
            X[idx] = (2 * rng.integers(0, 2, idx.size) - 1
                      + 1j * (2 * rng.integers(0, 2, idx.size) - 1))
            s = np.fft.ifft(X)
            out.append(np.r_[s[-ncp:], s])
        x = np.concatenate(out)[:n]
        return _unit(x), {"symbol_rate_hz": fs / (nfft + ncp),
                          "bandwidth_hz": fs * (used + 1) / nfft}
    raise ValueError(f"the local generator cannot make {key!r}; it makes "
                     "bpsk, qpsk, 8psk, 16qam, 64qam, ask4, 2fsk, gfsk, "
                     "c4fm_4800, c4fm_2400, dmr_4800, am, nfm, wfm, tone, "
                     "noise, ofdm")


def _native_signal(cls: str, fs: float, n: int, rng, symbol_rate=None,
                   bandwidth_hz=None, f_offset_hz: float = 0.0):
    """The toolkit's native generator (`synth.native.waveform`: the CLEAN
    signal of a class, unit power over its on-samples, at the carrier
    offset, with exact labels). -> ((x, label), "") or (None, why-not)."""
    try:
        from atk_diffusion.synth import native
    except ImportError as exc:
        return None, f"synth.native is not available ({exc})"
    fn = getattr(native, "waveform", None)
    if not callable(fn):
        return None, "synth.native has no waveform()"
    tries = []
    params = {}
    if symbol_rate:
        params["symbol_rate"] = float(symbol_rate)
    elif bandwidth_hz:
        params["bandwidth_hz"] = float(bandwidth_hz)
    if params:
        tries.append(params)
    tries.append(None)
    last = ""
    for prm in tries:
        try:
            x, lab = fn(cls, float(fs), int(n), rng,
                        carrier_offset_hz=float(f_offset_hz), params=prm)
        except Exception as exc:                           # noqa: BLE001
            last = str(exc)
            continue
        x = np.asarray(x, dtype=np.complex64).ravel()
        if x.size != int(n) or not np.all(np.isfinite(x)):
            last = "it returned the wrong length or non-finite samples"
            continue
        lab = dict(lab)
        if prm is None and params:
            lab["note"] = (f"the class's own {'symbol rate' if symbol_rate else 'bandwidth'}"
                           f" was used — it does not take that setting ({last})")
        return (x, lab), ""
    return None, f"synth.native could not make {cls}: {last}"


def make_signal(spec: dict, fs: float, rng) -> tuple[np.ndarray, dict]:
    """One signal from its spec: {class (the class table's name) or native
    (a local generator key), duration_s, f_offset_hz, symbol_rate_hz,
    bandwidth_hz, power_db, label}. -> (x at its offset, label facts).
    The native generator is used for every class it makes; the local one
    otherwise, and the label says which made it and why."""
    from atk_diffusion.detect import classes as _classes
    cls = str(spec.get("class") or spec.get("cls") or "")
    c = _classes.get(cls)
    n = max(16, int(round(float(spec.get("duration_s", 0.1)) * fs)))
    sr = spec.get("symbol_rate_hz")
    f_off = float(spec.get("f_offset_hz", 0.0))
    res, why = (None, "not a class with a native generator")
    if c is not None and c.native and not spec.get("native"):
        res, why = _native_signal(cls, fs, n, rng, sr, spec.get("bandwidth_hz"),
                                  f_off)
    if res is not None:
        x, nl = res
        generator = "atk_diffusion.synth.native"
        bw = nl.get("bandwidth_hz")
        rate = nl.get("symbol_rate_hz") or None
        extra = {"bursts": nl.get("bursts"), "native_label": {
            k: nl.get(k) for k in ("native", "params", "nominal_bandwidth_hz",
                                   "note") if nl.get(k) is not None}}
    else:
        key = str(spec.get("native") or (c.native if c else "") or cls)
        if not key:
            raise ValueError("a signal needs a class (from the class table) or "
                             "a native generator key")
        x, facts = local_signal(key, fs, n, rng, sr, spec.get("bandwidth_hz"))
        if f_off:
            t = np.arange(n) / fs
            x = (x * np.exp(2j * np.pi * f_off * t)).astype(np.complex64)
        generator = f"local fallback ({why})"
        bw = facts.get("bandwidth_hz")
        rate = facts.get("symbol_rate_hz")
        extra = {}
    gain = 10.0 ** (float(spec.get("power_db", 0.0)) / 20.0)
    if spec.get("bandwidth_hz") is not None and res is None:
        bw = float(spec["bandwidth_hz"])
    label = {"class": cls or str(spec.get("native")),
             "label": str(spec.get("label") or cls or spec.get("native")),
             "family": c.family if c else "unknown",
             "generator": generator, "f_offset_hz": f_off,
             "bandwidth_hz": None if bw is None else float(bw),
             "symbol_rate_hz": None if not rate else float(rate),
             "power_rel_db": float(spec.get("power_db", 0.0)), **extra}
    return (x * gain).astype(np.complex64), label


# ---------------------------------------------------------------------------
# Building the file
# ---------------------------------------------------------------------------
def build(out_dir, transmitter: str, tx_rate: float, signals: list[dict], *,
          rx_rate: float | None = None, marker: str = "chirp",
          marker_params: dict | None = None, gap_s: float = 0.02,
          end_marker: bool = True, backoff: float = 0.7, name: str = "",
          seed: int = 0, rf=None, tx_center_hz: float | None = None
          ) -> TxFile:
    """Write `<name><ext>` and `<name>.manifest.json` into `out_dir` (for
    a receiver profile `p`, the convention is `rf.cabled(p) / "tx"`)."""
    fam = str(transmitter).strip().lower()
    if fam not in FORMATS:
        raise ValueError(f"'{transmitter}' cannot be the loop's transmitter; "
                         "it is a bladeRF or a HackRF")
    lo, hi = TX_RATES[fam]
    fs = float(tx_rate)
    if not (lo <= fs <= hi):
        raise ValueError(f"a {_profiles.FAMILIES[fam]['label']} transmits at "
                         f"{lo:g}–{hi:g} S/s; {fs:g} is outside that")
    if not signals:
        raise ValueError("a transmit file needs at least one signal")
    if not (0.05 <= float(backoff) <= 0.95):
        raise ValueError("backoff is the peak as a fraction of full scale "
                         "(0.05–0.95)")
    if marker not in MARKERS:
        raise ValueError(f"unknown marker {marker!r} (one of {', '.join(MARKERS)})")
    mp = dict(marker_params or default_marker_params(marker, fs, rx_rate))
    rng = np.random.default_rng(int(seed))
    gap = np.zeros(int(round(float(gap_s) * fs)), dtype=np.complex64)
    mk = marker_waveform(marker, fs, mp)
    parts = [mk, gap]
    pos = mk.size + gap.size
    labels = []
    for spec in signals:
        x, lab = make_signal(spec, fs, rng)
        lab.update(start_sample=int(pos), count=int(x.size),
                   start_s=pos / fs, duration_s=x.size / fs)
        if lab.get("bursts"):
            # the generator's on-times (DMR's TDMA slots, ADS-B squitters):
            # the labels go on what was actually transmitted
            lab["bursts_s"] = [[(pos + int(b0)) / fs, int(bn) / fs]
                               for b0, bn in lab["bursts"]]
        if lab["bandwidth_hz"] is not None and \
                abs(lab["f_offset_hz"]) + lab["bandwidth_hz"] / 2 > 0.45 * fs:
            raise ValueError(f"{lab['label']} at {lab['f_offset_hz']:g} Hz, "
                             f"{lab['bandwidth_hz']:g} Hz wide, does not fit in "
                             f"the transmitter's {fs:g} S/s")
        labels.append(lab)
        parts += [x, gap]
        pos += x.size + gap.size
    end_pos = None
    if end_marker:
        end_pos = int(pos)
        parts.append(mk)
        pos += mk.size
    parts.append(gap)
    y = np.concatenate(parts).astype(np.complex64)
    peak_iq = float(np.max(np.maximum(np.abs(y.real), np.abs(y.imag))))
    scale = float(backoff) / peak_iq if peak_iq > 0 else 1.0
    y = (y * scale).astype(np.complex64)
    fmt, dt, ext = FORMATS[fam]
    raw = _iq.from_complex(y, dt)
    nm = name or f"loop_{fam}_{int(fs)}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{nm}{ext}"
    path.write_bytes(raw)
    back = _iq.to_complex(raw, dt)
    p_mean = float(np.mean(np.abs(back) ** 2))
    p_peak = float(np.max(np.abs(back) ** 2))
    manifest = {
        "what": "cabled-loop transmit file: ground truth for the labels",
        "transmitter": fam, "tx_rate": fs, "format": fmt, "datatype": dt,
        "file": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
        "n_samples": int(y.size), "duration_s": y.size / fs,
        "scale": scale, "backoff": float(backoff),
        "peak": float(np.max(np.maximum(np.abs(back.real), np.abs(back.imag)))),
        "rms_dbfs": 10 * math.log10(p_mean) if p_mean > 0 else None,
        "papr_db": 10 * math.log10(p_peak / p_mean) if p_mean > 0 else None,
        "clipped_fraction": _iq.clipped_fraction(back, dt),
        "marker": {"kind": marker, "params": mp, "start_sample": 0,
                   "length": int(mk.size), "start_s": 0.0},
        "end_marker": None if end_pos is None else
        {"start_sample": end_pos, "start_s": end_pos / fs},
        "gap_s": float(gap_s), "signals": labels, "seed": int(seed),
        "rx_rate_planned": None if rx_rate is None else float(rx_rate),
        "tx_center_hz": None if tx_center_hz is None else float(tx_center_hz),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    mpath = out / f"{nm}.manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if rf is not None:
        try:
            rf.record(path, "cabled-tx", f"{len(labels)} signals")
            rf.record(mpath, "cabled-tx-manifest", nm)
        except Exception:                                  # noqa: BLE001
            pass
    return TxFile(path, mpath, manifest)


def load_manifest(path_or_dict) -> dict:
    if isinstance(path_or_dict, dict):
        return path_or_dict
    p = Path(path_or_dict)
    m = json.loads(p.read_text(encoding="utf-8"))
    m["_dir"] = str(p.parent)
    return m


def verify_file(manifest: dict) -> tuple[bool, str]:
    """(ok, why): is the transmit file on disk the one the manifest
    describes? A changed file would put every label in the wrong place."""
    d = Path(manifest.get("_dir", "."))
    p = d / manifest["file"]
    if not p.exists():
        return False, f"the transmit file {p.name} is missing"
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != manifest.get("sha256"):
        return False, (f"{p.name} is not the file its manifest describes (its "
                       "hash changed); rebuild it — the labels would be wrong")
    return True, ""
