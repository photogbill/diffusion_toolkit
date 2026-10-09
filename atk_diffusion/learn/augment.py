# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""B4 generative augmentation — time-frequency diffusion, written from the
idea (plan §4.B4, §8) — and the parametric synthetic baseline it is judged
against.

Plan B4: *"RF-Diffusion's time-frequency diffusion (GPL-3 — run as a
separate process, or reimplement TFD from the paper) to add realism beyond
TorchSig's parametric impairments. Kept only if it closes the domain gap:
train with and without, test on cabled."*

TFD-LITE. RF-Diffusion (arXiv 2404.09140, MobiCom '24) degrades a signal in
two ways at once as t grows — Gaussian noise in time AND a blur of its
spectrum — and learns to undo both. This module is a reimplementation of
THAT IDEA under this repository's licence; no RF-Diffusion code is used or
was consulted (its code is GPL-3, plan §8, D1). The forward process here is

    x_t = √ᾱ_t · B_t(x0) + √(1 − ᾱ_t) · ε,

where B_t is a circular convolution of the spectrum with a Gaussian whose
width grows with t — done in time as a multiplication by the kernel's
transform m_t[n] (a real, positive envelope), so B_t is exact and
invertible. The network predicts x0 directly (natural when the degradation
has a deterministic part) and sampling re-applies the degradation at the
next step, DDIM-style. RF-Diffusion's hierarchical diffusion transformer is
NOT reproduced: the network is this package's small 1D U-Net, optionally
class-conditional. `blur=False` gives plain Gaussian diffusion — the
ablation that says whether the blur earns its place.

AUGMENTATION is SDEdit: a synthetic example is degraded to step t_aug and
brought back by a model trained on a LITTLE REAL data (cabled captures,
plan §3.5) — so it keeps its class and gains what real looks like. Plan
§3.6: *"learning what real looks like from a little real, so that the little
real goes further."* Every augmented example is INVENTED tier
(`provenance.tier_for("diffusion_augment")`) and flagged per row.

THE PARAMETRIC BASELINE, AND WHY IT LIVES HERE. Every learned tool in this
package needs signals with exact labels and noise that sounds like a
receiver. The package's generator (`synth.native`, plan §4.3) and receiver
impairment model (`dsp.impair`, §3.4) are other engineers' modules; this
module calls them when they are installed and otherwise falls back to its
own minimal versions, so no experiment here depends on them:

* `signal(kind, fs, n, rng)` — PTT key-up (NFM + CTCSS), POCSAG, DMR-like
  4FSK, ADS-B squitter, LoRa chirps, BPSK/QPSK, GFSK, OFDM, CW, LFM chirp,
  OOK, band-limited noise, and an 8-FSK the denoiser is NOT trained on.
  Everything Bill can make with his own bladeRF or capture around
  Nokesville; nothing needs a CSAR survival radio (plan change log).
* `burst(...)` places one in a window at a frequency offset, with its
  occupied bandwidth MEASURED (99 % power), not assumed.
* `LocalReceiver` — thermal floor, front-end droop and ripple (the floor
  shape), LO offset and phase noise, soft compression, IQ imbalance, DC
  spike, spurs, ADC quantisation. `for_profile` starts from the family's
  stated defaults (NOT measurements — say so) and takes the profile's
  measured `impairments` where their keys match.

HONEST LIMITS. The local generators are minimal: right modulation, right
rates, idealised framing (no real POCSAG codewords, no DMR sync words). The
local receiver defaults are typical values, not Bill's devices. TFD-lite at
the sizes tested here learns nothing useful; whether it closes the domain
gap on Bill's cabled captures is exactly what `experiments.augment_eval`
measures, and the plan's rule is that it is dropped if it does not.
"""

from __future__ import annotations

import inspect
import json
import math
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from atk_diffusion import provenance

# ---------------------------------------------------------------------------
# Local signal generators
# ---------------------------------------------------------------------------
#: kind -> (what it is, natural burst duration in seconds)
LOCAL_KINDS = {
    "nfm_keyup": ("a handheld's PTT key-up: NFM, ±2.5 kHz deviation, CTCSS "
                  "tone, voice-band tones, a 5 ms carrier ramp", 0.20),
    "pocsag": ("POCSAG pager burst: 2FSK ±4.5 kHz at 1200 bit/s, preamble "
               "then data", 0.10),
    "dmr": ("DMR-like 4FSK burst: 4800 sym/s, ±648 / ±1944 Hz, Gaussian "
            "shaped, one 27.5 ms TDMA slot", 0.0275),
    "adsb": ("ADS-B extended squitter: 8 µs preamble + 112 bits PPM at "
             "1 Mb/s (needs ≥ 2 MS/s)", 120e-6),
    "lora": ("LoRa chirps: 125 kHz, SF7 — 8 up-chirps, 2 down-chirps, "
             "payload symbols (synth.native's packet is 32 ms)", 0.032),
    "bpsk": ("BPSK, root-raised-cosine (β 0.35)", 0.02),
    "qpsk": ("QPSK, root-raised-cosine (β 0.35)", 0.02),
    "gfsk": ("GFSK (BT 0.5, h 0.5) — BLE-like", 0.02),
    "ofdm": ("OFDM, 64-point, 48 QPSK subcarriers, CP 1/4", 0.02),
    "tone": ("a CW tone", 0.02),
    "chirp": ("a linear-FM chirp (a radar-like pulse)", 0.002),
    "ook": ("on-off keyed carrier", 0.02),
    "noise": ("band-limited noise burst (spread / noise-like)", 0.02),
    "mfsk8": ("8-FSK, 1 kHz tone spacing at 1000 baud — the waveform the B3 "
              "denoiser is NOT trained on", 0.05),
}

#: class-table names (detect.classes) -> the local kind that stands in.
CLASS_TO_LOCAL = {
    "nfm_voice": "nfm_keyup", "noaa_wx": "nfm_keyup", "pocsag": "pocsag",
    "flex": "pocsag", "dmr": "dmr", "p25": "dmr", "nxdn96": "dmr",
    "adsb": "adsb", "lora": "lora", "ref_bpsk": "bpsk", "ref_qpsk": "qpsk",
    "ref_2fsk": "pocsag", "ref_gfsk": "gfsk", "ble": "gfsk",
    "ref_ofdm": "ofdm", "spur": "tone", "noise": "noise", "ref_ask": "ook",
    "gnss_jamming": "chirp",
}


def _rrc(beta: float, sps: int, span: int = 8) -> np.ndarray:
    """Root-raised-cosine taps, unit energy — ported from Bill's
    `atk/core/siga/synth.py` (singularities taken by their limits)."""
    sps = max(1, int(sps))
    n = np.arange(-span * sps / 2.0, span * sps / 2.0 + 1)
    t = n / float(sps)
    h = np.empty_like(t)
    sing = np.isclose(np.abs(t), 1.0 / (4.0 * beta), atol=1e-9)
    zero = np.isclose(t, 0.0, atol=1e-12)
    ok = ~(sing | zero)
    tt = t[ok]
    h[ok] = (np.sin(np.pi * tt * (1 - beta)) + 4 * beta * tt
             * np.cos(np.pi * tt * (1 + beta))) / (np.pi * tt * (1 - (4 * beta * tt) ** 2))
    h[zero] = 1.0 - beta + 4.0 * beta / np.pi
    if sing.any():
        h[sing] = (beta / np.sqrt(2.0)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                                           + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
    return h / np.sqrt(np.sum(h ** 2))


def _gauss_taps(bt: float, sps: float, span: int = 4) -> np.ndarray:
    sps = max(1.0, float(sps))
    t = np.arange(-span * sps / 2.0, span * sps / 2.0 + 1) / sps
    a = math.sqrt(math.log(2.0) / 2.0) / max(1e-6, float(bt))
    h = np.exp(-(np.pi ** 2) * t ** 2 / a ** 2)
    return h / h.sum()


def _cpfsk(rng, n: int, fs: float, baud: float, levels_hz, bt: float | None = None,
           symbols=None) -> np.ndarray:
    """Continuous-phase FSK at any rate: symbol k covers samples with
    floor(i·baud/fs) = k (non-integer samples per symbol are fine)."""
    idx = np.floor(np.arange(n) * float(baud) / float(fs)).astype(np.int64)
    nsym = int(idx[-1]) + 1
    levels = np.asarray(levels_hz, dtype=np.float64)
    sym = rng.integers(0, levels.size, nsym) if symbols is None else np.asarray(symbols)
    f = levels[sym[np.minimum(idx, sym.size - 1)]]
    if bt:
        # centred 'full' convolution trimmed to n: np.convolve(mode="same")
        # returns the LONGER length when the filter outgrows a short window
        taps = _gauss_taps(bt, fs / baud)
        lead = (taps.size - 1) // 2
        f = np.convolve(f, taps, mode="full")[lead: lead + n]
    return np.exp(2j * np.pi * np.cumsum(f) / fs)


def _need(fs: float, rate: float, what: str):
    if fs < rate:
        raise ValueError(f"{what} needs at least {rate / 1e6:g} MS/s to be "
                         f"represented; this profile runs at {fs / 1e6:g} MS/s")


def _lowpass_noise(rng, n, fs, bw):
    from scipy.signal import firwin, lfilter
    w = (rng.standard_normal(n + 256) + 1j * rng.standard_normal(n + 256)) / math.sqrt(2)
    cut = min(0.49 * fs, max(bw / 2.0, fs / 1000.0))
    taps = firwin(129, cut, fs=fs)
    return lfilter(taps, 1.0, w)[256:]


def _local(kind: str, fs: float, n: int, rng, **kw) -> tuple[np.ndarray, dict]:
    fs = float(fs)
    t = np.arange(n) / fs
    truth: dict = {"kind": kind, "generator": "local", "fs": fs}
    if kind == "nfm_keyup":
        dev = float(kw.get("deviation_hz", 2500.0))
        tones = rng.uniform(300.0, 2500.0, 3)
        audio = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) for f in tones) / 3.0
        audio = 0.85 * audio + 0.15 * np.sin(2 * np.pi * float(kw.get("ctcss_hz", 100.0)) * t)
        x = np.exp(2j * np.pi * dev * np.cumsum(audio) / fs)
        ramp = np.clip(t / 0.005, 0.0, 1.0)
        x = x * (0.5 - 0.5 * np.cos(np.pi * ramp))
        truth.update(deviation_hz=dev, ctcss_hz=float(kw.get("ctcss_hz", 100.0)),
                     nominal_bw_hz=2 * (dev + 3000.0))
    elif kind == "pocsag":
        baud = float(kw.get("baud", 1200.0))
        nbit = int(math.floor((n - 1) * baud / fs)) + 1
        bits = rng.integers(0, 2, nbit)
        bits[: max(1, int(0.3 * nbit))] = np.arange(max(1, int(0.3 * nbit))) % 2
        x = _cpfsk(rng, n, fs, baud, [-4500.0, 4500.0], symbols=bits)
        truth.update(baud=baud, deviation_hz=4500.0, nominal_bw_hz=9000.0 + baud)
    elif kind == "dmr":
        x = _cpfsk(rng, n, fs, 4800.0, [-1944.0, -648.0, 648.0, 1944.0], bt=0.5)
        truth.update(baud=4800.0, levels_hz=[-1944, -648, 648, 1944],
                     nominal_bw_hz=7600.0)
    elif kind == "adsb":
        _need(fs, 2e6, "ADS-B (1 Mb/s PPM)")
        bits = rng.integers(0, 2, 112)
        env = np.zeros(n)
        us = 1e-6 * fs

        def pulse(t0_us):
            a, b = int(round(t0_us * us)), int(round((t0_us + 0.5) * us))
            env[min(a, n): min(max(b, a + 1), n)] = 1.0
        for p in (0.0, 1.0, 3.5, 4.5):
            pulse(p)
        for i, bit in enumerate(bits):
            pulse(8.0 + i + (0.0 if bit else 0.5))
        x = env.astype(np.complex128) * np.exp(1j * rng.uniform(0, 6.28))
        truth.update(bits=112, nominal_bw_hz=2e6)
    elif kind == "lora":
        bw = float(kw.get("bw", 125e3))
        sf = int(kw.get("sf", 7))
        _need(fs, bw, "LoRa at this bandwidth")
        N = 2 ** sf
        tsym = N / bw
        k = np.floor(t / tsym).astype(np.int64)
        tau = t - k * tsym
        nsym = int(k[-1]) + 1
        shifts = rng.integers(0, N, nsym)
        shifts[: min(10, nsym)] = 0
        down = np.zeros(nsym, dtype=bool)
        down[8:10] = True
        frac = (shifts[k] / N + tau / tsym) % 1.0
        f = bw * frac - bw / 2.0
        f[down[k]] = -f[down[k]]
        x = np.exp(2j * np.pi * np.cumsum(f) / fs)
        truth.update(bw_hz=bw, sf=sf, symbol_s=tsym, nominal_bw_hz=bw)
    elif kind in ("bpsk", "qpsk"):
        sps = int(kw.get("sps", 8))
        beta = 0.35
        alpha = (np.array([1.0, -1.0]) if kind == "bpsk"
                 else np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / math.sqrt(2))
        h = _rrc(beta, sps)
        nsym = n // sps + 16
        up = np.zeros(nsym * sps, dtype=np.complex128)
        up[::sps] = alpha[rng.integers(0, alpha.size, nsym)]
        lead = (h.size - 1) // 2
        x = np.convolve(up, h)[lead: lead + n]
        truth.update(baud=fs / sps, rolloff=beta, nominal_bw_hz=fs / sps * (1 + beta))
    elif kind == "gfsk":
        baud = float(kw.get("baud", min(1e6, fs / 8.0)))
        x = _cpfsk(rng, n, fs, baud, [-baud / 4.0, baud / 4.0], bt=0.5)
        truth.update(baud=baud, h=0.5, nominal_bw_hz=1.5 * baud)
    elif kind == "ofdm":
        nfft = 64
        used = np.r_[1:25, 40:64]
        cp = 16
        nsym = int(math.ceil(n / (nfft + cp))) + 1
        X = np.zeros((nsym, nfft), dtype=np.complex128)
        qpsk = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / math.sqrt(2)
        X[:, used] = qpsk[rng.integers(0, 4, (nsym, used.size))]
        s = np.fft.ifft(X, axis=1) * math.sqrt(nfft)
        x = np.concatenate([s[:, -cp:], s], axis=1).reshape(-1)[:n]
        truth.update(nfft=nfft, cp=cp, nominal_bw_hz=fs * 48 / 64)
    elif kind == "tone":
        x = np.exp(2j * np.pi * float(kw.get("f_hz", 0.0)) * t + 1j * rng.uniform(0, 6.28))
        truth.update(nominal_bw_hz=fs / max(n, 1))
    elif kind == "chirp":
        bw = float(kw.get("bw", fs / 8.0))
        dur = n / fs
        x = np.exp(1j * np.pi * (bw / dur) * (t - dur / 2) ** 2)
        truth.update(bw_hz=bw, nominal_bw_hz=bw)
    elif kind == "ook":
        baud = float(kw.get("baud", min(10e3, fs / 16.0)))
        idx = np.floor(np.arange(n) * baud / fs).astype(np.int64)
        on = rng.integers(0, 2, int(idx[-1]) + 1)
        on[0] = 1
        env = np.convolve(on[idx].astype(float), np.ones(4) / 4.0, mode="full")[1: 1 + n]
        x = env * np.exp(1j * rng.uniform(0, 6.28))
        truth.update(baud=baud, nominal_bw_hz=2 * baud)
    elif kind == "noise":
        bw = float(kw.get("bw", fs / 10.0))
        x = _lowpass_noise(rng, n, fs, bw)
        truth.update(nominal_bw_hz=bw)
    elif kind == "mfsk8":
        x = _cpfsk(rng, n, fs, 1000.0, (np.arange(8) - 3.5) * 1000.0)
        truth.update(baud=1000.0, tones=8, nominal_bw_hz=9000.0)
    else:
        raise ValueError(f"unknown signal kind {kind!r} — one of "
                         f"{', '.join(sorted(LOCAL_KINDS))}")
    return np.asarray(x, dtype=np.complex128), truth


def _native_generate():
    """synth.native.generate when installed (another engineer's module)."""
    try:
        from atk_diffusion.synth import native
        return getattr(native, "generate", None)
    except Exception:                                      # noqa: BLE001
        return None


_NATIVE_ARGS = ("cls", "fs", "n_samples", "snr_db", "rng", "noise_dbfs")
_CLEAN_DB = 200.0         # ask synth.native for its noise this far down: clean


def _try_native(gen, cls_name: str, fs: float, n: int, rng):
    """synth.native.generate(cls, fs, n_samples, snr_db, rng, *,
    noise_dbfs=…) -> (iq, labels) always adds white noise; asking for it
    200 dB down (below float32 resolution next to the signal) gives the
    clean signal. A changed signature is not guessed at: None -> local."""
    try:
        names = list(inspect.signature(gen).parameters)
    except (TypeError, ValueError):
        return None
    if names[:5] != list(_NATIVE_ARGS[:5]) or "noise_dbfs" not in names:
        return None
    try:
        x, lab = gen(cls_name, float(fs), int(n), _CLEAN_DB, rng,
                     noise_dbfs=-_CLEAN_DB)
    except Exception:                                      # noqa: BLE001
        return None
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    if x.size != n:
        x = np.concatenate([x, np.zeros(max(0, n - x.size))])[:n]
    keep = {k: v for k, v in (lab or {}).items()
            if isinstance(v, (int, float, str, bool)) and k not in ("snr_db", "noise_dbfs")}
    s0 = int(keep.get("sample_start", 0))
    c0 = int(keep.get("sample_count", n)) or n
    truth = {"kind": cls_name, "generator": "synth.native", "fs": float(fs),
             "native": keep, "active": (s0, min(c0, n - s0))}
    if keep.get("nominal_bandwidth_hz"):
        truth["nominal_bw_hz"] = float(keep["nominal_bandwidth_hz"])
    return x, truth


def _is_class_name(kind: str) -> bool:
    from atk_diffusion.detect import classes as _classes
    return _classes.get(kind) is not None


def signal(kind: str, fs: float, n: int, rng=None, prefer_native: bool = True,
           **kw) -> tuple[np.ndarray, dict]:
    """A clean signal of `n` samples at `fs` with unit power while it is ON,
    and what it is (`truth`). A class-table name (pocsag, dmr, adsb, lora,
    nfm_voice, ble, …) goes first to `synth.native.generate` when it is
    installed (real sync words and framing); a local-only kind, or a name
    the native generator refuses, uses the local generator.
    `truth["generator"]` says which made it."""
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    n = int(n)
    if n < 2:
        raise ValueError("a signal needs at least two samples")
    x = None
    truth: dict = {}
    if prefer_native and _is_class_name(kind):
        gen = _native_generate()
        if gen is not None:
            got = _try_native(gen, kind, fs, n, rng)
            if got is not None:
                x, truth = got
    if x is None:
        local = CLASS_TO_LOCAL.get(kind, kind)
        x, truth = _local(local, fs, n, rng, **kw)
        truth["requested"] = kind
        truth["active"] = (0, n)
    # unit power over the ACTIVE span (energy / duration while on): a pulse
    # train's own gaps count, the silence the generator put around it does not
    s0, c0 = truth["active"]
    p = float(np.mean(np.abs(x[s0:s0 + c0]) ** 2)) if c0 > 0 else 0.0
    if p <= 0:
        raise ValueError(f"the generator made a silent {kind!r} record "
                         f"({n} samples at {fs:g} S/s) — lengthen it")
    return (x / math.sqrt(p)).astype(np.complex64), truth


def natural_duration(kind: str) -> float:
    local = CLASS_TO_LOCAL.get(kind, kind)
    return float(LOCAL_KINDS.get(local, ("", 0.02))[1])


def occupied_bandwidth(x, fs: float, frac: float = 0.99) -> float:
    """The bandwidth holding `frac` of the power (measured, Welch PSD)."""
    from scipy.signal import welch
    x = np.asarray(x)
    on = np.flatnonzero(np.abs(x) > 0)
    if on.size:                       # trim the ends only: interior gaps of
        x = x[on[0]: on[-1] + 1]      # a pulse train ARE its spectrum
    nper = int(min(1024, max(16, x.size)))
    f, p = welch(x, fs=fs, nperseg=nper, return_onesided=False, detrend=False)
    order = np.argsort(f)
    f, p = f[order], p[order]
    c = np.cumsum(p) / max(np.sum(p), 1e-30)
    lo = f[np.searchsorted(c, (1 - frac) / 2)]
    hi = f[min(np.searchsorted(c, 1 - (1 - frac) / 2), f.size - 1)]
    return float(max(hi - lo, fs / nper))


def shift(x, f_hz: float, fs: float) -> np.ndarray:
    n = np.arange(np.asarray(x).size)
    return (np.asarray(x) * np.exp(2j * np.pi * float(f_hz) * n / float(fs))).astype(np.complex64)


def burst_samples(kind: str, fs: float, nb: int, rng=None, **kw) -> tuple[np.ndarray, dict]:
    """The burst itself, at most `nb` samples, unit power while on. The
    generator is asked for a longer window and its labelled ACTIVE span is
    cut out, so a native generator that places its burst somewhere inside
    the window (ADS-B, BLE) is never clipped by the request."""
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    nb = int(max(16, nb))
    full, truth = signal(kind, fs, 2 * nb + 64, rng, **kw)
    s0, c0 = truth["active"]
    s = full[s0:s0 + min(int(c0), nb)]
    if s.size < 16:
        s = full[:nb]
    p = float(np.mean(np.abs(s) ** 2))
    if p <= 0:
        raise ValueError(f"the {kind!r} burst came out silent")
    truth["active"] = (0, int(s.size))
    return (s / math.sqrt(p)).astype(np.complex64), truth


def burst(kind: str, fs: float, n_total: int, rng=None, start: int | None = None,
          duration_s: float | None = None, offset_hz: float | None = None,
          edge_frac: float = 0.45, **kw) -> tuple[np.ndarray, dict]:
    """One burst placed in an `n_total`-sample window: random start and
    frequency offset (kept inside ±edge_frac·fs with its bandwidth) unless
    given. Unit power while on. truth carries sample_start/count, offset,
    the MEASURED 99 % bandwidth, and the clipped duration if the window was
    shorter than the burst's natural length."""
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    want = float(duration_s if duration_s is not None else natural_duration(kind))
    nb = int(max(16, min(int(n_total), round(want * fs))))
    s, truth = burst_samples(kind, fs, nb, rng, **kw)
    nb = int(s.size)
    bw = occupied_bandwidth(s, fs)
    if offset_hz is None:
        room = max(0.0, edge_frac * fs - bw / 2.0)
        offset_hz = float(rng.uniform(-room, room)) if room > 0 else 0.0
    if start is None:
        start = int(rng.integers(0, int(n_total) - nb + 1))
    x = np.zeros(int(n_total), dtype=np.complex64)
    x[start: start + nb] = shift(s, offset_hz, fs)
    truth.update(sample_start=int(start), sample_count=int(nb),
                 offset_hz=float(offset_hz), bw_hz=float(bw),
                 duration_s=nb / float(fs), clipped=bool(nb < round(want * fs)),
                 clean=s)
    return x, truth


# ---------------------------------------------------------------------------
# The local receiver model (fallback for dsp.impair)
# ---------------------------------------------------------------------------
@dataclass
class LocalReceiver:
    """What a receiver does to what it hears, in its own order: thermal
    noise at the input; the front end's response (droop toward the band
    edges, ripple — the floor's shape); LO offset and phase noise; soft
    compression; IQ imbalance; DC offset and spurs; the ADC. Levels are
    dBFS (complex power relative to full scale)."""
    name: str = "generic"
    adc_bits: int = 0                  # 0: no quantisation
    signal_dbfs: float = -20.0         # where a unit-power input lands
    noise_dbfs: float = -40.0          # thermal floor (complex power)
    dc_dbfs: float | None = None
    dc_phase_deg: float = 40.0
    iq_gain_db: float = 0.0
    iq_phase_deg: float = 0.0
    freq_offset_hz: float = 0.0
    linewidth_hz: float = 0.0          # Wiener phase noise
    edge_droop_db: float = 0.0         # attenuation at ±fs/2 vs centre
    ripple_db: float = 0.0
    ripple_cycles: float = 3.0
    spurs: tuple = ()                  # ((fraction of fs, dBFS), …)
    compression_dbfs: float | None = None
    notes: list = field(default_factory=list)

    def response(self, n: int) -> np.ndarray:
        """|H(f)| on the n FFT bins (unshifted order)."""
        f = np.fft.fftfreq(n)                       # −0.5 … 0.5 (cycles/sample)
        db = -self.edge_droop_db * (2.0 * np.abs(f)) ** 2
        if self.ripple_db:
            db = db + self.ripple_db * np.cos(2 * np.pi * self.ripple_cycles * f)
        return np.power(10.0, db / 20.0)

    def apply(self, x, fs: float, rng=None, thermal: bool = True) -> np.ndarray:
        rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
        x = np.asarray(x, dtype=np.complex128) * 10 ** (self.signal_dbfs / 20.0)
        n = x.size
        if thermal:
            sd = math.sqrt(10 ** (self.noise_dbfs / 10.0) / 2.0)
            x = x + sd * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        if self.edge_droop_db or self.ripple_db:
            x = np.fft.ifft(np.fft.fft(x) * self.response(n))
        ph = 2 * np.pi * self.freq_offset_hz * np.arange(n) / float(fs)
        if self.linewidth_hz:
            ph = ph + np.cumsum(rng.standard_normal(n)
                                * math.sqrt(2 * np.pi * self.linewidth_hz / float(fs)))
        if self.freq_offset_hz or self.linewidth_hz:
            x = x * np.exp(1j * ph)
        if self.compression_dbfs is not None:
            a = 10 ** (self.compression_dbfs / 20.0)
            p = 2.0
            x = x / (1.0 + (np.abs(x) / a) ** (2 * p)) ** (1.0 / (2 * p))
        if self.iq_gain_db or self.iq_phase_deg:
            # Bill's model (ATK iq_balance.py, as dsp.impair uses it):
            # Q_out = g·(Q cos φ + I sin φ), I untouched
            g = 10 ** (self.iq_gain_db / 20.0)
            phi = math.radians(self.iq_phase_deg)
            x = x.real + 1j * g * (x.imag * math.cos(phi) + x.real * math.sin(phi))
        if self.dc_dbfs is not None:
            x = x + 10 ** (self.dc_dbfs / 20.0) * np.exp(1j * math.radians(self.dc_phase_deg))
        k = np.arange(n)
        for frac, lvl in self.spurs:
            x = x + 10 ** (float(lvl) / 20.0) * np.exp(2j * np.pi * float(frac) * k)
        if self.adc_bits:
            q = 2.0 ** (int(self.adc_bits) - 1)
            re = np.clip(np.round(x.real * q), -q, q - 1) / q
            im = np.clip(np.round(x.imag * q), -q, q - 1) / q
            x = re + 1j * im
        return x.astype(np.complex64)

    def noise_only(self, n: int, fs: float, rng=None) -> np.ndarray:
        """What the receiver records with no antenna (terminated)."""
        return self.apply(np.zeros(int(n)), fs, rng)

    def to_json(self) -> dict:
        d = asdict(self)
        d["spurs"] = [list(s) for s in self.spurs]
        return d


#: Stated starting values per family — NOT measurements. dsp.impair
#: measures the real ones from a terminated capture (plan §3.4).
FAMILY_DEFAULTS = {
    "rtlsdr": dict(adc_bits=8, signal_dbfs=-20.0, noise_dbfs=-30.0, dc_dbfs=-35.0,
                   iq_gain_db=0.4, iq_phase_deg=2.0, edge_droop_db=2.0, ripple_db=0.3),
    "krakensdr": dict(adc_bits=8, signal_dbfs=-20.0, noise_dbfs=-30.0, dc_dbfs=-35.0,
                      iq_gain_db=0.4, iq_phase_deg=2.0, edge_droop_db=2.0, ripple_db=0.3),
    "hackrf": dict(adc_bits=8, signal_dbfs=-20.0, noise_dbfs=-30.0, dc_dbfs=-25.0,
                   iq_gain_db=0.6, iq_phase_deg=3.0, edge_droop_db=1.0),
    "bladerf1": dict(adc_bits=12, signal_dbfs=-25.0, noise_dbfs=-45.0, dc_dbfs=-50.0,
                     iq_gain_db=0.1, iq_phase_deg=0.5, edge_droop_db=0.5, ripple_db=0.1),
    "bladerf2": dict(adc_bits=12, signal_dbfs=-25.0, noise_dbfs=-45.0, dc_dbfs=-50.0,
                     iq_gain_db=0.1, iq_phase_deg=0.5, edge_droop_db=0.5, ripple_db=0.1),
    "airspy": dict(adc_bits=12, signal_dbfs=-25.0, noise_dbfs=-45.0, dc_dbfs=-60.0,
                   iq_gain_db=0.05, iq_phase_deg=0.2),
}


def receiver_for(profile) -> LocalReceiver:
    """The local receiver model for a profile id or ReceiverProfile: the
    family's stated defaults, then any measured `impairments` whose keys
    are fields of LocalReceiver. Unused keys are named in `.notes`."""
    from atk_diffusion import profiles as _profiles
    prof = profile
    if isinstance(profile, str):
        pid = _profiles.parse_profile_id(profile)
        family, imp = pid.family, {}
    else:
        family, imp = prof.pid.family, dict(getattr(prof, "impairments", {}) or {})
    base = dict(FAMILY_DEFAULTS.get(family, {}))
    rx = LocalReceiver(name=family, **base)
    rx.notes.append(f"{family} defaults are stated starting values, not "
                    "measurements of Bill's device (plan §3.4)" if base else
                    f"no defaults for {family}: a float receiver with a "
                    "-40 dBFS floor")
    known = {f.name for f in fields(LocalReceiver)}
    used, unused = [], []
    for k, v in imp.items():
        if k in known and k not in ("name", "notes"):
            setattr(rx, k, tuple(map(tuple, v)) if k == "spurs" else v)
            used.append(k)
        else:
            unused.append(k)
    if used:
        rx.notes.append("measured: " + ", ".join(sorted(used)))
    if unused:
        rx.notes.append("not used by the local model: " + ", ".join(sorted(unused)))
    return rx


def _external_impair():
    """dsp.impair (another engineer's module) when installed."""
    try:
        from atk_diffusion.dsp import impair as _imp
        if callable(getattr(_imp, "apply_impairments", None)):
            return _imp
    except Exception:                                      # noqa: BLE001
        return None
    return None


def _profile_parts(profile) -> tuple[dict, str]:
    """(measured impairments, datatype) of a profile id or ReceiverProfile.
    An id alone carries no measurement: the textbook receiver."""
    from atk_diffusion import profiles as _profiles
    if isinstance(profile, str):
        return {}, _profiles.parse_profile_id(profile).datatype
    return dict(getattr(profile, "impairments", {}) or {}), profile.datatype


def _local_from_measured(imp: dict) -> LocalReceiver:
    """The local model carrying what dsp.impair measured (fallback only)."""
    rx = LocalReceiver(name="measured", signal_dbfs=0.0)
    rx.iq_gain_db = float(imp.get("iq_gain_imbalance_db", 0.0) or 0.0)
    rx.iq_phase_deg = float(imp.get("iq_phase_imbalance_deg", 0.0) or 0.0)
    dci, dcq = imp.get("dc_offset_i"), imp.get("dc_offset_q")
    if dci is not None or dcq is not None:
        dc = complex(float(dci or 0.0), float(dcq or 0.0))
        if abs(dc) > 0:
            rx.dc_dbfs = 20.0 * math.log10(abs(dc))
            rx.dc_phase_deg = math.degrees(math.atan2(dc.imag, dc.real))
    rx.spurs = tuple((float(s["offset_hz"]) / float(imp.get("fs", 1.0) or 1.0),
                      float(s["level_db"])) for s in (imp.get("spurs") or [])
                     if s.get("level_db") is not None and imp.get("fs"))
    if imp.get("floor_db_per_bin"):
        rx.notes.append("the measured floor SHAPE is not applied by the local "
                        "fallback (dsp.impair applies it)")
    return rx


def impair(x, fs: float, profile, rng=None) -> tuple[np.ndarray, str]:
    """The profile's receiver applied to `x` — signals plus white noise
    already at the receiver's floor level, full-scale units (dsp.impair's
    contract). Measured impairments are applied when the profile has them;
    an unmeasured profile is the TEXTBOOK receiver (its datatype's
    quantisation only) and the returned words say so. Uses
    `dsp.impair.apply_impairments` when installed, else the local model.
    Returns (y, words)."""
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    imp, dt = _profile_parts(profile)
    ext = _external_impair()
    x = np.asarray(x, dtype=np.complex64)
    if ext is not None:
        y = ext.apply_impairments(x, float(fs), imp, rng, datatype=dt)
        return np.asarray(y, dtype=np.complex64), "dsp.impair: " + ext.describe(imp)
    from atk_diffusion.dsp import iq as _iq
    rx = _local_from_measured(imp)
    y = rx.apply(x, fs, rng, thermal=False)
    if dt and dt != "cf32":
        y = _iq.to_complex(_iq.from_complex(y, dt), dt)
    words = ("local receiver model (dsp.impair not installed): "
             + ("measured impairments" if imp else
                "receiver impairments not measured — a textbook receiver"))
    return np.asarray(y, dtype=np.complex64), words


def receiver_noise(n: int, fs: float, profile, rng=None) -> tuple[np.ndarray, str]:
    """Synthetic 'terminated capture' noise for a profile: white Gaussian
    noise at the measured floor power (dsp.impair.noise_power; −30 dBFS when
    unmeasured) through `impair`. Returns (noise, words)."""
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    imp, _dt = _profile_parts(profile)
    ext = _external_impair()
    p = ext.noise_power(imp) if ext is not None else 10 ** (
        float(imp.get("floor_mean_dbfs", -30.0)) / 10.0)
    w = (math.sqrt(p / 2.0) * (rng.standard_normal(int(n))
                               + 1j * rng.standard_normal(int(n)))).astype(np.complex64)
    return impair(w, fs, profile, rng)


def make_classification_set(kinds, n_per: int, length: int, fs: float, rng=None,
                            snr_db=(0.0, 10.0), receiver: LocalReceiver | None = None,
                            offset_frac: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """A tiny labelled IQ set: `n_per` windows of `length` samples per kind,
    each at a random SNR in `snr_db` (over the sampled bandwidth), optionally
    through a receiver model. Returns (X complex64 [N, L], y int64 [N])."""
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    X, Y = [], []
    for ci, kind in enumerate(kinds):
        for _ in range(int(n_per)):
            s, _t = signal(kind, fs, length, rng)
            if offset_frac:
                s = shift(s, rng.uniform(-offset_frac, offset_frac) * fs, fs)
            snr = rng.uniform(*snr_db) if np.ndim(snr_db) else float(snr_db)
            sd = math.sqrt(10 ** (-snr / 10.0) / 2.0)
            y = s + sd * (rng.standard_normal(length) + 1j * rng.standard_normal(length))
            if receiver is not None:
                y = receiver.apply(y, fs, rng, thermal=False)
            X.append(np.asarray(y, dtype=np.complex64))
            Y.append(ci)
    return np.stack(X), np.asarray(Y, dtype=np.int64)


def iq_to_channels(X) -> np.ndarray:
    """complex [N, L] -> float32 [N, 2, L] (I, Q)."""
    X = np.asarray(X)
    return np.stack([X.real, X.imag], axis=1).astype(np.float32)


def channels_to_iq(Z) -> np.ndarray:
    Z = np.asarray(Z)
    return (Z[:, 0] + 1j * Z[:, 1]).astype(np.complex64)


def unit_power(X) -> tuple[np.ndarray, np.ndarray]:
    """Scale each window to unit power per real element. Returns
    (scaled, rms per window)."""
    X = np.asarray(X)
    rms = np.sqrt(np.mean(np.abs(X) ** 2, axis=-1, keepdims=True) / 2.0)
    rms = np.maximum(rms, 1e-12)
    return X / rms, rms[..., 0]


# ---------------------------------------------------------------------------
# TFD-lite: time-frequency diffusion
# ---------------------------------------------------------------------------
provenance.METHOD_TIERS.setdefault("diffusion_augment", "invented")


def blur_envelope(t, T: int, length: int, blur_max: float = 0.3) -> np.ndarray:
    """m_t[n]: the time-domain form of a circular Gaussian blur of the
    spectrum. Its width grows linearly with t so that at t = T−1 the
    envelope falls to `blur_max` at the window's centre (0 < blur_max ≤ 1;
    1 = no blur). Returns [len(t), L] (positive, ≤ 1)."""
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    n = np.arange(int(length))
    d = np.minimum(n, length - n) / float(length)          # circular distance, ≤ 1/2
    # at t = T−1: exp(−k·(1/2)²) = blur_max  ->  k_max = −4 ln(blur_max)
    k_max = -4.0 * math.log(max(min(float(blur_max), 1.0), 1e-6))
    k = k_max * np.clip(t / max(T - 1, 1), 0.0, 1.0)
    return np.exp(-k[:, None] * d[None, :] ** 2)


@dataclass
class TFDConfig:
    T: int = 200
    schedule: str = "cosine"
    blur: bool = True
    blur_max: float = 0.3
    length: int = 256


class TFDLite:
    """The degradation and the sampler; the network is any x0-predictor
    f(x_t, t, y) on [B, 2, L] float tensors (this package's U-Net)."""

    def __init__(self, cfg: TFDConfig):
        from atk_diffusion.learn import diffusion as _d
        self.cfg = cfg
        self.sched = _d.make_schedule(cfg.schedule, cfg.T)
        self.ac = self.sched.alphas_cumprod
        self.env = (blur_envelope(np.arange(cfg.T), cfg.T, cfg.length, cfg.blur_max)
                    if cfg.blur else np.ones((cfg.T, cfg.length)))

    def _m(self, t, x):
        import torch
        e = self.env[np.asarray(t).reshape(-1)]           # [B, L]
        return torch.as_tensor(e[:, None, :], dtype=x.dtype, device=x.device)

    def degrade(self, x0, t, noise):
        """x_t = √ᾱ_t · m_t ⊙ x0 + √(1 − ᾱ_t) · ε (torch, t per example)."""
        import torch
        tn = np.asarray(t.detach().cpu() if hasattr(t, "detach") else t).reshape(-1)
        a = torch.as_tensor(self.ac[tn], dtype=x0.dtype, device=x0.device)[:, None, None]
        return torch.sqrt(a) * self._m(tn, x0) * x0 + torch.sqrt(1 - a) * noise

    def loss(self, model, x0, y=None, generator=None):
        import torch
        from atk_diffusion.learn import diffusion as _d
        t = _d.sample_t(x0.shape[0], self.cfg.T, generator=generator).to(x0.device)
        noise = torch.randn(x0.shape, generator=generator).to(x0.device)
        xt = self.degrade(x0, t, noise)
        pred = model(xt, t, y=y) if y is not None else model(xt, t)
        return torch.mean((pred - x0) ** 2)

    def sample(self, model, x_start, t_start: int, y=None, steps: int = 20,
               eta: float = 0.0, generator=None):
        """Reverse from x_start (a draw at t_start) to x̂0: predict x0,
        re-degrade to the next step with the implied noise (DDIM-style)."""
        import torch
        from atk_diffusion.learn import diffusion as _d
        seq = _d.timesteps(int(t_start), steps)
        x = x_start
        x0 = x
        with torch.no_grad():
            for t, tp in zip(seq[:-1], seq[1:]):
                tv = torch.full((x.shape[0],), t, dtype=torch.long, device=x.device)
                x0 = model(x, tv, y=y) if y is not None else model(x, tv)
                if tp < 0:
                    break
                a_t, a_p = float(self.ac[t]), float(self.ac[tp])
                m_t = self._m([t] * x.shape[0], x)
                m_p = self._m([tp] * x.shape[0], x)
                eps = (x - math.sqrt(a_t) * m_t * x0) / math.sqrt(1 - a_t)
                sig = eta * math.sqrt(max(0.0, (1 - a_p) / (1 - a_t) * (1 - a_t / a_p)))
                x = math.sqrt(a_p) * m_p * x0 + math.sqrt(max(0.0, 1 - a_p - sig ** 2)) * eps
                if sig > 0:
                    x = x + sig * torch.randn(x.shape, generator=generator).to(x.device)
        return x0


def train_augmenter(rf, profile: str, X_real, y_real=None, classes=None, *,
                    name: str | None = None, steps: int = 2000, batch: int = 32,
                    lr: float = 2e-4, unet: dict | None = None,
                    cfg: TFDConfig | None = None, seed: int = 0,
                    out_dir=None, device: str | None = None, progress=None) -> Path:
    """Train TFD-lite on a LITTLE REAL data (complex windows [N, L] at the
    profile's rate — cabled captures, plan §3.5) and save it with its card
    (kind "augmenter", tier invented). Class-conditional when `y_real` is
    given. Returns the model folder."""
    import torch
    from atk_diffusion import cards
    from atk_diffusion.learn import diffusion as _d
    from atk_diffusion.learn import unet as _u
    X = np.asarray(X_real)
    if X.ndim != 2 or not np.iscomplexobj(X):
        raise ValueError("TFD-lite trains on complex windows [N, L]")
    cfg = cfg or TFDConfig(length=X.shape[1])
    cfg.length = int(X.shape[1])
    ncls = 0 if y_real is None else int(len(classes) if classes else int(np.max(y_real)) + 1)
    conf = dict(unet or _u.TINY_1D)
    conf.update(in_ch=2, num_classes=ncls)
    if cfg.length % _u.multiple(conf):
        raise ValueError(f"window length {cfg.length} must be a multiple of "
                         f"{_u.multiple(conf)} for this network")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(seed))
    model = _u.build_unet(conf).to(dev)
    tfd = TFDLite(cfg)
    Xn, _rms = unit_power(X)
    data = torch.as_tensor(iq_to_channels(Xn)).to(dev)
    labels = None if y_real is None else torch.as_tensor(np.asarray(y_real, dtype=np.int64)).to(dev)
    g = torch.Generator().manual_seed(int(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    model.train()
    for step in range(int(steps)):
        idx = torch.randint(0, data.shape[0], (min(batch, data.shape[0]),), generator=g)
        loss = tfd.loss(model, data[idx.to(dev)],
                        None if labels is None else labels[idx.to(dev)], g)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if progress and (step % 100 == 0 or step == steps - 1):
            progress(f"TFD-lite step {step + 1} of {steps}: loss {losses[-1]:.4f}")
    model.eval()
    name = name or f"tfd_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"
    d = Path(out_dir) if out_dir else rf.models(profile, name)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "augmenter.pt")
    card = cards.new_card(
        name, "augmenter", profile,
        input={"domain": "iq", "window": cfg.length, "normalize": "unit power per window",
               "tfd": asdict(cfg), "schedule": tfd.sched.to_json(), "unet": model.config},
        classes=[{"name": str(c), "source": "trained",
                  "examples": int(np.sum(np.asarray(y_real) == i))}
                 for i, c in enumerate(classes or [])] if y_real is not None else [],
        datasets=[{"name": "real (cabled) windows", "n": int(X.shape[0]),
                   "sha256": provenance.sha256_bytes(np.ascontiguousarray(X).tobytes()),
                   "kind": "iq"}],
        metrics={"final_loss": float(np.mean(losses[-20:])) if losses else None,
                 "steps": int(steps)},
        license="all rights reserved (TFD reimplemented from the idea of "
                "arXiv 2404.09140; no RF-Diffusion code)",
        trained_on=str(dev),
        notes=["augmented examples are INVENTED tier; kept only if they close "
               "the domain gap (plan B4, experiments.augment_eval)"])
    cards.save(d, card, "augmenter.pt")
    return d


class Augmenter:
    """A trained TFD-lite model, loaded through its card."""

    def __init__(self, model, card, model_dir):
        self.model, self.card, self.dir = model, card, Path(model_dir)
        self.cfg = TFDConfig(**card.input["tfd"])
        self.tfd = TFDLite(self.cfg)

    @classmethod
    def load(cls, model_dir, for_profile: str | None = None, device: str = "cpu"):
        import torch
        from atk_diffusion import cards
        from atk_diffusion.learn import unet as _u
        card = cards.load(model_dir, expect_kind="augmenter", for_profile=for_profile)
        model = _u.build_unet(card.input["unet"])
        model.load_state_dict(torch.load(cards.weights_path(model_dir, card),
                                         map_location=device, weights_only=True))
        return cls(model.to(device).eval(), card, model_dir)

    def augment(self, X, y=None, strength: float = 0.4, steps: int = 10,
                seed: int = 0) -> np.ndarray:
        """SDEdit each window: degrade to t = strength·(T−1), bring it back
        with the model. Output keeps each input's RMS. INVENTED tier."""
        import torch
        X = np.asarray(X)
        if X.shape[1] != self.cfg.length:
            raise ValueError(f"this augmenter works on {self.cfg.length}-sample "
                             f"windows, not {X.shape[1]}")
        Xn, rms = unit_power(X)
        dev = next(self.model.parameters()).device
        x0 = torch.as_tensor(iq_to_channels(Xn)).to(dev)
        t0 = int(round(float(strength) * (self.cfg.T - 1)))
        g = torch.Generator().manual_seed(int(seed))
        xt = self.tfd.degrade(x0, np.full(x0.shape[0], t0),
                              torch.randn(x0.shape, generator=g).to(dev))
        yy = None
        if self.model.num_classes and y is not None:
            yy = torch.as_tensor(np.asarray(y, dtype=np.int64)).to(dev)
        out = self.tfd.sample(self.model, xt, t0, y=yy, steps=steps, generator=g)
        Z = channels_to_iq(out.cpu().numpy())
        return (Z * rms[:, None]).astype(np.complex64)


def augment_dataset(X, y, augmenter, strength: float = 0.4, n_per: int = 1,
                    steps: int = 10, seed: int = 0, keep_original: bool = True):
    """Grow a labelled IQ set with TFD-lite. Returns (X_out, y_out, info);
    `info["invented"]` is a boolean per row (True = made by the model) and
    `info["tier"]` is "invented" for those rows — never mixed silently into
    the record."""
    if isinstance(augmenter, (str, Path)):
        augmenter = Augmenter.load(augmenter)
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.int64)
    outs, labels, flags = [], [], []
    if keep_original:
        outs.append(X)
        labels.append(y)
        flags.append(np.zeros(len(y), dtype=bool))
    for k in range(int(n_per)):
        outs.append(augmenter.augment(X, y, strength=strength, steps=steps,
                                      seed=int(seed) + k))
        labels.append(y)
        flags.append(np.ones(len(y), dtype=bool))
    info = {"method": "diffusion_augment",
            "tier": provenance.tier_for("diffusion_augment"),
            "model": augmenter.card.name,
            "model_sha256": augmenter.card.weights.get("sha256", ""),
            "strength": float(strength), "steps": int(steps), "n_per": int(n_per),
            "invented": np.concatenate(flags),
            "words": provenance.TIER_WORDS["invented"]}
    return np.concatenate(outs), np.concatenate(labels), info


def card_json(card) -> str:
    return json.dumps(card.to_json(), indent=2)
