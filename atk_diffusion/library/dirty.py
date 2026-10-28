# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Dirty copies: a real original through simulated channel and receiver
effects, so a model trained on a clean bench recording still recognises the
signal when it arrives weak, faded, off-frequency, clipped and interfered
with (RFMLS Training, 2026-10-26).

Bill, 2026-10-26: *"generating signal cuts for training from recording
(slightly altering it, adding fake noise, etc) so it works against dirty
signals."*

THE EFFECTS, in the order a real signal meets them on its way into the cut:

  the transmitter and the path
    clock       the transmitter's clock a few ppm off (the symbol rate moves)
    fading      multipath: Rayleigh or Rician (a K-factor), a Jakes Doppler
                spectrum, the delay spread of the terrain (`synth.scene.
                apply_channel` — the same channel the scene composer uses)
    offset      frequency error and drift: the box centre missed by a little
                (the detector's own box is never exactly centred) and the
                oscillator wandering
    phase noise the oscillators' phase noise (a Wiener process of a given
                linewidth)
  at the antenna
    interferer  something else on or beside the channel: a carrier, a
                narrowband signal, a swept tone
    impulses    ignition and switching clicks
    noise       white noise until the in-band SNR is the one asked for —
                measured the way the library measures (`store._measure`),
                so "7 dB" means the same thing on every copy. Noise can only
                be ADDED: a copy is never cleaner than its original, and when
                the asked-for SNR is above the original's the copy keeps the
                original's and says so.
  in the receiver
    agc         the receiver's gain pumping
    dc          the DC spike inside the cut (a signal sitting on the radio's
                own centre frequency)
    image       I/Q imbalance's mirror image inside the cut (the same case)
    clip        the converter clipping on a strong signal (I and Q apart)
    bits        fewer effective bits than the converter has
    dropouts    samples lost on the way to the PC (USB), as zero gaps

and last, the cut's own low-pass (`atk:lowpass_hz`), because everything the
field cut holds has been through it — noise outside the box included. A copy
whose added noise was NOT filtered would teach a classifier that flat noise
to the edge of the cut means "a copy".

EVERY COPY SAYS WHAT WAS DONE TO IT (`atk:effects`: each effect's name, its
parameters and a sentence) and which part of which original it came from
(`atk:parent`, `atk:crop`). Deterministic in its seed.

PRESETS (`PRESETS`): light (a clean bench), typical (the field — the
default), harsh (weak and messy). Every number is the operator's to change;
the tab shows them.

LIMITS, stated. These are models of effects, not measurements of a place:
the fading is the statistics of a terrain class, the interferers are generic.
A copy of one transmitter's signal is still that transmitter's signal —
dirt teaches tolerance, not variety. Effects that depend on where the signal
sat in the radio's span (the DC spike, the image) are rare in the field and
off by default in "light".
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time

import numpy as np

from atk_diffusion.library import store as _store

#: Default length of a copy, in samples at the canonical rate: two of the
#: classifier's windows, so a dataset can take more than one view of it.
DEFAULT_LENGTH = 2 * _store.WINDOW
MAX_PER_ORIGINAL = 5000

#: The effects in the order they are applied, with what an operator reads.
EFFECT_ORDER = ("clock", "fading", "offset", "phase_noise", "interferer",
                "impulses", "noise", "agc", "dc", "image", "clip", "bits",
                "dropouts")

EFFECT_WORDS = {
    "clock": ("Transmitter clock error",
              "The transmitter's clock is a few parts per million off, so its "
              "symbol rate is too."),
    "fading": ("Fading (multipath)",
               "The signal arrives by several paths that add and cancel as "
               "things move — it swells and fades. Doppler is how fast; the "
               "K-factor is how strong the direct path is (high K: a clear "
               "line of sight; low K: none)."),
    "offset": ("Frequency error and drift",
               "The signal is not exactly where the box is centred, and it "
               "wanders a little — as a real radio's oscillator does, and as a "
               "detector's box never sits exactly on the signal."),
    "phase_noise": ("Phase noise",
                    "The oscillators' own jitter smears the signal slightly."),
    "interferer": ("Interference",
                   "Something else on or next to the channel: a steady "
                   "carrier, another narrow signal, or a swept tone."),
    "impulses": ("Impulse noise (clicks)",
                 "Short bursts of noise: engines, switches, lightning."),
    "noise": ("Noise (weaker signal)",
              "Background noise is added until the signal stands this many dB "
              "above it. Noise is only ever added — a copy is never cleaner "
              "than its original."),
    "agc": ("Gain pumping (AGC)",
            "The receiver's automatic gain moves the level up and down."),
    "dc": ("DC spike in the cut",
           "The radio's own spike at its centre frequency, inside the cut — "
           "as when the signal sits on the radio's centre."),
    "image": ("Mirror image (I/Q imbalance)",
              "A faint mirror of the signal from the radio's I/Q imbalance — "
              "as when the signal sits near the radio's centre."),
    "clip": ("Clipping (overload)",
             "A strong signal overloads the converter and its peaks are cut "
             "flat."),
    "bits": ("Fewer converter bits",
             "Coarser samples, as from a cheaper or overloaded converter."),
    "dropouts": ("Dropped samples (USB)",
                 "Moments where samples never reached the PC."),
}

_BASE = {
    "clock":       {"on": True, "p": 0.5, "ppm": 20.0},
    "fading":      {"on": True, "p": 0.5, "doppler_hz": [0.5, 20.0],
                    "k_db": [-5.0, 15.0], "delay_spread_us": [0.0, 3.0]},
    "offset":      {"on": True, "p": 1.0, "fraction_of_bw": 0.05,
                    "drift_hz_per_s": 20.0},
    "phase_noise": {"on": True, "p": 0.5, "linewidth_hz": [0.1, 10.0]},
    "interferer":  {"on": True, "p": 0.3, "inr_db": [-20.0, -6.0],
                    "kinds": ["tone", "narrowband", "sweep"]},
    "impulses":    {"on": True, "p": 0.2, "rate_hz": [1.0, 20.0],
                    "level_db": [6.0, 20.0]},
    "noise":       {"on": True, "p": 1.0, "snr_db": [0.0, 25.0]},
    "agc":         {"on": True, "p": 0.3, "depth_db": [1.0, 6.0],
                    "rate_hz": [0.5, 5.0]},
    "dc":          {"on": True, "p": 0.05, "level_db": [-30.0, -15.0]},
    "image":       {"on": True, "p": 0.05, "rejection_db": [20.0, 35.0]},
    "clip":        {"on": True, "p": 0.1, "headroom_db": [0.5, 6.0]},
    "bits":        {"on": True, "p": 0.15, "bits": [5, 10]},
    "dropouts":    {"on": True, "p": 0.1, "rate_hz": [0.2, 3.0],
                    "ms": [0.2, 5.0]},
}


def _preset(changes: dict) -> dict:
    eff = copy.deepcopy(_BASE)
    for k, v in changes.items():
        eff[k].update(v)
    return eff


PRESETS = {
    "light": {"label": "Light — a clean bench, a little dirt",
              "effects": _preset({
                  "clock": {"p": 0.2}, "fading": {"p": 0.2,
                                                  "doppler_hz": [0.1, 5.0],
                                                  "k_db": [5.0, 20.0]},
                  "offset": {"fraction_of_bw": 0.03, "drift_hz_per_s": 5.0},
                  "phase_noise": {"p": 0.2},
                  "interferer": {"p": 0.1, "inr_db": [-30.0, -15.0]},
                  "impulses": {"p": 0.05}, "noise": {"snr_db": [12.0, 30.0]},
                  "agc": {"p": 0.1}, "dc": {"on": False}, "image": {"on": False},
                  "clip": {"p": 0.02}, "bits": {"p": 0.05},
                  "dropouts": {"p": 0.02}})},
    "typical": {"label": "Typical field — the default",
                "effects": _preset({})},
    "harsh": {"label": "Harsh — weak, faded and messy",
              "effects": _preset({
                  "clock": {"p": 0.7, "ppm": 50.0},
                  "fading": {"p": 0.8, "doppler_hz": [1.0, 80.0],
                             "k_db": [-10.0, 10.0], "delay_spread_us": [0.0, 8.0]},
                  "offset": {"fraction_of_bw": 0.1, "drift_hz_per_s": 60.0},
                  "phase_noise": {"p": 0.7, "linewidth_hz": [1.0, 40.0]},
                  "interferer": {"p": 0.6, "inr_db": [-15.0, -3.0]},
                  "impulses": {"p": 0.4, "rate_hz": [5.0, 50.0]},
                  "noise": {"snr_db": [-6.0, 15.0]},
                  "agc": {"p": 0.5, "depth_db": [2.0, 10.0]},
                  "dc": {"p": 0.1}, "image": {"p": 0.1},
                  "clip": {"p": 0.25}, "bits": {"p": 0.3, "bits": [4, 8]},
                  "dropouts": {"p": 0.25, "ms": [0.5, 10.0]}})},
}
DEFAULT_PRESET = "typical"


def recipe(preset: str = DEFAULT_PRESET, overrides: dict | None = None,
           length: int = DEFAULT_LENGTH) -> dict:
    """A recipe: a preset's effects with an operator's changes merged in.
    `overrides` = {effect: {key: value}}; unknown effects or keys refuse."""
    if preset not in PRESETS:
        raise _store.LibraryError(f"a preset is one of {', '.join(PRESETS)}")
    eff = copy.deepcopy(PRESETS[preset]["effects"])
    for name, changes in (overrides or {}).items():
        if name not in eff:
            raise _store.LibraryError(f"'{name}' is not an effect "
                                      f"({', '.join(EFFECT_ORDER)})")
        for k, v in dict(changes).items():
            if k not in eff[name]:
                raise _store.LibraryError(f"the {name} effect has no "
                                          f"setting '{k}'")
            eff[name][k] = v
    _validate(eff)
    return {"preset": preset, "length": int(length), "effects": eff}


def _rng_pair(v, lo, hi, what):
    if isinstance(v, (list, tuple)):
        if len(v) != 2:
            raise _store.LibraryError(f"{what} is a range [low, high]")
        a, b = float(v[0]), float(v[1])
        if b < a:
            raise _store.LibraryError(f"{what}: the high end is below the low")
        if a < lo or b > hi:
            raise _store.LibraryError(f"{what} must stay within {lo:g}…{hi:g}")
    else:
        a = float(v)
        if a < lo or a > hi:
            raise _store.LibraryError(f"{what} must stay within {lo:g}…{hi:g}")


def _validate(eff: dict) -> None:
    limits = {"ppm": (0, 1000), "doppler_hz": (0, 2000), "k_db": (-30, 40),
              "delay_spread_us": (0, 50), "fraction_of_bw": (0, 0.5),
              "drift_hz_per_s": (0, 10_000), "linewidth_hz": (0, 1000),
              "inr_db": (-60, 10), "rate_hz": (0, 1000), "level_db": (-60, 40),
              "snr_db": (-20, 60), "depth_db": (0, 30),
              "rejection_db": (5, 80), "headroom_db": (0, 30), "bits": (2, 16),
              "ms": (0.01, 100)}
    for name, e in eff.items():
        p = float(e.get("p", 1.0))
        if not 0.0 <= p <= 1.0:
            raise _store.LibraryError(f"{name}: how often is 0–100 %")
        for k, v in e.items():
            if k in limits:
                _rng_pair(v, *limits[k], f"{EFFECT_WORDS[name][0]}: {k}")


def recipe_hash(rec: dict) -> str:
    return hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()[:12]


def describe(rec: dict) -> list[str]:
    """The recipe in words, one line per effect that can happen."""
    out = []
    for name in EFFECT_ORDER:
        e = rec["effects"][name]
        if not e.get("on") or float(e.get("p", 0)) <= 0:
            continue
        title = EFFECT_WORDS[name][0]
        how = f"{100 * float(e['p']):.0f} % of copies"
        detail = ", ".join(f"{k} {_fmt(v)}" for k, v in e.items()
                           if k not in ("on", "p"))
        out.append(f"{title}: {how}" + (f" — {detail}" if detail else ""))
    return out


def _fmt(v) -> str:
    if isinstance(v, (list, tuple)):
        return "–".join(_fmt(x) for x in v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _draw(rng, v):
    if isinstance(v, (list, tuple)):
        a, b = float(v[0]), float(v[1])
        return float(rng.uniform(a, b)) if b > a else a
    return float(v)


def _sym(rng, v):
    """A symmetric draw ±v."""
    return float(rng.uniform(-abs(float(v)), abs(float(v))))


# ---------------------------------------------------------------------------
# one copy
# ---------------------------------------------------------------------------
def make_one(x, fs: float, facts: dict, rec: dict, rng) -> tuple[np.ndarray, list]:
    """One dirty copy of `x` (complex, at the canonical rate `fs`). `facts`
    are the original's: bandwidth_hz, floor_per_hz, signal_power, snr_db,
    lowpass_hz. -> (copy, [applied effects])."""
    x = np.asarray(x, np.complex128).copy()
    n = x.size
    t = np.arange(n) / float(fs)
    eff = rec["effects"]
    applied: list[dict] = []
    bw = float(facts.get("bandwidth_hz") or 0.25 * fs)
    lp = facts.get("lowpass_hz")
    floor = float(facts.get("floor_per_hz") or 0.0)
    s_pow = float(facts.get("signal_power") or 0.0)
    snr0 = facts.get("snr_db")

    def happens(name: str) -> bool:
        e = eff[name]
        return bool(e.get("on")) and rng.uniform() < float(e.get("p", 1.0))

    def add(name, words, **params):
        applied.append({"name": name, "words": words,
                        "params": {k: (round(v, 6) if isinstance(v, float) else v)
                                   for k, v in params.items()}})

    rms = float(np.sqrt(np.mean(np.abs(x) ** 2))) or 1e-12
    if happens("clock"):
        ppm = _sym(rng, eff["clock"]["ppm"])
        r = 1.0 + ppm * 1e-6
        src = np.arange(n) * r
        src = np.clip(src, 0, n - 1)
        x = np.interp(src, np.arange(n), x.real) + 1j * np.interp(src, np.arange(n), x.imag)
        add("clock", f"transmitter clock {ppm:+.1f} ppm", ppm=ppm)
    if happens("fading"):
        from atk_diffusion.synth.scene import apply_channel
        e = eff["fading"]
        fd = _draw(rng, e["doppler_hz"])
        k = _draw(rng, e["k_db"])
        ds = _draw(rng, e["delay_spread_us"]) * 1e-6
        x, _los = apply_channel(x, fs, {"rms_delay_s": ds, "k_factor_db": k}, fd, rng)
        add("fading", f"fading: Doppler {fd:.1f} Hz, K {k:+.1f} dB, delay "
                      f"spread {ds * 1e6:.1f} µs", doppler_hz=fd, k_db=k,
            delay_spread_s=ds)
    if happens("offset"):
        e = eff["offset"]
        f0 = _sym(rng, float(e["fraction_of_bw"]) * bw)
        drift = _sym(rng, e["drift_hz_per_s"])
        # keep the signal inside the cut's low-pass
        if lp:
            room = max(0.0, float(lp) - 0.5 * bw)
            f0 = float(np.clip(f0, -room, room))
        x = x * np.exp(2j * np.pi * (f0 * t + 0.5 * drift * t * t))
        add("offset", f"frequency error {f0:+.0f} Hz, drift {drift:+.1f} Hz/s",
            offset_hz=f0, drift_hz_per_s=drift)
    if happens("phase_noise"):
        lw = _draw(rng, eff["phase_noise"]["linewidth_hz"])
        steps = rng.normal(0.0, math.sqrt(2 * math.pi * lw / float(fs)), n)
        x = x * np.exp(1j * np.cumsum(steps))
        add("phase_noise", f"phase noise, {lw:.1f} Hz linewidth", linewidth_hz=lw)
    sig_ref = s_pow if s_pow > 0 else rms ** 2
    if happens("interferer"):
        e = eff["interferer"]
        kinds = list(e.get("kinds") or ["tone"])
        kind = str(rng.choice(kinds))
        inr = _draw(rng, e["inr_db"])
        amp = math.sqrt(sig_ref * 10 ** (inr / 10.0))
        edge = float(lp) if lp else 0.45 * fs
        f_i = float(rng.uniform(-edge, edge))
        if kind == "tone":
            i_sig = amp * np.exp(2j * np.pi * f_i * t + 1j * rng.uniform(0, 2 * np.pi))
        elif kind == "sweep":
            span = float(rng.uniform(0.2, 1.0)) * edge
            rate = float(rng.uniform(0.5, 10.0))
            ph = 2 * np.pi * (f_i * t + span / (2 * np.pi * rate)
                              * np.sin(2 * np.pi * rate * t))
            i_sig = amp * np.exp(1j * ph)
        else:
            b = float(rng.uniform(0.05, 0.3)) * bw
            w = rng.standard_normal(n + 64) + 1j * rng.standard_normal(n + 64)
            w = _store.lowpass(w, max(b, 0.002 * fs), fs)[32:32 + n]
            w = w / (np.sqrt(np.mean(np.abs(w) ** 2)) or 1.0)
            i_sig = amp * w * np.exp(2j * np.pi * f_i * t)
        x = x + i_sig
        add("interferer", f"interference: {kind} at {f_i:+.0f} Hz, "
                          f"{inr:+.1f} dB relative to the signal",
            kind=kind, offset_hz=f_i, inr_db=inr)
    if happens("impulses"):
        e = eff["impulses"]
        rate = _draw(rng, e["rate_hz"])
        lvl = _draw(rng, e["level_db"])
        k = int(rng.poisson(rate * n / fs))
        a = math.sqrt(sig_ref) * 10 ** (lvl / 20.0)
        dec = max(1.0, 20e-6 * fs)
        for _ in range(k):
            at = int(rng.integers(0, n))
            m = min(n - at, int(8 * dec) + 1)
            env = np.exp(-np.arange(m) / dec)
            x[at:at + m] += a * env * np.exp(1j * rng.uniform(0, 2 * np.pi))
        add("impulses", f"{k} clicks, {lvl:+.0f} dB over the signal",
            count=k, level_db=lvl)
    if happens("noise"):
        target = _draw(rng, eff["noise"]["snr_db"])
        if snr0 is None or s_pow <= 0 or bw <= 0:
            add("noise", "noise not added: the original's SNR could not be "
                         "measured, so a target SNR would mean nothing",
                skipped=True)
        elif target >= float(snr0):
            add("noise", f"no noise added: asked for {target:.1f} dB, the "
                         f"original is already {float(snr0):.1f} dB",
                snr_db=float(snr0), asked_db=target)
        else:
            n_add = s_pow / (bw * 10 ** (target / 10.0)) - floor
            if n_add > 0:
                noise = _store.floor_noise(n, n_add, fs, lp, rng)
                x = x + noise
            add("noise", f"noise added: {target:.1f} dB SNR in the signal's "
                         f"band", snr_db=target)
    if happens("agc"):
        e = eff["agc"]
        depth = _draw(rng, e["depth_db"])
        rate = _draw(rng, e["rate_hz"])
        g = 10 ** ((depth / 20.0) * 0.5 * np.sin(2 * np.pi * rate * t
                                                  + rng.uniform(0, 2 * np.pi)))
        x = x * g
        add("agc", f"gain pumping ±{depth / 2:.1f} dB at {rate:.1f} Hz",
            depth_db=depth, rate_hz=rate)
    rms_now = float(np.sqrt(np.mean(np.abs(x) ** 2))) or 1e-12
    if happens("dc"):
        lvl = _draw(rng, eff["dc"]["level_db"])
        a = rms_now * 10 ** (lvl / 20.0)
        x = x + a * np.exp(1j * rng.uniform(0, 2 * np.pi))
        add("dc", f"DC spike {lvl:+.0f} dB relative to the signal", level_db=lvl)
    if happens("image"):
        rej = _draw(rng, eff["image"]["rejection_db"])
        x = x + 10 ** (-rej / 20.0) * np.conj(x)
        add("image", f"mirror image {rej:.0f} dB down", rejection_db=rej)
    if happens("clip"):
        hd = _draw(rng, eff["clip"]["headroom_db"])
        peak = max(float(np.max(np.abs(x.real))), float(np.max(np.abs(x.imag))),
                   1e-12)
        lvl = peak * 10 ** (-hd / 20.0)
        before = float(np.mean((np.abs(x.real) > lvl) | (np.abs(x.imag) > lvl)))
        x = np.clip(x.real, -lvl, lvl) + 1j * np.clip(x.imag, -lvl, lvl)
        add("clip", f"clipped {hd:.1f} dB below the peak ({100 * before:.1f} % "
                    f"of samples)", headroom_db=hd, clipped_fraction=before)
    if happens("bits"):
        b = int(round(_draw(rng, eff["bits"]["bits"])))
        fsc = 1.05 * max(float(np.max(np.abs(x.real))),
                         float(np.max(np.abs(x.imag))), 1e-12)
        q = fsc / (2 ** (b - 1))
        x = np.round(x.real / q) * q + 1j * np.round(x.imag / q) * q
        add("bits", f"{b} effective bits", bits=b)
    if happens("dropouts"):
        e = eff["dropouts"]
        rate = _draw(rng, e["rate_hz"])
        k = int(rng.poisson(rate * n / fs))
        lost = 0
        for _ in range(k):
            ms = _draw(rng, e["ms"])
            m = max(1, int(ms * 1e-3 * fs))
            at = int(rng.integers(0, max(1, n - m)))
            x[at:at + m] = 0
            lost += m
        add("dropouts", f"{k} dropouts, {lost} samples lost", count=k,
            samples_lost=lost)
    if lp and float(lp) < 0.49 * fs:
        x = _store.lowpass(x, float(lp), fs)
    return np.asarray(x, np.complex64), applied


def _facts(ex_meta: dict) -> dict:
    g = ex_meta.get("global", {})
    m = g.get("atk:measured") or {}
    return {"bandwidth_hz": m.get("bandwidth_hz"),
            "floor_per_hz": m.get("floor_per_hz"),
            "signal_power": m.get("signal_power"),
            "snr_db": m.get("snr_db"),
            "lowpass_hz": g.get("atk:lowpass_hz")}


def _crop(rng, span: tuple, length: int) -> tuple[int, int]:
    s0, n = int(span[0]), int(span[1])
    if n <= length:
        return s0, n
    a = int(rng.integers(s0, s0 + n - length + 1))
    return a, int(length)


def preview(lib: _store.TrainingLibrary, type_name: str, rec: dict,
            n: int = 6, seed: int = 0) -> list[dict]:
    """A few copies made in memory — nothing written — for the tab to show
    before anything is filed: [{x, fs, parent, effects (words)}]."""
    st = lib.get_type(type_name)
    origs = [e for e in lib.examples(st.name, "originals")
             if (e.role or "train") == "train"]
    if not origs:
        raise _store.LibraryError(f"'{st.words}' has no real original used for "
                                  "training yet — add one first (the ones "
                                  "kept aside to grade the model are never "
                                  "copied)")
    rng = np.random.default_rng(int(seed))
    out = []
    for i in range(int(n)):
        ex = origs[i % len(origs)]
        meta = lib.meta(ex)
        span = lib.train_span(ex)
        a, cnt = _crop(rng, span, int(rec.get("length", DEFAULT_LENGTH)))
        x = lib.load(ex, a, cnt)
        y, applied = make_one(x, ex.rate, _facts(meta), rec, rng)
        out.append({"x": y, "fs": ex.rate, "parent": ex.id,
                    "effects": [e["words"] for e in applied]})
    return out


def make_dirty(lib: _store.TrainingLibrary, type_name: str, per_original: int,
               rec: dict, seed: int = 0, progress=None, should_stop=None,
               originals: list[str] | None = None) -> dict:
    """`per_original` dirty copies of every TRAIN original of a type (or of
    the ids in `originals`), filed under <type>\\dirty\\. -> a summary."""
    st = lib.get_type(type_name)
    per = int(per_original)
    if per < 1 or per > MAX_PER_ORIGINAL:
        raise _store.LibraryError(f"copies per original is 1–{MAX_PER_ORIGINAL}")
    origs = [e for e in lib.examples(st.name, "originals")
             if (e.role or "train") == "train"
             and (originals is None or e.id in originals)]
    if not origs:
        raise _store.LibraryError(f"'{st.words}' has no real original used for "
                                  "training — add one first")
    rh = recipe_hash(rec)
    made, t0 = 0, time.time()
    total = per * len(origs)
    seeds = np.random.SeedSequence(int(seed)).spawn(len(origs))
    skipped = []
    for oi, ex in enumerate(origs):
        meta = lib.meta(ex)
        facts = _facts(meta)
        span = lib.train_span(ex)
        ids = lib.next_copy_ids(st.name, "dirty", ex.id, per)
        rng = np.random.default_rng(seeds[oi])
        for k, cid in enumerate(ids):
            if should_stop is not None and should_stop():
                return _summary(st, "dirty", made, total, t0, skipped, True)
            a, cnt = _crop(rng, span, int(rec.get("length", DEFAULT_LENGTH)))
            x = lib.load(ex, a, cnt)
            if x.size < _store.WINDOW:
                x = _store.pad_with_floor(x, _store.WINDOW,
                                          float(facts["floor_per_hz"] or 1e-12),
                                          ex.rate, facts["lowpass_hz"], rng)
            y, applied = make_one(x, ex.rate, facts, rec, rng)
            snr_out = facts["snr_db"]
            off = 0.0
            for e in applied:
                if e["name"] == "noise" and "snr_db" in e["params"]:
                    snr_out = e["params"]["snr_db"]
                if e["name"] == "offset":
                    off = float(e["params"].get("offset_hz", 0.0))
            extra = {"atk:parent": ex.id, "atk:crop": {"start": int(a),
                                                       "count": int(cnt)},
                     "atk:effects": applied, "atk:recipe": {
                         "preset": rec.get("preset"), "hash": rh},
                     "atk:seed": [int(seed), oi, k],
                     "atk:snr_db": snr_out,
                     "atk:carrier_offset_hz": off + float(
                         (meta["global"].get("atk:measured") or {}).get("centre_hz") or 0.0),
                     "atk:measured": meta["global"].get("atk:measured"),
                     "atk:lowpass_hz": facts["lowpass_hz"],
                     "atk:canonical_rate": ex.rate}
            centre = float(meta["captures"][0].get("core:frequency", 0.0))
            lib.write_copy(st.name, "dirty", cid, y, ex.rate, centre, extra,
                           f"dirty copy of {ex.id} ({st.words})")
            made += 1
            if progress is not None and (made == total or made % max(1, total // 50) == 0):
                progress(f"[{100 * made // total}%] dirty copies of {st.words}: "
                         f"{made}/{total}")
    return _summary(st, "dirty", made, total, t0, skipped, False)


def _summary(st, kind, made, total, t0, skipped, stopped) -> dict:
    return {"type": st.name, "kind": kind, "made": int(made),
            "asked": int(total), "seconds": round(time.time() - t0, 2),
            "skipped": skipped, "stopped": bool(stopped),
            "words": (f"{made} dirty cop{'ies' if made != 1 else 'y'} of "
                      f"{st.words}"
                      + (" (stopped before the end)" if stopped else ""))}
