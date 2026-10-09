# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Point-and-ask on the waterfall — the cheap version (plan §4.B5;
DETECTION_DESIGN §4.2, Route → point-and-ask).

Bill, 2026-10-08, on point-and-ask and teach: *"we definitely have to do"*.

WHAT IT IS. The analyst selects a VFO, a box or a cut; the crop of the
waterfall around it goes to ATK's primary model (Gemma 4 takes images)
together with the bench's measurements — centre, bandwidth, symbol rate, PRI,
the detector's class and confidence, the fingerprint if known — and the
analyst asks in words. RF-GPT (plan §9, 2602.14833) is the proper version: a
spectrogram encoder trained on profile data and aligned to the language
model. This is the cheap version that ships first, and the plan says it is
to be MEASURED against RF-GPT, not assumed equal to it (`compare_with_without`
is the first experiment's harness).

HOW, AND WHY THIS WAY.

* `build_payload` makes three things: the crop as a PNG, the prompt, and the
  facts. The PNG is encoded by a few lines of pure Python (zlib + struct —
  no Pillow, no Qt; ATK's core environment needs nothing new) in ATK's OWN
  waterfall colour ramp (blue → yellow, the stops in ATK's
  `atk/core/levels.py`, ported here), so the model is shown what the analyst
  sees on the screen and an exported crop reads the same way.
* The image alone is ambiguous, so the prompt DESCRIBES THE AXES in words —
  which way time runs, the frequency at each edge, and what the colours
  mean in dB. A vision model cannot read an axis that is not drawn.
* The bench's numbers are FACTS with tiers: a measurement is MEASURED
  (classical, checkable), the detector's class is PROPOSED until a decoder
  confirms it. The prompt lists what was not measured as "not measured", so
  a missing number is not mistaken for permission to invent one.
* Every prompt ends with the explicit way out — Bill's long-standing rule
  for small models: an instruction that ends with "or say I don't know" cuts
  hallucination. `ask` reports `said_dont_know` so the way out is counted,
  not just offered.
* `ask(payload, question, model)` calls a HOST-SUPPLIED
  `model(prompt, image_png) -> text` (ATK's cognitive core). The toolkit
  never loads a language model itself (ARCHITECTURE §6: point-and-ask's chat
  turn is ATK's). `message_for` builds ATK's own OpenAI-style message (a
  data URI, as `atk/core/vision.py` passes images) for hosts that prefer it.

THE TIER. An answer is a HYPOTHESIS, never an identification: it is returned
with tier PROPOSED and a sentence that says so. A language model looking at a
picture of a spectrum for a second is not a decoder; only a decoder confirms
(plan §2.1, DETECTION_DESIGN §5).

LIMITS, stated. The cheap version knows only what the crop and the listed
numbers show. It cannot tell QPSK from 8PSK from a spectrogram any better
than the 2D proposer can, and it has not seen Bill's receivers — so its
accuracy is a number to be measured on Bill's own captures (the first
experiment: ten signals, blind, with and without the bench measurements),
never a claim.
"""

from __future__ import annotations

import base64
import math
import re
import struct
import zlib
from typing import Callable

import numpy as np

from atk_diffusion import provenance

provenance.METHOD_TIERS.setdefault("point_ask", "proposed")

METHOD = "point_ask"
TIER = provenance.tier_for(METHOD)

#: The waterfall's colour stops, position 0..1 -> (r, g, b). Ported from ATK's
#: `atk/core/levels.py` (Bill's code): the ramp an operator already reads.
WF_STOPS = [
    (0.00, (8, 14, 40)),        # near-black blue: below the floor
    (0.18, (18, 46, 110)),      # deep blue: the floor itself
    (0.38, (30, 96, 170)),      # mid blue
    (0.55, (60, 150, 190)),     # cyan-ish: something is there
    (0.72, (140, 195, 150)),    # the blue-to-yellow crossover
    (0.88, (230, 220, 90)),     # yellow: a real signal
    (1.00, (255, 250, 190)),    # pale yellow-white: strong
]

#: The explicit way out every prompt ends with (Bill's rule for small models).
WAY_OUT = "I don't know"

DEFAULT_QUESTION = "What is this signal, and what tells you so?"

HYPOTHESIS_WORDS = ("A HYPOTHESIS from the language model, not an "
                    "identification (PROPOSED tier). Check it against a "
                    "decoder or the bench before relying on it.")

_PNG_SIG = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# Colour and PNG — pure Python
# ---------------------------------------------------------------------------
def ramp_rgb(level_db, lo_db: float = 0.0, hi_db: float = 40.0) -> np.ndarray:
    """Levels -> uint8 RGB in ATK's waterfall ramp, clamped to [lo, hi].
    Vectorised; identical to ATK's scalar `levels.rgb_for` (a test holds the
    two together)."""
    x = np.asarray(level_db, dtype=np.float64)
    span = float(hi_db) - float(lo_db)
    if span <= 0:
        t = np.zeros_like(x)
    else:
        t = (x - float(lo_db)) / span
    t = np.where(np.isfinite(t), t, 0.0)
    t = np.clip(t, 0.0, 1.0)
    pos = np.array([p for p, _c in WF_STOPS], dtype=np.float64)
    out = np.empty(t.shape + (3,), dtype=np.uint8)
    for ch in range(3):
        vals = np.array([c[ch] for _p, c in WF_STOPS], dtype=np.float64)
        out[..., ch] = np.round(np.interp(t, pos, vals)).astype(np.uint8)
    return out


def _chunk(kind: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))


def _latin1(text: str) -> bytes:
    s = str(text).replace("—", "-").replace("–", "-").replace("→", "->")
    return s.encode("latin-1", errors="replace")


def encode_png(rgb, text: dict | None = None, level: int = 6) -> bytes:
    """An 8-bit RGB PNG from an (H, W, 3) uint8 array, with optional tEXt
    chunks (keyword -> text). Filter type 0 on every row: simple, exact, and
    small enough for a crop."""
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("a PNG crop must be an (height, width, 3) RGB array")
    if a.shape[0] < 1 or a.shape[1] < 1:
        raise ValueError("an empty crop cannot be drawn")
    a = np.ascontiguousarray(a.astype(np.uint8, copy=False))
    h, w = a.shape[:2]
    raw = np.empty((h, 1 + 3 * w), dtype=np.uint8)
    raw[:, 0] = 0
    raw[:, 1:] = a.reshape(h, 3 * w)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    parts = [_PNG_SIG, _chunk(b"IHDR", ihdr)]
    for key, val in (text or {}).items():
        k = _latin1(key)[:79]
        parts.append(_chunk(b"tEXt", k + b"\x00" + _latin1(val)))
    parts.append(_chunk(b"IDAT", zlib.compress(raw.tobytes(), level)))
    parts.append(_chunk(b"IEND", b""))
    return b"".join(parts)


def decode_png(blob: bytes) -> tuple[np.ndarray, dict]:
    """Read back what `encode_png` writes (8-bit RGB, filter 0) — for tests
    and for a host that wants to check a payload. -> (rgb, text)."""
    if not blob.startswith(_PNG_SIG):
        raise ValueError("not a PNG")
    i = len(_PNG_SIG)
    w = h = 0
    idat = b""
    text = {}
    while i < len(blob):
        (n,) = struct.unpack(">I", blob[i:i + 4])
        kind = blob[i + 4:i + 8]
        data = blob[i + 8:i + 8 + n]
        (crc,) = struct.unpack(">I", blob[i + 8 + n:i + 12 + n])
        if zlib.crc32(kind + data) & 0xFFFFFFFF != crc:
            raise ValueError(f"PNG chunk {kind!r} has a bad CRC")
        if kind == b"IHDR":
            w, h, depth, ctype = struct.unpack(">IIBB", data[:10])
            if depth != 8 or ctype != 2:
                raise ValueError("only 8-bit RGB is read here")
        elif kind == b"IDAT":
            idat += data
        elif kind == b"tEXt":
            k, _, v = data.partition(b"\x00")
            text[k.decode("latin-1")] = v.decode("latin-1")
        i += 12 + n
    raw = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(h, 1 + 3 * w)
    if np.any(raw[:, 0] != 0):
        raise ValueError("only filter type 0 is read here")
    return raw[:, 1:].reshape(h, w, 3).copy(), text


def _resize_axis(a: np.ndarray, n_out: int, axis: int) -> np.ndarray:
    """Nearest-neighbour up, MAX-pool down — so a one-row burst is never
    averaged into the floor when the crop is shrunk."""
    n_in = a.shape[axis]
    if n_out == n_in:
        return a
    if n_out > n_in:
        idx = np.minimum((np.arange(n_out) * n_in) // n_out, n_in - 1)
        return np.take(a, idx, axis=axis)
    edges = np.linspace(0, n_in, n_out + 1).round().astype(int)
    parts = [np.max(np.take(a, np.arange(edges[k], max(edges[k] + 1, edges[k + 1])),
                            axis=axis), axis=axis, keepdims=True)
             for k in range(n_out)]
    return np.concatenate(parts, axis=axis)


def fit_size(level: np.ndarray, min_px: int = 256, max_px: int = 1024
             ) -> np.ndarray:
    """Scale a (rows, bins) level array so its short side is at least
    `min_px` and its long side at most `max_px` (a vision model reads a few
    hundred pixels well; a 30x40 crop is a smudge)."""
    a = np.asarray(level, dtype=np.float64)
    h, w = a.shape
    up = max(1.0, float(min_px) / float(min(h, w)))
    nh, nw = int(round(h * up)), int(round(w * up))
    down = min(1.0, float(max_px) / float(max(nh, nw)))
    nh, nw = max(1, int(round(nh * down))), max(1, int(round(nw * down)))
    return _resize_axis(_resize_axis(a, nh, 0), nw, 1)


def auto_levels(spec_db) -> tuple[float, float]:
    """A colour window that shows the crop: the 5th percentile to the 99.9th,
    at least 10 dB wide. Stated in the prompt, so the model knows the scale."""
    a = np.asarray(spec_db, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 40.0
    lo = float(np.percentile(a, 5.0))
    hi = float(np.percentile(a, 99.9))
    if hi - lo < 10.0:
        hi = lo + 10.0
    return lo, hi


# ---------------------------------------------------------------------------
# Crop and words
# ---------------------------------------------------------------------------
def hz_words(f) -> str:
    """'462.5625 MHz', '12.5 kHz', '1.09 GHz', '512 Hz'."""
    if f is None:
        return "not measured"
    v = float(f)
    a = abs(v)
    for scale, unit in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz")):
        if a >= scale:
            s = f"{v / scale:.6f}".rstrip("0").rstrip(".")
            return f"{s} {unit}"
    return f"{v:.6g} Hz"


def seconds_words(s) -> str:
    if s is None:
        return "not measured"
    v = float(s)
    if abs(v) >= 1.0:
        return f"{v:.3g} s"
    if abs(v) >= 1e-3:
        return f"{v * 1e3:.3g} ms"
    return f"{v * 1e6:.3g} µs"


def crop_around(spec_db, t_axis, f_axis, t0: float, t1: float, f_lo: float,
                f_hi: float, margin: float = 0.5, min_rows: int = 16,
                min_bins: int = 16) -> tuple[np.ndarray, dict]:
    """The waterfall around a box: `spec_db` is (frames, bins), `t_axis` the
    frame times (s), `f_axis` the bin centres (absolute Hz). `margin` is the
    fraction of the box added on each side. -> (crop, axes)."""
    s = np.asarray(spec_db)
    t = np.asarray(t_axis, dtype=np.float64)
    f = np.asarray(f_axis, dtype=np.float64)
    if s.shape != (t.size, f.size):
        raise ValueError(f"the spectrogram is {s.shape}; the axes say "
                         f"({t.size}, {f.size})")
    dt, df = max(t1 - t0, 0.0), max(f_hi - f_lo, 0.0)
    ta, tb = t0 - margin * dt, t1 + margin * dt
    fa, fb = f_lo - margin * df, f_hi + margin * df
    r0, r1 = int(np.searchsorted(t, ta, "left")), int(np.searchsorted(t, tb, "right"))
    b0, b1 = int(np.searchsorted(f, fa, "left")), int(np.searchsorted(f, fb, "right"))

    def _widen(a, b, n_min, n):
        if b - a >= n_min:
            return max(0, a), min(n, b)
        c = (a + b) // 2
        a2 = max(0, c - n_min // 2)
        return a2, min(n, a2 + n_min)
    r0, r1 = _widen(r0, r1, min_rows, t.size)
    b0, b1 = _widen(b0, b1, min_bins, f.size)
    crop = s[r0:r1, b0:b1]
    half_bin = 0.5 * (f[1] - f[0]) if f.size > 1 else 0.0
    half_row = 0.5 * (t[1] - t[0]) if t.size > 1 else 0.0
    axes = {"t0": float(t[r0] - half_row), "t1": float(t[r1 - 1] + half_row),
            "f_lo": float(f[b0] - half_bin), "f_hi": float(f[b1 - 1] + half_bin),
            "box": {"t0": t0, "t1": t1, "f_lo": f_lo, "f_hi": f_hi}}
    return crop, axes


_ALIASES = {
    "centre_hz": ("centre_hz", "center_hz", "center_freq_hz", "centre_freq_hz",
                  "f_center_hz", "fc_hz", "center", "centre"),
    "bandwidth_hz": ("bandwidth_hz", "occupied_bw_hz", "occupied_bandwidth_hz",
                     "obw_hz", "bw_hz", "bandwidth"),
    "symbol_rate_hz": ("symbol_rate_hz", "symbol_rate", "sym_rate_hz", "baud",
                       "symbol_rate_sps"),
    "pri_s": ("pri_s", "pri", "pri_seconds"),
    "snr_db": ("snr_db", "snr", "snr_above_floor_db"),
    "burst_s": ("burst_length_s", "burst_s", "pulse_width_s", "duration_s"),
    "duty": ("duty", "duty_cycle"),
    "carrier_offset_hz": ("carrier_offset_hz", "carrier_offset"),
}
_CLASS_KEYS = ("class", "cls", "classification", "label", "class_name")
_CONF_KEYS = ("confidence", "score", "class_confidence")
_FP_KEYS = ("fingerprint", "fingerprint_match", "emitter", "emitter_id")


def _pick(d: dict, keys) -> object:
    for k in keys:
        if isinstance(d, dict) and d.get(k) not in (None, ""):
            return d[k]
    return None


def _num(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def facts_from(detection_or_analysis=None, measurements: dict | None = None
               ) -> dict:
    """The bench's facts, flat, in SI units, with the tier of each.

    `detection_or_analysis` is a `detect.boxes.Detection`, its `to_json()`
    dict, or a cut's `analysis.json` dict; `measurements` (the bench's
    classical numbers) wins where both say something."""
    det = detection_or_analysis
    src: dict = {}
    meas: dict = {}
    if det is not None and hasattr(det, "to_json") and hasattr(det, "f_lo"):
        det = det.to_json()
    if isinstance(det, dict):
        src = dict(det)
        if {"f_lo", "f_hi"} <= set(src):
            meas["centre_hz"] = 0.5 * (float(src["f_lo"]) + float(src["f_hi"]))
            meas["bandwidth_hz"] = float(src["f_hi"]) - float(src["f_lo"])
            if src.get("t1") is not None and src.get("t0") is not None:
                meas["burst_s"] = float(src["t1"]) - float(src["t0"])
        if src.get("alpha_hz"):
            meas.setdefault("symbol_rate_hz", src.get("alpha_hz"))
        inner = src.get("measurements") or {}
        if isinstance(inner, dict):
            for key, names in _ALIASES.items():
                v = _pick(inner, names)
                if v is not None:
                    meas[key] = v
        for key, names in _ALIASES.items():
            v = _pick(src, names)
            if v is not None and key not in ("burst_s",):
                meas.setdefault(key, v)
    for key, names in _ALIASES.items():
        v = _pick(measurements or {}, names)
        if v is not None:
            meas[key] = v
    facts: dict = {k: _num(meas.get(k)) for k in _ALIASES}
    cls = _pick(src, _CLASS_KEYS)
    facts["class"] = str(cls) if cls not in (None, "") else None
    facts["confidence"] = _num(_pick(src, _CONF_KEYS))
    facts["state"] = str(src.get("state") or "proposed") if src else None
    facts["confirmed_by"] = str(src.get("confirmed_by") or "") or None
    fp = _pick(src, _FP_KEYS)
    if fp is None:
        fp = _pick(measurements or {}, _FP_KEYS)
    facts["fingerprint"] = fp if fp not in (None, "") else None
    facts["profile"] = str(src.get("profile") or "") or None
    facts["sources"] = list(src.get("sources") or []) or None
    facts["flags"] = list(src.get("flags") or []) or None
    tiers = {k: "measured" for k in _ALIASES if facts.get(k) is not None}
    if facts["class"]:
        tiers["class"] = "confirmed" if facts["state"] == "confirmed" else "proposed"
    if facts["fingerprint"] is not None:
        tiers["fingerprint"] = "proposed"
    facts["tiers"] = tiers
    return facts


def _fp_words(fp) -> str:
    if fp is None:
        return "not known"
    if isinstance(fp, dict):
        name = fp.get("name") or fp.get("id") or fp.get("emitter") or "an emitter"
        dist = fp.get("distance")
        return (f"matches {name}" + (f" (distance {float(dist):.3g})"
                                     if _num(dist) is not None else "")
                + " — a match, not a confirmation")
    return f"{fp} — a match, not a confirmation"


def fact_lines(facts: dict) -> list[str]:
    """The facts as the lines the prompt (and the Cuts viewer) shows."""
    f = facts or {}
    lines = ["THE BENCH'S MEASUREMENTS (classical, checkable):",
             f"- centre frequency: {hz_words(f.get('centre_hz'))}",
             f"- occupied bandwidth: {hz_words(f.get('bandwidth_hz'))}",
             "- symbol rate: " + (f"{f['symbol_rate_hz']:.6g} symbols/s"
                                  if f.get("symbol_rate_hz") is not None
                                  else "not measured"),
             "- pulse repetition interval (PRI): "
             + seconds_words(f.get("pri_s"))]
    if f.get("burst_s") is not None:
        lines.append(f"- burst length: {seconds_words(f['burst_s'])}")
    if f.get("duty") is not None:
        lines.append(f"- duty cycle: {100.0 * float(f['duty']):.3g} %")
    if f.get("snr_db") is not None:
        lines.append(f"- level: {float(f['snr_db']):.1f} dB above the noise floor")
    if f.get("carrier_offset_hz") is not None:
        lines.append(f"- carrier offset: {hz_words(f['carrier_offset_hz'])}")
    lines.append("THE DETECTOR'S OPINION (a proposal unless a decoder "
                 "confirmed it):")
    if f.get("class"):
        conf = (f", confidence {float(f['confidence']):.2f}"
                if f.get("confidence") is not None else "")
        if f.get("state") == "confirmed":
            how = (f"CONFIRMED by the {f['confirmed_by']} decoder"
                   if f.get("confirmed_by") else "CONFIRMED by a decoder")
        else:
            how = "PROPOSED — no decoder has confirmed it"
        lines.append(f"- class: {f['class']}{conf} — {how}")
    else:
        lines.append("- class: none given")
    lines.append(f"- fingerprint: {_fp_words(f.get('fingerprint'))}")
    return lines


def image_text(axes: dict, shape: tuple, lo_db: float, hi_db: float,
               rows_oldest_first: bool = True) -> str:
    """The words that make the picture readable: axes, direction, scale."""
    a = axes or {}
    h, w = int(shape[0]), int(shape[1])
    ref = str(a.get("db_ref", "above_floor"))
    ref_words = {"above_floor": "dB above the measured noise floor",
                 "dbfs": "dB relative to the converter's full scale (dBFS)"
                 }.get(ref, f"dB ({ref})")
    f_lo, f_hi = a.get("f_lo"), a.get("f_hi")
    t0, t1 = a.get("t0"), a.get("t1")
    if f_lo is not None and f_hi is not None:
        fw = (f"Frequency runs left to right, from {hz_words(f_lo)} at the "
              f"left edge to {hz_words(f_hi)} at the right edge "
              f"({hz_words(float(f_hi) - float(f_lo))} across).")
    else:
        fw = "Frequency runs left to right; its edges were not given."
    if t0 is not None and t1 is not None:
        first, last = (t0, t1) if rows_oldest_first else (t1, t0)
        order = ("earliest" if rows_oldest_first else "latest")
        tw = (f"Time runs top to bottom: the top row is the {order} moment "
              f"({float(first):.3f} s), the bottom row {float(last):.3f} s "
              f"({seconds_words(abs(float(t1) - float(t0)))} in all).")
    else:
        tw = "Time runs top to bottom; its span was not given."
    cw = (f"Colour is signal level in {ref_words}: near-black blue is "
          f"{lo_db:.1f} dB or less, deep blue is the floor, cyan means "
          f"something is there, yellow is a real signal and pale "
          f"yellow-white is {hi_db:.1f} dB or more.")
    box = a.get("box")
    bw = ""
    if isinstance(box, dict) and box.get("f_lo") is not None:
        bw = (f" The selected signal is the box from {hz_words(box['f_lo'])} "
              f"to {hz_words(box['f_hi'])}, near the middle of the picture.")
    return (f"THE IMAGE: a waterfall (spectrogram) crop, {w} pixels wide and "
            f"{h} pixels tall. {fw} {tw} {cw}{bw} Nothing else is drawn: no "
            "axes, labels or text.")


def compose_prompt(image_words: str, facts: dict, question: str,
                   include_facts: bool = True) -> str:
    """The whole prompt. It ENDS with the explicit way out."""
    q = (question or "").strip() or DEFAULT_QUESTION
    parts = ["You are helping a signals analyst look at one radio signal on "
             "a waterfall display.", "", image_words, ""]
    if include_facts:
        parts += fact_lines(facts) + [""]
    else:
        parts += ["No measurements are given with this picture; answer from "
                  "the picture alone.", ""]
    parts += [f"THE ANALYST'S QUESTION: {q}", "",
              "Answer in a few sentences. Say what in the picture"
              + (" or the measurements" if include_facts else "")
              + " supports your answer. Your answer is a hypothesis for the "
              "analyst to check, not an identification. Do not invent "
              "measurements that are not listed.",
              f"If the picture{' and the measurements' if include_facts else ''}"
              f" do not let you answer, say \"{WAY_OUT}\"."]
    return "\n".join(parts)


def build_payload(spec_db_crop, axes: dict, detection_or_analysis=None,
                  measurements: dict | None = None, *,
                  question: str = DEFAULT_QUESTION,
                  lo_db: float | None = None, hi_db: float | None = None,
                  rows_oldest_first: bool = True, min_px: int = 256,
                  max_px: int = 1024) -> dict:
    """-> {png, prompt, facts, image_text, axes, levels, tier, method}.

    `spec_db_crop` is (rows = time, bins = frequency) in dB; `axes` gives
    `t0`, `t1` (s), `f_lo`, `f_hi` (absolute Hz) and `db_ref`
    ('above_floor' | 'dbfs'); `crop_around` makes both from a waterfall.
    The colour window defaults to `auto_levels` and is stated in the prompt.
    """
    s = np.asarray(spec_db_crop, dtype=np.float64)
    if s.ndim != 2 or s.shape[0] < 1 or s.shape[1] < 1:
        raise ValueError("the crop must be a 2D (time x frequency) array with "
                         "at least one row and one bin")
    if lo_db is None or hi_db is None:
        alo, ahi = auto_levels(s)
        lo_db = alo if lo_db is None else lo_db
        hi_db = ahi if hi_db is None else hi_db
    if not rows_oldest_first:
        s = s[::-1]
    scaled = fit_size(np.where(np.isfinite(s), s, float(lo_db)), min_px, max_px)
    rgb = ramp_rgb(scaled, float(lo_db), float(hi_db))
    words = image_text(axes or {}, rgb.shape, float(lo_db), float(hi_db), True)
    facts = facts_from(detection_or_analysis, measurements)
    png = encode_png(rgb, {"Description": "ATK point-and-ask waterfall crop. "
                           + words,
                           "Software": "ATK Diffusion Toolkit (point_ask)"})
    return {"png": png, "prompt": compose_prompt(words, facts, question, True),
            "facts": facts, "image_text": words, "axes": dict(axes or {}),
            "levels": {"lo_db": float(lo_db), "hi_db": float(hi_db)},
            "size": {"width": int(rgb.shape[1]), "height": int(rgb.shape[0])},
            "tier": TIER, "method": METHOD}


# ---------------------------------------------------------------------------
# Asking
# ---------------------------------------------------------------------------
_THINK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.S | re.I)
_DONT_KNOW = re.compile(r"^\W*(i\s+(do\s+not|don'?t)\s+know|nstr)\s*"
                        r"(?:$|[.,;:!?\-\u2014\u2013])", re.I)
_DONT_KNOW_ANY = re.compile(r"\b(i\s+(do\s+not|don'?t)\s+know|nstr)\b", re.I)


def _clean(text: str) -> str:
    t = _THINK.sub("", str(text or ""))
    return t.replace("’", "'").replace("‘", "'").strip()


def said_dont_know(text: str) -> bool:
    """True when the reply TAKES the way out: it opens with "I don't know"
    (or NSTR), or is a short reply that says it. A long answer that merely
    contains the words ("I don't know the operator, but this is DMR") is an
    answer, not the way out."""
    t = _clean(text)
    if not t:
        return False
    if _DONT_KNOW.search(t):
        return True
    return bool(_DONT_KNOW_ANY.search(t)) and len(t.split()) <= 12


def message_for(payload: dict, question: str = DEFAULT_QUESTION,
                include_facts: bool = True) -> list:
    """ATK's chat message for this crop: the OpenAI-style content list with
    the PNG as a data URI — how `atk/core/vision.py` passes images."""
    prompt = compose_prompt(payload["image_text"], payload["facts"], question,
                            include_facts)
    uri = "data:image/png;base64," + base64.b64encode(payload["png"]).decode("ascii")
    return [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": uri}}]}]


def ask(payload: dict, question: str,
        model: Callable[[str, bytes], str], *,
        include_facts: bool = True) -> dict:
    """Ask the host's model about the crop. -> {answer, said_dont_know,
    tier: 'proposed', tier_words, question, prompt, with_facts, truncated,
    error}. Never raises for a model failure: the reason comes back in
    `error`, in words."""
    q = (question or "").strip() or DEFAULT_QUESTION
    prompt = compose_prompt(payload["image_text"], payload["facts"], q,
                            include_facts)
    out = {"answer": "", "said_dont_know": False, "tier": TIER,
           "tier_words": HYPOTHESIS_WORDS, "question": q, "prompt": prompt,
           "with_facts": bool(include_facts), "truncated": False, "error": "",
           "method": METHOD}
    try:
        reply = model(prompt, payload["png"])
    except Exception as exc:                               # noqa: BLE001
        out["error"] = (f"the model could not be asked ({type(exc).__name__}: "
                        f"{exc}); nothing was answered")
        return out
    out["truncated"] = bool(getattr(reply, "truncated", False))
    text = _clean(reply)
    out["answer"] = text
    out["said_dont_know"] = said_dont_know(text)
    if not text:
        out["error"] = ("the model returned nothing"
                        + (" — it was cut off at the token limit"
                           if out["truncated"] else ""))
    return out


def _mentions(answer: str, names) -> bool:
    a = _clean(answer).lower()
    for n in names:
        n = str(n).strip().lower()
        if n and re.search(r"(?<![a-z0-9])" + re.escape(n) + r"(?![a-z0-9])", a):
            return True
    return False


def compare_with_without(payloads: list, truths: list, model,
                         question: str = DEFAULT_QUESTION,
                         aliases: dict | None = None) -> dict:
    """The first experiment's harness (plan §4.B5): the same crops asked with
    and without the bench's measurements. An answer is correct when it names
    the true class (or one of its aliases) and did not take the way out.
    Scoring is by string match, so it is blind to the condition by
    construction. -> {n, with: {...}, without: {...}, rows}."""
    if len(payloads) != len(truths):
        raise ValueError("one truth per payload")
    aliases = aliases or {}
    rows = []
    tally = {True: [0, 0, 0], False: [0, 0, 0]}   # correct, dont_know, errors
    for p, truth in zip(payloads, truths):
        row = {"truth": truth}
        for cond in (True, False):
            r = ask(p, question, model, include_facts=cond)
            names = [truth] + list(aliases.get(truth, ()))
            ok = (not r["error"] and not r["said_dont_know"]
                  and _mentions(r["answer"], names))
            tally[cond][0] += int(ok)
            tally[cond][1] += int(r["said_dont_know"])
            tally[cond][2] += int(bool(r["error"]))
            row["with" if cond else "without"] = {
                "answer": r["answer"], "correct": ok,
                "said_dont_know": r["said_dont_know"], "error": r["error"]}
        rows.append(row)
    n = len(payloads)

    def _s(c):
        cor, dk, err = tally[c]
        return {"correct": cor, "accuracy": cor / n if n else 0.0,
                "dont_know": dk, "errors": err}
    return {"n": n, "with": _s(True), "without": _s(False), "rows": rows,
            "tier": TIER, "note": "answers are hypotheses; accuracy is "
                                  "string-matched against the truth labels"}
