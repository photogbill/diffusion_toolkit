# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Audio gap filling for the Media Lab — plan §4.D5 (the forensics repair's
audio half).

    *"…audio inpainting in the Media Lab"*  — plan §4.D5

WHAT A GAP IS. A run of identical samples — digital silence or a stuck
value — that INTERRUPTS audio: short (at most `max_ms`) and with active
sound on both sides. That last condition is the important one: DSD writes
exact zeros between overs, and a recorder pauses with zeros too; a long
silence, or one beside quiet, is the recording, not damage, and is left
alone. Gaps an analyst marks by hand (`gaps_from_times`) are filled as
marked.

HOW IT IS FILLED (the AR core is shared with `repair.iq_dropout`):

  * `janssen` (method `janssen_fill`) — Janssen, Veldhuis & Vries (1986):
    alternately fit an autoregressive model to the block around the gap and
    solve, by least squares, for the missing samples that make the
    prediction error smallest. The AR order is about three times the gap
    (their recommendation), capped at 400, never more than a third of the
    record around it; the forward–backward fill below is its starting
    point. Best for gaps up to a few tens of milliseconds.
  * `lpc` (method `lpc_fill`) — LPC (Burg) extrapolation forward from the
    audio before the gap and backward from the audio after it, CROSSFADED
    across the gap with a raised cosine (Etter 1996; Kauppinen & Roth
    2002). Cheaper; used by `janssen` for gaps too long to solve.
  * `linear` (method `interpolate`) — a straight line, the baseline.

Every filled span is INFERRED (provenance), listed with its start, length,
method and the fill's level against the audio around it; the WAV is written
beside a JSON sidecar that carries the list, because a WAV file has nowhere
of its own to say which samples are a guess. The audio outside the gaps is
never touched.

LIMITS, said plainly: a fill is the most likely continuation of the sound
around it. Across a gap longer than about one pitch period it cannot know
what was said — a syllable lost in a gap stays lost, and the fill is a
smooth bridge, not a recovery. A transcript made across a filled gap is a
transcript of a guess (`provenance.decoded_from_note`). Never for making a
voice identifiable (plan §2.4).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from atk_diffusion import provenance as _prov
from atk_diffusion.repair import iq_dropout as _ar

_prov.METHOD_TIERS.setdefault("janssen_fill", "inferred")

METHODS = {"janssen": "janssen_fill", "lpc": "lpc_fill", "linear": "interpolate"}
MAX_ORDER = 400


@dataclass
class AudioGap:
    start: int
    count: int
    kind: str = "zeros"           # zeros | stuck | marked
    detail: str = ""

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class FilledSpan:
    start: int
    count: int
    seconds: float
    method: str
    tier: str
    fill_to_context_db: float | None
    note: str = ""

    def to_json(self) -> dict:
        return asdict(self)


def _rms_db(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return -math.inf
    p = float(np.mean(x * x))
    return 10.0 * math.log10(p) if p > 0 else -math.inf


def find_gaps(x, fs: float, min_ms: float = 1.0, max_ms: float = 250.0,
              active_db: float = -45.0, context_ms: float = 20.0) -> list[AudioGap]:
    """Runs of identical samples (≥ `min_ms`, ≤ `max_ms`) with active audio
    (RMS above `active_db` dBFS over `context_ms`) on BOTH sides."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 3:
        return []
    min_len = max(3, int(round(min_ms * 1e-3 * fs)))
    max_len = int(round(max_ms * 1e-3 * fs))
    ctx = max(8, int(round(context_ms * 1e-3 * fs)))
    eq = x[1:] == x[:-1]
    s, e = _ar._runs(eq)
    out = []
    for a, b in zip(s.tolist(), e.tolist()):
        length = b - a + 1
        if length < min_len or length > max_len:
            continue
        before = x[max(0, a - ctx):a]
        after = x[a + length:a + length + ctx]
        if _rms_db(before) < active_db or _rms_db(after) < active_db:
            continue
        kind = "zeros" if x[a] == 0 else "stuck"
        out.append(AudioGap(int(a), int(length), kind,
                            f"{length / fs * 1e3:.1f} ms of identical samples "
                            "inside active audio"))
    return out


def gaps_from_times(spans_s, fs: float) -> list[AudioGap]:
    """Analyst-marked gaps [(start_s, end_s), …] -> AudioGaps."""
    out = []
    for a, b in spans_s:
        s0, s1 = int(round(float(a) * fs)), int(round(float(b) * fs))
        if s1 > s0:
            out.append(AudioGap(s0, s1 - s0, "marked", "marked by the analyst"))
    return out


def _fill_one(y, mask, s: int, c: int, method: str, fs: float) -> tuple[np.ndarray, str]:
    n = y.size
    p_want = int(min(MAX_ORDER, max(16, 3 * c)))
    ctx = int(max(3 * p_want + 1, 4 * c, int(0.05 * fs)))
    left, right = _ar._context_fast(y, mask, s, s + c, 0, n, ctx)
    if method == "interpolate":
        return _ar.linear_fill(left, right, c, np.float64), method
    if method == "lpc_fill":
        return _ar.fb_fill(left, right, c, p_want), method
    p = _ar._order_for(min(left.size, right.size), p_want)
    if p < 1 or left.size < p or right.size < p or c > 4096:
        return _ar.fb_fill(left, right, c, p_want), "lpc_fill"
    init = _ar.fb_fill(left, right, c, p)
    blk = np.concatenate([left, init, right])
    bm = np.zeros(blk.size, dtype=bool)
    bm[left.size:left.size + c] = True
    return _ar.janssen_fill(blk, bm, p, init=init)[left.size:left.size + c], method


def fill(x, fs: float, gaps, method: str = "janssen") -> tuple[np.ndarray, list[FilledSpan]]:
    """Fill `gaps` (AudioGaps or (start, count) pairs) in mono audio.
    Returns (y, spans) — y float32, the audio outside the gaps unchanged."""
    if method not in METHODS:
        raise ValueError(f"unknown audio fill {method!r} — one of {', '.join(METHODS)}")
    meth = METHODS[method]
    y = np.asarray(x, dtype=np.float64).ravel().copy()
    pairs = [(g.start, g.count) if isinstance(g, AudioGap) else (int(g[0]), int(g[1]))
             for g in gaps]
    mask = np.zeros(y.size, dtype=bool)
    for s, c in pairs:
        mask[max(0, s):min(y.size, s + c)] = True
    spans = []
    for s, c in sorted(pairs):
        s, c = max(0, s), min(c, y.size - max(0, s))
        if c <= 0:
            continue
        seg, used = _fill_one(y, mask, s, c, meth, fs)
        y[s:s + c] = seg
        ftc = _ar._fill_to_context_db(y, s, c, ctx=int(0.02 * fs))
        note = ""
        if used != meth:
            note = f"{meth} could not run here (gap too long or too little audio around it); {used} was used"
        if ftc is not None and ftc > 3.0:
            note = (note + " " if note else "") + (
                f"The fill is {ftc:.1f} dB louder than the audio around it — "
                "listen before trusting it.")
        spans.append(FilledSpan(int(s), int(c), c / float(fs), used,
                                _prov.tier_for(used), ftc, note))
    return y.astype(np.float32), spans


def inpaint_wav(in_path, out_path, method: str = "janssen", gaps=None,
                **find_kw) -> dict:
    """WAV -> gap-filled WAV + `<out>.json` listing every filled span
    (tier INFERRED), the method and the source's hash. `gaps=None` finds
    them; a list of (start_s, end_s) marks them."""
    from atk_diffusion.repair.speech import read_wav, write_wav
    x, fs, info = read_wav(in_path)
    if gaps is None:
        found = find_gaps(x, fs, **find_kw)
    else:
        found = gaps_from_times(gaps, fs)
    y, spans = fill(x, fs, found, method)
    w = write_wav(out_path, y, fs)
    tier = "inferred" if spans else "record"
    side = {"source": str(in_path), "source_sha256": _prov.sha256_path(in_path),
            "output": str(out_path), "rate": fs, "source_wav": info,
            "method": METHODS[method], "tier": tier,
            "tier_words": _prov.TIER_WORDS[tier],
            "gaps": [g.to_json() for g in found],
            "filled": [s.to_json() for s in spans],
            "seconds_inferred": round(sum(s.seconds for s in spans), 4),
            "clipped_samples": w["clipped"],
            "provenance": _prov.stamp("repair.audio_inpaint", method=method)}
    sp = Path(str(out_path) + ".json")
    sp.write_text(json.dumps(side, indent=2, default=float), encoding="utf-8")
    side["sidecar"] = str(sp)
    side["lines"] = [f"{len(spans)} gap(s) filled by {method} — "
                     f"{side['seconds_inferred'] * 1e3:.0f} ms of audio is INFERRED "
                     "and listed in the sidecar."] if spans else \
        ["No gaps found; the output is the input."]
    return side
