# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""IQ dropout detection and repair before DSD — the classical half of plan
§4.D1, with a hook for the learned inpainter.

    *"USB glitches break decoder sync and the voice is lost. Inpaint the gap
    in the IQ before the decoder sees it; the repaired span is marked in the
    processing mark."*  — plan §4.D1

WHAT IS FOUND. Three kinds of damage, each found by a rule that is derived,
not tuned:

  * **missing** — samples the receiver lost and the RECORDER KNEW it lost.
    ATK's `IqRecorder.note_gap` keeps the file contiguous and starts a new
    SigMF `captures` segment whose `core:datetime` is later by exactly the
    missing time; the global block carries `atk:gaps` (how many) and
    `atk:gap_samples` (how many samples in all). Those samples are NOT in the
    file. Repair INSERTS them, so every later sample is back on its own
    instant — which is what a TDMA decoder's symbol clock needs, whatever the
    fill holds. `recorder_gaps` reads the positions from the segments.
  * **stuck** / **zeros** — a run of identical samples (a zero-filled or
    repeated USB buffer). The run length that counts is derived from how
    often consecutive samples are equal in THIS capture, so that the expected
    number of false runs in the whole file is below `alpha` (0.01). Coarse
    8-bit quantisation of a weak signal repeats often, and the threshold
    rises with it.
  * **collapse** — power far below the receiver's own noise floor. A
    receiver cannot be quieter than its floor; a signal that switches off
    (a TDMA slot, the end of an over) falls TO the floor, never below it. The
    floor is the quietest 5 % of the capture's time–frequency cells (quiet
    slots and the bins beside a narrowband signal both count) or the
    profile's measured value; the threshold is the gamma (chi-square)
    quantile of a `win`-sample noise power at `pfa`, less a 10 dB margin.
    THE ONE WAY IT CAN BE FOOLED: a capture whose every cell, at every
    moment, is filled by one strong signal hides the floor, and a deep fade
    of that signal can then read as a collapse. Pass the profile's measured
    floor (`floor_power`) for such a capture; the voice-class cut DSD reads
    (a 12.5 kHz channel in 48 kHz) always has noise-only bins.

HOW IT IS FILLED. Every fill is `inferred` (provenance) and every span gets
its own SigMF annotation (`core:label` "repaired", `atk:method`, `atk:tier`),
so ATK's processing mark can show exactly which samples are a guess:

  * `linear` (method `interpolate`) — a straight line between the samples
    either side. The baseline every inpainter must beat (plan §7).
  * `ar` (method `ar_fill`) — Burg autoregressive models fitted on each side
    and extrapolated into the gap, forward and backward, crossfaded with a
    raised cosine (the weighted forward–backward predictor of Etter 1996 /
    Kauppinen & Roth 2002). The models are minimum-phase by construction, so
    an extrapolation decays toward zero instead of ringing up: in a gap of
    noise it fills with LESS than the noise, which is the honest failure.
  * `janssen` (method `janssen_fill`) — Janssen, Veldhuis & Vries (1986):
    alternate an AR fit on the block with a least-squares solve for the
    missing samples that minimises the prediction error. Best on gaps up to
    a few hundred samples; longer gaps fall back to `ar`, and the report says
    so.
  * `learned` (method `diffusion_inpaint`, tier INVENTED) — the diffusion
    inpainter from `atk_diffusion.learn.inpaint` (another part of the
    toolkit), imported only when asked for. Whatever it returns, only the
    masked samples are taken: a learned tool never touches the record
    outside the gap.
  * spans longer than `max_fill_s` are BLANKED (zeros, method `blank`,
    tier cleaned): an AR model cannot predict 100 ms of a wideband capture,
    and a confident-looking fill would be worse than an honest hole. A
    blanked missing gap still restores the timeline.

LIMITS, said plainly. Nothing here can recover information the receiver
never delivered; a fill is the most likely continuation of what surrounds
it, and a decode across it is a decode of a guess (`provenance.
decoded_from_note`). The recorder writes the FIRST segment's datetime to
whole seconds, so the first gap's length comes from `atk:gap_samples` less
the others; a lone gap is exact, several are each good to about one
microsecond of samples (their total is exact). A retune is never filled
across: the samples either side are at different frequencies.
`repair_capture` holds the whole capture in memory; a multi-gigabyte
recording is cut first (the signal cut takes the voice channel to 48 kS/s,
which is also where DSD wants it).

On a real capture there is no truth to score against, so each repaired span
reports `fill_to_context_db` — the fill's power against its surroundings. A
fill LOUDER than both sides is the tell of an invented signal and is said in
words. On synthetic data, `experiments.repair_eval` measures the
hallucination rate (energy put into gaps where the truth was only noise).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from atk_diffusion import provenance as _prov
from atk_diffusion import sigmf as _sigmf

# New method names this module writes (ARCHITECTURE §2 rule 5).
_prov.METHOD_TIERS.setdefault("janssen_fill", "inferred")
_prov.METHOD_TIERS.setdefault("blank", "cleaned")

#: Friendly method names -> the provenance method they write.
METHODS = {"linear": "interpolate", "interpolate": "interpolate",
           "ar": "ar_fill", "ar_fill": "ar_fill",
           "janssen": "janssen_fill", "janssen_fill": "janssen_fill",
           "learned": "diffusion_inpaint", "diffusion_inpaint": "diffusion_inpaint",
           "blank": "blank"}

#: Least to most removed from the record; a file's tier is its worst span's.
TIER_ORDER = ("record", "measured", "cleaned", "inferred", "invented")

DEFAULT_ORDER = 32            # AR order for IQ
DEFAULT_MAX_FILL_S = 0.05     # longer spans are blanked, not filled
JANSSEN_MAX = 4096            # longest single gap Janssen is asked to solve
MAX_COVERAGE = 0.5            # refuse when dropouts cover more of the file


def method_name(method: str) -> str:
    try:
        return METHODS[str(method).strip().lower()]
    except KeyError:
        raise ValueError(f"unknown repair method {method!r} — one of linear, "
                         "ar, janssen, learned") from None


def worst_tier(tiers) -> str:
    best = "record"
    for t in tiers:
        if TIER_ORDER.index(t) > TIER_ORDER.index(best):
            best = t
    return best


# ---------------------------------------------------------------------------
# The AR core (complex or real), shared with repair.audio_inpaint
# ---------------------------------------------------------------------------
def burg(x, order: int):
    """Burg's method. Returns (a, err) with a = [1, a1 … ap], the
    prediction-error filter e[n] = Σ a_i x[n−i], minimum-phase by
    construction (every reflection coefficient |k| < 1). Works on complex
    data (the conjugates are in the right places for IQ)."""
    x = np.asarray(x)
    cplx = np.iscomplexobj(x)
    dt = np.complex128 if cplx else np.float64
    f = x.astype(dt).copy()
    n = f.size
    p = int(order)
    if p < 1 or n <= p:
        raise ValueError(f"Burg needs more samples ({n}) than its order ({p})")
    b = f.copy()
    a = np.ones(1, dtype=dt)
    err = float(np.mean(np.abs(f) ** 2))
    for m in range(1, p + 1):
        ef = f[m:]
        eb = b[m - 1:n - 1]
        den = float(np.sum(np.abs(ef) ** 2) + np.sum(np.abs(eb) ** 2))
        if den <= 0.0:
            a = np.concatenate([a, np.zeros(p + 1 - a.size, dtype=dt)])
            break
        k = -2.0 * np.sum(ef * np.conj(eb)) / den
        f_new = ef + k * eb
        b_new = eb + np.conj(k) * ef
        f[m:] = f_new
        b[m:] = b_new
        ext = np.concatenate([a, np.zeros(1, dtype=dt)])
        a = ext + k * np.conj(ext[::-1])
        err *= max(0.0, 1.0 - float(np.abs(k)) ** 2)
    return a, err


def ar_extrapolate(a, history, n: int) -> np.ndarray:
    """Continue `history` (oldest first) by `n` samples with the AR model
    `a`: y[k] = −Σ a_i y[k−i], no excitation. scipy's lfilter runs it."""
    from scipy.signal import lfilter, lfiltic
    a = np.asarray(a)
    p = a.size - 1
    h = np.asarray(history)
    dt = np.result_type(a, h, np.float64)
    if n <= 0:
        return np.zeros(0, dtype=dt)
    if p == 0 or h.size == 0:
        return np.zeros(int(n), dtype=dt)
    past = np.zeros(p, dtype=dt)
    take = h[-p:][::-1].astype(dt)
    past[:take.size] = take
    zi = lfiltic(np.ones(1, dtype=dt), a.astype(dt), past)
    y, _ = lfilter(np.ones(1, dtype=dt), a.astype(dt), np.zeros(int(n), dtype=dt),
                   zi=zi)
    return y


def _order_for(ctx_len: int, order: int) -> int:
    return int(max(0, min(int(order), (int(ctx_len) - 1) // 3)))


def fb_fill(left, right, n: int, order: int = DEFAULT_ORDER) -> np.ndarray:
    """Weighted forward–backward AR prediction of an `n`-sample gap between
    `left` (record before it, oldest first) and `right` (record after it).
    Each side gets its own Burg model; the two extrapolations are
    crossfaded with a raised cosine. One usable side: that side alone.
    Neither: a straight line (or zeros)."""
    left = np.asarray(left)
    right = np.asarray(right)
    dt = np.result_type(left, right, np.float64)
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=dt)
    pl = _order_for(left.size, order)
    pr = _order_for(right.size, order)
    fwd = bwd = None
    if pl >= 1:
        fwd = ar_extrapolate(burg(left, pl)[0], left, n)
    if pr >= 1:
        rr = right[::-1]
        bwd = ar_extrapolate(burg(rr, pr)[0], rr, n)[::-1]
    if fwd is not None and bwd is not None:
        w = 0.5 * (1.0 + np.cos(np.pi * np.arange(1, n + 1) / (n + 1)))
        return (w * fwd + (1.0 - w) * bwd).astype(dt)
    if fwd is not None:
        return fwd.astype(dt)
    if bwd is not None:
        return bwd.astype(dt)
    return linear_fill(left[-1:] if left.size else None,
                       right[:1] if right.size else None, n, dt)


def linear_fill(before, after, n: int, dtype=np.complex128) -> np.ndarray:
    """A straight line from the last sample before to the first after. With
    only one side there is nothing to interpolate: zeros, said in the span's
    comment by the caller."""
    n = int(n)
    if before is None or len(before) == 0 or after is None or len(after) == 0:
        return np.zeros(n, dtype=dtype)
    a = complex(before[-1]) if np.iscomplexobj(before) else float(before[-1])
    b = complex(after[0]) if np.iscomplexobj(after) else float(after[0])
    f = np.arange(1, n + 1, dtype=np.float64) / (n + 1)
    return (a + (b - a) * f).astype(dtype)


def janssen_fill(block, mask, order: int, iterations: int = 10,
                 init=None, tol: float = 1e-6) -> np.ndarray:
    """Janssen, Veldhuis & Vries (1986), "Adaptive interpolation of
    discrete-time signals that can be modeled as autoregressive processes".

    `block` holds the record around the missing samples (`mask` True =
    missing). Each iteration fits AR(`order`) to the block as currently
    estimated (Burg), then solves for the missing samples that minimise the
    total prediction-error energy — a Hermitian Toeplitz system for one
    contiguous gap (Levinson), a banded one otherwise. The block must hold
    at least `order` known samples before the first missing sample and after
    the last (the caller chooses it so). Returns the whole block with the
    missing samples filled; the known samples are returned unchanged."""
    from scipy.linalg import solve, solve_toeplitz
    from scipy.signal import lfilter
    x = np.array(block, dtype=np.complex128 if np.iscomplexobj(block)
                 else np.float64)
    m = np.asarray(mask, dtype=bool)
    p = int(order)
    u = np.flatnonzero(m)
    if u.size == 0:
        return x
    n = x.size
    if p < 1 or u[0] < p or u[-1] > n - 1 - p:
        raise ValueError("janssen_fill needs `order` known samples on both "
                         "sides of the missing ones")
    if init is not None:
        x[m] = np.asarray(init)[: u.size] if np.size(init) == u.size else 0.0
    else:
        x[m] = 0.0
    contiguous = bool(u[-1] - u[0] + 1 == u.size)
    prev = x[m].copy()
    for _ in range(max(1, int(iterations))):
        a, _err = burg(x, p)
        r = np.correlate(a, a, mode="full")          # r[d + p] = Σ conj(a_i) a_{i+d}
        x0 = x.copy()
        x0[m] = 0.0
        e0 = lfilter(a, np.ones(1, dtype=a.dtype), x0)
        e0[:p] = 0.0
        z = -np.correlate(e0, a, mode="valid")[u]     # −A_u^H A_k x_k
        if contiguous:
            col = np.zeros(u.size, dtype=r.dtype)
            k = min(u.size, p + 1)
            col[:k] = r[p:p + k]
            col[0] = col[0].real + 1e-12 * max(1.0, abs(col[0]))
            xu = solve_toeplitz((col, np.conj(col)), z)
        else:
            d = np.subtract.outer(u, u)
            B = np.zeros((u.size, u.size), dtype=r.dtype)
            near = np.abs(d) <= p
            B[near] = r[d[near] + p]
            B[np.diag_indices_from(B)] += 1e-12 * max(1.0, abs(r[p]))
            xu = solve(B, z, assume_a="her")
        x[m] = xu
        change = np.linalg.norm(xu - prev) / max(np.linalg.norm(xu), 1e-30)
        prev = xu.copy()
        if change < tol:
            break
    return x


# ---------------------------------------------------------------------------
# Finding the damage
# ---------------------------------------------------------------------------
@dataclass
class Dropout:
    """One damaged or missing stretch. `start` is a sample index in the
    FILE. For in-file kinds (`zeros`, `stuck`, `collapse`) `count` samples
    from `start` are garbage; for `missing` they are absent and belong
    before sample `start`. A `retune` is reported, never filled."""
    start: int
    count: int
    kind: str
    in_file: bool = True
    detail: str = ""

    def to_json(self) -> dict:
        return asdict(self)


def _runs(flags: np.ndarray):
    """(starts, ends) of True runs in a bool array, ends exclusive."""
    f = np.asarray(flags, dtype=np.int8)
    if f.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    d = np.diff(np.concatenate(([0], f, [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def stuck_threshold(x, alpha: float = 0.01, floor_run: int = 4,
                    cap_run: int = 64) -> tuple[int, float]:
    """(run length, p_equal). The shortest run of identical samples such
    that a capture this long expects fewer than `alpha` such runs by chance,
    from how often consecutive samples are equal here (median over blocks,
    so the dropouts themselves do not raise it)."""
    x = np.asarray(x)
    if x.size < 3:
        return floor_run, 0.0
    eq = (x[1:] == x[:-1])
    blk = 4096
    nb = eq.size // blk
    if nb >= 3:
        p_eq = float(np.median(eq[: nb * blk].reshape(nb, blk).mean(axis=1)))
    else:
        p_eq = float(eq.mean())
    if p_eq <= 0.0:
        return floor_run, 0.0
    if p_eq >= 1.0:
        return cap_run, 1.0
    need = 1.0 + np.log(alpha / max(1.0, float(x.size))) / np.log(p_eq)
    return int(min(cap_run, max(floor_run, int(np.ceil(need))))), p_eq


def find_stuck_runs(x, min_run: int | None = None,
                    alpha: float = 0.01) -> list[Dropout]:
    """Runs of identical samples (zero-filled or repeated buffers)."""
    x = np.asarray(x)
    if x.size < 2:
        return []
    if min_run is None:
        L, p_eq = stuck_threshold(x, alpha)
    else:
        L, p_eq = int(min_run), float("nan")
    eq = (x[1:] == x[:-1])
    s, e = _runs(eq)
    out = []
    for a, b in zip(s.tolist(), e.tolist()):
        length = b - a + 1                 # samples a .. b inclusive
        if length < L:
            continue
        v = x[a]
        kind = "zeros" if v == 0 else "stuck"
        out.append(Dropout(int(a), int(length), kind, True,
                           f"{length} identical samples"
                           + ("" if kind == "zeros" else f" (stuck at {v:.4g})")
                           + f"; runs of {L}+ are not expected by chance here"
                           + (f" (consecutive samples equal {p_eq:.2%} of the "
                              "time)" if p_eq == p_eq else "")))
    return out


def noise_floor_power(x, nfft: int = 256, max_frames: int = 4096) -> float:
    """Per-sample noise power of the receiver, from the quietest 5 % of the
    capture's time–frequency cells. A noise-only cell's power is
    exponentially distributed about σ², so σ² = P5 / −ln(0.95). Quiet TDMA
    slots and the bins beside a narrowband signal both count; exactly-zero
    cells (a zero-filled dropout) do not. Up to `max_frames` frames spread
    over the capture are used, so a long capture costs no more than a short
    one."""
    x = np.asarray(x)
    n = x.size
    if n < 32:
        return float(np.mean(np.abs(x) ** 2)) if n else 0.0
    nfft = int(min(nfft, max(16, n // 16)))
    frames = n // nfft
    idx = np.unique(np.linspace(0, frames - 1, min(frames, max_frames)).astype(np.int64))
    w = np.hanning(nfft + 2)[1:-1]
    seg = x[(idx[:, None] * nfft + np.arange(nfft)[None, :])]
    X = np.fft.fft(seg * w[None, :], axis=1)
    P = (np.abs(X) ** 2).ravel() / float(np.sum(w ** 2))
    P = P[P > 0]
    if P.size == 0:
        return 0.0
    return float(np.percentile(P, 5) / -np.log(0.95))


def find_power_collapse(x, win: int = 64, margin_db: float = 10.0,
                        pfa: float = 1e-6,
                        floor_power: float | None = None) -> list[Dropout]:
    """Stretches whose power is far below the receiver's own noise floor.
    Windows of `win` samples, half-overlapping (boundaries good to win/2)."""
    from scipy.stats import gamma
    x = np.asarray(x)
    n = x.size
    win = int(max(8, win))
    if n < 4 * win:
        return []
    floor = float(floor_power) if floor_power else noise_floor_power(x)
    if floor <= 0:
        return []
    shape = win if np.iscomplexobj(x) else win / 2.0
    q = float(gamma.ppf(pfa, a=shape, scale=1.0 / shape))
    thr = floor * q * 10.0 ** (-float(margin_db) / 10.0)
    p = np.abs(x).astype(np.float64) ** 2
    c = np.concatenate(([0.0], np.cumsum(p)))
    hop = win // 2
    starts = np.arange(0, n - win + 1, hop)
    pw = (c[starts + win] - c[starts]) / win
    low = pw < thr
    if not low.any():
        return []
    flag = np.zeros(n, dtype=bool)
    for s0 in starts[low].tolist():
        flag[s0:s0 + win] = True
    # refine the edges to a few samples: an 8-sample average below its own
    # (looser) threshold, contiguous with the window-level detection
    short = 8
    q8 = float(gamma.ppf(1e-3, a=short if np.iscomplexobj(x) else short / 2.0,
                         scale=1.0 / (short if np.iscomplexobj(x) else short / 2.0)))
    thr8 = floor * q8 * 10.0 ** (-float(margin_db) / 10.0)
    # forward average over [i, i+8) finds a start exactly; backward over
    # (i-8, i] finds an end exactly (a centred one smears both by 4)
    cs = np.concatenate(([0.0], np.cumsum(p)))
    idx = np.arange(n)
    fwd = (cs[np.minimum(idx + short, n)] - cs[idx]) / np.minimum(short, n - idx)
    bwd = (cs[idx + 1] - cs[np.maximum(idx + 1 - short, 0)]) / np.minimum(short, idx + 1)
    f_ok = fwd < thr8
    b_ok = bwd < thr8
    s, e = _runs(flag)
    refined = np.zeros(n, dtype=bool)
    for a, b in zip(s.tolist(), e.tolist()):
        lo, hi = a, b
        while lo > max(0, a - win) and f_ok[lo - 1]:
            lo -= 1
        while hi < min(n, b + win) and b_ok[hi]:
            hi += 1
        first = np.flatnonzero(f_ok[lo:hi])
        last = np.flatnonzero(b_ok[lo:hi])
        if first.size and last.size and lo + last[-1] >= lo + first[0]:
            lo, hi = lo + int(first[0]), lo + int(last[-1]) + 1
        refined[lo:hi] = True
    s, e = _runs(refined)
    out = []
    for a, b in zip(s.tolist(), e.tolist()):
        seg_p = float(np.mean(p[a:b])) if b > a else 0.0
        depth = 10 * np.log10(max(seg_p, 1e-30) / floor)
        out.append(Dropout(int(a), int(b - a), "collapse", True,
                           f"power {depth:.0f} dB relative to the noise floor "
                           "— below what the receiver's own noise allows"))
    return out


def _merge(drops: list[Dropout]) -> list[Dropout]:
    """Union of overlapping in-file detections. The kind that covers most of
    the merged span names it (zeros before stuck before collapse on a tie)."""
    pri = {"zeros": 0, "stuck": 1, "collapse": 2}
    items = sorted((d for d in drops if d.in_file), key=lambda d: d.start)
    out: list[Dropout] = []
    cover: list[dict] = []
    for d in items:
        if out and d.start <= out[-1].start + out[-1].count:
            o = out[-1]
            o.count = max(o.start + o.count, d.start + d.count) - o.start
            c = cover[-1]
            c[d.kind] = c.get(d.kind, (0, d.detail))[0] + d.count, \
                c.get(d.kind, (0, d.detail))[1]
        else:
            out.append(Dropout(d.start, d.count, d.kind, True, d.detail))
            cover.append({d.kind: (d.count, d.detail)})
    for o, c in zip(out, cover):
        kind = max(c, key=lambda k: (c[k][0], -pri.get(k, 9)))
        o.kind, o.detail = kind, c[kind][1]
    return out


def detect_dropouts(x, kinds=("stuck", "collapse"), alpha: float = 0.01,
                    win: int = 64, margin_db: float = 10.0,
                    floor_power: float | None = None) -> list[Dropout]:
    """In-file damage in one channel (or the union over channels of a
    (channels, n) array — a USB dropout hits every channel of a coherent
    capture). Overlapping detections are merged."""
    x = np.asarray(x)
    chans = x if x.ndim == 2 else x[None, :]
    found: list[Dropout] = []
    for ch in chans:
        if "stuck" in kinds:
            found += find_stuck_runs(ch, alpha=alpha)
        if "collapse" in kinds:
            found += find_power_collapse(ch, win=win, margin_db=margin_db,
                                         floor_power=floor_power)
    return _merge(found)


# ---------------------------------------------------------------------------
# What ATK's recorder says about gaps (atk/core/iq_recorder.py)
# ---------------------------------------------------------------------------
def _epoch_us(s) -> int | None:
    """An RFC 3339 / SigMF datetime -> integer microseconds since the epoch
    (exact: a float of 1.8e9 s cannot hold a microsecond reliably)."""
    if not s:
        return None
    t = str(s).strip()
    if t.endswith("Z") or t.endswith("z"):
        t = t[:-1]
    elif len(t) > 6 and t[-6] in "+-" and t[-3] == ":":
        if t[-6:] not in ("+00:00", "-00:00"):
            try:
                dt = datetime.fromisoformat(t)
                return int(round(dt.timestamp() * 1e6))
            except ValueError:
                return None
        t = t[:-6]
    frac_us = 0
    if "." in t:
        t, frac = t.split(".", 1)
        digits = "".join(ch for ch in frac if ch.isdigit())[:9]
        if digits:
            frac_us = int(round(int(digits) * 10 ** (6 - len(digits))))
    try:
        base = datetime.strptime(t, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return int(base.timestamp()) * 1_000_000 + frac_us


def recorder_gaps(meta: dict) -> list[Dropout]:
    """The discontinuities ATK's recorder wrote into a capture's sidecar, in
    file sample positions: `missing` (samples lost; insert them) and
    `retune` (frequency changed; never filled across).

    Lengths come from consecutive segments' datetimes against their sample
    positions; the first gap (whose predecessor is the whole-second first
    segment) is `atk:gap_samples + atk:settle_samples` less the others.
    """
    caps = sorted(meta.get("captures", []) or [],
                  key=lambda c: int(c.get("core:sample_start", 0)))
    if len(caps) < 2:
        return []
    g = meta.get("global", {}) or {}
    rate = float(g.get("core:sample_rate", 0.0) or 0.0)
    if rate <= 0:
        return []
    total = None
    if "atk:gap_samples" in g or "atk:settle_samples" in g:
        total = int(g.get("atk:gap_samples", 0) or 0) + \
            int(g.get("atk:settle_samples", 0) or 0)
    times = [_epoch_us(c.get("core:datetime")) for c in caps]
    precise = ["." in str(c.get("core:datetime", "")) for c in caps]
    lengths: list[int | None] = [None] * len(caps)
    tol = rate * 2e-6 + 1.0
    for k in range(1, len(caps)):
        if times[k] is None or times[k - 1] is None:
            continue
        if not (precise[k] and precise[k - 1]):
            continue                      # whole-second times cannot size a gap
        dpos = int(caps[k]["core:sample_start"]) - int(caps[k - 1]["core:sample_start"])
        miss = (times[k] - times[k - 1]) * rate / 1e6 - dpos
        lengths[k] = 0 if abs(miss) <= tol else int(round(miss))
    first_note = ""
    if len(caps) >= 2 and lengths[1] is None:
        if total is not None:
            known = sum(v for v in lengths[2:] if v)
            lengths[1] = max(0, total - known)
            first_note = (" (length from atk:gap_samples less the later gaps: "
                          "the first segment's time is to whole seconds)")
        elif times[1] is not None and times[0] is not None:
            first_note = (" (the length is uncertain by up to a second: the "
                          "first segment's time is to whole seconds and the "
                          "file has no atk:gap_samples)")
            dpos = int(caps[1]["core:sample_start"])
            est = (times[1] - times[0]) * rate / 1e6 - dpos
            lengths[1] = None if est < 0 else int(round(est))
    out: list[Dropout] = []
    for k in range(1, len(caps)):
        pos = int(caps[k]["core:sample_start"])
        f0 = caps[k - 1].get("core:frequency")
        f1 = caps[k].get("core:frequency")
        retune = (f0 is not None and f1 is not None
                  and abs(float(f1) - float(f0)) > 0.5)
        miss = lengths[k]
        if retune:
            out.append(Dropout(pos, int(miss or 0), "retune", False,
                               f"retune from {float(f0) / 1e6:.6f} to "
                               f"{float(f1) / 1e6:.6f} MHz"
                               + (f" with {miss} samples not recorded" if miss
                                  else "")
                               + " — the samples either side are at different "
                               "frequencies; nothing is filled across it"))
        elif miss is None:
            out.append(Dropout(pos, 0, "missing", False,
                               "a time discontinuity whose length cannot be "
                               "worked out from this file — left as it is"
                               + first_note))
        elif miss > 0:
            out.append(Dropout(pos, int(miss), "missing", False,
                               f"{miss} samples ({miss / rate * 1e3:.3f} ms) "
                               "lost by the receiver and never written"
                               + (first_note if k == 1 else "")))
    return out


# ---------------------------------------------------------------------------
# Filling
# ---------------------------------------------------------------------------
@dataclass
class RepairedSpan:
    """One filled span in the OUTPUT's sample index."""
    start: int
    count: int
    kind: str                 # zeros | stuck | collapse | missing
    method: str               # provenance method
    tier: str
    inserted: bool            # the samples were absent from the file
    source_start: int         # where it was in the input file
    fill_to_context_db: float | None = None
    note: str = ""
    detail: str = ""

    def to_json(self) -> dict:
        return asdict(self)

    def annotation(self) -> _sigmf.Annotation:
        words = (f"REPAIRED — {self.count} samples "
                 + ("inserted where the receiver lost them" if self.inserted
                    else f"replaced ({self.kind})")
                 + f"; {self.method}, {self.tier.upper()}: "
                 + _prov.TIER_WORDS[self.tier])
        if self.note:
            words += " " + self.note
        extra = {"atk:method": self.method, "atk:tier": self.tier,
                 "atk:repair_kind": self.kind, "atk:inserted": bool(self.inserted),
                 "atk:source_sample_start": int(self.source_start)}
        if self.fill_to_context_db is not None:
            extra["atk:fill_to_context_db"] = round(float(self.fill_to_context_db), 2)
        return _sigmf.Annotation(int(self.start), int(self.count),
                                 label="repaired", comment=words, extra=extra)


def _context_fast(x, mask, a: int, b: int, lo: int, hi: int, length: int):
    """Record samples immediately before [a, b) and after it, up to `length`
    each side, stopping at masked samples and at the segment bounds
    [lo, hi) — a fill is built only from the record, never from another
    fill."""
    s0 = max(lo, a - length)
    left_mask = mask[s0:a]
    if left_mask.any():
        s0 = s0 + int(np.flatnonzero(left_mask)[-1]) + 1
    e0 = min(hi, b + length)
    right_mask = mask[b:e0]
    if right_mask.any():
        e0 = b + int(np.flatnonzero(right_mask)[0])
    return x[s0:a], x[b:e0]


def _learned_fill(x, mask, fs, inpainter=None, **kw):
    """The learned hook. `inpainter(x, mask, fs) -> y` may be supplied;
    otherwise `atk_diffusion.learn.inpaint.inpaint(x, mask, fs, **kw)` is
    imported here and only here. Only masked samples are taken from it."""
    if inpainter is None:
        try:
            from atk_diffusion.learn import inpaint as _inp   # lazy, heavy
        except ImportError as e:
            raise RuntimeError(
                "The learned IQ inpainter (atk_diffusion.learn.inpaint) is not "
                "available in this environment, so the 'learned' repair cannot "
                f"run ({e}). The classical repairs (linear, ar, janssen) still "
                "work.") from None
        fn = getattr(_inp, "inpaint", None)
        if not callable(fn):
            raise RuntimeError(
                "atk_diffusion.learn.inpaint has no inpaint(x, mask, fs, …) "
                "function, which is what the repair track calls.")
        y = fn(x, mask, fs, **kw)
    else:
        y = inpainter(x, mask, fs)
    if isinstance(y, tuple):
        y = y[0]
    if isinstance(y, dict):
        y = y.get("y", y.get("x"))
    y = np.asarray(y)
    if y.shape != np.asarray(x).shape:
        raise RuntimeError(f"the learned inpainter returned {y.shape} samples "
                           f"for {np.asarray(x).shape}; refusing to use it")
    out = np.array(x, copy=True)
    out[mask] = y[mask]
    return out


def fill_spans(x, spans, method: str = "ar", order: int = DEFAULT_ORDER,
               fs: float | None = None, max_fill: int | None = None,
               bounds=None, context: int | None = None,
               inpainter=None, learned_kw: dict | None = None):
    """Fill `spans` [(start, count), …] of a 1-D array (complex or real).

    Returns (y, used) where `used[i]` is the provenance method actually used
    for span i ('blank' when it was longer than `max_fill`, or the fallback
    method when the requested one could not run — the caller words it).
    `bounds` are indices where context must stop (retunes)."""
    x = np.asarray(x)
    y = np.array(x, dtype=np.result_type(x, np.complex64 if np.iscomplexobj(x)
                                          else np.float32), copy=True)
    n = y.size
    mask = np.zeros(n, dtype=bool)
    for s, c in spans:
        mask[int(s):int(s) + int(c)] = True
    cuts = sorted(set([0, n] + [int(b) for b in (bounds or []) if 0 < int(b) < n]))
    meth = method_name(method)
    used: list[str] = []
    learned_spans = []
    for s, c in spans:
        s, c = int(s), int(c)
        if c <= 0:
            used.append(meth)
            continue
        k = np.searchsorted(cuts, s, side="right")
        lo, hi = cuts[k - 1], cuts[k] if k < len(cuts) else n
        if max_fill is not None and c > int(max_fill) and meth != "blank":
            y[s:s + c] = 0
            used.append("blank")
            continue
        if meth == "blank":
            y[s:s + c] = 0
            used.append("blank")
            continue
        if meth == "diffusion_inpaint":
            learned_spans.append((s, c))
            used.append(meth)
            continue
        ctx = int(context or max(16 * order, 2 * c))
        ctx = min(ctx, 1 << 16)
        left, right = _context_fast(y, mask, s, s + c, lo, hi, ctx)
        if meth == "interpolate":
            y[s:s + c] = linear_fill(left, right, c, y.dtype)
            used.append(meth)
        elif meth == "ar_fill":
            y[s:s + c] = fb_fill(left, right, c, order)
            used.append(meth)
        elif meth == "janssen_fill":
            p = _order_for(min(left.size, right.size), order)
            if c <= JANSSEN_MAX and p >= 1 and left.size >= p and right.size >= p:
                init = fb_fill(left, right, c, order)
                blk = np.concatenate([left, init, right])
                bm = np.zeros(blk.size, dtype=bool)
                bm[left.size:left.size + c] = True
                filled = janssen_fill(blk, bm, p, init=init)
                y[s:s + c] = filled[left.size:left.size + c]
                used.append(meth)
            else:
                y[s:s + c] = fb_fill(left, right, c, order)
                used.append("ar_fill")
        else:                                              # pragma: no cover
            raise ValueError(meth)
    if learned_spans:
        lm = np.zeros(n, dtype=bool)
        for s, c in learned_spans:
            lm[s:s + c] = True
        y = _learned_fill(y, lm, fs, inpainter=inpainter, **(learned_kw or {}))
    return y, used


def _fill_to_context_db(y, s: int, c: int, ctx: int = 1024) -> float | None:
    a = y[max(0, s - ctx):s]
    b = y[s + c:s + c + ctx]
    ref = np.concatenate([a, b])
    if ref.size == 0 or c <= 0:
        return None
    pf = float(np.mean(np.abs(y[s:s + c]) ** 2))
    pr = float(np.mean(np.abs(ref) ** 2))
    if pr <= 0:
        return None
    return 10.0 * np.log10(max(pf, 1e-30) / pr)


def repair_array(x, dropouts: list[Dropout], method: str = "ar",
                 fs: float | None = None, order: int = DEFAULT_ORDER,
                 max_fill_s: float | None = DEFAULT_MAX_FILL_S,
                 inpainter=None, learned_kw: dict | None = None):
    """Repair an in-memory capture (1-D, or (channels, n) for a coherent
    multi-channel capture). `missing` dropouts are INSERTED (their positions
    are in the input's index); in-file ones are replaced. Returns
    (y, spans: list[RepairedSpan], index_map) where index_map(i) gives an
    input sample's index in y."""
    x = np.asarray(x)
    multi = x.ndim == 2
    chans = x if multi else x[None, :]
    n = chans.shape[1]
    meth = method_name(method)
    ins = sorted([d for d in dropouts if d.kind == "missing" and not d.in_file
                  and d.count > 0 and 0 < d.start <= n], key=lambda d: d.start)
    retunes = [d.start for d in dropouts if d.kind == "retune"]
    infile = [d for d in dropouts if d.in_file and d.count > 0]
    ins_pos = np.array([d.start for d in ins], dtype=np.int64)
    ins_cum = np.cumsum([d.count for d in ins]).astype(np.int64) if ins else np.zeros(0, np.int64)

    def index_map(i):
        i = np.asarray(i, dtype=np.int64)
        k = np.searchsorted(ins_pos, i, side="right")
        add = np.where(k > 0, ins_cum[np.maximum(k - 1, 0)] if ins_cum.size else 0, 0)
        return i + add

    total = n + int(ins_cum[-1] if ins_cum.size else 0)
    out = np.zeros((chans.shape[0], total), dtype=np.complex128
                   if np.iscomplexobj(x) else np.float64)
    src = index_map(np.arange(n))
    out[:, src] = chans
    plan = []      # (out_start, count, kind, inserted, src_start, detail)
    for d in ins:
        o = int(index_map(d.start)) - d.count
        plan.append((o, d.count, "missing", True, d.start, d.detail))
    for d in infile:
        a = max(0, d.start)
        b = min(n, d.start + d.count)
        if b <= a:
            continue
        o = int(index_map(a))
        plan.append((o, b - a, d.kind, False, a, d.detail))
    plan.sort(key=lambda t: t[0])
    merged = []
    for p in plan:
        if merged and p[0] <= merged[-1][0] + merged[-1][1]:
            q = merged[-1]
            end = max(q[0] + q[1], p[0] + p[1])
            merged[-1] = (q[0], end - q[0], q[2], q[3] or p[3], q[4],
                          q[5] + "; " + p[5])
        else:
            merged.append(p)
    max_fill = None
    if max_fill_s is not None and fs:
        max_fill = int(round(float(max_fill_s) * float(fs)))
    bounds = [int(index_map(r)) for r in retunes]
    span_pairs = [(m[0], m[1]) for m in merged]
    used_all = None
    for ci in range(out.shape[0]):
        yc, used = fill_spans(out[ci], span_pairs, meth, order=order, fs=fs,
                              max_fill=max_fill, bounds=bounds,
                              inpainter=inpainter, learned_kw=learned_kw)
        out[ci] = yc
        used_all = used if used_all is None else used_all
    spans = []
    for (o, c, kind, inserted, s0, detail), um in zip(merged, used_all or []):
        note = ""
        if um == "blank":
            note = (f"longer than {max_fill} samples — blanked (zeros), not "
                    "filled: nothing can predict that far"
                    + ("; the timeline is restored" if inserted else ""))
        elif um != meth:
            note = (f"{meth} could not run on this span (too long or too "
                    f"little record around it); {um} was used")
        ftc = _fill_to_context_db(out[0], o, c)
        if ftc is not None and ftc > 3.0 and um not in ("blank",):
            note = (note + " " if note else "") + (
                f"The fill is {ftc:.1f} dB LOUDER than the record around it — "
                "treat it with suspicion: a fill should not add energy.")
        spans.append(RepairedSpan(o, c, kind, um, _prov.tier_for(um), inserted,
                                  int(s0), ftc, note, detail))
    y = out if multi else out[0]
    return y.astype(np.complex64 if np.iscomplexobj(x) else np.float32), spans, index_map


# ---------------------------------------------------------------------------
# A capture on disk -> a repaired SigMF pair
# ---------------------------------------------------------------------------
def scan_capture(path, detect=("recorder", "stuck", "collapse"),
                 alpha: float = 0.01, margin_db: float = 10.0,
                 floor_power: float | None = None) -> dict:
    """Find the damage without changing anything: {dropouts, samples,
    sample_rate, coverage, lines}."""
    meta = _sigmf.read_meta(path)
    fs = _sigmf.sample_rate_of(meta)
    x = _sigmf.load(path, meta=meta)
    drops: list[Dropout] = []
    if "recorder" in detect:
        drops += recorder_gaps(meta)
    kinds = tuple(k for k in detect if k in ("stuck", "collapse"))
    if kinds:
        drops += detect_dropouts(x, kinds=kinds, alpha=alpha,
                                 margin_db=margin_db, floor_power=floor_power)
    n = x.shape[-1]
    bad = sum(d.count for d in drops if d.in_file)
    lines = [f"{len([d for d in drops if d.kind == 'missing'])} recorder gap(s), "
             f"{len([d for d in drops if d.in_file])} damaged stretch(es) in "
             f"the file ({bad} samples, {bad / max(1, n):.3%} of it), "
             f"{len([d for d in drops if d.kind == 'retune'])} retune(s)."]
    return {"dropouts": drops, "samples": n, "sample_rate": fs,
            "coverage": bad / max(1, n), "lines": lines, "meta": meta}


def repair_capture(path, out_base, method: str = "ar",
                   detect=("recorder", "stuck", "collapse"),
                   order: int = DEFAULT_ORDER,
                   max_fill_s: float | None = DEFAULT_MAX_FILL_S,
                   alpha: float = 0.01, margin_db: float = 10.0,
                   floor_power: float | None = None, rf=None,
                   inpainter=None, learned_kw: dict | None = None,
                   progress=None) -> dict:
    """Repair a SigMF capture into `<out_base>.sigmf-data/-meta` (cf32).

    The output's global block carries `atk:tier` (the worst span's tier),
    `atk:method`, `atk:method_params`, `atk:source_capture` and `atk:repair`
    (what was done, in numbers); the recorder's `atk:gaps` /
    `atk:gap_samples` count only the gaps still left. One annotation per
    repaired span (`core:label` "repaired", `atk:method`, `atk:tier`,
    `atk:inserted`) is added beside the capture's own annotations, which are
    carried across at their new sample positions. Retunes stay `captures`
    segments. Nothing is written except the pair (and, with `rf`, the write
    log). Returns a report dict with `lines` in plain words."""
    say = progress or (lambda _m: None)
    src = _sigmf.base_of(path)
    scan = scan_capture(src, detect, alpha, margin_db, floor_power)
    meta = scan["meta"]
    fs = scan["sample_rate"]
    drops: list[Dropout] = scan["dropouts"]
    if scan["coverage"] > MAX_COVERAGE:
        raise ValueError(
            f"{scan['coverage']:.0%} of {Path(src).name} looks like dropouts — "
            "there is not enough record left to repair from. (If the signal is "
            "very weak and coarsely quantised, consecutive samples repeat "
            "often and stuck-sample detection cannot work: raise the gain.)")
    say(scan["lines"][0])
    meth = method_name(method)
    x = _sigmf.load(src, meta=meta)
    y, spans, index_map = repair_array(x, drops, meth, fs=fs, order=order,
                                       max_fill_s=max_fill_s,
                                       inpainter=inpainter, learned_kw=learned_kw)
    n_out = y.shape[-1]
    g_in = dict(meta.get("global", {}) or {})
    caps_in = sorted(meta.get("captures", []) or [],
                     key=lambda c: int(c.get("core:sample_start", 0)))
    inserted_at = {d.start for d in drops if d.kind == "missing" and d.count > 0}
    # captures: the first segment, every retune, and any gap left unfilled
    caps_out = []
    for i, c in enumerate(caps_in):
        pos = int(c.get("core:sample_start", 0))
        keep = (i == 0)
        if not keep:
            f0 = caps_in[i - 1].get("core:frequency")
            f1 = c.get("core:frequency")
            retune = (f0 is not None and f1 is not None
                      and abs(float(f1) - float(f0)) > 0.5)
            keep = retune or pos not in inserted_at
        if keep:
            cc = dict(c)
            cc["core:sample_start"] = int(index_map(pos)) if pos > 0 else 0
            caps_out.append(cc)
    # the capture's own annotations, moved to the new index
    anns = []
    for a in _sigmf.annotations(meta):
        s_new = int(index_map(a.sample_start))
        e_new = int(index_map(max(a.sample_start, a.sample_start + a.sample_count - 1))) + 1
        a.sample_start = s_new
        a.sample_count = max(0, e_new - s_new) if a.sample_count > 0 else 0
        anns.append(a)
    anns += [s.annotation() for s in spans]
    tiers = [s.tier for s in spans]
    tier = worst_tier(tiers) if tiers else "cleaned"
    n_in = x.shape[-1]
    left_gaps = [d for d in drops if d.kind == "missing"
                 and not (d.count > 0 and 0 < d.start <= n_in)]
    try:
        from atk_diffusion import profiles as _profiles
        profile = _profiles.profile_from_meta(meta)
    except ValueError:
        profile = ""
    filled = sum(s.count for s in spans if s.method != "blank")
    blanked = sum(s.count for s in spans if s.method == "blank")
    inserted = sum(s.count for s in spans if s.inserted)
    summary = {"spans": len(spans), "samples_filled": int(filled),
               "samples_blanked": int(blanked), "samples_inserted": int(inserted),
               "fraction_inferred": round(filled / max(1, n_out), 6),
               "by_kind": {k: sum(1 for s in spans if s.kind == k)
                           for k in sorted({s.kind for s in spans})},
               "loud_fills": sum(1 for s in spans if s.fill_to_context_db is not None
                                 and s.fill_to_context_db > 3.0
                                 and s.method != "blank")}
    line = (f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} repaired "
            f"{Path(src).name}: {len(spans)} span(s), {filled} samples filled by "
            f"{meth}, {blanked} blanked, {inserted} inserted to restore the "
            "timeline")
    drop_keys = {"core:datatype", "atk:datatype", "atk:gaps", "atk:gap_samples",
                 "core:sample_rate", "core:version", "core:recorder",
                 "core:hw", "core:description", "core:num_channels"}
    extra = {k: v for k, v in g_in.items() if k not in drop_keys}
    extra.update({
        "atk:tier": tier, "atk:method": meth,
        "atk:method_params": {"order": int(order), "max_fill_s": max_fill_s,
                              "detect": list(detect), "alpha": alpha,
                              "margin_db": margin_db},
        "atk:source_capture": Path(src).name,
        "atk:repair": summary, "atk:repair_log": line,
    })
    if profile:
        extra["atk:receiver_profile"] = profile
    if "atk:gaps" in g_in or "atk:gap_samples" in g_in:
        extra["atk:source_gaps"] = {"atk:gaps": g_in.get("atk:gaps"),
                                    "atk:gap_samples": g_in.get("atk:gap_samples")}
    if left_gaps:
        extra["atk:gaps"] = len(left_gaps)
        extra["atk:gap_samples"] = int(sum(d.count for d in left_gaps))
    chans = y.shape[0] if y.ndim == 2 else 1
    dp, mp = _sigmf.write_pair(
        out_base, y, fs, _sigmf.center_of(meta), datatype="cf32",
        annotations=anns, extra_global=extra,
        hw=str(g_in.get("core:hw", "") or ""),
        description=f"repaired from {Path(src).name} — {tier.upper()} where "
                    "marked; the original is the record",
        channels=chans)
    m2 = _sigmf.read_meta(mp)
    m2["captures"] = caps_out or m2["captures"]
    m2["annotations"] = sorted(m2.get("annotations", []),
                               key=lambda a: int(a.get("core:sample_start", 0)))
    _sigmf.write_meta(mp, m2)
    if rf is not None:
        try:
            rf.record(dp, "repaired", line)
            rf.record(mp, "repaired-meta", line)
        except Exception:                                  # noqa: BLE001
            pass
    lines = list(scan["lines"])
    lines.append(f"{filled} samples filled ({meth}, {tier.upper()}), {blanked} "
                 f"blanked, {inserted} inserted so every later sample is back "
                 "on its own time.")
    if summary["loud_fills"]:
        lines.append(f"{summary['loud_fills']} fill(s) came out louder than "
                     "the record around them — look at those before trusting "
                     "a decode across them.")
    if left_gaps:
        lines.append(f"{len(left_gaps)} gap(s) could not be sized and were left "
                     "as captures segments.")
    lines.append(_prov.decoded_from_note(tier))
    say(lines[-2])
    return {"data": str(dp), "meta": str(mp), "tier": tier, "method": meth,
            "spans": [s.to_json() for s in spans],
            "dropouts": [d.to_json() for d in drops], "summary": summary,
            "log": line, "lines": [ln for ln in lines if ln]}
