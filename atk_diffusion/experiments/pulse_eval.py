# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""D3's experiment: deinterleaving and pulse-train completion on synthetic
interleaved emitters with fading dropouts (plan §4.D3, §6 Phase 3 exit:
*"PRI analysis correct with inferred pulses flagged"*, §7 hallucination
rate).

THE SCENE. Emitters of every kind the bench meets — constant, jittered,
two- and three-level stagger — interleaved, two of them on the SAME carrier
and width (separable only by PRI), one switching on and off like a scanning
beam. Each emitter fades through a Gilbert–Elliott channel (a good state
that drops 2 % of pulses and a bad one that drops 70 %, bad spells about
three pulses long — fading drops pulses in runs, not one at a time),
TOAs carry measurement noise, and random noise pulses (random carrier,
width and time) are mixed in.

THE NUMBERS, per scene and pooled over seeds:

  * **deinterleaving accuracy** — received pulses assigned to the right
    emitter (found emitters matched to true ones by the Hungarian method on
    their shared pulses), and how many emitters were found vs. present;
  * **PRI error with and without completion** — the naive reading a dropout
    wrecks (the MEAN of first differences of the received pulses, which a
    dropped pulse doubles) against the same reading of the completed train,
    and the classification (constant / jittered / staggered) of a
    first-difference classifier shaped like ATK's own `pulse.classify_pri`
    on received vs. completed pulses;
  * **false-pulse rate** — the hallucination rate: inferred pulses with no
    transmitted pulse of that emitter within max(3σ, 2 % of the PRI), over
    all inferred pulses (and per second), with inferred pulses during an
    emitter's OFF time counted separately — they are the worst kind;
  * **fill recall** — dropped pulses that were inferred, of those that
    could be (inside a burst, in a gap of at most `max_missing`).

On Bill's machine the same `score()` runs on synthetic scenes at the
bench's own sample rates; on a real recording there is no truth, so the
bench shows the completed train with its inferred pulses flagged.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from atk_diffusion import provenance as _prov
from atk_diffusion.repair import pulses as _pu

#: The default scene (seconds, hertz). Two emitters share carrier and width.
DEFAULT_EMITTERS = (
    {"name": "constant 1 ms", "pri": 1.0e-3, "freq": 0.0, "width": 5e-6},
    {"name": "jittered 1.37 ms ±6%", "pri": 1.37e-3, "jitter": 0.06,
     "freq": 300e3, "width": 2e-6},
    {"name": "stagger 0.6/0.82 ms", "levels": (0.6e-3, 0.82e-3),
     "freq": -400e3, "width": 10e-6},
    {"name": "stagger 0.9/1.3/1.1 ms", "levels": (0.9e-3, 1.3e-3, 1.1e-3),
     "freq": 150e3, "width": 3e-6},
    {"name": "same carrier as #1, 2.03 ms", "pri": 2.03e-3, "freq": 0.0,
     "width": 5e-6},
    {"name": "scanning 1.6 ms", "pri": 1.6e-3, "freq": -150e3, "width": 8e-6,
     "on": ((0.0, 0.08), (0.2, 0.28))},
)


def gilbert_elliott(n: int, rng, mean_drop: float = 0.15, bad_len: float = 3.0,
                    p_good: float = 0.02, p_bad: float = 0.7) -> np.ndarray:
    """Drop mask (True = dropped) from a two-state fading channel whose
    long-run drop rate is `mean_drop` and whose bad spells last `bad_len`
    pulses on average."""
    q2 = 1.0 / max(bad_len, 1.0)                       # bad -> good
    frac_bad = min(max((mean_drop - p_good) / (p_bad - p_good), 0.0), 0.95)
    q1 = q2 * frac_bad / max(1.0 - frac_bad, 1e-9)    # good -> bad
    bad = rng.random() < frac_bad
    out = np.empty(n, dtype=bool)
    for i in range(n):
        out[i] = rng.random() < (p_bad if bad else p_good)
        bad = (rng.random() >= q2) if bad else (rng.random() < q1)
    return out


def synth_scene(rng, emitters=DEFAULT_EMITTERS, duration_s: float = 0.3,
                mean_drop: float = 0.15, noise_rate_hz: float = 150.0,
                toa_sigma_s: float = 0.1e-6) -> tuple[list, list, list]:
    """(pdws, transmitted, specs): received PDWs in ATK's shape plus a
    `true` emitter id (−1 noise); every transmitted pulse with its fate."""
    pdws, sent = [], []
    specs = []
    for eid, spec in enumerate(emitters):
        seq = list(spec.get("levels") or [spec["pri"]])
        frame = float(sum(seq))
        jitter = float(spec.get("jitter", 0.0))
        on = spec.get("on")
        t = float(rng.uniform(0, seq[0]))
        times = []
        k = 0
        while t < duration_s:
            times.append(t)
            t += seq[k % len(seq)]
            k += 1
        times = np.array(times)
        mean_pri = frame / len(seq)
        if jitter:
            times = times + rng.uniform(-jitter, jitter, times.size) * mean_pri
        drop = gilbert_elliott(times.size, rng, mean_drop)
        specs.append({"id": eid, **{k2: v for k2, v in spec.items()
                                    if k2 not in ("on",)},
                      "on": [list(w) for w in on] if on else None,
                      "frame": frame, "mean_pri": mean_pri})
        for tt, dr in zip(times, drop):
            active = (not on) or any(a <= tt < b for a, b in on)
            sent.append({"toa_s": float(tt), "emitter": eid, "active": active,
                         "received": bool(active and not dr)})
            if active and not dr:
                pdws.append({"toa_s": float(tt + rng.normal(0, toa_sigma_s)),
                             "width_s": spec["width"] * (1 + 0.03 * rng.normal()),
                             "freq_hz": spec["freq"] + 2000.0 * rng.normal(),
                             "amplitude_db": 20.0, "amplitude": 10.0,
                             "chirp_hz": 0.0, "start": -1, "end": -1,
                             "true": eid})
    n_noise = rng.poisson(noise_rate_hz * duration_s)
    for _ in range(n_noise):
        pdws.append({"toa_s": float(rng.uniform(0, duration_s)),
                     "width_s": float(rng.uniform(0.5e-6, 20e-6)),
                     "freq_hz": float(rng.uniform(-1e6, 1e6)),
                     "amplitude_db": 12.0, "amplitude": 4.0, "chirp_hz": 0.0,
                     "start": -1, "end": -1, "true": -1})
    pdws.sort(key=lambda p: p["toa_s"])
    return pdws, sent, specs


def naive_kind(toas, jitter_tol: float = 0.05) -> dict:
    """A first-difference classifier shaped like ATK's `pulse.classify_pri`
    (median and MAD of first differences; stagger as separated clusters of
    intervals with 15 % support each) — the reading dropouts break."""
    t = np.sort(np.asarray(toas, dtype=np.float64))
    if t.size < 4:
        return {"kind": "too few", "pri_s": math.nan, "levels_s": []}
    d = np.diff(t)
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med))) * 1.4826
    s = np.sort(d)
    gaps = np.diff(s)
    boundary = max(jitter_tol * 2 * med, float(np.median(gaps)) * 6.0)
    splits = np.flatnonzero(gaps > boundary) + 1
    clusters = np.split(s, splits) if splits.size else [s]
    need = max(2, int(0.15 * d.size))
    levels = [float(np.mean(c)) for c in clusters if c.size >= need]
    if 1 < len(levels) <= 4:
        return {"kind": "staggered", "pri_s": float(np.mean(d)), "levels_s": levels}
    spread = mad / med if med > 0 else 1.0
    if spread <= jitter_tol:
        return {"kind": "constant", "pri_s": med, "levels_s": []}
    if spread < 0.5:
        return {"kind": "jittered", "pri_s": float(np.mean(d)), "levels_s": []}
    return {"kind": "irregular", "pri_s": float(np.mean(d)), "levels_s": []}


def _true_kind(spec) -> str:
    if spec.get("levels"):
        return "staggered"
    return "jittered" if spec.get("jitter", 0.0) > 0.02 else "constant"


def score(D: "_pu.Deinterleaved", sent: list, specs: list) -> dict:
    """Score a deinterleaving of a synthetic scene against its truth."""
    from scipy.optimize import linear_sum_assignment
    P = D.pdws
    n_true = len(specs)
    n_found = len(D.emitters)
    C = np.zeros((max(n_found, 1), max(n_true, 1)), dtype=np.int64)
    for e in D.emitters:
        for i in e.received:
            tr = P[i].get("true", -1)
            if tr >= 0:
                C[e.id, tr] += 1
    rows, cols = linear_sum_assignment(-C) if n_found and n_true else ([], [])
    match = {int(r): int(c) for r, c in zip(rows, cols) if C[r, c] > 0}
    rec_true = sum(1 for p in P if p.get("true", -1) >= 0)
    correct = sum(int(C[r, c]) for r, c in match.items())
    noise_assigned = sum(1 for e in D.emitters for i in e.received
                         if P[i].get("true", -1) < 0)
    per = []
    false_inf = off_inf = total_inf = 0
    recall_num = recall_den = 0
    dur = max((s["toa_s"] for s in sent), default=1.0)
    for e in D.emitters:
        total_inf += len(e.inferred)
        tr = match.get(e.id)
        if tr is None:
            false_inf += len(e.inferred)
            continue
        spec = specs[tr]
        tx = np.array(sorted(s["toa_s"] for s in sent if s["emitter"] == tr))
        act = np.array([s["active"] for s in sorted(
            (s for s in sent if s["emitter"] == tr), key=lambda s: s["toa_s"])])
        for p in e.inferred:
            tol = max(3.0 * p["sigma_toa_s"], 0.02 * spec["mean_pri"])
            j = int(np.argmin(np.abs(tx - p["toa_s"]))) if tx.size else -1
            if j < 0 or abs(tx[j] - p["toa_s"]) > tol:
                false_inf += 1
            elif not act[j]:
                false_inf += 1
                off_inf += 1
        received = [P[i] for i in e.received if P[i].get("true", -1) == tr]
        completed = sorted(received + list(e.inferred), key=lambda p: p["toa_s"])
        rec_t = _pu.toas(received)
        comp_t = _pu.toas(completed)
        nk_rec = naive_kind(rec_t)
        nk_comp = naive_kind(comp_t)
        truth_mean = spec["mean_pri"]
        # an honest naive error excludes the gaps BETWEEN bursts for both
        def mean_within(tt):
            if tt.size < 2:
                return math.nan
            d = np.diff(tt)
            return float(np.mean(d[d < 6 * truth_mean]))
        err_rec = abs(mean_within(rec_t) - truth_mean) / truth_mean
        err_comp = abs(mean_within(comp_t) - truth_mean) / truth_mean
        tk = _true_kind(spec)
        found_kind = e.kind.split(" ")[0]
        lv_err = math.nan
        if spec.get("levels") and e.levels_s and len(e.levels_s) == len(spec["levels"]):
            tl = np.array(spec["levels"])
            el = np.array(e.levels_s)
            lv_err = min(float(np.max(np.abs(np.roll(el, k) - tl)))
                         for k in range(len(tl))) / spec["frame"]
        # recall: dropped pulses inside a burst, in gaps the fill may cover
        dropped = [s for s in sent if s["emitter"] == tr and s["active"]
                   and not s["received"]]
        inf_t = np.array([p["toa_s"] for p in e.inferred])
        for s in dropped:
            recall_den += 1
            if inf_t.size and np.min(np.abs(inf_t - s["toa_s"])) <= max(
                    0.02 * spec["mean_pri"], 3e-7 + 0.07 * spec["mean_pri"] * spec.get("jitter", 0)):
                recall_num += 1
        per.append({"true": spec["name"], "found": e.words(),
                    "kind_true": tk, "kind_found": found_kind,
                    "kind_naive_received": nk_rec["kind"],
                    "kind_naive_completed": nk_comp["kind"],
                    "pri_err_naive_received": err_rec,
                    "pri_err_naive_completed": err_comp,
                    "frame_or_pri_err": abs(e.pri_s - (spec["frame"] if tk == "staggered"
                                                       else spec["mean_pri"]))
                    / (spec["frame"] if tk == "staggered" else spec["mean_pri"]),
                    "levels_err_of_frame": lv_err,
                    "received": len(received), "inferred": len(e.inferred)})
    return {"emitters_true": n_true, "emitters_found": n_found,
            "matched": len(match), "accuracy": correct / max(rec_true, 1),
            "noise_pulses_assigned": noise_assigned,
            "inferred": total_inf, "false_inferred": false_inf,
            "false_inferred_off_time": off_inf,
            "false_pulse_rate": false_inf / max(total_inf, 1),
            "false_pulses_per_s": false_inf / max(dur, 1e-9),
            "fill_recall": recall_num / max(recall_den, 1),
            "per_emitter": per}


def pulse_eval(seeds=range(5), emitters=DEFAULT_EMITTERS, duration_s: float = 0.3,
               mean_drop: float = 0.15, noise_rate_hz: float = 150.0,
               out_dir=None, progress=None, **deint_kw) -> dict:
    """Run `seeds` scenes, score each, pool the numbers, write a report
    (markdown + JSON) under `out_dir` when given."""
    runs = []
    t0 = time.time()
    for sd in seeds:
        rng = np.random.default_rng(int(sd))
        pdws, sent, specs = synth_scene(rng, emitters, duration_s, mean_drop,
                                        noise_rate_hz)
        D = _pu.deinterleave(pdws, **deint_kw)
        sc = score(D, sent, specs)
        sc["seed"] = int(sd)
        runs.append(sc)
        if progress:
            progress(f"seed {sd}: accuracy {sc['accuracy']:.1%}, "
                     f"false-pulse rate {sc['false_pulse_rate']:.1%}")
    pe = [p for r in runs for p in r["per_emitter"]]

    def mean(key, rows=pe):
        v = [r[key] for r in rows if r[key] == r[key]]
        return float(np.mean(v)) if v else math.nan

    pooled = {
        "scenes": len(runs), "seconds": round(time.time() - t0, 2),
        "accuracy": float(np.mean([r["accuracy"] for r in runs])),
        "emitters_found_vs_true": [(r["emitters_found"], r["emitters_true"]) for r in runs],
        "noise_pulses_assigned": int(sum(r["noise_pulses_assigned"] for r in runs)),
        "inferred": int(sum(r["inferred"] for r in runs)),
        "false_inferred": int(sum(r["false_inferred"] for r in runs)),
        "false_inferred_off_time": int(sum(r["false_inferred_off_time"] for r in runs)),
        "false_pulse_rate": (sum(r["false_inferred"] for r in runs)
                             / max(1, sum(r["inferred"] for r in runs))),
        "fill_recall": float(np.mean([r["fill_recall"] for r in runs])),
        "pri_err_naive_received": mean("pri_err_naive_received"),
        "pri_err_naive_completed": mean("pri_err_naive_completed"),
        "frame_or_pri_err": mean("frame_or_pri_err"),
        "kind_correct_found": float(np.mean([p["kind_found"] == p["kind_true"] for p in pe])) if pe else math.nan,
        "kind_correct_naive_received": float(np.mean([p["kind_naive_received"] == p["kind_true"] for p in pe])) if pe else math.nan,
        "kind_correct_naive_completed": float(np.mean([p["kind_naive_completed"] == p["kind_true"] for p in pe])) if pe else math.nan,
    }
    lines = [
        f"{pooled['scenes']} scene(s): {pooled['accuracy']:.1%} of received "
        "pulses went to the right emitter.",
        f"Mean-of-first-differences PRI error: {pooled['pri_err_naive_received']:.1%} "
        f"on the received pulses, {pooled['pri_err_naive_completed']:.2%} on the "
        "completed trains.",
        f"Classification right: {pooled['kind_correct_naive_received']:.0%} "
        "(first differences, received) → "
        f"{pooled['kind_correct_naive_completed']:.0%} (first differences, "
        f"completed); {pooled['kind_correct_found']:.0%} from the deinterleaver.",
        f"Hallucination: {pooled['false_inferred']} of {pooled['inferred']} "
        f"inferred pulses had no transmitted pulse there "
        f"({pooled['false_pulse_rate']:.1%}); "
        f"{pooled['false_inferred_off_time']} fell in an emitter's off time.",
        f"Fill recall: {pooled['fill_recall']:.1%} of dropped pulses were "
        "inferred where they were due.",
    ]
    result = {"experiment": "D3 pulse_eval", "pooled": pooled, "runs": runs,
              "lines": lines,
              "scene": {"emitters": [dict(e) for e in emitters],
                        "duration_s": duration_s, "mean_drop": mean_drop,
                        "noise_rate_hz": noise_rate_hz, "deinterleave": deint_kw},
              "provenance": _prov.stamp("experiments.pulse_eval")}
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        jp = out / "pulse_eval.json"
        jp.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        md = ["# D3 — deinterleaving and pulse-train completion", "",
              f"{pooled['scenes']} synthetic scene(s), {duration_s:g} s each, "
              f"Gilbert–Elliott fading at {mean_drop:.0%} mean drop, "
              f"{noise_rate_hz:g} noise pulses/s.", "",
              "| number | value |", "|---|---|"]
        for k, v in pooled.items():
            if k == "emitters_found_vs_true":
                v = ", ".join(f"{a}/{b}" for a, b in v)
            elif isinstance(v, float):
                v = f"{v:.4g}"
            md.append(f"| {k} | {v} |")
        md += ["", "## In words", ""] + [f"- {ln}" for ln in lines]
        mp = out / "pulse_eval.md"
        mp.write_text("\n".join(md) + "\n", encoding="utf-8")
        result["report_json"], result["report_md"] = str(jp), str(mp)
    return result
