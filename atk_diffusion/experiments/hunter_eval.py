# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""B6's first experiment: the hunter versus a fixed scan (plan §4.B6).

*"A bladeRF at low power on a cable loop into the hunter's receiver plays a
scripted sequence of bursts at random times and frequencies; time-to-find and
fraction found, hunter versus a fixed scan."*

`run()` does exactly that against `hunt.sim` — the same goal, the same
scripted band and the same receiver model for both policies, several seeds —
and reports, per policy: the fraction of the goal's bursts found, the median
and 90th-percentile time from a burst's start to the hunter marking it, false
marks per hour, and retunes. The classical comparator (the fixed scan) is
beside the learned-or-ruled one, as plan §7 requires.

On Bill's machine the same loop runs with ATK's receiver adapter in place of
the SimReceiver (`hunt.policy.run_hunt`) and the script played by the
bladeRF or HackRF through the cabled loop (`cabled.txfiles` makes the burst
file; its manifest is the ground truth, scored by `score_manifest`). Until
then the numbers are a comparison of POLICIES under a stated detection
curve, and the report says so.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np

from atk_diffusion.hunt.goal import parse_goal
from atk_diffusion.hunt.policy import (FixedScanPolicy, HuntLog,
                                       ReceiverLimits, RulePolicy, run_hunt)
from atk_diffusion.hunt.sim import SimReceiver, scripted_band

DEFAULT_GOAL = "anything narrowband and bursty between 400 and 470"


def _pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else None


def score(st, truth, duration_s: float) -> dict:
    """Score one hunt's finds against the script's truth."""
    truth_ids = {s.id: s for s in truth}
    first = {}
    false_marks = 0
    for f in st.found:
        sid = (f.measurements or {}).get("sim_signal", -1)
        if sid in truth_ids:
            first[sid] = min(first.get(sid, math.inf), f.found_at_s)
        else:
            false_marks += 1
    ttf = [first[i] - truth_ids[i].t0 for i in first]
    return {"truth": len(truth_ids), "found": len(first),
            "fraction_found": (len(first) / len(truth_ids)) if truth_ids else None,
            "time_to_find_s": ttf, "false_marks": false_marks,
            "false_marks_per_hour": false_marks * 3600.0 / max(duration_s, 1e-9)}


def score_manifest(found, manifest: dict, tx_start_s: float,
                   tx_center_hz: float, *, duration_s: float | None = None,
                   tol_s: float = 0.5) -> dict:
    """The REAL run's score: the hunter's finds (HuntState.found) against a
    cabled-loop transmit manifest (`cabled.txfiles`), whose bursts went out
    at `tx_start_s` on the hunt's clock, centred `tx_center_hz`. A burst is
    found by the first find that overlaps it in time (± tol_s) and contains
    its centre frequency; finds that match no burst are false marks."""
    bursts = []
    for sig in manifest.get("signals", []):
        on = sig.get("bursts_s") or [[sig["start_s"], sig["duration_s"]]]
        bw = float(sig.get("bandwidth_hz") or 0.0)
        f = float(tx_center_hz) + float(sig.get("f_offset_hz", 0.0))
        for b0, bd in on:
            bursts.append((tx_start_s + float(b0), tx_start_s + float(b0) + float(bd),
                           f, bw))
    first = {}
    used = set()
    for k, fd in enumerate(sorted(found, key=lambda x: x.found_at_s)):
        lo = fd.center_hz - 0.5 * max(fd.bw_hz, 1.0)
        hi = fd.center_hz + 0.5 * max(fd.bw_hz, 1.0)
        for i, (a, b, f, bw) in enumerate(bursts):
            if fd.t0 <= b + tol_s and fd.t1 >= a - tol_s and \
                    lo - 0.5 * bw <= f <= hi + 0.5 * bw:
                used.add(k)
                if i not in first:
                    first[i] = fd.found_at_s - a
                break
    ttf = sorted(first.values())
    false = len(found) - len(used)
    out = {"truth": len(bursts), "found": len(first),
           "fraction_found": len(first) / len(bursts) if bursts else None,
           "median_time_to_find_s": _pct(ttf, 50), "false_marks": false}
    if duration_s:
        out["false_marks_per_hour"] = false * 3600.0 / duration_s
    return out


def run(rf=None, profile: str = "rtlsdr_2400000_cu8",
        goal_text: str = DEFAULT_GOAL, *, duration_s: float = 900.0,
        seeds=(1, 2, 3), scan_dwell_s: float = 1.0, script: dict | None = None,
        receiver: dict | None = None, out_dir=None,
        progress: Callable[[str], None] | None = None) -> dict:
    """Hunter (RulePolicy) versus fixed scan on scripted bands.
    -> {goal, profile, policies: {name: summary}, per_seed, report_md, files}."""
    goal = parse_goal(goal_text)
    probs = goal.problems()
    if probs:
        raise ValueError("The experiment's goal cannot be hunted: " + "; ".join(probs))
    limits = ReceiverLimits.for_profile(profile)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = None
    if out_dir is not None:
        out = Path(out_dir)
    elif rf is not None:
        out = Path(rf.runs(profile)) / f"hunter_eval_{stamp}"
    logs_dir = out if out is not None else None
    per_seed = []
    pooled = {"hunter": {"ttf": [], "found": 0, "truth": 0, "false": 0,
                         "retunes": 0},
              "fixed_scan": {"ttf": [], "found": 0, "truth": 0, "false": 0,
                             "retunes": 0}}
    for seed in seeds:
        band = scripted_band(goal.f_lo_hz, goal.f_hi_hz, duration_s,
                             rng=np.random.default_rng(int(seed)),
                             **(script or {}))
        truth = band.goal_truth(goal)
        row = {"seed": int(seed), "truth": len(truth)}
        for name, policy in (("hunter", RulePolicy()),
                             ("fixed_scan", FixedScanPolicy(scan_dwell_s))):
            rx = SimReceiver(band, limits,
                             rng=np.random.default_rng(int(seed) + 1000),
                             profile=profile, **(receiver or {}))
            if logs_dir is not None:
                log = HuntLog(logs_dir / f"{name}_seed{seed}.jsonl", fsync=False)
            else:
                import tempfile
                log = HuntLog(Path(tempfile.mkdtemp(prefix="hunt_")) / "log.jsonl",
                              fsync=False)
            res, st = run_hunt(goal, rx, limits, policy, log,
                               duration_s=duration_s)
            sc = score(st, truth, duration_s)
            sc["retunes"] = res.retunes
            sc["refusals"] = res.refusals
            sc["log"] = str(log.path)
            row[name] = {k: v for k, v in sc.items() if k != "time_to_find_s"}
            row[name]["median_time_to_find_s"] = _pct(sc["time_to_find_s"], 50)
            p = pooled[name]
            p["ttf"] += sc["time_to_find_s"]
            p["found"] += sc["found"]
            p["truth"] += sc["truth"]
            p["false"] += sc["false_marks"]
            p["retunes"] += res.retunes
            if progress:
                progress(f"seed {seed} {name}: {sc['found']}/{sc['truth']} found")
        per_seed.append(row)
    hours = duration_s * len(seeds) / 3600.0
    summary = {}
    for name, p in pooled.items():
        summary[name] = {
            "fraction_found": (p["found"] / p["truth"]) if p["truth"] else None,
            "found": p["found"], "truth": p["truth"],
            "median_time_to_find_s": _pct(p["ttf"], 50),
            "p90_time_to_find_s": _pct(p["ttf"], 90),
            "false_marks_per_hour": p["false"] / hours if hours else None,
            "retunes_per_hour": p["retunes"] / hours if hours else None}
    result = {"experiment": "hunter_eval (plan §4.B6)", "goal": goal.describe(),
              "goal_notes": list(goal.notes), "profile": profile,
              "duration_s": duration_s, "seeds": list(seeds),
              "policies": summary, "per_seed": per_seed, "tier": "measured",
              "what_this_is": ("simulated band (hunt.sim) under a stated "
                               "detection curve: a comparison of policies, "
                               "not a prediction of the field"),
              "created": stamp}
    result["report_md"] = report_md(result)
    files = []
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        jp = out / "result.json"
        jp.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        mp = out / "report.md"
        mp.write_text(result["report_md"], encoding="utf-8")
        files = [str(jp), str(mp)]
        if rf is not None:
            for f in (jp, mp):
                try:
                    rf.record(f, "experiment", "hunter_eval")
                except Exception:                          # noqa: BLE001
                    pass
    result["files"] = files
    return result


def _f(v, fmt="{:.1f}"):
    return "—" if v is None else fmt.format(v)


def report_md(r: dict) -> str:
    lines = ["# Hunter versus fixed scan (plan §4.B6)", "",
             f"Goal: **{r['goal']}** · profile `{r['profile']}` · "
             f"{len(r['seeds'])} seeds × {r['duration_s']:g} s", "",
             f"*{r['what_this_is']}.*", "",
             "| policy | fraction found | median time-to-find | 90th pct | "
             "false marks / h | retunes / h |",
             "|---|---|---|---|---|---|"]
    for name in ("hunter", "fixed_scan"):
        s = r["policies"][name]
        lines.append(f"| {name.replace('_', ' ')} | "
                     f"{_f(s['fraction_found'], '{:.2f}')} "
                     f"({s['found']}/{s['truth']}) | "
                     f"{_f(s['median_time_to_find_s'])} s | "
                     f"{_f(s['p90_time_to_find_s'])} s | "
                     f"{_f(s['false_marks_per_hour'])} | "
                     f"{_f(s['retunes_per_hour'], '{:.0f}')} |")
    if r.get("goal_notes"):
        lines += ["", "How the goal was read:"] + [f"- {n}" for n in r["goal_notes"]]
    return "\n".join(lines) + "\n"
