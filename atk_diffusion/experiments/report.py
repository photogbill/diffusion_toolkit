# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""One way to write an experiment's result: markdown for Bill, JSON for the
next program (ARCHITECTURE §4.4: "each track's first experiment as a
function that returns a result dict and writes a report (markdown + JSON)
under the profile's runs\\"; plan §7: "published with the code, failures
included").

    md, js = write_report(out_dir, "domain_gap", result, lines)

* `lines` are the sentences a person reads first — what was measured, on
  what, and what it means — written by the experiment, in plain words.
* `result` is the whole result; the markdown repeats its scalar numbers in
  a table, and the JSON holds all of it. NaN and infinity become null (JSON
  has no NaN; a reader that chokes on one is not a reader Bill should
  need), numpy types become plain numbers.
* Every report carries its tier: an experiment's numbers are MEASURED —
  computed from data by a stated method — and say so (provenance §2.1), with
  the provenance stamp (what ran, where, when, by whom).

Standard library + numpy only; no PyTorch. Reused by every engineer's
experiments.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from atk_diffusion import provenance as _prov

TIER = "measured"


def experiment_dir(rf, profile: str, name: str) -> Path:
    """`rf.runs(profile)/<UTC stamp>_<name>/`, created (never reused)."""
    from atk_diffusion.learn.common import safe_name, utc_stamp
    base = Path(rf.runs(profile))
    base.mkdir(parents=True, exist_ok=True)
    stem = f"{utc_stamp()}_{safe_name(name)}"
    d = base / stem
    k = 2
    while d.exists():
        d = base / f"{stem}_{k}"
        k += 1
    d.mkdir(parents=True)
    return d


def clean(obj):
    """A JSON-safe copy: numpy -> Python, NaN/inf -> None, Paths -> str,
    tuples -> lists, unknown objects -> their str()."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _scalars(d, prefix: str = "", depth: int = 0, out=None) -> list:
    out = [] if out is None else out
    if depth > 3 or not isinstance(d, dict):
        return out
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, bool) or v is None:
            out.append((key, "—" if v is None else ("yes" if v else "no")))
        elif isinstance(v, (int, float)):
            out.append((key, f"{v:.4g}" if isinstance(v, float) else str(v)))
        elif isinstance(v, str) and len(v) <= 80:
            out.append((key, v))
        elif isinstance(v, dict):
            _scalars(v, key + ".", depth + 1, out)
    return out


def write_report(out_dir, name: str, result: dict, lines: list[str], *,
                 title: str | None = None, rf=None) -> tuple[Path, Path]:
    """Write `<out_dir>/<name>.md` and `<out_dir>/<name>.json`; return both
    paths. With an RfData, both files go into its write log."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    stamp = _prov.stamp(f"experiments.{name}")
    payload = {"name": name, "tier": TIER,
               "tier_words": _prov.TIER_WORDS[TIER], "provenance": stamp,
               "summary": [str(x) for x in lines], "result": clean(result)}
    js = d / f"{name}.json"
    tmp = js.with_name(js.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False),
                   encoding="utf-8")
    tmp.replace(js)
    md_lines = [f"# {title or name.replace('_', ' ')}", "",
                f"*{_prov.TIER_WORDS[TIER]}* Written "
                f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} by "
                f"{stamp['tool']} {stamp['version']} on {stamp['host']}.", ""]
    md_lines += [f"- {x}" for x in lines] or ["- (no summary)"]
    rows = _scalars(payload["result"])
    if rows:
        md_lines += ["", "## Numbers", "", "| quantity | value |", "|---|---|"]
        md_lines += [f"| {k} | {v} |" for k, v in rows]
    md_lines += ["", f"The full result is in `{js.name}`.", ""]
    md = d / f"{name}.md"
    md.write_text("\n".join(md_lines), encoding="utf-8")
    if rf is not None:
        for f in (md, js):
            try:
                rf.record(f, "report", name)
            except Exception:                              # noqa: BLE001
                pass
    return md, js
