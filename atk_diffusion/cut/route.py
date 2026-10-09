# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Route — the cut to every tool that accepts its class, and only those
(DETECTION_DESIGN §4.2 step 4; ARCHITECTURE §4.2 `cut`).

Bill, 2026-10-08: *"process the result via any tools that can process that
signal type, be it demodulator, decoder, DF, or even just saving the
original and the cleaned up one in the same folder for offline analysis."*

The toolkit owns the RULES; ATK owns the tools. A route is recorded in the
cut's analysis.json whether or not a tool ran; when the host passes a
`runner(iq, fs, meta) -> dict`, it is run on the chosen input and its result
is stored with the route, so a cut tells its whole story.

    the class's own tools   classes.tools_for(cls): DSD for P25/DMR/NXDN, the
                            pager decoder, multimon, ADS-B, LTE/NR search, and
                            the NFM / WFM / AM / SSB demodulators
    always                  bench (pulse, PRI, intra-pulse), df (atkdf — on
                            the cut itself when it holds all five Kraken
                            channels), ask (point-and-ask with the analysis
                            attached), teach (name the class), fingerprint,
                            save (the folder is the product)

A decode from anything but the record carries `provenance.decoded_from_note`
— *"decoded from a CLEANED signal, not from the record — confirm against the
original before relying on it"* — and matched-filter parameters, when the
cut has them, travel to the demodulator in `meta["atk:matched_parameters"]`.
"""

from __future__ import annotations

import json
import time

from atk_diffusion import provenance as _prov
from atk_diffusion.detect import classes as _classes

#: Offered for every cut, whatever its class.
ALWAYS = ("bench", "df", "ask", "teach", "fingerprint", "save")

ROUTE_WORDS = {
    "bench": "the Signals bench — pulse, PRI, intra-pulse analysis",
    "df": "direction finding (atkdf) — a bearing on the Kraken",
    "ask": "point-and-ask — the analysis goes to the primary model with the "
           "analyst's question",
    "teach": "teach — name the class; its examples become a prototype",
    "fingerprint": "fingerprint — add to the library or match against it",
    "save": "save only — the folder is the product",
}


def words(tool: str, multichannel: bool = False) -> str:
    if tool == "df" and multichannel:
        return ("direction finding on the cut itself — it holds every "
                "coherent Kraken channel")
    if tool in ROUTE_WORDS:
        return ROUTE_WORDS[tool]
    d = _classes.DECODERS.get(tool)
    if d:
        return d["label"] + (" — a decode CONFIRMS the class" if d["confirms"]
                             else " — audio only; it confirms nothing")
    return tool


def candidates_of(analysis: dict) -> list[str]:
    """Every class the cut might be: the classifier's, the source
    detection's, and the cyclic probe's candidates."""
    out: list[str] = []
    cl = (analysis.get("class") or {}).get("cls")
    if cl and cl != _classes.UNKNOWN:
        out.append(cl)
    src = (analysis.get("source") or {}).get("detection") or {}
    if src.get("cls"):
        out.append(src["cls"])
    for c in (src.get("measurements") or {}).get("candidates", []) or []:
        out.append(c)
    for c in ((analysis.get("cyclic") or {}).get("candidates") or []):
        out.append(c)
    seen: list[str] = []
    for c in out:
        if c not in seen and _classes.get(c) is not None:
            seen.append(c)
    return seen


def available(analysis: dict) -> list[str]:
    """The route menu: the tools of every candidate class, best first, then
    the ones every cut gets."""
    tools: list[str] = []
    for c in candidates_of(analysis):
        for t in _classes.tools_for(c):
            if t not in tools:
                tools.append(t)
    return tools + [t for t in ALWAYS if t not in tools]


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=_default))


def _default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 4096 else f"<array {o.shape}>"
    if isinstance(o, complex):
        return [o.real, o.imag]
    return str(o)


def perform(analysis: dict, tool: str, input_name: str, iq, fs: float,
            meta: dict, runner=None, who: str = "") -> dict:
    """Record (and, with a runner, run) one route. Returns the record that
    is appended to analysis["routes"]."""
    offered = available(analysis)
    if tool not in offered:
        raise ValueError(f"'{tool}' does not take this cut. The route menu "
                         "offers only the tools that accept its class: "
                         + ", ".join(offered) + ".")
    g = meta.get("global", {})
    tier = str(g.get("atk:tier", "record"))
    multichannel = getattr(iq, "ndim", 1) == 2 and iq.shape[0] > 1
    rec = {"tool": tool, "words": words(tool, multichannel),
           "input": input_name, "input_tier": tier,
           "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "by": who or _prov.who(), "ran": False, "result": None}
    note = _prov.decoded_from_note(tier) if tier in _prov.TIERS else ""
    if note and tool in _classes.DECODERS:
        rec["decoded_from_note"] = note
    elif tier != "record":
        rec["input_note"] = (f"this ran on a {tier.upper()} signal, not on the "
                             "record — the original is beside it in the folder")
    matched = [c for c in analysis.get("cleans", [])
               if c.get("method") == "matched"]
    meta_out = json.loads(json.dumps(meta, default=_default))
    if matched:
        meta_out.setdefault("global", {})["atk:matched_parameters"] = \
            matched[-1].get("parameters")
        rec["matched_parameters_passed"] = True
    if runner is not None:
        res = runner(iq, fs, meta_out)
        if not isinstance(res, dict):
            res = {"result": res}
        res = _jsonable(res)
        if note and tool in _classes.DECODERS:
            res["decoded_from_note"] = note
        rec["ran"] = True
        rec["result"] = res
    elif tool == "save":
        rec["result"] = {"saved": "the folder is the product — original, "
                                  "analysis and any cleaned files together"}
    return rec
