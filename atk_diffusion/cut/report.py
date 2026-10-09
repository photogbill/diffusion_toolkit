# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""report.md — the cut in words, written ONLY from the facts in
analysis.json (DETECTION_DESIGN §4.2 step 5; ARCHITECTURE §5).

Nothing in the report is computed here: every number is read from
analysis.json, with the tier and method it was stored with, so the report
can never say more than the folder holds. Re-running `CutFolder.report()`
after a clean or a route rewrites it from the updated facts.
"""

from __future__ import annotations

from atk_diffusion import provenance as _prov


def _f(v, fmt: str = "{:,.1f}", none: str = "—") -> str:
    if v is None:
        return none
    try:
        return fmt.format(v)
    except (TypeError, ValueError):
        return str(v)


def _plain(v) -> str | None:
    """A parameter as a short readable value, or None when it is not one
    (long lists, nested records — they are in analysis.json)."""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        return _f(v, "{:,.6g}")
    if isinstance(v, str):
        return v if len(v) <= 80 else None
    if (isinstance(v, list) and 0 < len(v) <= 8
            and all(isinstance(e, (int, float)) and not isinstance(e, bool)
                    for e in v)):
        return "[" + ", ".join(_f(e, "{:,.6g}") for e in v) + "]"
    return None


def render(a: dict) -> str:
    """analysis.json → markdown."""
    c = a.get("cut", {})
    box = c.get("box", {})
    L: list[str] = []
    L.append(f"# Signal cut {a.get('folder', '')}")
    L.append("")
    L.append(f"*{_prov.TIER_WORDS['record']}* `original` is the box as cut; "
             "everything else in this folder is labelled with its tier.")
    L.append("")
    L.append("## The cut")
    L.append("")
    L.append(f"- Receiver profile: `{a.get('profile', '')}` — "
             f"{c.get('profile_words', '')}")
    L.append(f"- Centre: {_f(c.get('center_hz'), '{:,.0f}')} Hz; box "
             f"{_f(box.get('f_lo_hz'), '{:,.0f}')}–"
             f"{_f(box.get('f_hi_hz'), '{:,.0f}')} Hz, "
             f"{_f(box.get('t0_s'), '{:.3f}')}–{_f(box.get('t1_s'), '{:.3f}')} "
             "s of the source")
    L.append(f"- Canonical rate: {_f(c.get('canonical_rate_hz'), '{:,.0f}')} S/s"
             f" ({c.get('canonical_class', '')} class), integer decimation "
             f"{c.get('decimation')} from {_f(c.get('source_sample_rate'), '{:,.0f}')}"
             " S/s" + (" — LIMITED: the receiver is narrower than the class"
                       if c.get("canonical_limited") else ""))
    L.append(f"- Channels: {c.get('channels', 1)}; samples: "
             f"{_f(c.get('samples'), '{:,}')}; "
             f"{_f(c.get('duration_s'), '{:.3f}')} s")
    L.append(f"- Source: {c.get('source_capture') or 'live stream'}, samples "
             f"{_f(c.get('source_sample_start'), '{:,}')} + "
             f"{_f(c.get('source_sample_count'), '{:,}')}")
    L.append(f"- Cut by {c.get('cut_by', '')} at {c.get('cut_at', '')}"
             + (f" — {c.get('note')}" if c.get("note") else ""))
    L.append(f"- Noise floor of the source span: "
             f"{_f(c.get('floor_db_per_hz'), '{:.1f}')} dBFS/Hz "
             f"({c.get('floor_method', '')})")
    m = a.get("measurements")
    if m:
        L.append("")
        L.append("## Measurements (MEASURED — classical, checkable)")
        L.append("")
        L.append("| Quantity | Value | Method |")
        L.append("|---|---|---|")
        ob = m.get("occupied_bandwidth", {})
        L.append(f"| Occupied bandwidth | {_f(ob.get('value_hz'), '{:,.0f}')} Hz"
                 f" (centre {_f(ob.get('centre_hz'), '{:+,.0f}')} Hz) | "
                 f"{ob.get('method', '')} |")
        sn = m.get("snr", {})
        L.append(f"| SNR above the floor | "
                 + (f"{_f(sn.get('snr_db'), '{:+.1f}')} dB ± "
                    f"{_f(sn.get('se_db'), '{:.2f}')}"
                    if sn.get("measurable") else
                    "not measurable by energy") + f" | {sn.get('method', '')} |")
        sr = m.get("symbol_rate", {})
        L.append(f"| Symbol rate | "
                 + (f"{_f(sr.get('value_hz'), '{:,.2f}')} Bd (confidence "
                    f"{_f(sr.get('confidence'), '{:.2f}')})" if sr.get("known")
                    else "none found") + f" | {sr.get('method', '')} |")
        co = m.get("carrier_offset", {})
        L.append(f"| Carrier offset | {_f(co.get('value_hz'), '{:+,.1f}')} Hz | "
                 f"{co.get('method', '')}; conjugate feature "
                 f"{co.get('conjugate_feature', '')} |")
        bs = m.get("bursts", {})
        L.append(f"| Bursts | {bs.get('count', 0)}; PRI "
                 f"{_f(bs.get('pri_s'), '{:.6f}')} s ({bs.get('pri_kind', '')}); "
                 f"duty {_f(bs.get('duty'), '{:.3f}')} | "
                 f"{bs.get('detector', bs.get('method', ''))} |")
        for k in ("occupied_bandwidth", "symbol_rate"):
            for cav in (m.get(k) or {}).get("caveats", []) or []:
                L.append(f"\n> {k.replace('_', ' ')}: {cav}")
    cy = a.get("cyclic")
    if cy:
        L.append("")
        L.append("## Cyclostationary analysis")
        L.append("")
        L.append("Peaks of the cyclic profile that cleared a threshold derived "
                 f"from a false-alarm rate of {cy.get('pfa', '')} "
                 "(scf.png, cyclic_profile.png):")
        L.append("")
        for p in cy.get("peaks", []):
            L.append(f"- α = {p['alpha_hz']:,.2f} Hz, statistic "
                     f"{p['statistic']:.1f} (threshold {p['threshold']:.1f}) — "
                     f"{p['words']}"
                     + (" — conjugate" if p.get("conj") else ""))
        if not cy.get("peaks"):
            L.append("- none: no symbol clock or carrier feature cleared its "
                     "threshold in this cut")
        if cy.get("candidates"):
            L.append("")
            L.append("Class-table probes matched: "
                     + ", ".join(cy["candidates"]) + ".")
        for w in cy.get("probe_words", []):
            L.append(f"- {w}")
    cl = a.get("class")
    if cl:
        L.append("")
        L.append("## Class")
        L.append("")
        L.append(f"- **{cl.get('cls', '')}** — "
                 + (f"confidence {cl['confidence']:.2f}, "
                    if cl.get("confidence") is not None else "")
                 + f"{cl.get('source', '')} ({_prov.TIER_WORDS.get(cl.get('tier', 'proposed'), '')})")
        if cl.get("why"):
            L.append(f"- {cl['why']}")
    if a.get("fingerprint"):
        L.append(f"- Fingerprint: {a['fingerprint']}")
    cleans = a.get("cleans") or []
    if cleans:
        L.append("")
        L.append("## Cleans")
        L.append("")
        for c2 in cleans:
            L.append(f"### {c2.get('id', '')} — {c2.get('method', '')} "
                     f"({str(c2.get('tier', '')).upper()})")
            L.append("")
            L.append(f"*{_prov.TIER_WORDS.get(c2.get('tier', ''), '')}*")
            L.append("")
            if c2.get("files"):
                L.append(f"- Files: {', '.join(c2['files'])}")
            pm = c2.get("parameters")
            if pm:
                L.append("- Parameters for the demodulator (MEASURED): symbol "
                         f"rate {_f(pm.get('symbol_rate_hz'), '{:,.2f}')} Bd, "
                         f"carrier {_f(pm.get('carrier_offset_hz'), '{:+,.1f}')}"
                         " Hz, first symbol centre "
                         + (_f(pm['timing_offset_s'] * 1e6, '{:,.1f}') + " µs"
                            if pm.get("timing_offset_s") is not None else "—")
                         + " after the cut's first sample")
            if c2.get("no_file"):
                L.append(f"- {c2['no_file']}")
            if c2.get("snr_before_db") is not None:
                L.append(f"- SNR: {_f(c2.get('snr_before_db'), '{:+.1f}')} → "
                         f"{_f(c2.get('snr_after_db'), '{:+.1f}')} dB "
                         f"(measured: {c2.get('snr_method', '')})")
            if c2.get("words"):
                L.append(f"- {c2['words']}")
            used = {k: _plain(v) for k, v in (c2.get("method_params") or {})
                    .items() if k not in ("floor_method", "rolloff_method")}
            used = {k: v for k, v in used.items() if v is not None}
            if used:
                L.append("- Parameters used: " + ", ".join(
                    f"{k} = {v}" for k, v in used.items()))
            if c2.get("note"):
                L.append(f"- {c2['note']}")
            if c2.get("sizing"):
                L.append(f"- Honest size: {c2['sizing']}")
            if c2.get("model_sha256"):
                L.append(f"- Model weights sha256: `{c2['model_sha256']}`")
            L.append("")
    routes = a.get("routes") or []
    if routes:
        L.append("## Routes")
        L.append("")
        for r in routes:
            L.append(f"- **{r['tool']}** on `{r['input']}` "
                     f"({str(r.get('input_tier', '')).upper()}) at {r['at']} by "
                     f"{r.get('by', '')} — {r.get('words', '')}"
                     + (": ran" if r.get("ran") else ": recorded"))
            if r.get("decoded_from_note"):
                L.append(f"  - ⚠ {r['decoded_from_note']}")
            if r.get("input_note"):
                L.append(f"  - {r['input_note']}")
            if r.get("result"):
                L.append(f"  - result: {r['result']}")
    L.append("")
    L.append("## Files")
    L.append("")
    for name, desc in (a.get("files") or {}).items():
        L.append(f"- `{name}` — {desc}")
    L.append("")
    return "\n".join(L)


def write_report(folder) -> "object":
    """Render folder.analysis into report.md, record it in the write log,
    return its path."""
    text = render(folder.analysis)
    p = folder.path / "report.md"
    p.write_text(text, encoding="utf-8")
    folder._record(p, "cut-report")
    return p
