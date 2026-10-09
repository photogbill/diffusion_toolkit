# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""D2's first experiment: word error rate with and without enhancement, on
the same clips, plus a hallucination check (plan §4.D2, §6 Phase 1, §7).

    *"First experiment: word error rate on Bill's transcripts with and
    without, on the same clips."*  — plan §4.D2
    *"Hallucination rate. For every reconstruction tool: how often does it
    produce a signal, pulse, word or character where the ground truth has
    none?"*  — plan §7

The host supplies the transcriber — `transcribe(wav_path) -> text` (ATK's
Whisper; a dict with "text" or an object with `.text` is accepted too) — so
this runs the real thing on Bill's machine and anything at all in a test.
Every enhancer is a callable `(in_wav, out_wav) -> dict`: the classical
ones from `repair.speech.classical_enhancers()`, the learned one an
`SgmseRunner` (its `.enhance`), any other an `ExternalEnhancer`.

WHAT COMES OUT. One row per method, RAW FIRST (classical baselines before
learned, plan §7): corpus WER and CER, substitutions / deletions /
insertions, and — paired — the raw WER on exactly the clips that method
processed, how many clips got better, worse or stayed the same. A method
that failed on a clip is counted as failed, not dropped silently. Then the
hallucination check: the same transcriber on NOISE-ONLY clips, where any
word at all is invented — reported per method as words per clip, the share
of clips with any word, and the words themselves (Whisper's own favourites
— "thank you", "you" — show up here for the raw audio too, which is the
baseline the enhancers are judged against).

Enhanced audio and the report (markdown + JSON) go under `out_dir`, which
the caller chooses; nothing is written anywhere else.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from atk_diffusion import provenance as _prov
from atk_diffusion.repair import wer as _wer

RAW = "raw"


def _say(progress, msg: str) -> None:
    if progress:
        progress(msg)


def _refs(clips, references) -> list[str]:
    if isinstance(references, dict):
        out = []
        for c in clips:
            p = Path(c)
            for key in (str(c), p.name, p.stem):
                if key in references:
                    out.append(str(references[key]))
                    break
            else:
                raise ValueError(f"no reference transcript for {p.name}")
        return out
    refs = [str(r) for r in references]
    if len(refs) != len(clips):
        raise ValueError(f"{len(clips)} clips but {len(refs)} reference "
                         "transcripts — they must pair one to one")
    return refs


def _tier_of(info, name: str) -> str:
    if isinstance(info, dict) and info.get("tier"):
        return str(info["tier"])
    try:
        return _prov.tier_for(name)
    except ValueError:
        return ""


def wer_with_without(clips, references, transcribe, enhancers=None,
                     out_dir=None, noise_clips=(), progress=None,
                     write_report: bool = True) -> dict:
    """Score `transcribe` on `clips` raw and through each enhancer.

    `references` pairs with `clips` (a list in the same order, or a dict
    keyed by path, file name or stem). `noise_clips` are noise-only WAVs for
    the hallucination check. Returns {rows, hallucination, per_clip, lines,
    report_md, report_json}."""
    clips = [Path(c) for c in clips]
    refs = _refs(clips, references)
    if enhancers is None:
        from atk_diffusion.repair.speech import classical_enhancers
        enhancers = classical_enhancers()
    enhancers = dict(enhancers)
    if RAW in enhancers:
        raise ValueError("'raw' is the unenhanced audio; name the enhancer "
                         "something else")
    if (enhancers or write_report) and out_dir is None:
        raise ValueError("an output folder is needed for the enhanced audio "
                         "and the report (nothing is written to a temporary "
                         "folder)")
    out = Path(out_dir) if out_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    methods = [RAW] + list(enhancers)

    def run_one(method, src: Path, dst_dir: Path | None):
        """-> (text, tier, error, seconds)"""
        t0 = time.time()
        path, tier = src, "record"
        if method != RAW:
            dst = dst_dir / method / f"{src.stem}.wav"
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                info = enhancers[method](src, dst)
            except Exception as e:                             # noqa: BLE001
                return None, _tier_of(None, method), f"{type(e).__name__}: {e}", \
                    time.time() - t0
            tier = _tier_of(info, method)
            path = Path(info.get("output", dst)) if isinstance(info, dict) else dst
        try:
            text = _wer.text_of(transcribe(str(path)))
        except Exception as e:                                 # noqa: BLE001
            return None, tier, f"transcription failed: {type(e).__name__}: {e}", \
                time.time() - t0
        return text, tier, "", time.time() - t0

    per_clip = []
    tiers = {RAW: "record"}
    for k, (clip, ref) in enumerate(zip(clips, refs)):
        _say(progress, f"clip {k + 1}/{len(clips)}: {clip.name}")
        row = {"clip": str(clip), "reference": ref, "methods": {}}
        for m in methods:
            text, tier, err, secs = run_one(m, clip, out)
            if tier:
                tiers.setdefault(m, tier)
            entry = {"text": text, "error": err, "seconds": round(secs, 3)}
            if text is not None:
                w = _wer.wer(ref, text)
                c = _wer.cer(ref, text)
                entry.update(wer=w.to_json(), cer=c.to_json())
            row["methods"][m] = entry
        per_clip.append(row)

    def score(m, subset):
        ws, cs = _wer.Score(), _wer.Score(unit="character")
        for r in subset:
            e = r["methods"][m]
            ws = ws + _wer.Score(**{k: e["wer"][k] for k in
                                   ("ref_len", "hyp_len", "substitutions",
                                    "deletions", "insertions", "hits")})
            cs = cs + _wer.Score(**{k: e["cer"][k] for k in
                                   ("ref_len", "hyp_len", "substitutions",
                                    "deletions", "insertions", "hits")})
        return ws, cs

    rows = []
    raw_ok = [r for r in per_clip if r["methods"][RAW]["text"] is not None]
    for m in methods:
        done = [r for r in raw_ok if r["methods"][m]["text"] is not None]
        failed = [r for r in per_clip if r["methods"][m]["text"] is None]
        ws, cs = score(m, done)
        rws, _ = score(RAW, done)
        better = worse = same = 0
        for r in done:
            a = r["methods"][m]["wer"]
            b = r["methods"][RAW]["wer"]
            ea = a["substitutions"] + a["deletions"] + a["insertions"]
            eb = b["substitutions"] + b["deletions"] + b["insertions"]
            better += ea < eb
            worse += ea > eb
            same += ea == eb
        rows.append({
            "method": m, "tier": tiers.get(m, ""), "clips": len(done),
            "failed": len(failed),
            "failures": [{"clip": r["clip"], "why": r["methods"][m]["error"]}
                         for r in failed],
            "wer": ws.rate if done else None, "cer": cs.rate if done else None,
            "substitutions": ws.substitutions, "deletions": ws.deletions,
            "insertions": ws.insertions, "reference_words": ws.ref_len,
            "raw_wer_same_clips": rws.rate if done else None,
            "delta_wer": (ws.rate - rws.rate) if done else None,
            "better": better, "worse": worse, "same": same,
            "seconds": round(sum(r["methods"][m]["seconds"] for r in done), 2)})

    halluc = []
    noise = [Path(p) for p in noise_clips]
    for m in methods:
        words_out, with_words, failed, texts = 0, 0, 0, []
        for nc in noise:
            text, _tier, err, _s = run_one(m, nc, (out / "noise_only") if out else None)
            if text is None:
                failed += 1
                continue
            w = _wer.words(text)
            words_out += len(w)
            with_words += bool(w)
            if w:
                texts.append({"clip": nc.name, "text": text})
        n_ok = len(noise) - failed
        halluc.append({"method": m, "clips": n_ok, "failed": failed,
                       "words": words_out,
                       "words_per_clip": (words_out / n_ok) if n_ok else None,
                       "clips_with_words": with_words,
                       "share_with_words": (with_words / n_ok) if n_ok else None,
                       "texts": texts[:20]})

    lines = _lines(rows, halluc)
    result = {"experiment": "D2 wer_with_without", "rows": rows,
              "hallucination": halluc, "per_clip": per_clip, "lines": lines,
              "normalisation": _wer.NORMALISATION,
              "provenance": _prov.stamp("experiments.wer_eval")}
    if write_report and out is not None:
        jp = out / "wer_report.json"
        jp.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        mp = out / "wer_report.md"
        mp.write_text(_markdown(result), encoding="utf-8")
        result["report_json"], result["report_md"] = str(jp), str(mp)
    return result


def _pct(v) -> str:
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.1%}"


def _lines(rows, halluc) -> list[str]:
    out = []
    raw = rows[0]
    out.append(f"Raw audio: WER {_pct(raw['wer'])} on {raw['clips']} clip(s) "
               f"({raw['insertions']} inserted words).")
    for r in rows[1:]:
        if not r["clips"]:
            out.append(f"{r['method']}: no clip could be processed "
                       f"({r['failed']} failed) — "
                       + (r["failures"][0]["why"] if r["failures"] else ""))
            continue
        d = r["delta_wer"]
        verb = "lowered" if d < 0 else ("raised" if d > 0 else "left")
        out.append(f"{r['method']} ({r['tier'].upper() or '?'}): {verb} the WER "
                   f"from {_pct(r['raw_wer_same_clips'])} to {_pct(r['wer'])} on "
                   f"the same {r['clips']} clip(s) — better on {r['better']}, "
                   f"worse on {r['worse']}"
                   + (f"; {r['failed']} clip(s) failed" if r["failed"] else "")
                   + ".")
    if halluc and halluc[0]["clips"]:
        for h in halluc:
            out.append(f"Noise only, {h['method']}: {h['words']} word(s) from "
                       f"{h['clips']} clip(s) with no speech "
                       f"({_pct(h['share_with_words'])} of clips produced "
                       "words).")
    return out


def _markdown(res: dict) -> str:
    L = ["# D2 — word error rate with and without enhancement", "",
         "Same clips, same transcriber. Raw first; classical before learned. "
         "INSERTIONS are words the speaker never said.", "",
         "| method | tier | clips | WER | raw WER, same clips | Δ | CER | S | D | I "
         "| better | worse | failed |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        d = r["delta_wer"]
        L.append(f"| {r['method']} | {r['tier']} | {r['clips']} | {_pct(r['wer'])} | "
                 f"{_pct(r['raw_wer_same_clips'])} | "
                 f"{'—' if d is None else f'{d * 100:+.1f} pts'} | {_pct(r['cer'])} | "
                 f"{r['substitutions']} | {r['deletions']} | {r['insertions']} | "
                 f"{r['better']} | {r['worse']} | {r['failed']} |")
    if res["hallucination"] and res["hallucination"][0]["clips"]:
        L += ["", "## Hallucination check — noise-only clips", "",
              "| method | clips | words | words/clip | clips with words |",
              "|---|---|---|---|---|"]
        for h in res["hallucination"]:
            wpc = h["words_per_clip"]
            L.append(f"| {h['method']} | {h['clips']} | {h['words']} | "
                     f"{'—' if wpc is None else f'{wpc:.2f}'} | "
                     f"{_pct(h['share_with_words'])} |")
        L.append("")
        for h in res["hallucination"]:
            for t in h["texts"][:5]:
                L.append(f"- {h['method']}, {t['clip']}: “{t['text']}”")
    L += ["", "## In words", ""] + [f"- {ln}" for ln in res["lines"]]
    L += ["", "Normalisation: " + " ".join(res.get("normalisation", "").split())]
    return "\n".join(L) + "\n"
