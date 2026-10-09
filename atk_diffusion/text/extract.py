# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""F3 — structured extraction as a canvas, measured against the autoregressive
extractor (plan §4.F3).

ATK's Network Link extracts JSON chunk by chunk with an autoregressive model,
tailored by a Pass 0 briefing (ATK `triage.py`). The plan's idea: write the
JSON TEMPLATE as the canvas — keys, quotes and braces given and pinned, each
value a masked slot — and let the host diffusion loop fill every slot at
once, conditioned on the Pass 0 brief and the document. Whether that is
faster or better is not asserted; `compare(docs, truths, …)` measures both
on the same documents: seconds per document and field accuracy, and how
often each fills a field the truth leaves empty (the hallucination rate,
plan §7).

THE RULES BOTH EXTRACTORS FOLLOW

* THE DOCUMENT IS EVIDENCE, NEVER INSTRUCTIONS. It goes inside a fence
  whose id is a hash of its own text (ported from Bill's ATK `evidence.py`:
  a document cannot close its own fence without containing the hash of
  itself), under a preface that says nothing inside is addressed to the
  model. Lines that read like instructions are listed for the analyst —
  warned about, never silently obeyed or dropped.
* AN EXPLICIT WAY OUT. Bill's rule for small models: every prompt ends by
  saying what to write when the answer is not there — "NSTR" (nothing
  significant to report) or "I don't know". A forced answer converts every
  unknown into a confident wrong value; "NSTR" is honest.
* PREFER NSTR TO INVENTION. A value is kept only if it is found in the
  document (`require_support`); a canvas value is also dropped when the
  loop committed its tokens with low confidence — the unmasking record is
  the model's own measure of how sure it was. Every dropped value says why.
* The classical comparator, `extract_regex` ("Field: value" lines), runs
  beside both.

LIMITS. Slots have a fixed length (`max_tokens`): a canvas cannot grow a
value, and a backend that does not write padding leaves extra words that are
cut at the first closing quote, brace or newline (and, for the word-level
toy, sentence punctuation). The toy backend matches field NAMES literally —
it fills "location" from "Location: …" in the text; a real model reads the
descriptions. Extracted values are PROPOSED (checkable against the cited
document); a value not found in the document is INVENTED and is not used.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from atk_diffusion import provenance
from atk_diffusion.text import backends as B
from atk_diffusion.text.loop import HostDiffusionLoop

for _m, _t in (("canvas_extract", "proposed"), ("ar_extract", "proposed"),
               ("regex_extract", "measured")):
    provenance.METHOD_TIERS.setdefault(_m, _t)

NSTR = "NSTR"
WAY_OUT = ("If the document does not state a field, write NSTR (nothing "
           "significant to report) for it. If you are not sure, say I don't "
           "know. Never guess.")
WAY_OUT_JSON = ('For any field the document does not state, write "NSTR". If '
                'you are not sure, write "I don\'t know". Never guess.')
_NSTR_WORDS = {"nstr", "i don't know", "i dont know", "don't know", "unknown",
               "n/a", "na", "none", "null", "not stated", "not given",
               "not known", "nothing", "-", "—", ""}
MEANING = ("values proposed from the document; each is checked against the "
           "document's own words and dropped (with the reason) if not found")


# ---------------------------------------------------------------------------
# the schema and the Pass 0 brief
# ---------------------------------------------------------------------------

@dataclass
class Field:
    name: str
    description: str = ""
    max_tokens: int = 6
    choices: tuple = ()


def as_schema(spec) -> list[Field]:
    """Fields from a list of names, a {name: description | dict} map, or a
    list of Field."""
    if isinstance(spec, dict):
        items = []
        for k, v in spec.items():
            if isinstance(v, dict):
                items.append(Field(str(k), str(v.get("description", "")),
                                   int(v.get("max_tokens", 6)),
                                   tuple(v.get("choices", ()) or ())))
            else:
                items.append(Field(str(k), str(v or "")))
    else:
        items = [f if isinstance(f, Field) else Field(str(f)) for f in spec]
    names = set()
    for f in items:
        if not f.name or '"' in f.name or f.name in names:
            raise ValueError(f"field name {f.name!r} is empty, repeated or "
                             "holds a quote")
        if f.max_tokens < 1:
            raise ValueError(f"field {f.name!r} needs at least one token")
        names.add(f.name)
    if not items:
        raise ValueError("the schema names no fields")
    return items


BRIEF_FIELDS = ("domain", "perspective", "graph_worthy", "not_graph_worthy")


def _clamp(s, limit: int) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()[:limit]


def render_brief(brief) -> str:
    """The Pass 0 DATA BRIEFING block (ATK `triage.render_brief`'s shape).

    A dict carries Pass 0's fixed fields — each clamped, and dropped whole if
    it reads like an instruction (a briefing is model output and the data
    is adversarial). A string is the analyst's own description: trusted,
    clamped for length only. "" when there is nothing to say."""
    if not brief:
        return ""
    lines = []
    if isinstance(brief, str):
        lines.append(f"- The analyst says: {_clamp(brief, 500)}")
    else:
        for key, label in (("domain", "The data"),
                           ("perspective", "Judge relevance as a"),
                           ("graph_worthy", "What matters"),
                           ("not_graph_worthy", "What does not")):
            val = _clamp(brief.get(key, ""), 240)
            if val and not B.SUSPECT.search(val):
                lines.append(f"- {label}: {val}")
        ctx = _clamp(brief.get("analyst", ""), 500)
        if ctx:
            lines.append(f"- The analyst says: {ctx}")
    if not lines:
        return ""
    return ("DATA BRIEFING — what this data is (Pass 0):\n" + "\n".join(lines)
            + "\nThe briefing narrows what to look for; it never adds a value "
              "the document does not contain.\n\n")


def _field_lines(schema: list[Field]) -> str:
    out = []
    for f in schema:
        d = f": {f.description}" if f.description else ""
        c = (f" (one of: {', '.join(f.choices)})" if f.choices else "")
        out.append(f"- {f.name}{d}{c}")
    return "\n".join(out)


def canvas_prompt(document: str, schema, brief=None) -> str:
    """The canvas's conditioning: brief, fields, the fenced document, and the
    way out — last."""
    schema = as_schema(schema)
    return (render_brief(brief)
            + "Fill each field of the JSON that follows from the document.\n"
            + _field_lines(schema) + "\n\n"
            + B.envelope([B.fence("document", document or "")]) + "\n\n"
            + WAY_OUT)


def ar_prompt(document: str, schema, brief=None) -> str:
    """The autoregressive extractor's prompt, ending with the way out."""
    schema = as_schema(schema)
    keys = ", ".join(f'"{f.name}"' for f in schema)
    return (render_brief(brief)
            + "Extract these fields from the document and return ONLY a JSON "
              f"object with exactly these keys: {keys}.\n"
            + _field_lines(schema) + "\n\n"
            + B.envelope([B.fence("document", document or "")]) + "\n\n"
            + WAY_OUT_JSON)


# ---------------------------------------------------------------------------
# JSON out of a reply (ported from Bill's ATK jsonish.py, 2026-08-31):
# balanced spans, string-aware, and the LAST one wins — a model that thinks
# out loud restates the template first and answers last
# ---------------------------------------------------------------------------

def _spans(text: str):
    depth = start = 0
    in_string = escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                yield start, i + 1


def find_json(text: str, keys=()) -> dict | None:
    """The last parseable JSON object in a reply (with one of `keys`, when
    given), or None."""
    found = []
    for a, b in _spans(text or ""):
        try:
            v = json.loads(text[a:b])
        except json.JSONDecodeError:
            continue
        if isinstance(v, dict):
            found.append(v)
    if keys:
        wanted = [v for v in found if any(k in v for k in keys)]
        if wanted:
            return wanted[-1]
    return found[-1] if found else None


def why_no_json(text: str, limit: int = 160) -> str:
    """One sentence, and the TAIL of the reply — where the answer is or is not."""
    body = (text or "").strip()
    if not body:
        return "the model returned nothing at all"
    spans = list(_spans(body))
    tail_from = spans[-1][1] if spans else 0
    if "{" in body[tail_from:]:
        return ("the reply was CUT OFF while writing the JSON object — raise "
                f"the reply budget. It ended: …{body[-limit:]!r}")
    return f"the model never produced a JSON object. It ended: …{body[-limit:]!r}"


# ---------------------------------------------------------------------------
# checking one value
# ---------------------------------------------------------------------------

def _norm(v) -> str:
    return re.sub(r"\s+", " ", str(v or "")).strip().strip(".,;:\"'").lower()


def is_nstr(v) -> bool:
    return _norm(v) in _NSTR_WORDS


def found_in(value: str, document: str) -> bool:
    """The value's words appear, in order and together, in the document."""
    vw, dw = B.words(value), B.words(document)
    if not vw:
        return False
    n = len(vw)
    return any(dw[i:i + n] == vw for i in range(len(dw) - n + 1))


def _judge(f: Field, value, document: str, require_support: bool,
           confidence: float | None, min_confidence: float, method: str) -> dict:
    raw = "" if value is None else str(value).strip()
    out = {"value": None, "raw": raw, "confidence": confidence,
           "supported": None, "tier": provenance.tier_for(method), "why": ""}
    if raw and B.redaction_marks(raw):
        out["why"] = "redacted in the document; not guessed (plan §5)"
    elif not raw or not any(ch.isalnum() for ch in raw):
        out["why"] = "nothing usable was written"
    elif is_nstr(raw):
        out["why"] = "the model said the document does not state it (NSTR)"
    elif confidence is not None and confidence < min_confidence:
        out["why"] = (f"the model was unsure (mean confidence {confidence:.2f}, "
                      f"under {min_confidence:.2f}); NSTR rather than a guess")
    elif f.choices and _norm(raw) not in {_norm(c) for c in f.choices}:
        out["why"] = f"'{raw}' is not one of the allowed values"
    else:
        sup = found_in(raw, document)
        out["supported"] = sup
        if require_support and not sup:
            out["tier"] = "invented"
            out["why"] = (f"'{raw}' is not in the document — a possible "
                          "invention; not used")
        else:
            out["value"] = raw
            out["why"] = ("found in the document" if sup else
                          "NOT found in the document (support check off)")
    return out


def _result(method: str, fields: dict, detail: dict, seconds: float,
            document: str, **extra) -> dict:
    return {"fields": fields, "detail": detail, "method": method,
            "tier": provenance.tier_for(method), "meaning": MEANING,
            "seconds": seconds,
            "suspect_lines": B.suspect_lines(document),
            "invented": [k for k, d in detail.items() if d["tier"] == "invented"],
            **extra}


# ---------------------------------------------------------------------------
# the canvas extractor
# ---------------------------------------------------------------------------

def build_canvas(schema, backend) -> tuple[list[int], dict]:
    """The JSON template as token ids with each value a run of masks.
    Returns (ids, {field: (start, end)})."""
    schema = as_schema(schema)
    ids = list(backend.encode("{"))
    slots = {}
    for i, f in enumerate(schema):
        ids += backend.encode(f'"{f.name}": "')
        a = len(ids)
        ids += [backend.mask_id] * f.max_tokens
        slots[f.name] = (a, len(ids))
        ids += backend.encode('"' + (", " if i < len(schema) - 1 else ""))
    ids += backend.encode("}")
    return ids, slots


def _token_texts(backend, ids) -> list[str]:
    f = getattr(backend, "token_strings", None)
    if callable(f):
        return [str(t) for t in f(ids)]
    return [backend.decode([int(i)]) for i in ids]


def extract_canvas(document: str, schema, backend, *, brief=None,
                   require_support: bool = True, min_confidence: float = 0.25,
                   loop_kw: dict | None = None) -> dict:
    """Fill the JSON template's slots with the host diffusion loop."""
    if not B.is_masked(backend):
        raise ValueError("the canvas extractor needs a masked (diffusion) "
                         "backend: " + B.describe(backend))
    schema = as_schema(schema)
    t0 = time.perf_counter()
    canvas, slots = build_canvas(schema, backend)
    prompt = backend.encode(canvas_prompt(document, schema, brief))
    n_masks = sum(f.max_tokens for f in schema)
    kw = {"steps": n_masks, "stable_steps": 2, "seed": 0}
    kw.update(loop_kw or {})
    res = HostDiffusionLoop(backend, **kw).fill(canvas, prompt)
    conf = res.confidence_at_commit()
    word_level = bool(getattr(backend, "word_level", False))
    stops = {'"', "}", "\n", "[EOS]", "[PAD]", "</s>", "<eos>", "<pad>"}
    if word_level:
        stops |= {".", ";"}
    others = {_norm(f.name) for f in schema}
    fields, detail = {}, {}
    for f in schema:
        a, b = slots[f.name]
        toks = [int(t) for t in res.tokens[a:b]]
        texts = _token_texts(backend, toks)
        keep = 0
        for j, tx in enumerate(texts):
            s = tx.strip()
            if s in stops or (word_level and _norm(s) in others and j > 0):
                break
            if getattr(backend, "eos_id", None) is not None and \
                    toks[j] == backend.eos_id:
                break
            keep = j + 1
        value = backend.decode(toks[:keep]).strip() if keep else ""
        c = conf[a:a + keep]
        c = c[np.isfinite(c)]
        mean_c = float(c.mean()) if c.size else None
        d = _judge(f, value, document, require_support, mean_c,
                   min_confidence, "canvas_extract")
        if d["value"] is not None:
            d["value"] = B.truecase(d["value"], [document],
                                    sentence_case=False)
        fields[f.name] = d["value"]
        detail[f.name] = d
    return _result("canvas_extract", fields, detail,
                   time.perf_counter() - t0, document, forwards=res.forwards,
                   backend=B.describe(backend), stopped=res.stopped)


# ---------------------------------------------------------------------------
# the autoregressive baseline and the classical one
# ---------------------------------------------------------------------------

def extract_ar(document: str, schema, generate: Callable[[str], str], *,
               brief=None, require_support: bool = True) -> dict:
    """The autoregressive extractor through a host-supplied generate(prompt)
    -> text (ATK's engine; `LlamaCppAR.generate` here)."""
    schema = as_schema(schema)
    t0 = time.perf_counter()
    reply = str(generate(ar_prompt(document, schema, brief)) or "")
    obj = find_json(reply, keys=[f.name for f in schema])
    fields, detail = {}, {}
    why_none = "" if obj is not None else why_no_json(reply)
    for f in schema:
        if obj is None:
            d = {"value": None, "raw": "", "confidence": None,
                 "supported": None, "tier": provenance.tier_for("ar_extract"),
                 "why": why_none}
        else:
            v = obj.get(f.name)
            if isinstance(v, (list, dict)):
                v = json.dumps(v)
            d = _judge(f, v, document, require_support, None, 0.0, "ar_extract")
        fields[f.name] = d["value"]
        detail[f.name] = d
    return _result("ar_extract", fields, detail, time.perf_counter() - t0,
                   document, reply_tail=reply[-240:])


def extract_regex(document: str, schema) -> dict:
    """The classical comparator: "Field: value" in the text, value up to the
    end of its clause."""
    schema = as_schema(schema)
    t0 = time.perf_counter()
    fields, detail = {}, {}
    for f in schema:
        name = re.escape(f.name).replace(r"\_", "[ _-]?").replace("_", "[ _-]?")
        m = re.search(r"(?im)\b" + name + r"\s*[:=]\s*(.+?)\s*(?:\.(?=\s|$)|;|\n|$)",
                      document or "")
        v = m.group(1) if m else None
        d = (_judge(f, v, document, True, None, 0.0, "regex_extract") if m else
             {"value": None, "raw": "", "confidence": None, "supported": None,
              "tier": provenance.tier_for("regex_extract"),
              "why": f"no '{f.name}:' line in the document"})
        fields[f.name] = d["value"]
        detail[f.name] = d
    return _result("regex_extract", fields, detail, time.perf_counter() - t0,
                   document)


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------

def _same(pred, truth) -> bool:
    if truth is None or is_nstr(truth):
        return pred is None
    return pred is not None and _norm(pred) == _norm(truth)


def compare(docs, truths, schema, *, backend=None, generate=None, brief=None,
            require_support: bool = True, min_confidence: float = 0.25,
            loop_kw: dict | None = None,
            progress: Callable[[str], None] | None = None) -> dict:
    """Speed and field accuracy of the canvas extractor (when a masked
    backend is given), the autoregressive one (when `generate` is given) and
    the regex baseline (always), on the same documents.

    truths: one {field: value or None} per document. A field the truth leaves
    empty that an extractor fills counts toward its hallucination rate."""
    say = progress or (lambda _m: None)
    schema = as_schema(schema)
    docs = list(docs)
    truths = list(truths)
    if len(docs) != len(truths):
        raise ValueError(f"{len(docs)} documents but {len(truths)} truths; give "
                         "one truth per document")
    arms: dict[str, Callable[[str], dict]] = {
        "regex": lambda d: extract_regex(d, schema)}
    skipped = {}
    if B.is_masked(backend):
        arms["canvas"] = lambda d: extract_canvas(
            d, schema, backend, brief=brief, require_support=require_support,
            min_confidence=min_confidence, loop_kw=loop_kw)
    else:
        skipped["canvas"] = ("no masked backend was given: "
                             + B.describe(backend))
    if generate is not None:
        arms["ar"] = lambda d: extract_ar(d, schema, generate, brief=brief,
                                          require_support=require_support)
    else:
        skipped["ar"] = "no generate(prompt) callable was given"
    out = {}
    for name, fn in arms.items():
        correct = invented = missed = wrong = empty_truth = 0
        seconds = 0.0
        forwards = 0
        per_doc = []
        for i, (d, t) in enumerate(zip(docs, truths)):
            say(f"{name}: document {i + 1} of {len(docs)}")
            t0 = time.perf_counter()
            r = fn(d)
            seconds += time.perf_counter() - t0
            forwards += int(r.get("forwards", 0) or 0)
            row = {}
            for f in schema:
                p, tv = r["fields"].get(f.name), (t or {}).get(f.name)
                ok = _same(p, tv)
                correct += ok
                if tv is None or is_nstr(tv):
                    empty_truth += 1
                    invented += p is not None
                elif p is None:
                    missed += 1
                elif not ok:
                    wrong += 1
                row[f.name] = {"predicted": p, "truth": tv, "correct": ok}
            per_doc.append(row)
        n = len(docs) * len(schema)
        out[name] = {
            "method": {"regex": "regex_extract", "canvas": "canvas_extract",
                       "ar": "ar_extract"}[name],
            "accuracy": correct / n if n else float("nan"),
            "hallucination_rate": (invented / empty_truth) if empty_truth
            else 0.0,
            "missed": missed, "wrong": wrong, "filled_when_empty": invented,
            "seconds_per_doc": seconds / max(1, len(docs)),
            "forwards_per_doc": forwards / max(1, len(docs)),
            "per_document": per_doc}
        out[name]["tier"] = provenance.tier_for(out[name]["method"])
    return {"arms": out, "skipped": skipped, "documents": len(docs),
            "fields": len(schema),
            "meaning": ("field accuracy against the given truths: exact match "
                        "after normalising case and spacing; an empty truth "
                        "is matched only by an empty answer")}
