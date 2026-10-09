# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""F4 — style fidelity as predictability, and terminology-consistent
translation post-edit (plan §4.F4).

STYLE AS A NUMBER. The Writing Workshop's fingerprints (ATK `authors.py`)
describe how somebody writes; the plan asks for them to "become a number".
Two numbers, each returned with what it means:

* `style_fidelity(passage, author_context, backend)` — PREDICTABILITY: the
  passage's pointwise mutual information with the author's writing, per
  token, log p(passage | author context) − log p(passage | generic context).
  Positive: the author's writing makes this passage more predictable to the
  model — it reads like them, to that model. Near zero: no more like this
  author than like plain prose. A masked backend scores both sides with the
  loop's surprise field; an autoregressive one with log-probabilities.
* `style_distance(text, reference)` — CLASSICAL stylometrics, always
  available: a Burrows-style Delta over function-word rates (how many of
  the author's own standard deviations the text sits from the author's mean,
  averaged over the commonest function words) and the Jensen–Shannon
  divergence between sentence-length distributions. `stylometrics(text)` is
  the profile behind both.

The author context is used for SCORING ONLY. Nothing here writes in anyone's
style, and nothing from the samples is copied into any output — ATK's rule
that a profile must not leak its author's sentences (`authors.py`) holds.

TERMINOLOGY. `terminology_check(glossary, target_segments, source_segments)`
finds where a translation renders a glossary term some other way: with the
source segments, every segment whose source holds the term must hold the
required rendering; without them, the glossary's listed variants are
flagged wherever they occur. A glossary entry with no required rendering
takes the document's own most frequent rendering — consistency with itself.
Each inconsistency comes with a proposed post-edit (the required rendering,
its capitalisation matched), and `apply_postedits` applies only the ones the
analyst accepts. With a masked backend, each edit is also checked for
agreement: the model's surprise at the words around it, before and after —
bidirectional context is what keeps a term rendered one way across a
document, and what notices "a army".

LIMITS. A Delta from fewer than ~1,000 words of reference or ~200 words of
text is noise, and the result says how reliable it is. PMI depends entirely
on the backend; a model that does not know the author's other work learns
the style only from the context it is given. Fuzzy variant search finds
near spellings and listed variants, not synonyms. Not a translation tool:
proposed post-edits are PROPOSED; the check itself is MEASURED.
"""

from __future__ import annotations

import collections
import difflib
import math
import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from atk_diffusion import provenance
from atk_diffusion.text import backends as B
from atk_diffusion.text.loop import HostDiffusionLoop

for _m, _t in (("style_pmi", "proposed"), ("stylometric", "measured"),
               ("term_check", "measured"), ("term_postedit", "proposed")):
    provenance.METHOD_TIERS.setdefault(_m, _t)

LN2 = math.log(2.0)
GENERIC_CONTEXT = ("The following is a passage of ordinary English prose, "
                   "written plainly.")
SENT_BINS = (1, 6, 11, 16, 21, 31, 46)
SENT_LABELS = ("1-5", "6-10", "11-15", "16-20", "21-30", "31-45", "46+")


# ---------------------------------------------------------------------------
# classical stylometrics
# ---------------------------------------------------------------------------

def _sentence_lengths(text: str) -> list[int]:
    return [n for n in (len(B.words(s)) for s, _a, _b in B.split_sentences(text))
            if n > 0]


def _hist(lengths) -> np.ndarray:
    edges = list(SENT_BINS) + [10 ** 9]
    h = np.zeros(len(SENT_LABELS))
    for n in lengths:
        for k in range(len(SENT_LABELS)):
            if edges[k] <= n < edges[k + 1]:
                h[k] += 1
                break
    return h / h.sum() if h.sum() else h


def _mattr(ws: list[str], window: int = 50) -> float:
    """Moving-average type-token ratio: vocabulary richness that does not
    fall just because a text is long."""
    if not ws:
        return 0.0
    if len(ws) <= window:
        return len(set(ws)) / len(ws)
    vals = [len(set(ws[i:i + window])) / window
            for i in range(0, len(ws) - window + 1)]
    return float(np.mean(vals))


def stylometrics(text: str) -> dict:
    """The measurable half of a style: sentence lengths, function-word rates
    (per 1,000 words), word length, vocabulary richness, commas."""
    ws = B.words(text)
    lens = _sentence_lengths(text)
    fw = collections.Counter(w for w in ws if w in B.FUNCTION_SET)
    n = len(ws)
    return {
        "words": n, "sentences": len(lens),
        "sentence_length": {
            "mean": float(np.mean(lens)) if lens else 0.0,
            "std": float(np.std(lens)) if lens else 0.0,
            "median": float(np.median(lens)) if lens else 0.0,
            "histogram": [float(x) for x in _hist(lens)],
            "bins": list(SENT_LABELS)},
        "function_words": {w: 1000.0 * fw[w] / n for w in B.FUNCTION_WORDS
                           if n and fw[w]},
        "mean_word_length": float(np.mean([len(w) for w in ws])) if ws else 0.0,
        "type_token_ratio": _mattr(ws),
        "commas_per_sentence": ((text or "").count(",") / len(lens)
                                if lens else 0.0),
        "method": "stylometric", "tier": provenance.tier_for("stylometric"),
    }


def _chunks(texts: Sequence[str], size: int) -> list[list[str]]:
    ws = [w for t in texts for w in B.words(t)]
    out = [ws[i:i + size] for i in range(0, len(ws), size)]
    if len(out) > 1 and len(out[-1]) < size // 2:
        out[-2].extend(out.pop())
    return [c for c in out if c]


def _rates(ws: list[str]) -> dict:
    c = collections.Counter(w for w in ws if w in B.FUNCTION_SET)
    n = max(1, len(ws))
    return {w: c[w] / n for w in B.FUNCTION_WORDS}


def _js_bits(p: np.ndarray, q: np.ndarray) -> float:
    if not p.sum() or not q.sum():
        return float("nan")
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        nz = a > 0
        return float((a[nz] * np.log2(a[nz] / b[nz])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def style_distance(text: str, reference, *, chunk_words: int = 200,
                   top: int = 30) -> dict:
    """How far `text` sits from a reference author's measured style.

    delta  Burrows-style: over the reference's `top` commonest function
           words, |rate in text − author's mean rate| / author's own standard
           deviation across chunks, averaged. About 1 = within the author's
           usual variation; 2 or more = writes differently.
    sentence_length_js  Jensen–Shannon divergence (bits, 0–1) between the
           sentence-length distributions."""
    refs = [reference] if isinstance(reference, str) else list(reference)
    chunks = _chunks(refs, chunk_words)
    ref_words = [w for c in chunks for w in c]
    tw = B.words(text)
    if not ref_words or not tw:
        raise ValueError("style_distance needs words in both the text and the "
                         "reference")
    rates = np.array([[_rates(c)[w] for w in B.FUNCTION_WORDS] for c in chunks])
    mu = rates.mean(axis=0)
    trate = np.array([_rates(tw)[w] for w in B.FUNCTION_WORDS])
    # the commonest function words of reference AND text together (Burrows):
    # a word the text leans on and the author never uses is evidence too
    pooled = _rates(ref_words + tw)
    order = sorted(range(len(B.FUNCTION_WORDS)),
                   key=lambda k: (-pooled[B.FUNCTION_WORDS[k]], k))
    use = [k for k in order[:top] if pooled[B.FUNCTION_WORDS[k]] > 0]
    sd = rates.std(axis=0, ddof=1) if len(chunks) > 1 else np.zeros_like(mu)
    # a floor from counting noise: a rate measured over `chunk_words` words,
    # and for a word the author never used, a rate below 1 in the reference
    mu_eff = np.maximum(mu, 0.5 / len(ref_words))
    sd = np.maximum(sd, np.sqrt(mu_eff / max(1, chunk_words)))
    # and the text's own counting noise: a short text cannot be far from
    # anyone with confidence, so it is not allowed to look far
    sd = np.sqrt(sd ** 2 + mu_eff / len(tw))
    delta = (float(np.mean(np.abs(trate[use] - mu[use]) / sd[use]))
             if use else float("nan"))
    js = _js_bits(_hist(_sentence_lengths(text)),
                  _hist([n for r in refs for n in _sentence_lengths(r)]))
    rel = ("good" if len(ref_words) >= 1000 and len(tw) >= 200 else
           "fair" if len(ref_words) >= 400 and len(tw) >= 80 else "low")
    return {
        "delta": delta, "sentence_length_js": js,
        "function_words_compared": len(use),
        "reference_words": len(ref_words), "text_words": len(tw),
        "reference_chunks": len(chunks), "reliability": rel,
        "meaning": (f"Delta {delta:.2f}: the text's function-word rates sit "
                    f"{delta:.2f} of the author's own standard deviations "
                    "from the author's mean (about 1 is within their usual "
                    "variation; 2 or more reads differently). Sentence-length "
                    f"divergence {js:.2f} bits (0 = the same distribution, "
                    f"1 = nothing in common). Reliability: {rel} — "
                    f"{len(ref_words)} reference words, {len(tw)} text words."),
        "method": "stylometric", "tier": provenance.tier_for("stylometric"),
    }


# ---------------------------------------------------------------------------
# style as predictability
# ---------------------------------------------------------------------------

def style_fidelity(passage: str, author_context: str, backend, *,
                   generic_context: str = GENERIC_CONTEXT,
                   passes: int | None = 4) -> dict:
    """PMI per token between the passage and the author's context, against a
    generic context: how much more predictable the author makes it."""
    if not (author_context or "").strip():
        raise ValueError("style fidelity needs some of the author's writing "
                         "(or the operator's own) as context")
    if B.is_masked(backend):
        ids = backend.encode(passage)
        if not ids:
            raise ValueError("the passage holds no tokens")
        loop = HostDiffusionLoop(backend, steps=2, seed=0)
        sa = loop.surprise_field(ids, backend.encode(author_context),
                                 passes=passes) / LN2
        sg = loop.surprise_field(ids, backend.encode(generic_context),
                                 passes=passes) / LN2
        n = len(ids)
        ba, bg = float(sa.mean()), float(sg.mean())
        how = "masked re-scoring (the loop's surprise field)"
    elif B.is_scoring(backend):
        f = getattr(backend, "scored_tokens", None)
        if callable(f):
            a = f(passage, author_context)
            g = f(passage, generic_context)
            n = max(1, len(a))
            ba = -sum(v for _t, v in a) / n / LN2
            bg = -sum(v for _t, v in g) / max(1, len(g)) / LN2
        else:
            n = max(1, len(B.tokenize(passage)))
            ba = -backend.logprob(passage, author_context) / n / LN2
            bg = -backend.logprob(passage, generic_context) / n / LN2
        how = "left-to-right log-probabilities"
    else:
        raise ValueError("style fidelity needs a model backend; without one, "
                         "use style_distance (classical stylometrics)")
    pmi = bg - ba
    if pmi > 0.25:
        reads = "reads MORE like the author than like generic prose"
    elif pmi < -0.25:
        reads = "reads LESS like the author than like generic prose"
    else:
        reads = "reads no more like the author than like generic prose"
    return {
        "pmi_bits_per_token": pmi, "bits_given_author": ba,
        "bits_given_generic": bg, "tokens": n, "scored_by": how,
        "backend": B.describe(backend),
        "meaning": (f"{pmi:+.2f} bits per token: to this model the passage "
                    f"{reads}. A model's view of predictability, not a "
                    "judgement of quality."),
        "method": "style_pmi", "tier": provenance.tier_for("style_pmi"),
    }


def style_report(passage: str, author_samples, backend=None) -> dict:
    """Both numbers: the classical distance always, the model's PMI when a
    backend is given."""
    samples = [author_samples] if isinstance(author_samples, str) else \
        list(author_samples)
    out = {"stylometric": style_distance(passage, samples),
           "profile": stylometrics(passage)}
    if backend is not None:
        out["predictability"] = style_fidelity(passage, " ".join(samples),
                                               backend)
    return out


# ---------------------------------------------------------------------------
# terminology-consistent post-edit
# ---------------------------------------------------------------------------

@dataclass
class TermRule:
    source: str
    rendering: str | None
    variants: tuple = ()


def as_glossary(glossary) -> list[TermRule]:
    """{source term: required rendering} or {source term: {"rendering",
    "variants"}}; a rendering of None takes the document's majority."""
    if not isinstance(glossary, dict) or not glossary:
        raise ValueError("the glossary must be a non-empty {source term: "
                         "rendering} map")
    out = []
    for src, v in glossary.items():
        if isinstance(v, dict):
            out.append(TermRule(str(src), v.get("rendering"),
                                tuple(v.get("variants", ()) or ())))
        else:
            out.append(TermRule(str(src), None if v is None else str(v)))
    for r in out:
        if not r.rendering and not r.variants:
            raise ValueError(f"glossary term {r.source!r} has neither a "
                             "rendering nor variants to choose among")
    return out


def _find(term: str, text: str) -> list[tuple[int, int]]:
    if not term:
        return []
    pat = r"(?<!\w)" + re.escape(term.strip()) + r"(?!\w)"
    return [(m.start(), m.end()) for m in re.finditer(pat, text or "", re.I)]


def _fuzzy(rendering: str, text: str, taken, min_similarity: float):
    """Word spans of `text` that look like `rendering` without being it."""
    spans = [(m.start(), m.end()) for m in B.WORD_RE.finditer(text or "")]
    k = len(B.words(rendering))
    want = rendering.lower()
    cands = []
    for n in range(max(1, k - 1), k + 2):
        for i in range(0, len(spans) - n + 1):
            a, b = spans[i][0], spans[i + n - 1][1]
            if any(not (b <= x or a >= y) for x, y in taken):
                continue
            s = text[a:b]
            if s.lower() == want:
                continue
            r = difflib.SequenceMatcher(None, s.lower(), want).ratio()
            if r >= min_similarity:
                cands.append((r, a, b))
    cands.sort(key=lambda t: (-t[0], t[1]))
    out = []
    for r, a, b in cands:
        if all(b <= x or a >= y for x, y, _r in out):
            out.append((a, b, r))
    return sorted(out)


def _match_case(found: str, rendering: str) -> str:
    if found[:1].isupper() and rendering[:1].islower():
        return rendering[:1].upper() + rendering[1:]
    return rendering


def agreement_check(before: str, after: str, start: int, end_before: int,
                    end_after: int, backend, window: int = 3) -> dict | None:
    """The model's surprise at the words around an edit, before and after it.
    A clear rise suggests the edit broke agreement around it ("a army")."""
    if not B.is_masked(backend):
        return None
    loop = HostDiffusionLoop(backend, steps=2, seed=0)

    def around(text, a, b):
        left = backend.encode(text[:a])
        mid = backend.encode(text[a:b])
        ids = left + mid + backend.encode(text[b:])
        s = loop.surprise_field(ids, [], passes=None) / LN2
        idx = list(range(max(0, len(left) - window), len(left))) + \
            list(range(len(left) + len(mid),
                       min(len(ids), len(left) + len(mid) + window)))
        return float(s[idx].mean()) if idx else 0.0
    b0 = around(before, start, end_before)
    b1 = around(after, start, end_after)
    rise = b1 - b0
    return {"bits_before": b0, "bits_after": b1, "rise_bits": rise,
            "flag": rise > 2.0,
            "meaning": ("the words around the edit became much less likely to "
                        "the model — check the grammar there" if rise > 2.0
                        else "the words around the edit read as before")}


def terminology_check(glossary, target_segments, source_segments=None, *,
                      min_similarity: float = 0.6, backend=None) -> dict:
    """Find inconsistent renderings of glossary terms and propose post-edits.

    target_segments  the translation: a string (split into sentences) or a
                     list of segments
    source_segments  the source, segment-aligned with the target (ATK's
                     translator keeps numbered segments aligned) — optional
    """
    rules = as_glossary(glossary)
    if isinstance(target_segments, str):
        targets = [s for s, _a, _b in B.split_sentences(target_segments)]
    else:
        targets = [str(t) for t in target_segments]
    sources = None
    if source_segments is not None:
        sources = [str(s) for s in source_segments]
        if len(sources) != len(targets):
            raise ValueError(f"{len(sources)} source segments but "
                             f"{len(targets)} target segments; they must be "
                             "aligned one to one")
    findings, terms = [], {}
    for rule in rules:
        required = rule.rendering
        majority_note = ""
        if not required:
            counts = {v: sum(len(_find(v, t)) for t in targets)
                      for v in rule.variants}
            required = max(rule.variants, key=lambda v: (counts[v], -len(v)))
            majority_note = (f"no rendering was fixed; the document's own most "
                             f"frequent one ('{required}', {counts[required]} "
                             "times) is taken")
        variants = [v for v in rule.variants if v.lower() != required.lower()]
        stat = {"required": required, "consistent": 0, "inconsistent": 0,
                "missing": 0, "note": majority_note}
        for i, tgt in enumerate(targets):
            uses = _find(required, tgt)
            stat["consistent"] += len(uses)
            vs = []
            for v in variants:
                vs += [(a, b, 1.0) for a, b in _find(v, tgt)
                       if all(b <= x or a >= y for x, y in uses)]
            expected = (len(_find(rule.source, sources[i])) if sources is not None
                        else None)
            if expected is not None:
                need = max(0, expected - len(uses))
                if need and len(vs) < need:
                    vs += _fuzzy(required, tgt, uses + [(a, b) for a, b, _r in vs],
                                 min_similarity)[:need - len(vs)]
                vs = sorted(vs)[:need]
                for _k in range(need - len(vs)):
                    stat["missing"] += 1
                    findings.append({
                        "segment": i, "kind": "missing", "source_term":
                        rule.source, "required": required, "found": "",
                        "start": None, "end": None,
                        "why": f"the source has '{rule.source}' here but no "
                               f"rendering of it was found in the translation"})
            for a, b, r in vs:
                stat["inconsistent"] += 1
                found = tgt[a:b]
                findings.append({
                    "segment": i, "kind": "variant", "source_term": rule.source,
                    "required": required, "found": found, "start": a, "end": b,
                    "similarity": r,
                    "why": f"'{found}' where the glossary requires "
                           f"'{required}' for '{rule.source}'"})
        terms[rule.source] = stat
    proposals = []
    for n, f in enumerate(findings):
        if f["kind"] != "variant":
            continue
        tgt = targets[f["segment"]]
        rep = _match_case(f["found"], f["required"])
        new = tgt[:f["start"]] + rep + tgt[f["end"]:]
        p = {"id": n, "segment": f["segment"], "before": tgt, "after": new,
             "replace": f["found"], "with": rep, "start": f["start"],
             "end": f["end"], "method": "term_postedit",
             "tier": provenance.tier_for("term_postedit")}
        if backend is not None:
            p["agreement"] = agreement_check(tgt, new, f["start"], f["end"],
                                             f["start"] + len(rep), backend)
        proposals.append(p)
    return {
        "findings": findings, "proposals": proposals, "terms": terms,
        "consistent": not findings, "segments": len(targets),
        "method": "term_check", "tier": provenance.tier_for("term_check"),
        "meaning": ("where the translation renders a glossary term another "
                    "way; each proposal is an edit for the analyst to accept "
                    "or reject"),
    }


def apply_postedits(target_segments, proposals, accept=None) -> dict:
    """Apply the accepted proposals (all when `accept` is None) and return the
    edited segments with a log. Edits in one segment apply right to left, so
    offsets stay valid."""
    targets = ([s for s, _a, _b in B.split_sentences(target_segments)]
               if isinstance(target_segments, str) else list(target_segments))
    chosen = [p for p in proposals if accept is None or p["id"] in set(accept)]
    by_seg: dict[int, list] = collections.defaultdict(list)
    for p in chosen:
        by_seg[p["segment"]].append(p)
    applied, skipped = [], []
    for seg, ps in by_seg.items():
        text = targets[seg]
        for p in sorted(ps, key=lambda q: -q["start"]):
            if text[p["start"]:p["end"]] != p["replace"]:
                skipped.append({"id": p["id"], "why": "the segment changed "
                                "since the proposal was made"})
                continue
            text = text[:p["start"]] + p["with"] + text[p["end"]:]
            applied.append(p["id"])
        targets[seg] = text
    return {"segments": targets, "applied": sorted(applied), "skipped": skipped,
            "method": "term_postedit",
            "tier": provenance.tier_for("term_postedit")}
