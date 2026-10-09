# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""F2 — revising in place, and composing as one canvas (plan §4.F2, §4.H).

THE HOUR LOG'S REVISION STEP. Bill, 2026-10-08: *"I need an hour by hour
automatic log producer that is exportable … over the course of years, they
always had me log what I did over that hour."* ATK builds the log; each hour
gets one entry derived ONLY from recorded events. Diffusion's part (plan §4.H)
is narrow: when an hour's facts complete late — a transcript finishes at :20
— the entry is revised IN PLACE rather than a correction appended. "Build the
honest log first; the model polishes it."

`revise_in_place(document, new_facts, backend)` therefore does four things,
and the fourth is the point:

1. ALIGN each new fact to the sentence it updates — by event id when the
   host knows it (an hour-log line is tied to its event), else by the text
   the fact says it `replaces`, else by TF-IDF similarity. A fact that
   updates nothing becomes a new sentence after its nearest neighbour, and
   a "no recorded activity" line is replaced outright.
2. FIND what is stale: the text a fact names as superseded, and numbers
   that changed ("3 bursts" → "5 bursts": same unit, new value). The new
   material is the part of the fact the sentence does not already say.
3. REWRITE the stale span. With a masked backend, the span becomes a masked
   slot and the host loop fills it conditioned on the fenced facts and the
   neighbouring sentences — the diffusion revision. Without one, the CLASSICAL
   SPLICE writes the fact's own words into the span — the baseline, which
   cannot add anything that is not in the facts.
4. CHECK SUPPORT — and refuse. Everything the revision asserts beyond the
   original sentence must be found in the facts it was revised from:
   content words (lightly stemmed), numbers with their stated precision
   (ported from Bill's ATK `fidelity.py`: `40` asserts 39.5–40.5), times and
   dates, negations. A revision that fails is REFUSED, the reason is given
   in words, and (by default) the splice is used instead. The share of model
   revisions refused is reported as the hallucination rate (plan §7).

Every changed sentence carries provenance: which fact(s) it came from, the
method and its tier — diffusion revisions are INVENTED (made by a generative
model), splices are CLEANED (a deterministic edit that adds nothing, but no
longer the record).

`compose(specialists, backend)` is Athena's Composer as a diffusion model:
the specialists' outputs fenced as context, the synthesis written as ONE
canvas by the loop, each sentence attributed to the specialists it draws on,
unsupported content flagged (or dropped, strict). Its classical comparator,
`compose_extractive`, picks the most central sentences (TF-IDF, maximal
marginal relevance) and adds nothing.

LIMITS. The support check compares words and numbers, not meaning: a
revision that rearranges recorded words into a false sentence passes it.
Stale-span detection is explicit (`replaces`) or arithmetic; contradictions
in wording ("still running" against "finished") are found only by a model —
the result lists the words a masked backend finds hard to reconcile with the
facts, for the analyst to judge. The toy backend lower-cases; its output is
re-cased from the facts. Not built (plan §5): guessing redacted text — a
redaction is never masked, and a fact that targets one is refused; and chat
inline editing ("trivial once F exists; not a track") — the host can call
`revise_in_place` on a chat turn, but no chat feature lives here.
"""

from __future__ import annotations

import collections
import difflib
import math
import re
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Callable

import numpy as np

from atk_diffusion import provenance
from atk_diffusion.text import backends as B
from atk_diffusion.text.loop import HostDiffusionLoop
from atk_diffusion.text.novelty import TfidfIndex

for _m, _t in (("diffusion_revise", "invented"), ("splice_revise", "cleaned"),
               ("diffusion_compose", "invented"),
               ("extractive_compose", "cleaned")):
    provenance.METHOD_TIERS.setdefault(_m, _t)

MEANING = ("a revision of the record's wording from recorded facts; every "
           "changed sentence names the facts it came from")
NO_ACTIVITY = re.compile(r"^\s*no recorded activity\.?\s*$", re.I)
LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# facts
# ---------------------------------------------------------------------------

@dataclass
class Fact:
    """A recorded fact. `replaces` names the text it supersedes (True: the
    whole sentence it updates); `event_id` ties it to a log line's event."""
    text: str
    id: str = ""
    event_id: str | None = None
    replaces: str | bool | None = None


def as_facts(new_facts) -> list[Fact]:
    items = [new_facts] if isinstance(new_facts, (str, dict, Fact)) else \
        list(new_facts or [])
    out = []
    for i, f in enumerate(items):
        if isinstance(f, Fact):
            fact = f
        elif isinstance(f, dict):
            fact = Fact(text=str(f.get("text", "")), id=str(f.get("id", "")),
                        event_id=f.get("event_id"), replaces=f.get("replaces"))
        else:
            fact = Fact(text=str(f))
        fact.text = (fact.text or "").strip()
        if not fact.text:
            raise ValueError(f"fact {i + 1} has no text; a revision can only be "
                             "made from something recorded")
        fact.id = fact.id or f"fact {i + 1}"
        if isinstance(fact.replaces, str) and B.redaction_marks(fact.replaces):
            raise B.RedactionRefused(f"{B.REDACTION_REFUSAL} {fact.id} names a "
                                     "redaction as the text it replaces.")
        out.append(fact)
    return out


# ---------------------------------------------------------------------------
# support: what the revision asserts must be in the facts
# (numbers with stated precision ported from Bill's ATK fidelity.py)
# ---------------------------------------------------------------------------

_NUMBER = re.compile(
    r"(?<![\w.:])(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)(?![\w:])"
    r"\s*(%|°[CF]?|[A-Za-zµ][A-Za-z0-9µ/\-]{0,11})?")
_TIME = re.compile(
    r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AaPp]\.?[Mm]\.?)?(?!\w)"
    r"|\b\d{4}\s?[Zz]\b|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_NOT_A_UNIT = frozenset(B.FUNCTION_WORDS) | {"was", "were", "is", "are"}
MEASURE_UNITS = frozenset({
    "m", "km", "cm", "mm", "mi", "ft", "yd", "nm", "kg", "g", "mg", "lb",
    "lbs", "t", "s", "sec", "ms", "min", "h", "hr", "hrs", "hz", "khz", "mhz",
    "ghz", "db", "dbm", "dbfs", "w", "kw", "mw", "v", "mv", "a", "ma", "%",
    "°", "°c", "°f", "l", "ml", "kt", "kts", "knot", "knots", "mph", "kph"})


def half_ulp(text: str) -> float:
    """Half the last written digit's place value — what a number asserts."""
    clean = text.replace(",", "").strip()
    if "." in clean:
        return 0.5 * (10 ** -len(clean.split(".", 1)[1]))
    return 0.5


@dataclass(frozen=True)
class Quantity:
    text: str
    value: float
    unit: str

    def same_claim(self, other: "Quantity") -> bool:
        # STRICT overlap of the stated-precision intervals. With <=, 40
        # ([39.5, 40.5]) and 41 ([40.5, 41.5]) touch at 40.5 and pass as one
        # claim — the boundary bug in ATK fidelity.py's `overlaps`, not
        # carried over.
        a, b = half_ulp(self.text), half_ulp(other.text)
        if not ((self.value - a) < (other.value + b)
                and (other.value - b) < (self.value + a)):
            return False
        if self.unit == other.unit:
            return True
        # "15 km" is not "15 kg"; "12 started" and "12 finished" are one 12
        return not (self.unit in MEASURE_UNITS and other.unit in MEASURE_UNITS)


def _norm_unit(unit: str) -> str:
    u = (unit or "").strip().lower().rstrip(".,;:")
    if u in _NOT_A_UNIT:
        return ""
    if len(u) > 3 and u.endswith("s") and not u.endswith("ss"):
        u = u[:-1]
    return u


def quantities(text: str) -> list[Quantity]:
    body = _TIME.sub(lambda m: " " * len(m.group(0)), text or "")
    out = []
    for m in _NUMBER.finditer(body):
        try:
            v = float(Decimal(m.group(1).replace(",", "")))
        except (InvalidOperation, ValueError):
            continue
        out.append(Quantity(m.group(1), v, _norm_unit(m.group(2) or "")))
    return out


def times(text: str) -> list[str]:
    return [re.sub(r"\s+", "", m.group(0)).lower()
            for m in _TIME.finditer(text or "")]


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    return w


def _content_stems(text: str) -> set:
    return {_stem(w) for w in B.content_words(text)
            if not any(ch.isdigit() for ch in w)}


@dataclass
class Support:
    supported: bool
    unsupported: list = field(default_factory=list)
    lost: list = field(default_factory=list)
    checked: tuple = ("content words", "numbers and their stated precision",
                      "times and dates", "negations")
    not_checked: tuple = ("meaning", "implication", "word order", "tone")
    reason: str = ""

    def why(self) -> str:
        if self.reason:
            return self.reason
        if self.supported:
            s = "every word, number and time it adds is in the recorded facts"
        else:
            s = ("not in the recorded facts: "
                 + ", ".join(f"'{u}'" for u in self.unsupported[:6]))
        if self.lost:
            s += "; dropped from the entry: " + ", ".join(
                f"'{x}'" for x in self.lost[:6])
        return s

    def as_dict(self) -> dict:
        return {"supported": self.supported, "unsupported": list(self.unsupported),
                "lost": list(self.lost), "checked": list(self.checked),
                "not_checked": list(self.not_checked), "why": self.why()}


def support_check(revised: str, original: str, sources,
                  superseded=()) -> Support:
    """Everything `revised` asserts beyond `original` must be found in
    `sources` (the recorded facts). Text in `superseded` (spans of the
    original a fact replaced) supports nothing any more: writing it back is
    asserting what the record has overtaken. Pure arithmetic over words and
    numbers; negations are content words, so an added "not" must be
    supported too."""
    sources = [sources] if isinstance(sources, str) else list(sources or [])
    kept = original or ""
    for old in superseded or ():
        if old:
            kept = kept.replace(old, " ", 1)
    allowed = kept + "\n" + "\n".join(sources)
    unsupported: list[str] = []
    stems = _content_stems(allowed)
    for w in B.content_words(revised):
        if any(ch.isdigit() for ch in w):
            continue
        if _stem(w) not in stems and w not in unsupported:
            unsupported.append(w)
    aq = quantities(allowed)
    for q in quantities(revised):
        if not any(q.same_claim(a) for a in aq):
            unsupported.append(q.text + (f" {q.unit}" if q.unit else ""))
    at = set(times(allowed))
    for t in times(revised):
        if t not in at and t not in unsupported:
            unsupported.append(t)
    lost = []
    rq = quantities(revised)
    for q in quantities(kept):
        if not any(q.same_claim(r) for r in rq):
            lost.append(q.text + (f" {q.unit}" if q.unit else ""))
    rt = set(times(revised))
    lost += [t for t in times(kept) if t not in rt]
    return Support(not unsupported, unsupported, lost)


# ---------------------------------------------------------------------------
# alignment and the edit
# ---------------------------------------------------------------------------

def _tok_spans(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0).lower(), m.start(), m.end())
            for m in B.TOKEN_RE.finditer(text or "")]


def _new_material(sentence: str, fact: str):
    """(new text from the fact, has_anchor). The new material is the fact's
    words after the first run of ≥2 tokens it shares with the sentence,
    without its closing punctuation."""
    st, ft = _tok_spans(sentence), _tok_spans(fact)
    sm = difflib.SequenceMatcher(a=[t for t, _a, _b in st],
                                 b=[t for t, _a, _b in ft], autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size >= 2]
    end = len(ft)
    while end and ft[end - 1][0] in ".!?;":
        end -= 1
    if not blocks:
        return fact[ft[0][1]:ft[end - 1][2]] if end else "", False
    # covered fact positions: inside any anchor block
    covered = set()
    for b in blocks:
        covered.update(range(b.b, b.b + b.size))
    first = blocks[0].b + blocks[0].size
    pieces = [i for i in range(first, end) if i not in covered]
    if not pieces:
        return "", True
    a, b = pieces[0], pieces[-1]
    return fact[ft[a][1]:ft[b][2]], True


def _stale_spans(sentence: str, fact: Fact) -> list[tuple[int, int, str]]:
    """Char spans of the sentence the fact supersedes: named text, and
    numbers of the same unit whose value changed."""
    spans = []
    if isinstance(fact.replaces, str) and fact.replaces.strip():
        i = sentence.lower().find(fact.replaces.strip().lower())
        if i >= 0:
            spans.append((i, i + len(fact.replaces.strip()),
                          f"{fact.id} names it as replaced"))
    fq = quantities(fact.text)
    body = _TIME.sub(lambda m: " " * len(m.group(0)), sentence)
    for m in _NUMBER.finditer(body):
        unit = _norm_unit(m.group(2) or "")
        if not unit:
            continue
        try:
            v = float(Decimal(m.group(1).replace(",", "")))
        except (InvalidOperation, ValueError):
            continue
        q = Quantity(m.group(1), v, unit)
        same_unit = [f for f in fq if f.unit == unit]
        if same_unit and not any(q.same_claim(f) for f in same_unit):
            if not any(a <= m.start(1) < b for a, b, _w in spans):
                spans.append((m.start(1), m.end(1),
                              f"{fact.id} gives a different {unit} count"))
    if any(B.redaction_marks(sentence[a:b]) for a, b, _w in spans):
        raise B.RedactionRefused(f"{B.REDACTION_REFUSAL} {fact.id} would "
                                 "rewrite a redacted span.")
    spans.sort()
    return spans


def _splice(sentence: str, new: str, stale) -> str:
    """The classical revision: the fact's own words into the stale span, or
    appended as a clause before the closing punctuation."""
    if not new and not stale:
        return sentence
    if stale:
        a, b, _w = stale[0]
        return sentence[:a] + new + sentence[b:]
    m = re.search(r"[.!?]+[\"'”’)\]]*\s*$", sentence)
    body, tail = (sentence[:m.start()], sentence[m.start():]) if m else \
        (sentence, ".")
    body = body.rstrip().rstrip(",;")
    return f"{body}{_joiner(new)}{new}{tail}"


def _joiner(new: str) -> str:
    """How appended material meets the sentence: a clause that continues it
    ("at 40 litres") joins with a space; a new statement with a semicolon."""
    first = B.words(new)[:1]
    return " " if first and first[0] in B.FUNCTION_SET else "; "


def _ensure_period(text: str) -> str:
    t = text.strip()
    return t if re.search(r"[.!?][\"'”’)\]]*$", t) else t + "."


# ---------------------------------------------------------------------------
# the diffusion revision of one span
# ---------------------------------------------------------------------------

def _prompt(facts: list[Fact], neighbours: str) -> str:
    blocks = [B.fence(f.id, f.text, "RECORDED FACT") for f in facts]
    if neighbours:
        blocks.append(B.fence("the entry around the revision", neighbours,
                              "CURRENT ENTRY"))
    return ("Revise the entry in place using ONLY the recorded facts. If the "
            "facts do not say it, leave it out.\n\n" + B.envelope(blocks))


def _diffuse(backend, sentence: str, new: str, stale, facts, neighbours,
             loop_kw) -> tuple[str | None, dict]:
    """Fill the stale span (or an inserted clause) with the host loop.
    (None, info) when the model wrote nothing usable or wrote the
    superseded words back."""
    if stale:
        a, b, _w = stale[0]
        prefix, suffix = sentence[:a], sentence[b:]
    else:
        m = re.search(r"[.!?]+[\"'”’)\]]*\s*$", sentence)
        body, tail = (sentence[:m.start()], sentence[m.start():]) if m else \
            (sentence, ".")
        prefix, suffix = body.rstrip().rstrip(",;") + _joiner(new), tail
    n = max(1, len(backend.encode(new))) if new else 1
    pre, suf = backend.encode(prefix), backend.encode(suffix)
    canvas = pre + [backend.mask_id] * n + suf
    kw = {"steps": max(2, n), "stable_steps": 2, "seed": 0}
    kw.update(loop_kw or {})
    loop = HostDiffusionLoop(backend, **kw)
    res = loop.fill(canvas, backend.encode(_prompt(facts, neighbours)))
    slot = res.tokens[len(pre):len(pre) + n]
    filled = backend.decode([int(t) for t in slot]).strip()
    info = {"forwards": res.forwards, "stopped": res.stopped,
            "notes": list(res.notes), "slot_tokens": n}
    # punctuation the slot shares with its surroundings is the surroundings'
    filled = filled.lstrip(",;: ")
    if suffix[:1] and suffix[:1] in ".!?;,":
        filled = filled.rstrip(".!?;, ")
    if not any(ch.isalnum() for ch in filled):
        return None, info
    if stale:
        old = sentence[stale[0][0]:stale[0][1]]
        if B.words(filled) == B.words(old):
            info["restored_stale"] = True
            return None, info
    filled = B.truecase(filled, [f.text for f in facts] + [sentence])
    if filled and filled[0].isupper() and prefix.strip() and \
            not re.search(r"[.!?]\s*$", prefix):
        # mid-sentence: keep a capital only where the facts write one
        first = filled.split()[0]
        if not any(first in f.text for f in facts):
            filled = filled[0].lower() + filled[1:]
    sep_l = "" if (not prefix or prefix.endswith((" ", "\n"))) else " "
    sep_r = "" if (not suffix or suffix[0] in ".,;:!?)]}\"'” \n") else " "
    revised = prefix + sep_l + filled + sep_r + suffix
    return re.sub(r"[ \t]{2,}", " ", revised), info


def _hard_words(backend, revised: str, facts: list[Fact], rise_bits: float):
    """Words of the revised sentence a masked backend finds much harder to
    predict once the facts are in the context — candidates for contradiction,
    for the analyst to judge."""
    if not B.is_masked(backend):
        return []
    ids = backend.encode(revised)
    if not ids:
        return []
    loop = HostDiffusionLoop(backend, steps=2, seed=0)
    ctx = backend.encode(" ".join(f.text for f in facts))
    a = loop.surprise_field(ids, ctx, passes=4) / LN2
    b = loop.surprise_field(ids, [], passes=4) / LN2
    names = (backend.token_strings(ids) if hasattr(backend, "token_strings")
             else [backend.decode([i]) for i in ids])
    out = []
    for nm, x, y in zip(names, a, b):
        nm = str(nm).strip()
        if x - y >= rise_bits and any(ch.isalnum() for ch in nm) and \
                nm.lower() not in B.FUNCTION_SET and nm not in out:
            out.append(nm)
    return out


# ---------------------------------------------------------------------------
# revise_in_place
# ---------------------------------------------------------------------------

def revise_in_place(document: str, new_facts, backend=None, *,
                    document_events=None, strict: bool = True,
                    fallback: str | None = "splice",
                    align_threshold: float = 0.2, rise_bits: float = 3.0,
                    context_sentences: int = 1, loop_kw: dict | None = None,
                    progress: Callable[[str], None] | None = None) -> dict:
    """Revise `document` in place from `new_facts`.

    document_events  optional: one list of event ids per sentence (an hour
                     log's lines), so a fact with `event_id` lands exactly
    strict           refuse any revision whose added content is not in the
                     facts (the hour log's rule). strict=False keeps an
                     unsupported model revision but flags it
    fallback         "splice": when a model revision is refused, use the
                     classical splice; None: keep the original sentence

    Returns the revised text, a sentence-level diff, provenance per changed
    sentence, what was refused and why, and the hallucination rate (model
    revisions refused for unsupported content / model revisions made)."""
    say = progress or (lambda _m: None)
    t0 = time.perf_counter()
    facts = as_facts(new_facts)
    spans = B.split_sentences(document or "")
    if document_events is not None and len(document_events) != len(spans):
        raise ValueError(f"document_events has {len(document_events)} entries "
                         f"but the document has {len(spans)} sentences; give "
                         "one list of event ids per sentence")
    use_model = B.is_masked(backend)
    method = "diffusion_revise" if use_model else "splice_revise"
    index = TfidfIndex([s for s, _a, _b in spans]) if spans else None
    notes: list[str] = []
    targets: dict[int, list[Fact]] = collections.defaultdict(list)
    adds: dict[int, list[Fact]] = collections.defaultdict(list)
    for f in facts:
        tgt = None
        if f.event_id is not None and document_events is not None:
            for i, evs in enumerate(document_events):
                if f.event_id in (evs or []):
                    tgt = i
                    break
            if tgt is None:
                notes.append(f"{f.id}: no line carries event {f.event_id}; "
                             "aligned by its wording instead")
        if tgt is None and isinstance(f.replaces, str) and f.replaces.strip():
            for i, (s, _a, _b) in enumerate(spans):
                if f.replaces.strip().lower() in s.lower():
                    tgt = i
                    break
            if tgt is None:
                notes.append(f"{f.id}: the text it replaces "
                             f"('{f.replaces.strip()}') is not in the document; "
                             "aligned by its wording instead")
        best_i, best = -1, 0.0
        if tgt is None and index is not None:
            sims = index.query(f.text)
            best_i = int(np.argmax(sims)) if sims.size else -1
            best = float(sims[best_i]) if best_i >= 0 else 0.0
            if best >= align_threshold:
                tgt = best_i
        if tgt is None:
            for i, (s, _a, _b) in enumerate(spans):
                if NO_ACTIVITY.match(s):
                    tgt = i
                    break
        if tgt is None:
            adds[best_i if best > 0 else len(spans) - 1].append(f)
        else:
            targets[tgt].append(f)

    revised_text = {}            # sentence index -> new text
    diff, provenance_rows, refused = [], [], []
    attempts = refusals = invented = 0
    for i, (s, _a, _b) in enumerate(spans):
        fs = targets.get(i, [])
        if not fs:
            continue
        say(f"revising sentence {i + 1} from {', '.join(f.id for f in fs)}")
        current = s
        model_made = False
        whys = []
        superseded: list[str] = []
        for f in fs:
            if f.replaces is True or NO_ACTIVITY.match(current):
                superseded.append(current)
                current = _ensure_period(f.text)
                whys.append(f"{f.id} replaces the whole sentence")
                continue
            new, anchored = _new_material(current, f.text)
            stale = _stale_spans(current, f)
            if not anchored and not stale:
                adds[i].append(f)
                whys.append(f"{f.id} shares no wording with this sentence; "
                            "added after it")
                continue
            if not new and not stale:
                whys.append(f"{f.id} is already stated here")
                continue
            if len(stale) > 1:
                whys.append(f"{f.id}: {len(stale) - 1} more stale span(s) left "
                            "in place for the analyst: " + ", ".join(
                                f"'{current[a:b]}' ({w})"
                                for a, b, w in stale[1:]))
            spliced = _splice(current, new, stale)
            gone = [current[a:b] for a, b, _w in stale[:1]]
            superseded += gone
            if not use_model:
                current = spliced
                whys.append(f"{f.id}: the fact's words were spliced in"
                            + (f" where {stale[0][2]}" if stale else ""))
                continue
            attempts += 1
            # the neighbours, never the sentence itself: its superseded words
            # would pull the model straight back to them
            lo = max(0, i - context_sentences)
            hi = i + 1 + context_sentences
            neighbours = " ".join(t for t, _x, _y in spans[lo:i] + spans[i + 1:hi])
            candidate, info = _diffuse(backend, current, new, stale, [f],
                                       neighbours, loop_kw)
            if candidate is None:
                sup = Support(False, reason=(
                    "the model wrote the superseded words back"
                    if info.get("restored_stale") else
                    "the model left the slot empty"))
                candidate = current
            else:
                sup = support_check(candidate, current, [f.text], gone)
            if sup.supported or (not strict and candidate != current):
                if not sup.supported:
                    invented += 1          # kept, flagged — still invention
                    whys.append(f"{f.id}: KEPT ALTHOUGH UNSUPPORTED (strict is "
                                f"off) — {sup.why()}")
                current = candidate
                model_made = True
                whys.append(f"{f.id}: the model rewrote the "
                            + ("stale span" if stale else "added clause"))
                continue
            refusals += 1
            if sup.unsupported:
                invented += 1
            refused.append({"sentence": i, "fact": f.id, "candidate": candidate,
                            "why": "refused: " + sup.why()})
            if fallback == "splice":
                current = spliced
                whys.append(f"{f.id}: the model's revision was refused "
                            f"({sup.why()}); the splice from the recorded fact "
                            "was used")
            else:
                whys.append(f"{f.id}: the model's revision was refused "
                            f"({sup.why()}); the sentence was left as it was")
        if current != s:
            revised_text[i] = current
            # the sentence's tier is its least-trusted part's
            used_method = "diffusion_revise" if model_made else "splice_revise"
            sup = support_check(current, s, [f.text for f in fs], superseded)
            hard = (_hard_words(backend, current, fs, rise_bits)
                    if use_model else [])
            diff.append({"index": i, "change": "revised", "before": s,
                         "after": current, "facts": [f.id for f in fs],
                         "method": used_method,
                         "tier": provenance.tier_for(used_method),
                         "support": sup.as_dict(), "why": "; ".join(whys),
                         "hard_to_reconcile": hard})
            provenance_rows.append({"text": current, "facts": [f.id for f in fs],
                                    "method": used_method,
                                    "tier": provenance.tier_for(used_method)})
        elif whys:
            diff.append({"index": i, "change": "unchanged", "before": s,
                         "after": s, "facts": [f.id for f in fs],
                         "why": "; ".join(whys)})

    # rebuild: replace changed sentences in place, insert additions
    out, pos = [], 0
    sep = "\n" if "\n" in (document or "") else " "
    inserted = []
    for i, (s, a, b) in enumerate(spans):
        out.append(document[pos:a])
        out.append(revised_text.get(i, s))
        pos = b
        if i in adds:
            for f in adds[i]:
                t = _ensure_period(f.text)
                out.append(sep + t)
                inserted.append((i, f, t))
    out.append((document or "")[pos:])
    if not spans:
        for f in adds.get(-1, []):
            t = _ensure_period(f.text)
            out.append((sep if out and "".join(out).strip() else "") + t)
            inserted.append((-1, f, t))
    for i, f, t in inserted:
        diff.append({"index": i, "change": "added", "before": "", "after": t,
                     "facts": [f.id], "method": "splice_revise",
                     "tier": provenance.tier_for("splice_revise"),
                     "support": support_check(t, "", [f.text]).as_dict(),
                     "why": f"{f.id} updates no sentence; it is added "
                            + ("after sentence %d" % (i + 1) if i >= 0
                               else "as the first sentence")})
        provenance_rows.append({"text": t, "facts": [f.id],
                                "method": "splice_revise",
                                "tier": provenance.tier_for("splice_revise")})
    diff.sort(key=lambda d: (d["index"], d["change"] == "added"))
    rate = (invented / attempts) if attempts else 0.0
    other = refusals - invented
    return {
        "text": "".join(out), "diff": diff, "provenance": provenance_rows,
        "refused": refused, "method": method,
        "tier": provenance.tier_for(method), "meaning": MEANING,
        "backend": B.describe(backend),
        "hallucination_rate": rate,
        "hallucination_note": (
            (f"{invented} of {attempts} model revision(s) added content not "
             "in the facts" + (" and were refused" if strict else
                               " (refused, or kept and flagged: strict is off)")
             + (f"; {other} more wrote nothing usable" if other else ""))
            if attempts else "no model revision was attempted; the splice "
                             "adds nothing that is not in the facts"),
        "seconds": time.perf_counter() - t0,
        "notes": notes + list(getattr(backend, "notes", []) or []),
    }


# ---------------------------------------------------------------------------
# composing: the specialists' outputs as context, one canvas
# ---------------------------------------------------------------------------

def _items(specialists) -> list[tuple[str, str]]:
    if isinstance(specialists, dict):
        items = list(specialists.items())
    else:
        items = [tuple(x) for x in specialists]
    out = [(str(n), str(t)) for n, t in items if str(t or "").strip()]
    if not out:
        raise ValueError("there is nothing to compose: every specialist's "
                         "output is empty")
    return out


def _attribute(sentence: str, items) -> tuple[list[str], list[str]]:
    """(specialists whose words the sentence draws on, unsupported words)."""
    cw = [w for w in B.content_words(sentence)
          if not any(ch.isdigit() for ch in w)]
    stems = {n: _content_stems(t) for n, t in items}
    shares = []
    for n, st in stems.items():
        share = (sum(1 for w in cw if _stem(w) in st) / len(cw)) if cw else 0.0
        shares.append((share, n))
    shares.sort(key=lambda x: -x[0])
    src = [n for sh, n in shares if sh >= 0.5]
    allst = set().union(*stems.values()) if stems else set()
    unsup = [w for w in cw if _stem(w) not in allst]
    sup = support_check(sentence, "", [t for _n, t in items])
    for u in sup.unsupported:
        numeric_or_negation = any(ch.isdigit() for ch in u) or u in B.NEGATIONS
        if numeric_or_negation and u not in unsup:
            unsup.append(u)
    return src, unsup


def compose_extractive(specialists, question: str = "",
                       budget_words: int | None = None,
                       diversity: float = 0.7) -> dict:
    """The classical composer: the most central sentences across the
    specialists' outputs, chosen by maximal marginal relevance. It adds
    nothing that a specialist did not say."""
    items = _items(specialists)
    sents = [(n, s) for n, t in items for s, _a, _b in B.split_sentences(t)]
    if not sents:
        raise ValueError("the specialists' outputs hold no sentences")
    texts = [s for _n, s in sents]
    idx = TfidfIndex(texts)
    sim = np.stack([idx.query(t) for t in texts]) if texts else np.zeros((0, 0))
    rel = sim.mean(axis=1)
    if question.strip():
        rel = rel + idx.query(question)
    if budget_words is None:
        budget_words = int(np.median([len(B.words(t)) for _n, t in items]))
        budget_words = max(12, budget_words)
    chosen: list[int] = []
    words = 0
    while len(chosen) < len(texts) and words < budget_words:
        best, best_v = -1, -np.inf
        for i in range(len(texts)):
            if i in chosen:
                continue
            red = max((sim[i, j] for j in chosen), default=0.0)
            v = diversity * rel[i] - (1 - diversity) * red
            if v > best_v:
                best, best_v = i, v
        chosen.append(best)
        words += len(B.words(texts[best]))
    chosen.sort()
    out = [{"text": texts[i], "from": [sents[i][0]], "unsupported": []}
           for i in chosen]
    return {"text": " ".join(texts[i] for i in chosen), "sentences": out,
            "method": "extractive_compose",
            "tier": provenance.tier_for("extractive_compose"),
            "hallucination_rate": 0.0,
            "meaning": "sentences chosen from the specialists' own words; "
                       "nothing added"}


def compose(specialists, backend=None, *, question: str = "",
            length: int | None = None, strict: bool = False,
            loop_kw: dict | None = None,
            progress: Callable[[str], None] | None = None) -> dict:
    """Athena's Composer as one canvas: the specialists' outputs (a dict or
    (name, text) pairs) fenced as context, the synthesis written by the host
    loop, every sentence attributed and its unsupported words flagged
    (strict: dropped). Without a masked backend, the extractive composer."""
    say = progress or (lambda _m: None)
    t0 = time.perf_counter()
    items = _items(specialists)
    baseline = compose_extractive(items, question)
    if not B.is_masked(backend):
        res = dict(baseline)
        res["notes"] = ["no masked backend: the classical extractive "
                        "synthesis is the result"]
        res["backend"] = B.describe(backend)
        res["seconds"] = time.perf_counter() - t0
        return res
    blocks = [B.fence(n, t, "SPECIALIST") for n, t in items]
    prompt = ((f"The question: {question.strip()}\n\n" if question.strip()
               else "")
              + B.envelope(blocks)
              + "\n\nWrite one synthesis using only the material above. Where "
                "the specialists disagree, say so. If nothing can be said, "
                "write: I don't know.")
    if length is None:
        length = int(np.median([len(backend.encode(t)) for _n, t in items]))
        length = int(min(256, max(16, length)))
    kw = {"steps": length, "stable_steps": 2, "seed": 0}
    kw.update(loop_kw or {})
    say(f"composing {length} tokens from {len(items)} specialist output(s)")
    loop = HostDiffusionLoop(backend, **kw)
    res = loop.generate(backend.encode(prompt), length)
    raw = backend.decode([int(t) for t in res.tokens])
    text = B.truecase(raw, [t for _n, t in items] + [question])
    sentences, kept, total_cw, unsup_cw = [], [], 0, 0
    for s, _a, _b in B.split_sentences(text):
        src, unsup = _attribute(s, items)
        total_cw += max(1, len(B.content_words(s)))
        unsup_cw += len(unsup)
        row = {"text": s, "from": src, "unsupported": unsup}
        if strict and unsup:
            row["dropped"] = True
        else:
            kept.append(s)
        sentences.append(row)
    return {
        "text": " ".join(kept), "sentences": sentences, "baseline": baseline,
        "method": "diffusion_compose",
        "tier": provenance.tier_for("diffusion_compose"),
        "backend": B.describe(backend),
        "hallucination_rate": (unsup_cw / total_cw) if total_cw else 0.0,
        "meaning": ("a synthesis written by a generative model from the "
                    "specialists' outputs; unsupported words are flagged"),
        "forwards": res.forwards, "stopped": res.stopped,
        "drafts": len(res.drafts), "seconds": time.perf_counter() - t0,
        "notes": list(res.notes) + list(getattr(backend, "notes", []) or []),
    }
