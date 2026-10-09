# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""F1 — what's new: redundancy is predictability (plan §4.F1).

Bill, 2026-10-08: *"diffusion also has a place in ATK as an analysis step
that catches duplication."* The plan's reading of that instinct: mask a
sentence, let the model fill it from everything already in the project;
perfect reconstruction means redundant, failure means new. Two tools:

* `novelty_rank(documents, project_corpus, backend)` — the novelty filter on
  ingest: rank N documents by what each ADDS to the project;
* `new_facts(document, project_corpus, backend)` — the new-facts highlight:
  the sentences inside one document that resisted reconstruction, each with
  its score and the reason, and character offsets to highlight in place.

HOW A SENTENCE IS SCORED. "Everything already in the project" reaches a
model as context: the project sentences most similar to this one (TF-IDF
retrieval, numpy) plus the document's own earlier sentences — a document
that repeats itself does not add anything by repeating.

* A masked (diffusion) backend re-scores the sentence with its tokens masked
  (the loop's `surprise_field`, context on both sides of every token) and,
  separately, masks a third of the tokens at a time and lets the loop fill
  them: the share it RESTORES is the plain-words reconstruction count. The
  score is the mean surprise in bits per token given the project.
* An autoregressive backend gives log p(sentence | context) and log p
  (sentence); their difference per token is the pointwise mutual
  information — how much the project explains the sentence. The score is
  the surprisal per token given the project, and the PMI is reported as the
  why: "explained by the project" and "predictable anyway" are different
  reasons for the same verdict.

THE LINE between redundant and new is calibrated on the project itself:
sample project sentences, score each once with itself in reach (a perfect
duplicate) and once without (an in-domain sentence the project does not
hold), and draw the line halfway between the two medians. A document's
contribution is the bits above that line, summed over its sentences —
length alone does not make a document new, and a long document of repeats
adds nothing.

CLASSICAL BASELINE, ALWAYS (plan §7): TF-IDF cosine to the nearest earlier
sentence and word-trigram containment, numpy only, scored and ranked beside
every model result. A model that does not beat them on Bill's documents is
not shipped.

EVERY RESULT SAYS: "a model's view of predictability, not a judgement of
truth". A false sentence the project does not contain is "new"; a true one
it already holds is "redundant". Typos and garbled text look new too. Model
scores carry the PROPOSED tier; the classical scores are MEASURED.
"""

from __future__ import annotations

import collections
import math
import time
from typing import Callable

import numpy as np

from atk_diffusion import provenance
from atk_diffusion.text import backends as B
from atk_diffusion.text.loop import HostDiffusionLoop

for _m, _t in (("novelty_masked", "proposed"), ("novelty_pmi", "proposed"),
               ("novelty_tfidf", "measured"),
               ("novelty_containment", "measured")):
    provenance.METHOD_TIERS.setdefault(_m, _t)

MEANING = "a model's view of predictability, not a judgement of truth"
CLASSICAL_MEANING = ("word overlap with what the project already holds — a "
                     "measurement of wording, not of meaning or truth")
LN2 = math.log(2.0)

#: Lines used when the project is too small to calibrate one.
DEFAULT_TAU = {"novelty_masked": 2.0, "novelty_pmi": 2.0,
               "novelty_tfidf": 0.35, "novelty_containment": 0.5}
UNITS = {"novelty_masked": "bits per token", "novelty_pmi": "bits per token",
         "novelty_tfidf": "1 − cosine similarity",
         "novelty_containment": "share of unseen word triples"}
MIN_CALIBRATION = 4


# ---------------------------------------------------------------------------
# ranking agreement (also used by the experiment)
# ---------------------------------------------------------------------------

def _ranks(x) -> np.ndarray:
    """Average ranks (ties share the mean rank), 1-based."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size)
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a, b) -> float:
    """Spearman's rank correlation with ties averaged; NaN when either side
    is constant."""
    ra, rb = _ranks(a), _ranks(b)
    if ra.size < 2 or ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def kendall_tau_b(a, b) -> float:
    """Kendall's tau-b (tie-corrected); NaN when either side is constant."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = a.size
    if n < 2:
        return float("nan")
    da = np.sign(a[:, None] - a[None, :])
    db = np.sign(b[:, None] - b[None, :])
    iu = np.triu_indices(n, 1)
    sa, sb = da[iu], db[iu]
    s = float((sa * sb).sum())
    n0 = sa.size
    ta = float((sa == 0).sum())
    tb = float((sb == 0).sum())
    denom = math.sqrt((n0 - ta) * (n0 - tb))
    return float("nan") if denom == 0 else s / denom


def auc(scores, labels) -> float:
    """Probability a random positive outranks a random negative (ties half)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    pos, neg = s[y], s[~y]
    if not pos.size or not neg.size:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (pos.size * neg.size))


# ---------------------------------------------------------------------------
# classical: TF-IDF retrieval and trigram containment (numpy only)
# ---------------------------------------------------------------------------

def _terms(text: str) -> list[str]:
    return B.content_words(text)


class TfidfIndex:
    """Sublinear TF-IDF over sentences, cosine by an inverted index."""

    def __init__(self, texts: list[str]):
        self.texts = list(texts)
        self.N = len(self.texts)
        toks = [_terms(t) for t in self.texts]
        df: collections.Counter = collections.Counter()
        for ts in toks:
            df.update(set(ts))
        self.idf = {w: math.log((1 + self.N) / (1 + c)) + 1.0
                    for w, c in df.items()}
        self.idf_unseen = math.log(1 + self.N) + 1.0
        post: dict = collections.defaultdict(lambda: ([], []))
        for r, ts in enumerate(toks):
            for w, x in self._vector(ts).items():
                post[w][0].append(r)
                post[w][1].append(x)
        self.post = {w: (np.asarray(rs, dtype=np.int64),
                         np.asarray(vs, dtype=np.float64))
                     for w, (rs, vs) in post.items()}

    def _vector(self, ts: list[str]) -> dict:
        tf = collections.Counter(ts)
        vec = {w: (1.0 + math.log(c)) * self.idf.get(w, self.idf_unseen)
               for w, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        return {w: v / norm for w, v in vec.items()} if norm else {}

    def query(self, text: str) -> np.ndarray:
        """Cosine similarity of `text` to every indexed sentence."""
        scores = np.zeros(self.N)
        for w, x in self._vector(_terms(text)).items():
            hit = self.post.get(w)
            if hit is not None:
                np.add.at(scores, hit[0], x * hit[1])
        return scores


def _grams(text: str, n: int = 3) -> set:
    ws = B.words(text)
    if len(ws) < n:
        return {tuple(ws)} if ws else set()
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def _corpus_sentences(project_corpus) -> list[str]:
    if project_corpus is None:
        return []
    docs = [project_corpus] if isinstance(project_corpus, str) else \
        list(project_corpus)
    out = []
    for d in docs:
        text = d.get("text", "") if isinstance(d, dict) else str(d)
        out.extend(s for s, _a, _b in B.split_sentences(text))
    return out


def _named(documents) -> list[tuple[str, str]]:
    out = []
    for i, d in enumerate(documents):
        if isinstance(d, dict):
            out.append((str(d.get("name", f"document {i + 1}")),
                        str(d.get("text", ""))))
        elif isinstance(d, (tuple, list)) and len(d) == 2:
            out.append((str(d[0]), str(d[1])))
        else:
            out.append((f"document {i + 1}", str(d)))
    return out


def _mode(backend, mode: str) -> str:
    m = (mode or "auto").lower()
    if m == "auto":
        if B.is_masked(backend):
            return "novelty_masked"
        if B.is_scoring(backend):
            return "novelty_pmi"
        return "novelty_tfidf"
    table = {"masked": "novelty_masked", "pmi": "novelty_pmi",
             "classical": "novelty_tfidf", "tfidf": "novelty_tfidf",
             "containment": "novelty_containment"}
    if m not in table:
        raise ValueError(f"unknown novelty mode {mode!r}: use auto, masked, "
                         "pmi, tfidf or containment")
    method = table[m]
    if method == "novelty_masked" and not B.is_masked(backend):
        raise ValueError("masked reconstruction needs a masked (diffusion) "
                         "backend, and this one is not: "
                         + B.describe(backend))
    if method == "novelty_pmi" and not B.is_scoring(backend):
        raise ValueError("PMI needs a backend that can give log p(text | "
                         "context), and this one cannot: "
                         + B.describe(backend))
    return method


# ---------------------------------------------------------------------------
# the scorer: one object per call, holding the pool and the index
# ---------------------------------------------------------------------------

class _Scorer:
    def __init__(self, project: list[str], extra: list[str], backend, method,
                 k_context: int, passes: int | None, rounds: int,
                 loop_kw: dict | None):
        self.project = project
        self.pool = project + extra            # rows: project, then documents
        self.index = TfidfIndex(self.pool)
        self.backend = backend
        self.method = method
        self.k = int(k_context)
        self.passes = passes
        self.rounds = int(rounds)
        self.loop_kw = dict(loop_kw or {})
        self.notes: list[str] = []
        self.loop = None
        if method == "novelty_masked":
            kw = {"steps": 8, "stable_steps": 2, "seed": 0}
            kw.update(self.loop_kw)
            self.loop = HostDiffusionLoop(backend, **kw)
        self._project_grams = set()
        for s in project:
            self._project_grams |= _grams(s)

    # -- retrieval ------------------------------------------------------------
    def nearest(self, text: str, allowed: np.ndarray):
        sims = self.index.query(text)
        sims = np.where(allowed, sims, -np.inf)
        order = np.argsort(-sims, kind="mergesort")
        top = [int(i) for i in order[:max(self.k, 1)] if np.isfinite(sims[i])
               and sims[i] > 0]
        best = (float(sims[order[0]]) if allowed.any() and
                np.isfinite(sims[order[0]]) else 0.0)
        return top, best, (int(order[0]) if allowed.any() else -1)

    def context_text(self, rows: list[int]) -> str:
        return " ".join(self.pool[i] for i in sorted(rows[:self.k]))

    # -- classical --------------------------------------------------------------
    def classical(self, text: str, allowed: np.ndarray, extra_grams: set):
        _top, best, near = self.nearest(text, allowed)
        g = _grams(text)
        seen = (self._project_grams | extra_grams)
        cont = (len(g & seen) / len(g)) if g else 1.0
        return {"novelty_tfidf": max(0.0, 1.0 - max(best, 0.0)),
                "novelty_containment": 1.0 - cont,
                "nearest": self.pool[near] if near >= 0 else "",
                "similarity": max(best, 0.0)}

    # -- model -------------------------------------------------------------------
    def model(self, text: str, ctx: str) -> dict | None:
        if self.method == "novelty_masked":
            return self._masked(text, ctx)
        if self.method == "novelty_pmi":
            return self._pmi(text, ctx)
        return None

    def _token_names(self, ids) -> list[str]:
        f = getattr(self.backend, "token_strings", None)
        if callable(f):
            return list(f(ids))
        return [self.backend.decode([int(i)]) for i in ids]

    def _masked(self, text: str, ctx: str) -> dict | None:
        be = self.backend
        ids = be.encode(text)
        if not ids:
            return None
        cids = be.encode(ctx) if ctx else []
        s_ctx = self.loop.surprise_field(ids, cids, passes=self.passes)
        s_alone = self.loop.surprise_field(ids, [], passes=self.passes)
        hits = total = 0
        if self.rounds > 0:
            arr = np.asarray(ids, dtype=np.int64)
            for r in range(self.rounds):
                pos = np.arange(r, arr.size, self.rounds)
                if not pos.size:
                    continue
                canvas = arr.copy()
                canvas[pos] = be.mask_id
                res = self.loop.fill(canvas, cids)
                hits += int((res.tokens[pos] == arr[pos]).sum())
                total += int(pos.size)
                for n in res.notes:
                    if n not in self.notes:
                        self.notes.append(n)
        names = self._token_names(ids)
        return self._pack(names, s_ctx / LN2, s_alone / LN2, hits, total)

    def _pmi(self, text: str, ctx: str) -> dict | None:
        be = self.backend
        if callable(getattr(be, "scored_tokens", None)):
            a = be.scored_tokens(text, ctx)
            b = be.scored_tokens(text, "")
            if not a:
                return None
            names = [t for t, _v in a]
            lc = np.array([v for _t, v in a])
            la = np.array([v for _t, v in b]) if len(b) == len(a) else \
                np.full(len(a), sum(v for _t, v in b) / max(1, len(b)))
            return self._pack(names, -lc / LN2, -la / LN2, 0, 0)
        n = max(1, len(B.tokenize(text)))
        lc = float(be.logprob(text, ctx))
        la = float(be.logprob(text, ""))
        return self._pack([], np.full(n, -lc / n / LN2),
                          np.full(n, -la / n / LN2), 0, 0)

    @staticmethod
    def _pack(names, bits_ctx, bits_alone, hits, total) -> dict:
        hard = []
        if names:
            order = np.argsort(-bits_ctx, kind="mergesort")
            for i in order:
                nm = str(names[i]).strip()
                if nm and any(ch.isalnum() for ch in nm) and nm not in hard:
                    hard.append(nm)
                if len(hard) == 3:
                    break
        return {"score": float(np.mean(bits_ctx)),
                "bits_alone": float(np.mean(bits_alone)),
                "pmi_bits": float(np.mean(bits_alone) - np.mean(bits_ctx)),
                "tokens": int(len(bits_ctx)),
                "recovered": [int(hits), int(total)] if total else None,
                "hardest": hard}


def _verdict(method: str, info: dict, tau: float) -> tuple[str, str]:
    """(verdict, why) for one sentence in plain words."""
    if method in ("novelty_tfidf", "novelty_containment"):
        nov = info[method]
        near = info.get("nearest", "")
        snip = (near[:117] + "…") if len(near) > 120 else near
        why = (f"nearest earlier sentence {info['similarity']:.2f} cosine"
               + (f" (\"{snip}\")" if snip else "")
               + f"; {1 - info['novelty_containment']:.0%} of its word "
               "triples already seen")
        return ("new" if nov > tau else "redundant"), why
    s, alone, pmi = info["score"], info["bits_alone"], info["pmi_bits"]
    rec = info.get("recovered")
    hard = ", ".join(f"'{h}'" for h in info.get("hardest", []))
    restored = (f"{rec[0]} of {rec[1]} masked tokens restored; " if rec else "")
    if s > tau:
        explain = (f"the project explains {pmi:.1f} bits/token of it"
                   if pmi > 0.05 else "nothing in the project helps predict it")
        lead = ("resisted reconstruction: " if method == "novelty_masked"
                else "not predictable from the project: ")
        return "new", (f"{lead}{restored}{s:.1f} bits/token against the line "
                       f"at {tau:.1f}; " + (f"hardest: {hard}; " if hard else "")
                       + explain)
    if alone <= tau:
        return "redundant", (f"predictable even without the project (generic "
                             f"wording): {restored}{s:.1f} bits/token")
    lead = ("restored from the project: " if method == "novelty_masked"
            else "explained by the project: ")
    return "redundant", (f"{lead}{restored}{s:.1f} bits/token, under the line at "
                         f"{tau:.1f}; the project explains {pmi:.1f} "
                         "bits/token of it")


# ---------------------------------------------------------------------------
# calibration: where redundant ends and new begins
# ---------------------------------------------------------------------------

def _calibrate(sc: _Scorer, method: str, sample: int) -> dict:
    P = len(sc.project)
    if P < MIN_CALIBRATION:
        return {"value": DEFAULT_TAU[method], "unit": UNITS[method],
                "how": f"the default line — the project has {P} sentence(s), "
                       f"fewer than {MIN_CALIBRATION} needed to calibrate one"}
    idx = sorted(set(np.linspace(0, P - 1, min(int(sample), P))
                     .round().astype(int).tolist()))
    selfs, loos = [], []
    base = np.zeros(len(sc.pool), dtype=bool)
    base[:P] = True
    for i in idx:
        text = sc.project[i]
        loo = base.copy()
        loo[i] = False
        if method in ("novelty_tfidf", "novelty_containment"):
            others = set()
            for j, s in enumerate(sc.project):
                if j != i:
                    others |= _grams(s)
            _t, best, _n = sc.nearest(text, loo)
            g = _grams(text)
            if method == "novelty_tfidf":
                selfs.append(0.0)
                loos.append(max(0.0, 1.0 - max(best, 0.0)))
            else:
                selfs.append(0.0)
                loos.append(1.0 - (len(g & others) / len(g) if g else 1.0))
            continue
        top_self, _b, _n = sc.nearest(text, base)
        if i not in top_self:
            top_self = [i] + top_self[:max(0, sc.k - 1)]
        top_loo, _b, _n = sc.nearest(text, loo)
        a = sc.model(text, sc.context_text(top_self))
        b = sc.model(text, sc.context_text(top_loo))
        if a is not None and b is not None:
            selfs.append(a["score"])
            loos.append(b["score"])
    if not selfs:
        return {"value": DEFAULT_TAU[method], "unit": UNITS[method],
                "how": "the default line — no project sentence could be scored"}
    ms, ml = float(np.median(selfs)), float(np.median(loos))
    tau = 0.5 * (ms + ml)
    how = (f"calibrated on {len(selfs)} project sentence(s): a sentence with "
           f"itself in reach scores {ms:.2f}, without it {ml:.2f}; the line is "
           "halfway")
    if ml <= ms:
        how += (" — WARNING: the project's own sentences are no harder without "
                "themselves than with, so this line separates little")
    return {"value": tau, "unit": UNITS[method], "how": how,
            "self_median": ms, "held_out_median": ml, "n": len(selfs)}


# ---------------------------------------------------------------------------
# scoring one document's sentences
# ---------------------------------------------------------------------------

def _score_document(sc: _Scorer, doc_rows: list[int], spans, method, tau,
                    ctau_t, ctau_c, include_earlier: bool) -> list[dict]:
    P = len(sc.project)
    allowed = np.zeros(len(sc.pool), dtype=bool)
    allowed[:P] = True
    extra_grams: set = set()
    out = []
    for j, (text, start, end) in enumerate(spans):
        row = doc_rows[j]
        cl = sc.classical(text, allowed, extra_grams)
        rec = {"index": j, "text": text, "start": start, "end": end,
               "classical": {"tfidf": cl["novelty_tfidf"],
                             "containment": cl["novelty_containment"],
                             "nearest": cl["nearest"],
                             "similarity": cl["similarity"]},
               "words": len(B.words(text))}
        cv_t, _w = _verdict("novelty_tfidf", cl, ctau_t)
        cv_c, _w2 = _verdict("novelty_containment", cl, ctau_c)
        rec["classical"]["verdict_tfidf"] = cv_t
        rec["classical"]["verdict_containment"] = cv_c
        if method in ("novelty_tfidf", "novelty_containment"):
            v, why = _verdict(method, cl, tau)
            rec.update(score=cl[method], verdict=v, why=why,
                       weight=rec["words"])
        else:
            top, _b, _n = sc.nearest(text, allowed)
            info = sc.model(text, sc.context_text(top))
            if info is None:
                rec.update(score=0.0, verdict="redundant",
                           why="nothing to score (no tokens)", weight=0)
            else:
                v, why = _verdict(method, info, tau)
                rec.update(score=info["score"], verdict=v, why=why,
                           bits_alone=info["bits_alone"],
                           pmi_bits=info["pmi_bits"],
                           recovered=info["recovered"],
                           hardest=info["hardest"], weight=info["tokens"])
        out.append(rec)
        if include_earlier:
            allowed[row] = True
            extra_grams |= _grams(text)
    return out


def _added(sentences: list[dict], tau: float) -> float:
    return float(sum(r["weight"] * max(0.0, r["score"] - tau)
                     for r in sentences))


def _classical_added(sentences: list[dict], key: str, tau: float) -> float:
    return float(sum(r["words"] * max(0.0, r["classical"][key] - tau)
                     for r in sentences))


def _setup(project_corpus, extra_texts, backend, mode, k_context, passes,
           rounds, loop_kw):
    method = _mode(backend, mode)
    project = _corpus_sentences(project_corpus)
    sc = _Scorer(project, extra_texts, backend, method, k_context, passes,
                 rounds if method == "novelty_masked" else 0, loop_kw)
    return method, sc


# ---------------------------------------------------------------------------
# the two tools
# ---------------------------------------------------------------------------

def new_facts(document: str, project_corpus, backend=None, *,
              mode: str = "auto", threshold: float | None = None,
              k_context: int = 4, passes: int | None = 4, rounds: int = 3,
              include_earlier: bool = True, calibrate_sample: int = 12,
              loop_kw: dict | None = None,
              progress: Callable[[str], None] | None = None) -> dict:
    """The sentences of `document` that resisted reconstruction from the
    project, with every sentence's score, verdict and reason (and its
    character span, for highlighting in place)."""
    say = progress or (lambda _m: None)
    t0 = time.perf_counter()
    spans = B.split_sentences(document or "")
    texts = [s for s, _a, _b in spans]
    method, sc = _setup(project_corpus, texts, backend, mode, k_context,
                        passes, rounds, loop_kw)
    P = len(sc.project)
    rows = list(range(P, P + len(texts)))
    say(f"calibrating the line between redundant and new on {P} project "
        "sentence(s)")
    cal = (_calibrate(sc, method, calibrate_sample) if threshold is None else
           {"value": float(threshold), "unit": UNITS[method],
            "how": "set by the caller"})
    ct = _calibrate(sc, "novelty_tfidf", calibrate_sample)
    cc = _calibrate(sc, "novelty_containment", calibrate_sample)
    say(f"scoring {len(texts)} sentence(s)")
    sents = _score_document(sc, rows, spans, method, cal["value"], ct["value"],
                            cc["value"], include_earlier)
    new = sorted((s for s in sents if s["verdict"] == "new"),
                 key=lambda s: (-s["score"], s["index"]))
    agree = (float(np.mean([s["verdict"] == s["classical"]["verdict_tfidf"]
                            for s in sents])) if sents else float("nan"))
    notes = list(getattr(backend, "notes", []) or []) + sc.notes
    return {
        "method": method, "tier": provenance.tier_for(method),
        "meaning": MEANING if method in ("novelty_masked", "novelty_pmi")
        else CLASSICAL_MEANING,
        "backend": B.describe(backend), "threshold": cal,
        "sentences": sents, "new": new,
        "classical": {"method": "novelty_tfidf",
                      "tier": provenance.tier_for("novelty_tfidf"),
                      "meaning": CLASSICAL_MEANING, "threshold": ct,
                      "containment_threshold": cc,
                      "new": [s["index"] for s in sents
                              if s["classical"]["verdict_tfidf"] == "new"]},
        "agreement_with_classical": agree,
        "seconds": time.perf_counter() - t0, "notes": notes,
    }


def novelty_rank(documents, project_corpus, backend=None, *,
                 mode: str = "auto", threshold: float | None = None,
                 k_context: int = 4, passes: int | None = 4, rounds: int = 3,
                 include_earlier: bool = True, calibrate_sample: int = 12,
                 loop_kw: dict | None = None,
                 progress: Callable[[str], None] | None = None) -> dict:
    """Rank documents by what each adds to the project (most first).

    `documents`: strings, {"name", "text"} dicts, or (name, text) pairs.
    Each document is judged against the project alone (and its own earlier
    sentences), not against the other incoming documents."""
    say = progress or (lambda _m: None)
    t0 = time.perf_counter()
    named = _named(documents)
    spans_all = [B.split_sentences(text) for _n, text in named]
    extra, rows_all = [], []
    for spans in spans_all:
        rows_all.append(list(range(len(extra), len(extra) + len(spans))))
        extra.extend(s for s, _a, _b in spans)
    method, sc = _setup(project_corpus, extra, backend, mode, k_context,
                        passes, rounds, loop_kw)
    P = len(sc.project)
    say(f"calibrating on {P} project sentence(s)")
    cal = (_calibrate(sc, method, calibrate_sample) if threshold is None else
           {"value": float(threshold), "unit": UNITS[method],
            "how": "set by the caller"})
    ct = _calibrate(sc, "novelty_tfidf", calibrate_sample)
    cc = _calibrate(sc, "novelty_containment", calibrate_sample)
    docs = []
    for d, ((name, _text), spans) in enumerate(zip(named, spans_all)):
        say(f"scoring document {d + 1} of {len(named)}: {name}")
        rows = [P + r for r in rows_all[d]]
        sents = _score_document(sc, rows, spans, method, cal["value"],
                                ct["value"], cc["value"], include_earlier)
        news = sorted((s for s in sents if s["verdict"] == "new"),
                      key=lambda s: (-s["score"], s["index"]))
        docs.append({
            "document": d, "name": name,
            "added": _added(sents, cal["value"]),
            "new_sentences": len(news), "sentences": len(sents),
            "top": [{"index": s["index"], "text": s["text"],
                     "score": s["score"], "why": s["why"]} for s in news[:3]],
            "classical_added": _classical_added(sents, "tfidf", ct["value"]),
            "containment_added": _classical_added(sents, "containment",
                                                  cc["value"]),
            "detail": sents})
    order = sorted(range(len(docs)),
                   key=lambda i: (-docs[i]["added"], -docs[i]["new_sentences"],
                                  i))
    corder = sorted(range(len(docs)),
                    key=lambda i: (-docs[i]["classical_added"], i))
    a = [x["added"] for x in docs]
    c = [x["classical_added"] for x in docs]
    unit = ("bits the project could not account for, above the line"
            if method in ("novelty_masked", "novelty_pmi")
            else "word-weighted novelty above the line")
    return {
        "method": method, "tier": provenance.tier_for(method),
        "meaning": MEANING if method in ("novelty_masked", "novelty_pmi")
        else CLASSICAL_MEANING,
        "backend": B.describe(backend), "threshold": cal, "added_unit": unit,
        "ranking": [docs[i] for i in order],
        "documents": docs,
        "classical": {"method": "novelty_tfidf",
                      "tier": provenance.tier_for("novelty_tfidf"),
                      "meaning": CLASSICAL_MEANING, "threshold": ct,
                      "ranking": [docs[i]["name"] for i in corder]},
        "rank_agreement_with_classical": {"spearman": spearman(a, c),
                                          "kendall_tau_b": kendall_tau_b(a, c)},
        "seconds": time.perf_counter() - t0,
        "notes": list(getattr(backend, "notes", []) or []) + sc.notes,
    }
