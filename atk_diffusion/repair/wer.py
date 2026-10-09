# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Word and character error rate — the number D2 and D5 are judged by
(plan §4.D2: *"word error rate on Bill's transcripts with and without, on
the same clips"*; §6 Phase 3: *"OCR accuracy on degraded scans
before/after"*).

    WER = (S + D + I) / N      S substitutions, D deletions, I insertions,
                               N words in the reference

The counts come from a Levenshtein alignment (unit costs), so a report can
say not just "18 % WER" but "4 substitutions, 1 deletion, 6 insertions" —
and INSERTIONS are the hallucination count: words in the transcript that
the speaker never said. A corpus WER is total errors over total reference
words (not the mean of per-clip rates, which over-weights short clips).

NORMALISATION — what is compared, stated so a number can be reproduced:

  1. Unicode NFKC; then case-folded (lower case).
  2. Curly quotes become straight; hyphens, dashes, slashes and underscores
     become spaces ("push-to-talk" = "push to talk").
  3. Every other character that is not a letter, a digit, a space or an
     apostrophe is removed (punctuation is not scored).
  4. An apostrophe is kept only BETWEEN letters or digits ("don't",
     "o'clock"); quotes around words are dropped.
  5. Runs of whitespace become one space; the ends are stripped.

NOT normalised, on purpose: numbers are not spelled out ("10" ≠ "ten") and
spelling variants are not merged — Whisper writes both forms, and hiding
that would flatter it. Pass `normalize=False` to score raw strings. CER is
computed on the normalised string with its single spaces counted as
characters.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import asdict, dataclass, field

import numpy as np

#: The normalisation in one line, for reports.
NORMALISATION = ("Unicode NFKC, case-folded; curly quotes straightened; "
                 "hyphens, dashes, slashes and underscores become spaces; other "
                 "punctuation removed; apostrophes kept only inside words; "
                 "whitespace collapsed. Numbers are not spelled out.")

_DASHES = "‐‑‒–—―−-/\\_"
_QUOTES = {"‘": "'", "’": "'", "‛": "'", "′": "'",
           "ʼ": "'", "`": "'", "´": "'",
           "“": '"', "”": '"', "„": '"'}
_APOS = re.compile(r"(?<![^\W_])'|'(?![^\W_])")


def normalize(text: str) -> str:
    """The documented normalisation (module docstring, steps 1–5)."""
    t = unicodedata.normalize("NFKC", str(text or "")).casefold()
    for k, v in _QUOTES.items():
        t = t.replace(k, v)
    for d in _DASHES:
        t = t.replace(d, " ")
    t = "".join(ch if (ch.isalnum() or ch.isspace() or ch == "'") else " "
                for ch in t)
    # an apostrophe survives only with a letter or digit on BOTH sides
    t = _APOS.sub(" ", t)
    return " ".join(t.split())


def words(text: str, norm: bool = True) -> list[str]:
    return (normalize(text) if norm else str(text or "")).split()


def chars(text: str, norm: bool = True) -> list[str]:
    t = normalize(text) if norm else " ".join(str(text or "").split())
    return list(t)


@dataclass
class Score:
    """Alignment counts and the rate. `rate` is NaN when the reference is
    empty and the hypothesis is not (an error rate over zero words is
    undefined; the insertions still count — that is the hallucination)."""
    ref_len: int = 0
    hyp_len: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    hits: int = 0
    unit: str = "word"
    ops: list = field(default_factory=list)

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        if self.ref_len == 0:
            return 0.0 if self.hyp_len == 0 else math.nan
        return self.errors / self.ref_len

    def __add__(self, other: "Score") -> "Score":
        return Score(self.ref_len + other.ref_len, self.hyp_len + other.hyp_len,
                     self.substitutions + other.substitutions,
                     self.deletions + other.deletions,
                     self.insertions + other.insertions,
                     self.hits + other.hits, self.unit)

    def to_json(self, with_ops: bool = False) -> dict:
        d = asdict(self)
        if not with_ops:
            d.pop("ops", None)
        d["errors"] = self.errors
        d["rate"] = None if math.isnan(self.rate) else self.rate
        return d

    def words(self) -> str:
        r = self.rate
        name = "WER" if self.unit == "word" else "CER"
        head = (f"{name} undefined (empty reference)" if math.isnan(r)
                else f"{name} {r:.1%}")
        return (f"{head}: {self.substitutions} substituted, {self.deletions} "
                f"deleted, {self.insertions} inserted, of {self.ref_len} "
                f"{self.unit}s")


def align(ref: list, hyp: list, keep_ops: bool = False) -> Score:
    """Levenshtein alignment with S/D/I counts. Each DP row is vectorised:
    the insertion chain C[i][j] = min_k (cand[k] + j − k) is a running
    minimum, so a 5,000-character page costs milliseconds per row. On equal
    cost the diagonal (hit/substitution) beats a deletion, and the latest
    origin of an insertion run wins. `keep_ops` records the alignment (for
    short inputs; it stores the whole table)."""
    n, m = len(ref), len(hyp)
    if n == 0 or m == 0:
        s = Score(n, m, 0, n, m, 0)
        if keep_ops:
            s.ops = [("D", r, None) for r in ref] + [("I", None, h) for h in hyp]
        return s
    vocab: dict = {}
    r_ids = np.array([vocab.setdefault(t, len(vocab)) for t in ref], dtype=np.int64)
    h_ids = np.array([vocab.setdefault(t, len(vocab)) for t in hyp], dtype=np.int64)
    J = np.arange(m + 1)
    cost = J.astype(np.int64).copy()
    S = np.zeros(m + 1, np.int64)
    D = np.zeros(m + 1, np.int64)
    I = J.astype(np.int64).copy()
    back = np.zeros((n + 1, m + 1), np.int8) if keep_ops else None
    org = np.zeros((n + 1, m + 1), np.int32) if keep_ops else None
    if keep_ops:
        back[0, 1:] = 3
        org[0, :] = 0
    for i in range(1, n + 1):
        mis = (h_ids != r_ids[i - 1]).astype(np.int64)
        diag = cost[:-1] + mis
        up = cost[1:] + 1
        use_diag = diag <= up
        cand = np.empty(m + 1, np.int64)
        cand[0] = cost[0] + 1
        cand[1:] = np.where(use_diag, diag, up)
        cS = np.empty(m + 1, np.int64)
        cD = np.empty(m + 1, np.int64)
        cI = np.empty(m + 1, np.int64)
        cS[0], cD[0], cI[0] = S[0], D[0] + 1, I[0]
        cS[1:] = np.where(use_diag, S[:-1] + mis, S[1:])
        cD[1:] = np.where(use_diag, D[:-1], D[1:] + 1)
        cI[1:] = np.where(use_diag, I[:-1], I[1:])
        key = cand - J
        run = np.minimum.accumulate(key)
        origin = np.maximum.accumulate(np.where(key == run, J, -1))
        cost = run + J
        S = cS[origin]
        D = cD[origin]
        I = cI[origin] + (J - origin)
        if keep_ops:
            kind = np.where(use_diag, np.where(mis == 0, 0, 1), 2)
            back[i, 0] = 2
            back[i, 1:] = kind
            org[i] = origin
    sc = Score(n, m, int(S[m]), int(D[m]), int(I[m]),
               int(n - S[m] - D[m]), "word")
    if keep_ops:
        ops = []
        i, j = n, m
        while i > 0 or j > 0:
            if i == 0:
                ops.append(("I", None, hyp[j - 1]))
                j -= 1
                continue
            o = int(org[i, j])
            while j > o:
                ops.append(("I", None, hyp[j - 1]))
                j -= 1
            k = int(back[i, j]) if j > 0 else 2
            if k == 0:
                ops.append(("=", ref[i - 1], hyp[j - 1]))
                i, j = i - 1, j - 1
            elif k == 1:
                ops.append(("S", ref[i - 1], hyp[j - 1]))
                i, j = i - 1, j - 1
            else:
                ops.append(("D", ref[i - 1], None))
                i -= 1
        sc.ops = ops[::-1]
    return sc


def wer(reference: str, hypothesis: str, normalize_text: bool = True,
        keep_ops: bool = False) -> Score:
    s = align(words(reference, normalize_text), words(hypothesis, normalize_text),
              keep_ops)
    s.unit = "word"
    return s


def cer(reference: str, hypothesis: str, normalize_text: bool = True,
        keep_ops: bool = False) -> Score:
    s = align(chars(reference, normalize_text), chars(hypothesis, normalize_text),
              keep_ops)
    s.unit = "character"
    return s


def corpus(pairs, unit: str = "word", normalize_text: bool = True) -> Score:
    """Total counts over (reference, hypothesis) pairs."""
    fn = wer if unit == "word" else cer
    total = Score(unit="word" if unit == "word" else "character")
    for r, h in pairs:
        total = total + fn(r, h, normalize_text)
    total.unit = "word" if unit == "word" else "character"
    return total


def text_of(result) -> str:
    """A host's result as text: a str, a dict with "text", or an object with
    `.text` (ATK's transcribe returns a dict; its OCR an OcrResult)."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return str(result.get("text", "") or "")
    return str(getattr(result, "text", "") or "")
