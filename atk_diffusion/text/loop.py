# SPDX-License-Identifier: MIT
# Copyright (c) 2026 William R. Duncan
# The host-side diffusion loop shared with Palimpsest (Palimpsest plan §8.6).
# This one file is MIT-licensed, on purpose, so that Palimpsest (an MIT
# project) can use it unchanged; the rest of the ATK Diffusion Toolkit is
# all rights reserved (see the toolkit's LICENSE, section 3).
"""The host-side masked-diffusion decoding loop — a few hundred lines over logits.

WHAT IT IS. A text diffusion model does not write left to right. It starts
from a canvas of mask tokens and, step by step, commits the positions it is
surest of while it reconsiders the rest. Palimpsest's plan §8.6 specifies the
loop in one sentence, and this module is that sentence made executable:

    "start from a masked canvas, forward, commit the lowest-entropy tokens
    under a bound, renoise the rest, self-condition on the last draft, stop
    when the canvas is stable — a few hundred lines over logits … the
    unmasking order is the metacognition signal, recall mid-draft is a pause
    between steps to inject a gist, the drafts are the inner speech."

WHY ON THE HOST. "A device-side loop cannot be interrupted … the loop must be
host-side." Every step returns to Python, so a host can watch it (`on_step`),
pause it and inject a recalled gist (`between_steps`), keep every draft (the
inner speech) and the order positions were committed in (the metacognition
signal), and stop it. A fused device kernel can do none of that. The loop is
written once and shared: ATK's text tools (ATK Diffusion Toolkit plan, track
F) and Palimpsest drive the same code.

THE BACKEND is anything with a mask token id and one method:

    logits(canvas_ids, prompt_ids, prev_draft=None) -> ndarray [len(canvas), vocab]

the model's scores for every canvas position, conditioned on the prompt (and,
for a backend that supports self-conditioning, on the last full draft).
Optional attributes the loop reads when present: `eos_id` (generation stops
at it), `forbidden_ids` (never committed: padding, BOS…), `uses_prev_draft`
(False means the backend ignores `prev_draft`, and the result says so rather
than pretending self-conditioning happened), `sees_own_token` (True for real
transformers, whose logits at an unmasked position can see that position's
own token — remasking then judges a token by its confidence when it was
committed, not by a forward that can see it), and `name`.

ONE STEP of a block, in order:

1. forward: `backend.logits(canvas, prompt, prev_draft)`;
2. per position: log-probabilities (mask and forbidden tokens excluded),
   entropy, the draft token (argmax, or a seeded Gumbel sample when
   temperature > 0) and its probability — the confidence;
3. remask (optional): committed tokens whose confidence fell under a
   threshold go back to the mask, a bounded number per step and per
   position, never a token the host gave or injected;
4. stability: when the full draft has not changed for `stable_steps` steps,
   committing more would not change the model's mind — the rest is
   committed as drafted and the block ends;
5. commit: the masked positions with the LOWEST ENTROPY, as many as the
   commit schedule allows this step (at least one), plus any whose entropy
   is under `entropy_bound`;
6. hooks: `on_step(state)` sees a snapshot; `between_steps(state)` may
   return an injection — tokens written into the canvas (pinned), tokens
   appended to the context (a recalled gist), or a request to stop;
7. the draft becomes `prev_draft` for the next forward (self-conditioning).

When the step budget runs out with positions still masked, one final
forward commits them all. Long outputs are written in BLOCKS of fixed size
(default 256, DiffusionGemma's canvas): each finished block joins the prompt
of the next — autoregressive across blocks, diffusion within.

SURPRISE AS A FIELD. `surprise_field(sequence)` re-scores a finished
sequence by masking its positions (one at a time, or in interleaved groups
for speed) and reading −log p of the true token at each masked position.
Each token is judged with context on BOTH sides, so surprise is a field over
the sequence rather than a stream; it is what the toolkit's novelty, style
and revision tools measure.

DETERMINISM. Greedy decoding is deterministic; sampling (temperature > 0)
draws from `numpy.random.default_rng(seed)` created afresh for every call,
so the same call with the same seed gives the same text. Ties in entropy are
broken by position.

LIMITS, SAID PLAINLY. The loop is only as good as the backend's logits: it
adds no knowledge. Fixed-size canvases cannot change their own length — a
backend that wants shorter output must write its end-of-sequence or padding
token, and `fill` never grows or shrinks a slot. Remasking relies on the
backend's confidences being meaningful; a backend that sees its own token
makes forward-time confidences useless, which is why commit-time confidence
is used for it. Pure Python and numpy; no toolkit imports, so the file can be
lifted into Palimpsest unchanged.

MIT License

Copyright (c) 2026 William R. Duncan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

import numpy as np

__all__ = [
    "LoopError", "DiffusionBackend", "Remask", "Injection", "LoopState",
    "Draft", "Commit", "Event", "LoopResult", "HostDiffusionLoop",
    "linear_schedule", "cosine_schedule", "SCHEDULES", "MAX_SURPRISE_NATS",
    "DEFAULT_BLOCK",
]

#: DiffusionGemma's canvas: 256 tokens per block.
DEFAULT_BLOCK = 256

#: A token the backend gives no probability at all would score +inf; the
#: surprise field reports this ceiling instead (about 72 bits), so a mean
#: over a sentence stays a number.
MAX_SURPRISE_NATS = 50.0


class LoopError(ValueError):
    """The loop cannot run as asked. The message says why, in words."""


@runtime_checkable
class DiffusionBackend(Protocol):
    """What the loop needs from a model: a mask token and per-position logits."""

    mask_id: int

    def logits(self, canvas_ids: Sequence[int], prompt_ids: Sequence[int],
               prev_draft: Sequence[int] | None = None) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# commit schedules: how many positions may be committed at a step
# ---------------------------------------------------------------------------

def linear_schedule(step: int, steps: int, n_masked: int, n_to_fill: int) -> int:
    """Spread what is still masked evenly over the steps that are left."""
    return int(math.ceil(n_masked / max(1, steps - step)))


def cosine_schedule(step: int, steps: int, n_masked: int, n_to_fill: int) -> int:
    """MaskGIT's cosine schedule: few commits early, many late."""
    still = int(math.floor(n_to_fill * math.cos(0.5 * math.pi * (step + 1)
                                                / max(1, steps))))
    return max(1, n_masked - still)


SCHEDULES: dict[str, Callable[[int, int, int, int], int]] = {
    "linear": linear_schedule, "cosine": cosine_schedule}


@dataclass(frozen=True)
class Remask:
    """Renoising committed tokens the model has stopped believing.

    threshold         a committed token whose confidence is under this goes
                      back to the mask
    max_per_step      at most this many per step
    max_per_position  a position is reopened at most this many times, so the
                      loop cannot oscillate forever
    """
    threshold: float = 0.1
    max_per_step: int = 2
    max_per_position: int = 1


@dataclass
class Injection:
    """What `between_steps` may hand back to the loop.

    canvas   {block position: token id} written into the canvas and pinned
             (never remasked) — a gist written into the draft
    context  token ids appended to the conditioning prompt for the rest of
             the generation — a recalled gist the model now reads
    stop     end the generation: what is still masked is committed as
             currently drafted
    note     a few words for the record (why the host did it)
    """
    canvas: dict = field(default_factory=dict)
    context: Sequence[int] = ()
    stop: bool = False
    note: str = ""


@dataclass
class LoopState:
    """A snapshot handed to the hooks. Changing it changes nothing."""
    block: int
    step: int
    steps: int
    canvas: np.ndarray
    masked: np.ndarray
    draft: np.ndarray
    entropy: np.ndarray
    confidence: np.ndarray
    newly_committed: list
    remasked: list
    prompt_len: int
    forwards: int


@dataclass
class Draft:
    """One step's full draft — the inner speech."""
    block: int
    step: int
    tokens: np.ndarray
    masked: np.ndarray          # what was still masked AFTER this step's commits
    mean_entropy: float         # over the positions that were masked at the forward


@dataclass
class Commit:
    """One position committed — the unmasking order is the metacognition signal.

    how: schedule | bound | stable | final | stopped | gist
    """
    block: int
    step: int
    tick: int                   # the forward pass (counted across blocks)
    position: int               # position in the whole output
    token: int
    entropy: float
    confidence: float
    how: str


@dataclass
class Event:
    """Something the loop did besides committing: remask, inject, stop, note."""
    block: int
    step: int
    kind: str
    positions: list
    detail: str = ""


@dataclass
class LoopResult:
    tokens: np.ndarray
    prompt_len: int
    drafts: list
    order: list
    events: list
    forwards: int
    blocks: int
    stopped: str
    notes: list

    def committed_order(self) -> list[int]:
        """Output positions in the order they were (last) committed."""
        last: dict[int, int] = {}
        for i, c in enumerate(self.order):
            last[c.position] = i
        return [p for p, _i in sorted(last.items(), key=lambda t: t[1])]

    def commit_steps(self) -> np.ndarray:
        """Per output position: the step (within its block) at which it was
        last committed; -1 for tokens the caller gave."""
        out = np.full(self.tokens.size, -1, dtype=np.int64)
        for c in self.order:
            if 0 <= c.position < out.size:
                out[c.position] = c.step
        return out

    def confidence_at_commit(self) -> np.ndarray:
        out = np.full(self.tokens.size, np.nan)
        for c in self.order:
            if 0 <= c.position < out.size:
                out[c.position] = c.confidence
        return out

    def hardness(self) -> np.ndarray:
        """Per output position, 0 for the first position committed in its
        block and 1 for the last — how hard the model found it. NaN for
        tokens the caller gave."""
        out = np.full(self.tokens.size, np.nan)
        by_block: dict[int, list[int]] = {}
        last: dict[int, tuple[int, int]] = {}
        for i, c in enumerate(self.order):
            last[c.position] = (c.block, i)
        for pos, (blk, i) in last.items():
            by_block.setdefault(blk, []).append((i, pos))
        for items in by_block.values():
            items.sort()
            m = len(items)
            for rank, (_i, pos) in enumerate(items):
                if 0 <= pos < out.size:
                    out[pos] = rank / (m - 1) if m > 1 else 0.0
        return out

    def inner_speech(self, decode: Callable[[list], str]) -> list[str]:
        """Every draft as text, through the caller's detokenizer."""
        return [decode([int(t) for t in d.tokens]) for d in self.drafts]

    def as_dict(self) -> dict:
        return {
            "tokens": [int(t) for t in self.tokens],
            "prompt_len": int(self.prompt_len),
            "forwards": int(self.forwards), "blocks": int(self.blocks),
            "stopped": self.stopped, "notes": list(self.notes),
            "order": [{"block": c.block, "step": c.step, "tick": c.tick,
                       "position": c.position, "token": c.token,
                       "entropy": _num(c.entropy),
                       "confidence": _num(c.confidence), "how": c.how}
                      for c in self.order],
            "events": [{"block": e.block, "step": e.step, "kind": e.kind,
                        "positions": [int(p) for p in e.positions],
                        "detail": e.detail} for e in self.events],
            "drafts": [{"block": d.block, "step": d.step,
                        "tokens": [int(t) for t in d.tokens],
                        "masked": [bool(m) for m in d.masked],
                        "mean_entropy": _num(d.mean_entropy)}
                       for d in self.drafts],
        }


def _num(x: float):
    x = float(x)
    return None if not math.isfinite(x) else x


class _Record:
    """What one call accumulates across its blocks."""

    def __init__(self):
        self.drafts: list[Draft] = []
        self.order: list[Commit] = []
        self.events: list[Event] = []
        self.notes: list[str] = []
        self.forwards = 0
        self.stopped = ""           # why the whole call ended
        self.block_reason = ""      # why the last block ended
        self.stop_all = False

    def note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------

class HostDiffusionLoop:
    """Masked-diffusion decoding driven from the host, step by step.

        loop = HostDiffusionLoop(backend, steps=32, commit_schedule="linear",
                                 remask=Remask(0.1), self_condition=True, seed=0)
        res = loop.generate(prompt_ids, length=64)    # a fully masked canvas
        res = loop.fill(canvas_with_masks, prompt_ids) # only the masked slots
        bits = loop.surprise_field(token_ids, prompt_ids) / math.log(2)

    steps            forward passes allowed per block (one more may be spent
                     on a final pass when the budget runs out)
    commit_schedule  "linear" | "cosine" | an int (fixed per step) | a
                     callable (step, steps, n_masked, n_to_fill) -> int
    remask           None/False (off), True (defaults), a float threshold,
                     or a Remask
    self_condition   pass the last full draft to the backend
    seed             for sampling (temperature > 0); greedy ignores it
    block_size       canvas size for long outputs (semi-autoregressive)
    temperature      0 = greedy
    stable_steps     stop a block once the draft has not changed for this
                     many steps (None = never stop early)
    entropy_bound    also commit every masked position whose entropy (nats)
                     is at most this (None = schedule only)
    on_step, between_steps   hooks (see the module docstring)
    record_drafts    keep every draft (the inner speech); off saves memory
    """

    def __init__(self, backend, steps: int = 32, commit_schedule="linear",
                 remask=None, self_condition: bool = True, seed: int = 0, *,
                 block_size: int = DEFAULT_BLOCK, temperature: float = 0.0,
                 stable_steps: int | None = 2,
                 entropy_bound: float | None = None,
                 on_step: Callable[[LoopState], Any] | None = None,
                 between_steps: Callable[[LoopState], Any] | None = None,
                 record_drafts: bool = True):
        if not callable(getattr(backend, "logits", None)):
            raise LoopError("the backend has no logits(canvas_ids, prompt_ids, "
                            "prev_draft) method, so the loop has nothing to "
                            "denoise with")
        mask = getattr(backend, "mask_id", None)
        if not isinstance(mask, (int, np.integer)) or int(mask) < 0:
            raise LoopError("the backend declares no mask token id (mask_id), "
                            "so it cannot fill a masked canvas")
        if int(steps) < 1:
            raise LoopError("the loop needs at least one step")
        if int(block_size) < 1:
            raise LoopError("a block must hold at least one token")
        if float(temperature) < 0:
            raise LoopError("temperature cannot be negative")
        if stable_steps is not None and int(stable_steps) < 1:
            raise LoopError("stable_steps must be at least 1, or None to "
                            "never stop early")
        self.backend = backend
        self.mask_id = int(mask)
        self.steps = int(steps)
        self.schedule = self._as_schedule(commit_schedule)
        self.schedule_name = (commit_schedule if isinstance(commit_schedule, str)
                              else ("fixed" if isinstance(commit_schedule, int)
                                    else "custom"))
        self.remask = self._as_remask(remask)
        self.self_condition = bool(self_condition)
        self.seed = int(seed)
        self.block_size = int(block_size)
        self.temperature = float(temperature)
        self.stable_steps = None if stable_steps is None else int(stable_steps)
        self.entropy_bound = (None if entropy_bound is None
                              else float(entropy_bound))
        self.on_step = on_step
        self.between_steps = between_steps
        self.record_drafts = bool(record_drafts)
        eos = getattr(backend, "eos_id", None)
        self.eos_id = int(eos) if isinstance(eos, (int, np.integer)) else None
        self.forbidden = tuple(int(t) for t in
                               (getattr(backend, "forbidden_ids", ()) or ()))
        self.sees_own_token = bool(getattr(backend, "sees_own_token", True))
        self.uses_prev_draft = bool(getattr(backend, "uses_prev_draft", True))
        self.last_forwards = 0

    # -- configuration helpers ----------------------------------------------
    @staticmethod
    def _as_schedule(s):
        if callable(s):
            return s
        if isinstance(s, bool):
            raise LoopError("commit_schedule must be 'linear', 'cosine', a "
                            "number per step, or a function")
        if isinstance(s, (int, np.integer)):
            k = int(s)
            if k < 1:
                raise LoopError("a fixed commit schedule must commit at least "
                                "one token per step")
            return lambda step, steps, n_masked, n_to_fill: k
        if isinstance(s, str) and s.lower() in SCHEDULES:
            return SCHEDULES[s.lower()]
        raise LoopError(f"unknown commit schedule {s!r}: use 'linear', "
                        "'cosine', a number per step, or a function")

    @staticmethod
    def _as_remask(r):
        if r is None or r is False:
            return None
        if r is True:
            return Remask()
        if isinstance(r, Remask):
            return r
        if isinstance(r, (int, float)) and not isinstance(r, bool):
            if not 0.0 < float(r) < 1.0:
                raise LoopError("a remask threshold is a probability between "
                                "0 and 1")
            return Remask(threshold=float(r))
        raise LoopError("remask must be None, True, a threshold between 0 and "
                        "1, or a Remask")

    def describe(self) -> str:
        """The loop's settings in one sentence."""
        name = getattr(self.backend, "name", type(self.backend).__name__)
        bits = [f"{self.steps} steps per {self.block_size}-token block",
                f"{self.schedule_name} commit schedule"]
        if self.entropy_bound is not None:
            bits.append(f"plus every position under {self.entropy_bound:g} "
                        "nats of entropy")
        bits.append("remasking under p=%g" % self.remask.threshold
                    if self.remask else "no remasking")
        bits.append("self-conditioned on the last draft" if self.self_condition
                    else "no self-conditioning")
        bits.append("greedy" if self.temperature == 0 else
                    f"sampled at temperature {self.temperature:g} (seed "
                    f"{self.seed})")
        if self.stable_steps:
            bits.append(f"a block ends once its draft is stable for "
                        f"{self.stable_steps} step(s)")
        return f"Host diffusion loop over {name}: " + ", ".join(bits) + "."

    # -- the numerics -----------------------------------------------------------
    def _forward(self, canvas: np.ndarray, prompt: list, prev_draft, rec) -> np.ndarray:
        prev = None
        if prev_draft is not None and self.self_condition:
            prev = [int(t) for t in prev_draft]
        out = self.backend.logits([int(t) for t in canvas],
                                  [int(t) for t in prompt], prev)
        rec.forwards += 1
        lg = np.array(out, dtype=np.float64, copy=True)
        if lg.ndim != 2 or lg.shape[0] != canvas.size:
            raise LoopError(f"the backend returned logits of shape "
                            f"{tuple(lg.shape)} for a canvas of {canvas.size} "
                            "tokens; one row per canvas position is required")
        if not 0 <= self.mask_id < lg.shape[1]:
            raise LoopError(f"the backend's vocabulary ({lg.shape[1]} tokens) "
                            f"does not contain its own mask token "
                            f"({self.mask_id})")
        if np.isnan(lg).any() or np.isposinf(lg).any():
            raise LoopError("the backend returned NaN or +infinite logits")
        return lg

    def _log_probs(self, lg: np.ndarray, forbid: bool = True) -> np.ndarray:
        lg = lg.copy()
        lg[:, self.mask_id] = -np.inf
        if forbid:
            for t in self.forbidden:
                if 0 <= t < lg.shape[1]:
                    lg[:, t] = -np.inf
        m = lg.max(axis=1, keepdims=True)
        if not np.all(np.isfinite(m)):
            raise LoopError("at some canvas position every token was either "
                            "forbidden or given no score at all")
        z = lg - m
        return z - np.log(np.exp(z).sum(axis=1, keepdims=True))

    @staticmethod
    def _entropy(logp: np.ndarray) -> np.ndarray:
        p = np.exp(logp)
        safe = np.where(np.isfinite(logp), logp, 0.0)
        return -(p * safe).sum(axis=1)

    def _draft_tokens(self, logp: np.ndarray, rng) -> np.ndarray:
        if self.temperature == 0.0:
            return np.argmax(logp, axis=1).astype(np.int64)
        g = rng.gumbel(size=logp.shape)
        return np.argmax(logp / self.temperature + g, axis=1).astype(np.int64)

    # -- one block --------------------------------------------------------------
    def _run_block(self, canvas: np.ndarray, prompt: list, block: int,
                   offset: int, rng, rec: _Record, on_step, between) -> np.ndarray:
        mask = self.mask_id
        canvas = np.array(canvas, dtype=np.int64, copy=True)
        n = canvas.size
        masked = canvas == mask
        pinned = ~masked
        commit_step = np.full(n, -1, dtype=np.int64)
        commit_conf = np.ones(n)
        remask_count = np.zeros(n, dtype=np.int64)
        n_to_fill = int(masked.sum())
        prev_draft = None
        stable = 0
        quiet = True        # nothing but commits happened since the last forward
        step = 0
        idx_all = np.arange(n)

        def commit(positions, how, ent, conf, draft):
            for p in positions:
                p = int(p)
                canvas[p] = draft[p]
                masked[p] = False
                commit_step[p] = step
                commit_conf[p] = float(conf[p])
                rec.order.append(Commit(block, step, rec.forwards, offset + p,
                                        int(draft[p]), float(ent[p]),
                                        float(conf[p]), how))

        if self.self_condition and not self.uses_prev_draft and n_to_fill:
            rec.note("self-conditioning was asked for, but this backend does "
                     "not read the last draft; the loop ran without it")

        rec.block_reason = ("nothing was masked, so nothing was filled"
                            if not n_to_fill else "")
        while masked.any():
            if step >= self.steps:
                # The budget is spent: one last forward, everything committed.
                lg = self._forward(canvas, prompt, prev_draft, rec)
                logp = self._log_probs(lg)
                ent = self._entropy(logp)
                draft = np.where(masked, self._draft_tokens(logp, rng), canvas)
                conf = np.exp(logp[idx_all, draft])
                left = np.flatnonzero(masked)
                order = left[np.lexsort((left, ent[left]))]
                commit(order, "final", ent, conf, draft)
                rec.events.append(Event(block, step, "final",
                                        [offset + int(p) for p in order],
                                        f"the {self.steps}-step budget ran out; "
                                        f"{order.size} position(s) committed "
                                        "from a final pass"))
                self._record_draft(rec, block, step, draft, masked, ent, order)
                if on_step is not None:
                    on_step(LoopState(block, step, self.steps, canvas.copy(),
                                      masked.copy(), draft.copy(), ent.copy(),
                                      conf.copy(), [offset + int(p) for p in order],
                                      [], len(prompt), rec.forwards))
                rec.block_reason = ("the step budget ran out; what was still "
                                    "masked was committed from a final pass")
                break

            lg = self._forward(canvas, prompt, prev_draft, rec)
            logp = self._log_probs(lg)
            ent = self._entropy(logp)
            drafted = self._draft_tokens(logp, rng)
            draft = np.where(masked, drafted, canvas)
            conf = np.exp(logp[idx_all, draft])
            masked_at_forward = masked.copy()

            # -- renoise: committed tokens the model no longer believes ------
            remasked = self._remask_positions(conf, commit_conf, commit_step,
                                              pinned, masked, remask_count, step)
            if remasked:
                for p in remasked:
                    canvas[p] = mask
                    masked[p] = True
                    remask_count[p] += 1
                    commit_step[p] = -2
                    draft[p] = drafted[p]
                rec.events.append(Event(block, step, "remask",
                                        [offset + p for p in remasked],
                                        "committed token(s) fell under "
                                        f"p={self.remask.threshold:g} and "
                                        "went back to the mask"))

            # -- stability ---------------------------------------------------
            if (prev_draft is not None and quiet and not remasked
                    and np.array_equal(draft, prev_draft)):
                stable += 1
            else:
                stable = 0
            quiet = True

            newly: list[int] = []
            ended_stable = bool(self.stable_steps) and stable >= self.stable_steps
            if ended_stable:
                left = np.flatnonzero(masked)
                order = left[np.lexsort((left, ent[left]))]
                commit(order, "stable", ent, conf, draft)
                newly = order.tolist()
                if newly:
                    rec.events.append(Event(
                        block, step, "stable", [offset + p for p in newly],
                        f"the draft had not changed for {stable} step(s); the "
                        "rest was committed as drafted"))
            else:
                cand = np.flatnonzero(masked & (commit_step != -2))
                if cand.size:
                    k = int(self.schedule(step, self.steps, int(masked.sum()),
                                          n_to_fill))
                    k = max(1, min(k, cand.size))
                    order = cand[np.lexsort((cand, ent[cand]))]
                    chosen = order[:k].tolist()
                    commit(chosen, "schedule", ent, conf, draft)
                    newly = list(chosen)
                    if self.entropy_bound is not None:
                        extra = [int(p) for p in order[k:]
                                 if ent[p] <= self.entropy_bound]
                        commit(extra, "bound", ent, conf, draft)
                        newly += extra
            # a reopened position waits one step before it can be recommitted
            commit_step[commit_step == -2] = -3

            self._record_draft(rec, block, step, draft, masked,
                               ent, np.flatnonzero(masked_at_forward))
            state = LoopState(block, step, self.steps, canvas.copy(),
                              masked.copy(), draft.copy(), ent.copy(),
                              conf.copy(), [offset + p for p in newly],
                              [offset + p for p in remasked], len(prompt),
                              rec.forwards)
            if on_step is not None:
                on_step(state)
            if ended_stable:
                rec.block_reason = ("the draft was stable, so the rest was "
                                    "committed as drafted")
                break
            if between is not None and masked.any():
                inj = self._as_injection(between(state))
                if inj is not None:
                    quiet = False
                    self._apply_injection(inj, canvas, masked, pinned,
                                          commit_step, prompt, block, step,
                                          offset, rec)
                    if inj.stop:
                        left = np.flatnonzero(masked)
                        order = left[np.lexsort((left, ent[left]))]
                        commit(order, "stopped", ent, conf, draft)
                        rec.events.append(Event(
                            block, step, "stop",
                            [offset + int(p) for p in order],
                            inj.note or "the host asked the loop to stop"))
                        rec.block_reason = rec.stopped = (
                            "the host asked to stop between steps"
                            + (f": {inj.note}" if inj.note else ""))
                        rec.stop_all = True
                        break
            prev_draft = draft
            step += 1

        if n_to_fill and not rec.block_reason:
            rec.block_reason = "every masked position was committed"
        return canvas

    def _record_draft(self, rec, block, step, draft, masked, ent, at_forward):
        if not self.record_drafts:
            return
        at = np.asarray(at_forward, dtype=np.int64)
        mean_ent = float(ent[at].mean()) if at.size else 0.0
        rec.drafts.append(Draft(block, step, draft.copy(), masked.copy(),
                                mean_ent))

    def _remask_positions(self, conf, commit_conf, commit_step, pinned, masked,
                          remask_count, step) -> list[int]:
        rm = self.remask
        if rm is None:
            return []
        eligible = ((~masked) & (~pinned) & (commit_step >= 0)
                    & (commit_step < step) & (remask_count < rm.max_per_position))
        basis = commit_conf if self.sees_own_token else conf
        low = np.flatnonzero(eligible & (basis < rm.threshold))
        if low.size == 0:
            return []
        order = low[np.lexsort((low, basis[low]))]
        return [int(p) for p in order[: max(0, int(rm.max_per_step))]]

    @staticmethod
    def _as_injection(value) -> Injection | None:
        if value is None:
            return None
        if isinstance(value, Injection):
            return value
        if isinstance(value, dict):
            return Injection(canvas=dict(value))
        raise LoopError("between_steps returned something the loop cannot use: "
                        "return None, a dict {position: token id}, or an "
                        "Injection")

    def _apply_injection(self, inj: Injection, canvas, masked, pinned,
                         commit_step, prompt: list, block, step, offset, rec):
        written = []
        for pos, tok in (inj.canvas or {}).items():
            p, t = int(pos), int(tok)
            if not 0 <= p < canvas.size:
                raise LoopError(f"an injection named canvas position {p}, but "
                                f"this block holds {canvas.size} positions")
            if t < 0 or t == self.mask_id:
                raise LoopError("an injection cannot write the mask token or a "
                                "negative token id")
            canvas[p] = t
            masked[p] = False
            pinned[p] = True
            commit_step[p] = step
            rec.order.append(Commit(block, step, rec.forwards, offset + p, t,
                                    float("nan"), float("nan"), "gist"))
            written.append(offset + p)
        if written:
            rec.events.append(Event(block, step, "inject-canvas", written,
                                    inj.note or "the host wrote a gist into "
                                    "the canvas"))
        ctx = [int(t) for t in (inj.context or ())]
        if ctx:
            if any(t < 0 or t == self.mask_id for t in ctx):
                raise LoopError("an injected context cannot hold the mask token "
                                "or a negative token id")
            prompt.extend(ctx)
            rec.events.append(Event(block, step, "inject-context", [],
                                    (inj.note or "the host appended a recalled "
                                     "gist to the context")
                                    + f" ({len(ctx)} tokens)"))

    # -- the public calls ---------------------------------------------------------
    def generate(self, prompt_ids: Sequence[int] = (), length: int = 64, *,
                 on_step=None, between_steps=None) -> LoopResult:
        """Write `length` tokens after the prompt, block by block, each block
        starting fully masked. Stops early at the backend's end-of-sequence
        token, or when a hook asks it to."""
        length = int(length)
        if length < 0:
            raise LoopError("cannot generate a negative number of tokens")
        rng = np.random.default_rng(self.seed)
        rec = _Record()
        running = [int(t) for t in prompt_ids]
        prompt_len = len(running)
        out: list[int] = []
        blocks = 0
        on_step = on_step or self.on_step
        between = between_steps or self.between_steps
        while len(out) < length and not rec.stop_all:
            size = min(self.block_size, length - len(out))
            canvas = np.full(size, self.mask_id, dtype=np.int64)
            toks = self._run_block(canvas, running, blocks, len(out), rng, rec,
                                   on_step, between)
            blocks += 1
            toks = [int(t) for t in toks]
            if self.eos_id is not None and self.eos_id in toks:
                cut = toks.index(self.eos_id)
                out.extend(toks[:cut])
                rec.stopped = "the model ended the text (end-of-sequence token)"
                break
            out.extend(toks)
            running.extend(toks)
        if length == 0:
            rec.stopped = "nothing was asked for"
        rec.stopped = rec.stopped or rec.block_reason
        self.last_forwards = rec.forwards
        return LoopResult(np.asarray(out, dtype=np.int64), prompt_len,
                          rec.drafts, rec.order, rec.events, rec.forwards,
                          blocks, rec.stopped, rec.notes)

    def fill(self, canvas_ids: Sequence[int], prompt_ids: Sequence[int] = (), *,
             on_step=None, between_steps=None) -> LoopResult:
        """Fill only the masked positions of a canvas; every other token is
        given and stays (it is never remasked). A canvas longer than one
        block is filled block by block, each block seeing the blocks before
        it as context."""
        canvas = np.asarray([int(t) for t in canvas_ids], dtype=np.int64)
        rng = np.random.default_rng(self.seed)
        rec = _Record()
        running = [int(t) for t in prompt_ids]
        prompt_len = len(running)
        out: list[int] = []
        blocks = 0
        on_step = on_step or self.on_step
        between = between_steps or self.between_steps
        for start in range(0, canvas.size, self.block_size):
            part = canvas[start:start + self.block_size]
            if rec.stop_all:
                # the host stopped the loop: later blocks keep their masks
                # out of the text — say so instead of leaving holes silently
                left = int((part == self.mask_id).sum())
                if left:
                    rec.note(f"{left} masked position(s) in later blocks were "
                             "not filled because the host stopped the loop")
                out.extend(int(t) for t in part)
                continue
            toks = self._run_block(part, running, blocks, start, rng, rec,
                                   on_step, between)
            blocks += 1
            out.extend(int(t) for t in toks)
            running.extend(int(t) for t in toks)
        if not (canvas == self.mask_id).any():
            rec.stopped = "nothing was masked, so nothing was filled"
        rec.stopped = rec.stopped or rec.block_reason
        self.last_forwards = rec.forwards
        return LoopResult(np.asarray(out, dtype=np.int64), prompt_len,
                          rec.drafts, rec.order, rec.events, rec.forwards,
                          blocks, rec.stopped, rec.notes)

    def surprise_field(self, sequence: Sequence[int],
                       prompt_ids: Sequence[int] = (), *,
                       passes: int | None = None) -> np.ndarray:
        """Per-token surprise, −log p in nats, under masked re-scoring.

        passes=None   each token is masked alone (one forward per token): the
                      exact pseudo-log-likelihood
        passes=k      tokens k apart are masked together (k forwards per
                      block): each masked token still sees the k−1 tokens
                      on either side of it

        A sequence longer than one block is scored block by block, each block
        seeing the prompt and the sequence before it. A token the backend
        gives no probability at all scores MAX_SURPRISE_NATS.
        """
        seq = np.asarray([int(t) for t in sequence], dtype=np.int64)
        out = np.zeros(seq.size)
        if seq.size == 0:
            return out
        if (seq == self.mask_id).any():
            raise LoopError("the sequence to score contains the mask token; "
                            "score finished text only")
        rec = _Record()
        base = [int(t) for t in prompt_ids]
        for start in range(0, seq.size, self.block_size):
            win = seq[start:start + self.block_size]
            ctx = base + [int(t) for t in seq[:start]]
            m = win.size
            k = m if passes is None else max(1, min(int(passes), m))
            for r in range(k):
                idx = np.arange(r, m, k)
                canvas = win.copy()
                canvas[idx] = self.mask_id
                logp = self._log_probs(self._forward(canvas, ctx, None, rec),
                                       forbid=False)
                s = -logp[idx, win[idx]]
                out[start + idx] = np.minimum(np.where(np.isfinite(s), s,
                                                       MAX_SURPRISE_NATS),
                                              MAX_SURPRISE_NATS)
        self.last_forwards = rec.forwards
        return out
