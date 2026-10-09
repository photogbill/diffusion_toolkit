# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The host-side diffusion loop (Palimpsest §8.6; plan F): commit order,
schedules, remasking, self-conditioning, hooks, blocks, records, surprise."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest

from atk_diffusion.text.loop import (HostDiffusionLoop, Injection, LoopError,
                                     MAX_SURPRISE_NATS, Remask, cosine_schedule)

LOOP_FILE = Path(__file__).resolve().parents[1] / "atk_diffusion" / "text" / "loop.py"
V = 10


class Table:
    """Position i prefers target[i] with logit sharp[i]; everything else 0.
    Records every call."""
    mask_id = 0
    name = "table"
    sees_own_token = False

    def __init__(self, target, sharp, eos_id=None):
        self.target, self.sharp = list(target), list(sharp)
        self.calls = []
        if eos_id is not None:
            self.eos_id = eos_id

    def logits(self, canvas, prompt, prev_draft=None):
        self.calls.append({"canvas": list(canvas), "prompt": list(prompt),
                           "prev": None if prev_draft is None else list(prev_draft)})
        out = np.zeros((len(canvas), V))
        for i in range(len(canvas)):
            out[i, self.target[i % len(self.target)]] = \
                self.sharp[i % len(self.sharp)]
        return out


def test_loop_file_is_mit_and_liftable_into_palimpsest():
    src = LOOP_FILE.read_text(encoding="utf-8")
    lines = src.splitlines()
    assert lines[0] == "# SPDX-License-Identifier: MIT"
    assert lines[1] == "# Copyright (c) 2026 William R. Duncan"
    assert "Palimpsest" in lines[2]
    assert "Permission is hereby granted, free of charge" in src
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in src
    mods = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
    assert mods <= {"__future__", "math", "dataclasses", "typing", "numpy"}, mods


def test_commits_lowest_entropy_first_one_per_step():
    be = Table([3, 4, 5, 6, 7, 8], [5, 1, 4, 2, 6, 3])
    res = HostDiffusionLoop(be, steps=6, commit_schedule=1,
                            stable_steps=None).generate([1, 2], 6)
    assert res.tokens.tolist() == [3, 4, 5, 6, 7, 8]
    # sharper logit = lower entropy = committed earlier
    assert res.committed_order() == [4, 0, 2, 5, 3, 1]
    assert [c.how for c in res.order] == ["schedule"] * 6
    assert res.forwards == 6 and res.blocks == 1
    ent = [c.entropy for c in res.order]
    assert ent == sorted(ent)
    assert res.hardness()[4] == 0.0 and res.hardness()[1] == 1.0
    assert res.commit_steps().tolist() == [1, 5, 2, 4, 0, 3]


def test_schedules_bound_each_step():
    be = Table(list(range(1, 9)), [1.0] * 8)
    seen = []
    HostDiffusionLoop(be, steps=4, commit_schedule="linear", stable_steps=None,
                      on_step=lambda s: seen.append(len(s.newly_committed))
                      ).generate([], 8)
    assert seen == [2, 2, 2, 2]
    seen.clear()
    HostDiffusionLoop(be, steps=4, commit_schedule="cosine", stable_steps=None,
                      on_step=lambda s: seen.append(len(s.newly_committed))
                      ).generate([], 8)
    expect, left = [], 8
    for st in range(4):
        k = max(1, min(cosine_schedule(st, 4, left, 8), left))
        expect.append(k)
        left -= k
    assert seen == expect and sum(seen) == 8
    assert seen[0] < seen[-1]          # few early, many late


def test_entropy_bound_commits_every_confident_position_at_once():
    be = Table([3, 4, 5, 6], [9, 9, 9, 0.1])
    res = HostDiffusionLoop(be, steps=8, commit_schedule=1, entropy_bound=0.05,
                            stable_steps=None).generate([], 4)
    first = [c for c in res.order if c.step == 0]
    assert {c.position for c in first} == {0, 1, 2}
    assert {c.how for c in first} == {"schedule", "bound"}


def test_a_stable_draft_ends_the_block_early():
    be = Table([3, 4, 5, 6, 7], [2, 2, 2, 2, 2])
    res = HostDiffusionLoop(be, steps=50, commit_schedule=1,
                            stable_steps=2).generate([], 5)
    assert res.tokens.tolist() == [3, 4, 5, 6, 7]
    assert res.forwards == 3
    assert "stable" in res.stopped
    assert any(c.how == "stable" for c in res.order)


def test_a_final_pass_commits_what_the_budget_left():
    be = Table([3, 4, 5, 6, 7], [5, 4, 3, 2, 1])
    res = HostDiffusionLoop(be, steps=2, commit_schedule=1,
                            stable_steps=None).generate([], 5)
    assert res.tokens.tolist() == [3, 4, 5, 6, 7]
    assert res.forwards == 3
    assert [c.how for c in res.order].count("final") == 3
    assert "budget" in res.stopped
    assert any(e.kind == "final" for e in res.events)


class ChangingMind:
    """Position 0 believes 3 until position 1 is committed, then 4."""
    mask_id = 0
    sees_own_token = False

    def logits(self, canvas, prompt, prev_draft=None):
        out = np.zeros((len(canvas), V))
        if canvas[1] == self.mask_id:
            out[0, 3] = 8.0
        else:
            out[0, 4], out[0, 3] = 8.0, -8.0
        out[1, 5] = 4.0
        out[2, 6] = 2.0
        return out


def test_remasking_reopens_a_token_the_model_stops_believing():
    plain = HostDiffusionLoop(ChangingMind(), steps=6, commit_schedule=1,
                              stable_steps=None).generate([], 3)
    assert plain.tokens.tolist() == [3, 5, 6]
    res = HostDiffusionLoop(ChangingMind(), steps=6, commit_schedule=1,
                            remask=Remask(threshold=0.1), stable_steps=None
                            ).generate([], 3)
    assert res.tokens.tolist() == [4, 5, 6]
    rm = [e for e in res.events if e.kind == "remask"]
    assert rm and rm[0].positions == [0]
    assert [c.position for c in res.order].count(0) == 2


def test_remask_never_reopens_a_given_or_injected_token():
    res = HostDiffusionLoop(ChangingMind(), steps=6, commit_schedule=1,
                            remask=0.1, stable_steps=None
                            ).fill([3, 0, 0], [])
    assert res.tokens.tolist()[0] == 3          # given, so pinned
    assert not any(e.kind == "remask" for e in res.events)


def test_self_conditioning_hands_the_backend_the_last_draft():
    be = Table([3, 4, 5, 6], [4, 3, 2, 1])
    drafts = []
    HostDiffusionLoop(be, steps=4, commit_schedule=1, stable_steps=None,
                      on_step=lambda s: drafts.append(s.draft.tolist())
                      ).generate([1], 4)
    assert be.calls[0]["prev"] is None
    assert [c["prev"] for c in be.calls[1:]] == drafts[:-1]
    be2 = Table([3, 4, 5, 6], [4, 3, 2, 1])
    HostDiffusionLoop(be2, steps=4, commit_schedule=1, stable_steps=None,
                      self_condition=False).generate([1], 4)
    assert all(c["prev"] is None for c in be2.calls)


def test_a_backend_that_ignores_the_draft_is_said_to():
    be = Table([3, 4], [2, 1])
    be.uses_prev_draft = False
    res = HostDiffusionLoop(be, steps=2).generate([], 2)
    assert any("does not read the last draft" in n for n in res.notes)


def test_hooks_see_every_step_and_a_gist_is_pinned():
    be = Table([3, 4, 5, 6, 7], [5, 4, 3, 2, 1])
    states = []

    def between(state):
        return {2: 9} if state.step == 0 else None
    res = HostDiffusionLoop(be, steps=5, commit_schedule=1, stable_steps=None,
                            remask=Remask(threshold=0.99, max_per_step=5),
                            on_step=states.append, between_steps=between
                            ).generate([], 5)
    assert len(states) == res.forwards
    assert res.tokens.tolist()[2] == 9
    gist = [c for c in res.order if c.how == "gist"]
    assert len(gist) == 1 and gist[0].position == 2 and gist[0].token == 9
    assert all(2 not in e.positions for e in res.events if e.kind == "remask")
    assert any(e.kind == "inject-canvas" for e in res.events)


def test_a_recalled_gist_joins_the_context():
    be = Table([3, 4, 5], [3, 2, 1])
    res = HostDiffusionLoop(be, steps=3, commit_schedule=1, stable_steps=None,
                            between_steps=lambda s: Injection(
                                context=[8, 8], note="recall") if s.step == 0
                            else None).generate([1], 3)
    assert be.calls[0]["prompt"] == [1]
    assert be.calls[1]["prompt"] == [1, 8, 8]
    assert any(e.kind == "inject-context" and "recall" in e.detail
               for e in res.events)


def test_the_host_can_stop_the_loop_between_steps():
    be = Table([3, 4, 5, 6], [4, 3, 2, 1])
    res = HostDiffusionLoop(be, steps=8, commit_schedule=1, stable_steps=None,
                            block_size=4,
                            between_steps=lambda s: Injection(
                                stop=True, note="operator pressed stop")
                            if s.step == 1 else None).generate([], 8)
    assert "operator pressed stop" in res.stopped
    assert res.tokens.size == 4 and res.blocks == 1
    assert [c.how for c in res.order].count("stopped") == 2


def test_blocks_are_autoregressive_across_and_diffusion_within():
    be = Table([3, 4, 5, 6], [4, 3, 2, 1])
    res = HostDiffusionLoop(be, steps=4, block_size=4).generate([1, 2], 10)
    assert res.blocks == 3 and res.tokens.size == 10
    plens = sorted({len(c["prompt"]) for c in be.calls})
    assert plens == [2, 6, 10]
    assert max(c.position for c in res.order) == 9


def test_end_of_sequence_ends_generation():
    be = Table([3, 4, 2, 6], [4, 3, 9, 1], eos_id=2)
    res = HostDiffusionLoop(be, steps=4).generate([], 4)
    assert res.tokens.tolist() == [3, 4]
    assert "end-of-sequence" in res.stopped


def test_sampling_is_reproducible_with_a_seed():
    be = Table([3, 4, 5, 6, 7, 8], [0.2] * 6)

    def run(seed):
        return HostDiffusionLoop(be, steps=6, temperature=1.0, seed=seed,
                                 stable_steps=None).generate([], 6).tokens.tolist()
    assert run(3) == run(3)
    assert len({tuple(run(s)) for s in range(6)}) > 1


class Neighbour:
    """Position i is sure of target[i] only when position i-1 is visible."""
    mask_id = 0

    def __init__(self, target):
        self.target = target

    def logits(self, canvas, prompt, prev_draft=None):
        out = np.zeros((len(canvas), V))
        for i in range(len(canvas)):
            seen = i == 0 or canvas[i - 1] != self.mask_id
            out[i, self.target[i]] = 6.0 if seen else 0.5
        return out


def test_surprise_field_is_minus_log_p_under_masking():
    be = Table([3, 4, 5], [2.0, 0.0, 5.0])
    loop = HostDiffusionLoop(be, steps=2)
    s = loop.surprise_field([3, 4, 5], [1])
    for i, (tok, sharp) in enumerate(zip([3, 4, 5], [2.0, 0.0, 5.0])):
        row = np.zeros(V)
        row[tok] = sharp
        row[0] = -np.inf
        expect = -(row[tok] - np.log(np.exp(row[np.isfinite(row)]).sum()))
        assert s[i] == pytest.approx(expect)
    assert loop.last_forwards == 3
    nb = Neighbour([3, 4, 5, 6])
    lp = HostDiffusionLoop(nb, steps=2)
    alone = lp.surprise_field([3, 4, 5, 6], passes=None)
    assert lp.last_forwards == 4
    together = lp.surprise_field([3, 4, 5, 6], passes=1)
    assert lp.last_forwards == 1
    assert alone[1:].mean() < together[1:].mean()
    assert alone[0] == pytest.approx(together[0])
    # a token the backend never scores is capped, not infinite
    capped = HostDiffusionLoop(Table([0, 4], [0, 0]), steps=1)
    assert np.all(capped.surprise_field([4, 4]) <= MAX_SURPRISE_NATS)
    with pytest.raises(LoopError, match="mask token"):
        loop.surprise_field([3, 0, 5])


def test_records_are_the_inner_speech_and_serialise():
    be = Table([3, 4, 5], [3, 2, 1])
    res = HostDiffusionLoop(be, steps=3, commit_schedule=1,
                            stable_steps=None).generate([], 3)
    assert len(res.drafts) == 3
    speech = res.inner_speech(lambda ids: " ".join(map(str, ids)))
    assert speech[-1] == "3 4 5"
    assert res.drafts[0].masked.tolist() == [False, True, True]
    json.dumps(res.as_dict())
    assert res.confidence_at_commit().shape == (3,)


def test_fill_keeps_every_given_token():
    be = Table([3, 4, 5, 6], [1, 1, 1, 1])
    res = HostDiffusionLoop(be, steps=4).fill([7, 0, 8, 0], [1])
    assert res.tokens.tolist() == [7, 4, 8, 6]
    assert res.commit_steps()[[0, 2]].tolist() == [-1, -1]
    none = HostDiffusionLoop(be, steps=4).fill([7, 8])
    assert none.forwards == 0 and "nothing was masked" in none.stopped


def test_errors_are_sentences():
    with pytest.raises(LoopError, match="no logits"):
        HostDiffusionLoop(object())

    class NoMask:
        def logits(self, *a, **k):
            return np.zeros((1, V))
    with pytest.raises(LoopError, match="mask token id"):
        HostDiffusionLoop(NoMask())
    with pytest.raises(LoopError, match="at least one step"):
        HostDiffusionLoop(Table([3], [1]), steps=0)
    with pytest.raises(LoopError, match="unknown commit schedule"):
        HostDiffusionLoop(Table([3], [1]), commit_schedule="fast")

    class Wrong:
        mask_id = 0

        def logits(self, canvas, prompt, prev_draft=None):
            return np.zeros((len(canvas) + 1, V))
    with pytest.raises(LoopError, match="shape"):
        HostDiffusionLoop(Wrong()).generate([], 3)
    with pytest.raises(LoopError, match="cannot use"):
        HostDiffusionLoop(Table([3, 4], [1, 1]), steps=2,
                          between_steps=lambda s: [1, 2]).generate([], 2)
    with pytest.raises(LoopError, match="mask token"):
        HostDiffusionLoop(Table([3, 4], [1, 1]), steps=2,
                          between_steps=lambda s: {1: 0}).generate([], 2)
    assert "steps per 256-token block" in HostDiffusionLoop(Table([3], [1])).describe()


def test_the_toy_backend_restores_a_corpus_sentence():
    from atk_diffusion.text.backends import ToyMaskedLM
    m = ToyMaskedLM(["At 0600 Raven 2 reported a convoy near Bridge 4.",
                     "The convoy moved north along Route 7."])
    ids = m.encode("At 0600 Raven 2 reported a [MASK] near [MASK] 4.")
    res = HostDiffusionLoop(m, steps=4).fill(ids)
    assert m.decode(res.tokens) == "at 0600 raven 2 reported a convoy near bridge 4."
    s = HostDiffusionLoop(m, steps=2).surprise_field(m.encode("the convoy moved north"))
    assert np.all(np.isfinite(s)) and s.mean() / math.log(2) < 3.0
