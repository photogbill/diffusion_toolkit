# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""F2 — revising in place from recorded facts (the hour log, plan §4.H) and
composing as one canvas, with the support check that refuses invention."""

from __future__ import annotations

import json

import pytest

from atk_diffusion.text import backends as B
from atk_diffusion.text import revise as R

LOG = ("14:00-15:00.\n"
       "Transcript of call 12 started at 14:05; transcription still running.\n"
       "RF session on 146.52 MHz recorded 3 bursts.\n"
       "Extraction run 7 finished at 14:40.")
FACTS = [
    {"id": "ev-31", "text": "Transcript of call 12 finished at 15:20.",
     "replaces": "still running"},
    {"id": "ev-32", "text": "RF session on 146.52 MHz recorded 5 bursts."},
    {"id": "ev-40", "text": "Analyst annotation added at 14:55."},
]
BACKGROUND = ["Transcript of call 9 finished at 13:10. RF session on 162.4 MHz "
              "recorded 2 bursts.",
              "Extraction run 3 finished at 11:40. Transcription still running "
              "at 12:00."]


@pytest.fixture
def toy():
    return B.ToyMaskedLM(BACKGROUND)


def test_support_check_is_arithmetic_over_words_and_numbers():
    ok = R.support_check("RF session recorded 5 bursts.",
                         "RF session recorded 3 bursts.",
                         ["RF session on 146.52 MHz recorded 5 bursts."])
    assert ok.supported and ok.lost == ["3 burst"]
    bad = R.support_check("RF session recorded 6 bursts near the river.",
                          "RF session recorded 3 bursts.", ["5 bursts recorded."])
    assert not bad.supported
    assert "river" in bad.unsupported and "6 burst" in bad.unsupported
    assert "not in the recorded facts" in bad.why()
    neg = R.support_check("The call was not recorded.", "The call was recorded.",
                          ["Call recorded."])
    assert neg.unsupported == ["not"]
    t = R.support_check("Finished at 15:20.", "Started.", ["finished at 15:25"])
    assert "15:20" in t.unsupported
    assert R.support_check("40.0 Nm", "", ["40 Nm"]).supported
    assert not R.support_check("41 Nm", "", ["40 Nm"]).supported
    # words a fact superseded support nothing any more
    old = "Transcription still running."
    assert R.support_check("Transcription still running at 15:20.", old,
                           ["finished at 15:20"]).supported
    assert not R.support_check("Transcription still running at 15:20.", old,
                               ["finished at 15:20"],
                               superseded=["still running"]).supported


def test_the_hour_log_is_revised_in_place_by_the_splice():
    r = R.revise_in_place(LOG, FACTS)
    lines = r["text"].split("\n")
    assert lines[1] == ("Transcript of call 12 started at 14:05; transcription "
                        "finished at 15:20.")
    assert lines[2] == "RF session on 146.52 MHz recorded 5 bursts."
    assert lines[4] == "Analyst annotation added at 14:55."
    assert len(lines) == 5 and "correction" not in r["text"].lower()
    assert r["method"] == "splice_revise" and r["tier"] == "cleaned"
    assert r["hallucination_rate"] == 0.0
    kinds = [(d["index"], d["change"]) for d in r["diff"]]
    assert kinds == [(1, "revised"), (2, "revised"), (3, "added")]
    prov = {p["text"]: p["facts"] for p in r["provenance"]}
    assert prov[lines[1]] == ["ev-31"] and prov[lines[2]] == ["ev-32"]
    assert all(d["support"]["supported"] for d in r["diff"])
    assert "different burst count" in r["diff"][1]["why"]
    json.dumps(r)


def test_the_model_revision_is_checked_and_refused_when_it_invents(toy):
    r = R.revise_in_place(LOG, FACTS, toy)
    assert r["method"] == "diffusion_revise" and r["tier"] == "invented"
    lines = r["text"].split("\n")
    assert lines[2] == "RF session on 146.52 MHz recorded 5 bursts."
    burst = next(d for d in r["diff"] if d["index"] == 2)
    assert burst["method"] == "diffusion_revise" and burst["tier"] == "invented"
    # the toy wrote words from a neighbouring line into the transcript entry;
    # the check refused it and the splice from the recorded fact was used
    call = next(d for d in r["diff"] if d["index"] == 1)
    assert lines[1].endswith("transcription finished at 15:20.")
    assert call["method"] == "splice_revise" and call["tier"] == "cleaned"
    assert r["refused"] and r["refused"][0]["fact"] == "ev-31"
    assert r["refused"][0]["why"].startswith("refused: not in the recorded facts")
    assert r["hallucination_rate"] == pytest.approx(0.5)
    assert "1 of 2 model revision(s)" in r["hallucination_note"]
    r0 = R.revise_in_place(LOG, FACTS, toy, context_sentences=0)
    why = r0["refused"][0]["why"]
    assert "running" in why          # superseded words are not support


class Inventor(B.ToyMaskedLM):
    """A backend that writes 'helicopter' into every gap."""

    def logits(self, canvas_ids, prompt_ids, prev_draft=None):
        out = super().logits(canvas_ids, prompt_ids, prev_draft)
        out[:, self.stoi["helicopter"]] += 100.0
        return out


def test_an_inventing_model_is_refused_strictly_and_flagged_otherwise():
    inv = Inventor(BACKGROUND)
    inv.encode("helicopter")
    facts = [{"id": "ev-32", "text": "RF session on 146.52 MHz recorded 5 bursts."}]
    kept = R.revise_in_place(LOG, facts, inv, fallback=None)
    assert kept["text"] == LOG
    assert kept["hallucination_rate"] == 1.0
    assert "helicopter" in kept["refused"][0]["why"]
    loose = R.revise_in_place(LOG, facts, inv, strict=False)
    d = next(x for x in loose["diff"] if x["index"] == 2)
    assert "helicopter" in d["after"].lower()
    assert "KEPT ALTHOUGH UNSUPPORTED" in d["why"]
    assert not d["support"]["supported"]
    assert loose["hallucination_rate"] == 1.0
    assert "kept and flagged" in loose["hallucination_note"]


def test_event_ids_place_a_fact_exactly():
    doc = "Patrol left the gate.\nGenerator fuel logged."
    events = [["ev-1"], ["ev-2"]]
    r = R.revise_in_place(doc, [{"id": "f", "event_id": "ev-2",
                                 "text": "Generator fuel logged at 40 litres.",
                                 }], document_events=events)
    assert r["text"].split("\n")[1] == "Generator fuel logged at 40 litres."
    with pytest.raises(ValueError, match="one list of event ids per sentence"):
        R.revise_in_place(doc, ["x"], document_events=[["ev-1"]])


def test_a_new_statement_is_appended_as_its_own_clause():
    r = R.revise_in_place("Transcript of call 12 started at 14:05.",
                          ["Transcript of call 12 finished at 15:20."])
    assert r["text"] == ("Transcript of call 12 started at 14:05; finished at "
                         "15:20.")


def test_a_quiet_hour_becomes_its_late_fact():
    r = R.revise_in_place("No recorded activity.", ["Kiwi session opened at 03:12."])
    assert r["text"] == "Kiwi session opened at 03:12."
    assert r["diff"][0]["change"] == "revised"


def test_facts_must_be_recorded_and_redactions_are_not_touched():
    with pytest.raises(ValueError, match="has no text"):
        R.as_facts([{"text": "  "}])
    with pytest.raises(B.RedactionRefused, match="§5"):
        R.revise_in_place("Met [REDACTED] at noon.",
                          [{"text": "Met Smith at noon.", "replaces": "[REDACTED]"}])
    facts = R.as_facts(["one", {"text": "two", "id": "x"}])
    assert [f.id for f in facts] == ["fact 1", "x"]


SPECIALISTS = {
    "signals": "Raven 2 reported a convoy near Bridge 4 at 0600. The convoy "
               "moved north.",
    "imagery": "Imagery shows six trucks near Bridge 4. The trucks moved north "
               "along Route 7.",
    "liaison": "Local liaison says the convoy carried food aid.",
}


def test_compose_writes_one_canvas_and_attributes_every_sentence(toy):
    r = R.compose(SPECIALISTS, toy, question="What moved near Bridge 4?")
    assert r["method"] == "diffusion_compose" and r["tier"] == "invented"
    assert r["text"] and r["sentences"]
    assert any(s["from"] for s in r["sentences"])
    assert 0.0 <= r["hallucination_rate"] <= 1.0
    assert r["baseline"]["method"] == "extractive_compose"
    assert r["baseline"]["tier"] == "cleaned"
    assert r["forwards"] >= 1 and r["drafts"] >= 1
    json.dumps(r)


def test_strict_compose_drops_what_no_specialist_said():
    inv = Inventor(BACKGROUND)
    inv.encode("helicopter")
    r = R.compose(SPECIALISTS, inv, strict=True, length=12)
    assert all(s.get("dropped") for s in r["sentences"] if s["unsupported"])
    assert "helicopter" not in r["text"].lower()
    assert r["hallucination_rate"] > 0.5


def test_without_a_model_the_composer_is_extractive():
    r = R.compose(SPECIALISTS, None)
    assert r["method"] == "extractive_compose" and r["hallucination_rate"] == 0.0
    texts = [s for t in SPECIALISTS.values() for s, _a, _b in
             B.split_sentences(t)]
    assert all(s["text"] in texts for s in r["sentences"])
    assert "no masked backend" in r["notes"][0]
    with pytest.raises(ValueError, match="nothing to compose"):
        R.compose({"a": " "}, None)
