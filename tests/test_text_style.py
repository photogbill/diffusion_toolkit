# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""F4 — style fidelity as predictability, classical stylometrics, and the
terminology-consistent translation post-edit (plan §4.F4)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from atk_diffusion.text import style as S
from atk_diffusion.text.backends import ToyMaskedLM

NOUNS = ["convoy", "truck", "bridge", "depot", "patrol", "radio", "road",
         "river", "market", "ford", "crowd", "smoke", "engine", "tower"]
VERBS = ["moved", "stopped", "held", "burned", "closed", "opened", "crossed",
         "waited", "returned", "failed"]
ADJS = ["old", "quiet", "grey", "narrow", "distant", "heavy", "small", "cold"]


def terse(rng, n):
    out = []
    for _ in range(n):
        a, b = rng.choice(NOUNS, 2, replace=False)
        k = rng.integers(3)
        out.append(f"{a.capitalize()} {rng.choice(VERBS)}." if k == 0 else
                   f"{a.capitalize()} {rng.choice(VERBS)} near {b}." if k == 1
                   else f"No {a} seen at the {b}.")
    return " ".join(out)


def florid(rng, n):
    out = []
    for _ in range(n):
        a, b, c = rng.choice(NOUNS, 3, replace=False)
        out.append(f"It was, as it had always been, the {rng.choice(ADJS)} {a} "
                   f"that {rng.choice(VERBS)} first, and the {b}, which had "
                   f"waited so long by the {c}, seemed at last to have "
                   f"{rng.choice(VERBS)} as if it knew.")
    return " ".join(out)


def test_stylometrics_measure_the_text():
    m = S.stylometrics("The convoy moved. It was, of course, late. Six trucks.")
    assert m["sentences"] == 3 and m["words"] == 10
    assert m["sentence_length"]["mean"] == pytest.approx(10 / 3)
    assert m["sentence_length"]["histogram"][0] == pytest.approx(1.0)
    assert m["function_words"]["the"] == pytest.approx(100.0)
    assert m["commas_per_sentence"] == pytest.approx(2 / 3)
    assert m["tier"] == "measured"


def test_style_distance_separates_two_voices(rng):
    t_ref, f_ref = terse(rng, 300), florid(rng, 40)
    t_txt, f_txt = terse(rng, 90), florid(rng, 10)
    same = S.style_distance(t_txt, t_ref)
    other = S.style_distance(f_txt, t_ref)
    assert same["delta"] < 2.0 < other["delta"]
    assert S.style_distance(f_txt, f_ref)["delta"] < \
        S.style_distance(t_txt, f_ref)["delta"]
    assert same["sentence_length_js"] < other["sentence_length_js"]
    assert same["reliability"] == "good" and "standard deviations" in same["meaning"]
    short = S.style_distance("Convoy moved.", t_ref)
    assert short["reliability"] == "low"
    with pytest.raises(ValueError, match="needs words"):
        S.style_distance("", t_ref)
    json.dumps(same)


def test_style_fidelity_is_pmi_with_the_author(rng):
    t_ref, f_ref = terse(rng, 80), florid(rng, 12)
    toy = ToyMaskedLM(["The report was filed. The road was open and the weather "
                       "was fair."])
    t_txt, f_txt = terse(rng, 4), florid(rng, 1)
    a = S.style_fidelity(t_txt, t_ref, toy)
    b = S.style_fidelity(f_txt, t_ref, toy)
    c = S.style_fidelity(f_txt, f_ref, toy)
    assert a["pmi_bits_per_token"] > 0.25 and c["pmi_bits_per_token"] > 0.25
    assert b["pmi_bits_per_token"] < a["pmi_bits_per_token"]
    assert "MORE like the author" in a["meaning"]
    assert "not a judgement of quality" in a["meaning"]
    assert a["method"] == "style_pmi" and a["tier"] == "proposed"
    assert "surprise field" in a["scored_by"]

    class LeftToRight:            # a scoring-only backend: the toy's AR side
        def __init__(self, m):
            self.m = m

        def logprob(self, text, context=""):
            return self.m.logprob(text, context)

        def fill(self, masked):
            return masked
    ar = S.style_fidelity(t_txt, t_ref, LeftToRight(toy))
    assert ar["scored_by"] == "left-to-right log-probabilities"
    assert ar["pmi_bits_per_token"] > 0
    with pytest.raises(ValueError, match="needs a model backend"):
        S.style_fidelity(t_txt, t_ref, None)
    with pytest.raises(ValueError, match="author's writing"):
        S.style_fidelity(t_txt, "  ", toy)
    rep = S.style_report(t_txt, [t_ref], toy)
    assert {"stylometric", "profile", "predictability"} <= set(rep)


GLOSSARY = {"Ejército de Liberación": {"rendering": "Liberation Army",
                                       "variants": ["Army of Liberation"]},
            "puesto de control": "checkpoint"}
SOURCE = ["El Ejército de Liberación cerró el puesto de control.",
          "El puesto de control abrió a las 0900.",
          "El Ejército de Liberación se retiró.",
          "Nadie vio el puesto de control."]
TARGET = ["The Liberation Army closed the checkpoint.",
          "The check point opened at 0900.",
          "The army of Liberation withdrew.",
          "Nobody saw anything."]


def test_terminology_finds_inconsistent_renderings_and_proposes_edits():
    r = S.terminology_check(GLOSSARY, TARGET, SOURCE)
    got = [(f["segment"], f["kind"], f["found"]) for f in r["findings"]]
    assert (2, "variant", "army of Liberation") in got
    assert (1, "variant", "check point") in got
    assert (3, "missing", "") in got
    assert not r["consistent"] and r["tier"] == "measured"
    t = r["terms"]
    assert t["Ejército de Liberación"]["consistent"] == 1
    assert t["Ejército de Liberación"]["inconsistent"] == 1
    assert t["puesto de control"]["missing"] == 1
    after = {p["segment"]: p["after"] for p in r["proposals"]}
    assert after[2] == "The Liberation Army withdrew."
    assert after[1] == "The checkpoint opened at 0900."
    assert all(p["tier"] == "proposed" for p in r["proposals"])
    one = [p["id"] for p in r["proposals"] if p["segment"] == 1]
    ed = S.apply_postedits(TARGET, r["proposals"], accept=one)
    assert ed["segments"][1] == "The checkpoint opened at 0900."
    assert ed["segments"][2] == TARGET[2]          # not accepted, not changed
    assert ed["applied"] == one
    with pytest.raises(ValueError, match="aligned one to one"):
        S.terminology_check(GLOSSARY, TARGET, SOURCE[:2])
    json.dumps(r)


def test_terminology_without_a_source_and_by_majority():
    text = ("The Army of Liberation crossed. The Liberation Army held. The "
            "Liberation Army left.")
    r = S.terminology_check({"Ejército de Liberación": {
        "rendering": None, "variants": ["Liberation Army", "Army of Liberation"]}},
        text)
    assert r["terms"]["Ejército de Liberación"]["required"] == "Liberation Army"
    assert "most frequent" in r["terms"]["Ejército de Liberación"]["note"]
    assert [p["after"] for p in r["proposals"]] == ["The Liberation Army crossed."]
    with pytest.raises(ValueError, match="neither a rendering nor variants"):
        S.terminology_check({"x": None}, "text")
    capital = S.terminology_check({"puesto": {"rendering": "checkpoint",
                                              "variants": ["post"]}},
                                  ["Post closed."])
    assert capital["proposals"][0]["after"] == "Checkpoint closed."


def test_agreement_is_checked_by_a_masked_model():
    toy = ToyMaskedLM(["The Liberation Army closed the checkpoint at noon.",
                       "The checkpoint opened at 0900."])
    r = S.terminology_check(GLOSSARY, TARGET, SOURCE, backend=toy)
    ag = [p["agreement"] for p in r["proposals"]]
    assert all(a is not None and np.isfinite(a["rise_bits"]) for a in ag)
    assert all(isinstance(a["flag"], bool) for a in ag)
    assert S.agreement_check("a b", "a c", 2, 3, 3, None) is None
