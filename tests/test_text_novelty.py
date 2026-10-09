# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""F1 — what's new: novelty_rank, new_facts, the classical baselines, the
ranking statistics, and the planted-truth experiment (plan §4.F1, §6)."""

from __future__ import annotations

import importlib.util
import json
import math

import pytest

from atk_diffusion.text import novelty as N
from atk_diffusion.text.backends import ToyMaskedLM

BACKGROUND = [
    "At 0500 Heron 4 reported a patrol near the quarry. The patrol moved east "
    "along the canal.",
    "At 0930 Osprey 7 reported a fire near the mill. The fire spread toward "
    "the school.",
    "At 1100 Heron 4 reported a roadblock near the station. The roadblock "
    "opened at 1200."]
PROJECT = [
    "At 0600 Raven 2 reported a convoy near Bridge 4. The convoy moved north "
    "along Route 7.",
    "At 0715 Falcon 1 reported smoke near the depot. Smoke was seen west of "
    "the depot.",
    "At 0820 Raven 2 reported a checkpoint near the market. The checkpoint "
    "closed at 0900.",
    "At 1000 Falcon 1 reported a crowd near the clinic. The crowd dispersed by "
    "1030."]
DOCS = {
    "dup": "At 0600 Raven 2 reported a convoy near Bridge 4. The checkpoint "
           "closed at 0900.",
    "one_new": "At 0600 Raven 2 reported a convoy near Bridge 4. At 1340 "
               "Kestrel 3 reported a fuel truck near Ford 9.",
    "two_new": "At 1340 Kestrel 3 reported a fuel truck near Ford 9. At 1415 "
               "Merlin 5 reported gunfire near the orchard.",
}


@pytest.fixture
def toy():
    return ToyMaskedLM(BACKGROUND)


def test_rank_statistics_match_their_definitions():
    a, b = [1, 2, 3, 4, 5], [2, 1, 4, 3, 5]
    assert N.spearman(a, a) == pytest.approx(1.0)
    assert N.kendall_tau_b(a, a[::-1]) == pytest.approx(-1.0)
    assert N.auc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0
    assert N.auc([0.5, 0.5], [1, 0]) == 0.5
    assert math.isnan(N.spearman([1, 1, 1], [1, 2, 3]))
    stats = pytest.importorskip("scipy.stats", reason="SciPy cross-check only")
    ties = ([1, 2, 2, 3, 5, 5], [2, 1, 3, 3, 4, 6])
    assert N.spearman(*ties) == pytest.approx(stats.spearmanr(*ties)[0])
    assert N.kendall_tau_b(*ties) == pytest.approx(stats.kendalltau(*ties)[0])
    assert N.spearman(a, b) == pytest.approx(stats.spearmanr(a, b)[0])


def test_tfidf_index_is_cosine():
    idx = N.TfidfIndex(["the convoy reached bridge 4", "smoke over the depot"])
    s = idx.query("the convoy reached bridge 4")
    assert s[0] == pytest.approx(1.0) and s[1] == pytest.approx(0.0)
    assert 0 < idx.query("a convoy near the depot")[0] < 1


def test_new_facts_names_the_new_sentence_and_why(toy):
    doc = DOCS["one_new"]
    r = N.new_facts(doc, PROJECT, toy)
    assert r["method"] == "novelty_masked" and r["tier"] == "proposed"
    assert r["meaning"] == "a model's view of predictability, not a judgement of truth"
    assert [s["verdict"] for s in r["sentences"]] == ["redundant", "new"]
    new = r["new"][0]
    assert doc[new["start"]:new["end"]] == new["text"]
    assert {"kestrel", "fuel", "truck", "ford", "9", "3", "1340"} & set(new["hardest"])
    assert "resisted reconstruction" in new["why"] and "bits/token" in new["why"]
    assert new["recovered"][0] < new["recovered"][1]
    red = r["sentences"][0]
    assert "restored from the project" in red["why"]
    assert red["recovered"][0] >= red["recovered"][1] - 1
    assert r["classical"]["tier"] == "measured" and r["classical"]["new"] == [1]
    assert r["agreement_with_classical"] == 1.0
    assert "calibrated on" in r["threshold"]["how"]
    json.dumps(r)


@pytest.mark.parametrize("mode", ["masked", "pmi", "tfidf", "containment"])
def test_novelty_rank_orders_by_what_each_adds(toy, mode):
    docs = [{"name": k, "text": v} for k, v in DOCS.items()]
    r = N.novelty_rank(docs, PROJECT, None if mode in ("tfidf", "containment")
                       else toy, mode=mode)
    names = [d["name"] for d in r["ranking"]]
    if mode in ("masked", "pmi"):
        assert names == ["two_new", "one_new", "dup"]
        assert r["tier"] == "proposed"
        dup = next(d for d in r["ranking"] if d["name"] == "dup")
        assert dup["added"] == 0.0 and dup["new_sentences"] == 0
    else:
        assert r["tier"] == "measured"
        key = "classical_added" if mode == "tfidf" else "containment_added"
        order = sorted(r["documents"], key=lambda d: -d[key])
        assert [d["name"] for d in order] == ["two_new", "one_new", "dup"]
    assert r["classical"]["ranking"] == ["two_new", "one_new", "dup"]
    agree = r["rank_agreement_with_classical"]
    assert agree["spearman"] == pytest.approx(1.0) or mode == "containment"


def test_without_a_model_the_classical_method_answers():
    r = N.new_facts(DOCS["one_new"], PROJECT, None)
    assert r["method"] == "novelty_tfidf" and r["tier"] == "measured"
    assert r["meaning"] == N.CLASSICAL_MEANING
    assert [s["verdict"] for s in r["sentences"]] == ["redundant", "new"]
    assert "nearest earlier sentence" in r["sentences"][1]["why"]


def test_a_document_that_repeats_itself_adds_nothing_twice(toy):
    fact = "At 1340 Kestrel 3 reported a fuel truck near Ford 9."
    r = N.new_facts(fact + " " + fact, PROJECT, toy)
    assert [s["verdict"] for s in r["sentences"]] == ["new", "redundant"]
    r2 = N.new_facts(fact + " " + fact, PROJECT, toy, include_earlier=False)
    assert [s["verdict"] for s in r2["sentences"]] == ["new", "new"]


def test_modes_and_small_projects_are_explained(toy):
    with pytest.raises(ValueError, match="masked .* backend"):
        N.new_facts("text.", PROJECT, None, mode="masked")
    with pytest.raises(ValueError, match="unknown novelty mode"):
        N.new_facts("text.", PROJECT, toy, mode="vibes")
    r = N.new_facts(DOCS["one_new"], PROJECT[:1], toy)
    assert "the default line" in r["threshold"]["how"]
    r = N.new_facts(DOCS["one_new"], PROJECT, toy, threshold=99.0)
    assert r["threshold"]["how"] == "set by the caller" and not r["new"]
    assert N.new_facts("", PROJECT, toy)["sentences"] == []


def test_the_planted_experiment_reports_agreement_per_arm(tmp_path):
    from atk_diffusion.experiments import novelty_eval as E
    data = E.synthetic_corpus(1, n_project=5, n_docs=5, doc_sentences=4)
    assert data["truth"] == [0, 1, 2, 3, 4]
    assert all(len(l) == 4 for l in data["labels"])
    assert sum(k.startswith("new") for k in data["labels"][-1]) == 4
    r = E.run(tmp_path / "run", data=data, doc_sentences=4,
              arms=("toy_masked", "classical_tfidf", "llama_pmi"))
    assert set(r["arms"]) == {"toy_masked", "classical_tfidf"}
    assert "GGUF" in r["skipped"]["llama_pmi"]
    for arm in r["arms"].values():
        assert -1.0 <= arm["spearman"] <= 1.0
        assert 0.0 <= arm["false_new_rate"] <= 1.0
        assert set(arm["auc_by_kind"]) == {"new:novel-entity",
                                           "new:recombination"}
    assert r["arms"]["toy_masked"]["auc_by_kind"]["new:novel-entity"] > 0.8
    assert r["arms"]["classical_tfidf"]["spearman"] > 0.8
    assert (tmp_path / "run" / "novelty_eval.md").is_file()
    payload = json.loads((tmp_path / "run" / "novelty_eval.json").read_text())
    assert payload["tier"] == "measured"
    assert any(x.startswith("toy_masked (proposed): Spearman")
               for x in payload["summary"])
    assert "| toy_masked (proposed) |" in r["report_md"]
    assert "skipped: no GGUF" in r["report_md"]
    assert E.run(data=data, doc_sentences=4, arms=("classical_tfidf",))[
        "report"]["writer"] is None


def test_the_real_run_needs_only_documents_and_a_ranking():
    from atk_diffusion.experiments import novelty_eval as E
    docs = [{"name": k, "text": v} for k, v in DOCS.items()]
    r = E.run(data={"project": PROJECT, "documents": docs, "truth": [0, 1, 2],
                    "truth_kind": "Bill's blind ranking"},
              arms=("toy_pmi", "classical_tfidf"))
    for arm in r["arms"].values():
        assert arm["spearman"] == pytest.approx(1.0)
        assert math.isnan(arm["sentence_auc"])
    assert any("no sentence labels" in n for n in r["arms"]["toy_pmi"]["notes"])
    assert any("trained on the project" in n for n in r["notes"])
    assert "Bill's blind ranking" in r["report_md"]


def test_the_shared_report_writer_when_it_exists(tmp_path):
    if importlib.util.find_spec("atk_diffusion.experiments.report") is None:
        pytest.skip("atk_diffusion.experiments.report is not written yet; the "
                    "experiment writes its report locally meanwhile")
    from atk_diffusion.experiments import novelty_eval as E
    data = E.synthetic_corpus(2, n_project=4, n_docs=3, doc_sentences=3)
    r = E.run(tmp_path, data=data, doc_sentences=3, arms=("classical_tfidf",))
    rep = r["report"]
    assert rep["writer"] == "atk_diffusion.experiments.report.write_report"
    md = (tmp_path / "novelty_eval.md").read_text(encoding="utf-8")
    assert md.startswith("# Novelty filter") and "MEASURED" in md
    assert "- classical_tfidf (measured): Spearman" in md
    assert (tmp_path / "novelty_eval_table.md").read_text().startswith(
        "# Novelty filter")
    payload = json.loads((tmp_path / "novelty_eval.json").read_text())
    assert payload["provenance"]["tool"] == "experiments.novelty_eval"


def test_the_report_is_written_locally_without_the_shared_writer(tmp_path,
                                                                  monkeypatch):
    import sys
    from atk_diffusion.experiments import novelty_eval as E
    monkeypatch.setitem(sys.modules, "atk_diffusion.experiments.report", None)
    data = E.synthetic_corpus(3, n_project=4, n_docs=3, doc_sentences=3)
    r = E.run(tmp_path, data=data, doc_sentences=3, arms=("classical_tfidf",))
    assert r["report"]["writer"] == "local"
    assert "not importable" in r["report"]["note"]
    assert json.loads((tmp_path / "novelty_eval.json").read_text())[
        "result"]["experiment"] == "novelty_eval"
    assert (tmp_path / "novelty_eval.md").read_text().startswith("# Novelty")
