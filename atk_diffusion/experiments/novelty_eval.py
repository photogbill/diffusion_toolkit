# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Phase 6's exit, in miniature: does the novelty filter rank documents the
way the truth does? (plan §6 Phase 6, §4.F1, §7)

The plan's exit for text diffusion is "novelty-filter ranking agrees with an
analyst's blind ranking". Bill's blind ranking needs Bill; this experiment is
the same measurement on a corpus whose truth is PLANTED, so the code path,
the numbers and the report exist before his documents do:

* a synthetic project corpus of field reports ("At 0715 Falcon 1 reported
  smoke near the depot."), and a background corpus in the same language
  with other names (what a pretrained model would already know);
* new documents of equal length, each holding a planted number of NEW facts
  (0 … all) among DUPLICATES of project facts — verbatim copies and
  reworded restatements. New facts are of two kinds: a NOVEL ENTITY (a
  unit, place or event the project never mentions) and a RECOMBINATION
  (known names in a combination the project never reported — every word
  familiar, the fact new; the case word overlap should miss);
* each arm ranks the documents with `novelty_rank`, and the ranking is
  compared with the planted counts: Spearman's rho and Kendall's tau-b.
  Sentence by sentence: the AUC of the scores for new against redundant, per
  kind; the FALSE-NEW rate (a duplicate called new — the novelty filter's
  hallucination rate) and the missed-new rate.

ARMS: the toy diffusion backend (masked reconstruction), the toy read left
to right (PMI), llama.cpp (PMI) when a GGUF path is given and llama-cpp-
python is installed, and the classical baselines (TF-IDF, trigram
containment), which always run. A model that does not beat the classical
arms on Bill's real documents is not shipped (plan §7).

LIMITS. Synthetic templates are easy: a perfect score here says the code
works, not that the method does. The real run is the same function with
`project=` and `documents=` (and `truth=` from the analyst's blind ranking)
pointed at Bill's own material.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np

UNITS = ["Raven 2", "Falcon 1", "Heron 4", "Osprey 7", "Kestrel 3",
         "Merlin 5", "Harrier 6", "Condor 8", "Swift 9", "Shrike 2",
         "Plover 5", "Curlew 1", "Lapwing 3", "Gannet 7", "Petrel 6",
         "Skua 4", "Tern 8", "Avocet 2"]
PLACES = ["Bridge 4", "the depot", "the market", "the clinic", "Route 7",
          "the quarry", "the mill", "the school", "Ford 9", "the orchard",
          "the station", "the canal", "Hill 312", "the airstrip",
          "the reservoir", "the stadium", "Gate 6", "the pumphouse",
          "the cemetery", "the ferry", "Junction 11", "the warehouse",
          "the barracks", "the hospital"]
EVENTS = ["a convoy", "smoke", "a checkpoint", "a crowd", "gunfire",
          "a fuel truck", "a roadblock", "a patrol", "a fire", "an ambulance",
          "a drone", "a generator", "a crane", "a ferry crossing",
          "a power cut", "a water queue", "a bus", "a tractor", "a sandbag wall",
          "a flare", "a siren", "a horse cart", "a tanker", "a radio mast"]
DIRS = ["north", "south", "east", "west"]


def _times(rng, n):
    """Distinct clock times (HHMM), so a time is never shared by accident."""
    vals = rng.choice(np.arange(24 * 60), size=min(n, 24 * 60), replace=False)
    return [f"{v // 60:02d}{v % 60:02d}" for v in vals]


def synthetic_corpus(seed: int = 0, *, n_project: int = 12,
                     facts_per_project_doc: int = 4, n_docs: int = 10,
                     doc_sentences: int = 5) -> dict:
    """The planted corpus. Names are split three ways — background, project,
    novel — so a novel entity is unseen by the project AND by the toy's
    training, and a recombination uses only project names."""
    rng = np.random.default_rng(seed)
    pools = {}
    for key, items in (("unit", UNITS), ("place", PLACES), ("event", EVENTS)):
        order = rng.permutation(len(items))
        k = len(items) // 3
        pools[key] = {"background": [items[i] for i in order[:k]],
                      "project": [items[i] for i in order[k:2 * k]],
                      "novel": [items[i] for i in order[2 * k:]]}
    used_times = iter(_times(rng, 24 * 60))

    def fact(part, used=None):
        while True:
            u = pools["unit"][part][rng.integers(len(pools["unit"][part]))]
            p = pools["place"][part][rng.integers(len(pools["place"][part]))]
            e = pools["event"][part][rng.integers(len(pools["event"][part]))]
            if used is None or (u, p, e) not in used:
                return {"time": next(used_times), "unit": u, "place": p,
                        "event": e}

    def say(f, reword=False):
        if reword:
            return (f"{f['unit']} saw {f['event']} by {f['place']} at "
                    f"{f['time']}.")
        return f"At {f['time']} {f['unit']} reported {f['event']} near {f['place']}."

    background = []
    for _ in range(max(6, n_project)):
        fs = [fact("background") for _ in range(facts_per_project_doc)]
        background.append(" ".join(say(f, reword=bool(rng.integers(2)))
                                   + f" The {rng.choice(DIRS)} road stayed open."
                                   for f in fs))
    project_facts, project = [], []
    seen = set()
    for _ in range(n_project):
        fs = []
        for _k in range(facts_per_project_doc):
            f = fact("project", seen)
            seen.add((f["unit"], f["place"], f["event"]))
            fs.append(f)
        project_facts.extend(fs)
        project.append(" ".join(say(f) for f in fs))
    docs, truth, labels = [], [], []
    for d in range(n_docs):
        k = int(round(d * doc_sentences / max(1, n_docs - 1)))
        sents, kinds = [], []
        for j in range(doc_sentences):
            if j < k:
                if j % 2 == 0:
                    f = fact("novel")
                    which = rng.choice(["unit", "place", "event"])
                    base = fact("project")
                    base[which] = f[which]          # one never-seen name
                    sents.append(say(base))
                    kinds.append("new:novel-entity")
                else:
                    # every word familiar — names AND time from the project —
                    # in a combination the project never reported
                    f = fact("project", seen)
                    seen.add((f["unit"], f["place"], f["event"]))
                    f["time"] = project_facts[rng.integers(len(project_facts))]["time"]
                    sents.append(say(f))
                    kinds.append("new:recombination")
            else:
                f = project_facts[rng.integers(len(project_facts))]
                if j % 2 == 0:
                    sents.append(say(f))
                    kinds.append("dup:verbatim")
                else:
                    sents.append(say(f, reword=True))
                    kinds.append("dup:reworded")
        order = rng.permutation(doc_sentences)
        sents = [sents[i] for i in order]
        kinds = [kinds[i] for i in order]
        docs.append({"name": f"doc{d:02d} ({k} new)", "text": " ".join(sents)})
        truth.append(k)
        labels.append(kinds)
    return {"background": background, "project": project, "documents": docs,
            "truth": truth, "labels": labels, "seed": seed}


# ---------------------------------------------------------------------------
# scoring one arm
# ---------------------------------------------------------------------------

def _arm(name, backend, mode, data, progress) -> dict:
    from atk_diffusion.text import novelty as N
    t0 = time.perf_counter()
    r = N.novelty_rank(data["documents"], data["project"], backend, mode=mode,
                       progress=progress)
    by_doc = sorted(r["documents"], key=lambda d: d["document"])
    if mode in ("tfidf", "containment"):
        key = "classical_added" if mode == "tfidf" else "containment_added"
        scores = [d[key] for d in by_doc]
    else:
        scores = [d["added"] for d in by_doc]
    truth = data["truth"]
    out = {
        "arm": name, "method": r["method"], "tier": r["tier"],
        "spearman": N.spearman(scores, truth),
        "kendall_tau_b": N.kendall_tau_b(scores, truth),
        "sentence_auc": float("nan"),
        "auc_by_kind": {"new:novel-entity": float("nan"),
                        "new:recombination": float("nan")},
        "false_new_rate": float("nan"), "false_new_reworded": float("nan"),
        "missed_new_rate": float("nan"),
        "threshold": r["threshold"] if mode not in ("tfidf", "containment")
        else r["classical"]["threshold"],
        "document_scores": scores, "seconds": time.perf_counter() - t0,
        "notes": list(r.get("notes", [])),
    }
    labels = data.get("labels")
    if not labels:
        out["notes"].append("no sentence labels were given: only the "
                            "document ranking is scored")
        return out
    sent_scores, sent_new, sent_kind, verdict_new = [], [], [], []
    for d, kinds in zip(by_doc, labels):
        if len(d["detail"]) != len(kinds):
            raise RuntimeError(f"{d['name']}: the sentence splitter found "
                               f"{len(d['detail'])} sentences but the labels "
                               f"name {len(kinds)}; they cannot be matched")
        for sent, kind in zip(d["detail"], kinds):
            if mode in ("tfidf", "containment"):
                sc = sent["classical"][mode]
                v = sent["classical"]["verdict_" + mode] == "new"
            else:
                sc, v = sent["score"], sent["verdict"] == "new"
            sent_scores.append(sc)
            sent_new.append(str(kind).startswith("new"))
            sent_kind.append(str(kind))
            verdict_new.append(v)
    s = np.asarray(sent_scores)
    y = np.asarray(sent_new, dtype=bool)
    k = np.asarray(sent_kind)
    v = np.asarray(verdict_new, dtype=bool)
    for kind in out["auc_by_kind"]:
        m = (k == kind) | ~y
        out["auc_by_kind"][kind] = N.auc(s[m], y[m])
    dups = ~y
    out.update(
        sentence_auc=N.auc(s, y),
        false_new_rate=float(v[dups].mean()) if dups.any() else 0.0,
        false_new_reworded=(float(v[k == "dup:reworded"].mean())
                            if (k == "dup:reworded").any() else float("nan")),
        missed_new_rate=float((~v[y]).mean()) if y.any() else 0.0)
    return out


def _fmt(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) \
        else f"{x:.2f}"


def report_md(result: dict) -> str:
    lines = [f"# {TITLE}",
             "",
             f"{result['documents']} documents of {result['doc_sentences']} "
             f"sentences against a project of {result['project_sentences']} "
             f"sentences (seed {result['seed']}). Truth: "
             f"{result.get('truth_kind', 'the planted number of new facts')}.",
             "",
             "| arm | Spearman | Kendall τb | sentence AUC | AUC novel entity "
             "| AUC recombination | false-new | false-new (reworded) | missed-new "
             "| seconds |", "|---|---|---|---|---|---|---|---|---|---|"]
    for a in result["arms"].values():
        lines.append(
            f"| {a['arm']} ({a['tier']}) | {_fmt(a['spearman'])} | "
            f"{_fmt(a['kendall_tau_b'])} | {_fmt(a['sentence_auc'])} | "
            f"{_fmt(a['auc_by_kind']['new:novel-entity'])} | "
            f"{_fmt(a['auc_by_kind']['new:recombination'])} | "
            f"{_fmt(a['false_new_rate'])} | {_fmt(a['false_new_reworded'])} | "
            f"{_fmt(a['missed_new_rate'])} | {a['seconds']:.2f} |")
    for name, why in result["skipped"].items():
        lines.append(f"| {name} | skipped: {why} | | | | | | | | |")
    lines += ["", "False-new is the novelty filter's hallucination rate: a "
              "duplicate called new. Model scores are a model's view of "
              "predictability, not a judgement of truth.", "",
              "Synthetic templates are easy: agreement here proves the code "
              "path, not the method. The real exit is agreement with Bill's "
              "blind ranking of his own documents."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# writing the report: the shared writer (experiments.report), else locally
# ---------------------------------------------------------------------------

TITLE = "Novelty filter — planted-truth ranking (plan §6 Phase 6, F1)"


def summary_lines(result: dict) -> list[str]:
    """The sentences a person reads first, one per arm."""
    out = [f"{result['documents']} documents against a project of "
           f"{result['project_sentences']} sentences (seed {result['seed']}); "
           f"truth: {result.get('truth_kind', 'the planted count of new facts')}.",
           "Agreement: Spearman's rho and Kendall's tau-b between each arm's "
           "document scores and the truth. Sentence AUC separates new from "
           "redundant sentences; false-new (a duplicate called new) is the "
           "novelty filter's hallucination rate."]
    for a in result["arms"].values():
        k = a["auc_by_kind"]
        out.append(
            f"{a['arm']} ({a['tier']}): Spearman {_fmt(a['spearman'])}, "
            f"Kendall τb {_fmt(a['kendall_tau_b'])}, sentence AUC "
            f"{_fmt(a['sentence_auc'])} (novel entity "
            f"{_fmt(k['new:novel-entity'])}, recombination "
            f"{_fmt(k['new:recombination'])}), false-new "
            f"{_fmt(a['false_new_rate'])} (reworded "
            f"{_fmt(a['false_new_reworded'])}), missed-new "
            f"{_fmt(a['missed_new_rate'])}, {a['seconds']:.2f} s.")
    for name, why in result["skipped"].items():
        out.append(f"{name}: skipped — {why}")
    out.append("Model scores are a model's view of predictability, not a "
               "judgement of truth. Synthetic templates are easy: agreement "
               "here proves the code path, not the method; the real exit is "
               "agreement with Bill's blind ranking of his own documents.")
    return out


def _write(out_dir, name: str, result: dict, lines: list[str],
           markdown: str) -> dict:
    """The shared writer when it is importable (markdown + JSON, MEASURED,
    with the provenance stamp), plus this experiment's own table beside it;
    a minimal local writer otherwise."""
    out = Path(out_dir)
    try:
        from atk_diffusion.experiments.report import write_report
    except ImportError:
        write_report = None
    note = ""
    if write_report is not None:
        try:
            md, js = write_report(out, name, result, lines, title=TITLE)
            table = out / f"{name}_table.md"
            table.write_text(markdown, encoding="utf-8")
            return {"writer": "atk_diffusion.experiments.report.write_report",
                    "markdown": str(md), "json": str(js), "table": str(table)}
        except Exception as e:                               # noqa: BLE001
            note = (f"the shared report writer failed ({e}); the report was "
                    "written locally instead")
    out.mkdir(parents=True, exist_ok=True)
    jp, mp = out / f"{name}.json", out / f"{name}.md"
    jp.write_text(json.dumps({"name": name, "tier": "measured",
                              "summary": lines, "result": result}, indent=2,
                             default=_jsonable), encoding="utf-8")
    mp.write_text(markdown + "\n" + "\n".join(f"- {x}" for x in lines) + "\n",
                  encoding="utf-8")
    return {"writer": "local", "json": str(jp), "markdown": str(mp),
            "note": note or "atk_diffusion.experiments.report is not "
                            "importable; wrote a minimal report locally"}


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


# ---------------------------------------------------------------------------
# the experiment
# ---------------------------------------------------------------------------

def run(out_dir=None, *, seed: int = 0, n_docs: int = 10,
        doc_sentences: int = 5, n_project: int = 12, llama_model=None,
        arms=("toy_masked", "toy_pmi", "llama_pmi", "classical_tfidf",
              "classical_containment"),
        data: dict | None = None, llama_kw: dict | None = None,
        progress: Callable[[str], None] | None = None) -> dict:
    """Run every arm on the planted corpus, or on `data`:

        {"project": [texts], "documents": [texts or {"name", "text"}],
         "truth": [one number per document — higher adds more; an
                   analyst's blind ranking or count],
         "labels": optional per-sentence kinds ("new…"/"dup…"),
         "background": optional corpus for the toy (else the project),
         "truth_kind": optional words for the report}

    `llama_kw` goes to LlamaCppAR (e.g. {"n_gpu_layers": -1}). Writes the
    report into `out_dir` when one is given; nothing is written otherwise."""
    from atk_diffusion.text.backends import (BackendUnavailable, LlamaCppAR,
                                             ToyMaskedLM)
    say = progress or (lambda _m: None)
    data = data or synthetic_corpus(seed, n_project=n_project, n_docs=n_docs,
                                    doc_sentences=doc_sentences)
    from atk_diffusion.text.backends import split_sentences
    psents = sum(len(split_sentences(p)) for p in data["project"])
    results, skipped = {}, {}
    toy = None
    notes = []
    background = data.get("background") or []
    if not background and any(a.startswith("toy") for a in arms):
        background = list(data["project"])
        notes.append("no background corpus was given: the toy was trained on "
                     "the project itself, so its 'alone' scores already know "
                     "the project")
    for arm in arms:
        say(f"arm: {arm}")
        if arm in ("toy_masked", "toy_pmi"):
            toy = toy or ToyMaskedLM(background)
            mode = "masked" if arm == "toy_masked" else "pmi"
            results[arm] = _arm(arm, toy, mode, data, progress)
        elif arm == "classical_tfidf":
            results[arm] = _arm(arm, None, "tfidf", data, progress)
        elif arm == "classical_containment":
            results[arm] = _arm(arm, None, "containment", data, progress)
        elif arm == "llama_pmi":
            if not llama_model:
                skipped[arm] = "no GGUF model path was given (llama_model=)"
                continue
            try:
                be = LlamaCppAR(llama_model, **(llama_kw or {}))
            except BackendUnavailable as e:
                skipped[arm] = str(e)
                continue
            results[arm] = _arm(arm, be, "pmi", data, progress)
        else:
            skipped[arm] = "unknown arm"
    result = {"experiment": "novelty_eval", "seed": seed,
              "documents": len(data["documents"]),
              "doc_sentences": doc_sentences, "project_sentences": psents,
              "truth": list(data["truth"]), "arms": results,
              "skipped": skipped, "notes": notes,
              "truth_kind": data.get("truth_kind", "the planted number of "
                                     "new facts per document"),
              "meaning": ("ranking agreement with the planted count of new "
                          "facts; a model's view of predictability, not a "
                          "judgement of truth")}
    md = report_md(result)
    lines = summary_lines(result)
    result["report_md"] = md
    result["summary"] = lines
    if out_dir is not None:
        result["report"] = _write(out_dir, "novelty_eval", result, lines, md)
    else:
        result["report"] = {"writer": None,
                            "note": "not written: no output folder was given"}
    return result
