# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The domain gap — the one number every RF track reports (plan §7, §3.5;
DETECTION_DESIGN §6.5, §11).

*"Trained on synthetic at the profile's rate, tested on cabled real captures
from that receiver. Augmentation, impairment models and profile discipline
are all judged by whether they shrink it."*

    r = domain_gap(rf, profile, model_dir, synthetic_dataset, cabled_dataset)

The same saved model — through its ONNX graph, as ATK will run it — is
scored on held-out synthetic data and on a cabled set of the same profile,
with the same truth rules (the card's box policy for the proposer, the
card's classes for the classifier):

* proposer2d — mAP@0.5 (families counted), and AP@0.5 with families aside
  (*where* without *what*); the gap is synthetic − cabled for each.
* classifier1d — accuracy and macro-F1 on the classes the model knows.

A positive gap is how much the model loses between the synthetic world and
the real receiver. The result is folded into the card (`metrics.
domain_gap`, `map_synthetic` / `map_cabled` or `accuracy_synthetic` /
`accuracy_cabled`, and the detail) — the AI Detect tab shows `domain_gap`
— and a report goes under `rf_data\\<profile>\\runs\\`.

HONEST LIMITS. A cabled set is real receiver impairment with perfect labels,
not a real environment: the over-the-air gap (multipath, interference,
unlabeled neighbours) is the minutes-to-acceptable experiment's question.
When the second dataset's manifest does not say it is cabled, the report
says the number is a gap between two datasets, not THE domain gap. Runs
without PyTorch (ONNX Runtime + numpy).
"""

from __future__ import annotations

from atk_diffusion import cards as _cards
from atk_diffusion.experiments.detector_eval import (evaluate_classifier,
                                                      evaluate_proposer)
from atk_diffusion.experiments.report import experiment_dir, write_report
from atk_diffusion.learn import export as X


def _is_cabled(summary: dict) -> bool:
    return (summary.get("generator") == "cabled"
            or "cabled" in (summary.get("label_sources") or []))


def _gap(a, b):
    return (a - b) if (a is not None and b is not None) else None


def domain_gap(rf, profile: str, model_dir, synthetic_dataset, cabled_dataset,
               *, split: str = "test", cabled_split: str = "all",
               update_card: bool = True, threads: int | None = 1,
               out_name: str = "domain_gap", progress=None) -> dict:
    """Score `model_dir` on both datasets and report the gap (module
    docstring). Returns the result dict with "report" paths."""
    card = _cards.load(model_dir, for_profile=profile)
    if card.kind not in ("proposer2d", "classifier1d"):
        raise ValueError(f"the domain gap is measured for the proposer and the "
                         f"classifier; {card.name} is a {card.kind}.")
    runner = X.OnnxRunner(model_dir, card.kind, for_profile=profile,
                          threads=threads)
    lines = []
    if card.kind == "proposer2d":
        s, _ = evaluate_proposer(runner, synthetic_dataset, profile, split, rf)
        c, _ = evaluate_proposer(runner, cabled_dataset, profile, cabled_split, rf)
        gap = _gap(s["map50"], c["map50"])
        gap_any = _gap(s["ap50_any"], c["ap50_any"])
        res = {"kind": card.kind, "model": card.name, "metric": "mAP@0.5",
               "synthetic": s, "cabled": c, "gap": gap,
               "gap_ap50_any": gap_any}
        lines.append(f"{card.name} (the 2D proposer) scored {_f(s['map50'])} "
                     f"mAP@0.5 on held-out synthetic tiles ({s['dataset']}, "
                     f"{s['tiles']} tiles) and {_f(c['map50'])} on the cabled "
                     f"set ({c['dataset']}, {c['tiles']} tiles).")
        lines.append(f"Domain gap: {_f(gap)} mAP@0.5 (families counted); "
                     f"{_f(gap_any)} AP@0.5 with families aside.")
        fold = {"map_synthetic": s["map50"], "map_cabled": c["map50"],
                "domain_gap": gap}
    else:
        s, _ = evaluate_classifier(runner, synthetic_dataset, profile, split, rf)
        c, _ = evaluate_classifier(runner, cabled_dataset, profile, cabled_split, rf)
        gap = _gap(s.get("accuracy"), c.get("accuracy"))
        gap_f1 = _gap(s.get("macro_f1"), c.get("macro_f1"))
        res = {"kind": card.kind, "model": card.name, "metric": "accuracy",
               "synthetic": s, "cabled": c, "gap": gap, "gap_macro_f1": gap_f1}
        lines.append(f"{card.name} (the 1D classifier) scored accuracy "
                     f"{_f(s.get('accuracy'))} on held-out synthetic cuts "
                     f"({s['dataset']}, {s['known_cuts']} cuts) and "
                     f"{_f(c.get('accuracy'))} on the cabled set ({c['dataset']}, "
                     f"{c['known_cuts']} cuts).")
        lines.append(f"Domain gap: {_f(gap)} accuracy; {_f(gap_f1)} macro-F1.")
        fold = {"accuracy_synthetic": s.get("accuracy"),
                "accuracy_cabled": c.get("accuracy"), "domain_gap": gap}
    if not _is_cabled(c):
        res["note"] = ("the second dataset's manifest does not say it is cabled "
                       f"(generator {c.get('generator')!r}); this is a gap "
                       "between two datasets, not the domain gap")
        lines.append("Note: " + res["note"] + ".")
    if _is_cabled(s):
        lines.append("Note: the first dataset says it is cabled, not synthetic.")
    if gap is not None:
        lines.append("A positive gap is what the model loses between synthetic "
                     "data and the real receiver; augmentation, impairment "
                     "models and self-supervised pretraining are kept only if "
                     "they shrink it (plan §7).")
    if update_card:
        met = dict(card.metrics or {})
        met.update(fold)
        met["domain_gap_detail"] = {k: res[k] for k in res
                                    if k not in ("synthetic", "cabled")}
        met["domain_gap_detail"]["synthetic_dataset"] = s["dataset"]
        met["domain_gap_detail"]["cabled_dataset"] = c["dataset"]
        card.metrics = met
        X.resave_card(model_dir, card, rf=rf)
        lines.append(f"Folded into {card.name}'s card.")
    out = experiment_dir(rf, profile, out_name)
    md, js = write_report(out, out_name, res, lines,
                          title=f"Domain gap — {card.name}", rf=rf)
    res["report"] = {"markdown": str(md), "json": str(js)}
    if progress:
        for x in lines:
            progress(x)
    return res


def _f(v) -> str:
    return "n/a" if v is None else f"{float(v):.3f}"
