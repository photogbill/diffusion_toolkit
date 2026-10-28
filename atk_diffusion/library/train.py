# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""One call per model, from the training library to a model ATK loads — and
a verdict in plain words (RFMLS Training, 2026-10-26).

    train_recognizer(rf, profile, canonical, preset=…)   the 1D classifier
    train_finder(rf, profile, preset=…)                  the 2D proposer
    recognizer_verdict(card, manifest)                   ready / usable / not
    finder_verdict(card)                                 ready / not ready
    models_overview(rf, profile)                         what the tab lists
    retire(rf, profile, name) / bring_back(...)          take one out of use

WHAT "IN USE" MEANS. ATK's live detector (`diffusion_host.detector_for`)
loads the NEWEST card of each kind for a receiver — one classifier per rate
class, one proposer. So a model is in use when it is the newest of its kind
(and rate class), and RETIRING one moves its folder to `models\\_retired\\`
where `cards.find` does not look; the one before it is in use again. Nothing
is deleted; Bring back moves it home.

THE GRADE IS ON REAL HELD-OUT RECORDINGS (`build.build_recognizer_dataset`):
validation and test hold real windows only. The verdict says how strong
that grade is per type — a type graded on a separate kept-aside recording is
graded properly; one graded only on the held-back end of its own training
recording is graded weakly; one with nothing held out is not graded at all —
and it never calls a model ready on a weak grade.

THE VERDICT'S BARS (stated so they can be argued with):
  ready      ≥ 90 % right on real held-out windows, every type ≥ 80 %, every
             type properly graded, better than the classical baseline, and
             inside the CPU budget
  usable     ≥ 75 % overall and every type ≥ 50 %
  not ready  below that
  unmeasured nothing was held out

KEEP AI EXAMPLES ONLY IF THEY HELP (plan B4's rule). `compare_generated`
trains twice — with and without the AI-generated copies — on the same real
held-out windows, and retires whichever scores lower, saying so.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from atk_diffusion import cards as _cards
from atk_diffusion import profiles as _profiles
from atk_diffusion.library import build as _build
from atk_diffusion.library import store as _store

RETIRED = "_retired"

RECOGNIZER_PRESETS = {
    "quick": {"epochs": 4, "rf_config": None, "search_epochs": 0,
              "train_cap_per_type": 600,
              "words": "Quick test — a few minutes. Proves the whole chain "
                       "works; not a field model."},
    "normal": {"epochs": 30, "rf_config": "search", "search_epochs": 3,
               "train_cap_per_type": 3000,
               "words": "Normal — tens of minutes on the GPU. The usual choice."},
    "thorough": {"epochs": 60, "rf_config": "search", "search_epochs": 5,
                 "train_cap_per_type": 8000,
                 "words": "Thorough — an hour or more. When Normal was close."},
}
FINDER_PRESETS = {
    "quick": {"scenes": 40, "epochs": 4,
              "words": "Quick test — a few minutes; proves the chain works."},
    "normal": {"scenes": 200, "epochs": 24,
               "words": "Normal — about an hour on the GPU."},
    "thorough": {"scenes": 600, "epochs": 48,
                 "words": "Thorough — several hours."},
}

READY_ACC, READY_MIN_TYPE = 0.90, 0.80
USABLE_ACC, USABLE_MIN_TYPE = 0.75, 0.50
FINDER_READY_AP, FINDER_USABLE_AP = 0.70, 0.50


def _stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _report_dir(rf, profile: str, name: str) -> Path:
    d = Path(rf.runs(profile)) / "training_reports" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_report(rf, profile: str, name: str, result: dict) -> Path:
    d = _report_dir(rf, profile, name)
    (d / "result.json").write_text(json.dumps(result, indent=2, default=str),
                                   encoding="utf-8")
    v = result.get("verdict") or {}
    lines = [f"# {name}", "", f"**{v.get('headline', '')}**", ""]
    for r in v.get("reasons", []):
        lines.append(f"- {r}")
    if v.get("per_type"):
        lines += ["", "| type | real held-out windows | graded | right |",
                  "|---|---|---|---|"]
        for t in v["per_type"]:
            acc = t.get("accuracy")
            lines.append(f"| {t['type']} | {t.get('test_windows', 0)} | "
                         f"{t.get('grade', '')} | "
                         f"{'' if acc is None else f'{100 * acc:.0f} %'} |")
    if result.get("compare"):
        lines += ["", f"Compared with and without AI-generated examples: "
                      f"{result['compare'].get('words', '')}"]
    lines += ["", f"Model: {result.get('model_dir', '')}",
              f"Dataset: {result.get('dataset_dir', '')}"]
    p = d / "report.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        rf.record(p, "training-report", name)
        rf.record(d / "result.json", "training-report", name)
    except OSError:
        pass
    return p


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------
def real_accuracy(per_class: dict, counts: dict, names) -> tuple:
    """(accuracy over the REAL signal types only, windows it is over).

    The model's own `accuracy` includes the noise class, whose validation
    and test windows are synthetic noise at the receiver's floor (2026-10-26
    review: a type with nothing held out read "100 % right on 0 real
    windows", and 1-of-2 right on each of two types plus 8-of-8 noise read
    83 %). The grade is the per-type accuracies weighted by each type's
    REAL held-out windows — the noise class and a type held out to test
    UNKNOWN are not in it."""
    num = den = 0.0
    for name in names:
        if name == _build.NOISE:
            continue
        acc = (per_class or {}).get(name)
        n = sum(((counts.get(name) or {}).get("test") or {}).values())
        if acc is None or not n:
            continue
        num += float(acc) * n
        den += n
    return (num / den if den else None), int(den)


def recognizer_verdict(card, manifest: dict | None = None) -> dict:
    """ready / usable / not_ready / unmeasured, with the reasons in words."""
    m = card.metrics or {}
    per = m.get("per_class_accuracy") or {}
    lib = (manifest or {}).get("library") or {}
    counts = lib.get("counts") or {}
    reasons: list[str] = []
    rows = []
    weak, ungraded = [], []
    for name in card.class_names():
        if name == _build.NOISE:
            continue
        c = counts.get(name) or {}
        tw = sum((c.get("test") or {}).values())
        grade = "none"
        if tw:
            # a separate kept-aside recording, or only the held-back end?
            grade = "proper" if _has_kept_aside_test(manifest, name) else "weak"
        if grade == "weak":
            weak.append(name)
        if grade == "none":
            ungraded.append(name)
        rows.append({"type": name, "test_windows": tw, "grade": grade,
                     "accuracy": per.get(name) if tw else None})
    acc, n_real = real_accuracy(per, counts, card.class_names())
    if acc is None:
        level = "unmeasured"
        headline = ("Not graded — none of your real signals was held out to "
                    "test it on. Add a second real recording of each type.")
    else:
        mins = [r["accuracy"] for r in rows if r["accuracy"] is not None]
        worst = min(mins) if mins else None
        beats = m.get("beats_classical")
        lat = m.get("latency_ms")
        budget = m.get("latency_budget_ms") or 20.0
        if (acc >= READY_ACC and (worst is None or worst >= READY_MIN_TYPE)
                and not weak and not ungraded and beats is not False
                and (lat is None or lat <= budget)):
            level = "ready"
        elif acc >= USABLE_ACC and (worst is None or worst >= USABLE_MIN_TYPE):
            level = "usable"
        else:
            level = "not_ready"
        word = {"ready": "Ready for field use", "usable": "Usable with care",
                "not_ready": "Not ready"}[level]
        headline = (f"{word} — {100 * acc:.0f} % right on {n_real} real "
                    "held-out windows it never trained on.")
        if worst is not None and worst < READY_MIN_TYPE:
            bad = [r["type"] for r in rows if r["accuracy"] is not None
                   and r["accuracy"] < READY_MIN_TYPE]
            reasons.append(f"Weakest: {', '.join(bad)} — add more real examples "
                           "of " + ("it" if len(bad) == 1 else "them") + ".")
        if beats is False:
            base = (m.get("classical_baseline") or {}).get("accuracy")
            reasons.append("It does not beat the classical baseline"
                           + (f" ({100 * base:.0f} %)" if base is not None else "")
                           + " — the learned model is not earning its place yet.")
        if lat is not None and lat > budget:
            reasons.append(f"It takes {lat:.1f} ms a cut on the CPU; the budget "
                           f"is {budget:.0f} ms.")
    if weak:
        reasons.append("Graded only on the held-back end of its own training "
                       f"recording (a weak test): {', '.join(weak)}. A second "
                       "real recording of each makes it a real test.")
    if ungraded:
        reasons.append(f"Not graded at all: {', '.join(ungraded)} — nothing of "
                       "it was held out.")
    osr = ((getattr(card, "calibration", None) or {}).get("open_set") or {})
    if osr.get("unknown_rejection") is not None:
        reasons.append(f"Said UNKNOWN for {100 * osr['unknown_rejection']:.0f} % "
                       "of a type it was never taught (held back on purpose).")
    if level == "ready":
        reasons.insert(0, "Every type is graded on a separate real recording "
                          "and every one scores well.")
    noise_acc = per.get(_build.NOISE)
    if noise_acc is not None:
        reasons.append(f"Said NOISE for {100 * float(noise_acc):.0f} % of the "
                       "noise it was shown (synthetic noise at your receiver's "
                       "floor — checked, but not part of the grade).")
    return {"level": level, "headline": headline, "reasons": reasons,
            "per_type": rows, "accuracy": acc, "real_windows": n_real,
            "accuracy_with_noise": m.get("accuracy"),
            "macro_f1": m.get("macro_f1"), "latency_ms": m.get("latency_ms"),
            "beats_classical": m.get("beats_classical"),
            "accuracy_vs_snr": m.get("accuracy_vs_snr"),
            "confusion": m.get("confusion")}


def _val_accuracy(rf, model_dir, profile: str):
    """A model's accuracy on its VALIDATION split, over the real types only
    (weighted by each type's real validation windows), or None."""
    try:
        card = _cards.load(model_dir, for_profile=profile, verify_weights=False)
    except Exception:                                     # noqa: BLE001
        return None
    v = (card.metrics or {}).get("validation") or {}
    per = v.get("per_class_accuracy") or {}
    man = _manifest_of(rf, profile, card)
    counts = ((man or {}).get("library") or {}).get("counts") or {}
    num = den = 0.0
    for name in card.class_names():
        if name == _build.NOISE:
            continue
        n = sum(((counts.get(name) or {}).get("val") or {}).values())
        acc = per.get(name)
        if acc is None or not n:
            continue
        num += float(acc) * n
        den += n
    if den:
        return num / den
    return None


def _has_kept_aside_test(manifest, type_name: str) -> bool:
    """Whether a type's test windows include a separate kept-aside recording
    (`build` counts them per type as `test_sources`)."""
    src = ((manifest or {}).get("library") or {}).get("test_sources") or {}
    return bool(src.get(type_name))


def finder_verdict(card, manifest: dict | None = None) -> dict:
    m = card.metrics or {}
    ap = m.get("ap50_any")
    mp = m.get("map50")
    how = (((manifest or {}).get("library") or {}).get("heldout_pool") or "")
    reasons = [{
        "kept_aside": "Graded on held-out synthetic scenes with your KEPT-ASIDE "
                      "real signals placed in them — signals it never trained "
                      "on — not on a real recording of a band.",
        "held_back_tails": "Graded on held-out scenes with the held-back ends "
                           "of your training recordings placed in them — the "
                           "same recordings it trained on, so a weak test. "
                           "Keep a recording aside for a real one.",
        "reused": "Graded on scenes with the SAME signals it trained on "
                  "(nothing was kept aside) — this says little about new "
                  "signals.",
    }.get(how, "Graded on held-out synthetic scenes with your library's real "
               "signals placed in them — not on a real recording of a band.")
        + " Watch it on the waterfall before trusting it."]
    if ap is None:
        level, headline = "unmeasured", "Not graded — no held-out scenes."
    else:
        level = ("ready" if ap >= FINDER_READY_AP else
                 "usable" if ap >= FINDER_USABLE_AP else "not_ready")
        if level == "ready" and how in ("held_back_tails", "reused"):
            level = "usable"           # a weak test never earns READY
        word = {"ready": "Finds most signals", "usable": "Finds many signals",
                "not_ready": "Misses too many signals"}[level]
        headline = (f"{word} — {100 * ap:.0f} % average precision at finding "
                    "where a signal is" + (f"; {100 * mp:.0f} % when its family "
                                            "must be right too" if mp is not None else "")
                    + ".")
    lat = m.get("latency_ms")
    if lat is not None and lat > 200:
        reasons.append(f"It takes {lat:.0f} ms a tile on the CPU; the budget is "
                       "200 ms.")
    return {"level": level, "headline": headline, "reasons": reasons,
            "ap50_any": ap, "map50": mp, "latency_ms": lat,
            "per_family": m.get("ap50_per_family")}


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train_recognizer(rf, profile: str, canonical: str, *, preset: str = "normal",
                     include=_build.COPY_KINDS, types=None,
                     held_out: str | None = None, name: str | None = None,
                     compare_generated: bool = False, device: str = "auto",
                     seed: int = 0, threads: int | None = None,
                     epochs: int | None = None, window: int | None = None,
                     train_cap_per_type: int | None = None,
                     max_items: int | None = None, progress=None,
                     should_stop=None) -> dict:
    """Build the dataset, train the classifier, measure it, judge it.
    -> {model_dir, dataset_dir, verdict, report, compare?}"""
    from atk_diffusion.learn import classifier1d
    if preset not in RECOGNIZER_PRESETS:
        raise _store.LibraryError(f"a length is one of {', '.join(RECOGNIZER_PRESETS)}")
    pz = RECOGNIZER_PRESETS[preset]
    say = progress or (lambda _m: None)
    base = name or f"recognizer_{canonical}_{_stamp()}"
    runs = [(base, tuple(include))]
    if compare_generated and "generated" in include:
        runs = [(base + "_without_ai", tuple(k for k in include if k != "generated")),
                (base, tuple(include))]
    results = []
    for i, (nm, inc) in enumerate(runs):
        tag = f"({i + 1} of {len(runs)}) " if len(runs) > 1 else ""
        say(f"{tag}building the dataset {nm} from the library …")
        man = _build.build_recognizer_dataset(
            rf, profile, canonical, nm, types=types, include=inc,
            window=int(window or _store.WINDOW),
            train_cap_per_type=int(train_cap_per_type or pz["train_cap_per_type"]),
            seed=seed, progress=say, should_stop=should_stop)
        held = (held_out,) if held_out else ()
        say(f"{tag}training the recognizer {nm} ({pz['words'].split(' —')[0]}) …")
        d = classifier1d.train(
            rf, profile, man["path"], nm, int(epochs or pz["epochs"]),
            rf_config=pz["rf_config"], search_epochs=pz["search_epochs"],
            held_out_classes=held, device=device, seed=seed, threads=threads,
            max_items=max_items, progress=say)
        card = _cards.load(d, for_profile=profile)
        v = recognizer_verdict(card, man)
        results.append({"model_dir": str(d), "dataset_dir": man["path"],
                        "model": nm, "include": list(inc), "verdict": v})
    out = dict(results[-1])
    if len(results) == 2:
        a, b = results[0], results[1]
        # chosen on the VALIDATION split (real windows only); the test split
        # only reports — choosing on it would flatter the winner's grade
        acc_a = _val_accuracy(rf, a["model_dir"], profile)
        acc_b = _val_accuracy(rf, b["model_dir"], profile)
        if acc_a is not None and acc_b is not None and acc_a > acc_b:
            retire(rf, profile, b["model"])
            out = dict(a)
            words = (f"on the validation recordings, without them it scored "
                     f"{100 * acc_a:.0f} %, with them {100 * acc_b:.0f} % — the "
                     "AI examples did not help, so the model trained WITHOUT "
                     "them is the one in use (the other is retired, not "
                     "deleted)")
        else:
            if acc_a is not None:
                retire(rf, profile, a["model"])
            words = (f"on the validation recordings, with them it scored "
                     f"{'?' if acc_b is None else f'{100 * acc_b:.0f} %'}, without "
                     f"{'?' if acc_a is None else f'{100 * acc_a:.0f} %'} — the "
                     "model trained WITH the AI examples is the one in use"
                     + (" (no validation recordings to tell them apart)"
                        if acc_a is None or acc_b is None else ""))
        out["compare"] = {"with": acc_b, "without": acc_a, "words": words}
    out["report"] = str(_write_report(rf, profile, out["model"], out))
    say(f"[100%] {out['verdict']['headline']}")
    return out


def train_finder(rf, profile: str, *, preset: str = "normal",
                 scenes: int | None = None, use_library: bool = True,
                 name: str | None = None, device: str = "auto", seed: int = 0,
                 threads: int | None = None, epochs: int | None = None,
                 max_tiles: int | None = None, progress=None,
                 should_stop=None) -> dict:
    """Build the finder's scenes, train the proposer, judge it."""
    from atk_diffusion.learn import proposer2d
    if preset not in FINDER_PRESETS:
        raise _store.LibraryError(f"a length is one of {', '.join(FINDER_PRESETS)}")
    pz = FINDER_PRESETS[preset]
    say = progress or (lambda _m: None)
    nm = name or f"finder_{_stamp()}"
    say(f"building {int(scenes or pz['scenes'])} scenes for the finder …")
    man = _build.build_finder_dataset(rf, profile, nm, int(scenes or pz["scenes"]),
                                      use_library=use_library, seed=seed,
                                      progress=say, should_stop=should_stop)
    say("training the finder …")
    d = proposer2d.train(rf, profile, man["path"], nm, int(epochs or pz["epochs"]),
                         device=device, seed=seed, threads=threads,
                         max_tiles=max_tiles, progress=say)
    card = _cards.load(d, for_profile=profile)
    v = finder_verdict(card, man)
    out = {"model_dir": str(d), "dataset_dir": man["path"], "model": nm,
           "verdict": v}
    out["report"] = str(_write_report(rf, profile, nm, out))
    say(f"[100%] {v['headline']}")
    return out


# ---------------------------------------------------------------------------
# the models the tab lists
# ---------------------------------------------------------------------------
_KIND_WORDS = {"classifier1d": "recognizer", "proposer2d": "finder",
               "augmenter": "generator"}


def models_overview(rf, profile: str) -> list[dict]:
    """Every recognizer, finder and library generator of a receiver, newest
    first, with whether it is in use and its verdict."""
    out = []
    base = Path(rf.models(profile))
    seen_in_use = set()
    for retired, folder in ((False, base), (True, base / RETIRED)):
        if not folder.is_dir():
            continue
        items = []
        for d in folder.iterdir():
            if not (d / _cards.CARD_FILE).exists():
                continue
            try:
                card = _cards.load(d, for_profile=profile, verify_weights=False)
            except (_cards.CardRefusal, _profiles.ProfileMismatch, ValueError):
                continue
            if card.kind not in _KIND_WORDS:
                continue
            if card.kind == "augmenter" and not (card.input or {}).get("library"):
                continue
            items.append((d, card))
        items.sort(key=lambda t: t[1].created, reverse=True)
        for d, card in items:
            canon = ((card.input or {}).get("canonical") or {})
            cls = canon.get("cls") or canon.get("class") or ""
            key = (card.kind, cls)
            in_use = (not retired) and key not in seen_in_use
            if in_use:
                seen_in_use.add(key)
            man = _manifest_of(rf, profile, card)
            if card.kind == "classifier1d":
                v = recognizer_verdict(card, man)
            elif card.kind == "proposer2d":
                v = finder_verdict(card, man)
            else:
                v = {"level": "", "headline": "Makes AI-generated variations for "
                                              f"{cls}-width signals.",
                     "reasons": []}
            out.append({"name": card.name, "dir": str(d), "kind": card.kind,
                        "what": _KIND_WORDS[card.kind], "canonical": cls,
                        "created": card.created, "classes": card.class_names(),
                        "in_use": in_use, "retired": retired, "verdict": v,
                        "trained_on": getattr(card, "trained_on", "")})
    return out


def _manifest_of(rf, profile: str, card) -> dict | None:
    """The manifest of the dataset a card was trained on (cards record the
    dataset's name and hash, and datasets live by name under rf_data)."""
    for ds in card.datasets or []:
        nm = ds.get("name") if isinstance(ds, dict) else None
        if not nm:
            continue
        p = Path(rf.datasets(profile, nm)) / "manifest.json"
        if p.exists():
            try:
                return json.loads(p.read_text("utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
    return None


def retire(rf, profile: str, name: str) -> Path:
    """Take a model out of use: its folder moves to models\\_retired\\."""
    src = Path(rf.models(profile, name))
    if not (src / _cards.CARD_FILE).exists():
        raise _store.LibraryError(f"there is no model called '{name}' in use")
    dest_dir = Path(rf.models(profile)) / RETIRED
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        dest = dest_dir / f"{src.name}__{_stamp()}"
    shutil.move(str(src), str(dest))
    return dest


def bring_back(rf, profile: str, name: str) -> Path:
    src = Path(rf.models(profile)) / RETIRED / name
    if not (src / _cards.CARD_FILE).exists():
        raise _store.LibraryError(f"there is no retired model called '{name}'")
    dest = Path(rf.models(profile, name.split("__")[0]))
    if dest.exists():
        raise _store.LibraryError(f"a model called '{dest.name}' is in use; "
                                  "retire it first")
    shutil.move(str(src), str(dest))
    # newest first decides what is in use: a brought-back model is in use
    # only when it is the newest of its kind — say so in the result
    return dest
