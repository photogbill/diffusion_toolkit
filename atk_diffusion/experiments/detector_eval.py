# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The detector's evaluation harness (DETECTION_DESIGN §11, §6.5; plan §7).

§11, item by item, and where each number comes from:

* **Detection versus SNR, per class** — *"the curve that is the product"*.
  The proposer's boxes at its operating threshold (the card's
  `calibration.min_score`) against every labelled box, by class and 5 dB
  SNR bin; a box is found when one overlaps it at IoU ≥ 0.5 (family aside:
  the question is *was it found*). SNR is the dataset's per-box SNR when it
  carries one (`box_snr_db`), else measured in the tile (`common.box_snr_db`).
* **mAP on the cabled set and on held-out synthetic, reported separately** —
  their difference is the domain gap (`experiments.domain_gap` folds it
  into the card).
* **False alarms per hour on empty captures** — boxes the proposer puts on
  noise-only tiles (terminated input, quiet bands), at its operating
  threshold, divided by the stream time those tiles cover.
* **Unknown rejection** — cuts of classes the classifier never trained on,
  through the ONNX and the prototype bank beside the card: how often they
  are called UNKNOWN rather than forced into a class.
* **Confirmation agreement** — how often the classifier's class matches the
  decoder's, from confirmed detections (`detect.boxes.Detection` JSON — a
  disagreement is the `disagreement` flag with `classifier_said` kept, by
  the core's rule) or from records that name both classes.
* **Latency** per tile and per cutout, on the CPU through ONNX Runtime.
* **Baselines**: *"CFAR alone; the energy detector ATK already has"*.
  `energy_baseline` is the energy detector in TILE space — a threshold in
  dB above the floor on a lightly smoothed tile, connected regions as boxes,
  scored by exactly the same box rule (`common.widen_boxes`) and truth. It
  is the energy idea applied to the very tiles the learned proposer sees —
  not ATK's CFAR proposer on IQ (`dsp.cfar`), which works on the PSD before
  tiling and is not scored here. The learned proposer is compared with it on
  AP (families aside), false alarms, and detection vs SNR; a learned tool
  that does not beat it is not shipped (plan §7).

NO PYTORCH. Everything here runs through ONNX Runtime and numpy/scipy, so
the harness also runs in ATK's core environment — the same numbers can be
re-measured on Bill's machine with the models that will actually run there.

`evaluate_detector(...)` runs whatever it is given (any subset of the
models and datasets) and writes one report under `rf_data\\<profile>\\runs\\`,
and — unless told not to — folds the card numbers §6.5 asks for (false
alarms per hour, detection vs SNR, unknown rejection, latency) into the
models' cards.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from atk_diffusion import profiles as _profiles
from atk_diffusion.detect import classes as _classes
from atk_diffusion.experiments.report import experiment_dir, write_report
from atk_diffusion.learn import common as K
from atk_diffusion.learn import export as X

DEFAULT_MIN_SCORE = 0.3          # what detect.onnx_models uses without a card value


# ---------------------------------------------------------------------------
# Shared evaluation of one model on one dataset (used by domain_gap and
# minutes_to_acceptable too)
# ---------------------------------------------------------------------------
def _splits(dataset_dir, kind: str, split: str) -> tuple[list[str], str]:
    have = [s for s in K.SPLITS if K.split_files(dataset_dir, s, kind)]
    if split == "all":
        return have, ""
    if split in have:
        return [split], ""
    if not have:
        return [], "the dataset has no examples in any split"
    return have, (f"the dataset has no {split!r} split; all of its splits "
                  f"({', '.join(have)}) were used")


def min_score(card) -> float:
    cal = card.calibration or {}
    return float(cal.get("min_score", (card.input or {}).get("min_score",
                                                             DEFAULT_MIN_SCORE)))


def evaluate_proposer(runner: X.OnnxRunner, dataset_dir, profile: str,
                      split: str = "test", rf=None, score_thr: float | None = None
                      ) -> tuple[dict, dict]:
    """A proposer on a wideband dataset of this profile, through its ONNX.
    Returns (summary, raw) — raw holds the predictions and truth for reuse."""
    card = runner.card
    m = K.open_dataset(dataset_dir, profile, "wideband", rf=rf)
    sps, note = _splits(dataset_dir, "wideband", split)
    policy = K.BoxPolicy.from_json(card.input.get("boxes") or {})
    fams = list(card.input.get("families") or _classes.FAMILIES)
    preds, gts, snrs, names = [], [], [], []
    for sp in sps:
        tiles = K.TileSet(dataset_dir, sp, m)
        g, s, n = K.proposer_truth(tiles, m, policy)
        preds += X.run_proposer(runner, tiles)
        gts += g
        snrs += s
        names += n
    thr = min_score(card) if score_thr is None else float(score_thr)
    fam = K.detection_map(preds, gts, len(fams))
    anyc = K.detection_map(preds, gts, 1, class_agnostic=True)
    op = K.operating_point(preds, gts, thr)
    summ = {"dataset": str(m.get("name")), "splits": sps,
            "generator": m.get("generator", ""),
            "label_sources": list(m.get("label_sources") or []),
            "tiles": len(gts), "boxes": int(sum(len(x["boxes"]) for x in gts)),
            "map50": K.finite_or_none(fam["map50"]),
            "map50_95": K.finite_or_none(fam["map50_95"]),
            "ap50_any": K.finite_or_none(anyc["map50"]),
            "ap50_per_family": {fams[c]: K.finite_or_none(v)
                                for c, v in fam["ap50"].items() if np.isfinite(v)},
            "operating_point": {k: (K.finite_or_none(v) if isinstance(v, float)
                                    else v) for k, v in op.items()}}
    if note:
        summ["note"] = note
    raw = {"preds": preds, "gts": gts, "snrs": snrs, "names": names,
           "manifest": m, "policy": policy, "threshold": thr, "splits": sps}
    return summ, raw


def evaluate_classifier(runner: X.OnnxRunner, dataset_dir, profile: str,
                        split: str = "test", rf=None, snr_bin_db: float = 5.0
                        ) -> tuple[dict, dict]:
    """A classifier on a narrowband dataset of this profile and canonical
    rate, through its ONNX. Cuts of classes the model does not have are
    counted, not scored (they are the unknown-rejection set)."""
    card = runner.card
    m = K.open_dataset(dataset_dir, profile, "narrowband", rf=rf)
    rate = float((m.get("canonical") or {}).get("rate") or 0)
    want = float((card.input.get("canonical") or {}).get("rate") or 0)
    if want and rate and abs(rate - want) > 0.5:
        raise K.DatasetRefused(f"{card.name} classifies cuts at {want:g} S/s; "
                               f"the dataset {m.get('name')!r} is at {rate:g} "
                               "S/s. A network never meets a rate it was not "
                               "trained at.")
    sps, note = _splits(dataset_dir, "narrowband", split)
    names = card.class_names()
    mnames = list(m.get("classes") or [])
    lut = {i: (names.index(n) if n in names else -1) for i, n in enumerate(mnames)}
    outs, labs, snr, raw_names = [], [], [], []
    for sp in sps:
        sh = K.ShardSet(dataset_dir, sp, m)
        if not len(sh):
            continue
        outs.append(X.run_classifier(runner, sh))
        lab = sh.labels["label"]
        labs.append(np.asarray([lut.get(int(v), -1) for v in lab]))
        snr.append(sh.labels["snr_db"])
        raw_names += [mnames[int(v)] if 0 <= int(v) < len(mnames) else "?"
                      for v in lab]
    if not outs:
        raise K.DatasetRefused(f"the dataset {m.get('name')!r} has no cuts to "
                               "score.")
    o = {k: np.concatenate([x[k] for x in outs]) for k in outs[0]}
    y = np.concatenate(labs)
    s = np.concatenate(snr)
    known = y >= 0
    pred = o["logits"].argmax(axis=1)
    summ = {"dataset": str(m.get("name")), "splits": sps,
            "generator": m.get("generator", ""),
            "label_sources": list(m.get("label_sources") or []),
            "cuts": int(len(y)), "known_cuts": int(known.sum()),
            "unknown_class_cuts": int((~known).sum())}
    if np.any(known):
        summ.update(K.classification_scores(pred[known], y[known], names,
                                            s[known], snr_bin_db))
        T = float((card.calibration or {}).get("temperature", 1.0) or 1.0)
        p = K.softmax(o["logits"][known] / T)
        from atk_diffusion.learn import calibrate as _cal
        summ["ece_calibrated"] = _cal.ece(p.max(axis=1),
                                          p.argmax(axis=1) == y[known])
        summ["temperature"] = T
    if note:
        summ["note"] = note
    raw = {"outputs": o, "labels": y, "snr": s, "names": raw_names,
           "manifest": m, "splits": sps}
    return summ, raw


# ---------------------------------------------------------------------------
# The energy baseline in tile space
# ---------------------------------------------------------------------------
def energy_baseline(spec_db, threshold_db: float = 6.0, smooth: int = 3,
                    min_area: int = 6, min_box_px: float = 0.0,
                    max_boxes: int = 200) -> dict:
    """Boxes where the (linearly) smoothed tile stands `threshold_db` above
    the floor: connected regions of at least `min_area` pixels. Score: a
    logistic of the region's mean dB above the threshold (it only ranks).
    Labels are 0 — energy knows no family."""
    from scipy import ndimage
    a = np.asarray(spec_db, np.float64)
    lin = 10.0 ** (a / 10.0)
    if smooth and smooth > 1:
        lin = ndimage.uniform_filter(lin, size=int(smooth), mode="nearest")
    mask = 10.0 * np.log10(np.maximum(lin, 1e-12)) > float(threshold_db)
    lab, n = ndimage.label(mask)
    boxes, scores = [], []
    for i, sl in enumerate(ndimage.find_objects(lab)):
        if sl is None:
            continue
        reg = lab[sl] == (i + 1)
        if int(reg.sum()) < int(min_area):
            continue
        mean_db = 10.0 * np.log10(float(np.mean(lin[sl][reg])))
        boxes.append([sl[0].start, sl[1].start, sl[0].stop, sl[1].stop])
        scores.append(1.0 / (1.0 + np.exp(-(mean_db - threshold_db) / 3.0)))
    b = np.asarray(boxes, np.float32).reshape(-1, 4)
    if min_box_px:
        b = K.widen_boxes(b, min_box_px, a.shape[0], a.shape[1])
    sc = np.asarray(scores, np.float32)
    order = np.argsort(-sc)[: int(max_boxes)]
    return {"boxes": b[order], "scores": sc[order],
            "labels": np.zeros(len(order), np.int64)}


def run_energy(tiles, threshold_db: float = 6.0, min_box_px: float = 0.0,
               **kw) -> list[dict]:
    return [energy_baseline(tiles.spec(i), threshold_db, min_box_px=min_box_px,
                            **kw) for i in range(len(tiles))]


# ---------------------------------------------------------------------------
# False alarms, unknown rejection, confirmation agreement
# ---------------------------------------------------------------------------
def tile_step_seconds(manifest: dict, profile: str) -> float:
    """Stream seconds each new tile adds: tile_seconds × (1 − overlap), from
    the dataset's STFT geometry, else the profile's default geometry."""
    st = manifest.get("stft") or {}
    if st.get("tile_seconds"):
        return float(st["tile_seconds"]) * (1.0 - float(st.get("tile_overlap", 0.0)))
    g = _profiles.default_stft(_profiles.parse_profile_id(profile).sample_rate)
    return float(g.tile_seconds) * (1.0 - float(g.tile_overlap))


def false_alarms(preds, step_s: float, score_thr: float = 0.0) -> dict:
    """Boxes scoring at least `score_thr` on noise-only tiles, per hour of
    stream those tiles cover."""
    n = sum(int(np.sum(np.asarray(p["scores"]) >= score_thr)) for p in preds)
    hours = len(preds) * float(step_s) / 3600.0
    return {"count": n, "tiles": len(preds), "hours": hours,
            "per_hour": (n / hours) if hours > 0 else None,
            "threshold": float(score_thr)}


def unknown_rejection(runner: X.OnnxRunner, model_dir, dataset_dir,
                      profile: str, split: str = "all", rf=None) -> dict:
    """Cuts of classes the classifier never trained on, called UNKNOWN by the
    prototype bank beside its card — and the false-unknown rate on the
    known classes in the same cuts."""
    from atk_diffusion.learn import calibrate as _cal
    card = runner.card
    canon = (card.input.get("canonical") or {})
    bank, impl = _cal.load_bank(model_dir, profile,
                                canon.get("class") or canon.get("cls"))
    _s, raw = evaluate_classifier(runner, dataset_dir, profile, split, rf)
    y = raw["labels"]
    emb = raw["outputs"]["embedding"]
    calls = [c for c, _d, _t in _cal._calls(bank, emb)]
    unk = y < 0
    names = card.class_names()
    out = {"implementation": impl, "dataset": raw["manifest"].get("name"),
           "splits": raw["splits"], "n_unknown": int(unk.sum()),
           "n_known": int((~unk).sum())}
    if unk.any():
        out["unknown_rejection"] = float(np.mean([calls[i] == _classes.UNKNOWN
                                                  for i in np.where(unk)[0]]))
        held = sorted({raw["names"][i] for i in np.where(unk)[0]})
        out["held_out_classes"] = held
    else:
        out["unknown_rejection"] = None
        out["note"] = ("every cut in this dataset is of a class the model "
                       "knows; give a dataset with classes it never saw")
    if (~unk).any():
        kn = np.where(~unk)[0]
        out["false_unknown_rate"] = float(np.mean([calls[i] == _classes.UNKNOWN
                                                   for i in kn]))
        acc = [calls[i] == names[int(y[i])] for i in kn
               if calls[i] != _classes.UNKNOWN]
        out["accuracy_on_accepted"] = float(np.mean(acc)) if acc else None
    return out


def _read_logs(logs) -> list:
    items = []
    for lg in logs if isinstance(logs, (list, tuple)) else [logs]:
        if isinstance(lg, (str, Path)) and Path(lg).exists():
            text = Path(lg).read_text(encoding="utf-8").strip()
            if text.startswith("["):
                items += json.loads(text)
            else:
                for line in text.splitlines():
                    line = line.strip()
                    if line:
                        try:
                            items.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        elif hasattr(lg, "to_json"):
            items.append(lg.to_json())
        elif isinstance(lg, dict):
            items.append(lg)
    return items


def confirmation_agreement(logs) -> dict:
    """How often the classifier's class matches the decoder's (§11).

    Accepts JSON-lines / JSON-array files, Detection objects or dicts:
    * records naming both — {"classifier_class", "decoder_class"} — are
      counted exactly;
    * confirmed Detection JSON (state "confirmed"): a `disagreement` flag
      with `measurements.classifier_said` is a disagreement; otherwise an
      agreement — the decoder was chosen by the classifier's class and
      decoded it. HONEST LIMIT: a decoder that states no class cannot
      contradict, so it counts as agreement; records with both classes do
      not have that weakness."""
    agree = disagree = skipped = 0
    by_class: dict[str, dict] = {}
    pairs = []
    for r in _read_logs(logs):
        if not isinstance(r, dict):
            skipped += 1
            continue
        if "classifier_class" in r and "decoder_class" in r:
            c, d = str(r["classifier_class"]), str(r["decoder_class"])
            if not c or not d:
                skipped += 1
                continue
            ok = c == d
        elif r.get("state") == "confirmed":
            meas = r.get("measurements") or {}
            if "disagreement" in (r.get("flags") or []) and meas.get("classifier_said"):
                c, d, ok = str(meas["classifier_said"]), str(r.get("cls", "")), False
            elif r.get("cls"):
                c = d = str(r["cls"])
                ok = True
            else:
                skipped += 1
                continue
        else:
            skipped += 1
            continue
        agree += int(ok)
        disagree += int(not ok)
        e = by_class.setdefault(d, {"agree": 0, "disagree": 0})
        e["agree" if ok else "disagree"] += 1
        if not ok:
            pairs.append({"classifier": c, "decoder": d})
    n = agree + disagree
    return {"agreement": (agree / n) if n else None, "confirmed": n,
            "agree": agree, "disagree": disagree, "skipped": skipped,
            "by_decoder_class": by_class, "disagreements": pairs[:50]}


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------
def evaluate_detector(rf, profile: str, *, proposer_dir=None,
                      classifier_dir=None, synthetic_dataset=None,
                      cabled_dataset=None, noise_dataset=None,
                      classifier_dataset=None, unknown_dataset=None,
                      confirm_logs=None, split: str = "test",
                      energy_threshold_db: float = 6.0, snr_bin_db: float = 5.0,
                      threads: int | None = None, latency_repeats: int = 20,
                      update_cards: bool = True,
                      out_name: str = "detector_eval", progress=None) -> dict:
    """Run DETECTION_DESIGN §11 on whatever is given and write one report.

    proposer_dir / classifier_dir  saved models of this profile
    synthetic_dataset              wideband, held-out synthetic (split)
    cabled_dataset                 wideband, cabled (every split)
    noise_dataset                  wideband tiles of empty captures; when
                                   absent, the tiles WITHOUT boxes of the
                                   given datasets are used, and the report
                                   says so
    classifier_dataset             narrowband, for accuracy (split)
    unknown_dataset                narrowband with classes the classifier
                                   never trained on (default:
                                   classifier_dataset)
    confirm_logs                   paths / Detection JSONs (see
                                   confirmation_agreement)
    Returns the result dict, with "report": {"markdown", "json"}."""
    def say(msg):
        lines.append(msg)
        if progress:
            progress(msg)

    lines: list[str] = []
    res: dict = {"profile": profile, "profile_words": _profiles.describe(profile)}
    if proposer_dir is not None:
        pr = X.OnnxRunner(proposer_dir, "proposer2d", for_profile=profile,
                          threads=threads)
        card = pr.card
        thr = min_score(card)
        mb = float((card.input.get("boxes") or {}).get("min_box_px", 0.0))
        P: dict = {"model": card.name, "operating_threshold": thr}
        sets = []
        if synthetic_dataset is not None:
            sets.append(("synthetic", synthetic_dataset, split))
        if cabled_dataset is not None:
            sets.append(("cabled", cabled_dataset, "all"))
        empty_preds, empty_energy, step = [], [], None
        curves = {}
        for tag, ds, sp in sets:
            summ, raw = evaluate_proposer(pr, ds, profile, sp, rf, thr)
            P[tag] = summ
            curves[tag] = K.detection_vs_snr(raw["preds"], raw["gts"],
                                             raw["snrs"], raw["names"], thr,
                                             width_db=snr_bin_db)
            # the energy baseline on the very same tiles and truth
            epreds = []
            for s_ in raw["splits"]:
                epreds += run_energy(K.TileSet(ds, s_, raw["manifest"]),
                                     energy_threshold_db, min_box_px=mb)
            eap = K.detection_map(epreds, raw["gts"], 1, class_agnostic=True)
            P.setdefault("energy_baseline", {})[tag] = {
                "ap50_any": K.finite_or_none(eap["map50"]),
                "operating_point": K.operating_point(epreds, raw["gts"], 0.0),
                "detection_vs_snr": K.detection_vs_snr(
                    epreds, raw["gts"], raw["snrs"], raw["names"], 0.0,
                    width_db=snr_bin_db)["_all"],
                "threshold_db": energy_threshold_db}
            say(f"proposer on {tag} ({summ['dataset']}, {summ['tiles']} tiles): "
                f"mAP@0.5 {_f(summ['map50'])}, AP@0.5 families aside "
                f"{_f(summ['ap50_any'])}; the energy baseline "
                f"{_f(P['energy_baseline'][tag]['ap50_any'])}")
            if noise_dataset is None:
                for i, g in enumerate(raw["gts"]):
                    if len(g["boxes"]) == 0:
                        empty_preds.append(raw["preds"][i])
                if step is None:
                    step = tile_step_seconds(raw["manifest"], profile)
                for s_ in raw["splits"]:
                    ts = K.TileSet(ds, s_, raw["manifest"])
                    for i, r in enumerate(ts.refs):
                        if len(r.boxes) == 0:
                            empty_energy.append(energy_baseline(
                                ts.spec(i), energy_threshold_db, min_box_px=mb))
        P["detection_vs_snr"] = curves
        if "synthetic" in P and "cabled" in P:
            a, b = P["synthetic"]["map50"], P["cabled"]["map50"]
            P["domain_gap"] = (a - b) if (a is not None and b is not None) else None
            say(f"domain gap (mAP@0.5, synthetic − cabled): {_f(P['domain_gap'])}")
        if noise_dataset is not None:
            mn = K.open_dataset(noise_dataset, profile, "wideband", rf=rf)
            step = tile_step_seconds(mn, profile)
            empty_preds, empty_energy = [], []
            for s_ in [s for s in K.SPLITS if K.split_files(noise_dataset, s,
                                                            "wideband")]:
                ts = K.TileSet(noise_dataset, s_, mn)
                empty_preds += X.run_proposer(pr, ts)
                empty_energy += run_energy(ts, energy_threshold_db, min_box_px=mb)
            src = f"the noise-only dataset {mn.get('name')!r}"
        else:
            src = "the tiles without labelled signals in the datasets above"
        if empty_preds:
            fa = false_alarms(empty_preds, step, thr)
            fe = false_alarms(empty_energy, step, 0.0)
            P["false_alarms"] = {"learned": fa, "energy": fe, "source": src}
            say(f"false alarms per hour on {src} ({fa['tiles']} tiles, "
                f"{fa['hours'] * 3600:.0f} s of stream): learned {_f(fa['per_hour'])}, "
                f"energy {_f(fe['per_hour'])}")
        else:
            P["false_alarms"] = {"note": "no noise-only tiles were available"}
            say("false alarms per hour: not measured — no noise-only tiles")
        tile = np.zeros((card.input["tile"]["rows"], card.input["tile"]["bins"]),
                        np.float32)
        P["latency"] = X.latency(pr.path, {"tile": tile[None, None]},
                                 repeats=latency_repeats, threads=threads)
        P["latency"]["budget_ms"] = 200.0
        say(f"proposer CPU latency {P['latency']['p50_ms']:.1f} ms a tile on "
            f"{P['latency']['machine']} (budget 200 ms on the 14-core machine)")
        ref = "synthetic" if "synthetic" in P else ("cabled" if "cabled" in P else None)
        if ref:
            ea = P["energy_baseline"][ref]["ap50_any"]
            la = P[ref]["ap50_any"]
            P["beats_energy"] = (bool(la > ea) if (la is not None and ea is not None)
                                 else None)
        res["proposer"] = P
        if update_cards:
            met = dict(card.metrics or {})
            fa = P["false_alarms"].get("learned") if isinstance(P["false_alarms"], dict) else None
            if fa and fa.get("per_hour") is not None:
                met["false_alarms_per_hour"] = fa["per_hour"]
                met["false_alarms"] = P["false_alarms"]
            if curves:
                met["detection_vs_snr"] = curves
            if "domain_gap" in P and P["domain_gap"] is not None:
                met["domain_gap"] = P["domain_gap"]
                met["map_cabled"] = P["cabled"]["map50"]
            met["energy_baseline"] = P.get("energy_baseline")
            card.metrics = met
            X.resave_card(proposer_dir, card, rf=rf)
    if classifier_dir is not None:
        cr = X.OnnxRunner(classifier_dir, "classifier1d", for_profile=profile,
                          threads=threads)
        ccard = cr.card
        C: dict = {"model": ccard.name}
        if classifier_dataset is not None:
            summ, _raw = evaluate_classifier(cr, classifier_dataset, profile,
                                             split, rf, snr_bin_db)
            C["heldout"] = summ
            say(f"classifier on {summ['dataset']} ({summ['known_cuts']} cuts of "
                f"known classes): accuracy {_f(summ.get('accuracy'))}, macro-F1 "
                f"{_f(summ.get('macro_f1'))}")
        uds = unknown_dataset or classifier_dataset
        if uds is not None:
            try:
                C["unknown"] = unknown_rejection(cr, classifier_dir, uds, profile,
                                                 "all" if unknown_dataset else split,
                                                 rf)
                u = C["unknown"]
                say(f"unknown rejection: {_f(u.get('unknown_rejection'))} on "
                    f"{u['n_unknown']} cuts of classes the classifier never saw"
                    f" ({u['implementation']}); known cuts called UNKNOWN: "
                    f"{_f(u.get('false_unknown_rate'))}")
            except FileNotFoundError as e:
                C["unknown"] = {"note": str(e)}
                say(str(e))
        iq_len = int(ccard.input.get("iq_len") or ccard.input.get("window"))
        shp = tuple(ccard.input.get("scf_shape") or (1, 1))
        C["latency"] = X.latency(cr.path, {"iq": np.zeros((1, 2, iq_len), np.float32),
                                           "scf": np.zeros((1, 1) + shp, np.float32)},
                                 repeats=latency_repeats, threads=threads)
        C["latency"]["budget_ms"] = 20.0
        say(f"classifier CPU latency {C['latency']['p50_ms']:.2f} ms a cutout "
            "(budget 20 ms)")
        res["classifier"] = C
        if update_cards:
            met = dict(ccard.metrics or {})
            u = C.get("unknown") or {}
            if u.get("unknown_rejection") is not None:
                met["unknown_rejection"] = u["unknown_rejection"]
                met["unknown_rejection_detail"] = u
            met["latency_measured"] = C["latency"]
            ccard.metrics = met
            X.resave_card(classifier_dir, ccard, rf=rf)
    if confirm_logs is not None:
        res["confirmation"] = confirmation_agreement(confirm_logs)
        c = res["confirmation"]
        say(f"confirmation agreement: {_f(c['agreement'])} over {c['confirmed']} "
            f"confirmed detections ({c['disagree']} disagreements)")
    if not lines:
        raise ValueError("nothing to evaluate: give a model and a dataset")
    out = experiment_dir(rf, profile, out_name)
    md, js = write_report(out, out_name, res, lines,
                          title="Detector evaluation (DETECTION_DESIGN §11)",
                          rf=rf)
    res["report"] = {"markdown": str(md), "json": str(js)}
    return res


def _f(v) -> str:
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)
