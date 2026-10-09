# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Minutes to acceptable — the environment version of the domain gap
(plan §3.6, §7; decision D8).

Bill, 2026-10-08: *"generative AI can be used to make realistic signal
samples for training. That way you could train without going out to that
exact region with your equipment and loitering when you don't need to."*
Plan §3.6: *"On arrival, a short capture — minutes, from a vehicle, no
antenna farm — is the adaptation set. The detector is fine-tuned on it (or
its thresholds re-fitted), and the log records how many minutes it took to
close the gap to the acceptance line … The number to drive down is minutes
on site to acceptable — that is what 'not loitering' means, measured."*

    r = minutes_to_acceptable(rf, profile, model_dir, onsite_dataset,
                              acceptance=0.7, mode="thresholds")
    r["minutes_to_acceptable"]      # e.g. 2.0, or None: not reached

THE PROTOCOL. The on-site dataset (same profile; wideband for the proposer,
narrowband for the classifier) is a capture ORDERED IN TIME: its items
carry `t_s` (seconds since the session began), or are taken as consecutive
in storage order (the report says which). Its `adapt_split` is the stream
the model adapts on; its `eval_split` is a fixed later stretch it is scored
on, never adapted on (without one, the last 30 % of the stream in time is
held out). For each k in `minutes`, the model adapts on the first k minutes
and is scored; k = 0 is the model as it arrived — trained on the composed
region. The answer is the first k whose score crosses the acceptance line.

THE ADAPTATIONS, cheapest first:
* `thresholds` (proposer) — re-fit the operating threshold for the best F1
  on the first k minutes; scored by F1 at that threshold. Seconds, no GPU.
* `prototypes` (classifier) — *teach on site*: the first k minutes' cuts
  are added to the prototype bank beside the card (DETECTION_DESIGN §4:
  "teach is a prototype, not a retrain") and its thresholds re-set on them;
  scored by open-set accuracy (right class, not UNKNOWN). Seconds, no GPU.
* `temperature` (both) — re-fit the confidence calibration (Platt for the
  proposer, temperature for the classifier); scored by expected calibration
  error (LOWER is better — the acceptance line is a ceiling).
* `finetune` (both) — fine-tune the network on the first k minutes
  (PyTorch; the GPU on Bill's machine); scored by mAP@0.5 (proposer) or
  accuracy (classifier).

`save_adapted=True` saves the adapted model at the crossing point as a new
model with its own card (`<name>_onsite_<k>min`), ready for the AI Detect
tab; the base model is never changed. The base card gets
`metrics.minutes_to_acceptable`.

LIMITS. On a synthetic stand-in (as in the tests) the curve proves the code
path, not the field. The acceptance line is Bill's to set per mission; this
code does not choose it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from atk_diffusion import cards as _cards
from atk_diffusion.experiments.detector_eval import min_score, tile_step_seconds
from atk_diffusion.experiments.report import experiment_dir, write_report
from atk_diffusion.learn import calibrate as _cal
from atk_diffusion.learn import common as K
from atk_diffusion.learn import export as X

MODES = {"proposer2d": ("thresholds", "temperature", "finetune"),
         "classifier1d": ("prototypes", "temperature", "finetune")}
DEFAULT_METRIC = {("proposer2d", "thresholds"): "f1",
                  ("proposer2d", "temperature"): "ece",
                  ("proposer2d", "finetune"): "map50",
                  ("classifier1d", "prototypes"): "open_set_accuracy",
                  ("classifier1d", "temperature"): "ece",
                  ("classifier1d", "finetune"): "accuracy"}
DEFAULT_MINUTES = (0, 0.5, 1, 2, 5, 10, 20, 30, 60)


def _higher_is_better(metric: str) -> bool:
    return metric != "ece"


def _crossed(value, acceptance: float, metric: str) -> bool:
    if value is None or not np.isfinite(value):
        return False
    return value >= acceptance if _higher_is_better(metric) else value <= acceptance


# ---------------------------------------------------------------------------
# The stream: items, their times, the held-out stretch
# ---------------------------------------------------------------------------
def _stream(card, dataset_dir, profile, adapt_split, eval_split, rf):
    """(manifest, adapt ids, adapt times (s), eval ids, eval split name,
    words) — ids are item indices of the adapt / eval splits."""
    kind = "wideband" if card.kind == "proposer2d" else "narrowband"
    m = K.open_dataset(dataset_dir, profile, kind, rf=rf)
    if kind == "wideband":
        ts = K.TileSet(dataset_dir, adapt_split, m)
        raw_t = np.asarray([r.t_s for r in ts.refs], np.float64)
        each = tile_step_seconds(m, profile)
    else:
        ts = K.ShardSet(dataset_dir, adapt_split, m)
        raw_t = ts.labels["t_s"]
        rate = float((m.get("canonical") or {}).get("rate") or m["sample_rate"])
        each = float(ts.L or 0) / rate
    if len(ts) == 0:
        raise K.DatasetRefused(f"the on-site dataset has no {adapt_split!r} "
                               "items to adapt on.")
    t, words = K.item_times(raw_t, each)
    ids = np.arange(len(t))
    if eval_split and K.split_files(dataset_dir, eval_split, kind):
        return m, ids, t, None, eval_split, words
    cut = float(np.quantile(t, 0.7))
    ev = ids[t >= cut]
    ad = ids[t < cut]
    words += ("; no held-out split, so the last 30 % of the stream in time "
              "was held out for scoring and never adapted on")
    return m, ad, t[ad], ev, adapt_split, words


# ---------------------------------------------------------------------------
# Adapters: each returns (metric value, state for saving)
# ---------------------------------------------------------------------------
class _ProposerOnnx:
    def __init__(self, runner, dataset_dir, m, adapt_split, eval_split, eval_ids):
        self.card = runner.card
        policy = K.BoxPolicy.from_json(self.card.input.get("boxes") or {})
        self.adapt = K.TileSet(dataset_dir, adapt_split, m)
        self.g_ad, _s, _n = K.proposer_truth(self.adapt, m, policy, False)
        self.p_ad = X.run_proposer(runner, self.adapt)
        if eval_ids is None:
            ev = K.TileSet(dataset_dir, eval_split, m)
            self.g_ev, _s, _n = K.proposer_truth(ev, m, policy, False)
            self.p_ev = X.run_proposer(runner, ev)
        else:
            self.g_ev = [self.g_ad[i] for i in eval_ids]
            self.p_ev = [self.p_ad[i] for i in eval_ids]

    def _pairs(self, preds, gts):
        s, t = [], []
        for p, g in zip(preds, gts):
            tp, _w, order = K.match_image(p["boxes"], p["scores"], g["boxes"], 0.5)
            s.append(np.asarray(p["scores"])[order])
            t.append(tp)
        return (np.concatenate(s) if s else np.zeros(0),
                np.concatenate(t) if t else np.zeros(0, bool))

    def thresholds(self, sel):
        if sel is None:
            thr = min_score(self.card)
        else:
            thr = float(K.best_threshold([self.p_ad[i] for i in sel],
                                         [self.g_ad[i] for i in sel])["threshold"])
        op = K.operating_point(self.p_ev, self.g_ev, thr)
        return K.finite_or_none(op["f1"]), {"min_score": thr, "operating_point": op}

    def temperature(self, sel):
        s_ev, t_ev = self._pairs(self.p_ev, self.g_ev)
        pl = (self.card.calibration or {}).get("platt") or {}
        if sel is None:
            a, b = (pl.get("a"), pl.get("b")) if "a" in pl else (None, None)
        else:
            s, t = self._pairs([self.p_ad[i] for i in sel],
                               [self.g_ad[i] for i in sel])
            if s.size < 2 or t.all() or not t.any():
                return None, {"skipped": "no mix of right and wrong boxes yet"}
            fit = _cal.fit_platt(s, t)
            a, b = fit["a"], fit["b"]
        if s_ev.size == 0:
            return None, {"skipped": "no boxes in the held-out stretch"}
        p = np.clip(s_ev, 0, 1) if a is None else _cal.apply_platt(s_ev, a, b)
        return _cal.ece(p, t_ev), {"platt": {"a": a, "b": b}}


class _ClassifierOnnx:
    def __init__(self, runner, model_dir, profile, dataset_dir, m, adapt_split,
                 eval_split, eval_ids):
        self.card = runner.card
        self.model_dir = model_dir
        self.profile = profile
        names = self.card.class_names()
        self.names = names
        mn = list(m.get("classes") or [])
        lut = {i: (names.index(n) if n in names else -1) for i, n in enumerate(mn)}
        sh = K.ShardSet(dataset_dir, adapt_split, m)
        self.o_ad = X.run_classifier(runner, sh)
        self.y_ad = np.asarray([lut.get(int(v), -1) for v in sh.labels["label"]])
        if eval_ids is None:
            ev = K.ShardSet(dataset_dir, eval_split, m)
            self.o_ev = X.run_classifier(runner, ev)
            self.y_ev = np.asarray([lut.get(int(v), -1) for v in ev.labels["label"]])
        else:
            self.o_ev = {k: v[eval_ids] for k, v in self.o_ad.items()}
            self.y_ev = self.y_ad[eval_ids]
        canon = self.card.input.get("canonical") or {}
        self.canon = canon.get("class") or canon.get("cls")

    def prototypes(self, sel):
        bank, impl = _cal.load_bank(self.model_dir, self.profile, self.canon)
        if sel is not None:
            ys = self.y_ad[sel]
            emb = self.o_ad["embedding"][sel]
            held = {}
            for i, n in enumerate(self.names):
                e = emb[ys == i]
                if len(e):
                    bank.add(n, e, "cabled")
                    held[n] = e
            if held:
                bank.calibrate_thresholds(held)
        known = self.y_ev >= 0
        calls = [c for c, _d, _t in _cal._calls(bank, self.o_ev["embedding"][known])]
        truth = [self.names[int(v)] for v in self.y_ev[known]]
        acc = float(np.mean([c == t for c, t in zip(calls, truth)])) if truth else None
        return acc, {"bank": bank, "implementation": impl}

    def temperature(self, sel):
        known_ev = self.y_ev >= 0
        if sel is None:
            T = float((self.card.calibration or {}).get("temperature", 1.0) or 1.0)
        else:
            ys = self.y_ad[sel]
            k = ys >= 0
            if k.sum() < 2:
                return None, {"skipped": "too few cuts of known classes yet"}
            T = _cal.fit_temperature(self.o_ad["logits"][sel][k], ys[k])["temperature"]
        if not known_ev.any():
            return None, {"skipped": "no known cuts in the held-out stretch"}
        e = _cal.multiclass_ece(self.o_ev["logits"][known_ev], self.y_ev[known_ev], T)
        return e, {"temperature": T}


def _finetune_value(card, model_dir, profile, dataset_dir, m, adapt_split,
                    eval_split, eval_ids, sel, metric, epochs, lr, batch_size,
                    device, seed):
    """Fine-tune a fresh copy of the PyTorch model on `sel` and score it on
    the held-out stretch, in PyTorch (the same graph within the export
    tolerance). Returns the metric value."""
    torch = K.require_torch()
    dev, _w = K.pick_device(device)
    K.seed_everything(seed)
    if card.kind == "proposer2d":
        from atk_diffusion.learn import data as D
        from atk_diffusion.learn import proposer2d as P2
        model, _c = P2.load_torch(model_dir, profile)
        policy = K.BoxPolicy.from_json(card.input.get("boxes") or {})
        norm = K.TileNorm.from_json(model.arch["norm"])
        ad = D.WidebandTiles(dataset_dir, adapt_split, profile, manifest=m,
                             policy=policy, norm=norm, train=True, seed=seed)
        if sel is not None and len(sel):
            P2.fit(model, torch.utils.data.Subset(ad, [int(i) for i in sel]),
                   epochs, lr=lr or 2e-4, batch_size=batch_size, device=dev,
                   seed=seed, tag="onsite")
        if eval_ids is None:
            ev = D.WidebandTiles(dataset_dir, eval_split, profile, manifest=m,
                                 policy=policy, norm=norm)
            gts = [{"boxes": ev.target_rc(i)[0], "labels": ev.target_rc(i)[1]}
                   for i in range(len(ev))]
            preds = P2.predict(model, ev, dev)
        else:
            ev = D.WidebandTiles(dataset_dir, adapt_split, profile, manifest=m,
                                 policy=policy, norm=norm)
            sub = torch.utils.data.Subset(ev, [int(i) for i in eval_ids])
            preds = []
            model.to(dev).eval()
            with torch.no_grad():
                for i in range(len(sub)):
                    det = model([sub[i][0].to(dev)])[0]
                    b = det["boxes"].cpu().numpy()
                    preds.append({"boxes": b[:, [1, 0, 3, 2]] if len(b) else b.reshape(0, 4),
                                  "scores": det["scores"].cpu().numpy(),
                                  "labels": det["labels"].cpu().numpy()})
            gts = [{"boxes": ev.target_rc(int(i))[0], "labels": ev.target_rc(int(i))[1]}
                   for i in eval_ids]
        if metric == "f1":
            return K.finite_or_none(K.operating_point(preds, gts, min_score(card))["f1"])
        sc = P2.score(preds, gts, list(card.input.get("families")))
        return sc.get(metric if metric in sc else "map50")
    from atk_diffusion.learn import classifier1d as C1
    from atk_diffusion.learn import data as D
    model, _c = C1.load_torch(model_dir, profile)
    names = card.class_names()
    use_scf = bool(model.arch["use_scf"])
    kw = dict(classes=names, window=int(card.input["iq_len"]), use_scf=use_scf,
              scf_norm=str(card.input.get("scf_norm", "max")))
    if sel is not None and len(sel):
        ad = D.NarrowbandDataset(dataset_dir, adapt_split, profile, manifest=m,
                                 train=True, seed=seed, known_only=True, **kw)
        keep = set(int(i) for i in sel)
        ad.items = np.asarray([i for i in ad.items if int(i) in keep], np.int64)
        if len(ad) >= 2:
            C1.fit(model, ad, epochs, lr=lr or 5e-4,
                   batch_size=min(batch_size, len(ad)), device=dev, seed=seed,
                   keep_best=False, tag="onsite")
    if eval_ids is None:
        ev = D.NarrowbandDataset(dataset_dir, eval_split, profile, manifest=m,
                                 known_only=True, **kw)
    else:
        ev = D.NarrowbandDataset(dataset_dir, adapt_split, profile, manifest=m,
                                 known_only=True, **kw)
        keep = set(int(i) for i in eval_ids)
        ev.items = np.asarray([i for i in ev.items if int(i) in keep], np.int64)
    return C1.evaluate_torch(model, ev, dev)["accuracy"] if len(ev) else None


# ---------------------------------------------------------------------------
def minutes_to_acceptable(rf, profile: str, model_dir, onsite_dataset, *,
                          acceptance: float, mode: str | None = None,
                          metric: str | None = None, minutes=None,
                          adapt_split: str = "train", eval_split: str = "test",
                          finetune_epochs: int = 4, finetune_lr: float | None = None,
                          batch_size: int = 8, device: str = "auto",
                          threads: int | None = 1, seed: int = 0,
                          save_adapted: bool = False, update_card: bool = True,
                          out_name: str = "minutes_to_acceptable",
                          progress=None) -> dict:
    """The curve of score against minutes on site, and the minutes at which
    it crosses `acceptance` (module docstring). Returns the result dict
    with "report" paths (and "adapted_model" when one was saved)."""
    card = _cards.load(model_dir, for_profile=profile)
    if card.kind not in MODES:
        raise ValueError(f"minutes-to-acceptable is measured for the proposer "
                         f"and the classifier; {card.name} is a {card.kind}.")
    mode = mode or MODES[card.kind][0]
    if mode not in MODES[card.kind]:
        raise ValueError(f"for a {card.kind} the adaptation is one of "
                         f"{', '.join(MODES[card.kind])}, not {mode!r}")
    metric = metric or DEFAULT_METRIC[(card.kind, mode)]
    m, ad_ids, t_ad, ev_ids, ev_split, time_words = _stream(
        card, onsite_dataset, profile, adapt_split, eval_split, rf)
    t_max = float(t_ad.max()) if len(t_ad) else 0.0
    span_min = t_max / 60.0
    ks = sorted(set(float(k) for k in (minutes or DEFAULT_MINUTES)))
    use_ks = []
    for k in ks:                      # stop at the first k past the stream's end
        use_ks.append(k)
        if k * 60.0 > t_max:
            break
    runner = X.OnnxRunner(model_dir, card.kind, for_profile=profile,
                          threads=threads)
    if mode != "finetune":
        if card.kind == "proposer2d":
            ad = _ProposerOnnx(runner, onsite_dataset, m, adapt_split, ev_split,
                               ev_ids)
        else:
            ad = _ClassifierOnnx(runner, model_dir, profile, onsite_dataset, m,
                                 adapt_split, ev_split, ev_ids)
        fn = getattr(ad, mode)
    curve, states = [], {}
    last_n, last_val, last_state = -1, None, None
    for k in use_ks:
        sel = None if k == 0 else ad_ids[t_ad < k * 60.0]
        n = 0 if sel is None else len(sel)
        if n == last_n and k != 0:
            val, state = last_val, last_state               # no new data
        elif mode == "finetune":
            val = _finetune_value(card, model_dir, profile, onsite_dataset, m,
                                  adapt_split, ev_split, ev_ids, sel, metric,
                                  finetune_epochs, finetune_lr, batch_size,
                                  device, seed)
            state = {"indices": None if sel is None else [int(i) for i in sel]}
        else:
            val, state = fn(sel)
        val = K.finite_or_none(val)
        curve.append({"minutes": k, "items": n, "value": val,
                      "crossed": _crossed(val, acceptance, metric)})
        states[k] = (sel, state)
        last_n, last_val, last_state = n, val, state
        if progress:
            progress(f"{k:g} min on site ({n} items): {metric} "
                     f"{'n/a' if val is None else f'{val:.3f}'}")
    hit = next((c for c in curve if c["crossed"]), None)
    mta = hit["minutes"] if hit else None
    word = "at least" if _higher_is_better(metric) else "at most"
    lines = [f"{card.name} ({card.kind}) arrived trained on "
             f"{', '.join(d.get('name', '?') for d in card.datasets) or 'its training set'}; "
             f"on-site data: {m.get('name')!r} ({len(ad_ids)} items over "
             f"{span_min:.1f} min; {time_words}).",
             f"Adaptation: {mode}; score: {metric} on the held-out stretch "
             f"(acceptance: {metric} {word} {acceptance:g}).",
             "Curve: " + "; ".join(f"{c['minutes']:g} min → "
                                   f"{'n/a' if c['value'] is None else format(c['value'], '.3f')}"
                                   for c in curve) + "."]
    if mta is None:
        lines.append(f"NOT REACHED within {use_ks[-1]:g} minutes — more on-site "
                     "data, a different adaptation, or a better prior is needed.")
    elif mta == 0:
        lines.append("Acceptable on arrival: the composed region was enough; no "
                     "minutes on site were needed.")
    else:
        lines.append(f"Minutes on site to acceptable: {mta:g}.")
    res = {"model": card.name, "kind": card.kind, "mode": mode, "metric": metric,
           "higher_is_better": _higher_is_better(metric),
           "acceptance": float(acceptance), "minutes_to_acceptable": mta,
           "curve": curve, "onsite_dataset": m.get("name"),
           "adapt_items": int(len(ad_ids)), "stream_minutes": span_min,
           "time_source": time_words, "eval_split": ev_split
           if ev_ids is None else "last 30 % of the stream"}
    if save_adapted and mta is not None and mta > 0:
        res["adapted_model"] = str(_save_adapted(
            rf, profile, model_dir, card, onsite_dataset, m, mode, mta,
            states[mta], adapt_split, finetune_epochs, finetune_lr, batch_size,
            device, seed, threads))
        lines.append(f"The adapted model was saved as "
                     f"{Path(res['adapted_model']).name}; the base is unchanged.")
    if update_card:
        met = dict(card.metrics or {})
        met["minutes_to_acceptable"] = {"value": mta, "acceptance": float(acceptance),
                                        "metric": metric, "mode": mode,
                                        "dataset": m.get("name"),
                                        "curve": [{"minutes": c["minutes"],
                                                   "value": c["value"]}
                                                  for c in curve]}
        card.metrics = met
        X.resave_card(model_dir, card, rf=rf)
    out = experiment_dir(rf, profile, out_name)
    md, js = write_report(out, out_name, res, lines,
                          title=f"Minutes to acceptable — {card.name}", rf=rf)
    res["report"] = {"markdown": str(md), "json": str(js)}
    return res


def _save_adapted(rf, profile, model_dir, card, dataset_dir, m, mode, k,
                  state, adapt_split, epochs, lr, batch_size, device, seed,
                  threads):
    """The adapted model at the crossing point, as a new model folder."""
    sel, st = state
    name = f"{card.name}_onsite_{k:g}min".replace(".", "p")
    note = f"adapted on site ({mode}) from the first {k:g} minutes of {m.get('name')}"
    if mode == "finetune":
        if card.kind == "proposer2d":
            from atk_diffusion.learn import proposer2d as P2
            return P2.finetune(rf, profile, model_dir, dataset_dir, name, epochs,
                               lr=lr or 2e-4, batch_size=batch_size,
                               split=adapt_split, indices=sel, device=device,
                               seed=seed, threads=threads, note=note)
        from atk_diffusion.learn import classifier1d as C1
        return C1.finetune(rf, profile, model_dir, dataset_dir, name, epochs,
                           lr=lr or 5e-4, batch_size=batch_size,
                           split=adapt_split, indices=sel, device=device,
                           seed=seed, threads=threads, note=note)
    new = K.new_model_dir(rf, profile, name)
    src = Path(model_dir)
    for f in (X.ONNX_FILE, X.TORCH_FILE):
        if (src / f).exists():
            shutil.copy2(src / f, new / f)
    c2 = _cards.ModelCard.from_json(card.to_json())
    c2.name = name
    cal = dict(c2.calibration or {})
    if mode == "thresholds":
        cal["min_score"] = float(st["min_score"])
        cal["operating_threshold"] = {**{kk: (K.finite_or_none(v) if isinstance(v, float)
                                              else v) for kk, v in
                                         st["operating_point"].items()},
                                      "fit_on": {"dataset": m.get("name"),
                                                 "minutes": k}}
    elif mode == "temperature":
        if "platt" in st:
            cal["platt"] = {**(cal.get("platt") or {}), **st["platt"],
                            "fit_on": {"dataset": m.get("name"), "minutes": k}}
        if "temperature" in st:
            cal["temperature"] = float(st["temperature"])
    elif mode == "prototypes":
        st["bank"].save(new)
        cal["open_set"] = {**(cal.get("open_set") or {}),
                           "implementation": st["implementation"],
                           "adapted_on": {"dataset": m.get("name"), "minutes": k}}
    c2.calibration = cal
    entry = K.dataset_entry(dataset_dir, m, (adapt_split,))
    entry["minutes_used"] = k
    c2.datasets = list(c2.datasets) + [entry]
    c2.notes = list(c2.notes) + [note]
    met = dict(c2.metrics or {})
    met.pop("minutes_to_acceptable", None)
    c2.metrics = met
    X.save_model(new, c2, rf=rf)
    return new
