# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Calibration: confidences that mean what they say, and an honest *unknown*
(DETECTION_DESIGN §4 "Open set — honest unknown", §6.4; plan §7).

§6.4: *"Temperature scaling of the detector's and classifier's confidences
on the cabled set, so that '0.8' means right four times in five. The
open-set thresholds are set here too."*

THE CLASSIFIER — temperature scaling (Guo et al. 2017): one number T, fitted
by L-BFGS on the negative log-likelihood of held-out logits; probabilities
become softmax(logits / T). It cannot change a single decision (argmax is
unchanged), only how sure the model says it is. Expected calibration error
(ECE, 15 equal-width confidence bins) is reported before and after — on the
set it was fitted on AND, when given, on a set it was not, because ECE on
the fitting set flatters.

THE PROPOSER — Platt scaling, p = σ(a·logit(s) + b), fitted the same way on
(score, was-it-a-real-signal) pairs from held-out tiles. WHY NOT TEMPERATURE
ALONE: an FCOS score is √(class probability × centre-ness), which runs low
— a box that is right 95 % of the time can score 0.4. Dividing logit(0.4)
by any T keeps it below 0.5, so temperature alone cannot make "0.8 means
four in five" true; the bias term can. "Real signal" means the box overlaps
a labelled one at IoU ≥ 0.5, family aside: the AI badge's confidence is
about *where*, and family accuracy is scored separately (AP per family).

L-BFGS FROM SCIPY, NOT TORCH. Both fits are one- or two-parameter problems
with closed-form gradients, so they run on `scipy.optimize` — which ATK's
core environment has — and calibration can be refreshed on new cabled
captures without the training environment.

OPEN SET. Classification is by nearest prototype in embedding space; a cut
farther from its nearest prototype than that class's threshold is UNKNOWN
(§4). The bank is `atk_diffusion.detect.prototypes.PrototypeBank`, written
by another engineer and imported lazily. When it is not importable the same
measurement runs on a built-in nearest-mean bank (cosine distance, the
threshold a quantile of held-out distances) and every report says which
was used — the number is real either way, but it is the bank ATK will run
that should set the shipped thresholds.

WHERE IT IS WRITTEN (the card's `calibration`, read by
`detect.onnx_models`):
  classifier  `temperature` (a number: logits are divided by it),
              `temperature_fit` (T, NLL and ECE before/after, on what),
              `open_set` (thresholds, unknown rejection, false-unknown rate);
              the bank itself beside the card (`prototypes.json` + `.npz`,
              bound to the weights' SHA-256 — `PrototypeBank.load` refuses
              a bank made by other weights)
  proposer    `min_score` (the operating threshold, best F1 on val — boxes
              below it are dropped), `operating_threshold` (its precision,
              recall, F1), `platt` (a, b, ECE before/after, and the formula)
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from atk_diffusion.detect import classes as _classes
from atk_diffusion.learn import common as K


# ---------------------------------------------------------------------------
# Calibration error
# ---------------------------------------------------------------------------
def reliability(confidence, correct, n_bins: int = 15) -> list[dict]:
    """Equal-width bins over [0, 1]: count, mean confidence, accuracy."""
    c = np.asarray(confidence, np.float64).reshape(-1)
    y = np.asarray(correct, np.float64).reshape(-1)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    out = []
    for k in range(int(n_bins)):
        lo, hi = edges[k], edges[k + 1]
        sel = (c > lo) & (c <= hi) if k else (c >= lo) & (c <= hi)
        n = int(sel.sum())
        out.append({"lo": float(lo), "hi": float(hi), "n": n,
                    "confidence": float(c[sel].mean()) if n else None,
                    "accuracy": float(y[sel].mean()) if n else None})
    return out


def ece(confidence, correct, n_bins: int = 15) -> float:
    """Expected calibration error: Σ (n_b / N) · |accuracy_b − confidence_b|.

    Multiclass: pass the top probability and whether the top class was
    right. Binary (the proposer): pass p(real) and whether it was real."""
    c = np.asarray(confidence, np.float64).reshape(-1)
    if c.size == 0:
        return float("nan")
    total = 0.0
    for b in reliability(c, correct, n_bins):
        if b["n"]:
            total += b["n"] * abs(b["accuracy"] - b["confidence"])
    return float(total / c.size)


def multiclass_ece(logits, labels, temperature: float = 1.0,
                   n_bins: int = 15) -> float:
    p = K.softmax(np.asarray(logits, np.float64) / float(temperature))
    y = np.asarray(labels).reshape(-1)
    return ece(p.max(axis=1), p.argmax(axis=1) == y, n_bins)


def _nll_t(logits, labels, T: float) -> float:
    z = np.asarray(logits, np.float64) / T
    m = z.max(axis=1, keepdims=True)
    lse = (m[:, 0] + np.log(np.exp(z - m).sum(axis=1)))
    return float(np.mean(lse - z[np.arange(len(z)), labels]))


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------
def fit_temperature(logits, labels, max_iter: int = 100,
                    n_bins: int = 15) -> dict:
    """Temperature T > 0 minimising the NLL of softmax(logits / T), by
    L-BFGS over log T. Returns T and NLL/ECE before and after."""
    from scipy.optimize import minimize
    z = np.asarray(logits, np.float64)
    y = np.asarray(labels, np.int64).reshape(-1)
    if z.ndim != 2 or len(z) != len(y) or len(y) == 0:
        raise ValueError("temperature scaling needs logits (N, C) and N labels")
    if np.any((y < 0) | (y >= z.shape[1])):
        raise ValueError("a label is outside the logits' classes (unknown "
                         "examples are not used to fit a temperature)")

    def f(v):
        T = math.exp(float(v[0]))
        q = z / T
        m = q.max(axis=1, keepdims=True)
        e = np.exp(q - m)
        s = e.sum(axis=1)
        p = e / s[:, None]
        nll = float(np.mean(m[:, 0] + np.log(s) - q[np.arange(len(q)), y]))
        # d nll / d T = (1/T²)·mean(z_y − Σ p·z); chain rule to log T
        g_T = float(np.mean(z[np.arange(len(z)), y] - (p * z).sum(axis=1))) / (T * T)
        return nll, np.array([g_T * T])

    res = minimize(f, x0=np.array([0.0]), jac=True, method="L-BFGS-B",
                   bounds=[(math.log(0.02), math.log(50.0))],
                   options={"maxiter": int(max_iter)})
    T = float(math.exp(float(res.x[0])))
    return {"method": "temperature scaling, L-BFGS on NLL (scipy)",
            "temperature": T, "n": int(len(y)),
            "nll_before": _nll_t(z, y, 1.0), "nll_after": _nll_t(z, y, T),
            "ece_before": multiclass_ece(z, y, 1.0, n_bins),
            "ece_after": multiclass_ece(z, y, T, n_bins),
            "converged": bool(res.success)}


def _logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def apply_platt(scores, a: float, b: float) -> np.ndarray:
    """p = σ(a · logit(score) + b)."""
    return 1.0 / (1.0 + np.exp(-(float(a) * _logit(scores) + float(b))))


def fit_platt(scores, is_true, max_iter: int = 100, n_bins: int = 15) -> dict:
    """Platt scaling of detection scores by L-BFGS on the binary NLL.
    Returns a, b and NLL/ECE before (raw score as the probability) and
    after."""
    from scipy.optimize import minimize
    x = _logit(scores).reshape(-1)
    y = np.asarray(is_true, np.float64).reshape(-1)
    if x.size == 0 or x.size != y.size:
        raise ValueError("Platt scaling needs one true/false per score")

    def bce(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def f(v):
        a, b = float(v[0]), float(v[1])
        p = 1.0 / (1.0 + np.exp(-(a * x + b)))
        r = p - y
        return bce(p), np.array([float(np.mean(r * x)), float(np.mean(r))])

    res = minimize(f, x0=np.array([1.0, 0.0]), jac=True, method="L-BFGS-B",
                   options={"maxiter": int(max_iter)})
    a, b = float(res.x[0]), float(res.x[1])
    raw = np.clip(np.asarray(scores, np.float64).reshape(-1), 0, 1)
    cal = apply_platt(raw, a, b)
    return {"method": "Platt scaling σ(a·logit(score) + b), L-BFGS on NLL "
                      "(scipy)",
            "a": a, "b": b, "n": int(x.size), "positives": int(y.sum()),
            "nll_before": bce(raw), "nll_after": bce(cal),
            "ece_before": ece(raw, y, n_bins), "ece_after": ece(cal, y, n_bins),
            "converged": bool(res.success),
            "apply": "p = 1 / (1 + exp(-(a * logit(score) + b)))"}


# ---------------------------------------------------------------------------
# Open set — prototypes and thresholds
# ---------------------------------------------------------------------------
class BuiltinBank:
    """Nearest-mean prototypes with per-class thresholds — used only when
    `detect.prototypes.PrototypeBank` is not importable (or `prefer=
    "builtin"`). Same method names, so the measurement code does not care
    which bank it has; saved as `prototypes_builtin.json`, which ATK's
    pipeline does not load — only the real bank ships.

    Distance is cosine distance (1 − cos) to the class's normalised mean
    embedding. A class's threshold is the `quantile` of its held-out
    distances; with fewer than `floor` held-out examples it is the largest
    distance seen times `widen`, and `notes` says the class is thin."""

    def __init__(self, profile: str, canonical_class: str,
                 quantile: float = 0.95, floor: int = 10, widen: float = 1.25):
        self.profile = profile
        self.canonical_class = canonical_class
        self.quantile = float(quantile)
        self.floor = int(floor)
        self.widen = float(widen)
        self.sums: dict[str, np.ndarray] = {}
        self.counts: dict[str, int] = {}
        self.sources: dict[str, str] = {}
        self.thresholds: dict[str, float] = {}
        self.notes: list[str] = []

    def add(self, cls: str, embeddings, source: str = "trained") -> None:
        e = np.asarray(embeddings, np.float64).reshape(-1, np.shape(embeddings)[-1])
        if e.size == 0:
            return
        self.sums[cls] = self.sums.get(cls, 0.0) + e.sum(axis=0)
        self.counts[cls] = self.counts.get(cls, 0) + len(e)
        self.sources.setdefault(cls, source)

    def prototypes(self) -> dict[str, np.ndarray]:
        out = {}
        for c, s in self.sums.items():
            v = np.asarray(s) / max(1, self.counts[c])
            n = float(np.linalg.norm(v))
            out[c] = v / n if n > 0 else v
        return out

    def _dist(self, e) -> dict[str, float]:
        e = np.asarray(e, np.float64).reshape(-1)
        n = float(np.linalg.norm(e))
        e = e / n if n > 0 else e
        return {c: float(1.0 - np.dot(e, p)) for c, p in self.prototypes().items()}

    def calibrate_thresholds(self, held_out: dict) -> dict:
        protos = self.prototypes()
        for c in protos:
            e = np.asarray(held_out.get(c, np.zeros((0, len(protos[c])))),
                           np.float64)
            d = [self._dist(x)[c] for x in e.reshape(-1, len(protos[c]))]
            if len(d) >= self.floor:
                self.thresholds[c] = float(np.quantile(d, self.quantile))
            else:
                base = max(d) if d else 0.5
                self.thresholds[c] = float(base * self.widen)
                self.notes.append(f"{c}: only {len(d)} held-out examples "
                                  f"(floor {self.floor}) — its threshold is "
                                  "wider than the others")
        return dict(self.thresholds)

    def classify(self, embedding):
        d = self._dist(embedding)
        if not d:
            return _classes.UNKNOWN, float("inf"), float("nan")
        c = min(d, key=d.get)
        thr = self.thresholds.get(c, float("inf"))
        return (c if d[c] <= thr else _classes.UNKNOWN), d[c], thr

    def save(self, out_dir) -> Path:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        p = d / "prototypes_builtin.json"
        K.write_json(p, {"kind": "builtin nearest-mean bank (calibrate.py)",
                         "profile": self.profile,
                         "canonical_class": self.canonical_class,
                         "distance": "cosine (1 - cos)",
                         "quantile": self.quantile, "floor": self.floor,
                         "prototypes": {c: v.tolist()
                                        for c, v in self.prototypes().items()},
                         "counts": self.counts, "sources": self.sources,
                         "thresholds": self.thresholds, "notes": self.notes})
        return p


def _builtin_from_json(path, profile: str) -> BuiltinBank:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    have, want = str(d.get("profile", "")).lower(), str(profile).lower()
    if have != want:
        from atk_diffusion import profiles as _p
        try:
            a, b = _p.describe(have), _p.describe(want)
        except ValueError:
            a, b = repr(have), repr(want)
        raise _p.ProfileMismatch(f"these prototypes were made for {a}; this is "
                                 f"{b}. Profiles never mix: calibrate the "
                                 "classifier again on this receiver's cuts.")
    b = BuiltinBank(d["profile"], d.get("canonical_class", ""),
                    quantile=float(d.get("quantile", 0.95)),
                    floor=int(d.get("floor", 10)))
    for c, v in (d.get("prototypes") or {}).items():
        n = int((d.get("counts") or {}).get(c, 1)) or 1
        b.sums[c] = np.asarray(v, np.float64) * n
        b.counts[c] = n
        b.sources[c] = (d.get("sources") or {}).get(c, "trained")
    b.thresholds = {c: float(t) for c, t in (d.get("thresholds") or {}).items()}
    b.notes = list(d.get("notes") or [])
    return b


def load_bank(model_dir, profile: str, canonical_class: str | None = None):
    """(bank, implementation words) saved beside a classifier's card:
    `detect.prototypes.PrototypeBank.load` when its files are there and the
    module imports, else the built-in bank's JSON. Raises FileNotFoundError
    in words when there is neither."""
    d = Path(model_dir)
    if (d / "prototypes.json").exists():
        try:
            from atk_diffusion.detect.prototypes import PrototypeBank
            return (PrototypeBank.load(d, profile, canonical_class),
                    "atk_diffusion.detect.prototypes.PrototypeBank")
        except ImportError:
            pass
    if (d / "prototypes_builtin.json").exists():
        return (_builtin_from_json(d / "prototypes_builtin.json", profile),
                "built-in nearest-mean bank")
    raise FileNotFoundError(f"{d.name} has no prototype bank beside its card, "
                            "so unknown rejection cannot be measured. "
                            "Calibrate the classifier first "
                            "(calibrate.calibrate_classifier).")


def make_bank(profile: str, canonical_class: str, prefer: str = "auto",
              quantile: float = 0.95, model_sha256: str = ""):
    """(bank, implementation words). 'auto' uses detect.prototypes when it
    can be imported, else the built-in bank; 'builtin' forces the latter;
    'bank' insists on detect.prototypes. The bank is bound to the weights
    whose embeddings it holds (`model_sha256`), as PrototypeBank.load
    checks."""
    if prefer != "builtin":
        try:
            from atk_diffusion.detect.prototypes import PrototypeBank
            return (PrototypeBank(profile, canonical_class, quantile=quantile,
                                  model_sha256=model_sha256),
                    "atk_diffusion.detect.prototypes.PrototypeBank")
        except ImportError:
            if prefer == "bank":
                raise
    return (BuiltinBank(profile, canonical_class, quantile=quantile),
            "built-in nearest-mean bank (detect.prototypes was not importable)")


def _calls(bank, emb) -> list[tuple]:
    """(cls, distance, threshold) per row — PrototypeBank returns `Match`
    objects that unpack to that triple; the built-in bank returns it."""
    e = np.asarray(emb)
    if e.size == 0:
        return []
    e = e.reshape(len(e), -1)
    if hasattr(bank, "classify_many"):
        return [tuple(m) for m in bank.classify_many(e)]
    return [tuple(bank.classify(x)) for x in e]


def _thresholds(bank) -> dict:
    if hasattr(bank, "card_classes"):
        return {d["name"]: K.finite_or_none(d["threshold"])
                for d in bank.card_classes()}
    th = getattr(bank, "thresholds", None)
    return {str(k): K.finite_or_none(v) for k, v in (th or {}).items()}


def open_set(profile: str, canonical_class: str, train_emb: dict,
             held_out: dict, unknown_emb=None, known_test: dict | None = None,
             out_dir=None, prefer: str = "auto", quantile: float = 0.95,
             model_sha256: str = "") -> dict:
    """Build the prototype bank, set its thresholds on held-out data, and
    MEASURE: unknown rejection (classes the model never saw, called
    UNKNOWN), the false-unknown rate on known classes, and accuracy on the
    known examples the bank accepts. `out_dir` — where the bank is saved:
    the classifier's model folder, beside its card (where
    `PrototypeBank.load` and ATK's pipeline look for it)."""
    bank, impl = make_bank(profile, canonical_class, prefer, quantile,
                           model_sha256)
    for cls, e in train_emb.items():
        e = np.asarray(e)
        if e.size:
            bank.add(cls, e, "trained")
    bank.calibrate_thresholds({c: np.asarray(v) for c, v in held_out.items()
                               if np.asarray(v).size})
    rep = {"implementation": impl, "quantile": float(quantile)}
    unk = np.asarray(unknown_emb) if unknown_emb is not None else np.zeros((0,))
    if unk.size:
        calls = [c for c, _d, _t in _calls(bank, unk)]
        rep["unknown_rejection"] = float(np.mean([c == _classes.UNKNOWN
                                                  for c in calls]))
        rep["n_unknown"] = int(len(calls))
        forced = {}
        for c in calls:
            if c != _classes.UNKNOWN:
                forced[c] = forced.get(c, 0) + 1
        rep["unknown_forced_into"] = forced
    else:
        rep["unknown_rejection"] = None
        rep["n_unknown"] = 0
        rep["unknown_note"] = ("no held-out unknown classes were available, so "
                               "unknown rejection was not measured")
    if known_test:
        n = rej = right = acc_n = 0
        for cls, e in known_test.items():
            for c, _d, _t in _calls(bank, e):
                n += 1
                if c == _classes.UNKNOWN:
                    rej += 1
                else:
                    acc_n += 1
                    right += int(c == cls)
        rep["false_unknown_rate"] = rej / n if n else None
        rep["accuracy_on_accepted"] = right / acc_n if acc_n else None
        rep["n_known"] = n
    rep["thresholds"] = _thresholds(bank)
    notes = list(getattr(bank, "notes", []) or [])
    if hasattr(bank, "describe"):
        try:
            notes = list(bank.describe()) + notes
        except Exception:                                  # noqa: BLE001
            pass
    if notes:
        rep["notes"] = notes
    if out_dir is not None:
        try:
            saved = bank.save(out_dir)
            rep["saved_to"] = Path(saved).name
        except Exception as e:                             # noqa: BLE001
            rep["save_error"] = f"the bank could not be saved: {e}"
    return rep


# ---------------------------------------------------------------------------
# Calibrating a saved model (ONNX on the CPU — what ships is what is measured)
# ---------------------------------------------------------------------------
def _known_unknown(labels_model):
    lab = np.asarray(labels_model)
    return lab >= 0, lab < 0


def calibrate_classifier(rf, profile: str, model_dir, dataset_dir,
                         split: str = "val", proto_split: str = "train",
                         test_split: str | None = "test", threads: int | None = 1,
                         prefer_bank: str = "auto", quantile: float = 0.95,
                         update_card: bool = True, progress=None):
    """Temperature and open-set thresholds for a saved classifier, measured
    through its ONNX graph on a dataset of this profile (the cabled set when
    there is one, §6.4). Writes `card.calibration` and returns the card."""
    from atk_diffusion.learn import export as X
    runner = X.OnnxRunner(model_dir, "classifier1d", for_profile=profile,
                          threads=threads)
    card = runner.card
    m = K.open_dataset(dataset_dir, profile, "narrowband", rf=rf)
    names = card.class_names()
    cn = card.input.get("canonical") or {}
    canon = cn.get("class") or cn.get("cls") or ""

    def outputs(sp):
        sh = K.ShardSet(dataset_dir, sp, m)
        if len(sh) == 0:
            return None, None, None
        o = X.run_classifier(runner, sh)
        lut = {i: (names.index(n) if n in names else -1)
               for i, n in enumerate(m.get("classes") or [])}
        lab = np.asarray([lut.get(int(v), -1) for v in sh.labels["label"]])
        return o, lab, sh

    o_fit, y_fit, _ = outputs(split)
    if o_fit is None:
        raise K.DatasetRefused(f"the {split!r} split of {Path(dataset_dir).name} "
                               "is empty; calibration needs held-out examples.")
    known, unk = _known_unknown(y_fit)
    if not np.any(known):
        raise K.DatasetRefused("none of the calibration examples is of a class "
                               "this model knows.")
    if progress:
        progress(f"fitting a temperature on {int(known.sum())} held-out cuts")
    temp = fit_temperature(o_fit["logits"][known], y_fit[known])
    temp["fit_on"] = {"dataset": str(m.get("name")), "split": split}
    o_test = y_test = None
    if test_split:
        o_test, y_test, _ = outputs(test_split)
    if o_test is not None:
        kt, _ = _known_unknown(y_test)
        if np.any(kt):
            temp["ece_heldout_before"] = multiclass_ece(o_test["logits"][kt],
                                                        y_test[kt], 1.0)
            temp["ece_heldout_after"] = multiclass_ece(o_test["logits"][kt],
                                                       y_test[kt],
                                                       temp["temperature"])
            temp["heldout"] = {"dataset": str(m.get("name")),
                               "split": test_split, "n": int(kt.sum())}
    o_pr, y_pr, _ = outputs(proto_split)
    train_emb = {}
    if o_pr is not None:
        for i, n in enumerate(names):
            sel = y_pr == i
            if np.any(sel):
                train_emb[n] = o_pr["embedding"][sel]
    held = {n: o_fit["embedding"][y_fit == i] for i, n in enumerate(names)
            if np.any(y_fit == i)}
    unknown = [o_fit["embedding"][unk]]
    known_test = None
    if o_test is not None:
        kt, ut = _known_unknown(y_test)
        unknown.append(o_test["embedding"][ut])
        known_test = {n: o_test["embedding"][y_test == i]
                      for i, n in enumerate(names) if np.any(y_test == i)}
    unknown = np.concatenate([u for u in unknown if u.size] or [np.zeros((0,))])
    osr = open_set(profile, canon, train_emb, held,
                   unknown_emb=unknown if unknown.size else None,
                   known_test=known_test, out_dir=Path(model_dir),
                   prefer=prefer_bank, quantile=quantile,
                   model_sha256=str(card.weights.get("sha256", "")))
    cal = dict(card.calibration or {})
    # a bare number: detect.onnx_models.Classifier1D divides logits by it
    cal["temperature"] = float(temp["temperature"])
    cal["temperature_fit"] = temp
    cal["open_set"] = osr
    cal["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    card.calibration = cal
    met = dict(card.metrics or {})
    if osr.get("unknown_rejection") is not None:
        met["unknown_rejection"] = osr["unknown_rejection"]
    met["false_unknown_rate"] = osr.get("false_unknown_rate")
    card.metrics = met
    if update_card:
        X.resave_card(model_dir, card, rf=rf)
    return card


def calibrate_proposer(rf, profile: str, model_dir, dataset_dir,
                       split: str = "val", threads: int | None = 1,
                       update_card: bool = True, progress=None):
    """Platt scaling and the operating threshold (best F1) for a saved
    proposer, measured through its ONNX graph on `split` of a wideband
    dataset of this profile. Writes `card.calibration`; returns the card."""
    from atk_diffusion.learn import export as X
    runner = X.OnnxRunner(model_dir, "proposer2d", for_profile=profile,
                          threads=threads)
    card = runner.card
    m = K.open_dataset(dataset_dir, profile, "wideband", rf=rf)
    tiles = K.TileSet(dataset_dir, split, m)
    if len(tiles) == 0:
        raise K.DatasetRefused(f"the {split!r} split of {Path(dataset_dir).name} "
                               "has no tiles; calibration needs held-out tiles.")
    policy = K.BoxPolicy.from_json(card.input.get("boxes") or {})
    gts, _snr, _names = K.proposer_truth(tiles, m, policy)
    if progress:
        progress(f"running the proposer on {len(tiles)} held-out tiles")
    preds = X.run_proposer(runner, tiles)
    scores, truth = [], []
    for p, g in zip(preds, gts):
        tp, _w, order = K.match_image(p["boxes"], p["scores"], g["boxes"], 0.5)
        scores.append(np.asarray(p["scores"])[order])
        truth.append(tp)
    s = np.concatenate(scores) if scores else np.zeros(0)
    t = np.concatenate(truth) if truth else np.zeros(0, bool)
    cal = dict(card.calibration or {})
    if s.size >= 2 and 0 < t.sum() < t.size:
        pl = fit_platt(s, t)
        pl["fit_on"] = {"dataset": str(m.get("name")), "split": split,
                        "tiles": len(tiles)}
        pl["what"] = ("the probability that a box marks a real signal (IoU ≥ "
                      "0.5 with a labelled one), family aside")
        cal["platt"] = pl
    else:
        cal["platt"] = {"skipped": True,
                        "why": ("the held-out tiles gave no mix of right and "
                                "wrong boxes to fit on (all right, all wrong, "
                                "or none)"), "n": int(s.size)}
    op = K.best_threshold(preds, gts, 0.5)
    op["fit_on"] = {"dataset": str(m.get("name")), "split": split}
    cal["operating_threshold"] = {k: (K.finite_or_none(v) if isinstance(v, float)
                                      else v) for k, v in op.items()}
    # the score below which detect.onnx_models.Proposer2D drops a box
    cal["min_score"] = float(op["threshold"])
    cal["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    card.calibration = cal
    if update_card:
        X.resave_card(model_dir, card, rf=rf)
    return card


def describe(cal: dict) -> list[str]:
    """Plain lines for the AI Detect tab."""
    out = []
    t = (cal or {}).get("temperature_fit")
    if isinstance(t, dict) and "temperature" in t:
        out.append(f"confidence calibrated by temperature {t['temperature']:.2f}: "
                   f"calibration error {t['ece_before']:.3f} → "
                   f"{t['ece_after']:.3f}")
    pl = (cal or {}).get("platt")
    if isinstance(pl, dict) and "a" in pl:
        out.append(f"box confidence calibrated (Platt): calibration error "
                   f"{pl['ece_before']:.3f} → {pl['ece_after']:.3f}")
    osr = (cal or {}).get("open_set")
    if isinstance(osr, dict) and osr.get("unknown_rejection") is not None:
        out.append(f"unknown signals called UNKNOWN {100 * osr['unknown_rejection']:.0f}"
                   f"% of the time ({osr['implementation']})")
    return out


def to_json_str(obj) -> str:
    return json.dumps(obj, indent=2, default=K.json_default)
