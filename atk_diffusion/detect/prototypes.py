# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Open set by prototypes; teach = add a prototype (DETECTION_DESIGN §4;
decision D6; plan B5; ARCHITECTURE §4.1).

*"Classification is by nearest prototype in embedding space: each known
class has one or more prototypes (the mean embedding of its examples, per
profile); a cutout whose distance to the nearest prototype exceeds that
class's threshold is UNKNOWN, and the threshold is set on held-out data so
that unknown-signal rejection is measured, not hoped. … a detector that
forces a class is a detector that lies."*

*"Teach-it-a-signal is a prototype, not a retrain. … No training run, no GPU,
seconds. The model card lists the class as taught with its example count; a
class with fewer examples than the floor says so, and its threshold is
wider."*

HOW.
* Embeddings are L2-normalised (the classifier's `embedding` output already
  is; this normalises again rather than trust it) and compared by COSINE
  DISTANCE, 1 − e·p, in [0, 2].
* A class has one prototype — the normalised mean of its examples — or, with
  `k_sub > 1` and at least 3 examples per sub-prototype, k spherical k-means
  sub-prototypes (a class with two looks: a pager on two baud rates). A
  sample's distance to a class is to its nearest sub-prototype.
* Per-class threshold: the `quantile` (default 0.95) of HELD-OUT within-class
  distances, set by `calibrate_thresholds` — so 95 % of that class's real
  examples are accepted and the rest is the price of an honest UNKNOWN.
  Until calibrated (and for a freshly taught class) it is the same quantile
  of the class's leave-one-out distances to its own mean, and the bank says
  the threshold is uncalibrated.
* Below the example floor (5) the threshold is WIDENED (x `widen`, 1.5) and
  every `classify` that lands on the class carries a note saying the class
  is thin. A class taught from ONE example has no spread to measure; it
  borrows the median threshold of the others (widened) and says so.
* Distance beyond the nearest class's threshold -> UNKNOWN, with the nearest
  class and its distance kept in the `Match`, so the analyst sees "nothing
  close enough; nearest DMR at 0.41 (threshold 0.22)".

`unknown_rejection(held_out_unknown)` is the number the card carries: the
fraction of embeddings of classes the bank never saw that it calls UNKNOWN.

PERSISTENCE: `prototypes.json` + `prototypes.npz` in the classifier's model
folder, beside its card. A bank is bound to its receiver PROFILE (a bank for
another profile is refused with ProfileMismatch, in words), its CANONICAL
CLASS, and the classifier WEIGHTS that made its embeddings: embeddings from
one network mean nothing to another, so a bank whose `model_sha256` is not
the card's weights hash is refused too. `update_card` writes the taught
classes and their example counts into the classifier's card.

LIMITS. A prototype is only as good as the embedding: two classes the
network maps to the same place cannot be separated by thresholds. Teaching
from five examples measured on the day they were taught is optimistic — the
plan's measure is the NEXT day's captures (plan B5, DETECTION_DESIGN §11).
"""

from __future__ import annotations

import json
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from atk_diffusion import profiles as _profiles
from atk_diffusion.detect.classes import UNKNOWN

EXAMPLE_FLOOR = 5
DEFAULT_THRESHOLD = 0.3          # used only when nothing at all can be measured
FILE_JSON = "prototypes.json"
FILE_NPZ = "prototypes.npz"
FORMAT = "atk-prototypes/1"
SOURCES = ("trained", "taught", "cabled", "synthetic", "confirmed")


def l2_normalize(x) -> np.ndarray:
    """Rows of `x` scaled to unit length; an all-zero row is refused."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError("embeddings are a vector or a [n, dim] array")
    nrm = np.linalg.norm(a, axis=1, keepdims=True)
    if np.any(nrm <= 1e-12) or not np.all(np.isfinite(a)):
        raise ValueError("an embedding of all zeros (or with NaN) cannot be "
                         "compared — the classifier produced nothing usable")
    return a / nrm


@dataclass
class Match:
    """The answer for one embedding. Unpacks as (cls, distance, threshold)."""
    cls: str
    distance: float
    threshold: float
    nearest: str = ""
    note: str = ""
    second: str = ""
    second_distance: float = float("nan")

    def __iter__(self):
        yield self.cls
        yield self.distance
        yield self.threshold

    @property
    def unknown(self) -> bool:
        return self.cls == UNKNOWN


def _spherical_kmeans(x: np.ndarray, k: int, seed: int, iters: int = 30) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    centers = [x[int(rng.integers(n))]]
    for _ in range(1, k):            # k-means++ on cosine distance
        d = np.min(1.0 - x @ np.array(centers).T, axis=1).clip(min=0)
        p = d / d.sum() if d.sum() > 0 else np.full(n, 1.0 / n)
        centers.append(x[int(rng.choice(n, p=p))])
    c = np.array(centers)
    for _ in range(iters):
        lab = np.argmax(x @ c.T, axis=1)
        new = np.array([x[lab == j].sum(axis=0) if np.any(lab == j) else c[j]
                        for j in range(k)])
        new = l2_normalize(new)
        if np.allclose(new, c):
            break
        c = new
    return c


class PrototypeBank:
    """Prototypes per class for one (profile, canonical class) — module
    docstring. `profile` is a profile id; `canonical_class` one of
    profiles.CLASS_ORDER (voice | wideband | spread)."""

    def __init__(self, profile: str, canonical_class: str, quantile: float = 0.95,
                 widen: float = 1.5, example_floor: int = EXAMPLE_FLOOR,
                 k_sub: int = 1, model_sha256: str = ""):
        pid = str(getattr(profile, "id", profile)).strip().lower()
        _profiles.parse_profile_id(pid)
        if canonical_class not in _profiles.CLASS_ORDER:
            raise ValueError(f"{canonical_class!r} is not a canonical class "
                             f"({', '.join(_profiles.CLASS_ORDER)})")
        if not 0.5 <= float(quantile) < 1.0:
            raise ValueError("the threshold quantile is between 0.5 and 1")
        self.profile = pid
        self.canonical_class = canonical_class
        self.quantile = float(quantile)
        self.widen = float(widen)
        self.example_floor = int(example_floor)
        self.k_sub = max(1, int(k_sub))
        self.model_sha256 = str(model_sha256 or "")
        self.dim: int | None = None
        self._ex: dict[str, np.ndarray] = {}
        self._proto: dict[str, np.ndarray] = {}
        self._thr: dict[str, float] = {}
        self._base: dict[str, float] = {}
        self._calibrated: dict[str, bool] = {}
        self._source: dict[str, str] = {}
        self._taught: dict[str, int] = {}
        self._taught_at: dict[str, str] = {}
        self.notes: list[str] = []

    # -- reading ---------------------------------------------------------------
    @property
    def classes(self) -> list[str]:
        return sorted(self._ex)

    def examples(self, cls: str) -> int:
        return int(self._ex[cls].shape[0]) if cls in self._ex else 0

    def threshold(self, cls: str) -> float:
        return float(self._thr[cls])

    def is_thin(self, cls: str) -> bool:
        return self.examples(cls) < self.example_floor

    def is_calibrated(self, cls: str) -> bool:
        return bool(self._calibrated.get(cls, False))

    def is_taught(self, cls: str) -> bool:
        """True when any of the class's examples were taught (plan B5)."""
        return self._source.get(cls) == "taught" or bool(self._taught.get(cls))

    def thin_note(self, cls: str) -> str:
        n = self.examples(cls)
        return (f"{cls} is thin: {n} example{'s' if n != 1 else ''}, below the "
                f"floor of {self.example_floor} — its threshold is widened to "
                f"{self._thr[cls]:.3f} and it may accept signals that are not "
                f"{cls}. Teach more examples.")

    def describe(self) -> list[str]:
        out = [f"prototype bank for {_profiles.describe(self.profile)}, "
               f"{self.canonical_class} cuts: {len(self._ex)} classes"]
        for c in self.classes:
            line = (f"{c}: {self.examples(c)} examples ({self._source[c]}"
                    + (f", {self._taught[c]} taught" if self._taught.get(c) else "")
                    + f"), {self._proto[c].shape[0]} prototype"
                    + ("s" if self._proto[c].shape[0] != 1 else "")
                    + f", threshold {self._thr[c]:.3f} "
                    + ("calibrated on held-out data" if self.is_calibrated(c)
                       else "UNCALIBRATED (leave-one-out)"))
            if self.is_thin(c):
                line += " — THIN"
            out.append(line)
        return out

    # -- building --------------------------------------------------------------
    def _check_dim(self, e: np.ndarray) -> None:
        if self.dim is None:
            self.dim = int(e.shape[1])
        elif e.shape[1] != self.dim:
            raise ValueError(f"these embeddings have {e.shape[1]} dimensions; "
                             f"the bank's have {self.dim} — they came from "
                             "a different classifier")

    def add(self, cls: str, embeddings, source: str = "trained",
            k: int | None = None) -> "PrototypeBank":
        """Add examples of `cls` (extending it if present) and rebuild its
        prototypes and (uncalibrated) threshold."""
        name = str(cls).strip()
        if not name or name == UNKNOWN:
            raise ValueError("a class needs a name, and UNKNOWN is not one")
        if source not in SOURCES:
            raise ValueError(f"unknown example source {source!r} ({', '.join(SOURCES)})")
        e = l2_normalize(embeddings)
        self._check_dim(e)
        if name in self._ex:
            self._ex[name] = np.vstack([self._ex[name], e])
        else:
            self._ex[name] = e
            self._source[name] = source
        if source == "taught":
            self._taught[name] = self._taught.get(name, 0) + e.shape[0]
            self._taught_at[name] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._rebuild(name, k)
        return self

    def teach(self, cls: str, embeddings, k: int | None = None) -> str:
        """Teach-it-a-signal (plan B5): add or extend `cls` from these
        examples, in seconds, no training. Returns what happened in words."""
        t0 = time.perf_counter()
        existed = str(cls) in self._ex
        self.add(cls, embeddings, source="taught", k=k)
        dt = time.perf_counter() - t0
        name = str(cls).strip()
        words = (f"{'extended' if existed else 'taught'} {name} from "
                 f"{np.asarray(embeddings).reshape(-1, self.dim).shape[0]} "
                 f"example(s) in {dt:.3f} s: {self.examples(name)} in all, "
                 f"threshold {self._thr[name]:.3f}"
                 + (" (calibrated earlier; recalibrate when held-out examples "
                    "exist)" if self.is_calibrated(name) else " (uncalibrated)"))
        if self.is_thin(name):
            words += ". " + self.thin_note(name)
        return words

    def _rebuild(self, cls: str, k: int | None = None) -> None:
        e = self._ex[cls]
        n = e.shape[0]
        kk = max(1, int(k if k is not None else self.k_sub))
        kk = min(kk, n // 3) if kk > 1 else 1
        if kk > 1:
            seed = zlib.crc32(cls.encode("utf-8")) & 0xFFFF
            self._proto[cls] = _spherical_kmeans(e, kk, seed)
        else:
            self._proto[cls] = l2_normalize(e.mean(axis=0))
        if self._calibrated.get(cls):
            base = self._base[cls]
        else:
            base = self._loo_threshold(cls)
        self._base[cls] = base
        self._thr[cls] = self._widened(cls, base)

    def _loo_threshold(self, cls: str) -> float:
        e = self._ex[cls]
        n = e.shape[0]
        if n < 2:
            others = [self._base[c] for c in self._ex if c != cls and c in self._base
                      and self._ex[c].shape[0] >= 2]
            return float(np.median(others)) if others else DEFAULT_THRESHOLD
        s = e.sum(axis=0)
        loo = l2_normalize(s[None, :] - e)            # mean of the others, per example
        d = 1.0 - np.sum(e * loo, axis=1)
        return float(np.clip(np.quantile(d, self.quantile), 1e-4, 2.0))

    def _widened(self, cls: str, base: float) -> float:
        if self.examples(cls) < self.example_floor:
            return float(min(2.0, base * self.widen))
        return float(base)

    def remove(self, cls: str) -> None:
        for d in (self._ex, self._proto, self._thr, self._base, self._calibrated,
                  self._source, self._taught, self._taught_at):
            d.pop(cls, None)

    # -- classifying -------------------------------------------------------------
    def _distances(self, e: np.ndarray) -> tuple[list[str], np.ndarray]:
        names = self.classes
        d = np.empty((e.shape[0], len(names)))
        for j, c in enumerate(names):
            d[:, j] = np.min(1.0 - e @ self._proto[c].T, axis=1)
        return names, d

    def classify(self, embedding) -> Match:
        """Nearest prototype, or UNKNOWN beyond that class's threshold."""
        return self.classify_many(embedding)[0]

    def classify_many(self, embeddings) -> list[Match]:
        e = l2_normalize(embeddings)
        if not self._ex:
            return [Match(UNKNOWN, float("nan"), float("nan"), "",
                          "the prototype bank is empty: nothing is known, so "
                          "everything is UNKNOWN") for _ in range(e.shape[0])]
        self._check_dim(e)
        names, d = self._distances(e)
        order = np.argsort(d, axis=1)
        out = []
        for i in range(e.shape[0]):
            j = int(order[i, 0])
            c, dist = names[j], float(d[i, j])
            thr = self._thr[c]
            sec, sd = ("", float("nan"))
            if len(names) > 1:
                j2 = int(order[i, 1])
                sec, sd = names[j2], float(d[i, j2])
            if dist <= thr:
                note = self.thin_note(c) if self.is_thin(c) else ""
                out.append(Match(c, dist, thr, c, note, sec, sd))
            else:
                out.append(Match(UNKNOWN, dist, thr, c,
                                 f"not close enough to any known class: nearest "
                                 f"is {c} at {dist:.3f} (its threshold "
                                 f"{thr:.3f})", sec, sd))
        return out

    # -- calibration and measurement -----------------------------------------------
    def calibrate_thresholds(self, held_out: dict, quantile: float | None = None
                             ) -> dict[str, float]:
        """Set each class's threshold to the `quantile` of its HELD-OUT
        examples' distances to its own nearest prototype. Classes without
        held-out examples keep their leave-one-out thresholds (and stay
        marked uncalibrated). Returns the thresholds set."""
        q = self.quantile if quantile is None else float(quantile)
        out = {}
        for cls, emb in (held_out or {}).items():
            if cls not in self._ex:
                self.notes.append(f"held-out examples of {cls} ignored: the bank "
                                  "has no such class")
                continue
            e = l2_normalize(emb)
            self._check_dim(e)
            d = np.min(1.0 - e @ self._proto[cls].T, axis=1)
            base = float(np.clip(np.quantile(d, q), 1e-4, 2.0))
            self._base[cls] = base
            self._calibrated[cls] = True
            self._thr[cls] = self._widened(cls, base)
            out[cls] = self._thr[cls]
        return out

    def unknown_rejection(self, held_out_unknown) -> float:
        """Fraction of embeddings of never-seen classes called UNKNOWN."""
        ms = self.classify_many(held_out_unknown)
        return float(np.mean([m.unknown for m in ms])) if ms else float("nan")

    def accuracy(self, held_out: dict) -> dict:
        """On held-out KNOWN classes: fraction right, fraction wrongly called
        another class, fraction rejected as UNKNOWN."""
        right = wrong = rejected = n = 0
        for cls, emb in (held_out or {}).items():
            for m in self.classify_many(emb):
                n += 1
                if m.cls == cls:
                    right += 1
                elif m.unknown:
                    rejected += 1
                else:
                    wrong += 1
        if n == 0:
            return {"n": 0}
        return {"n": n, "correct": right / n, "confused": wrong / n,
                "rejected": rejected / n}

    def evaluate(self, held_out_known: dict, held_out_unknown=None) -> dict:
        out = {"known": self.accuracy(held_out_known)}
        if held_out_unknown is not None and len(held_out_unknown):
            out["unknown_rejection"] = self.unknown_rejection(held_out_unknown)
        return out

    # -- the card --------------------------------------------------------------
    def card_classes(self) -> list[dict]:
        """Entries for the classifier card's class list."""
        return [{"name": c, "source": self._source[c],
                 "examples": self.examples(c), "taught": self._taught.get(c, 0),
                 "thin": self.is_thin(c), "threshold": round(self._thr[c], 4),
                 "calibrated": self.is_calibrated(c)} for c in self.classes]

    def update_card(self, model_dir) -> Path:
        """Write the taught classes (with example counts) into the card of
        the classifier in `model_dir` (plan B5)."""
        from atk_diffusion import cards as _cards
        card = _cards.load(model_dir, for_profile=self.profile, verify_weights=False)
        mine = {d["name"]: d for d in self.card_classes()}
        merged, seen = [], set()
        for c in card.classes:
            name = c["name"] if isinstance(c, dict) else str(c)
            entry = dict(c) if isinstance(c, dict) else {"name": name}
            if name in mine:
                entry.update({k: mine[name][k] for k in ("examples", "thin", "taught")})
                if mine[name]["taught"] and entry.get("source") != "trained":
                    entry["source"] = "taught"
            merged.append(entry)
            seen.add(name)
        for name, d in mine.items():
            if name not in seen:
                merged.append({"name": name, "source": d["source"],
                               "examples": d["examples"], "thin": d["thin"],
                               "taught": d["taught"]})
        card.classes = merged
        return _cards.save(model_dir, card)

    # -- persistence -------------------------------------------------------------
    def save(self, model_dir) -> Path:
        d = Path(model_dir)
        d.mkdir(parents=True, exist_ok=True)
        names = self.classes
        meta = {"format": FORMAT, "profile": self.profile,
                "canonical_class": self.canonical_class, "dim": self.dim,
                "quantile": self.quantile, "widen": self.widen,
                "example_floor": self.example_floor, "k_sub": self.k_sub,
                "model_sha256": self.model_sha256,
                "saved": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "notes": list(self.notes),
                "classes": [{"name": c, "index": i, "source": self._source[c],
                             "examples": self.examples(c),
                             "taught": self._taught.get(c, 0),
                             "taught_at": self._taught_at.get(c, ""),
                             "threshold": self._thr[c], "base": self._base[c],
                             "calibrated": self.is_calibrated(c),
                             "thin": self.is_thin(c)}
                            for i, c in enumerate(names)]}
        arrays = {}
        for i, c in enumerate(names):
            arrays[f"ex_{i}"] = self._ex[c].astype(np.float32)
            arrays[f"pr_{i}"] = self._proto[c].astype(np.float32)
        npz = d / FILE_NPZ
        tmp = d / (FILE_NPZ + ".tmp.npz")
        np.savez(tmp, **arrays) if arrays else np.savez(tmp, empty=np.zeros(0))
        tmp.replace(npz)
        js = d / FILE_JSON
        tmpj = d / (FILE_JSON + ".tmp")
        tmpj.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        tmpj.replace(js)
        return js

    @classmethod
    def load(cls, model_dir, profile, canonical_class: str | None = None,
             model_sha256: str | None = None) -> "PrototypeBank":
        """Load a bank, refusing (in words) another profile, another
        canonical class, or embeddings made by other classifier weights."""
        d = Path(model_dir)
        js = d / FILE_JSON
        if not js.exists():
            raise FileNotFoundError(f"{d.name} has no {FILE_JSON}: no classes "
                                    "have been taught or registered there")
        meta = json.loads(js.read_text(encoding="utf-8"))
        if meta.get("format") != FORMAT:
            raise ValueError(f"{js.name} is not a prototype bank this version "
                             f"reads (format {meta.get('format')!r})")
        want = str(getattr(profile, "id", profile)).strip().lower()
        have = str(meta.get("profile", "")).lower()
        if have != want:
            try:
                a, b = _profiles.describe(have), _profiles.describe(want)
            except ValueError:
                a, b = repr(have), repr(want)
            raise _profiles.ProfileMismatch(
                f"these prototypes were made for {a}; this is {b}. Profiles "
                "never mix: teach the classes again from this receiver's "
                "captures.")
        if canonical_class and meta.get("canonical_class") != canonical_class:
            raise ValueError(f"these prototypes are for {meta.get('canonical_class')} "
                             f"cuts, not {canonical_class} cuts")
        if model_sha256 is None and (d / "card.json").exists():
            try:
                card = json.loads((d / "card.json").read_text(encoding="utf-8"))
                model_sha256 = (card.get("weights") or {}).get("sha256") or None
            except (OSError, json.JSONDecodeError):
                model_sha256 = None
        saved_sha = str(meta.get("model_sha256", "") or "")
        if model_sha256 and saved_sha and saved_sha != model_sha256:
            raise ValueError(
                f"these prototypes were made from the embeddings of a different "
                f"classifier (weights {saved_sha[:12]}…, this one is "
                f"{model_sha256[:12]}…). Embeddings from one network mean "
                "nothing to another: re-embed the examples with this one.")
        bank = cls(want, meta["canonical_class"], quantile=meta.get("quantile", 0.95),
                   widen=meta.get("widen", 1.5),
                   example_floor=meta.get("example_floor", EXAMPLE_FLOOR),
                   k_sub=meta.get("k_sub", 1), model_sha256=saved_sha)
        bank.dim = meta.get("dim")
        bank.notes = list(meta.get("notes", []))
        with np.load(d / FILE_NPZ) as z:
            for c in meta.get("classes", []):
                i, name = c["index"], c["name"]
                bank._ex[name] = l2_normalize(z[f"ex_{i}"])
                bank._proto[name] = l2_normalize(z[f"pr_{i}"])
                bank._thr[name] = float(c["threshold"])
                bank._base[name] = float(c.get("base", c["threshold"]))
                bank._calibrated[name] = bool(c.get("calibrated", False))
                bank._source[name] = c.get("source", "trained")
                if c.get("taught"):
                    bank._taught[name] = int(c["taught"])
                if c.get("taught_at"):
                    bank._taught_at[name] = c["taught_at"]
        return bank
