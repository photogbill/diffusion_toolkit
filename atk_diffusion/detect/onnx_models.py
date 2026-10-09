# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The learned parts at inference time: ONNX Runtime on the CPU
(DETECTION_DESIGN §7, decision D4; ARCHITECTURE §3, §5).

*"The cognitive core owns the GPU under ATK's lease and the one AI queue; a
detector that watches a live waterfall cannot stand in that queue. So
inference runs on the CPU through ONNX Runtime, which also keeps the deployed
model free of a PyTorch dependency inside ATK. Budget: one tile per second of
capture, under 200 ms on the 14-core CPU for the detector and under 20 ms per
cutout for the classifier."*

`Proposer2D(model_dir)` and `Classifier1D(model_dir)` load ONLY through
`cards.load(..., expect_kind=..., for_profile=...)`: a model without a card,
for another profile, of another kind, or whose weights are not the ones its
card hashed, does not load — `CardRefusal` / `ProfileMismatch` propagate
unchanged, because their messages already are the plain sentences the AI
Detect tab shows. The ONNX graph's input and output NAMES are checked against
the contract (ARCHITECTURE §5) before the first run, and a model exported
without them is refused in words (a real hazard: the legacy exporter renames
an output it constant-folds).

    proposer2d    in   tile       float32 [1, 1, rows, bins]   dB above floor,
                                  normalised as card.input["normalize"] says
                  out  boxes      float32 [K, 4]   (row0, bin0, row1, bin1) pixels
                       scores     float32 [K]
                       labels     int64   [K]      index into card.input["families"]
    classifier1d  in   iq         float32 [B, 2, L]  (I, Q) at the canonical rate
                       scf        float32 [B, 1, H, W]
                  out  logits     [B, C]   C = len(card.classes)
                       embedding  [B, D]   L2-normalised
                       cycle      [B, 2]   (symbol rate, carrier offset) /
                                           card.input["cycle_scale"]

CARD FIELDS THESE WRAPPERS READ (the training side writes them):
  proposer2d    input.stft {fft_size, hop, window, tile_rows}, input.families,
                input.normalize, calibration.min_score (default 0.3)
  classifier1d  input.iq_len, input.canonical {cls (or class), rate, decimation},
                input.scf_used (False: the graph ignores its scf input; zeros
                are fed and no SCF is computed),
                input.fam (FamGeometry fields), input.cycle_scale [sr, cfo],
                input.iq_norm ("rms" | "max" | "none"), input.scf_norm ("max" |
                "none"), input.scf_shape [H, W] (else the graph's static shape;
                passed to cyclo.scf.scf_image as out_shape), input.scf_conj,
                calibration.temperature (default 1)
  `normalize`: {"clip_db": [lo, hi], "offset_db": o, "scale_db": s} -> (clip −
  o) / s; or {"mean": m, "std": s}; or None. `normalize_tile` is public so the
  training code applies exactly the same transform.

Every call is timed (`latency` — calls, last, mean, peak ms), because the
card's CPU-latency number has to be re-measurable on the machine that runs
it. Detections from the proposer are PROPOSED tier with sources ("learned",)
and a FAMILY, never a modulation (§3: the spectrogram cannot tell QPSK from
8PSK and does not pretend to); their SNR is measured classically from the
tile beside them.

LIMITS. A family index outside the card's list becomes "unknown" (counted in
`latency["bad_labels"]`). Scores are taken as calibrated only if the card's
calibration says they were; otherwise they are the network's own.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from atk_diffusion import cards as _cards
from atk_diffusion.detect import classes as _classes
from atk_diffusion.detect.boxes import Detection

PROPOSER_IO = (("tile",), ("boxes", "scores", "labels"))
CLASSIFIER_IO = (("iq", "scf"), ("logits", "embedding", "cycle"))


def _ort():
    try:
        import onnxruntime as ort
    except ImportError:
        from atk_diffusion.capabilities import can_infer
        raise RuntimeError(can_infer()[1] or "ONNX Runtime is not installed") from None
    return ort


def make_session(path, threads: int = 0):
    """A CPU-only InferenceSession. `threads` 0 lets ONNX Runtime choose."""
    ort = _ort()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if threads:
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=so,
                                providers=["CPUExecutionProvider"])


def check_io(session, inputs, outputs, what: str) -> None:
    ins = [i.name for i in session.get_inputs()]
    outs = [o.name for o in session.get_outputs()]
    if [n for n in inputs if n not in ins] or [n for n in outputs if n not in outs]:
        raise _cards.CardRefusal(
            f"{what}'s ONNX graph has inputs {ins} and outputs {outs}; the "
            f"contract (ARCHITECTURE §5) is inputs {list(inputs)} and outputs "
            f"{list(outputs)}. Re-export it with those names.")


class Latency:
    """Per-call wall time, kept for the last `keep` calls."""

    def __init__(self, keep: int = 200):
        self._ms: deque = deque(maxlen=int(keep))
        self.calls = 0
        self.extra: dict = {}

    def add(self, ms: float) -> None:
        self._ms.append(float(ms))
        self.calls += 1

    def summary(self) -> dict:
        if not self._ms:
            return {"calls": 0, **self.extra}
        a = np.array(self._ms)
        return {"calls": self.calls, "last_ms": float(a[-1]),
                "mean_ms": float(a.mean()), "peak_ms": float(a.max()), **self.extra}

    def __getitem__(self, k):
        return self.summary()[k]


def normalize_tile(spec, norm) -> np.ndarray:
    """The proposer's input transform, as the card states it (module
    docstring). Training code imports this so both sides agree exactly."""
    x = np.asarray(spec, dtype=np.float32)
    if not norm or norm == "none":
        return x
    if not isinstance(norm, dict):
        raise ValueError(f"the card's input.normalize {norm!r} is not a form "
                         "this toolkit applies ({'clip_db', 'offset_db', "
                         "'scale_db'} or {'mean', 'std'} or none)")
    keys = set(norm)
    if keys <= {"clip_db", "offset_db", "scale_db"}:
        if "clip_db" in norm:
            lo, hi = norm["clip_db"]
            x = np.clip(x, float(lo), float(hi))
        scale = float(norm.get("scale_db", 1.0)) or 1.0
        return ((x - float(norm.get("offset_db", 0.0))) / scale).astype(np.float32)
    if keys <= {"mean", "std"} and "mean" in keys:
        std = float(norm.get("std", 1.0)) or 1.0
        return ((x - float(norm["mean"])) / std).astype(np.float32)
    raise ValueError(f"the card's input.normalize has keys {sorted(keys)}; this "
                     "toolkit applies {'clip_db', 'offset_db', 'scale_db'} or "
                     "{'mean', 'std'}")


# ---------------------------------------------------------------------------
class Proposer2D:
    """The 2D AI proposer (FCOS-class) on the tile -> learned boxes."""

    def __init__(self, model_dir, profile: str | None = None, threads: int = 0,
                 min_score: float | None = None):
        self.model_dir = Path(model_dir)
        pid = None if profile is None else str(getattr(profile, "id", profile))
        self.card = _cards.load(self.model_dir, expect_kind="proposer2d",
                                for_profile=pid)
        if not self.card.weights:
            raise _cards.CardRefusal(f"{self.model_dir.name}'s card names no "
                                     "weights file")
        self.session = make_session(_cards.weights_path(self.model_dir, self.card),
                                    threads)
        check_io(self.session, *PROPOSER_IO, what=f"the proposer {self.card.name}")
        inp = self.card.input or {}
        self.families = list(inp.get("families") or _classes.FAMILIES)
        self.normalize = inp.get("normalize")
        normalize_tile(np.zeros((1, 1), np.float32), self.normalize)   # validate
        self.stft = dict(inp.get("stft") or {})
        cal = self.card.calibration or {}
        self.min_score = float(min_score if min_score is not None
                               else cal.get("min_score", inp.get("min_score", 0.3)))
        self.latency = Latency()
        self.latency.extra["bad_labels"] = 0
        shp = self.session.get_inputs()[0].shape
        self._static = tuple(int(v) for v in shp[2:4]) \
            if len(shp) == 4 and all(isinstance(v, int) for v in shp[2:4]) else None

    @property
    def name(self) -> str:
        return self.card.name

    def check_tile(self, tile) -> None:
        """Refuse, in words, a tile of a geometry the model was not trained on."""
        lay = tile.layout
        want = {"fft_size": lay.fft_size, "hop": lay.hop, "window": lay.window,
                "tile_rows": lay.rows}
        bad = [f"{k} {self.stft[k]!r} (this tile: {v!r})" for k, v in want.items()
               if k in self.stft and str(self.stft[k]).lower() != str(v).lower()]
        if self._static and self._static != (tile.rows, tile.bins):
            bad.append(f"input {self._static[0]} x {self._static[1]} (this tile: "
                       f"{tile.rows} x {tile.bins})")
        if bad:
            raise ValueError(f"the proposer {self.card.name} was trained on another "
                             "spectrogram geometry — " + "; ".join(bad)
                             + ". A model never meets a geometry it was not "
                             "trained on (DETECTION_DESIGN §2).")

    def run(self, tile, min_score: float | None = None) -> list[Detection]:
        self.check_tile(tile)
        x = normalize_tile(tile.spec, self.normalize)[None, None].astype(np.float32)
        t0 = time.perf_counter()
        boxes, scores, labels = self.session.run(list(PROPOSER_IO[1]), {"tile": x})
        self.latency.add((time.perf_counter() - t0) * 1e3)
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).ravel()
        labels = np.asarray(labels).ravel().astype(np.int64)
        if not (boxes.shape[0] == scores.size == labels.size):
            raise ValueError(f"the proposer {self.card.name} returned {boxes.shape[0]} "
                             f"boxes, {scores.size} scores and {labels.size} labels")
        thr = self.min_score if min_score is None else float(min_score)
        rv = tile.rows_valid or tile.rows
        out = []
        for (r0, b0, r1, b1), s, lab in zip(boxes, scores, labels):
            if not (np.isfinite([r0, b0, r1, b1, s]).all()) or s < thr:
                continue
            r0, r1 = sorted((min(max(r0, 0.0), rv), min(max(r1, 0.0), rv)))
            b0, b1 = sorted((min(max(b0, 0.0), tile.bins), min(max(b1, 0.0), tile.bins)))
            if r1 - r0 <= 0 or b1 - b0 <= 0:
                continue
            if 0 <= lab < len(self.families) and self.families[lab] in _classes.FAMILIES:
                fam = self.families[lab]
            else:
                fam = "unknown"
                self.latency.extra["bad_labels"] += 1
            t_lo, t_hi, f_lo, f_hi = tile.pixels_to_tf(r0, b0, r1, b1)
            snr = tile.box_snr_db(r0, b0, r1, b1)
            out.append(Detection(t0=t_lo, t1=t_hi, f_lo=f_lo, f_hi=f_hi,
                                 sources=("learned",), family=fam,
                                 confidence=round(float(s), 4),
                                 snr_db=None if snr is None else round(snr, 2),
                                 profile=self.card.profile or tile.profile,
                                 epoch=tile.epoch,
                                 measurements={"model": self.card.name}))
        return out


# ---------------------------------------------------------------------------
@dataclass
class ClassifierOutput:
    """One Classifier1D call. Rows follow the input windows."""
    probs: np.ndarray            # [B, C] calibrated softmax
    classes: list                # the card's class names (C)
    top: list                    # [B] argmax class names
    confidence: np.ndarray       # [B] probability of the top class
    embeddings: np.ndarray       # [B, D] L2-normalised
    cycle_hz: np.ndarray         # [B, 2] (symbol rate Hz, carrier offset Hz) — the
                                 # model's opinion; dsp.measure has the classical one
    latency_ms: float = 0.0
    notes: list = field(default_factory=list)


class Classifier1D:
    """The 1D classifier (IQ + SCF -> class, embedding, cycle parameters)."""

    def __init__(self, model_dir, profile: str | None = None, threads: int = 0):
        self.model_dir = Path(model_dir)
        pid = None if profile is None else str(getattr(profile, "id", profile))
        self.card = _cards.load(self.model_dir, expect_kind="classifier1d",
                                for_profile=pid)
        if not self.card.weights:
            raise _cards.CardRefusal(f"{self.model_dir.name}'s card names no "
                                     "weights file")
        self.session = make_session(_cards.weights_path(self.model_dir, self.card),
                                    threads)
        check_io(self.session, *CLASSIFIER_IO,
                 what=f"the classifier {self.card.name}")
        inp = self.card.input or {}
        if "iq_len" not in inp:
            raise _cards.CardRefusal(f"{self.card.name}'s card does not say its IQ "
                                     "window length (input.iq_len)")
        self.iq_len = int(inp["iq_len"])
        self.classes = self.card.class_names()
        if not self.classes:
            raise _cards.CardRefusal(f"{self.card.name}'s card lists no classes")
        self.canonical = dict(inp.get("canonical") or {})
        self.fam = dict(inp.get("fam") or {})
        cs = np.asarray(inp.get("cycle_scale", [1.0, 1.0]), dtype=np.float64).ravel()
        self.cycle_scale = np.resize(cs, 2) if cs.size else np.ones(2)
        self.temperature = float((self.card.calibration or {}).get("temperature", 1.0)) or 1.0
        self.iq_norm = str(inp.get("iq_norm", "rms"))
        self.scf_norm = str(inp.get("scf_norm", "max"))
        if self.iq_norm not in ("rms", "max", "none"):
            raise _cards.CardRefusal(f"{self.card.name}: unknown input.iq_norm "
                                     f"{self.iq_norm!r}")
        ins = {i.name: i for i in self.session.get_inputs()}
        iq_shape, scf_shape = ins["iq"].shape, ins["scf"].shape
        self.batch_fixed = iq_shape[0] if isinstance(iq_shape[0], int) else None
        if isinstance(iq_shape[-1], int) and iq_shape[-1] != self.iq_len:
            raise _cards.CardRefusal(f"{self.card.name}: the graph takes {iq_shape[-1]} "
                                     f"samples but the card says iq_len {self.iq_len}")
        if inp.get("scf_shape"):
            self.scf_shape = tuple(int(v) for v in inp["scf_shape"])
        elif len(scf_shape) == 4 and all(isinstance(v, int) for v in scf_shape[2:]):
            self.scf_shape = (int(scf_shape[2]), int(scf_shape[3]))
        else:
            self.scf_shape = None
        # a model trained without the SCF branch accepts and ignores the input:
        # it is fed zeros, and no SCF is computed for it
        self.scf_used = bool(inp.get("scf_used", True))
        self.latency = Latency()

    @property
    def name(self) -> str:
        return self.card.name

    @property
    def canonical_class(self) -> str:
        # the dataset manifest says "class"; either spelling is the same fact
        return str(self.canonical.get("cls") or self.canonical.get("class") or "")

    @property
    def canonical_rate(self) -> float:
        return float(self.canonical.get("rate", 0.0))

    # -- inputs ----------------------------------------------------------------
    def _iq_tensor(self, windows) -> np.ndarray:
        w = np.asarray(windows)
        if w.ndim == 1:
            w = w[None, :]
        if w.ndim != 2 or not np.iscomplexobj(w):
            raise ValueError("the classifier takes complex IQ windows, [B, L]")
        if w.shape[1] != self.iq_len:
            raise ValueError(f"the classifier {self.card.name} takes windows of "
                             f"{self.iq_len} samples at its canonical rate; got "
                             f"{w.shape[1]}")
        w = w.astype(np.complex64)
        if self.iq_norm == "rms":
            s = np.sqrt(np.mean(np.abs(w) ** 2, axis=1, keepdims=True))
            w = w / np.maximum(s, 1e-12)
        elif self.iq_norm == "max":
            s = np.max(np.abs(w), axis=1, keepdims=True)
            w = w / np.maximum(s, 1e-12)
        return np.stack([w.real, w.imag], axis=1).astype(np.float32)

    def _scf_tensor(self, windows, scf_images) -> np.ndarray:
        if scf_images is None and not self.scf_used:
            h, w = self.scf_shape or (1, 1)
            return np.zeros((np.asarray(windows).reshape(-1, self.iq_len).shape[0],
                             1, h, w), dtype=np.float32)
        if scf_images is None:
            imgs = self.scf_for(windows)
        else:
            imgs = np.asarray(scf_images, dtype=np.float32)
            if imgs.ndim == 3:
                imgs = imgs[:, None]
            if imgs.ndim != 4 or imgs.shape[1] != 1:
                raise ValueError("SCF images are [B, H, W] or [B, 1, H, W]")
        if self.scf_shape and tuple(imgs.shape[2:]) != self.scf_shape:
            raise ValueError(f"the SCF image is {imgs.shape[2]} x {imgs.shape[3]}; "
                             f"the classifier {self.card.name} was trained on "
                             f"{self.scf_shape[0]} x {self.scf_shape[1]} — a model "
                             "never meets an SCF of a geometry it was not trained on")
        if self.scf_norm == "max":
            m = np.max(np.abs(imgs), axis=(1, 2, 3), keepdims=True)
            imgs = imgs / np.maximum(m, 1e-12)
        return imgs.astype(np.float32)

    def scf_for(self, windows) -> np.ndarray:
        """The SCF image of each window with the card's FAM geometry, through
        atk_diffusion.cyclo.scf (imported here, only when needed)."""
        try:
            from atk_diffusion.cyclo.scf import scf_image
        except ImportError as e:
            raise RuntimeError(
                "the classifier needs each cut's spectral correlation image and "
                f"atk_diffusion.cyclo.scf is not available here ({e}); pass "
                "scf_images computed elsewhere") from None
        from atk_diffusion.profiles import FamGeometry
        geom = FamGeometry(**{k: v for k, v in self.fam.items()
                              if k in FamGeometry.__dataclass_fields__})
        rate = self.canonical_rate
        if rate <= 0:
            raise ValueError(f"{self.card.name}'s card does not state its canonical "
                             "rate (input.canonical.rate)")
        w = np.asarray(windows)
        if w.ndim == 1:
            w = w[None, :]
        kw = {"conj": bool((self.card.input or {}).get("scf_conj", False))}
        if self.scf_shape:
            kw["out_shape"] = tuple(self.scf_shape)     # the grid it was trained on
        imgs = []
        for row in w:
            out = scf_image(row.astype(np.complex64), rate, geom, **kw)
            img = out[0] if isinstance(out, tuple) else out
            img = np.asarray(img)
            imgs.append(np.abs(img) if np.iscomplexobj(img) else img)
        return np.asarray(imgs, dtype=np.float32)[:, None]

    # -- running ---------------------------------------------------------------
    def run(self, iq_windows, scf_images=None) -> ClassifierOutput:
        iq = self._iq_tensor(iq_windows)
        scf = self._scf_tensor(np.asarray(iq_windows).reshape(iq.shape[0], -1),
                               scf_images)
        if scf.shape[0] != iq.shape[0]:
            raise ValueError(f"{iq.shape[0]} IQ windows but {scf.shape[0]} SCF images")
        names = list(CLASSIFIER_IO[1])
        t0 = time.perf_counter()
        if self.batch_fixed:
            parts = []
            for i in range(0, iq.shape[0], self.batch_fixed):
                a, b = iq[i:i + self.batch_fixed], scf[i:i + self.batch_fixed]
                pad = self.batch_fixed - a.shape[0]
                if pad:
                    a = np.concatenate([a, np.zeros((pad,) + a.shape[1:], a.dtype)])
                    b = np.concatenate([b, np.zeros((pad,) + b.shape[1:], b.dtype)])
                r = self.session.run(names, {"iq": a, "scf": b})
                parts.append([np.asarray(v)[:self.batch_fixed - pad] for v in r])
            logits, emb, cyc = (np.concatenate([p[k] for p in parts]) for k in range(3))
        else:
            logits, emb, cyc = (np.asarray(v) for v in
                                self.session.run(names, {"iq": iq, "scf": scf}))
        ms = (time.perf_counter() - t0) * 1e3
        self.latency.add(ms)
        logits = np.asarray(logits, dtype=np.float64).reshape(iq.shape[0], -1)
        if logits.shape[1] != len(self.classes):
            raise ValueError(f"the classifier {self.card.name} outputs {logits.shape[1]} "
                             f"logits but its card lists {len(self.classes)} classes")
        z = logits / self.temperature
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        emb = np.asarray(emb, dtype=np.float64).reshape(iq.shape[0], -1)
        nrm = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(nrm, 1e-12)
        cyc = np.asarray(cyc, dtype=np.float64).reshape(iq.shape[0], -1)[:, :2]
        top_i = np.argmax(p, axis=1)
        return ClassifierOutput(
            probs=p, classes=list(self.classes),
            top=[self.classes[i] for i in top_i],
            confidence=p[np.arange(p.shape[0]), top_i],
            embeddings=emb, cycle_hz=cyc * self.cycle_scale[None, :],
            latency_ms=ms,
            notes=[] if self.temperature != 1.0 else
            ["confidences are the network's own (no temperature on the card)"])
