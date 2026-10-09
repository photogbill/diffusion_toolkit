# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""ONNX export, verification against PyTorch, and CPU latency
(DETECTION_DESIGN §7, decision D4; ARCHITECTURE §2.6, §5).

Inference inside ATK is ONNX Runtime on the CPU — the GPU belongs to the
cognitive core — so every trained model leaves this repo as an ONNX file and
a card. This module is the one way that happens:

* `export()` — the TorchScript-based exporter (`dynamo=False`), opset 17.
  WHY NOT THE NEWER EXPORTER: `torch.onnx.export(dynamo=True)` needs the
  `onnxscript` package, which the training environment does not ship, and
  torchvision's detection models carry tracing-aware code paths
  (`_topk_min`, the NMS coordinate trick, `_onnx_batch_images`) written for
  the TorchScript exporter. Opset 17: NonMaxSuppression (11+) and every op
  the models use; ONNX Runtime 1.30 runs it. The exported graph is checked
  with `onnx.checker`, and its input and output names are checked against
  the contract (ARCHITECTURE §5) — a renamed output is a refused export.
* `verify()` — runs the same inputs through PyTorch and ONNX Runtime and
  compares every output; a model whose graph disagrees with its weights is
  not saved (`ExportMismatch`).
* `latency()` — warm-up then timed runs on the CPU execution provider; the
  card records the median, the 95th percentile, the thread count and the
  machine, because a number measured on one machine is not a promise about
  another (the budget is set on Bill's 14-core CPU: 200 ms a tile, 20 ms a
  cutout).
* `OnnxRunner` — load a model through its card (refusing another profile or
  kind) and run it. Needs only numpy and onnxruntime, so evaluation runs in
  ATK's core environment as well as the training one.
* `save_model()` — `cards.save` with the ONNX file as the weights (its
  SHA-256 goes in the card), plus the PyTorch state kept beside it for
  fine-tuning (also hashed), and both recorded in the rf_data write log.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np

from atk_diffusion import cards as _cards
from atk_diffusion.learn import common as K
from atk_diffusion.provenance import sha256_path

DEFAULT_OPSET = 17
ONNX_FILE = "model.onnx"
TORCH_FILE = "model.pt"

#: The fixed I/O names (ARCHITECTURE §5).
IO = {"proposer2d": (("tile",), ("boxes", "scores", "labels")),
      "classifier1d": (("iq", "scf"), ("logits", "embedding", "cycle"))}


class ExportMismatch(RuntimeError):
    """The exported graph does not behave like the trained model."""


def export(module, example_inputs: tuple, path, input_names, output_names,
           dynamic_axes: dict | None = None, opset: int = DEFAULT_OPSET) -> Path:
    """Export `module` (eval mode, no grad) to `path` and check the graph."""
    torch = K.require_torch()
    import onnx
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    module.eval()
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(module, tuple(example_inputs), str(p),
                          input_names=list(input_names),
                          output_names=list(output_names),
                          dynamic_axes=dynamic_axes or None,
                          opset_version=int(opset), dynamo=False,
                          do_constant_folding=True)
    model = onnx.load(str(p))
    try:
        onnx.checker.check_model(model)
    except Exception as e:                                 # noqa: BLE001
        raise ExportMismatch(f"the exported graph failed ONNX's own check "
                             f"({e}); it was not saved as a model.") from None
    got_in = [i.name for i in model.graph.input
              if i.name not in {t.name for t in model.graph.initializer}]
    got_out = [o.name for o in model.graph.output]
    if got_in != list(input_names) or got_out != list(output_names):
        raise ExportMismatch(f"the exported graph has inputs {got_in} and "
                             f"outputs {got_out}; the contract is "
                             f"{list(input_names)} -> {list(output_names)} "
                             "(ARCHITECTURE §5).")
    return p


def session(path, threads: int | None = None):
    """An ONNX Runtime CPU session (the only provider ATK uses by default)."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so,
                                providers=["CPUExecutionProvider"])


def verify(path, module, feeds_list: list[dict], rtol: float = 1e-3,
           atol: float = 1e-4, threads: int | None = 1) -> dict:
    """Run each feed through PyTorch and ONNX Runtime and compare outputs.

    Returns {"ok", "cases", "max_abs_diff": {output: value}, "why"}. Shapes
    must agree exactly (a detector that finds a different NUMBER of boxes in
    the graph is a mismatch, not a tolerance question)."""
    torch = K.require_torch()
    sess = session(path, threads)
    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]
    worst = {n: 0.0 for n in out_names}
    why = ""
    module.eval()
    for case, feeds in enumerate(feeds_list):
        args = [torch.from_numpy(np.ascontiguousarray(feeds[n])) for n in in_names]
        with torch.no_grad():
            ref = module(*args)
        if isinstance(ref, torch.Tensor):
            ref = (ref,)
        got = sess.run(None, {n: np.ascontiguousarray(feeds[n]) for n in in_names})
        for name, r, g in zip(out_names, ref, got):
            r = r.detach().cpu().numpy()
            if r.shape != g.shape:
                why = (f"case {case}: output {name!r} has shape {g.shape} in "
                       f"ONNX Runtime and {r.shape} in PyTorch")
                return {"ok": False, "cases": case + 1, "max_abs_diff": worst,
                        "why": why}
            if r.size == 0:
                continue
            if np.issubdtype(r.dtype, np.integer):
                if not np.array_equal(r, g):
                    why = f"case {case}: integer output {name!r} differs"
                    return {"ok": False, "cases": case + 1,
                            "max_abs_diff": worst, "why": why}
                continue
            d = float(np.max(np.abs(r.astype(np.float64) - g.astype(np.float64))))
            worst[name] = max(worst[name], d)
            if not np.allclose(g, r, rtol=rtol, atol=atol):
                why = (f"case {case}: output {name!r} differs by up to {d:.3g} "
                       f"(tolerance rtol={rtol:g}, atol={atol:g})")
                return {"ok": False, "cases": case + 1, "max_abs_diff": worst,
                        "why": why}
    return {"ok": True, "cases": len(feeds_list), "max_abs_diff": worst,
            "why": "", "rtol": rtol, "atol": atol}


def latency(path, feeds: dict, repeats: int = 20, warmup: int = 3,
            threads: int | None = None) -> dict:
    """CPU latency of one run, in milliseconds (median, p95, mean, min)."""
    sess = session(path, threads)
    feeds = {k: np.ascontiguousarray(v) for k, v in feeds.items()}
    for _ in range(max(0, int(warmup))):
        sess.run(None, feeds)
    times = []
    for _ in range(max(1, int(repeats))):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        times.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(times)
    return {"p50_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)),
            "mean_ms": float(a.mean()), "min_ms": float(a.min()),
            "repeats": int(a.size), "warmup": int(warmup),
            "threads": int(threads) if threads else "onnxruntime default",
            "machine": K.machine_words()}


class OnnxRunner:
    """A model loaded through its card, run on the CPU.

        r = OnnxRunner(model_dir, "proposer2d", for_profile=profile)
        out = r.run(tile=x)          # {"boxes": …, "scores": …, "labels": …}
    """

    def __init__(self, model_dir, expect_kind: str, for_profile: str | None = None,
                 threads: int | None = None):
        self.dir = Path(model_dir)
        self.card = _cards.load(self.dir, expect_kind=expect_kind,
                                for_profile=for_profile)
        self.path = _cards.weights_path(self.dir, self.card)
        self.sess = session(self.path, threads)
        self.inputs = {i.name: i.shape for i in self.sess.get_inputs()}
        self.outputs = [o.name for o in self.sess.get_outputs()]

    def run(self, **feeds) -> dict:
        got = self.sess.run(None, {k: np.ascontiguousarray(v)
                                   for k, v in feeds.items()})
        return dict(zip(self.outputs, got))


def host_normalize(spec, norm) -> np.ndarray:
    """What the HOST does to a tile before the proposer, per the card's
    `input.normalize` — through `detect.onnx_models.normalize_tile` (the
    function ATK's pipeline runs) when it can be imported, so evaluation
    here feeds exactly what the pipeline feeds. The proposers trained here
    write `normalize: "none"` (their graph clips and normalises itself)."""
    try:
        from atk_diffusion.detect.onnx_models import normalize_tile
    except ImportError:
        if not norm or norm == "none":
            return np.asarray(spec, np.float32)
        raise RuntimeError("this card asks the host to normalise its tiles, "
                           "and detect.onnx_models (which knows how) is not "
                           "importable here") from None
    return normalize_tile(spec, norm)


def run_proposer(runner: OnnxRunner, tiles, indices=None) -> list[dict]:
    """Proposer outputs for tiles of a `common.TileSet` (or an array
    (N, rows, bins) of dB above floor): [{"boxes" (K, 4) row0, bin0, row1,
    bin1, "scores", "labels"}]."""
    inp = runner.card.input or {}
    shape = tuple((inp.get("tile") or {}).get("shape")
                  or runner.inputs.get("tile") or ())
    norm = inp.get("normalize")
    out = []
    n = len(tiles)
    for i in (range(n) if indices is None else indices):
        x = tiles.spec(i) if hasattr(tiles, "spec") else np.asarray(tiles[i])
        x = np.asarray(x, np.float32)
        if shape and tuple(x.shape) != tuple(shape[-2:]):
            raise ValueError(f"this proposer takes tiles of {shape[-2]}×"
                             f"{shape[-1]}; this tile is {x.shape[0]}×"
                             f"{x.shape[1]}. A model never meets a spectrogram "
                             "of a geometry it was not trained on.")
        r = runner.run(tile=host_normalize(x, norm)[None, None])
        out.append({"boxes": r["boxes"].reshape(-1, 4), "scores": r["scores"],
                    "labels": r["labels"]})
    return out


def run_classifier(runner: OnnxRunner, shards, indices=None,
                   batch: int = 64) -> dict:
    """Classifier outputs for examples of a `common.ShardSet`: {"logits",
    "embedding", "cycle"} stacked over the examples, fed as training fed
    them and as `detect.onnx_models.Classifier1D` feeds them: a centred
    window of `input.iq_len` samples at unit RMS, the SCF scaled by
    `input.scf_norm` (or a zero when the card says the SCF branch is
    unused)."""
    inp = runner.card.input or {}
    window = int(inp.get("iq_len") or inp.get("window") or shards.L)
    use_scf = bool(inp.get("scf_used", True))
    scf_norm = str(inp.get("scf_norm", "max"))
    scf_shape = tuple(inp.get("scf_shape") or (1, 1))
    idx = list(range(len(shards))) if indices is None else list(indices)
    outs = {"logits": [], "embedding": [], "cycle": []}
    for s in range(0, len(idx), max(1, int(batch))):
        chunk = idx[s:s + batch]
        iq = np.stack([K.iq_window(shards.iq(i), window) for i in chunk])
        if use_scf:
            scf = np.stack([K.scf_normalize(shards.scf(i), scf_norm)
                            for i in chunk])
        else:
            scf = np.zeros((len(chunk), 1) + scf_shape, np.float32)
        r = runner.run(iq=iq, scf=scf)
        for k in outs:
            outs[k].append(r[k])
    return {k: (np.concatenate(v) if v else np.zeros((0,))) for k, v in outs.items()}


def save_model(model_dir, card, onnx_file: str = ONNX_FILE,
               train_state: str | None = TORCH_FILE, rf=None) -> Path:
    """Write the card with the ONNX file as its weights (hashed) and, when
    present, the PyTorch state beside it (hashed into
    `card.weights["train_state"]`); record every file in the write log."""
    d = Path(model_dir)
    _cards.save(d, card, onnx_file)
    if train_state and (d / train_state).exists():
        w = d / train_state
        card.weights["train_state"] = {"file": train_state,
                                       "sha256": sha256_path(w),
                                       "bytes": w.stat().st_size,
                                       "format": "pytorch state_dict"}
        _cards.save(d, card)
    p = d / _cards.CARD_FILE
    if rf is not None:
        for f in (d / onnx_file, d / (train_state or ""), p):
            if f.is_file():
                try:
                    rf.record(f, f"model:{card.kind}", card.name)
                except Exception:                          # noqa: BLE001
                    pass
    return p


def resave_card(model_dir, card, rf=None) -> Path:
    """Write an updated card (metrics, calibration) for an existing model;
    the weights are re-checked against the card, never re-hashed blindly."""
    d = Path(model_dir)
    w = d / card.weights["file"]
    if sha256_path(w) != card.weights.get("sha256"):
        raise _cards.CardRefusal(f"{w.name} is not the file the card "
                                 "describes; the card was not updated.")
    _cards.save(d, card)
    p = d / _cards.CARD_FILE
    if rf is not None:
        try:
            rf.record(p, f"model:{card.kind}", card.name)
        except Exception:                                  # noqa: BLE001
            pass
    return p


def load_train_state(model_dir, card):
    """The PyTorch state saved beside the ONNX, checked against its hash."""
    torch = K.require_torch()
    ts = (card.weights or {}).get("train_state")
    if not ts:
        raise _cards.CardRefusal(f"{Path(model_dir).name} has no PyTorch state "
                                 "beside its ONNX file, so it cannot be "
                                 "fine-tuned (only run).")
    w = Path(model_dir) / ts["file"]
    if not w.exists():
        raise _cards.CardRefusal(f"{w.name} is missing from "
                                 f"{Path(model_dir).name}.")
    if ts.get("sha256") and sha256_path(w) != ts["sha256"]:
        raise _cards.CardRefusal(f"{w.name} is not the file the card describes "
                                 "(its hash changed). Retrain or restore it.")
    return torch.load(w, map_location="cpu", weights_only=True)
