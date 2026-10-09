# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Shared plumbing for training and evaluation (ARCHITECTURE §4.4, §5;
DETECTION_DESIGN §6, §11).

WHAT IS HERE, and why it is one module:

* **The run** — device choice, seeding, thread count, and a run folder under
  `rf_data\\<profile>\\runs\\<UTC stamp>_<name>\\` holding `run.log` (plain
  sentences, including which device trained and why), `metrics.jsonl` (one
  JSON object per measurement) and `checkpoints\\`. Plan §2.9: nothing happens
  silently — a CPU run on a machine that has a GPU says so, and says why.
* **Datasets** — the built-dataset layout of ARCHITECTURE §5 read with numpy:
  the manifest (refused, in words, when its profile is not the run's — the
  sample-rate law, plan §3.3), narrowband shards and wideband tile files,
  opened lazily so a 20 GB dataset is not read into memory to train on it.
* **What a model sees** — the tile normalisation (`TileNorm`, recorded in
  the card), the IQ window at unit RMS, and the SCF scaling, in numpy, so
  ONNX inference without PyTorch feeds exactly what training fed. The SCF
  scaling uses the words `detect.onnx_models.Classifier1D` applies
  (`scf_norm`), so training and inference can never drift apart.
* **Ground truth for the proposer** — one function decides how a dataset's
  boxes become training and scoring targets (the family list, explicit
  negatives, and the minimum box size, below), so training, the domain gap
  and the evaluation harness can never score against different truths.
* **Metrics** — detection AP/mAP (VOC all-point and COCO-style 0.5:0.95),
  precision/recall/F1 at an operating point, accuracy, macro-F1, curves over
  SNR. numpy only; pycocotools is not a dependency.

TORCH-FREE AT IMPORT. `torch` is imported inside the few functions that need
it. That keeps the evaluation harness (`experiments.detector_eval`,
`experiments.domain_gap`) runnable in ATK's core environment, which has
numpy, scipy and onnxruntime but no PyTorch (ARCHITECTURE §1).

THE MINIMUM BOX SIZE, stated because it changes what a box means. FCOS (the
2D proposer, decision D3) learns a box from the feature-map points that fall
INSIDE it; at the finest level those points are `stride` pixels apart. A
narrowband voice signal on an RTL-SDR tile is 3–5 FFT bins wide (2.34 kHz a
bin at 2.4 MS/s) — narrower than the stride — so it can contain no point at
all and could never be learned. Boxes narrower (or shorter) than
`min_box_px` are therefore widened symmetrically to `min_box_px` for training
AND for scoring, and the model card says so. A proposer box is *where* to
cut; the cut's classical measurement (DETECTION_DESIGN §4, "Measurement,
classical") gives the true occupied bandwidth.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import re
import time
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from atk_diffusion import capabilities as _caps
from atk_diffusion import profiles as _profiles
from atk_diffusion.detect import classes as _classes
from atk_diffusion.provenance import sha256_path

MANIFEST = "manifest.json"
SPLITS = ("train", "val", "test")
KINDS = ("narrowband", "wideband")

#: Narrowband per-example label arrays (ARCHITECTURE §5), plus `t_s` — an
#: optional extension this module reads when present: seconds since the
#: start of an on-site session, which orders a capture in time for the
#: minutes-to-acceptable experiment (plan §3.6).
NB_LABEL_FIELDS = ("label", "family", "snr_db", "symbol_rate_hz",
                   "carrier_offset_hz", "bandwidth_hz")


# ---------------------------------------------------------------------------
# PyTorch, on demand
# ---------------------------------------------------------------------------
def require_torch():
    """Import PyTorch or raise the plain sentence of capabilities.can_train()."""
    ok, why = _caps.can_train()
    if not ok:
        raise RuntimeError(why)
    import torch
    return torch


def pick_device(prefer: str = "auto"):
    """(torch.device, sentence). CUDA when PyTorch can see a GPU, else the
    CPU — and the sentence says which and why, for the run log."""
    torch = require_torch()
    p = str(prefer or "auto").lower()
    if p == "cpu":
        return torch.device("cpu"), "training on the CPU (asked for)"
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gb = props.total_memory / float(1 << 30)
        return (torch.device("cuda:0"),
                f"training on the GPU: {props.name}, {gb:.0f} GB")
    if p in ("cuda", "gpu"):
        raise RuntimeError("a GPU was asked for, but PyTorch can see no CUDA "
                           "device on this machine. Check that the training "
                           "environment has the CUDA build of PyTorch and that "
                           "the cognitive core is not holding the GPU.")
    built = getattr(torch.version, "cuda", None)
    if built:
        why = (f"no CUDA device is visible to PyTorch (this build supports "
               f"CUDA {built}; the driver or the GPU is not available)")
    else:
        why = "this PyTorch build has no CUDA support"
    return (torch.device("cpu"),
            f"training on the CPU — {why}. The RTX 3080 Ti is used when the "
            "training environment has a CUDA build and the GPU is free.")


def seed_everything(seed: int) -> int:
    """Seed Python, numpy and PyTorch (CPU and every GPU). Returns the seed."""
    s = int(seed)
    random.seed(s)
    np.random.seed(s % (2 ** 32))
    torch = require_torch()
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    return s


def set_threads(n: int | None) -> int:
    """`torch.set_num_threads(n)` when n is given; returns the count in force.
    Tests pass 1; a training run on Bill's machine leaves it to PyTorch."""
    torch = require_torch()
    if n:
        torch.set_num_threads(max(1, int(n)))
    return int(torch.get_num_threads())


def machine_words() -> str:
    """'x86_64, 2 logical CPUs' — said next to every latency number, because
    a latency measured here is not a latency on Bill's 14-core machine."""
    proc = platform.processor() or platform.machine() or "unknown CPU"
    return f"{proc}, {os.cpu_count() or 0} logical CPUs, {platform.system()}"


# ---------------------------------------------------------------------------
# Run folders
# ---------------------------------------------------------------------------
_NAME_OK = re.compile(r"[^A-Za-z0-9_\-.]+")


def utc_stamp(t: float | None = None) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(time.time() if t is None
                                                      else t))


def safe_name(name: str) -> str:
    """A folder name from anything: letters, digits, '-', '_', '.'."""
    s = _NAME_OK.sub("_", str(name).strip()).strip("._")
    if not s:
        raise ValueError("a run or model needs a name")
    return s[:80]


def json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def write_json(path, obj) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=json_default),
                   encoding="utf-8")
    tmp.replace(p)
    return p


class Run:
    """One training or evaluation run: `rf.runs(profile)/<stamp>_<name>/`.

        run = Run.start(rf, profile, "proposer2d_v1", config, progress)
        run.log("training on the CPU — …")
        run.metric(epoch=1, split="train", loss=0.82)
        run.save_checkpoint({"model": sd, "epoch": 1})
        run.finish({"map50": 0.71})
    """

    def __init__(self, folder: Path, profile: str, name: str, progress=None,
                 rf=None):
        self.dir = Path(folder)
        self.profile = profile
        self.name = name
        self.progress = progress
        self.rf = rf
        self.started = time.time()

    @classmethod
    def start(cls, rf, profile: str, name: str, config: dict | None = None,
              progress=None) -> "Run":
        base = Path(rf.runs(profile))
        base.mkdir(parents=True, exist_ok=True)
        stem = f"{utc_stamp()}_{safe_name(name)}"
        folder = base / stem
        k = 2
        while folder.exists():
            folder = base / f"{stem}_{k}"
            k += 1
        folder.mkdir(parents=True)
        run = cls(folder, profile, name, progress, rf)
        if config is not None:
            write_json(folder / "config.json", config)
        run.log(f"run {folder.name} for {_safe_describe(profile)} "
                f"({profile}) started")
        return run

    @property
    def log_path(self) -> Path:
        return self.dir / "run.log"

    @property
    def metrics_path(self) -> Path:
        return self.dir / "metrics.jsonl"

    def log(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}  {message}"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self.progress is not None:
            try:
                self.progress(str(message))
            except Exception:                              # noqa: BLE001
                pass

    def metric(self, **values) -> dict:
        rec = {"t": round(time.time() - self.started, 3), **values}
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=json_default) + "\n")
        return rec

    def metrics(self) -> list[dict]:
        if not self.metrics_path.exists():
            return []
        out = []
        for line in self.metrics_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def save_checkpoint(self, state: dict, tag: str = "last") -> Path:
        torch = require_torch()
        d = self.dir / "checkpoints"
        d.mkdir(exist_ok=True)
        p = d / f"{safe_name(tag)}.pt"
        tmp = p.with_name(p.name + ".tmp")
        torch.save(state, tmp)
        tmp.replace(p)
        return p

    def finish(self, summary: dict) -> Path:
        summary = dict(summary)
        summary.setdefault("seconds", round(time.time() - self.started, 2))
        p = write_json(self.dir / "summary.json", summary)
        self.log(f"run finished in {summary['seconds']:.1f} s")
        if self.rf is not None:
            for f in (p, self.log_path, self.metrics_path):
                if f.exists():
                    try:
                        self.rf.record(f, "run", self.name)
                    except Exception:                      # noqa: BLE001
                        pass
        return p


def _safe_describe(profile: str) -> str:
    try:
        return _profiles.describe(profile)
    except ValueError:
        return repr(profile)


def new_model_dir(rf, profile: str, out_name: str, overwrite: bool = False
                  ) -> Path:
    """`rf.models(profile, out_name)`, refusing to overwrite a model that is
    already there unless asked — a trained model is not replaced silently."""
    d = Path(rf.models(profile, safe_name(out_name)))
    if d.exists() and any(d.iterdir()):
        if not overwrite:
            raise FileExistsError(f"a model called {d.name!r} already exists "
                                  f"for {_safe_describe(profile)}. Choose another "
                                  "name, or pass overwrite=True to replace it.")
        # asked to replace: the old model's files go, so nothing of it (a
        # prototype bank, a backbone) is left beside the new card
        for f in MODEL_FILES:
            (d / f).unlink(missing_ok=True)
    d.mkdir(parents=True, exist_ok=True)
    return d


#: The files a model folder of this package can hold (cleared on overwrite).
MODEL_FILES = ("card.json", "model.onnx", "model.pt", "backbone.pt",
               "prototypes.json", "prototypes.npz", "prototypes_builtin.json")


# ---------------------------------------------------------------------------
# Datasets (ARCHITECTURE §5)
# ---------------------------------------------------------------------------
class DatasetRefused(ValueError):
    """A dataset cannot be used for this run. The message says why."""


def load_manifest(dataset_dir) -> dict:
    d = Path(dataset_dir)
    p = d / MANIFEST
    if not p.exists():
        raise DatasetRefused(f"{d} has no {MANIFEST}, so it is not a built "
                             "dataset (ARCHITECTURE §5). Build it with the "
                             "dataset builder first.")
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise DatasetRefused(f"{d.name}'s {MANIFEST} could not be read: "
                             f"{e}") from None
    if not isinstance(m, dict):
        raise DatasetRefused(f"{d.name}'s {MANIFEST} is not a JSON object")
    return m


def open_dataset(dataset_dir, profile: str, kind: str | None = None,
                 rf=None, verify: str = "exists") -> dict:
    """Read and check a dataset for a run on `profile`. Returns the manifest
    (with `_dir` added). Refuses, in words, when:

    * the manifest's profile is not the run's (`ProfileMismatch`, via
      `profiles.check_match` — the sample-rate law, plan §3.3);
    * its sample rate is not the profile's rate;
    * it is the wrong kind (narrowband vs wideband);
    * a file it lists is missing, or — when an RfData is given — a file the
      write log recorded has changed since ("named, not used", plan §3.2);
    * `verify="full"`: a file's SHA-256 is not the manifest's.
    """
    d = Path(dataset_dir)
    m = load_manifest(d)
    name = str(m.get("name") or d.name)
    missing = [k for k in ("profile", "sample_rate", "kind") if k not in m]
    if missing:
        raise DatasetRefused(f"the dataset {name!r} has an incomplete "
                             f"manifest (no {', '.join(missing)}).")
    try:
        _profiles.check_match(str(m["profile"]), profile,
                              what=f"the dataset {name!r}")
    except _profiles.ProfileMismatch:
        raise _profiles.ProfileMismatch(
            f"the dataset {name!r} was built for "
            f"{_safe_describe(str(m['profile']))}; this run is for "
            f"{_safe_describe(profile)}. Profiles never mix: build the "
            "dataset for this receiver at this rate, or resample the captures "
            "deliberately (a logged step) and build from those.") from None
    want_rate = float(_profiles.parse_profile_id(profile).sample_rate)
    if abs(float(m["sample_rate"]) - want_rate) > 0.5:
        raise DatasetRefused(f"the dataset {name!r} says its sample rate is "
                             f"{float(m['sample_rate']):g} S/s, but its "
                             f"profile {m['profile']} is {want_rate:g} S/s. "
                             "The manifest contradicts itself; rebuild it.")
    if m["kind"] not in KINDS:
        raise DatasetRefused(f"the dataset {name!r} has an unknown kind "
                             f"{m['kind']!r} (narrowband or wideband).")
    if kind and m["kind"] != kind:
        raise DatasetRefused(f"the dataset {name!r} is a {m['kind']} dataset; "
                             f"this needs a {kind} one.")
    files = m.get("files") or {}
    log_entries = {}
    root = None
    if rf is not None:
        try:
            log_entries = rf.log.entries()
            root = Path(rf.root).resolve()
        except Exception:                                  # noqa: BLE001
            log_entries = {}
    for rel, digest in files.items():
        p = d / rel
        if not p.exists():
            raise DatasetRefused(f"the dataset {name!r} lists {rel}, which is "
                                 "missing. The dataset is incomplete; rebuild "
                                 "it or restore the file.")
        if root is not None and log_entries:
            try:
                key = p.resolve().relative_to(root).as_posix()
            except ValueError:
                key = ""
            e = log_entries.get(key)
            if e is not None:
                st = p.stat()
                if st.st_size != e.get("size") or st.st_mtime_ns != e.get("mtime_ns"):
                    ok, why = rf.verify(p)
                    if not ok:
                        raise DatasetRefused(why)
        if verify == "full" and digest and sha256_path(p) != digest:
            raise DatasetRefused(f"{rel} in the dataset {name!r} is not the "
                                 "file the manifest describes (its hash "
                                 "changed). It is named, not used.")
    out = dict(m)
    out["_dir"] = str(d)
    return out


def dataset_entry(dataset_dir, manifest: dict, splits=()) -> dict:
    """What a model card records about a dataset it was trained or scored on:
    the manifest's SHA-256 (which itself lists every file's hash), and the
    facts that bear on trust — generator, label sources, resampled or not."""
    d = Path(dataset_dir)
    counts = manifest.get("splits") or {}
    return {"name": str(manifest.get("name") or d.name),
            "sha256": sha256_path(d / MANIFEST),
            "kind": manifest.get("kind", ""),
            "generator": manifest.get("generator", ""),
            "splits_used": list(splits),
            "n": {s: counts.get(s) for s in splits},
            "label_sources": list(manifest.get("label_sources") or []),
            "environment": manifest.get("environment", "") or "",
            "resampled": bool(manifest.get("resampled", False)),
            "files": len(manifest.get("files") or {})}


def split_files(dataset_dir, split: str, kind: str) -> list[Path]:
    pattern = "shard_*.npz" if kind == "narrowband" else "tiles_*.npz"
    return sorted((Path(dataset_dir) / split).glob(pattern))


def npz_member_shape(path, member: str) -> tuple | None:
    """The shape of one array inside an .npz without reading it (header
    only) — so indexing a 20 GB dataset reads kilobytes."""
    try:
        with zipfile.ZipFile(path) as z:
            name = member if member.endswith(".npy") else member + ".npy"
            if name not in z.namelist():
                return None
            with z.open(name) as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, _fo, _dt = np.lib.format.read_array_header_1_0(f)
                else:
                    shape, _fo, _dt = np.lib.format.read_array_header_2_0(f)
                return tuple(int(s) for s in shape)
    except (OSError, zipfile.BadZipFile, ValueError) as e:
        raise DatasetRefused(f"{Path(path).name} could not be read as an .npz "
                             f"file: {e}") from None


class _FileCache:
    """A tiny LRU of loaded npz members: the heavy arrays of a few files."""

    def __init__(self, files, members, size: int = 2):
        self.files = list(files)
        self.members = tuple(members)
        self.size = max(1, int(size))
        self._c: OrderedDict = OrderedDict()

    def get(self, i: int) -> dict:
        if i in self._c:
            self._c.move_to_end(i)
            return self._c[i]
        with np.load(self.files[i], allow_pickle=False) as z:
            arrs = {k: z[k] for k in self.members if k in z.files}
        self._c[i] = arrs
        while len(self._c) > self.size:
            self._c.popitem(last=False)
        return arrs


class ShardSet:
    """The narrowband examples of one split, read lazily.

    Label arrays are read up front (they are small); `iq` and `scf` are read
    one shard at a time and cached (`cache_files`), so a sampler that walks
    shard by shard (`shard_order`) reads each shard once per epoch."""

    def __init__(self, dataset_dir, split: str, manifest: dict,
                 cache_files: int = 2, max_items: int | None = None):
        self.dir = Path(dataset_dir)
        self.split = split
        self.manifest = manifest
        self.files = split_files(self.dir, split, "narrowband")
        refs, cols = [], {k: [] for k in NB_LABEL_FIELDS + ("t_s",)}
        self.L = None
        self.scf_shape = None
        for fi, f in enumerate(self.files):
            with np.load(f, allow_pickle=False) as z:
                if "iq" not in z.files or "label" not in z.files:
                    raise DatasetRefused(f"{f.name} has no 'iq' or 'label' "
                                         "array (ARCHITECTURE §5).")
                n = int(z["label"].shape[0])
                for k in NB_LABEL_FIELDS:
                    if k in z.files:
                        cols[k].append(np.asarray(z[k]).reshape(-1)[:n])
                    else:
                        cols[k].append(np.full(n, np.nan, np.float32))
                cols["t_s"].append(np.asarray(z["t_s"], np.float64).reshape(-1)[:n]
                                   if "t_s" in z.files
                                   else np.full(n, np.nan, np.float64))
            shp = npz_member_shape(f, "iq")
            if shp is None or len(shp) != 2 or shp[0] != n:
                raise DatasetRefused(f"{f.name}: 'iq' must be (N, L) with one "
                                     "row per label.")
            if self.L is None:
                self.L = int(shp[1])
            elif int(shp[1]) != self.L:
                raise DatasetRefused(f"{f.name} has windows of {shp[1]} samples "
                                     f"where the rest have {self.L}; a dataset "
                                     "has one window length.")
            sshp = npz_member_shape(f, "scf")
            if sshp is not None:
                if self.scf_shape is None:
                    self.scf_shape = tuple(sshp[1:])
            refs.extend((fi, j) for j in range(n))
        if max_items is not None:
            refs = refs[: int(max_items)]
        self.refs = refs
        n = len(refs)
        self.labels = {}
        for k, parts in cols.items():
            arr = np.concatenate(parts) if parts else np.zeros(0)
            self.labels[k] = arr[:n]
        self.labels["label"] = self.labels["label"].astype(np.int64)
        self.has_scf = self.scf_shape is not None and all(
            npz_member_shape(f, "scf") is not None for f in self.files)
        self._cache = _FileCache(self.files, ("iq", "scf"), cache_files)

    def __len__(self) -> int:
        return len(self.refs)

    def iq(self, i: int) -> np.ndarray:
        fi, j = self.refs[i]
        return np.asarray(self._cache.get(fi)["iq"][j], dtype=np.complex64)

    def scf(self, i: int) -> np.ndarray | None:
        fi, j = self.refs[i]
        a = self._cache.get(fi).get("scf")
        return None if a is None else np.asarray(a[j], dtype=np.float32)

    def shard_order(self, rng: np.random.Generator | None = None) -> list[int]:
        """Indices grouped by shard (shards shuffled, items shuffled within a
        shard): random enough to train on, and each shard is read once."""
        by_file: dict[int, list[int]] = {}
        for i, (fi, _j) in enumerate(self.refs):
            by_file.setdefault(fi, []).append(i)
        keys = list(by_file)
        if rng is not None:
            rng.shuffle(keys)
        out = []
        for k in keys:
            idx = list(by_file[k])
            if rng is not None:
                rng.shuffle(idx)
            out.extend(idx)
        return out


@dataclass
class TileRef:
    file: int
    local: int
    boxes: np.ndarray                  # (n, 4) row0, bin0, row1, bin1
    family: np.ndarray                 # (n,) index into manifest families
    cls: np.ndarray                    # (n,) index into manifest classes
    snr_db: np.ndarray                 # (n,) NaN when the dataset has none
    t_s: float = float("nan")


class TileSet:
    """The wideband tiles of one split, read lazily (spec arrays are cached
    one file at a time; boxes are read up front — they are small).

    Box convention (ARCHITECTURE §5): `[tile, row0, bin0, row1, bin1,
    family]`, `tile` indexing the `spec` array of the same file, edges in
    pixel units with `row1`/`bin1` one past the last row/bin covered."""

    def __init__(self, dataset_dir, split: str, manifest: dict,
                 cache_files: int = 2, max_tiles: int | None = None):
        self.dir = Path(dataset_dir)
        self.split = split
        self.manifest = manifest
        self.files = split_files(self.dir, split, "wideband")
        self.refs: list[TileRef] = []
        self.shape = None
        for fi, f in enumerate(self.files):
            shp = npz_member_shape(f, "spec")
            if shp is None or len(shp) != 3:
                raise DatasetRefused(f"{f.name} has no (N, rows, bins) 'spec' "
                                     "array (ARCHITECTURE §5).")
            if self.shape is None:
                self.shape = (int(shp[1]), int(shp[2]))
            elif (int(shp[1]), int(shp[2])) != self.shape:
                raise DatasetRefused(f"{f.name} has tiles of {shp[1]}×{shp[2]} "
                                     f"where the rest are {self.shape[0]}×"
                                     f"{self.shape[1]}; a dataset has one tile "
                                     "geometry.")
            n = int(shp[0])
            with np.load(f, allow_pickle=False) as z:
                boxes = (np.asarray(z["boxes"], np.float32).reshape(-1, 6)
                         if "boxes" in z.files else np.zeros((0, 6), np.float32))
                bcls = (np.asarray(z["box_class"], np.int64).reshape(-1)
                        if "box_class" in z.files
                        else np.full(len(boxes), -1, np.int64))
                bsnr = (np.asarray(z["box_snr_db"], np.float32).reshape(-1)
                        if "box_snr_db" in z.files
                        else np.full(len(boxes), np.nan, np.float32))
                tts = (np.asarray(z["t_s"], np.float64).reshape(-1)
                       if "t_s" in z.files else None)
            if len(bcls) != len(boxes):
                raise DatasetRefused(f"{f.name}: box_class has {len(bcls)} "
                                     f"entries for {len(boxes)} boxes.")
            tix = boxes[:, 0].astype(np.int64) if len(boxes) else np.zeros(0, np.int64)
            if len(tix) and (tix.min() < 0 or tix.max() >= n):
                raise DatasetRefused(f"{f.name}: a box points at tile "
                                     f"{int(tix.max())} but the file has {n} "
                                     "tiles (the tile column indexes the "
                                     "file's own spec array).")
            for j in range(n):
                sel = tix == j
                self.refs.append(TileRef(
                    fi, j, boxes[sel, 1:5].copy(),
                    boxes[sel, 5].astype(np.int64), bcls[sel], bsnr[sel],
                    float(tts[j]) if tts is not None and j < len(tts)
                    else float("nan")))
        if max_tiles is not None:
            self.refs = self.refs[: int(max_tiles)]
        self._cache = _FileCache(self.files, ("spec",), cache_files)

    def __len__(self) -> int:
        return len(self.refs)

    def spec(self, i: int) -> np.ndarray:
        r = self.refs[i]
        return np.asarray(self._cache.get(r.file)["spec"][r.local],
                          dtype=np.float32)

    def file_order(self, rng: np.random.Generator | None = None) -> list[int]:
        by_file: dict[int, list[int]] = {}
        for i, r in enumerate(self.refs):
            by_file.setdefault(r.file, []).append(i)
        keys = list(by_file)
        if rng is not None:
            rng.shuffle(keys)
        out = []
        for k in keys:
            idx = list(by_file[k])
            if rng is not None:
                rng.shuffle(idx)
            out.extend(idx)
        return out


def item_times(times: np.ndarray, seconds_each: float) -> tuple[np.ndarray, str]:
    """Seconds-since-start for items of an on-site set: the dataset's own
    `t_s` when every item has one, else the storage order times
    `seconds_each` — and a sentence saying which."""
    t = np.asarray(times, np.float64)
    if t.size and np.all(np.isfinite(t)):
        return t - float(np.min(t)), "item times from the dataset's t_s"
    return (np.arange(t.size, dtype=np.float64) * float(seconds_each),
            f"the dataset has no t_s; items are taken as consecutive in "
            f"storage order, {seconds_each:g} s apart")


# ---------------------------------------------------------------------------
# What a model sees: the tile normalisation, the IQ window, the SCF scaling
# (numpy, so ONNX inference in the core environment feeds exactly what
# training fed)
# ---------------------------------------------------------------------------
@dataclass
class TileNorm:
    """Clip dB-above-floor to [lo, hi], then (x − mean) / std. Recorded in
    the card; for the proposer it is applied INSIDE the ONNX graph."""
    clip_lo_db: float = -10.0
    clip_hi_db: float = 50.0
    mean_db: float = 0.0
    std_db: float = 10.0
    applied_by: str = "graph"          # 'graph' (in the ONNX) | 'host'

    def clip(self, x):
        return np.clip(np.asarray(x, np.float32), self.clip_lo_db,
                       self.clip_hi_db)

    def apply(self, x):
        return (self.clip(x) - np.float32(self.mean_db)) / np.float32(self.std_db)

    def to_json(self) -> dict:
        return {"input": "dB above the measured noise floor",
                "clip_db": [float(self.clip_lo_db), float(self.clip_hi_db)],
                "mean_db": float(self.mean_db), "std_db": float(self.std_db),
                "formula": "(clip(x, clip_db) - mean_db) / std_db",
                "applied_by": self.applied_by}

    @classmethod
    def from_json(cls, d: dict) -> "TileNorm":
        lo, hi = (d.get("clip_db") or [-10.0, 50.0])[:2]
        return cls(float(lo), float(hi), float(d.get("mean_db", 0.0)),
                   float(d.get("std_db", 10.0)),
                   str(d.get("applied_by", "graph")))

    @classmethod
    def fit(cls, tiles, n: int = 64, seed: int = 0, clip=(-10.0, 50.0),
            applied_by: str = "graph") -> "TileNorm":
        """Mean and standard deviation of the clipped dB values over up to
        `n` tiles of a TileSet."""
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(tiles))[: max(1, int(n))] if len(tiles) else []
        s = s2 = cnt = 0.0
        for i in sorted(int(j) for j in idx):
            x = np.clip(tiles.spec(i), clip[0], clip[1]).astype(np.float64)
            s += x.sum()
            s2 += (x * x).sum()
            cnt += x.size
        if cnt == 0:
            return cls(clip[0], clip[1], 0.0, 10.0, applied_by)
        mean = s / cnt
        std = max(float(np.sqrt(max(s2 / cnt - mean * mean, 0.0))), 1e-3)
        return cls(float(clip[0]), float(clip[1]), float(mean), std, applied_by)


def iq_window(x, window: int | None = None, offset: int | None = None,
              rms: bool = True) -> np.ndarray:
    """complex (L,) -> float32 [2, window]: a window of the cut (centred when
    no offset is given), scaled to unit RMS — the classifier is asked *what*,
    not *how loud*."""
    x = np.asarray(x, np.complex64).reshape(-1)
    L = x.size
    w = int(window or L)
    if w > L:
        raise ValueError(f"a window of {w} samples does not fit a cut of {L}")
    o = (L - w) // 2 if offset is None else int(offset)
    seg = x[o:o + w]
    if rms:
        p = float(np.sqrt(np.mean(np.abs(seg) ** 2)))
        if p > 0:
            seg = seg / p
    return np.stack([seg.real, seg.imag]).astype(np.float32)


#: SCF scalings, by the names `detect.onnx_models.Classifier1D` applies at
#: inference (`card.input["scf_norm"]`): the training side uses the same
#: words and the same arithmetic, so the two can never drift apart.
SCF_NORMS = ("max", "none")


def scf_normalize(img, mode: str = "max") -> np.ndarray:
    """SCF [H, W] -> float32 [1, H, W]. 'max': divided by its largest
    magnitude (an SCF surface is a magnitude, so this maps it to [0, 1]
    whatever scale it was cached in); 'none': as cached."""
    if mode not in SCF_NORMS:
        raise ValueError(f"scf_norm is one of {', '.join(SCF_NORMS)}, not {mode!r}")
    a = np.asarray(img, np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "max":
        m = float(np.max(np.abs(a))) if a.size else 0.0
        a = a / max(m, 1e-12)
    return a[None]


# ---------------------------------------------------------------------------
# Ground truth for the proposer — one definition for training and scoring
# ---------------------------------------------------------------------------
@dataclass
class BoxPolicy:
    """How a dataset's boxes become the proposer's targets.

    families     the proposer's family list (labels index into it)
    min_box_px   boxes narrower/shorter than this are widened to it (see the
                 module docstring); 0 disables
    negatives    'background' — boxes of an explicit-negative class (noise,
                 spur, DC spike; classes.CLASSES `negative=True`) are dropped,
                 so the AI proposer learns NOT to box them (energy still
                 does); 'unknown' — they are kept as family 'unknown'
    """
    families: tuple = tuple(_classes.FAMILIES)
    min_box_px: float = 10.0
    negatives: str = "background"

    def to_json(self) -> dict:
        return {"families": list(self.families), "min_box_px": self.min_box_px,
                "negatives": self.negatives}

    @classmethod
    def from_json(cls, d: dict) -> "BoxPolicy":
        return cls(tuple(d.get("families") or _classes.FAMILIES),
                   float(d.get("min_box_px", 10.0)),
                   str(d.get("negatives", "background")))


def family_map(manifest: dict, families) -> np.ndarray:
    """manifest family index -> index into `families` (refused in words for a
    family the proposer does not have)."""
    names = list(manifest.get("families") or [])
    if not names:
        names = list(_classes.FAMILIES)
    fams = list(families)
    out = np.zeros(len(names), np.int64)
    for i, n in enumerate(names):
        if n not in fams:
            raise DatasetRefused(f"the dataset uses the family {n!r}, which is "
                                 f"not one of the proposer's families "
                                 f"({', '.join(fams)}).")
        out[i] = fams.index(n)
    return out


def negative_classes(manifest: dict) -> np.ndarray:
    """Boolean per manifest class: True for an explicit negative."""
    names = list(manifest.get("classes") or [])
    out = np.zeros(len(names), bool)
    for i, n in enumerate(names):
        c = _classes.get(n)
        out[i] = bool(c is not None and c.negative)
    return out


def prepare_boxes(boxes_rc, fam_idx, cls_idx, tile_shape, policy: BoxPolicy,
                  fam_lut: np.ndarray, neg_lut: np.ndarray):
    """Dataset boxes -> (boxes (n, 4) row0, bin0, row1, bin1 float32, labels
    (n,) int64 into policy.families, keep (M,) bool over the input boxes).

    Clipped to the tile, negatives handled per the policy, degenerate boxes
    dropped, thin boxes widened to `min_box_px` (kept inside the tile)."""
    b = np.asarray(boxes_rc, np.float32).reshape(-1, 4).copy()
    fam = np.asarray(fam_idx, np.int64).reshape(-1)
    cls = np.asarray(cls_idx, np.int64).reshape(-1)
    rows, bins = int(tile_shape[0]), int(tile_shape[1])
    keep = np.ones(len(b), bool)
    if len(b) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.int64), keep
    if np.any((fam < 0) | (fam >= len(fam_lut))):
        raise DatasetRefused("a box has a family index outside the manifest's "
                             "family list.")
    labels = fam_lut[fam]
    is_neg = np.zeros(len(b), bool)
    ok_cls = (cls >= 0) & (cls < len(neg_lut))
    is_neg[ok_cls] = neg_lut[cls[ok_cls]]
    if policy.negatives == "background":
        keep &= ~is_neg
    elif policy.negatives == "unknown":
        unk = list(policy.families).index("unknown") \
            if "unknown" in policy.families else None
        if unk is None:
            keep &= ~is_neg
        else:
            labels = labels.copy()
            labels[is_neg] = unk
    else:
        raise ValueError(f"negatives must be 'background' or 'unknown', not "
                         f"{policy.negatives!r}")
    b[:, 0] = np.clip(b[:, 0], 0, rows)
    b[:, 2] = np.clip(b[:, 2], 0, rows)
    b[:, 1] = np.clip(b[:, 1], 0, bins)
    b[:, 3] = np.clip(b[:, 3], 0, bins)
    keep &= (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
    b = widen_boxes(b, float(policy.min_box_px or 0.0), rows, bins)
    return b[keep], labels[keep].astype(np.int64), keep


def widen_boxes(boxes_rc, min_px: float, rows: int, bins: int) -> np.ndarray:
    """Boxes (row0, bin0, row1, bin1) narrower or shorter than `min_px`
    widened symmetrically to it, kept inside the tile. One rule for the
    ground truth, the learned proposer and the energy baseline alike."""
    b = np.asarray(boxes_rc, np.float32).reshape(-1, 4).copy()
    m = float(min_px or 0.0)
    if m > 0 and len(b):
        for lo, hi, limit in ((0, 2, rows), (1, 3, bins)):
            size = b[:, hi] - b[:, lo]
            short = size < m
            if np.any(short):
                want = min(m, float(limit))
                c = 0.5 * (b[short, lo] + b[short, hi])
                new_lo = np.clip(c - 0.5 * want, 0.0, limit - want)
                b[short, lo] = new_lo
                b[short, hi] = new_lo + want
    return b


def box_snr_db(spec_db, box_rc) -> float:
    """In-box SNR measured from a tile: mean linear power above a unit floor,
    minus the floor, in dB. Approximate (it assumes the floor is the noise's
    mean power); used only when the dataset carries no per-box SNR."""
    a = np.asarray(spec_db, np.float64)
    r0 = int(max(0, math.floor(box_rc[0])))
    c0 = int(max(0, math.floor(box_rc[1])))
    r1 = int(min(a.shape[0], math.ceil(box_rc[2])))
    c1 = int(min(a.shape[1], math.ceil(box_rc[3])))
    if r1 <= r0 or c1 <= c0:
        return float("nan")
    p = float(np.mean(10.0 ** (a[r0:r1, c0:c1] / 10.0)))
    return float(10.0 * np.log10(max(p - 1.0, 1e-3)))


def proposer_truth(tiles: "TileSet", manifest: dict, policy: BoxPolicy,
                   measure_snr: bool = True):
    """Scoring truth for every tile of a TileSet under a policy: (gts
    [{"boxes", "labels"}], snrs [array per tile], names [class names per
    tile]). SNR is the dataset's per-box SNR when it has one, else measured
    from the tile (`box_snr_db`)."""
    fam_lut = family_map(manifest, policy.families)
    neg = negative_classes(manifest)
    cls_names = list(manifest.get("classes") or [])
    gts, snrs, names = [], [], []
    for i, r in enumerate(tiles.refs):
        b, lab, keep = prepare_boxes(r.boxes, r.family, r.cls, tiles.shape,
                                     policy, fam_lut, neg)
        gts.append({"boxes": b, "labels": lab})
        sn = np.asarray(r.snr_db, np.float64)[keep].copy()
        if measure_snr and sn.size and not np.all(np.isfinite(sn)):
            spec = tiles.spec(i)
            orig = r.boxes[keep]
            for k in np.where(~np.isfinite(sn))[0]:
                sn[k] = box_snr_db(spec, orig[k])
        snrs.append(sn)
        cl = r.cls[keep]
        names.append([cls_names[int(c)] if 0 <= int(c) < len(cls_names)
                      else str(policy.families[int(l)])
                      for c, l in zip(cl, lab)])
    return gts, snrs, names


# ---------------------------------------------------------------------------
# Metrics — detection
# ---------------------------------------------------------------------------
def box_iou(a, b) -> np.ndarray:
    """IoU of every box in `a` (Na, 4) with every box in `b` (Nb, 4); both in
    the same corner order (r0, c0, r1, c1)."""
    a = np.asarray(a, np.float64).reshape(-1, 4)
    b = np.asarray(b, np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    r0 = np.maximum(a[:, None, 0], b[None, :, 0])
    c0 = np.maximum(a[:, None, 1], b[None, :, 1])
    r1 = np.minimum(a[:, None, 2], b[None, :, 2])
    c1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(r1 - r0, 0, None) * np.clip(c1 - c0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = aa[:, None] + ab[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return out


def average_precision(tp_sorted, n_gt: int) -> float:
    """All-point interpolated AP (VOC 2010+) from TP flags in descending
    score order. NaN when there is no ground truth to find."""
    if n_gt <= 0:
        return float("nan")
    tp = np.asarray(tp_sorted, np.float64)
    if tp.size == 0:
        return 0.0
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1.0 - tp)
    rec = ctp / float(n_gt)
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def match_image(pb, ps, gb, iou_thr):
    """Greedy match of one image's predictions (sorted by score) to its GT.
    Returns (tp flags in score order, matched gt index or -1, order)."""
    order = np.argsort(-np.asarray(ps, np.float64), kind="stable")
    tp = np.zeros(len(order), bool)
    which = np.full(len(order), -1, np.int64)
    if len(order) == 0 or len(gb) == 0:
        return tp, which, order
    ious = box_iou(np.asarray(pb)[order], gb)
    taken = np.zeros(len(gb), bool)
    for k in range(len(order)):
        j = int(np.argmax(ious[k]))
        if ious[k, j] >= iou_thr and not taken[j]:
            taken[j] = True
            tp[k] = True
            which[k] = j
    return tp, which, order


def detection_ap(preds: list[dict], gts: list[dict], num_classes: int,
                 iou_thr: float = 0.5, class_agnostic: bool = False) -> dict:
    """AP per class and their mean at one IoU threshold.

    preds[i] = {"boxes": (K, 4), "scores": (K,), "labels": (K,)};
    gts[i] = {"boxes": (M, 4), "labels": (M,)}. Classes with no ground truth
    get AP NaN and are left out of the mean (their predictions are all false
    alarms, which `false_positives` counts)."""
    ncls = 1 if class_agnostic else int(num_classes)
    per = {}
    fps = {}
    for c in range(ncls):
        scores, tps = [], []
        n_gt = 0
        for p, g in zip(preds, gts):
            gl = np.asarray(g["labels"]).reshape(-1)
            pl = np.asarray(p["labels"]).reshape(-1)
            gsel = np.ones(len(gl), bool) if class_agnostic else gl == c
            psel = np.ones(len(pl), bool) if class_agnostic else pl == c
            gb = np.asarray(g["boxes"]).reshape(-1, 4)[gsel]
            pb = np.asarray(p["boxes"]).reshape(-1, 4)[psel]
            ps = np.asarray(p["scores"]).reshape(-1)[psel]
            n_gt += len(gb)
            tp, _w, order = match_image(pb, ps, gb, iou_thr)
            scores.append(ps[order])
            tps.append(tp)
        s = np.concatenate(scores) if scores else np.zeros(0)
        t = np.concatenate(tps) if tps else np.zeros(0, bool)
        o = np.argsort(-s, kind="stable")
        per[c] = average_precision(t[o], n_gt)
        fps[c] = int(np.sum(~t))
    vals = [v for v in per.values() if np.isfinite(v)]
    return {"map": float(np.mean(vals)) if vals else float("nan"),
            "ap": per, "false_positives": fps, "iou": float(iou_thr)}


def detection_map(preds, gts, num_classes: int, class_agnostic: bool = False
                  ) -> dict:
    """mAP@0.5 and the COCO-style mean over IoU 0.50:0.05:0.95."""
    at50 = detection_ap(preds, gts, num_classes, 0.5, class_agnostic)
    coco = []
    for thr in np.arange(0.5, 0.951, 0.05):
        r = detection_ap(preds, gts, num_classes, float(thr), class_agnostic)
        coco.append(r["map"])
    coco = [v for v in coco if np.isfinite(v)]
    return {"map50": at50["map"], "map50_95": float(np.mean(coco)) if coco
            else float("nan"), "ap50": at50["ap"],
            "false_positives50": at50["false_positives"]}


def operating_point(preds, gts, score_thr: float, iou_thr: float = 0.5
                    ) -> dict:
    """Class-agnostic precision, recall and F1 of the boxes scoring at least
    `score_thr` — *where*, not *what*: family confusion is scored by AP."""
    tp = fp = fn = 0
    for p, g in zip(preds, gts):
        s = np.asarray(p["scores"]).reshape(-1)
        sel = s >= score_thr
        pb = np.asarray(p["boxes"]).reshape(-1, 4)[sel]
        gb = np.asarray(g["boxes"]).reshape(-1, 4)
        t, _w, _o = match_image(pb, s[sel], gb, iou_thr)
        tp += int(t.sum())
        fp += int((~t).sum())
        fn += int(len(gb) - t.sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else float("nan")
    return {"threshold": float(score_thr), "precision": prec, "recall": rec,
            "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def best_threshold(preds, gts, iou_thr: float = 0.5,
                   candidates=None) -> dict:
    """The score threshold with the best F1 on this set (ties -> the higher
    threshold, i.e. fewer false alarms)."""
    if candidates is None:
        allscores = np.concatenate([np.asarray(p["scores"]).reshape(-1)
                                    for p in preds]) if preds else np.zeros(0)
        qs = np.unique(np.round(allscores, 4))
        candidates = qs if qs.size <= 200 else np.quantile(allscores,
                                                           np.linspace(0, 1, 201))
    best = None
    for thr in sorted(set(float(c) for c in candidates)):
        r = operating_point(preds, gts, thr, iou_thr)
        f = r["f1"] if np.isfinite(r["f1"]) else -1.0
        if best is None or f >= (best["f1"] if np.isfinite(best["f1"]) else -1.0):
            best = r
    return best or operating_point(preds, gts, 0.5, iou_thr)


def detection_vs_snr(preds, gts, gt_snr, gt_names, score_thr: float,
                     iou_thr: float = 0.5, width_db: float = 5.0) -> dict:
    """Probability of detection per class per SNR bin (the curve that is the
    product, DETECTION_DESIGN §11). A ground-truth box counts as detected
    when a box scoring at least `score_thr` overlaps it at `iou_thr` —
    class-agnostic, because the question is *was it found*."""
    found, snrs, names = [], [], []
    for p, g, sn, nm in zip(preds, gts, gt_snr, gt_names):
        s = np.asarray(p["scores"]).reshape(-1)
        sel = s >= score_thr
        pb = np.asarray(p["boxes"]).reshape(-1, 4)[sel]
        gb = np.asarray(g["boxes"]).reshape(-1, 4)
        ious = box_iou(gb, pb)
        hit = ious.max(axis=1) >= iou_thr if ious.size else np.zeros(len(gb), bool)
        found.extend(bool(h) for h in hit)
        snrs.extend(float(x) for x in np.asarray(sn).reshape(-1))
        names.extend(str(x) for x in nm)
    found = np.asarray(found, bool)
    snrs = np.asarray(snrs, np.float64)
    out = {}
    for name in sorted(set(names)):
        sel = np.asarray([n == name for n in names], bool)
        out[name] = curve_over_snr(snrs[sel], found[sel].astype(np.float64),
                                   width_db)
    out["_all"] = curve_over_snr(snrs, found.astype(np.float64), width_db)
    return out


def curve_over_snr(snr_db, values, width_db: float = 5.0) -> list[dict]:
    """Mean of `values` in SNR bins `width_db` wide. Items without an SNR are
    reported in one bin with snr_db None."""
    s = np.asarray(snr_db, np.float64).reshape(-1)
    v = np.asarray(values, np.float64).reshape(-1)
    out = []
    ok = np.isfinite(s)
    if np.any(ok):
        lo = math.floor(float(np.min(s[ok])) / width_db) * width_db
        hi = float(np.max(s[ok]))
        edge = lo
        while edge <= hi:
            sel = ok & (s >= edge) & (s < edge + width_db)
            if np.any(sel):
                out.append({"snr_db": edge + 0.5 * width_db, "lo_db": edge,
                            "hi_db": edge + width_db, "n": int(sel.sum()),
                            "value": float(np.mean(v[sel]))})
            edge += width_db
    if np.any(~ok):
        out.append({"snr_db": None, "lo_db": None, "hi_db": None,
                    "n": int((~ok).sum()), "value": float(np.mean(v[~ok]))})
    return out


# ---------------------------------------------------------------------------
# Metrics — classification
# ---------------------------------------------------------------------------
def accuracy(y_true, y_pred) -> float:
    t = np.asarray(y_true).reshape(-1)
    p = np.asarray(y_pred).reshape(-1)
    return float(np.mean(t == p)) if t.size else float("nan")


def confusion(y_true, y_pred, n: int) -> np.ndarray:
    m = np.zeros((n, n), np.int64)
    for t, p in zip(np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1)):
        if 0 <= t < n and 0 <= p < n:
            m[int(t), int(p)] += 1
    return m


def macro_f1(y_true, y_pred, n: int) -> float:
    """Mean F1 over the classes present in y_true (a class nobody labelled
    cannot be scored, and is left out rather than counted as zero)."""
    m = confusion(y_true, y_pred, n)
    f1s = []
    for c in range(n):
        if m[c].sum() == 0:
            continue
        tp = m[c, c]
        fp = m[:, c].sum() - tp
        fn = m[c].sum() - tp
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


def classification_scores(pred, y, classes, snr_db, snr_bin_db: float = 5.0
                          ) -> dict:
    """Accuracy, macro-F1, accuracy vs SNR, per-class accuracy, confusion."""
    pred = np.asarray(pred)
    y = np.asarray(y)
    corr = (pred == y).astype(np.float64)
    return {"accuracy": accuracy(y, pred),
            "macro_f1": macro_f1(y, pred, len(classes)),
            "accuracy_vs_snr": curve_over_snr(snr_db, corr, snr_bin_db),
            "per_class_accuracy": {c: finite_or_none(np.mean(pred[y == i] == i))
                                   for i, c in enumerate(classes) if np.any(y == i)},
            "confusion": {"classes": list(classes),
                          "matrix": confusion(y, pred, len(classes)).tolist()}}


def cycle_error(cycle_hz, symbol_rate_hz, carrier_offset_hz) -> dict:
    """Median errors of the regressed cycle parameters, in Hz."""
    cy = np.asarray(cycle_hz, np.float64).reshape(-1, 2)
    sr_t = np.asarray(symbol_rate_hz, np.float64)
    co_t = np.asarray(carrier_offset_hz, np.float64)
    ok_sr = np.isfinite(sr_t) & (sr_t > 0)
    ok_co = np.isfinite(co_t)
    return {"symbol_rate_median_abs_hz": finite_or_none(
                np.median(np.abs(cy[ok_sr, 0] - sr_t[ok_sr]))) if ok_sr.any() else None,
            "symbol_rate_median_rel": finite_or_none(
                np.median(np.abs(cy[ok_sr, 0] - sr_t[ok_sr]) / sr_t[ok_sr]))
            if ok_sr.any() else None,
            "carrier_offset_median_abs_hz": finite_or_none(
                np.median(np.abs(cy[ok_co, 1] - co_t[ok_co]))) if ok_co.any() else None,
            "note": "the classical cyclic-profile measurement (dsp.measure) is "
                    "the comparator; where the two disagree the classical one "
                    "is shown (DETECTION_DESIGN §4)"}


def softmax(logits, axis: int = -1) -> np.ndarray:
    z = np.asarray(logits, np.float64)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def finite_or_none(x):
    """JSON-safe number: NaN/inf become None, numpy scalars plain floats."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


@dataclass
class Timer:
    """`with Timer() as t: …` then `t.seconds`."""
    seconds: float = 0.0
    _t0: float = field(default=0.0, repr=False)

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0
        return False
