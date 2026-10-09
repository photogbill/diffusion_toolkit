# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""PyTorch Datasets over built datasets (ARCHITECTURE §5; DETECTION_DESIGN
§2, §4, §6).

Two kinds, one per learned part:

* `NarrowbandDataset` — for the classifier (DETECTION_DESIGN §4): each item
  is a cut at its canonical rate, `iq` as a [2, L] float32 (I and Q rows,
  scaled to unit RMS: the classifier is asked *what*, not *how loud*), the
  cached SCF as [1, H, W] (divided by its maximum — `scf_norm: "max"`, the
  scaling `detect.onnx_models.Classifier1D` applies at inference; the SCF
  is the representation that holds up at low SNR, §4.1), the class, and the
  regressed cycle targets — symbol rate and carrier offset divided by
  `cycle_scale` — with a mask, because an FM voice signal has no symbol
  rate to learn.
* `WidebandTiles` — for the 2D proposer (§3): each item is a tile, dB above
  the measured floor, as [1, rows, bins], with its boxes in pixel
  coordinates and family labels. The boxes pass through
  `common.prepare_boxes`, the one definition of ground truth that training
  and scoring share.

NORMALISATION IS WRITTEN DOWN, NOT IMPLIED (§2: "a model never meets a
spectrogram it was not trained on the geometry of" — nor the scaling).
`TileNorm` is what the card records: a clip of dB above floor to
[lo, hi], then (x − mean) / std. For the proposer the clip and the
mean/std are INSIDE the exported ONNX graph (`card.input.graph_normalize`,
`applied_by: "graph"`; `card.input.normalize` is "none" for the host), so
the host feeds raw dB above floor and cannot get the scaling wrong; the
dataset therefore applies only the clip during training and FCOS's own
transform applies mean/std, exactly as the graph does at inference.

WORKERS. Items draw random crops, phases and flips from the dataset's own
generator; `loader()` reseeds it in each DataLoader worker, so parallel
workers do not hand the network identical "random" views.

The SCF must be cached by the dataset builder (§4.1: "the dataset builder
computes and caches the SCF of every cut so training does not recompute
it"); a dataset without it can train the IQ branch alone (`use_scf=False`)
and the card says so.
"""

from __future__ import annotations

import numpy as np

from atk_diffusion.learn import common as K

torch = K.require_torch()            # a plain sentence in the core environment
from torch.utils.data import Dataset, Sampler  # noqa: E402


# The numpy pieces live in common.py so ONNX inference without PyTorch feeds
# exactly what training fed; they are re-exported here for convenience.
TileNorm = K.TileNorm
iq_tensor = K.iq_window
scf_tensor = K.scf_normalize


class NarrowbandDataset(Dataset):
    """Classifier examples of one split.

    classes     the model's class names in order; examples of a class not in
                the list get label -1 (they are the *unknown* examples the
                open-set measurement uses). Default: the manifest's classes.
    window      samples per example fed to the network (default: the
                dataset's own L); training takes a random window, scoring
                the centred one.
    use_scf     feed the cached SCF; False feeds a [1, 1, 1] zero.
    cycle_scale (symbol-rate scale, offset scale) in Hz; default the
                canonical rate for both.
    known_only  keep only examples of `classes` (training never sees the
                held-out classes the unknown-rejection measurement needs).
    """

    def __init__(self, dataset_dir, split: str, profile: str, *,
                 manifest: dict | None = None, classes=None,
                 window: int | None = None, train: bool = False,
                 use_scf: bool = True, cycle_scale=None,
                 max_items: int | None = None, seed: int = 0, rf=None,
                 cache_files: int = 4, known_only: bool = False,
                 scf_norm: str = "max"):
        self.manifest = manifest or K.open_dataset(dataset_dir, profile,
                                                   "narrowband", rf=rf)
        self.shards = K.ShardSet(dataset_dir, split, self.manifest,
                                 cache_files=cache_files, max_items=max_items)
        names = list(self.manifest.get("classes") or [])
        self.classes = list(classes) if classes is not None else names
        lut = np.full(max(len(names), 1), -1, np.int64)
        for i, n in enumerate(names):
            if n in self.classes:
                lut[i] = self.classes.index(n)
        self._lut = lut
        self.window = int(window or self.shards.L or 0)
        if self.shards.L is not None and self.window > self.shards.L:
            raise ValueError(f"a window of {self.window} samples is longer "
                             f"than the dataset's cuts ({self.shards.L}).")
        self.train = bool(train)
        self.use_scf = bool(use_scf)
        if scf_norm not in K.SCF_NORMS:
            raise ValueError(f"scf_norm is one of {', '.join(K.SCF_NORMS)}")
        self.scf_norm = scf_norm
        if self.use_scf and len(self.shards) and not self.shards.has_scf:
            raise K.DatasetRefused(
                "this dataset has no cached SCF ('scf' in its shards), so the "
                "classifier's SCF branch has nothing to learn from. Rebuild "
                "the dataset with SCF caching (DETECTION_DESIGN §4.1), or "
                "train the IQ branch alone with use_scf=False.")
        rate = float((self.manifest.get("canonical") or {}).get("rate")
                     or self.manifest.get("sample_rate"))
        self.cycle_scale = tuple(float(v) for v in (cycle_scale or (rate, rate)))
        self.rng = np.random.default_rng(seed)
        raw = self.shards.labels["label"]
        mapped = np.full(len(raw), -1, np.int64)
        ok = (raw >= 0) & (raw < len(self._lut))
        mapped[ok] = self._lut[raw[ok]]
        self._mapped = mapped
        self.items = (np.nonzero(mapped >= 0)[0] if known_only
                      else np.arange(len(raw)))

    def __len__(self) -> int:
        return int(len(self.items))

    def label_of(self, pos: int) -> int:
        return int(self._mapped[self.items[pos]])

    def model_labels(self) -> np.ndarray:
        """The model's label of every item (-1: a class the model lacks)."""
        return self._mapped[self.items].copy()

    def field(self, name: str) -> np.ndarray:
        """A per-item label array (snr_db, symbol_rate_hz, …) in item order."""
        return np.asarray(self.shards.labels[name])[self.items]

    def order(self, rng=None) -> list[int]:
        """Item positions grouped by shard (see common.ShardSet.shard_order)."""
        pos = {int(raw): p for p, raw in enumerate(self.items)}
        return [pos[i] for i in self.shards.shard_order(rng) if i in pos]

    def __getitem__(self, pos: int) -> dict:
        i = int(self.items[pos])
        x = self.shards.iq(i)
        L = x.size
        off = None
        if self.train and L > self.window:
            off = int(self.rng.integers(0, L - self.window + 1))
        if self.train:
            x = x * np.complex64(np.exp(1j * self.rng.uniform(0, 2 * np.pi)))
        iq = iq_tensor(x, self.window, off)
        if self.use_scf:
            scf = scf_tensor(self.shards.scf(i), self.scf_norm)
        else:
            scf = np.zeros((1, 1, 1), np.float32)
        lab = self.shards.labels
        sr = float(lab["symbol_rate_hz"][i])
        co = float(lab["carrier_offset_hz"][i])
        cyc = np.array([sr / self.cycle_scale[0] if np.isfinite(sr) else 0.0,
                        co / self.cycle_scale[1] if np.isfinite(co) else 0.0],
                       np.float32)
        mask = np.array([1.0 if (np.isfinite(sr) and sr > 0) else 0.0,
                         1.0 if np.isfinite(co) else 0.0], np.float32)
        return {"iq": torch.from_numpy(iq), "scf": torch.from_numpy(scf),
                "label": torch.tensor(int(self._mapped[i]), dtype=torch.int64),
                "cycle": torch.from_numpy(cyc),
                "cycle_mask": torch.from_numpy(mask),
                "snr_db": torch.tensor(float(lab["snr_db"][i]),
                                       dtype=torch.float32),
                "index": torch.tensor(pos, dtype=torch.int64)}


def collate_narrowband(batch: list[dict]) -> dict:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ---------------------------------------------------------------------------
# Wideband tiles
# ---------------------------------------------------------------------------
class WidebandTiles(Dataset):
    """Proposer examples of one split: (image [1, rows, bins] float32,
    target) where target = {"boxes": [n, 4] in torchvision order (bin0, row0,
    bin1, row1), "boxes_rc": [n, 4] (row0, bin0, row1, bin1), "labels": [n]
    int64 family index into `policy.families`, "index"}.

    normalize   'clip' — clip only (the proposer: its graph applies mean/std);
                'full' — clip and mean/std (self-supervised pretraining).
    train       random flips in time and frequency (a spectrogram tile is a
                plausible tile reversed in either axis; families are coarse
                enough that spectral inversion does not change them).
    """

    def __init__(self, dataset_dir, split: str, profile: str, *,
                 manifest: dict | None = None, policy: K.BoxPolicy | None = None,
                 norm: TileNorm | None = None, normalize: str = "clip",
                 train: bool = False, max_tiles: int | None = None,
                 seed: int = 0, rf=None, cache_files: int = 2):
        self.manifest = manifest or K.open_dataset(dataset_dir, profile,
                                                   "wideband", rf=rf)
        self.tiles = K.TileSet(dataset_dir, split, self.manifest,
                               cache_files=cache_files, max_tiles=max_tiles)
        self.policy = policy or K.BoxPolicy()
        self.norm = norm or TileNorm()
        if normalize not in ("clip", "full"):
            raise ValueError("normalize is 'clip' or 'full'")
        self.normalize = normalize
        self.train = bool(train)
        self.rng = np.random.default_rng(seed)
        self._fam = K.family_map(self.manifest, self.policy.families)
        self._neg = K.negative_classes(self.manifest)

    @property
    def shape(self):
        return self.tiles.shape

    def __len__(self) -> int:
        return len(self.tiles)

    def target_rc(self, i: int):
        """(boxes_rc (n, 4), labels (n,)) — scoring truth for tile i."""
        r = self.tiles.refs[i]
        b, lab, _keep = K.prepare_boxes(r.boxes, r.family, r.cls,
                                        self.tiles.shape, self.policy,
                                        self._fam, self._neg)
        return b, lab

    def __getitem__(self, i: int):
        x = self.tiles.spec(i)
        x = self.norm.apply(x) if self.normalize == "full" else self.norm.clip(x)
        b, lab = self.target_rc(i)
        rows, bins = x.shape
        if self.train:
            if self.rng.random() < 0.5:                 # time reversal
                x = x[::-1]
                b = b.copy()
                b[:, [0, 2]] = rows - b[:, [2, 0]]
            if self.rng.random() < 0.5:                 # spectral inversion
                x = x[:, ::-1]
                b = b.copy()
                b[:, [1, 3]] = bins - b[:, [3, 1]]
        img = torch.from_numpy(np.ascontiguousarray(x, np.float32))[None]
        rc = torch.from_numpy(np.ascontiguousarray(b, np.float32)).reshape(-1, 4)
        xyxy = rc[:, [1, 0, 3, 2]].contiguous()
        return img, {"boxes": xyxy, "boxes_rc": rc,
                     "labels": torch.from_numpy(np.asarray(lab, np.int64)),
                     "index": torch.tensor(i, dtype=torch.int64)}


def collate_tiles(batch):
    imgs = [b[0] for b in batch]
    tgts = [b[1] for b in batch]
    return imgs, tgts


class GroupedSampler(Sampler):
    """Walks a dataset file by file (files shuffled, items shuffled within a
    file) so each shard or tile file is read once per epoch."""

    def __init__(self, dataset, seed: int = 0, shuffle: bool = True):
        self.dataset = dataset
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch) if self.shuffle else None
        self.epoch += 1
        if hasattr(self.dataset, "order"):
            order = self.dataset.order(rng)
        elif isinstance(self.dataset, WidebandTiles):
            order = self.dataset.tiles.file_order(rng)
        else:
            order = list(range(len(self.dataset)))
            if rng is not None:
                rng.shuffle(order)
        return iter(order)

    def __len__(self) -> int:
        return len(self.dataset)


def loader(dataset, batch_size: int, *, train: bool, seed: int = 0,
           num_workers: int = 0, collate=None, drop_last: bool = False):
    """A DataLoader with the grouped sampler (train) or storage order."""
    base = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) \
        else dataset
    coll = collate or (collate_tiles if isinstance(base, WidebandTiles)
                       else collate_narrowband)
    sampler = GroupedSampler(dataset, seed=seed, shuffle=train)
    return torch.utils.data.DataLoader(dataset, batch_size=int(batch_size),
                                       sampler=sampler, collate_fn=coll,
                                       num_workers=int(num_workers),
                                       drop_last=bool(drop_last),
                                       worker_init_fn=_reseed_worker
                                       if int(num_workers) > 0 else None)


def _reseed_worker(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    ds = info.dataset if info is not None else None
    if isinstance(ds, torch.utils.data.Subset):
        ds = ds.dataset
    if ds is not None and hasattr(ds, "rng"):
        ds.rng = np.random.default_rng((info.seed + worker_id) % (2 ** 32))
