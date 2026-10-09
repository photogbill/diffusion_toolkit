# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Self-supervised pretraining, per receiver profile (DETECTION_DESIGN §6.1;
ARCHITECTURE §4.4).

§6.1: *"Unlabeled IQ is the one thing there is an unlimited supply of:
record. Masked-spectrogram modeling for the 2D backbone (mask patches of
the tile, reconstruct), and denoising / contrastive pretraining for the 1D
backbone. This teaches the networks what this receiver in this environment
looks like before a single label exists, and it is the second-largest
domain-gap reducer after noise-floor normalization."*

TWO OBJECTIVES, ONE PER BACKBONE
* `pretrain_2d` — masked spectrogram modelling for the proposer's backbone
  (`proposer2d.ThinResNet2d`). Square patches of each normalised tile are
  zeroed (zero is the mean after normalisation) and a light decoder on the
  stride-8 and stride-32 features reconstructs them; the loss is the mean
  squared error on the masked pixels only. Tiles come from a wideband
  dataset (its labels ignored) or from raw captures through the front end
  `dsp.stft.tiles` — the same tiles ATK's pipeline makes, so the backbone
  learns the pictures it will be shown.
* `pretrain_1d` — contrastive (SimCLR / NT-Xent) pretraining of the
  classifier's IQ branch (`classifier1d.ResNet1d`) with RF augmentations
  that change nothing about WHAT a signal is: a phase rotation, a small
  carrier offset, a time shift, and added noise at a random SNR. Two views
  of a cut must land together, apart from other cuts. Cuts come from a
  narrowband dataset (labels ignored) or are taken from raw captures at
  random times and frequencies, shifted and INTEGER-decimated to the
  profile's canonical rate for the chosen class (decision D2; the core
  `dsp.resample` path) — unlabeled cuts, at the rate the classifier sees.

MEASURED, NOT ASSERTED. The 2D card reports the masked-pixel error beside
two comparators: the trivial predictor (the mean) and a classical fill
(normalised convolution — interpolation from the unmasked neighbours,
plan §7: "every inpainter against interpolation"). The 1D card reports a
1-nearest-neighbour probe on labelled cuts when the data has labels, with
the same probe on an untrained encoder beside it: if pretraining did not
help, the card says so.

THE CARD. Kind `ssl_backbone` (tier *measured* — a backbone emits no
output an analyst sees). `backbone.pt` holds the encoder's state only;
`load_backbone_into(model, model_dir)` puts it into a proposer, a
classifier or a bare backbone, refusing (in words) another profile, the
other branch, another architecture, or a 1D backbone pretrained at another
canonical rate. `proposer2d.train(pretrained=…)` and
`classifier1d.train(pretrained=…)` call it, and adopt the backbone's
architecture (and, for 2D, its tile normalisation).

LIMITS. A CNN sees across a patch edge, so masked modelling with plain
convolutions is partly inpainting from context — the objective is still
self-supervised, but weaker than a sparse-convolution encoder's (SparK).
Contrastive pretraining needs batches of more than a few cuts to have
negatives worth the name; at test scale it proves the path, not the gain.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from atk_diffusion import cards as _cards
from atk_diffusion import profiles as _profiles
from atk_diffusion.learn import common as K

torch = K.require_torch()
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from atk_diffusion.learn import data as D  # noqa: E402
from atk_diffusion.learn.classifier1d import (DEFAULT_RF_CONFIGS,  # noqa: E402
                                              ResNet1d, receptive_field)
from atk_diffusion.learn.proposer2d import ThinResNet2d  # noqa: E402

KIND = "ssl_backbone"
WEIGHTS = "backbone.pt"


# ---------------------------------------------------------------------------
# 2D — masked spectrogram modelling
# ---------------------------------------------------------------------------
class MaskedSpecModel(nn.Module):
    """ThinResNet2d + a light decoder: stride-32 features upsampled onto the
    stride-8 ones, two convolutions, upsampled to the tile."""

    def __init__(self, width: int = 16, dec_channels: int = 64):
        super().__init__()
        self.body = ThinResNet2d(1, width)
        w = self.body.widths
        c = int(dec_channels)
        self.lat2 = nn.Conv2d(w[1], c, 1)
        self.lat4 = nn.Conv2d(w[3], c, 1)
        self.dec = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(True),
                                 nn.Conv2d(c, 1, 3, 1, 1))

    def forward(self, x):
        _c1, c2, _c3, c4 = self.body(x)
        p = self.lat2(c2) + F.interpolate(self.lat4(c4), size=c2.shape[-2:],
                                          mode="nearest")
        y = self.dec(p)
        return F.interpolate(y, size=x.shape[-2:], mode="bilinear",
                             align_corners=False)


def patch_mask(batch: int, shape, patch: int, ratio: float,
               generator=None, device=None):
    """[B, 1, H, W] float mask, 1 where masked: a `ratio` of the patch×patch
    cells of each tile, chosen at random."""
    H, W = int(shape[0]), int(shape[1])
    gh, gw = math.ceil(H / patch), math.ceil(W / patch)
    n = gh * gw
    k = max(1, int(round(ratio * n)))
    scores = torch.rand(batch, n, generator=generator)
    idx = scores.argsort(dim=1)[:, :k]
    m = torch.zeros(batch, n)
    m.scatter_(1, idx, 1.0)
    m = m.view(batch, 1, gh, gw)
    m = m.repeat_interleave(patch, dim=2).repeat_interleave(patch, dim=3)
    m = m[:, :, :H, :W]
    return m.to(device) if device is not None else m


def classical_fill(x, mask, patch: int):
    """Normalised convolution: each masked pixel is the box-weighted mean of
    the UNMASKED pixels around it — interpolation, the classical comparator."""
    k = 2 * int(patch) + 1
    ker = torch.ones(1, 1, k, k, device=x.device, dtype=x.dtype)
    keep = 1.0 - mask
    num = F.conv2d(x * keep, ker, padding=k // 2)
    den = F.conv2d(keep, ker, padding=k // 2)
    return torch.where(den > 0, num / den.clamp(min=1e-6), torch.zeros_like(x))


class _ArrayTiles(torch.utils.data.Dataset):
    """Normalised tiles held in memory (from captures)."""

    def __init__(self, arrays, norm: K.TileNorm, train: bool, seed: int = 0):
        self.arrays = list(arrays)
        self.norm = norm
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, i):
        x = self.norm.apply(self.arrays[i])
        if self.train and self.rng.random() < 0.5:
            x = x[::-1]
        if self.train and self.rng.random() < 0.5:
            x = x[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(x, np.float32))[None]


class _SpecOnly(torch.utils.data.Dataset):
    """WidebandTiles without the targets (labels are not used here)."""

    def __init__(self, base: D.WidebandTiles):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        return self.base[i][0]

    def order(self, rng=None):
        return self.base.tiles.file_order(rng)


def _stack(batch):
    return torch.stack(batch)


def capture_tiles(rf, profile: str, captures, geom=None,
                  max_seconds: float = 60.0, progress=None) -> tuple[list, dict]:
    """Tiles (dB above floor) of raw captures of `profile`, through the front
    end `dsp.stft.tiles` — refusing a capture of another profile. Returns
    (list of [rows, bins] arrays, the geometry used as a dict)."""
    try:
        from atk_diffusion.dsp import stft as _stft
    except ImportError as e:
        raise RuntimeError("tiles from raw captures need the front end "
                           f"(atk_diffusion.dsp.stft), which is not available: "
                           f"{e}. Pretrain from a built wideband dataset "
                           "instead.") from None
    from atk_diffusion import sigmf
    if geom is None:
        geom = _profiles.load_profile(rf, profile).stft
    out = []
    for c in captures:
        meta = sigmf.read_meta(c)
        _profiles.check_match(profile, _profiles.profile_from_meta(meta),
                              what="self-supervised pretraining for this profile")
        fs = sigmf.sample_rate_of(meta)
        n = min(sigmf.num_samples(c, meta), int(max_seconds * fs))
        x = sigmf.load(c, 0, n, meta=meta)
        if x.ndim > 1:
            x = x[0]
        for t in _stft.tiles(x, fs, sigmf.center_of(meta), geom,
                             profile=profile):
            out.append(np.asarray(t.spec, np.float32))
        if progress:
            progress(f"{Path(sigmf.base_of(c)).name}: {len(out)} tiles so far")
    g = {k: getattr(geom, k) for k in geom.__dataclass_fields__}
    return out, g


def pretrain_2d(rf, profile: str, out_name: str, *, dataset_dir=None,
                captures=None, stft=None, splits=("train",), epochs: int = 10,
                batch_size: int = 8, lr: float = 1e-3, width: int = 16,
                mask_ratio: float = 0.5, patch: int = 8, device: str = "auto",
                seed: int = 0, threads: int | None = None,
                max_tiles: int | None = None, val_fraction: float = 0.15,
                max_seconds: float = 60.0, overwrite: bool = False,
                progress=None) -> Path:
    """Masked-spectrogram pretraining of the proposer's backbone on tiles of
    `profile` — from a wideband dataset (`dataset_dir`; labels ignored; only
    `splits`, by default the training split, so a proposer scored on the
    test split is scored on tiles its backbone never saw) or from raw
    captures (`captures`, SigMF paths; `stft` overrides the profile's
    geometry). Returns `rf.models(profile, out_name)`."""
    if dataset_dir is None and not captures:
        raise ValueError("pretraining needs tiles: give a wideband dataset or "
                         "a list of captures of this profile")
    threads_now = K.set_threads(threads)
    K.seed_everything(seed)
    dev, dev_words = K.pick_device(device)
    model_dir = K.new_model_dir(rf, profile, out_name, overwrite=overwrite)
    run = K.Run.start(rf, profile, f"ssl2d_{out_name}",
                      {"dataset": str(dataset_dir) if dataset_dir else None,
                       "captures": [str(c) for c in captures or []],
                       "epochs": epochs, "mask_ratio": mask_ratio,
                       "patch": patch, "width": width, "seed": seed}, progress)
    run.log(dev_words)
    run.log(f"{threads_now} CPU threads for PyTorch")
    datasets, geom_used = [], {}
    if dataset_dir is not None:
        m = K.open_dataset(dataset_dir, profile, "wideband", rf=rf)
        parts = [D.WidebandTiles(dataset_dir, sp, profile, manifest=m,
                                 normalize="full", train=True, seed=seed)
                 for sp in splits if K.split_files(dataset_dir, sp, "wideband")]
        parts = [p for p in parts if len(p)]
        if not parts:
            raise K.DatasetRefused("the dataset has no tiles")
        norm = K.TileNorm.fit(parts[0].tiles, seed=seed, applied_by="host")
        for p in parts:
            p.norm = norm
        full = torch.utils.data.ConcatDataset([_SpecOnly(p) for p in parts])
        shape = parts[0].shape
        geom_used = m.get("stft") or {}
        datasets.append(K.dataset_entry(dataset_dir, m, [p.tiles.split
                                                          for p in parts]))
        arrays = None
    else:
        arrays, geom_used = capture_tiles(rf, profile, captures, stft,
                                          max_seconds, run.log)
        if not arrays:
            raise K.DatasetRefused("the captures gave no tiles")
        shape = arrays[0].shape

        class _T:                      # TileNorm.fit wants .spec and len
            def __len__(s):
                return len(arrays)

            def spec(s, i):
                return arrays[i]
        norm = K.TileNorm.fit(_T(), seed=seed, applied_by="host")
        full = _ArrayTiles(arrays, norm, train=True, seed=seed)
        for c in captures:
            from atk_diffusion import sigmf
            datasets.append({"name": Path(sigmf.base_of(c)).name,
                             "sha256": K.sha256_path(sigmf.data_path(c)),
                             "kind": "capture", "generator": "recorded",
                             "label_sources": [], "splits_used": []})
    n = len(full)
    if max_tiles is not None:
        n = min(n, int(max_tiles))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(full))[:n]
    n_val = max(1, int(round(val_fraction * n))) if n > 1 else 0
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    if len(tr_idx) == 0:
        raise K.DatasetRefused("too few tiles to pretrain on")
    tr = torch.utils.data.Subset(full, tr_idx.tolist())
    run.log(f"{len(tr_idx)} tiles to pretrain on, {len(val_idx)} held out; "
            f"tiles {shape[0]}×{shape[1]}; normalisation mean "
            f"{norm.mean_db:.2f} dB, std {norm.std_db:.2f} dB")
    model = MaskedSpecModel(width).to(dev)
    dl = torch.utils.data.DataLoader(tr, batch_size=int(batch_size), shuffle=True,
                                     collate_fn=_stack,
                                     generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    steps = max(1, len(dl) * int(epochs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(lr),
                                                total_steps=steps, pct_start=0.15)
    gen = torch.Generator().manual_seed(seed + 1)
    for ep in range(1, int(epochs) + 1):
        model.train()
        tot = nb = 0.0
        with K.Timer() as tm:
            for x in dl:
                x = x.to(dev)
                mk = patch_mask(x.shape[0], x.shape[-2:], patch, mask_ratio,
                                gen, dev)
                pred = model(x * (1 - mk))
                loss = ((pred - x) ** 2 * mk).sum() / mk.sum().clamp(min=1.0)
                if not torch.isfinite(loss):
                    raise FloatingPointError("the pretraining loss became "
                                             f"{float(loss)}; lower lr.")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                sched.step()
                tot += float(loss.detach())
                nb += 1
        rec = {"epoch": ep, "split": "pretrain", "masked_mse": tot / max(1, nb),
               "seconds": round(tm.seconds, 3)}
        run.metric(**rec)
        run.log(f"pretrain epoch {ep}/{epochs}: masked-pixel MSE "
                f"{rec['masked_mse']:.4f}; {tm.seconds:.1f} s")
    # -- measured on held-out tiles beside two comparators -------------------
    model.eval()
    vgen = torch.Generator().manual_seed(seed + 2)
    se = triv = fill = cnt = 0.0
    with torch.no_grad():
        for i in val_idx.tolist():
            x = (full[i] if arrays is None else
                 torch.from_numpy(norm.apply(arrays[i]))[None])[None].to(dev)
            mk = patch_mask(1, x.shape[-2:], patch, mask_ratio, vgen, dev)
            pred = model(x * (1 - mk))
            se += float((((pred - x) ** 2) * mk).sum())
            triv += float(((x ** 2) * mk).sum())
            fill += float((((classical_fill(x, mk, patch) - x) ** 2) * mk).sum())
            cnt += float(mk.sum())
    metrics = {"objective": "masked spectrogram modelling",
               "heldout_tiles": int(len(val_idx)),
               "masked_mse": se / cnt if cnt else None,
               "trivial_mse": triv / cnt if cnt else None,
               "classical_fill_mse": fill / cnt if cnt else None}
    if cnt:
        metrics["beats_trivial"] = bool(metrics["masked_mse"] < metrics["trivial_mse"])
        metrics["beats_classical_fill"] = bool(metrics["masked_mse"]
                                               < metrics["classical_fill_mse"])
        run.log(f"held-out masked-pixel MSE {metrics['masked_mse']:.4f}; "
                f"predicting the mean {metrics['trivial_mse']:.4f}; classical "
                f"fill {metrics['classical_fill_mse']:.4f}")
    torch.save(model.body.state_dict(), model_dir / WEIGHTS)
    card = _cards.new_card(
        out_name, KIND, profile,
        input={"branch": "2d", "arch": model.body.arch(),
               "normalize": norm.to_json(), "tile_shape": [int(shape[0]),
                                                           int(shape[1])],
               "stft": geom_used,
               "objective": {"name": "masked spectrogram modelling",
                             "mask_ratio": float(mask_ratio),
                             "patch": int(patch)}},
        datasets=datasets, metrics=metrics, trained_on=dev_words,
        license="all rights reserved (see LICENSE)",
        notes=["self-supervised: no labels were used",
               "load with ssl.load_backbone_into(model, this folder); the "
               "proposer adopts this tile normalisation"])
    _cards.save(model_dir, card, WEIGHTS)
    for f in (model_dir / WEIGHTS, model_dir / _cards.CARD_FILE):
        try:
            rf.record(f, f"model:{KIND}", out_name)
        except Exception:                                  # noqa: BLE001
            pass
    run.finish({"model_dir": str(model_dir), **{k: v for k, v in metrics.items()
                                                if isinstance(v, (int, float))}})
    return model_dir


# ---------------------------------------------------------------------------
# 1D — contrastive pretraining with RF augmentations
# ---------------------------------------------------------------------------
class ContrastiveIQ(nn.Module):
    def __init__(self, rf_config: dict | None = None, width: int = 32,
                 proj_dim: int = 64):
        super().__init__()
        cfg = dict(rf_config or DEFAULT_RF_CONFIGS[1])
        self.encoder = ResNet1d(2, width, cfg["kernel"], cfg["dilations"],
                                cfg.get("blocks", (1, 1, 1, 1)))
        d = self.encoder.out_dim
        self.proj = nn.Sequential(nn.Linear(d, d), nn.ReLU(True),
                                  nn.Linear(d, proj_dim))

    def forward(self, x):
        return F.normalize(self.proj(self.encoder(x)), dim=1)


def augment_iq(x, rate: float, max_cfo_hz: float, max_shift: int,
               snr_db=(0.0, 20.0), generator=None):
    """[B, 2, L] -> an RF-augmented view: phase rotation, carrier offset,
    circular time shift, noise at a random SNR; back to unit RMS. None of
    these changes what the signal IS."""
    B, _, L = x.shape
    dev = x.device
    z = torch.complex(x[:, 0], x[:, 1])
    g = generator
    phi = torch.rand(B, 1, generator=g).to(dev) * 2 * math.pi
    cfo = (torch.rand(B, 1, generator=g).to(dev) * 2 - 1) * float(max_cfo_hz)
    n = torch.arange(L, device=dev, dtype=torch.float32)[None]
    z = z * torch.exp(1j * (phi + 2 * math.pi * cfo * n / float(rate)))
    if max_shift > 0:
        shifts = torch.randint(-int(max_shift), int(max_shift) + 1, (B,),
                               generator=g).tolist()
        z = torch.stack([torch.roll(z[i], s) for i, s in enumerate(shifts)])
    lo, hi = float(snr_db[0]), float(snr_db[1])
    snr = lo + (hi - lo) * torch.rand(B, 1, generator=g).to(dev)
    sigma = torch.sqrt(10 ** (-snr / 10) / 2)
    noise = torch.complex(torch.randn(B, L, generator=g).to(dev),
                          torch.randn(B, L, generator=g).to(dev)) * sigma
    z = z + noise
    z = z / z.abs().pow(2).mean(dim=1, keepdim=True).sqrt().clamp(min=1e-8)
    return torch.stack([z.real, z.imag], dim=1).float()


def nt_xent(z1, z2, temperature: float = 0.2):
    """SimCLR's normalised-temperature cross-entropy over a batch of pairs."""
    z = torch.cat([z1, z2], dim=0)
    b = z1.shape[0]
    sim = z @ z.t() / float(temperature)
    sim = sim - torch.eye(2 * b, device=z.device) * 1e9
    target = torch.cat([torch.arange(b, 2 * b), torch.arange(0, b)]).to(z.device)
    return F.cross_entropy(sim, target)


def _centre(x, window: int) -> np.ndarray:
    """The centred `window` samples of a cut (complex)."""
    x = np.asarray(x, np.complex64).reshape(-1)
    if window > x.size:
        raise ValueError(f"a window of {window} samples does not fit a cut of "
                         f"{x.size}")
    o = (x.size - window) // 2
    return x[o:o + window]


class _IQWindows(torch.utils.data.Dataset):
    def __init__(self, arrays):
        self.arrays = arrays

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, i):
        return torch.from_numpy(K.iq_window(self.arrays[i]))


def capture_cuts(rf, profile: str, captures, canonical_class: str = "voice",
                 window: int = 1024, cuts_per_capture: int = 256,
                 seed: int = 0, max_seconds: float = 60.0) -> tuple[list, dict]:
    """Unlabeled cuts from raw captures: random times and frequency offsets,
    shifted and INTEGER-decimated (dsp.resample) to the profile's canonical
    rate for `canonical_class`. Returns (list of complex (window,), the
    canonical rate as {class, rate, decimation})."""
    from atk_diffusion import sigmf
    from atk_diffusion.dsp import resample as _rs
    fs = float(_profiles.parse_profile_id(profile).sample_rate)
    can = {c.cls: c for c in _profiles.canonical_rates(fs)}.get(canonical_class)
    if can is None:
        raise ValueError(f"{_profiles.describe(profile)} has no {canonical_class} "
                         "canonical rate")
    d = int(can.decimation)
    rng = np.random.default_rng(seed)
    need = int(window) * d + 64 * d
    out = []
    for c in captures:
        meta = sigmf.read_meta(c)
        _profiles.check_match(profile, _profiles.profile_from_meta(meta),
                              what="self-supervised pretraining for this profile")
        n = min(sigmf.num_samples(c, meta), int(max_seconds * fs))
        if n < need:
            raise K.DatasetRefused(f"{Path(sigmf.base_of(c)).name} is too short "
                                   f"for a {window}-sample cut at "
                                   f"{can.rate:g} S/s ({need} samples needed).")
        x = sigmf.load(c, 0, n, meta=meta)
        if x.ndim > 1:
            x = x[0]
        span = max(0.0, 0.5 * fs - 0.5 * can.rate)
        for _ in range(int(cuts_per_capture)):
            s0 = int(rng.integers(0, n - need + 1))
            off = float(rng.uniform(-span, span))
            seg = _rs.shift(x[s0:s0 + need], off, fs, n0=s0)
            y, _fs2 = _rs.decimate(seg, d, fs)
            if y.size >= window:
                out.append(y[(y.size - window) // 2:(y.size - window) // 2 + window])
    return out, {"class": can.cls, "rate": float(can.rate), "decimation": d}


@torch.no_grad()
def _features(encoder, arrays, device, batch: int = 256) -> np.ndarray:
    encoder.eval()
    outs = []
    for s in range(0, len(arrays), batch):
        x = torch.from_numpy(np.stack([K.iq_window(a) for a in arrays[s:s + batch]]))
        outs.append(F.normalize(encoder(x.to(device)), dim=1).cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0, encoder.out_dim))


def _knn_accuracy(f_tr, y_tr, f_te, y_te) -> float | None:
    if len(f_tr) == 0 or len(f_te) == 0:
        return None
    pred = y_tr[np.argmax(f_te @ f_tr.T, axis=1)]
    return float(np.mean(pred == y_te))


def pretrain_1d(rf, profile: str, out_name: str, *, dataset_dir=None,
                captures=None, splits=("train",), canonical_class: str = "voice",
                window: int | None = None, cuts_per_capture: int = 256,
                rf_config: dict | None = None, width: int = 32,
                epochs: int = 10, batch_size: int = 64, lr: float = 1e-3,
                temperature: float = 0.2, max_cfo_frac: float = 0.01,
                max_shift: int | None = None, snr_db=(0.0, 20.0),
                device: str = "auto", seed: int = 0, threads: int | None = None,
                max_seconds: float = 60.0, overwrite: bool = False,
                progress=None) -> Path:
    """Contrastive pretraining of the classifier's IQ branch on cuts of
    `profile` at one canonical rate — from a narrowband dataset
    (`dataset_dir`; only `splits`, by default the training split; labels
    ignored for training and used only by the 1-NN probe, train -> val) or
    raw captures (`captures`; `canonical_class` picks the rate). Returns
    `rf.models(profile, out_name)`."""
    if dataset_dir is None and not captures:
        raise ValueError("pretraining needs cuts: give a narrowband dataset or "
                         "a list of captures of this profile")
    threads_now = K.set_threads(threads)
    K.seed_everything(seed)
    dev, dev_words = K.pick_device(device)
    model_dir = K.new_model_dir(rf, profile, out_name, overwrite=overwrite)
    run = K.Run.start(rf, profile, f"ssl1d_{out_name}",
                      {"dataset": str(dataset_dir) if dataset_dir else None,
                       "captures": [str(c) for c in captures or []],
                       "epochs": epochs, "width": width, "seed": seed,
                       "temperature": temperature}, progress)
    run.log(dev_words)
    run.log(f"{threads_now} CPU threads for PyTorch")
    datasets = []
    probe = None
    if dataset_dir is not None:
        m = K.open_dataset(dataset_dir, profile, "narrowband", rf=rf)
        canonical = dict(m.get("canonical") or {})
        arrays = []
        labelled = {}
        for sp in K.SPLITS:
            sh = K.ShardSet(dataset_dir, sp, m)
            if not len(sh):
                continue
            a = [sh.iq(i) for i in range(len(sh))]
            if sp in splits:
                arrays.extend(a)
            labelled[sp] = (a, sh.labels["label"])
        L = len(arrays[0]) if arrays else 0
        window = int(window or L)
        arrays = [_centre(a, window) for a in arrays]
        if "train" in labelled and ("val" in labelled or "test" in labelled):
            te = labelled.get("val") or labelled.get("test")
            probe = (labelled["train"], te)
        datasets.append(K.dataset_entry(dataset_dir, m,
                                        [sp for sp in splits if sp in labelled]))
    else:
        window = int(window or 1024)
        arrays, canonical = capture_cuts(rf, profile, captures, canonical_class,
                                         window, cuts_per_capture, seed,
                                         max_seconds)
        from atk_diffusion import sigmf
        for c in captures:
            datasets.append({"name": Path(sigmf.base_of(c)).name,
                             "sha256": K.sha256_path(sigmf.data_path(c)),
                             "kind": "capture", "generator": "recorded",
                             "label_sources": [], "splits_used": []})
    if len(arrays) < 2:
        raise K.DatasetRefused("contrastive pretraining needs at least two cuts")
    rate = float(canonical.get("rate") or 0.0)
    if rate <= 0:
        raise K.DatasetRefused("the cuts' canonical rate is unknown (the "
                               "dataset manifest has no 'canonical')")
    cfg = dict(rf_config or DEFAULT_RF_CONFIGS[1])
    shift = int(max_shift if max_shift is not None else window // 8)
    max_cfo = float(max_cfo_frac) * rate
    run.log(f"{len(arrays)} unlabeled cuts of {window} samples at {rate:g} S/s "
            f"({canonical.get('class')}); receptive field "
            f"{receptive_field(cfg['kernel'], cfg['dilations'], cfg.get('blocks', (1, 1, 1, 1)))} "
            f"samples; augmentations: phase, carrier offset ±{max_cfo:g} Hz, "
            f"shift ±{shift} samples, noise {snr_db[0]:g}–{snr_db[1]:g} dB SNR")
    model = ContrastiveIQ(cfg, width).to(dev)
    knn_before = None
    if probe is not None:
        (a_tr, y_tr), (a_te, y_te) = probe
        a_tr = [_centre(x, window) for x in a_tr]
        a_te = [_centre(x, window) for x in a_te]
        knn_before = _knn_accuracy(_features(model.encoder, a_tr, dev), y_tr,
                                   _features(model.encoder, a_te, dev), y_te)
    ds = _IQWindows(arrays)
    bs = max(2, min(int(batch_size), len(ds)))
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True,
                                     drop_last=len(ds) >= 2 * bs,
                                     generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    steps = max(1, len(dl) * int(epochs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(lr),
                                                total_steps=steps, pct_start=0.15)
    gen = torch.Generator().manual_seed(seed + 1)
    last = float("nan")
    for ep in range(1, int(epochs) + 1):
        model.train()
        tot = nb = 0.0
        with K.Timer() as tm:
            for x in dl:
                if x.shape[0] < 2:
                    continue
                x = x.to(dev)
                v1 = augment_iq(x, rate, max_cfo, shift, snr_db, gen)
                v2 = augment_iq(x, rate, max_cfo, shift, snr_db, gen)
                loss = nt_xent(model(v1), model(v2), temperature)
                if not torch.isfinite(loss):
                    raise FloatingPointError("the contrastive loss became "
                                             f"{float(loss)}; lower lr.")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                tot += float(loss.detach())
                nb += 1
        last = tot / max(1, nb)
        run.metric(epoch=ep, split="pretrain", nt_xent=last,
                   seconds=round(tm.seconds, 3))
        run.log(f"pretrain epoch {ep}/{epochs}: contrastive loss {last:.4f}; "
                f"{tm.seconds:.1f} s")
    model.eval()
    # agreement of two views vs two different cuts, on up to 256 cuts
    with torch.no_grad():
        sel = np.random.default_rng(seed + 3).permutation(len(ds))[:256]
        x = torch.stack([ds[int(i)] for i in sel]).to(dev)
        g2 = torch.Generator().manual_seed(seed + 4)
        z1 = model(augment_iq(x, rate, max_cfo, shift, snr_db, g2))
        z2 = model(augment_iq(x, rate, max_cfo, shift, snr_db, g2))
        pos = float((z1 * z2).sum(dim=1).mean())
        sim = z1 @ z2.t()
        neg = float((sim.sum() - sim.diag().sum()) / max(1, sim.numel() - len(sel)))
    metrics = {"objective": "contrastive (NT-Xent) with RF augmentations",
               "final_loss": K.finite_or_none(last), "positive_cosine": pos,
               "negative_cosine": neg, "cuts": len(arrays)}
    if probe is not None:
        (a_tr, y_tr), (a_te, y_te) = probe
        a_tr = [_centre(x, window) for x in a_tr]
        a_te = [_centre(x, window) for x in a_te]
        knn_after = _knn_accuracy(_features(model.encoder, a_tr, dev), y_tr,
                                  _features(model.encoder, a_te, dev), y_te)
        metrics["knn_probe_accuracy"] = knn_after
        metrics["knn_probe_untrained"] = knn_before
        run.log(f"1-NN probe on labelled cuts: {knn_after:.3f} after "
                f"pretraining, {knn_before:.3f} with the untrained encoder")
    run.log(f"two views of a cut agree at cosine {pos:.3f}; different cuts "
            f"{neg:.3f}")
    torch.save(model.encoder.state_dict(), model_dir / WEIGHTS)
    card = _cards.new_card(
        out_name, KIND, profile,
        input={"branch": "1d", "arch": model.encoder.arch(), "window": int(window),
               "iq_len": int(window), "iq_norm": "rms",
               "canonical": {"class": canonical.get("class"),
                             "cls": canonical.get("class"), "rate": rate,
                             "decimation": canonical.get("decimation")},
               "augmentations": {"phase": "uniform 0–2π",
                                 "carrier_offset_hz": max_cfo,
                                 "time_shift_samples": shift,
                                 "snr_db": [float(snr_db[0]), float(snr_db[1])]},
               "objective": {"name": "NT-Xent", "temperature": float(temperature)}},
        datasets=datasets, metrics=metrics, trained_on=dev_words,
        license="all rights reserved (see LICENSE)",
        notes=["self-supervised: no labels were used for training (labels, "
               "where the data has them, only score the 1-NN probe)"])
    _cards.save(model_dir, card, WEIGHTS)
    for f in (model_dir / WEIGHTS, model_dir / _cards.CARD_FILE):
        try:
            rf.record(f, f"model:{KIND}", out_name)
        except Exception:                                  # noqa: BLE001
            pass
    run.finish({"model_dir": str(model_dir),
                **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}})
    return model_dir


# ---------------------------------------------------------------------------
# Loading a pretrained backbone into a model
# ---------------------------------------------------------------------------
def _target(model, branch: str):
    if branch == "2d":
        if isinstance(model, ThinResNet2d):
            return model
        bb = getattr(model, "backbone", None)
        if bb is not None and isinstance(getattr(bb, "body", None), ThinResNet2d):
            return bb.body
        if isinstance(getattr(model, "body", None), ThinResNet2d):
            return model.body
    else:
        if isinstance(model, ResNet1d):
            return model
        for name in ("iq", "encoder"):
            if isinstance(getattr(model, name, None), ResNet1d):
                return getattr(model, name)
    return None


def load_backbone_into(model, model_dir, profile: str | None = None,
                       canonical_rate: float | None = None) -> dict:
    """Load a pretrained backbone (`ssl_backbone` card) into `model` — a
    proposer (its FCOS backbone body), a classifier (its IQ branch), a
    MaskedSpecModel / ContrastiveIQ, or a bare backbone. Refuses in words:
    another profile (ProfileMismatch), the wrong branch, another
    architecture, or a 1D backbone pretrained at another canonical rate."""
    card = _cards.load(model_dir, expect_kind=KIND, for_profile=profile)
    branch = (card.input or {}).get("branch")
    target = _target(model, branch)
    if target is None:
        raise _cards.CardRefusal(f"{card.name} is a {branch} backbone and this "
                                 f"model ({type(model).__name__}) has no {branch} "
                                 "backbone to load it into.")
    want = (card.input or {}).get("arch") or {}
    have = target.arch()
    if want != have:
        raise _cards.CardRefusal(f"{card.name} was pretrained as {want}; this "
                                 f"model's backbone is {have}. The architectures "
                                 "must match — build the model with the "
                                 "backbone's width and configuration.")
    if branch == "1d" and canonical_rate is not None:
        r = float(((card.input or {}).get("canonical") or {}).get("rate") or 0)
        if r and abs(r - float(canonical_rate)) > 0.5:
            raise _cards.CardRefusal(
                f"{card.name} was pretrained on cuts at {r:g} S/s; this "
                f"classifier's cuts are at {float(canonical_rate):g} S/s. A "
                "network never meets a rate it was not trained at.")
    state = torch.load(_cards.weights_path(model_dir, card), map_location="cpu",
                       weights_only=True)
    target.load_state_dict(state, strict=True)
    return {"loaded": len(state), "branch": branch, "card": card.name,
            "target": type(target).__name__}


def backbone_arch(model_dir, profile: str | None = None) -> dict:
    """The architecture a pretrained backbone needs its model built with."""
    card = _cards.load(model_dir, expect_kind=KIND, for_profile=profile,
                       verify_weights=False)
    return dict((card.input or {}).get("arch") or {})
