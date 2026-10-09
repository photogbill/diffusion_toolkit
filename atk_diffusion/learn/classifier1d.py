# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The 1D classifier: a cut's IQ and its SCF -> class, embedding, cycle
parameters (DETECTION_DESIGN §4, §4.1, §6, §7; decisions D6, D9).

§4: *"Two inputs, one network: a 1D convolutional branch on a fixed-length
IQ window … with receptive fields chosen by search rather than habit —
RF-Next — and a 2D branch on the cut's spectral correlation function, fused
before the head. Three outputs: the modulation/protocol class, a normalized
embedding of the signal, and regressed cycle parameters — symbol rate,
carrier offset — supervised directly, since synthetic data knows them
exactly. The SCF branch is what holds accuracy up at low SNR; the IQ branch
is what tells QPSK from 8PSK."*

THE NETWORK
* IQ branch — `ResNet1d`: a plain 1D ResNet (stem /2, four stages at
  strides 1, 2, 2, 2) whose kernel size and per-stage dilations are a
  receptive-field configuration. Input [B, 2, L], unit RMS.
* SCF branch — three conv/pool stages and a global average, on the cached
  SCF [B, 1, H, W], divided by its maximum (`scf_norm: "max"`, the scaling
  `detect.onnx_models.Classifier1D` applies at inference).
* Fusion — concatenation -> a 128-unit layer (LayerNorm, so a batch of one
  behaves like a batch of many) -> three heads:
  `logits` [B, C] = s · cos(embedding, class weight) — a cosine classifier
  trained with an additive margin (CosFace), so that the class decision
  and the embedding live in the same space and a class's prototype (its
  mean embedding) sits where its weight points; `embedding` [B, 64],
  L2-normalised — what the prototype bank compares (D6, *unknown* and
  *teach*); `cycle` [B, 2] — symbol rate and carrier offset divided by
  `card.input["cycle_scale"]` (Hz = output × scale), masked in the loss
  where a class has no symbol rate.

RECEPTIVE FIELDS BY SEARCH (RF-Next, plan §9 — 2206.06637). RF-Next searches
dilation rates globally then locally inside training. Here the idea is kept
and the machinery is not: `search_receptive_fields()` trains the IQ branch
briefly under each of a few (kernel, dilations) configurations and picks
the one with the best validation accuracy (ties to the smaller receptive
field). Honest size: a small grid search, not RF-Next's optimiser; the
results table goes into the card and the run log, so the choice can be
read, not trusted.

THE CLASSICAL BASELINE, beside it (plan §7: "every learned tool has its
classical comparator"). `cumulant_baseline()` classifies the same cuts by
nearest centroid on higher-order cumulants — |C20|, |C40|, |C41|, C42 —
plus the amplitude's spread and the instantaneous frequency's, the classic
AMC features (Swami & Sadler). The cumulant arithmetic follows Bill's own
`atk/core/siga/csp/cumulants.py` at cycle frequency zero (ported, not
imported). The card records both accuracies.

WHAT `train()` WRITES — `rf_data\\<profile>\\models\\<name>\\`: `model.onnx`
(inputs `iq` [B, 2, L], `scf` [B, 1, H, W]; outputs `logits`, `embedding`,
`cycle`; batch dynamic — ARCHITECTURE §5), `model.pt`, `card.json` (kind
`classifier1d`), `prototypes.json` + `prototypes.npz` beside the card (the
`detect.prototypes.PrototypeBank`, from `calibrate.open_set`). One
model per (profile, canonical class): a dataset is cut at one canonical
rate (DETECTION_DESIGN §4, D2) and the card says which. Numbers in the card
are measured through the exported ONNX: accuracy and macro-F1 on held-out
cuts, accuracy versus SNR, cycle-parameter error, the temperature and
calibration error (fitted on val, reported on test), unknown rejection on
classes held out of training, CPU latency per cutout (budget 20 ms). The
input fields are the ones `detect.onnx_models.Classifier1D` reads —
`iq_len`, `iq_norm`, `scf_norm`, `scf_shape`, `canonical {cls, rate,
decimation}`, `cycle_scale` — and `calibration.temperature` is a number.

`finetune()` takes a saved classifier to the cabled set (§6.3) or the first
minutes on site and saves a NEW model, measured and calibrated the same way
(`save_trained`, shared with `train()`).

LIMITS. Trained here only at test scale; the class list is the dataset's.
A cut with no cached SCF can train the IQ branch alone (`use_scf=False`);
the graph then accepts and ignores its `scf` input and the card says
`scf_used: false` — but `detect.onnx_models.Classifier1D` does not read
that field yet, so such a model must be fed a [B, 1, 1, 1] zero by its
host. Models trained on datasets with cached SCF (the builder's default)
do not have this caveat.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from atk_diffusion import cards as _cards
from atk_diffusion.learn import common as K

torch = K.require_torch()
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from atk_diffusion.learn import data as D  # noqa: E402
from atk_diffusion.learn import export as X  # noqa: E402

KIND = "classifier1d"
LATENCY_BUDGET_MS = 20.0            # DETECTION_DESIGN §7, per cutout

#: The receptive-field configurations searched by default: plain small and
#: large kernels, and dilations that grow with depth.
DEFAULT_RF_CONFIGS = (
    {"name": "k3_d1", "kernel": 3, "dilations": [1, 1, 1, 1]},
    {"name": "k7_d1", "kernel": 7, "dilations": [1, 1, 1, 1]},
    {"name": "k3_d1248", "kernel": 3, "dilations": [1, 2, 4, 8]},
    {"name": "k5_d1224", "kernel": 5, "dilations": [1, 2, 2, 4]},
)


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------
class BasicBlock1d(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int, dilation: int,
                 stride: int):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv1 = nn.Conv1d(cin, cout, kernel, stride, pad, dilation,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, kernel, 1, pad, dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False),
                                      nn.BatchNorm1d(cout))

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + idt)


class ResNet1d(nn.Module):
    """The IQ branch. Output: [B, 2 · 4·width] (global average and max)."""

    STRIDES = (1, 2, 2, 2)

    def __init__(self, in_ch: int = 2, width: int = 32, kernel: int = 7,
                 dilations=(1, 1, 1, 1), blocks=(1, 1, 1, 1)):
        super().__init__()
        k = int(kernel)
        if k % 2 == 0 or k < 1:
            raise ValueError("the kernel size must be odd")
        self.kernel = k
        self.width = int(width)
        self.dilations = tuple(int(d) for d in dilations)
        self.blocks = tuple(int(b) for b in blocks)
        self.in_ch = int(in_ch)
        w = self.width
        self.widths = (w, 2 * w, 4 * w, 4 * w)
        self.stem = nn.Sequential(nn.Conv1d(in_ch, w, k, 2, (k - 1) // 2,
                                            bias=False),
                                  nn.BatchNorm1d(w), nn.ReLU(inplace=True))
        cin = w
        stages = []
        for c, d, n, s in zip(self.widths, self.dilations, self.blocks,
                              self.STRIDES):
            layers = [BasicBlock1d(cin, c, k, d, s)]
            layers += [BasicBlock1d(c, c, k, d, 1) for _ in range(n - 1)]
            stages.append(nn.Sequential(*layers))
            cin = c
        self.stages = nn.Sequential(*stages)
        self.out_dim = 2 * self.widths[-1]

    def arch(self) -> dict:
        return {"name": "resnet1d", "in_ch": self.in_ch, "width": self.width,
                "kernel": self.kernel, "dilations": list(self.dilations),
                "blocks": list(self.blocks)}

    def receptive_field(self) -> int:
        return receptive_field(self.kernel, self.dilations, self.blocks)

    def forward(self, x):
        y = self.stages(self.stem(x))
        return torch.cat([y.mean(dim=2), y.amax(dim=2)], dim=1)


def receptive_field(kernel: int, dilations, blocks=(1, 1, 1, 1)) -> int:
    """The IQ branch's receptive field in input samples (stem included)."""
    rf, jump = 1, 1
    rf += (kernel - 1) * jump           # stem, dilation 1
    jump *= 2
    for d, n, s in zip(dilations, blocks, ResNet1d.STRIDES):
        for b in range(int(n)):
            rf += (kernel - 1) * d * jump       # conv1 (stride s on the first)
            if b == 0:
                jump *= s
            rf += (kernel - 1) * d * jump       # conv2
    return int(rf)


class ScfBranch(nn.Module):
    def __init__(self, width: int = 16):
        super().__init__()
        w = int(width)
        self.width = w
        self.net = nn.Sequential(
            nn.Conv2d(1, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(True),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(w, 2 * w, 3, 1, 1, bias=False), nn.BatchNorm2d(2 * w),
            nn.ReLU(True), nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(2 * w, 4 * w, 3, 1, 1, bias=False), nn.BatchNorm2d(4 * w),
            nn.ReLU(True), nn.AdaptiveAvgPool2d(1))
        self.out_dim = 4 * w

    def forward(self, s):
        return self.net(s).flatten(1)


class NoScf(nn.Module):
    """Stands in for the SCF branch when the dataset has no cached SCF: the
    graph keeps its `scf` input (the contract) and multiplies it by zero."""
    out_dim = 1

    def forward(self, s):
        return s.flatten(1)[:, :1] * 0.0


class TwoBranchClassifier(nn.Module):
    def __init__(self, num_classes: int, rf_config: dict | None = None,
                 iq_width: int = 32, scf_width: int = 16, use_scf: bool = True,
                 embed_dim: int = 64, hidden: int = 128,
                 cosine_scale: float = 16.0, dropout: float = 0.1):
        super().__init__()
        cfg = dict(rf_config or DEFAULT_RF_CONFIGS[1])
        self.iq = ResNet1d(2, iq_width, cfg["kernel"], cfg["dilations"],
                           cfg.get("blocks", (1, 1, 1, 1)))
        self.scf = ScfBranch(scf_width) if use_scf else NoScf()
        self.fuse = nn.Sequential(nn.Linear(self.iq.out_dim + self.scf.out_dim,
                                            hidden),
                                  nn.LayerNorm(hidden), nn.ReLU(True),
                                  nn.Dropout(dropout))
        self.embed = nn.Linear(hidden, embed_dim)
        self.class_weight = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.1)
        self.cycle_head = nn.Linear(hidden, 2)
        self.cosine_scale = float(cosine_scale)
        self.arch = {"name": "two_branch_iq_scf", "num_classes": int(num_classes),
                     "rf_config": {"name": cfg.get("name", ""),
                                   "kernel": int(cfg["kernel"]),
                                   "dilations": list(cfg["dilations"]),
                                   "blocks": list(cfg.get("blocks", (1, 1, 1, 1)))},
                     "iq_width": int(iq_width), "scf_width": int(scf_width),
                     "use_scf": bool(use_scf), "embed_dim": int(embed_dim),
                     "hidden": int(hidden), "cosine_scale": float(cosine_scale),
                     "dropout": float(dropout)}

    def forward(self, iq, scf):
        h = self.fuse(torch.cat([self.iq(iq), self.scf(scf)], dim=1))
        e = F.normalize(self.embed(h), dim=1)
        w = F.normalize(self.class_weight, dim=1)
        logits = self.cosine_scale * (e @ w.t())
        return logits, e, self.cycle_head(h)


def build_from_arch(arch: dict) -> TwoBranchClassifier:
    a = dict(arch)
    return TwoBranchClassifier(a["num_classes"], a["rf_config"], a["iq_width"],
                               a["scf_width"], a["use_scf"], a["embed_dim"],
                               a["hidden"], a["cosine_scale"],
                               a.get("dropout", 0.1))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _loss(model, batch, margin: float, cycle_weight: float):
    logits, _e, cyc = model(batch["iq"], batch["scf"])
    y = batch["label"]
    onehot = F.one_hot(y.clamp(min=0), logits.shape[1]).float()
    lm = logits - model.cosine_scale * float(margin) * onehot
    ce = F.cross_entropy(lm, y)
    mask = batch["cycle_mask"]
    reg = (F.smooth_l1_loss(cyc, batch["cycle"], reduction="none", beta=0.01)
           * mask).sum() / mask.sum().clamp(min=1.0)
    return ce + float(cycle_weight) * reg, ce, reg, logits


@torch.no_grad()
def evaluate_torch(model, ds, device=None, batch_size: int = 128) -> dict:
    """Accuracy of a PyTorch classifier over the known items of `ds`."""
    device = device or torch.device("cpu")
    model.to(device).eval()
    dl = D.loader(ds, batch_size, train=False)
    right = n = 0
    for b in dl:
        b = {k: v.to(device) for k, v in b.items()}
        logits, _e, _c = model(b["iq"], b["scf"])
        known = b["label"] >= 0
        right += int((logits.argmax(1)[known] == b["label"][known]).sum())
        n += int(known.sum())
    return {"accuracy": right / n if n else float("nan"), "n": n}


def fit(model, train_ds, epochs: int, *, val_ds=None, lr: float = 2e-3,
        weight_decay: float = 1e-4, batch_size: int = 64, margin: float = 0.15,
        cycle_weight: float = 1.0, device=None, seed: int = 0,
        run: K.Run | None = None, amp: bool = True, num_workers: int = 0,
        keep_best: bool = True, checkpoint_every: int = 1, resume_from=None,
        tag: str = "train") -> list[dict]:
    """Train (or fine-tune) the classifier; keeps the epoch with the best
    validation accuracy when a validation set is given. With a run, the
    model, optimiser, schedule, history and the best state so far go to the
    run's `checkpoints/last.pt` every `checkpoint_every` epochs, and
    `resume_from` continues a run that stopped."""
    device = device or torch.device("cpu")
    model.to(device)
    dl = D.loader(train_ds, batch_size, train=True, seed=seed,
                  num_workers=num_workers,
                  drop_last=len(train_ds) > batch_size)
    steps = max(1, len(dl) * int(epochs))
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr),
                            weight_decay=float(weight_decay))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(lr),
                                                total_steps=steps,
                                                pct_start=0.15)
    use_amp = bool(amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    best_acc, best_state = -1.0, None
    hist, start = [], 1
    if resume_from:
        ck = torch.load(resume_from, map_location=device, weights_only=True)
        for k in ("model", "optimizer", "scheduler", "epoch"):
            if k not in ck:
                raise ValueError(f"{Path(resume_from).name} is not a training "
                                 f"checkpoint of this toolkit (no {k!r}).")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        hist = list(ck.get("history") or [])
        start = int(ck["epoch"]) + 1
        best_acc = float(ck.get("best_acc", -1.0))
        best_state = ck.get("best_state") or None
        if run is not None:
            run.log(f"resumed from {Path(resume_from).name}: epoch "
                    f"{ck['epoch']} done, continuing")
    for ep in range(start, int(epochs) + 1):
        model.train()
        tot = ce_s = reg_s = 0.0
        right = n = nb = 0
        with K.Timer() as tm:
            for b in dl:
                b = {k: v.to(device) for k, v in b.items()}
                with torch.autocast(device.type, dtype=torch.float16,
                                    enabled=use_amp):
                    loss, ce, reg, logits = _loss(model, b, margin, cycle_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"the training loss became {float(loss)} in epoch {ep}; "
                        "lower the learning rate (lr) and try again.")
                opt.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
                sched.step()
                tot += float(loss.detach())
                ce_s += float(ce.detach())
                reg_s += float(reg.detach())
                right += int((logits.argmax(1) == b["label"]).sum())
                n += int(b["label"].numel())
                nb += 1
        rec = {"epoch": ep, "split": tag, "loss": tot / max(1, nb),
               "ce": ce_s / max(1, nb), "cycle": reg_s / max(1, nb),
               "train_accuracy": right / max(1, n),
               "seconds": round(tm.seconds, 3)}
        if val_ds is not None and len(val_ds):
            rec["val_accuracy"] = evaluate_torch(model, val_ds, device)["accuracy"]
            if keep_best and rec["val_accuracy"] > best_acc:
                best_acc = rec["val_accuracy"]
                best_state = copy.deepcopy(model.state_dict())
        hist.append(rec)
        if run is not None:
            run.metric(**rec)
            run.log(f"{tag} epoch {ep}/{epochs}: loss {rec['loss']:.3f}, train "
                    f"accuracy {rec['train_accuracy']:.3f}"
                    + (f", val accuracy {rec['val_accuracy']:.3f}"
                       if "val_accuracy" in rec else "")
                    + f"; {tm.seconds:.1f} s")
            if checkpoint_every and ep % int(checkpoint_every) == 0:
                ck = {"model": model.state_dict(), "optimizer": opt.state_dict(),
                      "scheduler": sched.state_dict(), "epoch": ep,
                      "epochs": int(epochs), "history": hist,
                      "best_acc": float(best_acc)}
                if best_state is not None:
                    ck["best_state"] = best_state
                run.save_checkpoint(ck, "last")
    if best_state is not None:
        model.load_state_dict(best_state)
        if run is not None:
            run.log(f"kept the epoch with the best val accuracy ({best_acc:.3f})")
    model.eval()
    return hist


def search_receptive_fields(rf, profile: str, dataset_dir, configs=None,
                            epochs: int = 3, *, window: int | None = None,
                            iq_width: int = 16, batch_size: int = 64,
                            lr: float = 2e-3, seed: int = 0, device: str = "auto",
                            threads: int | None = None, held_out_classes=(),
                            max_items: int | None = None, run: K.Run | None = None,
                            progress=None) -> dict:
    """RF-Next, as a small search: train the IQ branch alone briefly under
    each (kernel, dilations) configuration and rank them by validation
    accuracy (ties to the smaller receptive field). Returns {"best",
    "results", "method"}; the results go into the run log."""
    K.set_threads(threads)
    dev, _w = K.pick_device(device)
    m = K.open_dataset(dataset_dir, profile, "narrowband", rf=rf)
    classes = [c for c in (m.get("classes") or []) if c not in set(held_out_classes)]
    rate = float((m.get("canonical") or {}).get("rate") or m["sample_rate"])
    tr = D.NarrowbandDataset(dataset_dir, "train", profile, manifest=m,
                             classes=classes, window=window, train=True,
                             use_scf=False, max_items=max_items, seed=seed,
                             known_only=True)
    va = D.NarrowbandDataset(dataset_dir, "val", profile, manifest=m,
                             classes=classes, window=window, use_scf=False,
                             max_items=max_items, known_only=True)
    if len(va) == 0:
        raise K.DatasetRefused("the receptive-field search ranks on the val "
                               "split, and it is empty.")
    results = []
    for cfg in (configs or DEFAULT_RF_CONFIGS):
        K.seed_everything(seed)
        model = TwoBranchClassifier(len(classes), cfg, iq_width=iq_width,
                                    use_scf=False)
        with K.Timer() as tm:
            fit(model, tr, epochs, val_ds=None, lr=lr, batch_size=batch_size,
                device=dev, seed=seed, keep_best=False, tag="search")
            acc = evaluate_torch(model, va, dev)["accuracy"]
        rfs = receptive_field(cfg["kernel"], cfg["dilations"],
                              cfg.get("blocks", (1, 1, 1, 1)))
        res = {"name": cfg.get("name", ""), "kernel": int(cfg["kernel"]),
               "dilations": list(cfg["dilations"]), "receptive_field": rfs,
               "receptive_field_s": rfs / rate, "val_accuracy": acc,
               "params": int(sum(p.numel() for p in model.iq.parameters())),
               "seconds": round(tm.seconds, 2)}
        results.append(res)
        line = (f"receptive-field search: {res['name']} (kernel {res['kernel']}, "
                f"dilations {res['dilations']}, {rfs} samples = "
                f"{1e3 * rfs / rate:.2f} ms) -> val accuracy {acc:.3f}")
        if run is not None:
            run.log(line)
        elif progress:
            progress(line)
    ranked = sorted(results, key=lambda r: (-(r["val_accuracy"]
                                              if np.isfinite(r["val_accuracy"])
                                              else -1.0), r["receptive_field"]))
    best = ranked[0]
    cfg = next(c for c in (configs or DEFAULT_RF_CONFIGS)
               if c.get("name", "") == best["name"])
    return {"best": dict(cfg), "results": results, "epochs": int(epochs),
            "method": ("RF-Next idea as a small grid search: the IQ branch alone "
                       f"trained {epochs} epochs per configuration, ranked by val "
                       "accuracy, ties to the smaller receptive field")}


# ---------------------------------------------------------------------------
# The classical baseline (cumulants, after Bill's csp/cumulants.py at α = 0)
# ---------------------------------------------------------------------------
def cumulant_features(iq) -> np.ndarray:
    """(N, L) complex -> (N, 7) features: |C20|, |C40|, |C41|, C42 of the
    unit-power signal, the amplitude's coefficient of variation, and the
    instantaneous frequency's standard deviation and kurtosis-like spread."""
    x = np.asarray(iq, np.complex128)
    if x.ndim == 1:
        x = x[None]
    x = x - x.mean(axis=1, keepdims=True)
    p = np.mean(np.abs(x) ** 2, axis=1, keepdims=True)
    x = x / np.sqrt(np.where(p > 0, p, 1.0))
    m20 = np.mean(x * x, axis=1)
    m21 = np.mean(np.abs(x) ** 2, axis=1)
    m40 = np.mean(x ** 4, axis=1)
    m41 = np.mean(x ** 3 * np.conj(x), axis=1)
    m42 = np.mean(np.abs(x) ** 4, axis=1)
    c40 = m40 - 3.0 * m20 * m20
    c41 = m41 - 3.0 * m20 * m21
    c42 = m42 - np.abs(m20) ** 2 - 2.0 * m21 * m21
    amp = np.abs(x)
    cov = amp.std(axis=1) / np.maximum(amp.mean(axis=1), 1e-12)
    dphi = np.angle(x[:, 1:] * np.conj(x[:, :-1]))
    fstd = dphi.std(axis=1)
    fk = np.mean(np.abs(dphi - dphi.mean(axis=1, keepdims=True)) ** 4, axis=1) / \
        np.maximum(fstd ** 4, 1e-12)
    return np.stack([np.abs(m20), np.abs(c40), np.abs(c41), np.real(c42), cov,
                     fstd, np.log1p(fk)], axis=1)


def cumulant_baseline(train_iq, train_y, test_iq, test_y, n_classes: int) -> dict:
    """Nearest centroid on standardised cumulant features: the classical
    automatic-modulation-classification baseline the learned classifier
    must beat (plan §7)."""
    ftr = cumulant_features(train_iq)
    fte = cumulant_features(test_iq)
    mu, sd = ftr.mean(axis=0), ftr.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    ftr, fte = (ftr - mu) / sd, (fte - mu) / sd
    ytr = np.asarray(train_y)
    cents = np.stack([ftr[ytr == c].mean(axis=0) if np.any(ytr == c)
                      else np.full(ftr.shape[1], np.inf) for c in range(n_classes)])
    d = ((fte[:, None, :] - cents[None]) ** 2).sum(axis=2)
    pred = np.argmin(np.nan_to_num(d, nan=np.inf, posinf=np.inf), axis=1)
    return {"method": "nearest centroid on |C20|, |C40|, |C41|, C42, amplitude "
                      "spread, instantaneous-frequency spread (cumulants after "
                      "ATK's csp/cumulants.py, α = 0)",
            "accuracy": K.accuracy(test_y, pred),
            "macro_f1": K.macro_f1(test_y, pred, n_classes),
            "n": int(len(test_y))}


# ---------------------------------------------------------------------------
# train()
# ---------------------------------------------------------------------------
def train(rf, profile: str, dataset_dir, out_name: str, epochs: int = 30, *,
          rf_config="search", search_epochs: int = 3, search_configs=None,
          window: int | None = None, use_scf: bool | None = None,
          held_out_classes=(), iq_width: int = 32, scf_width: int = 16,
          embed_dim: int = 64, hidden: int = 128, cosine_scale: float = 16.0,
          margin: float = 0.15, cycle_weight: float = 1.0,
          batch_size: int = 64, lr: float = 2e-3, weight_decay: float = 1e-4,
          pretrained=None, device: str = "auto", seed: int = 0,
          threads: int | None = None, max_items: int | None = None,
          latency_repeats: int = 50, latency_threads: int | None = None,
          snr_bin_db: float = 5.0, amp: bool = True, num_workers: int = 0,
          prefer_bank: str = "auto", overwrite: bool = False,
          resume_from=None, progress=None) -> Path:
    """Train the classifier for one (profile, canonical class) — the dataset
    fixes both — export it, verify the export, measure it, calibrate it,
    and save it with its card. Returns `rf.models(profile, out_name)`.

    held_out_classes — classes left out of training on purpose; their val
    and test cuts measure unknown rejection (§11).
    rf_config — 'search' (default: `search_receptive_fields`), a config
    dict {kernel, dilations}, or None (k7_d1).
    pretrained — a 1D self-supervised backbone (`ssl.pretrain_1d`) that
    initialises the IQ branch.
    resume_from — a `checkpoints/last.pt` of an earlier run that stopped
    (same settings): training continues from its last finished epoch."""
    threads_now = K.set_threads(threads)
    K.seed_everything(seed)
    dev, dev_words = K.pick_device(device)
    m = K.open_dataset(dataset_dir, profile, "narrowband", rf=rf)
    canonical = m.get("canonical") or {}
    if not canonical.get("rate"):
        raise K.DatasetRefused(f"the dataset {m.get('name')!r} does not say "
                               "which canonical rate its cuts are at "
                               "(manifest 'canonical'); a classifier is per "
                               "canonical class (DETECTION_DESIGN §4).")
    names = list(m.get("classes") or [])
    held = [c for c in held_out_classes]
    for c in held:
        if c not in names:
            raise K.DatasetRefused(f"the held-out class {c!r} is not in the "
                                   f"dataset ({', '.join(names)}).")
    classes = [c for c in names if c not in set(held)]
    if len(classes) < 2:
        raise K.DatasetRefused("a classifier needs at least two classes to "
                               "train on.")
    model_dir = K.new_model_dir(rf, profile, out_name, overwrite=overwrite)
    run = K.Run.start(rf, profile, f"{KIND}_{out_name}",
                      {"dataset": str(dataset_dir), "epochs": epochs,
                       "classes": classes, "held_out": held,
                       "rf_config": rf_config if isinstance(rf_config, (dict, str))
                       else None, "batch_size": batch_size, "lr": lr,
                       "seed": seed}, progress)
    run.log(dev_words)
    run.log(f"{threads_now} CPU threads for PyTorch")
    run.log(f"canonical class {canonical.get('class')} at "
            f"{float(canonical['rate']):g} S/s (decimation "
            f"{canonical.get('decimation')}); classes: {', '.join(classes)}"
            + (f"; held out for unknown rejection: {', '.join(held)}" if held else ""))
    probe = K.ShardSet(dataset_dir, "train", m, max_items=1)
    if use_scf is None:
        use_scf = bool(probe.has_scf)
        if not use_scf:
            run.log("the dataset has no cached SCF: training the IQ branch "
                    "alone (the card says so)")
    search = None
    ssl_arch = None
    if pretrained:
        from atk_diffusion.learn import ssl as _ssl
        ssl_arch = _ssl.backbone_arch(pretrained, profile)
        if "kernel" not in ssl_arch:
            raise _cards.CardRefusal(f"{Path(pretrained).name} is not a 1D "
                                     "backbone; the classifier needs one from "
                                     "ssl.pretrain_1d.")
        iq_width = int(ssl_arch["width"])
        rf_config = {"name": "pretrained", "kernel": int(ssl_arch["kernel"]),
                     "dilations": list(ssl_arch["dilations"]),
                     "blocks": list(ssl_arch.get("blocks", (1, 1, 1, 1)))}
        run.log(f"the IQ branch's architecture is the pretrained backbone's "
                f"(width {iq_width}, kernel {rf_config['kernel']}, dilations "
                f"{rf_config['dilations']}); no receptive-field search")
    if rf_config == "search":
        search = search_receptive_fields(rf, profile, dataset_dir,
                                         search_configs, search_epochs,
                                         window=window, iq_width=min(iq_width, 16),
                                         batch_size=batch_size, lr=lr, seed=seed,
                                         device=device, threads=threads,
                                         held_out_classes=held,
                                         max_items=max_items, run=run)
        cfg = search["best"]
        run.log(f"receptive field chosen: {cfg.get('name')} (kernel "
                f"{cfg['kernel']}, dilations {cfg['dilations']})")
    elif isinstance(rf_config, dict):
        cfg = dict(rf_config)
    else:
        cfg = dict(DEFAULT_RF_CONFIGS[1])
    K.seed_everything(seed)
    tr = D.NarrowbandDataset(dataset_dir, "train", profile, manifest=m,
                             classes=classes, window=window, train=True,
                             use_scf=use_scf, max_items=max_items, seed=seed,
                             known_only=True)
    va = D.NarrowbandDataset(dataset_dir, "val", profile, manifest=m,
                             classes=classes, window=window, use_scf=use_scf,
                             max_items=max_items, known_only=True)
    if len(tr) == 0:
        raise K.DatasetRefused("the training split has no examples of the "
                               "classes to train on.")
    model = TwoBranchClassifier(len(classes), cfg, iq_width, scf_width, use_scf,
                                embed_dim, hidden, cosine_scale)
    if pretrained:
        rep = _ssl.load_backbone_into(model, pretrained, profile=profile,
                                      canonical_rate=float(canonical["rate"]))
        run.log(f"IQ branch initialised from {Path(pretrained).name}: "
                f"{rep['loaded']} tensors loaded")
    nparams = sum(p.numel() for p in model.parameters())
    counts = np.bincount(tr.model_labels(), minlength=len(classes))
    run.log(f"{len(tr)} training cuts of {tr.window} samples, {len(va)} val; "
            f"{nparams:,} parameters; receptive field "
            f"{model.iq.receptive_field()} samples")
    hist = fit(model, tr, epochs, val_ds=va if len(va) else None, lr=lr,
               weight_decay=weight_decay, batch_size=batch_size, margin=margin,
               cycle_weight=cycle_weight, device=dev, seed=seed, run=run,
               amp=amp, num_workers=num_workers, resume_from=resume_from)
    test_split = _eval_split(dataset_dir)
    notes = []
    if pretrained:
        notes.append(f"IQ branch pretrained self-supervised: {Path(pretrained).name}")
    save_trained(rf, profile, model, model_dir, dataset_dir=dataset_dir,
                 manifest=m, classes=classes, train_ds=tr, run=run,
                 train_summary={"epochs": int(epochs), "cuts": len(tr),
                                "final_loss": K.finite_or_none(hist[-1]["loss"])
                                if hist else None, "device": dev_words},
                 counts=counts,
                 datasets=[K.dataset_entry(dataset_dir, m,
                                           ("train", "val") + ((test_split,)
                                                               if test_split else ()))],
                 extra_notes=notes, search=search, test_split=test_split,
                 calibrate_split="val" if len(va) else None,
                 latency_repeats=latency_repeats, latency_threads=latency_threads,
                 snr_bin_db=snr_bin_db, prefer_bank=prefer_bank,
                 trained_on=dev_words, name=out_name, max_items=max_items)
    return model_dir


def _eval_split(dataset_dir, want: str | None = None) -> str | None:
    if want:
        return want
    for sp in ("test", "val"):
        if K.split_files(dataset_dir, sp, "narrowband"):
            return sp
    return None


def save_trained(rf, profile: str, model, model_dir, *, dataset_dir,
                 manifest: dict, classes: list, train_ds, run: K.Run,
                 train_summary: dict, counts, datasets: list, extra_notes=(),
                 search: dict | None = None, test_split: str | None = None,
                 calibrate_split: str | None = "val",
                 latency_repeats: int = 50, latency_threads: int | None = None,
                 snr_bin_db: float = 5.0, prefer_bank: str = "auto",
                 trained_on: str = "", name: str | None = None,
                 max_items: int | None = None) -> Path:
    """Export a trained classifier to ONNX, verify the graph against PyTorch,
    measure latency and held-out scores THROUGH the ONNX (beside the
    cumulant baseline), calibrate it (temperature and the prototype bank on
    `calibrate_split`), and save it with its card. Shared by `train()`,
    `finetune()` and the minutes-to-acceptable experiment."""
    model_dir = Path(model_dir)
    name = name or model_dir.name
    m = manifest
    canonical = m.get("canonical") or {}
    model.to("cpu").eval()
    torch.save(model.state_dict(), model_dir / X.TORCH_FILE)
    use_scf = bool(model.arch["use_scf"])
    window = int(train_ds.window)
    te = None
    if test_split and K.split_files(dataset_dir, test_split, "narrowband"):
        te = D.NarrowbandDataset(dataset_dir, test_split, profile, manifest=m,
                                 classes=classes, window=window, use_scf=use_scf,
                                 max_items=max_items)
    src = te if te is not None and len(te) else train_ds
    sample = D.collate_narrowband([src[i] for i in range(min(3, len(src)))])
    onnx_path = model_dir / X.ONNX_FILE
    X.export(model, (sample["iq"], sample["scf"]), onnx_path, ("iq", "scf"),
             ("logits", "embedding", "cycle"),
             dynamic_axes={"iq": {0: "batch"}, "scf": {0: "batch"},
                           "logits": {0: "batch"}, "embedding": {0: "batch"},
                           "cycle": {0: "batch"}})
    feeds = [{"iq": sample["iq"].numpy(), "scf": sample["scf"].numpy()},
             {"iq": sample["iq"][:1].numpy(), "scf": sample["scf"][:1].numpy()}]
    check = X.verify(onnx_path, model, feeds, rtol=1e-3, atol=1e-4)
    if not check["ok"]:
        run.log(f"REFUSED: the ONNX graph does not reproduce PyTorch — "
                f"{check['why']}")
        for f in (onnx_path, model_dir / X.TORCH_FILE):
            f.unlink(missing_ok=True)
        raise X.ExportMismatch(f"the exported classifier does not reproduce the "
                               f"trained one ({check['why']}); it was not saved "
                               "as a model.")
    run.log(f"ONNX export verified against PyTorch ({check['cases']} batches, "
            f"largest difference {max(check['max_abs_diff'].values()):.2g})")
    lat = X.latency(onnx_path, feeds[1], repeats=latency_repeats,
                    threads=latency_threads)
    run.log(f"CPU latency {lat['p50_ms']:.2f} ms a cutout (median) on "
            f"{lat['machine']}, threads {lat['threads']}; the budget is "
            f"{LATENCY_BUDGET_MS:.0f} ms on Bill's 14-core CPU")
    scf_shape = (list(train_ds.shards.scf_shape)
                 if (use_scf and train_ds.shards.scf_shape) else [1, 1])
    rate = float(canonical["rate"])
    card = _cards.new_card(
        name, KIND, profile,
        input={"canonical": {"class": canonical.get("class"),
                             "cls": canonical.get("class"), "rate": rate,
                             "decimation": canonical.get("decimation")},
               "rate": rate, "iq_len": window, "window": window,
               # the words detect.onnx_models.Classifier1D applies at inference
               "iq_norm": "rms", "scf_norm": train_ds.scf_norm,
               "iq_window": "a window of iq_len samples at the canonical rate "
                            "(training: random offset; scoring: centred), unit RMS",
               "scf_used": use_scf, "scf_shape": scf_shape,
               "fam": m.get("fam") or {},
               "cycle_scale": [float(train_ds.cycle_scale[0]),
                               float(train_ds.cycle_scale[1])],
               "cycle_units": "Hz = cycle output × cycle_scale (symbol rate, "
                              "carrier offset)",
               "embedding_dim": int(model.arch["embed_dim"]),
               "cosine_scale": float(model.arch["cosine_scale"]),
               "receptive_field": {"samples": model.iq.receptive_field(),
                                   "seconds": model.iq.receptive_field() / rate,
                                   "config": model.arch["rf_config"],
                                   "search": search},
               "arch": model.arch},
        classes=[{"name": c, "source": "trained", "examples": int(n)}
                 for c, n in zip(classes, counts)],
        datasets=list(datasets),
        metrics={"latency_ms": lat["p50_ms"], "latency": lat,
                 "latency_budget_ms": LATENCY_BUDGET_MS, "onnx_check": check,
                 "params": int(sum(p.numel() for p in model.parameters())),
                 "train": train_summary},
        license="all rights reserved (see LICENSE)", trained_on=trained_on,
        notes=["logits are cosine similarities × cosine_scale; probabilities "
               "are softmax(logits / temperature) with the temperature in "
               "card.calibration",
               "open set: a cut farther from its nearest prototype than that "
               "class's threshold is UNKNOWN (the prototype bank is saved "
               "beside this card)"] + list(extra_notes))
    if not use_scf:
        card.notes.append("trained without the SCF branch (the dataset had no "
                          "cached SCF): the scf input is accepted and ignored")
    X.save_model(model_dir, card, rf=rf)

    # -- held-out numbers through the ONNX ------------------------------------
    met = dict(card.metrics)
    if te is not None and len(te):
        runner = X.OnnxRunner(model_dir, KIND, for_profile=profile, threads=1)
        out = X.run_classifier(runner, te.shards, te.items)
        y = te.model_labels()
        known = y >= 0
        pred = out["logits"].argmax(axis=1)
        if np.any(known):
            met.update(K.classification_scores(pred[known], y[known], classes,
                                              te.field("snr_db")[known],
                                              snr_bin_db))
            cy = out["cycle"][known] * np.asarray(train_ds.cycle_scale)[None]
            met["cycle_error"] = K.cycle_error(cy, te.field("symbol_rate_hz")[known],
                                             te.field("carrier_offset_hz")[known])
            met["heldout"] = {"dataset": str(m.get("name")), "split": test_split,
                              "cuts": int(known.sum()),
                              "generator": m.get("generator", ""),
                              "label_sources": m.get("label_sources", [])}
            if "cabled" in (m.get("label_sources") or []) or \
                    m.get("generator") == "cabled":
                met["accuracy_cabled"] = met["accuracy"]
            else:
                met["accuracy_synthetic"] = met["accuracy"]
            tr_iq = np.stack([train_ds.shards.iq(int(i)) for i in train_ds.items])
            te_iq = np.stack([te.shards.iq(int(i)) for i in te.items[known]])
            base = cumulant_baseline(tr_iq, train_ds.model_labels(), te_iq,
                                     y[known], len(classes))
            met["classical_baseline"] = base
            met["beats_classical"] = bool(met["accuracy"] > base["accuracy"])
            run.log(f"held-out ({test_split}): accuracy {met['accuracy']:.3f}, "
                    f"macro-F1 {met['macro_f1']:.3f}; the cumulant baseline "
                    f"{base['accuracy']:.3f}")
    else:
        run.log("nothing held out: the card carries no held-out score")
    card.metrics = met
    X.resave_card(model_dir, card, rf=rf)
    if calibrate_split and K.split_files(dataset_dir, calibrate_split, "narrowband"):
        from atk_diffusion.learn import calibrate as _cal
        card = _cal.calibrate_classifier(rf, profile, model_dir, dataset_dir,
                                         split=calibrate_split,
                                         proto_split="train",
                                         test_split=test_split, threads=1,
                                         prefer_bank=prefer_bank)
        t = card.calibration.get("temperature_fit", {})
        osr = card.calibration.get("open_set", {})
        run.log(f"temperature {t.get('temperature', float('nan')):.2f}: "
                f"calibration error {t.get('ece_before', float('nan')):.3f} -> "
                f"{t.get('ece_after', float('nan')):.3f} on {calibrate_split}"
                + (f", {t['ece_heldout_before']:.3f} -> "
                   f"{t['ece_heldout_after']:.3f} on {test_split}"
                   if "ece_heldout_after" in t else ""))
        if osr.get("unknown_rejection") is not None:
            run.log(f"unknown rejection {osr['unknown_rejection']:.3f} on "
                    f"{osr['n_unknown']} cuts of held-out classes "
                    f"({osr['implementation']})")
    else:
        run.log("no calibration split: no temperature and no open-set "
                "thresholds")
    run.finish({"model_dir": str(model_dir),
                "accuracy": card.metrics.get("accuracy"),
                "latency_ms": lat["p50_ms"]})
    return model_dir


def finetune(rf, profile: str, model_dir, dataset_dir, out_name: str,
             epochs: int = 8, *, lr: float = 5e-4, batch_size: int = 64,
             split: str = "train", indices=None, device: str = "auto",
             seed: int = 0, threads: int | None = None,
             test_split: str | None = None, calibrate_split: str | None = "val",
             latency_repeats: int = 50, latency_threads: int | None = None,
             amp: bool = True, prefer_bank: str = "auto",
             overwrite: bool = False, note: str = "", progress=None) -> Path:
    """Fine-tune a saved classifier on a narrowband dataset of the same
    profile and canonical rate — the cabled set (DETECTION_DESIGN §6.3) or
    the first minutes on site (plan §3.6) — and save it as a NEW model,
    measured and calibrated exactly as `train()` does. `indices` limits the
    cuts used (raw indices into the split). The base is never modified."""
    threads_now = K.set_threads(threads)
    K.seed_everything(seed)
    dev, dev_words = K.pick_device(device)
    model, base = load_torch(model_dir, profile)
    m = K.open_dataset(dataset_dir, profile, "narrowband", rf=rf)
    rate = float((m.get("canonical") or {}).get("rate") or 0)
    base_rate = float((base.input.get("canonical") or {}).get("rate") or 0)
    if abs(rate - base_rate) > 0.5:
        raise K.DatasetRefused(f"{base.name} classifies cuts at {base_rate:g} S/s; "
                               f"this dataset's cuts are at {rate:g} S/s. A "
                               "network never meets a rate it was not trained at.")
    classes = base.class_names()
    use_scf = bool(model.arch["use_scf"])
    full = D.NarrowbandDataset(dataset_dir, split, profile, manifest=m,
                               classes=classes, window=int(base.input["iq_len"]),
                               train=True, use_scf=use_scf, seed=seed,
                               known_only=True,
                               scf_norm=str(base.input.get("scf_norm", "max")))
    if indices is not None:
        keep = set(int(i) for i in indices)
        full.items = np.asarray([i for i in full.items if int(i) in keep], np.int64)
    if len(full) == 0:
        raise K.DatasetRefused("there are no cuts of the model's classes to "
                               "fine-tune on")
    new_dir = K.new_model_dir(rf, profile, out_name, overwrite=overwrite)
    run = K.Run.start(rf, profile, f"{KIND}_finetune_{out_name}",
                      {"base": str(model_dir), "dataset": str(dataset_dir),
                       "split": split, "cuts": len(full), "epochs": epochs,
                       "lr": lr, "seed": seed}, progress)
    run.log(dev_words)
    run.log(f"{threads_now} CPU threads for PyTorch")
    run.log(f"fine-tuning {base.name} on {len(full)} cuts of "
            f"{m.get('name')!r} ({split})" + (f" — {note}" if note else ""))
    hist = fit(model, full, epochs, val_ds=None, lr=lr,
               batch_size=min(int(batch_size), max(2, len(full))), device=dev,
               seed=seed, run=run, amp=amp, keep_best=False, tag="finetune")
    counts = np.bincount(full.model_labels(), minlength=len(classes))
    base_counts = {c["name"]: int(c.get("examples", 0)) for c in base.classes
                   if isinstance(c, dict)}
    counts = np.asarray([int(n) + base_counts.get(c, 0)
                         for c, n in zip(classes, counts)])
    ts = _eval_split(dataset_dir, test_split)
    entry = K.dataset_entry(dataset_dir, m, (split,) + ((ts,) if ts else ()))
    entry["cuts_used"] = len(full)
    full.train = False
    save_trained(rf, profile, model, new_dir, dataset_dir=dataset_dir,
                 manifest=m, classes=classes, train_ds=full, run=run,
                 train_summary={"fine_tuned_from": base.name,
                                "base_sha256": base.weights.get("sha256"),
                                "epochs": int(epochs), "cuts": len(full),
                                "lr": float(lr),
                                "final_loss": K.finite_or_none(hist[-1]["loss"])
                                if hist else None, "device": dev_words},
                 counts=counts, datasets=list(base.datasets) + [entry],
                 extra_notes=[f"fine-tuned from {base.name} on {m.get('name')} "
                              f"({len(full)} cuts)" + (f": {note}" if note else "")],
                 search=(base.input.get("receptive_field") or {}).get("search"),
                 test_split=ts, calibrate_split=calibrate_split,
                 latency_repeats=latency_repeats,
                 latency_threads=latency_threads, prefer_bank=prefer_bank,
                 trained_on=dev_words, name=out_name)
    return new_dir


def load_torch(model_dir, profile: str | None = None):
    """(PyTorch model, card) rebuilt from a saved classifier."""
    card = _cards.load(model_dir, expect_kind=KIND, for_profile=profile)
    model = build_from_arch(card.input["arch"])
    model.load_state_dict(X.load_train_state(model_dir, card))
    model.eval()
    return model, card


def gpu_recommendation() -> dict:
    """UNTESTED — what one full run on Bill's RTX 3080 Ti should look like."""
    return {"status": "UNTESTED recommendation (no GPU run has been made)",
            "per": "one model per (profile, canonical class): e.g. "
                   "rtlsdr_2400000_cu8 voice (48 kS/s), wideband (480 kS/s), "
                   "spread (2.4 MS/s)",
            "dataset": "≈2 000 cuts per class per split from TorchSig "
                       "narrowband at the canonical rate, SNR −10…+30 dB, SCF "
                       "cached; one cabled set",
            "model": "iq_width 32, window 2048–4096 samples, receptive-field "
                     "search 5 epochs per config",
            "training": "batch 256, lr 2e-3 one-cycle, 40 epochs, AMP — well "
                        "under an hour per canonical class (estimate)",
            "then": "calibrate on the cabled set; hold out two classes to "
                    "measure unknown rejection; latency on the 14-core CPU "
                    "(budget 20 ms a cutout)"}
