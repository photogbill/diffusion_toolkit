# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The 2D AI proposer: torchvision FCOS on the spectrogram tile
(DETECTION_DESIGN §3 "The learned detector", §6, §7; decision D3).

An anchor-free single-stage detector that puts boxes (time × frequency) on a
tile with a coarse FAMILY (FM-like · AM-like · FSK · PSK/QAM · OFDM · burst
· spread · unknown — `detect.classes.FAMILIES`) and a confidence. Family,
not modulation: §3 — *"the spectrogram cannot tell QPSK from 8PSK and
should not pretend to; the IQ classifier does that."*

WHY THESE CHOICES
* **FCOS from torchvision (BSD-3)**, decision D3, confirmed by Bill
  2026-10-08 (*"whichever permissive model you recommend in place of
  YOLO"*). Ultralytics YOLO is excluded (AGPL-3, §8) — not used, not
  vendored, not called.
* **A thin ResNet-18-class backbone** (the ResNet-18 layout — a 7×7/2 stem,
  a max-pool, four stages of two basic blocks — at a fraction of the
  channels, `width` 16 → 16/32/64/128 by default) with an FPN and FCOS's
  P6/P7 levels, single-channel input. Thin because inference is on the CPU
  inside ATK (§7, D4): the budget is 200 ms a tile on the 14-core machine,
  and it is measured into the card, not assumed.
* **Tiles are never rescaled.** torchvision's detection transform resizes
  every image to a `min_size`; a spectrogram's pixels ARE its geometry
  (§2: bins are Hz, rows are seconds), so the transform here is FCOS's own
  with the resize removed (`NoResizeTransform`, which also refuses a tile
  of the wrong shape) and min/max size set to the tile's sides. Its mean
  and standard deviation are the dataset's (`common.TileNorm`), and the clip
  to [lo, hi] dB is the first op of the exported graph — so the ONNX takes
  raw dB above the floor and the host cannot get the scaling wrong. The
  card says so in the words `detect.onnx_models` reads: `input.normalize`
  is "none" (the host feeds dB as it is) and `input.graph_normalize`
  records what the graph does inside.
* **Thin boxes are rescued.** FCOS learns a box from the feature-map points
  inside it; a voice channel 3–5 bins wide may contain none at the level
  its size maps to. Boxes are widened to `min_box_px` (common.BoxPolicy) and
  any box the standard assignment still leaves without a positive point is
  given the finest level's points inside it (`SignalFCOSHead`). The run log
  says how many were rescued. If thin boxes still score badly, the design's
  next step is RT-DETR (Apache 2.0, §3) — measured, not assumed.
* **Postprocessing is inside the graph.** Box decoding, the score floor,
  per-class NMS and the top-K are exported with the network (the tracing
  exporter handles torchvision's NMS), so the ONNX outputs exactly
  ARCHITECTURE §5: `boxes` [K, 4] (row0, bin0, row1, bin1 in tile pixels),
  `scores` [K], `labels` int64 [K] (index into `card.input["families"]`).
  The export is verified against PyTorch on held-out tiles before it is
  saved.

WHAT `train()` WRITES — `rf_data\\<profile>\\models\\<name>\\`: `model.onnx`
(the card's weights), `model.pt` (the PyTorch state, for fine-tuning on the
cabled set or on site), `card.json` (kind `proposer2d`), and a run folder
under `runs\\` with the log, per-epoch metrics and checkpoints. The card's
numbers are measured through the exported ONNX on held-out tiles: mAP@0.5
(family-aware), AP@0.5 with families ignored (*where*), mAP@0.5:0.95, the
operating threshold (`calibration.min_score`, the best F1 on `val` — the
score below which `detect.onnx_models.Proposer2D` drops a box) and Platt
calibration (`calibration.platt`), and CPU latency.

`finetune()` takes a saved proposer to the cabled set (DETECTION_DESIGN
§6.3) or to the first minutes on site (plan §3.6) and saves a NEW model
measured the same way (`save_trained`, shared by both); the base model is
never modified.

LIMITS, stated. Trained here only at test scale on the CPU; the numbers a
test produces say the code path works, not that a model is good. The
mAP on synthetic tiles is not the number that matters — the domain gap on
cabled captures is (`experiments.domain_gap`).
"""

from __future__ import annotations

import math
from collections import OrderedDict
from functools import partial
from pathlib import Path

import numpy as np

from atk_diffusion import cards as _cards
from atk_diffusion.detect import classes as _classes
from atk_diffusion.learn import common as K

torch = K.require_torch()
from torch import nn  # noqa: E402
from torchvision.models.detection import FCOS  # noqa: E402
from torchvision.models.detection.anchor_utils import AnchorGenerator  # noqa: E402
from torchvision.models.detection.fcos import (  # noqa: E402
    FCOSClassificationHead, FCOSHead, FCOSRegressionHead)
from torchvision.models.detection.transform import GeneralizedRCNNTransform  # noqa: E402
from torchvision.ops.feature_pyramid_network import (  # noqa: E402
    FeaturePyramidNetwork, LastLevelP6P7)

from atk_diffusion.learn import data as D  # noqa: E402
from atk_diffusion.learn import export as X  # noqa: E402

KIND = "proposer2d"
LATENCY_BUDGET_MS = 200.0           # DETECTION_DESIGN §7, Bill's 14-core CPU


# ---------------------------------------------------------------------------
# The backbone (shared with ssl.py's masked-spectrogram pretraining)
# ---------------------------------------------------------------------------
class BasicBlock2d(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                      nn.BatchNorm2d(cout))

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(y + idt)


class ThinResNet2d(nn.Module):
    """The ResNet-18 layout at `width` channels in the first stage (64 in the
    original): stem 7×7/2 + max-pool /2, stages at strides 4, 8, 16, 32."""

    def __init__(self, in_ch: int = 1, width: int = 16, blocks=(2, 2, 2, 2)):
        super().__init__()
        w = int(width)
        self.widths = (w, 2 * w, 4 * w, 8 * w)
        self.in_ch = int(in_ch)
        self.blocks = tuple(int(b) for b in blocks)
        self.conv1 = nn.Conv2d(self.in_ch, w, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(w)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        cin = w
        stages = []
        for i, (c, n) in enumerate(zip(self.widths, self.blocks)):
            layers = [BasicBlock2d(cin, c, 1 if i == 0 else 2)]
            layers += [BasicBlock2d(c, c, 1) for _ in range(n - 1)]
            stages.append(nn.Sequential(*layers))
            cin = c
        self.layer1, self.layer2, self.layer3, self.layer4 = stages
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")

    def arch(self) -> dict:
        return {"name": "thin_resnet18_2d", "in_ch": self.in_ch,
                "width": self.widths[0], "blocks": list(self.blocks)}

    def forward(self, x) -> list:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [c1, c2, c3, c4]


class ProposerBackbone(nn.Module):
    """ThinResNet2d + FPN + P6/P7: levels P{finest}…P7 at strides
    2**finest … 128 (finest 3: P3–P7 as torchvision's FCOS; finest 2 adds a
    stride-4 level for very thin signals, at a CPU cost)."""

    def __init__(self, body: ThinResNet2d, fpn_channels: int = 64,
                 finest_level: int = 3):
        super().__init__()
        if finest_level not in (2, 3):
            raise ValueError("finest_level is 2 (stride 4) or 3 (stride 8)")
        self.body = body
        self.returned = [s for s in (1, 2, 3, 4) if s + 1 >= finest_level]
        self.fpn = FeaturePyramidNetwork(
            [body.widths[s - 1] for s in self.returned], int(fpn_channels),
            extra_blocks=LastLevelP6P7(int(fpn_channels), int(fpn_channels)))
        self.out_channels = int(fpn_channels)

    def forward(self, x):
        feats = self.body(x)
        od = OrderedDict((str(i), feats[s - 1]) for i, s in enumerate(self.returned))
        return self.fpn(od)


class NoResizeTransform(GeneralizedRCNNTransform):
    """FCOS's transform without the resize: normalise (mean/std from the
    dataset) and pad to a multiple of 32, nothing else. A tile of any other
    shape than the one trained on is refused."""

    def __init__(self, tile_shape, mean: float, std: float):
        super().__init__(min(tile_shape), max(tile_shape), [float(mean)],
                         [float(std)], size_divisible=32)
        self.tile_shape = (int(tile_shape[0]), int(tile_shape[1]))

    def resize(self, image, target=None):
        got = (int(image.shape[-2]), int(image.shape[-1]))
        if got != self.tile_shape:
            raise ValueError(f"this proposer was trained on tiles of "
                             f"{self.tile_shape[0]}×{self.tile_shape[1]}; this "
                             f"tile is {got[0]}×{got[1]}. A model never meets a "
                             "spectrogram of a geometry it was not trained on.")
        return image, target


def rescue_unassigned(gt_boxes, anchors, matched):
    """Give every ground-truth box that FCOS's assignment left with NO
    positive point the finest level's unassigned points inside it.
    Returns (matched, number of boxes rescued)."""
    if gt_boxes.numel() == 0:
        return matched, 0
    M = gt_boxes.shape[0]
    has = torch.zeros(M, dtype=torch.bool, device=matched.device)
    pos = matched[matched >= 0]
    if pos.numel():
        has[pos.unique()] = True
    missing = torch.nonzero(~has).flatten()
    if missing.numel() == 0:
        return matched, 0
    sizes = anchors[:, 2] - anchors[:, 0]
    finest = sizes <= sizes.min() + 1e-6
    cx = 0.5 * (anchors[:, 0] + anchors[:, 2])
    cy = 0.5 * (anchors[:, 1] + anchors[:, 3])
    out = matched.clone()
    n = 0
    for j in missing.tolist():
        b = gt_boxes[j]
        inside = (finest & (cx > b[0]) & (cx < b[2]) & (cy > b[1]) & (cy < b[3])
                  & (out < 0))
        if bool(inside.any()):
            out[inside] = j
            n += 1
    return out, n


def _groups(c: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if c % g == 0 and c // g >= 2:
            return g
    return 1


class SignalFCOSHead(FCOSHead):
    """FCOS's head (classification, regression, centre-ness) with group
    norms sized for thin channels and the thin-box rescue in the loss."""

    def __init__(self, in_channels: int, num_classes: int, num_convs: int = 2):
        super().__init__(in_channels, 1, num_classes, num_convs)
        norm = partial(nn.GroupNorm, _groups(in_channels))
        self.classification_head = FCOSClassificationHead(
            in_channels, 1, num_classes, num_convs, norm_layer=norm)
        self.regression_head = FCOSRegressionHead(in_channels, 1, num_convs,
                                                  norm_layer=norm)
        self.rescued_boxes = 0

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        fixed = []
        for t, a, m in zip(targets, anchors, matched_idxs):
            m2, n = rescue_unassigned(t["boxes"], a, m)
            self.rescued_boxes += n
            fixed.append(m2)
        return super().compute_loss(targets, head_outputs, anchors, fixed)


def build_model(families=_classes.FAMILIES, tile_shape=(512, 1024),
                norm: K.TileNorm | None = None, width: int = 16,
                fpn_channels: int = 64, head_convs: int = 2,
                finest_level: int = 3, score_thresh: float = 0.05,
                nms_thresh: float = 0.5, detections_per_img: int = 100,
                topk_candidates: int = 1000,
                center_sampling_radius: float = 1.5):
    """The proposer, untrained. `model.arch` holds everything needed to
    build it again (it goes into the card)."""
    norm = norm or K.TileNorm()
    fams = list(families)
    body = ThinResNet2d(1, width)
    backbone = ProposerBackbone(body, fpn_channels, finest_level)
    n_levels = 7 - int(finest_level) + 1
    sizes = tuple((2 ** (int(finest_level) + i),) for i in range(n_levels))
    anchors = AnchorGenerator(sizes, ((1.0,),) * n_levels)
    head = SignalFCOSHead(int(fpn_channels), len(fams), int(head_convs))
    model = FCOS(backbone, num_classes=len(fams), min_size=min(tile_shape),
                 max_size=max(tile_shape), image_mean=[norm.mean_db],
                 image_std=[norm.std_db], anchor_generator=anchors, head=head,
                 center_sampling_radius=float(center_sampling_radius),
                 score_thresh=float(score_thresh), nms_thresh=float(nms_thresh),
                 detections_per_img=int(detections_per_img),
                 topk_candidates=int(topk_candidates))
    model.transform = NoResizeTransform(tile_shape, norm.mean_db, norm.std_db)
    model.arch = {"name": "fcos_thin_resnet18_fpn", "families": fams,
                  "tile_shape": [int(tile_shape[0]), int(tile_shape[1])],
                  "norm": norm.to_json(), "width": int(width),
                  "fpn_channels": int(fpn_channels),
                  "head_convs": int(head_convs),
                  "finest_level": int(finest_level),
                  "strides": [s[0] for s in sizes],
                  "score_thresh": float(score_thresh),
                  "nms_thresh": float(nms_thresh),
                  "detections_per_img": int(detections_per_img),
                  "topk_candidates": int(topk_candidates),
                  "center_sampling_radius": float(center_sampling_radius)}
    return model


def build_from_arch(arch: dict):
    a = dict(arch)
    return build_model(a["families"], tuple(a["tile_shape"]),
                       K.TileNorm.from_json(a["norm"]), a["width"],
                       a["fpn_channels"], a["head_convs"], a["finest_level"],
                       a["score_thresh"], a["nms_thresh"],
                       a["detections_per_img"], a["topk_candidates"],
                       a.get("center_sampling_radius", 1.5))


class ProposerExport(nn.Module):
    """What the ONNX holds: clip → FCOS (normalise, network, decode, NMS) →
    boxes reordered to (row0, bin0, row1, bin1)."""

    def __init__(self, model, norm: K.TileNorm):
        super().__init__()
        self.model = model
        self.lo = float(norm.clip_lo_db)
        self.hi = float(norm.clip_hi_db)

    def forward(self, tile):
        x = torch.clamp(tile[0], self.lo, self.hi)
        det = self.model([x])[0]
        b = det["boxes"]
        boxes = torch.stack([b[:, 1], b[:, 0], b[:, 3], b[:, 2]], dim=1)
        return boxes, det["scores"], det["labels"]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _schedule(opt, total_steps: int, warmup_frac: float = 0.1,
              floor: float = 0.05):
    warm = max(1, int(total_steps * warmup_frac))

    def f(step):
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, total_steps - warm)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def fit(model, train_ds, epochs: int, *, lr: float = 1e-3,
        weight_decay: float = 1e-4, batch_size: int = 4, device=None,
        seed: int = 0, run: K.Run | None = None, amp: bool = True,
        num_workers: int = 0, grad_clip: float = 10.0,
        checkpoint_every: int = 1, resume_from=None,
        tag: str = "train") -> list[dict]:
    """Train (or fine-tune) `model` on `train_ds` for `epochs`; returns the
    per-epoch history. Used by `train()`, `finetune()` and the
    minutes-to-acceptable experiment's on-site fine-tune.

    With a run, the model, optimiser, schedule and history are written to
    the run's `checkpoints/last.pt` every `checkpoint_every` epochs;
    `resume_from` (such a file) continues a run that stopped — a long GPU
    run is not lost to a crash or a reboot."""
    device = device or torch.device("cpu")
    model.to(device)
    model.train()
    dl = D.loader(train_ds, batch_size, train=True, seed=seed,
                  num_workers=num_workers)
    steps = max(1, len(dl) * int(epochs))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=float(lr), weight_decay=float(weight_decay))
    sched = _schedule(opt, steps)
    use_amp = bool(amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    hist, start = _resume(resume_from, model, opt, sched, run, device)
    for ep in range(start, int(epochs) + 1):
        sums: dict[str, float] = {}
        nb = 0
        model.head.rescued_boxes = 0
        with K.Timer() as tm:
            for imgs, tgts in dl:
                imgs = [im.to(device) for im in imgs]
                tg = [{"boxes": t["boxes"].to(device),
                       "labels": t["labels"].to(device)} for t in tgts]
                with torch.autocast(device.type, dtype=torch.float16,
                                    enabled=use_amp):
                    losses = model(imgs, tg)
                    loss = sum(losses.values())
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"the training loss became {float(loss)} in epoch {ep}; "
                        "lower the learning rate (lr) and try again.")
                opt.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    opt.step()
                sched.step()
                for k, v in losses.items():
                    sums[k] = sums.get(k, 0.0) + float(v.detach())
                sums["loss"] = sums.get("loss", 0.0) + float(loss.detach())
                nb += 1
        rec = {"epoch": ep, "split": tag, "batches": nb,
               "seconds": round(tm.seconds, 3),
               "rescued_boxes": int(model.head.rescued_boxes),
               "lr": float(opt.param_groups[0]["lr"])}
        rec.update({k: v / max(1, nb) for k, v in sums.items()})
        hist.append(rec)
        if run is not None:
            run.metric(**rec)
            run.log(f"{tag} epoch {ep}/{epochs}: loss {rec['loss']:.3f} "
                    f"(classification {rec.get('classification', 0):.3f}, box "
                    f"{rec.get('bbox_regression', 0):.3f}, centre "
                    f"{rec.get('bbox_ctrness', 0):.3f}); "
                    f"{rec['rescued_boxes']} thin boxes rescued; "
                    f"{tm.seconds:.1f} s")
            if checkpoint_every and ep % int(checkpoint_every) == 0:
                run.save_checkpoint({"model": model.state_dict(),
                                     "optimizer": opt.state_dict(),
                                     "scheduler": sched.state_dict(),
                                     "epoch": ep, "epochs": int(epochs),
                                     "history": hist}, "last")
    model.eval()
    return hist


def _resume(path, model, opt, sched, run, device):
    """(history, first epoch) — from a checkpoint when one is given."""
    if not path:
        return [], 1
    ck = torch.load(path, map_location=device, weights_only=True)
    for k in ("model", "optimizer", "scheduler", "epoch"):
        if k not in ck:
            raise ValueError(f"{Path(path).name} is not a training checkpoint "
                             f"of this toolkit (no {k!r}).")
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optimizer"])
    sched.load_state_dict(ck["scheduler"])
    if run is not None:
        run.log(f"resumed from {Path(path).name}: epoch {ck['epoch']} done, "
                "continuing")
    return list(ck.get("history") or []), int(ck["epoch"]) + 1


@torch.no_grad()
def predict(model, ds, device=None, batch_size: int = 8) -> list[dict]:
    """PyTorch predictions for every tile of a WidebandTiles, boxes in
    (row0, bin0, row1, bin1) — the same layout as the ONNX outputs."""
    device = device or torch.device("cpu")
    model.to(device).eval()
    out = []
    for s in range(0, len(ds), batch_size):
        imgs = [ds[i][0].to(device) for i in range(s, min(len(ds), s + batch_size))]
        for det in model(imgs):
            b = det["boxes"].cpu().numpy()
            out.append({"boxes": b[:, [1, 0, 3, 2]] if len(b) else b.reshape(0, 4),
                        "scores": det["scores"].cpu().numpy(),
                        "labels": det["labels"].cpu().numpy()})
    return out


def score(preds, gts, families) -> dict:
    """The proposer's scores on one set: family-aware mAP@0.5 and
    @0.5:0.95, class-agnostic AP@0.5 (*where*), AP per family."""
    fam = K.detection_map(preds, gts, len(families))
    anyc = K.detection_map(preds, gts, 1, class_agnostic=True)
    return {"map50": K.finite_or_none(fam["map50"]),
            "map50_95": K.finite_or_none(fam["map50_95"]),
            "ap50_any": K.finite_or_none(anyc["map50"]),
            "ap50_95_any": K.finite_or_none(anyc["map50_95"]),
            "ap50_per_family": {families[c]: K.finite_or_none(v)
                                for c, v in fam["ap50"].items()
                                if np.isfinite(v)},
            "tiles": len(gts),
            "boxes": int(sum(len(g["boxes"]) for g in gts))}


def _compare_detections(floor: float):
    """verify() comparator: boxes that clear the graph's score floor by a
    margin must agree in number, order and value (a box sitting exactly on
    the floor may legitimately fall either side in float arithmetic)."""
    def cmp(ref, got):
        rb, rs, rl = ref
        gb, gs, gl = got
        rk = rs >= floor + 1e-3
        gk = gs >= floor + 1e-3
        if int(rk.sum()) != int(gk.sum()):
            return False, {}, (f"PyTorch kept {int(rk.sum())} boxes and ONNX "
                               f"Runtime {int(gk.sum())}")
        if not rk.any():
            return True, {"boxes": 0.0, "scores": 0.0}, ""
        db = float(np.max(np.abs(rb[rk] - gb[gk])))
        ds_ = float(np.max(np.abs(rs[rk] - gs[gk])))
        ok = (np.allclose(rb[rk], gb[gk], rtol=1e-3, atol=1e-2)
              and np.allclose(rs[rk], gs[gk], rtol=1e-3, atol=1e-4)
              and np.array_equal(rl[rk], gl[gk]))
        return ok, {"boxes": db, "scores": ds_}, ("" if ok else
                                                  f"boxes differ by {db:.3g} px, "
                                                  f"scores by {ds_:.3g}")
    return cmp


def verify_export(path, wrapper, tiles: list[np.ndarray], floor: float) -> dict:
    """PyTorch vs ONNX Runtime on the same tiles, detection-aware."""
    sess = X.session(path, 1)
    cmp = _compare_detections(floor)
    worst = {"boxes": 0.0, "scores": 0.0}
    for i, t in enumerate(tiles):
        x = np.asarray(t, np.float32)[None, None]
        with torch.no_grad():
            ref = [v.cpu().numpy() for v in wrapper(torch.from_numpy(x))]
        got = sess.run(None, {"tile": x})
        ok, diff, why = cmp(ref, got)
        for k, v in diff.items():
            worst[k] = max(worst.get(k, 0.0), v)
        if not ok:
            return {"ok": False, "cases": i + 1, "max_abs_diff": worst,
                    "why": f"tile {i}: {why}"}
    return {"ok": True, "cases": len(tiles), "max_abs_diff": worst, "why": ""}


def train(rf, profile: str, dataset_dir, out_name: str, epochs: int = 24, *,
          batch_size: int = 8, lr: float = 1e-3, weight_decay: float = 1e-4,
          width: int = 16, fpn_channels: int = 64, head_convs: int = 2,
          finest_level: int = 3, min_box_px: float | None = None,
          negatives: str = "background", pretrained=None,
          device: str = "auto", seed: int = 0, threads: int | None = None,
          max_tiles: int | None = None, eval_split: str | None = None,
          score_thresh: float = 0.05, nms_thresh: float = 0.5,
          detections_per_img: int = 100, latency_repeats: int = 20,
          latency_threads: int | None = None, amp: bool = True,
          num_workers: int = 0, overwrite: bool = False,
          resume_from=None, progress=None) -> Path:
    """Train the 2D proposer on a wideband dataset of `profile`, export it to
    ONNX, verify the export, measure it, and save it with its card.

    Returns `rf.models(profile, out_name)`. Raises ProfileMismatch / a plain
    DatasetRefused for a dataset of another profile or kind, FileExistsError
    for a model name already taken (unless `overwrite`), ExportMismatch when
    the ONNX graph does not reproduce PyTorch.

    `pretrained` — a self-supervised backbone folder (`ssl.pretrain_2d`): its
    weights initialise the backbone, and its tile normalisation is adopted
    so the backbone sees what it was pretrained on.
    `resume_from` — a `checkpoints/last.pt` of an earlier run that stopped
    (same settings): training continues from its last finished epoch."""
    threads_now = K.set_threads(threads)
    K.seed_everything(seed)
    dev, dev_words = K.pick_device(device)
    manifest = K.open_dataset(dataset_dir, profile, "wideband", rf=rf)
    model_dir = K.new_model_dir(rf, profile, out_name, overwrite=overwrite)
    run = K.Run.start(rf, profile, f"{KIND}_{out_name}",
                      {"dataset": str(dataset_dir), "epochs": epochs,
                       "batch_size": batch_size, "lr": lr, "width": width,
                       "fpn_channels": fpn_channels, "head_convs": head_convs,
                       "finest_level": finest_level, "min_box_px": min_box_px,
                       "negatives": negatives, "seed": seed,
                       "pretrained": str(pretrained) if pretrained else None},
                      progress)
    run.log(dev_words)
    run.log(f"{threads_now} CPU threads for PyTorch")
    stride0 = 2 ** int(finest_level)
    policy = K.BoxPolicy(tuple(_classes.FAMILIES),
                         float(min_box_px if min_box_px is not None
                               else stride0 + 2), negatives)
    train_ds = D.WidebandTiles(dataset_dir, "train", profile, manifest=manifest,
                               policy=policy, train=True, max_tiles=max_tiles,
                               seed=seed)
    if len(train_ds) == 0:
        raise K.DatasetRefused(f"the dataset {manifest.get('name')!r} has no "
                               "training tiles.")
    rows, bins = train_ds.shape
    stft = manifest.get("stft") or {}
    if stft.get("tile_rows") and int(stft["tile_rows"]) != rows:
        run.log(f"note: the manifest's STFT geometry says {stft['tile_rows']} "
                f"rows a tile; the tiles have {rows}. The card records both.")
    ssl_card = None
    if pretrained:
        ssl_card = _cards.load(pretrained, expect_kind="ssl_backbone",
                               for_profile=profile)
        if (ssl_card.input or {}).get("branch") != "2d":
            raise _cards.CardRefusal(f"{ssl_card.name} is not a 2D backbone; the "
                                     "proposer needs one from ssl.pretrain_2d.")
        w = int(((ssl_card.input or {}).get("arch") or {}).get("width", width))
        if w != int(width):
            run.log(f"width {w} adopted from the pretrained backbone "
                    f"(asked for {width})")
            width = w
    if ssl_card is not None and (ssl_card.input or {}).get("normalize"):
        norm = K.TileNorm.from_json(ssl_card.input["normalize"])
        norm.applied_by = "graph"
        run.log("the tile normalisation is the pretrained backbone's "
                f"(mean {norm.mean_db:.2f} dB, std {norm.std_db:.2f} dB)")
    else:
        norm = K.TileNorm.fit(train_ds.tiles, seed=seed)
        run.log(f"tile normalisation from the training tiles: clip to "
                f"[{norm.clip_lo_db:g}, {norm.clip_hi_db:g}] dB, mean "
                f"{norm.mean_db:.2f} dB, std {norm.std_db:.2f} dB")
    train_ds.norm = norm
    model = build_model(policy.families, (rows, bins), norm, width,
                        fpn_channels, head_convs, finest_level, score_thresh,
                        nms_thresh, detections_per_img)
    if pretrained:
        from atk_diffusion.learn import ssl as _ssl
        rep = _ssl.load_backbone_into(model, pretrained, profile=profile)
        run.log(f"backbone initialised from {Path(pretrained).name}: "
                f"{rep['loaded']} tensors loaded")
    nparams = sum(p.numel() for p in model.parameters())
    counts = np.zeros(len(policy.families), np.int64)
    for i in range(len(train_ds)):
        _b, lab = train_ds.target_rc(i)
        for v in lab:
            counts[int(v)] += 1
    run.log(f"{len(train_ds)} training tiles of {rows}×{bins}, "
            f"{int(counts.sum())} boxes; {nparams:,} parameters")
    hist = fit(model, train_ds, epochs, lr=lr, weight_decay=weight_decay,
               batch_size=batch_size, device=dev, seed=seed, run=run, amp=amp,
               num_workers=num_workers, resume_from=resume_from)
    split = _eval_split(dataset_dir, eval_split)
    notes = []
    if ssl_card is not None:
        notes.append(f"backbone pretrained self-supervised: {ssl_card.name} "
                     f"({ssl_card.datasets[0]['name'] if ssl_card.datasets else 'captures'})")
    save_trained(rf, profile, model, model_dir, dataset_dir=dataset_dir,
                 manifest=manifest, policy=policy, run=run,
                 train_summary={"epochs": int(epochs), "tiles": len(train_ds),
                                "final_loss": K.finite_or_none(hist[-1]["loss"])
                                if hist else None,
                                "rescued_boxes_last_epoch":
                                    int(hist[-1]["rescued_boxes"]) if hist else 0,
                                "device": dev_words},
                 counts=counts,
                 datasets=[K.dataset_entry(dataset_dir, manifest,
                                           ("train",) + ((split,) if split else ()))],
                 extra_notes=notes, eval_split=split, seed=seed,
                 latency_repeats=latency_repeats,
                 latency_threads=latency_threads, trained_on=dev_words,
                 name=out_name)
    return model_dir


def _eval_split(dataset_dir, eval_split: str | None) -> str | None:
    """The held-out split to score on: the one asked for, else test, else
    val, else None (nothing held out — the card then says so)."""
    if eval_split:
        return eval_split
    for sp in ("test", "val"):
        if K.split_files(dataset_dir, sp, "wideband"):
            return sp
    return None


def save_trained(rf, profile: str, model, model_dir, *, dataset_dir,
                 manifest: dict, policy: K.BoxPolicy, run: K.Run,
                 train_summary: dict, counts, datasets: list,
                 extra_notes=(), eval_split: str | None = None, seed: int = 0,
                 latency_repeats: int = 20, latency_threads: int | None = None,
                 trained_on: str = "", name: str | None = None) -> Path:
    """Export a trained proposer to ONNX, verify the graph against PyTorch,
    measure its CPU latency and its held-out scores THROUGH the ONNX, fit its
    calibration on `val`, and save it with its card. Shared by `train()`,
    `finetune()` and the minutes-to-acceptable experiment."""
    model_dir = Path(model_dir)
    name = name or model_dir.name
    model.to("cpu").eval()
    torch.save(model.state_dict(), model_dir / X.TORCH_FILE)
    arch = model.arch
    norm = K.TileNorm.from_json(arch["norm"])
    rows, bins = (int(v) for v in arch["tile_shape"])
    score_thresh = float(arch["score_thresh"])
    stft = manifest.get("stft") or {}
    eval_ds = None
    if eval_split and K.split_files(dataset_dir, eval_split, "wideband"):
        eval_ds = D.WidebandTiles(dataset_dir, eval_split, profile,
                                  manifest=manifest, policy=policy, norm=norm,
                                  train=False)
    wrapper = ProposerExport(model, norm).eval()
    sample = ([eval_ds.tiles.spec(i) for i in range(min(3, len(eval_ds)))]
              if eval_ds is not None else [])
    if not sample:
        sample = [K.TileSet(dataset_dir, "train", manifest, max_tiles=1).spec(0)]
    rng = np.random.default_rng(seed)
    sample.append((10 * np.log10(rng.exponential(1.0, (rows, bins)))
                   ).astype(np.float32))
    onnx_path = model_dir / X.ONNX_FILE
    X.export(wrapper, (torch.from_numpy(sample[0][None, None]),), onnx_path,
             ("tile",), ("boxes", "scores", "labels"))
    check = verify_export(onnx_path, wrapper, sample, score_thresh)
    if not check["ok"]:
        run.log(f"REFUSED: the ONNX graph does not reproduce PyTorch — "
                f"{check['why']}")
        for f in (onnx_path, model_dir / X.TORCH_FILE):
            f.unlink(missing_ok=True)
        raise X.ExportMismatch(f"the exported proposer does not reproduce the "
                               f"trained one ({check['why']}); it was not "
                               "saved as a model.")
    run.log(f"ONNX export verified against PyTorch on {check['cases']} tiles "
            f"(largest box difference {check['max_abs_diff']['boxes']:.2g} px)")
    lat = X.latency(onnx_path, {"tile": sample[0][None, None]},
                    repeats=latency_repeats, threads=latency_threads)
    run.log(f"CPU latency {lat['p50_ms']:.1f} ms a tile (median; p95 "
            f"{lat['p95_ms']:.1f} ms) on {lat['machine']}, threads "
            f"{lat['threads']}; the budget is {LATENCY_BUDGET_MS:.0f} ms on "
            "Bill's 14-core CPU")
    counts = np.asarray(counts)
    card = _cards.new_card(
        name, KIND, profile,
        input={"tile": {"shape": [1, 1, rows, bins], "rows": rows, "bins": bins},
               "stft": stft,
               # the HOST normalises nothing (detect.onnx_models reads this);
               # the graph's own clip and mean/std are recorded beside it
               "normalize": "none", "graph_normalize": norm.to_json(),
               "families": list(policy.families),
               "boxes": {**policy.to_json(),
                         "order": ["row0", "bin0", "row1", "bin1"],
                         "units": "tile pixels; row1/bin1 one past the last "
                                  "row/bin covered"},
               "graph_score_floor": score_thresh,
               "nms_iou": float(arch["nms_thresh"]),
               "max_detections": int(arch["detections_per_img"]),
               "host_feeds": "float32 [1, 1, rows, bins], dB above the "
                             "measured floor; the graph clips and normalises",
               "arch": arch},
        classes=[{"name": f, "source": "trained", "examples": int(c)}
                 for f, c in zip(policy.families, counts) if c > 0],
        datasets=list(datasets),
        metrics={"latency_ms": lat["p50_ms"], "latency": lat,
                 "latency_budget_ms": LATENCY_BUDGET_MS, "onnx_check": check,
                 "params": int(sum(p.numel() for p in model.parameters())),
                 "train": train_summary},
        license="all rights reserved (see LICENSE); torchvision FCOS (BSD-3)",
        trained_on=trained_on,
        notes=[f"boxes narrower or shorter than {policy.min_box_px:g} px are "
               "widened to it for training and scoring; the cut's classical "
               "measurement gives the true bandwidth",
               f"explicit negatives (noise, spur, DC spike) are "
               f"{'background — the AI proposer learns not to box them' if policy.negatives == 'background' else 'kept as family unknown'}",
               "confidence is calibrated by Platt scaling on the val split "
               "(card.calibration.platt); the operating threshold is the best "
               "F1 there (card.calibration.min_score)",
               "Ultralytics YOLO is excluded (AGPL-3); this is torchvision "
               "FCOS (BSD-3)"] + list(extra_notes))
    X.save_model(model_dir, card, rf=rf)
    # -- held-out numbers, through the ONNX that ships ----------------------
    preds = gts = None
    if eval_ds is not None and len(eval_ds):
        runner = X.OnnxRunner(model_dir, KIND, for_profile=profile, threads=1)
        gts, _s, _n = K.proposer_truth(eval_ds.tiles, manifest, policy)
        preds = X.run_proposer(runner, eval_ds.tiles)
        sc = score(preds, gts, policy.families)
        met = dict(card.metrics)
        met.update(sc)
        met["heldout"] = {"dataset": str(manifest.get("name")),
                          "split": eval_split,
                          "generator": manifest.get("generator", ""),
                          "label_sources": manifest.get("label_sources", [])}
        if "cabled" not in (manifest.get("label_sources") or []) and \
                manifest.get("generator") != "cabled":
            met["map_synthetic"] = sc["map50"]
        else:
            met["map_cabled"] = sc["map50"]
        card.metrics = met
        X.resave_card(model_dir, card, rf=rf)
        run.log(f"held-out ({eval_split}): mAP@0.5 "
                f"{sc['map50'] if sc['map50'] is not None else float('nan'):.3f}, "
                f"AP@0.5 families aside "
                f"{sc['ap50_any'] if sc['ap50_any'] is not None else float('nan'):.3f}, "
                f"on {sc['tiles']} tiles / {sc['boxes']} boxes")
    else:
        run.log("nothing held out: the card carries no held-out score")
    if K.split_files(dataset_dir, "val", "wideband"):
        from atk_diffusion.learn import calibrate as _cal
        card = _cal.calibrate_proposer(rf, profile, model_dir, dataset_dir,
                                       split="val", threads=1)
        thr = (card.calibration.get("operating_threshold") or {}).get("threshold")
        if thr is not None and preds is not None:
            op = K.operating_point(preds, gts, float(thr))
            card.metrics["operating_point"] = {k: (K.finite_or_none(v)
                                                   if isinstance(v, float) else v)
                                               for k, v in op.items()}
            X.resave_card(model_dir, card, rf=rf)
    else:
        run.log("no val split: the confidence is not calibrated and no "
                "operating threshold was chosen")
    run.finish({"model_dir": str(model_dir),
                "map50": card.metrics.get("map50"),
                "ap50_any": card.metrics.get("ap50_any"),
                "latency_ms": lat["p50_ms"]})
    return model_dir


def finetune(rf, profile: str, model_dir, dataset_dir, out_name: str,
             epochs: int = 6, *, lr: float = 2e-4, batch_size: int = 8,
             split: str = "train", indices=None, device: str = "auto",
             seed: int = 0, threads: int | None = None,
             eval_split: str | None = None, latency_repeats: int = 20,
             latency_threads: int | None = None, amp: bool = True,
             overwrite: bool = False, note: str = "", progress=None) -> Path:
    """Fine-tune a saved proposer on a wideband dataset of the same profile —
    the cabled set (DETECTION_DESIGN §6.3: real receiver, perfect labels) or
    the first minutes on site (plan §3.6) — and save it as a NEW model with
    its own card, measured exactly as `train()` measures. `indices` limits
    the tiles used (the minutes-to-acceptable experiment passes the first k
    minutes). The base model is never modified."""
    threads_now = K.set_threads(threads)
    K.seed_everything(seed)
    dev, dev_words = K.pick_device(device)
    model, base = load_torch(model_dir, profile)
    manifest = K.open_dataset(dataset_dir, profile, "wideband", rf=rf)
    policy = K.BoxPolicy.from_json(base.input.get("boxes") or {})
    norm = K.TileNorm.from_json(model.arch["norm"])
    full = D.WidebandTiles(dataset_dir, split, profile, manifest=manifest,
                           policy=policy, norm=norm, train=True, seed=seed)
    use = list(range(len(full))) if indices is None else [int(i) for i in indices]
    if not use:
        raise K.DatasetRefused("there are no tiles to fine-tune on")
    ds = torch.utils.data.Subset(full, use)
    new_dir = K.new_model_dir(rf, profile, out_name, overwrite=overwrite)
    run = K.Run.start(rf, profile, f"{KIND}_finetune_{out_name}",
                      {"base": str(model_dir), "dataset": str(dataset_dir),
                       "split": split, "tiles": len(use), "epochs": epochs,
                       "lr": lr, "seed": seed}, progress)
    run.log(dev_words)
    run.log(f"{threads_now} CPU threads for PyTorch")
    run.log(f"fine-tuning {base.name} on {len(use)} tiles of "
            f"{manifest.get('name')!r} ({split})" + (f" — {note}" if note else ""))
    counts = np.zeros(len(policy.families), np.int64)
    for c in base.classes:
        if isinstance(c, dict) and c.get("name") in policy.families:
            counts[list(policy.families).index(c["name"])] += int(c.get("examples", 0))
    for i in use:
        for v in full.target_rc(i)[1]:
            counts[int(v)] += 1
    hist = fit(model, ds, epochs, lr=lr, batch_size=batch_size, device=dev,
               seed=seed, run=run, amp=amp, tag="finetune")
    split_eval = _eval_split(dataset_dir, eval_split)
    entry = K.dataset_entry(dataset_dir, manifest,
                            (split,) + ((split_eval,) if split_eval else ()))
    entry["tiles_used"] = len(use)
    save_trained(rf, profile, model, new_dir, dataset_dir=dataset_dir,
                 manifest=manifest, policy=policy, run=run,
                 train_summary={"fine_tuned_from": base.name,
                                "base_sha256": base.weights.get("sha256"),
                                "epochs": int(epochs), "tiles": len(use),
                                "lr": float(lr),
                                "final_loss": K.finite_or_none(hist[-1]["loss"])
                                if hist else None, "device": dev_words},
                 counts=counts, datasets=list(base.datasets) + [entry],
                 extra_notes=[f"fine-tuned from {base.name} on "
                              f"{manifest.get('name')} ({len(use)} tiles)"
                              + (f": {note}" if note else "")],
                 eval_split=split_eval, seed=seed,
                 latency_repeats=latency_repeats,
                 latency_threads=latency_threads, trained_on=dev_words,
                 name=out_name)
    return new_dir


def load_torch(model_dir, profile: str | None = None):
    """(PyTorch model, card) rebuilt from a saved proposer, for fine-tuning."""
    card = _cards.load(model_dir, expect_kind=KIND, for_profile=profile)
    model = build_from_arch(card.input["arch"])
    model.load_state_dict(X.load_train_state(model_dir, card))
    model.eval()
    return model, card


def gpu_recommendation() -> dict:
    """What one full run on Bill's RTX 3080 Ti (16 GB) should look like.
    UNTESTED — an engineering estimate written before any GPU run; the first
    real run replaces it with measured numbers."""
    return {"status": "UNTESTED recommendation (no GPU run has been made)",
            "profile": "rtlsdr_2400000_cu8 (tiles 512 rows × 1024 bins)",
            "dataset": "≈20 000 training tiles (TorchSig wideband scenes "
                       "through the receiver's impairment model) + ≈2 000 val, "
                       "≈2 000 test; one cabled set for the domain gap",
            "model": "width 32 (32/64/128/256), FPN 64, head_convs 3, "
                     "finest_level 3 (P3–P7); ≈3 M parameters",
            "training": "batch 16, AdamW lr 1e-3, cosine, 30–40 epochs, AMP "
                        "fp16 — a few hours (estimate)",
            "then": "fine-tune 5–10 epochs at lr 2e-4 on the cabled set; "
                    "measure the CPU latency on the 14-core machine "
                    "(budget 200 ms a tile) and step width down to 16 if over"}
