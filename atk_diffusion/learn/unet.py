# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Small U-Nets for the diffusion tools: 1D over IQ windows [B, 2, L] and
2D over spectrogram tiles [B, 1, H, W] (ARCHITECTURE §4.4; plan B3, D1, A6,
B4, R).

WHY A U-NET. The denoiser has to see a burst whole — its edges, its
bandwidth, the band around it — and still put every output sample back where
it was. A U-Net does both: the down path sees wide context cheaply, the skip
connections keep the fine detail. It is the canonical small model for this
job in RF (ETH's RFI U-Net, plan §9, 1609.09077) and in diffusion (Ho et
al.). Plan B3 asks for "a small U-Net on spectrogram tiles from Bill's own
captures, trained per profile"; the CPU budget (DETECTION_DESIGN §7) is why
it stays small.

WHAT IS IN IT. Residual blocks (GroupNorm, SiLU, 3-wide convolutions) with
the diffusion timestep added in through a sinusoidal embedding and a small
MLP; optional CONDITION CHANNELS concatenated to the input (the translator's
receiver-A window, plan A6; the cyclic parameters as a learned FRESH filter,
DETECTION_DESIGN §4.3 "Then the learned one"); an optional CLASS EMBEDDING
added to the time embedding (generative classification, plan §4.R; index
`num_classes` is the "no class" token). Nearest-neighbour upsampling + a
convolution, not transposed convolutions, so no checkerboard is learned —
a checkerboard on a waterfall is a pattern an analyst could mistake for a
signal.

FULLY CONVOLUTIONAL: trained on patches, run on whole tiles. Spatial sizes
must be multiples of `multiple(config)`; `pad_to_multiple` reflect-pads and
the caller crops back.

ONNX: `export_onnx` uses the TorchScript exporter (opset 17), which writes
ONE self-contained file — the hash in the model card covers every weight.
(torch 2.14's default dynamo exporter needs `onnxscript` and may split the
weights into a side file; neither is acceptable for a card-verified model.)

Limits, stated: these are small networks for a CPU; they will under-fit
what a GPU-sized model would learn on Bill's captures, and the tiny
configurations used in tests learn almost nothing — they prove the code
path, not the method.
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Test-sized configurations (seconds on one CPU thread). They prove the
#: code path; they are not useful denoisers.
TINY_1D = {"dims": 1, "in_ch": 2, "base": 8, "mults": [1, 2], "num_res": 1,
           "groups": 4}
TINY_2D = {"dims": 2, "in_ch": 1, "base": 8, "mults": [1, 2], "num_res": 1,
           "groups": 4}
#: Starting points for Bill's GPU (measured on his machine, then recorded
#: in the card — these are not tuned numbers).
SMALL_1D = {"dims": 1, "in_ch": 2, "base": 32, "mults": [1, 2, 2, 4],
            "num_res": 2, "groups": 8}
SMALL_2D = {"dims": 2, "in_ch": 1, "base": 32, "mults": [1, 2, 2, 4],
            "num_res": 2, "groups": 8}


def _conv(dims: int):
    if dims == 1:
        return nn.Conv1d
    if dims == 2:
        return nn.Conv2d
    raise ValueError("a U-Net here is 1D (IQ) or 2D (spectrogram tiles)")


def _groups(ch: int, want: int) -> int:
    g = math.gcd(int(ch), max(1, int(want)))
    return max(1, g)


def _expand(v, dims: int):
    for _ in range(dims):
        v = v.unsqueeze(-1)
    return v


class SinusoidalEmbedding(nn.Module):
    """The transformer position code applied to the diffusion timestep."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = int(dim)
        half = max(1, self.dim // 2)
        freqs = torch.exp(-math.log(max_period)
                          * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t):
        a = t.to(torch.float32)[:, None] * self.freqs[None, :]
        emb = torch.cat([torch.sin(a), torch.cos(a)], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return emb


class ResBlock(nn.Module):
    def __init__(self, dims: int, cin: int, cout: int, temb: int, groups: int,
                 dropout: float = 0.0):
        super().__init__()
        conv = _conv(dims)
        self.dims = dims
        self.n1 = nn.GroupNorm(_groups(cin, groups), cin)
        self.c1 = conv(cin, cout, 3, padding=1)
        self.t = nn.Linear(temb, cout)
        self.n2 = nn.GroupNorm(_groups(cout, groups), cout)
        self.drop = nn.Dropout(dropout)
        self.c2 = conv(cout, cout, 3, padding=1)
        self.skip = conv(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + _expand(self.t(F.silu(temb)), self.dims)
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, dims: int, ch: int):
        super().__init__()
        self.c = _conv(dims)(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.c(x)


class Upsample(nn.Module):
    def __init__(self, dims: int, ch: int):
        super().__init__()
        self.c = _conv(dims)(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.c(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class UNet(nn.Module):
    """ε_θ(x_t, t [, cond] [, y]). See the module docstring."""

    def __init__(self, dims: int = 1, in_ch: int = 2, out_ch: int | None = None,
                 cond_ch: int = 0, base: int = 32, mults=(1, 2, 2),
                 num_res: int = 1, groups: int = 8, num_classes: int = 0,
                 dropout: float = 0.0, time_dim: int | None = None):
        super().__init__()
        out_ch = int(out_ch or in_ch)
        mults = [int(m) for m in mults]
        temb = int(time_dim or 4 * base)
        self.config = {"dims": int(dims), "in_ch": int(in_ch), "out_ch": out_ch,
                       "cond_ch": int(cond_ch), "base": int(base),
                       "mults": mults, "num_res": int(num_res),
                       "groups": int(groups), "num_classes": int(num_classes),
                       "dropout": float(dropout), "time_dim": temb}
        conv = _conv(dims)
        self.dims, self.cond_ch, self.num_classes = int(dims), int(cond_ch), int(num_classes)
        self.temb = nn.Sequential(SinusoidalEmbedding(base), nn.Linear(base, temb),
                                  nn.SiLU(), nn.Linear(temb, temb))
        self.class_emb = nn.Embedding(num_classes + 1, temb) if num_classes else None
        self.inp = conv(in_ch + cond_ch, base, 3, padding=1)
        chans = [base * m for m in mults]
        skips = [base]
        cur = base
        self.down = nn.ModuleList()
        for i, ch in enumerate(chans):
            for _ in range(num_res):
                self.down.append(ResBlock(dims, cur, ch, temb, groups, dropout))
                cur = ch
                skips.append(cur)
            if i < len(chans) - 1:
                self.down.append(Downsample(dims, cur))
                skips.append(cur)
        self.mid1 = ResBlock(dims, cur, cur, temb, groups, dropout)
        self.mid2 = ResBlock(dims, cur, cur, temb, groups, dropout)
        self.up = nn.ModuleList()
        for i in reversed(range(len(chans))):
            for _ in range(num_res + 1):
                self.up.append(ResBlock(dims, cur + skips.pop(), chans[i], temb,
                                        groups, dropout))
                cur = chans[i]
            if i > 0:
                self.up.append(Upsample(dims, cur))
        self.out_norm = nn.GroupNorm(_groups(cur, groups), cur)
        self.out = conv(cur, out_ch, 3, padding=1)
        nn.init.zeros_(self.out.weight)       # start as "predict no noise"
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t, cond=None, y=None):
        if self.cond_ch:
            if cond is None:
                raise ValueError(f"this model was trained with {self.cond_ch} "
                                 "condition channels and none were given")
            x = torch.cat([x, cond], dim=1)
        emb = self.temb(t)
        if self.class_emb is not None:
            if y is None:
                y = torch.full((x.shape[0],), self.num_classes, dtype=torch.long,
                               device=x.device)
            emb = emb + self.class_emb(y)
        h = self.inp(x)
        hs = [h]
        for m in self.down:
            h = m(h, emb) if isinstance(m, ResBlock) else m(h)
            hs.append(h)
        h = self.mid2(self.mid1(h, emb), emb)
        for m in self.up:
            if isinstance(m, ResBlock):
                h = m(torch.cat([h, hs.pop()], dim=1), emb)
            else:
                h = m(h)
        return self.out(F.silu(self.out_norm(h)))


def build_unet(config: dict) -> UNet:
    """A U-Net from the config a card stores (`UNet.config`)."""
    return UNet(**dict(config))


def multiple(config: dict) -> int:
    """Spatial sizes must be multiples of this (2 per downsampling)."""
    return 2 ** (len(config.get("mults", [1])) - 1)


def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def pad_to_multiple(x, m: int, mode: str = "reflect"):
    """Pad the trailing spatial dims of a torch tensor [B, C, …] to a
    multiple of m. Returns (padded, original spatial shape)."""
    spatial = tuple(int(s) for s in x.shape[2:])
    pads = []
    for s in reversed(spatial):
        extra = (-s) % int(m)
        pads += [0, extra]
    if not any(pads):
        return x, spatial
    if mode == "reflect" and any(s < 2 for s in spatial):
        mode = "replicate"
    return F.pad(x, pads, mode=mode), spatial


def crop_to(x, spatial):
    idx = (slice(None), slice(None)) + tuple(slice(0, s) for s in spatial)
    return x[idx]


class _ConcatWrapper(nn.Module):
    """For export: x carries data channels then condition channels."""

    def __init__(self, model: UNet):
        super().__init__()
        self.model = model
        self.data_ch = model.config["in_ch"]

    def forward(self, x, t):
        if self.model.cond_ch:
            return self.model(x[:, :self.data_ch], t, cond=x[:, self.data_ch:])
        return self.model(x, t)


def export_onnx(model: UNet, path, example_shape, opset: int = 17,
                dynamic_spatial: bool = True):
    """Write `model` as ONE self-contained ONNX file with inputs "x" float32
    [B, C(+cond), …] and "t" int64 [B], output "eps" (ARCHITECTURE §5).
    Batch (and, by default, spatial) axes are dynamic. A class-conditional
    model is exported unconditional (its "no class" token): the ONNX contract
    has no label input."""
    model = model.eval()
    wrapper = _ConcatWrapper(model).eval()
    c = model.config["in_ch"] + model.config["cond_ch"]
    shape = (1, c) + tuple(int(s) for s in example_shape)
    x = torch.zeros(shape, dtype=torch.float32)
    t = torch.zeros((1,), dtype=torch.long)
    axes = {"x": {0: "B"}, "t": {0: "B"}, "eps": {0: "B"}}
    if dynamic_spatial:
        names = ["L"] if model.dims == 1 else ["H", "W"]
        for i, n in enumerate(names):
            axes["x"][2 + i] = n
            axes["eps"][2 + i] = n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(wrapper, (x, t), str(path), input_names=["x", "t"],
                          output_names=["eps"], opset_version=int(opset),
                          dynamic_axes=axes, dynamo=False)
    return path
