# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The learned radio-map residual — a small conditional diffusion model that
learns only what physics cannot (plan E2, E5's correction layer).

The plan, E5: *"the diffusion radio map learns the RESIDUAL between the
physics prediction and real measurements, so it only fills what physics
cannot."* So the model never predicts a received power. It predicts, on a
square patch, the field  measured - physics  given:

    channel 0   the terrain (DTED heights, standardised)
    channel 1   the physics prediction (`reach.predicted_reach`, standardised)
    channel 2   the sparse-sample mask (1 where the drive measured)
    channel 3   the measured residual at those cells (0 elsewhere)

as a denoising diffusion model (DDPM training, eps-prediction, cosine
schedule; DDIM sampling) — so it returns not one map but SAMPLES of the
residual, whose spread is the correction's uncertainty. ControlRadio and
Diffusion² (plan §9) generate whole radio maps; this one is deliberately
narrower, because a learned map that may contradict the physics where
nothing was measured is the hallucination the plan exists to prevent.

MEASURED AGAINST THE CLASSICAL ONE. `evaluate()` scores, on held-out
fields, the model's mean correction against ordinary kriging of the same
sparse residual samples (`geo.radiomap`, the classical baseline) and
against no correction at all — and reports the HALLUCINATION RATE: on
fields where the truth has no residual (physics is right), how often the
model invents a correction larger than `threshold_db`. A model that does
not beat kriging is not shipped (plan §7); the card carries the numbers.

The synthetic stand-in (`synthetic_fields`) makes the code path run here:
a "clutter" loss that physics does not model, tied to low ground (built-up
valleys), plus correlated shadowing; Bill's first experiment replaces it
with a known broadcaster's predicted coverage and a measured drive.

Training needs PyTorch (the training environment). Inference also runs in
ATK's core environment without PyTorch: `export_onnx` writes the denoiser
(inputs x, cond, t -> eps; the schedule in the card) and `sample_onnx`
runs the same DDIM loop in numpy over onnxruntime.

LIMIT. The model works at the fixed patch size it was trained at;
`correct_grid` resamples a coverage grid to the patch and the correction
back (bilinear) — detail finer than a patch cell is not invented.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from atk_diffusion import cards as _cards
from atk_diffusion import provenance as _prov

KIND = "radiomap"
METHOD = "diffusion_radiomap"
CHANNELS = ("terrain", "physics", "mask", "residual_samples")


# ---------------------------------------------------------------------------
# Synthetic residual fields (the stand-in for a measured drive)
# ---------------------------------------------------------------------------
def _smooth_field(rng, n, size, corr_cells):
    k = np.fft.fftfreq(size)
    kx, ky = np.meshgrid(k, k)
    filt = np.exp(-0.5 * (kx ** 2 + ky ** 2) * (2 * math.pi * corr_cells) ** 2)
    w = rng.normal(size=(n, size, size))
    f = np.real(np.fft.ifft2(np.fft.fft2(w) * filt))
    f /= f.std(axis=(1, 2), keepdims=True) + 1e-12
    return f


def synthetic_fields(n: int, size: int = 32, seed: int = 0, *,
                     clutter_db: float = 8.0, shadow_db: float = 3.0,
                     sample_frac: float = 0.08, noise_db: float = 1.0,
                     zero_residual: bool = False) -> dict:
    """n patches: terrain, physics (dBm), the true residual (dB), and the
    sparse measured residual. `zero_residual=True` makes fields where
    physics is exactly right (for the hallucination rate)."""
    rng = np.random.default_rng(seed)
    terrain = 100.0 + 60.0 * _smooth_field(rng, n, size, 4.0)
    yy, xx = np.mgrid[0:size, 0:size]
    phys = np.empty((n, size, size))
    for i in range(n):
        cx, cy = rng.uniform(-size, 2 * size, 2)
        d = np.hypot(xx - cx, yy - cy) * 100.0 + 50.0
        phys[i] = -40.0 - 30.0 * np.log10(d)
    if zero_residual:
        resid = np.zeros((n, size, size))
    else:
        low = (terrain < np.median(terrain, axis=(1, 2), keepdims=True)).astype(float)
        clutter = -clutter_db * low
        resid = clutter + shadow_db * _smooth_field(rng, n, size, 2.0)
    mask = (rng.random((n, size, size)) < sample_frac).astype(np.float64)
    samples = (resid + rng.normal(0, noise_db, resid.shape)) * mask
    return {"terrain": terrain, "physics": phys, "residual": resid,
            "mask": mask, "samples": samples, "size": size}


# ---------------------------------------------------------------------------
# Normalisation and the schedule
# ---------------------------------------------------------------------------
def _norm_stats(fields) -> dict:
    return {"terrain_mean": float(np.mean(fields["terrain"])),
            "terrain_std": float(np.std(fields["terrain"]) or 1.0),
            "physics_mean": float(np.mean(fields["physics"])),
            "physics_std": float(np.std(fields["physics"]) or 1.0)}


def _cond(fields, norm, scale_db) -> np.ndarray:
    t = (np.asarray(fields["terrain"]) - norm["terrain_mean"]) / norm["terrain_std"]
    p = (np.asarray(fields["physics"]) - norm["physics_mean"]) / norm["physics_std"]
    m = np.asarray(fields["mask"], dtype=np.float64)
    s = np.asarray(fields["samples"]) / scale_db * m
    return np.stack([t, p, m, s], axis=1).astype(np.float32)


def cosine_schedule(T: int, s: float = 0.008) -> np.ndarray:
    """alpha-bar for t = 0..T-1 (Nichol & Dhariwal)."""
    t = np.arange(T + 1) / T
    f = np.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    betas = np.clip(1 - ab[1:] / ab[:-1], 0, 0.999)
    return np.cumprod(1 - betas)


def ddim_steps(T: int, k: int) -> list[int]:
    return sorted({int(round(v)) for v in np.linspace(T - 1, 0, max(2, int(k)))},
                  reverse=True)


def ddim_loop(eps_fn, cond: np.ndarray, abar: np.ndarray, steps: int,
              rng, eta: float = 1.0, clip: float = 5.0) -> np.ndarray:
    """DDIM in numpy around any eps predictor (torch or onnxruntime):
    eps_fn(x, cond, t_int64) -> eps. Returns x0 samples (normalised)."""
    B, _, H, W = cond.shape
    x = rng.normal(size=(B, 1, H, W)).astype(np.float32)
    ts = ddim_steps(abar.size, steps)
    for i, t in enumerate(ts):
        a_t = abar[t]
        eps = eps_fn(x, cond, np.full(B, t, dtype=np.int64))
        x0 = np.clip((x - math.sqrt(1 - a_t) * eps) / math.sqrt(a_t), -clip, clip)
        if i + 1 == len(ts):
            x = x0
            break
        a_s = abar[ts[i + 1]]
        sig = eta * math.sqrt((1 - a_s) / (1 - a_t)) * math.sqrt(max(1 - a_t / a_s, 0.0))
        dirn = math.sqrt(max(1 - a_s - sig ** 2, 0.0)) * eps
        x = (math.sqrt(a_s) * x0 + dirn
             + sig * rng.normal(size=x.shape)).astype(np.float32)
    return x


# ---------------------------------------------------------------------------
# The network (PyTorch; built inside functions so this module imports
# without it)
# ---------------------------------------------------------------------------
def _build(channels: int = 32, temb: int = 64):
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self, cin, cout, tdim):
            super().__init__()
            self.n1 = nn.GroupNorm(min(8, cin), cin)
            self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
            self.t = nn.Linear(tdim, cout)
            self.n2 = nn.GroupNorm(min(8, cout), cout)
            self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
            self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

        def forward(self, x, te):
            h = self.c1(torch.nn.functional.silu(self.n1(x)))
            h = h + self.t(te)[:, :, None, None]
            h = self.c2(torch.nn.functional.silu(self.n2(h)))
            return h + self.skip(x)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.temb_dim = temb
            self.tmlp = nn.Sequential(nn.Linear(temb, temb), nn.SiLU(),
                                      nn.Linear(temb, temb))
            c = channels
            self.inc = nn.Conv2d(1 + len(CHANNELS), c, 3, padding=1)
            self.b1 = Block(c, c, temb)
            self.down = nn.Conv2d(c, 2 * c, 3, stride=2, padding=1)
            self.b2 = Block(2 * c, 2 * c, temb)
            self.up = nn.ConvTranspose2d(2 * c, c, 2, stride=2)
            self.b3 = Block(2 * c, c, temb)
            self.out = nn.Conv2d(c, 1, 3, padding=1)

        def embed(self, t):
            half = self.temb_dim // 2
            f = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32)
                          / half)
            a = t.float()[:, None] * f[None, :]
            return self.tmlp(torch.cat([torch.sin(a), torch.cos(a)], dim=1))

        def forward(self, x, cond, t):
            te = self.embed(t)
            h0 = self.inc(torch.cat([x, cond], dim=1))
            h1 = self.b1(h0, te)
            h2 = self.b2(self.down(h1), te)
            u = self.up(h2)
            h3 = self.b3(torch.cat([u, h1], dim=1), te)
            return self.out(torch.nn.functional.silu(h3))
    return Net()


# ---------------------------------------------------------------------------
# Training, saving through the card, loading
# ---------------------------------------------------------------------------
def train(fields: dict, out_dir, *, steps: int = 400, batch: int = 16,
          lr: float = 2e-3, T: int = 100, scale_db: float = 10.0,
          channels: int = 32, seed: int = 0, name: str = "radiomap-residual",
          heldout: dict | None = None, progress=None, threads: int = 1) -> Path:
    """Train on `fields` (synthetic_fields or real patches of the same
    layout); write weights + card to `out_dir`. With `heldout`, the card
    carries the comparison against kriging and the hallucination rate."""
    import torch
    torch.set_num_threads(int(threads))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    size = int(fields["size"])
    if size % 2:
        raise ValueError("the patch size must be even")
    norm = _norm_stats(fields)
    cond = torch.from_numpy(_cond(fields, norm, scale_db))
    x0 = torch.from_numpy((np.asarray(fields["residual"]) / scale_db)[:, None]
                          .astype(np.float32))
    abar = torch.from_numpy(cosine_schedule(T).astype(np.float32))
    net = _build(channels)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    losses = []
    say = progress or (lambda s: None)
    t0 = time.time()
    for it in range(int(steps)):
        idx = torch.from_numpy(rng.integers(0, x0.shape[0], batch))
        xb, cb = x0[idx], cond[idx]
        if rng.random() < 0.5:                          # flips: the map has no
            xb, cb = xb.flip(-1), cb.flip(-1)           # preferred direction
        t = torch.from_numpy(rng.integers(0, T, batch))
        eps = torch.randn_like(xb)
        a = abar[t][:, None, None, None]
        xt = a.sqrt() * xb + (1 - a).sqrt() * eps
        loss = torch.mean((net(xt, cb, t) - eps) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if progress and (it + 1) % 50 == 0:
            say(f"radiomap training: step {it + 1}/{steps}, loss "
                f"{np.mean(losses[-50:]):.4f}")
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), d / "model.pt")
    card = _cards.new_card(name, KIND, "",
                           input={"size": size, "channels": list(CHANNELS),
                                  "scale_db": scale_db, "normalize": norm,
                                  "schedule": {"kind": "cosine", "T": T},
                                  "width": channels},
                           datasets=[{"name": fields.get("name", "synthetic"),
                                      "n": int(x0.shape[0]),
                                      "kind": "residual patches"}],
                           trained_on=f"{int(steps)} steps, batch {batch}, "
                                      f"{time.time() - t0:.0f} s CPU",
                           notes=["learns measured - physics only; outputs "
                                  "are INVENTED tier",
                                  "classical baseline: ordinary kriging of the "
                                  "same sparse residuals (geo.radiomap)"])
    card.metrics = {"train_loss_first": float(np.mean(losses[:20])),
                    "train_loss_last": float(np.mean(losses[-20:]))}
    if heldout is not None:
        model = (net.eval(), card)
        card.metrics.update(evaluate(model, heldout, seed=seed))
        card.metrics["hallucination_rate"] = hallucination_rate(model, seed=seed)
    _cards.save(d, card, "model.pt")
    return d


def load(model_dir):
    """(network, card) — through the card, which must say radiomap."""
    import torch
    card = _cards.load(model_dir, expect_kind=KIND)
    net = _build(int(card.input.get("width", 32)))
    net.load_state_dict(torch.load(Path(model_dir) / card.weights["file"],
                                   map_location="cpu", weights_only=True))
    return net.eval(), card


def _torch_eps(net):
    import torch

    def fn(x, cond, t):
        with torch.no_grad():
            return net(torch.from_numpy(x), torch.from_numpy(cond),
                       torch.from_numpy(t)).numpy()
    return fn


def sample(model, terrain, physics, mask, samples_db, *, n_samples: int = 8,
           steps: int = 25, seed: int = 0, eta: float = 1.0):
    """Residual samples for one patch: (mean_db, std_db, draws_db)."""
    net, card = model
    size = int(card.input["size"])
    for a in (terrain, physics, mask, samples_db):
        if np.asarray(a).shape != (size, size):
            raise ValueError(f"this model works on {size} x {size} patches")
    f = {"terrain": np.asarray(terrain)[None], "physics": np.asarray(physics)[None],
         "mask": np.asarray(mask)[None], "samples": np.asarray(samples_db)[None]}
    cond = np.repeat(_cond(f, card.input["normalize"], card.input["scale_db"]),
                     n_samples, axis=0)
    abar = cosine_schedule(int(card.input["schedule"]["T"]))
    x = ddim_loop(_torch_eps(net), cond, abar, steps,
                  np.random.default_rng(seed), eta)
    draws = x[:, 0].astype(np.float64) * float(card.input["scale_db"])
    return draws.mean(axis=0), draws.std(axis=0), draws


def _kriging_patch(mask, samples, size):
    """Ordinary kriging of the sparse residual on a patch (cells as 100 m)."""
    from atk_diffusion.geo import radiomap as _rm
    from atk_diffusion.geo.products import GeoGrid
    deg = size * 100.0 / 111_195.0
    g = GeoGrid(0.0, 0.0, deg, deg, size, size)
    lat, lon = g.mesh()
    m = np.asarray(mask) > 0
    if m.sum() < 6:
        return np.zeros((size, size))
    meas = _rm.Measurements(lat[m], lon[m], np.asarray(samples)[m], "residual_db")
    return _rm.ordinary_kriging(meas, g).estimate


def evaluate(model, fields: dict, *, n_samples: int = 6, steps: int = 20,
             seed: int = 0, limit: int = 8) -> dict:
    """RMSE (dB) at the cells NOT measured: the model's mean correction,
    ordinary kriging of the same samples, and no correction at all."""
    n = min(limit, len(fields["residual"]))
    size = int(fields["size"])
    e_model, e_krig, e_zero = [], [], []
    for i in range(n):
        mean, _, _ = sample(model, fields["terrain"][i], fields["physics"][i],
                            fields["mask"][i], fields["samples"][i],
                            n_samples=n_samples, steps=steps, seed=seed + i)
        truth = fields["residual"][i]
        hole = fields["mask"][i] == 0
        k = _kriging_patch(fields["mask"][i], fields["samples"][i], size)
        e_model.append((mean - truth)[hole])
        e_krig.append((k - truth)[hole])
        e_zero.append(truth[hole])
    rm = lambda e: float(np.sqrt(np.mean(np.concatenate(e) ** 2)))   # noqa: E731
    out = {"rmse_model_db": rm(e_model), "rmse_kriging_db": rm(e_krig),
           "rmse_physics_only_db": rm(e_zero), "eval_fields": n}
    out["beats_kriging"] = out["rmse_model_db"] < out["rmse_kriging_db"]
    return out


def hallucination_rate(model, *, n: int = 4, threshold_db: float = 3.0,
                       seed: int = 0, steps: int = 20) -> float:
    """On fields where physics is RIGHT (true residual zero, samples only
    noise): the fraction of cells where the mean correction exceeds
    `threshold_db`. Structure invented where the truth has none."""
    _, card = model
    f = synthetic_fields(n, int(card.input["size"]), seed + 991, zero_residual=True)
    hits, total = 0, 0
    for i in range(n):
        mean, _, _ = sample(model, f["terrain"][i], f["physics"][i], f["mask"][i],
                            f["samples"][i], n_samples=4, steps=steps, seed=seed + i)
        hits += int(np.count_nonzero(np.abs(mean) > threshold_db))
        total += mean.size
    return hits / max(total, 1)


# ---------------------------------------------------------------------------
# A coverage grid, corrected
# ---------------------------------------------------------------------------
def _resample(a, shape):
    from scipy.ndimage import map_coordinates
    a = np.asarray(a, dtype=np.float64)
    r = np.linspace(0, a.shape[0] - 1, shape[0])
    c = np.linspace(0, a.shape[1] - 1, shape[1])
    R, C = np.meshgrid(r, c, indexing="ij")
    return map_coordinates(np.nan_to_num(a, nan=float(np.nanmean(a))), [R, C],
                           order=1, mode="nearest")


def correct_grid(model, physics_dbm, terrain_m, sample_mask, residual_db, *,
                 n_samples: int = 8, steps: int = 25, seed: int = 0):
    """The correction layer for a coverage grid of any shape: resample the
    inputs to the model's patch, sample, resample the mean and std back.
    Returns (correction_db, sigma_db), INVENTED tier."""
    _, card = model
    size = int(card.input["size"])
    shape = np.asarray(physics_dbm).shape
    m_small = (_resample(np.asarray(sample_mask, float), (size, size)) > 0.25)
    r_small = _resample(np.where(np.asarray(sample_mask) > 0, residual_db, 0.0),
                        (size, size))
    w_small = _resample(np.asarray(sample_mask, float), (size, size))
    r_small = np.where(m_small, r_small / np.maximum(w_small, 1e-6), 0.0)
    mean, std, _ = sample(model, _resample(terrain_m, (size, size)),
                          _resample(physics_dbm, (size, size)),
                          m_small.astype(float), r_small, n_samples=n_samples,
                          steps=steps, seed=seed)
    return _resample(mean, shape), _resample(std, shape)


# ---------------------------------------------------------------------------
# ONNX: inference in ATK's core environment
# ---------------------------------------------------------------------------
def export_onnx(model_dir, opset: int = 17) -> Path:
    """denoiser.onnx beside the weights: inputs x [B,1,S,S], cond [B,4,S,S],
    t int64 [B]; output eps. The card is updated with the file's hash."""
    import torch
    net, card = load(model_dir)
    S = int(card.input["size"])
    x = torch.zeros(1, 1, S, S)
    c = torch.zeros(1, len(CHANNELS), S, S)
    t = torch.zeros(1, dtype=torch.int64)
    out = Path(model_dir) / "denoiser.onnx"
    torch.onnx.export(net, (x, c, t), str(out), input_names=["x", "cond", "t"],
                      output_names=["eps"], opset_version=opset,
                      dynamic_axes={"x": {0: "b"}, "cond": {0: "b"},
                                    "t": {0: "b"}, "eps": {0: "b"}},
                      dynamo=False)
    card.input["onnx"] = {"file": out.name, "sha256": _prov.sha256_path(out),
                          "exporter": "torch.onnx (TorchScript)"}
    _cards.save(model_dir, card, card.weights["file"])
    return out


def sample_onnx(model_dir, terrain, physics, mask, samples_db, *,
                n_samples: int = 8, steps: int = 25, seed: int = 0):
    """The same sampler without PyTorch: numpy + onnxruntime."""
    import onnxruntime as ort
    card = _cards.load(model_dir, expect_kind=KIND)
    info = card.input.get("onnx")
    if not info:
        raise _cards.CardRefusal("this radiomap model has no ONNX export — run "
                                 "export_onnx in the training environment")
    path = Path(model_dir) / info["file"]
    if _prov.sha256_path(path) != info["sha256"]:
        raise _cards.CardRefusal(f"{path.name} is not the file the card "
                                 "describes (its hash changed)")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def fn(x, cond, t):
        return sess.run(None, {"x": x, "cond": cond, "t": t})[0]
    f = {"terrain": np.asarray(terrain)[None], "physics": np.asarray(physics)[None],
         "mask": np.asarray(mask)[None], "samples": np.asarray(samples_db)[None]}
    cond = np.repeat(_cond(f, card.input["normalize"], card.input["scale_db"]),
                     n_samples, axis=0)
    abar = cosine_schedule(int(card.input["schedule"]["T"]))
    x = ddim_loop(fn, cond, abar, steps, np.random.default_rng(seed))
    draws = x[:, 0].astype(np.float64) * float(card.input["scale_db"])
    return draws.mean(axis=0), draws.std(axis=0), draws
