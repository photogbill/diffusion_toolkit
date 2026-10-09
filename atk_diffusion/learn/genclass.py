# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Generative classification — research track (plan §4.R, decision D9: "left
on R"), and the small discriminative classifier this package's experiments
use as their downstream yardstick.

Plan §4.R: *"Generative classification for modulation recognition (a
denoiser per class; classify by which denoises best) — robust at low SNR;
niche beside TorchSig's classifiers."*

HOW. One CLASS-CONDITIONAL denoiser (the U-Net's class embedding) instead of
one per class — the same thing, cheaper. To classify a window x: draw a
fixed set of (t, ε) pairs, form x_t for each, and ask the model to predict
ε under every class label; the class whose denoising error E‖ε − ε_θ(x_t,
t, c)‖² is lowest wins (the "diffusion classifier" of Li et al., 2023). The
SAME draws are used for every class, so the comparison is paired and its
variance small. Steps are taken from the middle of the schedule, where the
class decides most of what the model can predict.

WHY IT MIGHT HOLD UP AT LOW SNR. A discriminative classifier learns a
boundary from the training SNRs; the generative one asks "which class's
model of the signal explains this best", and the forward process already
contains every noise level. That is the claim; `compare` measures it against
a small discriminative CNN on the same tiny synthetic data, accuracy per SNR,
and states the result whichever way it falls.

OUTPUT. A class from here is a PROPOSAL (card kind "genclass", tier
proposed): the confirmer and the decoder rules apply as to any classifier
(DETECTION_DESIGN §5).

LIMITS. Classification costs (classes × draws) network evaluations per
window — far slower than one forward pass; with tiny models and data it
learns little, and the test only proves the path. On R by Bill's choice.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from atk_diffusion import cards, profiles, provenance
from atk_diffusion.learn import diffusion as _diff

provenance.METHOD_TIERS.setdefault("diffusion_classify", "proposed")
METHOD = "diffusion_classify"


def _prep(X) -> np.ndarray:
    """complex [N, L] -> float32 [N, 2, L], unit power per real element."""
    X = np.asarray(X)
    rms = np.sqrt(np.mean(np.abs(X) ** 2, axis=-1, keepdims=True) / 2.0)
    Xn = X / np.maximum(rms, 1e-12)
    return np.stack([Xn.real, Xn.imag], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# The generative classifier
# ---------------------------------------------------------------------------
def train_genclass(rf, profile: str, X, y, classes, *, name: str | None = None,
                   schedule: str = "cosine", T: int = 1000, unet: dict | None = None,
                   steps: int = 20000, batch: int = 32, lr: float = 2e-4,
                   seed: int = 0, device: str | None = None, out_dir=None,
                   progress=None) -> Path:
    """Train the class-conditional denoiser on labelled windows (complex
    [N, L] at the profile's rate). Card kind "genclass" (tier proposed)."""
    import torch
    from atk_diffusion.learn import unet as _u
    say = progress or (lambda s: None)
    profiles.parse_profile_id(profile)
    Xc = _prep(X)
    yy = np.asarray(y, dtype=np.int64)
    C = len(classes)
    conf = dict(unet or _u.SMALL_1D)
    conf.update(dims=1, in_ch=2, num_classes=C)
    if Xc.shape[-1] % _u.multiple(conf):
        raise ValueError(f"the window {Xc.shape[-1]} must be a multiple of {_u.multiple(conf)}")
    torch.manual_seed(int(seed))
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _u.build_unet(conf).to(dev)
    sched = _diff.make_schedule(schedule, T)
    DX, DY = torch.as_tensor(Xc), torch.as_tensor(yy)
    gen = torch.Generator().manual_seed(int(seed))

    def batch_fn(step, g):
        i = torch.randint(0, DX.shape[0], (int(batch),), generator=g)
        return {"x0": DX[i].to(dev), "y": DY[i].to(dev)}
    say(f"training the generative classifier on {Xc.shape[0]} windows, {C} classes")
    hist = _diff.fit(model, batch_fn, sched.alphas_cumprod, steps, lr=lr, progress=say,
                     generator=gen)
    model = model.to("cpu").eval()
    name = name or f"genclass_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"
    d = Path(out_dir) if out_dir else rf.models(profile, name)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "model.pt")
    card = cards.new_card(
        name, "genclass", profile,
        input={"window": int(Xc.shape[-1]), "normalize": "unit power per window",
               "schedule": sched.to_json(), "unet": model.config,
               "t_fraction": [0.1, 0.7]},
        classes=[{"name": str(c), "source": "trained", "examples": int(np.sum(yy == i))}
                 for i, c in enumerate(classes)],
        datasets=[{"name": "labelled windows", "kind": "iq", "n": int(Xc.shape[0]),
                   "sha256": provenance.sha256_bytes(Xc.tobytes())}],
        metrics={"final_loss": float(np.mean(hist["loss"][-50:])) if hist["loss"] else None,
                 "steps": int(steps)},
        license="all rights reserved", trained_on=f"{dev} (torch {torch.__version__})",
        notes=["a class from here is PROPOSED; decoders confirm (DETECTION_DESIGN §5)",
               "research track (plan §4.R)"])
    cards.save(d, card, "model.pt")
    return d


class GenerativeClassifier:
    def __init__(self, model, card):
        self.model, self.card = model, card
        self.classes = card.class_names()
        self.ac = _diff.Schedule.from_json(card.input["schedule"]).alphas_cumprod
        self.L = int(card.input["window"])

    @classmethod
    def load(cls, model_dir, for_profile: str | None = None):
        import torch
        from atk_diffusion.learn import unet as _u
        card = cards.load(model_dir, expect_kind="genclass", for_profile=for_profile)
        model = _u.build_unet(card.input["unet"])
        model.load_state_dict(torch.load(cards.weights_path(model_dir, card),
                                         map_location="cpu", weights_only=True))
        return cls(model.eval(), card)

    def errors(self, X, n_t: int = 16, seed: int = 0, batch: int = 256) -> np.ndarray:
        """[N, C] mean denoising error per class, over n_t (t, ε) draws
        shared by every class (a paired comparison)."""
        import torch
        Xc = torch.as_tensor(_prep(X))
        N, C, T = Xc.shape[0], len(self.classes), self.ac.shape[0]
        lo, hi = self.card.input.get("t_fraction", [0.1, 0.7])
        ts = np.unique(np.linspace(lo * (T - 1), hi * (T - 1), int(n_t)).astype(np.int64))
        g = torch.Generator().manual_seed(int(seed))
        out = np.zeros((N, C))
        with torch.no_grad():
            for t in ts:
                eps = torch.randn(Xc.shape, generator=g)
                xt = _diff.q_sample(Xc, int(t), eps, self.ac)
                for c in range(C):
                    for s in range(0, N, int(batch)):
                        sl = slice(s, s + int(batch))
                        n = xt[sl].shape[0]
                        pred = self.model(xt[sl], torch.full((n,), int(t), dtype=torch.long),
                                          y=torch.full((n,), c, dtype=torch.long))
                        out[sl, c] += torch.mean((pred - eps[sl]) ** 2, dim=(1, 2)).numpy()
        return out / len(ts)

    def classify(self, X, n_t: int = 16, seed: int = 0):
        """(indices [N], class names, errors [N, C], margin [N] = runner-up
        error minus best). Every answer is a PROPOSAL."""
        e = self.errors(X, n_t=n_t, seed=seed)
        idx = np.argmin(e, axis=1)
        srt = np.sort(e, axis=1)
        margin = srt[:, 1] - srt[:, 0] if e.shape[1] > 1 else np.zeros(e.shape[0])
        return idx, [self.classes[i] for i in idx], e, margin


# ---------------------------------------------------------------------------
# The discriminative yardstick (shared by the experiments)
# ---------------------------------------------------------------------------
def _cnn(num_classes: int, width: int = 16):
    import torch.nn as nn

    class TinyCNN(nn.Module):
        """Three conv blocks, global average pool, a linear head: the small
        discriminative classifier every comparison here uses."""

        def __init__(self):
            super().__init__()
            w = int(width)
            self.f = nn.Sequential(
                nn.Conv1d(2, w, 7, padding=3), nn.GroupNorm(4, w), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(w, 2 * w, 5, padding=2), nn.GroupNorm(4, 2 * w), nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(2 * w, 2 * w, 3, padding=1), nn.GroupNorm(4, 2 * w), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1))
            self.head = nn.Linear(2 * w, int(num_classes))

        def forward(self, x):
            return self.head(self.f(x).squeeze(-1))
    return TinyCNN()


def train_discriminative(X, y, num_classes: int, steps: int = 300, batch: int = 32,
                         lr: float = 2e-3, width: int = 16, seed: int = 0,
                         device: str | None = None):
    """A small 1D CNN on unit-power IQ windows, cross-entropy, AdamW."""
    import torch
    torch.manual_seed(int(seed))
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _cnn(num_classes, width).to(dev)
    DX = torch.as_tensor(_prep(X))
    DY = torch.as_tensor(np.asarray(y, dtype=np.int64))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(int(seed))
    model.train()
    for _ in range(int(steps)):
        i = torch.randint(0, DX.shape[0], (min(int(batch), DX.shape[0]),), generator=g)
        loss = torch.nn.functional.cross_entropy(model(DX[i].to(dev)), DY[i].to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model.to("cpu").eval()


def predict_discriminative(model, X, batch: int = 512) -> np.ndarray:
    import torch
    DX = torch.as_tensor(_prep(X))
    out = []
    with torch.no_grad():
        for s in range(0, DX.shape[0], int(batch)):
            out.append(torch.argmax(model(DX[s:s + int(batch)]), dim=1).numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------
def compare(rf=None, profile: str = "rtlsdr_2400000_cu8", *,
            kinds=("bpsk", "qpsk", "gfsk", "ofdm"), snrs_db=(-6.0, -3.0, 0.0, 5.0),
            train_snr_db=(-6.0, 10.0), n_train_per: int = 200, n_test_per: int = 100,
            window: int = 128, fs: float = 240_000.0, gen_steps: int = 4000,
            disc_steps: int = 600, n_t: int = 16, unet: dict | None = None, T: int = 1000,
            seed: int = 0, out_dir=None, progress=None) -> dict:
    """Generative vs discriminative classification at low SNR on tiny
    synthetic data (both trained on the same windows at mixed SNRs,
    tested per SNR). SNRs are over the sampled bandwidth. Returns the
    result dict; writes a report when `out_dir` or `rf` is given."""
    from atk_diffusion.learn import augment as _aug
    say = progress or (lambda s: None)
    rng = np.random.default_rng(seed)
    kinds = list(kinds)
    Xtr, ytr = _aug.make_classification_set(kinds, n_train_per, window, fs, rng,
                                            snr_db=train_snr_db, offset_frac=0.05)
    if out_dir is None and rf is None:
        raise ValueError("compare trains a model, so it needs somewhere to put it: "
                         "pass rf= (the profile's models folder) or out_dir=")
    model_dir = (Path(out_dir) / "model") if out_dir else rf.models(
        profile, f"genclass_compare_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}")
    d = train_genclass(rf, profile, Xtr, ytr, kinds, steps=gen_steps, unet=unet, T=T,
                       seed=seed, out_dir=model_dir, progress=say)
    gc = GenerativeClassifier.load(d, for_profile=profile)
    disc = train_discriminative(Xtr, ytr, len(kinds), steps=disc_steps, seed=seed)
    acc_g, acc_d = [], []
    for snr in snrs_db:
        Xte, yte = _aug.make_classification_set(kinds, n_test_per, window, fs, rng,
                                                snr_db=float(snr), offset_frac=0.05)
        pg, _names, _e, _m = gc.classify(Xte, n_t=n_t, seed=seed)
        pd = predict_discriminative(disc, Xte)
        acc_g.append(float(np.mean(pg == yte)))
        acc_d.append(float(np.mean(pd == yte)))
        say(f"{snr:+.1f} dB: generative {acc_g[-1]:.2f}, discriminative {acc_d[-1]:.2f}")
    result = {"name": "genclass_compare", "profile": profile, "kinds": kinds,
              "snrs_db": [float(s) for s in snrs_db], "train_snr_db": list(train_snr_db),
              "window": int(window), "fs": float(fs), "n_train_per": int(n_train_per),
              "n_test_per": int(n_test_per), "chance": 1.0 / len(kinds),
              "accuracy": {"generative": acc_g, "discriminative": acc_d},
              "tier": provenance.tier_for(METHOD),
              "snr_definition": "signal power over noise power across the sampled band",
              "model": gc.card.name}
    low = int(np.argmin(result["snrs_db"]))
    better = acc_g[low] > acc_d[low]
    result["verdict"] = (f"at the lowest SNR ({snrs_db[low]:+.1f} dB) the generative "
                         f"classifier is {'ahead' if better else 'not ahead'} "
                         f"({acc_g[low]:.2f} vs {acc_d[low]:.2f}; chance "
                         f"{result['chance']:.2f})")
    if out_dir is not None or rf is not None:
        from atk_diffusion.experiments.weak_burst import report_dir, write_report
        target = Path(out_dir) if out_dir is not None else report_dir(rf, profile, "genclass_compare")
        table = ["| SNR (dB) | generative | discriminative |", "|---|---|---|"] + [
            f"| {s:+.1f} | {g:.2f} | {dd:.2f} |" for s, g, dd in zip(snrs_db, acc_g, acc_d)]
        md, js, det = write_report(target, "genclass_compare", result,
                                   [f"Generative vs discriminative classification of "
                                    f"{', '.join(kinds)} ({window}-sample windows).",
                                    result["verdict"] + ".",
                                    "A class from either is PROPOSED tier."],
                                   detail=["# Generative classification (plan §4.R)", ""] + table,
                                   rf=rf)
        result["report_md"], result["report_json"] = str(md), str(js)
    return result
