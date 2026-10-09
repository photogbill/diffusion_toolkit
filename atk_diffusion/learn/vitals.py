# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The small LSTM for vital signs from CSI (plan §4.I1; card kind "vitals").

PulseFi (arXiv 2510.24744) ends its pipeline in a small LSTM that regresses
heart and breathing rate from windows of the filtered CSI amplitude. This is
that model's shape, built per the architecture's rules: PyTorch imported
only inside functions; trained and saved through `cards` (a model without a
card does not load); and always beside its classical comparator,
`sensing.vitals.estimate` — the learned model is kept only if it beats the
classical pipeline on the same windows (plan §7; `experiments.vitals_eval`
scores both).

INPUT. `features()` is the classical front of the pipeline — amplitude, the
AGC divided out, DC removed, band-passed to cover both bands (0.1–2.17 Hz), Savitzky–Golay —
then the K subcarriers with the most in-band energy, in that order,
resampled to `FEATURE_FS` (10 Hz: the heart band's top edge is 2.17 Hz) and
normalised per channel. A window is `WINDOW_S` seconds.

OUTPUT. Breathing and heart rate per minute, de-normalised with the target
statistics stored in the card. Every prediction is labelled
`sensing.vitals.RESEARCH_LABEL`: a research-grade measurement, not a medical
device and not a diagnosis.

LIMITS. Trained here only on synthetic CSI as a code-path proof; it knows
nothing of Bill's room, his ESP32s or a real chest until it is trained on
his own recordings against a count. PulseFi's 118 participants are their
data, not this model's.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from atk_diffusion import cards as _cards
from atk_diffusion import provenance
from atk_diffusion.sensing import vitals as _v

KIND = "vitals"
FEATURE_FS = 10.0
FEATURE_K = 8
WINDOW_S = 30.0
BAND = (_v.BREATH_BAND[0], _v.HEART_BAND[1])
WEIGHTS = "vitals_lstm.pt"

provenance.METHOD_TIERS.setdefault("vitals_lstm", "measured")


def features(A, fs: float, k: int = FEATURE_K, target_fs: float = FEATURE_FS
             ) -> np.ndarray:
    """(time, subcarrier) CSI -> (T, k) float32 at `target_fs`."""
    from scipy.signal import resample_poly
    X = _v.normalise_agc(_v.amplitude(A))
    X = _v.savgol(_v.bandpass(_v.remove_dc(X), fs, BAND), fs, 0.25)
    energy = np.sum(X * X, axis=0)
    idx = np.argsort(-energy)[:k]
    Y = X[:, idx]
    if Y.shape[1] < k:
        Y = np.pad(Y, ((0, 0), (0, k - Y.shape[1])))
    up, down = _ratio(target_fs, fs)
    Y = resample_poly(Y, up, down, axis=0)
    sd = Y.std(axis=0)
    Y = (Y - Y.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    return Y.astype(np.float32)


def _ratio(fs_to: float, fs_from: float) -> tuple[int, int]:
    from fractions import Fraction
    fr = Fraction(fs_to).limit_denominator(1000) / Fraction(fs_from).limit_denominator(1000)
    return fr.numerator, fr.denominator


def _make_net(k: int, hidden: int, layers: int = 1):
    import torch
    from torch import nn

    class VitalsLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(k, hidden, num_layers=layers, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                      nn.Linear(hidden, 2))

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out.mean(dim=1))
    torch.manual_seed(0)
    return VitalsLSTM()


def train(X, y, out_dir, *, name: str = "vitals-lstm", epochs: int = 60,
          lr: float = 3e-3, hidden: int = 32, batch: int = 8, seed: int = 0,
          val_frac: float = 0.2, profile: str = "", trained_on: str = "",
          progress: Callable[[str], None] | None = None) -> tuple:
    """Train on windows X (N, T, k) with targets y (N, 2) = (breathing, heart)
    per minute. Saves `vitals_lstm.pt` and its card in `out_dir`.
    -> (card, metrics)."""
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(int(seed))
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if X.ndim != 3 or y.shape != (X.shape[0], 2):
        raise ValueError("X is (windows, time, channels) and y is (windows, 2)")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(X.shape[0])
    n_val = max(1, int(round(val_frac * X.shape[0]))) if X.shape[0] > 4 else 0
    va, tr = order[:n_val], order[n_val:]
    mu, sd = y[tr].mean(axis=0), y[tr].std(axis=0) + 1e-6
    net = _make_net(X.shape[2], hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy((y - mu) / sd)
    loss_fn = torch.nn.MSELoss()
    hist = []
    for ep in range(int(epochs)):
        net.train()
        perm = rng.permutation(tr)
        tot = 0.0
        for i in range(0, perm.size, batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(net(Xt[b]), yt[b])
            loss.backward()
            opt.step()
            tot += float(loss.item()) * b.size
        hist.append(tot / max(1, tr.size))
        if progress and (ep + 1) % 10 == 0:
            progress(f"epoch {ep + 1}: loss {hist[-1]:.4f}")
    net.eval()
    with torch.no_grad():
        pred = net(Xt).numpy() * sd + mu
    err = np.abs(pred - y)
    metrics = {"train_mae_breath_bpm": float(err[tr, 0].mean()),
               "train_mae_heart_bpm": float(err[tr, 1].mean()),
               "val_mae_breath_bpm": float(err[va, 0].mean()) if n_val else None,
               "val_mae_heart_bpm": float(err[va, 1].mean()) if n_val else None,
               "final_loss": hist[-1] if hist else None,
               "windows": int(X.shape[0])}
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), d / WEIGHTS)
    card = _cards.new_card(
        name, KIND, profile,
        input={"feature_fs": FEATURE_FS, "k": int(X.shape[2]),
               "window_s": X.shape[1] / FEATURE_FS, "band_hz": list(BAND),
               "hidden": int(hidden), "target_norm": {"mean": mu.tolist(),
                                                      "std": sd.tolist()},
               "outputs": ["breath_bpm", "heart_bpm"]},
        metrics=metrics, license="all rights reserved",
        trained_on=trained_on or "windows supplied to learn.vitals.train",
        notes=[_v.RESEARCH_LABEL,
               "kept only if it beats sensing.vitals.estimate on the same "
               "windows (plan §7)"])
    _cards.save(d, card, WEIGHTS)
    return card, metrics


class VitalsModel:
    """A loaded vitals LSTM. `predict(X) -> (N, 2)` per minute."""

    def __init__(self, card, net):
        self.card = card
        self.net = net
        tn = card.input.get("target_norm", {})
        self.mu = np.asarray(tn.get("mean", [0.0, 0.0]), dtype=np.float32)
        self.sd = np.asarray(tn.get("std", [1.0, 1.0]), dtype=np.float32)

    def predict(self, X) -> np.ndarray:
        import torch
        torch.set_num_threads(1)
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[None]
        with torch.no_grad():
            out = self.net(torch.from_numpy(X)).numpy()
        return out * self.sd + self.mu

    def predict_report(self, X) -> list[dict]:
        return [{"breath_bpm": float(b), "heart_bpm": float(h),
                 "label": _v.RESEARCH_LABEL, "tier": "measured",
                 "method": "vitals_lstm", "model": self.card.name}
                for b, h in self.predict(X)]


def load(model_dir) -> VitalsModel:
    """Through the card: refuses a model without one, of another kind, or
    whose weights changed."""
    import torch
    card = _cards.load(model_dir, expect_kind=KIND)
    net = _make_net(int(card.input["k"]), int(card.input["hidden"]))
    state = torch.load(Path(model_dir) / card.weights["file"], map_location="cpu",
                       weights_only=True)
    net.load_state_dict(state)
    net.eval()
    return VitalsModel(card, net)
