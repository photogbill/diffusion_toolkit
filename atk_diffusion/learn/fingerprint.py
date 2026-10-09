# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The learned RF fingerprint — a channel-resilient CNN on raw IQ, with an
honest UNKNOWN, and "denoise first, then fingerprint" measured (plan C1,
C2).

DeepRadioID (Restuccia et al., 2019) learns a transmitter's hardware
signature from raw IQ and names the confound: the channel is in every
sample too. Its answer, followed here, is to make the channel a nuisance
the network must learn to ignore: every training window passes through a
FRESH random channel — a random complex gain (the absolute carrier phase
means nothing), a short random multipath filter, a small frequency drift
of Doppler size (never as large as the radios' own crystal differences,
or the strongest fingerprint would be trained away), and noise at a random
SNR from 5 to 30 dB. The receiver is held constant by the profile; the
card is per profile and refuses any other.

THE NETWORK. Windows of `window` samples from the burst (from its key-up
on, so the turn-on transient is in the first window), each scaled to unit
RMS; a small 1-D CNN (four conv blocks, global average pool) to a
normalised embedding and the class logits; a burst is the mean of its
windows.

OPEN SET, HiNoVa-style (arXiv 2305.09594): a radio never enrolled must
come back UNKNOWN, not as the nearest known one. The hidden activations
(the normalised embedding, which the ONNX export also outputs) of each
known radio are summarised by a unit class mean on held-out bursts; a
burst's distance is its cosine distance to the nearest class mean, and a
burst beyond the threshold — the 95th percentile of LEAVE-ONE-OUT distances
of held-out known bursts — is UNKNOWN. `evaluate` measures how often a
held-out never-enrolled radio is rejected.

Why cosine and not Mahalanobis (2026-10-09, measured): the embedding has 64
dimensions and a held-out set has tens of bursts, so any covariance is
fitted to too few points and every NEW burst lands far outside it — the
first build called every test burst UNKNOWN. The cosine distance is one
number per class, its leave-one-out distribution matches new bursts', and
on the simulated pair it separated a never-enrolled radio (0.11-0.16) from
the enrolled ones (under 0.05) with nothing tuned.

C2 — DENOISE FIRST (the Liverpool paper, arXiv 2503.05514). `evaluate`
takes an injected `denoiser(iq, fs) -> iq` (the B3 diffusion denoiser, or
any classical one) and scores the same bursts raw and denoised; the
denoiser's HALLUCINATION RATE is measured on noise-only captures — how
often its output holds a burst where the input held none. A cleaner signal
that the fingerprint cannot tell apart better is not a gain.

Card kind "fingerprint" (PROPOSED tier: an identification nobody has
confirmed). ONNX export for inference in ATK's core environment follows the
classifier's I/O: in `iq` [B, 2, L]; out `logits`, `embedding`.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from atk_diffusion import cards as _cards
from atk_diffusion import provenance as _prov

KIND = "fingerprint"
METHOD = "learned_fingerprint"
UNKNOWN = "UNKNOWN"
_prov.METHOD_TIERS.setdefault(METHOD, "proposed")


# ---------------------------------------------------------------------------
# Windows and the channel
# ---------------------------------------------------------------------------
def windows(x, window: int = 256, hop: int | None = None, max_windows: int = 8,
            fs: float | None = None) -> np.ndarray:
    """(n, 2, window) float32 windows from the burst (key-up first), each
    at unit RMS."""
    from atk_diffusion.fingerprint.features import burst_bounds
    x = np.asarray(x, dtype=np.complex64)
    hop = int(hop or window)
    try:
        s, e, _, _ = burst_bounds(x, fs or 48_000.0)
    except ValueError:
        s, e = 0, x.size
    if e - s < window:
        s, e = max(0, e - window), max(e, window)
    seg = x[s:e]
    starts = list(range(0, max(1, seg.size - window + 1), hop))[:max_windows]
    out = np.empty((len(starts), 2, window), dtype=np.float32)
    for k, a in enumerate(starts):
        w = seg[a:a + window]
        if w.size < window:
            w = np.pad(w, (0, window - w.size))
        w = w / max(float(np.sqrt(np.mean(np.abs(w) ** 2))), 1e-12)
        out[k, 0], out[k, 1] = w.real, w.imag
    return out


def random_channel(xw: np.ndarray, rng, *, fs: float = 48_000.0,
                   snr_db=(5.0, 30.0), multipath: float = 0.3,
                   drift_hz: float = 20.0) -> np.ndarray:
    """A fresh random channel for every window (numpy, (B, 2, L))."""
    B, _, L = xw.shape
    z = xw[:, 0].astype(np.float64) + 1j * xw[:, 1]
    z = z * np.exp(1j * rng.uniform(0, 2 * math.pi, (B, 1)))
    if multipath > 0:
        taps = np.zeros((B, 3), complex)
        taps[:, 0] = 1.0
        taps[:, 1:] = multipath * (rng.normal(size=(B, 2)) + 1j * rng.normal(size=(B, 2))) \
            / math.sqrt(2) * np.array([1.0, 0.5])
        out = np.empty_like(z)
        for b in range(B):
            out[b] = np.convolve(z[b], taps[b])[:L]
        z = out
    n = np.arange(L)
    z = z * np.exp(2j * math.pi * rng.uniform(-drift_hz, drift_hz, (B, 1)) * n / fs)
    snr = rng.uniform(snr_db[0], snr_db[1], (B, 1))
    p = np.mean(np.abs(z) ** 2, axis=1, keepdims=True)
    z = z + np.sqrt(p / 10 ** (snr / 10) / 2) * (rng.normal(size=z.shape)
                                                 + 1j * rng.normal(size=z.shape))
    z = z / np.sqrt(np.mean(np.abs(z) ** 2, axis=1, keepdims=True))
    return np.stack([z.real, z.imag], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------
def _build(n_classes: int, width: int = 32, emb: int = 64):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()

            def blk(cin, cout, k, s):
                return nn.Sequential(nn.Conv1d(cin, cout, k, stride=s, padding=k // 2),
                                     nn.BatchNorm1d(cout), nn.ReLU())
            self.body = nn.Sequential(blk(2, width, 7, 1), blk(width, width, 5, 2),
                                      blk(width, 2 * width, 5, 2),
                                      blk(2 * width, 2 * width, 3, 2))
            self.proj = nn.Linear(2 * width, emb)
            self.head = nn.Linear(emb, n_classes)

        def features(self, x):
            return self.proj(self.body(x).mean(dim=-1))

        def forward(self, x):
            h = self.features(x)
            e = torch.nn.functional.normalize(h, dim=1)
            return self.head(h), e
    return Net()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _stack(bursts, labels, window, hop, max_windows, fs):
    xs, ys, owner = [], [], []
    for k, (x, y) in enumerate(zip(bursts, labels)):
        w = windows(x, window, hop, max_windows, fs)
        xs.append(w)
        ys += [int(y)] * len(w)
        owner += [k] * len(w)
    return np.concatenate(xs), np.array(ys), np.array(owner)


def _burst_features(net, bursts, window, hop, max_windows, fs):
    import torch
    feats, logits = [], []
    with torch.no_grad():
        for x in bursts:
            w = torch.from_numpy(windows(x, window, hop, max_windows, fs))
            lg, e = net(w)                      # the exported outputs
            feats.append(e.mean(0).numpy())
            logits.append(lg.mean(0).numpy())
    return np.array(feats, dtype=np.float64), np.array(logits, dtype=np.float64)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _cos_dist(f: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Cosine distance of one embedding to every unit class mean."""
    return 1.0 - _unit(np.asarray(f, dtype=np.float64)) @ means.T


def _open_set(feats: np.ndarray, labels: np.ndarray, n_classes: int,
              accept: float = 0.95) -> dict:
    """Unit class means from held-out bursts; the threshold is the `accept`
    quantile of LEAVE-ONE-OUT distances (each burst against its class mean
    computed without it), so it is set on distances the model has the same
    relationship to as a new burst's. An in-sample threshold is always too
    tight."""
    feats = np.asarray(feats, dtype=np.float64)
    means = np.stack([_unit(feats[labels == c].mean(0)) for c in range(n_classes)])
    d = []
    for k, (f, c) in enumerate(zip(feats, labels)):
        sel = labels == c
        sel[k] = False
        m = means.copy()
        if sel.any():
            m[c] = _unit(feats[sel].mean(0))
        d.append(float(_cos_dist(f, m)[c]))
    return {"means": means.tolist(), "distance": "cosine",
            "threshold": float(np.quantile(d, accept, method="higher")),
            "accept_target": accept, "n": int(len(feats)),
            "note": "threshold from leave-one-out held-out distances"}


def train(bursts, labels, out_dir, *, profile: str, class_names=None,
          fs: float = 48_000.0, window: int = 256, hop: int | None = None,
          max_windows: int = 8, steps: int = 400, batch: int = 64,
          lr: float = 2e-3, width: int = 32, seed: int = 0,
          heldout: tuple | None = None, unknown=None, threads: int = 1,
          name: str = "fingerprint-cnn", progress=None) -> Path:
    """Train on bursts (complex IQ at the profile's canonical rate) with
    integer labels. `heldout` = (bursts, labels) of the same radios, not
    trained on: sets the open-set threshold and the card's accuracy;
    `unknown` = bursts of radios never enrolled: the card's measured
    unknown rejection."""
    import torch
    torch.set_num_threads(int(threads))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=np.int64)
    n_classes = int(labels.max()) + 1
    names = list(class_names or [f"radio-{c}" for c in range(n_classes)])
    X, Y, _ = _stack(bursts, labels, window, hop, max_windows, fs)
    net = _build(n_classes, width)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    losses = []
    t0 = time.time()
    net.train()
    for it in range(int(steps)):
        idx = rng.integers(0, len(X), batch)
        xb = torch.from_numpy(random_channel(X[idx], rng, fs=fs))
        logits, _ = net(xb)
        loss = lossf(logits, torch.from_numpy(Y[idx]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if progress and (it + 1) % 100 == 0:
            progress(f"fingerprint CNN: step {it + 1}/{steps}, loss "
                     f"{np.mean(losses[-50:]):.3f}")
    net.eval()
    hb, hl = heldout if heldout is not None else (bursts, labels)
    hl = np.asarray(hl, dtype=np.int64)
    feats, logit = _burst_features(net, hb, window, hop, max_windows, fs)
    calib = _open_set(feats, hl, n_classes)
    metrics = {"train_loss_first": float(np.mean(losses[:20])),
               "train_loss_last": float(np.mean(losses[-20:])),
               "heldout_accuracy": float(np.mean(logit.argmax(1) == hl)),
               "heldout_is_training_data": heldout is None}
    if unknown:
        uf, _ = _burst_features(net, unknown, window, hop, max_windows, fs)
        means = np.array(calib["means"])
        metrics["unknown_rejection"] = float(np.mean(
            [_cos_dist(f, means).min() > calib["threshold"] for f in uf]))
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), d / "model.pt")
    counts = np.bincount(labels, minlength=n_classes)
    card = _cards.new_card(name, KIND, profile,
                           input={"window": window, "hop": hop or window,
                                  "max_windows": max_windows, "fs": fs,
                                  "normalize": "unit RMS per window",
                                  "width": width,
                                  "augmentation": "random gain/phase, 3-tap "
                                                  "multipath, +/-20 Hz drift, "
                                                  "5-30 dB SNR per window"},
                           classes=[{"name": names[c], "source": "trained",
                                     "examples": int(counts[c])}
                                    for c in range(n_classes)],
                           calibration={"open_set": calib},
                           trained_on=f"{steps} steps, {time.time() - t0:.0f} s CPU",
                           notes=["outputs are PROPOSED tier: an identification "
                                  "nobody has confirmed",
                                  "classical baseline: fingerprint.features + "
                                  "fingerprint.library on the same bursts"])
    card.metrics = metrics
    _cards.save(d, card, "model.pt")
    return d


class FingerprintModel:
    def __init__(self, net, card):
        self.net = net
        self.card = card
        inp = card.input
        self.window, self.hop = int(inp["window"]), int(inp["hop"])
        self.max_windows, self.fs = int(inp["max_windows"]), float(inp["fs"])
        os_ = card.calibration["open_set"]
        self.means = np.array(os_["means"])
        self.threshold = float(os_["threshold"])
        self.names = card.class_names()

    def identify(self, bursts) -> list[dict]:
        """[{class, name, distance, threshold, confidence}] per burst; the
        class is UNKNOWN beyond the open-set threshold."""
        feats, logits = _burst_features(self.net, bursts, self.window, self.hop,
                                        self.max_windows, self.fs)
        out = []
        for f, lg in zip(feats, logits):
            d = _cos_dist(f, self.means)
            c = int(np.argmin(d))
            p = np.exp(lg - lg.max())
            p = p / p.sum()
            unknown = float(d.min()) > self.threshold
            out.append({"class": UNKNOWN if unknown else c,
                        "name": UNKNOWN if unknown else self.names[c],
                        "distance": float(d.min()), "threshold": self.threshold,
                        "confidence": float(p[c]), "tier": "proposed"})
        return out


def load(model_dir, profile: str) -> FingerprintModel:
    """Through the card: a fingerprint model for exactly this profile."""
    import torch
    card = _cards.load(model_dir, expect_kind=KIND, for_profile=profile)
    net = _build(len(card.classes), int(card.input.get("width", 32)))
    net.load_state_dict(torch.load(Path(model_dir) / card.weights["file"],
                                   map_location="cpu", weights_only=True))
    return FingerprintModel(net.eval(), card)


def holds_burst(y, fs: float, smooth_ms: float = 2.0, rise_db: float = 6.0,
                min_ms: float = 2.0) -> bool:
    """Does `y` hold a burst: a run of at least `min_ms` whose smoothed power
    is `rise_db` over the capture's own median?

    The hallucination check needs its own test, not `burst_bounds`:
    burst_bounds assumes WHITE noise, and any smoothing denoiser colours the
    noise, which widens its power fluctuations until burst_bounds "finds" a
    burst in a moving average of pure noise (it did — 2026-10-09). A burst
    that is 6 dB over the median for 2 ms is not something stationary noise
    of any colour does once smoothed over 2 ms."""
    y = np.asarray(y)
    k = max(1, int(round(smooth_ms * 1e-3 * fs)))
    p = np.convolve(np.abs(y) ** 2, np.ones(k) / k, mode="valid")
    if p.size == 0:
        return False
    med = max(float(np.median(p)), 1e-30)
    hot = p > med * 10 ** (rise_db / 10)
    need = max(1, int(round(min_ms * 1e-3 * fs)))
    run = 0
    for h in hot:
        run = run + 1 if h else 0
        if run >= need:
            return True
    return False


def _noise_hallucination(denoiser, fs: float, n: int = 6, seed: int = 0,
                         length: int = 9600) -> float:
    """How often the denoiser's output of NOISE ALONE holds a burst. The
    input noise is checked too: a check that fires on its own input is a
    broken check, not a hallucinating denoiser."""
    rng = np.random.default_rng(seed + 77)
    hits = 0
    for _ in range(n):
        z = (rng.normal(size=length) + 1j * rng.normal(size=length)).astype(np.complex64)
        if holds_burst(z, fs):
            continue
        if holds_burst(np.asarray(denoiser(z, fs)), fs):
            hits += 1
    return hits / n


def evaluate(model: FingerprintModel, bursts, labels, snrs, *, denoiser=None,
             unknown=None, unknown_snr=None) -> dict:
    """Accuracy per SNR (and unknown rejection), raw — or `denoiser`ed first,
    with the denoiser's hallucination rate on noise-only captures."""
    if denoiser is not None:
        bursts = [denoiser(np.asarray(x), model.fs) for x in bursts]
        if unknown:
            unknown = [denoiser(np.asarray(x), model.fs) for x in unknown]
    res = model.identify(bursts)
    labels = np.asarray(labels)
    snrs = np.asarray(snrs, dtype=np.float64)
    per = {}
    for s in sorted(set(snrs.tolist())):
        sel = np.nonzero(snrs == s)[0]
        per[f"{s:g}"] = {"accuracy": float(np.mean([res[i]["class"] == labels[i]
                                                    for i in sel])),
                         "n": int(sel.size)}
    out = {"per_snr": per,
           "accuracy": float(np.mean([r["class"] == l for r, l in zip(res, labels)]))}
    if unknown:
        ur = model.identify(unknown)
        rej = [r["class"] == UNKNOWN for r in ur]
        out["unknown_rejection"] = float(np.mean(rej))
        if unknown_snr is not None:
            us = np.asarray(unknown_snr, dtype=np.float64)
            for s in per:
                sel = np.nonzero(us == float(s))[0]
                per[s]["unknown_rejected"] = float(np.mean([rej[i] for i in sel])) \
                    if sel.size else None
    if denoiser is not None:
        out["denoiser"] = getattr(denoiser, "__name__", type(denoiser).__name__)
        out["denoiser_hallucination_rate"] = _noise_hallucination(denoiser, model.fs)
    return out


def export_onnx(model_dir, profile: str, opset: int = 17) -> Path:
    """fingerprint.onnx: in `iq` [B, 2, L]; out `logits` [B, C], `embedding`
    [B, D] — the classifier's I/O (ARCHITECTURE §5)."""
    import torch
    m = load(model_dir, profile)
    x = torch.zeros(1, 2, m.window)
    out = Path(model_dir) / "fingerprint.onnx"
    torch.onnx.export(m.net, (x,), str(out), input_names=["iq"],
                      output_names=["logits", "embedding"], opset_version=opset,
                      dynamic_axes={"iq": {0: "b"}, "logits": {0: "b"},
                                    "embedding": {0: "b"}}, dynamo=False)
    m.card.input["onnx"] = {"file": out.name, "sha256": _prov.sha256_path(out)}
    _cards.save(model_dir, m.card, m.card.weights["file"])
    return out
