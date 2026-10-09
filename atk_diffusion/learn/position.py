# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Where am I, learned — a mixture-density network from RF fingerprint to
position (plan E3, the learned half beside `geo.whereami`).

Input: the same features the classical locators use (per-cell RSRP, FM
RSSI, in dB; "not heard" encoded as the floor plus a missing flag). Output:
a mixture of K Gaussians over position (local east/north kilometres around
the training area) — a POSTERIOR, multi-modal when the fingerprint is, not
a regression to one point. Trained by negative log-likelihood of the GPS
truth on Bill's drive data; its sigmas are then scaled on a held-out route
so the stated regions hold the truth at their stated rate (the same proper
scoring rule the classical locators are calibrated with), and `evaluate`
reports on another held-out route — error distribution and coverage — in
the same words (`geo.whereami.summarize`), so the two can be compared
like for like. The classical locators are the baseline it must beat.

Card kind "position" (INVENTED tier: a learned model's output); the
features, floors, reference point, mixture size and the calibration travel
in the card, and a model without its card does not load.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from atk_diffusion import cards as _cards
from atk_diffusion.geo import posterior as _post
from atk_diffusion.geo.products import GeoGrid
from atk_diffusion.geo.terrain import enu_m, from_enu, haversine_m

KIND = "position"
METHOD = "learned_position"
LEVELS = (0.5, 0.9, 0.95)


def encode(features: dict, keys, floors) -> np.ndarray:
    """dB features -> (value above floor / 30, missing flag) per key."""
    F = len(keys)
    out = np.zeros(2 * F, dtype=np.float32)
    for j, k in enumerate(keys):
        v = features.get(k)
        if v is None or not math.isfinite(float(v)):
            out[F + j] = 1.0
        else:
            out[j] = (float(v) - float(floors[j])) / 30.0
    return out


def _build(n_in: int, hidden: int, K: int):
    from torch import nn
    return nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(),
                         nn.Linear(hidden, hidden), nn.SiLU(),
                         nn.Linear(hidden, K * 5))


def _split(out, K: int):
    import torch
    o = out.view(-1, K, 5)
    logits = o[..., 0]
    mu = o[..., 1:3]
    log_s = torch.clamp(o[..., 3:5], -6.0, 3.0)
    return logits, mu, log_s


def _nll(out, y, K: int, scale: float = 1.0):
    import torch
    logits, mu, log_s = _split(out, K)
    log_s = log_s + math.log(scale)
    z = (y[:, None, :] - mu) / torch.exp(log_s)
    comp = -0.5 * (z ** 2).sum(-1) - log_s.sum(-1) - math.log(2 * math.pi)
    return -torch.logsumexp(torch.log_softmax(logits, -1) + comp, -1)


class PositionModel:
    """A trained MDN with its card: `locate(features)` -> GridPosterior."""

    def __init__(self, net, card):
        self.net = net
        self.card = card
        inp = card.input
        self.keys = list(inp["keys"])
        self.floors = np.asarray(inp["floors"], dtype=np.float64)
        self.lat0, self.lon0 = float(inp["ref_lat"]), float(inp["ref_lon"])
        self.K = int(inp["components"])
        self.sigma_scale = float(card.calibration.get("sigma_scale", 1.0))

    def mixture(self, features):
        """(weights (K,), means (K, 2) metres east/north, sigmas (K, 2) m)."""
        import torch
        x = torch.from_numpy(encode(features, self.keys, self.floors)[None])
        with torch.no_grad():
            logits, mu, log_s = _split(self.net(x), self.K)
        w = torch.softmax(logits, -1)[0].numpy().astype(np.float64)
        return (w, mu[0].numpy().astype(np.float64) * 1000.0,
                np.exp(log_s[0].numpy().astype(np.float64)) * 1000.0 * self.sigma_scale)

    def _density(self, w, mu, sg, e, n):
        d = np.zeros(np.broadcast(e, n).shape)
        for k in range(w.size):
            z = ((e - mu[k, 0]) / sg[k, 0]) ** 2 + ((n - mu[k, 1]) / sg[k, 1]) ** 2
            d = d + w[k] * np.exp(-0.5 * z) / (2 * math.pi * sg[k, 0] * sg[k, 1])
        return d

    def locate(self, features, max_side: int = 200) -> _post.GridPosterior:
        w, mu, sg = self.mixture(features)
        keep = w > 1e-4
        lo_e = np.min(mu[keep, 0] - 4 * sg[keep, 0])
        hi_e = np.max(mu[keep, 0] + 4 * sg[keep, 0])
        lo_n = np.min(mu[keep, 1] - 4 * sg[keep, 1])
        hi_n = np.max(mu[keep, 1] + 4 * sg[keep, 1])
        cell = max(max(hi_e - lo_e, hi_n - lo_n) / max_side,
                   float(np.min(sg[keep])) / 3.0, 2.0)
        lat_a, lon_a = from_enu(np.array([lo_e, hi_e]), np.array([lo_n, hi_n]),
                                self.lat0, self.lon0)
        grid = GeoGrid.covering(lat_a, lon_a, 0.0, cell)
        LAT, LON = grid.mesh()
        e, n = enu_m(LAT, LON, self.lat0, self.lon0)
        with np.errstate(divide="ignore"):
            logp = np.log(self._density(w, mu, sg, e, n))
        return _post.GridPosterior(grid, logp, method=METHOD,
                                   meta={"model": self.card.name,
                                         "components": self.K,
                                         "sigma_scale": self.sigma_scale})

    def credible_level(self, features, lat, lon, n: int = 2000, rng=None) -> float:
        """HPD level of a point, by Monte Carlo from the mixture."""
        rng = rng if rng is not None else np.random.default_rng(0)
        w, mu, sg = self.mixture(features)
        k = rng.choice(w.size, size=n, p=w / w.sum())
        se = mu[k, 0] + rng.normal(size=n) * sg[k, 0]
        sn = mu[k, 1] + rng.normal(size=n) * sg[k, 1]
        e, nn = enu_m(lat, lon, self.lat0, self.lon0)
        dt = self._density(w, mu, sg, e, nn)
        return float(np.mean(self._density(w, mu, sg, se, sn) > dt))

    def evaluate(self, records, levels=LEVELS) -> dict:
        from atk_diffusion.geo.whereami import summarize
        e_map, e_mean, cls_ = [], [], []
        rng = np.random.default_rng(0)
        for r in records:
            w, mu, sg = self.mixture(r.features)
            # MAP among the component means (a mixture's mode is near one)
            dens = [self._density(w, mu, sg, mu[k, 0], mu[k, 1]) for k in range(w.size)]
            k = int(np.argmax(dens))
            la, lo = from_enu(mu[k, 0], mu[k, 1], self.lat0, self.lon0)
            me, mn = float(np.sum(w * mu[:, 0])), float(np.sum(w * mu[:, 1]))
            ma, mo = from_enu(me, mn, self.lat0, self.lon0)
            e_map.append(float(haversine_m(la, lo, r.lat, r.lon)))
            e_mean.append(float(haversine_m(ma, mo, r.lat, r.lon)))
            cls_.append(self.credible_level(r.features, r.lat, r.lon, rng=rng))
        return summarize(np.array(e_map), np.array(e_mean), np.array(cls_), levels)


def train(records, out_dir, *, keys=None, floors=None, calib_records=None,
          components: int = 4, hidden: int = 64, epochs: int = 150,
          lr: float = 3e-3, batch: int = 128, seed: int = 0,
          name: str = "position-mdn", threads: int = 1, progress=None) -> Path:
    """Train on drive records (geo.whereami.DriveRecord); calibrate the
    sigmas on `calib_records` (a held-out route); write weights + card."""
    import torch
    from atk_diffusion.geo.whereami import FingerprintDB
    torch.set_num_threads(int(threads))
    torch.manual_seed(seed)
    recs = list(records)
    db = FingerprintDB.from_records(recs, keys=keys, floors=floors)
    keys, fl = db.keys, db.floors
    lat0, lon0 = float(np.mean(db.lat)), float(np.mean(db.lon))
    X = np.stack([encode(r.features, keys, fl) for r in recs])
    e, n = enu_m(db.lat, db.lon, lat0, lon0)
    Y = np.stack([e, n], axis=1).astype(np.float32) / 1000.0
    net = _build(X.shape[1], hidden, components)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    rng = np.random.default_rng(seed)
    losses = []
    t0 = time.time()
    for ep in range(int(epochs)):
        order = rng.permutation(len(X))
        for s in range(0, len(X), batch):
            idx = torch.from_numpy(order[s:s + batch])
            loss = _nll(net(Xt[idx]), Yt[idx], components).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        if progress and (ep + 1) % 25 == 0:
            progress(f"position MDN: epoch {ep + 1}/{epochs}, NLL "
                     f"{np.mean(losses[-10:]):.3f}")
    net.eval()
    scale, cal_note = 1.0, "not calibrated: no held-out route given"
    if calib_records:
        cal = list(calib_records)
        Xc = torch.from_numpy(np.stack([encode(r.features, keys, fl) for r in cal]))
        ce, cn = enu_m(np.array([r.lat for r in cal]), np.array([r.lon for r in cal]),
                       lat0, lon0)
        Yc = torch.from_numpy(np.stack([ce, cn], 1).astype(np.float32) / 1000.0)
        with torch.no_grad():
            out = net(Xc)
            best = min(((float(_nll(out, Yc, components, s).mean()), s)
                        for s in np.geomspace(0.5, 10.0, 25)), key=lambda t: t[0])
        scale = float(best[1])
        cal_note = (f"sigmas x {scale:.2f}, chosen on a held-out route of "
                    f"{len(cal)} points by the log score")
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), d / "model.pt")
    card = _cards.new_card(name, KIND, "",
                           input={"keys": keys, "floors": [float(v) for v in fl],
                                  "ref_lat": lat0, "ref_lon": lon0,
                                  "components": int(components),
                                  "hidden": int(hidden),
                                  "encoding": "(dB - floor)/30 and a missing flag"},
                           datasets=[{"name": "drive", "n": len(recs),
                                      "kind": "drive records with GPS truth"}],
                           calibration={"sigma_scale": scale, "note": cal_note},
                           trained_on=f"{epochs} epochs, {time.time() - t0:.0f} s CPU",
                           notes=["outputs are INVENTED tier: a learned posterior",
                                  "classical baselines: geo.whereami.KernelLocator "
                                  "and MapLocator on the same routes"])
    card.metrics = {"train_nll_first": float(np.mean(losses[:10])),
                    "train_nll_last": float(np.mean(losses[-10:]))}
    _cards.save(d, card, "model.pt")
    return d


def load(model_dir) -> PositionModel:
    import torch
    card = _cards.load(model_dir, expect_kind=KIND)
    net = _build(2 * len(card.input["keys"]), int(card.input["hidden"]),
                 int(card.input["components"]))
    net.load_state_dict(torch.load(Path(model_dir) / card.weights["file"],
                                   map_location="cpu", weights_only=True))
    return PositionModel(net.eval(), card)
