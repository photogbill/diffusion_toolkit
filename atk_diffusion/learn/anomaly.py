# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""CyberWolf anomaly scores by reconstruction error on flow features — under
CyberWolf's rule: CONTEXT, NEVER SUPPRESSION (plan §4.R).

The plan keeps this on the research track *"only under CyberWolf's context,
never suppression rule, and only after its false-positive history is
understood to be the detectors being right."* That history is understood
(ATK's FUTURE_PLANS, 2026-09-03): the Microsoft "false positives" were the
detectors being right — machine-generated, chatty infrastructure really does
look like a DGA and a C2 beacon — and the answer was context on the finding,
not a filter. In CyberWolf's own words: *"the finding keeps its severity and
its place, and the analyst gets the sentence that lets them weigh it."*

SO THIS MODULE CAN ONLY ADD CONTEXT. `annotate(findings, flows, model)`
returns exactly one finding for every finding it was given, in the same
order, with every field it had unchanged — severity included — and one more
entry in its `context` list: how predictable that flow's features were to a
model of the baseline traffic. There is no threshold that drops a finding,
no filter, no re-ranking, no flag that hides — not as an option, not as a
default. A low score does not mean "benign" and the words never say so; a
high score does not raise severity. The analyst weighs it.

THE MODEL. Reconstruction error of a flow's features under a model of
normal (baseline) traffic — the plan's mechanism table: "what is new" is
what the model of normal could not predict.

* `PcaModel` — the CLASSICAL baseline (plan §7): principal components of
  the baseline's standardised features; the error is what the components
  cannot reconstruct. numpy only.
* `AutoencoderModel` — a small PyTorch autoencoder (PyTorch inside its
  methods only), the learned comparator. It is kept only if it separates
  better than PCA on the same flows (`compare`).

Each score is turned into words against the baseline's own errors: "less
predictable than 99.3 % of the baseline flows". Models are saved and loaded
through cards (kind "anomaly", tier PROPOSED).

LIMITS. A flow model knows only the features it is given (sizes, packets,
timing, port class); it sees nothing of payload or reputation, and a slow,
quiet intruder that looks like the baseline scores like it — which is one
more reason the score is context and never a filter.
"""

from __future__ import annotations

import copy
import math
from bisect import bisect_left
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from atk_diffusion import cards as _cards
from atk_diffusion import provenance

provenance.METHOD_TIERS.setdefault("flow_reconstruction", "proposed")

KIND = "anomaly"
TIER = provenance.tier_for("flow_reconstruction")

#: Numeric flow fields used when present (log1p for counts and sizes).
NUMERIC = ("bytes_out", "bytes_in", "packets_out", "packets_in", "duration_s",
           "mean_iat_s", "std_iat_s", "distinct_ports", "connections")
#: Destination port, as its class (a port number is a name, not a quantity).
PORT_CLASSES = ("well_known", "registered", "dynamic", "none")


def featurize(flows, numeric=NUMERIC) -> tuple[np.ndarray, list[str]]:
    """Flow dicts -> (X, feature names). Missing numbers are 0; counts and
    sizes are log1p'd; the destination port is one-hot by class."""
    names = [f"log1p_{k}" for k in numeric] + [f"port_{c}" for c in PORT_CLASSES]
    rows = []
    for f in flows:
        f = f or {}
        v = []
        for k in numeric:
            try:
                x = float(f.get(k, 0.0) or 0.0)
            except (TypeError, ValueError):
                x = 0.0
            v.append(math.log1p(max(0.0, x)))
        port = f.get("dst_port")
        try:
            p = int(port)
            cls = "well_known" if p < 1024 else "registered" if p < 49152 else "dynamic"
        except (TypeError, ValueError):
            cls = "none"
        v += [1.0 if c == cls else 0.0 for c in PORT_CLASSES]
        rows.append(v)
    return np.asarray(rows, dtype=np.float64).reshape(len(rows), len(names)), names


class _Calibrated:
    """Shared: standardisation and the baseline's error distribution."""
    kind_name = ""

    def _standardise_fit(self, X):
        self.mu = X.mean(axis=0)
        sd = X.std(axis=0)
        self.sd = np.where(sd > 1e-9, sd, 1.0)

    def _z(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mu) / self.sd

    def _calibrate(self, X):
        self.baseline_errors = sorted(float(e) for e in self.errors(X))

    def percentile(self, err: float) -> float:
        """The share of baseline flows that reconstructed better (0–100)."""
        b = self.baseline_errors
        if not b:
            return 0.0
        return 100.0 * bisect_left(b, float(err)) / len(b)


class PcaModel(_Calibrated):
    """The classical baseline: reconstruction by the top principal
    components of the baseline's standardised features."""
    kind_name = "pca"

    def __init__(self, components: int = 3):
        self.components = int(components)

    def fit(self, X) -> "PcaModel":
        X = np.asarray(X, dtype=np.float64)
        if X.shape[0] < 10:
            raise ValueError("a baseline needs at least 10 flows")
        self._standardise_fit(X)
        Z = self._z(X)
        _u, _s, vt = np.linalg.svd(Z, full_matrices=False)
        self.V = vt[:min(self.components, vt.shape[0])].T
        self._calibrate(X)
        return self

    def errors(self, X) -> np.ndarray:
        Z = self._z(X)
        R = Z @ self.V @ self.V.T
        return np.sqrt(np.mean((Z - R) ** 2, axis=1))

    def state(self) -> dict:
        return {"mu": self.mu, "sd": self.sd, "V": self.V,
                "baseline_errors": np.asarray(self.baseline_errors)}

    @classmethod
    def from_state(cls, st) -> "PcaModel":
        m = cls(int(st["V"].shape[1]))
        m.mu, m.sd, m.V = st["mu"], st["sd"], st["V"]
        m.baseline_errors = [float(x) for x in st["baseline_errors"]]
        return m


class AutoencoderModel(_Calibrated):
    """The learned comparator: a small autoencoder on the same features."""
    kind_name = "autoencoder"

    def __init__(self, bottleneck: int = 3, hidden: int = 16, epochs: int = 200,
                 lr: float = 1e-2, seed: int = 0):
        self.bottleneck, self.hidden = int(bottleneck), int(hidden)
        self.epochs, self.lr, self.seed = int(epochs), float(lr), int(seed)
        self.net = None

    def _make(self, d: int):
        import torch
        from torch import nn
        torch.manual_seed(self.seed)
        return nn.Sequential(nn.Linear(d, self.hidden), nn.Tanh(),
                             nn.Linear(self.hidden, self.bottleneck), nn.Tanh(),
                             nn.Linear(self.bottleneck, self.hidden), nn.Tanh(),
                             nn.Linear(self.hidden, d))

    def fit(self, X) -> "AutoencoderModel":
        import torch
        torch.set_num_threads(1)
        X = np.asarray(X, dtype=np.float64)
        if X.shape[0] < 10:
            raise ValueError("a baseline needs at least 10 flows")
        self._standardise_fit(X)
        Z = torch.from_numpy(self._z(X).astype(np.float32))
        self.net = self._make(Z.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = torch.mean((self.net(Z) - Z) ** 2)
            loss.backward()
            opt.step()
        self.net.eval()
        self._calibrate(X)
        return self

    def errors(self, X) -> np.ndarray:
        import torch
        Z = torch.from_numpy(self._z(X).astype(np.float32))
        with torch.no_grad():
            R = self.net(Z)
        return torch.sqrt(torch.mean((R - Z) ** 2, dim=1)).numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Context — the only output
# ---------------------------------------------------------------------------
@dataclass
class Context:
    kind: str
    model: str
    score: float
    percentile: float
    words: str
    tier: str = TIER

    def to_json(self) -> dict:
        return asdict(self)


def context_for(flow: dict, model, model_name: str = "") -> Context:
    """The sentence for one flow. Never "benign", never "safe": the finding
    stands as raised, and this is how predictable its traffic was."""
    X, _ = featurize([flow])
    err = float(model.errors(X)[0])
    pct = model.percentile(err)
    name = model_name or getattr(model, "kind_name", "model")
    if pct >= 99.0:
        how = (f"its flow features were less predictable than {pct:.1f}% of "
               "the baseline flows — unusual for this network")
    elif pct >= 90.0:
        how = (f"its flow features were less predictable than {pct:.0f}% of "
               "the baseline flows")
    else:
        how = (f"its flow features look like the baseline's (less predictable "
               f"than only {pct:.0f}% of them)")
    words = (f"Context, not a verdict: {how}. The finding stands as raised; "
             f"its severity is unchanged. (reconstruction error {err:.3g}, "
             f"{name})")
    return Context("reconstruction_error", name, err, pct, words)


def annotate(findings, flows, model, model_name: str = "") -> list:
    """One finding out for every finding in, same order, every field as it
    was — plus one context entry. `flows[i]` is the flow behind
    `findings[i]` (None: the finding gets a context entry saying no flow was
    available). Nothing is dropped, hidden, re-ranked or re-rated."""
    findings = list(findings)
    flows = list(flows)
    if len(flows) != len(findings):
        raise ValueError(f"{len(findings)} findings but {len(flows)} flows — "
                         "give one flow (or None) per finding")
    out = []
    for f, flow in zip(findings, flows):
        if flow is None:
            ctx = Context("reconstruction_error", model_name or
                          getattr(model, "kind_name", "model"), float("nan"),
                          float("nan"), "Context: no flow features were "
                          "available for this finding. The finding stands as "
                          "raised.")
        else:
            ctx = context_for(flow, model, model_name)
        g = copy.deepcopy(f)
        if isinstance(g, dict):
            g["context"] = list(g.get("context") or []) + [ctx.to_json()]
        else:
            existing = list(getattr(g, "context", None) or [])
            setattr(g, "context", existing + [ctx.to_json()])
        out.append(g)
    return out


def compare(baseline_flows, normal_flows, unusual_flows, *,
            components: int = 3, use_autoencoder: bool = True) -> dict:
    """PCA (classical) versus the autoencoder: how far each puts known
    unusual flows above held-out normal ones (the AUC of the error). The
    learned model is kept only if it separates better (plan §7)."""
    Xb, _ = featurize(baseline_flows)
    Xn, _ = featurize(normal_flows)
    Xu, _ = featurize(unusual_flows)

    def auc(m):
        a, b = m.errors(Xu), m.errors(Xn)
        return float(np.mean(a[:, None] > b[None, :])
                     + 0.5 * np.mean(a[:, None] == b[None, :]))
    out = {"pca_auc": auc(PcaModel(components).fit(Xb)), "tier": TIER,
           "what": "AUC of reconstruction error, unusual versus normal "
                   "held-out flows (1.0 = perfectly separated)"}
    if use_autoencoder:
        try:
            out["autoencoder_auc"] = auc(AutoencoderModel(components).fit(Xb))
            out["keep_learned"] = out["autoencoder_auc"] > out["pca_auc"]
        except ImportError:
            out["autoencoder_auc"] = None
            out["keep_learned"] = False
            out["why"] = "PyTorch is not in this environment"
    return out


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------
def save(model, model_dir, name: str = "flow-anomaly", trained_on: str = "") -> Path:
    """Through a card (kind "anomaly"); PCA as .npz, the autoencoder as .pt."""
    d = Path(model_dir)
    d.mkdir(parents=True, exist_ok=True)
    common = {"mu": np.asarray(model.mu), "sd": np.asarray(model.sd),
              "baseline_errors": np.asarray(model.baseline_errors)}
    if isinstance(model, PcaModel):
        fname = "pca.npz"
        np.savez(d / fname, V=model.V, **common)
        inp = {"model": "pca", "components": int(model.V.shape[1])}
    else:
        import torch
        fname = "autoencoder.pt"
        torch.save({"state": model.net.state_dict(),
                    **{k: torch.from_numpy(np.asarray(v, dtype=np.float64))
                       for k, v in common.items()}}, d / fname)
        inp = {"model": "autoencoder", "bottleneck": model.bottleneck,
               "hidden": model.hidden, "d": int(np.asarray(model.mu).size)}
    inp["features"] = list(NUMERIC) + [f"port_{c}" for c in PORT_CLASSES]
    card = _cards.new_card(name, KIND, "", input=inp, license="all rights reserved",
                           trained_on=trained_on or "baseline flows",
                           notes=["context, never suppression: scores annotate "
                                  "findings and never drop, hide or re-rate one"])
    return _cards.save(d, card, fname)


def load(model_dir):
    card = _cards.load(model_dir, expect_kind=KIND)
    p = Path(model_dir) / card.weights["file"]
    if card.input.get("model") == "pca":
        z = np.load(p)
        return PcaModel.from_state({k: z[k] for k in z.files})
    import torch
    st = torch.load(p, map_location="cpu", weights_only=True)
    m = AutoencoderModel(card.input["bottleneck"], card.input["hidden"])
    m.mu = st["mu"].numpy()
    m.sd = st["sd"].numpy()
    m.baseline_errors = [float(x) for x in st["baseline_errors"].numpy()]
    m.net = m._make(int(card.input["d"]))
    m.net.load_state_dict(st["state"])
    m.net.eval()
    return m
