# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Model cards. A model without a card does not load (DETECTION_DESIGN §6.5).

A card says what a model is FOR, so the toolkit can refuse it for anything
else: the receiver profile and exact rate it was trained at, the STFT / FAM
geometry it has seen, its classes (trained or taught, with example counts),
the hashes of its datasets and weights, and the numbers that say how far to
trust it — the domain gap, false alarms per hour, unknown rejection, the
hallucination rate, the CPU latency. Plan §6–7.

    card = cards.load(model_dir, expect_kind="proposer2d",
                      for_profile="rtlsdr_2400000_cu8")

`load` refuses, in words, when the card is missing, malformed, for another
profile, for another kind, or when the weights on disk are not the weights
the card describes.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from atk_diffusion import profiles as _profiles
from atk_diffusion.provenance import TIERS, sha256_path

CARD_FILE = "card.json"

#: kind -> (what it is, the tier of what it outputs, whether it is per-profile)
KINDS = {
    "proposer2d":    ("the 2D AI proposer (boxes on the spectrogram)", "proposed", True),
    "classifier1d":  ("the 1D classifier (IQ + SCF -> class, embedding)", "proposed", True),
    "ssl_backbone":  ("a self-supervised backbone", "measured", True),
    "denoiser":      ("the diffusion denoiser (plan B3)", "invented", True),
    "inpainter":     ("the IQ dropout inpainter (plan D1)", "invented", True),
    "translator":    ("the receiver-to-receiver translator (plan A6)", "invented", False),
    "augmenter":     ("the time-frequency diffusion augmenter (plan B4)", "invented", True),
    "fingerprint":   ("the RF fingerprint embedder (plan C1)", "proposed", True),
    "radiomap":      ("the radio-map residual model (plan E2/E5)", "invented", False),
    "position":      ("the where-am-I position model (plan E3)", "invented", False),
    "speech_enhance": ("the speech enhancer before Whisper (plan D2)", "invented", False),
    "vitals":        ("the CSI vital-signs model (plan I1)", "measured", False),
    "presence":      ("the presence/motion model (plan I2)", "proposed", False),
    "genclass":      ("generative classification (research)", "proposed", True),
    "beacon":        ("the co-designed beacon (research)", "measured", True),
    "anomaly":       ("the flow-anomaly model (research)", "proposed", False),
    "track_inpaint": ("the track inpainter (plan D4)", "invented", False),
    "text_diffusion": ("a text diffusion backend (plan F)", "invented", False),
}

REQUIRED = ("name", "kind", "created", "weights")


class CardRefusal(ValueError):
    """The model may not be used here. The message says why, in words."""


@dataclass
class ModelCard:
    name: str
    kind: str
    profile: str = ""                 # receiver profile id ('' = not per-profile)
    sample_rate: float = 0.0
    input: dict = field(default_factory=dict)       # stft / fam / canonical / window
    classes: list = field(default_factory=list)     # {name, source, examples}
    datasets: list = field(default_factory=list)    # {name, sha256, n, kind}
    metrics: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)     # {file, sha256, format, bytes}
    tier: str = ""
    license: str = ""
    created: str = ""
    code_version: str = ""
    trained_on: str = ""
    notes: list = field(default_factory=list)

    # -- convenience -----------------------------------------------------------
    def class_names(self) -> list[str]:
        return [c["name"] if isinstance(c, dict) else str(c)
                for c in self.classes]

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "ModelCard":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


def new_card(name: str, kind: str, profile: str = "", **kw) -> ModelCard:
    if kind not in KINDS:
        raise ValueError(f"unknown model kind {kind!r}")
    from atk_diffusion import __version__
    tier = KINDS[kind][1]
    sr = 0.0
    if profile:
        sr = float(_profiles.parse_profile_id(profile).sample_rate)
    card = ModelCard(name=name, kind=kind, profile=str(profile).lower(),
                     sample_rate=sr, tier=tier,
                     created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     code_version=__version__)
    for k, v in kw.items():
        setattr(card, k, v)
    return card


def save(model_dir, card: ModelCard, weights_file: str | None = None) -> Path:
    """Write card.json; hash the weights file into it if one is named."""
    d = Path(model_dir)
    d.mkdir(parents=True, exist_ok=True)
    if weights_file:
        w = d / weights_file
        fmt = w.suffix.lstrip(".").lower()
        card.weights = {"file": weights_file, "sha256": sha256_path(w),
                        "bytes": w.stat().st_size, "format": fmt}
    problems = validate(card)
    if problems:
        raise CardRefusal("the card is incomplete: " + "; ".join(problems))
    p = d / CARD_FILE
    p.write_text(json.dumps(card.to_json(), indent=2), encoding="utf-8")
    return p


def validate(card: ModelCard) -> list[str]:
    out = []
    for k in REQUIRED:
        if not getattr(card, k, None):
            out.append(f"no {k}")
    if card.kind and card.kind not in KINDS:
        out.append(f"unknown kind {card.kind!r}")
    if card.tier and card.tier not in TIERS:
        out.append(f"unknown tier {card.tier!r}")
    per_profile = KINDS.get(card.kind, ("", "", False))[2]
    if per_profile and not card.profile:
        out.append(f"a {card.kind} model is per receiver profile and the card "
                   "names none")
    if card.profile:
        try:
            pid = _profiles.parse_profile_id(card.profile)
            if card.sample_rate and abs(float(card.sample_rate)
                                        - pid.sample_rate) > 0.5:
                out.append(f"the card's sample rate {card.sample_rate:g} is "
                           f"not its profile's ({pid.sample_rate})")
        except ValueError as e:
            out.append(str(e))
    w = card.weights or {}
    if w and not w.get("file"):
        out.append("weights has no file")
    return out


def load(model_dir, expect_kind: str | None = None,
         for_profile: str | None = None, verify_weights: bool = True
         ) -> ModelCard:
    """Read and check a card. Raises CardRefusal / ProfileMismatch."""
    d = Path(model_dir)
    p = d / CARD_FILE
    if not p.exists():
        raise CardRefusal(f"{d.name} has no {CARD_FILE}. A model without a "
                          "card does not load — its profile, rate and "
                          "measured accuracy are unknown.")
    try:
        card = ModelCard.from_json(json.loads(p.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, TypeError) as e:
        raise CardRefusal(f"{d.name}'s card could not be read: {e}") from None
    problems = validate(card)
    if problems:
        raise CardRefusal(f"{d.name}'s card is incomplete: "
                          + "; ".join(problems))
    if expect_kind and card.kind != expect_kind:
        raise CardRefusal(f"{d.name} is {KINDS.get(card.kind, (card.kind,))[0]}, "
                          f"not {KINDS.get(expect_kind, (expect_kind,))[0]}.")
    if for_profile and card.profile:
        _profiles.check_match(card.profile, for_profile,
                              what=f"this {card.kind} model ({card.name})")
    if verify_weights and card.weights:
        w = d / card.weights["file"]
        if not w.exists():
            raise CardRefusal(f"{d.name}'s weights file {w.name} is missing.")
        if card.weights.get("bytes") and w.stat().st_size != card.weights["bytes"]:
            raise CardRefusal(f"{w.name} is not the file the card describes "
                              "(its size changed). Retrain or restore it.")
        if card.weights.get("sha256") and sha256_path(w) != card.weights["sha256"]:
            raise CardRefusal(f"{w.name} is not the file the card describes "
                              "(its hash changed). Retrain or restore it.")
    return card


def weights_path(model_dir, card: ModelCard) -> Path:
    return Path(model_dir) / card.weights["file"]


def find(rf, profile: str, kind: str) -> list[tuple[Path, ModelCard]]:
    """Every loadable model of `kind` for `profile` under rf_data, newest
    first. Unloadable ones are skipped (the AI Detect tab lists why)."""
    out = []
    base = rf.models(profile)
    if not base.is_dir():
        return out
    for d in sorted(base.iterdir()):
        if not (d / CARD_FILE).exists():
            continue
        try:
            card = load(d, expect_kind=kind, for_profile=profile,
                        verify_weights=False)
        except (CardRefusal, _profiles.ProfileMismatch, ValueError):
            continue
        out.append((d, card))
    out.sort(key=lambda t: t[1].created, reverse=True)
    return out


def summary(card: ModelCard) -> list[str]:
    """Lines for the AI Detect tab: what it is, what for, how good."""
    what = KINDS.get(card.kind, (card.kind,))[0]
    lines = [f"{card.name} — {what}"]
    if card.profile:
        lines.append(f"for {_profiles.describe(card.profile)} "
                     f"({card.profile})")
    names = card.class_names()
    if names:
        taught = [c["name"] for c in card.classes
                  if isinstance(c, dict) and c.get("source") == "taught"]
        lines.append(f"{len(names)} classes"
                     + (f", {len(taught)} taught" if taught else ""))
    m = card.metrics or {}
    for key, label in (("domain_gap", "domain gap"),
                       ("map_cabled", "mAP on cabled captures"),
                       ("false_alarms_per_hour", "false alarms / hour"),
                       ("unknown_rejection", "unknown rejection"),
                       ("hallucination_rate", "hallucination rate"),
                       ("latency_ms", "CPU latency (ms)")):
        if key in m and m[key] is not None:
            v = m[key]
            lines.append(f"{label}: {v:.3g}" if isinstance(v, (int, float))
                         else f"{label}: {v}")
    if not m:
        lines.append("not yet measured — no number says how far to trust it")
    lines.append(f"outputs are {card.tier.upper()}")
    return lines
