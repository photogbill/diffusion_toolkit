# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Detections, tracks, and how a box looks on the waterfall
(DETECTION_DESIGN §1, §5, §12.2; decision D13).

`Detection` is the one currency of the detector: every proposer emits it, the
tracker associates it, the confirmer upgrades it, the cut reads it, and ATK's
waterfall draws it. A detection is PROPOSED until a decoder confirms it, and
a signal with no decoder stays Proposed — with its class and confidence —
forever if need be (plan §2.1, §4.B1).

THE BOXES — SOURCE IS THE EDGE, CLASS IS THE CAPTION, CONFIRMATION IS THE
WEIGHT. Bill, 2026-10-08: *"specific color coded detector boxes on the
waterfall in the event the signal detection is an AI signal detection, and
another if it's a cyclostationary assisted processing result."*

ONE CHANGE TO THE PLAN'S COLOURS, BY THE PLAN'S OWN RULE. §12.2 chose cyan
for cyclostationary and magenta for AI, and said: *"if the technology palette
ever gains a cyan or magenta, the source palette moves, not the technology
one."* It already had one. ATK's waterfall draws an LTE cell in `#3fd0e0` —
CIEDE2000 distance 5.5 from the plan's cyan, which is the same colour to any
eye. A cyan dashed box would read as an LTE tower. So:

    energy           grey-white, dashed      (unchanged — the quiet baseline)
    cyclostationary  VERMILION #ff5400, dashed (was cyan)
    AI               MAGENTA   #ff00dd, dashed (the plan's magenta, pinned)

Chosen by search, not by taste (tests/test_detect_boxes.py repeats it):
against ATK's thirteen technology colours, the processing green, the
selection amber and every stop of the blue-to-yellow waterfall ramp, these
two maximise the smallest perceptual distance (CIEDE2000 ≥ 16 to every
technology colour) while keeping at least 3:1 luminance contrast on the
noise floor — and they stay ≥ 40 apart from each other under simulated
protanopia and deuteranopia, and far from the energy grey. Magenta was
already the best AI colour available; the cyan was the one that had to go.
"""

from __future__ import annotations

import itertools
import math
import uuid
from dataclasses import asdict, dataclass, field

#: Proposer names. Every Detection carries the set that found it.
PROPOSERS = ("energy", "cyclic", "learned")

STATES = ("proposed", "confirmed")

#: Flags a detection can carry (each changes how it is drawn or worded).
FLAGS = ("denoised",      # came through the low-SNR diffusion path (§3)
         "post-clean",    # appeared only after a clean (§4.3)
         "escalated",     # found by cyclic detection over the IQ buffer (§4.3)
         "disagreement",  # proposers or decoder disagree on the class (§5)
         "taught")        # classified by a taught prototype


@dataclass
class Detection:
    """A box in time × frequency, with what found it and what it is.

    Times are seconds from the stream's origin (`t0`/`t1`); `epoch` is the
    wall-clock time of that origin when known, so ATK can map a box onto the
    waterfall rows it was drawn from. Frequencies are ABSOLUTE Hz."""
    t0: float
    t1: float
    f_lo: float
    f_hi: float
    sources: tuple = ()            # subset of PROPOSERS
    family: str = "unknown"        # classes.FAMILIES
    cls: str = ""                  # class name, classes.UNKNOWN, or ''
    confidence: float | None = None
    snr_db: float | None = None    # dB above the measured floor
    alpha_hz: float | None = None  # the cycle frequency that found it
    integration_s: float | None = None
    state: str = "proposed"
    confirmed_by: str = ""         # decoder key
    decoded: str = ""              # a short line of decoded content
    flags: tuple = ()
    track_id: str = ""
    measurements: dict = field(default_factory=dict)
    profile: str = ""
    epoch: float | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # -- derived -------------------------------------------------------------
    @property
    def center_hz(self) -> float:
        return 0.5 * (self.f_lo + self.f_hi)

    @property
    def bw_hz(self) -> float:
        return float(self.f_hi - self.f_lo)

    @property
    def duration_s(self) -> float:
        return float(self.t1 - self.t0)

    @property
    def wall_t0(self) -> float | None:
        return None if self.epoch is None else self.epoch + self.t0

    @property
    def wall_t1(self) -> float | None:
        return None if self.epoch is None else self.epoch + self.t1

    def to_json(self) -> dict:
        d = asdict(self)
        d["sources"] = list(self.sources)
        d["flags"] = list(self.flags)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Detection":
        known = set(cls.__dataclass_fields__)
        kw = {k: v for k, v in d.items() if k in known}
        kw["sources"] = tuple(kw.get("sources", ()))
        kw["flags"] = tuple(kw.get("flags", ()))
        return cls(**kw)

    def with_flag(self, flag: str) -> "Detection":
        if flag not in FLAGS:
            raise ValueError(f"unknown flag {flag!r}")
        if flag not in self.flags:
            self.flags = tuple(self.flags) + (flag,)
        return self

    def confirm(self, decoder: str, decoded: str = "",
                decoder_class: str = "") -> "Detection":
        """A decoder decoded it. The decoder wins the label (§5); a different
        class from the classifier is kept as a disagreement, not erased."""
        self.state = "confirmed"
        self.confirmed_by = str(decoder)
        self.decoded = str(decoded)[:500]
        if decoder_class and self.cls and decoder_class != self.cls:
            self.measurements = dict(self.measurements)
            self.measurements["classifier_said"] = self.cls
            self.with_flag("disagreement")
        if decoder_class:
            self.cls = decoder_class
        return self


def overlap_tf(a: Detection, b: Detection) -> float:
    """Intersection-over-union in time × frequency (0..1)."""
    t = max(0.0, min(a.t1, b.t1) - max(a.t0, b.t0))
    f = max(0.0, min(a.f_hi, b.f_hi) - max(a.f_lo, b.f_lo))
    inter = t * f
    if inter <= 0:
        return 0.0
    ua = a.duration_s * a.bw_hz + b.duration_s * b.bw_hz - inter
    return inter / ua if ua > 0 else 0.0


def merge(dets: list[Detection], iou: float = 0.3) -> list[Detection]:
    """Merge detections of the same signal from different proposers: their
    sources union, the box is the union, the best class and confidence are
    kept. Where proposers disagree about the class, both are kept and the
    detection is flagged — 'where they disagree, all are shown' (§1)."""
    out: list[Detection] = []
    for d in sorted(dets, key=lambda x: (x.t0, x.f_lo)):
        hit = None
        for o in out:
            if overlap_tf(o, d) >= iou:
                hit = o
                break
        if hit is None:
            out.append(Detection.from_json(d.to_json()))
            continue
        hit.t0, hit.t1 = min(hit.t0, d.t0), max(hit.t1, d.t1)
        hit.f_lo, hit.f_hi = min(hit.f_lo, d.f_lo), max(hit.f_hi, d.f_hi)
        hit.sources = tuple(p for p in PROPOSERS
                            if p in set(hit.sources) | set(d.sources))
        if d.cls and hit.cls and d.cls != hit.cls:
            hit.measurements = dict(hit.measurements)
            hit.measurements.setdefault("other_classes", []).append(d.cls)
            hit.with_flag("disagreement")
        if (d.confidence or 0) > (hit.confidence or 0):
            hit.cls = d.cls or hit.cls
            hit.confidence = d.confidence
            hit.family = d.family if d.family != "unknown" else hit.family
        if hit.family == "unknown" and d.family != "unknown":
            hit.family = d.family
        if d.alpha_hz and not hit.alpha_hz:
            hit.alpha_hz, hit.integration_s = d.alpha_hz, d.integration_s
        if d.snr_db is not None and (hit.snr_db is None or d.snr_db > hit.snr_db):
            hit.snr_db = d.snr_db
        for f in d.flags:
            hit.with_flag(f)
    return out


# ---------------------------------------------------------------------------
# The box styling (§12.2, D13) — data only; ATK's waterfall draws it
# ---------------------------------------------------------------------------
ENERGY_COLOUR = "#e0e0e0"
CYCLIC_COLOUR = "#ff5400"     # vermilion (the plan's cyan collided with LTE)
AI_COLOUR = "#ff00dd"         # magenta
PULSE_MS = 600

#: source -> how its Proposed box is drawn.
SOURCE_STYLES = {
    "energy": {"edge": ENERGY_COLOUR, "dash": "dash", "width": 1,
               "badge": "", "pulse": False,
               "words": "energy (CFAR) — something with power is here"},
    "cyclic": {"edge": CYCLIC_COLOUR, "dash": "dash", "width": 2,
               "badge": "α", "pulse": True,
               "words": "cyclostationary — a known cycle frequency was found"},
    "learned": {"edge": AI_COLOUR, "dash": "dash", "width": 2,
                "badge": "AI", "pulse": True,
                "words": "AI — the learned detector proposed it"},
}


def box_style(det: Detection, technology_colour: str | None = None) -> dict:
    """How ATK should draw this detection. Returns a dict:

        edge        colour(s) of the edge: one, or two that alternate when
                    proposers agree
        dash        'dash' | 'alternate' | 'solid'
        width       pen width in pixels
        badges      ['α 4.2 s', 'AI 0.87', '✓ dsd'] — stacked
        caption     the class in words, coloured `caption_colour`
        caption_colour  the technology colour (today's scheme), or None
        pulse_ms    one pulse on first appearance (0: none)
        hatch       True for post-clean (a hatched inner edge)
        tooltip     the full story (hover)
    """
    from atk_diffusion.detect import classes as _classes
    srcs = [s for s in PROPOSERS if s in set(det.sources)] or ["energy"]
    badges = []
    for s in srcs:
        b = SOURCE_STYLES[s]["badge"]
        if not b:
            continue
        if s == "cyclic" and det.integration_s:
            b = f"α {det.integration_s:.1f} s"
        elif s == "learned" and det.confidence is not None:
            b = f"AI {det.confidence:.2f}"
        badges.append(b)
    cls = _classes.get(det.cls)
    if det.cls == _classes.UNKNOWN:
        caption = "UNKNOWN"
    elif cls is not None:
        caption = cls.label
    else:
        caption = det.cls or _classes.FAMILY_WORDS.get(det.family, "")
    if "denoised" in det.flags:
        badges.append("denoised")
    if "post-clean" in det.flags:
        badges.append("post-clean")
    if det.state == "confirmed":
        badges.append(f"✓ {det.confirmed_by}" if det.confirmed_by else "✓")
        edge = [technology_colour or SOURCE_STYLES[srcs[-1]]["edge"]]
        dash, width = "solid", 3
        pulse = PULSE_MS
    else:
        coloured = [s for s in srcs if s != "energy"]
        if len(coloured) >= 2:
            edge = [SOURCE_STYLES[s]["edge"] for s in coloured]
            dash = "alternate"
        else:
            edge = [SOURCE_STYLES[(coloured or srcs)[0]]["edge"]]
            dash = "dash"
        width = max(SOURCE_STYLES[s]["width"] for s in srcs)
        pulse = PULSE_MS if any(SOURCE_STYLES[s]["pulse"] for s in srcs) else 0
    tip = [f"{caption or 'unclassified'} — {det.state.upper()}",
           "found by: " + ", ".join(SOURCE_STYLES[s]["words"] for s in srcs)]
    if det.confidence is not None:
        tip.append(f"confidence {det.confidence:.2f} (calibrated)")
    if det.snr_db is not None:
        tip.append(f"{det.snr_db:.1f} dB above the floor")
    if det.alpha_hz:
        tip.append(f"cycle frequency α = {det.alpha_hz:g} Hz"
                   + (f" after {det.integration_s:.1f} s of integration"
                      if det.integration_s else ""))
    for k, v in (det.measurements or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            tip.append(f"{k.replace('_', ' ')}: {v:g}")
    if det.state == "confirmed":
        tip.append(f"confirmed by {det.confirmed_by}"
                   + (f": {det.decoded}" if det.decoded else ""))
    if "disagreement" in det.flags:
        tip.append("the proposers or the decoder disagree about the class — "
                   "both are shown")
    tip.append("right-click → Cut signal…")
    return {"edge": edge, "dash": dash, "width": width, "badges": badges,
            "caption": caption,
            "caption_colour": technology_colour if cls and cls.technology else None,
            "pulse_ms": pulse, "hatch": "post-clean" in det.flags,
            "tooltip": "\n".join(tip)}


def legend() -> list[dict]:
    """The legend chip row (§12.2): one chip per edge style and badge."""
    return [
        {"edge": ENERGY_COLOUR, "dash": "dash", "text": "energy"},
        {"edge": CYCLIC_COLOUR, "dash": "dash", "text": "α cyclostationary"},
        {"edge": AI_COLOUR, "dash": "dash", "text": "AI proposed"},
        {"edge": [CYCLIC_COLOUR, AI_COLOUR], "dash": "alternate",
         "text": "both agree"},
        {"edge": None, "dash": "solid", "text": "✓ decoder confirmed "
                                                "(technology colour)"},
    ]


# ---------------------------------------------------------------------------
# Colour science used by the palette test (and nothing else)
# ---------------------------------------------------------------------------
def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _lin(c: float) -> float:
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _delin(c: float) -> int:
    c = min(1.0, max(0.0, c))
    v = 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
    return int(round(v * 255))


def rgb_lab(rgb) -> tuple[float, float, float]:
    r, g, b = (_lin(v) for v in rgb)
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = (0.2126 * r + 0.7152 * g + 0.0722 * b)
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t):
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    return 116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))


def delta_e(lab1, lab2) -> float:
    """CIEDE2000."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    C1, C2 = math.hypot(a1, b1), math.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(Cb ** 7 / (Cb ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dh = h2p - h1p
    if C1p * C2p == 0:
        dh = 0.0
    elif dh > 180:
        dh -= 360
    elif dh < -180:
        dh += 360
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dh / 2))
    Lbp, Cbp = (L1 + L2) / 2, (C1p + C2p) / 2
    if C1p * C2p == 0:
        hbp = h1p + h2p
    elif abs(h1p - h2p) > 180:
        hbp = (h1p + h2p + 360) / 2
    else:
        hbp = (h1p + h2p) / 2
    T = (1 - 0.17 * math.cos(math.radians(hbp - 30))
         + 0.24 * math.cos(math.radians(2 * hbp))
         + 0.32 * math.cos(math.radians(3 * hbp + 6))
         - 0.20 * math.cos(math.radians(4 * hbp - 63)))
    dth = 30 * math.exp(-((hbp - 275) / 25) ** 2)
    Rc = 2 * math.sqrt(Cbp ** 7 / (Cbp ** 7 + 25 ** 7))
    Sl = 1 + 0.015 * (Lbp - 50) ** 2 / math.sqrt(20 + (Lbp - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cbp, 1 + 0.015 * Cbp * T
    Rt = -math.sin(math.radians(2 * dth)) * Rc
    return math.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
                     + Rt * (dCp / Sc) * (dHp / Sh))


#: Machado et al. 2009, severity 1.0.
_CVD = {"protan": ((0.152286, 1.052583, -0.204868),
                   (0.114503, 0.786281, 0.099216),
                   (-0.003882, -0.048116, 1.051998)),
        "deutan": ((0.367322, 0.860646, -0.227968),
                   (0.280085, 0.672501, 0.047413),
                   (-0.011820, 0.042940, 0.968881))}


def simulate_cvd(rgb, kind: str) -> tuple[int, int, int]:
    m = _CVD[kind]
    lin = [_lin(v) for v in rgb]
    return tuple(_delin(sum(m[i][j] * lin[j] for j in range(3)))
                 for i in range(3))


def contrast_ratio(rgb1, rgb2) -> float:
    def lum(rgb):
        r, g, b = (_lin(v) for v in rgb)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    a, b = lum(rgb1), lum(rgb2)
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


def palette_report(technology: dict, waterfall_rgb: list) -> dict:
    """The numbers behind the choice, for the test and for anyone checking:
    min ΔE of each source colour to the technology palette and the ramp, the
    pairwise ΔE under protan/deutan simulation, and contrast on the floor."""
    out = {}
    srcs = {"energy": ENERGY_COLOUR, "cyclic": CYCLIC_COLOUR, "learned": AI_COLOUR}
    for name, hx in srcs.items():
        lab = rgb_lab(hex_rgb(hx))
        d_tech = min((delta_e(lab, rgb_lab(hex_rgb(v))), k)
                     for k, v in technology.items())
        d_wf = min(delta_e(lab, rgb_lab(c)) for c in waterfall_rgb)
        out[name] = {"nearest_technology": d_tech[1], "de_technology": d_tech[0],
                     "de_waterfall": d_wf,
                     "contrast_on_floor": contrast_ratio(hex_rgb(hx),
                                                         waterfall_rgb[1])}
    pairs = {}
    for (n1, h1), (n2, h2) in itertools.combinations(srcs.items(), 2):
        r1, r2 = hex_rgb(h1), hex_rgb(h2)
        pairs[f"{n1}/{n2}"] = {
            "normal": delta_e(rgb_lab(r1), rgb_lab(r2)),
            "protan": delta_e(rgb_lab(simulate_cvd(r1, "protan")),
                              rgb_lab(simulate_cvd(r2, "protan"))),
            "deutan": delta_e(rgb_lab(simulate_cvd(r1, "deutan")),
                              rgb_lab(simulate_cvd(r2, "deutan")))}
    out["pairs"] = pairs
    return out
