# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Labels, both ways: generator labels and TorchSig 2.2.0 per-signal metadata
⇄ SigMF annotations (DETECTION_DESIGN §10, §12 step 3; plan §4.A).

*"Labels are SigMF annotations … TorchSig's dataset metadata is converted to
and from this on the way in and out."* One label format, in the capture's own
`.sigmf-meta`, so a label never drifts from its data:

    core:sample_start  core:sample_count  core:freq_lower_edge
    core:freq_upper_edge  core:label (the class-table name)
    atk:family  atk:snr_db  atk:source = "synthetic"  atk:generator
    atk:symbol_rate (when known)  atk:carrier_offset_hz  [atk:environment]

WHAT TORCHSIG SAYS, KEPT BESIDE WHAT WE SAY. TorchSig 2.2.0 measures a
component's SNR as the peak of its time-averaged spectrum over the per-bin
noise floor, and redraws its box where the max-hold spectrum clears the floor
by 3 dB (`torchsig.utils.dsp.update_signal_snr_bandwidth`) — both depend on
the SNR and the FFT size. The toolkit's own definitions (`native.
SNR_DEFINITION`, the 99 % occupied band) are the ones in `atk:snr_db` and the
core edges, so the same number means the same thing whichever generator made
the data. TorchSig's own values are not thrown away: `atk:torchsig_class`,
`atk:torchsig_snr_db` and `atk:torchsig_edges` carry them.

TorchSig's coordinates: `center_freq` is a signed offset from the dataset's
frequency origin (not an RF frequency), `bandwidth` two-sided, and
lower_freq / upper_freq = center ∓ bandwidth/2 (`signal_types.py`). A capture's
centre frequency turns them into the absolute SigMF edges.

Pure Python and numpy: nothing here imports TorchSig, so ATK's core can read
and write TorchSig-shaped metadata without it.
"""

from __future__ import annotations

import math

from atk_diffusion import sigmf as _sigmf
from atk_diffusion.detect import classes as _classes

SOURCE = "synthetic"
TORCHSIG_GENERATOR = "torchsig 2.2.0"

for _k, _v in {
    "atk:torchsig_class": "TorchSig 2.2.0's own signal name for a synthetic label",
    "atk:torchsig_snr_db": "TorchSig's own SNR (peak spectrum bin over the "
                           "per-bin floor) — kept beside atk:snr_db",
    "atk:torchsig_edges": "TorchSig's own box [lower, upper] Hz, absolute",
    "atk:clipped": "a synthetic signal wider than the span, seen through it",
    "atk:bursts": "on-intervals [[start, count], …] inside the annotation",
    "atk:snr_definition": "how atk:snr_db is defined (synth.native.SNR_DEFINITION)",
    "atk:class_alternatives": "other classes a TorchSig name could also be",
    "atk:mobility": "synthetic scene: static | pedestrian | mobile | aircraft | drone",
    "atk:doppler_hz": "synthetic scene: the line-of-sight Doppler shift applied (Hz)",
    "atk:service": "synthetic scene: the allocation the emitter was placed in",
    "atk:dataset": "the dataset a synthetic scene belongs to",
    "atk:scene": "synthetic scene: floor, AGC, receiver and terrain used",
}.items():
    _sigmf.ATK_KEYS.setdefault(_k, _v)


class LabelError(ValueError):
    """A label cannot be converted; the message says why."""


# ---------------------------------------------------------------------------
# indices and names
# ---------------------------------------------------------------------------
def family_index(family: str) -> int:
    try:
        return _classes.FAMILIES.index(str(family))
    except ValueError:
        raise LabelError(f"{family!r} is not a detector family "
                         f"({', '.join(_classes.FAMILIES)})") from None


def torchsig_names(cls: str) -> tuple:
    c = _classes.get(cls)
    return tuple(c.torchsig) if c else ()


def classes_for_torchsig(name: str) -> list[str]:
    """Every class-table class TorchSig's `name` can stand for."""
    return [c.name for c in _classes.CLASSES if str(name) in c.torchsig]


def class_for_torchsig(name: str, bandwidth_hz: float | None = None
                       ) -> tuple[str, list[str]]:
    """(class, alternatives) for a TorchSig signal name. One candidate: it.
    Several: the one whose typical bandwidth is nearest (in ratio) to
    `bandwidth_hz` when given, else the TorchSig-family reference class
    (ref_*) when there is one, else the first; `alternatives` lists the
    rest so the ambiguity is visible. No candidate: UNKNOWN."""
    cands = classes_for_torchsig(name)
    if not cands:
        return _classes.UNKNOWN, []
    if len(cands) == 1:
        return cands[0], []
    pick = None
    if bandwidth_hz and bandwidth_hz > 0:
        sized = [(abs(math.log(_classes.get(c).bandwidth_hz / bandwidth_hz)), c)
                 for c in cands if _classes.get(c).bandwidth_hz > 0]
        if sized:
            pick = min(sized)[1]
    if pick is None:
        refs = [c for c in cands if c.startswith("ref_")]
        pick = refs[0] if refs else cands[0]
    return pick, [c for c in cands if c != pick]


# ---------------------------------------------------------------------------
# generator labels (synth.native / synth.torchsig_backend) <-> annotations
# ---------------------------------------------------------------------------
def _finite(v) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def label_to_annotation(label: dict, center_hz: float = 0.0, *,
                        generator: str | None = None, environment: str = "",
                        sample_offset: int = 0, extra: dict | None = None
                        ) -> _sigmf.Annotation:
    """A generator label dict -> SigMF annotation. Frequencies in the label
    are relative to the capture centre; the annotation's are absolute."""
    cls = str(label["cls"])
    c = _classes.get(cls)
    fam = label.get("family") or (c.family if c else "unknown")
    ex = {"atk:family": fam, "atk:source": SOURCE,
          "atk:generator": generator or label.get("generator", "native"),
          "atk:carrier_offset_hz": float(label.get("carrier_offset_hz", 0.0))}
    if _finite(label.get("snr_db")):
        ex["atk:snr_db"] = round(float(label["snr_db"]), 3)
    rs = label.get("symbol_rate_hz")
    if _finite(rs) and float(rs) > 0:
        ex["atk:symbol_rate"] = float(rs)
    if environment:
        ex["atk:environment"] = str(environment)
    if label.get("clipped"):
        ex["atk:clipped"] = True
    if label.get("torchsig_class"):
        ex["atk:torchsig_class"] = label["torchsig_class"]
    if _finite(label.get("torchsig_snr_db")):
        ex["atk:torchsig_snr_db"] = float(label["torchsig_snr_db"])
    if label.get("torchsig_edges"):
        lo, hi = label["torchsig_edges"]
        ex["atk:torchsig_edges"] = [float(center_hz) + float(lo),
                                    float(center_hz) + float(hi)]
    bursts = label.get("bursts") or []
    if len(bursts) > 1:
        s0 = int(label["sample_start"])
        ex["atk:bursts"] = [[int(s) - s0, int(k)] for s, k in bursts]
    ex.update(extra or {})
    has_band = cls != "noise" and label.get("f_hi_hz") is not None
    return _sigmf.Annotation(
        sample_start=int(label["sample_start"]) + int(sample_offset),
        sample_count=int(label["sample_count"]),
        freq_lower_edge=(float(center_hz) + float(label["f_lo_hz"])) if has_band else None,
        freq_upper_edge=(float(center_hz) + float(label["f_hi_hz"])) if has_band else None,
        label=cls, extra=ex)


def annotation_to_label(ann: _sigmf.Annotation, center_hz: float = 0.0) -> dict:
    """The inverse: an annotation -> a label dict (frequencies relative to
    `center_hz`). Missing numbers are NaN (SNR) or 0 (symbol rate)."""
    ex = ann.extra or {}
    c = _classes.get(ann.label)
    lo = None if ann.freq_lower_edge is None else float(ann.freq_lower_edge) - center_hz
    hi = None if ann.freq_upper_edge is None else float(ann.freq_upper_edge) - center_hz
    bw = (hi - lo) if (lo is not None and hi is not None) else 0.0
    off = ex.get("atk:carrier_offset_hz")
    if off is None:
        off = 0.5 * (lo + hi) if (lo is not None and hi is not None) else 0.0
    return {"cls": ann.label,
            "family": ex.get("atk:family") or (c.family if c else "unknown"),
            "bandwidth_hz": float(bw),
            "symbol_rate_hz": float(ex.get("atk:symbol_rate", 0.0) or 0.0),
            "carrier_offset_hz": float(off),
            "snr_db": float(ex["atk:snr_db"]) if _finite(ex.get("atk:snr_db"))
            else float("nan"),
            "sample_start": int(ann.sample_start),
            "sample_count": int(ann.sample_count),
            "f_lo_hz": lo, "f_hi_hz": hi,
            "source": ex.get("atk:source", ""),
            "generator": ex.get("atk:generator", "")}


# ---------------------------------------------------------------------------
# TorchSig per-signal metadata <-> annotations
# ---------------------------------------------------------------------------
def _meta_dict(meta) -> dict:
    """A TorchSig Signal / SignalMetadataObject, or a plain dict, -> dict."""
    if isinstance(meta, dict):
        return dict(meta)
    for attr in ("to_dict", "get_full_metadata"):
        f = getattr(meta, attr, None)
        if callable(f):
            return dict(f())
    raise LabelError("not TorchSig metadata (a dict or a torchsig Signal)")


def torchsig_to_annotation(meta, *, sample_rate: float, center_hz: float = 0.0,
                           cls: str | None = None, snr_db: float | None = None,
                           symbol_rate_hz: float | None = None,
                           box: tuple[float, float] | None = None,
                           generator: str = TORCHSIG_GENERATOR,
                           environment: str = "", extra: dict | None = None
                           ) -> _sigmf.Annotation:
    """One TorchSig component's metadata -> SigMF annotation.

    `cls` is the class-table class it was generated for (when known); else it
    is inferred from TorchSig's name (`class_for_torchsig`, alternatives in
    `atk:class_alternatives`). `box` (relative Hz) and `snr_db` are the
    toolkit's own measurements; when absent the core edges are TorchSig's box
    and atk:snr_db is left out (TorchSig's SNR is a different quantity and is
    kept as atk:torchsig_snr_db)."""
    m = _meta_dict(meta)
    for k in ("center_freq", "bandwidth", "start_in_samples", "duration_in_samples"):
        if k not in m:
            raise LabelError(f"TorchSig metadata has no {k!r} — not a placed "
                             "component signal")
    ts_name = str(m.get("class_name", "") or "")
    ts_center = float(m["center_freq"])
    ts_bw = float(m["bandwidth"])
    ts_lo, ts_hi = ts_center - ts_bw / 2.0, ts_center + ts_bw / 2.0
    ex = dict(extra or {})
    if cls is None:
        cls, alts = class_for_torchsig(ts_name, ts_bw)
        if alts:
            ex["atk:class_alternatives"] = alts
    c = _classes.get(cls)
    ex.update({"atk:family": c.family if c else "unknown",
               "atk:source": SOURCE, "atk:generator": generator,
               "atk:torchsig_class": ts_name,
               "atk:torchsig_edges": [float(center_hz) + ts_lo,
                                      float(center_hz) + ts_hi],
               "atk:carrier_offset_hz": ts_center})
    if _finite(m.get("snr_db")):
        ex["atk:torchsig_snr_db"] = float(m["snr_db"])
    if _finite(snr_db):
        ex["atk:snr_db"] = round(float(snr_db), 3)
    if _finite(symbol_rate_hz) and float(symbol_rate_hz) > 0:
        ex["atk:symbol_rate"] = float(symbol_rate_hz)
    if environment:
        ex["atk:environment"] = str(environment)
    lo, hi = (box if box is not None else (ts_lo, ts_hi))
    return _sigmf.Annotation(
        sample_start=int(m["start_in_samples"]),
        sample_count=int(m["duration_in_samples"]),
        freq_lower_edge=float(center_hz) + float(lo),
        freq_upper_edge=float(center_hz) + float(hi),
        label=str(cls), extra=ex)


def torchsig_name_for(cls: str) -> str:
    names = torchsig_names(cls)
    if not names:
        c = _classes.get(cls)
        what = c.label if c else repr(cls)
        raise LabelError(f"{what} has no TorchSig 2.2.0 signal type, so it "
                         "cannot be written as TorchSig metadata")
    return names[0]


def annotation_to_torchsig(ann: _sigmf.Annotation, *, sample_rate: float,
                           center_hz: float = 0.0,
                           num_iq_samples: int | None = None) -> dict:
    """A SigMF annotation -> TorchSig 2.2.0 per-signal metadata (the keys a
    `torchsig.signals.signal_types.Signal` carries). TorchSig's class name is
    the one recorded when TorchSig made it, else the class table's first."""
    if ann.freq_lower_edge is None or ann.freq_upper_edge is None:
        raise LabelError("the annotation has no frequency edges; TorchSig "
                         "metadata needs a centre and a bandwidth")
    ex = ann.extra or {}
    name = ex.get("atk:torchsig_class") or torchsig_name_for(ann.label)
    lo = float(ann.freq_lower_edge) - float(center_hz)
    hi = float(ann.freq_upper_edge) - float(center_hz)
    out = {"class_name": str(name),
           "center_freq": 0.5 * (lo + hi),
           "bandwidth": hi - lo,
           "start_in_samples": int(ann.sample_start),
           "duration_in_samples": int(ann.sample_count),
           "sample_rate": float(sample_rate)}
    snr = ex.get("atk:snr_db", ex.get("atk:torchsig_snr_db"))
    if _finite(snr):
        out["snr_db"] = float(snr)
    if num_iq_samples:
        out["num_iq_samples_dataset"] = int(num_iq_samples)
    if ex.get("atk:symbol_rate"):
        out["symbol_rate"] = float(ex["atk:symbol_rate"])
    return out
