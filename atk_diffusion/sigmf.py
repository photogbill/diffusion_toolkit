# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""SigMF pairs and annotations — the one label format (DETECTION_DESIGN §10).

Labels are SigMF annotations in the capture's own `.sigmf-meta`, so a label
never drifts from its data and anything that speaks SigMF can read it:

    core:sample_start  core:sample_count  core:freq_lower_edge
    core:freq_upper_edge  core:label
    atk:family  atk:snr_db  atk:source (synthetic | cabled | taught |
    confirmed | proposed)  atk:decoder  atk:symbol_rate  atk:carrier_offset_hz

Written by hand per the spec (as ATK's recorder does) — no dependency.
The `atk:` keys this toolkit writes are listed in `ATK_KEYS` so a reader can
tell ours from anybody else's.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from atk_diffusion.dsp import iq as _iq

SIGMF_VERSION = "1.0.0"

#: Every `atk:` key this toolkit reads or writes, with what it means.
ATK_KEYS = {
    "atk:receiver_profile": "the receiver profile (plan §3.1)",
    "atk:datatype": "ATK's exact datatype when it differs from the SigMF tag "
                    "(ci16q11: bladeRF full scale 2048)",
    "atk:tier": "record | measured | cleaned | inferred | invented | "
                "proposed | confirmed (provenance.TIERS)",
    "atk:resampled_from": "the profile this capture was resampled from",
    "atk:resample_log": "the logged resampling step that made this file",
    "atk:translated_from": "the profile a translator (plan A6) mapped from",
    "atk:source_capture": "the capture a cut came from",
    "atk:source_sample_start": "first sample of the cut in its source",
    "atk:source_sample_count": "length of the cut in its source, in samples",
    "atk:decimation": "integer decimation from the profile rate",
    "atk:canonical_class": "voice | wideband | spread",
    "atk:cut_by": "who made the cut",
    "atk:method": "the cleaning / reconstruction method",
    "atk:method_params": "its parameters",
    "atk:model_sha256": "hash of the model weights that made it",
    "atk:snr_before_db": "measured SNR before a clean",
    "atk:snr_after_db": "measured SNR after a clean",
    "atk:tx_power_dbm": "cabled loop: transmitter output",
    "atk:attenuation_db": "cabled loop: fixed attenuation in line",
    "atk:splitter_loss_db": "cabled loop: splitter loss in line",
    "atk:cable_loss_db": "cabled loop: cable loss",
    "atk:expected_input_dbm": "cabled loop: computed receiver input",
    "atk:environment": "environment profile a synthetic scene was composed for",
    "atk:generator": "synthetic generator (torchsig 2.2.0 | native)",
}

#: Annotation-level atk: keys.
ANNOTATION_KEYS = ("atk:family", "atk:snr_db", "atk:source", "atk:decoder",
                   "atk:symbol_rate", "atk:carrier_offset_hz", "atk:confidence",
                   "atk:proposer", "atk:track_id", "atk:class_id")

LABEL_SOURCES = ("synthetic", "cabled", "taught", "confirmed", "proposed")


def _iso(t: float) -> str:
    return datetime.fromtimestamp(float(t), tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ")


def base_of(path) -> Path:
    """'x.sigmf-data' / 'x.sigmf-meta' / 'x' -> Path('x')."""
    p = Path(path)
    if p.name.endswith(".sigmf-data") or p.name.endswith(".sigmf-meta"):
        return p.with_name(p.name.rsplit(".sigmf-", 1)[0])
    return p


def data_path(path) -> Path:
    b = base_of(path)
    return b.with_name(b.name + ".sigmf-data")


def meta_path(path) -> Path:
    b = base_of(path)
    return b.with_name(b.name + ".sigmf-meta")


# ---------------------------------------------------------------------------
@dataclass
class Annotation:
    sample_start: int
    sample_count: int
    freq_lower_edge: float | None = None
    freq_upper_edge: float | None = None
    label: str = ""
    comment: str = ""
    extra: dict = field(default_factory=dict)   # atk:* keys

    def to_sigmf(self) -> dict:
        d = {"core:sample_start": int(self.sample_start),
             "core:sample_count": int(self.sample_count)}
        if self.freq_lower_edge is not None:
            d["core:freq_lower_edge"] = float(self.freq_lower_edge)
        if self.freq_upper_edge is not None:
            d["core:freq_upper_edge"] = float(self.freq_upper_edge)
        if self.label:
            d["core:label"] = str(self.label)
        if self.comment:
            d["core:comment"] = str(self.comment)
        for k, v in (self.extra or {}).items():
            key = k if ":" in k else f"atk:{k}"
            d[key] = v
        return d

    @classmethod
    def from_sigmf(cls, d: dict) -> "Annotation":
        extra = {k: v for k, v in d.items() if not k.startswith("core:")}
        return cls(int(d.get("core:sample_start", 0)),
                   int(d.get("core:sample_count", 0)),
                   d.get("core:freq_lower_edge"), d.get("core:freq_upper_edge"),
                   str(d.get("core:label", "") or ""),
                   str(d.get("core:comment", "") or ""), extra)

    @property
    def source(self) -> str:
        return str(self.extra.get("atk:source", "") or "")

    @property
    def bandwidth(self) -> float | None:
        if self.freq_lower_edge is None or self.freq_upper_edge is None:
            return None
        return float(self.freq_upper_edge) - float(self.freq_lower_edge)


# ---------------------------------------------------------------------------
def read_meta(path) -> dict:
    return json.loads(Path(meta_path(path)).read_text(encoding="utf-8"))


def write_meta(path, meta: dict) -> Path:
    mp = meta_path(path)
    tmp = mp.with_name(mp.name + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2, default=_json_default),
                   encoding="utf-8")
    tmp.replace(mp)
    return mp


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def datatype_of(meta: dict) -> str:
    """The exact datatype to decode with (atk:datatype wins: Q11 scaling)."""
    g = meta.get("global", {})
    return _iq.norm_dt(g.get("atk:datatype") or g.get("core:datatype", "cf32_le"))


def sample_rate_of(meta: dict) -> float:
    return float(meta.get("global", {}).get("core:sample_rate", 0.0))


def center_of(meta: dict, sample: int = 0) -> float:
    """Centre frequency in force at `sample` (captures segments)."""
    caps = sorted(meta.get("captures", []) or [],
                  key=lambda c: int(c.get("core:sample_start", 0)))
    f = 0.0
    for c in caps:
        if int(c.get("core:sample_start", 0)) <= sample:
            f = float(c.get("core:frequency", f) or f)
    return f


def channels_of(meta: dict) -> int:
    return int(meta.get("global", {}).get("core:num_channels", 1) or 1)


def num_samples(path, meta: dict | None = None) -> int:
    meta = meta or read_meta(path)
    bps = _iq.bytes_per_sample(datatype_of(meta))
    return Path(data_path(path)).stat().st_size // bps // channels_of(meta)


def load(path, start: int = 0, count: int | None = None,
         channel: int | None = None, meta: dict | None = None) -> np.ndarray:
    """Samples [start, start+count) as complex64 (unit full scale). A
    multi-channel file returns (channels, n) unless `channel` is given."""
    meta = meta or read_meta(path)
    dt = datatype_of(meta)
    ch = channels_of(meta)
    bps = _iq.bytes_per_sample(dt)
    total = num_samples(path, meta)
    start = max(0, int(start))
    n = total - start if count is None else max(0, min(int(count), total - start))
    with open(data_path(path), "rb") as f:
        f.seek(start * bps * ch)
        raw = f.read(n * bps * ch)
    x = _iq.to_complex(raw, dt)
    if ch > 1:
        x = _iq.deinterleave(x, ch)
        return x[int(channel)] if channel is not None else x
    return x


def write_pair(base, samples, sample_rate: float, center_hz: float = 0.0,
               datatype: str = "cf32", t0_utc: float | None = None,
               annotations=None, extra_global: dict | None = None,
               hw: str = "", description: str = "",
               recorder: str = "ATK Diffusion Toolkit",
               channels: int = 1) -> tuple[Path, Path]:
    """Write `<base>.sigmf-data` + `<base>.sigmf-meta`. `samples` is complex
    (n,) or (channels, n); it is written in `datatype` (cf32 by default —
    cuts and products are processed data and are kept at full precision)."""
    base = base_of(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(samples)
    if x.ndim == 2:
        channels = x.shape[0]
        inter = np.empty(x.shape[0] * x.shape[1], dtype=np.complex64)
        for c in range(x.shape[0]):
            inter[c::x.shape[0]] = x[c]
        x = inter
    dt = _iq.norm_dt(datatype)
    dp = data_path(base)
    dp.write_bytes(_iq.from_complex(x, dt))
    sigmf_dt = {"cu8": "cu8", "ci8": "ci8", "ci16": "ci16_le",
                "ci16q11": "ci16_le", "cf32": "cf32_le"}[dt]
    g = {"core:datatype": sigmf_dt, "core:sample_rate": float(sample_rate),
         "core:version": SIGMF_VERSION, "core:recorder": recorder}
    if dt == "ci16q11":
        g["atk:datatype"] = "ci16q11"
    if hw:
        g["core:hw"] = hw
    if description:
        g["core:description"] = description
    if channels > 1:
        g["core:num_channels"] = int(channels)
    for k, v in (extra_global or {}).items():
        g[k if ":" in k else f"atk:{k}"] = v
    meta = {"global": g,
            "captures": [{"core:sample_start": 0,
                          "core:frequency": float(center_hz),
                          "core:datetime": _iso(time.time() if t0_utc is None
                                                else t0_utc)}],
            "annotations": [a.to_sigmf() if isinstance(a, Annotation) else dict(a)
                            for a in (annotations or [])]}
    mp = write_meta(base, meta)
    return dp, mp


def annotations(path_or_meta) -> list[Annotation]:
    meta = path_or_meta if isinstance(path_or_meta, dict) else read_meta(path_or_meta)
    return [Annotation.from_sigmf(a) for a in meta.get("annotations", []) or []]


def add_annotations(path, anns, replace_source: str | None = None) -> int:
    """Append annotations to a capture's meta (sorted by start, as SigMF
    asks). `replace_source` first drops existing annotations with that
    atk:source — re-running a proposer replaces its own boxes and leaves
    taught and confirmed labels alone."""
    meta = read_meta(path)
    cur = meta.get("annotations", []) or []
    if replace_source:
        cur = [a for a in cur if a.get("atk:source") != replace_source]
    cur += [a.to_sigmf() if isinstance(a, Annotation) else dict(a) for a in anns]
    cur.sort(key=lambda a: int(a.get("core:sample_start", 0)))
    meta["annotations"] = cur
    write_meta(path, meta)
    return len(cur)


def validate(meta: dict) -> list[str]:
    """Problems a reader would trip over, in words. Empty = fine."""
    out = []
    g = meta.get("global")
    if not isinstance(g, dict):
        return ["no 'global' object"]
    for k in ("core:datatype", "core:sample_rate", "core:version"):
        if k not in g:
            out.append(f"global is missing {k}")
    try:
        _iq.norm_dt(g.get("atk:datatype") or g.get("core:datatype", ""))
    except ValueError as e:
        out.append(str(e))
    if not isinstance(meta.get("captures", []), list):
        out.append("'captures' is not a list")
    for i, a in enumerate(meta.get("annotations", []) or []):
        if "core:sample_start" not in a:
            out.append(f"annotation {i} has no core:sample_start")
        lo, hi = a.get("core:freq_lower_edge"), a.get("core:freq_upper_edge")
        if lo is not None and hi is not None and float(hi) < float(lo):
            out.append(f"annotation {i} has its frequency edges reversed")
        src = a.get("atk:source")
        if src is not None and src not in LABEL_SOURCES:
            out.append(f"annotation {i} has an unknown atk:source {src!r}")
    return out
