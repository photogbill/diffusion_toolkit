# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The RF social graph — which radios co-occur, keep a schedule, move
together, and answer each other (plan C3).

Bill, 2026-10-08: *"I love the RF social graph idea."* Once every emitter
is fingerprinted (C1) and its sightings logged (`library.add_sighting`:
who, when, where, what frequency, any decoded identity), Network Link can
do for radios what it does for people. Four relations, each a statistical
test against chance with its evidence kept item by item:

* **co-occurrence** — both on the air in the same window more often than
  their activity rates predict (Poisson tail of the count against the
  expected count).
* **schedule** — a radio's activity is periodic (hourly / daily / weekly:
  the Rayleigh test of its sighting times wrapped on the period) and two
  radios share an hour-of-week profile (cosine similarity).
* **co-movement** — sightings close in time are close in space (allowing
  for the distance a vehicle covers between the two key-ups, from each
  radio's own track), while the radios actually moved — tested against how
  often the two are that close at all, so a pair parked at one building
  is co-located, not co-moving.
* **call-and-response** — B starts transmitting within a short delay
  after A stops, repeatedly, beyond B's base rate. Directed: A -> B.

A link carries its count, the count expected by chance, the lift, the
p-value and the list of evidence (each instance: when, how long a delay or
how far apart, where). The graph is INFERRED tier: an association measured
over PROPOSED identities (a fingerprint match is a proposal) — never a
fact about who talks to whom.

EXPORTS, ALL ROUND-TRIPPED (ATK FUTURE_PLANS §7, Bill's rule): an
i2-ready CSV set — one item per row, one field per column, never a packed
cell; unbounded lists (evidence, decoder ids, frequencies, schedule hours)
are CHILD tables keyed by id; RFC 4180 quoting, CRLF line ends, UTF-8 with
a BOM, ISO-8601 UTC times, an empty cell for "not known" — and the writer
and the reader walk ONE column specification, so they cannot drift.
`read_i2` rebuilds the graph and the test asserts it equal. Also GeoJSON
(emitters as points at their mean sighted position, links as lines) and
Network Link JSON in ATK's `.atkproj` shape (entity type "Emitter",
relations "Communication" / "Social", evidence in the edge's meta).
"""

from __future__ import annotations

import csv
import io
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from atk_diffusion import provenance as _prov
from atk_diffusion.geo import products as _products
from atk_diffusion.geo.terrain import haversine_m

_prov.METHOD_TIERS.setdefault("social_graph", "inferred")

KINDS = ("co_occurrence", "schedule", "co_movement", "call_response")
RELATION = {"call_response": "Communication", "co_occurrence": "Social",
            "co_movement": "Social", "schedule": "Social"}
KIND_WORDS = {"co_occurrence": "on the air together",
              "schedule": "keep the same schedule",
              "co_movement": "move together",
              "call_response": "answers"}
PERIODS_H = (1.0, 24.0, 168.0)


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------
def to_epoch(t) -> float:
    if t is None or t == "":
        raise ValueError("a sighting needs a time")
    if isinstance(t, (int, float, np.integer, np.floating)):
        return float(t)
    s = str(t).strip().replace("Z", "+00:00")
    d = datetime.fromisoformat(s)
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.timestamp()


def iso(t: float | None) -> str:
    """ISO-8601 UTC to the microsecond (the export's resolution)."""
    if t is None:
        return ""
    q = round(float(t), 6)
    sec = math.floor(q)
    us = int(round((q - sec) * 1e6))
    if us >= 1_000_000:
        sec, us = sec + 1, us - 1_000_000
    d = datetime.fromtimestamp(sec, tz=timezone.utc)
    return d.strftime("%Y-%m-%dT%H:%M:%S") + (f".{us:06d}" if us else "") + "Z"


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------
@dataclass
class Sighting:
    emitter_id: str
    t: float
    lat: float | None = None
    lon: float | None = None
    freq_hz: float | None = None
    duration_s: float = 0.0
    decoder_ids: tuple = ()

    @classmethod
    def from_dict(cls, d: dict) -> "Sighting":
        dec = d.get("decoder_ids") or ([d["decoder_id"]] if d.get("decoder_id") else [])
        return cls(str(d["emitter_id"]), to_epoch(d["t"]), d.get("lat"), d.get("lon"),
                   d.get("freq_hz"), float(d.get("duration_s") or 0.0), tuple(dec))


@dataclass
class Evidence:
    kind: str
    t: float | None
    value: float | None
    detail: str = ""
    lat: float | None = None
    lon: float | None = None


@dataclass
class Link:
    source: str
    target: str
    kind: str
    weight: float
    directed: bool
    count: int
    expected: float
    lift: float
    p_value: float
    evidence: list = field(default_factory=list)

    @property
    def link_id(self) -> str:
        arrow = "->" if self.directed else "--"
        return f"{self.kind}:{self.source}{arrow}{self.target}"

    def words(self) -> str:
        if self.kind == "call_response":
            return (f"{self.target} answers {self.source} ({self.count} times, "
                    f"{self.expected:.1f} expected by chance)")
        if self.kind == "schedule":
            return (f"{self.source} and {self.target} keep the same schedule "
                    f"(hour-of-week profiles {self.weight:.2f} alike; "
                    f"{self.count} active hours shared)")
        return (f"{self.source} and {self.target} {KIND_WORDS[self.kind]} "
                f"({self.count} instances, {self.expected:.1f} expected by chance)")


@dataclass
class EmitterNode:
    emitter_id: str
    label: str = ""
    sightings: int = 0
    first_seen: float | None = None
    last_seen: float | None = None
    lat: float | None = None
    lon: float | None = None
    moving: bool = False
    period_h: float | None = None
    period_p: float | None = None
    frequencies: dict = field(default_factory=dict)       # Hz -> sightings
    decoder_ids: list = field(default_factory=list)
    schedule: dict = field(default_factory=dict)          # hour of week -> fraction

    def schedule_words(self) -> str:
        if not self.period_h:
            return "no regular schedule"
        if not self.schedule:
            return f"repeats every {self.period_h:g} h"
        hours = sorted(int(h) for h in self.schedule)
        days = sorted({h // 24 for h in hours})
        hod = sorted({h % 24 for h in hours})
        names = "Mon Tue Wed Thu Fri Sat Sun".split()
        return (f"repeats every {self.period_h:g} h; active "
                f"{', '.join(names[d] for d in days)} at hours "
                f"{', '.join(str(h) for h in hod)} UTC")


class SocialGraph:
    def __init__(self, nodes=None, links=None, params=None):
        self.nodes: dict[str, EmitterNode] = dict(nodes or {})
        self.links: list[Link] = list(links or [])
        self.params: dict = dict(params or {})
        self.tier = "inferred"

    def link(self, a: str, b: str, kind: str) -> Link | None:
        for l in self.links:
            if l.kind == kind and ((l.source, l.target) == (a, b) or
                                   (not l.directed and (l.target, l.source) == (a, b))):
                return l
        return None

    def summary(self) -> dict:
        lines = [l.words() for l in sorted(self.links, key=lambda l: -l.weight)]
        return {"emitters": len(self.nodes), "links": len(self.links),
                "by_kind": dict(Counter(l.kind for l in self.links)),
                "words": lines}

    # -- GeoJSON ------------------------------------------------------------------
    def geojson_features(self) -> list:
        feats = []
        for n in self.nodes.values():
            if n.lat is None or n.lon is None:
                continue
            feats.append(_products.point_feature(n.lat, n.lon, {
                "emitter_id": n.emitter_id, "label": n.label or n.emitter_id,
                "sightings": n.sightings, "first_seen": iso(n.first_seen),
                "last_seen": iso(n.last_seen), "moving": n.moving,
                "schedule": n.schedule_words(), "role": "emitter"}))
        for l in self.links:
            a, b = self.nodes.get(l.source), self.nodes.get(l.target)
            if not a or not b or None in (a.lat, a.lon, b.lat, b.lon):
                continue
            if (a.lat, a.lon) == (b.lat, b.lon):
                continue
            feats.append(_products.line_feature([a.lat, b.lat], [a.lon, b.lon], {
                "link_id": l.link_id, "kind": l.kind, "directed": l.directed,
                "weight": round(l.weight, 4), "count": l.count,
                "expected": round(l.expected, 4), "p_value": l.p_value,
                "words": l.words(), "role": "link"}))
        return feats

    # -- Network Link (ATK's .atkproj) ------------------------------------------
    def to_network_link(self, name: str = "") -> dict:
        nodes = []
        for n in self.nodes.values():
            detail = [f"{n.sightings} sightings", n.schedule_words()]
            if n.decoder_ids:
                detail.append("decoded ids: " + ", ".join(n.decoder_ids))
            nodes.append({"id": f"emitter:{n.emitter_id}",
                          "label": n.label or n.emitter_id,
                          "entity_type": "Emitter", "x": 0.0, "y": 0.0,
                          "meta": {"detail": " · ".join(detail),
                                   "emitter_id": n.emitter_id,
                                   "first_seen": iso(n.first_seen),
                                   "last_seen": iso(n.last_seen),
                                   "frequencies_hz": {str(k): v for k, v in
                                                      n.frequencies.items()},
                                   "decoder_ids": list(n.decoder_ids),
                                   "moving": n.moving, "period_h": n.period_h,
                                   "tier": self.tier, "origin": "atk_diffusion.social",
                                   "source": "RF social graph (INFERRED: an "
                                             "association over proposed identities)"}})
        edges = []
        for l in self.links:
            edges.append({"source": f"emitter:{l.source}", "target": f"emitter:{l.target}",
                          "relation_type": RELATION[l.kind],
                          "meta": {"kind": l.kind, "label": l.words(),
                                   "directed": l.directed, "weight": l.weight,
                                   "count": l.count, "expected": l.expected,
                                   "lift": l.lift, "p_value": l.p_value,
                                   "tier": self.tier, "link_id": l.link_id,
                                   "evidence": [{"time": iso(e.t), "value": e.value,
                                                 "detail": e.detail, "lat": e.lat,
                                                 "lon": e.lon} for e in l.evidence]}})
        return {"atk_project_version": 1,
                "name": name or f"RF social graph {iso(time.time())}",
                "notes": "Built by the ATK Diffusion Toolkit (plan C3). Links "
                         "are statistical associations — INFERRED tier.",
                "exported_at": time.time(),
                "counts": {"nodes": len(nodes), "edges": len(edges)},
                "graph": {"nodes": nodes, "edges": edges}}

    # -- products --------------------------------------------------------------------
    def to_products(self, rf, run: str | None = None, label: str = "social") -> str:
        run = run or f"social_{_products.run_name(label)}"
        pr = _products.ProductRun(rf, "emitters", run, tier=self.tier,
                                  method="social_graph", params=self.params,
                                  description="; ".join(self.summary()["words"][:10]))
        pr.add_geojson("social.geojson", self.geojson_features(), tier=self.tier,
                       layer={"name": "RF social graph", "role": "graph"})
        pr.add_json("network_link.atkproj", self.to_network_link(), tier=self.tier,
                    role="network-link")
        for name, text in i2_texts(self).items():
            pr.add_text(name, text, tier=self.tier, role="i2")
        pr.finish(summary=self.summary())
        return str(pr.dir)


# ---------------------------------------------------------------------------
# The analyses
# ---------------------------------------------------------------------------
def _poisson_sf(k: int, mu: float) -> float:
    from scipy.stats import poisson
    return float(poisson.sf(k - 1, max(mu, 1e-12)))


def _rayleigh(times: np.ndarray, period_s: float) -> tuple[float, float]:
    """(resultant length R, p-value) of times wrapped on a period."""
    n = times.size
    if n < 3:
        return 0.0, 1.0
    a = 2 * math.pi * np.mod(times, period_s) / period_s
    R = float(abs(np.mean(np.exp(1j * a))))
    z = n * R * R
    p = math.exp(-z) * (1 + (2 * z - z * z) / (4 * n)
                        - (24 * z - 132 * z * z + 76 * z ** 3 - 9 * z ** 4) / (288 * n * n))
    return R, float(min(max(p, 0.0), 1.0))


def _hour_of_week(t: np.ndarray) -> np.ndarray:
    # 1970-01-01 was a Thursday: shift so 0 = Monday 00:00 UTC
    return ((np.floor(t / 3600.0).astype(np.int64) + 72) % 168)


def build(sightings, *, window_s: float = 60.0,
          response_s: tuple = (0.5, 10.0), co_move_m: float = 300.0,
          co_move_dt_s: float = 120.0, min_count: int = 3, alpha: float = 0.001,
          periods_h=PERIODS_H, schedule_similarity: float = 0.7,
          labels: dict | None = None) -> SocialGraph:
    """The graph from sightings (Sighting or dicts with emitter_id, t, lat,
    lon, freq_hz, duration_s, decoder_ids)."""
    sts = [s if isinstance(s, Sighting) else Sighting.from_dict(s) for s in sightings]
    if not sts:
        raise ValueError("no sightings")
    by = defaultdict(list)
    for s in sts:
        by[s.emitter_id].append(s)
    for v in by.values():
        v.sort(key=lambda s: s.t)
    t_all = np.array([s.t for s in sts])
    t0, t1 = float(t_all.min()), float(t_all.max())
    span = max(t1 - t0, window_s)
    n_bins = int(math.ceil(span / window_s)) + 1
    params = {"window_s": window_s, "response_s": list(response_s),
              "co_move_m": co_move_m, "co_move_dt_s": co_move_dt_s,
              "min_count": min_count, "alpha": alpha,
              "periods_h": list(periods_h), "schedule_similarity": schedule_similarity,
              "span": [iso(t0), iso(t1)], "sightings": len(sts)}
    g = SocialGraph(params=params)
    hw = {}
    for eid, ss in by.items():
        t = np.array([s.t for s in ss])
        pos = [(s.lat, s.lon) for s in ss if s.lat is not None and s.lon is not None]
        node = EmitterNode(eid, (labels or {}).get(eid, ""), len(ss), float(t.min()),
                           float(t.max()))
        if pos:
            la = np.array([p[0] for p in pos])
            lo = np.array([p[1] for p in pos])
            node.lat, node.lon = float(la.mean()), float(lo.mean())
            spread = haversine_m(node.lat, node.lon, la, lo)
            node.moving = bool(np.max(spread) > 3 * co_move_m)
        node.frequencies = dict(Counter(float(s.freq_hz) for s in ss if s.freq_hz))
        node.decoder_ids = sorted({d for s in ss for d in s.decoder_ids})
        best = None
        for P in periods_h:
            if span < 2.5 * P * 3600:               # need a few cycles
                continue
            R, p = _rayleigh(t, P * 3600.0)
            if p < alpha and (best is None or R > best[1]):
                best = (P, R, p)
        if best:
            node.period_h, node.period_p = best[0], best[2]
            prof = np.bincount(_hour_of_week(t), minlength=168) / len(t)
            node.schedule = {int(h): round(float(prof[h]), 4)
                             for h in np.nonzero(prof >= 0.5 / 24)[0]}
        hw[eid] = np.bincount(_hour_of_week(t), minlength=168).astype(float)
        g.nodes[eid] = node
    ids = sorted(by)
    bins = {}
    for eid in ids:
        b = set()
        for s in by[eid]:
            k0 = int((s.t - t0) // window_s)
            k1 = int((s.t + s.duration_s - t0) // window_s)
            b.update(range(k0, k1 + 1))
        bins[eid] = b
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            # co-occurrence
            both = sorted(bins[a] & bins[b])
            exp = len(bins[a]) * len(bins[b]) / n_bins
            if len(both) >= min_count:
                p = _poisson_sf(len(both), exp)
                if p < alpha:
                    ev = [Evidence("co_occurrence", t0 + k * window_s, None,
                                   f"both on the air in the {window_s:g} s window")
                          for k in both]
                    g.links.append(Link(a, b, "co_occurrence",
                                        math.log1p(len(both)) * min(len(both) / max(exp, 1e-9), 50),
                                        False, len(both), exp, len(both) / max(exp, 1e-9),
                                        p, ev))
            # shared schedule
            na, nb = g.nodes[a], g.nodes[b]
            if na.period_h and nb.period_h:
                va, vb = hw[a], hw[b]
                cos = float(va @ vb / max(np.linalg.norm(va) * np.linalg.norm(vb), 1e-12))
                if cos >= schedule_similarity:
                    shared = sorted(set(na.schedule) & set(nb.schedule))
                    ev = [Evidence("schedule", None, float(h),
                                   f"both active at hour-of-week {h} "
                                   f"({'Mon Tue Wed Thu Fri Sat Sun'.split()[h // 24]} "
                                   f"{h % 24:02d}:00 UTC)") for h in shared]
                    g.links.append(Link(a, b, "schedule", cos, False, len(shared),
                                        0.0, cos, max(na.period_p, nb.period_p), ev))
            # co-movement
            lnk = _co_movement(by[a], by[b], co_move_m, co_move_dt_s, min_count,
                               g.nodes[a].moving or g.nodes[b].moving, alpha)
            if lnk is not None:
                g.links.append(Link(a, b, **lnk))
    for a in ids:
        for b in ids:
            if a != b:
                lnk = _call_response(by[a], by[b], response_s, span, min_count, alpha)
                if lnk is not None:
                    g.links.append(Link(a, b, **lnk))
    return g


def _speed(ss, max_gap_s: float = 1800.0) -> float:
    """Median speed (m/s) between an emitter's own consecutive sightings."""
    v = []
    for a, b in zip(ss[:-1], ss[1:]):
        dt = b.t - a.t
        if 0 < dt <= max_gap_s:
            v.append(float(haversine_m(a.lat, a.lon, b.lat, b.lon)) / dt)
    return float(np.median(v)) if v else 0.0


def _co_movement(sa, sb, near_m, dt_s, min_count, moved, alpha=0.001):
    """Sightings close in time that are close in space — allowing for how
    far a vehicle moves in the time between the two key-ups — tested
    against the chance that the two radios are that close at all."""
    from scipy.stats import binom
    pa = [s for s in sa if s.lat is not None]
    pb = [s for s in sb if s.lat is not None]
    if not pa or not pb or not moved:
        return None
    v = max(_speed(pa), _speed(pb))
    tb = np.array([s.t for s in pb])
    pairs = []
    for s in pa:
        j = int(np.argmin(np.abs(tb - s.t)))
        dt = abs(tb[j] - s.t)
        if dt <= dt_s:
            d = float(haversine_m(s.lat, s.lon, pb[j].lat, pb[j].lon))
            pairs.append((s, pb[j], d, near_m + v * dt))
    if len(pairs) < min_count:
        return None
    close = [p for p in pairs if p[2] <= p[3]]
    # the null: how often ANY position of one is that close to ANY of the other
    la = np.array([s.lat for s in pa])[:, None]
    lo = np.array([s.lon for s in pa])[:, None]
    allowed = float(np.median([p[3] for p in pairs]))
    q0 = float(np.mean(haversine_m(la, lo, np.array([s.lat for s in pb])[None, :],
                                   np.array([s.lon for s in pb])[None, :]) <= allowed))
    q0 = min(max(q0, 1e-6), 1 - 1e-9)
    p = float(binom.sf(len(close) - 1, len(pairs), q0))
    if len(close) < min_count or len(close) / len(pairs) < 0.8 or p >= alpha:
        return None
    ev = [Evidence("co_movement", s.t, d,
                   f"{d:.0f} m apart, {abs(s.t - o.t):.0f} s apart", s.lat, s.lon)
          for s, o, d, _ in close]
    exp = q0 * len(pairs)
    return {"kind": "co_movement",
            "weight": math.log1p(len(close)) * min(len(close) / max(exp, 1e-9), 50),
            "directed": False, "count": len(close), "expected": exp,
            "lift": len(close) / max(exp, 1e-9), "p_value": p, "evidence": ev}


def _call_response(sa, sb, window, span, min_count, alpha):
    lo, hi = window
    tb = np.array([s.t for s in sb])
    hits = []
    for s in sa:
        end = s.t + s.duration_s
        j = np.nonzero((tb >= end + lo) & (tb <= end + hi))[0]
        if j.size:
            hits.append((s, sb[int(j[0])], float(tb[int(j[0])] - end)))
    rate_b = len(sb) / max(span, 1.0)
    exp = len(sa) * min(1.0, rate_b * (hi - lo))
    if len(hits) < min_count:
        return None
    p = _poisson_sf(len(hits), exp)
    if p >= alpha:
        return None
    ev = [Evidence("call_response", s.t + s.duration_s, d,
                   f"answered {d:.1f} s after the call ended", r.lat, r.lon)
          for s, r, d in hits]
    return {"kind": "call_response",
            "weight": math.log1p(len(hits)) * min(len(hits) / max(exp, 1e-9), 50),
            "directed": True, "count": len(hits), "expected": exp,
            "lift": len(hits) / max(exp, 1e-9), "p_value": p, "evidence": ev}


# ---------------------------------------------------------------------------
# i2: one column specification, shared by the writer and the reader
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Column:
    name: str
    kind: str          # str | int | float | bool | time | fopt


def _fmt(v, kind: str) -> str:
    if v is None or v == "":
        return ""
    if kind == "time":
        return iso(v)
    if kind in ("float", "fopt"):
        return repr(float(v))
    if kind == "int":
        return str(int(v))
    if kind == "bool":
        return "true" if v else "false"
    return str(v)


def _parse(s: str, kind: str):
    if s == "":
        return None if kind not in ("str",) else ""
    if kind == "time":
        return to_epoch(s)
    if kind in ("float", "fopt"):
        return float(s)
    if kind == "int":
        return int(s)
    if kind == "bool":
        return s.strip().lower() == "true"
    return s


#: table -> (columns, rows-from-graph, apply-row-to-builder). The writer and
#: the reader both walk this; a column added here is added to both.
ENTITY_COLS = (Column("entity_id", "str"), Column("entity_type", "str"),
               Column("label", "str"), Column("sightings", "int"),
               Column("first_seen", "time"), Column("last_seen", "time"),
               Column("latitude", "fopt"), Column("longitude", "fopt"),
               Column("moving", "bool"), Column("period_h", "fopt"),
               Column("period_p", "fopt"), Column("tier", "str"))
FREQ_COLS = (Column("entity_id", "str"), Column("frequency_hz", "float"),
             Column("sightings", "int"))
DECODER_COLS = (Column("entity_id", "str"), Column("decoder_id", "str"))
SCHEDULE_COLS = (Column("entity_id", "str"), Column("hour_of_week", "int"),
                 Column("fraction", "float"))
LINK_COLS = (Column("link_id", "str"), Column("source_id", "str"),
             Column("target_id", "str"), Column("relation", "str"),
             Column("direction", "str"), Column("label", "str"),
             Column("weight", "float"), Column("count", "int"),
             Column("expected", "float"), Column("lift", "float"),
             Column("p_value", "float"), Column("first_evidence", "time"),
             Column("last_evidence", "time"), Column("tier", "str"))
EVIDENCE_COLS = (Column("link_id", "str"), Column("evidence_index", "int"),
                 Column("kind", "str"), Column("time", "time"),
                 Column("value", "fopt"), Column("detail", "str"),
                 Column("latitude", "fopt"), Column("longitude", "fopt"))

I2_FILES = {"i2_entities.csv": ENTITY_COLS, "i2_entity_frequencies.csv": FREQ_COLS,
            "i2_entity_decoder_ids.csv": DECODER_COLS,
            "i2_entity_schedule.csv": SCHEDULE_COLS, "i2_links.csv": LINK_COLS,
            "i2_link_evidence.csv": EVIDENCE_COLS}


def _rows(g: SocialGraph) -> dict:
    ent, fr, dec, sch, lk, ev = [], [], [], [], [], []
    for n in g.nodes.values():
        ent.append({"entity_id": n.emitter_id, "entity_type": "Emitter",
                    "label": n.label, "sightings": n.sightings,
                    "first_seen": n.first_seen, "last_seen": n.last_seen,
                    "latitude": n.lat, "longitude": n.lon, "moving": n.moving,
                    "period_h": n.period_h, "period_p": n.period_p, "tier": g.tier})
        for f, c in sorted(n.frequencies.items()):
            fr.append({"entity_id": n.emitter_id, "frequency_hz": f, "sightings": c})
        for d in n.decoder_ids:
            dec.append({"entity_id": n.emitter_id, "decoder_id": d})
        for h, frac in sorted(n.schedule.items()):
            sch.append({"entity_id": n.emitter_id, "hour_of_week": h, "fraction": frac})
    for l in g.links:
        ts = [e.t for e in l.evidence if e.t is not None]
        lk.append({"link_id": l.link_id, "source_id": l.source, "target_id": l.target,
                   "relation": l.kind,
                   "direction": "source_to_target" if l.directed else "none",
                   "label": l.words(), "weight": l.weight, "count": l.count,
                   "expected": l.expected, "lift": l.lift, "p_value": l.p_value,
                   "first_evidence": min(ts) if ts else None,
                   "last_evidence": max(ts) if ts else None, "tier": g.tier})
        for i, e in enumerate(l.evidence):
            ev.append({"link_id": l.link_id, "evidence_index": i, "kind": e.kind,
                       "time": e.t, "value": e.value, "detail": e.detail,
                       "latitude": e.lat, "longitude": e.lon})
    return {"i2_entities.csv": ent, "i2_entity_frequencies.csv": fr,
            "i2_entity_decoder_ids.csv": dec, "i2_entity_schedule.csv": sch,
            "i2_links.csv": lk, "i2_link_evidence.csv": ev}


def i2_texts(g: SocialGraph) -> dict:
    """{file name: CSV text (BOM, CRLF, RFC 4180)} from the column spec."""
    out = {}
    for name, rows in _rows(g).items():
        cols = I2_FILES[name]
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL)
        w.writerow([c.name for c in cols])
        for r in rows:
            w.writerow([_fmt(r.get(c.name), c.kind) for c in cols])
        out[name] = "﻿" + buf.getvalue()
    return out


def write_i2(g: SocialGraph, folder) -> list[Path]:
    d = Path(folder)
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, text in i2_texts(g).items():
        p = d / name
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        paths.append(p)
    return paths


def _read_table(path: Path, cols) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        if header != [c.name for c in cols]:
            raise ValueError(f"{path.name}: the columns are {header}, the "
                             f"specification says {[c.name for c in cols]}")
        return [{c.name: _parse(v, c.kind) for c, v in zip(cols, row)} for row in r]


def read_i2(folder) -> SocialGraph:
    """The exact inverse of `write_i2` (the round trip is the test)."""
    d = Path(folder)
    t = {name: _read_table(d / name, cols) for name, cols in I2_FILES.items()}
    g = SocialGraph()
    for r in t["i2_entities.csv"]:
        g.nodes[r["entity_id"]] = EmitterNode(
            r["entity_id"], r["label"], r["sightings"], r["first_seen"], r["last_seen"],
            r["latitude"], r["longitude"], bool(r["moving"]), r["period_h"],
            r["period_p"])
        g.tier = r["tier"] or g.tier
    for r in t["i2_entity_frequencies.csv"]:
        g.nodes[r["entity_id"]].frequencies[r["frequency_hz"]] = r["sightings"]
    for r in t["i2_entity_decoder_ids.csv"]:
        g.nodes[r["entity_id"]].decoder_ids.append(r["decoder_id"])
    for r in t["i2_entity_schedule.csv"]:
        g.nodes[r["entity_id"]].schedule[r["hour_of_week"]] = r["fraction"]
    ev = defaultdict(list)
    for r in sorted(t["i2_link_evidence.csv"], key=lambda r: (r["link_id"],
                                                               r["evidence_index"])):
        ev[r["link_id"]].append(Evidence(r["kind"], r["time"], r["value"], r["detail"],
                                         r["latitude"], r["longitude"]))
    for r in t["i2_links.csv"]:
        g.links.append(Link(r["source_id"], r["target_id"], r["relation"], r["weight"],
                            r["direction"] == "source_to_target", r["count"],
                            r["expected"], r["lift"], r["p_value"], ev[r["link_id"]]))
    return g


def _t6(t):
    return None if t is None else round(float(t), 6)


def _node_key(n: EmitterNode) -> dict:
    d = dict(n.__dict__)
    d["first_seen"], d["last_seen"] = _t6(n.first_seen), _t6(n.last_seen)
    return d


def _link_key(l: Link) -> dict:
    d = {k: v for k, v in l.__dict__.items() if k != "evidence"}
    d["evidence"] = [(e.kind, _t6(e.t), e.value, e.detail, e.lat, e.lon)
                     for e in l.evidence]
    return d


def same_graph(a: SocialGraph, b: SocialGraph) -> list[str]:
    """Differences in words (empty = equal on entities, attributes, links,
    direction and evidence; times at the export's microsecond)."""
    out = []
    if set(a.nodes) != set(b.nodes):
        out.append(f"entities differ: {sorted(set(a.nodes) ^ set(b.nodes))}")
    for k in sorted(set(a.nodes) & set(b.nodes)):
        ka, kb = _node_key(a.nodes[k]), _node_key(b.nodes[k])
        for f in ka:
            if ka[f] != kb.get(f):
                out.append(f"entity {k}: {f} {ka[f]!r} != {kb.get(f)!r}")
    la = {l.link_id: l for l in a.links}
    lb = {l.link_id: l for l in b.links}
    if set(la) != set(lb):
        out.append(f"links differ: {sorted(set(la) ^ set(lb))}")
    for k in sorted(set(la) & set(lb)):
        ka, kb = _link_key(la[k]), _link_key(lb[k])
        for f in ka:
            if ka[f] != kb.get(f):
                out.append(f"link {k}: {f} differs")
    return out


# ---------------------------------------------------------------------------
# The first experiment, simulated: handhelds on a scripted weekly schedule
# ---------------------------------------------------------------------------
def scripted_week(seed: int = 0, start: str = "2026-10-05T00:00:00Z") -> list[Sighting]:
    """Six radios for seven days (from a Monday). A calls B on weekdays
    08-17 UTC and B answers within seconds; C and D ride one vehicle on a
    night patrol 20-02 UTC; E is a beacon on the hour; F is random."""
    rng = np.random.default_rng(seed)
    t0 = to_epoch(start)
    base = (38.70, -77.50)
    out: list[Sighting] = []

    def jitter(lat, lon, m):
        return (lat + rng.normal(0, m) / 111_195.0,
                lon + rng.normal(0, m) / (111_195.0 * math.cos(math.radians(lat))))
    for day in range(7):
        d0 = t0 + day * 86400
        if day < 5:
            t = d0 + 8 * 3600 + rng.uniform(0, 600)
            while t < d0 + 17 * 3600:
                dur = rng.uniform(4, 15)
                out.append(Sighting("A", t, *jitter(*base, 15), 151.94e6, dur,
                                    ("DMR:3110001",)))
                if rng.random() < 0.85:
                    tb = t + dur + rng.uniform(1.0, 4.0)
                    out.append(Sighting("B", tb, *jitter(base[0] + 0.02, base[1] + 0.03, 800),
                                        151.94e6, rng.uniform(3, 10), ("DMR:3110002",)))
                t += rng.uniform(900, 1800)
        # the night patrol (20:00 -> 02:00), a loop around town
        t = d0 + 20 * 3600 + rng.uniform(0, 300)
        while t < d0 + 26 * 3600:
            ph = 2 * math.pi * (t - d0) / 3600.0
            la = base[0] + 0.03 * math.sin(ph)
            lo = base[1] + 0.04 * math.cos(ph)
            out.append(Sighting("C", t, *jitter(la, lo, 10), 155.1e6, rng.uniform(2, 6)))
            tc = t + rng.uniform(30, 90)
            ph = 2 * math.pi * (tc - d0) / 3600.0
            out.append(Sighting("D", tc, *jitter(base[0] + 0.03 * math.sin(ph),
                                                 base[1] + 0.04 * math.cos(ph), 10),
                                155.1e6, rng.uniform(2, 6)))
            t += rng.uniform(500, 800)
        for h in range(24):
            out.append(Sighting("E", d0 + h * 3600 + rng.uniform(0, 5),
                                base[0] - 0.05, base[1], 162.475e6, 2.0))
    for _ in range(60):
        t = t0 + rng.uniform(0, 7 * 86400)
        out.append(Sighting("F", t, *jitter(base[0] - 0.02, base[1] - 0.02, 3000),
                            146.52e6, rng.uniform(2, 20)))
    return sorted(out, key=lambda s: s.t)
