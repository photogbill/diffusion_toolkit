# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Track inpainting — ADS-B coverage holes, GPS telemetry gaps, bearing gaps,
and "where does it reappear" (plan §4.D4).

    *"Track inpainting — ADS-B coverage holes, bearing gaps, GPS telemetry
    gaps, conditioned on the offline road graph for ground tracks. Also
    'where does it reappear.'"*  — plan §4.D4

EVERY FILLED POINT IS `inferred`, WITH ITS SIGMA. The record's fixes are
returned untouched (tier "record"); each point put into a gap carries
`tier: "inferred"`, its method and `sigma_m` — the 1σ radius along the
worst axis of its position uncertainty, which GROWS into the gap and shrinks
again toward the fix after it. A filled track is a lead, never a position
report; the GeoJSON says so on every point.

THE GEOMETRY. A sphere (mean radius 6,371,008.8 m — 0.3 % from the
ellipsoid, far inside any gap's uncertainty) and, per gap, an azimuthal
equidistant plane centred on the great-circle midpoint of the gap's ends.
In that plane the great circle between the ends is a straight line through
the centre with exact distances, so constant velocity in the plane IS
constant speed along the great circle — an aircraft crossing an ADS-B hole
flies a great circle, not a rhumb line.

THE FILLS:

  * `great_circle` — constant speed along the great circle between the two
    fixes either side (slerp). The baseline: right for an airliner at
    cruise, wrong through a turn.
  * `turn` — constant speed and constant TURN RATE (a coordinated turn):
    an arc flown forward from the smoothed state at the gap's start and one
    flown back from the state at its end, with the heading change chosen
    so the arcs meet, crossfaded with a raised cosine. The model for an
    aircraft in a procedure turn or a holding pattern, where the cubic
    curve of a CV smoother cuts the corner and changes speed.
  * `kalman` — a constant-velocity (or constant-acceleration) Kalman filter
    run forward through the gap and a Rauch–Tung–Striebel smoother run
    back. Its process noise is the maximum-likelihood value on the track's
    own record (the filter is told how this object moves, not a textbook
    number). Its covariance is the base of every method's `sigma_m`; a fill
    that is NOT the smoother's estimate adds its distance from that
    estimate (sigma² = smoother variance + offset²), so a great circle
    through a turn does not claim the smoother's precision.
  * `road` — for a ground track and an offline road graph: both gap ends
    snapped to the nearest road (refused beyond `max_snap_m`), the shortest
    road route between them by A* (both ends start part-way along their
    edge), positions along it at constant speed. Refused when the route
    would need a speed the vehicle could not have made.

REFUSALS ARE FINDINGS. A gap is not filled when (1) the implied speed is
beyond what the kind of object can do, or (2) the fix after the gap is
outside where the motion before it could have taken it — a Mahalanobis
gate on the Kalman prediction at p = 1e-4. Both are what a stitched track
looks like (two aircraft, one hex code; two phones; a GPS jump), and filling
them would invent a path that never happened: the experiment's
hallucination number for tracks is how often a stitched track is filled
anyway (`experiments.repair_eval`).

WHERE DOES IT REAPPEAR: `reappear()` predicts the state at a future time
from the end of the record — an ellipse at the asked probability (χ² with
2 degrees of freedom) as a GeoJSON polygon; `reappear_on_roads()` the
stretches of road a vehicle could have reached between the slowest and the
fastest plausible speed.

BEARINGS: `fill_bearings()` does the same for a DF bearing track — a 1-D
constant-rate Kalman/RTS on the unwrapped angle (so 359° → 1° is two
degrees, not 358), sigma in degrees.

LIMITS, plainly. A model of motion is a guess about what happened in the
gap; a turn the record gave no hint of will be missed, and the sigma then
understates the error (the experiment reports how often the truth falls
inside 2σ). A road graph only conditions where the roads are; it knows
nothing of which turn was taken. Nothing here identifies a person.
"""

from __future__ import annotations

import csv
import heapq
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from atk_diffusion import provenance as _prov

R_EARTH = 6_371_008.8

for _m in ("great_circle_fill", "turn_fill", "road_fill", "kalman_predict"):
    _prov.METHOD_TIERS.setdefault(_m, "inferred")

METHODS = {"great_circle": "great_circle_fill", "turn": "turn_fill",
           "kalman": "kalman_fill", "road": "road_fill"}

#: What each kind of object can do. `max_speed` (m/s) gates gaps;
#: `sigma_m` is the fix accuracy assumed when the record does not say.
#: `q_floor` (m²/s³, CV model) is the least process noise ever assumed:
#: the record's maximum-likelihood q is used, but never below it, because a
#: straight stretch of record says nothing about what happened in the gap
#: and no real vehicle holds its velocity to better than about
#: √(q·10 s) ≈ 0.7 m/s over ten seconds. A sigma should err pessimistic.
KINDS = {
    "aircraft":   {"max_speed": 350.0, "sigma_m": 30.0, "q_floor": 0.05,
                   "words": "an aircraft (ADS-B)"},
    "vehicle":    {"max_speed": 70.0, "sigma_m": 5.0, "q_floor": 0.05,
                   "words": "a road vehicle"},
    "vessel":     {"max_speed": 30.0, "sigma_m": 10.0, "q_floor": 0.005,
                   "words": "a vessel"},
    "pedestrian": {"max_speed": 4.0, "sigma_m": 5.0, "q_floor": 0.01,
                   "words": "a person on foot"},
}

GATE_P = 1e-4          # a fix after a gap this unlikely under the motion is refused


# ---------------------------------------------------------------------------
# Fixes
# ---------------------------------------------------------------------------
@dataclass
class Fix:
    t: float                    # seconds (Unix epoch or any common origin)
    lat: float
    lon: float
    alt_m: float | None = None
    sigma_m: float | None = None
    tier: str = "record"
    method: str = ""
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


_TIME_KEYS = ("t", "time", "epoch", "ts", "timestamp", "time_s")
_ALT_KEYS = ("alt_m", "altitude_m", "alt_geom_m")


def fixes_from_rows(rows, time_key: str | None = None, alt_key: str | None = None,
                    alt_scale: float = 1.0, sigma_key: str | None = None) -> list[Fix]:
    """Dicts -> Fixes, sorted by time. Understands ATK's GPS track CSV
    (`epoch`, `lat`, `lon`, `altitude_m`) and anything with t/time/epoch.
    ADS-B altitude in FEET: pass alt_key="alt", alt_scale=0.3048. Rows
    without a time or a position are skipped (never guessed)."""
    out = []
    for r in rows:
        tk = time_key or next((k for k in _TIME_KEYS if k in r and r[k] not in ("", None)), None)
        try:
            t = float(r[tk])
            lat, lon = float(r["lat"]), float(r["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        alt = None
        ak = alt_key or next((k for k in _ALT_KEYS if k in r), None)
        if ak and r.get(ak) not in ("", None):
            try:
                alt = float(r[ak]) * alt_scale
            except (TypeError, ValueError):
                alt = None
        sg = None
        if sigma_key and r.get(sigma_key) not in ("", None):
            try:
                sg = float(r[sigma_key])
            except (TypeError, ValueError):
                sg = None
        out.append(Fix(t, lat, lon, alt, sg))
    out.sort(key=lambda f: f.t)
    return out


def read_track_csv(path) -> list[Fix]:
    """ATK's `<capture>.track.csv` (iq_recorder.TRACK_FIELDS) or gps_track's CSV."""
    with open(path, newline="", encoding="utf-8") as fh:
        return fixes_from_rows(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Spherical geodesy and the azimuthal equidistant plane
# ---------------------------------------------------------------------------
def _unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo),
                     np.sin(la)], axis=-1)


def _latlon(v):
    v = np.asarray(v, dtype=np.float64)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return (np.degrees(np.arcsin(np.clip(v[..., 2], -1, 1))),
            np.degrees(np.arctan2(v[..., 1], v[..., 0])))


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def slerp(lat0, lon0, lat1, lon1, f):
    """Points a fraction f along the great circle (constant angular rate)."""
    a, b = _unit(lat0, lon0), _unit(lat1, lon1)
    om = math.acos(max(-1.0, min(1.0, float(np.dot(a, b)))))
    f = np.atleast_1d(np.asarray(f, dtype=np.float64))
    if om < 1e-12:
        v = np.repeat(a[None, :], f.size, axis=0)
    else:
        v = (np.sin((1 - f) * om)[:, None] * a + np.sin(f * om)[:, None] * b) / math.sin(om)
    return _latlon(v)


def midpoint(lat0, lon0, lat1, lon1):
    la, lo = slerp(lat0, lon0, lat1, lon1, [0.5])
    return float(la[0]), float(lo[0])


def centroid(lats, lons):
    v = _unit(np.asarray(lats), np.asarray(lons)).sum(axis=0)
    la, lo = _latlon(v)
    return float(la), float(lo)


class Plane:
    """Azimuthal equidistant projection on the sphere, centred at
    (lat0, lon0): distances and bearings FROM the centre are exact, and a
    great circle through the centre is a straight line."""

    def __init__(self, lat0: float, lon0: float):
        self.lat0, self.lon0 = float(lat0), float(lon0)
        self._p0 = math.radians(self.lat0)
        self._l0 = math.radians(self.lon0)

    def fwd(self, lat, lon):
        p = np.radians(np.asarray(lat, dtype=np.float64))
        dl = np.radians(np.asarray(lon, dtype=np.float64)) - self._l0
        a = (np.sin((p - self._p0) / 2) ** 2
             + math.cos(self._p0) * np.cos(p) * np.sin(dl / 2) ** 2)
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        k = np.where(c < 1e-12, 1.0, c / np.where(c < 1e-12, 1.0, np.sin(c)))
        x = R_EARTH * k * np.cos(p) * np.sin(dl)
        y = R_EARTH * k * (math.cos(self._p0) * np.sin(p)
                           - math.sin(self._p0) * np.cos(p) * np.cos(dl))
        return x, y

    def inv(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        rho = np.hypot(x, y)
        c = rho / R_EARTH
        sc, cc = np.sin(c), np.cos(c)
        safe = np.where(rho < 1e-9, 1.0, rho)
        lat = np.arcsin(np.clip(cc * math.sin(self._p0)
                                + y * sc * math.cos(self._p0) / safe, -1, 1))
        lon = self._l0 + np.arctan2(x * sc, safe * math.cos(self._p0) * cc
                                    - y * math.sin(self._p0) * sc)
        lat = np.where(rho < 1e-9, self._p0, lat)
        lon = np.where(rho < 1e-9, self._l0, lon)
        lon = (lon + np.pi) % (2 * np.pi) - np.pi
        return np.degrees(lat), np.degrees(lon)


# ---------------------------------------------------------------------------
# Kalman CV / CA with an RTS smoother
# ---------------------------------------------------------------------------
def _FQ(dt: float, q: float, model: str):
    if model == "ca":
        f1 = np.array([[1, dt, dt * dt / 2], [0, 1, dt], [0, 0, 1]])
        q1 = q * np.array([[dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                           [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
                           [dt ** 3 / 6, dt ** 2 / 2, dt]])
    else:
        f1 = np.array([[1, dt], [0, 1]])
        q1 = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
    d = f1.shape[0]
    F = np.zeros((2 * d, 2 * d))
    Q = np.zeros((2 * d, 2 * d))
    for ax in range(2):
        sl = slice(ax * d, (ax + 1) * d)
        F[sl, sl] = f1
        Q[sl, sl] = q1
    return F, Q


def _H(model: str):
    d = 3 if model == "ca" else 2
    H = np.zeros((2, 2 * d))
    H[0, 0] = 1.0
    H[1, d] = 1.0
    return H


def kalman_rts(ts, xy, sig, t_query=(), model: str = "cv", q: float = 1.0,
               gate_at=None) -> dict:
    """Forward Kalman + RTS smoother on fixes (ts, xy [n,2], sig [n] metres)
    and query times (predicted only). Returns {times, mean [m,2],
    cov [m,2,2], is_fix, loglik, gate} sorted by time. `gate_at` (index of
    a fix) records the Mahalanobis distance of that fix against its
    prediction BEFORE it updates the filter."""
    ts = np.asarray(ts, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float64)
    sig = np.asarray(sig, dtype=np.float64)
    tq = np.asarray(list(t_query), dtype=np.float64)
    times = np.concatenate([ts, tq])
    kind = np.concatenate([np.arange(ts.size), np.full(tq.size, -1)])
    order = np.lexsort((kind < 0, times))
    times, kind = times[order], kind[order]
    d = 3 if model == "ca" else 2
    n = 2 * d
    H = _H(model)
    x = np.zeros(n)
    x[0], x[d] = xy[0]
    if ts.size >= 2 and ts[1] > ts[0]:
        v = (xy[1] - xy[0]) / (ts[1] - ts[0])
        x[1], x[d + 1] = v
    # a diffuse prior on velocity (100 m/s 1σ) and acceleration (10 m/s²)
    per_axis = [sig[0] ** 2, 1e4, 1e2][:d]
    P = np.diag(per_axis * 2).astype(np.float64)
    m = times.size
    xf = np.zeros((m, n))
    Pf = np.zeros((m, n, n))
    xp = np.zeros((m, n))
    Pp = np.zeros((m, n, n))
    Fs = np.zeros((m, n, n))
    loglik = 0.0
    gate = None
    t_prev = times[0]
    first = True
    for i in range(m):
        dt = times[i] - t_prev
        if first:
            F, Q = np.eye(n), np.zeros((n, n))
            first = False
        else:
            F, Q = _FQ(max(dt, 0.0), q, model)
        x = F @ x
        P = F @ P @ F.T + Q
        xp[i], Pp[i], Fs[i] = x, P, F
        k = kind[i]
        if k >= 0 and not (i == 0):
            z = xy[k]
            R = np.eye(2) * sig[k] ** 2
            S = H @ P @ H.T + R
            r = z - H @ x
            Si = np.linalg.inv(S)
            d2 = float(r @ Si @ r)
            if gate_at is not None and k == gate_at:
                gate = d2
            loglik += -0.5 * (d2 + math.log(max(np.linalg.det(S), 1e-300))
                              + 2 * math.log(2 * math.pi))
            K = P @ H.T @ Si
            x = x + K @ r
            P = (np.eye(n) - K @ H) @ P
            P = 0.5 * (P + P.T)
        xf[i], Pf[i] = x, P
        t_prev = times[i]
    xs = xf.copy()
    Ps = Pf.copy()
    for i in range(m - 2, -1, -1):
        Pn = Pp[i + 1]
        try:
            C = Pf[i] @ Fs[i + 1].T @ np.linalg.inv(Pn)
        except np.linalg.LinAlgError:
            C = Pf[i] @ Fs[i + 1].T @ np.linalg.pinv(Pn)
        xs[i] = xf[i] + C @ (xs[i + 1] - xp[i + 1])
        Ps[i] = Pf[i] + C @ (Ps[i + 1] - Pn) @ C.T
        Ps[i] = 0.5 * (Ps[i] + Ps[i].T)
    sel = [0, d]
    mean = xs[:, sel]
    cov = Ps[:, sel][:, :, sel]
    vel = xs[:, [1, d + 1]]
    return {"times": times, "mean": mean, "cov": cov, "vel": vel,
            "is_fix": kind >= 0, "loglik": loglik, "gate": gate,
            "filtered_mean": xf[:, sel], "filtered_cov": Pf[:, sel][:, :, sel]}


def estimate_q(ts, xy, sig, model: str = "cv", grid=None,
               max_fixes: int = 300) -> tuple[float, dict]:
    """Maximum-likelihood process noise on the record: the q whose filter
    best predicts each next fix. (q, {q: loglik})."""
    ts = np.asarray(ts, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float64)
    sig = np.asarray(sig, dtype=np.float64)
    if ts.size > max_fixes:
        # q is a continuous-time intensity: an evenly thinned record
        # estimates the same thing, in a fraction of the time
        keep = np.unique(np.linspace(0, ts.size - 1, max_fixes).astype(np.int64))
        ts, xy, sig = ts[keep], xy[keep], sig[keep]
    if grid is None:
        grid = 10.0 ** np.linspace(-4, 3, 15) if model == "cv" else \
            10.0 ** np.linspace(-6, 1, 15)
    table = {}
    for q in grid:
        table[float(q)] = kalman_rts(ts, xy, sig, (), model, float(q))["loglik"]
    best = max(table, key=table.get)
    return best, table


def _sigma_axes(C):
    w, V = np.linalg.eigh(np.asarray(C))
    w = np.maximum(w, 0.0)
    major = math.sqrt(w[1])
    minor = math.sqrt(w[0])
    # angle of the major axis, degrees clockwise from north (y)
    ang = math.degrees(math.atan2(V[0, 1], V[1, 1])) % 180.0
    return major, minor, ang


# ---------------------------------------------------------------------------
# Gaps
# ---------------------------------------------------------------------------
@dataclass
class Gap:
    i0: int                    # last fix before
    i1: int                    # first fix after
    t0: float
    t1: float
    duration_s: float
    distance_m: float
    implied_speed_mps: float
    filled: bool = False
    method: str = ""
    why: str = ""
    gate_d2: float | None = None
    points: int = 0

    def to_json(self) -> dict:
        return asdict(self)


def find_gaps(fixes: list[Fix], factor: float = 3.0,
              min_gap_s: float | None = None) -> list[Gap]:
    """Intervals longer than `factor` × the track's median interval (and
    than `min_gap_s`, if given)."""
    if len(fixes) < 3:
        return []
    t = np.array([f.t for f in fixes])
    dt = np.diff(t)
    nominal = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 0.0
    thr = max(factor * nominal, float(min_gap_s or 0.0))
    out = []
    for i in np.flatnonzero(dt > thr):
        a, b = fixes[i], fixes[i + 1]
        dist = float(haversine_m(a.lat, a.lon, b.lat, b.lon))
        dur = float(b.t - a.t)
        out.append(Gap(int(i), int(i + 1), a.t, b.t, dur, dist,
                       dist / dur if dur > 0 else math.inf))
    return out


def _velocity_at(plane: Plane, fixes, idx, side: str, k: int = 4):
    """Least-squares velocity (m/s, in `plane`) from up to k fixes beside a
    gap end — before it (side 'before', ending at idx) or after."""
    if side == "before":
        sel = list(range(max(0, idx - k + 1), idx + 1))
    else:
        sel = list(range(idx, min(len(fixes), idx + k)))
    if len(sel) < 2:
        return None
    t = np.array([fixes[i].t for i in sel])
    x, y = plane.fwd([fixes[i].lat for i in sel], [fixes[i].lon for i in sel])
    tt = t - t.mean()
    den = float(np.sum(tt * tt))
    if den <= 0:
        return None
    return np.array([float(np.sum(tt * (x - x.mean())) / den),
                     float(np.sum(tt * (y - y.mean())) / den)])


def _arc(p0, speed: float, psi0: float, omega: float, tau):
    """Constant-speed, constant-turn-rate motion in the plane (x east, y
    north, heading clockwise from north)."""
    tau = np.asarray(tau, dtype=np.float64)
    if abs(omega) < 1e-9:
        return p0 + speed * np.stack([np.sin(psi0) * tau, np.cos(psi0) * tau], axis=1)
    psi = psi0 + omega * tau
    r = speed / omega
    return p0 + np.stack([r * (np.cos(psi0) - np.cos(psi)),
                          r * (np.sin(psi) - np.sin(psi0))], axis=1)


def _turn_path(p0, v0, p1, v1, T: float, f):
    """Coordinated-turn fill between two smoothed states: the heading
    change (choosing among ±2π wraps the one whose forward arc lands
    nearest the far end), constant mean speed, a forward arc from the start
    and a backward arc from the end, crossfaded with a raised cosine."""
    p0, v0, p1, v1 = (np.asarray(a, dtype=np.float64) for a in (p0, v0, p1, v1))
    s0, s1 = float(np.hypot(*v0)), float(np.hypot(*v1))
    speed = 0.5 * (s0 + s1)
    if speed < 1e-6 or T <= 0:
        f = np.asarray(f)[:, None]
        return p0 + f * (p1 - p0)
    psi0 = math.atan2(v0[0], v0[1])
    psi1 = math.atan2(v1[0], v1[1])
    d = (psi1 - psi0 + math.pi) % (2 * math.pi) - math.pi
    best = None
    for k in (-1, 0, 1):
        om = (d + 2 * math.pi * k) / T
        end = _arc(p0, speed, psi0, om, [T])[0]
        err = float(np.hypot(*(end - p1)))
        if best is None or err < best[0]:
            best = (err, om)
    om = best[1]
    tau = np.asarray(f, dtype=np.float64) * T
    fwd = _arc(p0, speed, psi0, om, tau)
    # backward: fly the reverse heading from the end, turning the other way
    bwd = _arc(p1, speed, psi1 + math.pi, -om, T - tau)
    w = (0.5 * (1 + np.cos(np.pi * np.asarray(f))))[:, None]
    return w * fwd + (1 - w) * bwd


# ---------------------------------------------------------------------------
# The road graph
# ---------------------------------------------------------------------------
class RoadGraph:
    """An offline road graph: `nodes` {id: (lat, lon)}, `edges` [(a, b)] or
    [(a, b, speed_mps)], two-way unless (a, b) is in `oneway`. Straight
    segments between nodes (a curved road is several nodes). ATK's own road
    store (`atk/core/roads`) can hand over the sub-graph around a gap in
    this shape."""

    def __init__(self, nodes: dict, edges, oneway=()):
        if not nodes:
            raise ValueError("a road graph needs nodes")
        self.nodes = {k: (float(v[0]), float(v[1])) for k, v in nodes.items()}
        la, lo = centroid([v[0] for v in self.nodes.values()],
                          [v[1] for v in self.nodes.values()])
        self.plane = Plane(la, lo)
        ids = list(self.nodes)
        x, y = self.plane.fwd([self.nodes[i][0] for i in ids],
                              [self.nodes[i][1] for i in ids])
        self.xy = {i: (float(a), float(b)) for i, a, b in zip(ids, x, y)}
        ow = {tuple(e) for e in oneway}
        self.adj: dict = {i: [] for i in ids}
        self.edges = []
        for e in edges:
            a, b = e[0], e[1]
            sp = float(e[2]) if len(e) > 2 and e[2] else None
            if a not in self.nodes or b not in self.nodes:
                raise ValueError(f"edge {a}-{b} names a node that is not in the graph")
            L = math.dist(self.xy[a], self.xy[b])
            self.edges.append((a, b, L, sp))
            self.adj[a].append((b, L, sp))
            if (a, b) not in ow:
                self.adj[b].append((a, L, sp))
        ex = np.array([[*self.xy[a], *self.xy[b]] for a, b, _L, _s in self.edges])
        self._seg = ex if ex.size else np.zeros((0, 4))

    @classmethod
    def from_geojson(cls, gj: dict, snap_m: float = 1.0) -> "RoadGraph":
        """LineString / MultiLineString features -> a graph; vertices within
        `snap_m` of each other become one node (shared junctions)."""
        nodes, edges, key = {}, [], {}

        def node(lon, lat):
            k = (round(lat / (snap_m / 111_000.0)), round(lon / (snap_m / 111_000.0)))
            if k not in key:
                key[k] = len(nodes)
                nodes[key[k]] = (lat, lon)
            return key[k]
        for f in gj.get("features", []):
            g = f.get("geometry") or {}
            lines = ([g["coordinates"]] if g.get("type") == "LineString" else
                     g.get("coordinates", []) if g.get("type") == "MultiLineString" else [])
            sp = (f.get("properties") or {}).get("speed_mps")
            for line in lines:
                ids = [node(c[0], c[1]) for c in line]
                for a, b in zip(ids[:-1], ids[1:]):
                    if a != b:
                        edges.append((a, b, sp))
        return cls(nodes, edges)

    def snap(self, lat: float, lon: float) -> dict:
        """The nearest point on any edge: {edge index, a, b, frac (from a),
        dist_m, x, y, lat, lon}."""
        px, py = self.plane.fwd(lat, lon)
        px, py = float(px), float(py)
        S = self._seg
        ax, ay, bx, by = S[:, 0], S[:, 1], S[:, 2], S[:, 3]
        dx, dy = bx - ax, by - ay
        L2 = np.maximum(dx * dx + dy * dy, 1e-12)
        u = np.clip(((px - ax) * dx + (py - ay) * dy) / L2, 0, 1)
        qx, qy = ax + u * dx, ay + u * dy
        d = np.hypot(px - qx, py - qy)
        j = int(np.argmin(d))
        la, lo = self.plane.inv(qx[j], qy[j])
        a, b, L, _sp = self.edges[j]
        return {"edge": j, "a": a, "b": b, "frac": float(u[j]), "len": L,
                "dist_m": float(d[j]), "x": float(qx[j]), "y": float(qy[j]),
                "lat": float(la), "lon": float(lo)}

    def _seeds(self, s: dict) -> list:
        """(node, cost) seeds for a snap: both ends of its edge, each at the
        length of road between the snap and that end — you are never at a
        junction, you are part-way along a road. (A one-way restriction is
        not applied to these two partial legs.)"""
        return [(s["a"], s["frac"] * s["len"]), (s["b"], (1 - s["frac"]) * s["len"])]

    def route(self, sa: dict, sb: dict) -> tuple[list, float]:
        """Shortest road route between two snaps by A* (cost = length; the
        straight-line distance to the goal is the admissible heuristic).
        Returns (polyline [(lat, lon)], length_m); ([], inf) when no route."""
        if sa["edge"] == sb["edge"]:
            L = abs(sb["frac"] - sa["frac"]) * sa["len"]
            return [(sa["lat"], sa["lon"]), (sb["lat"], sb["lon"])], L
        goal_cost = {sb["a"]: sb["frac"] * sb["len"],
                     sb["b"]: (1 - sb["frac"]) * sb["len"]}
        gx, gy = sb["x"], sb["y"]

        def h(nid):
            x, y = self.xy[nid]
            return math.hypot(x - gx, y - gy)
        dist, prev, pq = {}, {}, []
        for nid, c in self._seeds(sa):
            if c < dist.get(nid, math.inf):
                dist[nid] = c
                prev[nid] = None
                heapq.heappush(pq, (c + h(nid), c, nid))
        best, best_end = math.inf, None
        done = set()
        while pq:
            f, g, u = heapq.heappop(pq)
            if u in done or g > dist.get(u, math.inf):
                continue
            if f >= best:
                break
            done.add(u)
            if u in goal_cost and g + goal_cost[u] < best:
                best, best_end = g + goal_cost[u], u
            for v, L, _sp in self.adj[u]:
                ng = g + L
                if ng < dist.get(v, math.inf):
                    dist[v] = ng
                    prev[v] = u
                    heapq.heappush(pq, (ng + h(v), ng, v))
        if best_end is None:
            return [], math.inf
        path = [best_end]
        while prev.get(path[-1]) is not None:
            path.append(prev[path[-1]])
        path.reverse()
        poly = [(sa["lat"], sa["lon"])] + [self.nodes[n] for n in path] + \
               [(sb["lat"], sb["lon"])]
        return poly, best

    def reachable(self, s: dict, d_lo: float, d_hi: float,
                  step_m: float = 10.0) -> list[list]:
        """Stretches of road at network distance between d_lo and d_hi from
        a snap (Dijkstra), as polylines [(lat, lon), …]."""
        dist, pq = {}, []
        for nid, c in self._seeds(s):
            if c < dist.get(nid, math.inf):
                dist[nid] = c
                heapq.heappush(pq, (c, nid))
        while pq:
            g, u = heapq.heappop(pq)
            if g > dist.get(u, math.inf) or g > d_hi:
                continue
            for v, L, _sp in self.adj[u]:
                if g + L < dist.get(v, math.inf):
                    dist[v] = g + L
                    heapq.heappush(pq, (g + L, v))
        pieces = []
        for j, (a, b, L, _sp) in enumerate(self.edges):
            da, db = dist.get(a, math.inf), dist.get(b, math.inf)
            if j == s["edge"]:
                da = min(da, s["frac"] * L)
                db = min(db, (1 - s["frac"]) * L)
            if min(da, db) > d_hi:
                continue
            n = max(2, int(math.ceil(L / step_m)) + 1)
            u = np.linspace(0, 1, n)
            if j == s["edge"]:
                dd = np.abs(u - s["frac"]) * L
            else:
                dd = np.minimum(da + u * L, db + (1 - u) * L)
            ok = (dd >= d_lo) & (dd <= d_hi)
            if not ok.any():
                continue
            ax, ay = self.xy[a]
            bx, by = self.xy[b]
            xs, ys = ax + u * (bx - ax), ay + u * (by - ay)
            la, lo = self.plane.inv(xs, ys)
            run = []
            for k in range(n):
                if ok[k]:
                    run.append((float(la[k]), float(lo[k])))
                elif run:
                    pieces.append(run)
                    run = []
            if run:
                pieces.append(run if len(run) > 1 else run * 2)
        return pieces


def _resample_polyline(plane: Plane, poly, fracs):
    """Points at fractions of a polyline's length (in a plane)."""
    x, y = plane.fwd([p[0] for p in poly], [p[1] for p in poly])
    seg = np.hypot(np.diff(x), np.diff(y))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    s = np.asarray(fracs) * total
    xi = np.interp(s, cum, x)
    yi = np.interp(s, cum, y)
    return plane.inv(xi, yi), total


# ---------------------------------------------------------------------------
# Filling
# ---------------------------------------------------------------------------
@dataclass
class TrackFill:
    fixes: list            # record + inferred, in time order
    gaps: list
    q: float
    model: str
    kind: str
    lines: list

    def inferred(self) -> list:
        return [f for f in self.fixes if f.tier == "inferred"]

    def to_geojson(self, name: str = "", ellipses_every: int = 0) -> dict:
        return to_geojson(self.fixes, name=name, ellipses_every=ellipses_every,
                          gaps=self.gaps)


def _context(fixes, g: Gap, window: int = 30):
    lo = max(0, g.i0 - window + 1)
    hi = min(len(fixes), g.i1 + window)
    return list(range(lo, hi))


def fill_gaps(fixes: list[Fix], method: str = "kalman", kind: str = "vehicle",
              step_s: float | None = None, model: str = "cv",
              q: float | None = None, factor: float = 3.0,
              min_gap_s: float | None = None, graph: RoadGraph | None = None,
              max_snap_m: float = 60.0, gate: float | None = GATE_P,
              max_speed: float | None = None) -> TrackFill:
    """Find the gaps in a track and fill each with `method` (great_circle,
    turn, kalman, road). Every filled point is tier "inferred" with
    `sigma_m` from the Kalman smoother's covariance at that instant (and,
    in `extra`, the ellipse: sigma_major_m, sigma_minor_m, ellipse_deg)."""
    if method not in METHODS:
        raise ValueError(f"unknown fill {method!r} — one of {', '.join(METHODS)}")
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r} — one of {', '.join(KINDS)}")
    if method == "road" and graph is None:
        raise ValueError("a road fill needs a road graph")
    fixes = sorted(fixes, key=lambda f: f.t)
    if len(fixes) < 3:
        return TrackFill(list(fixes), [], 0.0, model, kind,
                         ["fewer than three fixes — nothing to fill from"])
    meth = METHODS[method]
    vmax = float(max_speed or KINDS[kind]["max_speed"])
    sig0 = KINDS[kind]["sigma_m"]
    sig = np.array([f.sigma_m if f.sigma_m else sig0 for f in fixes])
    t_all = np.array([f.t for f in fixes])
    dts = np.diff(t_all)
    step = float(step_s or max(float(np.median(dts[dts > 0])) if np.any(dts > 0) else 1.0, 1e-3))
    la_c, lo_c = centroid([f.lat for f in fixes], [f.lon for f in fixes])
    gplane = Plane(la_c, lo_c)
    gx, gy = gplane.fwd([f.lat for f in fixes], [f.lon for f in fixes])
    gxy = np.stack([gx, gy], axis=1)
    gaps = find_gaps(fixes, factor, min_gap_s)
    q_fitted = None
    if q is None:
        # the record only: estimate on the fixes, gaps included (they are
        # intervals, not fixes, and the likelihood copes with them)
        q_fitted, _tbl = estimate_q(t_all, gxy, sig, model)
        floor = KINDS[kind]["q_floor"] if model == "cv" else KINDS[kind]["q_floor"] / 100.0
        q = max(q_fitted, floor)
    out = list(fixes)
    lines = []
    for g in gaps:
        a, b = fixes[g.i0], fixes[g.i1]
        if g.implied_speed_mps > vmax:
            g.why = (f"not filled: crossing it needs {g.implied_speed_mps:.0f} m/s "
                     f"and {KINDS[kind]['words']} does at most {vmax:.0f} m/s — "
                     "two different objects, or a bad fix")
            lines.append(g.why)
            continue
        ctx = _context(fixes, g)
        la_m, lo_m = midpoint(a.lat, a.lon, b.lat, b.lon)
        plane = Plane(la_m, lo_m)
        cx, cy = plane.fwd([fixes[i].lat for i in ctx], [fixes[i].lon for i in ctx])
        cxy = np.stack([cx, cy], axis=1)
        n_in = max(1, int(math.floor((g.t1 - g.t0) / step - 1e-9)))
        tq = g.t0 + step * np.arange(1, n_in + 1)
        tq = tq[tq < g.t1 - 1e-9]
        if tq.size == 0:
            tq = np.array([0.5 * (g.t0 + g.t1)])
        k_after = ctx.index(g.i1)
        kr = kalman_rts(t_all[ctx], cxy, sig[ctx], tq, model, q, gate_at=k_after)
        g.gate_d2 = kr["gate"]
        if gate is not None and kr["gate"] is not None:
            from scipy.stats import chi2
            lim = float(chi2.ppf(1.0 - gate, 2))
            if kr["gate"] > lim:
                g.why = (f"not filled: the fix after the gap is "
                         f"{math.sqrt(kr['gate']):.1f}σ from where the motion "
                         "before it predicts — a turn the record gave no hint "
                         "of, or two different objects")
                lines.append(g.why)
                continue
        qmask = ~kr["is_fix"]
        mean_q = kr["mean"][qmask]
        cov_q = kr["cov"][qmask]
        f_frac = (tq - g.t0) / (g.t1 - g.t0)
        if meth == "kalman_fill":
            lat_q, lon_q = plane.inv(mean_q[:, 0], mean_q[:, 1])
        elif meth == "great_circle_fill":
            lat_q, lon_q = slerp(a.lat, a.lon, b.lat, b.lon, f_frac)
        elif meth == "turn_fill":
            fm, vm = kr["mean"], kr["vel"]
            i_a = int(np.flatnonzero(kr["is_fix"] & (np.abs(kr["times"] - g.t0) < 1e-9))[0])
            i_b = int(np.flatnonzero(kr["is_fix"] & (np.abs(kr["times"] - g.t1) < 1e-9))[0])
            P = _turn_path(fm[i_a], vm[i_a], fm[i_b], vm[i_b], g.t1 - g.t0, f_frac)
            lat_q, lon_q = plane.inv(P[:, 0], P[:, 1])
        else:                                            # road
            sa, sb = graph.snap(a.lat, a.lon), graph.snap(b.lat, b.lon)
            if max(sa["dist_m"], sb["dist_m"]) > max_snap_m:
                g.why = (f"not filled by road: a gap end is "
                         f"{max(sa['dist_m'], sb['dist_m']):.0f} m from the "
                         f"nearest road in the graph (limit {max_snap_m:.0f} m)")
                lines.append(g.why)
                continue
            poly, L = graph.route(sa, sb)
            if not poly:
                g.why = "not filled by road: the graph has no route between the gap ends"
                lines.append(g.why)
                continue
            v_need = L / (g.t1 - g.t0)
            if v_need > vmax:
                g.why = (f"not filled by road: the road route is {L / 1000:.2f} km, "
                         f"which needs {v_need:.0f} m/s in the time available — "
                         f"more than {KINDS[kind]['words']} can do")
                lines.append(g.why)
                continue
            (lat_q, lon_q), _tot = _resample_polyline(graph.plane, poly, f_frac)
            # along-route uncertainty: a speed that varies by ~20 % about the
            # mean, as a Brownian bridge; plus the snap offsets
            along = 0.2 * L * np.sqrt(f_frac * (1 - f_frac))
            road_sig = np.sqrt(along ** 2 + 5.0 ** 2 + max(sa["dist_m"], sb["dist_m"]) ** 2)
        if meth in ("great_circle_fill", "turn_fill"):
            ox, oy = plane.fwd(np.atleast_1d(lat_q), np.atleast_1d(lon_q))
            offset = np.hypot(ox - mean_q[:, 0], oy - mean_q[:, 1])
        else:
            offset = np.zeros(tq.size)
        for j in range(tq.size):
            major, minor, ang = _sigma_axes(cov_q[j])
            sg = math.hypot(major, float(offset[j]))
            ex = {"sigma_major_m": round(major, 2), "sigma_minor_m": round(minor, 2),
                  "ellipse_deg": round(ang, 1), "gap": [g.t0, g.t1]}
            if offset[j] > 0:
                ex["offset_from_smoother_m"] = round(float(offset[j]), 2)
            if meth == "road_fill":
                sg = float(road_sig[j])
                ex["sigma_basis"] = "road: 20 % speed variation as a bridge + snap"
                ex["kalman_sigma_m"] = round(major, 2)
            out.append(Fix(float(tq[j]), float(np.atleast_1d(lat_q)[j]),
                           float(np.atleast_1d(lon_q)[j]), None, float(sg),
                           "inferred", meth, ex))
        g.filled, g.method, g.points = True, meth, int(tq.size)
        g.why = (f"filled with {tq.size} inferred point(s) by {method}; sigma "
                 f"up to {max(f.sigma_m for f in out[-tq.size:]):.0f} m")
        lines.append(g.why)
    out.sort(key=lambda f: f.t)
    head = (f"{len(gaps)} gap(s) in {len(fixes)} fixes; "
            f"{sum(g.filled for g in gaps)} filled, "
            f"{sum(not g.filled for g in gaps)} refused. Process noise "
            f"q = {q:.3g} {'m²/s³' if model == 'cv' else 'm²/s⁵'}"
            + ("" if q_fitted is None else
               (" (fitted to this track's own record)" if q == q_fitted else
                f" (the floor for {KINDS[kind]['words']}; the record alone "
                f"fitted {q_fitted:.2g})")) + ".")
    return TrackFill(out, gaps, float(q), model, kind, [head] + lines)


# ---------------------------------------------------------------------------
# Where does it reappear
# ---------------------------------------------------------------------------
@dataclass
class Region:
    t: float
    lat: float
    lon: float
    sigma_major_m: float
    sigma_minor_m: float
    ellipse_deg: float
    prob: float
    ring: list                       # [(lat, lon), …] closed
    tier: str = "inferred"
    method: str = "kalman_predict"

    def to_feature(self) -> dict:
        return {"type": "Feature",
                "geometry": {"type": "Polygon",
                             "coordinates": [[[lo, la] for la, lo in self.ring]]},
                "properties": {"tier": self.tier, "method": self.method,
                               "t": self.t, "prob": self.prob,
                               "sigma_major_m": self.sigma_major_m,
                               "sigma_minor_m": self.sigma_minor_m,
                               "words": f"where it may reappear, with "
                                        f"{self.prob:.0%} probability under a "
                                        "constant-velocity model — INFERRED"}}


def reappear(fixes: list[Fix], t_future: float, prob: float = 0.95,
             model: str = "cv", q: float | None = None, kind: str = "vehicle",
             n_ring: int = 72) -> Region:
    """The predicted region at `t_future` from the record's end."""
    fixes = sorted(fixes, key=lambda f: f.t)
    if len(fixes) < 2:
        raise ValueError("at least two fixes are needed to predict motion")
    if t_future <= fixes[-1].t:
        raise ValueError("the time asked about is not after the last fix")
    sig0 = KINDS.get(kind, KINDS["vehicle"])["sigma_m"]
    sig = np.array([f.sigma_m if f.sigma_m else sig0 for f in fixes])
    last = fixes[-1]
    plane = Plane(last.lat, last.lon)
    x, y = plane.fwd([f.lat for f in fixes], [f.lon for f in fixes])
    xy = np.stack([x, y], axis=1)
    ts = np.array([f.t for f in fixes])
    if q is None:
        q, _ = estimate_q(ts, xy, sig, model)
        floor = KINDS.get(kind, KINDS["vehicle"])["q_floor"]
        q = max(q, floor if model == "cv" else floor / 100.0)
    kr = kalman_rts(ts, xy, sig, [t_future], model, q)
    j = int(np.flatnonzero(~kr["is_fix"])[0])
    # the prediction is the FILTERED state run forward (no future data)
    m, C = kr["filtered_mean"][j], kr["filtered_cov"][j]
    from scipy.stats import chi2
    kk = math.sqrt(float(chi2.ppf(prob, 2)))
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, 0)
    ang = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    pts = (V @ (np.sqrt(w)[:, None] * kk * np.stack([np.cos(ang), np.sin(ang)]))).T + m
    la, lo = plane.inv(pts[:, 0], pts[:, 1])
    ring = [(float(a), float(b)) for a, b in zip(la, lo)]
    ring.append(ring[0])
    cla, clo = plane.inv(m[0], m[1])
    major, minor, ell = _sigma_axes(C)
    return Region(float(t_future), float(cla), float(clo), major, minor, ell,
                  float(prob), ring)


def reappear_on_roads(fixes: list[Fix], t_future: float, graph: RoadGraph,
                      v_min: float | None = None, v_max: float | None = None,
                      kind: str = "vehicle") -> dict:
    """The roads a vehicle could be on at `t_future`: network distance from
    the last fix between v_min·Δt and v_max·Δt (default: 0.5× and 1.5× its
    last measured speed, capped at the kind's maximum). A GeoJSON Feature
    (MultiLineString), tier inferred."""
    fixes = sorted(fixes, key=lambda f: f.t)
    last = fixes[-1]
    dt = float(t_future - last.t)
    if dt <= 0:
        raise ValueError("the time asked about is not after the last fix")
    plane = Plane(last.lat, last.lon)
    v = _velocity_at(plane, fixes, len(fixes) - 1, "before")
    speed = float(np.hypot(*v)) if v is not None else 0.0
    vmax_kind = KINDS[kind]["max_speed"]
    lo = 0.5 * speed if v_min is None else float(v_min)
    hi = min(vmax_kind, max(1.5 * speed, lo + 1.0)) if v_max is None else float(v_max)
    s = graph.snap(last.lat, last.lon)
    pieces = graph.reachable(s, lo * dt, hi * dt)
    return {"type": "Feature",
            "geometry": {"type": "MultiLineString",
                         "coordinates": [[[b, a] for a, b in p] for p in pieces]},
            "properties": {"tier": "inferred", "method": "road_reach",
                           "t": float(t_future), "v_min_mps": lo, "v_max_mps": hi,
                           "words": f"roads reachable in {dt:.0f} s at "
                                    f"{lo:.0f}–{hi:.0f} m/s from the last fix "
                                    "— INFERRED, not a sighting"}}


_prov.METHOD_TIERS.setdefault("road_reach", "inferred")


# ---------------------------------------------------------------------------
# Bearings
# ---------------------------------------------------------------------------
def fill_bearings(times, bearings_deg, sigma_deg: float = 2.0,
                  factor: float = 3.0, step_s: float | None = None,
                  q: float | None = None) -> dict:
    """Gaps in a DF bearing track, filled by a 1-D constant-rate Kalman/RTS
    on the UNWRAPPED angle. Returns {record: [...], inferred: [{t,
    bearing_deg, sigma_deg, tier, method}], q, gaps}."""
    t = np.asarray(times, dtype=np.float64)
    order = np.argsort(t)
    t = t[order]
    b = np.unwrap(np.radians(np.asarray(bearings_deg, dtype=np.float64)[order]))
    b = np.degrees(b)
    if t.size < 3:
        raise ValueError("at least three bearings are needed")
    dt = np.diff(t)
    nominal = float(np.median(dt[dt > 0]))
    step = float(step_s or nominal)
    gaps = [(int(i), int(i + 1)) for i in np.flatnonzero(dt > factor * nominal)]
    tq = []
    for i0, i1 in gaps:
        k = np.arange(t[i0] + step, t[i1] - 1e-9, step)
        tq += list(k)
    xy = np.stack([b, np.zeros_like(b)], axis=1)
    sg = np.full(t.size, float(sigma_deg))
    if q is None:
        q, _ = estimate_q(t, xy, sg, "cv", grid=10.0 ** np.linspace(-6, 2, 17))
    kr = kalman_rts(t, xy, sg, tq, "cv", q)
    qm = ~kr["is_fix"]
    inferred = []
    for tt, m, C in zip(kr["times"][qm], kr["mean"][qm], kr["cov"][qm]):
        inferred.append({"t": float(tt), "bearing_deg": float(m[0] % 360.0),
                         "sigma_deg": float(math.sqrt(max(C[0, 0], 0.0))),
                         "tier": "inferred", "method": "kalman_fill"})
    rec = [{"t": float(a), "bearing_deg": float(c % 360.0), "tier": "record"}
           for a, c in zip(t, b)]
    return {"record": rec, "inferred": inferred, "q": float(q),
            "gaps": [(float(t[i0]), float(t[i1])) for i0, i1 in gaps]}


# ---------------------------------------------------------------------------
# GeoJSON — LONGITUDE FIRST (the recorder's own warning)
# ---------------------------------------------------------------------------
def _iso(t: float) -> str:
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(t)))
    except (OverflowError, ValueError, OSError):
        return ""


def to_geojson(fixes: list[Fix], name: str = "", ellipses_every: int = 0,
               gaps=(), regions=()) -> dict:
    """FeatureCollection: the record as a LineString (tier record), every
    inferred point as a Point with tier/method/sigma, optional 1σ ellipses
    for every n-th inferred point, and any regions. GeoJSON order is
    [longitude, latitude]."""
    rec = [f for f in fixes if f.tier == "record"]
    feats = []
    if len(rec) >= 2:
        feats.append({"type": "Feature",
                      "geometry": {"type": "LineString",
                                   "coordinates": [[f.lon, f.lat] for f in rec]},
                      "properties": {"tier": "record", "name": name,
                                     "t_start": rec[0].t, "t_end": rec[-1].t,
                                     "fixes": len(rec)}})
    inf = [f for f in fixes if f.tier != "record"]
    for k, f in enumerate(inf):
        feats.append({"type": "Feature",
                      "geometry": {"type": "Point", "coordinates": [f.lon, f.lat]},
                      "properties": {"tier": f.tier, "method": f.method, "t": f.t,
                                     "time_utc": _iso(f.t), "sigma_m": f.sigma_m,
                                     "words": _prov.TIER_WORDS.get(f.tier, ""),
                                     **{k2: v for k2, v in f.extra.items()
                                        if k2 in ("sigma_major_m", "sigma_minor_m",
                                                  "ellipse_deg")}}})
        if ellipses_every and k % ellipses_every == 0 and \
                f.extra.get("sigma_major_m") is not None:
            plane = Plane(f.lat, f.lon)
            ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
            th = math.radians(f.extra.get("ellipse_deg", 0.0))
            a, b = f.extra["sigma_major_m"], f.extra["sigma_minor_m"]
            ex = a * np.cos(ang)
            ey = b * np.sin(ang)
            x = ex * math.sin(th) + ey * math.cos(th)
            y = ex * math.cos(th) - ey * math.sin(th)
            la, lo = plane.inv(x, y)
            ring = [[float(o), float(l)] for l, o in zip(la, lo)]
            ring.append(ring[0])
            feats.append({"type": "Feature",
                          "geometry": {"type": "Polygon", "coordinates": [ring]},
                          "properties": {"tier": f.tier, "t": f.t,
                                         "what": "1-sigma uncertainty"}})
    for r in regions:
        feats.append(r.to_feature() if isinstance(r, Region) else r)
    return {"type": "FeatureCollection", "name": name,
            "properties": {"made_by": "ATK Diffusion Toolkit — repair.tracks",
                           "note": "INFERRED points are a lead, never a position "
                                   "report; the LineString is the record.",
                           "gaps": [g.to_json() for g in gaps]},
            "features": feats}


def write_geojson(path, gj: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(gj, indent=1), encoding="utf-8")
    tmp.replace(p)
    return p
