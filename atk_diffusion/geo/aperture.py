# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The KrakenSDR as a moving synthetic aperture (plan E4).

Bill, 2026-10-08: *"I love … kraken as a moving synthetic aperture."* Five
coherent channels make a small array; driven, with every snapshot stamped
with GPS position and heading, it becomes a large one for locating an
emitter — the passive cousin of SAR.

DIRECT POSITION DETERMINATION, NOT BEARINGS-THEN-TRIANGULATION. The usual
two-step method estimates a bearing per snapshot and intersects them; each
step throws information away (a broad or two-peaked beam becomes one
number). Here every candidate position on the map is scored against every
snapshot's whole beam pattern at once — Weiss's direct position
determination (IEEE SPL 11(5), 2004), in its form for unknown waveforms:

    for each snapshot l:  R_l = X_l X_l^H / N           (5 x 5, coherent)
                          P_l(az) = a^H R_l a / max      (Bartlett, true azimuth:
                                                          the array's steering is
                                                          rotated by the heading)
    for each cell p:      L(p) = sum_l kappa_l * (P_l(bearing from snapshot l to p) - 1)

Coherent across the five channels, INCOHERENT across snapshots: an unknown
emitter's carrier phase does not survive minutes of driving, so snapshots
add as independent looks (no carrier-phase SAR is claimed). `kappa_l` turns
a beam pattern into a likelihood with a stated bearing error: near its
peak P ~ 1 - c (az - az0)^2, and kappa = 1 / (2 c sigma^2) makes that a
Gaussian of `sigma_deg` — the array-calibration, heading and multipath
error the operator states — while far from the peak the penalty saturates,
so one snapshot's multipath lobe cannot veto the right place. A linear
array (or a circle too wide for the frequency) has a two-peaked beam and
the posterior keeps both lobes.

The surface is a `posterior.GridPosterior` (coarse-to-fine grid): the
estimate, the cloud, the 50 / 90 / 95 % regions. `stationary_fix` gives the
comparator the plan names — one parked position's bearing, which has no
range at all — and `two_step` the classical bearings-then-fusion.

THE DIFFUSION SAMPLER IS NOT HERE. The plan says the inverse problem "is
where the diffusion posterior sampler earns its place". With no driven
data to learn a prior from, the posterior here is exact on a grid under a
stated error model; a learned sampler would have to beat it, measured.

GEOMETRY CONVENTIONS. Bearings are true, clockwise from north. The array
lies in the vehicle frame (x forward, y left): a uniform circle with
element 0 at `first_element_deg` counter-clockwise from forward, or a line
along `axis_deg`; `mount_offset_deg` is how far the array's forward mark is
turned from the vehicle's nose. Heading is the GPS course (valid while
moving) unless a compass supplies it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from atk_diffusion import provenance as _prov
from atk_diffusion.geo import posterior as _post
from atk_diffusion.geo import products as _products
from atk_diffusion.geo import terrain as _terrain

C_LIGHT = 299_792_458.0
_prov.METHOD_TIERS.setdefault("dpd", "inferred")


@dataclass
class ArrayGeometry:
    kind: str = "uca"
    n: int = 5
    radius_m: float = 0.5
    spacing_m: float = 0.0
    first_element_deg: float = 0.0
    clockwise: bool = False
    axis_deg: float = 90.0
    mount_offset_deg: float = 0.0

    @classmethod
    def uca(cls, radius_m: float, n: int = 5, **kw) -> "ArrayGeometry":
        return cls("uca", int(n), float(radius_m), 0.0, **kw)

    @classmethod
    def ula(cls, spacing_m: float, n: int = 5, **kw) -> "ArrayGeometry":
        return cls("ula", int(n), 0.0, float(spacing_m), **kw)

    def body_xy(self) -> np.ndarray:
        """(n, 2) element positions, metres: (forward, left)."""
        k = np.arange(self.n)
        if self.kind == "uca":
            sgn = -1.0 if self.clockwise else 1.0
            ang = np.radians(self.first_element_deg + sgn * 360.0 * k / self.n)
            xy = self.radius_m * np.stack([np.cos(ang), np.sin(ang)], axis=1)
        elif self.kind == "ula":
            off = (k - (self.n - 1) / 2.0) * self.spacing_m
            a = math.radians(self.axis_deg)
            xy = np.stack([off * math.cos(a), off * math.sin(a)], axis=1)
        else:
            raise ValueError("array kind is 'uca' or 'ula'")
        m = math.radians(self.mount_offset_deg)
        rot = np.array([[math.cos(m), -math.sin(m)], [math.sin(m), math.cos(m)]])
        return xy @ rot.T

    def enu(self, heading_deg: float) -> np.ndarray:
        """(n, 2) element positions (east, north) for a vehicle heading."""
        h = math.radians(heading_deg)
        fwd = np.array([math.sin(h), math.cos(h)])
        left = np.array([-math.cos(h), math.sin(h)])
        b = self.body_xy()
        return b[:, :1] * fwd[None, :] + b[:, 1:] * left[None, :]

    def neighbour_spacing_m(self) -> float:
        if self.kind == "uca":
            return 2.0 * self.radius_m * math.sin(math.pi / self.n)
        return self.spacing_m

    def ambiguous(self, freq_hz: float) -> bool:
        """True when a bearing has a twin: any linear array (front/back), or
        neighbours more than half a wavelength apart."""
        lam = C_LIGHT / float(freq_hz)
        return self.kind == "ula" or self.neighbour_spacing_m() > 0.5 * lam

    def to_json(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Snapshot:
    """One coherent capture from the array with where and how it sat."""
    iq: np.ndarray                  # (channels, samples) complex
    lat: float
    lon: float
    heading_deg: float
    t: float = 0.0


def steering(geom: ArrayGeometry, heading_deg: float, az_deg, freq_hz: float) -> np.ndarray:
    """(n, len(az)) far-field steering vectors toward true azimuths."""
    p = geom.enu(heading_deg)
    az = np.radians(np.atleast_1d(np.asarray(az_deg, dtype=np.float64)))
    u = np.stack([np.sin(az), np.cos(az)])                 # (2, A)
    return np.exp(2j * math.pi * float(freq_hz) / C_LIGHT * (p @ u))


def sample_covariance(iq: np.ndarray) -> np.ndarray:
    x = np.asarray(iq, dtype=np.complex128)
    if x.ndim != 2:
        raise ValueError("a snapshot is (channels, samples)")
    return (x @ x.conj().T) / x.shape[1]


def spectrum(R: np.ndarray, A: np.ndarray, method: str = "bartlett") -> np.ndarray:
    """Spatial spectrum over the steering columns, normalised to peak 1."""
    if method == "bartlett":
        p = np.real(np.einsum("ia,ij,ja->a", A.conj(), R, A))
    elif method == "music":
        w, v = np.linalg.eigh(R)
        En = v[:, :-1]                                     # one source
        p = 1.0 / np.maximum(np.sum(np.abs(En.conj().T @ A) ** 2, axis=0), 1e-12)
    else:
        raise ValueError("method is 'bartlett' or 'music'")
    p = np.maximum(p, 0.0)
    return p / max(float(p.max()), 1e-30)


def _peak(P: np.ndarray, az: np.ndarray) -> tuple[float, float]:
    """(peak azimuth, curvature c in 1/rad^2 with P ~ 1 - c d^2), refined by
    a parabola through the peak and its neighbours (azimuth periodic)."""
    i = int(np.argmax(P))
    n = P.size
    a, b, c0 = P[(i - 1) % n], P[(i + 1) % n], P[i]
    step = math.radians(float(az[1] - az[0]))
    den = a - 2 * c0 + b
    off = 0.0 if den >= 0 else 0.5 * (a - b) / den
    curv = max(-den / (2 * step * step), 1e-6)
    return float((az[i] + off * (az[1] - az[0])) % 360.0), curv


@dataclass
class ApertureResult:
    posterior: _post.GridPosterior
    estimate: tuple
    snapshot_bearings: list
    route: tuple
    words: str
    meta: dict = field(default_factory=dict)
    gains: np.ndarray | None = None          # self-calibrated channel gains

    def to_products(self, rf, *, run: str | None = None, label: str = "",
                    truth: tuple | None = None, truth_label: str = "") -> str:
        lat, lon = self.route
        extra = [("route.geojson",
                  [_products.line_feature(lat, lon, {"role": "driven route",
                                                     "snapshots": len(lat)})]
                  if len(lat) >= 2 else [], "measured"),
                 ("snapshot_bearings.geojson",
                  _post.bearing_features(self.snapshot_bearings,
                                         self.meta.get("search_radius_m", 20e3)),
                  "measured")]
        if truth is not None:
            extra.append(("truth.geojson",
                          [_products.point_feature(truth[0], truth[1],
                                                   {"role": "known position",
                                                    "label": truth_label})],
                          "measured"))
        return self.posterior.to_products(rf, kind="tracks", run=run,
                                          label=label or "synthetic-aperture",
                                          extra_features=extra,
                                          params=self.meta)


def _snapshot_terms(snapshots, geom, freq_hz, sigma_deg, method, az_step_deg,
                    gains=None):
    az = np.arange(0.0, 360.0, float(az_step_deg))
    g = np.ones(geom.n, dtype=np.complex128) if gains is None else np.asarray(gains)
    terms, bearings = [], []
    for s in snapshots:
        R = sample_covariance(s.iq)
        A = g[:, None] * steering(geom, s.heading_deg, az, freq_hz)
        P = spectrum(R, A, "bartlett")
        peak_az, c = _peak(P, az)
        if method == "music":       # sharper single bearing for the two-step
            peak_az, _ = _peak(spectrum(R, A, "music"), az)
        kappa = 1.0 / (2.0 * c * math.radians(sigma_deg) ** 2)
        terms.append((s, P, kappa))
        bearings.append(_post.Bearing(s.lat, s.lon, peak_az, sigma_deg, s.t,
                                      "kraken", geom.ambiguous(freq_hz)))
    return az, terms, bearings


def relative_angle_spread(snapshots, lat: float, lon: float) -> float:
    """How much of the array's own circle of directions the drive showed the
    emitter from, in degrees (0..360): 360 minus the largest gap between the
    array-relative bearings."""
    rel = np.sort([(float(_terrain.initial_bearing_deg(s.lat, s.lon, lat, lon))
                    - s.heading_deg) % 360.0 for s in snapshots])
    if rel.size < 2:
        return 0.0
    gaps = np.diff(np.concatenate([rel, [rel[0] + 360.0]]))
    return float(360.0 - gaps.max())


def self_calibrate(snapshots, geom: ArrayGeometry, freq_hz: float, lat: float,
                   lon: float) -> np.ndarray:
    """The channels' complex gains that best explain every snapshot as one
    emitter at (lat, lon): with D_l = diag(steering toward it),
    D_l^H R_l D_l ~ P g g^H + noise, so g is the principal eigenvector of
    their sum (each R_l scaled to unit power). Normalised to unit mean
    magnitude and zero phase on channel 0 (a common gain or phase changes
    no bearing)."""
    M = np.zeros((geom.n, geom.n), dtype=np.complex128)
    for s in snapshots:
        R = sample_covariance(s.iq)
        R = R / max(float(np.real(np.trace(R))), 1e-30)
        th = float(_terrain.initial_bearing_deg(s.lat, s.lon, lat, lon))
        a = steering(geom, s.heading_deg, th, freq_hz)[:, 0]
        M += (a.conj()[:, None] * R) * a[None, :]
    w, v = np.linalg.eigh(M)
    g = v[:, -1]
    g = g * np.exp(-1j * np.angle(g[0]))
    return g / np.mean(np.abs(g))


def locate_moving(snapshots, geom: ArrayGeometry, freq_hz: float, *,
                  sigma_deg: float = 3.0, search_radius_m: float = 20_000.0,
                  cell_m: float | None = None, max_cells: int = 200_000,
                  az_step_deg: float = 0.25, method: str = "bartlett",
                  calibrate: bool = True, iterations: int = 3,
                  min_spread_deg: float = 180.0) -> ApertureResult:
    """Direct position determination over a driven set of snapshots.

    `calibrate=True` (the default) also solves for the array's channel
    errors, alternating with the position: residual phase errors of a few
    degrees make bearing errors that depend on the angle the emitter is
    seen at relative to the array, and on a loop that angle follows the
    position — a systematic error that independent-bearing statistics
    cannot see and that moves the fix by kilometres. A drive that showed
    the emitter from less than `min_spread_deg` of the array's directions
    cannot separate the two, and calibration is skipped with a note."""
    snaps = list(snapshots)
    if not snaps:
        raise ValueError("no snapshots")
    if sigma_deg <= 0:
        raise ValueError("sigma_deg must be positive")
    meta = {"geometry": geom.to_json(), "freq_hz": float(freq_hz),
            "snapshots": len(snaps), "sigma_deg": float(sigma_deg),
            "search_radius_m": float(search_radius_m),
            "combination": "coherent across channels, incoherent across snapshots",
            "ambiguous_array": geom.ambiguous(freq_hz)}
    notes: list = []

    def solve(gains):
        az, terms, bearings = _snapshot_terms(snaps, geom, freq_hz, sigma_deg,
                                              method, az_step_deg, gains)
        az_ext = np.concatenate([az, [360.0]])

        def evaluate(grid):
            lat, lon = grid.mesh()
            ll = np.zeros(lat.shape)
            for s, P, kappa in terms:
                b = _terrain.initial_bearing_deg(s.lat, s.lon, lat, lon)
                ll += kappa * (np.interp(b, az_ext, np.concatenate([P, P[:1]])) - 1.0)
            return ll
        post = _post.adaptive_posterior(evaluate, [s.lat for s in snaps],
                                        [s.lon for s in snaps], method="dpd",
                                        search_radius_m=search_radius_m,
                                        cell_m=cell_m, max_cells=max_cells,
                                        meta=meta)
        return post, bearings
    post, bearings = solve(None)
    gains = None
    if calibrate:
        spread = relative_angle_spread(snaps, *post.map_point())
        meta["relative_angle_spread_deg"] = round(spread, 1)
        if spread < min_spread_deg:
            notes.append(f"the drive showed the emitter from only {spread:.0f} "
                         "degrees of the array's directions; the array was not "
                         "self-calibrated (drive a loop around or past it)")
        else:
            for _ in range(int(iterations)):
                gains = self_calibrate(snaps, geom, freq_hz, *post.map_point())
                post, bearings = solve(gains)
            meta["channel_phase_deg"] = [round(float(v), 2)
                                         for v in np.degrees(np.angle(gains))]
            meta["channel_gain_db"] = [round(float(v), 2)
                                       for v in 20 * np.log10(np.abs(gains))]
            notes.append("the array was self-calibrated on this drive "
                         f"(channel phases {meta['channel_phase_deg']} deg)")
    post.meta.update(meta)
    post.notes = list(notes)
    est = post.map_point()
    summ = post.summary()
    words = (f"{len(snaps)} snapshots over the drive, direct position "
             f"determination: {summ['words']}")
    res = ApertureResult(post, est, bearings,
                         ([s.lat for s in snaps], [s.lon for s in snaps]),
                         words, meta, gains)
    return res


def stationary_fix(snapshot: Snapshot, geom: ArrayGeometry, freq_hz: float, *,
                   sigma_deg: float = 3.0, search_radius_m: float = 20_000.0
                   ) -> tuple[_post.Bearing, _post.GridPosterior]:
    """The comparator: one parked position. A bearing, and its posterior —
    a wedge with no range, which the summary calls unbounded."""
    _, _, bearings = _snapshot_terms([snapshot], geom, freq_hz, sigma_deg,
                                     "bartlett", 0.25)
    b = bearings[0]
    return b, _post.locate([b], search_radius_m=search_radius_m,
                           method="dpd")


def two_step(result: ApertureResult, **kw) -> _post.GridPosterior:
    """The classical way, for comparison: each snapshot's peak bearing,
    fused as independent bearings (posterior.locate)."""
    return _post.locate(result.snapshot_bearings,
                        search_radius_m=result.meta.get("search_radius_m", 20e3),
                        **kw)


# ---------------------------------------------------------------------------
# Simulation: a known tower, a driven loop
# ---------------------------------------------------------------------------
def loop_route(lat: float, lon: float, radius_m: float, n: int,
               start_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """n points on a circle, driven clockwise (bearing from the centre
    increasing)."""
    az = (start_deg + 360.0 * np.arange(n) / n) % 360.0
    return _terrain.destination(lat, lon, az, radius_m)


def simulate_drive(tower_lat: float, tower_lon: float, route_lat, route_lon,
                   geom: ArrayGeometry, freq_hz: float, *, snr_db: float = 10.0,
                   n_samples: int = 256, heading_sigma_deg: float = 1.0,
                   gps_sigma_m: float = 3.0, phase_sigma_deg: float = 5.0,
                   gain_sigma_db: float = 0.5, multipath: tuple | None = None,
                   rng=None) -> list[Snapshot]:
    """Snapshots of an unknown-waveform emitter at a known position, seen by
    the array along a route. The array's channel gain and phase errors are
    fixed for the whole drive (a calibration error, not noise); position and
    heading carry GPS-like noise; `multipath=(relative_amplitude,
    extra_degrees)` adds a reflected path from a bearing offset."""
    rng = rng if rng is not None else np.random.default_rng()
    lat = np.asarray(route_lat, dtype=np.float64)
    lon = np.asarray(route_lon, dtype=np.float64)
    n = lat.size
    g = 10 ** (rng.normal(0, gain_sigma_db, geom.n) / 20.0) * \
        np.exp(1j * np.radians(rng.normal(0, phase_sigma_deg, geom.n)))
    noise = 10 ** (-snr_db / 20.0)
    out = []
    for i in range(n):
        j = (i + 1) % n
        heading = float(_terrain.initial_bearing_deg(lat[i], lon[i], lat[j], lon[j]))
        theta = float(_terrain.initial_bearing_deg(lat[i], lon[i], tower_lat, tower_lon))
        a = g * steering(geom, heading, theta, freq_hz)[:, 0]
        s = (rng.normal(size=n_samples) + 1j * rng.normal(size=n_samples)) / math.sqrt(2)
        x = np.outer(a, s)
        if multipath:
            amp, off = multipath
            am = g * steering(geom, heading, theta + off, freq_hz)[:, 0]
            sm = np.roll(s, 3) * amp * np.exp(1j * rng.uniform(0, 2 * math.pi))
            x = x + np.outer(am, sm)
        x = x + noise * (rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape)) / math.sqrt(2)
        dn, de = rng.normal(0, gps_sigma_m, 2)
        rl, ro = _terrain.from_enu(de, dn, lat[i], lon[i])
        out.append(Snapshot(x.astype(np.complex64), float(rl), float(ro),
                            (heading + rng.normal(0, heading_sigma_deg)) % 360.0,
                            float(i)))
    return out
