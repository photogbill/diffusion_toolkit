# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Propagation models, physics first (plan E5): free space, two-ray ground
reflection, knife-edge diffraction (Bullington and Deygout), and the NTIA
Irregular Terrain Model (Longley-Rice) through `itmlogic`.

Bill, 2026-10-08: *"we can use space loss and other propagation models to
show on the map the likely reach given the radio, power levels, antenna
patterns."* The plan's order is the order here: free space and two-ray as
the sanity line, then terrain-aware models.

* **Free space** — 20 log10(4 pi d / lambda). The floor no real path beats.
* **Two-ray** — the direct ray plus one reflected from flat ground, with the
  complex Fresnel reflection coefficient of the ground (permittivity and
  conductivity, either polarisation), not the asymptotic 40 log d. Flat
  earth: valid inside the radio horizon over open ground.
* **Knife edges** — the profile's obstructions as knife edges, loss J(v)
  from ITU-R P.526: Bullington (one equivalent edge where the steepest
  horizon rays from each end cross) and Deygout (the main edge, then the
  worst edge either side of it). Added to free space.
* **ITM / Longley-Rice** — the NTIA model (version 1.2.2) in point-to-point
  mode, through `itmlogic` 1.2 (MIT; Oughton et al., JOSS 2020). Its
  inputs follow the NTIA C++ `point_to_point()` exactly: mode-of-
  variability 12, the system elevation as the mean of the path's middle
  80 %, free space as 32.45 + 20 log f(MHz) + 20 log d(km). Validated in
  the tests against the ITS QKPFL test 1 published table (Crystal Palace
  to Mursley, 41.5 MHz, 77.8 km): all fifteen quantiles within 0.15 dB.
  If `itmlogic` is absent the model REFUSES in words; nothing falls back
  silently to a different model under ITM's name.

ITM'S OWN LIMITS, KEPT. 20 MHz to 20 GHz; 1 km to 2000 km; antennas 0.5 m
to 3000 m. ITM's error flag (`kwx`) travels with every result as a
sentence. One upstream quirk is recorded rather than patched: itmlogic's
`qlrpfl` reads the second-to-last profile point (not the last) for the
receiver's effective height on line-of-sight paths — a sub-metre effect on
a densely sampled profile.

Every number here is INFERRED tier (provenance): a model's prediction, not
a measurement. Tiers for Bullington and Deygout are registered at import.
"""

from __future__ import annotations

import cmath
import importlib.util
import math
from dataclasses import dataclass, field

import numpy as np

from atk_diffusion import provenance as _prov
from atk_diffusion.geo.terrain import K_STANDARD, Profile, earth_bulge_m

C_LIGHT = 299_792_458.0

_prov.METHOD_TIERS.setdefault("bullington", "inferred")
_prov.METHOD_TIERS.setdefault("deygout", "inferred")

#: Ground electrical constants (ITU-R P.527 style): (relative permittivity,
#: conductivity S/m). "average" is ITM's default.
GROUNDS = {
    "average": (15.0, 0.005), "poor": (4.0, 0.001), "good": (25.0, 0.020),
    "fresh_water": (81.0, 0.010), "sea_water": (81.0, 5.0),
    "city": (5.0, 0.001),
}

#: model name -> provenance method key
MODELS = {"fspl": "free_space", "two_ray": "two_ray",
          "bullington": "bullington", "deygout": "deygout", "itm": "itm"}

MODEL_WORDS = {
    "fspl": "free space (no terrain, no ground)",
    "two_ray": "two-ray flat-earth ground reflection",
    "bullington": "free space + Bullington knife-edge diffraction over the "
                  "DTED profile",
    "deygout": "free space + Deygout multiple knife-edge diffraction over the "
               "DTED profile",
    "itm": "the NTIA Irregular Terrain Model (Longley-Rice) over the DTED "
           "profile",
}


def wavelength_m(freq_hz: float) -> float:
    if freq_hz <= 0:
        raise ValueError("a frequency must be positive")
    return C_LIGHT / float(freq_hz)


# ---------------------------------------------------------------------------
# Free space and two-ray
# ---------------------------------------------------------------------------
def fspl_db(d_m, freq_hz: float) -> np.ndarray:
    """Free-space path loss, dB (distance floored at 1 m)."""
    d = np.maximum(np.asarray(d_m, dtype=np.float64), 1.0)
    return 20.0 * np.log10(4.0 * math.pi * d * float(freq_hz) / C_LIGHT)


def reflection_coefficient(psi_rad, freq_hz: float, eps_r: float,
                           sigma_s_m: float, polarization: str) -> np.ndarray:
    """Fresnel reflection coefficient of flat ground at grazing angle psi."""
    lam = wavelength_m(freq_hz)
    eta = complex(eps_r, -60.0 * lam * sigma_s_m)
    psi = np.asarray(psi_rad, dtype=np.float64)
    s, c2 = np.sin(psi), np.cos(psi) ** 2
    root = np.sqrt(eta - c2 + 0j)
    pol = polarization.lower()
    if pol.startswith("v"):
        return (eta * s - root) / (eta * s + root)
    if pol.startswith("h"):
        return (s - root) / (s + root)
    raise ValueError("polarization is 'vertical' or 'horizontal'")


def two_ray_db(d_m, freq_hz: float, h_tx_m, h_rx_m, eps_r: float = 15.0,
               sigma_s_m: float = 0.005,
               polarization: str = "vertical") -> np.ndarray:
    """Two-ray (direct + ground-reflected) path loss over flat ground, dB."""
    d = np.maximum(np.asarray(d_m, dtype=np.float64), 1.0)
    ht = np.maximum(np.asarray(h_tx_m, dtype=np.float64), 0.01)
    hr = np.maximum(np.asarray(h_rx_m, dtype=np.float64), 0.01)
    r1 = np.sqrt(d ** 2 + (ht - hr) ** 2)
    r2 = np.sqrt(d ** 2 + (ht + hr) ** 2)
    dr = 4.0 * ht * hr / (r1 + r2)          # r2 - r1 without cancellation
    k = 2.0 * math.pi * float(freq_hz) / C_LIGHT
    gamma = reflection_coefficient(np.arctan2(ht + hr, d), freq_hz, eps_r,
                                   sigma_s_m, polarization)
    field_ratio = np.abs(1.0 + gamma * (r1 / r2) * np.exp(-1j * k * dr))
    return fspl_db(r1, freq_hz) - 20.0 * np.log10(np.maximum(field_ratio, 1e-6))


# ---------------------------------------------------------------------------
# Knife-edge diffraction
# ---------------------------------------------------------------------------
def knife_edge_loss_db(v) -> np.ndarray:
    """ITU-R P.526 J(v): loss of a single knife edge, 0 for v <= -0.78."""
    v = np.asarray(v, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        j = 6.9 + 20.0 * np.log10(np.sqrt((v - 0.1) ** 2 + 1.0) + v - 0.1)
    return np.where(v > -0.78, j, 0.0)


@dataclass
class Diffraction:
    loss_db: float
    method: str
    edges: list = field(default_factory=list)      # [(distance_m, v, J_db)]
    line_of_sight: bool = True

    def words(self) -> str:
        if not self.edges:
            return f"{self.method}: no obstruction (the path is clear)"
        e = ", ".join(f"{d / 1000:.2f} km (v={v:.2f}, {j:.1f} dB)"
                      for d, v, j in self.edges)
        return f"{self.method}: {self.loss_db:.1f} dB over edges at {e}"


def _eff(prof: Profile, h_tx: float, h_rx: float, k: float):
    d = prof.distance_m
    D = prof.length_m
    ze = prof.ground_m + earth_bulge_m(d, D, k)
    hts = float(prof.ground_m[0]) + float(h_tx)
    hrs = float(prof.ground_m[-1]) + float(h_rx)
    return d, D, ze, hts, hrs


def bullington(prof: Profile, h_tx: float, h_rx: float, freq_hz: float,
               k: float = K_STANDARD, correction: bool = False) -> Diffraction:
    """Bullington's equivalent knife edge (ITU-R P.526 §4.5.1 form).
    `correction=True` adds P.526-13's empirical term
    [1 - exp(-L/6)] (10 + 0.02 d_km) for the spherical-earth part."""
    lam = wavelength_m(freq_hz)
    d, D, ze, hts, hrs = _eff(prof, h_tx, h_rx, k)
    di, zi = d[1:-1], ze[1:-1]
    if di.size == 0 or D <= 0:
        return Diffraction(0.0, "Bullington")
    stim = np.max((zi - hts) / di)
    str_ = (hrs - hts) / D
    if stim < str_:                       # line of sight: the worst edge
        h = zi - (hts * (D - di) + hrs * di) / D
        v = h * np.sqrt(2.0 * D / (lam * di * (D - di)))
        j = int(np.argmax(v))
        vb, db, los = float(v[j]), float(di[j]), True
    else:
        srim = np.max((zi - hrs) / (D - di))
        db = (hrs - hts + srim * D) / (stim + srim)
        db = min(max(db, di[0]), di[-1])
        hb = hts + stim * db - (hts * (D - db) + hrs * db) / D
        vb = float(hb * math.sqrt(2.0 * D / (lam * db * (D - db))))
        los = False
    luc = float(knife_edge_loss_db(vb))
    loss = luc
    if correction:
        loss += (1.0 - math.exp(-luc / 6.0)) * (10.0 + 0.02 * D / 1000.0)
    edges = [(db, vb, luc)] if luc > 0 else []
    return Diffraction(loss, "Bullington", edges, los)


def deygout(prof: Profile, h_tx: float, h_rx: float, freq_hz: float,
            k: float = K_STANDARD, depth: int = 1) -> Diffraction:
    """Deygout: the edge with the largest v over the whole path, then the
    largest-v edge of each side path, recursively to `depth` (1 = the
    textbook three edges). Losses add. The full-path effective profile
    serves every sub-path exactly: adding a straight line to the heights
    does not change any edge's height above its own chord."""
    lam = wavelength_m(freq_hz)
    d, D, ze, hts, hrs = _eff(prof, h_tx, h_rx, k)
    edges: list = []

    def worst(i0, i1, h0, h1):
        if i1 - i0 < 2:
            return None
        idx = np.arange(i0 + 1, i1)
        d1 = d[idx] - d[i0]
        d2 = d[i1] - d[idx]
        h = ze[idx] - (h0 + (h1 - h0) * d1 / (d1 + d2))
        v = h * np.sqrt(2.0 * (d1 + d2) / (lam * d1 * d2))
        j = int(np.argmax(v))
        return int(idx[j]), float(v[j])

    def recurse(i0, i1, h0, h1, level):
        w = worst(i0, i1, h0, h1)
        if w is None or w[1] <= -0.78:
            return
        m, v = w
        edges.append((float(d[m]), v, float(knife_edge_loss_db(v))))
        if level < depth:
            recurse(i0, m, h0, float(ze[m]), level + 1)
            recurse(m, i1, float(ze[m]), h1, level + 1)
    recurse(0, d.size - 1, hts, hrs, 0)
    los = not edges or edges[0][1] < 0
    edges.sort(key=lambda e: e[0])
    return Diffraction(float(sum(e[2] for e in edges)), "Deygout", edges, los)


# ---------------------------------------------------------------------------
# ITM / Longley-Rice
# ---------------------------------------------------------------------------
class ItmUnavailable(RuntimeError):
    """The Longley-Rice model cannot run here. The message says why."""


KWX_WORDS = {
    0: "",
    1: "ITM warning: some parameters are nearly out of range — use the "
       "result with caution",
    2: "ITM note: default values were substituted for impossible parameters",
    3: "ITM warning: a combination of parameters is out of range — the result "
       "is probably invalid",
    4: "ITM warning: some parameters are out of range — the result is "
       "probably invalid",
}


def itm_available() -> tuple[bool, str]:
    try:
        ok = importlib.util.find_spec("itmlogic") is not None
    except (ImportError, ValueError):
        ok = False
    if not ok:
        return False, ("The Longley-Rice model (ITM) needs the itmlogic package "
                       "(MIT; the Python port of NTIA ITM 1.2.2), which is not "
                       "installed in this environment. Free space, two-ray "
                       "and knife-edge diffraction still work; ITM does not "
                       "run under another model's name.")
    return True, ""


@dataclass
class ItmResult:
    loss_db: float                 # basic transmission loss at the quantile
    fspl_db: float
    mode: str
    kwx: int
    warning: str
    dist_m: float
    he_m: tuple
    dh_m: float
    reliability: float
    confidence: float
    quantiles: dict = field(default_factory=dict)

    @property
    def excess_db(self) -> float:
        return self.loss_db - self.fspl_db


def _zsys(pfl: list) -> float:
    """System elevation as the NTIA point_to_point() computes it: the mean
    of the profile's middle part (indices from the C++ reference)."""
    npts = int(pfl[0])
    ja = int(3.0 + 0.1 * npts)
    jb = npts - ja + 6
    vals = pfl[ja - 1:jb]
    return float(np.mean(vals)) if len(vals) else float(np.mean(pfl[2:]))


def itm_p2p(prof: Profile, freq_hz: float, h_tx_m: float, h_rx_m: float, *,
            polarization: str = "vertical", eps_r: float = 15.0,
            sigma_s_m: float = 0.005, n0: float = 301.0, climate: int = 5,
            zsys: float | None = None, reliability: float = 50.0,
            confidence: float = 50.0, mdvar: int = 12,
            quantiles=None) -> ItmResult:
    """Longley-Rice point-to-point over `prof`. `reliability` and
    `confidence` are percentages (50/50 is the median). `quantiles`, an
    optional list of (reliability, confidence) pairs, adds those too.
    Raises ItmUnavailable (in words) when itmlogic is missing."""
    ok, why = itm_available()
    if not ok:
        raise ItmUnavailable(why)
    from itmlogic.misc.qerfi import qerfi
    from itmlogic.preparatory_subroutines.qlrpfl import qlrpfl
    from itmlogic.preparatory_subroutines.qlrps import qlrps
    from itmlogic.statistics.avar import avar
    f_mhz = float(freq_hz) / 1e6
    if not 20.0 <= f_mhz <= 20000.0:
        raise ValueError(f"ITM is defined from 20 MHz to 20 GHz; "
                         f"{f_mhz:g} MHz is outside it")
    if prof.ground_m.size < 3 or prof.length_m <= 0:
        raise ValueError("ITM needs a profile of at least three points")
    pfl = prof.itm_pfl()
    zs = _zsys(pfl) if zsys is None else float(zsys)
    ipol = 1 if polarization.lower().startswith("v") else 0
    wn, gme, ens, zgnd = qlrps(f_mhz, zs, float(n0), ipol, float(eps_r),
                               float(sigma_s_m))
    prop = {"wn": wn, "gme": gme, "ens": ens, "zgnd": zgnd,
            "hg": [float(h_tx_m), float(h_rx_m)], "pfl": pfl,
            "klimx": int(climate), "mdvarx": int(mdvar), "lvar": 0,
            "kwx": 0, "klim": int(climate), "mdvar": int(mdvar), "mdp": -1}
    prop = qlrpfl(prop)
    dist = float(prop["dist"])
    fs = 32.45 + 20.0 * math.log10(f_mhz) + 20.0 * math.log10(dist / 1000.0)
    q = dist - prop["dla"]
    if int(q) < 0:
        mode = "line of sight"
    else:
        mode = "single horizon" if int(q) == 0 else "double horizon"
        dx = prop.get("dx", float("inf"))
        if dist <= prop["dlsa"] or dist <= dx:
            mode += ", diffraction dominant"
        else:
            mode += ", troposcatter dominant"
    pairs = [(float(reliability), float(confidence))] + \
        [(float(r), float(c)) for r, c in (quantiles or [])]
    out = {}
    for r, c in pairs:
        zr = qerfi([r / 100.0])[0]
        zc = qerfi([c / 100.0])[0]
        a, prop = avar(zr, 0, zc, prop)
        out[(r, c)] = float(fs + a)
    kwx = int(prop.get("kwx", 0))
    return ItmResult(loss_db=out[pairs[0]], fspl_db=float(fs), mode=mode,
                     kwx=kwx, warning=KWX_WORDS.get(kwx, f"ITM error {kwx}"),
                     dist_m=dist, he_m=tuple(float(h) for h in prop["he"]),
                     dh_m=float(prop["dh"]), reliability=float(reliability),
                     confidence=float(confidence), quantiles=out)


# ---------------------------------------------------------------------------
# One door for every model
# ---------------------------------------------------------------------------
@dataclass
class PathLoss:
    loss_db: float
    model: str
    method: str
    tier: str
    notes: list = field(default_factory=list)


def path_loss(model: str, prof: Profile, freq_hz: float, h_tx_m: float,
              h_rx_m: float, *, k: float = K_STANDARD,
              ground: str = "average", polarization: str = "vertical",
              itm: dict | None = None) -> PathLoss:
    """Path loss between two antennas over a profile, by name."""
    if model not in MODELS:
        raise ValueError(f"unknown propagation model {model!r} — one of "
                         f"{', '.join(MODELS)}")
    method = MODELS[model]
    tier = _prov.tier_for(method)
    D = prof.length_m
    eps, sig = GROUNDS.get(ground, GROUNDS["average"])
    notes = list(prof.notes)
    if model == "fspl":
        loss = float(fspl_db(D, freq_hz))
    elif model == "two_ray":
        loss = float(two_ray_db(D, freq_hz, h_tx_m, h_rx_m, eps, sig,
                                polarization))
    elif model in ("bullington", "deygout"):
        fn = bullington if model == "bullington" else deygout
        dif = fn(prof, h_tx_m, h_rx_m, freq_hz, k)
        loss = float(fspl_db(D, freq_hz)) + dif.loss_db
        notes.append(dif.words())
    else:
        r = itm_p2p(prof, freq_hz, h_tx_m, h_rx_m, polarization=polarization,
                    eps_r=eps, sigma_s_m=sig, **(itm or {}))
        loss = r.loss_db
        notes.append(f"ITM {r.mode}")
        if r.warning:
            notes.append(r.warning)
    return PathLoss(loss, model, method, tier, notes)
