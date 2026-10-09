# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The emitter library — fingerprints of known radios, matched with an
honest UNKNOWN (plan C1; the RF side of ATK's watchlist).

Lives in `rf_data\\products\\emitters\\` (plan §3.7), so it outlives an
install and travels to a new one:

    library.json        every emitter: id, name, receiver profile, the
                        running mean and spread of each feature, counts,
                        first/last seen, decoder ids, notes
    emitters.csv        the same, one emitter-feature per row (one item per
                        row, one field per column — Bill's export rule)
    sightings.geojson   every time an emitter was heard: where, when, on
                        what frequency, how close the match was
    sightings.csv       the same, one sighting per row
    manifest.json       hashes of all of it (products.ProductRun)

MATCHING. A fingerprint is compared only with emitters recorded through
the SAME receiver profile — the receiver's own imperfections are in every
feature (plan §3.1), so a match across receivers is refused in words,
never attempted. The distance is the RMS z-score over the features both
sides have: each feature scaled by the emitter's own spread when it has
enough observations, else the spread pooled over the library, never below
a stated floor (a feature measured identically twice is not infinitely
certain). A distance beyond the threshold is UNKNOWN: by default the
99.9 % point of chi-square for the number of features compared, and
`calibrate_threshold` sets it from held-out known and unknown radios so
the rejection rate is measured, not hoped (DETECTION_DESIGN §4: "a
detector that forces a class is a detector that lies").

A match is a PROPOSAL (provenance tier "proposed"): hardware fingerprints
drift with temperature, battery and age, and only a decode of the radio's
own identity confirms who it is.
"""

from __future__ import annotations

import csv
import io
import json
import math
import time
from dataclasses import dataclass, field

import numpy as np

from atk_diffusion import profiles as _profiles
from atk_diffusion import provenance as _prov
from atk_diffusion.fingerprint.features import NAMES, Fingerprint
from atk_diffusion.geo import products as _products

_prov.METHOD_TIERS.setdefault("fingerprint_match", "proposed")

UNKNOWN = "UNKNOWN"
FORMAT = 1
#: A feature's spread is never taken as smaller than this.
FLOORS = {"cfo_ppm": 0.03, "tx_irr_db": 0.5, "tx_dc_db": 1.0,
          "rise_time_ms": 0.03, "overshoot_pct": 1.0, "keyup_offset_hz": 20.0,
          "phase_noise_rms_hz": 2.0, "acpr_upper_db": 0.5,
          "acpr_lower_db": 0.5, "symbol_clock_ppm": 0.5}
MIN_OWN = 5        # observations before an emitter's own spread is trusted


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class Emitter:
    emitter_id: str
    name: str = ""
    profile: str = ""
    n: list = field(default_factory=lambda: [0] * len(NAMES))
    mean: list = field(default_factory=lambda: [0.0] * len(NAMES))
    m2: list = field(default_factory=lambda: [0.0] * len(NAMES))
    first_seen: str = ""
    last_seen: str = ""
    decoder_ids: list = field(default_factory=list)
    notes: str = ""

    def add(self, vec: np.ndarray, when: str = ""):
        for j, v in enumerate(vec):
            if not math.isfinite(v):
                continue
            self.n[j] += 1
            d = v - self.mean[j]
            self.mean[j] += d / self.n[j]
            self.m2[j] += d * (v - self.mean[j])
        when = when or _now()
        self.first_seen = self.first_seen or when
        self.last_seen = when

    def var(self) -> np.ndarray:
        n = np.asarray(self.n, dtype=np.float64)
        return np.where(n > 1, np.asarray(self.m2) / np.maximum(n - 1, 1), np.nan)

    def observations(self) -> int:
        return int(max(self.n)) if self.n else 0

    def to_json(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Match:
    emitter_id: str                  # or UNKNOWN
    name: str
    distance: float
    threshold: float
    candidates: list                 # [(id, name, distance)], nearest first
    features_compared: int
    tier: str = "proposed"
    words: str = ""


class ProfileRefused(ValueError):
    """A fingerprint was offered to emitters recorded through another
    receiver."""


class EmitterLibrary:
    """Load with `EmitterLibrary(rf)`; `save()` writes the product."""

    def __init__(self, rf, threshold: float | None = None):
        self.rf = rf
        self.dir = _products.product_dir(rf, "emitters")
        self.emitters: dict[str, Emitter] = {}
        self.sightings: list[dict] = []
        self.threshold = threshold
        self.calibration: dict = {}
        p = self.dir / "library.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("feature_names") and list(d["feature_names"]) != list(NAMES):
                raise ValueError("this library was written with another feature "
                                 "list; re-enrol its emitters")
            for e in d.get("emitters", []):
                em = Emitter(**{k: v for k, v in e.items()
                                if k in Emitter.__dataclass_fields__})
                self.emitters[em.emitter_id] = em
            self.sightings = list(d.get("sightings", []))
            if threshold is None:
                self.threshold = d.get("threshold")
            self.calibration = d.get("calibration", {})

    # -- enrolment -------------------------------------------------------------
    def _new_id(self) -> str:
        i = len(self.emitters) + 1
        while f"EMT-{i:04d}" in self.emitters:
            i += 1
        return f"EMT-{i:04d}"

    def add(self, fp: Fingerprint, emitter_id: str | None = None, *,
            name: str = "", when: str = "", decoder_id: str = "") -> str:
        """Enrol a fingerprint: to an existing emitter (by id) or a new one."""
        if not fp.profile:
            raise ValueError("a fingerprint without a receiver profile cannot "
                             "be enrolled — fingerprints are per receiver")
        if emitter_id and emitter_id in self.emitters:
            em = self.emitters[emitter_id]
            _profiles.check_match(em.profile, fp.profile,
                                  what=f"emitter {emitter_id}'s fingerprint")
        else:
            em = Emitter(emitter_id or self._new_id(), name, fp.profile)
            self.emitters[em.emitter_id] = em
        if name and not em.name:
            em.name = name
        em.add(fp.vector(), when)
        if decoder_id and decoder_id not in em.decoder_ids:
            em.decoder_ids.append(decoder_id)
        return em.emitter_id

    # -- matching --------------------------------------------------------------
    def _pooled_var(self, profile: str) -> np.ndarray:
        num = np.zeros(len(NAMES))
        den = np.zeros(len(NAMES))
        for em in self.emitters.values():
            if em.profile != profile:
                continue
            v = em.var()
            n = np.asarray(em.n, dtype=np.float64)
            ok = np.isfinite(v) & (n > 1)
            num[ok] += (n[ok] - 1) * v[ok]
            den[ok] += n[ok] - 1
        return np.where(den > 0, num / np.maximum(den, 1), np.nan)

    def distances(self, fp: Fingerprint) -> list[tuple[str, float, int]]:
        x = fp.vector()
        pooled = self._pooled_var(fp.profile)
        floors = np.array([FLOORS[nm] for nm in NAMES]) ** 2
        out = []
        for em in self.emitters.values():
            if em.profile != fp.profile:
                continue
            n = np.asarray(em.n, dtype=np.float64)
            own = em.var()
            var = np.where((n >= MIN_OWN) & np.isfinite(own), own, pooled)
            var = np.where(np.isfinite(var), var, 0.0) + floors
            var = var * (1.0 + 1.0 / np.maximum(n, 1))      # the mean's own error
            ok = np.isfinite(x) & (n > 0)
            if not ok.any():
                continue
            z2 = (x[ok] - np.asarray(em.mean)[ok]) ** 2 / var[ok]
            out.append((em.emitter_id, float(math.sqrt(np.mean(z2))), int(ok.sum())))
        out.sort(key=lambda t: t[1])
        return out

    def default_threshold(self, k: int) -> float:
        from scipy.stats import chi2
        k = max(int(k), 1)
        return float(math.sqrt(chi2.ppf(0.999, k) / k))

    def match(self, fp: Fingerprint) -> Match:
        """The nearest enrolled emitter of the same receiver profile, or
        UNKNOWN. Raises ProfileRefused when the library has emitters but
        none through this receiver."""
        same = [e for e in self.emitters.values() if e.profile == fp.profile]
        if not same:
            others = sorted({e.profile for e in self.emitters.values()})
            if others:
                raise ProfileRefused(
                    f"the library holds emitters recorded through "
                    f"{', '.join(others)}; this fingerprint was measured through "
                    f"{fp.profile or 'an unnamed receiver'}. Fingerprints carry "
                    "the receiver's own imperfections and are compared only "
                    "within one receiver profile.")
            return Match(UNKNOWN, "", float("inf"), float("nan"), [], 0,
                         words="the library is empty")
        ds = self.distances(fp)
        if not ds:
            return Match(UNKNOWN, "", float("inf"), float("nan"), [], 0,
                         words="no feature in common with any emitter")
        best_id, best_d, k = ds[0]
        thr = float(self.threshold) if self.threshold else self.default_threshold(k)
        cands = [(i, self.emitters[i].name, d) for i, d, _ in ds[:5]]
        if best_d > thr:
            return Match(UNKNOWN, "", best_d, thr, cands, k,
                         words=(f"UNKNOWN: the nearest emitter ({best_id}) is "
                                f"{best_d:.2f} away, beyond the {thr:.2f} "
                                "threshold"))
        em = self.emitters[best_id]
        words = (f"proposed: {em.name or best_id} at distance {best_d:.2f} "
                 f"(threshold {thr:.2f}, {k} features)")
        if len(ds) > 1 and ds[1][1] < 1.25 * best_d:
            words += (f"; {ds[1][0]} is nearly as close ({ds[1][1]:.2f}) — "
                      "treat as ambiguous")
        return Match(best_id, em.name, best_d, thr, cands, k, words=words)

    def calibrate_threshold(self, known: list, unknown: list | None = None,
                            target_accept: float = 0.95) -> dict:
        """Set the threshold from held-out fingerprints of ENROLLED radios
        (`known`: [(Fingerprint, true_id)]) so `target_accept` of them are
        accepted; report what that does to held-out UNKNOWN radios."""
        d_known = []
        for fp, true_id in known:
            ds = dict((i, d) for i, d, _ in self.distances(fp))
            if true_id in ds:
                d_known.append(ds[true_id])
        if not d_known:
            raise ValueError("no held-out fingerprint of an enrolled emitter")
        thr = float(np.quantile(d_known, target_accept, method="higher"))
        self.threshold = thr
        res = {"threshold": thr, "known_accepted": float(np.mean(np.array(d_known) <= thr)),
               "n_known": len(d_known)}
        if unknown:
            d_un = [self.distances(fp)[0][1] if self.distances(fp) else float("inf")
                    for fp in unknown]
            res["unknown_rejected"] = float(np.mean(np.array(d_un) > thr))
            res["n_unknown"] = len(d_un)
        self.calibration = res
        return res

    # -- sightings -------------------------------------------------------------
    def add_sighting(self, emitter_id: str, t, lat: float | None, lon: float | None,
                     freq_hz: float | None = None, *, distance: float | None = None,
                     decoder_id: str = "", profile: str = "", **props) -> dict:
        s = {"emitter_id": emitter_id, "t": t, "lat": lat, "lon": lon,
             "freq_hz": freq_hz, "distance": distance, "decoder_id": decoder_id,
             "profile": profile, **props}
        self.sightings.append(s)
        em = self.emitters.get(emitter_id)
        if em is not None:
            when = t if isinstance(t, str) else time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(t))) if t is not None else ""
            em.last_seen = max(em.last_seen or "", when or "")
            em.first_seen = em.first_seen or when
            if decoder_id and decoder_id not in em.decoder_ids:
                em.decoder_ids.append(decoder_id)
        return s

    # -- the product ------------------------------------------------------------
    def save(self) -> str:
        pr = _products.ProductRun(self.rf, "emitters", tier="proposed",
                                  method="fingerprint_match",
                                  params={"feature_names": list(NAMES),
                                          "threshold": self.threshold,
                                          "floors": FLOORS})
        lib = {"format": FORMAT, "updated": _now(), "feature_names": list(NAMES),
               "threshold": self.threshold, "calibration": self.calibration,
               "emitters": [e.to_json() for e in self.emitters.values()],
               "sightings": self.sightings}
        pr.add_json("library.json", lib, tier="proposed", role="library")
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\r\n")
        w.writerow(["emitter_id", "name", "profile", "feature", "mean", "std", "n"])
        for em in self.emitters.values():
            v = em.var()
            for j, nm in enumerate(NAMES):
                if em.n[j]:
                    w.writerow([em.emitter_id, em.name, em.profile, nm,
                                repr(float(em.mean[j])),
                                "" if not math.isfinite(v[j]) else repr(float(math.sqrt(v[j]))),
                                em.n[j]])
        pr.add_text("emitters.csv", "﻿" + buf.getvalue(), tier="proposed",
                    role="table")
        feats, rows = [], io.StringIO()
        w = csv.writer(rows, lineterminator="\r\n")
        w.writerow(["emitter_id", "t", "lat", "lon", "freq_hz", "distance",
                    "decoder_id", "profile"])
        for s in self.sightings:
            w.writerow(["" if s.get(k) is None else s.get(k) for k in
                        ("emitter_id", "t", "lat", "lon", "freq_hz", "distance",
                         "decoder_id", "profile")])
            if s.get("lat") is not None and s.get("lon") is not None:
                props = {k: v for k, v in s.items() if k not in ("lat", "lon")}
                feats.append(_products.point_feature(s["lat"], s["lon"], props))
        pr.add_text("sightings.csv", "﻿" + rows.getvalue(), tier="proposed",
                    role="table")
        pr.add_geojson("sightings.geojson", feats, tier="proposed",
                       layer={"name": "emitter sightings", "role": "sightings"})
        pr.finish(emitters=len(self.emitters), sightings=len(self.sightings))
        return str(pr.dir)
