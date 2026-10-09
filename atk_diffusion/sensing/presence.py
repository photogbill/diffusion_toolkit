# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Presence and motion from CSI, against an empty-room baseline (plan §4.I2).

Bill likes both halves of passive sensing; this is the CSI half of I2 — the
same measurement as I1's, with a bigger target: a person walking changes the
multipath, and the CSI amplitudes move.

TWO STATISTICS PER WINDOW, both compared with the SAME statistic measured in
the empty room (`fit_baseline`), so the thresholds come from the room's own
noise and are not tuned by hand:

* MOTION — how much the amplitudes move inside the window: the median over
  subcarriers of the standard deviation of amplitude relative to the
  empty-room mean. A person walking moves it; an empty room does not.
* PRESENCE — how different the window's average amplitude PROFILE across
  subcarriers is from the empty room's: 1 − the correlation between the two.
  A person standing still changes the static multipath even without moving.

Each threshold is the (1 − pfa) quantile of the statistic over the
empty-room windows, times a margin (1.5 by default, because a quantile from a
few minutes of baseline is a rough estimate — stated, not hidden). With a
short baseline the false-alarm rate is only approximately `pfa`.

WHAT IT IS NOT. A proposal ("something moved here"), tier PROPOSED — not a
count of people, not who, and never identification of a person (plan §2.4).
Through walls and rubble, physics pushes back hardest (plan §4.I2): the
baseline must be taken in the same room with the same placement, and a
changed room (a door opened, a chair moved) is a changed baseline. The
FM/Kraken passive-radar half waits for B3 and belongs with atkpr.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from atk_diffusion import provenance

provenance.METHOD_TIERS.setdefault("csi_presence", "proposed")

METHOD = "csi_presence"
TIER = provenance.tier_for(METHOD)
LABEL = ("PROPOSED — presence/motion from Wi-Fi CSI against an empty-room "
         "baseline: not a count of people, not identification.")


def _amp(H) -> np.ndarray:
    X = np.asarray(H)
    A = np.abs(X) if np.iscomplexobj(X) else X.astype(np.float64)
    return np.asarray(A, dtype=np.float64)


def _windows(n: int, w: int, hop: int) -> list[tuple[int, int]]:
    if n < w:
        return []
    return [(a, a + w) for a in range(0, n - w + 1, hop)]


def _stats(A: np.ndarray, ref_profile: np.ndarray, live: np.ndarray
           ) -> tuple[float, float]:
    rel = A[:, live] / ref_profile[live]
    motion = float(np.median(np.std(rel, axis=0)))
    prof = A[:, live].mean(axis=0)
    c = np.corrcoef(prof, ref_profile[live])[0, 1] if live.sum() > 2 else 1.0
    return motion, float(1.0 - c)


@dataclass
class Baseline:
    fs: float
    window_s: float
    profile: list                    # empty-room mean amplitude per subcarrier
    live: list                       # subcarriers that carry signal
    motion_stats: list
    presence_stats: list
    motion_threshold: float
    presence_threshold: float
    pfa: float
    margin: float
    note: str = ""

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Baseline":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def fit_baseline(H, fs: float, *, window_s: float = 2.0, pfa: float = 0.01,
                 margin: float = 1.5) -> Baseline:
    """The empty room: its mean profile and the distribution of both
    statistics over its windows, and thresholds from them."""
    A = _amp(H)
    w = int(round(window_s * fs))
    wins = _windows(A.shape[0], w, max(1, w // 2))
    if len(wins) < 5:
        raise ValueError(f"an empty-room baseline needs at least "
                         f"{3 * window_s:.0f} s of CSI with nobody in the room")
    prof = A.mean(axis=0)
    live = prof > 0.05 * np.max(prof)
    ms, ps = [], []
    for a, b in wins:
        m, p = _stats(A[a:b], prof, live)
        ms.append(m)
        ps.append(p)
    q = 1.0 - float(pfa)
    note = ""
    if len(wins) < int(round(2.0 / max(pfa, 1e-6))):
        note = (f"the baseline has {len(wins)} windows — too few to pin a "
                f"{100 * pfa:g}% false-alarm rate; the margin of {margin:g} "
                "covers that, roughly. Record a longer empty room to tighten it.")
    return Baseline(fs=float(fs), window_s=float(window_s), profile=prof.tolist(),
                    live=live.tolist(), motion_stats=ms, presence_stats=ps,
                    motion_threshold=float(np.quantile(ms, q)) * margin,
                    presence_threshold=float(np.quantile(ps, q)) * margin,
                    pfa=float(pfa), margin=float(margin), note=note)


@dataclass
class PresenceReport:
    windows: list = field(default_factory=list)
    motion_fraction: float = 0.0
    presence_fraction: float = 0.0
    label: str = LABEL
    tier: str = TIER
    method: str = METHOD
    notes: list = field(default_factory=list)

    def lines(self) -> list[str]:
        n = len(self.windows)
        return [self.label,
                f"motion in {100 * self.motion_fraction:.0f}% of {n} windows; "
                f"a changed room (presence) in {100 * self.presence_fraction:.0f}%"
                ] + self.notes

    def to_json(self) -> dict:
        d = asdict(self)
        d["lines"] = self.lines()
        return d


def detect(H, fs: float, baseline: Baseline) -> PresenceReport:
    """Each window of `H` against the baseline. -> PresenceReport."""
    if abs(float(fs) - baseline.fs) > 1e-6:
        raise ValueError(f"the baseline was measured at {baseline.fs:g} CSI "
                         f"frames/s; this is {fs:g} — resample to the same grid")
    A = _amp(H)
    prof = np.asarray(baseline.profile, dtype=np.float64)
    live = np.asarray(baseline.live, dtype=bool)
    if A.shape[1] != prof.size:
        raise ValueError(f"the baseline has {prof.size} subcarriers; this CSI "
                         f"has {A.shape[1]} — a different frame format")
    w = int(round(baseline.window_s * fs))
    rep = PresenceReport()
    for a, b in _windows(A.shape[0], w, max(1, w // 2)):
        m, p = _stats(A[a:b], prof, live)
        rep.windows.append({"t0_s": a / fs, "t1_s": b / fs, "motion": m,
                            "presence": p,
                            "moving": m > baseline.motion_threshold,
                            "changed": p > baseline.presence_threshold,
                            "tier": TIER, "label": LABEL})
    n = len(rep.windows)
    if n:
        rep.motion_fraction = sum(x["moving"] for x in rep.windows) / n
        rep.presence_fraction = sum(x["changed"] for x in rep.windows) / n
    if baseline.note:
        rep.notes.append(baseline.note)
    return rep
