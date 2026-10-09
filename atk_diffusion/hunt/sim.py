# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""A simulated band and receiver for the hunter (plan §4.B6, first
experiment — without hardware).

The plan's first experiment puts a bladeRF at low power on a cable loop into
the hunter's receiver, playing a scripted sequence of bursts at random times
and frequencies, and measures time-to-find and fraction found, hunter versus
a fixed scan. This is the same experiment with the cable replaced by
arithmetic, so the whole code path — goal, policy, validation, log, the
receiver interface — is exercised and measured before any RF is involved.

THE SCRIPT (`scripted_band`). Radio traffic is not uniformly random: a net
that keyed up once keys up again on the same channel. So the script has
EMITTERS — each on a fixed narrowband channel, keying up at random times
(Poisson, a mean interval of tens of seconds, bursts of 1–5 s) — plus
one-off bursts on random frequencies that never repeat, plus distractors the
goal is NOT about: continuous carriers and wideband bursts. A hunter that
remembers where matches were has something to exploit; one that does not
(the fixed scan) does not. If memory does not help on Bill's band, the real
experiment will say so — the script only decides what is tested.

THE RECEIVER (`SimReceiver`) implements `hunt.policy.Receiver`: retune costs
the profile family's retune latency (ATK's sweep.py numbers) during which it
hears nothing; a dwell sees the signals whose centre lies inside the current
span for at least `min_overlap_s`, each detected with a probability that
rises with SNR (a logistic around `snr50_db`), and reports them as
`detect.boxes.Detection`s cut to the look (so a burst that started or ended
inside the look is visibly a burst); false alarms arrive as a Poisson
process. Each true detection carries `measurements["sim_signal"]` — the
ground truth the experiment scores against, never shown to the policy.

LIMITS. Gain is accepted and ignored. Detection probability is a stated
curve, not a model of Bill's detector; the experiment's numbers are
comparisons between policies under the same curve, not predictions of the
field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from atk_diffusion.detect.boxes import Detection


@dataclass
class SimSignal:
    id: int
    t0: float
    t1: float                 # math.inf for continuous
    f_hz: float
    bw_hz: float
    snr_db: float
    family: str = "fm"
    cls: str = ""
    emitter: str = ""
    kind: str = "burst"       # burst | oneoff | continuous | wide_burst

    @property
    def duration(self) -> float:
        return self.t1 - self.t0


@dataclass
class SimBand:
    f_lo_hz: float
    f_hi_hz: float
    duration_s: float
    signals: list = field(default_factory=list)

    def goal_truth(self, goal) -> list[SimSignal]:
        """The signals the goal is looking for, judged on their TRUE
        parameters (band, bandwidth, burstiness, family/class)."""
        out = []
        for s in self.signals:
            if goal.f_lo_hz is not None and not (goal.f_lo_hz <= s.f_hz <= goal.f_hi_hz):
                continue
            if goal.bw_max_hz is not None and s.bw_hz > goal.bw_max_hz:
                continue
            if goal.bw_min_hz is not None and s.bw_hz < goal.bw_min_hz:
                continue
            bursty = math.isfinite(s.t1) and s.duration <= 5.0
            if goal.burstiness == "bursty" and not bursty:
                continue
            if goal.burstiness == "continuous" and bursty:
                continue
            if goal.classes and s.cls not in goal.classes:
                continue
            if goal.families and s.family not in goal.families:
                continue
            if s.t0 >= self.duration_s:
                continue
            out.append(s)
        return out


def scripted_band(f_lo_hz: float, f_hi_hz: float, duration_s: float, *,
                  rng=None, n_emitters: int = 6, mean_interval_s=(30.0, 120.0),
                  burst_s=(1.0, 5.0), n_oneoff: int = 6, n_continuous: int = 2,
                  n_wide_bursts: int = 3, snr_db=(8.0, 25.0),
                  channel_bw_hz=(6_250.0, 12_500.0, 25_000.0)) -> SimBand:
    """A scripted band (see the module docstring)."""
    rng = rng or np.random.default_rng(0)
    sig: list[SimSignal] = []
    nid = 0
    edge = 50_000.0
    lo, hi = f_lo_hz + edge, f_hi_hz - edge
    fams = (("nfm_voice", "fm"), ("dmr", "fsk"), ("p25", "fsk"), ("pocsag", "fsk"))
    for e in range(n_emitters):
        f = float(rng.uniform(lo, hi))
        bw = float(rng.choice(channel_bw_hz))
        cls, fam = fams[int(rng.integers(len(fams)))]
        mean = float(rng.uniform(*mean_interval_s))
        t = float(rng.exponential(mean))
        snr = float(rng.uniform(*snr_db))
        while t < duration_s:
            d = float(rng.uniform(*burst_s))
            sig.append(SimSignal(nid, t, t + d, f, bw, snr + float(rng.normal(0, 1.5)),
                                 fam, cls, f"emitter{e}", "burst"))
            nid += 1
            t += d + float(rng.exponential(mean))
    for _ in range(n_oneoff):
        t = float(rng.uniform(0, duration_s))
        cls, fam = fams[int(rng.integers(len(fams)))]
        sig.append(SimSignal(nid, t, t + float(rng.uniform(*burst_s)),
                             float(rng.uniform(lo, hi)),
                             float(rng.choice(channel_bw_hz)),
                             float(rng.uniform(*snr_db)), fam, cls, "", "oneoff"))
        nid += 1
    for _ in range(n_continuous):
        sig.append(SimSignal(nid, 0.0, math.inf, float(rng.uniform(lo, hi)),
                             float(rng.choice((12_500.0, 200_000.0))),
                             float(rng.uniform(15, 30)), "fm", "", "", "continuous"))
        nid += 1
    for _ in range(n_wide_bursts):
        t = float(rng.uniform(0, duration_s))
        sig.append(SimSignal(nid, t, t + float(rng.uniform(0.5, 3.0)),
                             float(rng.uniform(lo, hi)), 1_000_000.0,
                             float(rng.uniform(10, 25)), "ofdm", "", "",
                             "wide_burst"))
        nid += 1
    sig.sort(key=lambda s: s.t0)
    return SimBand(f_lo_hz, f_hi_hz, duration_s, sig)


class SimReceiver:
    """`hunt.policy.Receiver` over a `SimBand` (see the module docstring)."""

    def __init__(self, band: SimBand, limits, *, rng=None,
                 snr50_db: float = 6.0, slope_db: float = 1.5,
                 min_overlap_s: float = 0.05,
                 false_alarms_per_s_per_mhz: float = 0.002,
                 start_hz: float | None = None, profile: str = ""):
        self.band = band
        self.limits = limits
        self.rng = rng or np.random.default_rng(0)
        self.snr50_db = snr50_db
        self.slope_db = slope_db
        self.min_overlap_s = min_overlap_s
        self.fa_rate = false_alarms_per_s_per_mhz
        self.t = 0.0
        self.center_hz = float(start_hz if start_hz is not None
                               else 0.5 * (band.f_lo_hz + band.f_hi_hz))
        self.span_hz = float(limits.usable_span_hz)
        self.gain_db = None
        self.profile = profile
        self.retunes = 0
        self.first_seen: dict[int, float] = {}
        self.calls: list[tuple] = []

    # -- the Receiver interface ----------------------------------------------
    def now(self) -> float:
        return self.t

    def retune(self, center_hz: float) -> None:
        self.calls.append(("retune", float(center_hz)))
        if abs(float(center_hz) - self.center_hz) > 0.5:
            self.t += float(self.limits.retune_latency_s)   # deaf while it moves
            self.retunes += 1
        self.center_hz = float(center_hz)

    def set_span(self, span_hz: float) -> None:
        self.calls.append(("set_span", float(span_hz)))
        self.span_hz = min(float(span_hz), float(self.limits.usable_span_hz))

    def set_gain(self, gain_db: float) -> None:
        self.calls.append(("set_gain", float(gain_db)))
        self.gain_db = float(gain_db)

    def p_detect(self, snr_db: float) -> float:
        return 1.0 / (1.0 + math.exp(-(float(snr_db) - self.snr50_db) / self.slope_db))

    def dwell(self, seconds: float) -> list[Detection]:
        self.calls.append(("dwell", float(seconds)))
        a, b = self.t, self.t + float(seconds)
        lo = self.center_hz - 0.5 * self.span_hz
        hi = self.center_hz + 0.5 * self.span_hz
        out = []
        for s in self.band.signals:
            if s.t0 >= b or s.t1 <= a or not (lo <= s.f_hz <= hi):
                continue
            t0, t1 = max(s.t0, a), min(s.t1, b)
            if t1 - t0 < self.min_overlap_s:
                continue
            if self.rng.random() > self.p_detect(s.snr_db):
                continue
            meas_bw = s.bw_hz * float(self.rng.uniform(0.85, 1.15))
            fc = s.f_hz + float(self.rng.normal(0.0, 0.05 * s.bw_hz))
            named = self.rng.random() < 0.7
            out.append(Detection(
                t0=t0, t1=t1, f_lo=fc - 0.5 * meas_bw, f_hi=fc + 0.5 * meas_bw,
                sources=("energy",), family=s.family if named else "unknown",
                cls=s.cls if named and s.cls else "",
                confidence=float(self.rng.uniform(0.4, 0.95)) if named else None,
                snr_db=float(s.snr_db + self.rng.normal(0, 1.0)),
                profile=self.profile,
                measurements={"sim_signal": int(s.id)}))
            self.first_seen.setdefault(s.id, b)
        n_fa = int(self.rng.poisson(self.fa_rate * float(seconds)
                                    * self.span_hz / 1e6))
        for _ in range(n_fa):
            fc = float(self.rng.uniform(lo, hi))
            bw = float(self.rng.uniform(3_000, 20_000))
            t0 = float(self.rng.uniform(a, b))
            out.append(Detection(t0=t0, t1=min(b, t0 + 0.1), f_lo=fc - bw / 2,
                                 f_hi=fc + bw / 2, sources=("energy",),
                                 family="unknown", snr_db=float(
                                     self.rng.uniform(5, 8)),
                                 profile=self.profile,
                                 measurements={"sim_signal": -1}))
        self.t = b
        return out
