# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Sifting — the deep scan for signals BELOW the noise floor
(DETECTION_DESIGN §4.4).

Bill, 2026-10-12: *"an automated 'sifting' mode that is specifically for
looking for and automatically recording signals below the noise floor. Its
an LPI/LPD detector. The Operator would set a bandwidth and a list of
frequencies to hop through, and it would take long listens looking for
signals."* This module is the engine of it; ATK drives the radio
(`atk/ui/sift_page.py`, `atk/core/sift_worker.py`).

WHAT IS DIFFERENT FROM THE CYCLIC PROPOSER. `cyclo.proposer` finds signals
whose symbol rate is in the class table, and nothing else. Sifting is
BLIND: every cycle frequency from a few hertz to half the channel rate, at
several channel widths, conjugate and non-conjugate, with a threshold
derived from a false-alarm rate. It finds "a modulated signal with a clock
at α is here" without knowing what it is — which is the only kind of
finding an LPI search can promise.

THE FIVE DECISIONS (each argued in §4.4):

1. RECORD FIRST, DECIDE AFTER. A detection only exists at the end of a
   dwell, so a buffer that starts on a detection is always too late. Every
   dwell is written to disk as it arrives (`DwellStore`, in the radio's own
   bytes); the search runs on the file; a dwell with nothing new is
   deleted; one with something new is HELD while the same stop is listened
   to again (the confirmation dwell), and both are kept only if the second
   reproduces the feature. Two independent looks square the false-alarm
   probability and give a longer recording of anything real.
2. SEGMENTED INTEGRATION. Coherent integration pays only while a feature
   stays inside one α bin (1/T). Clock and LO drift end that within
   seconds, so a dwell is cut into coherent segments (`coherent_s`) whose
   statistics are SUMMED. Per segment, per α bin, per lag the statistic is
   exponential(1) under the null, so the sum of K is Gamma(K, 1) and the
   threshold comes from `gammainccinv` and the number of cells searched —
   derived, not tuned. Past the segment length each doubling of dwell buys
   ~0.75 dB instead of ~1.5 dB, which `plan` says before the run.
3. SEVERAL CHANNEL WIDTHS. Cyclic sensitivity is set by IN-BAND SNR, so a
   25 kHz signal searched as part of a 1 MHz channel loses ~√(1 MHz/25 kHz)
   ≈ 8 dB. The stop is cut by the proposer's polyphase channelizer at
   widths a factor `LEVEL_FACTOR` apart (the stop's bandwidth, ÷4, ÷16, …,
   down to `min_width_hz`), half-overlapping, so any signal sits whole in
   a channel within a factor of 4 of its own width (≤ 3 dB from matched).
4. THE ENVIRONMENT IS THE REAL FALSE-ALARM SOURCE. Below the floor a blind
   search finds every periodic thing nearby: switch-mode supplies, LED
   drivers, USB, the radio's own spurs, I/Q-imbalance images. The derived
   threshold controls statistical false alarms only. So every feature is
   CLASSIFIED (`Ledger.classify`), and only NEW ones are recorded:
     receiver   — learned from dwells with the antenna replaced by a 50 Ω
                  terminator (`calibrate`); stored per receiver profile
     local      — the same α in three or more places further apart than
                  the stop's bandwidth (so at different stops): a broadband
                  local source, not an emitter
     background — seen at this stop during the baseline pass(es)
     dc         — a conjugate line at α≈0 in the channel holding the tuned
                  centre: indistinguishable from the radio's DC offset
     visible    — the channel's ENERGY is measurably above the floor: the
                  ordinary detector sees it; not a sub-floor find
     explained  — α matches a class-table rate (a distant P25, a pager)
     new        — none of the above. Recorded.
   Nothing is hidden from view: every class is listed, colour-coded; only
   the recording decision depends on it.
5. FALSE RECORDINGS PER DAY, NOT A Pfa. The operator says how many false
   recordings a day are acceptable; `false_alarm_split` turns that and the
   dwell rate into the first-look and confirmation false-alarm
   probabilities.

WHAT IT CANNOT DO, stated in `LIMITS` and in the plan's words: a burst
shorter than a few hundred symbols leaves no line; a frequency hopper is
seen in fragments (beaten by bandwidth, not dwell); and a waveform built
without cyclic features (dithered timing, noise-like) can hide from it.

Pure numpy + scipy; no Qt, no hardware. Every function that takes long
takes `progress` and `cancel` callables. `log` is any `logging.Logger`
(ATK passes its own `atk.sift`); every decision ALSO goes to the run's
JSON-lines log (`SiftLog`).
"""

from __future__ import annotations

import functools
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from atk_diffusion import sigmf as _sigmf
from atk_diffusion.dsp import iq as _iq

_LOG = logging.getLogger("atk_diffusion.sift")

# ---------------------------------------------------------------------------
# Constants — each with its reason
# ---------------------------------------------------------------------------
#: Channel widths are this factor apart. 4 keeps every signal within a
#: factor of 4 of a matched channel — at most √4 = 3 dB of required SNR —
#: at the price of one more level than a factor of 8 would cost.
LEVEL_FACTOR = 4
#: Channels at a level are spaced half their width apart, so a signal up to
#: half the width lies whole inside one of them.
OVERLAP = 2
#: Non-conjugate lags, in samples at the channel rate. A channel is sampled
#: at ≥ 2× its width and a signal it holds is between ¼ and all of that
#: width, so its symbol is ~2.7–11 samples long; these lags span up to about
#: a symbol, where the symbol-rate line of every linear and FSK modulation
#: lives (lag 0 is the squared envelope).
LAGS_NC = (0, 1, 2, 4, 8)
#: Conjugate lags. The carrier line of BPSK / AM / ASK / MSK is strongest at
#: lag 0 (x² has a line at 2·f_c); 2 catches the pulse-shaped ones. (Five
#: and two lags, not six and three, since 2026-10-12: neighbouring lags of
#: a channel sampled at 2.5× its width are strongly correlated — measured,
#: every lag of a QPSK line within 30 % of the best — and the dropped ones
#: were a fifth of the search.)
LAGS_C = (0, 2)
#: The lowest non-conjugate α searched, in α bins: below it the record's
#: own power term leaks (the window's main lobe is ~4 bins each side).
ALPHA_MIN_BINS = 8
#: A feature seen in this many places, each further from the others than
#: the stop's bandwidth, is a local broadband source (a switch-mode supply,
#: a USB cable), not an emitter. Further apart than the BANDWIDTH, not than
#: the feature's own channel: one wideband emitter fills several narrow
#: channels of ONE stop, and was being called local by itself (review,
#: 2026-10-12) — which then labelled every other emitter with its rate.
LOCAL_SIGHTINGS = 3
#: At most this many dwells may wait for analysis before the hopper holds.
MAX_PENDING_ANALYSIS = 2
#: The most stops one list may hold.
MAX_STOPS = 500
#: A clipped-sample fraction above this is said: an overloaded front end
#: makes intermodulation products, and they are cyclostationary too.
CLIP_WARN = 1e-4
#: Seconds discarded after every retune (PLL settling, the DC servo, gain
#: transients) — on top of whatever the driver already discarded.
SETTLE_S = 0.15
#: A near miss: the best peak's family p-value under this earns one relook.
NEAR_MISS_P = 0.1
#: How far a CONJUGATE line (a carrier: an absolute frequency) may move
#: between two looks and still be the same line, in parts per million of
#: its frequency. Both oscillators drift — the emitter's and the radio's —
#: and an α bin is a fraction of a hertz, so a tolerance in bins (a few
#: parts per BILLION at RF) failed real confirmations (review, 2026-10-12:
#: a carrier moved 2 Hz at 433 MHz was "not seen again"). A confirmation
#: is a minute or two after the first look: 0.5 ppm (217 Hz at 434 MHz)
#: covers a TCXO at both ends; the cost is counted — the confirmation's
#: threshold is set by the cells inside this window (`focus_trials`).
CONFIRM_DRIFT_PPM = 0.5
#: The same for the run's ledger — the baseline and the receiver's
#: calibration are hours old by the end of a run: 2 ppm (868 Hz at
#: 434 MHz). A new carrier closer than this to a known one is taken for it;
#: that is the price of not re-recording the known one every pass.
LEDGER_DRIFT_PPM = 2.0

#: Colours of each classification, wherever ATK shows a sift result. NEW
#: is the one colour no other ATK detector uses — Bill, 2026-10-08: *"the
#: bright flashy pretty things are all they will see"* — and the classes
#: that are NOT recorded are muted so the eye goes to what was kept.
CLASS_COLOURS = {
    "new": "#ffd21f",          # bright gold — recorded
    "confirmed": "#ffe866",    # brighter gold — recorded, seen twice
    "explained": "#7fb3ff",    # blue — a known class at a distance
    "visible": "#e0e0e0",      # white — the ordinary detector sees it
    "background": "#8a8f98",   # grey — was here in the baseline
    "local": "#b07a4f",        # brown — a nearby broadband source
    "receiver": "#5c6370",     # dim — the radio's own
    "dc": "#5c6370",           # dim — the radio's DC offset
}
CLASS_WORDS = {
    "new": "NEW — not explained by the receiver, the place or the class "
           "table: recorded",
    "confirmed": "NEW and seen again on the confirmation dwell: kept",
    "explained": "its cycle frequency matches a known class (a distant "
                 "signal of a listed kind)",
    "visible": "its channel's energy is above the floor: the ordinary "
               "detector sees this one",
    "background": "it was present at this stop during the baseline pass",
    "local": "the same cycle frequency at three or more stops, further "
             "apart than the bandwidth: a local broadband source, not an "
             "emitter",
    "receiver": "the radio's own (seen with the antenna replaced by a "
                "terminator)",
    "dc": "a conjugate line at α≈0 at the tuned centre: the radio's DC "
          "offset looks exactly like this",
}
#: The order a feature's labels are ranked in: the first that applies is
#: its class. A receiver feature is the radio's whatever else is true.
CLASS_ORDER = ("receiver", "dc", "local", "background", "visible",
               "explained", "new")

LIMITS = (
    "Sifting finds signals whose statistics repeat — a symbol clock, a chip "
    "rate, a carrier a squarer can see, a sweep or pulse rate — and nothing "
    "else. A burst shorter than a few hundred symbols leaves no line. A "
    "frequency hopper is seen only in the fragments that land at a stop "
    "while it is listened to (bandwidth beats it, not dwell). A waveform "
    "built without cyclic features (dithered timing, noise-like spreading) "
    "can hide from it; against that only a carefully calibrated "
    "radiometer helps.")


# ---------------------------------------------------------------------------
# The operator's frequency list
# ---------------------------------------------------------------------------
_UNIT = {"hz": 1.0, "khz": 1e3, "k": 1e3, "mhz": 1e6, "m": 1e6, "ghz": 1e9,
         "g": 1e9}
_NUM = r"(\d+(?:\.\d*)?|\.\d+)"
_TOKEN = re.compile(
    rf"^{_NUM}\s*([a-z]*)\s*(?:(?:-|to|–)\s*{_NUM}\s*([a-z]*)"
    rf"(?:\s*(?:step|/|by)\s*{_NUM}\s*([a-z]*))?)?$")


def parse_stops(text: str, bandwidth_hz: float,
                default_unit: str = "mhz") -> tuple[list[float], list[str]]:
    """`(stops_hz, problems)` from what the operator typed.

    Accepted, comma- or semicolon-separated: a frequency (`433.92`,
    `433.92 MHz`, `868300 kHz`), a range with a step (`430-440 step 1`,
    `430-440/0.5`), or a range alone (`430-440`), which is TILED at the
    bandwidth so it is covered edge to edge. A bare number is in MHz.
    Duplicates are dropped; the order typed is kept. Problems are words,
    one per token that could not be read."""
    bw = float(bandwidth_hz)
    out: list[float] = []
    problems: list[str] = []
    seen = set()
    unit0 = _UNIT.get(default_unit.lower(), 1e6)
    for raw in re.split(r"[,;\n]+", str(text or "")):
        tok = raw.strip().lower()
        if not tok:
            continue
        m = _TOKEN.match(tok)
        if not m:
            problems.append(f"\"{raw.strip()}\" is not a frequency or a "
                            "range (e.g. 433.92, 430-440 step 1)")
            continue
        a, ua, b, ub, s, us = m.groups()

        def hz(v, u, fallback):
            if u and u not in _UNIT:
                raise ValueError(u)
            return float(v) * (_UNIT[u] if u else fallback)
        try:
            unit_b = _UNIT.get(ub, None) if ub else None
            # "430-440 MHz": the unit after the range applies to both ends
            first = hz(a, ua, unit_b or unit0)
            if b is None:
                freqs = [first]
            else:
                last = hz(b, ub, _UNIT.get(ua, unit0) if ua else unit0)
                step = (hz(s, us, unit_b or (_UNIT.get(ua) if ua else unit0))
                        if s is not None else bw)
                if step <= 0:
                    raise ValueError("step")
                lo, hi = sorted((first, last))
                if s is None:
                    # tile: the first stop's band starts at lo
                    lo = lo + 0.5 * bw
                n = int(math.floor((hi - lo) / step + 1e-9)) + 1
                if s is None and lo + (n - 1) * step + 0.5 * bw < hi - 1e-6:
                    n += 1
                if n > MAX_STOPS:
                    problems.append(f"\"{raw.strip()}\" makes {n} stops; the "
                                    f"most one list may hold is {MAX_STOPS}")
                    continue
                freqs = [lo + i * step for i in range(max(1, n))]
        except ValueError as exc:
            problems.append(f"\"{raw.strip()}\": {exc} is not a unit ATK "
                            "knows (Hz, kHz, MHz, GHz)" if str(exc) not in
                            ("step",) else
                            f"\"{raw.strip()}\": the step must be positive")
            continue
        for f in freqs:
            key = round(f, 0)
            if key in seen:
                continue
            seen.add(key)
            out.append(float(key))
    if len(out) > MAX_STOPS:
        problems.append(f"{len(out)} stops; the most one list may hold is "
                        f"{MAX_STOPS}")
        out = out[:MAX_STOPS]
    return out, problems


# ---------------------------------------------------------------------------
# Configuration and plan
# ---------------------------------------------------------------------------
@dataclass
class SiftConfig:
    """What the operator chose. `problems()` says, in words, what is wrong."""
    stops_hz: list
    bandwidth_hz: float = 500_000.0
    dwell_s: float = 30.0
    coherent_s: float = 2.0
    false_per_day: float = 1.0
    confirm: bool = True
    conjugate: bool = True
    min_width_hz: float = 12_500.0
    disk_budget_gb: float = 20.0
    keep_visible: bool = False
    keep_explained: bool = False
    baseline_passes: int = 1
    settle_s: float = SETTLE_S

    def problems(self, sample_rate: float, usable_span_hz: float | None = None,
                 f_min_hz: float | None = None,
                 f_max_hz: float | None = None) -> list[str]:
        out = []
        fs = float(sample_rate)
        span = float(usable_span_hz or 0.75 * fs)
        if not self.stops_hz:
            out.append("there are no frequencies to listen to")
        if self.bandwidth_hz <= 0:
            out.append("the bandwidth must be positive")
        elif self.bandwidth_hz > span + 1e-6:
            out.append(f"the bandwidth ({self.bandwidth_hz / 1e6:g} MHz) is "
                       f"more than this radio sees cleanly at once "
                       f"({span / 1e6:g} MHz at {fs / 1e6:g} MS/s)")
        if self.min_width_hz <= 0 or self.min_width_hz > self.bandwidth_hz:
            out.append("the narrowest channel must be positive and no wider "
                       "than the bandwidth")
        if self.coherent_s < 0.25:
            out.append("a coherent segment shorter than 0.25 s resolves too "
                       "little in α to be worth searching")
        if self.dwell_s < self.coherent_s:
            out.append(f"the dwell ({self.dwell_s:g} s) must hold at least one "
                       f"coherent segment ({self.coherent_s:g} s); the "
                       f"{self.settle_s:g} s settling time after a retune is "
                       "added to it, not taken from it")
        if self.dwell_s > 600:
            out.append("a dwell longer than 10 minutes is not supported")
        if self.false_per_day <= 0:
            out.append("false recordings per day must be positive (say 0.1 "
                       "for one every ten days)")
        if self.disk_budget_gb <= 0:
            out.append("the disk budget must be positive")
        if not 0 <= int(self.baseline_passes) <= 5:
            out.append("baseline passes must be 0–5")
        for f in self.stops_hz:
            if f_min_hz is not None and f < f_min_hz:
                out.append(f"{f / 1e6:g} MHz is below what this radio tunes "
                           f"({f_min_hz / 1e6:g} MHz)")
                break
            if f_max_hz is not None and f > f_max_hz:
                out.append(f"{f / 1e6:g} MHz is above what this radio tunes "
                           f"({f_max_hz / 1e6:g} MHz)")
                break
        return out

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "SiftConfig":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in dict(d).items() if k in known})


@dataclass
class Level:
    """One channel width searched at every stop."""
    width_hz: float
    spacing_hz: float
    decim: int
    fs_c: float
    offsets_hz: list            # channel centres, from the stop's centre
    seg_len: int                # samples per coherent segment, per channel
    nfft: int                   # α transform length (≥ seg_len)
    cells_nc: int               # resolvable non-conjugate α cells / channel
    cells_c: int                # resolvable conjugate α cells / channel
    enbw_hz: float              # noise bandwidth of the channel filter
    stage: int = 1              # always 1 since every level is channelised
    trans_hz: float = 0.0       # the channel filter's transition width

    @property
    def channels(self) -> int:
        return len(self.offsets_hz)

    @property
    def dalpha_hz(self) -> float:
        return self.fs_c / max(1, self.seg_len)


@dataclass
class SiftPlan:
    """Everything a run will do, decided before it starts and said in words
    (`words`) — the bench's idiom: a cost you were told about is a cost, one
    you were not is a fault report."""
    config: SiftConfig
    sample_rate: float
    datatype: str
    levels: list
    segments: int
    trials: float
    pfa_first: float
    pfa_confirm: float
    dwells_per_day: float
    revisit_s: float
    bytes_per_dwell: int
    memory_bytes: int
    analysis_s: float | None = None
    sensitivity: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def to_json(self) -> dict:
        d = asdict(self)
        d["words"] = self.words()
        return d

    def words(self) -> str:
        c = self.config
        n = len(c.stops_hz)
        lines = []
        lines.append(
            f"{n} stop{'s' if n != 1 else ''} × {c.dwell_s:g} s, "
            f"{c.bandwidth_hz / 1e6:g} MHz each: one pass every "
            f"{_dur(self.revisit_s)}"
            + (f" (the analysis, {_dur(self.analysis_s)} a dwell, is slower "
               f"than the dwell, so it sets the pace)"
               if self.analysis_s and self.analysis_s > c.dwell_s else "")
            + ".")
        lines.append(
            f"Each dwell is {self.segments} coherent segment(s) of "
            f"{c.coherent_s:g} s, searched at {len(self.levels)} channel "
            f"width(s) ("
            + ", ".join(_hz(lv.width_hz) for lv in self.levels) + ")"
            + (", conjugate and non-conjugate" if c.conjugate
               else ", non-conjugate only") + ".")
        if self.sensitivity:
            sym = ", ".join(f"{s['symbol_db']:+.0f} dB in {_hz(s['width_hz'])}"
                            for s in self.sensitivity)
            car = [s for s in self.sensitivity if s.get("carrier_db") is not None]
            lines.append(
                "Sensitivity (in-band SNR, a signal filling its channel, "
                "found 9 times in 10): a QPSK-like symbol-rate line at "
                + sym
                + (f"; a BPSK-like carrier line about "
                   f"{car[-1]['symbol_db'] - car[-1]['carrier_db']:.0f} dB "
                   f"deeper" if car else "")
                + ". Wider channels integrate more samples, so a WIDE weak "
                  "signal is found deeper than a narrow one. Measured "
                  "references scaled by the algebra — not a promise for "
                  "every waveform.")
        if c.confirm:
            lines.append(
                f"About {c.false_per_day:g} false recording(s) a day from "
                f"chance: a first look at Pfa {self.pfa_first:.1e} (≈"
                f"{self.dwells_per_day * self.pfa_first:.1f} a day, each "
                f"costing one confirmation dwell) and a confirmation at "
                f"Pfa {self.pfa_confirm:.1e}, looking again only where the "
                f"first look found something. Environmental features are "
                f"handled by the receiver calibration, the baseline and the "
                f"local rule, not by this number.")
        else:
            lines.append(
                f"About {c.false_per_day:g} false recording(s) a day from "
                f"chance (Pfa {self.pfa_first:.1e} a dwell, no confirmation).")
        gb_dwell = self.bytes_per_dwell / 1e9
        lines.append(
            f"Every dwell is written to disk ({gb_dwell:.2f} GB) and deleted "
            f"unless something new is in it; kept recordings — every sift "
            f"with this radio, together — stop at {c.disk_budget_gb:g} GB. "
            f"Analysis needs up to {self.memory_bytes / 1e6:,.0f} MB of "
            f"memory ({level_threads(len(self.levels))} channel width(s) "
            f"searched at once on this machine).")
        if c.baseline_passes:
            lines.append(
                f"The first {c.baseline_passes} pass(es) learn the place: "
                "anything found then is background, listed but never "
                "recorded — an emitter already on when sifting starts will "
                "not be called new.")
        lines.extend(self.notes)
        return "\n".join(lines)


def _hz(f: float) -> str:
    f = float(f)
    if f >= 1e6:
        return f"{f / 1e6:g} MHz"
    if f >= 1e3:
        return f"{f / 1e3:g} kHz"
    return f"{f:g} Hz"


def _dur(s) -> str:
    if s is None:
        return "?"
    s = float(s)
    if s < 120:
        return f"{s:.0f} s"
    if s < 7200:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} h"


def false_alarm_split(false_per_day: float, dwells_per_day: float,
                      confirm: bool) -> tuple[float, float]:
    """`(pfa_first, pfa_confirm)` for an accepted number of false recordings
    a day.

    A false RECORDING needs a false first look AND (with confirmation) a
    false confirmation, so F = D · p₁ · p₂. The split is even, p₁ = p₂ =
    √(F/D), capped so a first look is never more than 2 % likely to be
    false (each one costs a confirmation dwell: at 2 % and 30 s dwells that
    is ~3 % of the run's time) and the confirmation never looser than 0.5.
    Without confirmation p₁ = F/D."""
    F = max(1e-9, float(false_per_day))
    D = max(1.0, float(dwells_per_day))
    if not confirm:
        return float(min(0.5, max(1e-12, F / D))), 1.0
    p1 = min(0.02, max(1e-12, math.sqrt(F / D)))
    p2 = min(0.5, max(1e-12, F / (D * p1)))
    return float(p1), float(p2)


def _channel_filter(fs: float, spacing_hz: float, width_hz: float,
                    taps_per_phase: int = 8) -> np.ndarray:
    """The prototype filter `proposer.channelise` builds — the same call,
    so the noise bandwidth computed from it is the one the channels have."""
    from scipy.signal import firwin
    M = max(2, int(round(fs / float(spacing_hz))))
    L = int(taps_per_phase) * M
    trans = max(0.15 * width_hz, 3.5 * fs / L)
    return firwin(L, 0.5 * width_hz + 0.5 * trans, fs=fs,
                  window=("kaiser", 8.0)).astype(np.float64)


#: A channel is sampled at this multiple of its width. The channelizer's
#: transition band is ~0.22 of the width each side (`_transition`), and a
#: conjugate line from content in it lands at up to width + 2·transition;
#: at 2.5× the width its alias (that, minus the channel rate) stays out of
#: |α| ≤ width — measured otherwise: a BPSK at +93.75 kHz showed its
#: carrier line again in the 150 kHz channel at 0 Hz, aliased to −145.8 kHz.
#: The cyclic proposer probes at the same multiple for its own reason.
RATE_FACTOR = 2.5


def _transition(fs_in: float, spacing_hz: float, width_hz: float) -> float:
    """The channel filter's transition width (as `proposer.channelise`)."""
    M = max(2, int(round(fs_in / float(spacing_hz))))
    return float(max(0.15 * width_hz, 3.5 * fs_in / (8 * M)))


def alpha_limit_hz(width_hz: float, fs_c: float, conj: bool = False,
                   trans_hz: float = 0.0) -> float:
    """The largest |α| a channel can hold a REAL feature at.

    A non-conjugate feature needs both f + α/2 and f − α/2 inside the
    passband, so |α| ≤ width (and ≤ fs_c/2). A conjugate line sits at twice
    a carrier inside the passband, so |α| ≤ width too — and only where no
    alias can land: content in the transition band makes lines up to
    width + 2·transition, which alias to that minus fs_c, so the clean
    region is |α| ≤ fs_c − width − 2·transition. Zero means this channel
    cannot be searched conjugately at all (said in the plan)."""
    if not conj:
        return float(min(width_hz, 0.5 * fs_c))
    return float(max(0.0, min(width_hz, fs_c - width_hz - 2.0 * trans_hz)))


def first_resampling(fs: float, bandwidth_hz: float) -> tuple[int, int]:
    """`(up, down)` taking the radio's rate to the stop's: the smallest
    rational rate fs·up/down (down ≤ 64) that keeps `RATE_FACTOR` × the
    bandwidth. Every level is channelised from THAT, so the search's cost
    follows the bandwidth, not the radio — a HackRF at 20 MS/s searching
    1 MHz works at 2.5 MS/s, and so does a bladeRF at 4 MS/s (measured
    2026-10-12: integer decimation left the bladeRF at 4 MS/s, 1.6× the
    work, for nothing). (1, 1) when the radio's rate is already at or
    below it."""
    fs = float(fs)
    need = RATE_FACTOR * float(bandwidth_hz)
    if need >= fs:
        return 1, 1
    best = (1, 1)
    best_rate = fs
    for down in range(2, 65):
        up = int(math.ceil(need * down / fs - 1e-9))
        if up >= down:
            continue
        rate = fs * up / down
        if rate < best_rate - 1e-6:
            g = math.gcd(up, down)
            best, best_rate = (up // g, down // g), rate
    return best


def _stage0_filter(down: int) -> np.ndarray:
    """The anti-alias filter `scipy.signal.resample_poly` uses for this
    ratio (its documented default: Kaiser β 5, 10 zero crossings a side)."""
    from scipy.signal import firwin
    return firwin(2 * 10 * down + 1, 1.0 / down, window=("kaiser", 5.0))


def stage0(x: np.ndarray, ratio) -> np.ndarray:
    """The stop's neighbourhood at fs·up/down: the radio's samples resampled
    by scipy's polyphase resampler (C, not a Python loop). The channel
    filters that define every level run on THIS."""
    up, down = (ratio, 1) if isinstance(ratio, int) else ratio
    if up == down:
        return np.asarray(x, dtype=np.complex64)
    from scipy.signal import resample_poly
    return resample_poly(np.asarray(x, dtype=np.complex64), int(up),
                         int(down)).astype(np.complex64, copy=False)


def first_channel(y0: np.ndarray, fs0: float, lv: "Level") -> np.ndarray:
    """The widest level — ONE channel, centred — as a plain FIR with the
    same prototype `channelise` would build, by scipy's FFT convolution:
    the filterbank spent an M-point FFT per output sample to keep one bin
    (measured: a third of a segment's work). Same alignment as
    `channelise` ('valid', a window starting at each output index)."""
    from scipy.signal import oaconvolve
    h = _channel_filter(fs0, lv.spacing_hz, lv.width_hz).astype(np.float32)
    y = oaconvolve(y0, h, mode="valid")
    return y[:: max(1, int(lv.decim))].astype(np.complex64, copy=False)


def make_levels(fs: float, bandwidth_hz: float, min_width_hz: float,
                coherent_s: float) -> list[Level]:
    """The channel widths searched at each stop, widest first. Every level
    is a `channelise` of the stop's neighbourhood at fs/d0 (`stage0`) —
    the widest a single channel the stop's own width, so it has a real
    filter at the stop's edges, not just the resampler's."""
    from scipy.fft import next_fast_len
    fs = float(fs)
    bw = float(bandwidth_hz)
    up, down = first_resampling(fs, bw)
    fs0 = fs * up / down
    n_in = int(round(coherent_s * fs))
    n0 = -(-n_in * up // down)                # resample_poly's output length
    usable = 0.5 * bw / fs0
    out = []
    w = bw
    while w >= float(min_width_hz) * (1 - 1e-9) or not out:
        spacing = w / OVERLAP
        decim = max(1, int(fs0 // (RATE_FACTOR * w)))
        fs_c = fs0 / decim
        M = max(2, int(round(fs0 / spacing)))
        ks = np.arange(M)
        centres = np.where(ks < M / 2, ks, ks - M) * fs0 / M
        keep = np.abs(centres) + 0.5 * w <= usable * fs0 * (1 + 1e-9)
        offs = sorted(float(c) for c in centres[keep]) or [0.0]
        # exactly what `channelise` returns for one segment: its sliding
        # window of taps_per_phase·M samples, every decim-th
        L = max(256, (n0 - 8 * M) // decim + 1)
        nfft = int(next_fast_len(L))
        h = _channel_filter(fs0, spacing, w)
        enbw = fs0 * float(np.sum(h ** 2)) / float(np.sum(h)) ** 2
        trans = _transition(fs0, spacing, w)
        lim_nc = alpha_limit_hz(w, fs_c)
        lim_c = alpha_limit_hz(w, fs_c, conj=True, trans_hz=trans)
        lo = ALPHA_MIN_BINS * fs_c / L
        cells_nc = max(8, int(L * max(0.0, lim_nc - lo) / fs_c))
        cells_c = int(L * min(1.0, 2.0 * lim_c / fs_c)) if lim_c > 0 else 0
        out.append(Level(width_hz=w, spacing_hz=spacing, decim=decim,
                         fs_c=fs_c, offsets_hz=offs, seg_len=L, nfft=nfft,
                         cells_nc=cells_nc, cells_c=cells_c, enbw_hz=enbw,
                         stage=1, trans_hz=trans))
        w = w / LEVEL_FACTOR
    return out


def level_threads(n_levels: int, threads: int | None = None) -> int:
    """How many channel widths `search` works on at once: half the cores
    (the FFTs inside each use the rest), never more than there are levels.
    `make_plan` uses the same number, so the memory it states is the memory
    the search can take on THIS machine."""
    return max(1, min(int(n_levels), int(threads or 0)
                      or max(1, (os.cpu_count() or 2) // 2)))


def _level_memory(lv: "Level", fs: float, coherent_s: float,
                  conjugate: bool) -> int:
    """Peak bytes of `_search_level` for one width: the accumulators and one
    segment's statistics (float32, held together while they are summed),
    the segment in flight (the raw read, the first resampling, the channels,
    one lag's product and its FFT), and the end of the dwell (the Gamma
    ratio, its local median in float64, the max over lags and its argmax).
    Checked against tracemalloc (tests/test_cyclo_sift.py)."""
    C = lv.channels
    m_nc = lv.nfft // 2 + 1
    cells = m_nc * len(LAGS_NC) + (lv.nfft * len(LAGS_C)
                                   if conjugate and lv.cells_c > 0 else 0)
    acc = C * cells * 4
    seg = int(coherent_s * fs)
    in_flight = (seg * 8                       # the raw read, complex64
                 + seg * 16                    # the first resampling
                 + C * lv.seg_len * 16         # the channels
                 + C * lv.nfft * 8 * 2)        # one lag's product and FFT
    end = (C * max(m_nc, lv.nfft) * (8 + 8 + 4)  # median (f64), argmax, max
           + acc)                              # the Gamma ratio
    return int(2 * acc + max(in_flight, end))


def analysis_memory(levels: list, fs: float, coherent_s: float,
                    conjugate: bool, threads: int | None = None) -> int:
    """Peak bytes of one dwell's search, at most: the levels that run AT
    ONCE (`level_threads`) can all hold their memory together. The plan once
    took the largest single level while the search ran every level at once
    — 2.3× under at 2.4 MS/s and ~5× at 40 MS/s (review, 2026-10-12). Now an
    upper bound: measured 1.1× over for one level, 1.8× for three (their
    peaks rarely coincide, but they can)."""
    per = sorted((_level_memory(lv, fs, coherent_s, conjugate)
                  for lv in levels), reverse=True)
    return int(sum(per[:level_threads(len(per), threads)]))


def make_plan(config: SiftConfig, sample_rate: float, datatype: str,
              retune_s: float = 0.15, benchmark: bool = False) -> SiftPlan:
    """The plan for `config` on a stream at `sample_rate` / `datatype`.
    `benchmark=True` times one batched FFT on this machine (~0.1 s) so the
    analysis time is a measurement, not a guess."""
    fs = float(sample_rate)
    c = config
    levels = make_levels(fs, c.bandwidth_hz, c.min_width_hz, c.coherent_s)
    # the recorder skips the settling time BEFORE it writes, so a dwell file
    # holds exactly dwell_s: every whole segment of it is searched (taking
    # the settling time off again wasted a segment a dwell — review,
    # 2026-10-12: 14 of 15 at the defaults)
    K = max(1, int(c.dwell_s / c.coherent_s + 1e-9))
    trials = 0.0
    for lv in levels:
        trials += lv.channels * lv.cells_nc * len(LAGS_NC)
        if c.conjugate and lv.cells_c > 0:
            trials += lv.channels * lv.cells_c * len(LAGS_C)
    mem = analysis_memory(levels, fs, c.coherent_s, c.conjugate)
    n_stops = max(1, len(c.stops_hz))
    cycle = c.dwell_s + retune_s
    bps = _iq.bytes_per_sample(datatype)
    plan = SiftPlan(config=c, sample_rate=fs, datatype=str(datatype),
                    levels=levels, segments=K, trials=float(trials),
                    pfa_first=0.0, pfa_confirm=0.0,
                    dwells_per_day=86400.0 / cycle,
                    revisit_s=n_stops * cycle,
                    bytes_per_dwell=int(c.dwell_s * fs * bps),
                    memory_bytes=int(mem))
    plan.pfa_first, plan.pfa_confirm = false_alarm_split(
        c.false_per_day, plan.dwells_per_day, c.confirm)
    if benchmark:
        plan.analysis_s = estimate_analysis_s(plan)
        if plan.analysis_s > c.dwell_s:
            plan.revisit_s = n_stops * (plan.analysis_s + retune_s)
    plan.sensitivity = [
        {"width_hz": lv.width_hz,
         "symbol_db": predicted_snr_db(lv, K, plan.trials, plan.pfa_first,
                                       kappa=KAPPA_SYMBOL),
         "carrier_db": (predicted_snr_db(lv, K, plan.trials, plan.pfa_first,
                                         kappa=KAPPA_CARRIER)
                        if (c.conjugate and lv.cells_c > 0) else None)}
        for lv in levels]
    if c.conjugate and any(lv.cells_c == 0 for lv in levels):
        plan.notes.append(
            "The bandwidth is close to the radio's sample rate, so the widest "
            "channel cannot be searched conjugately without aliases (it needs "
            f"{RATE_FACTOR:g}× its width); it is searched non-conjugate only.")
    if c.coherent_s > 4:
        plan.notes.append(
            f"A {c.coherent_s:g} s coherent segment asks the radio's clock and "
            "LO to hold one α bin for that long; a cheap oscillator drifts "
            "out of it, and the feature smears instead of growing.")
    return plan


# ---------------------------------------------------------------------------
# Sensitivity — the measured reference, scaled by the algebra
# ---------------------------------------------------------------------------
#: κ — a feature's strength in this engine's normalised units: the mean
#: statistic of one segment at the true α is 1 + κ·L·η, where L is the
#: segment length in channel samples and η = s²/(1+s)² for in-band SNR s.
#: MEASURED 2026-10-12 (tests/test_cyclo_sift.py holds the model to the
#: engine), each on a signal FILLING a 25 kHz channel (fs 1 MS/s, 1 s
#: segments, 4 seeds), at −5 dB in-band, where the low-SNR form applies:
#:   KAPPA_SYMBOL   QPSK, RRC 0.35 — its non-conjugate symbol-rate line,
#:                  the weakest line a linear modulation has (it lives only
#:                  in the 35 % excess bandwidth): 6.3e-3 (8.2e-3 at 0 dB)
#:   KAPPA_CARRIER  BPSK, RRC 0.35 — its conjugate carrier line: 0.098
#:                  (0.126 at 0 dB), ~6 dB easier to find than the above
#: (Re-measured with the five and two lags of LAGS_NC / LAGS_C: the QPSK
#: line is unchanged, the carrier line 0.113 → 0.098.)
#: A signal filling only part of its channel scores less (measured: a third
#: of a 100 kHz channel, about half the matched κ) — the levels exist so
#: that some channel is within a factor of 4 of every signal.
KAPPA_SYMBOL = 6.3e-3
KAPPA_CARRIER = 0.098


def predicted_snr_db(level: Level, segments: int, trials: float, pfa: float,
                     pd: float = 0.9, kappa: float = KAPPA_SYMBOL) -> float:
    """In-band SNR (dB) at which the reference waveform, filling a channel
    of `level`, is detected with probability ≈ `pd`.

    Summed over K segments the null is Gamma(K, 1) — mean K, variance K —
    and the signal adds K·κ·L·η. Detection at `pd` needs that mean to
    clear the threshold t by z_pd standard deviations:
        K·κ·L·η = t − K + z_pd·√K
    and η = s²/(1+s)² is solved for s. Per doubling of L this falls by
    ~1.5 dB; per doubling of K, by ~0.75 dB (t − K grows like √K)."""
    from scipy.special import gammainccinv
    from scipy.stats import norm
    K = max(1, int(segments))
    t = float(gammainccinv(K, min(0.5, max(1e-300, pfa / max(1.0, trials)))))
    need = max(1e-12, t - K + float(norm.ppf(pd)) * math.sqrt(K))
    eta = need / (K * kappa * level.seg_len)
    if eta >= 0.25:                  # s/(1+s) ≥ ½ → s ≥ 1 (0 dB) and up
        r = math.sqrt(min(eta, 0.999))
        s = r / (1 - r)
    else:
        r = math.sqrt(eta)
        s = r / (1 - r)
    return float(10 * math.log10(max(s, 1e-12)))


def estimate_analysis_s(plan: SiftPlan) -> float:
    """Seconds to analyse one dwell, MEASURED on this machine: noise through
    the real search with one coherent segment and with two. The difference
    is the per-segment work (the channelizer, the lag products, their
    FFTs); the rest is paid once a dwell (the local medians, the peak
    picking). A first run warms the window cache and the FFT plans, which a
    running sift pays once. Costs about four segments of analysis — run it
    off the GUI thread."""
    import copy
    c = copy.copy(plan.config)

    def run(k: int) -> float:
        one = copy.copy(plan)
        one.config = c
        one.segments = k
        n = int(round((k * c.coherent_s + c.settle_s) * plan.sample_rate)) + 16
        rng = np.random.default_rng(k)
        x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)
             ).astype(np.complex64)
        t = time.perf_counter()
        search(x, plan.sample_rate, 0.0, one, 1e-6,
               floor_per_hz=2.0 / plan.sample_rate, floor_spread_db=0.0)
        return time.perf_counter() - t
    run(1)                                      # warm
    t1 = run(1)
    t2 = run(2)
    # a segment cannot cost more than the whole one-segment run; when the
    # difference says otherwise it is the machine's noise, not the search
    per_seg = min(max(0.0, t2 - t1), t1)
    fixed = max(0.0, t1 - per_seg)
    return float(fixed + per_seg * plan.segments)


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------
@dataclass
class Feature:
    """One cycle frequency found at one place. Frequencies are ABSOLUTE."""
    conj: bool
    alpha_hz: float
    f_lo_hz: float
    f_hi_hz: float
    width_hz: float
    stop_hz: float
    statistic: float
    threshold: float
    p_family: float
    lag: int
    segments: int
    dalpha_hz: float
    energy_snr_db: float | None = None
    energy_visible: bool = False
    harmonics_hz: list = field(default_factory=list)
    sidebands_hz: list = field(default_factory=list)
    widths_seen_hz: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    klass: str = "new"
    explained_by: list = field(default_factory=list)
    confirmed: bool = False
    id: str = ""
    #: the line's ABSOLUTE position: the carrier (Hz) for a conjugate line
    #: — channel centre + α/2, the same wherever it was seen — and α for a
    #: non-conjugate one. Conjugate lines are matched on this, never on α,
    #: which depends on the channel they were seen in.
    line_hz: float = 0.0

    @property
    def margin(self) -> float:
        return self.statistic / max(1e-12, self.threshold)

    @property
    def centre_hz(self) -> float:
        return 0.5 * (self.f_lo_hz + self.f_hi_hz)

    def words(self) -> str:
        kind = "conjugate" if self.conj else "cycle"
        what = (f"conjugate line: a carrier at {self.line_hz / 1e6:.6f} MHz"
                if self.conj else
                f"{self.alpha_hz:,.2f} Hz {kind} frequency")
        return (f"{what} in {self.f_lo_hz / 1e6:.4f}–{self.f_hi_hz / 1e6:.4f} "
                f"MHz · statistic {self.statistic:,.0f} vs {self.threshold:,.0f}"
                f" · p {self.p_family:.1e} · {self.klass}")

    def to_json(self) -> dict:
        d = asdict(self)
        d["words"] = self.words()
        d["colour"] = CLASS_COLOURS.get("confirmed" if self.confirmed
                                        else self.klass, "#ffffff")
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Feature":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in dict(d).items() if k in known})


@dataclass
class DwellResult:
    stop_hz: float
    kind: str                       # dwell | confirm | relook | calibrate
    features: list
    near_miss: dict | None
    floor_db_per_hz: float | None
    clipped_fraction: float
    seconds_analysed: float
    analysis_s: float
    segments: int
    warnings: list = field(default_factory=list)
    dwell_id: str = ""

    def to_json(self) -> dict:
        d = asdict(self)
        d["features"] = [f.to_json() if isinstance(f, Feature) else f
                         for f in self.features]
        return d


@functools.lru_cache(maxsize=64)
def _window(n: int) -> np.ndarray:
    """Blackman-Harris (as the lag profile uses: the α = 0 power term would
    otherwise leak sidelobes that read as features). Cached — one length per
    level and lag, computed once a run (it was 15 % of the search)."""
    a = (0.35875, 0.48829, 0.14128, 0.01168)
    k = np.arange(int(n))
    w = np.zeros(int(n))
    for i, ai in enumerate(a):
        w += ((-1) ** i) * ai * np.cos(2 * np.pi * i * k / max(1, n - 1))
    w = w.astype(np.float32)
    w.setflags(write=False)
    return w


def _segment_stats(Y: np.ndarray, lags, conj: bool, nfft: int,
                   workers: int = -1) -> np.ndarray:
    """[lags, C, bins] — N·|R(α,τ)|²/P², the lag-domain statistic, for ONE
    coherent segment of every channel at once. Non-conjugate keeps α ≥ 0
    (bins 0..nfft/2); conjugate keeps the whole axis. Under the null each
    value is exponential with a lag-dependent mean; the local median removes
    that scale (see `_gamma_ratio`)."""
    import scipy.fft as sfft
    C, L = Y.shape
    P = np.mean(np.abs(Y) ** 2, axis=1).astype(np.float64)
    P = np.where(P > 0, P, 1.0)
    bins = nfft // 2 + 1 if not conj else nfft
    out = np.empty((len(lags), C, bins), dtype=np.float32)
    for i, t in enumerate(lags):
        t = int(t)
        a = Y[:, t:]
        b = Y[:, : L - t]
        prod = a * (b if conj else np.conj(b))
        if not conj:
            # the α = 0 term of a NON-conjugate product is the ordinary
            # autocorrelation (large for band-limited noise at every lag,
            # not only lag 0) — not a cyclic feature, and its window leakage
            # is what the first α bins would otherwise report
            prod -= prod.mean(axis=1, keepdims=True)
        w = _window(L - t)
        prod *= w[None, :]
        F = sfft.fft(prod, n=nfft, axis=1, workers=workers)
        if not conj:
            F = F[:, :bins]
        np.multiply(F.real, F.real, out=out[i])
        out[i] += F.imag * F.imag
        out[i] /= (float(np.sum(w.astype(np.float64) ** 2))
                   * (P ** 2))[:, None].astype(np.float32)
    return out


def _local_median(A: np.ndarray) -> np.ndarray:
    """Per row of [C, n], each bin's neighbourhood median (±n/32) — CFAR in
    α, as `scf.local_ratio` — as the median of BLOCK medians: one vectorised
    pass over the data (blocks of n/128), then a median over the 9 blocks
    around each, interpolated. The same floor as a sliding median to within
    the sample median's own scatter, for a twentieth of the work (the
    sliding version was a quarter of the search)."""
    C, n = A.shape
    nb = int(min(128, max(8, n // 64)))
    blk = n // nb
    if blk < 8:
        floor = np.median(A, axis=1, keepdims=True)
        return np.maximum(np.broadcast_to(floor, A.shape).astype(np.float64),
                          1e-30)
    bm = np.median(A[:, : nb * blk].reshape(C, nb, blk), axis=2)
    half = 4                                     # ±4 blocks ≈ ±n/32
    pad = np.pad(bm, ((0, 0), (half, half)), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1, axis=1)
    floors = np.median(win, axis=2)              # [C, nb]
    centres = (np.arange(nb) + 0.5) * blk
    xs = np.arange(n)
    out = np.empty((C, n), dtype=np.float64)
    for r in range(C):
        out[r] = np.interp(xs, centres, floors[r])
    return np.maximum(out, 1e-30)


def _gamma_ratio(acc: np.ndarray, K: int) -> np.ndarray:
    """[lags, C, bins] summed statistics → Gamma(K, 1) units: each lag's
    row over its local median, times the median of Gamma(K, 1)."""
    from scipy.special import gammaincinv
    gmed = float(gammaincinv(K, 0.5))
    out = np.empty_like(acc, dtype=np.float32)
    for i in range(acc.shape[0]):
        out[i] = (acc[i] / _local_median(acc[i]) * gmed).astype(np.float32)
    return out


def _peaks(row: np.ndarray, thr: float, valid: np.ndarray,
           guard: int) -> list[int]:
    """Indices of local maxima above `thr` (one per contiguous run), at
    least `guard` bins apart."""
    over = (row > thr) & valid
    if not over.any():
        return []
    idx = np.flatnonzero(over)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > guard) + 1)
    return [int(r[np.argmax(row[r])]) for r in runs if r.size]


def _refine(row: np.ndarray, k: int) -> float:
    """Sub-bin offset of a peak by a parabola through three bins."""
    n = row.size
    if k <= 0 or k >= n - 1:
        return 0.0
    a, b, c = float(row[k - 1]), float(row[k]), float(row[k + 1])
    den = a - 2 * b + c
    return float(0.5 * (a - c) / den) if den < 0 else 0.0


class _Cancelled(Exception):
    pass


def search(x_or_reader, fs: float, center_hz: float, plan: SiftPlan,
           pfa: float, *, floor_per_hz: float | None = None,
           floor_spread_db: float | None = None, progress=None, cancel=None,
           log: logging.Logger | None = None, workers: int = -1,
           threads: int | None = None, skip_s: float | None = None,
           focus: list | None = None, pfa_focus: float | None = None) -> dict:
    """The blind search over one dwell. Returns {features, near_miss,
    floor, segments, seconds}.

    `x_or_reader` is complex samples at `fs` (a test) or a callable
    `reader(start, count) -> complex64` (a dwell file: `DwellReader`), read
    one coherent segment at a time so memory is the accumulators, never
    the dwell. Levels are the OUTER loop, run `level_threads` at a time:
    each re-reads the file (sequential, from the OS cache), so a level holds
    only its own statistics (`analysis_memory` counts the ones at once).

    `focus` (a confirmation dwell): the held lines' windows
    (`focus_windows`). Inside them the threshold is set by `pfa_focus` and
    the cells inside the windows alone (`focus_trials`) — the question
    there is "is THAT line still here?", not "is anything anywhere?" —
    and everywhere else by `pfa` and the whole search, as on a first
    look."""
    from scipy.special import gammainccinv
    log = log or _LOG
    say = progress or (lambda _m: None)
    stop = cancel or (lambda: False)
    c = plan.config
    fs = float(fs)
    if callable(x_or_reader):
        reader = x_or_reader
        total = int(getattr(x_or_reader, "samples", 0))
    else:
        xa = np.asarray(x_or_reader).astype(np.complex64, copy=False)
        total = xa.size

        def reader(start, count, _x=xa):
            return _x[start:start + count]
    # the settling time after a retune is skipped HERE for raw samples; a
    # dwell file from the recorder has already had it skipped (skip_s=0)
    skip = c.settle_s if skip_s is None else float(skip_s)
    settle = int(round(skip * fs)) if total > int(skip * fs) * 4 else 0
    seg_full = int(round(c.coherent_s * fs))
    K = max(1, min(plan.segments, (total - settle) // max(1, seg_full)))
    if K < 1 or total - settle < seg_full:
        raise ValueError(f"the dwell holds {total / fs:.2f} s; one coherent "
                         f"segment needs {c.coherent_s:g} s")
    # the floor, from the first segment of the stop (full rate)
    from atk_diffusion.dsp import measure as _m
    first = np.asarray(reader(settle, seg_full))
    if floor_per_hz is None:
        nf = _m.noise_floor(first[: min(first.size, 1 << 20)], fs, nfft=1024)
        floor_per_hz = nf["floor_per_hz"]
        floor_spread_db = nf["spread_db"]
    spread = float(floor_spread_db or 0.0)
    d0 = first_resampling(fs, c.bandwidth_hz)
    fs0 = fs * d0[0] / d0[1]
    trials = max(2.0, plan.trials)
    p_cell = min(0.5, max(1e-300, float(pfa) / trials))
    thr = float(gammainccinv(K, p_cell))
    log.info("sift search: %.6f MHz, %d segment(s) of %g s, %d level(s), "
             "%.3g trials, pfa %.2g -> threshold %.2f (Gamma(%d))",
             center_hz / 1e6, K, c.coherent_s, len(plan.levels), trials, pfa,
             thr, K)
    focus = list(focus or [])
    thr_focus, trials_focus = None, 0.0
    if focus and pfa_focus is not None:
        trials_focus = max(1.0, focus_trials(plan, focus, center_hz))
        thr_focus = min(thr, float(gammainccinv(
            K, min(0.5, max(1e-300, float(pfa_focus) / trials_focus)))))
        log.info("sift search: %d held line(s) looked for again: %.3g cells "
                 "in their windows, pfa %.2g -> threshold %.2f there",
                 len(focus), trials_focus, pfa_focus, thr_focus)
    t0 = time.perf_counter()
    n_threads = level_threads(len(plan.levels), threads)
    fft_workers = max(1, (os.cpu_count() or 2) // n_threads)
    ctx = {"reader": reader, "settle": settle, "seg_full": seg_full, "K": K,
           "d0": d0, "fs0": fs0, "thr": thr, "trials": trials,
           "floor_per_hz": floor_per_hz, "spread": spread,
           "center_hz": float(center_hz), "c": c, "stop": stop, "say": say,
           "workers": fft_workers if workers == -1 else workers, "fs": fs,
           "focus": focus if thr_focus is not None else [],
           "thr_focus": thr_focus, "trials_focus": trials_focus}
    results = []
    if n_threads == 1:
        for lv in plan.levels:
            results.append(_search_level(lv, ctx))
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_threads,
                                thread_name_prefix="sift-level") as ex:
            results = list(ex.map(lambda lv: _search_level(lv, ctx),
                                  plan.levels))
    raw: list[Feature] = []
    best_miss = None
    for feats_l, miss_l in results:
        raw.extend(feats_l)
        if miss_l is not None and (best_miss is None
                                   or miss_l["p_family"] < best_miss["p_family"]):
            best_miss = miss_l
    feats = _merge(raw)
    out = {"features": feats, "near_miss": best_miss if not feats else None,
           "floor_per_hz": floor_per_hz,
           "floor_db_per_hz": (10 * math.log10(floor_per_hz)
                               if floor_per_hz and floor_per_hz > 0 else None),
           "segments": K, "seconds": K * seg_full / fs, "threshold": thr,
           "threshold_focus": thr_focus, "trials_focus": trials_focus,
           "analysis_s": time.perf_counter() - t0}
    log.info("sift search done: %d feature(s) in %.1f s (%d level thread(s))",
             len(feats), out["analysis_s"], n_threads)
    return out



def _search_level(lv: Level, ctx: dict) -> tuple[list, dict | None]:
    """One channel width over every coherent segment of the dwell:
    (raw features, the best near miss). Runs on a worker thread; reads its
    own segments, so levels share nothing but the read-only plan."""
    from scipy.special import gammaincc
    from atk_diffusion.cyclo.proposer import channelise
    reader, settle, seg_full, K = (ctx["reader"], ctx["settle"],
                                   ctx["seg_full"], ctx["K"])
    d0, fs0, thr, trials = ctx["d0"], ctx["fs0"], ctx["thr"], ctx["trials"]
    c, stop, say, workers = ctx["c"], ctx["stop"], ctx["say"], ctx["workers"]
    center_hz, fs = ctx["center_hz"], ctx["fs"]
    floor_per_hz, spread = ctx["floor_per_hz"], ctx["spread"]
    focus, thr_f, trials_f = (ctx.get("focus") or [], ctx.get("thr_focus"),
                              ctx.get("trials_focus") or 0.0)
    C = lv.channels
    acc_nc = None
    acc_c = None
    powers = np.zeros(C)
    raw: list[Feature] = []
    best_miss = None
    for k in range(K):
        if stop():
            raise _Cancelled()
        say(f"width {_hz(lv.width_hz)}: segment {k + 1}/{K}")
        x = np.asarray(reader(settle + k * seg_full, seg_full))
        y0 = stage0(x, d0)
        del x
        if lv.channels == 1 and lv.offsets_hz == [0.0]:
            Y = first_channel(y0, fs0, lv)[None, :]
        else:
            taps = 8 * max(2, int(round(fs0 / lv.spacing_hz)))
            Y, _centres, _fs_c, _info = channelise(
                y0, fs0, lv.spacing_hz, lv.width_hz, lv.decim,
                usable=min(0.5, 0.5 * c.bandwidth_hz / fs0 + 1e-9),
                chunk=max(16, min(2048, (1 << 24) // taps)))
        del y0
        if Y.shape[0] != C:
            # the grid is the plan's; a mismatch is a bug, said loudly
            raise RuntimeError(f"channelizer gave {Y.shape[0]} channels, "
                               f"the plan has {C}")
        Y = Y[:, : lv.seg_len]
        if Y.shape[1] < lv.seg_len:
            Y = np.pad(Y, ((0, 0), (0, lv.seg_len - Y.shape[1])))
        powers += np.mean(np.abs(Y) ** 2, axis=1)
        s = _segment_stats(Y, LAGS_NC, False, lv.nfft, workers)
        acc_nc = s if acc_nc is None else acc_nc + s
        if c.conjugate and lv.cells_c > 0:
            s = _segment_stats(Y, LAGS_C, True, lv.nfft, workers)
            acc_c = s if acc_c is None else acc_c + s
        del Y, s
    powers /= K
    noise_p = floor_per_hz * lv.enbw_hz
    for conj, acc, lags in ((False, acc_nc, LAGS_NC), (True, acc_c, LAGS_C)):
        if acc is None:
            continue
        G = _gamma_ratio(acc, K)                     # [lags, C, bins]
        T = G.max(axis=0)
        Targ = G.argmax(axis=0)
        del G
        alpha_axis, valid = _alpha_grid(lv, conj)
        if alpha_axis.size != T.shape[1]:
            raise RuntimeError(f"α grid of {alpha_axis.size} bins for "
                               f"statistics of {T.shape[1]}")
        guard = max(4, int(4 * lv.nfft / lv.seg_len))
        for ch in range(C):
            row = T[ch]
            fm = (_focus_mask(lv, center_hz + lv.offsets_hz[ch], conj,
                              alpha_axis, focus) if focus else None)
            if fm is not None:
                fm &= valid
                thr_row = np.where(fm, thr_f, thr)
            else:
                thr_row = thr
            rv = np.where(valid, row, 0.0)
            kbest = int(np.argmax(rv))
            pb = min(1.0, trials * float(gammaincc(K, float(rv[kbest]))))
            if pb < NEAR_MISS_P and rv[kbest] <= thr and (
                    best_miss is None or pb < best_miss["p_family"]):
                best_miss = {"conj": conj, "width_hz": lv.width_hz,
                             "offset_hz": lv.offsets_hz[ch],
                             "alpha_hz": float(alpha_axis[kbest]),
                             "statistic": float(rv[kbest]),
                             "threshold": thr, "p_family": pb}
            pks = _peaks(row, thr_row, valid, guard)
            if not pks:
                continue
            excess = powers[ch] / max(noise_p, 1e-300) - 1.0
            bins_in = max(1.0, lv.width_hz / (fs / 1024))
            se_floor = (10 ** (spread / 10.0) - 1.0) / math.sqrt(bins_in)
            se = math.sqrt((1.0 / math.sqrt(K * lv.seg_len)) ** 2
                           + se_floor ** 2) * (1.0 + max(excess, 0.0))
            # VISIBLE means at or above the noise floor: the signal's power
            # density at least the noise's in this channel (0 dB) — the
            # literal meaning of "below the floor", which is what sifting
            # is for — AND measurably so. A perfect radiometer over a whole
            # dwell of white noise "measures" a −15 dB channel; a waterfall
            # does not show it, so that is not what visible may mean.
            measurable = bool(excess > 3.0 * se and excess >= 1.0)
            snr_db = 10 * math.log10(excess) if excess > 0 else None
            off = lv.offsets_hz[ch]
            for kk in pks:
                d = _refine(row, kk)
                a = float(alpha_axis[kk] + d * lv.fs_c / lv.nfft)
                stat = float(row[kk])
                held = fm is not None and bool(fm[kk])
                thr_k = thr_f if held else thr
                p = min(1.0, (trials_f if held else trials)
                        * float(gammaincc(K, stat)))
                centre = center_hz + off
                raw.append(Feature(
                    conj=conj, alpha_hz=a,
                    line_hz=(centre + 0.5 * a) if conj else a,
                    f_lo_hz=center_hz + off - 0.5 * lv.width_hz,
                    f_hi_hz=center_hz + off + 0.5 * lv.width_hz,
                    width_hz=lv.width_hz, stop_hz=float(center_hz),
                    statistic=stat, threshold=thr_k, p_family=p,
                    lag=int(lags[int(Targ[ch, kk])]), segments=K,
                    dalpha_hz=lv.dalpha_hz, energy_snr_db=snr_db,
                    energy_visible=measurable,
                    widths_seen_hz=[lv.width_hz]))
    return raw, best_miss

def _alpha_grid(lv: Level, conj: bool) -> tuple[np.ndarray, np.ndarray]:
    """(α of every statistic bin, which bins are searched) for one width:
    the ONE definition `_search_level` and `focus_trials` both use, so the
    cells a threshold is set from are the cells searched."""
    lim = alpha_limit_hz(lv.width_hz, lv.fs_c, conj, lv.trans_hz)
    if conj:
        axis = np.fft.fftfreq(lv.nfft, 1.0 / lv.fs_c)
        return axis, np.abs(axis) <= lim
    bins = np.arange(lv.nfft // 2 + 1)
    axis = bins * lv.fs_c / lv.nfft
    return axis, ((bins >= int(ALPHA_MIN_BINS * lv.nfft / lv.seg_len))
                  & (axis <= lim))


def focus_windows(held: list, drift_ppm: float = CONFIRM_DRIFT_PPM
                  ) -> list[dict]:
    """The windows a confirmation dwell looks in again: one per held
    feature — its α (non-conjugate) or its carrier (conjugate, allowing
    `drift_ppm` of drift), at its place. `confirms` matches with the same
    tolerances."""
    return [{"conj": bool(f.conj), "alpha_hz": float(f.alpha_hz),
             "line_hz": float(f.line_hz), "dalpha_hz": float(f.dalpha_hz),
             "f_lo_hz": float(f.f_lo_hz), "f_hi_hz": float(f.f_hi_hz),
             "drift_ppm": float(drift_ppm)} for f in held]


def _focus_mask(lv: Level, ch_centre_hz: float, conj: bool,
                alpha_axis: np.ndarray, windows: list) -> np.ndarray | None:
    """The bins of one channel that lie inside any window (None: none
    can)."""
    mask = None
    lo = ch_centre_hz - 0.5 * lv.width_hz
    hi = ch_centre_hz + 0.5 * lv.width_hz
    for w in windows:
        if bool(w["conj"]) != conj:
            continue
        a_tol = max(3.0 * w["dalpha_hz"], 3.0 * lv.dalpha_hz,
                    1e-6 * abs(w["alpha_hz"]))
        if conj:
            # a carrier at offset o from this channel's centre is α = 2·o;
            # `valid` already keeps only lines inside the channel
            tol = max(0.5 * a_tol, w["drift_ppm"] * 1e-6 * abs(w["line_hz"]))
            m = np.abs(alpha_axis - 2.0 * (w["line_hz"] - ch_centre_hz)) \
                <= 2.0 * tol
        else:
            if not (lo < w["f_hi_hz"] and hi > w["f_lo_hz"]):
                continue
            m = np.abs(alpha_axis - w["alpha_hz"]) <= a_tol
        mask = m if mask is None else (mask | m)
    return mask


def focus_trials(plan: SiftPlan, windows: list, center_hz: float) -> float:
    """How many cells (bins × lags, every width and channel) lie inside the
    windows: the confirmation's threshold is set from THESE, not from the
    whole search. With the whole search's count (millions of cells) the
    second look had to clear the first look's threshold, so the chance
    recordings were ~10⁶ below what the operator asked for and weak real
    finds were thrown away — measured, 4 of 10 confirmed at −15 dB in-band
    where the windows confirm 9 (review, 2026-10-12)."""
    n = 0
    for lv in plan.levels:
        for conj, lags in ((False, LAGS_NC), (True, LAGS_C)):
            if conj and not (plan.config.conjugate and lv.cells_c > 0):
                continue
            axis, valid = _alpha_grid(lv, conj)
            for off in lv.offsets_hz:
                m = _focus_mask(lv, float(center_hz) + off, conj, axis,
                                windows)
                if m is not None:
                    n += int(np.count_nonzero(m & valid)) * len(lags)
    return float(n)


def line_tol_hz(f: Feature, other_dalpha_hz: float = 0.0,
                drift_ppm: float = 0.0) -> float:
    """How far apart two CONJUGATE lines (carriers) may be and still be one:
    half the α tolerance (α is twice the carrier's offset), or the drift
    allowance — `drift_ppm` of the carrier frequency — if that is larger."""
    return max(0.5 * _alpha_tol(f, other_dalpha_hz),
               float(drift_ppm) * 1e-6 * abs(f.line_hz))


def same_line(f: Feature, g: Feature, tol_hz: float | None = None,
              drift_ppm: float = 0.0) -> bool:
    """Do two features describe the same line? Conjugate: the same carrier
    (`line_tol_hz`; `drift_ppm` when the two are from different looks).
    Non-conjugate: the same α and an overlapping place."""
    if f.conj != g.conj:
        return False
    if f.conj:
        tol = (0.5 * tol_hz if tol_hz is not None
               else line_tol_hz(f, g.dalpha_hz, drift_ppm))
        return abs(f.line_hz - g.line_hz) <= tol
    tol = tol_hz if tol_hz is not None else _alpha_tol(f, g.dalpha_hz)
    return (abs(f.alpha_hz - g.alpha_hz) <= tol
            and f.f_lo_hz < g.f_hi_hz and f.f_hi_hz > g.f_lo_hz)


def _merge(raw: list) -> list:
    """One feature per line and place.

    Conjugate lines are one line per CARRIER (`same_line`): seen in two
    overlapping channels or at several widths, they are merged, keeping the
    narrowest channel for the place.

    Non-conjugate rates (`_merge_rates`) are merged per WIDTH first — the
    same α in overlapping channels of one width is one run, its place their
    union — and only then across widths, narrowest first, a wider sighting
    joining the narrower feature it overlaps. Merging across widths first
    let the widest channel (which overlaps everything in the stop) chain
    separate emitters with one rate together, and narrowing to one channel
    split a wideband emitter into a feature per narrow channel (review,
    2026-10-12: 11 features for one 70 kHz emitter).

    Then harmonics of a rate (inside its place, `_within`) and sidebands
    of a carrier are folded into the line they belong to (`_fold`)."""
    raw = sorted(raw, key=lambda f: -f.margin)
    merged = (_merge_carriers([f for f in raw if f.conj])
              + _merge_rates([f for f in raw if not f.conj]))
    out = _fold(sorted(merged, key=lambda f: -f.margin))
    for f in out:
        f.id = f.id or uuid.uuid4().hex[:8]
    return sorted(out, key=lambda f: -f.margin)


def _take_stats(into: Feature, f: Feature) -> None:
    """`into` takes `f`'s significance (statistic, threshold, p, lag) —
    the place and α stay `into`'s."""
    into.statistic, into.threshold = f.statistic, f.threshold
    into.p_family, into.lag, into.segments = f.p_family, f.lag, f.segments


def _merge_carriers(raw: list) -> list:
    """Conjugate lines, strongest first: one per carrier, at the narrowest
    channel it was seen in (that channel's α, so the carrier stays right),
    with the strongest sighting's significance."""
    merged: list[Feature] = []
    for f in raw:
        hit = next((g for g in merged if same_line(g, f)), None)
        if hit is None:
            merged.append(f)
            continue
        if f.width_hz not in hit.widths_seen_hz:
            hit.widths_seen_hz.append(f.width_hz)
        if f.width_hz < hit.width_hz:
            hit.f_lo_hz, hit.f_hi_hz, hit.width_hz = (f.f_lo_hz, f.f_hi_hz,
                                                      f.width_hz)
            hit.alpha_hz, hit.dalpha_hz = f.alpha_hz, f.dalpha_hz
            hit.energy_snr_db = f.energy_snr_db
            hit.energy_visible = f.energy_visible
    return merged


def _same_rate(a: Feature, b: Feature) -> bool:
    return abs(a.alpha_hz - b.alpha_hz) <= _alpha_tol(a, b.dalpha_hz)


def _overlap(a: Feature, b: Feature) -> bool:
    return a.f_lo_hz < b.f_hi_hz and a.f_hi_hz > b.f_lo_hz


def _merge_rates(raw: list) -> list:
    """Non-conjugate lines (see `_merge`). `raw` is strongest first."""
    by_width: dict = {}
    for f in raw:
        by_width.setdefault(float(f.width_hz), []).append(f)
    feats: list[Feature] = []
    for width in sorted(by_width):
        # 1. one width: the same rate in overlapping channels is one run
        runs: list[Feature] = []
        for f in by_width[width]:                 # strongest first
            joined = [r for r in runs if _same_rate(r, f) and _overlap(r, f)]
            if not joined:
                runs.append(f)
                continue
            r = joined[0]
            for other in joined[1:]:              # f bridges two runs
                _grow(r, other)
                runs.remove(other)
            _grow(r, f)
        # 2. across widths: a run joins the narrower feature it overlaps
        # (the strongest, if several: it cannot tell them apart)
        for r in sorted(runs, key=lambda q: -q.margin):
            hits = [g for g in feats if _same_rate(g, r) and _overlap(g, r)]
            if not hits:
                feats.append(r)
                continue
            g = max(hits, key=lambda q: q.margin)
            for w in r.widths_seen_hz or [r.width_hz]:
                if w not in g.widths_seen_hz:
                    g.widths_seen_hz.append(w)
            if r.margin > g.margin:
                _take_stats(g, r)
            g.energy_visible = g.energy_visible or r.energy_visible
    return feats


def _grow(r: Feature, f: Feature) -> None:
    """Run `r` (one width) takes in `f`: the union of the places, the
    stronger significance, visible if either channel is."""
    r.f_lo_hz, r.f_hi_hz = min(r.f_lo_hz, f.f_lo_hz), max(r.f_hi_hz,
                                                          f.f_hi_hz)
    if f.margin > r.margin:
        _take_stats(r, f)
        r.alpha_hz, r.dalpha_hz = f.alpha_hz, f.dalpha_hz
    if f.energy_snr_db is not None and (r.energy_snr_db is None
                                        or f.energy_snr_db > r.energy_snr_db):
        r.energy_snr_db = f.energy_snr_db
    r.energy_visible = r.energy_visible or f.energy_visible


def _within(inner: Feature, outer: Feature) -> bool:
    """`inner`'s place lies inside `outer`'s. A harmonic of a rate is seen
    where its fundamental is, so it is folded only when its place is inside
    the fundamental's. Overlap was not enough: a rate seen only in a wide
    channel overlaps everything in it, and another emitter's rate was
    folded in as a "harmonic" and hidden with it (review, 2026-10-12)."""
    tol = 1e-6 * max(abs(outer.f_lo_hz), abs(outer.f_hi_hz), 1.0)
    return (inner.f_lo_hz >= outer.f_lo_hz - tol
            and inner.f_hi_hz <= outer.f_hi_hz + tol)


def _multiple(hi: Feature, lo: Feature) -> int:
    """hi's α as a whole multiple of lo's (0 when it is not one)."""
    q = hi.alpha_hz / max(1e-9, lo.alpha_hz)
    n = int(round(q))
    return n if n >= 1 and abs(q - n) < 0.01 else 0


def _fold(merged: list) -> list:
    """Strongest first. Sidebands into their carrier; harmonics (×2…×6)
    into their rate — only when seen inside the rate's place (`_within`)."""
    out = []
    used = set()
    for i, f in enumerate(merged):
        if i in used:
            continue
        if f.conj:
            # f is the strongest line not yet claimed. Its SIDEBANDS
            # (2f_c ± R for BPSK, ± f_m for AM) are weaker lines in mirror
            # pairs about it — the Hermitian symmetry of a real envelope's
            # square. Claim every such pair.
            used.add(i)
            tol = 1.5 * f.dalpha_hz
            others = [j for j, g in enumerate(merged)
                      if j not in used and g.conj and g.margin <= f.margin]
            for j in others:
                if j in used:
                    continue
                a = merged[j].line_hz
                mirror = 2 * f.line_hz - a
                k = next((k for k in others if k != j and k not in used
                          and abs(merged[k].line_hz - mirror) <= tol), None)
                if k is not None:
                    f.sidebands_hz += [a, merged[k].line_hz]
                    used.update((j, k))
            f.sidebands_hz.sort()
            out.append(f)
            continue
        free = [j for j, g in enumerate(merged)
                if j not in used and not g.conj]
        # the rate f is a harmonic of (or f itself): the lowest one whose
        # place holds f's
        base = min((j for j in free if _within(f, merged[j])
                    and 1 <= _multiple(f, merged[j]) <= 6),
                   key=lambda j: merged[j].alpha_hz)
        b = merged[base]
        members = [base]
        for j in free:
            if j == base or not _within(merged[j], b):
                continue
            n = _multiple(merged[j], b)
            if 1 <= n <= 6:
                a = merged[j].alpha_hz
                if n >= 2 and all(abs(h - a) > _alpha_tol(b)
                                  for h in b.harmonics_hz):
                    b.harmonics_hz.append(a)
                members.append(j)
        if i not in members:              # cannot happen; never lose f
            members.append(i)
        used.update(members)
        strongest = max(members, key=lambda j: merged[j].margin)
        if strongest != base:
            _take_stats(b, merged[strongest])
        b.harmonics_hz.sort()
        out.append(b)
    return out


# ---------------------------------------------------------------------------
# Classification — the ledger
# ---------------------------------------------------------------------------
def _alpha_tol(a: Feature, b_dalpha: float = 0.0) -> float:
    return max(3.0 * a.dalpha_hz, 3.0 * b_dalpha, 1e-6 * abs(a.alpha_hz))


class Ledger:
    """What is already explained, per run (background, local sightings)
    and per receiver (its own features, persisted in rf_data).

    Thread ownership: ONE thread (ATK's analyzer) uses a Ledger."""

    def __init__(self, receiver_path: Path | None = None,
                 class_rates: list | None = None, apart_hz: float = 0.0):
        self.receiver_path = Path(receiver_path) if receiver_path else None
        #: sightings must be further apart than this to count as separate
        #: places for the local rule: the run's bandwidth (so, other stops)
        self.apart_hz = float(apart_hz or 0.0)
        self.receiver: list[dict] = []
        self.receiver_meta: dict = {}
        self.background: dict = {}           # stop key -> [feature dicts]
        self.sightings: list[dict] = []      # every feature seen, for 'local'
        self.class_rates = list(class_rates if class_rates is not None
                                else _class_rates())
        if self.receiver_path is not None and self.receiver_path.is_file():
            try:
                d = json.loads(self.receiver_path.read_text(encoding="utf-8"))
                self.receiver = list(d.get("features") or [])
                self.receiver_meta = {k: v for k, v in d.items()
                                      if k != "features"}
            except (OSError, ValueError):
                self.receiver = []

    # -- the receiver's own ----------------------------------------------------
    def learn_receiver(self, features: list, stop_hz: float) -> int:
        n = 0
        for f in features:
            self.receiver.append({"conj": f.conj, "alpha_hz": f.alpha_hz,
                                  "dalpha_hz": f.dalpha_hz,
                                  "offset_hz": f.centre_hz - stop_hz,
                                  "line_offset_hz": (f.line_hz - stop_hz
                                                     if f.conj else 0.0),
                                  "width_hz": f.width_hz,
                                  "statistic": f.statistic})
            n += 1
        return n

    def save_receiver(self, note: str = "") -> Path | None:
        if self.receiver_path is None:
            return None
        self.receiver_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"calibrated": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                            time.gmtime()),
                "note": note, "features": self.receiver}
        tmp = self.receiver_path.with_name(self.receiver_path.name + ".tmp")
        tmp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        tmp.replace(self.receiver_path)
        self.receiver_meta = {k: v for k, v in meta.items() if k != "features"}
        return self.receiver_path

    def forget_receiver(self) -> None:
        self.receiver = []
        self.receiver_meta = {}
        if self.receiver_path is not None:
            try:
                self.receiver_path.unlink()
            except OSError:
                pass

    def receiver_words(self) -> str:
        if not self.receiver_meta:
            return ("Not calibrated: the radio's own features will be "
                    "reported as found until it is (Calibrate…).")
        return (f"{len(self.receiver)} receiver feature(s), calibrated "
                f"{self.receiver_meta.get('calibrated', '?')}.")

    # -- the place -------------------------------------------------------------
    def learn_background(self, features: list, stop_hz: float) -> None:
        key = str(int(round(stop_hz)))
        lst = self.background.setdefault(key, [])
        for f in features:
            lst.append({"conj": f.conj, "alpha_hz": f.alpha_hz,
                        "dalpha_hz": f.dalpha_hz, "f_lo_hz": f.f_lo_hz,
                        "f_hi_hz": f.f_hi_hz, "line_hz": f.line_hz})

    def _sighted_apart(self, f: Feature) -> int:
        """How many sightings of this α sit in places separate from each
        other — at different stops, and further apart than the run's
        bandwidth (`apart_hz`) and than either place. Non-conjugate only: a
        conjugate line IS a frequency, so it cannot be seen 'elsewhere'."""
        if f.conj:
            return 0
        f_span = f.f_hi_hz - f.f_lo_hz
        places = []                       # (centre, stop)
        for s in self.sightings:
            if s["conj"] != f.conj:
                continue
            if abs(s["alpha_hz"] - f.alpha_hz) > _alpha_tol(f, s["dalpha_hz"]):
                continue
            c, at = s["centre_hz"], s.get("stop_hz")
            sep = max(self.apart_hz, f_span, s.get("span_hz", s["width_hz"]))
            # never two places from one stop: one emitter's pieces are there
            if all(abs(c - p) > sep and (at is None or at != q)
                   for p, q in places):
                places.append((c, at))
        return len(places)

    # -- classification --------------------------------------------------------
    def classify(self, f: Feature, stop_hz: float, record: bool = True
                 ) -> Feature:
        """Labels and class for `f` (see the module docstring for the
        order). `record=False` classifies without adding a sighting (a
        confirmation dwell re-sees the same thing)."""
        labels = []
        off = f.centre_hz - stop_hz
        for r in self.receiver:
            if r["conj"] != f.conj:
                continue
            tol = _alpha_tol(f, r.get("dalpha_hz", 0.0))
            if f.conj:
                # hours after the calibration, at another stop: the line may
                # have moved by the radio's own drift (`LEDGER_DRIFT_PPM`)
                hit = abs(r.get("line_offset_hz", 0.0)
                          - (f.line_hz - stop_hz)) <= max(
                    0.5 * tol, LEDGER_DRIFT_PPM * 1e-6 * abs(stop_hz))
            else:
                hit = (abs(r["alpha_hz"] - f.alpha_hz) <= tol
                       and abs(r["offset_hz"] - off)
                       <= 0.5 * max(r["width_hz"], f.width_hz))
            if hit:
                labels.append("receiver")
                break
        if f.conj and abs(f.line_hz - stop_hz) <= 1.5 * f.dalpha_hz:
            labels.append("dc")
        if record:
            self.sightings.append({"conj": f.conj, "alpha_hz": f.alpha_hz,
                                   "dalpha_hz": f.dalpha_hz,
                                   "centre_hz": f.centre_hz,
                                   "width_hz": f.width_hz,
                                   "span_hz": f.f_hi_hz - f.f_lo_hz,
                                   "stop_hz": float(stop_hz)})
            if len(self.sightings) > 20000:
                del self.sightings[:5000]
        if self._sighted_apart(f) >= LOCAL_SIGHTINGS:
            labels.append("local")
        for b in self.background.get(str(int(round(stop_hz))), []):
            if b["conj"] != f.conj:
                continue
            tol = _alpha_tol(f, b.get("dalpha_hz", 0.0))
            if f.conj:
                hit = abs(b.get("line_hz", 0.0) - f.line_hz) <= max(
                    0.5 * tol, LEDGER_DRIFT_PPM * 1e-6 * abs(f.line_hz))
            else:
                hit = (abs(b["alpha_hz"] - f.alpha_hz) <= tol
                       and f.f_lo_hz < b["f_hi_hz"]
                       and f.f_hi_hz > b["f_lo_hz"])
            if hit:
                labels.append("background")
                break
        if f.energy_visible:
            labels.append("visible")
        if not f.conj:
            ex = []
            for name, rate in self.class_rates:
                for n in (1, 2, 3, 4):
                    if abs(f.alpha_hz - n * rate) <= max(
                            _alpha_tol(f), 0.002 * n * rate):
                        ex.append(name)
                        break
            if ex:
                labels.append("explained")
                f.explained_by = sorted(set(ex))
        f.labels = labels
        f.klass = next((k for k in CLASS_ORDER if k in labels), "new")
        return f

    def to_json(self) -> dict:
        return {"receiver": self.receiver, "receiver_meta": self.receiver_meta,
                "background": self.background,
                "sightings": self.sightings[-2000:]}


def _class_rates() -> list:
    try:
        from atk_diffusion.detect import classes as _classes
        rates = list(_classes.cycle_frequencies())
        for name, lag in _classes.cp_lags():
            c = _classes.get(name)
            if c is not None and c.symbol_rates:
                rates.append((name, float(c.symbol_rates[0])))
        return rates
    except Exception:                                     # noqa: BLE001
        return []


def confirms(first: Feature, second: list,
             drift_ppm: float = CONFIRM_DRIFT_PPM) -> Feature | None:
    """The feature in `second` (the confirmation dwell's) that reproduces
    `first`: same conjugation and either the same carrier within the drift
    allowance (`line_tol_hz`) or the same rate (α within ×3 the larger
    resolution) at an overlapping place — exactly the windows
    `focus_windows` gave the confirmation's search. None if nothing does."""
    best = None
    for g in second:
        if not same_line(first, g, drift_ppm=drift_ppm):
            continue
        if best is None or g.margin > best.margin:
            best = g
    return best


# ---------------------------------------------------------------------------
# Dwell files
# ---------------------------------------------------------------------------
class DwellReader:
    """`reader(start, count) -> complex64` over a raw dwell file, for
    `search`. Reads only what is asked; `samples` is the file's length."""

    def __init__(self, path, datatype: str):
        self.path = Path(path)
        self.datatype = _iq.norm_dt(datatype)
        self.bps = _iq.bytes_per_sample(self.datatype)
        self.samples = self.path.stat().st_size // self.bps

    def __call__(self, start: int, count: int) -> np.ndarray:
        start = max(0, int(start))
        count = max(0, min(int(count), self.samples - start))
        with open(self.path, "rb") as f:
            f.seek(start * self.bps)
            raw = f.read(count * self.bps)
        return _iq.to_complex(raw, self.datatype)


def _retry_os(fn, *args, tries: int = 6, wait_s: float = 0.25):
    """Run a file operation, retrying briefly on PermissionError: on Windows
    a virus scanner or the search indexer holds a just-written file for a
    moment, and a rename or delete in that moment fails though nothing is
    wrong. The last failure is raised."""
    for i in range(int(tries)):
        try:
            return fn(*args)
        except PermissionError:
            if i == int(tries) - 1:
                raise
            time.sleep(wait_s)


class DwellStore:
    """The run's dwell files: written while the radio is on a stop, then
    DISCARDED, HELD (waiting for a confirmation) or KEPT (a SigMF pair in
    captures\\sift\\<run>). The recorder thread writes; the analyzer thread
    decides; a lock guards the bookkeeping between them.

    A full disk budget refuses to KEEP — it never truncates a dwell being
    written (a hole in a capture shifts every later sample; ATK's recorder
    learned that), and the refusal is said."""

    def __init__(self, work_dir, keep_dir, datatype: str, sample_rate: float,
                 profile: str, budget_bytes: int, hw: str = "",
                 rf=None, log: logging.Logger | None = None):
        self.work_dir = Path(work_dir)
        self.keep_dir = Path(keep_dir)
        self.datatype = _iq.norm_dt(datatype)
        self.sample_rate = float(sample_rate)
        self.profile = str(profile)
        self.budget = int(budget_bytes)
        self.hw = str(hw)
        self.rf = rf
        self.log = log or _LOG
        self._lock = threading.Lock()
        self._files: dict = {}            # dwell_id -> info
        self._seq = 0
        self.kept_bytes = self.already_kept()

    def already_kept(self) -> int:
        """Bytes already kept by every sift with this radio (the keep
        folder's siblings): the budget is for them all together. It was per
        run, so a sift restarted every day could fill the drive budget by
        budget (review, 2026-10-12)."""
        total = 0
        try:
            for p in self.keep_dir.parent.glob("*/*.sigmf-data"):
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        except OSError:
            pass
        return int(total)

    # -- writing (the recorder thread) -----------------------------------------
    def open(self, stop_hz: float, kind: str, t0_utc: float | None = None
             ) -> tuple[str, object]:
        """A new dwell file. Returns (dwell_id, file object to write)."""
        with self._lock:
            self._seq += 1
            did = f"{self._seq:05d}_{stop_hz / 1e6:.6f}MHz_{kind}"
            self.work_dir.mkdir(parents=True, exist_ok=True)
            path = self.work_dir / f"{did}.part"
            fh = open(path, "wb")
            self._files[did] = {"path": path, "stop_hz": float(stop_hz),
                                "kind": kind, "state": "writing",
                                "t0_utc": float(t0_utc or time.time()),
                                "bytes": 0}
        return did, fh

    def closed(self, dwell_id: str, nbytes: int) -> Path:
        with self._lock:
            info = self._files[dwell_id]
            info["bytes"] = int(nbytes)
            info["state"] = "written"
            return info["path"]

    def info(self, dwell_id: str) -> dict:
        with self._lock:
            return dict(self._files.get(dwell_id) or {})

    # -- deciding (the analyzer thread) ------------------------------------------
    def discard(self, dwell_id: str, why: str = "") -> None:
        with self._lock:
            info = self._files.pop(dwell_id, None)
        if not info:
            return
        try:
            _retry_os(Path(info["path"]).unlink)
        except FileNotFoundError:
            pass
        except OSError as exc:
            self.log.warning("sift: could not delete %s: %s", info["path"], exc)
        self.log.debug("sift: discarded %s (%s)", dwell_id, why)

    def hold(self, dwell_id: str) -> None:
        with self._lock:
            info = self._files.get(dwell_id)
            if info:
                info["state"] = "held"

    def held(self) -> list[str]:
        with self._lock:
            return [k for k, v in self._files.items() if v["state"] == "held"]

    def can_keep(self, nbytes: int) -> bool:
        return self.kept_bytes + int(nbytes) <= self.budget

    def can_keep_all(self, dwell_ids) -> bool:
        """Would the budget take ALL of these (a first look and its
        confirmation are kept together or not at all)?"""
        with self._lock:
            n = sum(int((self._files.get(d) or {}).get("bytes", 0))
                    for d in dwell_ids)
        return self.can_keep(n)

    def keep(self, dwell_id: str, features: list, why: str,
             extra_global: dict | None = None) -> tuple[Path, Path] | None:
        """Rename the dwell into the keep folder as `<id>.sigmf-data` and
        write its meta, with one annotation per feature. None (and the file
        left where it was) when the budget refuses."""
        with self._lock:
            info = self._files.get(dwell_id)
        if not info:
            return None
        if not self.can_keep(info["bytes"]):
            self.log.warning("sift: disk budget reached; %s not kept", dwell_id)
            return None
        self.keep_dir.mkdir(parents=True, exist_ok=True)
        base = self.keep_dir / dwell_id
        data = _sigmf.data_path(base)
        _retry_os(os.replace, info["path"], data)
        n = info["bytes"] // _iq.bytes_per_sample(self.datatype)
        sigmf_dt = {"cu8": "cu8", "ci8": "ci8", "ci16": "ci16_le",
                    "ci16q11": "ci16_le", "cf32": "cf32_le"}[self.datatype]
        g = {"core:datatype": sigmf_dt, "core:sample_rate": self.sample_rate,
             "core:version": _sigmf.SIGMF_VERSION,
             "core:recorder": "ATK Diffusion Toolkit — sift",
             "core:description": f"Sift dwell kept: {why}",
             "atk:receiver_profile": self.profile, "atk:tier": "record"}
        if self.datatype == "ci16q11":
            g["atk:datatype"] = "ci16q11"
        if self.hw:
            g["core:hw"] = self.hw
        for k, v in (extra_global or {}).items():
            g[k if ":" in k else f"atk:{k}"] = v
        anns = []
        for f in features:
            ex = {"atk:proposer": "sift",
                  "atk:source": "confirmed" if f.confirmed else "proposed",
                  "atk:confidence": float(max(0.0, min(1.0, 1.0 - f.p_family))),
                  "atk:sift_conjugate": bool(f.conj),
                  "atk:sift_alpha_hz": float(f.alpha_hz),
                  "atk:sift_statistic": float(f.statistic),
                  "atk:sift_threshold": float(f.threshold),
                  "atk:sift_p_family": float(f.p_family),
                  "atk:sift_class": f.klass,
                  "atk:sift_segments": int(f.segments)}
            if f.conj:
                ex["atk:carrier_offset_hz"] = float(f.line_hz - info["stop_hz"])
            else:
                ex["atk:symbol_rate"] = float(f.alpha_hz)
            anns.append(_sigmf.Annotation(
                sample_start=0, sample_count=int(n),
                freq_lower_edge=float(f.f_lo_hz),
                freq_upper_edge=float(f.f_hi_hz),
                label=f"sift {f.klass}", comment=f.words(), extra=ex))
        meta = {"global": g,
                "captures": [{"core:sample_start": 0,
                              "core:frequency": float(info["stop_hz"]),
                              "core:datetime": _sigmf._iso(info["t0_utc"])}],
                "annotations": [a.to_sigmf() for a in anns]}
        mp = _sigmf.write_meta(base, meta)
        with self._lock:
            self.kept_bytes += info["bytes"]
            self._files.pop(dwell_id, None)
        if self.rf is not None:
            try:
                self.rf.record(data, "sift-dwell", note=why)
                self.rf.record(mp, "sift-dwell-meta", note=why)
            except Exception as exc:                      # noqa: BLE001
                self.log.warning("sift: write log entry failed: %s", exc)
        self.log.info("sift: kept %s (%s)", dwell_id, why)
        return data, mp

    def discard_all(self, why: str) -> int:
        with self._lock:
            ids = list(self._files)
        for d in ids:
            self.discard(d, why)
        return len(ids)


# ---------------------------------------------------------------------------
# The schedule
# ---------------------------------------------------------------------------
@dataclass
class Step:
    kind: str           # dwell | confirm | relook | calibrate | wait | done
    stop_hz: float = 0.0
    reason: str = ""
    feature_ids: list = field(default_factory=list)
    held_dwell: str = ""
    baseline: bool = False


class SiftSchedule:
    """Which stop next, and why — pure, so it is tested without a radio.

    A normal pass visits the stops in order. A NEW feature on a first look
    queues a CONFIRMATION of that stop at the front; a near miss queues one
    RELOOK (once per stop per pass); and the hopper WAITS while more than
    `MAX_PENDING_ANALYSIS` dwells are queued for analysis, because a dwell
    nobody will analyse in time is disk spent for nothing."""

    def __init__(self, config: SiftConfig, calibrate: bool = False):
        self.config = config
        self.calibrating = bool(calibrate)
        self.stops = list(config.stops_hz)
        if self.calibrating:
            # three stops spread across the list are enough to see the
            # radio's own features (they follow the radio, not the band)
            n = len(self.stops)
            pick = sorted({0, n // 2, n - 1}) if n > 2 else list(range(n))
            self.stops = [self.stops[i] for i in pick]
        self.pos = 0
        self.passes = 0
        self.queue: deque = deque()
        self.pending = 0
        self.relooked: set = set()
        self.dwells = 0

    @property
    def baseline(self) -> bool:
        return (not self.calibrating
                and self.passes < int(self.config.baseline_passes))

    def next_step(self) -> Step:
        if self.pending >= MAX_PENDING_ANALYSIS:
            return Step("wait", reason=f"{self.pending} dwell(s) waiting for "
                                       "analysis")
        if self.queue:
            return self.queue.popleft()
        if not self.stops:
            return Step("done", reason="no stops")
        if self.calibrating and self.pos >= len(self.stops):
            return Step("done", reason="calibration dwells finished")
        if self.pos >= len(self.stops):
            self.pos = 0
            self.passes += 1
            self.relooked.clear()
        f = self.stops[self.pos]
        self.pos += 1
        if self.calibrating:
            return Step("calibrate", f, "calibration: learning the radio's "
                                        "own features (antenna replaced by a "
                                        "terminator)")
        why = (f"baseline pass {self.passes + 1} of "
               f"{self.config.baseline_passes}: learning this place"
               if self.baseline else f"pass {self.passes + 1}, stop "
                                     f"{self.pos} of {len(self.stops)}")
        return Step("dwell", f, why, baseline=self.baseline)

    def dwell_started(self) -> None:
        self.pending += 1
        self.dwells += 1

    def dwell_lost(self) -> None:
        """A dwell that will never be analysed (a gap, an abort)."""
        self.pending = max(0, self.pending - 1)

    def analysed(self, step: Step, result: DwellResult) -> list[str]:
        """Feed back one analysed dwell; returns what was queued, in words."""
        self.pending = max(0, self.pending - 1)
        said = []
        if step.kind in ("calibrate", "confirm") or step.baseline:
            return said
        new = [f for f in result.features if recordable(f, self.config)]
        if new and self.config.confirm:
            self.queue.appendleft(Step(
                "confirm", step.stop_hz,
                f"confirm {len(new)} new feature(s) at "
                f"{step.stop_hz / 1e6:.6f} MHz",
                feature_ids=[f.id for f in new], held_dwell=result.dwell_id))
            said.append("confirmation queued")
        elif not new and result.near_miss and step.kind == "dwell" \
                and step.stop_hz not in self.relooked:
            self.relooked.add(step.stop_hz)
            nm = result.near_miss
            self.queue.append(Step(
                "relook", step.stop_hz,
                f"near miss: α {nm['alpha_hz']:,.1f} Hz at p "
                f"{nm['p_family']:.2f} — one more look"))
            said.append("relook queued")
        return said


# ---------------------------------------------------------------------------
# The run's log
# ---------------------------------------------------------------------------
class SiftLog:
    """JSON lines, one per decision, under runs\\sifts\\<run>\\sift_log.jsonl
    (the hunt log's shape). Every hop, dwell, analysis, classification,
    keep and discard is a line with its reason."""

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()

    @classmethod
    def for_run(cls, run_dir) -> "SiftLog":
        return cls(Path(run_dir) / "sift_log.jsonl")

    def put(self, event: str, **fields) -> dict:
        row = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "event": str(event), **fields}
        line = json.dumps(row, ensure_ascii=False, default=_jdefault)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")
        return row

    def entries(self) -> list[dict]:
        out = []
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            return out
        for line in text.splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                out.append(row)
        return out


def _jdefault(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "to_json"):
        return o.to_json()
    return str(o)


def run_dirs(rf, profile: str, run_id: str | None = None) -> dict:
    """The folders one run uses, under the ONE rf_data root:
    runs\\sifts\\<run>\\ (the log, the plan, the work files) and
    captures\\sift\\<run>\\ (what was kept), plus the receiver's own
    features at <profile>\\sift\\receiver_features.json."""
    rid = run_id or (time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-"
                     + uuid.uuid4().hex[:6])
    run = Path(rf.runs(profile)) / "sifts" / rid
    return {"run_id": rid, "run": run, "work": run / "work",
            "keep": Path(rf.captures(profile)) / "sift" / rid,
            "receiver": receiver_path(rf, profile)}


def receiver_path(rf, profile: str) -> Path:
    """The receiver's own features (`Ledger.save_receiver`): per receiver
    profile, because they follow the radio, not the place or the run."""
    return Path(rf.profile_dir(profile)) / "sift" / "receiver_features.json"


# ---------------------------------------------------------------------------
# One dwell, end to end (what ATK's analyzer thread calls)
# ---------------------------------------------------------------------------
def analyse_dwell(path, datatype: str, fs: float, stop_hz: float,
                  plan: SiftPlan, kind: str, ledger: Ledger, *,
                  baseline: bool = False, progress=None, cancel=None,
                  log: logging.Logger | None = None,
                  held: list | None = None) -> DwellResult:
    """Search one dwell file and classify what it holds. The DECISION
    (keep / hold / discard) is the caller's (`decide`), so it can be tested
    on its own. A confirmation dwell passes the first look's `held`
    features: they are looked for again at the confirmation's own Pfa in
    their windows only (`focus_windows`); the rest of the dwell is searched
    as a first look is."""
    log = log or _LOG
    t0 = time.perf_counter()
    reader = DwellReader(path, datatype)
    warnings = []
    probe = reader(0, min(reader.samples, int(fs)))
    clip = float(_iq.clipped_fraction(probe, datatype)) if probe.size else 0.0
    if clip > CLIP_WARN:
        warnings.append(f"{100 * clip:.3f} % of samples are at full scale: an "
                        "overloaded front end makes intermodulation products, "
                        "and they are cyclostationary too — lower the gain")
    focus = focus_windows(held or []) if kind == "confirm" else []
    res = search(reader, fs, stop_hz, plan, plan.pfa_first, progress=progress,
                 cancel=cancel, log=log, skip_s=0.0, focus=focus,
                 pfa_focus=plan.pfa_confirm if focus else None)
    feats = res["features"]
    for f in feats:
        ledger.classify(f, stop_hz, record=(kind != "confirm"))
    if kind == "calibrate":
        ledger.learn_receiver(feats, stop_hz)
    elif baseline:
        ledger.learn_background(feats, stop_hz)
    return DwellResult(stop_hz=float(stop_hz), kind=kind, features=feats,
                       near_miss=res["near_miss"],
                       floor_db_per_hz=res["floor_db_per_hz"],
                       clipped_fraction=clip,
                       seconds_analysed=res["seconds"],
                       analysis_s=time.perf_counter() - t0,
                       segments=res["segments"], warnings=warnings)


def recordable(f: Feature, config: SiftConfig) -> bool:
    """Is this feature's class one the operator asked to record?"""
    return (f.klass == "new"
            or (f.klass == "visible" and config.keep_visible)
            or (f.klass == "explained" and config.keep_explained))


def decide(step_kind: str, result: DwellResult, config: SiftConfig,
           baseline: bool, held_features: list | None = None) -> dict:
    """{action: keep | hold | discard, features, why} for one analysed
    dwell. A confirmation decides for the held dwell too: `confirmed` lists
    the held features it reproduced."""
    if step_kind == "calibrate":
        return {"action": "discard", "features": [],
                "why": f"calibration dwell: {len(result.features)} receiver "
                       "feature(s) learned"}
    if baseline:
        return {"action": "discard", "features": [],
                "why": f"baseline pass: {len(result.features)} feature(s) "
                       "learned as background"}
    if step_kind == "confirm":
        firsts = list(held_features or [])
        confirmed, matched = [], []
        for f in firsts:
            g = confirms(f, result.features)
            if g is not None:
                f.confirmed = True
                g.confirmed = True
                g.klass = f.klass
                confirmed.append(f)
                matched.append(g)
        if confirmed:
            return {"action": "keep", "features": confirmed,
                    "matched": matched,
                    "why": f"{len(confirmed)} of {len(firsts)} feature(s) "
                           "seen again on the confirmation dwell"}
        return {"action": "discard", "features": [],
                "why": "not seen again on the confirmation dwell (chance, or "
                       "a transient): the held dwell is discarded too"}
    rec = [f for f in result.features if recordable(f, config)]
    if not rec:
        return {"action": "discard", "features": [],
                "why": ("nothing new" if result.features else "nothing found")}
    if config.confirm:
        return {"action": "hold", "features": rec,
                "why": f"{len(rec)} new feature(s): held for confirmation"}
    return {"action": "keep", "features": rec,
            "why": f"{len(rec)} new feature(s) (no confirmation asked)"}
