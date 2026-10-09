# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The self-hunting receiver's policy, its bounded action set, and its log
(plan §4.B6; DARPA RFMLS task 4 and the agentic-RF paper, plan §9).

Bill: *"I definitely want to do"* the hunter — built on the receiver control
ATK already has, the detector (B1) as its eyes. And: **"it never
transmits."**

THE ACTION SET IS BOUNDED, AND TRANSMIT IS NOT IN IT.

    retune   (center_hz)          set_span (span_hz)     set_gain (gain_db)
    dwell    (seconds)            cut      (detection_id)
    mark     (detection_id, label)            skip     (cell, seconds)

`validate_action` refuses anything else — an unknown action, an unknown
parameter, a frequency outside the receiver's tuning range, a span wider than
the receiver sees at once, a gain outside its range, a dwell outside the
limits, a cut or mark of a detection that does not exist — with a sentence.
There is no transmit action to refuse into existence: a policy (rules or a
language model) can only CHOOSE from this list, and every choice is
validated again by the loop before the receiver is touched.

THE SAMPLE-RATE LAW HOLDS. `set_span` narrows how much of the capture the
hunter attends to; it can never be wider than the profile's rate shows at
once (the usable fraction of it, as ATK's swept view uses), and nothing here
changes the sample rate — the rate is the profile's (plan §3.3).

TWO POLICIES, RULES FIRST (plan §4.B6: "a policy (rules first, then the
primary model as the agent choosing from a bounded action set)").

* `RulePolicy` — cover the goal band in looks one usable span wide; look at
  unexplored cells first; then prefer cells that were RECENTLY ACTIVE with
  goal matches (a net that keyed up once keys up again) and cells not seen
  for a while; extend the dwell when what was seen is AMBIGUOUS (a
  narrowband signal that filled the look cannot yet be called a burst); skip
  a cell for a while when it showed only signals the goal is not about; skip
  cells outside the receiver's tuning range, once, saying why.
* `LlmPolicy` — the state summary and the allowed actions go to the
  host's model as JSON; its choice is parsed (ATK's last-balanced-JSON rule)
  and VALIDATED; anything invalid, or "I don't know", falls back to the
  rules, and the log says so. Every prompt ends with the way out.

THE LOG. `HuntLog` is JSON lines, one per action — time, action, params,
reason, which policy chose it, what was seen, and any refusal or fallback —
appended and fsynced as ATK's hunt store does, so a crash leaves the record
behind. Every retune is logged with its reason (plan §4.B6).

`run_hunt` is the loop: policy → validate → the receiver (a host object over
ATK's SdrController and the detector pipeline; `hunt.sim.SimReceiver` in
tests and the first experiment) → observe → log.

LIMITS. The rule policy's numbers (dwell lengths, the activity half-life,
the revisit time) are stated starting values, not tuned results — the
hunter_eval experiment is where they are measured. Gain is validated and
logged; the simulator does not model it.
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Callable, Protocol

from atk_diffusion import profiles as _profiles
from atk_diffusion.hunt.goal import BURST_MAX_S, Goal, _hz, burst_evidence, last_json

#: The hunter's whole vocabulary. There is no transmit.
ACTIONS = ("retune", "set_span", "set_gain", "dwell", "cut", "mark", "skip")

#: action -> {param: type}. Every parameter is required except where the
#: OPTIONAL table says otherwise; unknown parameters are refused.
ACTION_PARAMS = {
    "retune": {"center_hz": "number"},
    "set_span": {"span_hz": "number"},
    "set_gain": {"gain_db": "number"},
    "dwell": {"seconds": "number"},
    "cut": {"detection_id": "string"},
    "mark": {"detection_id": "string", "label": "string"},
    "skip": {"cell": "integer", "seconds": "number"},
}
OPTIONAL = {"mark": ("label",), "skip": ("seconds",)}

#: Words that ask for a transmission. Refused with the rule, by name.
_TRANSMIT_WORDS = ("transmit", "tx", "send", "key", "ptt", "jam", "beacon",
                   "emit", "radiate", "play", "replay", "inject")

#: Fraction of the captured bandwidth used per look (ATK's sweep.py).
USABLE_FRACTION = 0.75

#: Tuning ranges and gains, from ATK's own radio table (atk/core/radios.py,
#: Bill's code) and retune costs from atk/core/sweep.py. Gains: the RTL's
#: R820T range; the HackRF's amp + LNA + VGA as ATK's single gain number
#: splits it; bladeRF and others are clamped by the host.
TUNING = {
    "rtlsdr":    {"f_min": 24e6, "f_max": 1_766e6, "gain": (0.0, 49.6),
                  "retune_s": 0.16},
    "krakensdr": {"f_min": 24e6, "f_max": 1_766e6, "gain": (0.0, 49.6),
                  "retune_s": 0.16},
    "hackrf":    {"f_min": 1e6, "f_max": 6_000e6, "gain": (0.0, 116.0),
                  "retune_s": 0.12},
    "bladerf1":  {"f_min": 300e6, "f_max": 3_800e6, "gain": None,
                  "retune_s": 0.10, "f_min_xb200": 60e3},
    "bladerf2":  {"f_min": 47e6, "f_max": 6_000e6, "gain": None,
                  "retune_s": 0.10},
}


class Receiver(Protocol):
    """What the hunter drives. ATK's adapter implements it over the
    SdrController (retune, gain) and the detector pipeline (dwell returns
    the detections of that look); `hunt.sim.SimReceiver` in tests."""
    def now(self) -> float: ...
    def retune(self, center_hz: float) -> None: ...
    def set_span(self, span_hz: float) -> None: ...
    def set_gain(self, gain_db: float) -> None: ...
    def dwell(self, seconds: float) -> list: ...


@dataclass
class ReceiverLimits:
    f_min_hz: float
    f_max_hz: float
    sample_rate: float
    gain_min_db: float | None = None
    gain_max_db: float | None = None
    dwell_min_s: float = 0.05
    dwell_max_s: float = 30.0
    retune_latency_s: float = 0.15
    usable_fraction: float = USABLE_FRACTION
    label: str = ""

    @property
    def usable_span_hz(self) -> float:
        return float(self.sample_rate) * float(self.usable_fraction)

    @classmethod
    def for_profile(cls, profile: str, *, xb200: bool = False,
                    **overrides) -> "ReceiverLimits":
        p = _profiles.parse_profile_id(profile)
        t = TUNING.get(p.family)
        if t is None:
            raise ValueError(f"no tuning range is known for "
                             f"{_profiles.describe(profile)}; give the "
                             "receiver's limits explicitly")
        g = t.get("gain") or (None, None)
        f_min = t.get("f_min_xb200", t["f_min"]) if xb200 else t["f_min"]
        kw = dict(f_min_hz=f_min, f_max_hz=t["f_max"],
                  sample_rate=float(p.sample_rate), gain_min_db=g[0],
                  gain_max_db=g[1], retune_latency_s=t["retune_s"],
                  label=_profiles.describe(profile))
        kw.update(overrides)
        return cls(**kw)

    def to_json(self) -> dict:
        d = asdict(self)
        d["usable_span_hz"] = self.usable_span_hz
        return d


@dataclass
class Action:
    kind: str
    params: dict = field(default_factory=dict)
    reason: str = ""
    policy: str = "rules"

    def to_json(self) -> dict:
        return {"action": self.kind, "params": dict(self.params),
                "reason": self.reason, "policy": self.policy}

    @classmethod
    def from_json(cls, d: dict, policy: str = "llm") -> "Action":
        kind = d.get("action", d.get("kind", ""))
        params = d.get("params") or {}
        return cls(str(kind), dict(params) if isinstance(params, dict) else {},
                   str(d.get("reason", "") or ""), policy)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@dataclass
class Cell:
    index: int
    center_hz: float
    f_lo_hz: float
    f_hi_hz: float
    in_range: bool = True
    visits: int = 0
    last_visit_s: float | None = None
    activity: float = 0.0          # goal matches, decayed
    activity_t: float = 0.0
    matches: int = 0
    last_match_s: float | None = None
    quiet_nonmatch: int = 0        # consecutive looks with only non-goal signals
    cooldown_until_s: float = -math.inf
    skipped_for_range: bool = False
    looked_s: float = 0.0          # seconds of looking here, decayed
    looked_t: float = 0.0

    def activity_now(self, now: float, half_life: float) -> float:
        if self.activity <= 0:
            return 0.0
        return self.activity * 0.5 ** (max(0.0, now - self.activity_t) / half_life)

    def looked_now(self, now: float, half_life: float) -> float:
        if self.looked_s <= 0:
            return 0.0
        return self.looked_s * 0.5 ** (max(0.0, now - self.looked_t) / half_life)


def make_cells(goal: Goal, limits: ReceiverLimits) -> tuple[list[Cell], float]:
    """The goal band in looks one usable span wide (ATK's sweep layout: evenly
    spread, slightly overlapping). -> (cells, span)."""
    lo, hi = float(goal.f_lo_hz), float(goal.f_hi_hz)
    width = hi - lo
    usable = limits.usable_span_hz
    if width <= usable:
        c = 0.5 * (lo + hi)
        cells = [Cell(0, c, lo, hi)]
        span = max(width, 1.0)
    else:
        n = int(math.ceil(width / usable))
        step = width / n
        cells = [Cell(i, lo + step * (i + 0.5), lo + step * i, lo + step * (i + 1))
                 for i in range(n)]
        span = usable
    for c in cells:
        c.in_range = limits.f_min_hz <= c.center_hz <= limits.f_max_hz
    return cells, span


@dataclass
class Found:
    detection_id: str
    center_hz: float
    bw_hz: float
    t0: float
    t1: float
    found_at_s: float
    cell: int
    why: str
    cls: str = ""
    measurements: dict = field(default_factory=dict)


class HuntState:
    """Everything the policies read; updated by `observe` after each look."""

    def __init__(self, goal: Goal, limits: ReceiverLimits):
        self.goal = goal
        self.limits = limits
        self.cells, self.target_span = make_cells(goal, limits)
        self.now_s = 0.0
        self.center_hz: float | None = None
        self.span_hz: float | None = None
        self.gain_db: float | None = None
        self.last_action: Action | None = None
        self.looked_here = False
        self.last_look: list = []
        self.last_look_window: tuple | None = None
        self.ambiguous: list = []
        self.extensions = 0
        self.pending: deque = deque()
        self.detections: dict = {}
        self.found: list[Found] = []
        self.marked: set = set()
        self.cut_ids: set = set()
        #: signals that filled consecutive looks: (centre, bw, first t0, last t1)
        self._carry: list = []

    def cell_at(self, f_hz: float | None) -> Cell | None:
        """The cell whose band holds `f_hz` (the receiver's centre)."""
        if f_hz is None:
            return None
        for c in self.cells:
            if c.f_lo_hz - 1.0 <= f_hz <= c.f_hi_hz + 1.0:
                return c
        return None

    def remember(self, det) -> None:
        self.detections[det.id] = det
        if len(self.detections) > 2000:            # bounded memory
            for k in list(self.detections)[:500]:
                if k not in self.marked:
                    self.detections.pop(k, None)

    def _same_as_found(self, det) -> Found | None:
        c = 0.5 * (det.f_lo + det.f_hi)
        bw = det.f_hi - det.f_lo
        for f in reversed(self.found[-50:]):
            if abs(f.center_hz - c) <= 0.5 * max(f.bw_hz, bw) and \
                    det.t0 <= f.t1 + 0.5 and det.t1 >= f.t0 - 0.5:
                return f
        return None

    def observe(self, dets: list, t0: float, t1: float, half_life: float = 600.0
                ) -> dict:
        """Take one look's detections: matches are queued for mark and cut
        (once per signal — a burst seen across two looks is one find);
        ambiguous ones are kept for the policy; the cell's activity moves."""
        self.now_s = t1
        self.last_look = list(dets)
        self.last_look_window = (t0, t1)
        self.ambiguous = []
        cell = self.cell_at(self.center_hz)
        if cell is not None:
            cell.visits += 1
            cell.last_visit_s = t1
            cell.looked_s = cell.looked_now(t1, half_life) + max(0.0, t1 - t0)
            cell.looked_t = t1
        n_match = n_non = 0
        verdicts = []
        carry = []
        for d in dets:
            self.remember(d)
            ok, why = self.goal.matches(d, (t0, t1))
            if ok is None and self.goal.burstiness != "any" \
                    and burst_evidence(d, (t0, t1)) is None:
                # it filled the look: was it there through the last look too?
                c, bw = 0.5 * (d.f_lo + d.f_hi), d.f_hi - d.f_lo
                first = d.t0
                for pc, pbw, pfirst, plast in self._carry:
                    if abs(pc - c) <= 0.5 * max(pbw, bw) and d.t0 <= plast + 0.25:
                        first = min(first, pfirst)
                        break
                carry.append((c, bw, first, d.t1))
                if d.t1 - first > BURST_MAX_S:
                    whole = dc_replace(d, t0=first)
                    ok, why = self.goal.matches(whole, None)
                    if ok is not True:
                        why = f"{why} (present for {d.t1 - first:.1f} s across looks)"
            verdicts.append((d, ok, why))
            if ok is True:
                n_match += 1
                prev = self._same_as_found(d)
                if prev is not None:
                    prev.t1 = max(prev.t1, d.t1)
                    continue
                f = Found(d.id, 0.5 * (d.f_lo + d.f_hi), d.f_hi - d.f_lo, d.t0,
                          d.t1, t1, cell.index if cell else -1, why,
                          str(getattr(d, "cls", "") or ""),
                          dict(getattr(d, "measurements", {}) or {}))
                self.found.append(f)
                self.pending.append(Action("mark", {"detection_id": d.id,
                                                    "label": self.goal.describe()},
                                           why))
                self.pending.append(Action("cut", {"detection_id": d.id},
                                           "cut what was found, for the "
                                           "analyst and the route"))
            elif ok is None:
                self.ambiguous.append((d, why))
            else:
                n_non += 1
        self._carry = carry
        if cell is not None:
            if n_match:
                a = cell.activity_now(t1, half_life)
                cell.activity, cell.activity_t = a + n_match, t1
                cell.matches += n_match
                cell.last_match_s = t1
                cell.quiet_nonmatch = 0
            elif n_non and not self.ambiguous:
                cell.quiet_nonmatch += 1
            elif not dets:
                cell.quiet_nonmatch = 0
        return {"matches": n_match, "ambiguous": len(self.ambiguous),
                "other": n_non, "verdicts": verdicts}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and \
        math.isfinite(float(v))


def validate_action(action, limits: ReceiverLimits,
                    state: HuntState | None = None) -> tuple[bool, str]:
    """(ok, why). Anything outside the bounded set, or outside the receiver's
    limits, is refused in a sentence."""
    if isinstance(action, dict):
        action = Action.from_json(action)
    if not isinstance(action, Action):
        return False, "that is not an action"
    kind = str(action.kind or "").strip().lower()
    if kind not in ACTIONS:
        if any(w == kind or kind.startswith(w + "_") for w in _TRANSMIT_WORDS):
            return False, (f"'{kind}' is not an action the hunter has — it "
                           "never transmits; it can only "
                           + ", ".join(ACTIONS) + ".")
        return False, (f"'{kind or '(nothing)'}' is not one of the hunter's "
                       f"actions ({', '.join(ACTIONS)}).")
    params = action.params if isinstance(action.params, dict) else {}
    spec = ACTION_PARAMS[kind]
    extra = sorted(set(params) - set(spec))
    if extra:
        return False, (f"{kind} does not take {', '.join(extra)}; it takes "
                       f"{', '.join(spec)}.")
    for p, typ in spec.items():
        if p not in params:
            if p in OPTIONAL.get(kind, ()):
                continue
            return False, f"{kind} needs {p}."
        v = params[p]
        if typ == "number" and not _is_number(v):
            return False, f"{kind}: {p} must be a number, not {v!r}."
        if typ == "integer" and not (isinstance(v, int) and not isinstance(v, bool)):
            return False, f"{kind}: {p} must be a whole number, not {v!r}."
        if typ == "string" and not isinstance(v, str):
            return False, f"{kind}: {p} must be text, not {v!r}."
    if kind == "retune":
        f = float(params["center_hz"])
        if not (limits.f_min_hz <= f <= limits.f_max_hz):
            return False, (f"{_hz(f)} is outside the receiver's tuning range "
                           f"({_hz(limits.f_min_hz)} to {_hz(limits.f_max_hz)}).")
    elif kind == "set_span":
        s = float(params["span_hz"])
        if s <= 0:
            return False, "a span must be wider than nothing."
        if s > limits.usable_span_hz + 1.0:
            return False, (f"a span of {_hz(s)} is wider than the receiver "
                           f"sees at once ({_hz(limits.usable_span_hz)} at "
                           "this profile's rate). The hunter never changes "
                           "the sample rate — the rate is the profile's.")
    elif kind == "set_gain":
        g = float(params["gain_db"])
        lo, hi = limits.gain_min_db, limits.gain_max_db
        if (lo is not None and g < lo) or (hi is not None and g > hi):
            return False, (f"a gain of {g:g} dB is outside the receiver's "
                           f"range ({lo:g} to {hi:g} dB).")
    elif kind == "dwell":
        s = float(params["seconds"])
        if not (limits.dwell_min_s <= s <= limits.dwell_max_s):
            return False, (f"a dwell of {s:g} s is outside "
                           f"{limits.dwell_min_s:g}–{limits.dwell_max_s:g} s.")
    elif kind in ("cut", "mark"):
        did = params["detection_id"]
        if state is not None and did not in state.detections:
            return False, f"there is no detection {did!r} to {kind}."
    elif kind == "skip":
        i = params["cell"]
        if state is not None and not (0 <= i < len(state.cells)):
            return False, f"there is no cell {i} (there are {len(state.cells)})."
        if "seconds" in params and float(params["seconds"]) < 0:
            return False, "a skip cannot last less than nothing."
    return True, ""


# ---------------------------------------------------------------------------
# The rule policy
# ---------------------------------------------------------------------------
class RulePolicy:
    """Rules first. Each look is chosen by one of three rules, in order:

    1. STAY when the last look was ambiguous (up to `max_extensions` more
       looks of `extension_s`): a narrowband signal that filled the look may
       be a burst or a carrier, and only more time tells.
    2. EXPLOIT — go where goal matches have been seen: among cells with
       recent matches, the one with the most bursts likely to be ON NOW,
       `rate x min(time since last look, horizon_s)`. The rate is matches per
       second of looking, both decayed with `half_life_s`, so the estimate
       follows a band that changes. Saturating the age at the burst horizon
       makes the receiver ROTATE among active channels with short looks — a
       burst that is on when it arrives is caught, held until it ends, and
       the receiver moves on — rather than sit on the busiest one.
    3. EXPLORE — every `explore_every`-th move, and whenever nothing is
       active: the unexplored cell nearest the receiver, else the cell
       unvisited longest. Without this a hunter finds the first busy channel
       and never learns of the second.

    Plus: skip cells outside the receiver's range (once, saying why) and rest
    a cell that showed only what the goal is not about for
    `quiet_looks_to_skip` looks. Every number here is a stated starting value
    to be measured by experiments.hunter_eval, not a tuned result."""
    name = "rules"

    def __init__(self, base_dwell_s: float = 1.0, extension_s: float = 1.0,
                 max_extensions: int = 6, half_life_s: float = 600.0,
                 horizon_s: float = 5.0, explore_every: int = 4,
                 cooldown_s: float = 60.0, quiet_looks_to_skip: int = 3,
                 active_threshold: float = 0.25):
        self.base_dwell_s = base_dwell_s
        self.extension_s = extension_s
        self.max_extensions = max_extensions
        self.half_life_s = half_life_s
        self.horizon_s = horizon_s
        self.explore_every = max(2, int(explore_every))
        self.cooldown_s = cooldown_s
        self.quiet_looks_to_skip = quiet_looks_to_skip
        self.active_threshold = active_threshold
        self._moves = 0

    def _dwell(self, st: HuntState) -> float:
        s = self.base_dwell_s * (2.0 if st.goal.prefer_weak else 1.0)
        return min(max(s, st.limits.dwell_min_s), st.limits.dwell_max_s)

    def rate(self, cell: Cell, now: float) -> float:
        """Goal matches per second of looking at this cell (decayed)."""
        m = cell.activity_now(now, self.half_life_s)
        t = cell.looked_now(now, self.half_life_s)
        return m / max(t, 1.0)

    def _age(self, cell: Cell, now: float) -> float:
        return math.inf if cell.last_visit_s is None else now - cell.last_visit_s

    def next_action(self, st: HuntState) -> Action:
        if st.pending:
            return st.pending.popleft()
        if st.span_hz is None or abs(st.span_hz - st.target_span) > 1.0:
            n = len(st.cells)
            return Action("set_span", {"span_hz": float(st.target_span)},
                          f"cover {_hz(st.goal.f_lo_hz)}–{_hz(st.goal.f_hi_hz)} "
                          f"in {n} look{'s' if n != 1 else ''} of "
                          f"{_hz(st.target_span)}")
        here = st.cell_at(st.center_hz)
        last = st.last_action.kind if st.last_action else ""
        now = st.now_s
        # 1. ambiguous: stay and look longer
        if last == "dwell" and st.ambiguous and here is not None \
                and st.extensions < self.max_extensions:
            st.extensions += 1
            d, why = st.ambiguous[0]
            return Action("dwell", {"seconds": float(self.extension_s)},
                          f"ambiguous at {_hz(0.5 * (d.f_lo + d.f_hi))}: {why} "
                          f"— staying {self.extension_s:g} s longer "
                          f"({st.extensions} of {self.max_extensions})")
        # just arrived: look
        if last == "retune" and here is not None:
            st.extensions = 0
            return Action("dwell", {"seconds": self._dwell(st)},
                          f"look at {_hz(here.f_lo_hz)}–{_hz(here.f_hi_hz)}")
        # out-of-range cells: skip once, saying why
        for c in st.cells:
            if not c.in_range and not c.skipped_for_range:
                c.skipped_for_range = True
                c.cooldown_until_s = math.inf
                return Action("skip", {"cell": c.index},
                              f"{_hz(c.center_hz)} is outside the receiver's "
                              f"tuning range ({_hz(st.limits.f_min_hz)}–"
                              f"{_hz(st.limits.f_max_hz)})")
        # a cell that shows only what the goal is not about: rest it
        if here is not None and here.quiet_nonmatch >= self.quiet_looks_to_skip \
                and here.cooldown_until_s < now:
            here.quiet_nonmatch = 0
            here.cooldown_until_s = now + self.cooldown_s
            return Action("skip", {"cell": here.index,
                                   "seconds": float(self.cooldown_s)},
                          f"only signals the goal is not about at "
                          f"{_hz(here.center_hz)} for {self.quiet_looks_to_skip}"
                          f" looks — back in {self.cooldown_s:g} s")
        live = [c for c in st.cells if c.in_range and c.cooldown_until_s <= now]
        if not live:
            live = [c for c in st.cells if c.in_range]
        if not live:
            return Action("dwell", {"seconds": float(st.limits.dwell_min_s)},
                          "no cell of the goal band is in the receiver's range")
        st.extensions = 0
        active = [c for c in live
                  if c.activity_now(now, self.half_life_s) >= self.active_threshold]
        self._moves += 1
        explore = (not active) or (self._moves % self.explore_every == 0)
        if explore:
            ref = st.center_hz if st.center_hz is not None else live[0].center_hz
            fresh = [c for c in live if c.visits == 0]
            pool = [c for c in live if c not in active] or live
            if fresh:
                pick = min(fresh, key=lambda c: (abs(c.center_hz - ref), c.index))
                why = "unexplored"
            else:
                pick = max(pool, key=lambda c: (self._age(c, now), -c.index))
                why = f"exploring: not looked at for {self._age(pick, now):.0f} s"
        else:
            def prio(c):
                return self.rate(c, now) * min(self._age(c, now), self.horizon_s)
            pick = max(active, key=lambda c: (prio(c), -c.index))
            r = self.rate(pick, now)
            why = (f"recently active: {pick.matches} goal match"
                   f"{'es' if pick.matches != 1 else ''} here (about "
                   f"{60.0 * r:.1f} a minute of looking), last looked at "
                   f"{self._age(pick, now):.0f} s ago")
        if here is not None and pick.index == here.index:
            return Action("dwell", {"seconds": self._dwell(st)}, f"stay: {why}")
        return Action("retune", {"center_hz": float(pick.center_hz)},
                      f"{why}: {_hz(pick.f_lo_hz)}–{_hz(pick.f_hi_hz)}")


class FixedScanPolicy:
    """The classical comparator: step through the cells in a fixed order, the
    same dwell everywhere, no memory of where matches were — but, like any
    scanner (and ATK's PTT scanner), it STOPS ON ACTIVITY: while a look shows
    a goal match or something ambiguous it holds for up to `max_hold` more
    looks. Finds are marked and cut as the hunter's are, so both policies
    are scored the same way."""
    name = "fixed_scan"

    def __init__(self, dwell_s: float = 1.0, max_hold: int = 6):
        self.dwell_s = dwell_s
        self.max_hold = max_hold
        self._hold = 0
        self._i = -1

    def next_action(self, st: HuntState) -> Action:
        if st.pending:
            return st.pending.popleft()
        if st.span_hz is None or abs(st.span_hz - st.target_span) > 1.0:
            return Action("set_span", {"span_hz": float(st.target_span)},
                          "the scan's span", policy=self.name)
        live = [c for c in st.cells if c.in_range]
        if not live:
            return Action("dwell", {"seconds": float(st.limits.dwell_min_s)},
                          "nothing in range", policy=self.name)
        last = st.last_action.kind if st.last_action else ""
        if last == "retune":
            self._hold = 0
            return Action("dwell", {"seconds": float(self.dwell_s)},
                          "scan dwell", policy=self.name)
        active = st.ambiguous or any(
            st.goal.matches(d, st.last_look_window)[0] is True
            for d in st.last_look)
        if last == "dwell" and active and self._hold < self.max_hold:
            self._hold += 1
            return Action("dwell", {"seconds": float(self.dwell_s)},
                          "scanner hold: activity here", policy=self.name)
        self._hold = 0
        self._i = (self._i + 1) % len(live)
        c = live[self._i]
        if st.center_hz is not None and abs(c.center_hz - st.center_hz) < 1.0:
            return Action("dwell", {"seconds": float(self.dwell_s)},
                          "scan dwell", policy=self.name)
        return Action("retune", {"center_hz": float(c.center_hz)},
                      f"scan step {self._i + 1} of {len(live)}",
                      policy=self.name)


# ---------------------------------------------------------------------------
# The language-model policy
# ---------------------------------------------------------------------------
def state_summary(st: HuntState, max_cells: int = 12) -> dict:
    """What the model is shown: the goal, the receiver, the busiest and the
    stalest cells, and the last look."""
    cells = []
    for c in st.cells:
        if not c.in_range:
            continue
        cells.append({
            "cell": c.index, "centre_mhz": round(c.center_hz / 1e6, 4),
            "visits": c.visits,
            "seconds_since_visit": (None if c.last_visit_s is None
                                    else round(st.now_s - c.last_visit_s, 1)),
            "recent_goal_matches": round(c.activity_now(st.now_s, 600.0), 2),
            "resting": c.cooldown_until_s > st.now_s})
    cells.sort(key=lambda d: (-(d["recent_goal_matches"]),
                              -(d["seconds_since_visit"] or 1e9)))
    seen = [{"detection_id": d.id,
             "centre_mhz": round(0.5 * (d.f_lo + d.f_hi) / 1e6, 5),
             "bandwidth_khz": round((d.f_hi - d.f_lo) / 1e3, 2),
             "class": d.cls or None, "family": d.family,
             "confidence": d.confidence} for d in st.last_look[:10]]
    return {"goal": st.goal.describe(),
            "receiver": {"label": st.limits.label,
                         "centre_mhz": (None if st.center_hz is None
                                        else round(st.center_hz / 1e6, 4)),
                         "span_khz": (None if st.span_hz is None
                                      else round(st.span_hz / 1e3, 1)),
                         "gain_db": st.gain_db,
                         "tunes_from_mhz": st.limits.f_min_hz / 1e6,
                         "tunes_to_mhz": st.limits.f_max_hz / 1e6,
                         "widest_span_khz": round(st.limits.usable_span_hz / 1e3, 1)},
            "cells": cells[:max_cells],
            "last_look": seen,
            "ambiguous": [why for _d, why in st.ambiguous[:5]],
            "seconds_hunting": round(st.now_s, 1)}


def llm_prompt(st: HuntState) -> str:
    allowed = {k: v for k, v in ACTION_PARAMS.items() if k not in ("cut", "mark")}
    return (
        "You steer a radio receiver that hunts for signals an analyst "
        "described. It only listens: there is no action that transmits.\n"
        "STATE (JSON):\n" + json.dumps(state_summary(st), indent=1) + "\n"
        "ALLOWED ACTIONS and their parameters (JSON):\n"
        + json.dumps(allowed) + "\n"
        "Choose the ONE next action that best finds what the goal describes: "
        "look where matches were recently seen, look where nobody has looked "
        "for a while, stay longer when something is ambiguous.\n"
        'Reply with one JSON object: {"action": "...", "params": {...}, '
        '"reason": "one sentence"}.\n'
        "If you are unsure which action is best, say \"I don't know\" and the "
        "rules will choose.")


class LlmPolicy:
    """The host's model chooses from the bounded set; the rules catch every
    choice that is invalid, unreadable, or "I don't know"."""
    name = "llm"

    def __init__(self, llm: Callable[[str], str],
                 fallback: RulePolicy | None = None):
        self.llm = llm
        self.fallback = fallback or RulePolicy()
        self.last_fallback = ""
        self.counts = {"llm": 0, "fallback": 0}

    def next_action(self, st: HuntState) -> Action:
        self.last_fallback = ""
        if st.pending:                       # finds are marked by rule, always
            return st.pending.popleft()
        if st.span_hz is None:
            return self.fallback.next_action(st)
        try:
            reply = str(self.llm(llm_prompt(st)) or "")
        except Exception as exc:                           # noqa: BLE001
            return self._fall(st, f"the model could not be asked "
                                  f"({type(exc).__name__})")
        obj = last_json(reply)
        if obj is None:
            low = reply.lower().replace("’", "'")
            why = ("the model said it did not know" if "don't know" in low
                   or "do not know" in low else "the model gave no JSON action")
            return self._fall(st, why)
        act = Action.from_json(obj, policy="llm")
        if act.kind in ("cut", "mark"):
            return self._fall(st, "marks and cuts are made by rule, not chosen")
        ok, why = validate_action(act, st.limits, st)
        if not ok:
            return self._fall(st, f"the model's choice was refused: {why}")
        if not act.reason:
            act.reason = "the model's choice (it gave no reason)"
        self.counts["llm"] += 1
        return act

    def _fall(self, st: HuntState, why: str) -> Action:
        self.counts["fallback"] += 1
        self.last_fallback = why
        a = self.fallback.next_action(st)
        a.policy = "rules (fallback)"
        a.reason = f"{a.reason} [{why}; the rules chose]"
        return a


# ---------------------------------------------------------------------------
# The log
# ---------------------------------------------------------------------------
def _det_summary(d) -> dict:
    return {"id": d.id, "f_lo": d.f_lo, "f_hi": d.f_hi, "t0": d.t0, "t1": d.t1,
            "family": d.family, "cls": d.cls, "confidence": d.confidence,
            "snr_db": d.snr_db, "sources": list(d.sources), "state": d.state}


class HuntLog:
    """JSON lines, one per action (and a start and a stop event), appended
    and fsynced. A torn last line from a crash is skipped on reading."""

    def __init__(self, path, fsync: bool = True):
        self.path = Path(path)
        self.fsync = fsync

    @classmethod
    def for_run(cls, rf, profile: str, run_id: str | None = None,
                fsync: bool = True) -> "HuntLog":
        rid = run_id or (time.strftime("%Y%m%d-%H%M%S", time.gmtime())
                         + "-" + uuid.uuid4().hex[:6])
        return cls(Path(rf.runs(profile)) / "hunts" / rid / "hunt_log.jsonl",
                   fsync=fsync)

    def _append(self, row: dict) -> dict:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            if self.fsync:
                f.flush()
                os.fsync(f.fileno())
        return row

    def event(self, name: str, t_s: float = 0.0, **fields) -> dict:
        return self._append({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                   time.gmtime()),
                             "t": round(float(t_s), 4), "event": name, **fields})

    def write(self, t_s: float, action: Action, seen: list | None = None,
              refused: str = "", result: str = "", **fields) -> dict:
        row = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "t": round(float(t_s), 4), "action": action.kind,
               "params": dict(action.params), "reason": action.reason,
               "policy": action.policy}
        if seen is not None:
            row["seen"] = [_det_summary(d) for d in seen]
        if refused:
            row["refused"] = refused
        if result:
            row["result"] = result
        row.update(fields)
        return self._append(row)

    def entries(self) -> list[dict]:
        out = []
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            return out
        for line in text.splitlines():
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(row, dict):
                out.append(row)
        return out

    def retunes(self) -> list[dict]:
        return [e for e in self.entries() if e.get("action") == "retune"
                and not e.get("refused")]


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------
@dataclass
class HuntResult:
    steps: int
    found: list
    retunes: int
    refusals: int
    duration_s: float
    stopped_because: str
    log_path: str
    fallbacks: int = 0

    def lines(self) -> list[str]:
        return [f"Hunted {self.duration_s:.0f} s in {self.steps} steps: "
                f"{len(self.found)} find{'s' if len(self.found) != 1 else ''}, "
                f"{self.retunes} retunes, {self.refusals} refused actions"
                + (f", {self.fallbacks} model choices replaced by the rules"
                   if self.fallbacks else "") + ".",
                f"Stopped because {self.stopped_because}.",
                f"Every action and its reason: {self.log_path}"]


def run_hunt(goal: Goal, receiver, limits: ReceiverLimits, policy=None,
             log: HuntLog | None = None, *, duration_s: float | None = None,
             max_steps: int = 100_000, stop: Callable[[], bool] | None = None,
             max_refusals_in_a_row: int = 20,
             progress: Callable[[str], None] | None = None
             ) -> tuple[HuntResult, HuntState]:
    """Run the hunter until `duration_s` of receiver time has passed, or
    `max_steps`, or `stop()` says so. Refuses to start on a goal with
    problems (the sentence says which)."""
    probs = goal.problems()
    if probs:
        raise ValueError("The hunt cannot start: " + "; ".join(probs) + ".")
    policy = policy or RulePolicy()
    st = HuntState(goal, limits)
    if log is None:
        raise ValueError("a hunt needs a log — every retune is written down "
                         "with its reason (HuntLog.for_run)")
    t_start = float(receiver.now())
    st.now_s = t_start
    log.event("start", t_start, goal=goal.to_json(), limits=limits.to_json(),
              policy=getattr(policy, "name", "?"),
              cells=len(st.cells), span_hz=st.target_span)
    steps = refusals = in_a_row = retunes = 0
    why_stop = f"it reached its step limit ({max_steps})"
    while steps < max_steps:
        now = float(receiver.now())
        if duration_s is not None and now - t_start >= duration_s:
            why_stop = f"its time was up ({duration_s:g} s)"
            break
        if stop is not None and stop():
            why_stop = "the analyst stopped it"
            break
        st.now_s = now
        action = policy.next_action(st)
        steps += 1
        ok, why = validate_action(action, limits, st)
        if not ok:
            refusals += 1
            in_a_row += 1
            log.write(now, action, refused=why)
            st.last_action = action
            if in_a_row >= max_refusals_in_a_row:
                why_stop = (f"{in_a_row} actions in a row were refused (the last:"
                            f" {why})")
                break
            continue
        in_a_row = 0
        k, p = action.kind, action.params
        seen = None
        result = ""
        if k == "retune":
            receiver.retune(float(p["center_hz"]))
            st.center_hz = float(p["center_hz"])
            retunes += 1
        elif k == "set_span":
            receiver.set_span(float(p["span_hz"]))
            st.span_hz = float(p["span_hz"])
        elif k == "set_gain":
            receiver.set_gain(float(p["gain_db"]))
            st.gain_db = float(p["gain_db"])
        elif k == "dwell":
            t0 = float(receiver.now())
            seen = list(receiver.dwell(float(p["seconds"])) or [])
            t1 = float(receiver.now())
            obs = st.observe(seen, t0, t1)
            result = (f"{obs['matches']} match, {obs['ambiguous']} ambiguous, "
                      f"{obs['other']} other")
        elif k == "cut":
            det = st.detections[p["detection_id"]]
            st.cut_ids.add(det.id)
            fn = getattr(receiver, "cut", None)
            result = str(fn(det)) if callable(fn) else \
                "cut requested (the host's signal cut takes it from here)"
        elif k == "mark":
            det = st.detections[p["detection_id"]]
            st.marked.add(det.id)
            fn = getattr(receiver, "mark", None)
            result = str(fn(det, p.get("label", ""))) if callable(fn) else "marked"
        elif k == "skip":
            c = st.cells[int(p["cell"])]
            secs = float(p.get("seconds", math.inf))
            c.cooldown_until_s = max(c.cooldown_until_s, now + secs)
            result = "resting" if math.isfinite(secs) else "skipped"
        extra = {}
        fb = getattr(policy, "last_fallback", "")
        if fb:
            extra["fallback"] = fb
        log.write(float(receiver.now()), action, seen=seen, result=result, **extra)
        st.last_action = action
        if progress and k == "retune":
            progress(f"retune to {_hz(float(p['center_hz']))}: {action.reason}")
    t_end = float(receiver.now())
    fallbacks = int(getattr(policy, "counts", {}).get("fallback", 0)) \
        if hasattr(policy, "counts") else 0
    res = HuntResult(steps=steps, found=[asdict(f) for f in st.found],
                     retunes=retunes, refusals=refusals,
                     duration_s=t_end - t_start, stopped_because=why_stop,
                     log_path=str(log.path), fallbacks=fallbacks)
    log.event("stop", t_end, why=why_stop, finds=len(st.found),
              retunes=retunes, refusals=refusals)
    return res, st
