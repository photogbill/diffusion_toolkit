# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Transmit safety on the cabled calibration loop — the arithmetic and the
refusals, enforced by code, not by memory (plan §3.5, §4.A6 "Transmit safety
on the loop"; decision D10).

ATK's RF work is passive and receive-only. The cabled loop is the ONE
approved exception, and it must be impossible to run over the air by
mistake. Bill, 2026-10-08: *"we'd have to use either the bladeRF or HackRF as
the transmitter … careful about transmit power to avoid blowing a
receiver."*

THE RULES, each a refusal in a sentence when broken:

1. The transmitter is a bladeRF (1.0 or 2.0) or a HackRF — the families
   whose `profiles.FAMILIES[...]["transmit"]` is true. Nothing else.
2. `cabled=True` and `dc_block=True` must be CONFIRMED explicitly (the value
   `True`, not something truthy). Never over the air; the DC block keeps a
   bias tee or a DC offset off the other radio's front end.
3. The receiver's maximum safe input comes from its `ReceiverProfile.
   safe_input` — entered once FROM THE DATA SHEET with its source and the
   date. Missing, or missing its source or date: refused. Never guessed.
4. The transmitter's output power at a gain setting is entered the same way
   (`TxPower`: the power at one gain setting, with its source and date, and
   dB per gain step — 1 dB for both radios' TX gain controls).
5. Expected input = TX power at the setting − fixed attenuation − splitter
   loss (the Kraken is fed through a splitter; the loss PER OUTPUT is
   required for it, and cannot be less than the physics of an N-way split,
   10·log10 N) − cable loss.
6. Hard ceiling = the data-sheet maximum − 20 dB. Above it: refused, with the
   attenuation needed to get under it.
7. Target about −40 dBm at the receiver — linear, well inside the ADC. Far
   from it (more than 10 dB): a warning with the attenuation change that
   would reach it.
8. THE RAMP. The first run of a new setup is at the MINIMUM TX gain. Before
   the gain goes up, the receiver's own level reading (dBFS) and its
   clipping (`dsp.iq.clipped_fraction`) at the previous step must be
   consistent with the computed change: clipping at all — refused; a level
   change that disagrees with the arithmetic by more than 3 dB — refused
   (compression, or a chain that is not what the numbers say); a level not
   yet measurable above the receiver's noise — the gain may rise by at most
   6 dB per run until it is. A change to the setup (attenuators, splitter,
   cable, frequency, TX power entry) is a NEW setup and starts again at the
   minimum.

Every verdict carries its numbers into the capture's SigMF metadata
(`Verdict.metadata`): TX power, attenuation, splitter loss, cable loss and
the computed input.

BALLPARK, IN DOCUMENTATION ONLY — CONFIRM AGAINST THE DATA SHEETS, NEVER
MEMORY: the plan's own estimate is that a HackRF at its highest TX gain
wants 50–60 dB of fixed attenuation and a bladeRF 1.0 at full TX 40–50 dB.
No such number is used by the code; the code uses only what was entered,
with its source.

LIMITS. The arithmetic is only as good as the entries; that is why the ramp
checks the receiver's own reading against it before the gain goes up. TX
output varies with frequency — so the frequency is part of a setup's
identity. A cable is not a shield: the transmitter's own leakage is not
modelled.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from atk_diffusion import profiles as _profiles

#: The families that can be the loop's transmitter (plan §3.5).
TRANSMITTERS = tuple(f for f, d in _profiles.FAMILIES.items() if d.get("transmit"))

MARGIN_DB = 20.0               # under the data-sheet maximum (D10)
TARGET_DBM = -40.0             # at the receiver (D10)
TARGET_WINDOW_DB = 10.0        # warn when further than this from the target
RAMP_TOLERANCE_DB = 3.0        # measured vs computed change between steps
MEASURABLE_DB = 6.0            # above the TX-off reading, to count as measured
MAX_BLIND_STEP_DB = 6.0        # per run, while the level cannot yet be checked
MAX_CLIP_FRACTION = 1e-4       # any more at the rails is clipping
ABS_TOLERANCE_DB = 6.0         # measured dBm vs computed, when full scale is known

#: TX gain ranges the tool knows from the tool's own documentation. The
#: bladeRF's range depends on the board and libbladeRF and is entered.
KNOWN_TX_GAIN = {
    "hackrf": (0.0, 47.0, "hackrf_transfer -h: '-x gain_db  TX VGA (IF) gain, "
                          "0-47dB'"),
}


@dataclass
class TxPower:
    """The transmitter's output power at ONE gain setting, at this frequency,
    entered from the data sheet (or measured) with where it came from."""
    ref_gain: float
    ref_dbm: float
    source: str = ""
    entered: str = ""
    db_per_step: float = 1.0

    def at(self, gain: float) -> float:
        return float(self.ref_dbm) + (float(gain) - float(self.ref_gain)) \
            * float(self.db_per_step)


@dataclass
class LoopSetup:
    """One cabled-loop setup and the gain asked for this run."""
    transmitter: str                       # bladerf1 | bladerf2 | hackrf
    receiver_profile: str                  # the receiver under test
    frequency_hz: float
    tx_gain: float                         # the setting for THIS run
    tx_power: TxPower | None = None
    attenuation_db: float | None = None    # fixed attenuators in line, total
    cable_loss_db: float = 0.0
    splitter_loss_db: float | None = None  # per output (required for the Kraken)
    splitter_ways: int | None = None
    cabled: bool = False                   # operator's explicit confirmation
    dc_block: bool = False                 # operator's explicit confirmation
    tx_gain_min: float | None = None
    tx_gain_max: float | None = None
    tx_serial: str = ""
    target_dbm: float = TARGET_DBM
    note: str = ""

    def gain_range(self) -> tuple[float | None, float | None, str]:
        lo, hi, src = self.tx_gain_min, self.tx_gain_max, "entered"
        known = KNOWN_TX_GAIN.get(str(self.transmitter).lower())
        if known:
            lo = known[0] if lo is None else lo
            hi = known[1] if hi is None else hi
            src = known[2]
        return lo, hi, src

    def fingerprint(self) -> str:
        """The physical setup's identity — everything but the gain. A change
        to any of it is a NEW setup and starts again at minimum gain."""
        tp = self.tx_power
        key = {"tx": str(self.transmitter).lower(), "serial": self.tx_serial,
               "rx": str(self.receiver_profile).lower(),
               "f_khz": round(float(self.frequency_hz) / 1e3),
               "att": self.attenuation_db, "cable": self.cable_loss_db,
               "split": self.splitter_loss_db, "ways": self.splitter_ways,
               "tx_power": None if tp is None else
               [tp.ref_gain, tp.ref_dbm, tp.db_per_step]}
        return hashlib.sha256(json.dumps(key, sort_keys=True).encode()
                              ).hexdigest()[:16]

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "LoopSetup":
        d = dict(d)
        tp = d.pop("tx_power", None)
        known = set(cls.__dataclass_fields__)
        s = cls(**{k: v for k, v in d.items() if k in known})
        s.tx_power = TxPower(**tp) if isinstance(tp, dict) else tp
        return s


def expected_input_dbm(tx_power_dbm: float, attenuation_db: float,
                       splitter_loss_db: float = 0.0,
                       cable_loss_db: float = 0.0) -> float:
    """TX power − attenuation − splitter loss − cable loss (all in dB)."""
    return (float(tx_power_dbm) - float(attenuation_db)
            - float(splitter_loss_db or 0.0) - float(cable_loss_db or 0.0))


def ideal_split_loss_db(ways: int) -> float:
    """The least loss an N-way power splitter can have per output: 10·log10 N
    (physics; a real one adds its excess loss on top)."""
    return 10.0 * math.log10(max(1, int(ways)))


@dataclass
class Verdict:
    ok: bool
    refusals: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    tx_power_dbm: float | None = None
    expected_input_dbm: float | None = None
    ceiling_dbm: float | None = None
    max_safe_dbm: float | None = None
    margin_db: float | None = None          # ceiling − expected (positive = safe)
    suggest_attenuation_change_db: float | None = None   # + add, − remove
    ramp: str = ""
    setup: dict = field(default_factory=dict)

    def lines(self) -> list[str]:
        out = []
        if self.expected_input_dbm is not None:
            out.append(f"Expected input at the receiver: "
                       f"{self.expected_input_dbm:.1f} dBm (TX "
                       f"{self.tx_power_dbm:.1f} dBm less the attenuation, "
                       "splitter and cable losses).")
        if self.ceiling_dbm is not None:
            out.append(f"Ceiling: {self.ceiling_dbm:.1f} dBm — the receiver's "
                       f"data-sheet maximum ({self.max_safe_dbm:+.1f} dBm) less "
                       f"{MARGIN_DB:g} dB of margin.")
        out += [f"REFUSED: {r}" for r in self.refusals]
        out += [f"Warning: {w}" for w in self.warnings]
        if self.ramp:
            out.append(self.ramp)
        out.append("Safe to run on the cable." if self.ok else
                   "Not run. Every refusal above must be cleared first.")
        return out

    def to_json(self) -> dict:
        d = asdict(self)
        d["lines"] = self.lines()
        return d

    def metadata(self) -> dict:
        """The loop's numbers for the capture's SigMF global object."""
        s = self.setup or {}
        return {"atk:tx_power_dbm": self.tx_power_dbm,
                "atk:attenuation_db": s.get("attenuation_db"),
                "atk:splitter_loss_db": s.get("splitter_loss_db") or 0.0,
                "atk:cable_loss_db": s.get("cable_loss_db") or 0.0,
                "atk:expected_input_dbm": self.expected_input_dbm,
                "atk:receiver_max_safe_dbm": self.max_safe_dbm,
                "atk:loop_ceiling_dbm": self.ceiling_dbm,
                "atk:loop_target_dbm": s.get("target_dbm"),
                "atk:tx_family": s.get("transmitter"),
                "atk:tx_gain": s.get("tx_gain"),
                "atk:loop_frequency_hz": s.get("frequency_hz")}


def _finite(v) -> bool:
    try:
        return v is not None and not isinstance(v, bool) and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def check(setup: LoopSetup, receiver: _profiles.ReceiverProfile,
          ramp: "Ramp | None" = None) -> Verdict:
    """Every rule in the module docstring. All refusals are collected, so the
    operator sees everything wrong at once."""
    ref: list[str] = []
    warn: list[str] = []
    v = Verdict(ok=False, setup=setup.to_json())
    fam = str(setup.transmitter or "").strip().lower()
    if fam not in _profiles.FAMILIES:
        ref.append(f"'{setup.transmitter}' is not a known radio. The loop's "
                   "transmitter is a bladeRF or a HackRF.")
    elif fam not in TRANSMITTERS:
        ref.append(f"the {_profiles.FAMILIES[fam]['label']} cannot transmit. "
                   "The loop's transmitter is a bladeRF or a HackRF — the only "
                   "radios here that can.")
    if setup.cabled is not True:
        ref.append("the cable is not confirmed. This loop runs only with the "
                   "transmitter connected by cable, through the attenuators, "
                   "to the receiver — never to an antenna, never over the air. "
                   "Confirm it (cabled=True).")
    if setup.dc_block is not True:
        ref.append("the DC block is not confirmed. Put a DC block in line "
                   "between the radios and confirm it (dc_block=True).")
    if str(receiver.id).lower() != str(setup.receiver_profile).lower():
        ref.append(f"the setup names {setup.receiver_profile} but the "
                   f"receiver's record is {receiver.id}.")
    si = receiver.safe_input
    max_safe = si.max_dbm if si is not None else None
    if not _finite(max_safe):
        ref.append(f"the receiver's maximum safe input is not entered for "
                   f"{receiver.id}. Enter it once from the receiver's data "
                   "sheet, with the source and the date, in "
                   f"profiles\\{receiver.id}.json (safe_input). The loop "
                   "will not guess it.")
        max_safe = None
    elif not str(si.source or "").strip() or not str(si.entered or "").strip():
        ref.append("the receiver's maximum safe input has no "
                   + ("source" if not str(si.source or "").strip() else "date")
                   + ". It must be entered from the data sheet with where it "
                   "came from and when — a number without its source is a "
                   "number from memory.")
    tp = setup.tx_power
    if tp is None or not _finite(tp.ref_dbm) or not _finite(tp.ref_gain):
        ref.append("the transmitter's output power is not entered. Enter its "
                   "output at one gain setting, at this frequency, from its "
                   "data sheet or a measurement, with the source and the "
                   "date (TxPower).")
        tp = None
    elif not str(tp.source or "").strip() or not str(tp.entered or "").strip():
        ref.append("the transmitter's output power has no "
                   + ("source" if not str(tp.source or "").strip() else "date")
                   + ". Enter where it came from and when.")
    elif not _finite(tp.db_per_step) or float(tp.db_per_step) <= 0:
        ref.append("the transmitter's dB per gain step must be a positive "
                   "number.")
        tp = None
    gmin, gmax, gsrc = setup.gain_range()
    if not _finite(setup.tx_gain):
        ref.append("the TX gain for this run is not a number.")
    if gmin is None:
        ref.append(f"the {fam or 'transmitter'}'s minimum TX gain is not known. "
                   "Enter the TX gain range for this board as its tool "
                   "reports it — the first run of a setup is at the minimum.")
    elif _finite(setup.tx_gain) and (float(setup.tx_gain) < float(gmin)
                                     or (gmax is not None
                                         and float(setup.tx_gain) > float(gmax))):
        ref.append(f"a TX gain of {float(setup.tx_gain):g} is outside the "
                   f"transmitter's range ({gmin:g} to "
                   f"{'?' if gmax is None else f'{gmax:g}'}; {gsrc}).")
    if not _finite(setup.attenuation_db):
        ref.append("the fixed attenuation in line is not entered (total dB of "
                   "the attenuators between the radios).")
    elif float(setup.attenuation_db) < 0:
        ref.append("attenuation cannot be negative.")
    if not _finite(setup.cable_loss_db) or float(setup.cable_loss_db) < 0:
        ref.append("the cable loss must be zero or more dB.")
    rx_family = _profiles.parse_profile_id(receiver.id).family \
        if _profiles.is_profile_id(receiver.id) else ""
    split = setup.splitter_loss_db
    if rx_family == "krakensdr" and not _finite(split):
        ref.append("the KrakenSDR is fed through a splitter, and the "
                   "splitter's loss per output is not entered. Enter it from "
                   "the splitter's data sheet — it is part of the arithmetic.")
    if _finite(split):
        if float(split) < 0:
            ref.append("a splitter cannot have negative loss.")
        elif setup.splitter_ways:
            ideal = ideal_split_loss_db(setup.splitter_ways)
            if float(split) < ideal - 0.05:
                ref.append(f"a {int(setup.splitter_ways)}-way splitter loses at "
                           f"least {ideal:.1f} dB per output; {float(split):g} "
                           "dB cannot be right. Check the entry.")
    if tp is not None and _finite(setup.tx_gain) and _finite(setup.attenuation_db):
        v.tx_power_dbm = tp.at(setup.tx_gain)
        v.expected_input_dbm = expected_input_dbm(
            v.tx_power_dbm, setup.attenuation_db,
            split if _finite(split) else 0.0,
            setup.cable_loss_db if _finite(setup.cable_loss_db) else 0.0)
    if max_safe is not None:
        v.max_safe_dbm = float(max_safe)
        v.ceiling_dbm = float(max_safe) - MARGIN_DB
    e = v.expected_input_dbm
    if e is not None and v.ceiling_dbm is not None:
        v.margin_db = v.ceiling_dbm - e
        if e > v.ceiling_dbm:
            ref.append(f"at TX gain {float(setup.tx_gain):g} the receiver would "
                       f"get {e:.1f} dBm — above the ceiling of "
                       f"{v.ceiling_dbm:.1f} dBm (its data-sheet maximum "
                       f"{v.max_safe_dbm:+.1f} dBm less {MARGIN_DB:g} dB of "
                       f"margin). Add at least {e - v.ceiling_dbm:.1f} dB of "
                       "attenuation, or lower the TX gain.")
        if gmax is not None and tp is not None:
            e_max = expected_input_dbm(tp.at(gmax), setup.attenuation_db,
                                       split if _finite(split) else 0.0,
                                       setup.cable_loss_db or 0.0)
            if e_max > v.ceiling_dbm:
                warn.append(f"at the transmitter's highest gain ({gmax:g}) the "
                            f"input would be {e_max:.1f} dBm, above the "
                            "ceiling; the loop will refuse before the gain "
                            "gets there.")
    if e is not None:
        diff = e - float(setup.target_dbm)
        v.suggest_attenuation_change_db = diff
        if abs(diff) > TARGET_WINDOW_DB:
            if diff < 0:
                warn.append(f"the expected input {e:.1f} dBm is {-diff:.1f} dB "
                            f"below the {setup.target_dbm:g} dBm target: remove "
                            f"about {-diff:.0f} dB of attenuation (or raise the "
                            "TX gain through the ramp) for a strong, clean "
                            "signal.")
            else:
                warn.append(f"the expected input {e:.1f} dBm is {diff:.1f} dB "
                            f"above the {setup.target_dbm:g} dBm target: add "
                            f"about {diff:.0f} dB of attenuation.")
    if ramp is not None:
        ok, why = ramp.permit(setup)
        v.ramp = why
        if not ok:
            ref.append(why)
    v.refusals, v.warnings = ref, warn
    v.ok = not ref
    return v


# ---------------------------------------------------------------------------
# The ramp
# ---------------------------------------------------------------------------
@dataclass
class Step:
    gain: float
    expected_input_dbm: float | None
    at: str
    measured_dbfs: float | None = None
    noise_dbfs: float | None = None
    clipped_fraction: float | None = None
    rx_full_scale_dbm: float | None = None

    @property
    def measured(self) -> bool:
        return self.measured_dbfs is not None and self.clipped_fraction is not None

    @property
    def above_noise_db(self) -> float | None:
        if self.measured_dbfs is None or self.noise_dbfs is None:
            return None
        return float(self.measured_dbfs) - float(self.noise_dbfs)

    @property
    def measurable(self) -> bool:
        a = self.above_noise_db
        return a is not None and a >= MEASURABLE_DB


class Ramp:
    """The per-setup record of runs and the receiver's readings (persisted
    as JSON beside the cabled captures: `rf.cabled(profile)/loop_ramp.json`)."""

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else None
        self._d = {"setups": {}}
        if self.path is not None and self.path.exists():
            try:
                self._d = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._d = {"setups": {}}
        self._d.setdefault("setups", {})

    @classmethod
    def for_profile(cls, rf, profile: str) -> "Ramp":
        return cls(Path(rf.cabled(profile)) / "loop_ramp.json")

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._d, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def steps(self, setup: LoopSetup) -> list[Step]:
        e = self._d["setups"].get(setup.fingerprint(), {})
        return [Step(**s) for s in e.get("steps", [])]

    def _put(self, setup: LoopSetup, steps: list[Step]) -> None:
        fp = setup.fingerprint()
        s = setup.to_json()
        s.pop("tx_gain", None)
        self._d["setups"][fp] = {"setup": s, "steps": [asdict(x) for x in steps]}
        self._save()

    def reset(self, setup: LoopSetup) -> None:
        self._d["setups"].pop(setup.fingerprint(), None)
        self._save()

    def begin(self, setup: LoopSetup, expected_input_dbm: float | None) -> Step:
        """Record a run that is about to transmit (before it does, so a crash
        mid-run still leaves a step that needs its reading)."""
        st = self.steps(setup)
        step = Step(float(setup.tx_gain), expected_input_dbm,
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        st.append(step)
        self._put(setup, st)
        return step

    def record(self, setup: LoopSetup, gain: float, measured_dbfs: float,
               clipped_fraction: float, noise_dbfs: float | None = None,
               rx_full_scale_dbm: float | None = None,
               expected_input_dbm: float | None = None) -> Step:
        """Attach the receiver's reading to the newest unread run at `gain`
        (or add the run, when it was made outside `loop.execute`)."""
        st = self.steps(setup)
        target = None
        for s in reversed(st):
            if abs(s.gain - float(gain)) < 1e-9 and not s.measured:
                target = s
                break
        if target is None:
            target = Step(float(gain), expected_input_dbm,
                          time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            st.append(target)
        target.measured_dbfs = float(measured_dbfs)
        target.clipped_fraction = float(clipped_fraction)
        target.noise_dbfs = None if noise_dbfs is None else float(noise_dbfs)
        target.rx_full_scale_dbm = None if rx_full_scale_dbm is None \
            else float(rx_full_scale_dbm)
        self._put(setup, st)
        return target

    def permit(self, setup: LoopSetup) -> tuple[bool, str]:
        """(ok, words): may this setup run at `setup.tx_gain` now?"""
        gmin, _gmax, _src = setup.gain_range()
        gain = float(setup.tx_gain)
        steps = self.steps(setup)
        dps = float(setup.tx_power.db_per_step) if setup.tx_power else 1.0
        if not steps:
            if gmin is None:
                return False, ("the minimum TX gain is not known, so the first "
                               "run of this setup cannot be placed at it.")
            if abs(gain - float(gmin)) > 1e-9:
                return False, (f"this is the first run of this setup — it runs "
                               f"at the minimum TX gain ({float(gmin):g}), not "
                               f"{gain:g}. Record that run's level, then raise "
                               "the gain.")
            return True, ("Ramp: the first run of this setup, at the minimum TX "
                          "gain.")
        last = steps[-1]
        if not last.measured:
            if gain <= last.gain + 1e-9:
                return True, "Ramp: not above the last run's gain."
            return False, (f"the run at gain {last.gain:g} has no level reading "
                           "yet. Record the receiver's level and clipping for "
                           "it (loop.measure_rx) before the gain goes up.")
        meas = [s for s in steps if s.measured]
        L = meas[-1]
        if gain <= L.gain + 1e-9:
            return True, "Ramp: at or below a gain already measured — down is " \
                         "always allowed."
        if L.clipped_fraction is not None and L.clipped_fraction > MAX_CLIP_FRACTION:
            return False, (f"the receiver was clipping at gain {L.gain:g} "
                           f"({100.0 * L.clipped_fraction:.3g}% of samples at "
                           "the converter's rails). Lower the gain or add "
                           "attenuation — it does not go up from here.")
        if L.measurable and L.rx_full_scale_dbm is not None \
                and L.expected_input_dbm is not None:
            got = float(L.measured_dbfs) + float(L.rx_full_scale_dbm)
            if abs(got - float(L.expected_input_dbm)) > ABS_TOLERANCE_DB:
                return False, (f"at gain {L.gain:g} the receiver read {got:.1f} "
                               f"dBm where the arithmetic said "
                               f"{float(L.expected_input_dbm):.1f} dBm — the "
                               "chain is not what the numbers say. Check the "
                               "attenuators and the TX power entry before the "
                               "gain goes up.")
        prev = [s for s in meas[:-1] if s.measurable and s.gain < L.gain - 1e-9]
        verified = False
        if L.measurable and prev:
            P = prev[-1]
            m = float(L.measured_dbfs) - float(P.measured_dbfs)
            c = (L.gain - P.gain) * dps
            if abs(m - c) > RAMP_TOLERANCE_DB:
                how = ("— it may be compressing; " if m < c else
                       "— the chain is not what the arithmetic says; ")
                return False, (f"from gain {P.gain:g} to {L.gain:g} the "
                               f"receiver's level rose {m:.1f} dB where the "
                               f"arithmetic says {c:.1f} dB {how}check the "
                               "setup before raising the gain.")
            verified = True
        step_db = (gain - L.gain) * dps
        if not verified and step_db > MAX_BLIND_STEP_DB + 1e-9:
            if L.noise_dbfs is None:
                reason = "there is no TX-off noise reading to compare it with"
            elif not L.measurable:
                reason = (f"it was only {L.above_noise_db:.1f} dB above the "
                          f"receiver's noise (at least {MEASURABLE_DB:g} dB "
                          "counts as measured)")
            else:
                reason = "there is only one measured step to compare with"
            return False, (f"the level at gain {L.gain:g} could not yet be "
                           f"checked against the arithmetic ({reason}), so the "
                           f"gain rises at most {MAX_BLIND_STEP_DB:g} dB per run "
                           f"until it can — ask for {L.gain + MAX_BLIND_STEP_DB / dps:g}"
                           " or less.")
        if verified:
            return True, (f"Ramp: the receiver's level followed the arithmetic "
                          f"to within {RAMP_TOLERANCE_DB:g} dB; the gain may go "
                          "up while the ceiling holds.")
        return True, (f"Ramp: a step of {step_db:.1f} dB from the last measured "
                      "run, within the cautious limit.")
