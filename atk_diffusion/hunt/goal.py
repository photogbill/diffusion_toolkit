# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The hunter's goal, from the analyst's words (plan §4.B6).

Bill: *"I definitely want to do"* the self-hunting receiver — *"anything
narrowband and bursty between 400 and 470"*. That sentence is the interface,
so it must be read the same way every time and say how it was read.

RULES FIRST. `parse_goal` reads units and ranges (Hz, kHz, MHz, GHz;
"between … and …", "from … to …", "400-470", "above", "below", "around"),
bandwidth phrases ("narrower than 25 kHz", "at least 200 kHz wide"),
burstiness words (bursty, intermittent, key-up, push-to-talk; continuous,
steady), the detector's families (FM, AM, FSK, PSK/QAM, OFDM, spread) and the
class table's names (DMR, P25, pagers, LoRa, ADS-B …). Every assumption it
makes is written into `Goal.notes` in words — a bare "between 400 and 470" is
read as MHz and the note says so, because a reading the analyst cannot see is
a reading the analyst cannot correct.

THEN, OPTIONALLY, THE MODEL. An `llm(prompt) -> text` callable (ATK's
cognitive core) may read the sentence too and return JSON. Its answer is
VALIDATED against a schema (types, ranges, the family and class tables) and
anything invalid is discarded in favour of the rules, with the reason in the
notes. Two further rules keep a fluent wrong answer out:

* the FREQUENCIES THE ANALYST TYPED are read by the rules, never re-read by
  the model — the model may only supply a band when no numbers were typed
  (e.g. "pagers" → its band), and the note says the band came from it;
* the model can add families and classes only from the class table.

The prompt ends with the explicit way out (Bill's rule for small models):
*say "I don't know"*. A model that takes it changes nothing.

`Goal.matches(detection, look)` is the hunter's test of what it saw:
True (a match), False, or None — AMBIGUOUS, e.g. a narrowband signal that
filled the whole look, so whether it is a burst cannot be told yet. The
policy answers ambiguity by dwelling longer (policy.py).

LIMITS. The class-implied bands (ADS-B at 1090 MHz, pagers at 929–932 MHz,
LoRa at 902–928 MHz, …) are the US band plan, where ATK is used today; a
region elsewhere needs its own (plan §3.6). The rules read English.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Callable

from atk_diffusion import profiles as _profiles
from atk_diffusion.detect import classes as _classes

#: A detection no longer than this, seen to start or stop, is a burst.
BURST_MAX_S = 5.0
#: Highest frequency any receiver family here tunes (HackRF / bladeRF 2.0).
F_MAX_HZ = 6.0e9
F_MIN_HZ = 1.0e3
#: "around X" means X ± this fraction of X (at least AROUND_MIN_HZ).
AROUND_FRACTION = 0.005
AROUND_MIN_HZ = 50_000.0

#: Narrowband / wideband by the detector's own bandwidth classes.
NARROW_MAX_HZ = _profiles.BANDWIDTH_CLASSES["voice"]["max_bw"]       # 30 kHz
WIDE_MIN_HZ = NARROW_MAX_HZ
SPREAD_MIN_HZ = _profiles.BANDWIDTH_CLASSES["wideband"]["max_bw"]    # 400 kHz

#: Where a class lives when no band was typed (US band plan — stated in the
#: goal's notes whenever it is used).
CLASS_BANDS = {
    "adsb": (1_089e6, 1_091e6, "ADS-B is at 1090 MHz"),
    "noaa_wx": (162.375e6, 162.575e6, "NOAA weather radio is at 162.400–162.550 MHz"),
    "lora": (902e6, 928e6, "US LoRa is in the 902–928 MHz ISM band"),
    "fm_broadcast": (88e6, 108e6, "broadcast FM is 88–108 MHz"),
    "pocsag": (929e6, 932e6, "US paging is at 929–932 MHz"),
    "flex": (929e6, 932e6, "US paging is at 929–932 MHz"),
    "ble": (2_400e6, 2_483.5e6, "Bluetooth LE is in the 2.4 GHz ISM band"),
    "wifi_24": (2_400e6, 2_483.5e6, "2.4 GHz Wi-Fi is in the 2.4 GHz ISM band"),
    "gnss_jamming": (1_574.42e6, 1_576.42e6, "GPS L1 is at 1575.42 MHz"),
    "atsc": (470e6, 608e6, "US UHF television is 470–608 MHz"),
    "drone_fpv_analog": (5_650e6, 5_925e6, "analog FPV video is at 5.65–5.925 GHz"),
}

#: Words -> class names (beyond the table's own names and labels).
CLASS_WORDS = {
    "pager": ("pocsag", "flex"), "pagers": ("pocsag", "flex"),
    "paging": ("pocsag", "flex"),
    "digital voice": ("dmr", "p25", "nxdn96", "nxdn48"),
    "nxdn": ("nxdn96", "nxdn48"), "lte": ("lte_dl",), "5g": ("nr_dl",),
    "wifi": ("wifi_24",), "wi-fi": ("wifi_24",), "bluetooth": ("ble",),
    "drone": ("drone_fpv_analog", "drone_digital"),
    "drones": ("drone_fpv_analog", "drone_digital"),
    "gps jamming": ("gnss_jamming",), "gnss jamming": ("gnss_jamming",),
    "jammer": ("gnss_jamming",), "weather radio": ("noaa_wx",),
    "noaa": ("noaa_wx",), "ads-b": ("adsb",), "adsb": ("adsb",),
    "broadcast fm": ("fm_broadcast",), "fm broadcast": ("fm_broadcast",),
    "lora": ("lora",), "dmr": ("dmr",), "p25": ("p25",), "pocsag": ("pocsag",),
    "flex": ("flex",), "atsc": ("atsc",), "ble": ("ble",),
}

FAMILY_WORDS = {
    "fm": "fm", "nfm": "fm", "am": "am", "fsk": "fsk", "gfsk": "fsk",
    "4fsk": "fsk", "psk": "psk_qam", "qam": "psk_qam", "bpsk": "psk_qam",
    "qpsk": "psk_qam", "ofdm": "ofdm", "spread": "spread", "chirp": "spread",
    "chirps": "spread", "spread spectrum": "spread",
}

BURSTY_WORDS = ("bursty", "burst", "bursts", "intermittent", "key-up",
                "keyup", "key up", "push-to-talk", "push to talk", "ptt",
                "transient", "short", "occasional", "keying")
CONTINUOUS_WORDS = ("continuous", "steady", "always on", "always-on",
                    "constant", "carrier", "persistent")


@dataclass
class Goal:
    """What to hunt for, in numbers, with how the words were read."""
    text: str = ""
    f_lo_hz: float | None = None
    f_hi_hz: float | None = None
    bw_min_hz: float | None = None
    bw_max_hz: float | None = None
    burstiness: str = "any"               # bursty | continuous | any
    families: tuple = ()
    classes: tuple = ()
    prefer_weak: bool = False             # dwell longer; escalate (§4.3)
    novel: bool = False                   # prefer what the detector calls UNKNOWN
    source: str = "rules"                 # rules | llm+rules
    notes: list = field(default_factory=list)

    # -- checks -----------------------------------------------------------------
    def problems(self) -> list[str]:
        """Why the hunter cannot start on this goal, in words. Empty = fine."""
        out = []
        if self.f_lo_hz is None or self.f_hi_hz is None:
            out.append("the goal gives no frequency range — say where to look "
                       "(for example 'between 400 and 470 MHz')")
        elif not (self.f_hi_hz > self.f_lo_hz > 0):
            out.append("the goal's frequency range is empty or reversed")
        if (self.bw_min_hz is not None and self.bw_max_hz is not None
                and self.bw_min_hz > self.bw_max_hz):
            out.append("the goal asks for a bandwidth both wider and narrower "
                       "than it can be")
        return out

    def describe(self) -> str:
        parts = []
        what = "anything"
        if self.classes:
            what = " or ".join(_class_label(c) for c in self.classes)
        elif self.families:
            what = "anything " + " or ".join(_classes.FAMILY_WORDS.get(f, f)
                                             for f in self.families)
        parts.append(what)
        bw = []
        if self.bw_max_hz is not None:
            bw.append(f"no wider than {_hz(self.bw_max_hz)}")
        if self.bw_min_hz is not None:
            bw.append(f"at least {_hz(self.bw_min_hz)} wide")
        if bw:
            parts.append(" and ".join(bw))
        if self.burstiness != "any":
            parts.append(self.burstiness)
        if self.novel:
            parts.append("that the detector cannot name")
        s = ", ".join(parts)
        if self.f_lo_hz is not None and self.f_hi_hz is not None:
            s += f", between {_hz(self.f_lo_hz)} and {_hz(self.f_hi_hz)}"
        return s

    def to_json(self) -> dict:
        d = asdict(self)
        d["families"], d["classes"] = list(self.families), list(self.classes)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Goal":
        known = set(cls.__dataclass_fields__)
        kw = {k: v for k, v in (d or {}).items() if k in known}
        kw["families"] = tuple(kw.get("families", ()) or ())
        kw["classes"] = tuple(kw.get("classes", ()) or ())
        return cls(**kw)

    # -- the hunter's test ----------------------------------------------------
    def matches(self, det, look: tuple | None = None) -> tuple:
        """(True | False | None, why). None = AMBIGUOUS — the policy dwells
        longer. `look` is the (t0, t1) of the dwell the detection came from."""
        f_c = 0.5 * (float(det.f_lo) + float(det.f_hi))
        bw = float(det.f_hi) - float(det.f_lo)
        if (self.f_lo_hz is not None and f_c < self.f_lo_hz) or \
                (self.f_hi_hz is not None and f_c > self.f_hi_hz):
            return False, f"{_hz(f_c)} is outside the goal band"
        if self.bw_max_hz is not None and bw > self.bw_max_hz:
            return False, f"{_hz(bw)} wide — wider than {_hz(self.bw_max_hz)}"
        if self.bw_min_hz is not None and bw < self.bw_min_hz:
            return False, f"{_hz(bw)} wide — narrower than {_hz(self.bw_min_hz)}"
        cls = str(getattr(det, "cls", "") or "")
        fam = str(getattr(det, "family", "unknown") or "unknown")
        unnamed = cls in ("", _classes.UNKNOWN)
        if self.novel and not unnamed:
            return False, f"the detector already names it ({cls})"
        ambiguous = []
        if self.classes and cls not in self.classes:
            if not unnamed:
                return False, f"it is {cls}, not {'/'.join(self.classes)}"
            want_fams = {_classes.get(c).family for c in self.classes
                         if _classes.get(c)}
            if fam != "unknown" and want_fams and fam not in want_fams:
                return False, f"its family is {fam}"
            ambiguous.append("its class is not known yet")
        if self.families and fam not in self.families:
            if fam != "unknown":
                return False, f"its family is {fam}, not {'/'.join(self.families)}"
            ambiguous.append("its family is not known yet")
        if self.burstiness != "any":
            b = burst_evidence(det, look)
            if b is None:
                ambiguous.append("it filled the whole look, so whether it is "
                                 "a burst cannot be told yet")
            elif (b and self.burstiness == "continuous") or \
                    (not b and self.burstiness == "bursty"):
                return False, ("it is a burst" if b else "it is continuous")
        if ambiguous:
            return None, "; ".join(ambiguous)
        return True, "matches the goal: " + _det_words(det)


def burst_evidence(det, look: tuple | None) -> bool | None:
    """True: a burst (a track duty under 0.5, or it started or stopped inside
    the look); False: continuous (seen for longer than BURST_MAX_S); None:
    it filled a short look and cannot be told yet."""
    duty = (getattr(det, "measurements", None) or {}).get("duty")
    if duty is not None:
        try:
            return float(duty) < 0.5
        except (TypeError, ValueError):
            pass
    dur = float(det.t1) - float(det.t0)
    if look is None:
        return True if dur <= BURST_MAX_S else False
    l0, l1 = float(look[0]), float(look[1])
    eps = max(1e-3, 0.02 * (l1 - l0))
    inside = (float(det.t0) > l0 + eps) or (float(det.t1) < l1 - eps)
    if inside:
        return True
    return False if dur > BURST_MAX_S else None


def _class_label(name: str) -> str:
    c = _classes.get(name)
    return c.label if c else name


def _hz(f) -> str:
    v = float(f)
    for scale, unit in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz")):
        if abs(v) >= scale:
            return f"{v / scale:.6f}".rstrip("0").rstrip(".") + f" {unit}"
    return f"{v:g} Hz"


def _det_words(det) -> str:
    bw = float(det.f_hi) - float(det.f_lo)
    c = 0.5 * (float(det.f_lo) + float(det.f_hi))
    name = str(getattr(det, "cls", "") or "") or \
        _classes.FAMILY_WORDS.get(getattr(det, "family", "unknown"), "a signal")
    return f"{name} {_hz(bw)} wide at {_hz(c)}"


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
_UNIT = r"(ghz|mhz|khz|hz|g|m|k)?"
_NUM = r"(\d+(?:\.\d+)?)"
#: A number that is a rate, a time or a level is not a frequency.
_TAIL = (r"(?![\w.])(?!\s*(?:baud|bps|sym|symbols|sps|sec|secs|seconds|s\b|ms\b"
         r"|min\b|minutes|db|dbm|percent|%))")
_SCALE = {"ghz": 1e9, "g": 1e9, "mhz": 1e6, "m": 1e6, "khz": 1e3, "k": 1e3,
          "hz": 1.0}


def _unit_scale(u: str | None) -> float | None:
    return _SCALE.get((u or "").lower()) if u else None


def _guess_scale(values: list[float], notes: list, what: str) -> float:
    """Unitless numbers: MHz (what an analyst means by 'between 400 and
    470'); kHz when MHz would be above every receiver here and kHz lands in
    HF; Hz for numbers of a hundred thousand and more. Always noted."""
    if all(v >= 1e5 for v in values):
        notes.append(f"no unit was given for {what}; numbers that large are "
                     "read as Hz")
        return 1.0
    if all(v * 1e6 <= F_MAX_HZ for v in values):
        notes.append(f"no unit was given for {what}; read as MHz")
        return 1e6
    if all(v * 1e3 <= 30e6 for v in values):
        notes.append(f"no unit was given for {what}; as MHz it would be above "
                     "every receiver here, so it is read as kHz (HF)")
        return 1e3
    notes.append(f"no unit was given for {what}; read as MHz")
    return 1e6


_BW_PATTERNS = (
    # narrower than / less than / under ... wide
    (re.compile(r"(?:narrower than|thinner than|no wider than|not wider than)\s*"
                + _NUM + r"\s*" + _UNIT + r"\b"), "max"),
    (re.compile(r"(?:wider than|broader than|no narrower than)\s*"
                + _NUM + r"\s*" + _UNIT + r"\b"), "min"),
    (re.compile(r"(?:under|below|less than|at most|up to|<=?)\s*" + _NUM
                + r"\s*" + _UNIT + r"\s*(?:wide|of bandwidth|bandwidth|bw)\b"),
     "max"),
    (re.compile(r"(?:over|above|more than|at least|>=?)\s*" + _NUM + r"\s*"
                + _UNIT + r"\s*(?:wide|of bandwidth|bandwidth|bw)\b"), "min"),
    (re.compile(r"(?:bandwidth|bw)\s*(?:under|below|less than|of at most|<=?)\s*"
                + _NUM + r"\s*" + _UNIT + r"\b"), "max"),
    (re.compile(r"(?:bandwidth|bw)\s*(?:over|above|more than|of at least|>=?)\s*"
                + _NUM + r"\s*" + _UNIT + r"\b"), "min"),
    (re.compile(_NUM + r"\s*" + _UNIT + r"\s*(?:wide|bandwidth|bw)\b"), "about"),
)

_RANGE_PATTERNS = (
    re.compile(r"between\s*" + _NUM + r"\s*" + _UNIT + r"\s*(?:and|-|to)\s*"
               + _NUM + r"\s*" + _UNIT + _TAIL),
    re.compile(r"from\s*" + _NUM + r"\s*" + _UNIT + r"\s*(?:to|-|through|thru"
               r"|until)\s*" + _NUM + r"\s*" + _UNIT + _TAIL),
    re.compile(r"(?<![\w.])" + _NUM + r"\s*" + _UNIT + r"\s*(?:-|to|through|thru)"
               r"\s*" + _NUM + r"\s*" + _UNIT + _TAIL),
)
_ABOVE = re.compile(r"(?:above|over|higher than|greater than|>)\s*" + _NUM
                    + r"\s*" + _UNIT + _TAIL)
_BELOW = re.compile(r"(?:below|under|lower than|less than|<)\s*" + _NUM
                    + r"\s*" + _UNIT + _TAIL)
_AROUND = re.compile(r"(?:around|near|about|at|on|close to)\s*" + _NUM
                     + r"\s*" + _UNIT + _TAIL)
_BARE = re.compile(r"(?<![\w.])" + _NUM + r"\s*(ghz|mhz|khz|hz)\b")


def _words_in(text: str, words) -> bool:
    for w in words:
        if re.search(r"(?<![\w-])" + re.escape(w) + r"(?![\w-])", text):
            return True
    return False


def parse_rules(text: str) -> Goal:
    """The rules' reading of the analyst's sentence (see the module doc)."""
    raw = str(text or "")
    t = raw.lower().replace("–", "-").replace("—", "-").replace(",", " ")
    t = re.sub(r"\s+", " ", t)
    g = Goal(text=raw)
    notes = g.notes
    explicit_band = False

    # -- bandwidth phrases first, removed before frequencies are read ---------
    for pat, kind in _BW_PATTERNS:
        for m in list(pat.finditer(t)):
            v = float(m.group(1))
            sc = _unit_scale(m.group(2))
            if sc is None:
                sc = 1e3
                notes.append(f"no unit was given for the bandwidth {m.group(1)}; "
                             "read as kHz")
            bw = v * sc
            if kind == "max":
                g.bw_max_hz = bw if g.bw_max_hz is None else min(g.bw_max_hz, bw)
            elif kind == "min":
                g.bw_min_hz = bw if g.bw_min_hz is None else max(g.bw_min_hz, bw)
            else:
                g.bw_min_hz, g.bw_max_hz = 0.5 * bw, 2.0 * bw
                notes.append(f"'{m.group(0).strip()}' is read as {_hz(0.5 * bw)}"
                             f" to {_hz(2 * bw)} — bandwidth estimates vary by "
                             "a factor of two between methods")
            t = t[:m.start()] + " " * (m.end() - m.start()) + t[m.end():]

    # -- frequency ranges --------------------------------------------------------
    for pat in _RANGE_PATTERNS:
        m = pat.search(t)
        if not m:
            continue
        a, ua, b, ub = float(m.group(1)), m.group(2), float(m.group(3)), m.group(4)
        sa, sb = _unit_scale(ua), _unit_scale(ub)
        if sa is None and sb is None:
            sa = sb = _guess_scale([a, b], notes, f"'{m.group(0).strip()}'")
        sa = sa if sa is not None else sb
        sb = sb if sb is not None else sa
        lo, hi = sorted((a * sa, b * sb))
        g.f_lo_hz, g.f_hi_hz = lo, hi
        explicit_band = True
        t = t[:m.start()] + " " * (m.end() - m.start()) + t[m.end():]
        break
    if not explicit_band:
        ma, mb = _ABOVE.search(t), _BELOW.search(t)
        if ma or mb:
            vals = [float(x.group(1)) for x in (ma, mb) if x]
            units = [_unit_scale(x.group(2)) for x in (ma, mb) if x]
            known = [u for u in units if u is not None]
            guess = known[0] if known else _guess_scale(vals, notes, "the band edge")
            if ma:
                g.f_lo_hz = float(ma.group(1)) * (_unit_scale(ma.group(2)) or guess)
            if mb:
                g.f_hi_hz = float(mb.group(1)) * (_unit_scale(mb.group(2)) or guess)
            explicit_band = True
            if g.f_lo_hz is None:
                g.f_lo_hz = F_MIN_HZ
                notes.append("no lower edge was given; the hunt starts at the "
                             "bottom of the receiver's range")
            if g.f_hi_hz is None:
                g.f_hi_hz = F_MAX_HZ
                notes.append("no upper edge was given; the hunt runs to the top "
                             "of the receiver's range — a wide hunt is a slow one")
        else:
            m = _AROUND.search(t) or _BARE.search(t)
            if m:
                v = float(m.group(1))
                sc = _unit_scale(m.group(2)) or _guess_scale([v], notes,
                                                             f"'{m.group(0).strip()}'")
                c = v * sc
                half = max(AROUND_FRACTION * c, AROUND_MIN_HZ)
                g.f_lo_hz, g.f_hi_hz = c - half, c + half
                explicit_band = True
                notes.append(f"'{m.group(0).strip()}' is read as "
                             f"{_hz(c - half)} to {_hz(c + half)}")

    # -- words -------------------------------------------------------------------
    if _words_in(t, ("narrowband", "narrow-band", "narrow band", "narrow")):
        if g.bw_max_hz is None:
            g.bw_max_hz = NARROW_MAX_HZ
            notes.append(f"'narrowband' is read as no wider than "
                         f"{_hz(NARROW_MAX_HZ)} (the detector's voice class)")
    if _words_in(t, ("wideband", "wide-band", "wide band", "broadband")):
        if g.bw_min_hz is None:
            g.bw_min_hz = WIDE_MIN_HZ
            notes.append(f"'wideband' is read as wider than {_hz(WIDE_MIN_HZ)}")
    bursty = _words_in(t, BURSTY_WORDS)
    steady = _words_in(t, CONTINUOUS_WORDS)
    if bursty and not steady:
        g.burstiness = "bursty"
    elif steady and not bursty:
        g.burstiness = "continuous"
    elif bursty and steady:
        notes.append("both 'bursty' and 'continuous' words were used; "
                     "either is accepted")
    fams = []
    for w, fam in FAMILY_WORDS.items():
        if w in ("am", "fm"):
            # 'am' is also English ("I am looking for …"): only the upper-case
            # abbreviation or the spelled-out modulation counts
            spelled = ("amplitude modulat" if w == "am" else "frequency modulat")
            hit = (re.search(r"(?<![\w-])" + w.upper() + r"(?![\w])", raw)
                   or spelled in t or f"{w}-like" in t)
        else:
            hit = _words_in(t, (w,))
        if hit and fam not in fams:
            fams.append(fam)
    cls = []
    for w, names in CLASS_WORDS.items():
        if _words_in(t, (w,)):
            cls += [n for n in names if n not in cls]
    for c in _classes.CLASSES:
        if not c.negative and _words_in(t, (c.name,)) and c.name not in cls:
            cls.append(c.name)
    # a family word that is also a class's own word (fm in 'fm broadcast')
    # is not a family filter when the class was named
    if cls:
        fams = [f for f in fams if not any(
            _classes.get(c) and _classes.get(c).family == f for c in cls)]
    g.families, g.classes = tuple(fams), tuple(cls)
    if _words_in(t, ("weak", "faint", "low snr", "quiet", "below the noise")):
        g.prefer_weak = True
    if _words_in(t, ("unknown", "unidentified", "unfamiliar", "unrecognised",
                     "unrecognized", "unnamed")):
        g.novel = True
    if not explicit_band and g.classes:
        bands = [CLASS_BANDS[c] for c in g.classes if c in CLASS_BANDS]
        if bands and len(bands) == len(g.classes):
            g.f_lo_hz = min(b[0] for b in bands)
            g.f_hi_hz = max(b[1] for b in bands)
            notes.append("no band was given; " + "; ".join(sorted({b[2] for b in bands}))
                         + " (US band plan) — looking there")
    return g


# ---------------------------------------------------------------------------
# JSON from a model's reply (ATK's jsonish rule: the LAST balanced value)
# ---------------------------------------------------------------------------
def last_json(text: str, want=dict):
    """The LAST parseable balanced JSON value of `want`'s type in `text`, or
    None. String-aware; the last, because a model that thinks out loud
    restates the example shape first and answers last (ported from ATK's
    `atk/core/jsonish.py`)."""
    opener = "{" if want is dict else "["
    closer = "}" if want is dict else "]"
    s = str(text or "")
    depth = start = 0
    in_str = esc = False
    found = None
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            if depth == 0:
                start = i
            depth += 1
        elif ch == closer and depth:
            depth -= 1
            if depth == 0:
                try:
                    v = json.loads(s[start:i + 1])
                except json.JSONDecodeError:
                    continue
                if isinstance(v, want):
                    found = v
    return found


GOAL_SCHEMA = {
    "f_lo_hz": "number or null — lowest frequency in Hz",
    "f_hi_hz": "number or null — highest frequency in Hz",
    "bw_min_hz": "number or null — narrowest bandwidth in Hz",
    "bw_max_hz": "number or null — widest bandwidth in Hz",
    "burstiness": '"bursty" | "continuous" | "any"',
    "families": "list, from: " + ", ".join(f for f in _classes.FAMILIES
                                            if f != "unknown"),
    "classes": "list, from the class names given",
}


def goal_prompt(text: str) -> str:
    """The prompt for the model's reading. Ends with the way out."""
    names = ", ".join(_classes.names(include_negatives=False))
    return (
        "An analyst told a radio receiver what to hunt for. Turn the request "
        "into JSON with exactly these keys:\n"
        + json.dumps(GOAL_SCHEMA, indent=1) + "\n"
        f"Class names you may use: {names}.\n"
        "Use null for anything the request does not say. Frequencies and "
        "bandwidths are in Hz. Do not add anything the request does not ask "
        "for.\n"
        f"REQUEST: {text}\n"
        "Reply with the JSON object only. If you cannot tell what the request "
        "asks for, say \"I don't know\".")


def _num_or_none(v, what: str) -> float | None:
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{what} is not a number")
    if not math.isfinite(float(v)):
        raise ValueError(f"{what} is not finite")
    return float(v)


def validate_goal_json(obj) -> tuple[dict | None, str]:
    """(fields, '') for a valid reading, (None, why) otherwise."""
    if not isinstance(obj, dict):
        return None, "the model gave no JSON object"
    if obj.get("unknown") is True:
        return None, "the model said it could not tell"
    try:
        lo = _num_or_none(obj.get("f_lo_hz"), "f_lo_hz")
        hi = _num_or_none(obj.get("f_hi_hz"), "f_hi_hz")
        bmin = _num_or_none(obj.get("bw_min_hz"), "bw_min_hz")
        bmax = _num_or_none(obj.get("bw_max_hz"), "bw_max_hz")
    except ValueError as exc:
        return None, str(exc)
    for v, what in ((lo, "f_lo_hz"), (hi, "f_hi_hz")):
        if v is not None and not (F_MIN_HZ <= v <= F_MAX_HZ):
            return None, f"{what} {v:g} Hz is outside 1 kHz – 6 GHz"
    if (lo is None) != (hi is None):
        return None, "the model gave only one edge of the band"
    if lo is not None and not hi > lo:
        return None, "the model's band is empty or reversed"
    for v, what in ((bmin, "bw_min_hz"), (bmax, "bw_max_hz")):
        if v is not None and not (0 < v <= 100e6):
            return None, f"{what} {v:g} Hz is not a plausible bandwidth"
    if bmin is not None and bmax is not None and bmin > bmax:
        return None, "the model's bandwidth bounds are reversed"
    burst = obj.get("burstiness", "any")
    burst = "any" if burst is None else str(burst).strip().lower()
    if burst not in ("bursty", "continuous", "any"):
        return None, f"burstiness {burst!r} is not bursty, continuous or any"
    fams = obj.get("families") or []
    cls = obj.get("classes") or []
    if not isinstance(fams, list) or not isinstance(cls, list):
        return None, "families and classes must be lists"
    bad_f = [f for f in fams if f not in _classes.FAMILIES or f == "unknown"]
    if bad_f:
        return None, f"unknown families {bad_f}"
    valid = set(_classes.names(include_negatives=False))
    bad_c = [c for c in cls if c not in valid]
    if bad_c:
        return None, f"classes not in the class table: {bad_c}"
    return {"f_lo_hz": lo, "f_hi_hz": hi, "bw_min_hz": bmin, "bw_max_hz": bmax,
            "burstiness": burst, "families": tuple(fams),
            "classes": tuple(cls)}, ""


_DONT_KNOW = re.compile(r"\bi\s+(do\s+not|don'?t)\s+know\b", re.I)


def parse_goal(text: str, llm: Callable[[str], str] | None = None) -> Goal:
    """The goal from the analyst's words: rules, then (optionally) the model,
    validated, with the rules' typed numbers always winning."""
    g = parse_rules(text)
    if llm is None:
        return g
    try:
        reply = llm(goal_prompt(text))
    except Exception as exc:                               # noqa: BLE001
        g.notes.append(f"the model could not be asked ({type(exc).__name__}); "
                       "the rules' reading is used")
        return g
    reply = str(reply or "")
    obj = last_json(reply)
    if obj is None and _DONT_KNOW.search(reply.replace("’", "'")):
        g.notes.append("the model said it did not know; the rules' reading "
                       "is used")
        return g
    fields, why = validate_goal_json(obj)
    if fields is None:
        g.notes.append(f"the model's reading was not used ({why}); the rules' "
                       "reading is used")
        return g
    typed = bool(re.search(r"\d", str(text or "")))
    if g.f_lo_hz is None and fields["f_lo_hz"] is not None:
        if typed:
            g.notes.append("the model offered a band, but numbers were typed "
                           "and the rules could not read them — say the band "
                           "with units")
        else:
            g.f_lo_hz, g.f_hi_hz = fields["f_lo_hz"], fields["f_hi_hz"]
            g.notes.append(f"the band {_hz(g.f_lo_hz)}–{_hz(g.f_hi_hz)} is the "
                           "model's reading (no frequencies were typed)")
    if g.bw_min_hz is None and g.bw_max_hz is None:
        g.bw_min_hz, g.bw_max_hz = fields["bw_min_hz"], fields["bw_max_hz"]
    if g.burstiness == "any" and fields["burstiness"] != "any":
        g.burstiness = fields["burstiness"]
    g.families = tuple(dict.fromkeys(tuple(g.families) + fields["families"]))
    g.classes = tuple(dict.fromkeys(tuple(g.classes) + fields["classes"]))
    g.source = "llm+rules"
    return g
