# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Receiver profiles and the sample-rate law (plan §3.1, §3.3; decision D4).

Nothing learned about RF survives a change of receiver or rate. Bill,
2026-10-08: *"the bladeRF files won't work for the RTL-SDR or KrakenSDR or
HackRF files"* and *"OmniSIG only works if the sample rate is identical in
training … as in the field"*. So every capture, dataset and model is keyed by
a PROFILE, and a model is offered only for a capture whose profile matches.

    <family>_<rate>_<datatype>[_<variant>]
        rtlsdr_2400000_cu8          krakensdr_2400000_cu8_ch0
        bladerf1_4000000_ci16       hackrf_8000000_ci8

THE DATATYPE IS THE ONE ON DISK. The plan's examples wrote `rtlsdr_…_ci8`;
an RTL-SDR's samples are UNSIGNED offset-binary bytes (`cu8`), which is what
ATK's recorder has always tagged them, and a profile that names a datatype the
files do not have would be the first lie in the chain. The bladeRF's SC16 Q11
samples are `ci16` (ATK keeps the Q11 scaling in `atk:datatype`). Endianness
is not part of the name (everything ATK records is little-endian).

THE CANONICAL RATES (DETECTION_DESIGN §4, decision D2). A signal cut is
decimated to one of three rates per profile — voice-class, wideband digital,
spread/OFDM — each an INTEGER decimation of the profile's own rate, never a
fractional resample. The rule, stated so it can be checked rather than
remembered: for each class, the LOWEST exact integer-decimated rate that is at
least the class minimum (48 kHz, 480 kHz, 2 MHz) — a rate that holds the
class and no more. It reproduces the plan's bladeRF (50 k / 500 k / 2 M) and
HackRF (50 k / 500 k / 2 M) rates exactly. For the RTL at 2.4 MS/s it gives
48 k / 480 k / 2.4 M where the plan sketched 48 k / 240 k / 1.2 M — because
240 kHz cannot hold a 250 kHz LoRa chirp or a 200 kHz broadcast FM channel
with a guard, and 1.2 MHz cannot hold a 2 MHz ADS-B or BLE signal at all.

THE REFUSAL is a sentence, not an error code: *"this detector was trained for
the RTL-SDR at 2.4 MS/s; this capture is a bladeRF at 4 MS/s."*

Standard library only (ATK's recorder shares these rules; see paths.py).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: The receiver families. `adc_bits` is the converter's resolution (the
#: quantisation noise a detector learns); `datatype` what the files carry;
#: `transmit` whether the device can be the source in the cabled loop
#: (plan §3.5 — only the bladeRF and the HackRF can).
FAMILIES: dict[str, dict] = {
    "rtlsdr":    {"label": "RTL-SDR", "adc_bits": 8, "datatype": "cu8",
                  "transmit": False, "channels": 1},
    "krakensdr": {"label": "KrakenSDR", "adc_bits": 8, "datatype": "cu8",
                  "transmit": False, "channels": 5},
    "hackrf":    {"label": "HackRF One", "adc_bits": 8, "datatype": "ci8",
                  "transmit": True, "channels": 1},
    "bladerf1":  {"label": "bladeRF 1.0 (x40/x115)", "adc_bits": 12,
                  "datatype": "ci16", "transmit": True, "channels": 1},
    "bladerf2":  {"label": "bladeRF 2.0 micro", "adc_bits": 12,
                  "datatype": "ci16", "transmit": True, "channels": 1},
    "kiwisdr":   {"label": "KiwiSDR (remote)", "adc_bits": 14,
                  "datatype": "ci16", "transmit": False, "channels": 1},
    "spyserver": {"label": "SpyServer (remote)", "adc_bits": 0,
                  "datatype": "ci16", "transmit": False, "channels": 1},
    "airspy":    {"label": "Airspy", "adc_bits": 12, "datatype": "ci16",
                  "transmit": False, "channels": 1},
    "esp32csi":  {"label": "ESP32 CSI sensor", "adc_bits": 0,
                  "datatype": "cf32", "transmit": False, "channels": 1},
    "sigmf-import": {"label": "an imported SigMF file", "adc_bits": 0,
                     "datatype": "", "transmit": False, "channels": 1},
}

DATATYPES = ("cu8", "ci8", "ci16", "cf32")

#: SigMF / ATK datatype tags -> the profile's datatype.
_DT_ALIASES = {
    "cu8": "cu8", "cu8_le": "cu8",
    "ci8": "ci8", "cs8": "ci8", "ci8_le": "ci8",
    "ci16": "ci16", "ci16_le": "ci16", "ci16q11": "ci16", "cs16_le": "ci16",
    "sc16": "ci16", "sc16q11": "ci16",
    "cf32": "cf32", "cf32_le": "cf32", "cfloat32": "cf32",
}

#: ATK's `settings["sdr"]["source"]` keys -> family.
ATK_SOURCES = {"rtl-sdr": "rtlsdr", "rtlsdr": "rtlsdr", "hackrf": "hackrf",
               "kraken": "krakensdr", "krakensdr": "krakensdr",
               "spyserver": "spyserver", "kiwi": "kiwisdr",
               "kiwisdr": "kiwisdr", "airspy": "airspy"}

_PID = re.compile(r"^(?P<family>[a-z0-9][a-z0-9\-]*?)_(?P<rate>\d+)_"
                  r"(?P<dt>cu8|ci8|ci16|cf32)(?:_(?P<variant>[a-z0-9][a-z0-9\-_]*))?$")
_VARIANT = re.compile(r"^[a-z0-9][a-z0-9\-_]*$")

# ---------------------------------------------------------------------------
# Bandwidth classes and canonical rates (DETECTION_DESIGN §4, D2)
# ---------------------------------------------------------------------------
#: class -> (minimum rate that holds the class, widest occupied bandwidth the
#: class is for). A cut is placed in the narrowest class whose bandwidth limit
#: holds it.
BANDWIDTH_CLASSES = {
    "voice":    {"min_rate": 48_000.0, "max_bw": 30_000.0,
                 "label": "voice-class (narrowband FM, DMR/P25/NXDN, pagers)"},
    "wideband": {"min_rate": 480_000.0, "max_bw": 400_000.0,
                 "label": "wideband digital (LoRa, broadcast FM, wide data)"},
    "spread":   {"min_rate": 2_000_000.0, "max_bw": float("inf"),
                 "label": "spread / OFDM (LTE, Wi-Fi, BLE, ADS-B)"},
}
CLASS_ORDER = ("voice", "wideband", "spread")


@dataclass(frozen=True)
class CanonicalRate:
    cls: str            # voice | wideband | spread
    rate: float         # Hz, exact: profile rate / decimation
    decimation: int     # integer factor from the profile's own rate
    limited: bool = False   # True: the receiver's whole rate is below the
                            # class minimum, so the class runs at full rate


def canonical_rates(sample_rate: float) -> list[CanonicalRate]:
    """The three canonical rates of a profile (fewer when the receiver is
    too narrow for a class — a 12 kHz KiwiSDR has only a limited voice
    class). Pure arithmetic; a test pins the plan's examples."""
    fs = int(round(float(sample_rate)))
    if fs <= 0:
        raise ValueError("a sample rate must be positive")
    out: list[CanonicalRate] = []
    for cls in CLASS_ORDER:
        lo = BANDWIDTH_CLASSES[cls]["min_rate"]
        if fs < lo:
            if cls == "voice":
                out.append(CanonicalRate(cls, float(fs), 1, limited=True))
            continue
        best = 1
        for d in range(int(fs // lo), 0, -1):
            if fs % d == 0 and fs / d >= lo:
                best = d
                break
        out.append(CanonicalRate(cls, fs / best, best))
    return out


def bandwidth_class(bw_hz: float) -> str:
    """The narrowest class whose bandwidth limit holds `bw_hz`."""
    bw = abs(float(bw_hz))
    for cls in CLASS_ORDER:
        if bw <= BANDWIDTH_CLASSES[cls]["max_bw"]:
            return cls
    return "spread"


def canonical_for(sample_rate: float, bw_hz: float) -> CanonicalRate:
    """The canonical rate a cut of width `bw_hz` goes to. When the profile
    has no rate for the cut's class (too narrow a receiver), the widest
    class it does have is used and `limited` says so."""
    want = bandwidth_class(bw_hz)
    rates = {c.cls: c for c in canonical_rates(sample_rate)}
    if want in rates:
        return rates[want]
    for cls in reversed(CLASS_ORDER):
        if cls in rates:
            c = rates[cls]
            return CanonicalRate(c.cls, c.rate, c.decimation, limited=True)
    raise ValueError("no canonical rate")       # unreachable: voice exists


# ---------------------------------------------------------------------------
# Geometry fixed per profile (DETECTION_DESIGN §2, §4.1; D5, D7)
# ---------------------------------------------------------------------------
@dataclass
class StftGeometry:
    """The spectrogram a detector sees. Part of the profile and of every
    model card: a model never meets a spectrogram of a geometry it was not
    trained on. D7 (tile size and duration) is pending measurement; these
    are the stated starting values, written into the profile so a change is
    a recorded change."""
    fft_size: int = 1024
    hop: int = 1024             # samples between frames (no overlap by default)
    window: str = "hann"
    tile_seconds: float = 1.0
    tile_rows: int = 512        # frames are max-pooled in time to this many
    tile_overlap: float = 0.25  # fraction of a tile shared with the next

    def rbw_hz(self, sample_rate: float) -> float:
        return float(sample_rate) / float(self.fft_size)


@dataclass
class FamGeometry:
    """FFT accumulation method parameters for the cut's SCF (§4.1).
    `channel_fft` (N′) sets the spectral resolution fs/N′; `hop` is N′/4 by
    the usual choice; the α resolution follows the cut length."""
    channel_fft: int = 64
    hop: int = 16
    window: str = "hamming"
    max_seconds: float = 2.0     # longest stretch of a cut the SCF image uses


def default_stft(sample_rate: float) -> StftGeometry:
    fs = float(sample_rate)
    if fs <= 3_200_000:
        n = 1024
    elif fs <= 12_800_000:
        n = 2048
    else:
        n = 4096
    return StftGeometry(fft_size=n, hop=n)


@dataclass
class ProfileId:
    family: str
    sample_rate: int
    datatype: str
    variant: str = ""

    def __str__(self) -> str:
        return make_profile_id(self.family, self.sample_rate, self.datatype,
                               self.variant)


def normalize_datatype(dt: str) -> str:
    """SigMF/ATK datatype tag -> the profile's datatype; '' if unknown."""
    return _DT_ALIASES.get(str(dt or "").strip().lower(), "")


def make_profile_id(family: str, sample_rate, datatype: str,
                    variant: str = "") -> str:
    fam = str(family).strip().lower()
    if fam not in FAMILIES:
        raise ValueError(f"unknown receiver family {family!r} — known: "
                         f"{', '.join(sorted(FAMILIES))}")
    rate = float(sample_rate)
    if rate <= 0 or abs(rate - round(rate)) > 1e-6:
        raise ValueError(f"a profile rate must be a whole number of Hz, "
                         f"got {sample_rate!r}")
    dt = normalize_datatype(datatype)
    if not dt:
        raise ValueError(f"unknown datatype {datatype!r} — one of "
                         f"{', '.join(DATATYPES)}")
    out = f"{fam}_{int(round(rate))}_{dt}"
    var = str(variant or "").strip().lower()
    if var:
        if not _VARIANT.match(var):
            raise ValueError(f"a variant is lower-case letters, digits, '-' "
                             f"and '_': {variant!r}")
        out += f"_{var}"
    return out


def parse_profile_id(pid: str) -> ProfileId:
    m = _PID.match(str(pid).strip().lower())
    if not m or m.group("family") not in FAMILIES:
        raise ValueError(f"{pid!r} is not a receiver profile "
                         "(<family>_<rate>_<datatype>[_<variant>])")
    return ProfileId(m.group("family"), int(m.group("rate")), m.group("dt"),
                     m.group("variant") or "")


def is_profile_id(pid: str) -> bool:
    try:
        parse_profile_id(pid)
        return True
    except ValueError:
        return False


def _rate_words(rate: float) -> str:
    r = float(rate)
    if r >= 1e6:
        s = f"{r / 1e6:.3f}".rstrip("0").rstrip(".")
        return f"{s} MS/s"
    if r >= 1e3:
        s = f"{r / 1e3:.3f}".rstrip("0").rstrip(".")
        return f"{s} kS/s"
    return f"{r:g} S/s"


def describe(pid: str) -> str:
    """'the RTL-SDR at 2.4 MS/s' — for refusals and status lines."""
    p = parse_profile_id(pid)
    label = FAMILIES[p.family]["label"]
    words = f"the {label} at {_rate_words(p.sample_rate)}"
    if p.variant:
        if p.variant.startswith("ch") and p.variant[2:].isdigit():
            words += f", channel {p.variant[2:]}"
        else:
            words += f" ({p.variant})"
    return words


class ProfileMismatch(ValueError):
    """A model, dataset or tool met a capture of another profile."""


def check_match(expected: str, actual: str, what: str = "this model") -> None:
    """Raise ProfileMismatch with the plain sentence the plan asks for."""
    if str(expected).strip().lower() == str(actual).strip().lower():
        return
    try:
        a = describe(expected)
    except ValueError:
        a = repr(expected)
    try:
        b = describe(actual)
    except ValueError:
        b = repr(actual)
    raise ProfileMismatch(
        f"{what} was trained for {a}; this capture is {b}. Profiles never "
        "mix: use a model trained for this receiver at this rate, or "
        "resample the capture deliberately (a logged step) and use a model "
        "trained on resampled data.")


# ---------------------------------------------------------------------------
# Deriving a profile from what ATK or a SigMF file knows
# ---------------------------------------------------------------------------
def family_from_atk(source: str, bladerf_model: str = "x115") -> str:
    """ATK's settings["sdr"]["source"] (+ bladerf_model) -> family."""
    s = str(source or "").strip().lower()
    if s == "bladerf":
        return "bladerf2" if str(bladerf_model).lower() == "micro" else "bladerf1"
    if s in ATK_SOURCES:
        return ATK_SOURCES[s]
    raise ValueError(f"ATK source {source!r} has no receiver family")


def family_from_hw(hw: str) -> str:
    """A SigMF `core:hw` string -> family, or 'sigmf-import'."""
    h = str(hw or "").lower()
    if "kraken" in h:
        return "krakensdr"
    if "hackrf" in h:
        return "hackrf"
    if "bladerf" in h:
        return "bladerf2" if ("2.0" in h or "micro" in h or "xa" in h) \
            else "bladerf1"
    if "rtl" in h:
        return "rtlsdr"
    if "kiwi" in h:
        return "kiwisdr"
    if "spyserver" in h:
        return "spyserver"
    if "airspy" in h:
        return "airspy"
    return "sigmf-import"


_KRAKEN_CH = re.compile(r"channel\s*(\d)")


def profile_from_meta(meta: dict) -> str:
    """The profile of a SigMF capture: `atk:receiver_profile` when the
    recorder wrote one, else derived from core:hw / rate / datatype. A
    derived profile is honest about its origin only through the family —
    an unknown device becomes 'sigmf-import', never a guess."""
    g = (meta or {}).get("global", meta or {})
    pid = str(g.get("atk:receiver_profile", "") or "").strip()
    if pid and is_profile_id(pid):
        return pid
    rate = g.get("core:sample_rate")
    dt = g.get("atk:datatype") or g.get("core:datatype", "")
    if not rate:
        raise ValueError("the capture has no core:sample_rate")
    hw = str(g.get("core:hw", "") or "")
    fam = family_from_hw(hw)
    variant = ""
    if fam == "krakensdr":
        m = _KRAKEN_CH.search(hw.lower())
        if m:
            variant = f"ch{m.group(1)}"
    return make_profile_id(fam, rate, dt, variant)


# ---------------------------------------------------------------------------
# The profile record (rf_data\profiles\<profile>.json)
# ---------------------------------------------------------------------------
@dataclass
class SafeInput:
    """The receiver's maximum safe RF input, entered ONCE from the data
    sheet, with where it came from and when (plan §4.A6, D10). The cabled
    loop refuses to run without it — never filled from memory."""
    max_dbm: float | None = None
    source: str = ""            # "HackRF One data sheet rev …", a URL, a page
    entered: str = ""           # ISO date it was entered


@dataclass
class ReceiverProfile:
    id: str
    label: str = ""
    adc_bits: int = 0
    channels: int = 1
    device_serial: str = ""
    firmware: str = ""
    gain_preset: str = ""
    stft: StftGeometry = field(default_factory=StftGeometry)
    fam: FamGeometry = field(default_factory=FamGeometry)
    cfar_pfa: float = 1e-4            # per-cell false-alarm probability
    escalate_snr_db: float = 6.0      # below this, escalate to the buffer (§4.3)
    impairments: dict = field(default_factory=dict)   # measured (§3.4)
    safe_input: SafeInput = field(default_factory=SafeInput)
    created: str = ""
    updated: str = ""
    notes: list = field(default_factory=list)

    @property
    def pid(self) -> ProfileId:
        return parse_profile_id(self.id)

    @property
    def sample_rate(self) -> int:
        return self.pid.sample_rate

    @property
    def datatype(self) -> str:
        return self.pid.datatype

    def canonical_rates(self) -> list[CanonicalRate]:
        return canonical_rates(self.sample_rate)

    def to_json(self) -> dict:
        d = asdict(self)
        d["canonical_rates"] = [asdict(c) for c in self.canonical_rates()]
        d["description"] = describe(self.id)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "ReceiverProfile":
        d = dict(d)
        d.pop("canonical_rates", None)     # derived, never trusted from disk
        d.pop("description", None)
        st = StftGeometry(**(d.pop("stft", None) or {}))
        fm = FamGeometry(**(d.pop("fam", None) or {}))
        si = SafeInput(**(d.pop("safe_input", None) or {}))
        known = {f for f in cls.__dataclass_fields__}
        extra = {k: v for k, v in d.items() if k not in known}
        d = {k: v for k, v in d.items() if k in known}
        prof = cls(stft=st, fam=fm, safe_input=si, **d)
        parse_profile_id(prof.id)          # validates
        if extra:
            prof.notes = list(prof.notes) + [f"ignored unknown keys: "
                                             f"{', '.join(sorted(extra))}"]
        return prof


def new_profile(pid: str, **kw) -> ReceiverProfile:
    """A fresh profile with the family's facts and default geometry."""
    p = parse_profile_id(pid)
    fam = FAMILIES[p.family]
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    prof = ReceiverProfile(id=str(pid).lower(), label=fam["label"],
                           adc_bits=int(fam["adc_bits"]),
                           channels=int(fam["channels"]),
                           stft=default_stft(p.sample_rate),
                           created=now, updated=now)
    for k, v in kw.items():
        setattr(prof, k, v)
    return prof


def load_profile(path_or_rf, pid: str | None = None) -> ReceiverProfile:
    """Load `profiles\\<pid>.json` (pass an RfData and the id), or a path.
    A profile that has never been saved is created with defaults — it is
    the receiver's facts plus stated defaults, and saying so is the job of
    `ReceiverProfile.notes`."""
    if pid is not None:
        path = Path(path_or_rf.profile_json(pid))
    else:
        path = Path(path_or_rf)
    if not path.exists():
        if pid is None:
            raise FileNotFoundError(path)
        prof = new_profile(pid)
        prof.notes.append("defaults — no profile file yet; the impairments "
                          "are unmeasured (plan §3.4)")
        return prof
    return ReceiverProfile.from_json(json.loads(path.read_text("utf-8")))


def save_profile(rf, prof: ReceiverProfile) -> Path:
    prof.updated = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if not prof.created:
        prof.created = prof.updated
    path = Path(rf.profile_json(prof.id))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(prof.to_json(), indent=2), encoding="utf-8")
    tmp.replace(path)
    try:
        rf.record(path, "profile")
    except Exception:                                      # noqa: BLE001
        pass
    return path
