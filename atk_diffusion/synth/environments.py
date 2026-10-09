# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Environment profiles: what a region's spectrum looks like, without a
capture from it (plan §3.6, decision D8).

Bill, 2026-10-08: *"generative AI can be used to make realistic signal samples
for training. That way you could train without going out to that exact region
with your equipment and loitering when you don't need to."* Receiver and rate
say WHAT HEARS; an environment profile says WHERE: the ITU region and the
national allocation table, the cellular downlink bands, the broadcast
services, the terrain class and the channel model that goes with it, and the
interference population. The scene composer (`synth.scene`) lays signals into
a receiver's span according to it.

A PRIOR, NOT THE PLACE. Every number here — occupancy, SNR ranges, which
classes live in which band — is a starting belief, written down so it can be
scored: the domain gap between a detector trained on a composed region and the
first minutes on site (plan §7, *minutes to acceptable*) is the number that
says how good the prior was. The `caveat` field says so on every profile.

WHAT IS AND IS NOT IN THE BUILT-IN PROFILE. `builtin("us-va-nokesville")` —
Bill's own area — is built from well-known US allocations (47 CFR §2.106 and
the service rules cited per entry) and from ATK's own LTE and NR band tables
(`atk/core/lte_bands.py`, 3GPP TS 36.101 Table 5.5-1; `atk/core/nr_bands.py`,
TS 38.104 Table 5.2-1 — their US downlink rows are copied below, and a test
holds them to ATK's files when those are present). It names NO operator, NO
station, NO tower and NO channel assignment: which FM stations are on air,
which LTE carriers an operator runs where, which NOAA channel serves the
county — none of that is asserted. The composer draws carriers and channels
inside the allocations at the stated occupancy; the real ones are learned on
site.

Stored as `rf_data\\environments\\<region>.json` (plan §3.6), written through
the rf write log.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from atk_diffusion.detect import classes as _classes

M = 1e6
CAVEAT = ("a prior, not the place: occupancy, SNR and class mix are beliefs "
          "to be scored against the first minutes on site (plan §3.6, §7)")

#: Terrain class -> channel model (representative values in the style of
#: COST 207 / ITU-R M.1225; a prior, not a measurement). K-factor None is
#: Rayleigh (no line of sight); rms delay spread sets the exponential power
#: delay profile; a second cluster models hilly terrain's late echoes.
TERRAIN_CHANNELS: dict[str, dict] = {
    "urban":    {"k_factor_db": None, "rms_delay_s": 1.0e-6,
                 "source": "COST 207 Typical Urban (~1 µs rms)"},
    "suburban": {"k_factor_db": 3.0, "rms_delay_s": 0.5e-6,
                 "source": "between COST 207 TU and RA"},
    "rural":    {"k_factor_db": 6.0, "rms_delay_s": 0.1e-6,
                 "source": "COST 207 Rural Area (0.1 µs rms), Rician"},
    "mountain": {"k_factor_db": None, "rms_delay_s": 3.0e-6,
                 "second_cluster_s": 15e-6, "second_cluster_db": -10.0,
                 "source": "COST 207 Hilly Terrain (two clusters)"},
    "coastal":  {"k_factor_db": 9.0, "rms_delay_s": 0.05e-6,
                 "source": "open water: strong line of sight"},
    "open":     {"k_factor_db": 12.0, "rms_delay_s": 0.02e-6,
                 "source": "open terrain, strong line of sight"},
}

#: Mobility -> speed range (m/s) for the Doppler of an emitter; a static
#: emitter still fades slowly with moving scatterers.
MOBILITY_SPEED = {"static": (0.0, 0.5), "pedestrian": (0.5, 2.0),
                  "mobile": (0.0, 30.0), "aircraft": (100.0, 250.0),
                  "drone": (0.0, 20.0)}


class EnvironmentError(ValueError):
    """An environment profile is malformed or missing; the message says why."""


@dataclass
class Allocation:
    f_lo_hz: float
    f_hi_hz: float
    service: str
    classes: list                 # class-table names that live here
    occupancy: float              # P(a channel is active at a random instant)
    channel_hz: float = 0.0       # raster; 0 = a single frequency (the centre)
    snr_db: list = field(default_factory=lambda: [5.0, 30.0])
    weights: dict = field(default_factory=dict)   # class -> relative weight
    mobility: str = "static"
    source: str = ""
    notes: str = ""

    def overlaps(self, lo: float, hi: float) -> bool:
        return self.f_hi_hz > lo and self.f_lo_hz < hi


@dataclass
class CellularBand:
    band: str                     # "B2", "n71"
    dl_lo_hz: float
    dl_hi_hz: float
    technology: str               # lte | nr
    duplex: str                   # FDD | TDD | SDL
    occupancy: float = 0.5        # fraction of the downlink carrying carriers
    source: str = ""


@dataclass
class EnvironmentProfile:
    region: str
    label: str = ""
    itu_region: int = 0
    country: str = ""
    allocations: list = field(default_factory=list)
    cellular_bands: list = field(default_factory=list)
    broadcasters: list = field(default_factory=list)   # {service, f_lo_hz, f_hi_hz, notes}
    terrain_class: str = "rural"
    channel_model: dict = field(default_factory=dict)
    interference: list = field(default_factory=list)   # {kind, classes, occupancy, notes}
    notes: list = field(default_factory=list)
    caveat: str = CAVEAT
    source: str = ""
    created: str = ""
    updated: str = ""

    # -- views ---------------------------------------------------------------
    def allocations_in(self, lo: float, hi: float) -> list:
        return [a for a in self.allocations if a.overlaps(lo, hi)]

    def cellular_in(self, lo: float, hi: float) -> list:
        return [b for b in self.cellular_bands
                if b.dl_hi_hz > lo and b.dl_lo_hz < hi]

    def channel(self) -> dict:
        return dict(self.channel_model or TERRAIN_CHANNELS.get(self.terrain_class, {}))

    def describe(self) -> str:
        return (f"{self.label or self.region} — ITU Region {self.itu_region}, "
                f"{self.country}, {self.terrain_class} terrain, "
                f"{len(self.allocations)} allocations, "
                f"{len(self.cellular_bands)} cellular downlink bands. {self.caveat}")

    # -- JSON ------------------------------------------------------------------
    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "EnvironmentProfile":
        d = dict(d)
        known = set(cls.__dataclass_fields__)
        allocs = [Allocation(**{k: v for k, v in a.items()
                                if k in Allocation.__dataclass_fields__})
                  for a in d.pop("allocations", []) or []]
        cells = [CellularBand(**{k: v for k, v in b.items()
                                 if k in CellularBand.__dataclass_fields__})
                 for b in d.pop("cellular_bands", []) or []]
        env = cls(allocations=allocs, cellular_bands=cells,
                  **{k: v for k, v in d.items() if k in known})
        problems = validate(env)
        if problems:
            raise EnvironmentError(f"environment {env.region!r} is malformed: "
                                   + "; ".join(problems))
        return env


def validate(env: EnvironmentProfile) -> list[str]:
    """Problems in words; empty = fine. Every class must be in the class
    table; every band must have its edges the right way round; occupancy is
    a probability."""
    out = []
    if not env.region or "/" in env.region or "\\" in env.region:
        out.append("the region id is empty or not a plain name")
    if env.terrain_class not in TERRAIN_CHANNELS:
        out.append(f"unknown terrain class {env.terrain_class!r} "
                   f"({', '.join(TERRAIN_CHANNELS)})")
    for i, a in enumerate(env.allocations):
        if not a.f_hi_hz > a.f_lo_hz:
            out.append(f"allocation {i} ({a.service}) has its edges reversed")
        if not 0.0 <= float(a.occupancy) <= 1.0:
            out.append(f"allocation {i} ({a.service}) occupancy is not 0–1")
        for c in a.classes:
            if _classes.get(c) is None:
                out.append(f"allocation {i} ({a.service}) names {c!r}, which "
                           "is not in the class table")
        if a.mobility not in MOBILITY_SPEED:
            out.append(f"allocation {i} ({a.service}) mobility {a.mobility!r} "
                       f"is not one of {', '.join(MOBILITY_SPEED)}")
    for b in env.cellular_bands:
        if not b.dl_hi_hz > b.dl_lo_hz:
            out.append(f"band {b.band} has its edges reversed")
        if b.technology not in ("lte", "nr"):
            out.append(f"band {b.band} technology must be lte or nr")
    for it in env.interference:
        for c in it.get("classes", []):
            if _classes.get(c) is None:
                out.append(f"interference {it.get('kind')} names {c!r}, which "
                           "is not in the class table")
    return out


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------
def path_for(rf, region: str) -> Path:
    from atk_diffusion.paths import _safe
    return Path(rf.environments_dir()) / f"{_safe(region)}.json"


def save(rf, env: EnvironmentProfile) -> Path:
    problems = validate(env)
    if problems:
        raise EnvironmentError("not saved — " + "; ".join(problems))
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    env.updated = now
    env.created = env.created or now
    p = path_for(rf, env.region)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(env.to_json(), indent=2), encoding="utf-8")
    tmp.replace(p)
    rf.record(p, "environment", env.describe())
    return p


def load(rf, region: str) -> EnvironmentProfile:
    """The saved profile, else the built-in one of that name (noted), else
    a refusal that lists what exists."""
    p = path_for(rf, region)
    if p.exists():
        return EnvironmentProfile.from_json(json.loads(p.read_text("utf-8")))
    if region in BUILTIN:
        env = builtin(region)
        env.notes = list(env.notes) + ["built-in (not yet saved under rf_data)"]
        return env
    have = list_environments(rf)
    raise EnvironmentError(f"no environment profile {region!r} — saved: "
                           f"{', '.join(have) or 'none'}; built in: "
                           f"{', '.join(BUILTIN)}")


def list_environments(rf) -> list[str]:
    d = Path(rf.environments_dir())
    return sorted(p.stem for p in d.glob("*.json")) if d.is_dir() else []


def resolve(rf, env) -> EnvironmentProfile:
    """An EnvironmentProfile, or a region id to load."""
    if isinstance(env, EnvironmentProfile):
        return env
    if isinstance(env, str):
        return load(rf, env) if rf is not None else builtin(env)
    raise EnvironmentError("an environment is an EnvironmentProfile or a "
                           "region id such as 'us-va-nokesville'")


# ---------------------------------------------------------------------------
# the built-in prior for Bill's area
# ---------------------------------------------------------------------------
#: US downlink rows of ATK's atk/core/lte_bands.py (_RAW, REGIONS["US"]):
#: (band, mode, dl_low MHz, dl_high MHz). Copied, not imported (ATK is not a
#: run-time dependency); tests/test_synth_scene.py checks them against ATK's
#: file when it is present.
US_LTE_DL = (
    (2, "FDD", 1930, 1990), (4, "FDD", 2110, 2155), (5, "FDD", 869, 894),
    (12, "FDD", 729, 746), (13, "FDD", 746, 756), (14, "FDD", 758, 768),
    (17, "FDD", 734, 746), (25, "FDD", 1930, 1995), (26, "FDD", 859, 894),
    (29, "SDL", 717, 728), (30, "FDD", 2350, 2360), (41, "TDD", 2496, 2690),
    (46, "TDD", 5150, 5925), (48, "TDD", 3550, 3700), (66, "FDD", 2110, 2200),
    (71, "FDD", 617, 652), (85, "FDD", 728, 746),
)
#: US downlink rows of ATK's atk/core/nr_bands.py (_RAW, REGIONS["US"]).
US_NR_DL = (
    (2, "FDD", 1930, 1990), (5, "FDD", 869, 894), (12, "FDD", 729, 746),
    (13, "FDD", 746, 756), (14, "FDD", 758, 768), (25, "FDD", 1930, 1995),
    (26, "FDD", 859, 894), (29, "SDL", 717, 728), (30, "FDD", 2350, 2360),
    (41, "TDD", 2496, 2690), (48, "TDD", 3550, 3700), (66, "FDD", 2110, 2200),
    (70, "FDD", 1995, 2020), (71, "FDD", 617, 652), (77, "TDD", 3300, 4200),
)

_LMR = ["nfm_voice", "dmr", "p25", "nxdn96", "nxdn48"]


def _us_va_nokesville() -> EnvironmentProfile:
    A = Allocation
    allocs = [
        A(88.0 * M, 108.0 * M, "FM broadcast", ["fm_broadcast"], 0.45,
          200e3, [15.0, 45.0], mobility="static",
          source="47 CFR §73.201 (channels at odd tenths, 88.1–107.9 MHz)",
          notes="near a large metro market most channels carry a station; "
                "which ones is not asserted"),
        A(162.3875 * M, 162.5625 * M, "NOAA Weather Radio", ["noaa_wx"], 0.3,
          25e3, [5.0, 30.0], mobility="static",
          source="NWS NOAA Weather Radio: seven channels 162.400–162.550 MHz",
          notes="which channel serves the county is not asserted"),
        A(150.8 * M, 174.0 * M, "VHF land mobile (Part 90; federal above 162 MHz)",
          _LMR + ["pocsag"], 0.03, 12.5e3, [0.0, 30.0],
          weights={"nfm_voice": 3, "dmr": 2, "p25": 2, "nxdn96": 1, "nxdn48": 1,
                   "pocsag": 1},
          mobility="mobile", source="47 CFR Part 90; NTIA federal 162–174 MHz",
          notes="push-to-talk: a channel is busy a few per cent of the time"),
        A(450.0 * M, 470.0 * M, "UHF land mobile (Part 90)", _LMR, 0.03, 12.5e3,
          [0.0, 30.0], weights={"nfm_voice": 3, "dmr": 3, "p25": 2,
                                "nxdn96": 1, "nxdn48": 1},
          mobility="mobile", source="47 CFR Part 90 (450–470 MHz)"),
        A(769.0 * M, 775.0 * M, "700 MHz public-safety narrowband (base)",
          ["p25"], 0.05, 12.5e3, [0.0, 30.0], mobility="static",
          source="47 CFR §90.531 (769–775 / 799–805 MHz)",
          notes="trunked control channels transmit continuously"),
        A(851.0 * M, 869.0 * M, "800 MHz land mobile / public safety (base)",
          ["p25", "nfm_voice"], 0.05, 12.5e3, [0.0, 30.0],
          weights={"p25": 3, "nfm_voice": 1}, mobility="static",
          source="47 CFR §90.613 (806–824 / 851–869 MHz)",
          notes="trunked control channels transmit continuously"),
        A(902.0 * M, 928.0 * M, "900 MHz ISM (LoRa and others)", ["lora"], 0.01,
          200e3, [0.0, 25.0], mobility="static",
          source="47 CFR §15.247; LoRaWAN US915 channel plan"),
        A(929.0 * M, 932.0 * M, "paging (929–932 MHz)", ["pocsag", "flex"], 0.2,
          25e3, [5.0, 35.0], weights={"flex": 2, "pocsag": 1},
          mobility="static", source="47 CFR §22.531 / §90.494"),
        A(1089.0 * M, 1091.0 * M, "ADS-B 1090ES / Mode S", ["adsb"], 0.05, 0.0,
          [0.0, 35.0], mobility="aircraft",
          source="ICAO Annex 10 Vol IV; 14 CFR §91.225",
          notes="squitters from every aircraft in range; near a major airport "
                "the rate is high"),
        A(174.0 * M, 216.0 * M, "VHF-high TV broadcast (ch 7–13)", ["atsc"], 0.3,
          6e6, [10.0, 40.0], mobility="static", source="47 CFR §73.603"),
        A(470.0 * M, 608.0 * M, "UHF TV broadcast (ch 14–36)", ["atsc"], 0.3,
          6e6, [10.0, 40.0], mobility="static",
          source="47 CFR §73.603 (after the 2020 repack)"),
        A(1574.42 * M, 1576.42 * M, "GNSS L1 (jamming only — L1 C/A is below "
          "the floor)", ["gnss_jamming"], 0.002, 0.0, [0.0, 30.0],
          mobility="mobile",
          source="47 CFR §2.106 RNSS 1559–1610 MHz; jamming is illegal "
                 "(47 U.S.C. §333)",
          notes="a jammer is rare; when present it is strong"),
        A(2400.0 * M, 2483.5 * M, "2.4 GHz ISM", ["wifi_24", "ble", "drone_digital"],
          0.1, 2e6, [0.0, 30.0], weights={"wifi_24": 3, "ble": 3,
                                         "drone_digital": 1},
          mobility="pedestrian", source="47 CFR §15.247"),
        A(5725.0 * M, 5850.0 * M, "5.8 GHz ISM / amateur",
          ["drone_fpv_analog", "drone_digital"], 0.005, 20e6, [5.0, 30.0],
          mobility="drone", source="47 CFR §15.247; Part 97 (5650–5925 MHz)"),
    ]
    cells = []
    for band, mode, lo, hi in US_LTE_DL:
        cells.append(CellularBand(f"B{band}", lo * M, hi * M, "lte", mode, 0.5,
                                  "ATK lte_bands.py (3GPP TS 36.101 Table 5.5-1)"))
    for band, mode, lo, hi in US_NR_DL:
        cells.append(CellularBand(f"n{band}", lo * M, hi * M, "nr", mode, 0.3,
                                  "ATK nr_bands.py (3GPP TS 38.104 Table 5.2-1)"))
    broadcasters = [
        {"service": "FM broadcast", "f_lo_hz": 88.0 * M, "f_hi_hz": 108.0 * M,
         "notes": "no station list is asserted"},
        {"service": "TV broadcast (ATSC 1.0)", "f_lo_hz": 174.0 * M,
         "f_hi_hz": 608.0 * M, "notes": "VHF-high and UHF; no station list"},
        {"service": "NOAA Weather Radio", "f_lo_hz": 162.4 * M,
         "f_hi_hz": 162.55 * M, "notes": "seven channels"},
    ]
    interference = [
        {"kind": "unintentional emitters (switching supplies, LED drivers, "
                 "digital clocks)", "classes": ["spur"], "occupancy": 0.02,
         "notes": "CW-like spurs anywhere; strength varies"},
        {"kind": "GNSS jammers", "classes": ["gnss_jamming"], "occupancy": 0.002,
         "notes": "illegal, mobile, rare"},
    ]
    return EnvironmentProfile(
        region="us-va-nokesville",
        label="Nokesville, Virginia (Prince William County)",
        itu_region=2, country="US", allocations=allocs, cellular_bands=cells,
        broadcasters=broadcasters, terrain_class="rural",
        channel_model=dict(TERRAIN_CHANNELS["rural"]),
        interference=interference,
        notes=["semi-rural, rolling terrain; suburban toward Manassas and "
               "Bristow — 'rural' is the prior's terrain class",
               "allocations from the US table; no operator, station, tower or "
               "channel assignment is asserted",
               "cellular downlink bands copied from ATK's own LTE/NR tables"],
        source="toolkit built-in prior, 2026-10-08")


BUILTIN = {"us-va-nokesville": _us_va_nokesville}


def builtin(region: str) -> EnvironmentProfile:
    try:
        env = BUILTIN[region]()
    except KeyError:
        raise EnvironmentError(f"no built-in environment {region!r}; built in: "
                               f"{', '.join(BUILTIN)}") from None
    env.created = env.updated = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return env
