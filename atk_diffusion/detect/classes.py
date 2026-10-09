# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The class table, v1 (DETECTION_DESIGN §9, decision D8).

One table that every part of the detector reads, so nothing has its own
private idea of what DMR is:

* the cyclic proposer scans each class's cycle frequencies (§3, §4.1),
* the classifier's label set and the 2D proposer's coarse FAMILY come from
  it (§3: family, not modulation — the spectrogram cannot tell QPSK from
  8PSK and does not pretend to),
* the confirmer looks up which decoder can confirm a class (§5),
* the cut's Route step offers only the tools that accept the class (§4.2),
* the waterfall caption takes the class's ATK TECHNOLOGY key, so a detected
  DMR box is DMR's orange — the colour ATK has always used for DMR.

What Bill can capture around Nokesville and make with his own hardware, plus
what TorchSig can synthesise. The field adds the rest through *teach*.
Nothing here depends on samples Bill cannot get (no CSAR radios — plan change
log, 2026-10-08).
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: The 2D proposer's coarse families (DETECTION_DESIGN §3).
FAMILIES = ("fm", "am", "fsk", "psk_qam", "ofdm", "burst", "spread", "unknown")

FAMILY_WORDS = {"fm": "FM-like", "am": "AM-like", "fsk": "FSK",
                "psk_qam": "PSK/QAM", "ofdm": "OFDM", "burst": "burst",
                "spread": "spread / noise-like", "unknown": "unknown"}

#: Decoders and demodulators ATK has, by the key the confirmer and the
#: Route step use. `confirms` = a successful run upgrades Proposed to
#: Confirmed (§5); a demodulator only produces audio and confirms nothing.
DECODERS = {
    "dsd":        {"label": "dsd-neo (P25 / DMR / NXDN)", "confirms": True},
    "pager":      {"label": "ATK's pager decoder (POCSAG / FLEX)", "confirms": True},
    "multimon":   {"label": "multimon-ng", "confirms": True},
    "adsb":       {"label": "the ADS-B path (dump1090)", "confirms": True},
    "lte_search": {"label": "ATK's LTE cell search (PSS/SSS)", "confirms": True},
    "nr_search":  {"label": "ATK's 5G NR SSB search", "confirms": True},
    "nfm":        {"label": "NFM demodulator", "confirms": False},
    "wfm":        {"label": "WFM demodulator", "confirms": False},
    "am":         {"label": "AM demodulator", "confirms": False},
    "ssb":        {"label": "SSB demodulator", "confirms": False},
}


@dataclass(frozen=True)
class SignalClass:
    name: str                       # the label (SigMF core:label)
    label: str                      # what an analyst calls it
    family: str                     # one of FAMILIES
    bandwidth_hz: float             # typical occupied bandwidth
    symbol_rates: tuple = ()        # cycle frequencies the cyclic proposer scans
    cp_lag_s: float = 0.0           # OFDM: the useful-symbol lag (CP probe)
    conjugate: bool = False         # conjugate cyclic features (BPSK/AM/MSK)
    decoders: tuple = ()            # keys of DECODERS, best first
    technology: str = ""            # ATK's PROTOCOL_COLOURS key ('' = none)
    torchsig: tuple = ()            # TorchSig 2.2 signal names that synthesise it
    native: str = ""                # the native generator's key ('' = none)
    negative: bool = False          # an explicit negative (noise, spur, DC)
    profiles: tuple = ()            # receiver families it is captured with
    notes: str = ""

    @property
    def confirmable(self) -> bool:
        return any(DECODERS[d]["confirms"] for d in self.decoders)


_ALL_RX = ("rtlsdr", "krakensdr", "hackrf", "bladerf1", "bladerf2")

CLASSES: tuple[SignalClass, ...] = (
    SignalClass("nfm_voice", "narrowband FM voice", "fm", 11_000.0,
                decoders=("nfm",), technology="analog",
                torchsig=("fm",), native="nfm", profiles=_ALL_RX,
                notes="amateur, business, analog public safety; no cyclic "
                      "feature of its own — energy and the learned detector "
                      "find it"),
    SignalClass("p25", "P25 Phase 1 (C4FM)", "fsk", 8_100.0,
                symbol_rates=(4_800.0,), decoders=("dsd",), technology="p25",
                torchsig=("p25",), native="c4fm_4800", profiles=_ALL_RX),
    SignalClass("dmr", "DMR (4FSK TDMA)", "fsk", 7_600.0,
                symbol_rates=(4_800.0,), decoders=("dsd",), technology="dmr",
                torchsig=("dmr",), native="dmr_4800", profiles=_ALL_RX,
                notes="two-slot TDMA: 27.5 ms bursts on a 30 ms frame"),
    SignalClass("nxdn96", "NXDN 9600 (4800 sym/s)", "fsk", 8_300.0,
                symbol_rates=(4_800.0,), decoders=("dsd",), technology="nxdn",
                torchsig=("4fsk",), native="c4fm_4800", profiles=_ALL_RX),
    SignalClass("nxdn48", "NXDN 4800 (2400 sym/s)", "fsk", 4_000.0,
                symbol_rates=(2_400.0,), decoders=("dsd",), technology="nxdn",
                torchsig=("4fsk",), native="c4fm_2400", profiles=_ALL_RX),
    SignalClass("pocsag", "POCSAG pager", "fsk", 12_000.0,
                symbol_rates=(512.0, 1_200.0, 2_400.0),
                decoders=("pager", "multimon"), technology="pocsag",
                torchsig=("2fsk",), native="pocsag", profiles=_ALL_RX),
    SignalClass("flex", "FLEX pager", "fsk", 12_000.0,
                symbol_rates=(1_600.0, 3_200.0), decoders=("pager",),
                technology="pocsag", torchsig=("2fsk", "4fsk"), native="flex",
                profiles=_ALL_RX,
                notes="shares the pager amber; the caption says FLEX"),
    SignalClass("adsb", "ADS-B 1090 MHz", "burst", 2_000_000.0,
                symbol_rates=(1_000_000.0,), decoders=("adsb",),
                torchsig=("adsb-long", "adsb-short"), native="adsb",
                profiles=("rtlsdr", "hackrf", "bladerf1", "bladerf2"),
                notes="PPM squitters, 120 µs; the air picture confirms"),
    SignalClass("fm_broadcast", "broadcast FM", "fm", 200_000.0,
                decoders=("wfm",), technology="analog", torchsig=("fm",),
                native="wfm", profiles=_ALL_RX),
    SignalClass("noaa_wx", "NOAA weather radio", "fm", 12_000.0,
                decoders=("nfm",), technology="analog", torchsig=("fm",),
                native="nfm", profiles=_ALL_RX),
    SignalClass("lora", "LoRa (915 MHz ISM chirps)", "spread", 125_000.0,
                torchsig=("lora",), native="lora",
                profiles=("rtlsdr", "hackrf", "bladerf1", "bladerf2"),
                notes="125/250/500 kHz chirp spread spectrum"),
    SignalClass("lte_dl", "LTE downlink", "ofdm", 9_000_000.0,
                symbol_rates=(14_000.0,), cp_lag_s=1.0 / 15_000.0,
                decoders=("lte_search",), technology="lte",
                torchsig=("ofdm-600", "ofdm-900"), native="ofdm_lte",
                profiles=_ALL_RX,
                notes="15 kHz SCS: the cyclic prefix makes the signal "
                      "correlate with itself at 66.7 µs; ATK's PSS/SSS search "
                      "confirms the cell"),
    SignalClass("nr_dl", "5G NR downlink (30 kHz SCS)", "ofdm", 20_000_000.0,
                symbol_rates=(28_000.0,), cp_lag_s=1.0 / 30_000.0,
                decoders=("nr_search",), technology="lte",
                torchsig=("ofdm-1200",), native="ofdm_nr30",
                profiles=("hackrf", "bladerf1", "bladerf2"),
                notes="also 15 kHz SCS in low bands — then it looks like LTE "
                      "to the CP probe and the SSB search tells them apart"),
    SignalClass("wifi_24", "Wi-Fi (2.4 GHz OFDM)", "ofdm", 16_600_000.0,
                symbol_rates=(250_000.0,), cp_lag_s=3.2e-6,
                torchsig=("80211a",), native="ofdm_wifi",
                profiles=("hackrf", "bladerf1", "bladerf2")),
    SignalClass("ble", "Bluetooth LE", "fsk", 1_000_000.0,
                symbol_rates=(1_000_000.0,), torchsig=("btle",),
                native="ble", profiles=("hackrf", "bladerf1", "bladerf2")),
    SignalClass("atsc", "ATSC 1.0 (8VSB)", "psk_qam", 5_380_000.0,
                symbol_rates=(10_762_238.0,), conjugate=True,
                torchsig=(), native="vsb8", profiles=("hackrf", "bladerf1",
                                                      "bladerf2"),
                notes="the pilot gives a conjugate feature at the carrier"),
    SignalClass("drone_fpv_analog", "analog FPV video (5.8 GHz)", "fm",
                18_000_000.0, torchsig=("fm",), native="fpv_analog",
                profiles=("hackrf",),
                notes="Ludovika compact-CNN paper; HackRF profile"),
    SignalClass("drone_digital", "digital drone link (OFDM / FHSS)", "ofdm",
                10_000_000.0, torchsig=("ofdm-600",), native="ofdm_drone",
                profiles=("hackrf", "bladerf1", "bladerf2")),
    SignalClass("gnss_jamming", "GNSS jamming at L1", "spread", 2_000_000.0,
                torchsig=("lfm-data", "chirpss"), native="chirp_jammer",
                profiles=("rtlsdr", "bladerf1", "bladerf2"),
                notes="the trigger for where-am-I (plan E3)"),
    # -- the bladeRF-generated reference set (TorchSig families) ------------
    SignalClass("ref_bpsk", "reference BPSK", "psk_qam", 0.0,
                conjugate=True, torchsig=("bpsk",), native="bpsk"),
    SignalClass("ref_qpsk", "reference QPSK", "psk_qam", 0.0,
                torchsig=("qpsk",), native="qpsk"),
    SignalClass("ref_8psk", "reference 8PSK", "psk_qam", 0.0,
                torchsig=("8psk",), native="8psk"),
    SignalClass("ref_16qam", "reference 16QAM", "psk_qam", 0.0,
                torchsig=("16qam",), native="16qam"),
    SignalClass("ref_64qam", "reference 64QAM", "psk_qam", 0.0,
                torchsig=("64qam",), native="64qam"),
    SignalClass("ref_2fsk", "reference 2FSK", "fsk", 0.0,
                torchsig=("2fsk",), native="2fsk"),
    SignalClass("ref_gfsk", "reference GFSK", "fsk", 0.0,
                torchsig=("2gfsk",), native="gfsk"),
    SignalClass("ref_ofdm", "reference OFDM", "ofdm", 0.0,
                torchsig=("ofdm-256",), native="ofdm"),
    SignalClass("ref_ask", "reference 4ASK", "am", 0.0,
                conjugate=True, torchsig=("4ask",), native="ask4"),
    SignalClass("ref_am", "reference AM (DSB)", "am", 0.0,
                conjugate=True, torchsig=("am-dsb",), native="am"),
    # -- explicit negatives --------------------------------------------------
    SignalClass("noise", "noise only", "unknown", 0.0, negative=True,
                native="noise"),
    SignalClass("spur", "a spur (CW tone from the receiver itself)", "unknown",
                0.0, negative=True, torchsig=("tone",), native="tone"),
    SignalClass("dc_spike", "the DC spike", "unknown", 0.0, negative=True,
                native="dc"),
)

BY_NAME: dict[str, SignalClass] = {c.name: c for c in CLASSES}

#: The honest answer when nothing is near enough (DETECTION_DESIGN §4).
UNKNOWN = "UNKNOWN"


def get(name: str) -> SignalClass | None:
    return BY_NAME.get(str(name))


def names(include_negatives: bool = True) -> list[str]:
    return [c.name for c in CLASSES if include_negatives or not c.negative]


def for_profile_family(family: str) -> list[SignalClass]:
    """Classes this receiver family can actually capture (or that are
    generic references/negatives with no family restriction)."""
    return [c for c in CLASSES if not c.profiles or family in c.profiles]


def cycle_frequencies(sample_rate: float | None = None) -> list[tuple[str, float]]:
    """(class, α) for every listed symbol-rate cycle frequency — what the
    cyclic proposer scans. With a sample rate, only α the rate can show
    (α < fs/2) are listed."""
    out = []
    for c in CLASSES:
        for a in c.symbol_rates:
            if sample_rate is None or a < 0.5 * float(sample_rate):
                out.append((c.name, float(a)))
    return out


def cp_lags(sample_rate: float | None = None) -> list[tuple[str, float]]:
    """(class, lag seconds) for the OFDM cyclic-prefix probe; only lags of
    at least four samples at `sample_rate` are listed."""
    out = []
    for c in CLASSES:
        if c.cp_lag_s > 0 and (sample_rate is None
                               or c.cp_lag_s * float(sample_rate) >= 4):
            out.append((c.name, c.cp_lag_s))
    return out


def technology(name: str) -> str:
    c = get(name)
    return c.technology if c else ""


def tools_for(name: str) -> list[str]:
    """Decoder/demodulator keys that accept this class (Route, §4.2)."""
    c = get(name)
    return list(c.decoders) if c else []
