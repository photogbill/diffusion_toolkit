# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""TorchSig 2.2.0 at the profile's exact rate (plan §3.3, §4.A, §4.B;
DETECTION_DESIGN §6 step 2, §10; ARCHITECTURE §4.3). The ONLY module that
imports TorchSig, and only inside functions.

Bill, 2026-10-08: *"OmniSIG only works if the sample rate is identical in
training … as in the field."* So TorchSig's `sample_rate` — and its FFT size,
its frequency limits and every per-class bandwidth range — is DERIVED FROM
THE RECEIVER PROFILE here, never typed: every public function takes the
profile (a `ReceiverProfile` or its id), not a rate.

Bill, the same night, on TorchSig: *"it only works with a specific version of
python … it was a pain the last time."* `available()` answers in words before
anything is attempted: installed or not, the version (this file is written
against 2.2.0's API — the 1.x/0.x APIs are different packages in all but
name), and whether it imports. Nothing in the toolkit depends on it: the
native generator makes every class without it.

WHAT TORCHSIG 2.2.0 ACTUALLY DOES (read from its source, verified by tests):
* Its "bandwidth" parameter means different things per builder. For the
  builders that oversample by 4 at baseband and resample (constellations,
  DMR, P25, BTLE, ADS-B, LoRa, OFDM, 802.11a) it fixes the SYMBOL (or chip,
  or channel) rate exactly; for FM it is a Carson bandwidth; for FSK a 3 dB
  bandwidth reached through a RANDOM, unrecorded modulation index.
* Every builder refuses bandwidth > fs/2 (FM: > fs). So at an RTL's 2.4 MS/s
  TorchSig cannot make a 2 Mchip/s ADS-B squitter or a 9 MHz LTE carrier
  even though both are, physically, inside what the RTL samples. `coverage()`
  says which classes, at which rates, and why.
* Its metadata carries NO symbol rate. Where the builder's arithmetic fixes
  it, this module recovers it EXACTLY: the drawn "bandwidth" is captured by
  a component transform before TorchSig overwrites it, and the resampler's
  realised ratio comes from TorchSig's own `multistage_polyphase_resampler_
  actual_rate`. Where the builder draws an unrecorded parameter (FSK index,
  chirp length, FM index) the label has no symbol rate and says so.
* Its SNR is the peak bin of the time-averaged spectrum over the per-bin
  floor, and it re-draws each box where the max-hold spectrum clears the
  floor by 3 dB — both SNR- and FFT-dependent. So the sample is REBUILT:
  TorchSig's own noise realisation (captured as it is drawn) plus each
  TorchSig component, re-levelled to the toolkit's SNR definition
  (`native.SNR_DEFINITION`) and boxed at its 99 % occupied band. With unit
  gains the rebuild reproduces TorchSig's sample exactly (a test proves it);
  TorchSig's own SNR and box ride along in the label.

IMPAIRMENT LEVELS are TorchSig's own `Impairments(level)`: 0 perfect, 1
transmitter hardware ("cabled"), 2 transmitter + fading channel ("wireless").
Only its SIGNAL (transmit/channel) transforms are used. Its dataset-level
transforms are not: the RECEIVER side is this profile's MEASURED impairments
(`dsp.impair`, plan §3.4 — "so synthetic data sounds like this receiver and
not a textbook one"), applied by the caller; and its ML augmentations
(time reversal, I/Q swap, slope, dropped samples) are training-time
augmentations, not something to bake into a dataset on disk. At levels ≥ 1
TorchSig's clock drift moves the symbol rate by its ppm setting; the label
keeps the nominal value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from atk_diffusion import capabilities, provenance
from atk_diffusion import profiles as _profiles
from atk_diffusion.detect import classes as _classes
from atk_diffusion.synth import labels as _labels
from atk_diffusion.synth import native as _native

provenance.METHOD_TIERS.setdefault("synthetic_torchsig", "invented")

GENERATOR = _labels.TORCHSIG_GENERATOR
METHOD = "synthetic_torchsig"
TESTED_VERSION = "2.2.0"

_AVAILABLE: tuple[bool, str] | None = None
_DS_CLASS = None


class TorchsigUnavailable(RuntimeError):
    """TorchSig cannot be used here; the message says why and what still works."""


class TorchsigRefusal(ValueError):
    """TorchSig cannot make this class at this rate; the message says why."""


def available(refresh: bool = False) -> tuple[bool, str]:
    """(ok, why) — installed, the right version, and importable."""
    global _AVAILABLE
    if _AVAILABLE is not None and not refresh:
        return _AVAILABLE
    if not capabilities.has("torchsig"):
        res = (False, "TorchSig is not installed in this environment. It belongs "
                      "in the training environment (envs\\atk_diffusion); the "
                      "native generator makes every class without it.")
    else:
        v = capabilities.version("torchsig")
        if not v.startswith("2.2."):
            res = (False, f"TorchSig {v or '(version unknown)'} is installed, but "
                          f"this backend is written and tested against TorchSig "
                          f"{TESTED_VERSION}, whose API is not the older one. "
                          "Install 2.2.0 in the training environment, or use "
                          "the native generator, which needs no TorchSig.")
        else:
            try:
                import torchsig.datasets.datasets  # noqa: F401
                import torchsig.utils.dsp  # noqa: F401
                res = (True, f"TorchSig {v}")
            except Exception as e:                       # noqa: BLE001
                res = (False, f"TorchSig {v} is installed but does not import here "
                              f"({type(e).__name__}: {e}). The native generator "
                              "works without it.")
    _AVAILABLE = res
    return res


def _need() -> None:
    ok, why = available()
    if not ok:
        raise TorchsigUnavailable(why)


# ---------------------------------------------------------------------------
# the profile is the only source of the rate
# ---------------------------------------------------------------------------
def _profile(profile) -> _profiles.ReceiverProfile:
    if isinstance(profile, _profiles.ReceiverProfile):
        return profile
    if isinstance(profile, str):
        return _profiles.new_profile(profile)
    raise TypeError("TorchSig's sample rate comes from a receiver profile (a "
                    "ReceiverProfile or its id, e.g. 'rtlsdr_2400000_cu8') — "
                    "never a number typed in (plan §3.3).")


def dataset_metadata(profile, n_iq: int, *, num_signals=(1, 1),
                     snr_db=(20.0, 20.0), center_range=None,
                     duration=None, overlap_probability: float = 0.0) -> dict:
    """TorchSig 2.2.0 dataset metadata, every rate-dependent value derived
    from the profile: sample_rate (the profile's exact rate), fft_size and
    stride (the profile's STFT geometry), frequency limits (±fs/2), default
    bandwidth range (fs/64 … fs/8), centre range (inside ±0.45 fs)."""
    prof = _profile(profile)
    fs = int(prof.sample_rate)
    n = int(n_iq)
    nfft = int(prof.stft.fft_size)
    c_lo, c_hi = center_range if center_range is not None else (-0.45 * fs, 0.45 * fs)
    d_lo, d_hi = duration if duration is not None else (n, n)
    return {"num_iq_samples_dataset": n,
            "num_signals_min": int(num_signals[0]),
            "num_signals_max": int(num_signals[1]),
            "fft_size": nfft, "fft_stride": nfft,
            "sample_rate": fs, "noise_power_db": 0.0,
            "snr_db_min": float(snr_db[0]), "snr_db_max": float(snr_db[1]),
            "cochannel_overlap_probability": float(overlap_probability),
            "signal_duration_in_samples_min": int(d_lo),
            "signal_duration_in_samples_max": int(d_hi),
            "bandwidth_min": max(1, fs // 64), "bandwidth_max": max(2, fs // 8),
            "signal_center_freq_min": float(c_lo),
            "signal_center_freq_max": float(c_hi),
            "frequency_min": -fs / 2.0, "frequency_max": fs / 2.0 - 1.0}


# ---------------------------------------------------------------------------
# recipes: how each class-table class is made by TorchSig 2.2.0
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Recipe:
    """`names`: TorchSig names used (a subset of the class table's), in
    preference order. `bandwidth`: TorchSig's 'bandwidth' value when fixed by
    the class (its builder's own meaning, `meaning`); None = chosen per
    example. `rate`: how the symbol rate follows from the builder ('' = it
    does not: `why_no_rate`). `limit`: TorchSig's own ceiling, × fs."""
    names: tuple
    meaning: str
    bandwidth: float | None = None
    rate: str = ""
    limit: float = 0.5
    burst_s: float | None = None
    why_no_rate: str = ""
    note: str = ""
    extra: dict = field(default_factory=dict)


_FSK_NO_RATE = ("TorchSig's FSK builders draw a random modulation index (and "
                "Gaussian BT) and do not record it, so the symbol rate behind "
                "a given 3 dB bandwidth is unknown")
_ANALOG = "analog: no symbol rate"

RECIPES: dict[str, Recipe] = {
    "nfm_voice": Recipe(("fm",), "Carson bandwidth", 11_000.0, limit=1.0,
                        why_no_rate=_ANALOG),
    "noaa_wx": Recipe(("fm",), "Carson bandwidth", 12_000.0, limit=1.0,
                      why_no_rate=_ANALOG),
    "fm_broadcast": Recipe(("fm",), "Carson bandwidth", 200_000.0, limit=1.0,
                           why_no_rate=_ANALOG,
                           note="TorchSig's FM is a noise-modulated carrier: no "
                                "stereo pilot or RDS"),
    "drone_fpv_analog": Recipe(("fm",), "Carson bandwidth", 18e6, limit=1.0,
                               why_no_rate=_ANALOG,
                               note="a noise-modulated FM carrier, not video"),
    "p25": Recipe(("p25",), "symbol rate", 4800.0, rate="x4"),
    "dmr": Recipe(("dmr",), "symbol rate", 4800.0, rate="x4",
                  note="TorchSig's DMR has no TDMA timing (its own docstring); "
                       "the native generator has the 27.5 ms bursts"),
    "nxdn96": Recipe(("4fsk",), "3 dB bandwidth", 8_300.0, why_no_rate=_FSK_NO_RATE,
                     note="generic 4FSK, not NXDN's deviations"),
    "nxdn48": Recipe(("4fsk",), "3 dB bandwidth", 4_000.0, why_no_rate=_FSK_NO_RATE,
                     note="generic 4FSK, not NXDN's deviations"),
    "pocsag": Recipe(("2fsk",), "3 dB bandwidth", 12_000.0, why_no_rate=_FSK_NO_RATE,
                     note="generic 2FSK: no preamble or codewords"),
    "flex": Recipe(("2fsk", "4fsk"), "3 dB bandwidth", 12_000.0,
                   why_no_rate=_FSK_NO_RATE, note="generic 2/4FSK"),
    "adsb": Recipe(("adsb-long", "adsb-short"), "chip rate", 2.0e6, rate="adsb",
                   burst_s=120e-6, note="random payload, no CRC"),
    "lora": Recipe(("lora",), "chip rate", None, rate="lora",
                   note="stepped (not continuous-phase) chirps"),
    "lte_dl": Recipe(("ofdm-900", "ofdm-600"), "channel bandwidth", None,
                     rate="ofdm", note="generic OFDM at LTE's 15 kHz spacing: "
                                       "random CP, no PSS or reference signals"),
    "nr_dl": Recipe(("ofdm-1200",), "channel bandwidth", 1200 * 30e3, rate="ofdm"),
    "wifi_24": Recipe(("80211a",), "channel bandwidth", 20e6, rate="wifi"),
    "ble": Recipe(("btle",), "symbol rate", 1.0e6, rate="x4"),
    "drone_digital": Recipe(("ofdm-600",), "channel bandwidth", 10e6, rate="ofdm"),
    "gnss_jamming": Recipe(("lfm-data", "chirpss"), "chirp span", None,
                           why_no_rate="TorchSig's chirp builders draw the chirp "
                                       "length at random and do not record it"),
    "ref_bpsk": Recipe(("bpsk",), "symbol rate", None, rate="x4"),
    "ref_qpsk": Recipe(("qpsk",), "symbol rate", None, rate="x4"),
    "ref_8psk": Recipe(("8psk",), "symbol rate", None, rate="x4"),
    "ref_16qam": Recipe(("16qam",), "symbol rate", None, rate="x4"),
    "ref_64qam": Recipe(("64qam",), "symbol rate", None, rate="x4"),
    "ref_ask": Recipe(("4ask",), "symbol rate", None, rate="x4"),
    "ref_2fsk": Recipe(("2fsk",), "3 dB bandwidth", None, why_no_rate=_FSK_NO_RATE),
    "ref_gfsk": Recipe(("2gfsk",), "3 dB bandwidth", None, why_no_rate=_FSK_NO_RATE),
    "ref_ofdm": Recipe(("ofdm-256",), "channel bandwidth", None, rate="ofdm"),
    "ref_am": Recipe(("am-dsb",), "message bandwidth (two-sided)", 6_000.0,
                     why_no_rate=_ANALOG),
    "spur": Recipe(("tone",), "none", why_no_rate="a tone has no symbol rate"),
    "noise": Recipe((), "none", why_no_rate="noise only"),
}

#: the lookup name a TorchSig name is built from ("fm" is a family in 2.2.0)
_BUILD = {"fm": "fm-data"}

_REASONS_NONE = {
    "atsc": "TorchSig 2.2.0 has no 8-VSB builder",
    "dc_spike": "the DC spike is a receiver artefact; TorchSig has no generator "
                "for it (the receiver impairment model adds the measured one)",
}

_LORA_BW = (500e3, 250e3, 125e3)
_OFDM_N = {"ofdm-900": 900, "ofdm-600": 600, "ofdm-1200": 1200, "ofdm-256": 256}


@dataclass
class _Plan:
    cls: str
    name: str = ""            # TorchSig name
    bw: int = 0               # TorchSig 'bandwidth' (its builder's meaning)
    extra: dict = field(default_factory=dict)
    duration: int | None = None
    recipe: Recipe | None = None


def _ref_bw(r: Recipe, cls: str, p: dict, fs: float, rng) -> float:
    """TorchSig's 'bandwidth' for a reference class. For the 4×-oversampled
    builders it IS the symbol rate, so a target occupied band is divided by
    a mid roll-off (1.3); the FSK/OFDM/AM builders take the band itself.
    With neither given, the band is drawn log-uniformly in fs/64 … fs/8."""
    if p.get("symbol_rate"):
        if r.rate != "x4":
            raise TorchsigRefusal(
                f"{_classes.get(cls).label}: TorchSig's builder cannot be asked "
                "for a symbol rate (it draws its modulation index at random); "
                "give bandwidth_hz instead")
        return float(p["symbol_rate"])
    if p.get("bandwidth_hz"):
        target = float(p["bandwidth_hz"])
    else:
        target = math.exp(rng.uniform(math.log(fs / 64), math.log(fs / 8)))
    return target / 1.3 if r.rate == "x4" else target


def _plan(cls: str, fs: float, rng, params: dict | None) -> _Plan:
    """Choose TorchSig's name and parameters for one example of `cls` at
    `fs`, or raise TorchsigRefusal with the reason."""
    c = _classes.get(cls)
    if c is None:
        raise TorchsigRefusal(f"{cls!r} is not in the class table")
    if cls in _REASONS_NONE or cls not in RECIPES:
        raise TorchsigRefusal(f"{c.label}: {_REASONS_NONE.get(cls, 'no TorchSig recipe')}")
    r = RECIPES[cls]
    p = dict(params or {})
    pl = _Plan(cls, recipe=r)
    if cls == "noise":
        return pl
    lim = r.limit * fs
    if cls == "lte_dl":
        for name in r.names:
            bw = _OFDM_N[name] * 15e3
            if bw <= lim:
                pl.name, pl.bw = name, int(bw)
                break
        else:
            raise TorchsigRefusal(
                f"{c.label} at 15 kHz spacing needs {_native._hz_words(600 * 15e3)} "
                f"(ofdm-600) at least; TorchSig's builders hold a signal to half "
                f"the sample rate, {_native._hz_words(lim)} at "
                f"{_native._rate_words(fs)}")
    elif cls == "lora":
        want = p.get("bw")
        opts = [b for b in _LORA_BW if b <= lim]
        if want is not None:
            if float(want) > lim:
                raise TorchsigRefusal(f"LoRa at {_native._hz_words(want)} exceeds "
                                      f"TorchSig's limit of half the rate "
                                      f"({_native._hz_words(lim)})")
            opts = [float(want)]
        if not opts:
            raise TorchsigRefusal(f"LoRa's narrowest channel (125 kHz) exceeds "
                                  f"TorchSig's limit of {_native._hz_words(lim)}")
        pl.name, pl.bw = "lora", int(rng.choice(opts) if want is None else opts[0])
        pl.extra["sf"] = int(p.get("sf") or rng.integers(7, 13))
    elif cls == "gnss_jamming":
        sweep = float(p.get("sweep_hz", 2.0e6))
        for name, bw in (("lfm-data", sweep), ("chirpss", sweep / 2.0)):
            if bw <= lim:
                pl.name, pl.bw = name, int(bw)
                break
        else:
            raise TorchsigRefusal(f"a {_native._hz_words(sweep)} sweep exceeds "
                                  f"TorchSig's chirp limits at {_native._rate_words(fs)}")
    elif cls == "adsb":
        fr = p.get("frame") or ("long" if rng.uniform() < 0.8 else "short")
        pl.name = "adsb-long" if fr == "long" else "adsb-short"
        pl.bw = int(r.bandwidth)
        pl.duration = int(round((120e-6 if fr == "long" else 64e-6) * fs))
    elif r.bandwidth is not None:
        pl.name = r.names[0] if len(r.names) == 1 else str(rng.choice(r.names))
        pl.bw = int(round(r.bandwidth))
    elif cls == "spur":
        pl.name = "tone"
    else:                                   # references: chosen per example
        pl.name = r.names[0]
        pl.bw = int(round(_ref_bw(r, cls, p, fs, rng)))
    if pl.name != "tone" and pl.bw > lim:
        raise TorchsigRefusal(
            f"{c.label} needs TorchSig's {r.meaning} of {_native._hz_words(pl.bw)}; "
            f"its builders refuse anything over {'the' if r.limit == 1.0 else 'half the'} "
            f"sample rate — {_native._hz_words(lim)} at {_native._rate_words(fs)}")
    if pl.name != "tone" and pl.bw < 1:
        raise TorchsigRefusal(f"{c.label}: a bandwidth below 1 Hz is not a signal")
    return pl


def can_generate(cls: str, profile, params: dict | None = None
                 ) -> tuple[bool, str]:
    fs = float(_profile(profile).sample_rate)
    try:
        _plan(cls, fs, np.random.default_rng(0), params)
        return True, ""
    except TorchsigRefusal as e:
        return False, str(e)


def coverage(profile) -> dict[str, str]:
    """class -> '' when TorchSig 2.2.0 can make it at this profile's rate,
    else the reason, in words."""
    return {c.name: can_generate(c.name, profile)[1] for c in _classes.CLASSES}


# ---------------------------------------------------------------------------
# TorchSig objects (lazy)
# ---------------------------------------------------------------------------
class _Recorder:
    """A component transform that keeps the builder's drawn 'bandwidth'
    before TorchSig overwrites it with its measured box."""

    def __call__(self, signal):
        try:
            signal["atk_nominal_bandwidth"] = float(signal["bandwidth"])
        except Exception:                                 # noqa: BLE001
            pass
        return signal


def _dataset_class():
    global _DS_CLASS
    if _DS_CLASS is None:
        from torchsig.datasets.datasets import TorchSigIterableDataset

        class _NoiseKeeping(TorchSigIterableDataset):
            """TorchSig's dataset, keeping each sample's noise realisation so
            the sample can be rebuilt exactly (re-levelled)."""

            def _build_noise_floor(self):
                x = super()._build_noise_floor()
                self._atk_noise = np.array(x, copy=True)
                return x

        _DS_CLASS = _NoiseKeeping
    return _DS_CLASS


def _make(fs: float, n: int, plans: list[_Plan], seed: int, *, profile,
          num_signals, impairment_level: int, center_range, overlap: float,
          duration=None):
    from torchsig.transforms.impairments import Impairments
    from torchsig.utils.signal_building import lookup_signal_generator_by_string
    if int(impairment_level) not in (0, 1, 2):
        raise TorchsigRefusal("TorchSig's impairment level is 0 (perfect), 1 "
                              "(cabled: transmitter hardware) or 2 (wireless: "
                              "transmitter + fading)")
    comp = [_Recorder()]
    if int(impairment_level) > 0:
        comp.append(Impairments(int(impairment_level)).signal_transforms)
    md = dataset_metadata(profile, n, num_signals=num_signals,
                          center_range=center_range, duration=duration,
                          overlap_probability=overlap)
    ds = _dataset_class()(signal_generators=[], component_transforms=comp,
                          target_labels=None, metadata=md, seed=int(seed))
    for pl in plans:
        if not pl.name:
            continue
        gen = lookup_signal_generator_by_string(_BUILD.get(pl.name, pl.name))
        if pl.name != "tone":
            gen["bandwidth_min"] = int(pl.bw)
            gen["bandwidth_max"] = int(pl.bw)
        for k, v in pl.extra.items():
            gen[k] = v
        if pl.duration:
            gen["signal_duration_in_samples_min"] = int(pl.duration)
            gen["signal_duration_in_samples_max"] = int(pl.duration)
        gen["atk_torchsig_name"] = pl.name
        ds.add_signal_generator(gen, class_name=pl.cls)
    if not ds.signal_generators:
        ds.add_signal_generator(lookup_signal_generator_by_string("tone"),
                                class_name="spur")
    return ds


def _symbol_rate(pl: _Plan, fs: float, meta: dict) -> float:
    """The symbol rate TorchSig's builder realised, from its own arithmetic:
    baseband at 4 samples per symbol (chip / OFDM sample), resampled by
    TorchSig's multistage resampler whose realised ratio is its own
    `multistage_polyphase_resampler_actual_rate`."""
    kind = pl.recipe.rate if pl.recipe else ""
    if not kind:
        return 0.0
    from torchsig.utils.dsp import multistage_polyphase_resampler_actual_rate
    bw = float(meta.get("atk_nominal_bandwidth") or pl.bw)
    r = multistage_polyphase_resampler_actual_rate((fs / bw) / 4.0)
    base = fs / r                         # TorchSig's baseband rate, realised
    if kind == "x4":
        return base / 4.0
    if kind == "adsb":
        return base / 4.0 / 2.0           # 2 chips a bit
    if kind == "lora":
        return base / 4.0 / 2 ** int(pl.extra.get("sf", 7))
    if kind == "ofdm":
        n = _OFDM_N[pl.name]
        cp = int(meta.get("cyclic_prefix_len", 0) or 0)
        return base / (4.0 * (n + cp))
    if kind == "wifi":
        return base / (4.0 * 80.0)        # 80 samples a symbol at the channel rate
    return 0.0


def _component_label(comp, pl: _Plan, fs: float, n: int, level: int) -> dict:
    meta = _labels._meta_dict(comp)
    s = np.asarray(comp.data, dtype=np.complex128)
    start = int(meta["start_in_samples"])
    count = int(s.size)
    lo, hi = _native.occupied_band(s, fs) if count else (0.0, 0.0)
    c = _classes.get(pl.cls)
    ts_lo = float(meta["center_freq"]) - float(meta["bandwidth"]) / 2.0
    ts_hi = float(meta["center_freq"]) + float(meta["bandwidth"]) / 2.0
    rs = _symbol_rate(pl, fs, meta)
    params = {k: v for k, v in meta.items()
              if k in ("atk_nominal_bandwidth", "pulse_shape_name", "alpha_rolloff",
                       "has_cyclic_prefix", "cyclic_prefix_len")}
    params.update(pl.extra)
    params["torchsig_bandwidth_meaning"] = pl.recipe.meaning if pl.recipe else ""
    if not rs and pl.recipe and pl.recipe.why_no_rate:
        params["no_symbol_rate"] = pl.recipe.why_no_rate
    if pl.recipe and pl.recipe.note:
        params["note"] = pl.recipe.note
    if pl.name.startswith("ofdm") and rs:
        params["cp_lag_s"] = (_OFDM_N[pl.name] / (fs / multistage_rate(fs, meta, pl))
                              * 4.0)
    if pl.name == "80211a" and rs:
        params["cp_lag_s"] = 64.0 * 4.0 / (fs / multistage_rate(fs, meta, pl))
    return {"cls": pl.cls, "family": c.family,
            "bandwidth_hz": float(hi - lo), "symbol_rate_hz": float(rs),
            "carrier_offset_hz": float(meta["center_freq"]), "snr_db": None,
            "sample_start": start, "sample_count": count,
            "f_lo_hz": float(lo), "f_hi_hz": float(hi),
            "bursts": [[start, count]] if count else [],
            "native": "", "fs": float(fs), "generator": GENERATOR,
            "torchsig_class": pl.name,
            "torchsig_snr_db": float(meta.get("snr_db", float("nan"))),
            "torchsig_edges": [ts_lo, ts_hi],
            "impairment_level": int(level),
            "params": _native._jsonable(params), "clipped": False}


def multistage_rate(fs: float, meta: dict, pl: _Plan) -> float:
    """TorchSig's realised resampling ratio for this component."""
    from torchsig.utils.dsp import multistage_polyphase_resampler_actual_rate
    bw = float(meta.get("atk_nominal_bandwidth") or pl.bw)
    return multistage_polyphase_resampler_actual_rate((fs / bw) / 4.0)


def _sample(profile, plans, n, rng, *, num_signals, level, center_range,
            overlap=0.0, duration=None):
    _need()
    prof = _profile(profile)
    fs = float(prof.sample_rate)
    seed = int(rng.integers(0, 2 ** 62))
    ds = _make(fs, n, plans, seed, profile=prof, num_signals=num_signals,
               impairment_level=level, center_range=center_range,
               overlap=overlap, duration=duration)
    sample = next(ds)
    noise = np.asarray(ds._atk_noise, dtype=np.complex128)
    by_cls = {pl.cls: pl for pl in plans}
    return fs, sample, noise, by_cls


def rebuild(noise: np.ndarray, components, gains) -> np.ndarray:
    """TorchSig's sample, rebuilt: its noise realisation plus each placed
    component times its gain. Unit gains reproduce TorchSig's own sample."""
    x = np.array(noise, dtype=np.complex128, copy=True)
    for comp, g in zip(components, gains):
        s = np.asarray(comp.data, dtype=np.complex128)
        a = int(comp["start_in_samples"])
        x[a:a + s.size] += g * s
    return x


# ---------------------------------------------------------------------------
# public generation
# ---------------------------------------------------------------------------
def component(profile, cls: str, n_samples: int, rng: np.random.Generator, *,
              carrier_offset_hz: float = 0.0, params: dict | None = None,
              impairment_level: int = 0) -> tuple[np.ndarray, dict]:
    """One clean TorchSig-made signal of `cls`, n samples at the profile's
    rate, unit power while on, at `carrier_offset_hz` — the TorchSig twin of
    `native.waveform` (the scene composer uses either)."""
    _need()
    prof = _profile(profile)
    fs = float(prof.sample_rate)
    pl = _plan(cls, fs, rng, params)
    if cls == "noise":
        lab = _native.generate("noise", fs, 1, 0.0, np.random.default_rng(0))[1]
        lab.update(sample_count=int(n_samples), generator=GENERATOR)
        return np.zeros(int(n_samples), np.complex64), lab
    half = (pl.bw * (2.0 if pl.name == "chirpss" else 1.0)) / 2.0
    if pl.name != "tone" and abs(carrier_offset_hz) + half > fs / 2.0:
        raise TorchsigRefusal(f"{_classes.get(cls).label} at "
                              f"{carrier_offset_hz / 1e3:+.1f} kHz would reach past "
                              f"the band edge at {_native._rate_words(fs)}")
    off = float(carrier_offset_hz)
    fs_, sample, noise, _ = _sample(prof, [pl], int(n_samples), rng,
                                    num_signals=(1, 1), level=impairment_level,
                                    center_range=(off, off), duration=None
                                    if pl.duration is None else (pl.duration, pl.duration))
    comp = sample.component_signals[0]
    lab = _component_label(comp, pl, fs, int(n_samples), impairment_level)
    x = np.zeros(int(n_samples), dtype=np.complex128)
    s = np.asarray(comp.data, dtype=np.complex128)
    a = lab["sample_start"]
    pw = float(np.mean(np.abs(s) ** 2)) if s.size else 0.0
    x[a:a + s.size] = s / math.sqrt(pw) if pw > 0 else s
    return x.astype(np.complex64), lab


def generate_narrowband(profile, cls: str, n_samples: int, snr_db: float,
                        rng: np.random.Generator, *,
                        carrier_offset_hz: float = 0.0,
                        params: dict | None = None, impairment_level: int = 0,
                        noise_dbfs: float = _native.DEFAULT_NOISE_DBFS
                        ) -> tuple[np.ndarray, dict]:
    """One signal (TorchSig num_signals = 1) of `cls` at the profile's rate,
    in TorchSig's own noise realisation scaled to `noise_dbfs`, re-levelled
    to `snr_db` by the toolkit's SNR definition. The noise class is TorchSig's
    noise floor alone."""
    _need()
    prof = _profile(profile)
    fs = float(prof.sample_rate)
    n = int(n_samples)
    pn = 10.0 ** (float(noise_dbfs) / 10.0)
    pl = _plan(cls, fs, rng, params)
    if cls == "noise":
        fs_, sample, noise, _ = _sample(prof, [_plan("spur", fs, rng, None)], n, rng,
                                        num_signals=(0, 0), level=0,
                                        center_range=(0.0, 0.0))
        x = noise * math.sqrt(pn / float(np.mean(np.abs(noise) ** 2)))
        lab = _native.generate("noise", fs, 1, 0.0, np.random.default_rng(0))[1]
        lab.update(sample_count=n, generator=GENERATOR, noise_dbfs=float(noise_dbfs))
        return x.astype(np.complex64), lab
    off = float(carrier_offset_hz)
    half = (pl.bw * (2.0 if pl.name == "chirpss" else 1.0)) / 2.0
    if pl.name != "tone" and abs(off) + half > fs / 2.0:
        raise TorchsigRefusal(f"{_classes.get(cls).label} at {off / 1e3:+.1f} kHz "
                              f"would reach past the band edge at "
                              f"{_native._rate_words(fs)}")
    fs_, sample, noise, _ = _sample(prof, [pl], n, rng, num_signals=(1, 1),
                                    level=impairment_level, center_range=(off, off),
                                    duration=None if pl.duration is None
                                    else (pl.duration, pl.duration))
    comp = sample.component_signals[0]
    lab = _component_label(comp, pl, fs, n, impairment_level)
    s = np.asarray(comp.data, dtype=np.complex128)
    pw = float(np.mean(np.abs(s) ** 2))
    g_n = math.sqrt(pn / float(np.mean(np.abs(noise) ** 2)))
    a = _native.signal_amplitude(snr_db, lab["bandwidth_hz"], pn, fs)
    x = rebuild(noise * g_n, [comp], [a / math.sqrt(pw)])
    lab["snr_db"] = float(snr_db)
    lab["noise_dbfs"] = float(noise_dbfs)
    return x.astype(np.complex64), lab


def generate_wideband(profile, n_samples: int, rng: np.random.Generator, *,
                      classes: list[str] | None = None, num_signals=(1, 6),
                      snr_range=(5.0, 30.0), impairment_level: int = 0,
                      noise_dbfs: float = _native.DEFAULT_NOISE_DBFS,
                      overlap_probability: float = 0.0, center_hz: float = 0.0,
                      params: dict | None = None
                      ) -> tuple[np.ndarray, list[dict], list]:
    """A TorchSig wideband sample (num_signals components placed by TorchSig
    at random in time and frequency) at the profile's rate, each component
    re-levelled to an SNR drawn uniformly from `snr_range` (the toolkit's
    definition) and boxed at its 99 % band. Classes TorchSig cannot make at
    this rate are left out and named in the first label's 'skipped'. Returns
    (x, labels, SigMF annotations)."""
    _need()
    prof = _profile(profile)
    fs = float(prof.sample_rate)
    n = int(n_samples)
    pn = 10.0 ** (float(noise_dbfs) / 10.0)
    want = list(classes) if classes else [c.name for c in _classes.CLASSES
                                         if c.name not in ("noise",)]
    plans, skipped = [], {}
    for cls in want:
        try:
            pl = _plan(cls, fs, rng, (params or {}).get(cls))
        except TorchsigRefusal as e:
            skipped[cls] = str(e)
            continue
        if pl.name:
            if pl.duration is None and pl.recipe and pl.recipe.burst_s:
                pl.duration = int(round(pl.recipe.burst_s * fs))
            plans.append(pl)
    if not plans:
        raise TorchsigRefusal("none of the requested classes can be made by "
                              f"TorchSig at {_native._rate_words(fs)}: "
                              + "; ".join(f"{k}: {v}" for k, v in skipped.items()))
    # TorchSig 2.2.0 draws every component's centre from ONE dataset-wide
    # range; shrinking it to fit the widest class would cram the narrow ones
    # together. So the range is wide and a component that crosses the band
    # edge is cut by TorchSig's own up-conversion anti-alias filter — its box
    # is then the part that is in the band, and its label says 'clipped'.
    fs_, sample, noise, by_cls = _sample(
        prof, plans, n, rng, num_signals=num_signals, level=impairment_level,
        center_range=(-0.4 * fs, 0.4 * fs), overlap=overlap_probability,
        duration=(int(0.3 * n), n))
    comps = list(sample.component_signals)
    g_n = math.sqrt(pn / float(np.mean(np.abs(noise) ** 2)))
    labels, gains = [], []
    for comp in comps:
        pl = by_cls[str(comp["class_name"])]
        lab = _component_label(comp, pl, fs, n, impairment_level)
        snr = float(rng.uniform(*snr_range))
        s = np.asarray(comp.data, dtype=np.complex128)
        pw = float(np.mean(np.abs(s) ** 2))
        a = _native.signal_amplitude(snr, lab["bandwidth_hz"], pn, fs)
        gains.append(a / math.sqrt(pw) if pw > 0 else 0.0)
        lab["snr_db"] = snr
        lab["noise_dbfs"] = float(noise_dbfs)
        lab["clipped"] = bool(lab["f_lo_hz"] < -0.49 * fs or lab["f_hi_hz"] > 0.49 * fs)
        labels.append(lab)
    x = rebuild(noise * g_n, comps, gains)
    if labels and skipped:
        labels[0]["skipped"] = skipped
    anns = [_labels.label_to_annotation(lab, center_hz, generator=GENERATOR)
            for lab in labels]
    return x.astype(np.complex64), labels, anns
