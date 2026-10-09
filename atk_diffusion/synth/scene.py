# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The scene composer: a wideband capture of a region nobody has recorded
yet, at a receiver profile's exact rate (plan §3.6; DETECTION_DESIGN §6 step
2; ARCHITECTURE §4.3).

    x, annotations = compose_scene(env, profile, center_hz, duration_s, rng,
                                   generator="native" | "torchsig")

Bill, 2026-10-08: *"That way you could train without going out to that exact
region with your equipment and loitering when you don't need to."* So:

1. THE RIGHT BANDS, THE RIGHT KINDS OF SIGNAL. Every allocation of the
   environment profile that overlaps the receiver's span gets channels on its
   raster, each active with the allocation's occupancy; an active channel
   carries one of the allocation's classes (by its weights) at an SNR drawn
   from the allocation's range. Cellular downlink bands get LTE/NR carriers of
   standard widths at the band's occupancy, placed once per technology over
   the union of overlapping bands (B2 and B25 are the same spectrum). A
   carrier wider than the span is rendered as the slice a receiver tuned here
   would see (`native.waveform(clip=True)`). Only classes the receiver family
   can capture (`detect.classes.for_profile_family`) are placed.
2. ACTIVITY. Broadcast, cellular, TV and jammers are on for the whole scene;
   push-to-talk and pager transmissions key up and down inside it; ADS-B, BLE,
   Wi-Fi and LoRa come as bursts.
3. THE TERRAIN'S CHANNEL (environments.TERRAIN_CHANNELS): a tapped delay
   line with an exponential power-delay profile of the terrain's rms delay
   spread, a Rician first tap (K-factor) where there is line of sight,
   Rayleigh otherwise, every tap fading with a Jakes (Clarke) Doppler spectrum
   whose maximum is v·f/c for the emitter's mobility at its RF frequency, the
   line of sight shifted by its own Doppler. The channel has unit mean power
   gain, so the label's SNR is the MEAN SNR over the fading.
4. THE RECEIVER. White noise at the profile's measured floor level (−30 dBFS
   when unmeasured), an AGC step if the scene would overload the converter
   (total power above −12 dBFS: everything is scaled down together, as a
   receiver's AGC would, and the scene says by how much), then the profile's
   MEASURED impairments (`dsp.impair.apply_impairments`: floor shape, spurs,
   I/Q imbalance, DC, quantisation in the profile's own datatype) when they
   have been measured. Unmeasured, the scene says it used a textbook receiver.

Annotations carry the label (DETECTION_DESIGN §10) plus atk:environment, the
emitter's mobility and Doppler; TDMA bursts and squitters get one annotation
per burst (the box a detector should draw).

LIMITS, stated: occupancy and SNR are a prior (environments.CAVEAT); no
propagation model places emitters in distance or terrain (E5 owns that); the
channel is the terrain class's statistics, not this valley's; co-channel
collisions are allowed where the prior puts two emitters on one channel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from atk_diffusion import profiles as _profiles
from atk_diffusion.detect import classes as _classes
from atk_diffusion.dsp import impair as _impair
from atk_diffusion.synth import environments as _env
from atk_diffusion.synth import labels as _labels
from atk_diffusion.synth import native as _native

C_LIGHT = 299_792_458.0
AGC_CEILING_DBFS = -12.0
MAX_EMITTERS = 48

#: receiver family -> tuning range (Hz), from the devices' data sheets; ATK's
#: own device tables win where they differ.
FAMILY_TUNING = {
    "rtlsdr": (24e6, 1766e6), "krakensdr": (24e6, 1766e6),
    "hackrf": (1e6, 6000e6), "bladerf1": (300e6, 3800e6),
    "bladerf2": (47e6, 6000e6), "airspy": (24e6, 1800e6),
    "kiwisdr": (10e3, 30e6), "spyserver": (0.0, 6000e6),
    "esp32csi": (2400e6, 2500e6), "sigmf-import": (0.0, 1e12),
}

#: class -> how it is on the air
CLASS_ACTIVITY = {
    "fm_broadcast": "continuous", "noaa_wx": "continuous", "atsc": "continuous",
    "lte_dl": "continuous", "nr_dl": "continuous", "gnss_jamming": "continuous",
    "drone_fpv_analog": "continuous", "drone_digital": "continuous",
    "spur": "continuous", "dc_spike": "continuous",
    "nfm_voice": "ptt", "p25": "ptt", "dmr": "ptt", "nxdn96": "ptt",
    "nxdn48": "ptt", "pocsag": "ptt", "flex": "ptt",
    "adsb": "burst", "ble": "burst", "wifi_24": "burst", "lora": "burst",
}

LTE_CARRIERS = ((100, 20e6), (75, 15e6), (50, 10e6), (25, 5e6))
NR_CARRIERS = ((106, 40e6), (51, 20e6), (24, 10e6))


@dataclass
class Emitter:
    cls: str
    offset_hz: float              # carrier offset from the scene centre
    rf_hz: float                  # absolute
    snr_db: float
    start: int                    # first sample of its window in the scene
    count: int                    # window length
    mobility: str = "static"
    speed_mps: float = 0.0
    params: dict = field(default_factory=dict)
    service: str = ""


@dataclass
class SceneResult:
    x: np.ndarray
    annotations: list
    labels: list
    info: dict


# ---------------------------------------------------------------------------
# where to tune, and what is on the air there
# ---------------------------------------------------------------------------
def _profile(profile) -> _profiles.ReceiverProfile:
    if isinstance(profile, _profiles.ReceiverProfile):
        return profile
    if isinstance(profile, str):
        return _profiles.new_profile(profile)
    raise TypeError("a scene is composed for a receiver profile (a "
                    "ReceiverProfile or its id)")


def capturable(profile) -> set:
    prof = _profile(profile)
    return {c.name for c in _classes.for_profile_family(prof.pid.family)}


def choose_center(env: _env.EnvironmentProfile, profile, rng,
                  classes: list | None = None) -> float:
    """A tune frequency inside the receiver's range whose span overlaps an
    allocation (or cellular band) carrying a class this receiver captures."""
    prof = _profile(profile)
    lo, hi = FAMILY_TUNING.get(prof.pid.family, (0.0, 1e12))
    cap = capturable(prof) if classes is None else set(classes) & capturable(prof)
    cands = []
    for a in env.allocations:
        if set(a.classes) & cap and a.f_hi_hz > lo and a.f_lo_hz < hi:
            cands.append((max(a.f_lo_hz, lo), min(a.f_hi_hz, hi)))
    for b in env.cellular_bands:
        cls = "lte_dl" if b.technology == "lte" else "nr_dl"
        if cls in cap and b.dl_hi_hz > lo and b.dl_lo_hz < hi:
            cands.append((max(b.dl_lo_hz, lo), min(b.dl_hi_hz, hi)))
    if not cands:
        raise _env.EnvironmentError(
            f"nothing in {env.region} that {_profiles.describe(prof.id)} can "
            f"capture lies inside its tuning range ({lo / 1e6:g}–{hi / 1e6:g} MHz)")
    a, b = cands[int(rng.integers(0, len(cands)))]
    return float(rng.uniform(a, b))


def _merge(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _activity(cls: str, n: int, fs: float, rng) -> tuple[int, int]:
    """(start, count) of a transmission inside an n-sample scene: whole for
    continuous classes and bursts (placed inside by the generator); a
    push-to-talk or pager transmission is keyed through the scene 60 % of the
    time, otherwise it keys up and/or down inside it (at least 50 ms long)."""
    if CLASS_ACTIVITY.get(cls, "continuous") != "ptt":
        return 0, n
    min_len = max(1, int(0.05 * fs))
    if n <= 2 * min_len or rng.uniform() < 0.6:
        return 0, n
    a = int(rng.integers(0, n - min_len))
    b = int(rng.integers(a + min_len, n + 1))
    return a, b - a


def plan_scene(env: _env.EnvironmentProfile, profile, center_hz: float,
               duration_s: float, rng, max_emitters: int = MAX_EMITTERS
               ) -> list[Emitter]:
    """The emitters a scene will hold (nothing generated yet)."""
    prof = _profile(profile)
    fs = float(prof.sample_rate)
    n = int(round(float(duration_s) * fs))
    lo, hi = center_hz - 0.5 * fs, center_hz + 0.5 * fs
    cap = capturable(prof)
    out: list[Emitter] = []

    def add(cls, f, snr, mob, service, params=None):
        a, k = _activity(cls, n, fs, rng)
        speed = float(rng.uniform(*_env.MOBILITY_SPEED.get(mob, (0.0, 0.0))))
        out.append(Emitter(cls, float(f - center_hz), float(f), float(snr), a, k,
                           mob, speed, dict(params or {}), service))

    for al in env.allocations_in(lo, hi):
        classes = [c for c in al.classes if c in cap]
        if not classes:
            continue
        w = np.array([float(al.weights.get(c, 1.0)) for c in classes])
        w = w / w.sum()
        if al.channel_hz > 0:
            n_ch = int((al.f_hi_hz - al.f_lo_hz) // al.channel_hz)
            centers = al.f_lo_hz + al.channel_hz / 2.0 + al.channel_hz * np.arange(n_ch)
            # channels whose band reaches into the span (a TV channel centred
            # just outside an RTL's span still shows its edge)
            vis = centers[(centers + al.channel_hz / 2 > lo)
                          & (centers - al.channel_hz / 2 < hi)]
            active = vis[rng.uniform(size=vis.size) < al.occupancy]
        else:
            c0 = 0.5 * (al.f_lo_hz + al.f_hi_hz)
            k = int(rng.poisson(al.occupancy * 20.0)) if lo < c0 < hi else 0
            active = np.full(k, c0)
        for f in active:
            cls = str(rng.choice(classes, p=w))
            params = {}
            if cls == "adsb":
                params = {"squitters": int(rng.integers(1, 5))}
            elif cls == "ble":
                params = {"packets": int(rng.integers(1, 4))}
            elif cls == "wifi_24":
                params = {"frames": int(rng.integers(1, 4))}
            add(cls, f, rng.uniform(*al.snr_db), al.mobility, al.service, params)
    # cellular: carriers over the union of each technology's downlink bands
    placed: list[tuple[float, float]] = []
    for tech, cls, menu in (("lte", "lte_dl", LTE_CARRIERS),
                            ("nr", "nr_dl", NR_CARRIERS)):
        if cls not in cap:
            continue
        bands = [b for b in env.cellular_bands if b.technology == tech]
        occ = float(np.mean([b.occupancy for b in bands])) if bands else 0.0
        for a, b in _merge([(x.dl_lo_hz, x.dl_hi_hz) for x in bands]):
            if b <= lo - 50e6 or a >= hi + 50e6:
                continue
            width = b - a
            fits = [(r, w) for r, w in menu if w <= width]
            if not fits:
                continue
            budget = occ * width
            tries = 0
            while budget > 0 and tries < 40:
                tries += 1
                r, w = fits[int(rng.integers(0, len(fits)))]
                f0 = a + w / 2 + rng.uniform(0, width - w)
                f0 = round(f0 / 100e3) * 100e3            # the 100 kHz raster
                if any(f0 - w / 2 < q and f0 + w / 2 > p for p, q in placed):
                    continue
                placed.append((f0 - w / 2, f0 + w / 2))
                budget -= w
                if f0 + w / 2 > lo and f0 - w / 2 < hi:
                    add(cls, f0, rng.uniform(5.0, 25.0), "static", f"{tech} downlink",
                        {"n_rb": r})
    for it in env.interference:
        for cls in it.get("classes", []):
            if cls not in cap or any(cls in al.classes for al in env.allocations):
                continue
            for _ in range(int(rng.poisson(float(it.get("occupancy", 0)) * 20.0))):
                add(cls, rng.uniform(lo + 0.02 * fs, hi - 0.02 * fs),
                    rng.uniform(5.0, 30.0), "static", it.get("kind", "interference"))
    if len(out) > max_emitters:
        keep = rng.choice(len(out), max_emitters, replace=False)
        out = [out[i] for i in sorted(keep)]
    return out


# ---------------------------------------------------------------------------
# the channel
# ---------------------------------------------------------------------------
def _jakes(t: np.ndarray, f_d: float, rng, m: int = 16) -> np.ndarray:
    """Unit-power complex Gaussian fading process with Clarke's spectrum
    (sum of m sinusoids at random arrival angles)."""
    if f_d <= 0:
        return np.full(t.size, np.exp(1j * rng.uniform(0, 2 * np.pi)))
    alpha = rng.uniform(0, 2 * np.pi, m)
    phi = rng.uniform(0, 2 * np.pi, m)
    g = np.exp(1j * (2 * np.pi * f_d * np.cos(alpha)[:, None] * t[None, :]
                     + phi[:, None]))
    return g.sum(axis=0) / math.sqrt(m)


def apply_channel(s: np.ndarray, fs: float, model: dict, f_d: float, rng
                  ) -> tuple[np.ndarray, float]:
    """Pass `s` through the terrain's channel with maximum Doppler `f_d`.
    Returns (y, line-of-sight Doppler in Hz). Unit mean power gain."""
    s = np.asarray(s, dtype=np.complex128)
    n = s.size
    if n == 0:
        return s, 0.0
    tau = float(model.get("rms_delay_s", 0.0) or 0.0)
    kdb = model.get("k_factor_db")
    k_lin = 0.0 if kdb is None else 10.0 ** (float(kdb) / 10.0)
    # taps: exponential power-delay profile, at least a sample apart
    if tau * fs < 0.5:
        delays = np.array([0])
        powers = np.array([1.0])
    else:
        step = max(1, int(round(0.5 * tau * fs)))
        delays = np.arange(0, int(4 * tau * fs) + 1, step)
        powers = np.exp(-delays / (tau * fs))
        if model.get("second_cluster_s"):
            d2 = int(round(float(model["second_cluster_s"]) * fs))
            delays = np.append(delays, d2)
            powers = np.append(powers, powers[0] * 10 ** (float(
                model.get("second_cluster_db", -10.0)) / 10.0))
    powers = powers / powers.sum()
    # slow fading: evaluate on a coarse grid, interpolate
    T = n / fs
    npts = int(min(n, max(64, math.ceil(T * f_d * 20))))
    tg = np.linspace(0, T, npts)
    ts = np.arange(n) / fs
    y = np.zeros(n, dtype=np.complex128)
    los_dop = 0.0
    for i, (d, pw) in enumerate(zip(delays, powers)):
        g = _jakes(tg, f_d, rng)
        if i == 0 and k_lin > 0:
            los_dop = float(f_d * math.cos(rng.uniform(0, 2 * np.pi)))
            los = np.exp(1j * (2 * np.pi * los_dop * tg + rng.uniform(0, 2 * np.pi)))
            g = math.sqrt(k_lin / (k_lin + 1)) * los + math.sqrt(1 / (k_lin + 1)) * g
        h = np.interp(ts, tg, g.real) + 1j * np.interp(ts, tg, g.imag)
        d = int(d)
        if d >= n:
            continue
        y[d:] += math.sqrt(pw) * h[d:] * s[:n - d]
    return y, los_dop


# ---------------------------------------------------------------------------
# composing
# ---------------------------------------------------------------------------
def _render(em: Emitter, prof, fs: float, rng, generator: str):
    """(clean waveform, label, generator words) for one emitter."""
    if generator == "torchsig":
        try:
            from atk_diffusion.synth import torchsig_backend as _ts
            ok, why = _ts.available()
            if not ok:
                raise _ts.TorchsigUnavailable(why)
            p = {k: v for k, v in em.params.items() if k in ("bw", "sf", "frame")}
            s, lab = _ts.component(prof, em.cls, em.count, rng,
                                   carrier_offset_hz=em.offset_hz, params=p)
            return s, lab, _ts.GENERATOR
        except Exception as e:                            # noqa: BLE001
            note = f"native (TorchSig could not: {e})"
            s, lab = _native.waveform(em.cls, fs, em.count, rng,
                                      carrier_offset_hz=em.offset_hz,
                                      params=em.params or None, clip=True)
            return s, lab, note[:300]
    s, lab = _native.waveform(em.cls, fs, em.count, rng,
                              carrier_offset_hz=em.offset_hz,
                              params=em.params or None, clip=True)
    return s, lab, _native.GENERATOR


def compose_scene_detailed(env, profile, center_hz: float, duration_s: float,
                           rng: np.random.Generator, generator: str = "native",
                           rf=None, max_emitters: int = MAX_EMITTERS
                           ) -> SceneResult:
    """Everything compose_scene does, plus the labels and a scene report."""
    if generator not in ("native", "torchsig"):
        raise ValueError("generator is 'native' or 'torchsig'")
    env = _env.resolve(rf, env)
    prof = _profile(profile)
    fs = float(prof.sample_rate)
    n = int(round(float(duration_s) * fs))
    if n <= 0:
        raise ValueError("a scene needs a positive duration")
    emitters = plan_scene(env, prof, float(center_hz), duration_s, rng,
                          max_emitters)
    measured = _impair.is_measured(prof.impairments)
    p_floor = _impair.noise_power(prof.impairments if measured else None)
    x = _native.noise(n, p_floor, rng).astype(np.complex128)
    model = env.channel()
    labels, anns, report = [], [], []
    for em in emitters:
        try:
            s, lab, gen_words = _render(em, prof, fs, rng, generator)
        except _native.SynthRefusal as e:
            report.append({"cls": em.cls, "rf_hz": em.rf_hz, "skipped": str(e)})
            continue
        f_d = em.speed_mps * em.rf_hz / C_LIGHT
        y, los = apply_channel(s, fs, model, f_d, rng)   # LOS carries its Doppler
        a = _native.signal_amplitude(em.snr_db, lab["bandwidth_hz"], p_floor, fs)
        x[em.start:em.start + em.count] += a * y[:n - em.start]
        lab = dict(lab)
        lab["snr_db"] = em.snr_db
        lab["f_lo_hz"] += los
        lab["f_hi_hz"] += los
        lab["carrier_offset_hz"] = em.offset_hz + los
        lab["sample_start"] += em.start
        lab["bursts"] = [[s0 + em.start, k] for s0, k in lab.get("bursts", [])]
        labels.append(lab)
        extra = {"atk:mobility": em.mobility, "atk:doppler_hz": round(los, 3),
                 "atk:service": em.service}
        bursts = lab["bursts"] if lab["cls"] in ("dmr", "adsb", "ble", "wifi_24") \
            and len(lab["bursts"]) > 1 else [[lab["sample_start"], lab["sample_count"]]]
        for s0, k in bursts:
            one = dict(lab, sample_start=s0, sample_count=k, bursts=[[s0, k]])
            anns.append(_labels.label_to_annotation(
                one, center_hz, generator=gen_words, environment=env.region,
                extra=extra))
        report.append({"cls": em.cls, "rf_hz": em.rf_hz, "snr_db": em.snr_db,
                       "start": em.start, "count": em.count,
                       "mobility": em.mobility, "doppler_hz": round(los, 3),
                       "generator": gen_words, "clipped": lab.get("clipped", False)})
    x, agc_db, measured = _receive(x, prof, fs, p_floor, rng)
    anns.sort(key=lambda a: a.sample_start)
    info = {"environment": env.region, "profile": prof.id, "center_hz": float(center_hz),
            "sample_rate": fs, "samples": n, "generator": generator,
            "noise_dbfs": 10.0 * math.log10(p_floor), "agc_gain_db": round(agc_db, 2),
            "receiver": _impair.describe(prof.impairments),
            "receiver_impairments_applied": bool(measured),
            "terrain": env.terrain_class, "channel_model": model,
            "emitters": report, "caveat": env.caveat}
    return SceneResult(np.asarray(x, dtype=np.complex64), anns, labels, info)


def compose_scene(env, profile, center_hz: float, duration_s: float,
                  rng: np.random.Generator, generator: str = "native"
                  ) -> tuple[np.ndarray, list]:
    """(x at the profile's exact rate, [sigmf.Annotation]) — see the module
    docstring. `env` is an EnvironmentProfile or a built-in region id."""
    r = compose_scene_detailed(env, profile, center_hz, duration_s, rng, generator)
    return r.x, r.annotations


def _receive(x: np.ndarray, prof, fs: float, p_floor: float, rng) -> tuple:
    """The receiver end of a scene: AGC above AGC_CEILING_DBFS, then the
    profile's measured impairments. Returns (x, agc_db, measured)."""
    measured = _impair.is_measured(prof.impairments)
    p_tot = float(np.mean(np.abs(x) ** 2)) if x.size else 0.0
    agc_db = 0.0
    ceiling = 10.0 ** (AGC_CEILING_DBFS / 10.0)
    if p_tot > ceiling:
        g = math.sqrt(ceiling / p_tot)
        x = x * g
        agc_db = 20.0 * math.log10(g)
    if measured:
        x = _impair.apply_impairments(x, fs, prof.impairments, rng,
                                      adc_bits=prof.adc_bits or None,
                                      datatype=prof.datatype)
    return x, agc_db, measured


def compose_generic(profile, center_hz: float, duration_s: float,
                    rng: np.random.Generator, generator: str = "native",
                    classes: list | None = None, n_emitters=(1, 6),
                    snr_range=(5.0, 30.0)) -> SceneResult:
    """A scene with no environment profile: 1–6 emitters of random classes
    (from `classes`, else every class this receiver family captures) at
    random offsets, SNRs and times; no terrain channel. With
    generator="torchsig" the placement is TorchSig's own wideband sampler
    (`torchsig_backend.generate_wideband`). The receiver end (AGC, measured
    impairments) is the same as compose_scene's."""
    prof = _profile(profile)
    fs = float(prof.sample_rate)
    n = int(round(float(duration_s) * fs))
    if n <= 0:
        raise ValueError("a scene needs a positive duration")
    cap = capturable(prof)
    pool = [c for c in (classes or sorted(cap)) if c in cap
            and c not in ("noise", "dc_spike")]
    if not pool:
        raise ValueError("none of those classes is captured by "
                         f"{_profiles.describe(prof.id)}")
    measured = _impair.is_measured(prof.impairments)
    p_floor = _impair.noise_power(prof.impairments if measured else None)
    report: list = []
    if generator == "torchsig":
        from atk_diffusion.synth import torchsig_backend as _ts
        x, labels, anns = _ts.generate_wideband(
            prof, n, rng, classes=pool, num_signals=tuple(n_emitters),
            snr_range=tuple(snr_range), noise_dbfs=10.0 * math.log10(p_floor),
            center_hz=center_hz)
        x = x.astype(np.complex128)
        for lab in labels:
            report.append({"cls": lab["cls"], "snr_db": lab["snr_db"],
                           "generator": _ts.GENERATOR,
                           "skipped": lab.get("skipped", {})})
    elif generator == "native":
        x = _native.noise(n, p_floor, rng).astype(np.complex128)
        labels, anns = [], []
        k = int(rng.integers(int(n_emitters[0]), int(n_emitters[1]) + 1))
        for _ in range(k):
            cls = str(rng.choice(pool))
            off = float(rng.uniform(-0.45 * fs, 0.45 * fs))
            snr = float(rng.uniform(*snr_range))
            a0, cnt = _activity(cls, n, fs, rng)
            try:
                sg, lab = _native.waveform(cls, fs, cnt, rng,
                                           carrier_offset_hz=off, clip=True)
            except _native.SynthRefusal as e:
                report.append({"cls": cls, "skipped": str(e)})
                continue
            amp = _native.signal_amplitude(snr, lab["bandwidth_hz"], p_floor, fs)
            x[a0:a0 + cnt] += amp * sg
            lab = dict(lab, snr_db=snr, sample_start=lab["sample_start"] + a0,
                       bursts=[[b0 + a0, bk] for b0, bk in lab["bursts"]])
            labels.append(lab)
            bursts = lab["bursts"] if cls in ("dmr", "adsb", "ble", "wifi_24") \
                and len(lab["bursts"]) > 1 else [[lab["sample_start"], lab["sample_count"]]]
            for b0, bk in bursts:
                anns.append(_labels.label_to_annotation(
                    dict(lab, sample_start=b0, sample_count=bk, bursts=[[b0, bk]]),
                    center_hz, generator=_native.GENERATOR))
            report.append({"cls": cls, "snr_db": snr, "start": a0, "count": cnt,
                           "generator": _native.GENERATOR})
    else:
        raise ValueError("generator is 'native' or 'torchsig'")
    x, agc_db, measured = _receive(x, prof, fs, p_floor, rng)
    anns.sort(key=lambda a: a.sample_start)
    info = {"environment": "", "profile": prof.id, "center_hz": float(center_hz),
            "sample_rate": fs, "samples": n, "generator": generator,
            "noise_dbfs": 10.0 * math.log10(p_floor), "agc_gain_db": round(agc_db, 2),
            "receiver": _impair.describe(prof.impairments),
            "receiver_impairments_applied": bool(measured), "terrain": "",
            "channel_model": {}, "emitters": report,
            "caveat": "no environment profile: random classes, offsets and "
                      "times — not a region"}
    return SceneResult(np.asarray(x, dtype=np.complex64), anns, labels, info)
