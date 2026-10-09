# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The fingerprint track's first experiment, in simulation (plan C, C1-C2).

The plan: *"two handheld radios of the same model, cabled; can the system
tell them apart at the SNRs the field gives?"* This is that experiment with
the radios simulated, so the whole code path runs here and the same
functions take Bill's cabled captures (`run(..., bursts=...)`) on his
machine.

THE RADIOS. `RadioModel` is one handheld's hardware: its crystal (carrier
offset in ppm, the symbol clock on the same reference), its I/Q modulator
(gain and phase imbalance, LO leakage), its power amplifier (Rapp AM/AM
smoothness, AM/PM), its key-up (the PA ramp's speed and damping, the
synthesiser's settling), its phase noise. `same_model_pair` draws two
radios whose differences are the size real units of one model show —
fractions of a ppm, tenths of a dB, a millisecond of ramp.

THE CHANNEL. Every burst goes through a fresh random channel: Rician
fading (a new gain and phase per burst), a short multipath tail, Doppler,
and noise at the field SNRs 5 / 10 / 15 / 20 / 30 dB — the confound the
plan names; the receiver is held constant.

THE SCORE. Enrol each radio from a few bursts; on held-out bursts at each
SNR: the classical library's accuracy (`fingerprint.library`) and, with
PyTorch, the CNN's (`learn.fingerprint`) on raw IQ and on IQ denoised first
(C2, with an injected denoiser); and a third radio of the same model that
was never enrolled — how often it is called UNKNOWN. Report: markdown +
JSON under the profile's `runs\\fingerprint_eval\\`.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SNRS = (5.0, 10.0, 15.0, 20.0, 30.0)
DEFAULT_PROFILE = "hackrf_8000000_ci8"


@dataclass
class RadioModel:
    name: str
    cfo_ppm: float = 0.0
    iq_gain_db: float = 0.0
    iq_phase_deg: float = 0.0
    lo_leak_db: float = -35.0
    pa_smoothness: float = 3.0
    pa_backoff_db: float = 1.0
    am_pm_deg: float = 3.0
    ramp_ms: float = 1.0
    ramp_damping: float = 0.6
    pll_offset_hz: float = 600.0
    pll_tau_ms: float = 2.0
    linewidth_hz: float = 20.0
    clock_ppm: float | None = None


def same_model_pair(seed: int = 0, n: int = 2) -> list[RadioModel]:
    """n radios of one model: the same nominal design, unit-to-unit spread."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        out.append(RadioModel(
            name=f"radio-{chr(65 + i)}",
            cfo_ppm=float(rng.uniform(-2.0, 2.0)),
            iq_gain_db=float(rng.uniform(0.1, 0.8)),
            iq_phase_deg=float(rng.uniform(0.5, 3.0)),
            lo_leak_db=float(rng.uniform(-38, -28)),
            pa_smoothness=float(rng.uniform(2.0, 4.0)),
            pa_backoff_db=float(rng.uniform(0.5, 2.0)),
            am_pm_deg=float(rng.uniform(1.0, 5.0)),
            ramp_ms=float(rng.uniform(0.6, 1.6)),
            ramp_damping=float(rng.uniform(0.45, 0.9)),
            pll_offset_hz=float(rng.uniform(-900, 900)),
            pll_tau_ms=float(rng.uniform(1.0, 3.5)),
            linewidth_hz=float(rng.uniform(5, 40))))
    return out


def _c4fm(rng, n_sym, sps, dev_hz, fs):
    sym = rng.choice([-3.0, -1.0, 1.0, 3.0], n_sym)
    f = np.repeat(sym, sps) * dev_hz / 3.0
    k = np.hanning(sps + 1)
    k /= k.sum()
    return np.convolve(f, k, mode="same")


def simulate_burst(radio: RadioModel, rng, *, fs: float = 48_000.0,
                   center_hz: float = 446.0e6, duration_s: float = 0.2,
                   symbol_rate: float = 4800.0, modulation: str = "c4fm",
                   snr_db: float = 20.0, lead_s: float = 0.02,
                   fading_k_db: float = 6.0, doppler_hz: float = 10.0,
                   multipath: float = 0.15) -> np.ndarray:
    """One key-up of `radio` through a fresh random channel, with noise
    before the burst (so the transient and the floor are in the capture)."""
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    sps = max(2, int(round(fs / symbol_rate)))
    if modulation == "c4fm":
        clk = radio.clock_ppm if radio.clock_ppm is not None else radio.cfo_ppm
        sps_true = fs / (symbol_rate * (1 + clk * 1e-6))
        n_sym = int(n / sps_true) + 2
        sym_f = _c4fm(rng, n_sym, sps, 1800.0, fs)
        # resample the symbol stream onto the transmitter's own clock
        src = np.arange(sym_f.size) * (sps_true / sps)
        f_mod = np.interp(np.arange(n), src, sym_f)
    elif modulation == "carrier":
        f_mod = np.zeros(n)
    elif modulation == "fm":
        f_mod = 2500.0 * np.sin(2 * math.pi * 1000.0 * t + rng.uniform(0, 6.28))
    else:
        raise ValueError("modulation is 'c4fm', 'fm' or 'carrier'")
    f_pll = radio.pll_offset_hz * np.exp(-t / (radio.pll_tau_ms * 1e-3))
    phase = 2 * math.pi * np.cumsum(f_mod + f_pll) / fs
    phase += np.cumsum(rng.normal(0, math.sqrt(2 * math.pi * radio.linewidth_hz / fs), n))
    s = np.exp(1j * phase)
    # the I/Q modulator: y = K1 s + K2 s*, plus LO leakage
    g = 10 ** (radio.iq_gain_db / 20)
    ph = math.radians(radio.iq_phase_deg)
    k1 = (1 + g * np.exp(-1j * ph)) / 2
    k2 = (1 - g * np.exp(1j * ph)) / 2
    y = k1 * s + k2 * np.conj(s) + 10 ** (radio.lo_leak_db / 20) * np.exp(1j * 0.7)
    # the key-up ramp: a second-order step response
    wn = 2.2 / (radio.ramp_ms * 1e-3)
    z = radio.ramp_damping
    if z < 1:
        wd = wn * math.sqrt(1 - z * z)
        ramp = 1 - np.exp(-z * wn * t) * (np.cos(wd * t) + z / math.sqrt(1 - z * z)
                                           * np.sin(wd * t))
    else:
        ramp = 1 - (1 + wn * t) * np.exp(-wn * t)
    y = y * ramp
    # the PA: Rapp AM/AM and a little AM/PM
    sat = 10 ** (radio.pa_backoff_db / 20)
    a = np.abs(y)
    p = radio.pa_smoothness
    a_out = a / (1 + (a / sat) ** (2 * p)) ** (1 / (2 * p))
    y = a_out * np.exp(1j * (np.angle(y) + math.radians(radio.am_pm_deg) * (a / sat) ** 2))
    # the crystal
    y = y * np.exp(2j * math.pi * radio.cfo_ppm * 1e-6 * center_hz * t)
    # the channel: Rician gain, Doppler, a short multipath tail
    k = 10 ** (fading_k_db / 10)
    hlos = math.sqrt(k / (k + 1))
    hsc = math.sqrt(1 / (k + 1)) * (rng.normal() + 1j * rng.normal()) / math.sqrt(2)
    h = (hlos * np.exp(1j * rng.uniform(0, 2 * math.pi)) + hsc)
    y = h * y * np.exp(2j * math.pi * rng.uniform(-doppler_hz, doppler_hz) * t)
    if multipath > 0:
        taps = np.array([1.0, multipath * (rng.normal() + 1j * rng.normal()) / math.sqrt(2),
                         0.5 * multipath * (rng.normal() + 1j * rng.normal()) / math.sqrt(2)])
        y = np.convolve(y, taps)[:n]
    lead = int(lead_s * fs)
    sig = np.concatenate([np.zeros(lead, complex), y, np.zeros(lead // 2, complex)])
    p_sig = float(np.mean(np.abs(y[n // 4:]) ** 2))
    noise = math.sqrt(p_sig / 10 ** (snr_db / 10) / 2)
    sig = sig + noise * (rng.normal(size=sig.size) + 1j * rng.normal(size=sig.size))
    return sig.astype(np.complex64)


def bursts(radios, per_radio: int, snrs=SNRS, seed: int = 0, **kw):
    """[(iq, radio index, snr)], each through its own random channel."""
    rng = np.random.default_rng(seed)
    out = []
    for i, r in enumerate(radios):
        for k in range(per_radio):
            snr = float(snrs[k % len(snrs)])
            out.append((simulate_burst(r, rng, snr_db=snr, **kw), i, snr))
    return out


# ---------------------------------------------------------------------------
def classical(radios, enrol, test, unknown, *, rf, profile: str, fs: float,
              center_hz: float, symbol_rate: float | None) -> dict:
    """Enrol, calibrate the UNKNOWN threshold, score per SNR."""
    from atk_diffusion.fingerprint import features as F
    from atk_diffusion.fingerprint.library import UNKNOWN, EmitterLibrary
    lib = EmitterLibrary(rf)
    ids = {}

    def fp(x):
        return F.extract(x, fs, center_hz, symbol_rate=symbol_rate,
                         profile=profile, rx_ppm=0.0)
    # Every third enrolment burst is HELD BACK to set the threshold: a
    # threshold set on bursts the library was built from is set on
    # distances no new burst will ever be that close to (2026-10-09: the
    # first build did that and called every test burst UNKNOWN).
    fit = [b for k, b in enumerate(enrol) if k % 3 != 1]
    held = [b for k, b in enumerate(enrol) if k % 3 == 1]
    for x, i, _ in fit:
        ids[i] = lib.add(fp(x), ids.get(i), name=radios[i].name)
    held_known = [(fp(x), ids[i]) for x, i, _ in held if i in ids]
    cal = lib.calibrate_threshold(held_known, [fp(x) for x, _, _ in unknown[::2]])
    per = {}
    for snr in sorted({s for _, _, s in test}):
        sub = [(x, i) for x, i, s in test if s == snr]
        right = sum(lib.match(fp(x)).emitter_id == ids[i] for x, i in sub)
        un = [x for x, _, s in unknown if s == snr]
        rej = sum(lib.match(fp(x)).emitter_id == UNKNOWN for x in un)
        per[f"{snr:g}"] = {"accuracy": right / max(len(sub), 1),
                           "unknown_rejected": rej / max(len(un), 1),
                           "n": len(sub)}
    lib.save()
    return {"per_snr": per, "calibration": cal,
            "library_dir": str(lib.dir), "emitters": len(lib.emitters)}


def run(rf, *, profile: str = DEFAULT_PROFILE, seed: int = 0,
        enrol_per_radio: int = 15, test_per_radio: int = 25,
        learned: bool = True, denoiser=None, fs: float = 48_000.0,
        center_hz: float = 446.0e6, symbol_rate: float = 4800.0,
        cnn_steps: int = 300, progress=None) -> dict:
    """The experiment (see the module docstring). Writes the report."""
    say = progress or (lambda s: None)
    radios = same_model_pair(seed, 3)
    known, stranger = radios[:2], radios[2]
    kw = {"fs": fs, "center_hz": center_hz, "symbol_rate": symbol_rate}
    enrol = bursts(known, enrol_per_radio, seed=seed + 1, **kw)
    test = bursts(known, test_per_radio, seed=seed + 2, **kw)
    unknown = [(x, 99, s) for x, _, s in
               bursts([stranger], test_per_radio, seed=seed + 3, **kw)]
    say("fingerprint: classical features and the library")
    result = {"experiment": "C1/C2 same-model radios", "profile": profile,
              "radios": [asdict(r) for r in radios], "snrs": list(SNRS),
              "classical": classical(known, enrol, test, unknown, rf=rf,
                                     profile=profile, fs=fs, center_hz=center_hz,
                                     symbol_rate=symbol_rate)}
    if learned:
        try:
            import importlib.util
            if importlib.util.find_spec("torch") is None:
                raise ImportError("PyTorch is not in this environment")
            from atk_diffusion.learn import fingerprint as LF
            say("fingerprint: the channel-resilient CNN")
            train = bursts(known, enrol_per_radio * 4, seed=seed + 4, **kw)
            d = LF.train([x for x, _, _ in train], [i for _, i, _ in train],
                         Path(rf.models(profile)) / f"fingerprint-sim-{seed}",
                         profile=profile, class_names=[r.name for r in known],
                         steps=cnn_steps, seed=seed,
                         heldout=([x for x, _, _ in enrol], [i for _, i, _ in enrol]),
                         unknown=[x for x, _, _ in unknown[::2]])
            model = LF.load(d, profile)
            result["cnn"] = LF.evaluate(model, [x for x, _, _ in test],
                                        [i for _, i, _ in test],
                                        [s for _, _, s in test],
                                        unknown=[x for x, _, _ in unknown],
                                        unknown_snr=[s for _, _, s in unknown])
            if denoiser is not None:
                result["cnn_denoised_first"] = LF.evaluate(
                    model, [x for x, _, _ in test], [i for _, i, _ in test],
                    [s for _, _, s in test], denoiser=denoiser,
                    unknown=[x for x, _, _ in unknown],
                    unknown_snr=[s for _, _, s in unknown])
            result["cnn_model_dir"] = str(d)
        except ImportError as e:
            result["cnn"] = {"skipped": str(e)}
    result["report"] = write_report(rf, result, profile=profile)
    return result


def write_report(rf, result: dict, *, profile: str = DEFAULT_PROFILE) -> dict:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    d = Path(rf.runs(profile)) / "fingerprint_eval" / stamp
    d.mkdir(parents=True, exist_ok=True)
    jp = d / "result.json"
    jp.write_text(json.dumps({k: v for k, v in result.items() if k != "report"},
                             indent=2, default=str), encoding="utf-8")
    lines = ["# Same-model radios: can the system tell them apart?", "",
             f"Run {stamp} (UTC), simulated radios through random channels "
             "(Rician fading, multipath, Doppler). Numbers from result.json.", "",
             "| SNR dB | classical accuracy | classical UNKNOWN rejection | "
             "CNN accuracy | CNN UNKNOWN rejection | CNN after denoising |",
             "|---|---|---|---|---|---|"]
    cl = result["classical"]["per_snr"]
    cnn = result.get("cnn", {}).get("per_snr", {})
    den = result.get("cnn_denoised_first", {}).get("per_snr", {})
    for snr, row in cl.items():
        c = cnn.get(snr, {})
        dn = den.get(snr, {})
        lines.append(f"| {snr} | {row['accuracy']:.0%} | {row['unknown_rejected']:.0%} | "
                     + (f"{c['accuracy']:.0%} | {c['unknown_rejected']:.0%} | "
                        if c else "— | — | ")
                     + (f"{dn['accuracy']:.0%} |" if dn else "— |"))
    if "skipped" in result.get("cnn", {}):
        lines += ["", f"CNN: {result['cnn']['skipped']}"]
    lines += ["", "Chance is 50 % for two radios. A match is a PROPOSAL; only a "
              "decode of the radio's own identity confirms it."]
    mp = d / "report.md"
    mp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for p in (jp, mp):
        rf.record(p, "experiment", "fingerprint_eval")
    return {"dir": str(d), "json": str(jp), "markdown": str(mp)}
