# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The cyclic proposer — the detector that finds known signals BELOW the
energy floor (DETECTION_DESIGN §3, §4.1; ARCHITECTURE §4.2 `cyclo.proposer`).

    cyclic_proposer(x, fs, center_hz, profile, classes=None, regions=None,
                    t0=0.0, epoch=None, pfa=1e-3) -> list[Detection]

Three proposers because they fail differently (§1): energy catches anything
with power and nothing structured below the floor; THIS one catches signals
below the floor whose symbol rate, chip rate or OFDM prefix it knows from
the class table, and nothing it does not; the learned detector catches
structure and can hallucinate. Its boxes are drawn with the cyclostationary
edge and the α badge (`detect.boxes`).

WHAT IT DOES
  * `regions` given (absolute (f_lo, f_hi) pairs, e.g. CFAR's boxes): each
    region is cut to the profile's canonical rate for its width
    (`dsp.resample.cut_to_canonical`), band-limited to the bandwidth of the
    classes being tested, and probed: `probes.symbol_rate_line` at every
    listed symbol rate the cut can show (`classes.cycle_frequencies`),
    `probes.cp_probe` at every OFDM prefix lag it can show
    (`classes.cp_lags`).
  * `regions=None` — the span is swept on a CHANNEL GRID sized for the
    class bandwidths: channels as wide as each group of classes, half
    overlapping, so any such signal lies whole inside one. The grid is cut
    by a polyphase (weighted overlap-add) channelizer that produces, for
    every channel at once, what `cut_to_canonical` produces for one box —
    shifted to 0 Hz, low-passed to the channel, at the canonical rate —
    because cutting 400 channels one at a time would cost minutes a
    second.

THE FALSE-ALARM RATE IS PER CALL. `pfa` is the probability that one call
returns ANY detection on noise alone — across every region or channel,
every rate and every lag tested (Bonferroni over the probe calls; each
probe is exact within itself). On a 2.4 MS/s span swept for voice-class
signals that is ~800 probe calls, so each runs at pfa/800; the price is a
higher threshold, about 1 dB of SNR. The tests check the rate on noise.

OFDM BY ITS PREFIX. A cyclic-prefix hit (LTE: lag 1/15 kHz = 66.67 µs, its
symbol rate 14 kHz the period of the prefix train) carries the van de Beek
carrier offset referred to the CAPTURE's centre (modulo the subcarrier
spacing — every region and grid channel is cut with its own shift, which is
added back), and `measurements["cp_cells"]`: how many distinct,
non-time-aligned transmitters share the channel (`probes.
cp_timing_phases` — the cell-tower survey; time-synchronised cells count
as one, and the words say so).

THE CLASS IS NEVER GUESSED. 4800 sym/s is P25, DMR and NXDN96 alike: the
detection then carries family "fsk", no class, and
measurements["candidates"] = ["p25", "dmr", "nxdn96"] — the decoder says
which (§5). A rate that is an integer multiple of another detected rate in
the same place is folded into it (a POCSAG 1200 also shows lines at 2400 and
4800; it is one signal).

WHAT IT COSTS (measured, one core, numpy): a 240 kHz span for 5 s swept
for every voice-class rate — ~1 s; a 2.4 MS/s span for 1 s — ~5–8 s for the
voice grid (two channelizer passes, ~770 channels × 6 rates), ~50 ms for
the LTE prefix probe on the whole span. The regions path costs a canonical
cut and a few probes per region — milliseconds. The voice-grid sweep of a
wide span is therefore an ESCALATION over the buffer (`cyclo.escalate`), not
something to run on every live tile; the report says what each call cost.

LIMITS. It finds only the rates it is given. FSK voice (P25/DMR/NXDN) has a
weak symbol-rate line: ~0 dB in-band SNR in 3–5 s, not 10 dB under the
floor (see `probes`). Very short bursts (fewer than ~32 symbols) give no
line at all — those fall to the learned detector.
"""

from __future__ import annotations

import math
import time

import numpy as np

from atk_diffusion import profiles as _profiles
from atk_diffusion.cyclo import probes as _probes
from atk_diffusion.detect import classes as _classes
from atk_diffusion.detect.boxes import Detection

#: Fraction of the span (±) a channel may reach: receivers roll off at the
#: band edges, so the outer 10 % is not swept.
USABLE = 0.45

#: Near misses are reported (for escalation) when a probe call's family
#: p-value is below this.
NEAR_MISS_P = 0.1

#: Grid channels are as wide as the class group and spaced a third of that
#: apart, so a signal anywhere in the span sits within W/6 of some
#: channel's centre and keeps all but its outer edge.
GRID_STEP = 3


def probe_decimation(fs: float, width_hz: float) -> int:
    """The integer decimation at which a channel of `width_hz` is probed:
    the lowest output rate (≥ 2.5× the width) the span's rate divides
    into: four or more samples a symbol, so the lag products can sit near
    the 1¼-symbol lag where FSK puts its line (at 2 samples a symbol the
    lags quantise badly — measured). A probe needs the channel, not the
    canonical cut's guard band —
    and every sample saved is saved for every rate and lag."""
    return max(1, int(float(fs) // (2.5 * float(width_hz))))


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def _as_profile(profile) -> _profiles.ReceiverProfile:
    if isinstance(profile, _profiles.ReceiverProfile):
        return profile
    return _profiles.new_profile(str(profile))


def _class_list(classes, family: str) -> list:
    if classes is None:
        cl = _classes.for_profile_family(family)
    else:
        cl = []
        for c in classes:
            obj = c if isinstance(c, _classes.SignalClass) else _classes.get(c)
            if obj is None:
                raise ValueError(f"{c!r} is not in the class table "
                                 f"({', '.join(_classes.names())})")
            cl.append(obj)
    return [c for c in cl if not c.negative and (c.symbol_rates or c.cp_lag_s)]


def _rate_groups(cl: list, fs: float) -> list[dict]:
    """Symbol-rate probes grouped by the band they need: {bw, rates,
    families, classes_by_rate}. A rate's band is the widest class that
    lists it (P25 8.1 k, DMR 7.6 k, NXDN96 8.3 k → 8.3 kHz for 4800)."""
    by_rate: dict = {}
    for c in cl:
        if c.cp_lag_s > 0:
            continue          # OFDM: its "symbol rate" is the prefix train
        for r in c.symbol_rates:
            if r <= 0 or r >= 0.5 * fs:
                continue
            e = by_rate.setdefault(float(r), {"bw": 0.0, "classes": []})
            e["bw"] = max(e["bw"], float(c.bandwidth_hz or 2.5 * r))
            e["classes"].append(c)
    groups: dict = {}
    for r, e in by_rate.items():
        bw = min(e["bw"], 0.9 * fs)
        key = (_profiles.bandwidth_class(bw), round(bw, -2))
        g = groups.setdefault(key, {"bw": bw, "rates": [], "classes": {},
                                    "families": {}})
        g["rates"].append(r)
        g["classes"][r] = [c.name for c in e["classes"]]
        fams = {c.family for c in e["classes"]}
        g["families"][r] = fams.pop() if len(fams) == 1 else ""
    return sorted(groups.values(), key=lambda g: g["bw"])


def _cp_entries(cl: list) -> list[dict]:
    out: dict = {}
    for c in cl:
        if c.cp_lag_s > 0:
            e = out.setdefault(round(c.cp_lag_s, 12), {
                "lag_s": float(c.cp_lag_s), "bw": 0.0, "classes": [],
                "hint": (c.symbol_rates[0] if c.symbol_rates else None)})
            e["bw"] = max(e["bw"], float(c.bandwidth_hz))
            e["classes"].append(c.name)
    return list(out.values())


# ---------------------------------------------------------------------------
# The channel grid — a polyphase (WOLA) channelizer
# ---------------------------------------------------------------------------
def channelise(x, fs: float, spacing_hz: float, width_hz: float,
               decim: int, usable: float = USABLE, taps_per_phase: int = 8,
               chunk: int = 2048) -> tuple:
    """Every channel of a grid at once. Returns (Y [C, n_out] complex64,
    centres_hz [C] (offsets from the span centre), fs_out, info).

    Channel k is x shifted by −k·fs/M, low-passed to ±width/2 by one
    prototype filter (Kaiser, `taps_per_phase`·M taps) and decimated by
    `decim` — the same operation `cut_to_canonical` performs on one box,
    for all M channels with one M-point FFT per output sample (weighted
    overlap-add). M = round(fs/spacing); the true spacing is fs/M. Channels
    reaching beyond ±usable·fs are dropped. The output is delayed by the
    filter's (L−1)/2 input samples, which the info records."""
    from numpy.lib.stride_tricks import sliding_window_view
    from scipy.signal import firwin
    x = np.asarray(x).astype(np.complex64, copy=False)
    fs = float(fs)
    M = max(2, int(round(fs / float(spacing_hz))))
    D = max(1, int(decim))
    L = int(taps_per_phase) * M
    trans = max(0.15 * width_hz, 3.5 * fs / L)
    h = firwin(L, 0.5 * width_hz + 0.5 * trans, fs=fs,
               window=("kaiser", 8.0)).astype(np.float32)
    ks = np.arange(M)
    centres_all = np.where(ks < M / 2, ks, ks - M) * fs / M
    keep = np.flatnonzero(np.abs(centres_all) + 0.5 * width_hz <= usable * fs)
    keep = keep[np.argsort(centres_all[keep])]
    if x.size < L + D:
        return (np.zeros((keep.size, 0), np.complex64), centres_all[keep],
                fs / D, {"M": M, "taps": L, "delay_samples": (L - 1) / 2})
    view = sliding_window_view(x, L)[::D]
    n_out = view.shape[0]
    Y = np.empty((keep.size, n_out), dtype=np.complex64)
    for s in range(0, n_out, chunk):
        e = min(n_out, s + chunk)
        seg = view[s:e] * h[None, :]
        fold = seg.reshape(e - s, int(taps_per_phase), M).sum(axis=1)
        F = np.fft.fft(fold, axis=1)[:, keep]
        m = np.arange(s, e, dtype=np.float64)
        ph = np.exp(-2j * np.pi * np.mod(np.outer(m * D, keep) / M, 1.0))
        Y[:, s:e] = (F * ph).T.astype(np.complex64)
    return Y, centres_all[keep], fs / D, {"M": M, "taps": L,
                                          "delay_samples": (L - 1) / 2,
                                          "spacing_hz": fs / M}


# ---------------------------------------------------------------------------
# The proposer
# ---------------------------------------------------------------------------
def cyclic_proposer(x, fs, center_hz, profile, classes=None, regions=None,
                    t0=0.0, epoch=None, pfa=1e-3, *, report=None):
    """Cyclic-feature detections in x (see the module docstring).

    x          IQ at the PROFILE's rate: (n,) or (channels, n) — a
               multi-channel capture is probed on channel 0
    fs         its sample rate; must equal the profile's (the sample-rate
               law: a mismatch is refused in words)
    center_hz  the centre frequency of x (absolute)
    profile    a ReceiverProfile or a profile id
    classes    class names / SignalClass objects to look for (default: every
               class the profile's receiver can capture that has a symbol
               rate or an OFDM prefix in the class table)
    regions    absolute (f_lo_hz, f_hi_hz) pairs, or None for the grid
    t0, epoch  stream time of x[0] and the wall clock of that origin
    pfa        false-alarm probability for the WHOLE call
    report     a dict to fill with what was tested, near misses, timings

    Returns detections with sources=("cyclic",), alpha_hz, integration_s,
    confidence (from the measured significance), snr_db when energy can
    measure it, and the probe's numbers in measurements."""
    tic = time.perf_counter()
    prof = _as_profile(profile)
    fs = float(fs)
    if abs(fs - float(prof.sample_rate)) > 0.5:
        raise _profiles.ProfileMismatch(
            f"this IQ is at {fs:,.0f} S/s and the profile "
            f"{prof.id} is {_profiles.describe(prof.id)}. The cyclic proposer "
            "runs only at the profile's own rate (the sample-rate law); "
            "resample deliberately, as a logged step, if you must.")
    xa = np.asarray(x)
    note = ""
    if xa.ndim == 2:
        note = f"multi-channel input ({xa.shape[0]} channels): channel 0 probed"
        xa = xa[0]
    xa = xa.astype(np.complex64, copy=False)
    T = xa.size / fs
    cl = _class_list(classes, prof.pid.family)
    groups = _rate_groups(cl, fs)
    cps = _cp_entries(cl)
    rep = {"profile": prof.id, "integration_s": T, "pfa": pfa, "calls": [],
           "near_misses": [], "notes": [note] if note else [],
           "mode": "regions" if regions is not None else "grid"}
    floor = _span_floor(xa, fs, prof)
    rep["floor_per_hz"] = floor
    if regions is not None:
        plan = _plan_regions(regions, center_hz, fs, groups, cps)
    else:
        plan = _plan_grid(fs, groups, cps)
    n_calls = max(1, sum(p["calls"] for p in plan))
    pfa_call = float(pfa) / n_calls
    rep["probe_calls"] = n_calls
    rep["pfa_per_call"] = pfa_call
    raw: list[dict] = []
    for item in plan:
        t_item = time.perf_counter()
        if item["kind"] == "region":
            raw += _run_region(xa, fs, center_hz, item, pfa_call, floor, rep)
        elif item["kind"] == "grid":
            raw += _run_grid(xa, fs, center_hz, item, pfa_call, floor, rep)
        else:
            raw += _run_cp_grid(xa, fs, center_hz, item, pfa_call, floor, rep)
        rep["calls"].append({"what": item["label"],
                             "seconds": time.perf_counter() - t_item})
    dets = _to_detections(raw, center_hz, t0, T, epoch, prof.id, pfa, n_calls)
    rep["seconds"] = time.perf_counter() - tic
    rep["detections"] = len(dets)
    if isinstance(report, dict):
        report.update(rep)
    return dets


def _span_floor(x: np.ndarray, fs: float, prof) -> float:
    from atk_diffusion.dsp import measure as _m
    n = min(x.size, int(fs * 0.5) + 4096)
    nfft = int(getattr(prof.stft, "fft_size", 1024))
    return float(_m.noise_floor(x[:n], fs, nfft=nfft)["floor_per_hz"])


def _plan_regions(regions, center_hz, fs, groups, cps) -> list:
    plan = []
    for (lo, hi) in regions:
        lo, hi = sorted((float(lo), float(hi)))
        W = max(hi - lo, 1.0)
        can = _profiles.canonical_for(fs, W)
        fs_c = can.rate
        gsel = []
        for g in groups:
            rates = [r for r in g["rates"] if r < 0.5 * fs_c]
            if rates and g["bw"] <= 2.5 * W and g["bw"] >= W / 8.0:
                gsel.append({**g, "rates": rates})
        csel = [c for c in cps if c["lag_s"] * fs_c >= 4 and W >= 50e3]
        plan.append({"kind": "region", "lo": lo, "hi": hi, "groups": gsel,
                     "cps": csel, "calls": len(gsel) + len(csel),
                     "label": f"region {lo:,.0f}–{hi:,.0f} Hz"})
    return plan


def _plan_grid(fs, groups, cps) -> list:
    plan = []
    for g in groups:
        W = g["bw"]
        D = probe_decimation(fs, W)
        if W >= 2 * USABLE * fs * 0.95:          # one channel: the span
            centres = np.array([0.0])
        else:
            M = max(2, int(round(fs / (W / GRID_STEP))))
            k = np.arange(M)
            c = np.where(k < M / 2, k, k - M) * fs / M
            centres = np.sort(c[np.abs(c) + 0.5 * W <= USABLE * fs])
        rates = [r for r in g["rates"] if r < 0.5 * fs / D]
        if not rates or centres.size == 0:
            continue
        plan.append({"kind": "grid", "group": {**g, "rates": rates},
                     "decim": D, "fs_c": fs / D,
                     "n_channels": int(centres.size),
                     "calls": int(centres.size),
                     "label": (f"grid {W / 1e3:,.1f} kHz channels × "
                               f"{centres.size} for {', '.join(f'{r:g}' for r in rates)} "
                               "sym/s")})
    for c in cps:
        W = min(c["bw"], 2 * USABLE * fs)
        can = _profiles.canonical_for(fs, W)
        W = min(W, 0.8 * can.rate)
        if c["lag_s"] * can.rate < 4:
            continue
        if W >= 2 * USABLE * fs * 0.95:
            n = 1
        else:
            M = max(2, int(round(fs / (0.5 * W))))
            k = np.arange(M)
            cc = np.where(k < M / 2, k, k - M) * fs / M
            n = int(np.sum(np.abs(cc) + 0.5 * W <= USABLE * fs))
        if n == 0:
            continue
        plan.append({"kind": "cp", "entry": c, "W": W, "decim": can.decimation,
                     "fs_c": can.rate, "calls": n,
                     "label": (f"OFDM prefix {c['lag_s'] * 1e6:,.2f} µs on "
                               f"{n} channel(s) of {W / 1e6:,.2f} MHz")})
    return plan


def _snr(y: np.ndarray, fs_c: float, W: float, floor: float) -> dict:
    from atk_diffusion.dsp import measure as _m
    try:
        return _m.snr_above_floor(y, fs_c, band=(-0.5 * W, 0.5 * W),
                                  floor_per_hz=floor)
    except Exception:                                      # noqa: BLE001
        return {"snr_db": None, "measurable": False}


def _line_hits(res: dict, group: dict, lo: float, hi: float, fs_c: float,
               y: np.ndarray, floor: float, pfa_call: float, rep: dict
               ) -> list:
    out = []
    if res.get("p_family", 1.0) < NEAR_MISS_P and not res["detected"]:
        rep["near_misses"].append({
            "f_lo": lo, "f_hi": hi, "probe": "symbol_rate_line",
            "rate_hz": res["best_rate_hz"], "statistic": res["statistic"],
            "threshold": res["threshold"], "p_call": res.get("p_family"),
            "confidence": res.get("confidence")})
    if not res["detected"]:
        return out
    hits = [r for r in res["per_rate"] if r["detected"]]
    snr = _snr(y, fs_c, hi - lo, floor)
    for r in hits:
        out.append({"probe": "symbol_rate_line", "lo": lo, "hi": hi,
                    "rate": r["rate_hz"], "alpha": r["alpha_hz"],
                    "statistic": r["statistic"], "threshold": r["threshold"],
                    "p_test": r["p_value"], "trials": res["trials"],
                    "candidates": list(group["classes"][r["rate_hz"]]),
                    "family": group["families"].get(r["rate_hz"], ""),
                    "lags": r["lags"], "snr": snr, "fs_c": fs_c})
    return out


def _run_region(x, fs, center_hz, item, pfa_call, floor, rep) -> list:
    from atk_diffusion.dsp.resample import cut_to_canonical
    lo, hi = item["lo"], item["hi"]
    y, fs_c, info = cut_to_canonical(x, fs, 0.5 * (lo + hi) - float(center_hz),
                                     hi - lo)
    out = []
    W = hi - lo
    for g in item["groups"]:
        bw = g["bw"] if g["bw"] < W else None
        res = _probes.symbol_rate_line(y, fs_c, g["rates"], pfa=pfa_call,
                                       bandwidth_hz=bw, family=g["families"])
        out += _line_hits(res, g, lo, hi, fs_c, y, floor, pfa_call, rep)
    for c in item["cps"]:
        res = _probes.cp_probe(y, fs_c, c["lag_s"], pfa=pfa_call,
                               period_hint_hz=c["hint"])
        out += _cp_hit(res, c, lo, hi, fs_c, y, floor, rep,
                       shift_hz=0.5 * (lo + hi) - float(center_hz))
    return out


def _run_grid(x, fs, center_hz, item, pfa_call, floor, rep) -> list:
    g = item["group"]
    W = g["bw"]
    Y, centres, fs_c, info = channelise(x, fs, W / GRID_STEP, W,
                                        item["decim"])
    if Y.shape[1] == 0:
        return []
    res_all = _probes.symbol_rate_line(Y, fs_c, g["rates"], pfa=pfa_call,
                                       family=g["families"])
    out = []
    for i, res in enumerate(res_all):
        c = float(center_hz) + float(centres[i])
        out += _line_hits(res, g, c - 0.5 * W, c + 0.5 * W, fs_c, Y[i], floor,
                          pfa_call, rep)
    rep.setdefault("grids", []).append({"channels": int(Y.shape[0]),
                                        "width_hz": W, "fs_c": fs_c,
                                        "rates": g["rates"], **info})
    return out


def _wrap(f: float, period: float) -> float:
    """f into [−period/2, period/2)."""
    return float((float(f) + 0.5 * period) % period - 0.5 * period)


def _cp_hit(res, c, lo, hi, fs_c, y, floor, rep, shift_hz: float = 0.0
            ) -> list:
    if res.get("p_value", 1.0) < NEAR_MISS_P and not res["detected"]:
        rep["near_misses"].append({"f_lo": lo, "f_hi": hi, "probe": "cp_probe",
                                   "lag_s": c["lag_s"],
                                   "statistic": res["statistic"],
                                   "threshold": res["threshold"],
                                   "confidence": res.get("confidence")})
    if not res["detected"]:
        return []
    # the prefix's phase measures the carrier offset from the centre of the
    # cut it was probed in; every cut here was shifted by `shift_hz` from
    # the capture's centre, so the offset from the CAPTURE's centre is the
    # measured one plus the shift, modulo the subcarrier spacing (the first
    # version left the shift in: 7,066 Hz reported for a 900 Hz cell)
    res = dict(res)
    spacing = 1.0 / float(c["lag_s"])
    if res.get("cfo_hz") is not None:
        res["cfo_hz"] = _wrap(res["cfo_hz"] + shift_hz, spacing)
        if isinstance(res.get("estimates"), dict):
            res["estimates"] = dict(res["estimates"], cfo_hz=res["cfo_hz"])
    return [{"probe": "cp_probe", "lo": lo, "hi": hi, "lag_s": c["lag_s"],
             "statistic": res["statistic"], "threshold": res["threshold"],
             "p_test": res["p_value"], "trials": 1,
             "candidates": list(c["classes"]), "family": "ofdm",
             "cp": res, "snr": _snr(y, fs_c, hi - lo, floor), "fs_c": fs_c,
             "y": y, "shift_hz": shift_hz, "hint": c.get("hint")}]


def _cp_cells(m: dict, pfa: float) -> dict:
    """How many distinct (non-time-aligned) transmitters share the detected
    OFDM channel — probes.cp_timing_phases on the hit's own cut, with the
    carrier offsets referred to the capture's centre like the detection's."""
    y = m.get("y")
    if y is None:
        return {}
    hint = m.get("hint")
    try:
        r = _probes.cp_timing_phases(y, m["fs_c"], m["lag_s"],
                                     period_s=(1.0 / hint if hint else None),
                                     pfa=pfa)
    except ValueError as e:
        return {"cp_cells": None, "cp_cells_words": f"not counted: {e}"}
    spacing = 1.0 / float(m["lag_s"])
    cells = []
    for q in r.get("cells", []):
        q = dict(q)
        q["cfo_hz"] = _wrap(q["cfo_hz"] + m.get("shift_hz", 0.0), spacing)
        cells.append(q)
    return {"cp_cells": int(r["n_cells"]), "cp_cells_detail": cells,
            "cp_cells_words": r["words"], "cp_cells_limit": r["limit"]}


def _run_cp_grid(x, fs, center_hz, item, pfa_call, floor, rep) -> list:
    c, W = item["entry"], item["W"]
    if item["calls"] == 1:
        if item["decim"] == 1:
            Y, centres, fs_c = x[None, :], np.array([0.0]), fs
        else:
            from atk_diffusion.dsp.resample import cut_to_canonical
            y, fs_c, _info = cut_to_canonical(x, fs, 0.0, W)
            Y, centres = y[None, :], np.array([0.0])
    else:
        Y, centres, fs_c, _info = channelise(x, fs, 0.5 * W, W, item["decim"])
    out = []
    for i in range(Y.shape[0]):
        res = _probes.cp_probe(Y[i], fs_c, c["lag_s"], pfa=pfa_call,
                               period_hint_hz=c["hint"])
        mid = float(center_hz) + float(centres[i])
        out += _cp_hit(res, c, mid - 0.5 * W, mid + 0.5 * W, fs_c, Y[i],
                       floor, rep, shift_hz=float(centres[i]))
    return out


# ---------------------------------------------------------------------------
# Raw hits -> Detections: merge overlapping channels, fold harmonics
# ---------------------------------------------------------------------------
def _to_detections(raw, center_hz, t0, T, epoch, pid, pfa, n_calls) -> list:
    # 1. merge the same rate/lag across overlapping channels
    merged: list = []
    for h in sorted(raw, key=lambda d: (d.get("rate", d.get("lag_s", 0)),
                                        d["lo"])):
        key = ("r", round(h["rate"], 3)) if "rate" in h else \
            ("c", round(h["lag_s"], 12))
        tgt = next((m for m in merged if m["key"] == key
                    and h["lo"] <= m["hi"] and h["hi"] >= m["lo"]), None)
        if tgt is None:
            merged.append({**h, "key": key, "n": 1})
        else:
            tgt["lo"], tgt["hi"] = min(tgt["lo"], h["lo"]), max(tgt["hi"],
                                                                 h["hi"])
            tgt["n"] += 1
            if h["statistic"] / h["threshold"] > \
                    tgt["statistic"] / tgt["threshold"]:
                keep = {k: tgt[k] for k in ("lo", "hi", "n", "key")}
                tgt.update(h)
                tgt.update(keep)
    # 2. fold harmonics: a rate that is an integer multiple (2..6) of
    #    another detected rate overlapping it is the same signal
    rates = [m for m in merged if m["key"][0] == "r"]
    for m in rates:
        for base in rates:
            if base is m or base["rate"] >= m["rate"]:
                continue
            q = m["rate"] / base["rate"]
            if (abs(q - round(q)) < 0.01 and 2 <= round(q) <= 6
                    and m["lo"] <= base["hi"] and m["hi"] >= base["lo"]):
                m["harmonic_of"] = base["rate"]
                base.setdefault("harmonics", []).append(m["rate"])
    dets = []
    for m in merged:
        if m.get("harmonic_of"):
            continue
        dets.append(_detection(m, t0, T, epoch, pid, pfa, n_calls))
    return dets


def _detection(m: dict, t0, T, epoch, pid, pfa, n_calls) -> Detection:
    p_call = _probes.family_p(m["p_test"], m["trials"])
    p_fw = min(1.0, p_call * n_calls)
    conf = _probes.confidence_from_p(p_fw, pfa)
    cands = sorted(set(m["candidates"]))
    fams = {(_classes.get(c).family if _classes.get(c) else "") for c in cands}
    family = fams.pop() if len(fams) == 1 else (m.get("family") or "unknown")
    cls = cands[0] if len(cands) == 1 else ""
    meas = {"probe": m["probe"], "statistic": float(m["statistic"]),
            "threshold": float(m["threshold"]),
            "margin": float(m["statistic"] / m["threshold"]),
            "p_value_call": float(p_call), "channels_merged": int(m["n"]),
            "canonical_rate_hz": float(m["fs_c"]),
            "candidates": cands}
    snr = m.get("snr") or {}
    snr_db = snr.get("snr_db") if snr.get("measurable") else None
    if snr_db is None:
        meas["snr_note"] = ("the energy is within the floor's own uncertainty "
                            "— below what an energy measurement can see")
    if m["probe"] == "symbol_rate_line":
        alpha = float(m["alpha"])
        meas.update({"symbol_rate_hz": alpha, "listed_rate_hz": m["rate"],
                     "lags": list(m["lags"])})
        if m.get("harmonics"):
            meas["harmonics_hz"] = sorted(m["harmonics"])
        words = (f"{alpha:,.1f} sym/s "
                 + _classes.FAMILY_WORDS.get(family, family)
                 + (f" — {_classes.get(cls).label}" if cls else
                    f" — one of {', '.join(cands)} (the decoder says which)"))
    else:
        cp = m["cp"]
        alpha = (cp.get("symbol_rate_hz") if cp.get("period_detected")
                 else None)
        meas.update({"cp_lag_s": m["lag_s"], "cfo_hz": cp.get("cfo_hz"),
                     "cfo_reference": ("the capture's centre frequency, modulo "
                                       f"the {1e-3 / m['lag_s']:,.3g} kHz "
                                       "subcarrier spacing"),
                     "cfo_se_hz": cp.get("cfo_se_hz"), "rho": cp.get("rho"),
                     "symbol_period_s": cp.get("symbol_period_s"),
                     "period_detected": cp.get("period_detected")})
        meas.update(_cp_cells(m, pfa))
        if cp.get("snr_db") is not None:
            # the prefix's own SNR, not energy: an OFDM carrier that fills
            # more than half the span is IN the span's median floor, and the
            # energy reading then comes out ~9 dB low (measured: −14.5 dB
            # for a −6 dB cell whose prefix said −6.0)
            if snr_db is not None:
                meas["snr_energy_db"] = snr_db
            snr_db = cp["snr_db"]
            meas["snr_method"] = cp.get("snr_method")
            meas.pop("snr_note", None)
        words = (f"OFDM, {1e-3 / m['lag_s']:,.3g} kHz subcarrier spacing "
                 f"(cyclic prefix at {m['lag_s'] * 1e6:,.2f} µs) — "
                 + (_classes.get(cls).label if cls else ", ".join(cands)))
        if cls and _classes.get(cls) and _classes.get(cls).notes:
            meas["note"] = _classes.get(cls).notes
    meas["class_words"] = words
    return Detection(t0=float(t0), t1=float(t0) + T, f_lo=float(m["lo"]),
                     f_hi=float(m["hi"]), sources=("cyclic",), family=family,
                     cls=cls, confidence=conf, snr_db=snr_db, alpha_hz=alpha,
                     integration_s=T, measurements=meas, profile=pid,
                     epoch=epoch)
