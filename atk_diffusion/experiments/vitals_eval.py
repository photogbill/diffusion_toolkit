# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""I1's experiment on synthetic CSI with KNOWN rates (plan §4.I1, §7).

Bill's first experiment is his own breathing against a count, three
postures, one ESP32 pair. Until the pair arrives, the same code is measured
here on CSI made from physics with known truth:

    H_k(t) = Σ_p a_p e^{-j2π f_k τ_p}                (static multipath)
           + a_b e^{-j2π f_k L(t)/c} e^{jφ}          (the path off the chest)
    L(t)   = L0 + 2·(breathing(t) + heartbeat(t))    (a reflection: twice the
                                                      chest's displacement)

over the subcarriers of a 20 MHz Wi-Fi channel (312.5 kHz apart), with an AGC
that wanders, receiver noise, the ESP32's int8 quantisation, timestamp
jitter and dropped frames — then through `sensing.csi.resample_uniform` and
`sensing.vitals.estimate`, exactly as a real recording would go.

What it reports, per condition: the breathing-rate and heart-rate error
(bpm) against the truth, how often a window reported no rate (the "I don't
know" rate), and on EMPTY-ROOM CSI the rate at which a breathing or heart
rate is reported where there is no person — this pipeline's hallucination
rate (plan §7). When PyTorch is present, the small LSTM (`learn.vitals`) is
trained on synthetic windows and scored on held-out conditions beside the
classical estimator — the classical baseline first (plan §7).

Every number is labelled what it is: a research-grade measurement on
synthetic data, not a medical device and not a diagnosis.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np

from atk_diffusion import profiles as _profiles
from atk_diffusion.sensing import csi as _csi
from atk_diffusion.sensing import vitals as _vitals

C = 299_792_458.0


def synth_csi(duration_s: float = 60.0, fs: float = 50.0, *,
              breath_bpm: float = 15.0, heart_bpm: float = 72.0,
              person: bool = True, n_sc: int = 52, f_c: float = 2.437e9,
              spacing: float = 312.5e3, chest_mm: float = 5.0,
              heart_mm: float = 0.3, body_gain: float = 0.35,
              static_paths: int = 6, noise: float = 0.01,
              agc_db: float = 0.15, jitter: float = 0.1, drop: float = 0.02,
              apnea: tuple | None = None, walker: bool = False,
              rng=None) -> tuple[np.ndarray, np.ndarray, dict]:
    """(t seconds, H complex64 (frames, subcarriers), truth). See the module
    docstring for the model; `apnea=(t0, t1)` stops breathing for that span;
    `walker` adds a person walking across the room (motion, no vitals)."""
    rng = rng or np.random.default_rng(0)
    n = int(duration_s * fs)
    t = np.arange(n) / fs + rng.uniform(-jitter, jitter, n) / fs
    t = np.sort(t)
    k = np.arange(n_sc) - n_sc / 2
    fk = f_c + k * spacing
    H = np.exp(-2j * np.pi * fk * (5.0 / C))[None, :] * np.ones((n, 1))
    for _ in range(static_paths):
        tau = rng.uniform(15e-9, 80e-9)
        a = rng.uniform(0.1, 0.6)
        H = H + a * np.exp(1j * rng.uniform(0, 2 * np.pi)) * \
            np.exp(-2j * np.pi * fk * tau)[None, :]
    truth = {"breath_bpm": breath_bpm if person else None,
             "heart_bpm": heart_bpm if person else None, "person": person,
             "apnea": apnea, "walker": walker}
    if person:
        fb = breath_bpm / 60.0 * (1 + 0.02 * np.sin(2 * np.pi * t / 47.0))
        fh = heart_bpm / 60.0 * (1 + 0.02 * np.sin(2 * np.pi * t / 31.0))
        pb = 2 * np.pi * np.cumsum(np.r_[0, np.diff(t)] * fb)
        ph = 2 * np.pi * np.cumsum(np.r_[0, np.diff(t)] * fh)
        env = np.ones(n)
        if apnea is not None:
            env[(t >= apnea[0]) & (t < apnea[1])] = 0.0
            from scipy.ndimage import uniform_filter1d
            env = uniform_filter1d(env, max(1, int(fs)))
        disp = env * (chest_mm * 1e-3 / 2) * (1 - np.cos(pb) + 0.15 * np.sin(2 * pb))
        disp = disp + heart_mm * 1e-3 * (np.sin(ph) + 0.3 * np.sin(2 * ph))
        L0 = rng.uniform(4.0, 7.0)
        L = L0 + 2 * disp
        H = H + body_gain * np.exp(1j * rng.uniform(0, 2 * np.pi)) * \
            np.exp(-2j * np.pi * fk[None, :] * (L[:, None] / C))
    if walker:
        pos = np.cumsum(rng.normal(0, 0.03, n))          # metres of path change
        L = rng.uniform(6.0, 9.0) + np.abs(pos)
        H = H + 0.4 * np.exp(-2j * np.pi * fk[None, :] * (L[:, None] / C))
    gain = 10 ** (np.cumsum(rng.normal(0, agc_db / np.sqrt(fs), n)) / 20.0)
    gain = gain / gain.mean()
    H = H * gain[:, None]
    H = H + noise * np.mean(np.abs(H)) * (rng.standard_normal(H.shape)
                                         + 1j * rng.standard_normal(H.shape))
    scale = 40.0 / np.max(np.abs(np.r_[H.real.ravel(), H.imag.ravel()]))
    H = np.round(H.real * scale) + 1j * np.round(H.imag * scale)   # int8 CSI
    keep = rng.random(n) >= drop
    return t[keep] - t[keep][0], H[keep].astype(np.complex64), truth


def classical(t, H, fs: float = 50.0, **kw) -> _vitals.VitalsReport:
    """A synthetic (or real) recording through the same path a real one
    takes: uniform grid with the gaps reported, then the classical pipeline."""
    tg, Ag, info = _csi.resample_uniform(t, np.abs(H), fs)
    return _vitals.estimate(Ag, fs, gaps=info["gaps"], **kw)


DEFAULT_CONDITIONS = (
    {"breath_bpm": 12.0, "heart_bpm": 60.0},
    {"breath_bpm": 16.0, "heart_bpm": 75.0},
    {"breath_bpm": 22.0, "heart_bpm": 95.0},
    {"breath_bpm": 9.0, "heart_bpm": 110.0, "noise": 0.03},
)


def run(rf=None, *, conditions=DEFAULT_CONDITIONS, duration_s: float = 90.0,
        fs: float = 50.0, seeds=(1, 2), empty_seeds=(11, 12),
        lstm: bool = True, out_dir=None,
        progress: Callable[[str], None] | None = None) -> dict:
    """Score the classical pipeline (and the LSTM, when PyTorch is present)
    on synthetic CSI with known rates. -> result dict; writes report.md and
    result.json under the ESP32 profile's runs folder when `rf` is given."""
    rows = []
    errs_b, errs_h = [], []
    none_b = none_h = nwin = 0
    for cond in conditions:
        for seed in seeds:
            kw = dict(cond)
            t, H, truth = synth_csi(duration_s, fs, rng=np.random.default_rng(seed),
                                    **kw)
            rep = classical(t, H, fs, apnea=False)
            eb = [abs(w["breath_bpm"] - truth["breath_bpm"]) for w in rep.windows
                  if w["breath_bpm"] is not None]
            eh = [abs(w["heart_bpm"] - truth["heart_bpm"]) for w in rep.windows
                  if w["heart_bpm"] is not None]
            nwin += len(rep.windows)
            none_b += sum(w["breath_bpm"] is None for w in rep.windows)
            none_h += sum(w["heart_bpm"] is None for w in rep.windows)
            errs_b += eb
            errs_h += eh
            rows.append({**cond, "seed": seed,
                         "breath_est": rep.breath_bpm, "heart_est": rep.heart_bpm,
                         "breath_mae": float(np.mean(eb)) if eb else None,
                         "heart_mae": float(np.mean(eh)) if eh else None})
            if progress:
                progress(f"{cond} seed {seed}: breath {rep.breath_bpm}, "
                         f"heart {rep.heart_bpm}")
    # empty room: what is reported where nobody is
    fb = fh = ne = 0
    for seed in empty_seeds:
        t, H, _ = synth_csi(duration_s, fs, person=False,
                            rng=np.random.default_rng(seed))
        rep = classical(t, H, fs, apnea=False)
        ne += len(rep.windows)
        fb += sum(w["breath_bpm"] is not None for w in rep.windows)
        fh += sum(w["heart_bpm"] is not None for w in rep.windows)
    # a breathing pause
    t, H, truth = synth_csi(120.0, fs, breath_bpm=14.0, heart_bpm=70.0,
                            apnea=(50.0, 70.0), rng=np.random.default_rng(99))
    rep = classical(t, H, fs)
    pause_found = any(abs(e["t0_s"] - 50.0) < 8.0 for e in rep.apnea)
    result = {
        "experiment": "vitals_eval (plan §4.I1) — synthetic CSI",
        "label": _vitals.RESEARCH_LABEL, "tier": "measured",
        "classical": {
            "breath_mae_bpm": float(np.mean(errs_b)) if errs_b else None,
            "heart_mae_bpm": float(np.mean(errs_h)) if errs_h else None,
            "breath_p90_bpm": float(np.percentile(errs_b, 90)) if errs_b else None,
            "heart_p90_bpm": float(np.percentile(errs_h, 90)) if errs_h else None,
            "windows": nwin,
            "no_breath_rate_fraction": none_b / nwin if nwin else None,
            "no_heart_rate_fraction": none_h / nwin if nwin else None},
        "empty_room": {"windows": ne,
                       "breath_reported_fraction": fb / ne if ne else None,
                       "heart_reported_fraction": fh / ne if ne else None,
                       "what": "a rate reported where nobody is — this "
                               "pipeline's hallucination rate"},
        "pause": {"injected": [50.0, 70.0], "found": bool(pause_found),
                  "events": rep.apnea},
        "rows": rows, "duration_s": duration_s, "fs": fs,
        "created": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}
    if lstm:
        result["lstm"] = _lstm_part(conditions, duration_s, fs, seeds)
    result["report_md"] = report_md(result)
    files = []
    out = Path(out_dir) if out_dir is not None else (
        Path(rf.runs(profile_for(fs))) / f"vitals_eval_{result['created']}"
        if rf is not None else None)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        jp, mp = out / "result.json", out / "report.md"
        jp.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        mp.write_text(result["report_md"], encoding="utf-8")
        files = [str(jp), str(mp)]
        if rf is not None:
            for f in (jp, mp):
                try:
                    rf.record(f, "experiment", "vitals_eval")
                except Exception:                          # noqa: BLE001
                    pass
    result["files"] = files
    return result


def profile_for(fs: float) -> str:
    """The ESP32 CSI sensor as a receiver profile (rate = CSI frames/s)."""
    return _profiles.make_profile_id("esp32csi", int(round(fs)), "cf32")


def _lstm_part(conditions, duration_s, fs, seeds) -> dict:
    try:
        import torch  # noqa: F401
    except ImportError:
        return {"available": False,
                "why": "PyTorch is not in this environment; the LSTM runs in "
                       "the training environment"}
    from atk_diffusion.learn import vitals as _lv
    X, y = [], []
    rng = np.random.default_rng(1234)
    for _ in range(24):
        b = float(rng.uniform(8, 24))
        h = float(rng.uniform(55, 115))
        t, H, _tr = synth_csi(40.0, fs, breath_bpm=b, heart_bpm=h,
                              rng=np.random.default_rng(int(rng.integers(1 << 30))))
        _tg, A, _info = _csi.resample_uniform(t, np.abs(H), fs)
        F = _lv.features(A, fs)
        X.append(F[:int(30 * _lv.FEATURE_FS)])
        y.append([b, h])
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="vitals_lstm_"))
    card, metrics = _lv.train(np.stack(X), np.array(y), d, name="vitals-synthetic",
                              epochs=60, seed=0)
    model = _lv.load(d)
    errs_b, errs_h = [], []
    for cond in conditions:
        for seed in seeds:
            t, H, truth = synth_csi(40.0, fs, rng=np.random.default_rng(seed + 500),
                                    **cond)
            _tg, A, _i = _csi.resample_uniform(t, np.abs(H), fs)
            F = _lv.features(A, fs)[:int(30 * _lv.FEATURE_FS)]
            pb, ph = model.predict(F[None])[0]
            errs_b.append(abs(pb - truth["breath_bpm"]))
            errs_h.append(abs(ph - truth["heart_bpm"]))
    return {"available": True, "breath_mae_bpm": float(np.mean(errs_b)),
            "heart_mae_bpm": float(np.mean(errs_h)),
            "train": metrics, "note": "trained on 24 synthetic windows — a "
                                      "code-path proof, not a model to use"}


def _f(v, fmt="{:.1f}"):
    return "—" if v is None else fmt.format(v)


def report_md(r: dict) -> str:
    c, e = r["classical"], r["empty_room"]
    lines = ["# Vital signs from CSI — synthetic evaluation (plan §4.I1)", "",
             f"**{r['label']}**", "",
             "| estimator | breathing MAE | 90th pct | heart MAE | 90th pct |",
             "|---|---|---|---|---|",
             f"| classical (PulseFi pipeline) | {_f(c['breath_mae_bpm'])} bpm | "
             f"{_f(c['breath_p90_bpm'])} | {_f(c['heart_mae_bpm'])} bpm | "
             f"{_f(c['heart_p90_bpm'])} |"]
    lm = r.get("lstm") or {}
    if lm.get("available"):
        lines.append(f"| small LSTM (learn.vitals) | {_f(lm['breath_mae_bpm'])} bpm"
                     f" | — | {_f(lm['heart_mae_bpm'])} bpm | — |")
    elif lm:
        lines.append(f"| small LSTM | {lm.get('why')} | | | |")
    lines += ["",
              f"Windows with no clear breathing rhythm (no rate reported): "
              f"{_f(100 * (c['no_breath_rate_fraction'] or 0), '{:.0f}')}%; "
              f"no clear heart rhythm: "
              f"{_f(100 * (c['no_heart_rate_fraction'] or 0), '{:.0f}')}%.",
              f"Empty room — a rate reported where nobody is (the "
              f"hallucination rate): breathing "
              f"{_f(100 * (e['breath_reported_fraction'] or 0), '{:.0f}')}%, "
              f"heart {_f(100 * (e['heart_reported_fraction'] or 0), '{:.0f}')}% "
              f"of {e['windows']} windows.",
              f"A 20 s breathing pause injected at 50 s: "
              f"{'found' if r['pause']['found'] else 'NOT found'}."]
    return "\n".join(lines) + "\n"
