# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""A6's experiment — does a learned receiver-to-receiver translation make
better training data for receiver B than the parametric model of B?
(plan §4.A6, §7.)

Plan A6's first experiment: *"translate bladeRF -> RTL-SDR; train a detector
on the translated set alone; test on real RTL cabled captures; compare with
a detector trained on synthetic-plus-impairments."* This is that experiment
at tiny scale on synthetic stand-ins, every piece the real one uses:

* RECEIVER A and RECEIVER B — two impairment models applied to the SAME
  signal (the cabled loop through a splitter, simulated). "Real B" carries
  what a terminated measurement cannot see — soft compression, LO phase
  noise, a front-end droop — on top of quantisation, DC and IQ image.
* SYNTHETIC-PLUS-IMPAIRMENTS for B — what the package's dataset builder
  would make: clean synthetic signals plus white noise at B's floor, through
  `dsp.impair.apply_impairments` with B's impairments MEASURED from a
  terminated capture of real B (`dsp.impair.measure_impairments` — exactly
  the plan §3.4 workflow; both are another engineer's module, imported
  lazily, with the local receiver model as the fallback).
* THE TRANSLATOR — `learn.translator`, trained on paired (A, B) windows, or
  the one given; and its classical comparator, the widely-linear
  least-squares FIR.

THE NUMBERS.
* Statistics against real B, on held-out pairs — NMSE (paired), the
  average-spectrum distance in dB (floor shape, DC spike, image), and the
  receiver fingerprints on NOISE-ONLY windows (DC level, IQ improperness,
  quantisation levels) — for raw A, the linear translation, the learned
  translation and synthetic-plus-impairments.
* The DOWNSTREAM test: a small classifier (`learn.genclass.
  train_discriminative`) trained on (i) A translated by the model, (ii) A
  through the linear translator, (iii) synthetic-plus-impairments, and (iv)
  real B itself (the upper line), all tested on held-out real B. The DOMAIN
  GAP of each is the upper line's accuracy minus its own (plan §7: the one
  number every RF track reports).
* The translator's HALLUCINATION RATE: noise-only A windows translated, and
  the energy detector asked whether the translation holds a signal real B's
  noise does not.

LIMITS. Stand-in receivers, not Bill's; a tiny translator learns nothing in
a test. Both receivers run at one rate here — different rates go through
the logged resampler first (plan §3.3). A translated dataset is INVENTED
tier and labelled `atk:translated_from`; it is judged like any synthetic
data, by the gap.
"""

from __future__ import annotations

import math
import time

import numpy as np

from atk_diffusion import profiles, provenance
from atk_diffusion.dsp import denoise_classical as _cl
from atk_diffusion.learn import augment as _aug
from atk_diffusion.learn import translator as _tr

DEFAULT_KINDS = ("bpsk", "qpsk", "gfsk", "ofdm")


def stand_in_receivers():
    """(real A, real B): a bladeRF-like and an RTL-SDR-like receiver model.
    Stated stand-ins, not measurements of Bill's devices."""
    a = _aug.LocalReceiver(name="A (bladeRF-like)", adc_bits=12, signal_dbfs=-25.0,
                           noise_dbfs=-45.0, dc_dbfs=-50.0, iq_gain_db=0.1,
                           iq_phase_deg=0.5, edge_droop_db=0.5)
    b = _aug.LocalReceiver(name="B (RTL-SDR-like)", adc_bits=8, signal_dbfs=-20.0,
                           noise_dbfs=-32.0, dc_dbfs=-32.0, iq_gain_db=0.8,
                           iq_phase_deg=4.0, edge_droop_db=3.0, ripple_db=0.4,
                           compression_dbfs=-17.0, linewidth_hz=300.0)
    return a, b


def parametric_b(real_b: _aug.LocalReceiver, fs: float, rng, n_term: int = 1 << 15):
    """Synthetic-plus-impairments for B: B measured TERMINATED (what
    dsp.impair can see), then applied to clean synthetic signals. Returns
    (fn(clean unit-power windows) -> windows, words)."""
    term = real_b.noise_only(n_term, fs, rng)
    gain = 10 ** (real_b.signal_dbfs / 20.0)
    try:
        from atk_diffusion.dsp import impair as _imp
        imp = _imp.measure_impairments(term, fs, "cu8" if real_b.adc_bits == 8 else "cf32")
        p_n = _imp.noise_power(imp)

        def fn(S, r):
            out = []
            for s in np.atleast_2d(S):
                w = math.sqrt(p_n / 2) * (r.standard_normal(s.size) + 1j * r.standard_normal(s.size))
                out.append(_imp.apply_impairments(gain * s + w, fs, imp, r))
            return np.stack(out).astype(np.complex64)
        return fn, "dsp.impair: B measured terminated, then applied (" + _imp.describe(imp) + ")"
    except ImportError:
        par = _aug.LocalReceiver(name="parametric B", adc_bits=real_b.adc_bits,
                                 signal_dbfs=real_b.signal_dbfs, noise_dbfs=real_b.noise_dbfs,
                                 dc_dbfs=real_b.dc_dbfs, iq_gain_db=real_b.iq_gain_db,
                                 iq_phase_deg=real_b.iq_phase_deg)

        def fn(S, r):
            return np.stack([par.apply(s, fs, r) for s in np.atleast_2d(S)])
        return fn, ("local parametric model of B (dsp.impair not installed): DC, IQ "
                    "image, quantisation; no droop, compression or phase noise")


def _detects(x, fs, F, pfa) -> bool:
    spec = _cl.db_above(_cl.stft_power(x, 32), F)
    from atk_diffusion.learn import denoiser as _dn
    geom = {"fft_size": 32, "hop": 32, "window": "hann", "pool": 1, "pool_mode": "mean",
            "fs": fs}
    return _dn.detect_any(spec, pfa, geom)


def run(rf=None, from_profile: str = "bladerf1_2400000_ci16",
        to_profile: str = "rtlsdr_2400000_cu8", *, translator=None,
        kinds=DEFAULT_KINDS, window: int = 256, n_pairs: int = 400, n_test: int = 100,
        n_downstream_per: int = 60, snr_db=(5.0, 25.0), train_steps: int = 4000,
        unet: dict | None = None, T: int = 1000, sample_steps: int = 50,
        classifier_steps: int = 400, hallucination_trials: int = 40, pfa: float = 1e-3,
        seed: int = 0, out_dir=None, name: str = "translator_eval", progress=None) -> dict:
    """The A6 experiment (see the module docstring). Trains a translator on
    the stand-in pairs unless one is given (then its card must translate
    from `from_profile`). Returns the result; writes the report when `rf`
    or `out_dir` is given (the trained translator goes under `rf`'s target
    profile models, or `out_dir/model`)."""
    from atk_diffusion.learn import genclass as _gc
    say = progress or (lambda s: None)
    t_start = time.time()
    pa, pb = profiles.parse_profile_id(from_profile), profiles.parse_profile_id(to_profile)
    if pa.sample_rate != pb.sample_rate:
        raise profiles.ProfileMismatch(
            "this experiment runs both receivers at one rate; resample A's captures "
            "to B's rate first (dsp.resample.resample_capture, a logged step)")
    fs = float(pb.sample_rate)
    rng = np.random.default_rng(seed)
    real_a, real_b = stand_in_receivers()
    par_b, par_words = parametric_b(real_b, fs, rng)
    kinds = list(kinds)

    def pairs(n, snr):
        """The same clean signals into both receivers (a splitter). A level
        drawn in `snr` dB, less 20 dB, scales the receivers' input: real B
        (−20 dBFS per unit input over a −32 dBFS floor) then sees full-band
        SNRs of snr − 8 dB, and its compression point is reached at the top."""
        S, y = _aug.make_classification_set(kinds, max(1, n // len(kinds)), window, fs, rng,
                                            snr_db=(60.0, 60.0), offset_frac=0.2)
        lvl = 10 ** (np.asarray(rng.uniform(*snr, len(S))) / 20.0) * 0.1
        A = np.stack([real_a.apply(s * g, fs, rng) for s, g in zip(S, lvl)])
        B = np.stack([real_b.apply(s * g, fs, rng) for s, g in zip(S, lvl)])
        return S * lvl[:, None], y, A, B
    _S, _y, A_tr, B_tr = pairs(n_pairs, snr_db)
    if translator is None:
        if rf is None and out_dir is None:
            raise ValueError("training a translator needs somewhere to put it: pass rf= "
                             "or out_dir=, or pass a trained translator")
        if rf is not None:
            d = _tr.train_translator(rf, from_profile, to_profile, A_tr, B_tr,
                                     steps=train_steps, unet=unet, T=T,
                                     sample_steps=sample_steps, seed=seed, progress=say,
                                     name=f"{name}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}")
        else:
            from atk_diffusion.paths import RfData
            d = _tr.train_translator(RfData(__import__("pathlib").Path(out_dir) / "rf_data",
                                            create=True), from_profile, to_profile, A_tr, B_tr,
                                     steps=train_steps, unet=unet, T=T,
                                     sample_steps=sample_steps, seed=seed, progress=say,
                                     name=name)
        translator = d
    tr = translator if isinstance(translator, _tr.Translator) else \
        _tr.Translator.load(translator, from_profile=from_profile)
    if tr.input.get("to_profile") != str(to_profile).lower():
        raise profiles.ProfileMismatch(
            f"this translator makes {profiles.describe(tr.input['to_profile'])}, not "
            f"{profiles.describe(to_profile)}")
    if tr.L != int(window):
        raise ValueError(f"the translator works on {tr.L}-sample windows; this run asked "
                         f"for {window}")
    lin = _tr.fit_linear(A_tr, B_tr)
    # held-out statistics
    S_te, _y_te, A_te, B_te = pairs(n_test, snr_db)
    T_te = tr.translate_windows(A_te, steps=sample_steps, seed=seed + 1)
    L_te = _tr.apply_linear(lin, A_te)
    P_te = par_b(S_te, rng)
    sources = {"raw A": A_te, "linear": L_te, "diffusion": T_te,
               "synthetic + impairments": P_te}
    sig = {k: {"nmse_db": (_tr.nmse_db(v, B_te) if k != "synthetic + impairments" else None),
               "psd_distance_db": _tr.psd_distance_db(v, B_te)} for k, v in sources.items()}
    zA = np.stack([real_a.noise_only(window, fs, rng) for _ in range(max(8, n_test // 4))])
    zB = np.stack([real_b.noise_only(window, fs, rng) for _ in range(zA.shape[0])])
    zP = par_b(np.zeros_like(zA), rng)
    noise_src = {"real B": zB, "raw A": zA, "linear": _tr.apply_linear(lin, zA),
                 "diffusion": tr.translate_windows(zA, steps=sample_steps, seed=seed + 2),
                 "synthetic + impairments": zP}
    noise_stats = {k: _tr.stats(v) for k, v in noise_src.items()}
    for k in noise_stats:
        noise_stats[k]["psd_distance_db"] = (_tr.psd_distance_db(noise_src[k], zB)
                                             if k != "real B" else 0.0)
    # hallucination: A's noise translated, judged by the energy detector
    F = _cl.stft_power(np.concatenate(list(zB)), 32).mean(axis=0)
    hall = {"eligible": 0, "hallucinated": 0}
    for i in range(int(hallucination_trials)):
        za = real_a.noise_only(window, fs, rng)
        zb = real_b.noise_only(window, fs, rng)
        if _detects(zb, fs, F, pfa):
            continue                          # real B's own noise already 'detects'
        hall["eligible"] += 1
        if _detects(tr.translate_windows(za[None], steps=sample_steps, seed=seed + 10 + i)[0],
                    fs, F, pfa):
            hall["hallucinated"] += 1
    hall["rate"] = hall["hallucinated"] / hall["eligible"] if hall["eligible"] else None
    # downstream: classifiers trained four ways, tested on real B
    S_d, y_d, A_d, B_d = pairs(n_downstream_per * len(kinds), snr_db)
    S_x, y_x, _A_x, B_x = pairs(n_downstream_per * len(kinds), snr_db)
    train_sets = {"translated (diffusion)": tr.translate_windows(A_d, steps=sample_steps,
                                                                 seed=seed + 3),
                  "translated (linear)": _tr.apply_linear(lin, A_d),
                  "synthetic + impairments": par_b(S_d, rng),
                  "real B (upper line)": B_d}
    acc = {}
    for k, X in train_sets.items():
        m = _gc.train_discriminative(X, y_d, len(kinds), steps=classifier_steps, seed=seed)
        acc[k] = float(np.mean(_gc.predict_discriminative(m, B_x) == y_x))
        say(f"trained on {k}: accuracy on real B {acc[k]:.2f}")
    upper = acc["real B (upper line)"]
    gap = {k: upper - v for k, v in acc.items() if k != "real B (upper line)"}
    better = gap["translated (diffusion)"] < gap["synthetic + impairments"]
    result = {"name": name, "from_profile": str(from_profile).lower(),
              "to_profile": str(to_profile).lower(),
              "words": f"{profiles.describe(from_profile)} -> {profiles.describe(to_profile)}",
              "fs": fs, "window": int(window), "kinds": kinds,
              "receivers": {"A": real_a.to_json(), "B": real_b.to_json()},
              "parametric_b": par_words,
              "translator": {"name": tr.card.name, "sha256": tr.card.weights.get("sha256"),
                             "backend": tr.backend, "sample_steps": int(sample_steps)},
              "tier_translated": provenance.tier_for("diffusion_translate"),
              "signal_statistics_vs_real_b": sig, "noise_fingerprints": noise_stats,
              "hallucination": {**hall, "pfa": pfa,
                                "rule": "A's noise translated; the energy detector finds a "
                                        "signal where real B's own noise gave none"},
              "downstream_accuracy_on_real_b": acc, "domain_gap": gap,
              "chance": 1.0 / len(kinds),
              "verdict": (f"Trained on the learned translation the classifier's domain gap is "
                          f"{gap['translated (diffusion)']:.2f}; on synthetic-plus-impairments "
                          f"{gap['synthetic + impairments']:.2f}; on the linear translation "
                          f"{gap['translated (linear)']:.2f}. The translation "
                          f"{'closes' if better else 'does NOT close'} more of the gap than "
                          "the parametric model."),
              "seconds": None}
    result["seconds"] = round(time.time() - t_start, 1)
    target = out_dir
    if target is None and rf is not None:
        from atk_diffusion.experiments.weak_burst import report_dir
        target = report_dir(rf, str(to_profile).lower(), name)
    if target is not None:
        from atk_diffusion.experiments.weak_burst import write_report
        md, js, det = write_report(target, name, result, [
            f"A6 on stand-in receivers: {result['words']}, {len(kinds)} kinds, "
            f"{window}-sample windows.",
            f"Synthetic-plus-impairments: {par_words}.",
            result["verdict"],
            (f"The translator put a signal into {hall['rate']:.0%} of A's noise windows "
             f"({hall['hallucinated']}/{hall['eligible']}).") if hall["rate"] is not None
            else "No hallucination trials were eligible.",
            "Translated data is INVENTED tier, labelled atk:translated_from."],
            detail=report_lines(result), rf=rf)
        result["report_md"], result["report_json"] = str(md), str(js)
        result["report_detail_md"] = str(det)
    return result


def report_lines(result: dict) -> list[str]:
    L = [f"# A6 — receiver-to-receiver translation — {result['words']}", "",
         f"- parametric B: {result['parametric_b']}",
         f"- translator: {result['translator']['name']} ({result['translator']['backend']})",
         "", "## Against real B (held-out pairs)", "",
         "| source | NMSE (dB) | spectrum distance (dB) |", "|---|---|---|"]
    for k, v in result["signal_statistics_vs_real_b"].items():
        n = "—" if v["nmse_db"] is None else f"{v['nmse_db']:.1f}"
        L.append(f"| {k} | {n} | {v['psd_distance_db']:.2f} |")
    L += ["", "## Receiver fingerprints on noise-only windows", "",
          "| source | DC (dB re power) | improperness | levels per RMS | spectrum distance to real B (dB) |",
          "|---|---|---|---|---|"]
    for k, v in result["noise_fingerprints"].items():
        L.append(f"| {k} | {v['dc_db']:.1f} | {v['improperness']:.3f} | "
                 f"{v['levels_per_rms']} | {v['psd_distance_db']:.2f} |")
    L += ["", "## Downstream: a classifier trained on each, tested on real B", "",
          "| trained on | accuracy | domain gap |", "|---|---|---|"]
    for k, a in result["downstream_accuracy_on_real_b"].items():
        g = result["domain_gap"].get(k)
        L.append(f"| {k} | {a:.2f} | {'—' if g is None else f'{g:+.2f}'} |")
    h = result["hallucination"]
    L += ["", f"Hallucination: {h['rule']} — "
          + (f"{h['rate']:.3f} ({h['hallucinated']}/{h['eligible']})" if h["rate"] is not None
             else "no eligible trials"),
          "", result["verdict"]]
    return L
