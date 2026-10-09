# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""B4's experiment — is TFD-lite augmentation kept? (plan §4.B4, §6 Phase
2, §7.)

Plan B4: *"Kept only if it closes the domain gap (§7): train with and
without, test on cabled."* Phase 2's exit: *"the domain gap with and without
augmentation."* This is that test at tiny scale on synthetic stand-ins:

* SYNTHETIC — what the dataset builder makes for the profile: clean signals
  plus white noise through the profile's MEASURED impairments
  (`learn.augment.impair` -> `dsp.impair`; a textbook receiver when the
  profile is unmeasured, and the result says so).
* CABLED-LIKE — the held-out test set: the same kinds through a receiver
  with impairments the synthetic lacked (front-end droop and ripple, soft
  compression, LO phase noise and offset, IQ image, DC, 8-bit converter) —
  the stand-in for the cabled loop's recordings (plan §3.5).
* A LITTLE REAL — a few cabled-like examples per class: what TFD-lite learns
  "what real looks like" from (plan §3.6), class-conditionally.

Classifiers (`learn.genclass.train_discriminative`) are trained on: the
synthetic set alone; synthetic + TFD-augmented copies of it; the little real
alone; synthetic + the little real (the honest alternative — if just adding
the real examples does as well, augmentation has not earned its place); and
plenty of cabled-like (the upper line). All are tested on the held-out
cabled-like set. The DOMAIN GAP of each is the upper line's accuracy minus
its own. KEPT means: augmentation shrinks the gap against both the plain
synthetic and the synthetic + little-real baselines.

The augmentation's own honesty number: the LABEL-FLIP RATE — augmented
examples the upper-line classifier assigns to another class, beyond its
error on the same examples before augmentation. An augmenter that changes
what a signal IS has invented a different signal (plan §2.1, in its
training-data form).

LIMITS. Stand-in receivers; tiny data and models in tests. One seed is one
draw: on Bill's machine run several seeds and read the spread before
calling a difference real.
"""

from __future__ import annotations

import time

import numpy as np

from atk_diffusion import profiles
from atk_diffusion.learn import augment as _aug

DEFAULT_KINDS = ("bpsk", "qpsk", "gfsk", "ofdm")


def cabled_like() -> _aug.LocalReceiver:
    """The stand-in for the cabled recordings: impairments synthetic data
    built from a terminated measurement does not carry."""
    return _aug.LocalReceiver(name="cabled-like (stand-in)", adc_bits=8, signal_dbfs=-20.0,
                              noise_dbfs=-34.0, dc_dbfs=-36.0, iq_gain_db=0.7,
                              iq_phase_deg=3.5, freq_offset_hz=0.0, linewidth_hz=400.0,
                              edge_droop_db=3.0, ripple_db=0.5, ripple_cycles=4.0,
                              compression_dbfs=-17.0)


def run(rf=None, profile: str = "rtlsdr_2400000_cu8", *, augmenter=None,
        kinds=DEFAULT_KINDS, window: int = 128, n_synth_per: int = 200,
        n_real_per: int = 8, n_test_per: int = 100, snr_db=(0.0, 20.0),
        tfd_steps: int = 3000, tfd: _aug.TFDConfig | None = None, unet: dict | None = None,
        strength: float = 0.4, n_aug_per: int = 1, sample_steps: int = 10,
        classifier_steps: int = 400, seed: int = 0, out_dir=None,
        name: str = "augment_eval", progress=None) -> dict:
    """The B4 experiment (module docstring). Trains TFD-lite on the little
    real unless `augmenter` (a model folder or Augmenter) is given."""
    from atk_diffusion.learn import genclass as _gc
    say = progress or (lambda s: None)
    t_start = time.time()
    pid = str(profile).lower()
    prof = profiles.load_profile(rf, pid) if rf is not None else profiles.new_profile(pid)
    fs = float(prof.sample_rate)
    rng = np.random.default_rng(seed)
    kinds = list(kinds)
    cab = cabled_like()

    def synthetic(n_per):
        X, y = _aug.make_classification_set(kinds, n_per, window, fs, rng, snr_db=snr_db,
                                            offset_frac=0.1)
        out, words = [], ""
        for x in X:
            z, words = _aug.impair(0.1 * x, fs, prof, rng)
            out.append(z)
        return np.stack(out).astype(np.complex64), y, words

    def cabled(n_per):
        X, y = _aug.make_classification_set(kinds, n_per, window, fs, rng, snr_db=(60.0, 60.0),
                                            offset_frac=0.1)
        lvl = 10 ** (rng.uniform(*snr_db, len(X)) / 20.0) * 10 ** (-14 / 20.0)
        return np.stack([cab.apply(x * g, fs, rng) for x, g in zip(X, lvl)]), y

    Xs, ys, imp_words = synthetic(n_synth_per)
    Xr, yr = cabled(n_real_per)
    Xt, yt = cabled(n_test_per)
    Xu, yu = cabled(n_synth_per)
    if augmenter is None:
        if rf is None and out_dir is None:
            raise ValueError("training the augmenter needs somewhere to put it: pass rf= "
                             "or out_dir=, or pass a trained augmenter")
        cfg = tfd or _aug.TFDConfig(length=window)
        cfg.length = int(window)
        from pathlib import Path
        d = _aug.train_augmenter(rf, pid, Xr, yr, kinds, steps=tfd_steps, unet=unet, cfg=cfg,
                                 seed=seed, progress=say,
                                 out_dir=(Path(out_dir) / "augmenter") if out_dir else None,
                                 name=f"{name}_tfd_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}")
        augmenter = d
    aug = augmenter if isinstance(augmenter, _aug.Augmenter) else \
        _aug.Augmenter.load(augmenter, for_profile=pid)
    Xa, ya, info = _aug.augment_dataset(Xs, ys, aug, strength=strength, n_per=n_aug_per,
                                        steps=sample_steps, seed=seed, keep_original=True)
    sets = {"synthetic": (Xs, ys),
            "synthetic + augmented": (Xa, ya),
            "little real only": (Xr, yr),
            "synthetic + little real": (np.concatenate([Xs, Xr]), np.concatenate([ys, yr])),
            "cabled-like (upper line)": (Xu, yu)}
    acc, models = {}, {}
    for k, (X, y) in sets.items():
        m = _gc.train_discriminative(X, y, len(kinds), steps=classifier_steps, seed=seed)
        models[k] = m
        acc[k] = float(np.mean(_gc.predict_discriminative(m, Xt) == yt))
        say(f"trained on {k}: accuracy on held-out cabled-like {acc[k]:.2f}")
    upper = acc["cabled-like (upper line)"]
    gap = {k: upper - v for k, v in acc.items() if k != "cabled-like (upper line)"}
    inv = info["invented"]
    judge = models["cabled-like (upper line)"]
    err_before = float(np.mean(_gc.predict_discriminative(judge, Xs) != ys))
    err_after = float(np.mean(_gc.predict_discriminative(judge, Xa[inv]) != ya[inv]))
    flip = max(0.0, err_after - err_before)
    kept = (gap["synthetic + augmented"] < gap["synthetic"]
            and gap["synthetic + augmented"] < gap["synthetic + little real"])
    result = {"name": name, "profile": pid, "profile_words": profiles.describe(pid),
              "fs": fs, "window": int(window), "kinds": kinds,
              "synthetic_impairments": imp_words,
              "cabled_like_receiver": cab.to_json(),
              "n": {"synthetic_per_class": int(n_synth_per), "little_real_per_class": int(n_real_per),
                    "test_per_class": int(n_test_per), "augmented_per_example": int(n_aug_per)},
              "augmenter": {"name": aug.card.name, "sha256": aug.card.weights.get("sha256"),
                            "blur": bool(aug.cfg.blur), "strength": float(strength),
                            "sample_steps": int(sample_steps)},
              "tier_augmented": info["tier"], "accuracy_on_cabled_like": acc,
              "domain_gap": gap, "chance": 1.0 / len(kinds),
              "label_flip_rate": flip,
              "label_flip": {"upper_line_error_before": err_before,
                             "upper_line_error_after": err_after},
              "kept": bool(kept),
              "verdict": (f"Augmentation is {'KEPT' if kept else 'NOT kept'}: the domain gap is "
                          f"{gap['synthetic + augmented']:+.2f} with it, "
                          f"{gap['synthetic']:+.2f} without, and "
                          f"{gap['synthetic + little real']:+.2f} when the little real is simply "
                          f"added to the synthetic set. One seed is one draw."),
              "seconds": round(time.time() - t_start, 1)}
    target = out_dir
    if target is None and rf is not None:
        from atk_diffusion.experiments.weak_burst import report_dir
        target = report_dir(rf, pid, name)
    if target is not None:
        from atk_diffusion.experiments.weak_burst import write_report
        table = ["# B4 — TFD-lite augmentation and the domain gap", "",
                 f"- synthetic: clean signals through {imp_words}",
                 f"- test: held-out cabled-like ({cab.name})", "",
                 "| trained on | accuracy on cabled-like | domain gap |", "|---|---|---|"]
        for k, a in acc.items():
            g = gap.get(k)
            table.append(f"| {k} | {a:.2f} | {'—' if g is None else f'{g:+.2f}'} |")
        table += ["", f"Label-flip rate of augmented examples: {flip:.3f} "
                  f"(upper-line classifier error {err_before:.3f} before, {err_after:.3f} after).",
                  "", result["verdict"]]
        md, js, det = write_report(target, name, result, [
            f"B4 on {result['profile_words']}: {len(kinds)} kinds, {window}-sample windows; "
            f"TFD-lite trained on {n_real_per} cabled-like examples per class.",
            result["verdict"],
            f"Augmented examples changed class {flip:.1%} of the time beyond the "
            "judge's own error (the label-flip rate).",
            "Augmented examples are INVENTED tier and flagged per row."],
            detail=table, rf=rf)
        result["report_md"], result["report_json"] = str(md), str(js)
        result["report_detail_md"] = str(det)
    return result
