# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""AI-generated variations: a diffusion model taught on the training
library, then asked for variations of each real original (RFMLS Training,
2026-10-26).

Bill, 2026-10-26: *"from one signal, we should be able to use generative ai
to generate hundreds of examples for training."*

HOW, AND WHY IT WORKS FROM ONE SIGNAL. A diffusion model trained on ONE
recording can only give that recording back. So the generator for a receiver
and rate class is taught on everything the library and the generator know
at that rate:

  * every type's REAL originals (their train spans only — never a kept-aside
    original), each window labelled with its type;
  * their dirty copies, when asked (more views of the same transmissions);
  * synthetic examples of every class-table class the native generator can
    make at that rate (`fresh.make_one`) — the model's general knowledge of
    what FSK, PSK, OFDM, FM … look like at THIS receiver's rate,

with the label dropped one time in ten, so the model also learns what a
signal at this rate looks like in general. Then a variation is made the
SDEdit way (`learn.augment.Augmenter.augment`, TFD-lite, plan B4): a real
window is noised (and its spectrum blurred) part of the way to pure noise —
`strength` says how far — and the model brings it back, steered by the type's
label. Close to 0 the variation is the original; towards 1 the model
invents more. The coarse shape that survives the noising — the band, the
burst rhythm, the envelope — is kept; the fine detail is the model's.

THE QUALITY GATE. A variation that no longer MEASURES like its parent is
thrown away, counted and reported (`quality`): its occupied bandwidth must be
within 0.6–1.6 × the parent window's, its centre within a quarter of that
bandwidth, and the shape of its spectrum (dB, inside the cut's low-pass)
must correlate with the parent's at r ≥ 0.7. What passes is filtered with
the cut's own low-pass — whatever the model put outside the box, the field
cut would have removed.

TIERS. Every variation is INVENTED (`provenance`: "made by a generative
model … a lead, never a reading"), marked atk:generator_model with the
weights' hash, and NEVER used to grade a model — the dataset builder keeps
them out of validation and test. The plan's rule for B4 stands: generated
examples are kept only if they help; the RFMLS tab offers the comparison
(train with and without).

LIMITS, stated. The network is this package's small 1D U-Net (TFD-lite);
it learns the look of a signal, not its protocol — a generated DMR window
need not hold a valid sync word, and nothing here claims it does. With
very few real examples the variations stay close to them; the honest cure is
more real recordings. Training needs PyTorch (the training environment) and
is meant for the GPU; tests run it tiny on the CPU.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from atk_diffusion import cards as _cards
from atk_diffusion import provenance as _prov
from atk_diffusion.detect import classes as _classes
from atk_diffusion.library import store as _store

KIND = "augmenter"
WEIGHTS = "augmenter.pt"
LABEL_DROPOUT = 0.1

#: How far a variation is noised before the model brings it back.
STRENGTH_RANGE = (0.05, 0.9)
DEFAULT_STRENGTH = 0.35

#: The quality gate (module docstring).
BW_RATIO = (0.6, 1.6)
CENTRE_FRACTION = 0.25
PSD_R_MIN = 0.7

#: Training sizes: steps and the network, by the tab's three lengths.
PRESETS = {
    "quick":    {"steps": 600, "unet": "small", "windows_per_original": 24,
                 "synthetic_per_class": 24},
    "normal":   {"steps": 4000, "unet": "small", "windows_per_original": 64,
                 "synthetic_per_class": 96},
    "thorough": {"steps": 12000, "unet": "small", "windows_per_original": 128,
                 "synthetic_per_class": 192},
}


class _Like:
    """Enough of a SignalType for fresh.make_one (a class to make)."""

    def __init__(self, link: str, canonical: str):
        self.link, self.canonical, self.name = link, canonical, link
        self.words = link


def find(rf, profile: str, canonical: str) -> tuple[Path, object] | None:
    """The newest library generator for (profile, rate class), or None."""
    for d, card in _cards.find(rf, profile, KIND):
        inp = card.input or {}
        if inp.get("library") and (inp.get("canonical") or {}).get("cls") == canonical:
            return d, card
    return None


def stale_originals(lib, card_input: dict) -> list[str]:
    """The originals a generator was taught on that no longer train
    ("type/id"): set aside to grade, or removed, since."""
    out = []
    for tname, ids in ((card_input or {}).get("originals") or {}).items():
        try:
            now = {e.id: e for e in lib.examples(tname, "originals",
                                                  include_removed=True)}
        except Exception:                                 # noqa: BLE001
            now = {}
        for i in ids:
            e = now.get(i)
            if e is None or e.removed or (e.role or "train") != "train":
                out.append(f"{tname}/{i}")
    return out


def describe(rf, profile: str, canonical: str) -> str:
    found = find(rf, profile, canonical)
    if found is None:
        return (f"No generator has been taught for {canonical}-width signals "
                "on this receiver yet.")
    d, card = found
    inp = card.input or {}
    types = inp.get("types") or []
    words = (f"Generator {card.name}, taught {card.created} on "
             f"{len(types)} type{'s' if len(types) != 1 else ''} "
             f"({', '.join(types[:6])}{'…' if len(types) > 6 else ''}) and "
             f"{len(inp.get('synthetic_classes') or [])} synthetic classes; "
             f"{(card.metrics or {}).get('steps', '?')} steps.")
    try:
        stale = stale_originals(_store.TrainingLibrary(rf, profile), inp)
    except Exception:                                     # noqa: BLE001
        stale = []
    if stale:
        words += (f" ⚠ {len(stale)} of the real examples it learned from now "
                  "grade the model (or were removed) — teach it again before "
                  "making more.")
    return words


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------
def _windows_from(lib, ex, span, W: int, k: int, rng, facts) -> list[np.ndarray]:
    s0, n = span
    out = []
    if n <= W:
        x = lib.load(ex, s0, n)
        out.append(_store.pad_with_floor(x, W, float(facts.get("floor_per_hz") or 1e-12),
                                         ex.rate, facts.get("lowpass_hz"), rng))
        return out
    for _ in range(int(k)):
        a = int(rng.integers(s0, s0 + n - W + 1))
        out.append(lib.load(ex, a, W))
    return out


def _facts(meta: dict) -> dict:
    g = meta.get("global", {})
    m = g.get("atk:measured") or {}
    return {"floor_per_hz": m.get("floor_per_hz"),
            "lowpass_hz": g.get("atk:lowpass_hz"),
            "bandwidth_hz": m.get("bandwidth_hz")}


def _synthetic_classes(lib, canonical: str) -> list[str]:
    """Class-table classes the native generator makes at this rate that are
    cut to `canonical` (and fit it)."""
    from atk_diffusion.library import fresh as _fresh
    from atk_diffusion import profiles as _profiles
    from atk_diffusion.synth import native as _native
    can = lib.rate_of(canonical)
    out = []
    for c in _classes.CLASSES:
        if c.negative or not c.native:
            continue
        if c.bandwidth_hz > 0:
            if _profiles.canonical_for(lib.fs, c.bandwidth_hz).cls != canonical:
                continue
            if c.bandwidth_hz > 0.9 * can.rate:
                continue
        ok, _why = _native.can_generate(c.name, lib.fs)
        if ok:
            out.append(c.name)
    _ = _fresh
    return out


def gather(lib: _store.TrainingLibrary, canonical: str, W: int,
           windows_per_original: int, synthetic_per_class: int,
           include_dirty: bool = True, seed: int = 0, progress=None,
           used: dict | None = None
           ) -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict]:
    """The generator's teaching set: (X [N, W] complex, y [N], types,
    synthetic classes, counts). `used`, when given, is filled with
    {type: [original ids]} — what the card records, so a later role change
    is caught (`generate` refuses a generator taught on what now grades)."""
    from atk_diffusion.library import fresh as _fresh
    rng = np.random.default_rng(int(seed))
    types = [t for t in lib.types() if t.canonical == canonical]
    X, y, counts = [], [], {}
    names: list[str] = []
    for st in types:
        origs = [e for e in lib.examples(st.name, "originals")
                 if (e.role or "train") == "train"]
        if not origs:
            continue
        ti = len(names)
        names.append(st.name)
        if used is not None:
            used[st.name] = sorted(e.id for e in origs)
        n0 = len(X)
        for ex in origs:
            facts = _facts(lib.meta(ex))
            for w in _windows_from(lib, ex, lib.train_span(ex), W,
                                   windows_per_original, rng, facts):
                X.append(w)
                y.append(ti)
        if include_dirty:
            dirt = lib.examples(st.name, "dirty")
            keep = {e.id for e in origs}
            dirt = [d for d in dirt if d.parent in keep]
            rng.shuffle(dirt)
            for d in dirt[: windows_per_original * len(origs)]:
                facts = _facts(lib.meta(d))
                X.extend(_windows_from(lib, d, (0, d.samples), W, 1, rng, facts))
                y.append(ti)
        counts[st.name] = len(X) - n0
    if not names:
        raise _store.LibraryError(
            f"no type of {canonical}-width signals has a real original used for "
            "training — add real examples first (the generator learns from them)")
    synth = _synthetic_classes(lib, canonical) if synthetic_per_class > 0 else []
    for ci, cls in enumerate(synth):
        like = _Like(cls, canonical)
        for _ in range(int(synthetic_per_class)):
            try:
                x, _f = _fresh.make_one(lib, like, {}, W, (5.0, 25.0), rng)
            except Exception:                              # noqa: BLE001
                continue
            X.append(x)
            y.append(len(names) + ci)
        counts[f"synthetic:{cls}"] = int(synthetic_per_class)
        if progress is not None:
            progress(f"synthetic {cls}: {synthetic_per_class} windows")
    return (np.stack(X).astype(np.complex64), np.asarray(y, np.int64), names,
            synth, counts)


# ---------------------------------------------------------------------------
# teaching the generator
# ---------------------------------------------------------------------------
def teach(rf, profile: str, canonical: str, *, preset: str = "normal",
          steps: int | None = None, window: int = _store.WINDOW,
          unet: str | dict | None = None, include_dirty: bool = True,
          synthetic_per_class: int | None = None,
          windows_per_original: int | None = None, batch: int = 32,
          lr: float = 2e-4, seed: int = 0, device: str = "auto",
          name: str | None = None, progress=None, should_stop=None) -> Path:
    """Teach the generator for (profile, rate class) and save it with its
    card. Returns the model folder."""
    from atk_diffusion.learn import augment as A
    from atk_diffusion.learn import common as K
    from atk_diffusion.learn import unet as U
    torch = K.require_torch()
    if preset not in PRESETS:
        raise _store.LibraryError(f"a length is one of {', '.join(PRESETS)}")
    pz = PRESETS[preset]
    steps = int(steps or pz["steps"])
    lib = _store.TrainingLibrary(rf, profile)
    W = int(window)
    conf = dict(U.SMALL_1D if (unet or pz["unet"]) == "small" else
                U.TINY_1D if (unet or pz["unet"]) == "tiny" else unet)
    if W % U.multiple(conf):
        raise _store.LibraryError(f"the window ({W}) must be a multiple of "
                                  f"{U.multiple(conf)} for this network")
    say = progress or (lambda _m: None)
    say(f"gathering the teaching set for {canonical}-width signals …")
    used: dict = {}
    X, y, names, synth, counts = gather(
        lib, canonical, W,
        int(windows_per_original or pz["windows_per_original"]),
        int(pz["synthetic_per_class"] if synthetic_per_class is None
            else synthetic_per_class),
        include_dirty=include_dirty, seed=seed, progress=say, used=used)
    classes = names + [f"synthetic:{c}" for c in synth]
    conf.update(in_ch=2, num_classes=len(classes))
    dev, dev_words = K.pick_device(device)
    say(dev_words)
    say(f"{len(X)} windows of {W} samples: " + ", ".join(
        f"{k} {v}" for k, v in counts.items()))
    torch.manual_seed(int(seed))
    model = U.build_unet(conf).to(dev)
    cfg = A.TFDConfig(length=W)
    tfd = A.TFDLite(cfg)
    Xn, _rms = A.unit_power(X)
    data = torch.as_tensor(A.iq_to_channels(Xn)).to(dev)
    labels = torch.as_tensor(y).to(dev)
    g = torch.Generator().manual_seed(int(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []
    model.train()
    t0 = time.time()
    stopped = False
    for step in range(steps):
        if should_stop is not None and should_stop():
            stopped = True
            break
        idx = torch.randint(0, data.shape[0], (min(batch, data.shape[0]),),
                            generator=g).to(dev)
        yy = labels[idx].clone()
        drop = torch.rand(yy.shape[0], generator=g).to(dev) < LABEL_DROPOUT
        yy[drop] = len(classes)                      # the "no label" token
        loss = tfd.loss(model, data[idx], yy, g)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if step % max(1, steps // 100) == 0 or step == steps - 1:
            say(f"[{100 * (step + 1) // steps}%] teaching the generator: step "
                f"{step + 1} of {steps}, loss {np.mean(losses[-20:]):.4f}")
    model.eval()
    if stopped and not losses:
        raise _store.LibraryError("stopped before the generator learned anything")
    from atk_diffusion.learn.common import new_model_dir
    name = name or f"generator_{canonical}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"
    d = new_model_dir(rf, profile, name)
    torch.save(model.state_dict(), d / WEIGHTS)
    can = lib.rate_of(canonical)
    card = _cards.new_card(
        name, KIND, profile,
        input={"library": True, "domain": "iq", "window": W,
               "normalize": "unit power per window",
               "canonical": {"cls": canonical, "class": canonical,
                             "rate": float(can.rate),
                             "decimation": int(can.decimation)},
               "types": names, "synthetic_classes": synth,
               "originals": used,
               "label_dropout": LABEL_DROPOUT,
               "tfd": {"T": cfg.T, "schedule": cfg.schedule, "blur": cfg.blur,
                       "blur_max": cfg.blur_max, "length": cfg.length},
               "schedule": tfd.sched.to_json(), "unet": model.config},
        classes=[{"name": c, "source": "synthetic" if c.startswith("synthetic:")
                  else "library", "examples": int(np.sum(y == i))}
                 for i, c in enumerate(classes)],
        datasets=[{"name": f"training library {profile} ({canonical})",
                   "n": int(len(X)),
                   "sha256": _prov.sha256_bytes(np.ascontiguousarray(X).tobytes()),
                   "kind": "iq", "counts": counts}],
        metrics={"final_loss": float(np.mean(losses[-50:])) if losses else None,
                 "steps": len(losses), "steps_asked": steps,
                 "stopped": stopped, "seconds": round(time.time() - t0, 1)},
        license="all rights reserved (TFD reimplemented from the idea of "
                "arXiv 2404.09140; no RF-Diffusion code)",
        trained_on=str(dev_words),
        notes=["AI-generated variations are INVENTED tier and never grade a "
               "model; a variation that does not measure like its parent is "
               "thrown away (library.generative.quality)",
               "taught on real originals' train spans only — kept-aside "
               "originals never reach the generator"])
    _cards.save(d, card, WEIGHTS)
    try:
        rf.record(d / WEIGHTS, "model-weights", name)
        rf.record(d / _cards.CARD_FILE, "model-card", name)
    except OSError:
        pass
    say(f"generator saved: {d}")
    return d


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
def _psd_db(x, fs, nfft=256):
    from scipy.signal import welch
    f, p = welch(np.asarray(x, np.complex128), fs=fs, nperseg=min(nfft, len(x)),
                 return_onesided=False, detrend=False)
    order = np.argsort(f)
    return f[order], 10 * np.log10(p[order] + 1e-30)


def quality(src, gen, fs: float, floor_per_hz: float | None,
            lowpass_hz: float | None) -> dict:
    """Does `gen` still measure like `src`? (module docstring)"""
    from atk_diffusion.library.store import measure
    ms = measure(src, fs, floor_per_hz)
    mg = measure(gen, fs, floor_per_hz)
    out = {"ok": False, "bandwidth_ratio": None, "centre_shift_hz": None,
           "psd_r": None}
    bs, bg = ms.get("bandwidth_hz"), mg.get("bandwidth_hz")
    if not bs:
        out["words"] = "the parent window holds no measurable signal"
        out["no_parent"] = True
        return out
    if not bg:
        out["words"] = "no measurable signal in the variation"
        return out
    ratio = float(bg) / float(bs)
    shift = abs(float(mg.get("centre_hz") or 0.0) - float(ms.get("centre_hz") or 0.0))
    f, ps = _psd_db(src, fs)
    _f, pg = _psd_db(gen, fs)
    band = np.abs(f) <= (float(lowpass_hz) if lowpass_hz else 0.45 * fs)
    r = float(np.corrcoef(ps[band], pg[band])[0, 1]) if band.sum() > 3 else 0.0
    out.update(bandwidth_ratio=round(ratio, 3), centre_shift_hz=round(shift, 1),
               psd_r=round(r, 3))
    why = []
    if not BW_RATIO[0] <= ratio <= BW_RATIO[1]:
        why.append(f"bandwidth {ratio:.2f}× the parent's")
    if shift > CENTRE_FRACTION * float(bs):
        why.append(f"centre moved {shift:,.0f} Hz")
    if not (r >= PSD_R_MIN):
        why.append(f"spectrum shape r = {r:.2f}")
    out["ok"] = not why
    out["words"] = ("measures like its parent" if not why else
                    "thrown away: " + ", ".join(why))
    return out


# ---------------------------------------------------------------------------
# making variations
# ---------------------------------------------------------------------------
def generate(rf, profile: str, type_name: str, per_original: int, *,
             strength: float = DEFAULT_STRENGTH, steps: int = 20,
             seed: int = 0, model_dir=None, device: str = "auto",
             batch: int = 32, progress=None, should_stop=None) -> dict:
    """`per_original` AI-generated variations of every TRAIN original of a
    type, filed under <type>\\generated\\. -> a summary with the gate's
    count of what was thrown away and why."""
    from atk_diffusion.learn import augment as A
    from atk_diffusion.learn import common as K
    K.require_torch()
    lib = _store.TrainingLibrary(rf, profile)
    st = lib.get_type(type_name)
    s = float(strength)
    if not STRENGTH_RANGE[0] <= s <= STRENGTH_RANGE[1]:
        raise _store.LibraryError(f"strength is {STRENGTH_RANGE[0]}–"
                                  f"{STRENGTH_RANGE[1]}")
    per = int(per_original)
    if per < 1 or per > 2000:
        raise _store.LibraryError("variations per original is 1–2000")
    if model_dir is None:
        found = find(rf, profile, st.canonical)
        if found is None:
            raise _store.LibraryError(
                f"no generator has been taught for {st.canonical}-width signals "
                "on this receiver — teach it first")
        model_dir = found[0]
    dev, _w = K.pick_device(device)
    aug = A.Augmenter.load(model_dir, for_profile=profile, device=str(dev))
    inp = aug.card.input or {}
    types = list(inp.get("types") or [])
    if st.name not in types:
        raise _store.LibraryError(
            f"the generator {aug.card.name} was taught before '{st.words}' had "
            "real examples — teach it again so it knows this type")
    label = types.index(st.name)
    stale = stale_originals(lib, inp)
    if stale:
        raise _store.LibraryError(
            f"the generator {aug.card.name} was taught on "
            f"{', '.join(stale[:6])}{' …' if len(stale) > 6 else ''}, which "
            "are now kept aside to grade the model (or removed). Teach it "
            "again, so nothing it makes comes from a grading recording.")
    W = int(aug.cfg.length)
    origs = [e for e in lib.examples(st.name, "originals")
             if (e.role or "train") == "train"]
    if not origs:
        raise _store.LibraryError(f"'{st.words}' has no real original used for "
                                  "training")
    rng = np.random.default_rng(int(seed))
    t0 = time.time()
    made, thrown, reasons = 0, 0, {}
    total = per * len(origs)
    wsha = (aug.card.weights or {}).get("sha256", "")
    for oi, ex in enumerate(origs):
        meta = lib.meta(ex)
        facts = _facts(meta)
        span = lib.train_span(ex)
        got = 0
        tries = 0
        ids = lib.next_copy_ids(st.name, "generated", ex.id, per)
        while got < per and tries < 4 * per:
            if should_stop is not None and should_stop():
                return _done(st, made, total, thrown, reasons, t0, True, aug)
            k = min(batch, per - got + 4)
            src = _windows_from(lib, ex, span, W, k, rng, facts)
            if len(src) < k:                 # a short original: one window, repeated
                src = [src[0]] * k
            Xs = np.stack(src).astype(np.complex64)
            Z = aug.augment(Xs, np.full(len(Xs), label), strength=s, steps=int(steps),
                            seed=int(rng.integers(0, 2 ** 31)))
            for xs, z in zip(Xs, Z):
                tries += 1
                if got >= per:
                    break
                lp = facts.get("lowpass_hz")
                if lp and float(lp) < 0.49 * ex.rate:
                    z = _store.lowpass(z, float(lp), ex.rate)
                q = quality(xs, z, ex.rate, facts.get("floor_per_hz"), lp)
                if q.get("no_parent"):
                    continue
                if not q["ok"]:
                    thrown += 1
                    for part in q["words"].replace("thrown away: ", "").split(", "):
                        key = part.split(" ")[0]
                        reasons[key] = reasons.get(key, 0) + 1
                    continue
                extra = {"atk:parent": ex.id, "atk:generator_model": aug.card.name,
                         "atk:model_sha256": wsha, "atk:strength": s,
                         "atk:sampling_steps": int(steps), "atk:quality": q,
                         "atk:seed": [int(seed), oi, got],
                         "atk:snr_db": (meta["global"].get("atk:measured") or {}).get("snr_db"),
                         "atk:measured": meta["global"].get("atk:measured"),
                         "atk:lowpass_hz": lp, "atk:canonical_rate": ex.rate,
                         "atk:carrier_offset_hz": float((meta["global"].get(
                             "atk:measured") or {}).get("centre_hz") or 0.0)}
                centre = float(meta["captures"][0].get("core:frequency", 0.0))
                lib.write_copy(st.name, "generated", ids[got], z, ex.rate, centre,
                               extra, f"AI-generated variation of {ex.id} "
                                      f"({st.words}) — INVENTED")
                got += 1
                made += 1
                if progress is not None and (made == total or made % max(1, total // 50) == 0):
                    progress(f"[{100 * made // max(1, total)}%] AI variations of "
                             f"{st.words}: {made}/{total} kept, {thrown} thrown away")
    return _done(st, made, total, thrown, reasons, t0, False, aug)


def _done(st, made, total, thrown, reasons, t0, stopped, aug) -> dict:
    tried = made + thrown
    words = (f"{made} AI-generated variation{'s' if made != 1 else ''} of "
             f"{st.words} kept"
             + (f"; {thrown} thrown away by the quality gate "
                f"({', '.join(f'{k} {v}' for k, v in reasons.items())})"
                if thrown else "")
             + (" (stopped before the end)" if stopped else ""))
    return {"type": st.name, "kind": "generated", "made": made, "asked": total,
            "thrown_away": thrown, "reasons": reasons,
            "kept_fraction": round(made / tried, 3) if tried else None,
            "model": aug.card.name, "seconds": round(time.time() - t0, 2),
            "stopped": stopped, "words": words,
            "tier": _prov.TIER_WORDS["invented"]}
