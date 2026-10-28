# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""From the training library to the datasets the trainers read (RFMLS
Training, 2026-10-26): the RECOGNIZER's narrowband dataset, and the
FINDER's wideband scenes with the library's signals placed in them.

THE RECOGNIZER (`build_recognizer_dataset`) — the 1D classifier's dataset
in ARCHITECTURE §5's format, one rate class at a time (a classifier is per
canonical class, DETECTION_DESIGN §4, D2):

  * TRAIN  windows of the TRAIN originals' train spans (several views of
           each), and one window of each copy the operator chose to include
           (dirty, fresh, AI-generated) — but only copies whose parent is a
           TRAIN original NOW: a role flipped to "test" takes its copies out
           with it.
  * VAL    the originals kept aside for calibration, and the first half of
           every held-out segment — REAL only.
  * TEST   the originals kept aside to grade, and the second half of every
           held-out segment — REAL only.

So the grade on the card is a grade on real recordings the model never saw,
never on a copy of something it trained on. Every row carries its `tier`
(0 real, 1 dirty, 2 fresh, 3 AI-generated, 4 noise) and its `origin` (the
example it came from), and the manifest's `library` block counts both per
type and split — the tab shows them, and the verdict reads them.

A NOISE CLASS is added (`noise_class=True`): windows of the receiver's own
floor through a cut's low-pass, in every split — so the recognizer has an
honest "nothing here" answer and one type alone still trains.

THE FINDER (`build_finder_dataset`) — wideband tiles for the 2D proposer:
scenes from the native generator (`synth.scene.compose_generic`, every
class this receiver captures, at the profile's exact rate, through its
measured front end) and, with `use_library`, the library's own real signals
and their dirty copies placed in them: interpolated back to the profile's
rate by the INTEGER factor they were decimated by, put at a random frequency
and time, scaled to an SNR, each burst boxed with its type's family. Real
originals already carry the receiver's impairments, so they are placed after
the receiver end. An injected example's own noise rides with it, so its SNR
is capped 6 dB under the SNR it was recorded at — its noise pedestal stays at
least 6 dB under the scene's floor. Only the TILES are written by default
(the trainer reads only those): a scene at 4 MS/s is 16 MB, and a few
hundred of them are not worth keeping (`keep_scenes`).

LIMITS, stated. Held-out segments of a train original are the same
transmission as its train span — a weak test, said in the verdict. The
finder's library signals are placed on a synthetic background, not a real
one.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from atk_diffusion import cards as _cards
from atk_diffusion import profiles as _profiles
from atk_diffusion import provenance as _prov
from atk_diffusion import sigmf as _sigmf
from atk_diffusion.detect import classes as _classes
from atk_diffusion.library import store as _store
from atk_diffusion.synth import datasets as _D
from atk_diffusion.synth import labels as _labels

_prov.METHOD_TIERS.setdefault("training_library", "invented")

TIER_CODES = {"originals": 0, "dirty": 1, "fresh": 2, "generated": 3, "noise": 4}
TIER_NAMES = {v: k for k, v in TIER_CODES.items()}
COPY_KINDS = ("dirty", "fresh", "generated")
NOISE = "noise"
MAX_HELDOUT_WINDOWS = 64          # per held-out source
DEFAULT_TRAIN_CAP = 3000          # train windows per type


# ---------------------------------------------------------------------------
# what a build would hold (the tab shows this before anything runs)
# ---------------------------------------------------------------------------
def _type_sources(lib, st, include, W: int):
    """(train, val, test) source lists for one type: each source is
    (example, start, count, tier code)."""
    origs = lib.examples(st.name, "originals")
    train_ids = {e.id for e in origs if (e.role or "train") == "train"}
    tr, va, te = [], [], []
    for e in origs:
        role = e.role or "train"
        if role == "train":
            s0, n = lib.train_span(e)
            tr.append((e, s0, n, 0))
            if e.holdout:
                h0, hn = int(e.holdout["start"]), int(e.holdout["count"])
                if hn >= 2 * W:
                    half = hn // 2
                    va.append((e, h0, half, 0))
                    te.append((e, h0 + half, hn - half, 0))
                else:
                    te.append((e, h0, hn, 0))
        elif role == "val":
            a, n = _store.signal_span(e)
            va.append((e, a, n, 0))
        else:
            a, n = _store.signal_span(e)
            te.append((e, a, n, 0))
    stale_gen: dict = {}
    for kind in COPY_KINDS:
        if kind not in include:
            continue
        for c in lib.examples(st.name, kind):
            if kind != "fresh" and c.parent not in train_ids:
                continue
            if kind == "generated" and _generator_is_stale(
                    lib, lib.meta(c)["global"].get("atk:generator_model", ""),
                    stale_gen):
                continue
            tr.append((c, 0, c.samples, TIER_CODES[kind]))
    return tr, va, te


def _generator_is_stale(lib, name: str, memo: dict) -> bool:
    """True when the generator that made an AI-generated copy learned from
    a real example that now grades the model (or was removed) — its copies
    would carry a grading recording into training, so they are left out."""
    if not name:
        return False
    if name not in memo:
        from atk_diffusion.library import generative as _gen
        stale = False
        for d in (Path(lib.rf.models(lib.pid)) / name,
                  Path(lib.rf.models(lib.pid)) / "_retired" / name):
            if (d / _cards.CARD_FILE).exists():
                try:
                    card = _cards.load(d, for_profile=lib.pid, verify_weights=False)
                    stale = bool(_gen.stale_originals(lib, card.input or {}))
                except Exception:                         # noqa: BLE001
                    stale = False
                break
        memo[name] = stale
    return memo[name]


def _separate_test_recordings(lib, st) -> int:
    """How many recordings grade `st` that no training example came from —
    what makes a grade a REAL test rather than the held-back end of a
    recording it also trained on."""
    origs = lib.examples(st.name, "originals")
    train_groups = {e.group for e in origs if (e.role or "train") == "train"}
    return len({e.group for e in origs if (e.role or "") == "test"
                and e.group not in train_groups})


def _windows_in(n: int, W: int) -> int:
    return max(1, min(MAX_HELDOUT_WINDOWS, n // W)) if n >= W else 1


def plan(lib: _store.TrainingLibrary, canonical: str, types=None,
         include=COPY_KINDS, window: int = _store.WINDOW,
         train_cap_per_type: int = DEFAULT_TRAIN_CAP) -> dict:
    """What a recognizer build would hold, per type, and what is wrong with
    it — in words, before anything runs."""
    W = int(window)
    rows, warnings = [], []
    sel = [t for t in lib.types() if t.canonical == canonical
           and (types is None or t.name in types)]
    for st in sel:
        tr, va, te = _type_sources(lib, st, include, W)
        n_orig_train = sum(1 for s in tr if s[3] == 0)
        copies = {k: sum(1 for s in tr if s[3] == TIER_CODES[k]) for k in COPY_KINDS}
        val_w = sum(_windows_in(s[2], W) for s in va)
        test_w = sum(_windows_in(s[2], W) for s in te)
        row = {"type": st.name, "label": st.words, "family": st.family,
               "train_originals": n_orig_train, "copies": copies,
               "val_windows": val_w, "test_windows": test_w,
               "kept_aside": sum(1 for s in va + te if s[0].role in ("val", "test"))}
        if n_orig_train == 0:
            row["problem"] = "no real original used for training"
            warnings.append(f"{st.words}: no real original is used for training "
                            "— it will be left out")
        elif test_w == 0:
            warnings.append(f"{st.words}: nothing is held out to grade it — "
                            "add a second real example (it is kept aside "
                            "automatically) or a longer recording")
        elif not _separate_test_recordings(lib, st):
            warnings.append(f"{st.words}: graded only on the held-back end of "
                            "its own training recording — a weak test; a "
                            "second real recording makes it a real one")
        total_copies = sum(copies.values())
        if n_orig_train and total_copies == 0:
            warnings.append(f"{st.words}: no copies yet — consider dirty copies "
                            "so it holds up on weak and messy signals")
        rows.append(row)
    usable = [r for r in rows if not r.get("problem")]
    if not usable:
        warnings.insert(0, f"no {canonical}-width type has a real original used "
                           "for training — nothing to train")
    return {"canonical": canonical, "types": rows, "usable": len(usable),
            "warnings": warnings, "window": W,
            "train_cap_per_type": int(train_cap_per_type)}


# ---------------------------------------------------------------------------
# the recognizer's dataset
# ---------------------------------------------------------------------------
def _label_facts(lib, ex, st) -> dict:
    g = lib.meta(ex)["global"]
    m = g.get("atk:measured") or {}
    c = _classes.get(st.link) if st.link else None
    sr = 0.0
    if c is not None and len(c.symbol_rates) == 1:
        sr = float(c.symbol_rates[0])
    elif m.get("symbol_rate_hz"):
        sr = float(m["symbol_rate_hz"])
    snr = g.get("atk:snr_db", m.get("snr_db"))
    co = g.get("atk:carrier_offset_hz", m.get("centre_hz"))
    return {"snr_db": float("nan") if snr is None else float(snr),
            "symbol_rate_hz": sr,
            "carrier_offset_hz": 0.0 if co is None else float(co),
            "bandwidth_hz": float(m.get("bandwidth_hz") or 0.0),
            "floor_per_hz": float(m.get("floor_per_hz") or 0.0),
            "lowpass_hz": g.get("atk:lowpass_hz")}


def _cut_windows(lib, src, W: int, k: int, rng, facts, deterministic: bool):
    ex, s0, n, _tier = src
    if n < W:
        x = lib.load(ex, s0, n)
        return [_store.pad_with_floor(x, W, facts["floor_per_hz"] or 1e-12,
                                      ex.rate, facts["lowpass_hz"], rng)]
    if deterministic:
        m = min(k, n // W)
        step = (n - W) // max(1, m - 1) if m > 1 else 0
        return [lib.load(ex, s0 + i * step, W) for i in range(m)]
    return [lib.load(ex, int(rng.integers(s0, s0 + n - W + 1)), W)
            for _ in range(int(k))]


def _flush(rows: list, d: Path, split: str, idx: int, with_scf: bool) -> Path:
    arr = {"iq": np.stack([r["iq"] for r in rows]).astype(np.complex64),
           "label": np.array([r["label"] for r in rows], np.int32),
           "family": np.array([r["family"] for r in rows], np.int32),
           "tier": np.array([r["tier"] for r in rows], np.int8),
           "origin": np.array([r["origin"] for r in rows], np.int32)}
    for k in ("snr_db", "symbol_rate_hz", "carrier_offset_hz", "bandwidth_hz"):
        arr[k] = np.array([r[k] for r in rows], np.float32)
    if with_scf:
        arr["scf"] = np.stack([r["scf"] for r in rows]).astype(np.float16)
    p = d / split / f"shard_{idx:03d}.npz"
    _D._save_npz(p, arr)
    return p


def build_recognizer_dataset(rf, profile: str, canonical: str, name: str, *,
                             types=None, include=COPY_KINDS,
                             window: int = _store.WINDOW,
                             train_cap_per_type: int = DEFAULT_TRAIN_CAP,
                             views_per_original: int | None = None,
                             noise_class: bool = True, compute_scf: bool = True,
                             seed: int = 0, progress=None, overwrite: bool = False,
                             should_stop=None) -> dict:
    """The library -> a narrowband dataset for one rate class (module
    docstring). Returns the manifest (with 'path')."""
    lib = _store.TrainingLibrary(rf, profile)
    prof = lib.profile
    can = lib.rate_of(canonical)
    W = int(window)
    if W < 64:
        raise _store.LibraryError("a window shorter than 64 samples holds no signal")
    include = tuple(k for k in include if k in COPY_KINDS)
    pl = plan(lib, canonical, types, include, W, train_cap_per_type)
    sel = [t for t in lib.types() if t.canonical == canonical
           and (types is None or t.name in types)
           and any(r["type"] == t.name and not r.get("problem") for r in pl["types"])]
    if not sel:
        raise _store.LibraryError(pl["warnings"][0] if pl["warnings"] else
                                  "nothing to train")
    classes = [t.name for t in sel] + ([NOISE] if noise_class else [])
    if len(classes) < 2:
        raise _store.LibraryError("a recognizer needs two answers to choose "
                                  "between: add a second type, or keep the "
                                  "noise class on")
    say = progress or (lambda _m: None)
    rng = np.random.default_rng(int(seed))
    scf = None
    scf_why = "not computed: compute_scf=False"
    if compute_scf:
        scf, scf_why = _D._scf_fn()
    notes: list[str] = []
    d = _D._prepare(rf, prof, name, overwrite, notes)
    bufs = {s: [] for s in _D.SPLITS}
    idx = {s: 0 for s in _D.SPLITS}
    written: list[Path] = []
    origins: list[str] = []
    counts = {}
    floors, lowpasses = [], []
    test_sources: dict = {}
    t0 = time.time()

    def put(split: str, win, ci: int, fam: int, tier: int, origin: int, facts):
        row = {"iq": np.ascontiguousarray(win, np.complex64), "label": ci,
               "family": fam, "tier": tier, "origin": origin,
               "snr_db": facts["snr_db"], "symbol_rate_hz": facts["symbol_rate_hz"],
               "carrier_offset_hz": facts["carrier_offset_hz"],
               "bandwidth_hz": facts["bandwidth_hz"]}
        if scf is not None:
            img, _, _ = scf(row["iq"], float(can.rate), prof.fam,
                            out_shape=_D.SCF_SHAPE)
            row["scf"] = img.astype(np.float16)
        bufs[split].append(row)
        if len(bufs[split]) >= _D.SHARD_SIZE:
            written.append(_flush(bufs[split], d, split, idx[split], scf is not None))
            idx[split] += 1
            bufs[split] = []

    try:
        n_types = len(sel)
        for ti, st in enumerate(sel):
            fam = _labels.family_index(st.family)
            tr, va, te = _type_sources(lib, st, include, W)
            c = {"train": {}, "val": {}, "test": {}}
            orig_tr = [s for s in tr if s[3] == 0]
            copy_tr = [s for s in tr if s[3] != 0]
            views = int(views_per_original or max(8, min(
                256, int(train_cap_per_type) // max(1, 4 * len(orig_tr)))))
            # originals first: they are never dropped by the cap
            plan_tr = [(s, views) for s in orig_tr]
            room = max(0, int(train_cap_per_type) - views * len(orig_tr))
            if len(copy_tr) > room:
                pick = rng.choice(len(copy_tr), size=room, replace=False)
                copy_tr = [copy_tr[i] for i in sorted(pick)]
                notes.append(f"{st.name}: {room} of its copies used (the "
                             f"per-type cap is {train_cap_per_type} windows)")
            plan_tr += [(s, 1) for s in copy_tr]
            for split, items, det in (("train", plan_tr, False),
                                      ("val", [(s, MAX_HELDOUT_WINDOWS) for s in va], True),
                                      ("test", [(s, MAX_HELDOUT_WINDOWS) for s in te], True)):
                for k, (src, nv) in enumerate(items):
                    if should_stop is not None and should_stop():
                        raise KeyboardInterrupt("stopped")
                    facts = _label_facts(lib, src[0], st)
                    if src[3] == 0:
                        if facts["floor_per_hz"]:
                            floors.append(facts["floor_per_hz"])
                        if facts["lowpass_hz"]:
                            lowpasses.append(float(facts["lowpass_hz"]))
                    origin = len(origins)
                    origins.append(f"{st.name}/{src[0].kind}/{src[0].id}")
                    for win in _cut_windows(lib, src, W, nv, rng, facts, det):
                        put(split, win, ti, fam, src[3], origin, facts)
                        key = _store.KIND_WORDS[TIER_NAMES[src[3]]]
                        c[split][key] = c[split].get(key, 0) + 1
            counts[st.name] = c
            test_sources[st.name] = _separate_test_recordings(lib, st)
            say(f"[{int(90 * (ti + 1) / n_types)}%] {st.words}: "
                f"{sum(c['train'].values())} train, {sum(c['val'].values())} "
                f"val, {sum(c['test'].values())} test windows")
        if noise_class:
            fl = float(np.median(floors)) if floors else 1e-9
            lps = lowpasses or [0.25 * float(can.rate)]
            per = {s: int(np.median([sum(counts[t][s].values()) for t in counts]))
                   for s in _D.SPLITS}
            per["val"] = max(per["val"], 8)
            per["test"] = max(per["test"], 8)
            ci = classes.index(NOISE)
            origin = len(origins)
            origins.append("noise (the receiver's floor through a cut's low-pass)")
            c = {"train": {}, "val": {}, "test": {}}
            for split in _D.SPLITS:
                for _ in range(per[split]):
                    lp = float(rng.choice(lps))
                    win = _store.floor_noise(W, fl, float(can.rate), lp, rng)
                    put(split, win, ci, _labels.family_index("unknown"), 4, origin,
                        {"snr_db": float("nan"), "symbol_rate_hz": 0.0,
                         "carrier_offset_hz": 0.0, "bandwidth_hz": 0.0})
                c[split]["noise"] = per[split]
            counts[NOISE] = c
        for s in _D.SPLITS:
            if bufs[s]:
                written.append(_flush(bufs[s], d, s, idx[s], scf is not None))
    except BaseException:
        _D._abandon(d)
        raise
    m = _D._common_manifest(prof, name, "narrowband", "the training library",
                            "training_library")
    split_n = {s: sum(sum(counts[t][s].values()) for t in counts) for s in _D.SPLITS}
    heldout = {t: sum(counts[t]["val"].values()) + sum(counts[t]["test"].values())
               for t in counts if t != NOISE}
    m.update({
        "generator": "the training library (real originals and their copies)",
        "label_sources": ["record"] + [f"{k}_copy" for k in include],
        "canonical": {"class": can.cls, "rate": float(can.rate),
                      "decimation": int(can.decimation), "limited": bool(can.limited)},
        "window": {"samples": W, "seconds": W / float(can.rate)},
        "classes": classes, "splits": split_n,
        "class_families": {t.name: t.family for t in sel},
        "scf": (f"computed: cyclo.scf.scf_image {_D.SCF_SHAPE[0]}x{_D.SCF_SHAPE[1]}"
                if scf is not None else scf_why),
        "library": {"counts": counts, "heldout_real_windows": heldout,
                    "test_sources": test_sources,
                    "include": list(include), "origins": origins,
                    "tier_codes": {v: k for k, v in TIER_CODES.items()},
                    "rule": "validation and test hold REAL windows only; copies "
                            "train only, and only when their parent is a train "
                            "original",
                    "plan_warnings": pl["warnings"]},
        "params": {"canonical": canonical, "window": W, "seed": int(seed),
                   "types": [t.name for t in sel], "include": list(include),
                   "train_cap_per_type": int(train_cap_per_type),
                   "noise_class": bool(noise_class),
                   "build_seconds": round(time.time() - t0, 2)},
        "notes": notes or ["built from the training library"],
        "tier": "invented" if any(k in include for k in COPY_KINDS) else "record",
    })
    say("[95%] writing the manifest")
    return _D._finish(rf, d, m, written, "dataset-shard", overwrite, prof)


# ---------------------------------------------------------------------------
# the finder's scenes
# ---------------------------------------------------------------------------
def _library_pools(lib, include_dirty: bool = True) -> dict:
    """The signals placed in finder scenes, per split (2026-10-26 review:
    one pool for every split graded the finder on signals it trained on).

    train: the train originals' train spans, and their dirty copies.
    held:  the KEPT-ASIDE originals (other recordings — a real test); else
           the held-back tails of the train originals (the same recordings —
           a weak test); else nothing, and the scenes reuse training signals
           (said in the manifest, and the verdict caps itself)."""
    train, aside, tails = [], [], []
    for st in lib.types():
        if not st.canonical:
            continue
        origs = lib.examples(st.name, "originals")
        tr = [e for e in origs if (e.role or "train") == "train"]
        ids = {e.id for e in tr}
        for e in tr:
            train.append((st, e, lib.train_span(e)))
            if e.holdout:
                tails.append((st, e, (int(e.holdout["start"]),
                                      int(e.holdout["count"]))))
        for e in origs:
            if (e.role or "train") != "train":
                aside.append((st, e, _store.signal_span(e)))
        if include_dirty:
            for c in lib.examples(st.name, "dirty"):
                if c.parent in ids:
                    train.append((st, c, (0, c.samples)))
    if aside:
        held, how = aside, "kept_aside"
    elif tails:
        held, how = tails, "held_back_tails"
    else:
        held, how = train, "reused"
    return {"train": train, "held": held, "held_how": how}


def _inject(x, fs: float, lib, st, ex, span, p_floor: float, rng,
            snr_range=(5.0, 25.0)) -> list:
    """Place one library example in the scene `x` (in place). -> its
    annotations (one per burst)."""
    from scipy.signal import resample_poly
    from atk_diffusion.dsp import measure as _m
    g = lib.meta(ex)["global"]
    m = g.get("atk:measured") or {}
    bw = float(m.get("bandwidth_hz") or 0.0)
    s_pow = float(m.get("signal_power") or 0.0)
    snr0 = g.get("atk:snr_db", m.get("snr_db"))
    if bw <= 0 or s_pow <= 0 or snr0 is None:
        return []
    can = lib.rate_of(st.canonical)
    d = int(can.decimation)
    n_scene = x.size
    max_c = max(1, n_scene // d)
    s0, n = span
    cnt = min(n, max_c)
    a = int(rng.integers(s0, s0 + n - cnt + 1)) if n > cnt else s0
    y = lib.load(ex, a, cnt)
    up = resample_poly(np.asarray(y, np.complex128), d, 1) if d > 1 else y.astype(np.complex128)
    up = up[:n_scene]
    snr = min(float(rng.uniform(*snr_range)), float(snr0) - 6.0)
    if snr < 0:
        return []
    n0 = p_floor / fs                              # scene floor, per Hz
    amp = math.sqrt(10 ** (snr / 10.0) * n0 * bw / s_pow)
    room = 0.45 * fs - 0.5 * bw
    if room <= 0:
        return []
    f_place = float(rng.uniform(-room, room))
    start = int(rng.integers(0, max(1, n_scene - up.size + 1)))
    t = (np.arange(up.size) + start) / fs
    x[start:start + up.size] += amp * up * np.exp(2j * np.pi * f_place * t)
    # one box per burst (the envelope of the canonical-rate example)
    try:
        eb = _m.envelope_bursts(y, float(can.rate))
    except Exception:                                     # noqa: BLE001
        eb = {"bursts": []}
    bursts = [(int(round(b["t0_s"] * fs)), int(round((b["t1_s"] - b["t0_s"]) * fs)))
              for b in eb.get("bursts") or [] if b.get("t1_s", 0) > b.get("t0_s", 0)]
    if not bursts or len(bursts) > 64:
        bursts = [(0, up.size)]
    centre = f_place + float(m.get("centre_hz") or 0.0)
    anns = []
    for b0, bk in bursts:
        b0 = max(0, min(up.size - 1, b0))
        bk = max(1, min(up.size - b0, bk))
        anns.append(_sigmf.Annotation(
            sample_start=start + b0, sample_count=bk,
            freq_lower_edge=centre - 0.5 * bw, freq_upper_edge=centre + 0.5 * bw,
            label=st.name, extra={"atk:family": st.family, "atk:snr_db": snr,
                                  "atk:source": "library",
                                  "atk:example": f"{st.name}/{ex.kind}/{ex.id}"}))
    return anns


def build_finder_dataset(rf, profile: str, name: str, n_scenes: int, *,
                         use_library: bool = True, synthetic: bool = True,
                         per_scene=(1, 4), include_dirty: bool = True,
                         scene_seconds: float | None = None, seed: int = 0,
                         splits=(0.8, 0.1, 0.1), keep_scenes: bool = False,
                         progress=None, overwrite: bool = False,
                         should_stop=None) -> dict:
    """Wideband tiles for the 2D proposer (module docstring)."""
    from atk_diffusion.dsp import impair as _impair
    from atk_diffusion.synth import native as _native
    from atk_diffusion.synth import scene as _scene
    lib = _store.TrainingLibrary(rf, profile)
    prof = lib.profile
    fs = float(prof.sample_rate)
    ns = int(n_scenes)
    if ns < 1:
        raise _store.LibraryError("the finder needs at least one scene")
    if not use_library and not synthetic:
        raise _store.LibraryError("a scene needs the library's signals, the "
                                  "generator's, or both")
    pools = _library_pools(lib, include_dirty) if use_library else \
        {"train": [], "held": [], "held_how": ""}
    pool = pools["train"]
    if use_library and not pool:
        if not synthetic:
            raise _store.LibraryError("the library has no real original used for "
                                      "training to place in scenes")
        use_library = False
    secs = float(scene_seconds if scene_seconds is not None else prof.stft.tile_seconds)
    if secs * fs < prof.stft.fft_size:
        raise _store.LibraryError("a scene shorter than one FFT frame")
    cls_list = [c.name for c in _classes.CLASSES] + sorted(
        {st.name for st, _e, _s in pool + pools["held"]})
    counts = _D._split_counts(ns, splits)
    notes: list[str] = []
    d = _D._prepare(rf, prof, name, overwrite, notes)
    order = np.random.default_rng(int(seed)).permutation(ns)
    split_of = {}
    a_, b_ = counts["train"], counts["train"] + counts["val"]
    for r, i in enumerate(order):
        split_of[int(i)] = "train" if r < a_ else ("val" if r < b_ else "test")
    dt = prof.datatype
    if dt == "ci16" and prof.adc_bits and prof.adc_bits <= 12:
        dt = "ci16q11"
    written: list[Path] = []
    n_lib = n_syn = 0
    floor_words = ""
    measured = _impair.is_measured(prof.impairments)
    try:
        for i in range(ns):
            if should_stop is not None and should_stop():
                raise KeyboardInterrupt("stopped")
            rng = np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=(11, i)))
            if synthetic:
                r = _scene.compose_generic(prof, 0.0, secs, rng, "native")
                x = r.x.astype(np.complex128)
                anns = list(r.annotations)
                p_scene = 10 ** (r.info["noise_dbfs"] / 10) * 10 ** (r.info["agc_gain_db"] / 10)
                n_syn += len(anns)
            else:
                n = int(round(secs * fs))
                p_scene = _impair.noise_power(prof.impairments if measured else None)
                x = _native.noise(n, p_scene, rng).astype(np.complex128)
                if measured:
                    x = _impair.apply_impairments(x, fs, prof.impairments, rng,
                                                  adc_bits=prof.adc_bits or None,
                                                  datatype=prof.datatype)
                anns = []
            sp = split_of[i]
            use = pool if sp == "train" else (pools["held"] or pool)
            if use_library and use:
                k = int(rng.integers(int(per_scene[0]), int(per_scene[1]) + 1))
                for _ in range(k):
                    st, ex, span = use[int(rng.integers(0, len(use)))]
                    got = _inject(x, fs, lib, st, ex, span, p_scene, rng)
                    anns.extend(got)
                    n_lib += len(got)
            x = _impair.quantise(np.asarray(x, np.complex64), prof.datatype,
                                 prof.adc_bits or None)
            anns.sort(key=lambda a: a.sample_start)
            if keep_scenes:
                base = d / sp / f"scene_{i:03d}"
                dp, mp = _sigmf.write_pair(
                    base, x, fs, 0.0, datatype=dt, annotations=anns,
                    extra_global={"atk:receiver_profile": prof.id,
                                  "atk:tier": "invented",
                                  "atk:method": "training_library",
                                  "atk:dataset": name},
                    description=f"finder scene {i} — synthetic background"
                                + (" with real library signals" if use_library else ""))
                written += [dp, mp]
            tl, why = _D._tile_parts(x, fs, 0.0, prof, p_scene, rng)
            if tl is None:
                notes.append(f"scene {i}: {why}")
                continue
            floor_words = why
            boxes, box_cls = _D._boxes(tl, anns, fs, cls_list)
            tp = d / sp / f"tiles_{i:03d}.npz"
            _D._save_npz(tp, {
                "spec": np.stack([t.spec for t in tl]).astype(np.float16),
                "boxes": boxes, "box_class": box_cls,
                "tile_t0": np.array([t.t0 for t in tl], np.float64),
                "tile_f0": np.array([t.f0 for t in tl], np.float64),
                "row_period": np.float64(tl[0].row_period),
                "bin_hz": np.float64(tl[0].bin_hz)})
            written.append(tp)
            if progress is not None:
                progress(f"[{int(90 * (i + 1) / ns)}%] finder scene {i + 1}/{ns}: "
                         f"{len(anns)} boxes")
    except BaseException:
        _D._abandon(d)
        raise
    m = _D._common_manifest(prof, name, "wideband", "the training library + "
                            "native scenes", "training_library")
    m.update({
        "classes": cls_list, "splits": counts,
        "families": list(_classes.FAMILIES),
        "scene": {"seconds": secs, "samples": int(round(secs * fs)),
                  "datatype": dt, "kept": bool(keep_scenes)},
        "tiles": f"computed: dsp.stft.tiles at the profile's STFT geometry; floor {floor_words}",
        "library": {"used": bool(use_library), "boxes_from_library": n_lib,
                    "boxes_synthetic": n_syn,
                    "types": sorted({st.name for st, _e, _s in pool}),
                    "heldout_pool": pools["held_how"] if use_library else "",
                    "rule": "train scenes hold training signals; validation "
                            "and test scenes hold the kept-aside real "
                            "signals when there are any"},
        "params": {"n_scenes": ns, "seed": int(seed), "use_library": bool(use_library),
                   "synthetic": bool(synthetic), "per_scene": list(per_scene),
                   "scene_seconds": secs},
        "notes": notes or ["synthetic background; library signals placed at random "
                           "frequencies and times, their SNR capped 6 dB under "
                           "their own"],
    })
    return _D._finish(rf, d, m, written, "dataset-scene", overwrite, prof)
