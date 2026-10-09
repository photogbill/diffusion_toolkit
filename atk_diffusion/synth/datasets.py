# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The dataset builder: training data at a receiver profile's exact rate, in
the layout ARCHITECTURE §5 fixes, and the cabled-set ingester (plan §3.2, §3.3,
§3.5, §4.A; DETECTION_DESIGN §6, §10).

    build_narrowband(rf, profile, name, classes, n_per_class, snr_range,
                     canonical_class, generator="native"|"torchsig", …)
    build_wideband(rf, profile, name, n_scenes, env=None, generator=…, …)
    ingest_cabled(rf, profile, capture_paths, name, …)
    load_manifest(path)   iter_shards(path, split)   verify(path) -> (ok, problems)

Everything lands in `<rf_data>\\<profile>\\datasets\\<name>\\`:

    manifest.json    name, profile, sample_rate, kind, generator, canonical,
                     stft, fam, classes, families, splits, label_sources,
                     environment, resampled (always false), created, params,
                     files {relpath: sha256} — and the tier, the receiver it
                     was made through, what was and was not computed (SCF,
                     tiles) and why
    train\\ val\\ test\\
      narrowband: shard_NNN.npz   iq (N, L) complex64 at the canonical rate,
                  label (N,) int32, family (N,) int32, snr_db, symbol_rate_hz,
                  carrier_offset_hz, bandwidth_hz (N,) float32,
                  [scf (N, H, W) float16]
      wideband:   scene_NNN.sigmf-data/-meta (labels as SigMF annotations)
                  + tiles_NNN.npz  spec (N, rows, bins) float16 dB above the
                  floor, boxes (M, 6) float32 [tile, row0, bin0, row1, bin1,
                  family], box_class (M,) int32

THE SAMPLE-RATE LAW, ENFORCED (Bill: *"OmniSIG only works if the sample rate
is identical in training … as in the field"*). A dataset belongs to ONE
profile: it is generated at that profile's exact rate, through that profile's
measured receiver (or a stated textbook one), quantised to the profile's own
datatype, and a narrowband example reaches its canonical rate the way the
field cut does — shift, low-pass to the box with a 1.25 guard, INTEGER
decimation (`dsp.resample.decimate`, the factor from `profiles.
canonical_rates`). Nothing is fractionally resampled. A dataset folder that
already belongs to another profile, a capture from another receiver or rate,
or a capture that was resampled (`atk:resampled_from`) is refused in words.

THE NARROWBAND CUT IMITATES THE FIELD. The signal is generated somewhere in
the receiver's band (where the DC spike, spurs and floor roll-off differ),
and the "box" it is cut from is its 99 % band (never narrower than two STFT
bins — the smallest box a proposer draws) with its centre missed by a little
(σ = 5 % of the box), so the classifier's regression targets — symbol rate
and the carrier's offset from the cut centre — are what a field cut presents.
The noise class is cut through boxes as wide as the other classes' so the
low-pass width gives nothing away.

DONE BY OTHERS, USED LAZILY. The SCF image comes from `cyclo.scf.scf_image`
and the tiles from `dsp.stft.tiles` against a floor measured by
`dsp.floor.NoiseFloor` from a terminated stand-in of the same synthetic
receiver. If either module cannot be imported the dataset is still built and
the manifest says, in words, what was not computed and why.

Every file written is hashed into the manifest AND recorded in the rf_data
write log. Re-building over an existing dataset is refused unless asked for
(`overwrite=True`): a dataset that took a night to build is not deleted by
accident (plan §3.2).

BUILT BESIDE, MOVED INTO PLACE. A build writes into `<name>.partial\\` next
to the dataset's folder and moves it into place only when every file is
written. So an overwrite that fails half-way (a generator error, a crash, a
Ctrl-C at hour three) leaves the old dataset whole, and an interrupted build
never leaves a half-filled folder that blocks the name: the next build of
that name clears the `.partial` folder (only files this builder writes) and
says so. A dataset name is letters, digits, '-' and '_' — Windows drops a
trailing dot or space and refuses ':', and a folder whose name is not the
dataset's name would be the first lie in a run.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from atk_diffusion import profiles as _profiles
from atk_diffusion import provenance
from atk_diffusion import sigmf as _sigmf
from atk_diffusion.detect import classes as _classes
from atk_diffusion.dsp import impair as _impair
from atk_diffusion.dsp import resample as _resample
from atk_diffusion.paths import sha256_file
from atk_diffusion.synth import labels as _labels
from atk_diffusion.synth import native as _native

MANIFEST = "manifest.json"
SPLITS = ("train", "val", "test")
DEFAULT_WINDOW = 4096
SHARD_SIZE = 1024
CUT_MARGIN = 32                 # canonical samples each side for the filter
BOX_CENTRE_ERROR = 0.05         # σ of the box-centre miss, × box width
SCF_SHAPE = (64, 128)
PARTIAL = ".partial"            # a build in progress: <name>.partial\
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$")
#: the only files a build writes (so only these are cleared from .partial)
_OURS = re.compile(r"^(shard|scene|tiles)_\d{3,}(\.npz|\.sigmf-data|\.sigmf-meta)"
                   r"(\.tmp)?$")


class DatasetRefusal(ValueError):
    """The dataset will not be built or read; the message says why."""


# ---------------------------------------------------------------------------
# shared plumbing
# ---------------------------------------------------------------------------
def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _profile(rf, profile) -> _profiles.ReceiverProfile:
    """A ReceiverProfile, or an id loaded from rf_data (so its measured
    impairments, if any, are used)."""
    if isinstance(profile, _profiles.ReceiverProfile):
        return profile
    if isinstance(profile, str):
        return _profiles.load_profile(rf, profile.strip().lower())
    raise DatasetRefusal("a dataset is built for a receiver profile (a "
                         "ReceiverProfile or its id, e.g. 'rtlsdr_2400000_cu8')")


def _split_counts(n: int, splits) -> dict:
    fr = [float(v) for v in splits]
    if len(fr) != 3 or any(v < 0 for v in fr) or abs(sum(fr) - 1.0) > 1e-6:
        raise DatasetRefusal("splits are three fractions (train, val, test) "
                             f"that add up to 1; got {tuple(splits)}")
    raw = [n * v for v in fr]
    cnt = [int(math.floor(v)) for v in raw]
    for i in sorted(range(3), key=lambda i: raw[i] - cnt[i], reverse=True):
        if sum(cnt) >= n:
            break
        cnt[i] += 1
    return dict(zip(SPLITS, cnt))


def _target(rf, prof, name: str, overwrite: bool) -> tuple[Path, dict | None]:
    """The dataset's final folder and its current manifest (None when it has
    none). Checks only — nothing is created or deleted — so a build can be
    refused before any work is done, and checked again before the move."""
    if not _NAME.match(str(name)):
        raise DatasetRefusal(f"{name!r} is not a plain dataset name (letters, "
                             "digits, '-', '_'; starting with a letter or a "
                             "digit; at most 100 characters)")
    d = Path(rf.datasets(prof.id, name))
    mf = d / MANIFEST
    if mf.exists():
        old = json.loads(mf.read_text("utf-8"))
        if str(old.get("profile", "")).lower() != prof.id:
            raise _profiles.ProfileMismatch(
                f"the dataset folder {name} already holds data of "
                f"{old.get('profile')!r}; this build is for "
                f"{_profiles.describe(prof.id)}. Profiles never mix — choose "
                "another name.")
        if not overwrite:
            raise DatasetRefusal(
                f"dataset {name} already exists for {prof.id} "
                f"({old.get('created', '?')}); choose a new name, or pass "
                "overwrite=True to rebuild it (a night's work is not deleted "
                "by accident)")
        listed = set(old.get("files", {})) | {MANIFEST}
        for f in d.rglob("*"):
            if f.is_file() and f.relative_to(d).as_posix() not in listed:
                raise DatasetRefusal(f"{name} holds {f.relative_to(d)}, which "
                                     "is not in its manifest; move it out "
                                     "before rebuilding")
        return d, old
    if d.exists() and any(f.is_file() for f in d.rglob("*")):
        raise DatasetRefusal(f"{d} holds files but no {MANIFEST}, so it is not "
                             "a dataset built here; choose another name")
    return d, None


def _clear_partial(work: Path) -> int:
    """Remove an interrupted build's `<name>.partial` folder — only when
    every file in it is one this builder writes. -> files removed."""
    files = [f for f in work.rglob("*") if f.is_file()]
    for f in files:
        if not _OURS.match(f.name):
            raise DatasetRefusal(f"{work} holds {f.relative_to(work)}, which "
                                 "this builder did not write; move it out "
                                 "first")
    for f in files:
        f.unlink()
    for sub in sorted((p for p in work.rglob("*") if p.is_dir()),
                      key=lambda p: len(p.parts), reverse=True):
        sub.rmdir()
    work.rmdir()
    return len(files)


def _prepare(rf, prof, name: str, overwrite: bool, notes: list | None = None
             ) -> Path:
    """The folder a build writes into: `<name>.partial` beside the dataset's
    own, cleared of an interrupted build first (said in `notes`)."""
    d, _old = _target(rf, prof, name, overwrite)
    work = d.with_name(d.name + PARTIAL)
    if work.exists():
        k = _clear_partial(work)
        if notes is not None:
            notes.append(f"an interrupted build of {name} ({k} file(s) in "
                         f"{work.name}) was cleared before this one")
    for s in SPLITS:
        (work / s).mkdir(parents=True, exist_ok=True)
    return work


def _abandon(work: Path) -> None:
    """A build that failed: its partial folder goes (best effort); the
    dataset it was to replace was never touched."""
    try:
        if work.exists():
            _clear_partial(work)
    except (OSError, DatasetRefusal):
        pass


def _save_npz(path: Path, arrays: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    tmp.replace(path)


def _finish(rf, work: Path, manifest: dict, written: list[Path], kind: str,
            overwrite: bool, prof) -> dict:
    """Move the finished build into place (replacing the old dataset only
    now, when overwrite allows it), hash every file into the manifest,
    write it, and record everything in the write log."""
    d, old = _target(rf, prof, manifest["name"], overwrite)
    if old is not None:
        for rel in list(old.get("files", {})) + [MANIFEST]:
            p = d / rel
            if p.exists():
                p.unlink()
    if d.exists():
        for sub in sorted((p for p in d.rglob("*") if p.is_dir()),
                          key=lambda p: len(p.parts), reverse=True):
            sub.rmdir()
        d.rmdir()
    work.rename(d)
    files = {}
    for p in written:
        rel = p.relative_to(work).as_posix()
        q = d / rel
        files[rel] = sha256_file(q)
        rf.record(q, kind, f"dataset {manifest['name']} ({manifest['profile']})")
    manifest["files"] = dict(sorted(files.items()))
    mp = d / MANIFEST
    tmp = mp.with_name(MANIFEST + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, default=_sigmf._json_default),
                   encoding="utf-8")
    tmp.replace(mp)
    rf.record(mp, "dataset-manifest", f"dataset {manifest['name']}")
    out = dict(manifest)
    out["path"] = str(d)
    return out


def _scf_fn():
    """(callable or None, words)."""
    try:
        from atk_diffusion.cyclo.scf import scf_image
    except Exception as e:                              # noqa: BLE001
        return None, (f"not computed: atk_diffusion.cyclo.scf is not available "
                      f"({type(e).__name__}: {e})")
    return scf_image, ""


def _common_manifest(prof, name, kind, generator_words, method) -> dict:
    from atk_diffusion import __version__
    measured = _impair.is_measured(prof.impairments)
    return {"name": name, "profile": prof.id,
            "sample_rate": float(prof.sample_rate), "kind": kind,
            "generator": generator_words,
            "stft": asdict(prof.stft), "fam": asdict(prof.fam),
            "families": list(_classes.FAMILIES),
            "label_sources": ["synthetic"], "environment": "",
            "resampled": False, "created": _now(),
            "tier": provenance.tier_for(method), "method": method,
            "receiver": _impair.describe(prof.impairments),
            "receiver_impairments_applied": bool(measured),
            "quantised_to": prof.datatype,
            "snr_definition": _native.SNR_DEFINITION,
            "code_version": __version__,
            "provenance": provenance.stamp("atk_diffusion.synth.datasets",
                                           dataset=name, profile=prof.id)}


# ---------------------------------------------------------------------------
# narrowband
# ---------------------------------------------------------------------------
def _canonical(prof, canonical_class: str) -> _profiles.CanonicalRate:
    rates = {c.cls: c for c in prof.canonical_rates()}
    if canonical_class not in _profiles.BANDWIDTH_CLASSES:
        raise DatasetRefusal(f"canonical_class is one of "
                             f"{', '.join(_profiles.CLASS_ORDER)}; got "
                             f"{canonical_class!r}")
    if canonical_class not in rates:
        raise DatasetRefusal(
            f"{_profiles.describe(prof.id)} has no {canonical_class} canonical "
            f"rate (its rate is below the class's minimum); it has "
            f"{', '.join(rates)}")
    return rates[canonical_class]


def _gen_words(generator: str) -> str:
    if generator == "native":
        return _native.GENERATOR
    if generator == "torchsig":
        from atk_diffusion.synth import torchsig_backend as _ts
        ok, why = _ts.available()
        if not ok:
            raise DatasetRefusal(f"generator='torchsig' is not usable here: {why}")
        return _ts.GENERATOR
    raise DatasetRefusal("generator is 'native' or 'torchsig'")


def _check_classes(prof, classes, can, generator: str) -> None:
    if not classes:
        raise DatasetRefusal("a dataset needs at least one class")
    fs = float(prof.sample_rate)
    problems = []
    for cls in classes:
        c = _classes.get(cls)
        if c is None:
            problems.append(f"{cls!r} is not in the class table")
            continue
        bw = c.bandwidth_hz
        if bw > 0.9 * can.rate:
            problems.append(f"{c.label} (≈{_native._hz_words(bw)}) does not fit "
                            f"the {can.cls} canonical rate of "
                            f"{_native._rate_words(can.rate)}")
            continue
        if generator == "native":
            ok, why = _native.can_generate(cls, fs)
        else:
            from atk_diffusion.synth import torchsig_backend as _ts
            ok, why = _ts.can_generate(cls, prof)
        if not ok:
            problems.append(f"{c.label}: {why}")
    if problems:
        raise DatasetRefusal("not built — " + "; ".join(problems))


def _example(prof, cls: str, ci: int, j: int, seed: int, can, L: int,
             snr_range, generator: str, impairment_level: int,
             bw_pool: list, scf=None) -> dict:
    """One narrowband example (deterministic in (seed, ci, j))."""
    rng = np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=(ci, j)))
    fs = float(prof.sample_rate)
    d, rate = int(can.decimation), float(can.rate)
    n = (L + 2 * CUT_MARGIN) * d
    w0, w1 = CUT_MARGIN * d, (CUT_MARGIN + L) * d
    c = _classes.get(cls)
    measured = _impair.is_measured(prof.impairments)
    floor_dbfs = 10.0 * math.log10(_impair.noise_power(
        prof.impairments if measured else None))
    # reference classes have no nominal width: draw one that fills a fair
    # part of the canonical band — where the generator takes a width at all
    # (the AM reference is a voice-band DSB; its width is its audio's)
    takes_bw = (generator != "native"
                or "bandwidth_hz" in _native._ACCEPT.get(c.native or "", set()))
    for _attempt in range(6):
        snr = float(rng.uniform(*snr_range))
        params = None
        if c.bandwidth_hz <= 0 and not c.negative:
            target = math.exp(rng.uniform(math.log(0.1 * rate), math.log(0.6 * rate)))
            if takes_bw:
                params = {"bandwidth_hz": target}
        width = max(c.bandwidth_hz, (params or {}).get("bandwidth_hz", 0.0), 0.0)
        span = 0.45 * fs - 0.5 * max(width, rate)
        f_c = 0.0 if (cls == "dc_spike" or span <= 0) else float(rng.uniform(-span, span))
        if generator == "native":
            x, lab = _native.generate(cls, fs, n, snr, rng, carrier_offset_hz=f_c,
                                      noise_dbfs=floor_dbfs, params=params)
        else:
            from atk_diffusion.synth import torchsig_backend as _ts
            x, lab = _ts.generate_narrowband(prof, cls, n, snr, rng,
                                             carrier_offset_hz=f_c, params=params,
                                             impairment_level=impairment_level,
                                             noise_dbfs=floor_dbfs)
        if cls == "noise" or any(s < w1 and s + k > w0 for s, k in lab["bursts"]):
            break                                    # the window holds it
    if measured:
        x = _impair.apply_impairments(x, fs, prof.impairments, rng,
                                      adc_bits=prof.adc_bits or None,
                                      datatype=prof.datatype)
    else:
        x = _impair.quantise(x, prof.datatype, prof.adc_bits or None)
    rbw = prof.stft.rbw_hz(fs)
    if cls == "noise":
        box_bw = float(rng.choice(bw_pool)) if bw_pool else 0.25 * rate
        centre = f_c
    else:
        box_bw = max(float(lab["bandwidth_hz"]), 2.0 * rbw)
        centre = 0.5 * (lab["f_lo_hz"] + lab["f_hi_hz"])
    err = float(np.clip(rng.normal(0.0, BOX_CENTRE_ERROR * box_bw),
                        -3 * BOX_CENTRE_ERROR * box_bw, 3 * BOX_CENTRE_ERROR * box_bw))
    f_cut = centre + err
    cutoff = min(0.5 * box_bw * 1.25, 0.49 * rate)
    y, fs_c = _resample.decimate(_resample.shift(x, f_cut, fs), d, fs,
                                 cutoff_hz=cutoff)
    win = np.ascontiguousarray(y[CUT_MARGIN:CUT_MARGIN + L], dtype=np.complex64)
    if win.size < L:
        win = np.concatenate([win, np.zeros(L - win.size, np.complex64)])
    out = {"iq": win, "family": _labels.family_index(c.family),
           "snr_db": float("nan") if cls == "noise" else snr,
           "symbol_rate_hz": float(lab.get("symbol_rate_hz", 0.0) or 0.0),
           "carrier_offset_hz": 0.0 if cls == "noise"
           else float(lab["carrier_offset_hz"]) - f_cut,
           "bandwidth_hz": float(lab.get("bandwidth_hz", 0.0) or 0.0)}
    if scf is not None:
        img, _, _ = scf(win, fs_c, prof.fam, out_shape=SCF_SHAPE)
        out["scf"] = img.astype(np.float16)
    return out


def _flush(buf: list, d: Path, split: str, idx: int, with_scf: bool) -> Path:
    arr = {"iq": np.stack([b["iq"] for b in buf]).astype(np.complex64),
           "label": np.array([b["label"] for b in buf], np.int32),
           "family": np.array([b["family"] for b in buf], np.int32)}
    for k in ("snr_db", "symbol_rate_hz", "carrier_offset_hz", "bandwidth_hz"):
        arr[k] = np.array([b[k] for b in buf], np.float32)
    if with_scf:
        arr["scf"] = np.stack([b["scf"] for b in buf]).astype(np.float16)
    p = d / split / f"shard_{idx:03d}.npz"
    _save_npz(p, arr)
    return p


def build_narrowband(rf, profile, name: str, classes: list, n_per_class: int,
                     snr_range, canonical_class: str, generator: str = "native",
                     seed: int = 0, window: int | None = None,
                     splits=(0.8, 0.1, 0.1), compute_scf: bool = True,
                     progress=None, *, impairment_level: int = 0,
                     shard_size: int = SHARD_SIZE, overwrite: bool = False
                     ) -> dict:
    """A narrowband classification dataset (ARCHITECTURE §5): `n_per_class`
    examples of every class, each generated at the profile's exact rate and
    cut to the profile's canonical rate for `canonical_class` by integer
    decimation, `window` samples long (default 4096). Returns the manifest
    (with 'path'). `impairment_level` is TorchSig's (generator='torchsig')."""
    prof = _profile(rf, profile)
    can = _canonical(prof, canonical_class)
    L = int(window or DEFAULT_WINDOW)
    if L < 64:
        raise DatasetRefusal("a window shorter than 64 samples holds no signal")
    n_per = int(n_per_class)
    if n_per < 1:
        raise DatasetRefusal("n_per_class must be at least 1")
    lo, hi = (float(v) for v in snr_range)
    if hi < lo:
        raise DatasetRefusal("snr_range is (low, high)")
    gw = _gen_words(generator)
    classes = list(classes)
    if len(set(classes)) != len(classes):
        raise DatasetRefusal("a class is listed twice")
    _check_classes(prof, classes, can, generator)
    per_split = _split_counts(n_per, splits)
    notes: list[str] = []
    d = _prepare(rf, prof, name, overwrite, notes)
    scf, scf_why = (None, "not computed: compute_scf=False")
    if compute_scf:
        scf, scf_why = _scf_fn()
    bw_pool = [c.bandwidth_hz for c in map(_classes.get, classes)
               if c.bandwidth_hz > 0] or [0.25 * can.rate]
    rng = np.random.default_rng(int(seed))
    assign: dict = {}
    for ci, cls in enumerate(classes):
        perm = np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=(ci,))
                                     ).permutation(n_per)
        rank = np.empty(n_per, np.int64)
        rank[perm] = np.arange(n_per)               # example j's place in perm
        a = per_split["train"]
        b = a + per_split["val"]
        for j in range(n_per):
            r = int(rank[j])
            assign[(ci, j)] = "train" if r < a else ("val" if r < b else "test")
    order = [(ci, j) for ci in range(len(classes)) for j in range(n_per)]
    order = [order[i] for i in rng.permutation(len(order))]
    bufs = {s: [] for s in SPLITS}
    shard_idx = {s: 0 for s in SPLITS}
    written: list[Path] = []
    total = len(order)
    t_start = time.time()
    try:
        for k, (ci, j) in enumerate(order, 1):
            ex = _example(prof, classes[ci], ci, j, seed, can, L, (lo, hi), generator,
                          impairment_level, bw_pool, scf)
            ex["label"] = ci
            sp = assign[(ci, j)]
            bufs[sp].append(ex)
            if len(bufs[sp]) >= int(shard_size):
                written.append(_flush(bufs[sp], d, sp, shard_idx[sp], scf is not None))
                shard_idx[sp] += 1
                bufs[sp] = []
            if progress is not None and (k == total or k % max(1, total // 20) == 0):
                progress(f"{name}: {k}/{total} examples ({100 * k // total} %), "
                         f"{k / max(1e-9, time.time() - t_start):.1f} examples/s")
        for sp in SPLITS:
            if bufs[sp]:
                written.append(_flush(bufs[sp], d, sp, shard_idx[sp], scf is not None))
    except BaseException:
        _abandon(d)
        raise
    method = "synthetic_native" if generator == "native" else "synthetic_torchsig"
    m = _common_manifest(prof, name, "narrowband", gw, method)
    m.update({
        "canonical": {"class": can.cls, "rate": float(can.rate),
                      "decimation": int(can.decimation), "limited": bool(can.limited)},
        "window": {"samples": L, "seconds": L / float(can.rate)},
        "classes": classes,
        "splits": {s: per_split[s] * len(classes) for s in SPLITS},
        "scf": (f"computed: cyclo.scf.scf_image {SCF_SHAPE[0]}x{SCF_SHAPE[1]}, "
                "the profile's FAM geometry") if scf is not None else scf_why,
        "params": {"n_per_class": n_per, "snr_range": [lo, hi], "seed": int(seed),
                   "canonical_class": canonical_class, "generator": generator,
                   "impairment_level": int(impairment_level),
                   "window": L, "shard_size": int(shard_size),
                   "cut": {"margin_samples": CUT_MARGIN, "lowpass_guard": 1.25,
                           "box_centre_error_sigma": BOX_CENTRE_ERROR,
                           "min_box_stft_bins": 2},
                   "build_seconds": round(time.time() - t_start, 2)},
        "arrays": {"iq": "(N, L) complex64 at the canonical rate",
                   "label": "(N,) int32 index into classes",
                   "family": "(N,) int32 index into families",
                   "snr_db": "(N,) float32 by snr_definition; NaN = no signal",
                   "symbol_rate_hz": "(N,) float32; 0 = none or not provided",
                   "carrier_offset_hz": "(N,) float32, the carrier's offset from "
                                        "the cut centre",
                   "bandwidth_hz": "(N,) float32, 99 % occupied band; 0 = none",
                   "scf": f"(N, {SCF_SHAPE[0]}, {SCF_SHAPE[1]}) float16, 0..1"},
        "notes": [f"generated at {prof.sample_rate} S/s and cut to "
                  f"{can.rate:g} S/s by integer decimation ÷{can.decimation}"]
        + notes,
    })
    return _finish(rf, d, m, written, "dataset-shard", overwrite, prof)


# ---------------------------------------------------------------------------
# wideband
# ---------------------------------------------------------------------------
def _tile_parts(x, fs, center_hz, prof, p_floor_scene, rng):
    """(tiles npz arrays, None) or (None, why-not)."""
    try:
        from atk_diffusion.dsp import stft as _stft
    except Exception as e:                              # noqa: BLE001
        return None, (f"not computed: atk_diffusion.dsp.stft is not available "
                      f"({type(e).__name__}: {e})")
    floor = None
    floor_words = "estimated from each scene (dsp.floor unavailable)"
    try:
        from atk_diffusion.dsp.floor import NoiseFloor
        # a terminated stand-in of the same synthetic receiver, at the
        # scene's own floor level (after any AGC)
        n_ref = int(min(max(x.size, 32 * prof.stft.fft_size), 2 * fs))
        ref = _native.noise(n_ref, p_floor_scene, rng)
        if _impair.is_measured(prof.impairments):
            ref = _impair.apply_impairments(ref, fs, prof.impairments, rng,
                                            adc_bits=prof.adc_bits or None,
                                            datatype=prof.datatype)
        else:
            ref = _impair.quantise(ref, prof.datatype, prof.adc_bits or None)
        floor = NoiseFloor.from_terminated(ref, fs, prof.stft)
        floor_words = "measured (dsp.floor) from a terminated stand-in of the scene's receiver"
    except Exception as e:                              # noqa: BLE001
        floor = None
        floor_words = f"estimated from each scene ({type(e).__name__}: {e})"
    try:
        tl = list(_stft.tiles(x, fs, center_hz, prof.stft, floor=floor,
                              profile=prof.id))
    except Exception as e:                              # noqa: BLE001
        return None, (f"not computed: dsp.stft.tiles failed "
                      f"({type(e).__name__}: {e})")
    if not tl:
        return None, "not computed: the scene is shorter than one tile frame"
    return tl, floor_words


def _boxes(tiles, anns, fs, classes: list) -> tuple[np.ndarray, np.ndarray]:
    rows, cls_idx = [], []
    for ti, t in enumerate(tiles):
        for a in anns:
            if a.freq_lower_edge is None or a.freq_upper_edge is None:
                continue
            t0 = a.sample_start / fs
            t1 = (a.sample_start + a.sample_count) / fs
            if t1 <= t.t0 or t0 >= t.t1:
                continue
            r0, b0, r1, b1 = t.tf_to_pixels(t0, t1, a.freq_lower_edge,
                                            a.freq_upper_edge, clip=True)
            if r1 - r0 <= 0 or b1 - b0 <= 0:
                continue
            fam = (a.extra or {}).get("atk:family") or _classes.get(a.label).family
            rows.append([ti, r0, b0, r1, b1, _labels.family_index(fam)])
            cls_idx.append(classes.index(a.label) if a.label in classes else -1)
    return (np.array(rows, np.float32).reshape(-1, 6),
            np.array(cls_idx, np.int32))


def build_wideband(rf, profile, name: str, n_scenes: int, env=None,
                   generator: str = "native", seed: int = 0,
                   scene_seconds: float | None = None, *,
                   splits=(0.8, 0.1, 0.1), classes: list | None = None,
                   compute_tiles: bool = True, progress=None,
                   overwrite: bool = False) -> dict:
    """A wideband detection dataset (ARCHITECTURE §5): `n_scenes` scenes of
    `scene_seconds` (default the profile's tile length) at the profile's
    exact rate — composed for an environment profile (`env`: an
    EnvironmentProfile or a region id) or, with env=None, generic — written
    as SigMF pairs in the profile's own datatype with their labels as
    annotations, plus their tiles and pixel boxes. Returns the manifest."""
    from atk_diffusion.synth import environments as _env
    from atk_diffusion.synth import scene as _scene
    prof = _profile(rf, profile)
    gw = _gen_words(generator)
    ns = int(n_scenes)
    if ns < 1:
        raise DatasetRefusal("n_scenes must be at least 1")
    fs = float(prof.sample_rate)
    secs = float(scene_seconds if scene_seconds is not None else prof.stft.tile_seconds)
    if secs * fs < prof.stft.fft_size:
        raise DatasetRefusal(f"a {secs:g} s scene is shorter than one "
                             f"{prof.stft.fft_size}-point frame")
    envp = _env.resolve(rf, env) if env is not None else None
    if classes is not None:
        bad = [c for c in classes if _classes.get(c) is None]
        if bad:
            raise DatasetRefusal(f"not in the class table: {', '.join(bad)}")
    cls_list = list(classes) if classes else [c.name for c in _classes.CLASSES]
    counts = _split_counts(ns, splits)
    notes: list[str] = []
    d = _prepare(rf, prof, name, overwrite, notes)
    order = np.random.default_rng(int(seed)).permutation(ns)
    split_of = {}
    a, b = counts["train"], counts["train"] + counts["val"]
    for r, i in enumerate(order):
        split_of[int(i)] = "train" if r < a else ("val" if r < b else "test")
    dt = prof.datatype
    if dt == "ci16" and prof.adc_bits and prof.adc_bits <= 12:
        dt = "ci16q11"
    written: list[Path] = []
    tiles_missing: list[tuple[int, str]] = []
    floor_words = ""
    t_start = time.time()
    n_ann = 0
    try:
        for i in range(ns):
            rng = np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=(7, i)))
            if envp is not None:
                centre = _scene.choose_center(envp, prof, rng, classes)
                r = _scene.compose_scene_detailed(envp, prof, centre, secs, rng, generator)
            else:
                centre = 0.0
                r = _scene.compose_generic(prof, centre, secs, rng, generator,
                                           classes=classes)
            if not _impair.is_measured(prof.impairments):
                r.x = _impair.quantise(r.x, prof.datatype, prof.adc_bits or None)
            sp = split_of[i]
            base = d / sp / f"scene_{i:03d}"
            g_extra = {"atk:receiver_profile": prof.id,
                       "atk:tier": provenance.tier_for(
                           "synthetic_native" if generator == "native"
                           else "synthetic_torchsig"),
                       "atk:method": "synthetic_native" if generator == "native"
                       else "synthetic_torchsig",
                       "atk:generator": gw,
                       "atk:environment": envp.region if envp else "",
                       "atk:dataset": name,
                       "atk:scene": {k: r.info[k] for k in
                                     ("noise_dbfs", "agc_gain_db", "receiver",
                                      "receiver_impairments_applied", "terrain",
                                      "caveat")},
                       "atk:snr_definition": _native.SNR_DEFINITION}
            dp, mp = _sigmf.write_pair(base, r.x, fs, centre, datatype=dt,
                                       annotations=r.annotations, extra_global=g_extra,
                                       hw=f"synthetic: ATK Diffusion Toolkit ({gw})",
                                       description=f"synthetic scene {i} of dataset "
                                                   f"{name} — never received")
            written += [dp, mp]
            n_ann += len(r.annotations)
            if compute_tiles:
                p_scene = 10 ** (r.info["noise_dbfs"] / 10) \
                    * 10 ** (r.info["agc_gain_db"] / 10)
                # the tiles are made from the record as written (its datatype)
                xr = _sigmf.load(base)
                tl, why = _tile_parts(xr, fs, centre, prof, p_scene, rng)
                if tl is None:
                    tiles_missing.append((i, why))
                else:
                    floor_words = why
                    boxes, box_cls = _boxes(tl, r.annotations, fs, cls_list)
                    tp = d / sp / f"tiles_{i:03d}.npz"
                    _save_npz(tp, {
                        "spec": np.stack([t.spec for t in tl]).astype(np.float16),
                        "boxes": boxes, "box_class": box_cls,
                        "tile_t0": np.array([t.t0 for t in tl], np.float64),
                        "tile_f0": np.array([t.f0 for t in tl], np.float64),
                        "row_period": np.float64(tl[0].row_period),
                        "bin_hz": np.float64(tl[0].bin_hz)})
                    written.append(tp)
            if progress is not None:
                progress(f"{name}: scene {i + 1}/{ns} ({len(r.annotations)} labels)")
    except BaseException:
        _abandon(d)
        raise
    if not compute_tiles:
        tiles_note = "not computed: compute_tiles=False"
    elif len(tiles_missing) == ns:
        tiles_note = tiles_missing[0][1]
    else:
        tiles_note = (f"computed: dsp.stft.tiles at the profile's STFT geometry; "
                      f"floor {floor_words}")
        if tiles_missing:
            tiles_note += (f" — but NOT for {len(tiles_missing)} of {ns} scenes "
                           f"(scene {tiles_missing[0][0]}: {tiles_missing[0][1]})")
    method = "synthetic_native" if generator == "native" else "synthetic_torchsig"
    m = _common_manifest(prof, name, "wideband", gw, method)
    m.update({
        "classes": cls_list, "splits": counts,
        "environment": envp.region if envp else "",
        "scene": {"seconds": secs, "samples": int(round(secs * fs)),
                  "datatype": dt, "annotations": n_ann},
        "tiles": tiles_note,
        "params": {"n_scenes": ns, "seed": int(seed), "generator": generator,
                   "scene_seconds": secs, "classes": classes,
                   "environment": envp.region if envp else "",
                   "build_seconds": round(time.time() - t_start, 2)},
        "arrays": {"spec": "(N, rows, bins) float16 dB above the floor",
                   "boxes": "(M, 6) float32 [tile, row0, bin0, row1, bin1, family]",
                   "box_class": "(M,) int32 index into classes (-1: not listed)"},
        "notes": ([envp.caveat] if envp else
                  ["no environment profile: random classes, offsets and times"])
        + notes,
    })
    return _finish(rf, d, m, written, "dataset-scene", overwrite, prof)


# ---------------------------------------------------------------------------
# cabled
# ---------------------------------------------------------------------------
def ingest_cabled(rf, profile, capture_paths, name: str, *,
                  canonical_class: str | None = None, window: int | None = None,
                  splits=(0.0, 0.0, 1.0), compute_scf: bool = True,
                  seed: int = 0, progress=None, overwrite: bool = False) -> dict:
    """A narrowband dataset from cabled captures (plan §3.5): every SigMF
    annotation with atk:source = "cabled" — the ground truth of what the
    transmitter played — is cut from the RECORD to the canonical rate the
    way the field cut does. All examples go to 'test' by default (the
    domain-gap set, plan §7). The captures stay where they are: the manifest
    lists them with their hashes ('sources') and verify() checks them."""
    prof = _profile(rf, profile)
    fs = float(prof.sample_rate)
    paths = [_sigmf.base_of(p) for p in capture_paths]
    if not paths:
        raise DatasetRefusal("no captures given")
    _split_counts(1, splits)                    # refuse bad splits up front
    _target(rf, prof, name, overwrite)          # ... and a name in use
    notes, sources, items = [], [], []
    for p in paths:
        meta = _sigmf.read_meta(p)
        g = meta.get("global", {})
        cap_pid = _profiles.profile_from_meta(meta)
        if cap_pid != prof.id:
            raise _profiles.ProfileMismatch(
                f"dataset {name} is a dataset of {_profiles.describe(prof.id)}; "
                f"{p.name} was recorded on {_profiles.describe(cap_pid)}. "
                "Profiles never mix: record the cabled set on this receiver at "
                "this rate.")
        if g.get("atk:resampled_from"):
            raise DatasetRefusal(
                f"{p.name} was resampled from {g['atk:resampled_from']}; a "
                "dataset is built only from captures recorded at its profile's "
                "own rate (plan §3.3). Record the cabled set at this rate.")
        tier = str(g.get("atk:tier", "") or "").strip().lower()
        if g.get("atk:translated_from") or tier not in ("", "record"):
            raise DatasetRefusal(
                f"{p.name} is not a recording (atk:tier {tier or 'invented'}"
                + (f", translated from {g['atk:translated_from']}"
                   if g.get("atk:translated_from") else "")
                + "). A cabled set is RECORD tier — what the receiver actually "
                "heard; a translated or cleaned capture is judged like "
                "synthetic data, never as the cabled set (plan §4.A6, §7).")
        dp = _sigmf.data_path(p)
        ok, why = rf.verify(dp)
        if not ok and "changed after it was recorded" in why:
            raise DatasetRefusal(f"{why} A capture that changed is named, not "
                                 "used (plan §3.2).")
        if not ok:
            notes.append(f"{p.name}: {why} — used, but its integrity since "
                         "recording cannot be checked")
        anns = [a for a in _sigmf.annotations(meta) if a.source == "cabled"]
        other = len(_sigmf.annotations(meta)) - len(anns)
        if other:
            notes.append(f"{p.name}: {other} annotation(s) not atk:source=cabled "
                         "were left out")
        centre = _sigmf.center_of(meta)
        total = _sigmf.num_samples(p, meta)
        for a in anns:
            if _classes.get(a.label) is None:
                notes.append(f"{p.name}: label {a.label!r} is not in the class "
                             "table — left out")
                continue
            if a.freq_lower_edge is None or a.freq_upper_edge is None:
                notes.append(f"{p.name}: an annotation of {a.label} has no "
                             "frequency edges — left out")
                continue
            items.append((p, meta, centre, total, a))
        try:
            rel = Path(dp).resolve().relative_to(Path(rf.root).resolve()).as_posix()
        except ValueError:
            rel = Path(dp).as_posix()
        sources.append({"capture": rel, "sha256": sha256_file(dp),
                        "profile": cap_pid, "annotations": len(anns),
                        "loop": {k: g[k] for k in
                                 ("atk:tx_power_dbm", "atk:attenuation_db",
                                  "atk:splitter_loss_db", "atk:cable_loss_db",
                                  "atk:expected_input_dbm") if k in g}})
    if not items:
        raise DatasetRefusal("the captures hold no annotation with "
                             "atk:source = 'cabled' — nothing to learn from")
    if canonical_class is None:
        votes: dict = {}
        for *_x, a in items:
            c = _profiles.canonical_for(fs, a.freq_upper_edge - a.freq_lower_edge).cls
            votes[c] = votes.get(c, 0) + 1
        canonical_class = max(votes, key=votes.get)
    can = _canonical(prof, canonical_class)
    L = int(window or DEFAULT_WINDOW)
    d_ = int(can.decimation)
    keep = []
    for it in items:
        bw = it[4].freq_upper_edge - it[4].freq_lower_edge
        if bw > 0.9 * can.rate:
            notes.append(f"{it[0].name}: a {it[4].label} of "
                         f"{_native._hz_words(bw)} does not fit the "
                         f"{can.cls} rate — left out")
            continue
        keep.append(it)
    if not keep:
        raise DatasetRefusal(f"no cabled annotation fits the {can.cls} canonical "
                             f"rate of {_native._rate_words(can.rate)}")
    scf, scf_why = (None, "not computed: compute_scf=False")
    if compute_scf:
        scf, scf_why = _scf_fn()
    need = (L + 2 * CUT_MARGIN) * d_
    examples = []
    for k, (p, meta, centre, total, a) in enumerate(keep, 1):
        if total < need:
            notes.append(f"{p.name}: a {a.label} is in a capture shorter than "
                         f"one window ({need} samples at {fs:g} S/s for a "
                         f"{L}-sample window) — left out")
            continue
        mid = a.sample_start + a.sample_count // 2
        s0 = int(min(max(0, mid - need // 2), total - need))
        x = _sigmf.load(p, s0, need, meta=meta)
        if x.ndim > 1:
            x = x[0]
        lo = float(a.freq_lower_edge) - centre
        hi = float(a.freq_upper_edge) - centre
        f_cut = 0.5 * (lo + hi)
        cutoff = min(0.5 * (hi - lo) * 1.25, 0.49 * can.rate)
        y, fs_c = _resample.decimate(_resample.shift(x, f_cut, fs), d_, fs,
                                     cutoff_hz=cutoff)
        win = np.ascontiguousarray(y[CUT_MARGIN:CUT_MARGIN + L], np.complex64)
        ex = a.extra or {}
        c = _classes.get(a.label)
        e = {"iq": win, "cls": a.label,
             "family": _labels.family_index(ex.get("atk:family") or c.family),
             "snr_db": float(ex["atk:snr_db"]) if ex.get("atk:snr_db") is not None
             else float("nan"),
             "symbol_rate_hz": float(ex.get("atk:symbol_rate", 0.0) or 0.0),
             "carrier_offset_hz": (float(ex["atk:carrier_offset_hz"]) - f_cut)
             if ex.get("atk:carrier_offset_hz") is not None else 0.0,
             "bandwidth_hz": hi - lo}
        if scf is not None:
            img, _, _ = scf(win, fs_c, prof.fam, out_shape=SCF_SHAPE)
            e["scf"] = img.astype(np.float16)
        examples.append(e)
        if progress is not None:
            progress(f"{name}: cut {k}/{len(keep)} cabled signals")
    if not examples:
        raise DatasetRefusal("no cabled signal could be cut: " + "; ".join(
            n for n in notes if "shorter than one window" in n))
    # the class list is what was actually cut, in the class table's order
    classes = [c.name for c in _classes.CLASSES
               if any(e["cls"] == c.name for e in examples)]
    for e in examples:
        e["label"] = classes.index(e.pop("cls"))
    d = _prepare(rf, prof, name, overwrite, notes)
    by_cls: dict = {}
    for i, e in enumerate(examples):
        by_cls.setdefault(e["label"], []).append(i)
    split_of = {}
    counts = {s: 0 for s in SPLITS}
    for ci, idx in by_cls.items():
        perm = np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=(ci,))
                                     ).permutation(len(idx))
        cnt = _split_counts(len(idx), splits)
        a_, b_ = cnt["train"], cnt["train"] + cnt["val"]
        for r, pi in enumerate(perm):
            s = "train" if r < a_ else ("val" if r < b_ else "test")
            split_of[idx[pi]] = s
            counts[s] += 1
    written = []
    try:
        for s in SPLITS:
            buf = [examples[i] for i in range(len(examples)) if split_of.get(i) == s]
            for si in range(0, len(buf), SHARD_SIZE):
                written.append(_flush(buf[si:si + SHARD_SIZE], d, s, si // SHARD_SIZE,
                                      scf is not None))
    except BaseException:
        _abandon(d)
        raise
    m = _common_manifest(prof, name, "narrowband", "cabled", "cabled_ingest")
    m.update({
        "generator": "cabled", "label_sources": ["cabled"],
        "tier": "record", "method": "cabled_ingest",
        "receiver": "the real receiver (cabled captures are the record)",
        "receiver_impairments_applied": False, "quantised_to": prof.datatype,
        "canonical": {"class": can.cls, "rate": float(can.rate),
                      "decimation": int(can.decimation), "limited": bool(can.limited)},
        "window": {"samples": L, "seconds": L / float(can.rate)},
        "classes": classes, "splits": counts, "sources": sources,
        "scf": (f"computed: cyclo.scf.scf_image {SCF_SHAPE[0]}x{SCF_SHAPE[1]}"
                if scf is not None else scf_why),
        "params": {"canonical_class": can.cls, "window": L, "seed": int(seed),
                   "splits": list(splits)},
        "notes": notes or ["every cabled annotation was used"],
    })
    return _finish(rf, d, m, written, "dataset-shard", overwrite, prof)


provenance.METHOD_TIERS.setdefault("cabled_ingest", "record")


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def _dir(path) -> Path:
    p = Path(path)
    return p.parent if p.name == MANIFEST else p


def load_manifest(path, expect_profile: str | None = None) -> dict:
    """The manifest of a dataset folder (or of its manifest.json). With
    `expect_profile`, a dataset of another profile is refused in words."""
    d = _dir(path)
    mf = d / MANIFEST
    if not mf.exists():
        raise DatasetRefusal(f"{d} has no {MANIFEST} — it is not a dataset "
                             "built by the toolkit")
    m = json.loads(mf.read_text("utf-8"))
    if expect_profile and str(m.get("profile", "")).lower() != str(expect_profile).lower():
        raise _profiles.ProfileMismatch(
            f"dataset {m.get('name')} was built for "
            f"{_profiles.describe(m['profile'])}; this is "
            f"{_profiles.describe(expect_profile)}. Profiles never mix.")
    if m.get("resampled"):
        raise DatasetRefusal(f"dataset {m.get('name')} says it was resampled; "
                             "the toolkit never builds one so — it was edited")
    m["path"] = str(d)
    return m


def iter_shards(path, split: str):
    """Narrowband: each shard of `split` as a dict of arrays. Wideband: each
    scene of `split` as {scene, meta, annotations, tiles (dict or None)}."""
    if split not in SPLITS:
        raise DatasetRefusal(f"split is one of {', '.join(SPLITS)}")
    m = load_manifest(path)
    d = _dir(path)
    files = sorted(m.get("files", {}))
    if m["kind"] == "narrowband":
        for rel in files:
            if rel.startswith(f"{split}/shard_") and rel.endswith(".npz"):
                with np.load(d / rel) as z:
                    yield {k: z[k] for k in z.files}
        return
    for rel in files:
        if rel.startswith(f"{split}/scene_") and rel.endswith(".sigmf-meta"):
            base = d / rel[: -len(".sigmf-meta")]
            meta = _sigmf.read_meta(base)
            tp = d / split / ("tiles_" + base.name.split("_", 1)[1] + ".npz")
            tiles = None
            if tp.exists():
                with np.load(tp) as z:
                    tiles = {k: z[k] for k in z.files}
            yield {"scene": base, "meta": meta,
                   "annotations": _sigmf.annotations(meta), "tiles": tiles}


def verify(path) -> tuple[bool, list[str]]:
    """(ok, problems): every file in the manifest present with its sha256,
    nothing unlisted in the split folders, and — for a cabled set — every
    source capture unchanged since the dataset was built."""
    problems: list[str] = []
    try:
        m = load_manifest(path)
    except (DatasetRefusal, json.JSONDecodeError) as e:
        return False, [str(e)]
    d = _dir(path)
    files = m.get("files", {})
    for rel, h in files.items():
        p = d / rel
        if not p.exists():
            problems.append(f"{rel} is missing")
        elif sha256_file(p) != h:
            problems.append(f"{rel} changed since the dataset was built (its "
                            "hash no longer matches)")
    for s in SPLITS:
        sd = d / s
        if sd.is_dir():
            for f in sd.rglob("*"):
                if f.is_file() and f.relative_to(d).as_posix() not in files:
                    problems.append(f"{f.relative_to(d).as_posix()} is not in "
                                    "the manifest")
    for src in m.get("sources", []) or []:
        sp = Path(src["capture"])
        if not sp.is_absolute():
            rf_root = d.parents[2] if len(d.parents) >= 3 else d
            sp = rf_root / sp
        if not sp.exists():
            problems.append(f"source capture {src['capture']} is missing")
        elif sha256_file(sp) != src.get("sha256"):
            problems.append(f"source capture {src['capture']} changed since the "
                            "dataset was built")
    return (not problems), problems


# ---------------------------------------------------------------------------
# throughput
# ---------------------------------------------------------------------------
def measure_throughput(profile, generator: str = "native",
                       classes: list | None = None, n: int = 8,
                       canonical_class: str = "voice", seed: int = 0,
                       window: int | None = None, compute_scf: bool = False,
                       rf=None) -> dict:
    """Examples a second this machine generates (nothing is written): the
    first number to read before planning a night's build."""
    prof = _profile(rf, profile) if rf is not None or not isinstance(profile, str) \
        else _profiles.new_profile(profile)
    can = _canonical(prof, canonical_class)
    L = int(window or DEFAULT_WINDOW)
    _gen_words(generator)
    if classes is None:
        fit = [c.name for c in _classes.CLASSES if 0 < c.bandwidth_hz <= 0.9 * can.rate]
        classes = fit[:4] or ["ref_qpsk"]
    _check_classes(prof, classes, can, generator)
    scf = _scf_fn()[0] if compute_scf else None
    bw_pool = [c.bandwidth_hz for c in map(_classes.get, classes) if c.bandwidth_hz > 0]
    t = time.time()
    k = 0
    for j in range(int(n)):
        for ci, cls in enumerate(classes):
            _example(prof, cls, ci, j, seed, can, L, (0.0, 20.0), generator, 0,
                     bw_pool, scf)
            k += 1
    dt = time.time() - t
    return {"profile": prof.id, "generator": generator, "examples": k,
            "seconds": round(dt, 3), "examples_per_s": round(k / max(dt, 1e-9), 2),
            "canonical": can.cls, "window": L, "scf": bool(scf), "classes": classes}
