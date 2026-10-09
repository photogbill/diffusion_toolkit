# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The toolkit's command line: `python -m atk_diffusion <command> …`
(`atkdiff.bat <command> …` on Windows runs exactly that).

WHY A COMMAND LINE. The toolkit has no window of its own — ATK owns the tabs
(ARCHITECTURE §6) — but every first experiment in the plan (§4) has to be
runnable by Bill on his own machine before any tab exists, and ATK's
`diffusion_host.run_job` runs training and experiments as a separate process
the same way: `<envs\\atk_diffusion python> -m atk_diffusion <args…>`. So
this module is the one door: each command CALLS THE TOOLKIT'S PUBLIC
FUNCTIONS (nothing is reimplemented here) and says what happened in plain
words — a sentence per result and per failure (plan §2.9, Bill-proof).

    status                         what is installed, the rf_data root, the profiles
    rfdata init | verify           make the rf_data tree; check every file in the write log
    profile list | show | new | set-safe-input | geometry
    impair measure <capture>       a terminated capture -> the profile's impairments + floor
    synth narrowband | wideband | cabled | throughput | list
    train <kind>                   proposer classifier ssl calibrate denoiser inpainter
                                   translator augmenter fingerprint radiomap position
                                   vitals genclass beacon anomaly
    export onnx <model>            make (or check) a model's ONNX file for ATK's CPU path
    models list                    every model card under rf_data, in words
    detect <capture>               the detector over a recorded capture
    cut <capture> --t0 --t1 --f-lo --f-hi [--clean METHOD]
    resample <capture> --to PROFILE
    experiment --list | <name>     every first experiment (experiments.*)
    cabled check | plan | txfile | measure | align
    products list | import | reach
    hunt simulate                  the self-hunting receiver against a simulated band
    vitals replay <csi log>        breathing / heart rate from a saved ESP32 CSI log
    repair audio <wav>             fill dropouts in a WAV (INFERRED tier, listed)

EXIT CODES: 0 done; 1 refused or failed (the sentence says why); 2 the
command line itself was wrong (argparse's usage message).

THE rf_data ROOT: `--rf-data PATH`, else the environment variable
ATK_RF_DATA, else beside the code (`paths.default_root()`) — never AppData,
never a sync folder (refused in words).

TRANSMITTING. The cabled loop is the ONE approved exception to receive-only
(plan §3.5, D10). This command line never transmits unless BOTH
`--execute` and `--i-confirm-cabled-with-attenuators` are given AND the
safety verdict (`cabled.safety.check`, with the ramp) permits — and then it
runs exactly the command `cabled.loop.plan_run` printed.

NO CONSOLE SCRIPT. ATK's rule: `python -m …` always, never a Scripts\\*.exe
shim (pyproject.toml declares none).

LIMITS. Long jobs print their progress as they go; a training run is as
long as its arguments make it. Commands that need PyTorch say so in words in
an environment without it (ATK's core), and the rest still work there.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2

#: Training kinds, in the order `train --help` lists them, with what each is.
TRAIN_KINDS = {
    "proposer":    "the 2D AI proposer (FCOS boxes on the spectrogram) from a wideband dataset",
    "classifier":  "the 1D classifier (IQ + SCF -> class, embedding) from a narrowband dataset",
    "ssl":         "self-supervised pretraining of a backbone (2d tiles or 1d cuts)",
    "calibrate":   "temperature scaling and open-set thresholds for a trained model",
    "denoiser":    "the B3 diffusion denoiser for weak bursts",
    "inpainter":   "the D1 IQ dropout inpainter",
    "translator":  "the A6 receiver-to-receiver translator from paired cabled recordings",
    "augmenter":   "the B4 time-frequency diffusion augmenter from cabled recordings",
    "fingerprint": "the C1 fingerprint embedder from labelled bursts",
    "radiomap":    "the E2/E5 radio-map residual model",
    "position":    "the E3 where-am-I position model from drive logs",
    "vitals":      "the I1 CSI vital-signs LSTM",
    "genclass":    "generative classification (research)",
    "beacon":      "the beacon co-designed with its detector (research)",
    "anomaly":     "the flow-anomaly model (research, CyberWolf)",
}


class Fail(Exception):
    """A refusal or failure the operator must read; the message is the
    sentence. Exit code 1."""


# ---------------------------------------------------------------------------
# output and context
# ---------------------------------------------------------------------------
class Ctx:
    """What every command gets: the parsed arguments, the printer, the
    rf_data root (opened lazily, created only by commands that write)."""

    def __init__(self, args, out=None, err=None):
        self.a = args
        self.json = bool(getattr(args, "json", False))
        self.out = out or sys.stdout
        self.err = err or sys.stderr
        self._rf = None

    # -- words -----------------------------------------------------------
    def say(self, line: str = "") -> None:
        """A sentence for the operator (stderr in --json mode, so stdout is
        pure JSON)."""
        stream = self.err if self.json else self.out
        print(str(line), file=stream, flush=True)

    def progress(self, msg: str) -> None:
        self.say(f"  ... {msg}")

    # -- rf_data -----------------------------------------------------------
    def root(self) -> Path:
        from atk_diffusion import paths
        given = getattr(self.a, "rf_data", None)
        if given:
            return Path(given)
        return paths.default_root()

    def rf(self, create: bool = False):
        """The RfData. Commands that write pass create=True: the tree is made
        (and said) when it is not there yet."""
        from atk_diffusion import paths
        if self._rf is None:
            root = self.root()
            try:
                self._rf = paths.RfData(root)
            except paths.RootRefused as e:
                raise Fail(f"the rf_data folder was refused: {e}") from None
        if create and not (self._rf.root / "README.txt").exists():
            existed = self._rf.root.is_dir()
            self._rf.ensure()
            if not existed:
                self.say(f"Created a new rf_data folder at {self._rf.root}.")
        return self._rf


def _json_safe(obj):
    from atk_diffusion.experiments.report import clean
    return clean(obj)


def _describe(pid: str) -> str:
    from atk_diffusion import profiles
    try:
        return profiles.describe(pid)
    except ValueError:
        return pid


def _rate_words(r: float) -> str:
    r = float(r)
    if r >= 1e6:
        return f"{r / 1e6:.6g} MS/s"
    if r >= 1e3:
        return f"{r / 1e3:.6g} kS/s"
    return f"{r:g} S/s"


def _hz(f: float) -> str:
    f = float(f)
    a = abs(f)
    if a >= 1e9:
        return f"{f / 1e9:.6g} GHz"
    if a >= 1e6:
        return f"{f / 1e6:.6g} MHz"
    if a >= 1e3:
        return f"{f / 1e3:.6g} kHz"
    return f"{f:g} Hz"


def _floats(text, what: str) -> list[float]:
    """'1,2.5,3' -> [1.0, 2.5, 3.0], refused in words."""
    try:
        return [float(v) for v in str(text).replace(" ", "").split(",") if v != ""]
    except ValueError:
        raise Fail(f"{what} is a comma-separated list of numbers, e.g. 1,2,3; "
                   f"got {text!r}") from None


def _words(text) -> list[str]:
    return [v.strip() for v in str(text).split(",") if v.strip()]


def _profile_id(text: str) -> str:
    from atk_diffusion import profiles
    pid = str(text).strip().lower()
    try:
        profiles.parse_profile_id(pid)
    except ValueError as e:
        raise Fail(f"{e}. Examples: rtlsdr_2400000_cu8, hackrf_8000000_ci8, "
                   "bladerf1_4000000_ci16, krakensdr_2400000_cu8_ch0. "
                   "`atkdiff.bat profile list` shows yours.") from None
    return pid


def _capture(path: str) -> Path:
    """A SigMF capture given as base, .sigmf-data or .sigmf-meta."""
    from atk_diffusion import sigmf
    p = Path(path)
    if not sigmf.meta_path(p).exists():
        raise Fail(f"no SigMF capture at {p}: {sigmf.meta_path(p).name} is "
                   "missing (a capture is a .sigmf-data and .sigmf-meta pair)")
    if not sigmf.data_path(p).exists():
        raise Fail(f"{sigmf.data_path(p).name} is missing beside its "
                   ".sigmf-meta — the samples are not there")
    return sigmf.base_of(p)


def _capture_profile(cap: Path, given: str | None = None) -> str:
    """The capture's receiver profile, from its own metadata; when the
    operator named one, a different one is refused in words."""
    from atk_diffusion import profiles, sigmf
    meta = sigmf.read_meta(cap)
    pid = profiles.profile_from_meta(meta)
    if given:
        want = _profile_id(given)
        if want != pid:
            raise profiles.ProfileMismatch(
                f"{cap.name} is a capture of {_describe(pid)} ({pid}), not of "
                f"{_describe(want)} ({want}). Profiles never mix: use the "
                "capture's own profile, or resample it deliberately "
                "(`atkdiff.bat resample`, a logged step).")
    return pid


def _dataset_dir(rf, pid: str, name_or_path) -> Path:
    if name_or_path is None:
        return None
    p = Path(name_or_path)
    if (p / "manifest.json").exists() or p.name == "manifest.json":
        return p.parent if p.name == "manifest.json" else p
    d = Path(rf.datasets(pid, str(name_or_path)))
    if not (d / "manifest.json").exists():
        raise Fail(f"there is no dataset {name_or_path!r} for {pid} (looked in "
                   f"{d}). `atkdiff.bat synth list --profile {pid}` lists them.")
    return d


def _model_dir(rf, pid: str | None, name_or_path) -> Path:
    if name_or_path is None:
        return None
    p = Path(name_or_path)
    if (p / "card.json").exists():
        return p
    cands = []
    if pid:
        cands.append(Path(rf.models(pid, str(name_or_path))))
    cands.append(Path(rf.shared()) / "models" / str(name_or_path))
    for d in cands:
        if (d / "card.json").exists():
            return d
    raise Fail(f"there is no model {name_or_path!r}"
               + (f" for {pid}" if pid else "") + f" (looked in "
               + " and ".join(str(c) for c in cands)
               + "). `atkdiff.bat models list` lists them.")


def _shared_models(rf, name: str) -> Path:
    """Where a model that is not per receiver profile lives."""
    return Path(rf.shared()) / "models" / name


def _shared_run(rf, name: str) -> Path:
    """A fresh run folder for an experiment that is not per receiver
    profile (speech, text, pulses): <rf_data>\\shared\\runs\\<stamp>_<name>."""
    base = Path(rf.shared()) / "runs"
    stem = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{name}"
    d = base / stem
    k = 2
    while d.exists():
        d = base / f"{stem}_{k}"
        k += 1
    d.mkdir(parents=True)
    return d


def _need_torch(what: str) -> None:
    from atk_diffusion import capabilities
    ok, why = capabilities.can_train()
    if not ok:
        raise Fail(f"{what} needs PyTorch. {why} On a stand-alone toolkit, "
                   "install.bat builds the .venv that has it.")


def _card_lines(model_dir) -> list[str]:
    from atk_diffusion import cards
    try:
        card = cards.load(model_dir, verify_weights=False)
    except (cards.CardRefusal, ValueError) as e:
        return [f"(its card could not be read: {e})"]
    return cards.summary(card)


def _saved_model(ctx, d, what: str) -> dict:
    d = Path(d)
    if d.name == "card.json":            # some savers return the card's path
        d = d.parent
    ctx.say(f"Saved {what} to {d}.")
    for line in _card_lines(d):
        ctx.say(f"  {line}")
    return {"model_dir": str(d)}


# ---------------------------------------------------------------------------
# status, rfdata
# ---------------------------------------------------------------------------
def _profile_counts(rf, pid: str) -> dict:
    def n_dirs(base, marker):
        base = Path(base)
        if not base.is_dir():
            return 0
        return sum(1 for d in base.iterdir() if d.is_dir() and (d / marker).exists())
    caps = Path(rf.captures(pid))
    n_caps = len(list(caps.glob("*.sigmf-meta"))) if caps.is_dir() else 0
    cab = Path(rf.cabled(pid))
    n_cab = len(list(cab.glob("*.sigmf-meta"))) if cab.is_dir() else 0
    cuts = Path(rf.cuts(pid))
    n_cuts = sum(1 for d in cuts.iterdir() if d.is_dir()) if cuts.is_dir() else 0
    return {"captures": n_caps, "cabled": n_cab,
            "datasets": n_dirs(rf.datasets(pid), "manifest.json"),
            "models": n_dirs(rf.models(pid), "card.json"), "cuts": n_cuts}


def _profile_brief(rf, pid: str) -> str:
    from atk_diffusion import profiles
    from atk_diffusion.dsp import impair
    try:
        prof = profiles.load_profile(rf, pid)
    except (ValueError, OSError) as e:
        return f"{pid} — not a usable profile ({e})"
    c = _profile_counts(rf, pid)
    meas = "impairments measured" if impair.is_measured(prof.impairments) \
        else "impairments NOT measured"
    si = prof.safe_input
    safe = "safe input entered" if si and si.max_dbm is not None else \
        "safe input not entered"
    filed = "" if Path(rf.profile_json(pid)).exists() else " (no profile file yet)"
    return (f"{pid} — {profiles.describe(pid)}{filed}; {meas}; {safe}; "
            f"{c['captures']} captures, {c['cabled']} cabled, {c['datasets']} "
            f"datasets, {c['models']} models, {c['cuts']} cuts")


def cmd_status(ctx) -> dict:
    from atk_diffusion import __version__, capabilities, paths, profiles
    rep = capabilities.report()
    ctx.say(f"ATK Diffusion Toolkit {__version__} — `python -m atk_diffusion`, "
            f"running on Python {rep['python']} ({sys.executable}).")
    ctx.say(f"Toolkit code: {paths.toolkit_root()}")
    for line in capabilities.lines(rep):
        ctx.say(line)
    can_t, why_t = capabilities.can_train()
    can_i, why_i = capabilities.can_infer()
    ctx.say("Training and dataset generation: possible in this environment."
            if can_t else f"Training: not in this environment — {why_t}")
    ctx.say("Trained models (ONNX) can run here on the CPU." if can_i
            else f"Trained models: {why_i}")
    gpu = capabilities.gpu() if can_t else None
    if gpu is not None:
        ctx.say(gpu["words"])
    root = ctx.root()
    ok, why = paths.check_root(root)
    out = {"version": __version__, "python": rep["python"], "capabilities": rep,
           "can_train": can_t, "can_infer": can_i, "gpu": gpu, "rf_data": str(root),
           "rf_data_ok": ok, "profiles": []}
    if not ok:
        ctx.say(f"[ERR] rf_data: {why}")
        raise Fail("the rf_data folder is refused; pass --rf-data with a folder "
                   "on the data drive (outside AppData and sync folders)")
    exists = root.is_dir()
    ctx.say(f"rf_data: {root}" + ("" if exists else
            " — does not exist yet (`atkdiff.bat rfdata init` makes it)")
            + (f" ({why})" if why else ""))
    out["rf_data_exists"] = exists
    if exists:
        rf = ctx.rf()
        known = rf.known_profiles()
        if not known:
            ctx.say("No receiver profiles yet. Make one, e.g. "
                    "`atkdiff.bat profile new rtlsdr_2400000_cu8`.")
        else:
            ctx.say(f"Receiver profiles ({len(known)}):")
            for pid in known:
                ctx.say(f"  {_profile_brief(rf, pid)}")
        out["profiles"] = known
    ctx.say("Receiver families: " + ", ".join(
        f"{k} ({v['label']}, {v['datatype'] or 'any datatype'})"
        for k, v in profiles.FAMILIES.items()))
    return out


def cmd_rfdata_init(ctx) -> dict:
    rf = ctx.rf(create=True)
    ctx.say(f"rf_data is ready at {rf.root} (README.txt explains the layout; "
            "install, update and uninstall never touch it).")
    if rf.note:
        ctx.say(f"Note: {rf.note}")
    for sub in ("profiles", "environments", "products", "shared"):
        ctx.say(f"  {sub}\\ — present")
    return {"root": str(rf.root), "profiles": rf.known_profiles()}


def cmd_rfdata_verify(ctx) -> dict:
    rf = ctx.rf()
    if not rf.root.is_dir():
        raise Fail(f"{rf.root} does not exist — nothing to verify "
                   "(`atkdiff.bat rfdata init` makes it)")
    entries = rf.log.entries()
    good, changed, missing = [], [], []
    for rel in sorted(entries):
        p = Path(rel)
        p = p if p.is_absolute() else rf.root / p
        ok, why = rf.log.verify(p, full=bool(ctx.a.full))
        if ok:
            good.append(rel)
        elif "missing" in why:
            missing.append((rel, why))
        else:
            changed.append((rel, why))
    ctx.say(f"Checked {len(entries)} files in the write log of {rf.root}"
            + (" (every file re-hashed)." if ctx.a.full else
               " (size and time; a file whose size or time moved is re-hashed)."))
    ctx.say(f"{len(good)} unchanged, {len(changed)} changed since they were "
            f"written, {len(missing)} missing.")
    for rel, why in changed[:50]:
        ctx.say(f"[ERR] CHANGED: {rel} — {why}")
    for rel, why in missing[:50]:
        ctx.say(f"[ERR] MISSING: {rel}")
    if len(changed) + len(missing) > 100:
        ctx.say(f"(and {len(changed) + len(missing) - 100} more)")
    res = {"checked": len(entries), "unchanged": len(good),
           "changed": [r for r, _ in changed], "missing": [r for r, _ in missing]}
    if changed or missing:
        raise Fail(f"{len(changed)} changed and {len(missing)} missing file(s): "
                   "a file that changed after it was written is named, not used")
    return res


# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------
def _profile_lines(rf, prof) -> list[str]:
    from atk_diffusion import profiles
    from atk_diffusion.dsp import impair
    st, fm = prof.stft, prof.fam
    rates = ", ".join(f"{c.cls} {_rate_words(c.rate)} (÷{c.decimation}"
                      + (", limited" if c.limited else "") + ")"
                      for c in prof.canonical_rates())
    lines = [f"{prof.id} — {profiles.describe(prof.id)}",
             f"  ADC {prof.adc_bits or '?'} bits, {prof.channels} channel(s); "
             f"serial {prof.device_serial or 'not recorded'}, firmware "
             f"{prof.firmware or 'not recorded'}, gain preset "
             f"{prof.gain_preset or 'not recorded'}",
             f"  canonical rates for cuts: {rates}",
             f"  spectrogram: {st.fft_size}-point FFT, hop {st.hop}, {st.window} "
             f"window, {st.tile_seconds:g} s tiles max-pooled to {st.tile_rows} "
             f"rows, {100 * st.tile_overlap:g} % overlap (RBW "
             f"{_hz(st.rbw_hz(prof.sample_rate))})",
             f"  SCF (FAM): channel FFT {fm.channel_fft}, hop {fm.hop}, "
             f"{fm.window}, at most {fm.max_seconds:g} s",
             f"  CFAR false-alarm probability {prof.cfar_pfa:g} per cell; "
             f"escalate below {prof.escalate_snr_db:g} dB SNR",
             f"  impairments: {impair.describe(prof.impairments)}"]
    imp = prof.impairments or {}
    if imp.get("floor_db_per_bin") is not None:
        lines.append(f"  detector floor: measured {imp.get('floor_measured', '?')} "
                     f"({imp.get('floor_source', '')})")
    else:
        lines.append("  detector floor: not measured (estimated from live data "
                     "until `atkdiff.bat impair measure` files one)")
    si = prof.safe_input
    if si is None or si.max_dbm is None:
        lines.append("  maximum safe input: NOT entered — the cabled loop refuses "
                     "until it is entered from the data sheet "
                     "(`atkdiff.bat profile set-safe-input`)")
    else:
        lines.append(f"  maximum safe input: {si.max_dbm:+g} dBm, from "
                     f"\"{si.source}\", entered {si.entered}")
    for n in prof.notes:
        lines.append(f"  note: {n}")
    return lines


def cmd_profile_list(ctx) -> dict:
    rf = ctx.rf()
    known = rf.known_profiles() if rf.root.is_dir() else []
    if not known:
        ctx.say(f"No receiver profiles in {rf.root} yet. Make one with "
                "`atkdiff.bat profile new <family>_<rate>_<datatype>`, e.g. "
                "rtlsdr_2400000_cu8.")
    for pid in known:
        ctx.say(_profile_brief(rf, pid))
    return {"profiles": known}


def cmd_profile_show(ctx) -> dict:
    from atk_diffusion import profiles
    rf = ctx.rf()
    pid = _profile_id(ctx.a.profile_id)
    prof = profiles.load_profile(rf, pid)
    for line in _profile_lines(rf, prof):
        ctx.say(line)
    if not Path(rf.profile_json(pid)).exists():
        ctx.say(f"(no profile file yet — these are the family's defaults; "
                f"`atkdiff.bat profile new {pid}` files it)")
    return prof.to_json()


def cmd_profile_new(ctx) -> dict:
    from atk_diffusion import profiles
    a = ctx.a
    if a.profile_id:
        pid = _profile_id(a.profile_id)
    else:
        if not (a.family and a.rate):
            raise Fail("name the profile (e.g. rtlsdr_2400000_cu8), or give "
                       "--family and --rate (and --datatype, --variant)")
        fam = str(a.family).lower()
        if fam not in profiles.FAMILIES:
            raise Fail(f"unknown receiver family {a.family!r} — one of "
                       + ", ".join(profiles.FAMILIES))
        dt = a.datatype or profiles.FAMILIES[fam]["datatype"]
        if not dt:
            raise Fail(f"{fam} has no default datatype; give --datatype "
                       "(cu8, ci8, ci16 or cf32)")
        pid = profiles.make_profile_id(fam, float(a.rate), dt, a.variant or "")
    rf = ctx.rf(create=True)
    path = Path(rf.profile_json(pid))
    if path.exists() and not a.overwrite:
        raise Fail(f"{pid} already exists ({path}). `atkdiff.bat profile show "
                   f"{pid}` shows it; --overwrite starts it again from the "
                   "family's defaults (its measured impairments and its "
                   "safe-input entry are lost)")
    prof = profiles.new_profile(pid)
    for attr in ("device_serial", "firmware", "gain_preset"):
        v = getattr(a, attr, None)
        if v:
            setattr(prof, attr, str(v))
    p = profiles.save_profile(rf, prof)
    Path(rf.profile_dir(pid)).mkdir(parents=True, exist_ok=True)
    ctx.say(f"Profile {pid} — {profiles.describe(pid)} — saved to {p}.")
    ctx.say("Next: record a TERMINATED capture (antenna port terminated, 50 Ω) "
            f"and run `atkdiff.bat impair measure <capture>` to measure this "
            "receiver's impairments and floor.")
    for line in _profile_lines(rf, prof)[1:]:
        ctx.say(line)
    return {"profile": pid, "path": str(p)}


def cmd_profile_safe_input(ctx) -> dict:
    import datetime as _dt
    from atk_diffusion import profiles
    a = ctx.a
    pid = _profile_id(a.profile_id)
    if not math.isfinite(float(a.max_dbm)):
        raise Fail("--max-dbm is a number in dBm, from the receiver's data sheet")
    src = str(a.source or "").strip()
    if not src:
        raise Fail("--source is required: where the number came from (the data "
                   "sheet's name and revision, a page, a URL). A number "
                   "without its source is a number from memory")
    try:
        when = _dt.date.fromisoformat(str(a.date).strip())
    except ValueError:
        raise Fail(f"--date is the day it was entered, as YYYY-MM-DD; got "
                   f"{a.date!r}") from None
    rf = ctx.rf(create=True)
    prof = profiles.load_profile(rf, pid)
    prof.safe_input = profiles.SafeInput(float(a.max_dbm), src, when.isoformat())
    prof.notes = [n for n in prof.notes if not str(n).startswith("defaults")]
    prof.notes.append(f"maximum safe input {float(a.max_dbm):+g} dBm entered "
                      f"{when.isoformat()} from {src}")
    p = profiles.save_profile(rf, prof)
    ctx.say(f"{pid}: maximum safe input {float(a.max_dbm):+g} dBm, from \"{src}\", "
            f"entered {when.isoformat()} — saved to {p}.")
    ctx.say(f"The cabled loop's ceiling for this receiver is "
            f"{float(a.max_dbm) - 20.0:+g} dBm (20 dB under the data-sheet "
            "maximum, plan D10).")
    return {"profile": pid, "safe_input": {"max_dbm": float(a.max_dbm),
                                           "source": src, "entered": when.isoformat()}}


_GEOM = {"fft_size": ("stft", int), "hop": ("stft", int), "window": ("stft", str),
         "tile_seconds": ("stft", float), "tile_rows": ("stft", int),
         "tile_overlap": ("stft", float), "fam_channel_fft": ("fam", int),
         "fam_hop": ("fam", int), "fam_window": ("fam", str),
         "fam_max_seconds": ("fam", float)}


def cmd_profile_geometry(ctx) -> dict:
    from dataclasses import asdict
    from atk_diffusion import profiles
    a = ctx.a
    pid = _profile_id(a.profile_id)
    rf = ctx.rf()
    prof = profiles.load_profile(rf, pid)
    changes = {}
    for key, (where, typ) in _GEOM.items():
        v = getattr(a, key, None)
        if v is None:
            continue
        attr = key[4:] if key.startswith("fam_") else key
        obj = getattr(prof, where)
        old = getattr(obj, attr)
        new = typ(v)
        if new != old:
            setattr(obj, attr, new)
            changes[f"{where}.{attr}"] = [old, new]
    if changes:
        if prof.stft.hop <= 0 or prof.stft.fft_size <= 0 or prof.stft.tile_rows <= 0:
            raise Fail("the FFT size, hop and tile rows must be positive")
        if not 0.0 <= prof.stft.tile_overlap < 1.0:
            raise Fail("the tile overlap is a fraction from 0 up to (not including) 1")
        rf = ctx.rf(create=True)
        prof.notes.append(f"geometry changed {time.strftime('%Y-%m-%d')}: "
                          + "; ".join(f"{k} {o} -> {n}" for k, (o, n) in changes.items()))
        profiles.save_profile(rf, prof)
        for k, (o, n) in changes.items():
            ctx.say(f"{pid}: {k} changed from {o} to {n}.")
        ctx.say("A model is refused for a spectrogram geometry it was not trained "
                "on: models of this profile trained before this change will be "
                "refused (in words) until they are retrained, and a measured "
                "detector floor of another FFT size is re-measured.")
    st = prof.stft
    ctx.say(f"{pid} spectrogram: {st.fft_size}-point FFT, hop {st.hop}, "
            f"{st.window}, {st.tile_seconds:g} s tiles of {st.tile_rows} rows, "
            f"{100 * st.tile_overlap:g} % overlap; RBW "
            f"{_hz(st.rbw_hz(prof.sample_rate))}.")
    ctx.say(f"{pid} SCF (FAM): channel FFT {prof.fam.channel_fft}, hop "
            f"{prof.fam.hop}, {prof.fam.window}, at most {prof.fam.max_seconds:g} s.")
    return {"profile": pid, "stft": asdict(prof.stft), "fam": asdict(prof.fam),
            "changed": changes}


# ---------------------------------------------------------------------------
# impairments
# ---------------------------------------------------------------------------
def cmd_impair_measure(ctx) -> dict:
    from atk_diffusion import profiles
    from atk_diffusion.dsp import impair
    from atk_diffusion.dsp.floor import NoiseFloor
    a = ctx.a
    cap = _capture(a.capture)
    pid = _capture_profile(cap, a.profile)
    rf = ctx.rf(create=True)
    ctx.say(f"Measuring {cap.name} as a TERMINATED capture of {_describe(pid)} "
            "(no antenna: everything in it is the receiver).")
    meas = impair.measure_sigmf(cap, skip_seconds=float(a.skip_seconds))
    path = impair.store(rf, pid, meas, device_serial=a.serial or "",
                        firmware=a.firmware or "")
    ctx.say(f"Impairments: {impair.describe(meas)}.")
    ctx.say(f"  floor {meas['floor_mean_dbfs']:.1f} dBFS; DC offset "
            f"({meas.get('dc_offset_i', 0):+.4f}, {meas.get('dc_offset_q', 0):+.4f}) "
            f"full scale; IQ imbalance {meas.get('iq_gain_imbalance_db', 0):+.2f} dB, "
            f"{meas.get('iq_phase_imbalance_deg', 0):+.2f}°; ENOB ≈ "
            f"{meas.get('enob_est', 0):.1f} bits at this gain")
    for s in (meas.get("spurs") or [])[:10]:
        ctx.say(f"  spur at {_hz(s['offset_hz'])} from centre, "
                f"{s['level_above_floor_db']:.1f} dB above the floor")
    out = {"profile": pid, "profile_file": str(path), "impairments": meas}
    if not a.no_floor:
        prof = profiles.load_profile(rf, pid)
        fl = NoiseFloor.from_terminated(cap, profile=prof,
                                        max_seconds=float(a.max_seconds))
        prof.impairments = dict(prof.impairments or {})
        prof.impairments.update(fl.to_impairments())
        profiles.save_profile(rf, prof)
        ctx.say(f"Detector floor: {fl.describe()} — filed in the profile at its "
                f"{prof.stft.fft_size}-point geometry.")
        out["floor"] = fl.describe()
    ctx.say(f"Saved to {path}. Synthetic data for {pid} now goes through THIS "
            "receiver's measured impairments. Re-measure when the device, its "
            "firmware or the gain preset changes.")
    return out


# ---------------------------------------------------------------------------
# synthetic data
# ---------------------------------------------------------------------------
def _splits(text) -> tuple:
    v = _floats(text, "--splits")
    if len(v) != 3:
        raise Fail("--splits is three fractions train,val,test that add up to "
                   "1, e.g. 0.8,0.1,0.1")
    return tuple(v)


def _dataset_lines(ctx, m: dict) -> None:
    from atk_diffusion import provenance
    sp = m.get("splits", {})
    ctx.say(f"Dataset {m['name']} ({m['kind']}) for {_describe(m['profile'])}: "
            f"{sp.get('train', 0)} train, {sp.get('val', 0)} val, "
            f"{sp.get('test', 0)} test; generator {m['generator']}.")
    if m.get("canonical"):
        c = m["canonical"]
        ctx.say(f"  cut to the {c['class']} canonical rate {_rate_words(c['rate'])} "
                f"by integer decimation ÷{c['decimation']}"
                + (f"; {m['window']['samples']}-sample windows" if m.get("window") else ""))
    if m.get("scene"):
        s = m["scene"]
        ctx.say(f"  scenes of {s['seconds']:g} s ({s['samples']} samples, written "
                f"as {s['datatype']}), {s['annotations']} labels in all")
    cls = m.get("classes", []) or []
    ctx.say(f"  classes: {', '.join(cls)}" if len(cls) <= 12 else
            f"  classes: {len(cls)} — the whole class table")
    tier = m.get("tier", "")
    if tier:
        ctx.say(f"  tier: {provenance.TIER_WORDS.get(tier, tier)}")
    if m.get("receiver"):
        ctx.say(f"  receiver: {m['receiver']}")
    for key in ("scf", "tiles"):
        if m.get(key):
            ctx.say(f"  {'SCF images' if key == 'scf' else 'tiles'}: {m[key]}")
    for n in m.get("notes", []) or []:
        ctx.say(f"  note: {n}")
    ctx.say(f"  {len(m.get('files', {}))} files, each hashed in manifest.json and "
            f"the rf_data write log: {m['path']}")


def cmd_synth_narrowband(ctx) -> dict:
    from atk_diffusion.synth import datasets as D
    a = ctx.a
    pid = _profile_id(a.profile)
    snr = _floats(a.snr, "--snr")
    if len(snr) != 2:
        raise Fail("--snr is LOW,HIGH in dB, e.g. 0,20")
    lo, hi = snr
    rf = ctx.rf(create=True)
    m = D.build_narrowband(rf, pid, a.name, _words(a.classes), int(a.n_per_class),
                           (float(lo), float(hi)), a.canonical,
                           generator=a.generator, seed=int(a.seed),
                           window=a.window, splits=_splits(a.splits),
                           compute_scf=not a.no_scf, progress=ctx.progress,
                           impairment_level=int(a.impairment_level),
                           overwrite=bool(a.overwrite))
    _dataset_lines(ctx, m)
    return m


def cmd_synth_wideband(ctx) -> dict:
    from atk_diffusion.synth import datasets as D
    a = ctx.a
    pid = _profile_id(a.profile)
    rf = ctx.rf(create=True)
    m = D.build_wideband(rf, pid, a.name, int(a.scenes), env=a.env or None,
                         generator=a.generator, seed=int(a.seed),
                         scene_seconds=a.seconds, splits=_splits(a.splits),
                         classes=_words(a.classes) if a.classes else None,
                         compute_tiles=not a.no_tiles, progress=ctx.progress,
                         overwrite=bool(a.overwrite))
    _dataset_lines(ctx, m)
    return m


def cmd_synth_cabled(ctx) -> dict:
    from atk_diffusion.synth import datasets as D
    a = ctx.a
    pid = _profile_id(a.profile)
    rf = ctx.rf(create=True)
    caps = [_capture(c) for c in a.captures]
    m = D.ingest_cabled(rf, pid, caps, a.name, canonical_class=a.canonical,
                        window=a.window, splits=_splits(a.splits),
                        compute_scf=not a.no_scf, seed=int(a.seed),
                        progress=ctx.progress, overwrite=bool(a.overwrite))
    _dataset_lines(ctx, m)
    return m


def cmd_synth_throughput(ctx) -> dict:
    from atk_diffusion.synth import datasets as D
    a = ctx.a
    pid = _profile_id(a.profile)
    rf = ctx.rf() if ctx.root().is_dir() else None
    r = D.measure_throughput(pid, generator=a.generator,
                             classes=_words(a.classes) if a.classes else None,
                             n=int(a.n), canonical_class=a.canonical,
                             window=a.window, compute_scf=bool(a.scf), rf=rf)
    ctx.say(f"{r['examples']} examples in {r['seconds']:.2f} s: "
            f"{r['examples_per_s']:.1f} examples a second with the {r['generator']} "
            f"generator ({r['canonical']} cuts of {r['window']} samples"
            f"{', with SCF images' if r['scf'] else ''}; classes "
            f"{', '.join(r['classes'])}). Nothing was written.")
    per_hour = r["examples_per_s"] * 3600
    ctx.say(f"At this rate one CPU process makes about {per_hour:,.0f} examples "
            "an hour — the number to read before planning a night's build.")
    return r


def cmd_synth_list(ctx) -> dict:
    from atk_diffusion.synth import datasets as D
    rf = ctx.rf()
    pids = [_profile_id(ctx.a.profile)] if ctx.a.profile else \
        (rf.known_profiles() if rf.root.is_dir() else [])
    out = []
    for pid in pids:
        base = Path(rf.datasets(pid))
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            if not (d / "manifest.json").exists():
                continue
            try:
                m = D.load_manifest(d)
            except (D.DatasetRefusal, ValueError) as e:
                ctx.say(f"[ERR] {pid}/{d.name}: {e}")
                continue
            sp = m.get("splits", {})
            line = (f"{pid}/{m['name']}: {m['kind']}, {m['generator']}, "
                    f"{sp.get('train', 0)}/{sp.get('val', 0)}/{sp.get('test', 0)} "
                    f"train/val/test, built {m.get('created', '?')}")
            row = {"profile": pid, "name": m["name"], "kind": m["kind"],
                   "path": str(d)}
            if ctx.a.verify:
                ok, problems = D.verify(d)
                line += " — verified" if ok else f" — {len(problems)} problem(s): " \
                                                 + "; ".join(problems[:3])
                row["ok"], row["problems"] = ok, problems
            ctx.say(line)
            out.append(row)
    if not out:
        ctx.say("No datasets yet. `atkdiff.bat synth narrowband …` or "
                "`synth wideband …` builds one.")
    return {"datasets": out}


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def _windows_from_captures(rf, pid: str, captures, window: int, sources,
                           max_per_label: int = 64):
    """Labelled complex windows [N, L] at the profile's own rate, shifted to
    baseband, from the SigMF annotations of recorded captures (the cabled
    loop's, or taught ones). -> (X, y, classes)."""
    import numpy as np
    from atk_diffusion import profiles, sigmf
    from atk_diffusion.detect import classes as _classes
    from atk_diffusion.dsp import resample
    fs = float(profiles.parse_profile_id(pid).sample_rate)
    X, labels = [], []
    for c in captures:
        cap = _capture(c)
        meta = sigmf.read_meta(cap)
        profiles.check_match(pid, profiles.profile_from_meta(meta),
                             what=f"the {pid} model being trained")
        centre = sigmf.center_of(meta)
        total = sigmf.num_samples(cap, meta)
        for ann in sigmf.annotations(meta):
            if sources and ann.source not in sources:
                continue
            if _classes.get(ann.label) is None or ann.freq_lower_edge is None \
                    or ann.freq_upper_edge is None:
                continue
            f_off = 0.5 * (ann.freq_lower_edge + ann.freq_upper_edge) - centre
            s, end = int(ann.sample_start), min(total, int(ann.sample_start
                                                           + ann.sample_count))
            k = 0
            while s + window <= end and k < max_per_label:
                x = sigmf.load(cap, s, window, meta=meta)
                if x.ndim > 1:
                    x = x[0]
                X.append(resample.shift(x, f_off, fs, n0=s))
                labels.append(ann.label)
                s += window
                k += 1
    if not X:
        raise Fail("the captures hold no labelled signal long enough for a "
                   f"{window}-sample window (annotations with atk:source in "
                   f"{', '.join(sources)} and a class from the class table)")
    names = [c.name for c in _classes.CLASSES if c.name in set(labels)]
    y = np.array([names.index(lb) for lb in labels], dtype=np.int64)
    return np.stack(X).astype(np.complex64), y, names


def _train_proposer(ctx, rf, pid):
    from atk_diffusion.learn import proposer2d
    a = ctx.a
    d = proposer2d.train(rf, pid, _dataset_dir(rf, pid, a.dataset), a.name,
                         int(a.epochs), batch_size=int(a.batch_size),
                         lr=float(a.lr or 1e-3), width=int(a.width),
                         pretrained=_model_dir(rf, pid, a.pretrained) if a.pretrained else None,
                         device=a.device, seed=int(a.seed), threads=a.threads,
                         max_tiles=a.max_tiles, amp=not a.no_amp,
                         overwrite=bool(a.overwrite), resume_from=a.resume,
                         progress=ctx.progress)
    return _saved_model(ctx, d, "the 2D proposer")


def _train_classifier(ctx, rf, pid):
    from atk_diffusion.learn import classifier1d
    a = ctx.a
    d = classifier1d.train(rf, pid, _dataset_dir(rf, pid, a.dataset), a.name,
                           int(a.epochs), rf_config=a.rf_config,
                           search_epochs=int(a.search_epochs), window=a.window,
                           use_scf=False if a.no_scf else None,
                           held_out_classes=tuple(_words(a.held_out)) if a.held_out else (),
                           batch_size=int(a.batch_size), lr=float(a.lr or 2e-3),
                           pretrained=_model_dir(rf, pid, a.pretrained) if a.pretrained else None,
                           device=a.device, seed=int(a.seed), threads=a.threads,
                           max_items=a.max_items, amp=not a.no_amp,
                           overwrite=bool(a.overwrite), resume_from=a.resume,
                           progress=ctx.progress)
    return _saved_model(ctx, d, "the 1D classifier")


def _train_ssl(ctx, rf, pid):
    from atk_diffusion.learn import ssl
    a = ctx.a
    ds = _dataset_dir(rf, pid, a.dataset) if a.dataset else None
    caps = [_capture(c) for c in (a.captures or [])] or None
    if ds is None and caps is None:
        raise Fail("ssl pretraining needs --dataset (its training split) or "
                   "--captures (raw SigMF captures of this profile)")
    if a.mode == "2d":
        d = ssl.pretrain_2d(rf, pid, a.name, dataset_dir=ds, captures=caps,
                            epochs=int(a.epochs), batch_size=int(a.batch_size),
                            lr=float(a.lr or 1e-3), device=a.device, seed=int(a.seed),
                            threads=a.threads, max_tiles=a.max_tiles,
                            overwrite=bool(a.overwrite), progress=ctx.progress)
    else:
        d = ssl.pretrain_1d(rf, pid, a.name, dataset_dir=ds, captures=caps,
                            canonical_class=a.canonical, window=a.window,
                            epochs=int(a.epochs), batch_size=int(a.batch_size),
                            lr=float(a.lr or 1e-3), device=a.device, seed=int(a.seed),
                            threads=a.threads, overwrite=bool(a.overwrite),
                            progress=ctx.progress)
    return _saved_model(ctx, d, f"the {a.mode} self-supervised backbone")


def _train_calibrate(ctx, rf, pid):
    from atk_diffusion import cards
    from atk_diffusion.learn import calibrate
    a = ctx.a
    md = _model_dir(rf, pid, a.model)
    ds = _dataset_dir(rf, pid, a.dataset)
    card = cards.load(md, for_profile=pid)
    if card.kind == "classifier1d":
        card = calibrate.calibrate_classifier(rf, pid, md, ds, split=a.split,
                                              threads=a.threads or 1,
                                              progress=ctx.progress)
    elif card.kind == "proposer2d":
        card = calibrate.calibrate_proposer(rf, pid, md, ds, split=a.split,
                                            threads=a.threads or 1,
                                            progress=ctx.progress)
    else:
        raise Fail(f"{card.name} is a {card.kind}; calibration is for the "
                   "classifier1d and proposer2d models")
    ctx.say(f"Calibrated {card.name} on {ds.name} ({a.split} split); the card "
            f"now carries the numbers:")
    for line in calibrate.describe(card.calibration or {}):
        ctx.say(f"  {line}")
    return {"model_dir": str(md), "calibration": card.calibration}


def _train_denoiser(ctx, rf, pid):
    from atk_diffusion.learn import denoiser
    a = ctx.a
    noise = [_capture(c) for c in (a.noise_capture or [])]
    data = [_capture(c) for c in (a.captures or [])] or None
    d = denoiser.train_denoiser(rf, pid, data, a.domain, name=a.name,
                                noise_captures=noise, synthetic=int(a.synthetic),
                                canonical=a.canonical, steps=int(a.steps),
                                batch=int(a.batch), seed=int(a.seed),
                                device=a.device, overwrite=bool(a.overwrite),
                                progress=ctx.progress)
    return _saved_model(ctx, d, "the B3 denoiser")


def _train_inpainter(ctx, rf, pid):
    from atk_diffusion.learn import inpaint
    a = ctx.a
    d = inpaint.train_inpainter(rf, pid, name=a.name, canonical=a.canonical,
                                window=int(a.window or 256),
                                synthetic=int(a.synthetic), steps=int(a.steps),
                                batch=int(a.batch), seed=int(a.seed),
                                device=a.device, overwrite=bool(a.overwrite),
                                progress=ctx.progress)
    return _saved_model(ctx, d, "the D1 inpainter")


def _train_translator(ctx, rf, pid):
    from atk_diffusion import profiles, sigmf
    from atk_diffusion.learn import translator
    a = ctx.a
    ca, cb = _capture(a.from_capture), _capture(a.to_capture)
    ma, mb = sigmf.read_meta(ca), sigmf.read_meta(cb)
    pa, pb = profiles.profile_from_meta(ma), profiles.profile_from_meta(mb)
    if a.from_profile:
        pa = _profile_id(a.from_profile)
    if pid != pb:
        profiles.check_match(pid, pb, what="the target profile you named")
    xa, xb = sigmf.load(ca, meta=ma), sigmf.load(cb, meta=mb)
    xa = xa[0] if xa.ndim > 1 else xa
    xb = xb[0] if xb.ndim > 1 else xb
    A_al, B_al, info = translator.align_pair(xa, xb)
    L = int(a.window or 256)
    A, B = translator.windows(A_al, L), translator.windows(B_al, L)
    n = min(len(A), len(B))
    if n < 8:
        raise Fail(f"the aligned overlap holds only {n} windows of {L} samples; "
                   "record longer paired captures")
    ctx.say(f"Aligned {ca.name} ({pa}) to {cb.name} ({pb}): {n} paired windows "
            f"of {L} samples ({info.get('words', '') or 'lag found from the envelopes'}).")
    d = translator.train_translator(
        rf, pa, pb, A[:n], B[:n],
        resampled_from=ma.get("global", {}).get("atk:resampled_from"),
        name=a.name, steps=int(a.steps), batch=int(a.batch), seed=int(a.seed),
        device=a.device, overwrite=bool(a.overwrite), progress=ctx.progress)
    return _saved_model(ctx, d, "the A6 translator")


def _train_augmenter(ctx, rf, pid):
    from atk_diffusion.learn import augment
    a = ctx.a
    X, y, names = _windows_from_captures(rf, pid, a.captures, int(a.window or 256),
                                         ("cabled",))
    ctx.say(f"{len(X)} cabled windows of {X.shape[1]} samples, {len(names)} classes: "
            + ", ".join(names))
    d = augment.train_augmenter(rf, pid, X, y, names, name=a.name,
                                steps=int(a.steps), batch=int(a.batch),
                                seed=int(a.seed), device=a.device,
                                progress=ctx.progress)
    return _saved_model(ctx, d, "the B4 augmenter")


def _train_genclass(ctx, rf, pid):
    from atk_diffusion.learn import genclass
    a = ctx.a
    X, y, names = _windows_from_captures(
        rf, pid, a.captures, int(a.window or 128),
        ("cabled", "taught", "confirmed", "synthetic"))
    ctx.say(f"{len(X)} labelled windows of {X.shape[1]} samples, "
            f"{len(names)} classes: " + ", ".join(names))
    d = genclass.train_genclass(rf, pid, X, y, names, name=a.name,
                                steps=int(a.steps), batch=int(a.batch),
                                seed=int(a.seed), device=a.device,
                                progress=ctx.progress)
    return _saved_model(ctx, d, "the generative classifier")


def _npz(path, keys: tuple, what: str) -> dict:
    import numpy as np
    p = Path(path)
    if not p.exists():
        raise Fail(f"{p} does not exist")
    with np.load(p, allow_pickle=False) as z:
        missing = [k for k in keys if k not in z.files]
        if missing:
            raise Fail(f"{p.name} is not {what}: it has no "
                       + ", ".join(missing) + f" (it has {', '.join(z.files)})")
        return {k: z[k] for k in z.files}


def _train_fingerprint(ctx, rf, pid):
    import numpy as np
    from atk_diffusion.learn import fingerprint
    a = ctx.a
    if a.bursts:
        z = _npz(a.bursts, ("bursts", "labels"), "a bursts file (arrays "
                 "'bursts' [N, L] complex and 'labels' [N], optional 'fs', "
                 "'class_names')")
        bursts = list(z["bursts"])
        labels = z["labels"]
        if labels.dtype.kind in "US":
            names = sorted(set(labels.tolist()))
            labels = np.array([names.index(v) for v in labels.tolist()])
        else:
            names = [str(v) for v in z["class_names"]] if "class_names" in z else None
        fs = float(z["fs"]) if "fs" in z else float(a.fs)
        trained_on = Path(a.bursts).name
    else:
        from atk_diffusion.experiments import fingerprint_eval as FE
        radios = FE.same_model_pair(int(a.seed), 2)
        data = FE.bursts(radios, int(a.synthetic), seed=int(a.seed))
        bursts = [x for x, _i, _s in data]
        labels = np.array([i for _x, i, _s in data])
        names = [r.name for r in radios] if hasattr(radios[0], "name") else None
        fs = 48_000.0
        trained_on = "synthetic same-model radios (experiments.fingerprint_eval)"
        ctx.say(f"Synthetic bursts: {len(bursts)} from {len(radios)} same-model radios.")
    out = rf.models(pid, a.name)
    d = fingerprint.train(bursts, labels, out, profile=pid, class_names=names,
                          fs=fs, steps=int(a.steps), seed=int(a.seed),
                          name=a.name, progress=ctx.progress)
    ctx.say(f"Trained on {trained_on}.")
    return _saved_model(ctx, d, "the C1 fingerprint embedder")


def _train_radiomap(ctx, rf, pid):
    from atk_diffusion.learn import radiomap
    a = ctx.a
    if a.fields:
        fields = _npz(a.fields, ("terrain", "physics", "mask", "samples"),
                      "a fields file (learn.radiomap.synthetic_fields' arrays)")
    else:
        fields = radiomap.synthetic_fields(int(a.synthetic), int(a.size),
                                           seed=int(a.seed))
        ctx.say(f"Synthetic fields: {int(a.synthetic)} patches of {int(a.size)}x"
                f"{int(a.size)} (the stand-in for a measured drive).")
    d = radiomap.train(fields, _shared_models(rf, a.name), steps=int(a.steps),
                       batch=int(a.batch), seed=int(a.seed), name=a.name,
                       progress=ctx.progress, threads=int(a.threads or 1))
    return _saved_model(ctx, d, "the radio-map residual model")


def _train_position(ctx, rf, pid):
    from atk_diffusion.geo import whereami
    from atk_diffusion.learn import position
    a = ctx.a
    if a.drive:
        recs = []
        for f in a.drive:
            recs += whereami.records_from_csv(f)
        calib = whereami.records_from_csv(a.calib) if a.calib else None
        trained_on = ", ".join(Path(f).name for f in a.drive)
    else:
        world = whereami.DriveWorld(seed=int(a.seed))
        recs = sum((world.drive(int(a.synthetic), seed=int(a.seed) + s)
                    for s in range(1, 4)), [])
        calib = world.drive(max(20, int(a.synthetic) // 3), seed=int(a.seed) + 50)
        trained_on = "the synthetic drive world (geo.whereami.DriveWorld)"
    if len(recs) < 20:
        raise Fail(f"{len(recs)} drive records is too few to train on")
    ctx.say(f"{len(recs)} drive records from {trained_on}"
            + (f"; {len(calib)} for calibration" if calib else ""))
    d = position.train(recs, _shared_models(rf, a.name), calib_records=calib,
                       epochs=int(a.epochs), seed=int(a.seed), name=a.name,
                       progress=ctx.progress)
    return _saved_model(ctx, d, "the where-am-I position model")


def _train_vitals(ctx, rf, pid):
    import numpy as np
    from atk_diffusion.experiments import vitals_eval
    from atk_diffusion.learn import vitals as LV
    from atk_diffusion.sensing import csi
    a = ctx.a
    fs = float(a.fs)
    if a.data:
        z = _npz(a.data, ("X", "y"), "a vitals file (X [windows, time, "
                 "channels] of learn.vitals.features, y [windows, 2] = "
                 "breathing, heart per minute)")
        X, y = z["X"], z["y"]
        trained_on = Path(a.data).name
    else:
        rng = np.random.default_rng(int(a.seed))
        Xs, ys = [], []
        for _ in range(int(a.synthetic)):
            b, h = float(rng.uniform(8, 24)), float(rng.uniform(55, 115))
            t, H, _tr = vitals_eval.synth_csi(40.0, fs, breath_bpm=b, heart_bpm=h,
                                              rng=np.random.default_rng(
                                                  int(rng.integers(1 << 30))))
            _tg, A, _info = csi.resample_uniform(t, np.abs(H), fs)
            Xs.append(LV.features(A, fs)[:int(30 * LV.FEATURE_FS)])
            ys.append([b, h])
        X, y = np.stack(Xs), np.array(ys)
        trained_on = (f"{int(a.synthetic)} synthetic CSI windows "
                      "(experiments.vitals_eval.synth_csi) — a code-path proof")
    prof = vitals_eval.profile_for(fs)
    out = rf.models(prof, a.name)
    from atk_diffusion.sensing.vitals import RESEARCH_LABEL
    _card, metrics = LV.train(X, y, out, name=a.name, epochs=int(a.epochs),
                              seed=int(a.seed), profile=prof, trained_on=trained_on,
                              progress=ctx.progress)
    ctx.say(f"Trained on {trained_on}. {RESEARCH_LABEL}")
    res = _saved_model(ctx, out, "the CSI vital-signs LSTM")
    res["metrics"] = metrics
    return res


def _train_beacon(ctx, rf, pid):
    from atk_diffusion.learn import beacon
    a = ctx.a
    out = rf.models(pid, a.name)
    m = beacon.train(pid, out, k=int(a.k), n=int(a.n), steps=int(a.steps),
                     seed=int(a.seed), progress=ctx.progress)
    ev = beacon.evaluate(m, blocks=int(a.eval_blocks))
    for line in beacon.report_md(ev).splitlines()[:12]:
        if line.strip():
            ctx.say(f"  {line}")
    res = _saved_model(ctx, out, "the co-designed beacon")
    res["evaluation"] = ev
    return res


def _train_anomaly(ctx, rf, pid):
    from atk_diffusion.learn import anomaly
    a = ctx.a
    p = Path(a.flows)
    if not p.exists():
        raise Fail(f"{p} does not exist")
    text = p.read_text(encoding="utf-8")
    try:
        flows = json.loads(text)
    except json.JSONDecodeError:
        flows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(flows, dict):
        flows = flows.get("flows", [])
    X, cols = anomaly.featurize(flows)
    model = anomaly.PcaModel(int(a.components)) if a.model == "pca" \
        else anomaly.AutoencoderModel(seed=int(a.seed))
    model.fit(X)
    d = anomaly.save(model, _shared_models(rf, a.name), name=a.name,
                     trained_on=p.name)
    ctx.say(f"{len(X)} baseline flows, features: {', '.join(cols)}. A score is "
            "CONTEXT for an analyst, never a suppression (CyberWolf's rule).")
    return _saved_model(ctx, d, f"the flow-anomaly model ({a.model})")


_TRAINERS = {"proposer": _train_proposer, "classifier": _train_classifier,
             "ssl": _train_ssl, "calibrate": _train_calibrate,
             "denoiser": _train_denoiser, "inpainter": _train_inpainter,
             "translator": _train_translator, "augmenter": _train_augmenter,
             "fingerprint": _train_fingerprint, "radiomap": _train_radiomap,
             "position": _train_position, "vitals": _train_vitals,
             "genclass": _train_genclass, "beacon": _train_beacon,
             "anomaly": _train_anomaly}
#: kinds that are not per receiver profile (no --profile needed)
_NO_PROFILE = ("radiomap", "position", "vitals", "anomaly")


def cmd_train(ctx) -> dict:
    a = ctx.a
    kind = a.kind
    no_torch = kind == "calibrate" or (kind == "anomaly" and a.model == "pca")
    if not no_torch:
        _need_torch(f"training the {kind}")
    pid = None
    if kind not in _NO_PROFILE:
        if not getattr(a, "profile", None):
            raise Fail(f"train {kind} needs --profile (the receiver profile the "
                       "model is for)")
        pid = _profile_id(a.profile)
    rf = ctx.rf(create=True)
    try:
        import torch
        torch.set_num_threads(int(a.threads) if getattr(a, "threads", None) else
                              max(1, (os.cpu_count() or 2) - 1))
    except ImportError:
        pass
    ctx.say(f"Training {TRAIN_KINDS[kind]}" + (f" for {_describe(pid)}" if pid else "")
            + ". Progress follows; a long run can be stopped with Ctrl-C.")
    return _TRAINERS[kind](ctx, rf, pid)


# ---------------------------------------------------------------------------
# export, models
# ---------------------------------------------------------------------------
_ONNX_AT_TRAINING = ("proposer2d", "classifier1d", "denoiser", "inpainter",
                     "translator")


def cmd_export_onnx(ctx) -> dict:
    from atk_diffusion import cards
    a = ctx.a
    rf = ctx.rf()
    pid = _profile_id(a.profile) if a.profile else None
    md = _model_dir(rf, pid, a.model)
    card = cards.load(md, for_profile=pid)
    if card.kind == "fingerprint":
        from atk_diffusion.learn import fingerprint
        p = fingerprint.export_onnx(md, card.profile)
    elif card.kind == "radiomap":
        from atk_diffusion.learn import radiomap
        p = radiomap.export_onnx(md)
    elif card.kind in _ONNX_AT_TRAINING:
        found = sorted(Path(md).glob("*.onnx"))
        if not found:
            raise Fail(f"{card.name} ({card.kind}) has no .onnx file; this kind is "
                       "exported when it is trained — retrain it")
        p = found[0]
    else:
        raise Fail(f"{card.name} is a {card.kind} model; it runs in the training "
                   "environment and has no ONNX export (ATK's CPU path does not "
                   "use it)")
    from atk_diffusion import capabilities
    ok, why = capabilities.can_infer()
    info = {"onnx": str(p), "kind": card.kind}
    if ok:
        from atk_diffusion.learn import export as X
        sess = X.session(p, threads=1)
        ins = [f"{i.name} {i.shape}" for i in sess.get_inputs()]
        outs = [o.name for o in sess.get_outputs()]
        ctx.say(f"{p} loads in ONNX Runtime: inputs {', '.join(ins)}; outputs "
                f"{', '.join(outs)}.")
        info.update(inputs=ins, outputs=outs)
    else:
        ctx.say(f"{p} written; {why}")
    ctx.say(f"ONNX for {card.name} ({card.kind}, {card.tier.upper()} outputs): {p}")
    return info


def cmd_models_list(ctx) -> dict:
    from atk_diffusion import cards
    rf = ctx.rf()
    bases = []
    pids = [_profile_id(ctx.a.profile)] if ctx.a.profile else \
        (rf.known_profiles() if rf.root.is_dir() else [])
    for pid in pids:
        bases.append((pid, Path(rf.models(pid))))
    if not ctx.a.profile:
        bases.append(("shared", Path(rf.shared()) / "models"))
    out = []
    for pid, base in bases:
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            if not (d / cards.CARD_FILE).exists():
                continue
            try:
                card = cards.load(d, verify_weights=bool(ctx.a.verify))
                lines = cards.summary(card)
                ok = True
            except (cards.CardRefusal, ValueError) as e:
                lines, ok = [f"{d.name}: REFUSED — {e}"], False
            ctx.say(f"{pid}/{d.name}:")
            for line in lines:
                ctx.say(f"  {line}")
            out.append({"where": pid, "name": d.name, "path": str(d), "ok": ok})
    if not out:
        ctx.say("No trained models yet (`atkdiff.bat train <kind> …` makes one).")
    return {"models": out}


# ---------------------------------------------------------------------------
# detect, cut, resample
# ---------------------------------------------------------------------------
def cmd_detect(ctx) -> dict:
    from atk_diffusion import cards, sigmf
    from atk_diffusion.detect.pipeline import DetectorPipeline
    a = ctx.a
    cap = _capture(a.capture)
    pid = _capture_profile(cap, a.profile)
    rf = ctx.rf()
    proposer = _model_dir(rf, pid, a.proposer) if a.proposer else None
    if a.learned and proposer is None:
        found = cards.find(rf, pid, "proposer2d") if rf.root.is_dir() else []
        if not found:
            ctx.say(f"[--] --learned: there is no trained proposer for {pid} yet; "
                    "the energy (and cyclic) proposers run alone.")
        else:
            proposer = found[0][0]
            ctx.say(f"Learned proposer: {proposer.name} (the newest for {pid}).")
    classifier = _model_dir(rf, pid, a.classifier) if a.classifier else None
    bank = None
    if classifier is not None:
        try:
            from atk_diffusion.learn import calibrate
            bank, words = calibrate.load_bank(classifier, pid)
            ctx.say(f"Prototype bank: {words}.")
        except (FileNotFoundError, ImportError, ValueError) as e:
            ctx.say(f"[--] {e}")
    pipe = DetectorPipeline(pid, rf=rf if rf.root.is_dir() else None,
                            proposers={"cyclic": bool(a.cyclic),
                                       "learned": proposer is not None},
                            proposer2d=proposer, classifier=classifier,
                            prototypes=bank, cyclic_mode=a.cyclic_mode)
    res = pipe.run_on_capture(cap, write_annotations=bool(a.write_annotations),
                              chunk_seconds=float(a.chunk_seconds),
                              channel=int(a.channel), max_seconds=a.max_seconds,
                              progress=ctx.progress)
    dets = res["detections"]
    st = res["status"]
    for line in st.get("lines", []):
        ctx.say(line)
    ctx.say(f"{len(dets)} detections in {res['seconds']:.2f} s of {cap.name} "
            f"({res['wall_seconds']:.2f} s wall, {res['cpu_seconds']:.2f} s CPU). "
            "Every one is PROPOSED until a decoder confirms it.")
    for d in sorted(dets, key=lambda d: (d.t0, d.f_lo))[:int(a.list)]:
        what = d.cls or d.family
        snr = f", {d.snr_db:.1f} dB above the floor" if d.snr_db is not None else ""
        conf = f", confidence {d.confidence:.2f}" if d.confidence is not None else ""
        ctx.say(f"  {d.t0:8.3f}–{d.t1:8.3f} s  {_hz(d.f_lo)} – {_hz(d.f_hi)} "
                f"({_hz(d.bw_hz)} wide): {what} by {'+'.join(d.sources)}{snr}{conf}"
                f" [{d.state.upper()}]")
    if len(dets) > int(a.list):
        ctx.say(f"  (and {len(dets) - int(a.list)} more; --list N shows more, "
                "--json gives all)")
    if a.write_annotations:
        mp = sigmf.meta_path(cap)
        try:                     # keep the write log true for a capture under rf_data
            if rf.root.is_dir() and rf.root.resolve() in mp.resolve().parents:
                rf.record(mp, "annotations", "proposed detections written")
        except OSError:
            pass
        ctx.say(f"{res['annotations_written']} annotations now in {mp.name} "
                "(atk:source \"proposed\"; earlier proposed ones replaced, taught "
                "and confirmed labels untouched).")
    return {"capture": str(cap), "profile": pid, "seconds": res["seconds"],
            "wall_seconds": res["wall_seconds"], "cpu_seconds": res["cpu_seconds"],
            "detections": [d.to_json() for d in dets],
            "annotations_written": res["annotations_written"],
            "status": {k: v for k, v in st.items() if k != "last_tile"}}


def cmd_cut(ctx) -> dict:
    from atk_diffusion import provenance
    from atk_diffusion.cut.cut import cut_from_capture
    a = ctx.a
    cap = _capture(a.capture)
    pid = _capture_profile(cap, a.profile)
    rf = ctx.rf(create=True)
    box = {"t0_s": float(a.t0), "t1_s": float(a.t1),
           "f_lo_hz": float(a.f_lo), "f_hi_hz": float(a.f_hi)}
    cf = cut_from_capture(cap, box, rf, who=a.who or "", note=a.note or "",
                          margin_s=a.margin)
    an = cf.analysis
    c = an.get("cut", {}) or {}
    ctx.say(f"Cut {_hz(box['f_lo_hz'])} – {_hz(box['f_hi_hz'])}, "
            f"{box['t0_s']:g}–{box['t1_s']:g} s of {cap.name} ({_describe(pid)}) "
            f"into {cf.path}.")
    if c:
        ctx.say(f"  original: {c.get('samples', '?')} samples at the "
                f"{c.get('canonical_class', '?')} canonical rate "
                f"{_rate_words(c.get('canonical_rate_hz', 0))} (÷{c.get('decimation', '?')}), "
                "RECORD tier — it is never changed")
    out = {"cut": str(cf.path), "profile": pid}
    if not a.no_analyze:
        cf.analyze()
        an = cf.analysis
        meas = an.get("measurements") or {}
        ob = meas.get("occupied_bandwidth") or {}
        if ob.get("value_hz") is not None:
            ctx.say(f"  occupied bandwidth (99 %): {_hz(ob['value_hz'])} — MEASURED")
        sn = meas.get("snr") or {}
        if sn.get("snr_db") is not None:
            ctx.say(f"  SNR above the floor: {sn['snr_db']:.1f} dB — MEASURED")
        peaks = (an.get("cyclic") or {}).get("peaks") or []
        if peaks:
            ctx.say("  cyclic features at α = " + ", ".join(
                _hz(p["alpha_hz"]) + (" (conjugate)" if p.get("conj") else "")
                for p in peaks[:4]) + " (each above its derived threshold)")
        cls = an.get("class") or {}
        if cls:
            ctx.say(f"  class: {cls.get('cls', '?')}"
                    + (f" (confidence {cls['confidence']:.2f})"
                       if cls.get("confidence") is not None else "")
                    + (f" — {cls['why']}" if cls.get("why") else ""))
        out["analysis"] = str(Path(cf.path) / "analysis.json")
    if a.clean:
        rec = cf.clean(a.clean)
        tier = rec.get("tier", "")
        ctx.say(f"  clean '{a.clean}': " + provenance.TIER_WORDS.get(tier, tier))
        if rec.get("snr_before_db") is not None and rec.get("snr_after_db") is not None:
            ctx.say(f"    measured SNR {rec['snr_before_db']:.1f} dB before, "
                    f"{rec['snr_after_db']:.1f} dB after")
        if rec.get("words"):
            ctx.say(f"    {rec['words']}")
        out["clean"] = _json_safe(rec)
    rp = cf.report()
    ctx.say(f"  report: {rp}")
    try:
        routes = cf.routes_available()
        ctx.say(f"  can be routed to: {', '.join(routes)}")
        out["routes"] = routes
    except Exception:                                      # noqa: BLE001
        pass
    out["report"] = str(rp)
    return out


def cmd_resample(ctx) -> dict:
    from atk_diffusion import profiles, sigmf
    from atk_diffusion.dsp import resample
    a = ctx.a
    cap = _capture(a.capture)
    src = _capture_profile(cap)
    tgt = _profile_id(a.to)
    if tgt == src:
        raise Fail(f"{cap.name} is already {tgt}; nothing to resample")
    rf = ctx.rf(create=True)
    out = Path(a.out) if a.out else Path(rf.captures(tgt)) / f"{cap.name}_from_{src}"
    if sigmf.meta_path(out).exists() and not a.overwrite:
        raise Fail(f"{sigmf.meta_path(out)} already exists; --out names another, "
                   "or --overwrite replaces it")
    r = resample.resample_capture(cap, tgt, out, rf=rf, who=a.who or "",
                                  reason=a.reason or "")
    ctx.say(r["log"])
    ctx.say(f"Wrote {r['data']} ({r['samples']} samples at {_rate_words(r['fs'])}, "
            f"cf32, CLEANED tier, atk:resampled_from = {r['from']}). A model "
            "trained on it is a model of RESAMPLED data and says so; it is "
            f"logged in {Path(rf.runs(tgt)) / 'resample_log.txt'}.")
    if not r["exact"]:
        ctx.say("[--] the rate ratio is APPROXIMATE (not a small rational) — "
                "the log says so.")
    return r


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------
def _report_paths(result) -> list[str]:
    """Every report file an experiment says it wrote (the experiments name
    them differently: report{}, files[], report_md/report_json paths)."""
    out = []

    def add(v):
        if isinstance(v, str) and "\n" not in v and len(v) < 1000:
            try:
                if Path(v).is_file() and v not in out:
                    out.append(v)
            except OSError:
                pass

    if not isinstance(result, dict):
        return out
    rep = result.get("report")
    if isinstance(rep, dict):
        for v in rep.values():
            add(v)
    for k in ("report_md", "report_json", "report_detail_md", "json", "markdown"):
        add(result.get(k))
    for v in result.get("files", []) or []:
        add(v)
    return out


def _summary_from_reports(paths) -> list[str]:
    for p in paths:
        if p.endswith(".json"):
            try:
                d = json.loads(Path(p).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(d, dict) and isinstance(d.get("summary"), list):
                return [str(s) for s in d["summary"]]
    return []


def _exp_weak_burst(ctx, rf):
    from atk_diffusion.experiments import weak_burst as W
    a = ctx.a
    pid = _profile_id(a.profile)
    r = W.run(rf, pid, noise_capture=_capture(a.noise_capture) if a.noise_capture else None,
              bursts=tuple(_words(a.bursts)), snrs_db=tuple(_floats(a.snrs, "--snrs")),
              trials=int(a.trials), noise_trials=int(a.noise_trials), pfa=float(a.pfa),
              denoiser=_model_dir(rf, pid, a.denoiser) if a.denoiser else None,
              rows=int(a.rows), bins=int(a.bins), seed=int(a.seed),
              progress=ctx.progress)
    return r, W.report_lines(r)


def _exp_augment(ctx, rf):
    from atk_diffusion.experiments import augment_eval as AE
    a = ctx.a
    pid = _profile_id(a.profile)
    r = AE.run(rf, pid, augmenter=_model_dir(rf, pid, a.augmenter) if a.augmenter else None,
               kinds=tuple(_words(a.kinds)), window=int(a.window),
               n_synth_per=int(a.n_synth), n_real_per=int(a.n_real),
               n_test_per=int(a.n_test), tfd_steps=int(a.tfd_steps),
               classifier_steps=int(a.classifier_steps), seed=int(a.seed),
               progress=ctx.progress)
    return r, [r["verdict"], f"label-flip rate {r['label_flip_rate']:.3f}"]


def _exp_detector(ctx, rf):
    from atk_diffusion.experiments import detector_eval as DE
    a = ctx.a
    pid = _profile_id(a.profile)
    ds = lambda v: _dataset_dir(rf, pid, v) if v else None     # noqa: E731
    r = DE.evaluate_detector(
        rf, pid, proposer_dir=_model_dir(rf, pid, a.proposer) if a.proposer else None,
        classifier_dir=_model_dir(rf, pid, a.classifier) if a.classifier else None,
        synthetic_dataset=ds(a.synthetic_dataset), cabled_dataset=ds(a.cabled_dataset),
        noise_dataset=ds(a.noise_dataset), classifier_dataset=ds(a.classifier_dataset),
        unknown_dataset=ds(a.unknown_dataset), confirm_logs=a.confirm_logs or None,
        split=a.split, energy_threshold_db=float(a.energy_threshold_db),
        update_cards=not a.no_update_cards, progress=ctx.progress)
    return r, []


def _exp_domain_gap(ctx, rf):
    from atk_diffusion.experiments import domain_gap as DG
    a = ctx.a
    pid = _profile_id(a.profile)
    r = DG.domain_gap(rf, pid, _model_dir(rf, pid, a.model),
                      _dataset_dir(rf, pid, a.synthetic_dataset),
                      _dataset_dir(rf, pid, a.cabled_dataset), split=a.split,
                      cabled_split=a.cabled_split, update_card=not a.no_update_card,
                      progress=ctx.progress)
    return r, []


def _exp_minutes(ctx, rf):
    from atk_diffusion.experiments import minutes_to_acceptable as MA
    a = ctx.a
    pid = _profile_id(a.profile)
    r = MA.minutes_to_acceptable(
        rf, pid, _model_dir(rf, pid, a.model), _dataset_dir(rf, pid, a.onsite_dataset),
        acceptance=float(a.acceptance), mode=a.mode, metric=a.metric,
        minutes=_floats(a.minutes, "--minutes") if a.minutes else None,
        finetune_epochs=int(a.finetune_epochs), device=a.device,
        save_adapted=bool(a.save_adapted), update_card=not a.no_update_card,
        progress=ctx.progress)
    return r, []


def _exp_fingerprint(ctx, rf):
    from atk_diffusion.experiments import fingerprint_eval as FE
    a = ctx.a
    pid = _profile_id(a.profile)
    r = FE.run(rf, profile=pid, seed=int(a.seed), enrol_per_radio=int(a.enrol),
               test_per_radio=int(a.test), learned=not a.no_learned,
               denoiser=_model_dir(rf, pid, a.denoiser) if a.denoiser else None,
               cnn_steps=int(a.cnn_steps), progress=ctx.progress)
    lines = []
    cnn = (r.get("cnn") or {}).get("per_snr", {})
    for snr, row in ((r.get("classical") or {}).get("per_snr") or {}).items():
        c = cnn.get(snr) or {}
        lines.append(f"at {snr} dB SNR: classical features told the two radios apart "
                     f"{row['accuracy']:.0%} of the time and rejected the stranger "
                     f"{row['unknown_rejected']:.0%}"
                     + (f"; the CNN {c['accuracy']:.0%} and {c['unknown_rejected']:.0%}"
                        if c else ""))
    if "skipped" in (r.get("cnn") or {}):
        lines.append(f"CNN: {r['cnn']['skipped']}")
    lines.append("Chance is 50 % for two radios. A match is a PROPOSAL; only a "
                 "decode of the radio's own identity confirms it.")
    return r, lines


def _geo_report(ctx, rf, r, name):
    from atk_diffusion.experiments import geo_eval as G
    pid = _profile_id(ctx.a.profile)
    r["report"] = G.write_report(rf, r, profile=pid, name=name)
    return r


def _exp_e1(ctx, rf):
    from atk_diffusion.experiments import geo_eval as G
    a = ctx.a
    r = G.df_coverage(int(a.trials), seed=int(a.seed),
                      failure_trials=a.failure_trials, max_cells=int(a.max_cells),
                      progress=ctx.progress)
    lines = [r["words"]] + [
        f"{row['label']}: 90 % regions held the truth {row['rates']['0.9']:.0%} of "
        f"{row['trials']} times — " + ("ok" if row["passed"] else "FAIL")
        + (" (meant to fail)" if row["expect_failure"] else "") for row in r["rows"]]
    return _geo_report(ctx, rf, r, "e1_df_coverage"), lines


def _exp_e3(ctx, rf):
    from atk_diffusion.experiments import geo_eval as G
    from atk_diffusion.geo import whereami
    a = ctx.a
    given = [a.drive, a.calib, a.test]
    if any(given) and not all(given):
        raise Fail("give all three of --drive, --calib and --test (drive CSVs with "
                   "time, lat, lon and one column per feature), or none for the "
                   "synthetic drive world")
    recs = [whereami.records_from_csv(f) for f in given] if all(given) else [None] * 3
    r = G.whereami_experiment(*recs, seed=int(a.seed), learned=not a.no_learned,
                              mdn_epochs=int(a.mdn_epochs), progress=ctx.progress,
                              rf=rf, profile=_profile_id(a.profile))
    lines = []
    for m, res in r["methods"].items():
        if "median_error_m" in res:
            lines.append(f"{m}: median error {res['median_error_m']:.0f} m, "
                         f"90th percentile {res.get('p90_error_m', float('nan')):.0f} m")
        elif "skipped" in res:
            lines.append(f"{m}: skipped — {res['skipped']}")
    return _geo_report(ctx, rf, r, "e3_whereami"), lines


def _exp_e4(ctx, rf):
    from atk_diffusion.experiments import geo_eval as G
    a = ctx.a
    r = G.aperture_experiment(freq_hz=float(a.freq_hz), radius_m=float(a.radius_m),
                              seed=int(a.seed), sigma_deg=float(a.sigma_deg))
    lines = [f"moving aperture (self-calibrated): {r['dpd_self_calibrated_error_m']:.0f} m "
             f"from the tower; two-step {r['two_step_error_m']:.0f} m; parked bearing "
             f"off by {r['parked_bearing_error_deg']:.1f}° (no range from one place)",
             str(r.get("words", ""))]
    return _geo_report(ctx, rf, r, "e4_aperture"), lines


def _exp_e5(ctx, rf):
    from atk_diffusion.experiments import geo_eval as G
    a = ctx.a
    r = G.reach_residual_experiment(seed=int(a.seed), learned=not a.no_learned, rf=rf,
                                    model=a.model, learned_steps=int(a.learned_steps),
                                    learned_fields=int(a.learned_fields),
                                    profile=_profile_id(a.profile))
    lines = [f"residual at held-out drive points: {r['residual_before_rmse_db']:.1f} dB "
             f"RMS from physics alone, {r['residual_after_kriging_rmse_db']:.1f} dB "
             "after kriging"]
    if "residual_after_learned_rmse_db" in r:
        lines.append(f"after the learned correction: "
                     f"{r['residual_after_learned_rmse_db']:.1f} dB, hallucination "
                     f"rate {r['learned_hallucination_rate']:.2f} "
                     f"({r['learned_budget']['steps']} training steps)")
    elif r.get("learned"):
        lines.append(f"learned correction {r['learned']}")
    return _geo_report(ctx, rf, r, "e5_reach_residual"), lines


def _exp_hf(ctx, rf):
    from atk_diffusion.experiments import hf_eval as HE
    a = ctx.a
    spots = None
    if a.spots:
        from atk_diffusion.hf import wspr
        spots = []
        for f in a.spots:
            spots += wspr.parse_file(f)
        ctx.say(f"{len(spots)} WSPR spots from {len(a.spots)} file(s).")
    kiwis = HE.DEFAULT_KIWIS
    if a.kiwi:
        kiwis = []
        for k in a.kiwi:
            if ":" not in k:
                raise Fail(f"--kiwi is NAME:GRID (e.g. England:IO91), got {k!r}")
            name, grid = k.rsplit(":", 1)
            kiwis.append({"name": name, "grid": grid})
    r = HE.run(rf, spots, here_grid=a.here, kiwis=tuple(kiwis),
               bands=tuple(_words(a.bands)),
               hours=tuple(int(h) for h in _floats(a.hours, "--hours")),
               year=int(a.year), month=int(a.month), ssn=float(a.ssn),
               progress=ctx.progress)
    return r, [r.get("what", "")]


def _exp_hunter(ctx, rf):
    from atk_diffusion.experiments import hunter_eval as HU
    a = ctx.a
    pid = _profile_id(a.profile)
    r = HU.run(rf, pid, a.goal, duration_s=float(a.duration),
               seeds=tuple(int(s) for s in _floats(a.seeds, "--seeds")),
               scan_dwell_s=float(a.scan_dwell), progress=ctx.progress)
    lines = []
    for name, s in r["policies"].items():
        ff = s["fraction_found"]
        ttf = s["median_time_to_find_s"]
        lines.append(f"{name}: found {s['found']} of {s['truth']} bursts"
                     + (f" ({ff:.0%})" if ff is not None else "")
                     + (f", median {ttf:.1f} s to find" if ttf is not None else "")
                     + f", {s['false_marks_per_hour']:.1f} false marks an hour")
    lines.append(r["what_this_is"])
    return r, lines


def _exp_inpaint(ctx, rf):
    from atk_diffusion.experiments import inpaint_eval as IE
    a = ctx.a
    pid = _profile_id(a.profile)
    r = IE.run(rf, pid, inpainter=_model_dir(rf, pid, a.inpainter) if a.inpainter else None,
               canonical=a.canonical, trials=int(a.trials),
               gaps_ms=tuple(_floats(a.gaps_ms, "--gaps-ms")), snr_db=float(a.snr_db),
               silence_trials=int(a.silence_trials), seed=int(a.seed),
               progress=ctx.progress)
    return r, IE.report_lines(r)


def _exp_novelty(ctx, rf):
    from atk_diffusion.experiments import novelty_eval as NE
    a = ctx.a
    data = None
    if a.data:
        data = json.loads(Path(a.data).read_text(encoding="utf-8"))
    out = _shared_run(rf, "novelty_eval")
    r = NE.run(out, seed=int(a.seed), n_docs=int(a.n_docs), llama_model=a.llama_model,
               data=data, progress=ctx.progress)
    return r, NE.summary_lines(r)


def _exp_pulse(ctx, rf):
    from atk_diffusion.experiments import pulse_eval as PE
    a = ctx.a
    out = _shared_run(rf, "pulse_eval")
    r = PE.pulse_eval(seeds=range(int(a.seeds)), duration_s=float(a.duration),
                      mean_drop=float(a.mean_drop), noise_rate_hz=float(a.noise_rate),
                      out_dir=out, progress=ctx.progress)
    return r, [str(x) for x in r.get("lines", [])]


def _exp_translator(ctx, rf):
    from atk_diffusion.experiments import translator_eval as TE
    a = ctx.a
    fp, tp = _profile_id(a.from_profile), _profile_id(a.to_profile)
    r = TE.run(rf, fp, tp, translator=_model_dir(rf, tp, a.translator) if a.translator else None,
               window=int(a.window), n_pairs=int(a.n_pairs), n_test=int(a.n_test),
               train_steps=int(a.train_steps), seed=int(a.seed), progress=ctx.progress)
    return r, TE.report_lines(r)


def _exp_vitals(ctx, rf):
    from atk_diffusion.experiments import vitals_eval as VE
    a = ctx.a
    r = VE.run(rf, duration_s=float(a.duration), fs=float(a.fs), lstm=not a.no_lstm,
               progress=ctx.progress)
    c = r["classical"]
    lines = [r["label"],
             f"classical: breathing error {c['breath_mae_bpm']:.1f} per minute, heart "
             f"{c['heart_mae_bpm']:.1f} per minute (mean absolute)"
             if c["breath_mae_bpm"] is not None and c["heart_mae_bpm"] is not None
             else "classical: no rate measured",
             f"empty room: a breathing rate reported in "
             f"{r['empty_room']['breath_reported_fraction']:.0%} of windows "
             "(the hallucination rate)"]
    return r, lines


def _command_transcriber(template: str):
    """`--transcriber "whisper-cli.exe -m model.bin -nt -f {wav}"`: the
    command's standard output is the transcript."""
    import shlex
    import subprocess
    if "{wav}" not in template:
        raise Fail("--transcriber is a command with {wav} where the audio file "
                   "goes, e.g. \"C:\\whisper\\whisper-cli.exe -m ggml-base.en.bin "
                   "-nt -f {wav}\"")
    parts = shlex.split(template, posix=(os.name != "nt"))

    def run(wav):
        args = [p.replace("{wav}", str(wav)) for p in parts]
        p = subprocess.run(args, capture_output=True, text=True, timeout=600)
        if p.returncode != 0:
            raise RuntimeError(f"the transcriber ended with code {p.returncode}: "
                               f"{(p.stderr or '').strip()[-300:]}")
        return p.stdout.strip()
    return run


def _exp_wer(ctx, rf):
    from atk_diffusion.experiments import wer_eval as WE
    a = ctx.a
    clips = []
    for c in a.clips:
        p = Path(c)
        clips += sorted(p.glob("*.wav")) if p.is_dir() else [p]
    if not clips:
        raise Fail("--clips names WAV files or a folder of them")
    rp = Path(a.references)
    if rp.suffix.lower() == ".json":
        refs = json.loads(rp.read_text(encoding="utf-8"))
    else:
        lines = [ln for ln in rp.read_text(encoding="utf-8").splitlines() if ln.strip()]
        refs = {}
        for ln in lines:
            if "\t" not in ln:
                raise Fail(f"{rp.name}: each line is <clip name><TAB><what was "
                           "said> (or give a .json {clip: text})")
            k, v = ln.split("\t", 1)
            refs[k.strip()] = v.strip()
    noise = []
    for c in a.noise_clips or []:
        p = Path(c)
        noise += sorted(p.glob("*.wav")) if p.is_dir() else [p]
    transcribe = _command_transcriber(a.transcriber)
    out = Path(a.out) if a.out else _shared_run(rf, "wer_eval")
    r = WE.wer_with_without(clips, refs, transcribe, out_dir=out, noise_clips=noise,
                            progress=ctx.progress)
    return r, [str(x) for x in r.get("lines", [])]


#: name -> (handler, plan track, entry point, what it needs)
EXPERIMENTS = {
    "weak-burst": (_exp_weak_burst, "B3", "experiments.weak_burst.run",
                   "runs here on synthetic noise; your terminated capture "
                   "(--noise-capture) and a trained denoiser (GPU) for the real number"),
    "augment":    (_exp_augment, "B4", "experiments.augment_eval.run",
                   "trains a small TFD-lite on stand-ins; GPU for real sizes"),
    "detector":   (_exp_detector, "B1/B2", "experiments.detector_eval.evaluate_detector",
                   "trained proposer/classifier and datasets of the profile"),
    "domain-gap": (_exp_domain_gap, "§7", "experiments.domain_gap.domain_gap",
                   "a trained model, a synthetic and a CABLED dataset (bladeRF/HackRF loop)"),
    "minutes":    (_exp_minutes, "§3.6/§7",
                   "experiments.minutes_to_acceptable.minutes_to_acceptable",
                   "a model trained for the region and an on-site dataset"),
    "fingerprint": (_exp_fingerprint, "C1/C2", "experiments.fingerprint_eval.run",
                    "runs here on simulated same-model radios"),
    "e1-coverage": (_exp_e1, "E1", "experiments.geo_eval.df_coverage",
                    "runs here (simulated DF geometries)"),
    "e3-whereami": (_exp_e3, "E3", "experiments.geo_eval.whereami_experiment",
                    "synthetic drive world here; your drive CSVs with GPS truth"),
    "e4-aperture": (_exp_e4, "E4", "experiments.geo_eval.aperture_experiment",
                    "simulated drive here; Kraken snapshots on a driven loop for real"),
    "e5-reach":   (_exp_e5, "E5", "experiments.geo_eval.reach_residual_experiment",
                   "synthetic terrain and drive here; DTED1 and a measured drive for real"),
    "hf":         (_exp_hf, "J", "experiments.hf_eval.run",
                   "synthetic evening here; WSPR spot files from your Kiwis; voacapl optional"),
    "hunter":     (_exp_hunter, "B6", "experiments.hunter_eval.run",
                   "runs here against a simulated band"),
    "inpaint":    (_exp_inpaint, "D1", "experiments.inpaint_eval.run",
                   "classical fills here; a trained inpainter (GPU) for the learned row"),
    "novelty":    (_exp_novelty, "F1", "experiments.novelty_eval.run",
                   "planted corpus here; your documents (--data) and a GGUF model optional"),
    "pulse":      (_exp_pulse, "D3", "experiments.pulse_eval.pulse_eval",
                   "runs here on synthetic pulse trains"),
    "translator": (_exp_translator, "A6", "experiments.translator_eval.run",
                   "stand-in receivers here; paired cabled captures for real (GPU)"),
    "vitals":     (_exp_vitals, "I1", "experiments.vitals_eval.run",
                   "synthetic CSI here; the ESP32 pair (not in hand yet) for real"),
    "wer":        (_exp_wer, "D2", "experiments.wer_eval.wer_with_without",
                   "your clips, their transcripts, and a transcriber command (Whisper)"),
}


def cmd_experiment(ctx) -> dict:
    a = ctx.a
    if a.list or not a.name:
        ctx.say("Experiments — the plan's first experiments (§4), each a function "
                "that measures and writes a report. Run one with "
                "`atkdiff.bat experiment <name> [options]` "
                "(`atkdiff.bat experiment <name> --help` lists its options).")
        for name, (_h, track, entry, needs) in EXPERIMENTS.items():
            ctx.say(f"  {name:<12} {track:<8} {entry} — {needs}")
        return {"experiments": {n: {"track": v[1], "entry": v[2], "needs": v[3]}
                                for n, v in EXPERIMENTS.items()}}
    handler, track, entry, _needs = EXPERIMENTS[a.name]
    rf = ctx.rf(create=True)
    ctx.say(f"Experiment {a.name} (plan {track}, {entry}).")
    t0 = time.time()
    result, lines = handler(ctx, rf)
    paths = _report_paths(result)
    lines = [ln for ln in (lines or _summary_from_reports(paths)) if ln]
    for ln in lines:
        ctx.say(ln)
    ctx.say(f"Done in {time.time() - t0:.1f} s. The numbers are MEASURED (computed "
            "from data by a stated method); the report says on what.")
    for p in paths:
        ctx.say(f"  report: {p}")
    if not paths:
        ctx.say("  (this experiment wrote no report file)")
    return {"experiment": a.name, "track": track, "lines": lines,
            "reports": paths, "result": result}


# ---------------------------------------------------------------------------
# the cabled loop
# ---------------------------------------------------------------------------
def _loop_setup(ctx, receiver_id: str):
    from atk_diffusion.cabled import safety as S
    a = ctx.a
    tp = None
    if a.tx_power_dbm is not None:
        tp = S.TxPower(ref_gain=float(a.tx_power_at_gain), ref_dbm=float(a.tx_power_dbm),
                       source=a.tx_power_source or "", entered=a.tx_power_date or "",
                       db_per_step=float(a.db_per_step))
    return S.LoopSetup(transmitter=str(a.tx).lower(), receiver_profile=receiver_id,
                       frequency_hz=float(a.freq), tx_gain=float(a.tx_gain),
                       tx_power=tp, attenuation_db=a.attenuation_db,
                       cable_loss_db=float(a.cable_loss_db),
                       splitter_loss_db=a.splitter_loss_db,
                       splitter_ways=a.splitter_ways, cabled=bool(a.cabled),
                       dc_block=bool(a.dc_block), tx_gain_min=a.tx_gain_min,
                       tx_gain_max=a.tx_gain_max, tx_serial=a.tx_serial or "")


_LOOP_HINTS = (
    ("cable is not confirmed", "--cabled confirms the cable"),
    ("DC block is not confirmed", "--dc-block confirms the DC block"),
    ("maximum safe input", "`atkdiff.bat profile set-safe-input <receiver profile> "
                           "--max-dbm … --source \"<data sheet>\" --date YYYY-MM-DD`"),
    ("output power", "--tx-power-dbm with --tx-power-at-gain, --tx-power-source and "
                     "--tx-power-date"),
    ("attenuation in line", "--attenuation-db (the attenuators' total)"),
    ("splitter", "--splitter-loss-db (per output) and --splitter-ways"),
    ("minimum TX gain", "--tx-gain-min / --tx-gain-max (the bladeRF's range as its "
                        "tool reports it)"),
    ("first run of this setup", "--tx-gain <the minimum> for the first run"),
)


def _loop_hints(ctx, refusals) -> None:
    said = []
    for r in refusals:
        for key, hint in _LOOP_HINTS:
            if key in r and hint not in said:
                said.append(hint)
    if said:
        ctx.say("To clear them on this command line: " + "; ".join(said) + ".")


def cmd_cabled_check(ctx) -> dict:
    from atk_diffusion import profiles
    from atk_diffusion.cabled import safety as S
    a = ctx.a
    rx = _profile_id(a.rx_profile)
    rf = ctx.rf(create=True)
    receiver = profiles.load_profile(rf, rx)
    setup = _loop_setup(ctx, rx)
    ramp = S.Ramp.for_profile(rf, rx)
    v = S.check(setup, receiver, ramp)
    ctx.say(f"Cabled loop check: the {setup.transmitter} at TX gain "
            f"{setup.tx_gain:g} into {_describe(rx)} at {_hz(setup.frequency_hz)}.")
    for line in v.lines():
        ctx.say(line if not line.startswith("REFUSED") else f"[ERR] {line}")
    _loop_hints(ctx, v.refusals)
    ctx.say("Nothing was transmitted — `check` only does the arithmetic.")
    out = v.to_json()
    if not v.ok:
        raise Fail(f"{len(v.refusals)} refusal(s) — the loop will not run until "
                   "every one is cleared")
    return out


_PRESET = ("ref_bpsk", "ref_qpsk", "ref_2fsk", "ref_gfsk", "nfm_voice", "dmr",
           "p25", "pocsag")


def _signal_specs(a, fs_low: float) -> list[dict]:
    specs = []
    for s in a.signal or []:
        parts = [p.strip() for p in s.split(",")]
        try:
            spec = {"class": parts[0], "duration_s": float(parts[1]) if len(parts) > 1 else 0.1,
                    "f_offset_hz": float(parts[2]) if len(parts) > 2 else 0.0,
                    "power_db": float(parts[3]) if len(parts) > 3 else 0.0}
        except ValueError:
            raise Fail(f"--signal is CLASS[,seconds[,offset Hz[,power dB]]], e.g. "
                       f"ref_qpsk,0.1,200e3; got {s!r}") from None
        specs.append(spec)
    if not specs and a.preset:
        n = len(_PRESET)
        for k, cls in enumerate(_PRESET):
            off = (-0.3 + 0.6 * k / (n - 1)) * fs_low
            specs.append({"class": cls, "duration_s": 0.1, "f_offset_hz": round(off)})
    if not specs:
        raise Fail("name the signals to play: --signal CLASS[,seconds[,offset Hz"
                   "[,power dB]]] (repeat it), or --preset for the reference set ("
                   + ", ".join(_PRESET) + ")")
    return specs


def _txfile(ctx, rf, rx: str):
    from atk_diffusion import profiles
    from atk_diffusion.cabled import txfiles
    a = ctx.a
    rx_rate = float(profiles.parse_profile_id(rx).sample_rate)
    tx_rate = float(a.tx_rate or rx_rate)
    specs = _signal_specs(a, min(tx_rate, rx_rate))
    tf = txfiles.build(Path(rf.cabled(rx)) / "tx", a.tx, tx_rate, specs,
                       rx_rate=rx_rate, marker=a.marker, name=a.name or "",
                       seed=int(a.seed), rf=rf,
                       tx_center_hz=float(a.freq) if getattr(a, "freq", None) else None)
    return tf


def cmd_cabled_txfile(ctx) -> dict:
    a = ctx.a
    rx = _profile_id(a.rx_profile)
    rf = ctx.rf(create=True)
    tf = _txfile(ctx, rf, rx)
    for line in tf.lines():
        ctx.say(line)
    for s in tf.manifest["signals"]:
        bw = s.get("bandwidth_hz")
        ctx.say(f"  {s['label']}: {s['duration_s'] * 1e3:.0f} ms at "
                f"{_hz(s['f_offset_hz'])} from the TX centre"
                + (f", {_hz(bw)} wide" if bw else "") + f" ({s['generator']})")
    ctx.say(f"Transmit file {tf.path} and its ground-truth manifest "
            f"{tf.manifest_path}. Nothing was transmitted.")
    return {"file": str(tf.path), "manifest": str(tf.manifest_path),
            "manifest_data": tf.manifest}


def cmd_cabled_plan(ctx) -> dict:
    from atk_diffusion import profiles
    from atk_diffusion.cabled import loop
    from atk_diffusion.cabled import safety as S
    a = ctx.a
    rx = _profile_id(a.rx_profile)
    rf = ctx.rf(create=True)
    receiver = profiles.load_profile(rf, rx)
    setup = _loop_setup(ctx, rx)
    if a.tx_file:
        manifest = Path(a.tx_file)
        if not manifest.name.endswith(".manifest.json"):
            cand = manifest.with_name(manifest.stem + ".manifest.json")
            manifest = cand if cand.exists() else manifest
        if not manifest.exists():
            raise Fail(f"no transmit manifest at {manifest}")
    else:
        tf = _txfile(ctx, rf, rx)
        manifest = tf.manifest_path
        ctx.say(f"Built the transmit file {tf.path.name} ({len(tf.manifest['signals'])} "
                "signals).")
    ramp = S.Ramp.for_profile(rf, rx)
    plan = loop.plan_run(setup, receiver, manifest, ramp=ramp,
                         bandwidth_hz=a.bandwidth_hz)
    for line in plan.lines():
        ctx.say(line if not line.startswith("REFUSED") else f"[ERR] {line}")
    _loop_hints(ctx, list(plan.verdict.refusals) + list(plan.refusals))
    out = plan.to_json()
    if not a.execute:
        ctx.say("Planned only — NOTHING WAS TRANSMITTED. To transmit, connect the "
                "transmitter by cable through the attenuators and a DC block to "
                "the receiver (never an antenna) and run this again with "
                "--execute --i-confirm-cabled-with-attenuators.")
        if not plan.runnable:
            raise Fail("the plan is not runnable until every refusal is cleared")
        return out
    if not a.i_confirm_cabled_with_attenuators:
        raise Fail("Refused: --execute needs --i-confirm-cabled-with-attenuators "
                   "too — your confirmation, now, that the transmitter is connected "
                   "by cable through the attenuators and a DC block to the "
                   "receiver. Nothing was transmitted.")
    res = loop.execute(plan, confirm_cabled=True, runner=loop.subprocess_runner,
                       dry_run=False, ramp=ramp)
    ctx.say(res["why"] if res["ran"] else f"[ERR] {res['why']}")
    if res.get("output"):
        ctx.say(f"  the tool said: {res['output'].strip()[-600:]}")
    out["execute"] = res
    if not res["ran"] or res.get("returncode") not in (0,):
        raise Fail("the run did not complete" if res["ran"] else
                   "nothing was transmitted")
    ctx.say("Now record the receiver's level for the ramp: `atkdiff.bat cabled "
            "measure <the recording> --tx-file <manifest> …` with the same setup.")
    return out


def cmd_cabled_measure(ctx) -> dict:
    from atk_diffusion import profiles
    from atk_diffusion.cabled import loop
    from atk_diffusion.cabled import safety as S
    a = ctx.a
    cap = _capture(a.capture)
    rx = _capture_profile(cap, a.rx_profile)
    rf = ctx.rf(create=True)
    setup = _loop_setup(ctx, rx)
    r = loop.measure_rx(cap, a.tx_file, f_tx_hz=float(a.freq))
    ctx.say(f"{cap.name}: {r['words']}.")
    out = {"measurement": r, "recorded": False}
    if r["level_dbfs"] is None:
        raise Fail("no level was read, so nothing was recorded for the ramp")
    receiver = profiles.load_profile(rf, rx)
    v = S.check(setup, receiver)
    ramp = S.Ramp.for_profile(rf, rx)
    st = ramp.record(setup, setup.tx_gain, r["level_dbfs"], r["clipped_fraction"],
                     noise_dbfs=r["noise_dbfs"], rx_full_scale_dbm=a.rx_full_scale_dbm,
                     expected_input_dbm=v.expected_input_dbm)
    ctx.say(f"Recorded for the ramp at TX gain {st.gain:g}: {st.measured_dbfs:.1f} dBFS"
            + (f", {st.above_noise_db:.1f} dB above the TX-off noise"
               if st.above_noise_db is not None else "")
            + f", {100 * st.clipped_fraction:.3g} % clipped. The next `cabled check` "
            "says how far the gain may go up.")
    out["recorded"] = True
    return out


def cmd_cabled_align(ctx) -> dict:
    from atk_diffusion import profiles
    from atk_diffusion.cabled import loop
    from atk_diffusion.cabled import safety as S
    a = ctx.a
    cap = _capture(a.capture)
    rx = _capture_profile(cap, a.rx_profile)
    rf = ctx.rf(create=True)
    setup = _loop_setup(ctx, rx)
    v = S.check(setup, profiles.load_profile(rf, rx))
    res = loop.align_labels(cap, a.tx_file, rf=rf, setup=setup, verdict=v,
                            f_tx_hz=float(a.freq))
    for line in res.lines():
        ctx.say(line)
    if not res.found:
        raise Fail(res.why or "the marker was not found; no labels were written")
    ctx.say(f"Next: `atkdiff.bat synth cabled --profile {rx} --name <set> "
            f"{res.capture or cap}` turns the labelled recording into the cabled "
            "(domain-gap) dataset.")
    return {"found": True, "labels": res.labels, "capture": res.capture,
            "start_sample": res.start_sample, "cfo_hz": res.cfo_hz,
            "drift_ppm": res.drift_ppm}


# ---------------------------------------------------------------------------
# products, hunt, vitals, repair
# ---------------------------------------------------------------------------
def cmd_products_list(ctx) -> dict:
    from atk_diffusion.geo import products as P
    rf = ctx.rf()
    kinds = [ctx.a.kind] if ctx.a.kind else list(P.PRODUCT_KINDS)
    out = []
    for k in kinds:
        if k not in P.PRODUCT_KINDS:
            raise Fail(f"{k!r} is not a product kind — one of "
                       + ", ".join(P.PRODUCT_KINDS))
        for d, man in P.list_runs(rf, k) if rf.root.is_dir() else []:
            files = man.get("files") or {}
            line = (f"{k}/{d.name}: {man.get('description') or man.get('method') or ''} "
                    f"— {len(files)} files, tier {man.get('tier', '?')}, created "
                    f"{man.get('created', '?')}").replace(":  —", ": —")
            row = {"kind": k, "run": d.name, "path": str(d),
                   "tier": man.get("tier"), "files": len(files)}
            if ctx.a.verify:
                ok, problems = P.verify_run(d)
                line += " — verified" if ok else " — " + "; ".join(problems[:3])
                row["ok"] = ok
            ctx.say(line)
            out.append(row)
    if not out:
        ctx.say(f"No products yet under {rf.products()} (maps, tracks, coverage "
                "and HF matrices land there).")
    return {"products": out}


def cmd_products_import(ctx) -> dict:
    from atk_diffusion.geo import products as P
    rf = ctx.rf(create=True)
    dest, words = P.import_run(ctx.a.folder, rf, kind=ctx.a.kind)
    ctx.say(words)
    return {"imported": str(dest), "words": words}


def cmd_products_reach(ctx) -> dict:
    """Plan E5 on real terrain: the predicted reach of one radio over DTED,
    written as a product the Geospatial map's Layers tab shows. Physics
    only (the INFERRED layer); a measured drive and the learned residual
    are added by `geo.reach.add_measurement_layer` / `add_correction_layer`
    (and the e5-reach experiment). Needs no training."""
    import json as _json
    from atk_diffusion.geo import antenna as A
    from atk_diffusion.geo import dted as D
    from atk_diffusion.geo import propagation as PR
    from atk_diffusion.geo import reach as R
    from atk_diffusion.geo import terrain as T
    a = ctx.a
    rf = ctx.rf(create=True)
    if a.model not in PR.MODELS:
        raise Fail(f"--model {a.model!r} is not a propagation model — one of "
                   + ", ".join(PR.MODELS))
    if a.flat:
        terrain = T.FlatTerrain(float(a.flat_height))
        if a.model in ("itm", "bullington", "deygout"):
            ctx.say("Note: flat ground was asked for, so the terrain models see "
                    "no hills — this is the sanity line, not a reach map.")
    else:
        folder = Path(a.dted) if a.dted else rf.shared() / "dted"
        if not folder.is_dir():
            raise Fail(f"no DTED folder at {folder}. Put DTED level 1 tiles "
                       "(the usual w078\\n38.dt1 tree, from NGA) there, or name "
                       "the folder with --dted; --flat runs without terrain.")
        terrain = D.DtedMosaic.from_folder(folder, progress=None)
        ctx.say(f"Terrain: {terrain.describe()}")
        for s in terrain.skipped[:5]:
            ctx.say(f"  skipped {s}")
        z = terrain.elevation(float(a.lat), float(a.lon))
        if not np_isfinite(z):
            raise Fail(f"the DTED under {folder} does not cover "
                       f"{float(a.lat):.4f}, {float(a.lon):.4f} (or the post "
                       "there is void) — add the tile that holds the transmitter.")
    if a.antenna_file:
        try:
            ant = A.from_json(_json.loads(Path(a.antenna_file).read_text("utf-8")))
        except (OSError, ValueError, KeyError) as e:
            raise Fail(f"{a.antenna_file} is not an antenna pattern: {e}") from None
    elif a.antenna == "isotropic":
        ant = A.Isotropic(float(a.gain_dbi or 0.0))
    elif a.antenna == "directional":
        if a.gain_dbi is None or a.beamwidth is None:
            raise Fail("a directional antenna needs --gain-dbi and --beamwidth "
                       "(the -3 dB azimuth beamwidth, degrees)")
        ant = A.Directional(float(a.gain_dbi), float(a.beamwidth))
    else:
        ant = (A.Dipole("vertical", float(a.gain_dbi)) if a.gain_dbi is not None
               else A.Dipole("vertical"))
    if (a.power_w is None) == (a.power_dbm is None):
        raise Fail("give the transmit power once: --power-w or --power-dbm")
    kw = dict(height_agl_m=float(a.height), antenna=ant,
              azimuth_deg=float(a.azimuth), tilt_deg=float(a.tilt),
              line_loss_db=float(a.line_loss), name=a.label or "")
    tx = (R.Transmitter.watts(float(a.lat), float(a.lon), float(a.freq),
                              float(a.power_w), **kw)
          if a.power_w is not None else
          R.Transmitter(float(a.lat), float(a.lon), float(a.freq),
                        float(a.power_dbm), **kw))
    rx = R.Receiver(height_agl_m=float(a.rx_height),
                    sensitivity_dbm=float(a.sensitivity_dbm),
                    fade_margin_db=float(a.fade_margin))
    res = R.predicted_reach(tx, rx, terrain, float(a.radius_km), model=a.model,
                            grid_m=float(a.grid_m), rf=rf, run=a.name or None,
                            label=a.label or "", ground=a.ground,
                            progress=ctx.progress)
    ctx.say(res.words())
    for n in res.notes[:8]:
        ctx.say(f"  {n}")
    ctx.say(f"Written to {res.run_dir} — tier {res.tier.upper()}: a prediction "
            "for planning, never a promise of contact. It is on the Geospatial "
            "map's Layers tab in ATK.")
    return {"run_dir": res.run_dir, "tier": res.tier, "model": res.model,
            "summary": res.summary}


def np_isfinite(z) -> bool:
    import numpy as np
    return bool(np.all(np.isfinite(np.asarray(z, dtype=float))))


def cmd_hunt_simulate(ctx) -> dict:
    import numpy as np
    from atk_diffusion.experiments.hunter_eval import score
    from atk_diffusion.hunt.goal import parse_goal
    from atk_diffusion.hunt.policy import (FixedScanPolicy, HuntLog,
                                           ReceiverLimits, RulePolicy, run_hunt)
    from atk_diffusion.hunt.sim import SimReceiver, scripted_band
    a = ctx.a
    pid = _profile_id(a.profile)
    goal = parse_goal(a.goal)
    probs = goal.problems()
    if probs:
        raise Fail("the goal cannot be hunted: " + "; ".join(probs))
    ctx.say(f"Goal: {goal.describe()}")
    for n in goal.notes:
        ctx.say(f"  {n}")
    rf = ctx.rf(create=True)
    limits = ReceiverLimits.for_profile(pid)
    band = scripted_band(goal.f_lo_hz, goal.f_hi_hz, float(a.duration),
                         rng=np.random.default_rng(int(a.seed)))
    truth = band.goal_truth(goal)
    policy = RulePolicy() if a.policy == "hunter" else FixedScanPolicy(float(a.dwell))
    rx = SimReceiver(band, limits, rng=np.random.default_rng(int(a.seed) + 1000),
                     profile=pid)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    log = HuntLog.for_run(rf, pid, run_id=f"sim-{stamp}-{a.policy}-seed{int(a.seed)}",
                          fsync=False)
    ctx.say(f"Hunting a simulated {_hz(goal.f_hi_hz - goal.f_lo_hz)} band for "
            f"{float(a.duration):g} s with the {a.policy} policy (every retune and "
            "its reason goes to the hunt log) …")
    res, st = run_hunt(goal, rx, limits, policy, log, duration_s=float(a.duration))
    sc = score(st, truth, float(a.duration))
    for line in res.lines():
        ctx.say(line)
    ttf = sc["time_to_find_s"]
    ctx.say(f"Simulated band: the {a.policy} found {sc['found']} of {sc['truth']} "
            f"bursts the goal is about"
            + (f" (median {float(np.median(ttf)):.1f} s after each began)" if ttf else "")
            + f", with {sc['false_marks']} false marks. It never transmits.")
    ctx.say("This is a SIMULATION under a stated detection curve (hunt.sim) — a "
            "comparison of policies, not a prediction of the field.")
    return {"goal": goal.to_json(), "policy": a.policy, "score": sc,
            "log": str(log.path), "retunes": res.retunes}


def cmd_vitals_replay(ctx) -> dict:
    from atk_diffusion.experiments.vitals_eval import classical, profile_for
    from atk_diffusion.sensing import csi
    a = ctx.a
    p = Path(a.log)
    if not p.exists():
        raise Fail(f"{p} does not exist (a saved ESP32 CSI console log)")
    parser = csi.CsiParser()
    frames = list(csi.replay(p, parser))
    if len(frames) < 2:
        raise Fail(f"{p.name} holds {len(frames)} CSI frame(s) the parser could "
                   "read; it needs a recording of a minute or more")
    t, H, info = csi.to_matrix(frames, mac=a.mac or None)
    ctx.say(f"{info['frames']} frames from {info['mac']} ({info['subcarriers']} "
            f"subcarriers, {info['clock']} clock) over {float(t[-1]):.1f} s"
            + (f"; {info['dropped_other_format']} frames of another format left out"
               if info["dropped_other_format"] else "") + ".")
    rep = classical(t, H, float(a.fs))
    for line in rep.lines():
        ctx.say(line)
    out = {"frames": info, "report": rep.to_json()}
    if ctx.root().is_dir() or a.save:
        rf = ctx.rf(create=True)
        d = Path(rf.runs(profile_for(float(a.fs)))) / \
            f"vitals_replay_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        d.mkdir(parents=True, exist_ok=True)
        jp = d / "result.json"
        jp.write_text(json.dumps(_json_safe(out), indent=2), encoding="utf-8")
        rf.record(jp, "vitals", p.name)
        ctx.say(f"  report: {jp}")
        out["report_file"] = str(jp)
    return out


def cmd_repair_audio(ctx) -> dict:
    from atk_diffusion.repair import audio_inpaint
    a = ctx.a
    src = Path(a.wav)
    if not src.exists():
        raise Fail(f"{src} does not exist")
    out = Path(a.out) if a.out else src.with_name(src.stem + "_filled.wav")
    if out.resolve() == src.resolve():
        raise Fail("--out must not be the input: the original is never changed")
    gaps = None
    if a.gap:
        gaps = []
        for g in a.gap:
            v = _floats(g, "--gap")
            if len(v) != 2:
                raise Fail("--gap is START,END in seconds, e.g. 12.30,12.34")
            gaps.append((v[0], v[1]))
    side = audio_inpaint.inpaint_wav(src, out, method=a.method, gaps=gaps)
    for line in side["lines"]:
        ctx.say(line)
    for s in side["filled"][:20]:
        ctx.say(f"  {s['start'] / side['rate']:.3f} s, {s['seconds'] * 1e3:.1f} ms, "
                f"{s['method']} ({s['tier'].upper()})" + (f" — {s['note']}" if s["note"] else ""))
    ctx.say(f"Wrote {out} and its sidecar {side['sidecar']} (the list of every "
            "inferred sample). The original is unchanged.")
    return side


# ---------------------------------------------------------------------------
# the parser
# ---------------------------------------------------------------------------
class _Parser(argparse.ArgumentParser):
    """argparse, with its errors said in words and exit code 2."""

    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"[ERR] {message}. Add --help to that command to see what it takes.",
              file=sys.stderr)
        raise SystemExit(EXIT_USAGE)


def _common() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--rf-data", default=argparse.SUPPRESS, metavar="PATH",
                   help="the rf_data folder (else ATK_RF_DATA, else beside the code)")
    p.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                   help="print the result as JSON on stdout (words go to stderr)")
    p.add_argument("--debug", action="store_true", default=argparse.SUPPRESS,
                   help="print the full traceback of an unexpected error")
    return p


def _add_loop_args(p, rx_required: bool = True) -> None:
    g = p.add_argument_group("the cabled loop setup (all entered from data sheets "
                             "or measurements, with their source and date)")
    g.add_argument("--tx", required=True, help="the transmitter: hackrf, bladerf1 or bladerf2")
    g.add_argument("--rx-profile", required=rx_required,
                   help="the receiver under test's profile, e.g. rtlsdr_2400000_cu8"
                        + ("" if rx_required else " (default: the capture's)"))
    g.add_argument("--freq", type=float, required=True, help="centre frequency, Hz")
    g.add_argument("--tx-gain", type=float, default=0.0, help="TX gain setting for this run")
    g.add_argument("--tx-power-dbm", type=float, default=None,
                   help="the transmitter's output (dBm) at --tx-power-at-gain, at this "
                        "frequency, from its data sheet or a measurement")
    g.add_argument("--tx-power-at-gain", type=float, default=0.0,
                   help="the gain setting --tx-power-dbm was measured at")
    g.add_argument("--tx-power-source", default="", help="where --tx-power-dbm came from")
    g.add_argument("--tx-power-date", default="", help="when it was entered (YYYY-MM-DD)")
    g.add_argument("--db-per-step", type=float, default=1.0, help="dB per TX gain step")
    g.add_argument("--attenuation-db", type=float, default=None,
                   help="total fixed attenuation in line, dB")
    g.add_argument("--cable-loss-db", type=float, default=0.0)
    g.add_argument("--splitter-loss-db", type=float, default=None,
                   help="splitter loss per output (required for the KrakenSDR)")
    g.add_argument("--splitter-ways", type=int, default=None)
    g.add_argument("--tx-gain-min", type=float, default=None)
    g.add_argument("--tx-gain-max", type=float, default=None)
    g.add_argument("--tx-serial", default="")
    g.add_argument("--cabled", action="store_true",
                   help="CONFIRM: the transmitter is connected by cable, through the "
                        "attenuators, to the receiver — never an antenna")
    g.add_argument("--dc-block", action="store_true",
                   help="CONFIRM: a DC block is in line between the radios")


def _add_txfile_args(p) -> None:
    g = p.add_argument_group("the transmit file")
    g.add_argument("--tx-rate", type=float, default=None,
                   help="TX sample rate (default: the receiver's rate)")
    g.add_argument("--signal", action="append",
                   help="CLASS[,seconds[,offset Hz[,power dB]]], repeatable "
                        "(class names from the class table, e.g. ref_qpsk, dmr)")
    g.add_argument("--preset", action="store_true",
                   help="the reference set: " + ", ".join(_PRESET))
    g.add_argument("--marker", default="chirp", choices=("chirp", "pn"))
    g.add_argument("--name", default="", help="the file's name (default: dated)")
    g.add_argument("--seed", type=int, default=0)


def _add_train_parsers(sub, common) -> None:
    tp = sub.add_parser("train", parents=[common], help="train a model (needs PyTorch)",
                        description="Train one of the toolkit's models. Each kind "
                        "saves its weights WITH a model card (a model without a card "
                        "does not load). Kinds: " + "; ".join(
                            f"{k} — {v}" for k, v in TRAIN_KINDS.items()))
    ks = tp.add_subparsers(dest="kind", metavar="KIND", required=True)

    def kind(name, profile=True, name_default=None):
        p = ks.add_parser(name, parents=[common], help=TRAIN_KINDS[name],
                          description=f"Train {TRAIN_KINDS[name]}.")
        if profile:
            p.add_argument("--profile", required=True, help="receiver profile id")
        p.add_argument("--name", default=name_default or name,
                       help="the model folder's name")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--threads", type=int, default=None,
                       help="CPU threads for PyTorch (default: all but one)")
        return p

    def dev(p):
        p.add_argument("--device", default="auto",
                       help="auto (the GPU when there is one), cuda or cpu")

    p = kind("proposer")
    p.add_argument("--dataset", required=True, help="a wideband dataset (name or folder)")
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--width", type=int, default=16, help="backbone width")
    p.add_argument("--max-tiles", type=int, default=None, help="cap the training tiles")
    p.add_argument("--pretrained", default=None, help="an ssl backbone model to start from")
    p.add_argument("--resume", default=None, help="a checkpoint to resume from")
    p.add_argument("--no-amp", action="store_true", help="no mixed precision on the GPU")
    p.add_argument("--overwrite", action="store_true")
    dev(p)

    p = kind("classifier")
    p.add_argument("--dataset", required=True, help="a narrowband dataset (name or folder)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--rf-config", default="search",
                   help="'search' the receptive fields first (default) or a config name")
    p.add_argument("--search-epochs", type=int, default=3)
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--no-scf", action="store_true", help="the IQ branch only")
    p.add_argument("--held-out", default="", help="classes kept out of training "
                   "(comma list) to measure unknown rejection")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--pretrained", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    dev(p)

    p = kind("ssl")
    p.add_argument("--mode", choices=("2d", "1d"), default="2d",
                   help="2d: the proposer's backbone on tiles; 1d: the classifier's on cuts")
    p.add_argument("--dataset", default=None)
    p.add_argument("--captures", nargs="*", default=None, help="raw SigMF captures")
    p.add_argument("--canonical", default="voice", help="1d: the cut class")
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-tiles", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    dev(p)

    p = kind("calibrate")
    p.add_argument("--model", required=True, help="the classifier or proposer to calibrate")
    p.add_argument("--dataset", required=True, help="a held-out dataset of the profile")
    p.add_argument("--split", default="val")

    p = kind("denoiser")
    p.add_argument("--domain", choices=("spectrogram", "iq"), default="spectrogram")
    p.add_argument("--noise-capture", nargs="*", default=None,
                   help="terminated or quiet captures of THIS profile (the real noise)")
    p.add_argument("--captures", nargs="*", default=None,
                   help="your own captures as ambient examples (unlabelled)")
    p.add_argument("--synthetic", type=int, default=4000, help="synthetic examples")
    p.add_argument("--canonical", default=None, help="iq domain: a canonical class")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default=None)

    p = kind("inpainter")
    p.add_argument("--canonical", default="voice")
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--synthetic", type=int, default=4000)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default=None)

    p = kind("translator")
    p.add_argument("--from-capture", required=True,
                   help="receiver A's recording of the cabled loop")
    p.add_argument("--to-capture", required=True,
                   help="receiver B's recording of the SAME transmission")
    p.add_argument("--from-profile", default=None,
                   help="A's profile when it was resampled to B's rate")
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default=None)

    p = kind("augmenter")
    p.add_argument("--captures", nargs="+", required=True,
                   help="cabled recordings with their atk:source=cabled labels")
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default=None)

    p = kind("fingerprint")
    p.add_argument("--bursts", default=None,
                   help="an .npz with 'bursts' [N, L] complex, 'labels' [N] (which "
                        "radio), optional 'fs' and 'class_names'")
    p.add_argument("--synthetic", type=int, default=40,
                   help="without --bursts: bursts per simulated radio")
    p.add_argument("--fs", type=float, default=48_000.0)
    p.add_argument("--steps", type=int, default=400)

    p = kind("radiomap", profile=False)
    p.add_argument("--fields", default=None, help="an .npz of fields "
                   "(learn.radiomap.synthetic_fields' arrays)")
    p.add_argument("--synthetic", type=int, default=160, help="synthetic patches")
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=16)

    p = kind("position", profile=False)
    p.add_argument("--drive", nargs="*", default=None,
                   help="drive CSVs (time, lat, lon, one column per feature in dB)")
    p.add_argument("--calib", default=None, help="a held-out drive CSV for calibration")
    p.add_argument("--synthetic", type=int, default=400,
                   help="without --drive: points per synthetic drive")
    p.add_argument("--epochs", type=int, default=150)

    p = kind("vitals", profile=False)
    p.add_argument("--data", default=None, help="an .npz with X and y")
    p.add_argument("--synthetic", type=int, default=24, help="synthetic windows")
    p.add_argument("--fs", type=float, default=50.0, help="CSI frames a second")
    p.add_argument("--epochs", type=int, default=60)

    p = kind("genclass")
    p.add_argument("--captures", nargs="+", required=True,
                   help="captures with class labels (cabled, taught or confirmed)")
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default=None)

    p = kind("beacon")
    p.add_argument("--k", type=int, default=4, help="bits per block")
    p.add_argument("--n", type=int, default=8, help="channel uses per block")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--eval-blocks", type=int, default=2000)

    p = kind("anomaly", profile=False)
    p.add_argument("--flows", required=True, help="baseline flows: JSON list or JSON lines")
    p.add_argument("--model", choices=("pca", "autoencoder"), default="pca")
    p.add_argument("--components", type=int, default=3)


def _add_experiment_parsers(sub, common) -> None:
    ep = sub.add_parser("experiment", parents=[common],
                        help="run one of the plan's first experiments",
                        description="The plan's first experiments (§4). "
                        "`experiment --list` lists them with what each needs.")
    ep.add_argument("--list", action="store_true", help="list the experiments")
    es = ep.add_subparsers(dest="name", metavar="NAME")

    def exp(name, profile_default=None):
        h = EXPERIMENTS[name]
        p = es.add_parser(name, parents=[common], help=f"{h[1]}: {h[3]}",
                          description=f"plan {h[1]} — {h[2]}. Needs: {h[3]}.")
        p.set_defaults(list=False)
        if profile_default is not None:
            p.add_argument("--profile", default=profile_default,
                           help=f"receiver profile (default {profile_default})")
        p.add_argument("--seed", type=int, default=0)
        return p

    p = exp("weak-burst", "rtlsdr_2400000_cu8")
    p.add_argument("--noise-capture", default=None, help="a terminated capture of the profile")
    p.add_argument("--bursts", default="mfsk8,nfm_keyup,pocsag,adsb,lora")
    p.add_argument("--snrs", default="-15,-12,-9,-6,-3,0,3,6", help="in-band SNRs, dB")
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--noise-trials", type=int, default=200)
    p.add_argument("--pfa", type=float, default=0.01, help="tile false-alarm rate")
    p.add_argument("--denoiser", default=None, help="a trained B3 denoiser")
    p.add_argument("--rows", type=int, default=64)
    p.add_argument("--bins", type=int, default=128)

    p = exp("augment", "rtlsdr_2400000_cu8")
    p.add_argument("--augmenter", default=None, help="a trained augmenter (else one is trained)")
    p.add_argument("--kinds", default="bpsk,qpsk,gfsk,ofdm")
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--n-synth", type=int, default=200)
    p.add_argument("--n-real", type=int, default=8)
    p.add_argument("--n-test", type=int, default=100)
    p.add_argument("--tfd-steps", type=int, default=3000)
    p.add_argument("--classifier-steps", type=int, default=400)

    p = exp("detector", "rtlsdr_2400000_cu8")
    for k in ("proposer", "classifier", "synthetic-dataset", "cabled-dataset",
              "noise-dataset", "classifier-dataset", "unknown-dataset"):
        p.add_argument(f"--{k}", default=None)
    p.add_argument("--confirm-logs", nargs="*", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--energy-threshold-db", type=float, default=6.0)
    p.add_argument("--no-update-cards", action="store_true")

    p = exp("domain-gap", "rtlsdr_2400000_cu8")
    p.add_argument("--model", required=True)
    p.add_argument("--synthetic-dataset", required=True)
    p.add_argument("--cabled-dataset", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--cabled-split", default="all")
    p.add_argument("--no-update-card", action="store_true")

    p = exp("minutes", "rtlsdr_2400000_cu8")
    p.add_argument("--model", required=True, help="the model trained for the region")
    p.add_argument("--onsite-dataset", required=True, help="the first minutes on site")
    p.add_argument("--acceptance", type=float, required=True,
                   help="the acceptance line on the metric (e.g. 0.6 mAP)")
    p.add_argument("--mode", default=None, help="thresholds | prototypes | finetune")
    p.add_argument("--metric", default=None)
    p.add_argument("--minutes", default=None, help="comma list of minutes to try")
    p.add_argument("--finetune-epochs", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--save-adapted", action="store_true")
    p.add_argument("--no-update-card", action="store_true")

    p = exp("fingerprint", "hackrf_8000000_ci8")
    p.add_argument("--enrol", type=int, default=15, help="bursts per radio to enrol")
    p.add_argument("--test", type=int, default=25, help="bursts per radio to test")
    p.add_argument("--no-learned", action="store_true", help="classical only")
    p.add_argument("--denoiser", default=None)
    p.add_argument("--cnn-steps", type=int, default=300)

    p = exp("e1-coverage", "krakensdr_2400000_cu8")
    p.add_argument("--trials", type=int, default=300)
    p.add_argument("--failure-trials", type=int, default=None)
    p.add_argument("--max-cells", type=int, default=40_000)

    p = exp("e3-whereami", "krakensdr_2400000_cu8")
    p.add_argument("--drive", default=None, help="drive CSV for the database")
    p.add_argument("--calib", default=None, help="a held-out route for calibration")
    p.add_argument("--test", default=None, help="a held-out route to score")
    p.add_argument("--no-learned", action="store_true")
    p.add_argument("--mdn-epochs", type=int, default=120)

    p = exp("e4-aperture", "krakensdr_2400000_cu8")
    p.add_argument("--freq-hz", type=float, default=98.1e6)
    p.add_argument("--radius-m", type=float, default=1.0, help="the array's radius")
    p.add_argument("--sigma-deg", type=float, default=3.0)

    p = exp("e5-reach", "krakensdr_2400000_cu8")
    p.add_argument("--model", default="deygout", help="the propagation model")
    p.add_argument("--no-learned", action="store_true")
    p.add_argument("--learned-steps", type=int, default=500)
    p.add_argument("--learned-fields", type=int, default=160)

    p = exp("hf")
    p.add_argument("--spots", nargs="*", default=None, help="ALL_WSPR.TXT-style files")
    p.add_argument("--here", default="FM18", help="your Maidenhead square")
    p.add_argument("--kiwi", action="append", default=None, help="NAME:GRID, repeatable")
    p.add_argument("--bands", default="80m,40m,30m,20m")
    p.add_argument("--hours", default="20,21,22,23,0,1,2,3", help="UTC hours")
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--month", type=int, default=10)
    p.add_argument("--ssn", type=float, default=100.0, help="sunspot number for VOACAP")

    p = exp("hunter", "rtlsdr_2400000_cu8")
    p.add_argument("--goal", default="anything narrowband and bursty between 400 and 470")
    p.add_argument("--duration", type=float, default=900.0, help="seconds per seed")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--scan-dwell", type=float, default=1.0)

    p = exp("inpaint", "rtlsdr_2400000_cu8")
    p.add_argument("--inpainter", default=None, help="a trained D1 inpainter")
    p.add_argument("--canonical", default="voice")
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--gaps-ms", default="0.5,1,2")
    p.add_argument("--snr-db", type=float, default=20.0)
    p.add_argument("--silence-trials", type=int, default=40)

    p = exp("novelty")
    p.add_argument("--n-docs", type=int, default=10)
    p.add_argument("--data", default=None, help="a JSON of your project and documents")
    p.add_argument("--llama-model", default=None, help="a GGUF model for the llama arm")

    p = exp("pulse")
    p.add_argument("--seeds", type=int, default=5, help="how many scenes")
    p.add_argument("--duration", type=float, default=0.3)
    p.add_argument("--mean-drop", type=float, default=0.15)
    p.add_argument("--noise-rate", type=float, default=150.0)

    p = exp("translator")
    p.add_argument("--from-profile", default="bladerf1_2400000_ci16")
    p.add_argument("--to-profile", default="rtlsdr_2400000_cu8")
    p.add_argument("--translator", default=None, help="a trained A6 translator")
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--n-pairs", type=int, default=400)
    p.add_argument("--n-test", type=int, default=100)
    p.add_argument("--train-steps", type=int, default=4000)

    p = exp("vitals")
    p.add_argument("--duration", type=float, default=90.0)
    p.add_argument("--fs", type=float, default=50.0)
    p.add_argument("--no-lstm", action="store_true")

    p = exp("wer")
    p.add_argument("--clips", nargs="+", required=True, help="WAV files or folders")
    p.add_argument("--references", required=True,
                   help="what was said: .json {clip: text} or lines <clip><TAB><text>")
    p.add_argument("--transcriber", required=True,
                   help="a command with {wav}, whose output is the transcript")
    p.add_argument("--noise-clips", nargs="*", default=None,
                   help="noise-only WAVs for the hallucination check")
    p.add_argument("--out", default=None)


def build_parser() -> argparse.ArgumentParser:
    common = _common()
    ap = _Parser(prog="python -m atk_diffusion", parents=[common],
                 description="The ATK Diffusion Toolkit's command line. Every "
                 "command says what it did in plain words; exit code 0 = done, "
                 "1 = refused or failed (the sentence says why), 2 = the command "
                 "line was wrong.")
    sub = ap.add_subparsers(dest="command", metavar="COMMAND", parser_class=_Parser)

    sub.add_parser("status", parents=[common],
                   help="what is installed, the rf_data root, the profiles")

    p = sub.add_parser("rfdata", parents=[common], help="the rf_data folder")
    s = p.add_subparsers(dest="sub", metavar="ACTION", required=True)
    s.add_parser("init", parents=[common], help="make the rf_data tree and its README")
    v = s.add_parser("verify", parents=[common],
                     help="check every file in the write log against its hash")
    v.add_argument("--full", action="store_true", help="re-hash every file")

    p = sub.add_parser("profile", parents=[common], help="receiver profiles")
    s = p.add_subparsers(dest="sub", metavar="ACTION", required=True)
    s.add_parser("list", parents=[common], help="the profiles under rf_data")
    v = s.add_parser("show", parents=[common], help="everything a profile holds")
    v.add_argument("profile_id")
    v = s.add_parser("new", parents=[common], help="make a profile")
    v.add_argument("profile_id", nargs="?", default=None,
                   help="e.g. rtlsdr_2400000_cu8 (or give --family and --rate)")
    v.add_argument("--family", default=None)
    v.add_argument("--rate", type=float, default=None, help="sample rate, S/s")
    v.add_argument("--datatype", default=None, help="cu8, ci8, ci16 or cf32")
    v.add_argument("--variant", default=None, help="e.g. ch0 for a Kraken channel")
    v.add_argument("--serial", dest="device_serial", default=None)
    v.add_argument("--firmware", default=None)
    v.add_argument("--gain-preset", default=None)
    v.add_argument("--overwrite", action="store_true")
    v = s.add_parser("set-safe-input", parents=[common],
                     help="enter the receiver's maximum safe input from its data sheet")
    v.add_argument("profile_id")
    v.add_argument("--max-dbm", type=float, required=True)
    v.add_argument("--source", required=True, help="the data sheet (name, revision, page)")
    v.add_argument("--date", required=True, help="YYYY-MM-DD")
    v = s.add_parser("geometry", parents=[common],
                     help="show (or change) the spectrogram and SCF geometry")
    v.add_argument("profile_id")
    for k, (_w, typ) in _GEOM.items():
        v.add_argument("--" + k.replace("_", "-"), type=typ, default=None)

    p = sub.add_parser("impair", parents=[common], help="receiver impairments")
    s = p.add_subparsers(dest="sub", metavar="ACTION", required=True)
    v = s.add_parser("measure", parents=[common],
                     help="measure a TERMINATED capture into its profile")
    v.add_argument("capture")
    v.add_argument("--profile", default=None, help="check the capture is this profile")
    v.add_argument("--serial", default=None, help="the device's serial number")
    v.add_argument("--firmware", default=None)
    v.add_argument("--skip-seconds", type=float, default=0.1, help="tuner settling")
    v.add_argument("--max-seconds", type=float, default=10.0, help="for the floor")
    v.add_argument("--no-floor", action="store_true", help="impairments only")

    p = sub.add_parser("synth", parents=[common], help="datasets at a profile's exact rate")
    s = p.add_subparsers(dest="sub", metavar="ACTION", required=True)
    v = s.add_parser("narrowband", parents=[common],
                     help="a classification dataset (one signal per example)")
    v.add_argument("--profile", required=True)
    v.add_argument("--name", required=True)
    v.add_argument("--classes", required=True, help="comma list of class-table names")
    v.add_argument("--n-per-class", type=int, required=True)
    v.add_argument("--snr", default="0,20", help="LOW,HIGH dB (default 0,20)")
    v.add_argument("--canonical", default="voice", choices=("voice", "wideband", "spread"))
    v.add_argument("--generator", default="native", choices=("native", "torchsig"))
    v.add_argument("--window", type=int, default=None, help="samples (default 4096)")
    v.add_argument("--splits", default="0.8,0.1,0.1")
    v.add_argument("--impairment-level", type=int, default=0, help="TorchSig's (0-2)")
    v.add_argument("--no-scf", action="store_true", help="skip the SCF images")
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--overwrite", action="store_true")
    v = s.add_parser("wideband", parents=[common],
                     help="a detection dataset (scenes, tiles, boxes)")
    v.add_argument("--profile", required=True)
    v.add_argument("--name", required=True)
    v.add_argument("--scenes", type=int, required=True)
    v.add_argument("--env", default=None, help="an environment profile, e.g. us-va-nokesville")
    v.add_argument("--seconds", type=float, default=None, help="scene length (default: a tile)")
    v.add_argument("--classes", default=None, help="comma list (default: all)")
    v.add_argument("--generator", default="native", choices=("native", "torchsig"))
    v.add_argument("--splits", default="0.8,0.1,0.1")
    v.add_argument("--no-tiles", action="store_true")
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--overwrite", action="store_true")
    v = s.add_parser("cabled", parents=[common],
                     help="a dataset from cabled recordings (the domain-gap set)")
    v.add_argument("captures", nargs="+")
    v.add_argument("--profile", required=True)
    v.add_argument("--name", required=True)
    v.add_argument("--canonical", default=None, choices=("voice", "wideband", "spread"))
    v.add_argument("--window", type=int, default=None)
    v.add_argument("--splits", default="0,0,1")
    v.add_argument("--no-scf", action="store_true")
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--overwrite", action="store_true")
    v = s.add_parser("throughput", parents=[common],
                     help="examples a second on this machine (writes nothing)")
    v.add_argument("--profile", required=True)
    v.add_argument("--generator", default="native", choices=("native", "torchsig"))
    v.add_argument("--classes", default=None)
    v.add_argument("--n", type=int, default=8)
    v.add_argument("--canonical", default="voice", choices=("voice", "wideband", "spread"))
    v.add_argument("--window", type=int, default=None)
    v.add_argument("--scf", action="store_true", help="include the SCF images")
    v = s.add_parser("list", parents=[common], help="the datasets under rf_data")
    v.add_argument("--profile", default=None)
    v.add_argument("--verify", action="store_true", help="check every file's hash")

    _add_train_parsers(sub, common)

    p = sub.add_parser("export", parents=[common], help="ONNX for ATK's CPU path")
    s = p.add_subparsers(dest="sub", metavar="ACTION", required=True)
    v = s.add_parser("onnx", parents=[common], help="make or check a model's ONNX file")
    v.add_argument("model", help="the model folder (or its name with --profile)")
    v.add_argument("--profile", default=None)

    p = sub.add_parser("models", parents=[common], help="trained models")
    s = p.add_subparsers(dest="sub", metavar="ACTION", required=True)
    v = s.add_parser("list", parents=[common], help="every model card, in words")
    v.add_argument("--profile", default=None)
    v.add_argument("--verify", action="store_true", help="check the weights' hashes")

    p = sub.add_parser("detect", parents=[common], help="the detector over a capture")
    p.add_argument("capture")
    p.add_argument("--profile", default=None, help="check the capture is this profile")
    p.add_argument("--write-annotations", action="store_true",
                   help="write the detections into the capture's .sigmf-meta")
    p.add_argument("--cyclic", action="store_true", help="switch the cyclic proposer on")
    p.add_argument("--cyclic-mode", default="regions", choices=("regions", "span"))
    p.add_argument("--learned", action="store_true",
                   help="use the newest trained proposer of the profile")
    p.add_argument("--proposer", default=None, help="a trained proposer (name or folder)")
    p.add_argument("--classifier", default=None, help="a trained classifier")
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--chunk-seconds", type=float, default=1.0)
    p.add_argument("--channel", type=int, default=0, help="a Kraken capture's channel")
    p.add_argument("--list", type=int, default=40, help="detections to print")

    p = sub.add_parser("cut", parents=[common], help="the signal cut (B7)")
    p.add_argument("capture")
    p.add_argument("--t0", type=float, required=True, help="seconds from the capture's start")
    p.add_argument("--t1", type=float, required=True)
    p.add_argument("--f-lo", type=float, required=True, help="absolute Hz")
    p.add_argument("--f-hi", type=float, required=True)
    p.add_argument("--profile", default=None)
    p.add_argument("--clean", default=None,
                   choices=("matched", "fresh", "wiener", "rfi_mask_interp", "score"),
                   help="a classical clean beside the original (score: Kraken "
                        "cuts). The diffusion clean takes a loaded denoiser and "
                        "runs from ATK's Cuts tab, not from here")
    p.add_argument("--no-analyze", action="store_true")
    p.add_argument("--margin", type=float, default=None, help="seconds each side")
    p.add_argument("--who", default="")
    p.add_argument("--note", default="")

    p = sub.add_parser("resample", parents=[common],
                       help="a capture to another profile's rate (a logged step)")
    p.add_argument("capture")
    p.add_argument("--to", required=True, help="the target profile")
    p.add_argument("--out", default=None, help="the new capture's base path")
    p.add_argument("--who", default="")
    p.add_argument("--reason", default="")
    p.add_argument("--overwrite", action="store_true")

    _add_experiment_parsers(sub, common)

    p = sub.add_parser("cabled", parents=[common],
                       help="the cabled calibration loop (the ONE transmit exception)")
    s = p.add_subparsers(dest="sub", metavar="ACTION", required=True)
    v = s.add_parser("check", parents=[common], help="the safety arithmetic (transmits nothing)")
    _add_loop_args(v)
    v = s.add_parser("txfile", parents=[common], help="build a transmit file (transmits nothing)")
    v.add_argument("--tx", required=True, help="hackrf, bladerf1 or bladerf2")
    v.add_argument("--rx-profile", required=True)
    v.add_argument("--freq", type=float, default=None, help="planned centre, Hz")
    _add_txfile_args(v)
    v = s.add_parser("plan", parents=[common],
                     help="the verdict and the exact command; transmits ONLY with "
                          "--execute --i-confirm-cabled-with-attenuators")
    _add_loop_args(v)
    _add_txfile_args(v)
    v.add_argument("--tx-file", default=None, help="an existing transmit manifest")
    v.add_argument("--bandwidth-hz", type=float, default=None)
    v.add_argument("--execute", action="store_true", help="run the plan (with the confirm flag)")
    v.add_argument("--i-confirm-cabled-with-attenuators", action="store_true",
                   help="confirm, now: cable + attenuators + DC block, never an antenna")
    v = s.add_parser("measure", parents=[common],
                     help="the receiver's level from a loop recording, for the ramp")
    v.add_argument("capture")
    v.add_argument("--tx-file", required=True, help="the transmit manifest")
    v.add_argument("--rx-full-scale-dbm", type=float, default=None,
                   help="the receiver's full-scale input at its gain, when known")
    _add_loop_args(v, rx_required=False)
    v = s.add_parser("align", parents=[common],
                     help="line a loop recording up with its manifest; write the labels")
    v.add_argument("capture")
    v.add_argument("--tx-file", required=True)
    _add_loop_args(v, rx_required=False)

    p = sub.add_parser("products", parents=[common], help="maps, tracks, coverage")
    s = p.add_subparsers(dest="sub", metavar="ACTION", required=True)
    v = s.add_parser("list", parents=[common], help="the products under rf_data")
    v.add_argument("--kind", default=None)
    v.add_argument("--verify", action="store_true")
    v = s.add_parser("import", parents=[common],
                     help="bring a product folder from another rf_data")
    v.add_argument("folder")
    v.add_argument("--kind", default=None)
    v = s.add_parser("reach", parents=[common],
                     help="predicted reach of a radio over DTED (plan E5) — a map layer")
    v.add_argument("--lat", type=float, required=True, help="transmitter latitude")
    v.add_argument("--lon", type=float, required=True, help="transmitter longitude")
    v.add_argument("--freq", type=float, required=True, help="Hz")
    v.add_argument("--power-w", type=float, default=None, help="transmit power, W")
    v.add_argument("--power-dbm", type=float, default=None)
    v.add_argument("--height", type=float, default=2.0, help="antenna above ground, m")
    v.add_argument("--antenna", default="dipole",
                   choices=("dipole", "isotropic", "directional"))
    v.add_argument("--antenna-file", default=None,
                   help="a pattern as JSON (geo.antenna to_json / from ATK's designer)")
    v.add_argument("--gain-dbi", type=float, default=None)
    v.add_argument("--beamwidth", type=float, default=None,
                   help="directional: -3 dB azimuth beamwidth, degrees")
    v.add_argument("--azimuth", type=float, default=0.0, help="boresight, degrees true")
    v.add_argument("--tilt", type=float, default=0.0)
    v.add_argument("--line-loss", type=float, default=0.0, help="dB")
    v.add_argument("--rx-height", type=float, default=1.5, help="m")
    v.add_argument("--sensitivity-dbm", type=float, default=-110.0)
    v.add_argument("--fade-margin", type=float, default=0.0, help="dB")
    v.add_argument("--radius-km", type=float, default=20.0)
    v.add_argument("--grid-m", type=float, default=250.0)
    v.add_argument("--model", default="itm",
                   help="itm (Longley-Rice), deygout, bullington, two_ray, fspl")
    v.add_argument("--ground", default="average",
                   help="average, poor, good, city, fresh_water, sea_water")
    v.add_argument("--dted", default=None,
                   help="the DTED folder (default rf_data\\shared\\dted)")
    v.add_argument("--flat", action="store_true", help="no terrain (the sanity line)")
    v.add_argument("--flat-height", type=float, default=0.0)
    v.add_argument("--name", default=None, help="the product run's folder name")
    v.add_argument("--label", default="", help="what to call the radio on the map")

    p = sub.add_parser("hunt", parents=[common], help="the self-hunting receiver (B6)")
    s = p.add_subparsers(dest="sub", metavar="ACTION", required=True)
    v = s.add_parser("simulate", parents=[common], help="hunt a simulated band")
    v.add_argument("--goal", default="anything narrowband and bursty between 400 and 470")
    v.add_argument("--profile", default="rtlsdr_2400000_cu8")
    v.add_argument("--duration", type=float, default=300.0, help="seconds")
    v.add_argument("--policy", default="hunter", choices=("hunter", "fixed"))
    v.add_argument("--dwell", type=float, default=1.0, help="the fixed scan's dwell")
    v.add_argument("--seed", type=int, default=1)

    p = sub.add_parser("vitals", parents=[common], help="CSI vital signs (I1)")
    s = p.add_subparsers(dest="sub", metavar="ACTION", required=True)
    v = s.add_parser("replay", parents=[common], help="a saved ESP32 CSI log")
    v.add_argument("log")
    v.add_argument("--fs", type=float, default=50.0, help="the analysis rate, frames/s")
    v.add_argument("--mac", default=None, help="the sender (default: the most common)")
    v.add_argument("--save", action="store_true", help="write the report under rf_data")

    p = sub.add_parser("repair", parents=[common], help="repair tools (D5)")
    s = p.add_subparsers(dest="sub", metavar="ACTION", required=True)
    v = s.add_parser("audio", parents=[common], help="fill dropouts in a WAV")
    v.add_argument("wav")
    v.add_argument("--out", default=None)
    v.add_argument("--method", default="janssen", choices=("janssen", "lpc", "linear"))
    v.add_argument("--gap", action="append", default=None,
                   help="START,END seconds of a gap you marked (repeatable); "
                        "without it gaps are found")
    return ap


_COMMANDS = {
    ("status", None): cmd_status,
    ("rfdata", "init"): cmd_rfdata_init,
    ("rfdata", "verify"): cmd_rfdata_verify,
    ("profile", "list"): cmd_profile_list,
    ("profile", "show"): cmd_profile_show,
    ("profile", "new"): cmd_profile_new,
    ("profile", "set-safe-input"): cmd_profile_safe_input,
    ("profile", "geometry"): cmd_profile_geometry,
    ("impair", "measure"): cmd_impair_measure,
    ("synth", "narrowband"): cmd_synth_narrowband,
    ("synth", "wideband"): cmd_synth_wideband,
    ("synth", "cabled"): cmd_synth_cabled,
    ("synth", "throughput"): cmd_synth_throughput,
    ("synth", "list"): cmd_synth_list,
    ("train", None): cmd_train,
    ("export", "onnx"): cmd_export_onnx,
    ("models", "list"): cmd_models_list,
    ("detect", None): cmd_detect,
    ("cut", None): cmd_cut,
    ("resample", None): cmd_resample,
    ("experiment", None): cmd_experiment,
    ("cabled", "check"): cmd_cabled_check,
    ("cabled", "txfile"): cmd_cabled_txfile,
    ("cabled", "plan"): cmd_cabled_plan,
    ("cabled", "measure"): cmd_cabled_measure,
    ("cabled", "align"): cmd_cabled_align,
    ("products", "list"): cmd_products_list,
    ("products", "import"): cmd_products_import,
    ("products", "reach"): cmd_products_reach,
    ("hunt", "simulate"): cmd_hunt_simulate,
    ("vitals", "replay"): cmd_vitals_replay,
    ("repair", "audio"): cmd_repair_audio,
}


def main(argv=None) -> int:
    """Run one command. Returns the exit code (0 done, 1 refused or failed,
    2 a wrong command line) — never raises for an operator's mistake."""
    ap = build_parser()
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else EXIT_USAGE
    for k, default in (("json", False), ("debug", False), ("rf_data", None)):
        if not hasattr(a, k):
            setattr(a, k, default)
    if not a.command:
        ap.print_help()
        return EXIT_USAGE
    key = (a.command, getattr(a, "sub", None))
    fn = _COMMANDS.get(key) or _COMMANDS.get((a.command, None))
    ctx = Ctx(a)
    try:
        data = fn(ctx)
        code = EXIT_OK
        error = ""
    except Fail as e:
        data, code, error = None, EXIT_FAIL, str(e)
    except KeyboardInterrupt:
        data, code, error = None, EXIT_FAIL, "stopped (Ctrl-C) — nothing after this point ran"
    except (ValueError, FileNotFoundError, NotADirectoryError, PermissionError,
            RuntimeError, ImportError, OSError) as e:
        # the toolkit's refusals are ValueError subclasses that already speak
        # in sentences (ProfileMismatch, DatasetRefusal, CardRefusal, CutError …)
        data, code = None, EXIT_FAIL
        error = str(e) or type(e).__name__
        if isinstance(e, ImportError):
            error = (f"a part of the toolkit's environment is missing ({e}). "
                     "`atkdiff.bat status` says what is installed.")
        if a.debug:
            traceback.print_exc()
    except Exception as e:                                 # noqa: BLE001
        data, code = None, EXIT_FAIL
        error = (f"unexpected error ({type(e).__name__}: {e}). This is a fault in "
                 "the toolkit, not something you did; run the same command with "
                 "--debug and keep the output.")
        if a.debug:
            traceback.print_exc()
    if code != EXIT_OK:
        ctx.say(f"[ERR] {error or 'failed'}")
    if a.json:
        payload = {"ok": code == EXIT_OK, "command": " ".join(
            x for x in (a.command, getattr(a, "sub", None) or getattr(a, "kind", None)
                        or getattr(a, "name", None)) if x)}
        if code == EXIT_OK:
            payload["result"] = _json_safe(data)
        else:
            payload["error"] = error
        print(json.dumps(payload, indent=1, allow_nan=False), file=ctx.out)
    return code
