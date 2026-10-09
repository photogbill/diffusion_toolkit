# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The learned speech enhancer, held at arm's length — plan §4.D2 and §8.

    *"Diffusion speech enhancement models exist off the shelf (SGMSE-class).
    A clean before transcribe step on the DSD and PTT audio paths."*
    — plan §4.D2.   *"Weights pinned by hash; license shown."* — plan §8.

WHY A SEPARATE PROCESS. SGMSE (sp-uhh/sgmse, MIT) brings its own PyTorch,
PyTorch Lightning and torchaudio versions. It runs in ITS OWN environment as
a separate process and is never imported here, so its dependencies can never
break ATK's core or this toolkit's training environment, and a crash costs
one clip, never the app (plan §2.7 — "failing loudly and costing one
feature"). Nothing is downloaded: Bill fetches the code and a checkpoint,
hashes the checkpoint once (`write_card`), and from then on the runner
REFUSES to run a checkpoint whose SHA-256 is not that one.

WHAT RUNS. `SgmseRunner.enhance(in_wav, out_wav)` copies the clip into a
work folder BESIDE the output (never the system temp folder, which on
Windows is in the user profile), runs

    <python_exe> <repo>/enhancement.py --test_dir <work>/in
        --enhanced_dir <work>/out --ckpt <checkpoint> --N <steps>
        --corrector <ald> --corrector_steps <1> --snr <0.5> --device <cpu|cuda>

with a timeout, moves the result to `out_wav`, deletes the work folder, and
writes `<out_wav>.json` — tier INVENTED, the checkpoint's hash, its licence,
the command line. The argument names are sgmse's `enhancement.py` as
published; `extra_args` and `script` cover a version that differs.

TIER: INVENTED, always. A score-based diffusion enhancer RECONSTRUCTS clean
speech from a prior over speech; at low SNR it can produce phonemes that were
never spoken, which Whisper will then transcribe with confidence. That is the
plan's §2.1 hazard in its purest form, and why `experiments.wer_eval` reports
a hallucination check (words from noise-only clips) beside the word error
rate. Never for identifying a speaker (plan §2.4): this is noise reduction
for transcription.

`ExternalEnhancer` is the same discipline for any command-line enhancer
(DeepFilterNet's `deepFilter` as the optional comparator): a template with
`{in}`, `{out}`, `{out_dir}`, `{stem}`, `{model}`; a timeout; a work folder
beside the output; an optional model hash; tier from its method.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from atk_diffusion import provenance as _prov

DEFAULT_LICENSE = ("sgmse code: MIT (sp-uhh/sgmse). The checkpoint's licence is "
                   "the one on the page it was downloaded from — record it in "
                   "the card.")

#: Environment variables that must not leak from this process into the
#: enhancer's own Python (it has its own packages).
_STRIP_ENV = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV",
              "CONDA_PREFIX")


class EnhancerRefusal(ValueError):
    """The enhancer may not run (missing files, hash mismatch). Why, in words."""


class EnhancerFailed(RuntimeError):
    """The enhancer ran and did not produce an output. Why, in words."""


def _env() -> dict:
    env = dict(os.environ)
    for k in _STRIP_ENV:
        env.pop(k, None)
    return env


def _tail(text: str, lines: int = 4) -> str:
    rows = [r for r in (text or "").strip().splitlines() if r.strip()]
    return " | ".join(rows[-lines:])[:600]


def _workdir(out_wav: Path, tag: str) -> Path:
    w = out_wav.parent / f".{out_wav.stem}.{tag}_work_{os.getpid()}_{int(time.time() * 1000)}"
    w.mkdir(parents=True, exist_ok=False)
    return w


def _write_sidecar(out_wav: Path, info: dict) -> Path:
    sp = out_wav.with_name(out_wav.name + ".json")
    sp.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return sp


def _probe_wav(path: Path) -> dict:
    from atk_diffusion.repair.speech import read_wav
    try:
        y, fs, info = read_wav(path)
    except (ValueError, OSError) as e:
        raise EnhancerFailed(f"the enhancer wrote {path.name}, but it cannot be "
                             f"read as audio ({e})") from None
    return {"rate": fs, "seconds": round(y.size / float(fs), 3), **info}


class _HashCache:
    """Re-hash a large file only when its size or time moved (paths.WriteLog's
    rule): verifying is mandatory, re-reading 250 MB per clip is not."""

    def __init__(self):
        self._key = None
        self._ok = False

    def check(self, path: Path, want: str) -> bool:
        st = path.stat()
        key = (str(path), st.st_size, st.st_mtime_ns, want)
        if key != self._key:
            self._ok = _prov.sha256_path(path).lower() == want.lower()
            self._key = key
        return self._ok


class SgmseRunner:
    """The SGMSE-class enhancer as a separate process in its own environment.

    `python_exe` is that environment's interpreter, `repo_dir` the sgmse
    checkout, `checkpoint` the downloaded weights and `sha256` their pinned
    hash (from `write_card`, or typed from the release page)."""

    method = "speech_enhance"

    def __init__(self, python_exe, repo_dir, checkpoint, sha256: str,
                 script: str = "enhancement.py", device: str = "cuda",
                 steps: int = 30, corrector: str = "ald",
                 corrector_steps: int = 1, snr: float = 0.5,
                 timeout_s: float = 600.0, extra_args=(),
                 license: str = DEFAULT_LICENSE, name: str = "sgmse"):
        self.python_exe = Path(python_exe)
        self.repo_dir = Path(repo_dir)
        self.checkpoint = Path(checkpoint)
        self.sha256 = str(sha256 or "").strip().lower()
        self.script = str(script)
        self.device = str(device)
        self.steps = int(steps)
        self.corrector = str(corrector)
        self.corrector_steps = int(corrector_steps)
        self.snr = float(snr)
        self.timeout_s = float(timeout_s)
        self.extra_args = [str(a) for a in extra_args]
        self.license = str(license)
        self.name = str(name)
        self._hash = _HashCache()

    # -- what it is ----------------------------------------------------------
    @classmethod
    def from_card(cls, model_dir, python_exe, repo_dir, **kw) -> "SgmseRunner":
        """A runner for the checkpoint a `speech_enhance` card describes; the
        card's hash is the one enforced (cards.load verifies it too)."""
        from atk_diffusion import cards
        card = cards.load(model_dir, expect_kind="speech_enhance")
        ck = cards.weights_path(model_dir, card)
        kw.setdefault("license", card.license or DEFAULT_LICENSE)
        kw.setdefault("name", card.name)
        return cls(python_exe, repo_dir, ck, card.weights.get("sha256", ""), **kw)

    def describe(self) -> list[str]:
        return [f"{self.name} — a diffusion speech enhancer run as a separate "
                f"process ({self.python_exe})",
                f"checkpoint {self.checkpoint.name}, SHA-256 {self.sha256[:16]}…",
                f"licence: {self.license}",
                f"{self.steps} reverse steps on {self.device}; output is "
                + _prov.TIER_WORDS["invented"]]

    def verify_checkpoint(self) -> tuple[bool, str]:
        if not self.sha256:
            return False, ("No SHA-256 was given for the checkpoint. A learned "
                           "model's weights are pinned by hash (plan §8): hash "
                           "the file once with write_card and use that card.")
        if not self.checkpoint.is_file():
            return False, (f"The checkpoint {self.checkpoint} does not exist. "
                           "Download it yourself and point the runner at it; "
                           "nothing is downloaded here.")
        if not self._hash.check(self.checkpoint, self.sha256):
            return False, (f"The checkpoint {self.checkpoint.name} does not match "
                           "its pinned SHA-256 hash, so it is not the model that "
                           "was checked. Restore the right file or make a new "
                           "card for this one on purpose.")
        return True, ""

    def available(self) -> tuple[bool, str]:
        """(ok, why) — every reason it cannot run, as a sentence."""
        if not self.python_exe.is_file():
            return False, (f"The enhancer's Python ({self.python_exe}) was not "
                           "found. SGMSE runs in its own environment; set the "
                           "path to that environment's python executable.")
        if not self.repo_dir.is_dir():
            return False, (f"The SGMSE code folder ({self.repo_dir}) was not "
                           "found. Point the runner at your sgmse checkout.")
        if not (self.repo_dir / self.script).is_file():
            return False, (f"{self.script} is not in {self.repo_dir}, so this "
                           "does not look like an sgmse checkout.")
        return self.verify_checkpoint()

    def command(self, in_dir, out_dir) -> list[str]:
        return [str(self.python_exe), str(self.repo_dir / self.script),
                "--test_dir", str(in_dir), "--enhanced_dir", str(out_dir),
                "--ckpt", str(self.checkpoint), "--N", str(self.steps),
                "--corrector", self.corrector,
                "--corrector_steps", str(self.corrector_steps),
                "--snr", f"{self.snr:g}", "--device", self.device,
                *self.extra_args]

    # -- running -------------------------------------------------------------
    def enhance(self, in_wav, out_wav) -> dict:
        """Enhance one WAV into `out_wav` (+ `<out_wav>.json`). Raises
        EnhancerRefusal before running, EnhancerFailed after."""
        src, out = Path(in_wav), Path(out_wav)
        ok, why = self.available()
        if not ok:
            raise EnhancerRefusal(why)
        if not src.is_file():
            raise EnhancerRefusal(f"the clip {src} does not exist")
        out.parent.mkdir(parents=True, exist_ok=True)
        work = _workdir(out, "sgmse")
        try:
            (work / "in").mkdir()
            (work / "out").mkdir()
            shutil.copyfile(src, work / "in" / f"{src.stem}.wav")
            cmd = self.command(work / "in", work / "out")
            t0 = time.time()
            try:
                proc = subprocess.run(cmd, cwd=str(self.repo_dir), env=_env(),
                                      capture_output=True, text=True,
                                      timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                raise EnhancerFailed(
                    f"the SGMSE enhancer did not finish within "
                    f"{self.timeout_s:g} s and was stopped; nothing was "
                    "written. (On CPU, fewer reverse steps or a shorter clip "
                    "help.)") from None
            except OSError as e:
                raise EnhancerFailed(f"the SGMSE enhancer could not be started: "
                                     f"{e}") from None
            secs = time.time() - t0
            if proc.returncode != 0:
                raise EnhancerFailed(
                    f"the SGMSE enhancer failed (exit code {proc.returncode}): "
                    f"{_tail(proc.stderr) or _tail(proc.stdout) or 'no message'}")
            made = work / "out" / f"{src.stem}.wav"
            if not made.is_file():
                found = sorted((work / "out").rglob("*.wav"))
                if not found:
                    raise EnhancerFailed("the SGMSE enhancer exited normally but "
                                         "wrote no WAV file")
                made = found[0]
            probe = _probe_wav(made)
            os.replace(made, out)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        info = {"tier": _prov.tier_for(self.method), "method": self.method,
                "tier_words": _prov.TIER_WORDS[_prov.tier_for(self.method)],
                "enhancer": self.name, "model_sha256": self.sha256,
                "checkpoint": str(self.checkpoint), "license": self.license,
                "command": cmd, "seconds_to_run": round(secs, 2),
                "output": str(out), "output_wav": probe,
                "source": str(src), "source_sha256": _prov.sha256_path(src),
                "stdout_tail": _tail(proc.stdout),
                "provenance": _prov.stamp("repair.speech_learned",
                                          enhancer=self.name)}
        info["sidecar"] = str(_write_sidecar(out, info))
        return info

    __call__ = enhance


def write_card(model_dir, checkpoint_file: str, name: str = "sgmse",
               license: str = DEFAULT_LICENSE, notes=(), metrics=None,
               trained_on: str = "") -> Path:
    """Pin a downloaded checkpoint: a `speech_enhance` model card holding its
    SHA-256 and licence (cards.save hashes the file). Not per profile —
    audio has no receiver profile."""
    from atk_diffusion import cards
    card = cards.new_card(name, "speech_enhance", license=license,
                          notes=list(notes) or ["a pre-trained SGMSE-class "
                                                "checkpoint, not trained here"],
                          metrics=dict(metrics or {}), trained_on=trained_on)
    return cards.save(model_dir, card, checkpoint_file)


class ExternalEnhancer:
    """Any command-line enhancer at arm's length. `argv` is a template list:
    `{in}` the input WAV, `{out}` a file path the tool should write, or
    `{out_dir}` a folder it writes into (then `output_glob`, which may use
    `{stem}`, finds the result), `{model}` the model path."""

    def __init__(self, name: str, argv, output_glob: str | None = None,
                 timeout_s: float = 600.0, method: str = "speech_enhance",
                 model_path=None, sha256: str = "", license: str = "",
                 cwd=None):
        self.name = str(name)
        self.argv = [str(a) for a in argv]
        self.output_glob = output_glob
        self.timeout_s = float(timeout_s)
        self.method = str(method)
        _prov.tier_for(self.method)          # an undeclared method may not write
        self.model_path = Path(model_path) if model_path else None
        self.sha256 = str(sha256 or "").lower()
        self.license = str(license)
        self.cwd = Path(cwd) if cwd else None
        self._hash = _HashCache()

    def available(self) -> tuple[bool, str]:
        exe = self.argv[0] if self.argv else ""
        if not exe or not (Path(exe).is_file() or shutil.which(exe)):
            return False, (f"{self.name}: the program {exe!r} was not found. "
                           "Install it in its own environment and give its "
                           "full path.")
        if self.model_path is not None:
            if not self.model_path.is_file():
                return False, f"{self.name}: the model {self.model_path} was not found."
            if self.sha256 and not self._hash.check(self.model_path, self.sha256):
                return False, (f"{self.name}: the model {self.model_path.name} "
                               "does not match its pinned SHA-256 hash.")
        return True, ""

    def enhance(self, in_wav, out_wav) -> dict:
        src, out = Path(in_wav), Path(out_wav)
        ok, why = self.available()
        if not ok:
            raise EnhancerRefusal(why)
        out.parent.mkdir(parents=True, exist_ok=True)
        work = _workdir(out, "ext")
        try:
            out_dir = work / "out"
            out_dir.mkdir()
            target = out_dir / f"{src.stem}.wav"
            subs = {"in": str(src), "out": str(target), "out_dir": str(out_dir),
                    "stem": src.stem,
                    "model": str(self.model_path) if self.model_path else ""}
            cmd = [a.format(**subs) for a in self.argv]
            t0 = time.time()
            try:
                proc = subprocess.run(cmd, cwd=str(self.cwd) if self.cwd else None,
                                      env=_env(), capture_output=True, text=True,
                                      timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                raise EnhancerFailed(f"{self.name} did not finish within "
                                     f"{self.timeout_s:g} s and was stopped; "
                                     "nothing was written.") from None
            except OSError as e:
                raise EnhancerFailed(f"{self.name} could not be started: {e}") from None
            if proc.returncode != 0:
                raise EnhancerFailed(f"{self.name} failed (exit code "
                                     f"{proc.returncode}): "
                                     f"{_tail(proc.stderr) or 'no message'}")
            if target.is_file():
                made = target
            else:
                pat = (self.output_glob or "*.wav").format(stem=src.stem)
                found = sorted(glob.glob(str(out_dir / "**" / pat), recursive=True),
                               key=lambda p: os.path.getmtime(p))
                if not found:
                    raise EnhancerFailed(f"{self.name} exited normally but "
                                         f"wrote no file matching {pat!r}")
                made = Path(found[-1])
            probe = _probe_wav(made)
            os.replace(made, out)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        tier = _prov.tier_for(self.method)
        info = {"tier": tier, "method": self.method,
                "tier_words": _prov.TIER_WORDS[tier], "enhancer": self.name,
                "model_sha256": self.sha256, "license": self.license,
                "command": cmd, "seconds_to_run": round(time.time() - t0, 2),
                "output": str(out), "output_wav": probe, "source": str(src),
                "source_sha256": _prov.sha256_path(src),
                "provenance": _prov.stamp("repair.speech_learned",
                                          enhancer=self.name)}
        info["sidecar"] = str(_write_sidecar(out, info))
        return info

    __call__ = enhance


def deepfilternet(exe: str = "deepFilter", **kw) -> ExternalEnhancer:
    """DeepFilterNet's command-line tool as the optional learned comparator.
    The template follows its published CLI (`deepFilter <wav> --output-dir
    <dir>`, writing `<stem>_DeepFilterNet3.wav`); check it against the
    installed version — the argv list is the one thing to edit."""
    return ExternalEnhancer("deepfilternet",
                            [exe, "{in}", "--output-dir", "{out_dir}"],
                            output_glob="{stem}_DeepFilterNet*.wav",
                            license=kw.pop("license", "MIT / Apache-2.0 "
                                           "(DeepFilterNet)"), **kw)


def available(python_exe=None, repo_dir=None, checkpoint=None,
              sha256: str = "") -> tuple[bool, str]:
    """(ok, why) for the learned enhancer as configured — one line for the
    tab. Unconfigured is a plain 'not set up' sentence, not an error."""
    missing = [n for n, v in (("the enhancer's python", python_exe),
                              ("the sgmse code folder", repo_dir),
                              ("the checkpoint", checkpoint)) if not v]
    if missing:
        return False, ("The learned speech enhancer is not set up: "
                       + ", ".join(missing) + " not given. It runs as a "
                       "separate process in its own environment with a "
                       "hash-pinned checkpoint; the classical enhancers work "
                       "without it.")
    return SgmseRunner(python_exe, repo_dir, checkpoint, sha256).available()
