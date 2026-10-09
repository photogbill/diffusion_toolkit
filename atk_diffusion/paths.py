# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Where RF data lives: `D:\\Analyst_Toolkit\\rf_data\\`, one folder per
receiver profile, never mixed (plan §3.2, decision D2).

Bill, 2026-10-08: *"anything RF related should have separate storage areas
based on the SDR utilized"* and *"I like the rf_data folder idea."*

THE ROOT IS BESIDE THE CODE, NEVER IN A USER PROFILE. Bill's standing rule
for software written for him: recordings and data are never written to C:\\ or
to Windows user-profile folders (AppData and the rest) by default. So the
default root is found from where this code is, not from the home folder:

  * the toolkit beside ATK      D:\\Analyst_Toolkit\\ATK_Diffusion_Toolkit
                                  -> D:\\Analyst_Toolkit\\rf_data
  * the toolkit vendored in ATK D:\\Analyst_Toolkit\\ATK\\vendor\\diffusion_toolkit
                                  -> D:\\Analyst_Toolkit\\rf_data (the same one)
  * ATK_RF_DATA set              -> that folder (tests, a second drive)

A root inside AppData or under a sync tool (OneDrive, Dropbox, Google Drive,
iCloud) is REFUSED, not warned about: a sync client that uploads a 4 GB
capture half-written, or rewrites a file under a training run, corrupts the
one thing this folder is for (plan §3.2: "Never under a sync tool").

THE WRITE LOG. Every file the toolkit writes under the root is hashed into
`write_log.jsonl` at the moment it is written. A capture that changed after it
was recorded is NAMED, not used (plan §3.2). Verification is cheap in the
ordinary case — size and modification time first — and only re-hashes a file
whose size or time moved, because a full SHA-256 of a 10 GB capture before
every use would make the rule a reason to turn it off.

Standard library only, so ATK's recorder can share these rules without the
rest of the toolkit (ATK keeps its own copy, `atk/core/rf_data.py`, held to
this one by a contract test on both sides).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

ROOT_ENV = "ATK_RF_DATA"
ROOT_NAME = "rf_data"
WRITE_LOG = "write_log.jsonl"

#: Path segments that make a root unacceptable, compared case-insensitively.
#: AppData is Bill's rule; the rest are sync tools (plan §3.2).
FORBIDDEN_SEGMENTS = ("appdata", "onedrive", "dropbox", "google drive",
                      "googledrive", "icloud drive", "iclouddrive", "box sync")

README_TEXT = """\
rf_data — RF captures, datasets, models and products for ATK and the ATK
Diffusion Toolkit.

ONE FOLDER PER RECEIVER PROFILE, AND PROFILES NEVER MIX.
A receiver profile is <family>_<sample rate>_<datatype>[_<variant>], for
example rtlsdr_2400000_cu8 or bladerf1_4000000_ci16. Everything learned about
RF is learned at one exact sample rate through one receiver's front end, so a
model trained on bladeRF captures is not offered for an RTL-SDR capture, and a
capture at another rate is never silently resampled. (Bill, 2026-10-08:
"anything RF related should have separate storage areas based on the SDR
utilized".)

  profiles\\<profile>.json     the receiver's measured impairments, data-sheet
                              limits, and the fixed STFT/FAM geometry
  environments\\<region>.json  what a region's spectrum looks like (plan §3.6)
  <profile>\\captures\\         SigMF pairs, as recorded
  <profile>\\cabled\\           calibration captures through a cable (plan §3.5)
  <profile>\\synthetic\\        generated datasets at this profile's exact rate
  <profile>\\datasets\\         training/validation/test splits + manifest.json
  <profile>\\models\\           weights + model card (a model without a card
                              does not load)
  <profile>\\runs\\             training and evaluation logs
  <profile>\\cuts\\             signal cuts: original + cleaned + analysis
  products\\                   maps, tracks, fingerprints — open formats
  shared\\                     class lists, label schemas — nothing per-profile

Install, update and uninstall of ATK or the toolkit never touch this folder.
Keep it off OneDrive/Dropbox and out of AppData. write_log.jsonl holds a hash
of every file written here; a file that changed afterwards is named, not used.
"""


class RootRefused(ValueError):
    """The chosen rf_data root breaks a storage rule. The message says which."""


def toolkit_root() -> Path:
    """The repository folder (the one holding `atk_diffusion\\`)."""
    return Path(__file__).resolve().parents[1]


def default_root() -> Path:
    """The rf_data folder this installation uses when nothing says otherwise.

    `ATK_RF_DATA` wins. Otherwise the folder beside the code's home: the
    toolkit's parent, or — when the toolkit is vendored inside ATK
    (`ATK\\vendor\\<repo>`) — ATK's parent, so both layouts share ONE root.
    """
    env = os.environ.get(ROOT_ENV, "").strip()
    if env:
        return Path(env)
    here = toolkit_root()
    parent = here.parent
    if parent.name.lower() == "vendor" and (parent.parent / "atk").is_dir():
        return parent.parent.parent / ROOT_NAME
    return parent / ROOT_NAME


def check_root(path) -> tuple[bool, str]:
    """(ok, why). A root is refused under AppData or a sync tool; a root on
    C:\\ or in the user profile is allowed only because somebody chose it, and
    the reason line says so (it is never the default)."""
    p = Path(path)
    try:
        resolved = p.resolve()
    except OSError:
        resolved = p
    parts = [s.lower() for s in resolved.parts]
    for seg in FORBIDDEN_SEGMENTS:
        if any(seg == part or part.startswith(seg + " -") for part in parts):
            if seg == "appdata":
                return False, (f"{p} is inside AppData. RF data is kept beside "
                               "the toolkit on the data drive, never in the "
                               "user profile.")
            return False, (f"{p} is inside a sync folder ({seg}). A sync "
                           "client can upload a half-written capture or "
                           "rewrite a file under a training run. Choose a "
                           "folder outside it.")
    note = ""
    try:
        home = Path.home().resolve()
        if resolved == home or home in resolved.parents:
            note = ("note: this is inside the user profile — allowed because "
                    "it was chosen, but the default is beside the code.")
    except (OSError, RuntimeError):
        pass
    drive = (resolved.drive or "").upper()
    if not note and drive == "C:":
        note = ("note: this is on C:\\ — allowed because it was chosen; the "
                "default is the data drive beside the code.")
    return True, note


class RfData:
    """The layout of one rf_data root (plan §3.2). Creating the object
    touches nothing; `ensure()` makes the root and its README."""

    def __init__(self, root=None, *, create: bool = False):
        self.root = Path(root) if root is not None else default_root()
        ok, why = check_root(self.root)
        if not ok:
            raise RootRefused(why)
        self.note = why
        self._log = WriteLog(self.root)
        if create:
            self.ensure()

    # -- the tree ------------------------------------------------------------
    def ensure(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in ("profiles", "environments", "products", "shared"):
            (self.root / sub).mkdir(exist_ok=True)
        readme = self.root / "README.txt"
        if not readme.exists():
            readme.write_text(README_TEXT, encoding="utf-8")
        return self.root

    def profiles_dir(self) -> Path:
        return self.root / "profiles"

    def profile_json(self, profile: str) -> Path:
        return self.profiles_dir() / f"{_safe(profile)}.json"

    def environments_dir(self) -> Path:
        return self.root / "environments"

    def profile_dir(self, profile: str) -> Path:
        return self.root / _safe(profile)

    def captures(self, profile: str) -> Path:
        return self.profile_dir(profile) / "captures"

    def labeled(self, profile: str, cls: str) -> Path:
        """Taught examples of one class (plan B5): captures\\labeled\\<class>."""
        return self.captures(profile) / "labeled" / _safe(cls)

    def cabled(self, profile: str) -> Path:
        return self.profile_dir(profile) / "cabled"

    def synthetic(self, profile: str, dataset: str = "") -> Path:
        base = self.profile_dir(profile) / "synthetic"
        return base / _safe(dataset) if dataset else base

    def datasets(self, profile: str, dataset: str = "") -> Path:
        base = self.profile_dir(profile) / "datasets"
        return base / _safe(dataset) if dataset else base

    def models(self, profile: str, model: str = "") -> Path:
        base = self.profile_dir(profile) / "models"
        return base / _safe(model) if model else base

    def runs(self, profile: str) -> Path:
        return self.profile_dir(profile) / "runs"

    def cuts(self, profile: str) -> Path:
        return self.profile_dir(profile) / "cuts"

    def products(self, kind: str = "") -> Path:
        base = self.root / "products"
        return base / _safe(kind) if kind else base

    def shared(self) -> Path:
        return self.root / "shared"

    def known_profiles(self) -> list[str]:
        """Profiles that have a folder or a profile file, sorted."""
        out = set()
        if self.root.is_dir():
            for d in self.root.iterdir():
                if d.is_dir() and d.name not in ("profiles", "environments",
                                                 "products", "shared"):
                    out.add(d.name)
        if self.profiles_dir().is_dir():
            out.update(p.stem for p in self.profiles_dir().glob("*.json"))
        return sorted(out)

    # -- the write log -------------------------------------------------------
    @property
    def log(self) -> "WriteLog":
        return self._log

    def record(self, path, kind: str, note: str = "") -> dict:
        return self._log.record(path, kind, note)

    def verify(self, path) -> tuple[bool, str]:
        return self._log.verify(path)


def _safe(name: str) -> str:
    """A path component that cannot climb out of the root."""
    s = str(name).strip().replace("\\", "_").replace("/", "_")
    s = s.replace("..", "_")
    if not s or s in (".", ""):
        raise ValueError("an empty name cannot be a folder")
    return s


# ---------------------------------------------------------------------------
def sha256_file(path, chunk: int = 4 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class WriteLog:
    """Append-only JSON lines: one entry per file written under the root.

    The newest entry for a path wins (a re-generated dataset is written
    again, deliberately). Paths are stored RELATIVE to the root, so the log
    survives the folder moving to another drive.
    """

    _lock = threading.Lock()

    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / WRITE_LOG

    def _rel(self, path) -> str:
        p = Path(path).resolve()
        try:
            return p.relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return p.as_posix()

    def record(self, path, kind: str, note: str = "") -> dict:
        p = Path(path)
        st = p.stat()
        entry = {"path": self._rel(p), "kind": str(kind),
                 "sha256": sha256_file(p), "size": st.st_size,
                 "mtime_ns": st.st_mtime_ns,
                 "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime()),
                 "note": str(note)[:500]}
        self.root.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        return entry

    def entries(self) -> dict:
        """path -> newest entry."""
        out: dict = {}
        if not self.path.exists():
            return out
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue      # a torn last line from a crash: skip it
                if isinstance(e, dict) and e.get("path"):
                    out[e["path"]] = e
        return out

    def entry(self, path) -> dict | None:
        return self.entries().get(self._rel(path))

    def verify(self, path, full: bool = False) -> tuple[bool, str]:
        """(ok, why). Unlogged files are reported as such — "never recorded"
        is a different fact from "changed since it was recorded"."""
        p = Path(path)
        e = self.entry(p)
        if e is None:
            return False, f"{p.name} is not in the write log — it was not " \
                          "written by ATK or the toolkit, or was added by hand."
        if not p.exists():
            return False, f"{p.name} was recorded but is missing now."
        st = p.stat()
        if not full and st.st_size == e.get("size") \
                and st.st_mtime_ns == e.get("mtime_ns"):
            return True, ""
        if st.st_size != e.get("size"):
            return False, (f"{p.name} changed after it was recorded "
                           f"({e.get('size')} bytes then, {st.st_size} now). "
                           "It is named, not used.")
        if sha256_file(p) != e.get("sha256"):
            return False, (f"{p.name} changed after it was recorded (its "
                           "contents no longer match the hash written at the "
                           "time). It is named, not used.")
        return True, "contents match; only the file time moved"
