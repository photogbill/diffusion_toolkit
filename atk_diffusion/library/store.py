# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The training library: ONE folder of training examples per signal type,
per receiver profile (ATK's Setup → RFMLS Training, 2026-10-26).

Bill, 2026-10-26: *"It should deposit all training files into a single
training folder by signal type."* And his standing rule (2026-10-08):
*"anything RF related should have separate storage areas based on the SDR
utilized"* — so the single folder is per receiver profile:

    <rf_data>\\<profile>\\training\\
      README.txt
      <type>\\type.json        what the type is (label, family, the class it
                               is a kind of, its rate class, notes)
      <type>\\originals\\      REAL signals cut from recordings — the record
      <type>\\dirty\\          dirty copies: an original through simulated
                               channel and receiver effects
      <type>\\fresh\\          fresh copies: re-made by the native generator
                               from the type's measured parameters
      <type>\\generated\\      AI-generated variations (a diffusion model)
      _removed\\<type>\\<kind>\\  examples taken out — Restore puts them back

Every example is a SigMF pair at the profile's CANONICAL rate for the type's
bandwidth class — the rate the live detector cuts that signal to (DETECTION_
DESIGN §4, D2) — so a classifier trained from this folder meets, in the
field, exactly the kind of cut it learned on. The cut is made the way the
field cut is (`dsp.resample.cut_to_canonical`: shift to 0 Hz, low-pass to the
box with a 1.25 guard, INTEGER decimation), and then made again around the
signal's MEASURED 99 % band, because the live proposer draws its box around
the signal, not around the operator's generous rectangle.

ROLES — HOW THE GRADE STAYS HONEST. A model graded on copies of its own
training examples scores well and means nothing. So every real original has
a role, set when it is added and changeable by the operator:

  * the 2nd, 7th, 12th … original of a type is kept aside for TEST and the
    4th, 9th, 14th … for VALIDATION (calibration); the rest TRAIN;
  * a TRAIN original long enough for it (`HOLDOUT_MIN_SAMPLES`) keeps its
    last 20 % back as a held-out segment, so even a type with one recording
    gets a real (if weak: same transmission) test;
  * copies are made ONLY from the train span of TRAIN originals, and the
    dataset builder drops any copy whose parent is not (now) a TRAIN
    original — so flipping a role can never leak a copy into the grade.

THE ORIGINAL IS NEVER CHANGED. Every file is written to a temporary name and
moved into place, then hashed into the rf_data write log. Removing an
example moves it to `_removed\\` (nothing is deleted here), and Restore moves
it back.

LIMITS, stated. The library cannot know whether an operator's label is right
— a wrong label teaches a wrong answer with full confidence; the tab says
so. One recording of one transmitter is a narrow view of a signal type: the
copies teach tolerance of dirt, not the variety of real transmitters.
"""

from __future__ import annotations

import json
import os
import math
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from atk_diffusion import profiles as _profiles
from atk_diffusion import provenance as _prov
from atk_diffusion import sigmf as _sigmf
from atk_diffusion.detect import classes as _classes

VERSION = 1
TYPE_FILE = "type.json"
REMOVED = "_removed"
README = "README.txt"

#: The four kinds of example, in the order the tab shows them.
KINDS = ("originals", "dirty", "fresh", "generated")

#: What an operator calls each kind (the tab's chips use these words).
KIND_WORDS = {"originals": "real", "dirty": "dirty copy",
              "fresh": "fresh copy", "generated": "AI-generated"}

#: Each kind's provenance method -> tier (provenance.METHOD_TIERS).
KIND_METHOD = {"originals": "training_original", "dirty": "dirty_copy",
               "fresh": "fresh_copy", "generated": "generated_copy"}
_prov.METHOD_TIERS.setdefault("training_original", "record")
_prov.METHOD_TIERS.setdefault("dirty_copy", "invented")
_prov.METHOD_TIERS.setdefault("fresh_copy", "invented")
_prov.METHOD_TIERS.setdefault("generated_copy", "invented")

#: The id prefix of each kind's files (orig_0001, orig_0001_d00001, …).
_KIND_TAG = {"dirty": "d", "fresh": "f", "generated": "g"}

ROLES = ("train", "val", "test")
ROLE_WORDS = {"train": "used for training",
              "val": "kept aside to calibrate the model",
              "test": "kept aside to grade the model"}

#: The classifier's window (synth.datasets.DEFAULT_WINDOW): examples are at
#: least this long once padded, and a holdout segment holds at least one.
WINDOW = 4096
#: A train original holds a segment back only when it is at least this long
#: (four windows): one window held out, three or more left to grow from.
HOLDOUT_MIN_SAMPLES = 4 * WINDOW
HOLDOUT_FRACTION = 0.2
#: The shortest original worth keeping, at its canonical rate.
MIN_SAMPLES = 256
#: The longest original kept (samples at the canonical rate); a longer box is
#: cut to this and the meta says so. 2^22 cf32 = 32 MB.
MAX_SAMPLES = 1 << 22
#: Seconds of margin each side of a box (the cut's own rule: the larger of
#: this and 10 % of the box).
MIN_MARGIN_S = 0.05
#: Re-cut around the measured band only when it is at least this many STFT
#: bins wide — narrower is a carrier the measurement cannot shape.
MIN_BOX_BINS = 2

_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,59}$")

README_TEXT = """\
training — ATK's training library for this receiver (Setup -> RFMLS Training).

ONE FOLDER PER SIGNAL TYPE. Inside each:
  originals\\   REAL signals cut from recordings. These are the record and
               are never changed.
  dirty\\       dirty copies: a real original with simulated noise, fading,
               frequency error, clipping, interference and dropouts added.
  fresh\\       fresh copies: the signal re-made from its measured parameters.
  generated\\   AI-generated variations made by a diffusion model.
  type.json    what the signal type is.

Every file is a SigMF pair (.sigmf-data + .sigmf-meta) that GNU Radio,
inspectrum and SigDigger open directly. Copies say in their .sigmf-meta which
original they came from and exactly what was done to them (atk:effects).

Some real originals are KEPT ASIDE (atk:role = test or val): they grade the
model and are never copied or trained on. That is what keeps the grade honest.

_removed\\ holds examples taken out of the library; ATK can put them back.
Everything here is per receiver: a bladeRF's library does not train an
RTL-SDR's model.
"""


class LibraryError(ValueError):
    """The library will not do this; the message says why, in words."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def slug(text: str) -> str:
    """A type name from what an operator typed: lower case, spaces to '_',
    anything else that a folder name cannot hold dropped."""
    s = re.sub(r"[^a-z0-9_-]+", "_", str(text or "").strip().lower())
    s = re.sub(r"_+", "_", s).strip("_-")
    return s[:60]


def role_for_index(i: int) -> str:
    """The role of the i-th original of a type (1-based): 2, 7, 12 … test;
    4, 9, 14 … val; the rest train. With two originals one grades the other;
    with four there is one of each kept aside."""
    r = int(i) % 5
    return "test" if r == 2 else ("val" if r == 4 else "train")


# ---------------------------------------------------------------------------
# signal types
# ---------------------------------------------------------------------------
@dataclass
class SignalType:
    name: str                       # the folder name (a slug)
    label: str = ""                 # what an analyst calls it
    family: str = "unknown"         # one of detect.classes.FAMILIES
    link: str = ""                  # a class-table class it is a kind of
    canonical: str = ""             # voice | wideband | spread ('' until known)
    notes: str = ""
    created: str = ""
    created_by: str = ""
    next_index: int = 1             # the next original's number
    version: int = VERSION

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "SignalType":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in dict(d).items() if k in known})

    @property
    def words(self) -> str:
        return self.label or self.name

    def linked_class(self):
        return _classes.get(self.link) if self.link else None


def _check_family(family: str) -> str:
    f = str(family or "unknown").strip().lower()
    if f not in _classes.FAMILIES:
        raise LibraryError(f"'{family}' is not a signal family ATK knows "
                           f"({', '.join(_classes.FAMILY_WORDS[x] for x in _classes.FAMILIES)})")
    return f


def _check_link(link: str) -> str:
    k = str(link or "").strip()
    if k and _classes.get(k) is None:
        raise LibraryError(f"'{link}' is not a signal class in ATK's class "
                           "table; leave it empty for a type of your own")
    if k and _classes.get(k).negative:
        raise LibraryError(f"'{link}' is a negative class (noise, a spur, the "
                           "DC spike); the recognizer learns noise by itself")
    return k


# ---------------------------------------------------------------------------
# examples
# ---------------------------------------------------------------------------
@dataclass
class Example:
    """One example on disk, with the facts the tab lists (read from its
    .sigmf-meta; the samples are read only by `TrainingLibrary.load`)."""
    id: str
    kind: str
    type: str
    base: str                       # the SigMF base path (no extension)
    samples: int = 0
    rate: float = 0.0
    role: str = ""                  # originals only
    parent: str = ""                # copies: the original's id ('' for fresh)
    tier: str = ""
    method: str = ""
    snr_db: float | None = None
    bandwidth_hz: float | None = None
    created: str = ""
    note: str = ""
    removed: bool = False
    holdout: dict | None = None     # originals: {start, count} or None
    effects: list = field(default_factory=list)
    #: originals: where the signal is inside the cut — the drawn box, trimmed
    #: to where it stands above the noise around it — {start, count}; None
    #: means the whole original (an original filed before 2026-10-26 late)
    signal_span: dict | None = None
    #: originals: the recording it was cut from. Every original of one
    #: recording shares ONE role, so a recording that trains never grades.
    group: str = ""

    @property
    def seconds(self) -> float:
        return self.samples / self.rate if self.rate else 0.0

    def to_json(self) -> dict:
        return asdict(self)


def _example_from_meta(base: Path, kind: str, tname: str, meta: dict,
                       removed: bool = False) -> Example:
    g = meta.get("global", {})
    try:
        n = _sigmf.num_samples(base, meta)
    except OSError:
        n = 0
    meas = g.get("atk:measured") or {}
    snr = g.get("atk:snr_db", meas.get("snr_db"))
    bw = meas.get("bandwidth_hz")
    eff = g.get("atk:effects") or []
    return Example(id=base.name, kind=kind, type=tname, base=str(base),
                   samples=int(n), rate=float(g.get("core:sample_rate", 0.0)),
                   role=str(g.get("atk:role", "") or ""),
                   parent=str(g.get("atk:parent", "") or ""),
                   tier=str(g.get("atk:tier", "") or ""),
                   method=str(g.get("atk:method", "") or ""),
                   snr_db=None if snr is None else float(snr),
                   bandwidth_hz=None if bw is None else float(bw),
                   created=str(g.get("atk:created", "") or ""),
                   note=str(g.get("atk:note", "") or ""), removed=removed,
                   holdout=g.get("atk:holdout"),
                   effects=[str(e.get("words", e.get("name", "")))
                            for e in eff if isinstance(e, dict)],
                   signal_span=g.get("atk:signal_span"),
                   group=str(g.get("atk:group", "") or g.get("atk:source_cut", "")
                             or g.get("atk:source_capture", "") or base.name))


# ---------------------------------------------------------------------------
# the library
# ---------------------------------------------------------------------------
class TrainingLibrary:
    """The training folder of one receiver profile. Creating the object
    touches nothing; the first write makes the folder and its README."""

    def __init__(self, rf, profile):
        self.rf = rf
        if isinstance(profile, _profiles.ReceiverProfile):
            self.profile = profile
        else:
            pid = str(profile).strip().lower()
            if not _profiles.is_profile_id(pid):
                raise LibraryError(f"'{profile}' is not a receiver profile id "
                                   "(for example rtlsdr_2400000_cu8)")
            self.profile = _profiles.load_profile(rf, pid)
        self.pid = self.profile.id
        self.fs = float(self.profile.sample_rate)
        self.root = Path(rf.training(self.pid))

    # -- the folder ----------------------------------------------------------
    def ensure(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        readme = self.root / README
        if not readme.exists():
            readme.write_text(README_TEXT, encoding="utf-8")
        return self.root

    def type_dir(self, name: str) -> Path:
        return self.root / str(name)

    def kind_dir(self, name: str, kind: str) -> Path:
        if kind not in KINDS:
            raise LibraryError(f"kind is one of {', '.join(KINDS)}")
        return self.type_dir(name) / kind

    def removed_dir(self, name: str, kind: str) -> Path:
        return self.root / REMOVED / str(name) / kind

    def canonical_rates(self) -> dict:
        return {c.cls: c for c in self.profile.canonical_rates()}

    def rate_of(self, cls: str) -> _profiles.CanonicalRate:
        rates = self.canonical_rates()
        if cls not in rates:
            raise LibraryError(f"{_profiles.describe(self.pid)} has no "
                               f"{cls} rate class")
        return rates[cls]

    # -- types ---------------------------------------------------------------
    def types(self) -> list[SignalType]:
        out = []
        if not self.root.is_dir():
            return out
        for d in sorted(self.root.iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            tf = d / TYPE_FILE
            if not tf.exists():
                continue
            try:
                out.append(SignalType.from_json(json.loads(
                    tf.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, TypeError, OSError):
                continue
        return out

    def get_type(self, name: str) -> SignalType:
        tf = self.type_dir(name) / TYPE_FILE
        if not tf.exists():
            raise LibraryError(f"there is no signal type '{name}' in the "
                               f"library of {_profiles.describe(self.pid)}")
        try:
            return SignalType.from_json(json.loads(tf.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError) as e:
            raise LibraryError(f"the type file of '{name}' could not be read "
                               f"({e})") from None

    def _save_type(self, st: SignalType) -> Path:
        d = self.type_dir(st.name)
        d.mkdir(parents=True, exist_ok=True)
        p = d / TYPE_FILE
        tmp = p.with_name(TYPE_FILE + ".tmp")
        tmp.write_text(json.dumps(st.to_json(), indent=2), encoding="utf-8")
        tmp.replace(p)
        try:
            self.rf.record(p, "training-type", f"{self.pid} {st.name}")
        except OSError:
            pass
        return p

    def new_type(self, name: str, label: str = "", family: str = "",
                 link: str = "", notes: str = "", who: str = "") -> SignalType:
        """A new signal type. `name` becomes the folder (letters, digits,
        '_' and '-'); `link` is the class-table class it is a kind of (its
        family, rate class and fresh copies then follow from it)."""
        nm = slug(name)
        if not nm or not _NAME.match(nm):
            raise LibraryError("a signal type needs a name of letters and "
                               "digits (for example county_fire_dmr)")
        if nm in (REMOVED.strip("_"), "noise", _classes.UNKNOWN.lower()):
            raise LibraryError(f"'{nm}' is reserved: the recognizer makes its "
                               "own noise and UNKNOWN answers")
        existing = {t.name.lower() for t in self.types()}
        if nm in existing:
            raise LibraryError(f"the library already has a type called '{nm}'")
        lk = _check_link(link)
        c = _classes.get(lk) if lk else None
        fam = _check_family(family or (c.family if c else "unknown"))
        canon = ""
        if c is not None and c.bandwidth_hz > 0:
            canon = _profiles.canonical_for(self.fs, c.bandwidth_hz).cls
        st = SignalType(name=nm, label=str(label or "").strip()[:80] or
                        (c.label if c else nm), family=fam, link=lk,
                        canonical=canon, notes=str(notes or "")[:2000],
                        created=_now(), created_by=who or _prov.who())
        self.ensure()
        self._save_type(st)
        for k in KINDS:
            self.kind_dir(nm, k).mkdir(parents=True, exist_ok=True)
        return st

    def update_type(self, name: str, *, label: str | None = None,
                    notes: str | None = None, family: str | None = None,
                    link: str | None = None) -> SignalType:
        st = self.get_type(name)
        if label is not None:
            st.label = str(label).strip()[:80] or st.name
        if notes is not None:
            st.notes = str(notes)[:2000]
        if family is not None:
            st.family = _check_family(family)
        if link is not None:
            lk = _check_link(link)
            c = _classes.get(lk) if lk else None
            if c is not None and c.bandwidth_hz > 0 and st.canonical:
                want = _profiles.canonical_for(self.fs, c.bandwidth_hz).cls
                if want != st.canonical:
                    raise LibraryError(
                        f"{c.label} is a {want}-width signal; the examples of "
                        f"'{st.name}' are {st.canonical}-width cuts. Make a "
                        "new type for it.")
            st.link = lk
            if c is not None and family is None:
                st.family = c.family
        self._save_type(st)
        return st

    def remove_type(self, name: str) -> Path:
        """Move the whole type (and everything in it) to _removed\\."""
        d = self.type_dir(name)
        if not (d / TYPE_FILE).exists():
            raise LibraryError(f"there is no signal type '{name}'")
        dest = self.root / REMOVED / f"{name}__type_{_stamp()}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(d), str(dest))
        return dest

    def removed_types(self) -> list[tuple[str, Path]]:
        base = self.root / REMOVED
        if not base.is_dir():
            return []
        return [(d.name.split("__type_")[0], d) for d in sorted(base.iterdir())
                if d.is_dir() and "__type_" in d.name
                and (d / TYPE_FILE).exists()]

    def restore_type(self, removed_folder) -> SignalType:
        src = Path(removed_folder)
        if not (src / TYPE_FILE).exists() or src.parent != self.root / REMOVED:
            raise LibraryError(f"{src} is not a removed signal type of this "
                               "library")
        st = SignalType.from_json(json.loads((src / TYPE_FILE).read_text("utf-8")))
        if self.type_dir(st.name).exists():
            raise LibraryError(f"a type called '{st.name}' exists again; "
                               "rename or remove it first")
        shutil.move(str(src), str(self.type_dir(st.name)))
        return st

    # -- examples ------------------------------------------------------------
    def examples(self, name: str, kind: str | None = None,
                 include_removed: bool = False) -> list[Example]:
        out = []
        for k in ([kind] if kind else KINDS):
            for removed, d in ((False, self.kind_dir(name, k)),
                               (True, self.removed_dir(name, k))):
                if removed and not include_removed:
                    continue
                if not d.is_dir():
                    continue
                for mp in sorted(d.glob("*.sigmf-meta")):
                    if mp.name.startswith("."):
                        continue        # a write that was stopped part-way
                    base = _sigmf.base_of(mp)
                    if not _sigmf.data_path(base).exists():
                        continue
                    try:
                        meta = _sigmf.read_meta(base)
                    except (OSError, json.JSONDecodeError):
                        continue
                    out.append(_example_from_meta(base, k, name, meta, removed))
        return out

    def example(self, name: str, ex_id: str) -> Example:
        for e in self.examples(name, include_removed=True):
            if e.id == ex_id:
                return e
        raise LibraryError(f"'{name}' has no example {ex_id}")

    def counts(self, name: str) -> dict:
        """Numbers the tab shows for a type."""
        ex = self.examples(name)
        out = {k: 0 for k in KINDS}
        roles = {r: 0 for r in ROLES}
        seconds = 0.0
        for e in ex:
            out[e.kind] += 1
            if e.kind == "originals":
                roles[e.role or "train"] = roles.get(e.role or "train", 0) + 1
                seconds += e.seconds
        out["roles"] = roles
        out["real_seconds"] = round(seconds, 3)
        out["copies"] = out["dirty"] + out["fresh"] + out["generated"]
        out["holdout_segments"] = sum(1 for e in ex if e.kind == "originals"
                                      and (e.role or "train") == "train"
                                      and e.holdout)
        return out

    def summary(self) -> list[dict]:
        out = []
        for st in self.types():
            out.append({"type": st.to_json(), "counts": self.counts(st.name)})
        return out

    def load(self, ex: Example, start: int = 0, count: int | None = None
             ) -> np.ndarray:
        x = _sigmf.load(ex.base, start, count)
        if x.ndim > 1:
            x = x[0]
        return np.asarray(x, dtype=np.complex64)

    def meta(self, ex: Example) -> dict:
        return _sigmf.read_meta(ex.base)

    def train_span(self, ex: Example) -> tuple[int, int]:
        """(start, count) of the samples copies may be made from: the whole
        original minus its held-out segment. Refuses a kept-aside original."""
        if ex.kind != "originals":
            raise LibraryError("copies are made from real originals")
        if (ex.role or "train") != "train":
            raise LibraryError(f"{ex.id} is {ROLE_WORDS[ex.role]} — it is never "
                               "copied, so it can grade the model honestly")
        s0, n = signal_span(ex)
        if ex.holdout:
            h0 = int(ex.holdout["start"])
            return s0, max(0, h0 - s0)
        return s0, n

    def recording_of(self, name: str, ex: Example) -> list[Example]:
        """Every original of `name` cut from the same recording as `ex`
        (removed ones included — they may be brought back)."""
        return [e for e in self.examples(name, "originals", include_removed=True)
                if e.group == ex.group]

    def set_role(self, name: str, ex_id: str, role: str) -> Example:
        """Set the role of an original — and of EVERY original cut from the
        same recording, so one recording never both trains and grades (two
        boxes on one capture can overlap in time, or share the same burst)."""
        if role not in ROLES:
            raise LibraryError(f"a role is one of {', '.join(ROLES)}")
        ex = self.example(name, ex_id)
        if ex.kind != "originals":
            raise LibraryError("only real originals have a role")
        for e in self.recording_of(name, ex):
            meta = _sigmf.read_meta(e.base)
            g = meta["global"]
            if g.get("atk:role") == role:
                continue
            g["atk:role"] = role
            if role == "train" and not g.get("atk:holdout"):
                sp = g.get("atk:signal_span") or {"start": 0, "count": int(e.samples)}
                hd = _holdout_for(int(sp["start"]), int(sp["count"]))
                if hd:
                    g["atk:holdout"] = hd
            hist = list(g.get("atk:history") or [])
            what = f"role -> {role}" + (f" (with {ex.id}: same recording)"
                                         if e.id != ex.id else "")
            hist.append({"at": _now(), "by": _prov.who(), "what": what})
            g["atk:history"] = hist[-50:]
            mp = _sigmf.write_meta(e.base, meta)
            try:
                self.rf.record(mp, "training-example-meta", f"{e.id} role {role}")
            except OSError:
                pass
        return self.example(name, ex_id)

    def remove_example(self, name: str, ex_id: str) -> Path:
        ex = self.example(name, ex_id)
        if ex.removed:
            raise LibraryError(f"{ex_id} is already removed")
        dest = self.removed_dir(name, ex.kind)
        dest.mkdir(parents=True, exist_ok=True)
        base = Path(ex.base)
        for p in (_sigmf.data_path(base), _sigmf.meta_path(base)):
            shutil.move(str(p), str(dest / p.name))
        return dest / base.name

    def restore_example(self, name: str, ex_id: str) -> Path:
        ex = self.example(name, ex_id)
        if not ex.removed:
            raise LibraryError(f"{ex_id} is not removed")
        dest = self.kind_dir(name, ex.kind)
        dest.mkdir(parents=True, exist_ok=True)
        base = Path(ex.base)
        for p in (_sigmf.data_path(base), _sigmf.meta_path(base)):
            if (dest / p.name).exists():
                raise LibraryError(f"{p.name} exists in the library again")
        for p in (_sigmf.data_path(base), _sigmf.meta_path(base)):
            shutil.move(str(p), str(dest / p.name))
        return dest / base.name

    # -- writing -------------------------------------------------------------
    def _write(self, folder: Path, ex_id: str, x: np.ndarray, rate: float,
               centre_hz: float, g: dict, description: str,
               annotations=None, t0_utc: float | None = None) -> Path:
        """Write a pair under a temporary name, move it into place, record
        both files in the write log. -> the base path."""
        folder.mkdir(parents=True, exist_ok=True)
        final = folder / ex_id
        if _sigmf.meta_path(final).exists() or _sigmf.data_path(final).exists():
            raise LibraryError(f"{final.name} already exists")
        tmp = folder / f".{ex_id}.partial"
        dp, mp = _sigmf.write_pair(tmp, np.asarray(x, np.complex64), rate,
                                   centre_hz, datatype="cf32",
                                   extra_global=g, description=description,
                                   recorder="ATK RFMLS Training",
                                   annotations=annotations, t0_utc=t0_utc)
        # the meta names its data file by the base: rename data first
        dp.replace(_sigmf.data_path(final))
        mp.replace(_sigmf.meta_path(final))
        note = f"{self.pid} training {g.get('atk:training_type', '')} {ex_id}"
        for p in (_sigmf.data_path(final), _sigmf.meta_path(final)):
            try:
                self.rf.record(p, "training-example", note)
            except OSError:
                pass
        return final

    def _role_for_new(self, st: SignalType, group: str,
                      box_abs: tuple | None) -> tuple[str, int]:
        """(role, recording number) for a new original of `st` from the
        recording `group`. A recording already in the library keeps its role
        — all its originals train together or grade together — and a box
        overlapping one already cut from it (same time AND same frequencies)
        is refused. A new recording gets the next recording number's role
        (`role_for_index`: the 2nd and 4th of every five are kept aside)."""
        groups: dict = {}
        for e in self.examples(st.name, "originals", include_removed=True):
            groups.setdefault(e.group, []).append(e)
        same = groups.get(group, [])
        if box_abs is not None:
            a0, a1, f0, f1 = box_abs
            for e in same:
                g = self.meta(e).get("global", {})
                b = g.get("atk:source_box_samples")
                bx = g.get("atk:box") or {}
                if not b or not bx:
                    continue
                if (a0 < int(b[1]) and int(b[0]) < a1
                        and f0 < float(bx.get("f_hi_hz", 0)) and
                        float(bx.get("f_lo_hz", 0)) < f1):
                    raise LibraryError(
                        f"this box overlaps {e.id}, already cut from the same "
                        "recording at the same time and frequency"
                        + (" (removed — bring it back instead)" if e.removed else "")
                        + ". Each stretch of a recording goes in once.")
        if same:
            live = [e for e in same if not e.removed] or same
            roles = [e.role or "train" for e in live]
            role = max(set(roles), key=roles.count)
            gidx = int(self.meta(live[0]).get("global", {}).get(
                "atk:group_index", 0) or 0)
            return role, gidx
        gidx = len(groups) + 1
        return role_for_index(gidx), gidx

    def _next_original_id(self, st: SignalType) -> tuple[str, int]:
        i = max(1, int(st.next_index))
        taken = {e.id for e in self.examples(st.name, "originals",
                                             include_removed=True)}
        while f"orig_{i:04d}" in taken:
            i += 1
        return f"orig_{i:04d}", i

    def next_copy_ids(self, name: str, kind: str, parent: str, n: int
                      ) -> list[str]:
        """n unused ids for copies of `parent` ('' for fresh copies)."""
        tag = _KIND_TAG[kind]
        prefix = f"{parent}_{tag}" if parent else f"fresh_{_stamp()}_{tag}"
        # names only — reading every copy's meta here made filing N copies
        # cost N² file reads
        taken = set()
        for d in (self.kind_dir(name, kind), self.removed_dir(name, kind)):
            if d.is_dir():
                taken.update(_sigmf.base_of(p).name
                             for p in d.glob(prefix + "*.sigmf-meta"))
        out, i = [], 1
        while len(out) < int(n):
            cand = f"{prefix}{i:05d}"
            if cand not in taken:
                out.append(cand)
            i += 1
        return out

    def write_copy(self, name: str, kind: str, ex_id: str, x: np.ndarray,
                   rate: float, centre_hz: float, extra: dict,
                   description: str = "") -> Path:
        """Write one copy (dirty / fresh / generated) with its facts."""
        if kind not in _KIND_TAG:
            raise LibraryError("copies are dirty, fresh or generated")
        st = self.get_type(name)
        g = {"atk:receiver_profile": self.pid,
             "atk:tier": _prov.tier_for(KIND_METHOD[kind]),
             "atk:method": KIND_METHOD[kind], "atk:training_type": st.name,
             "atk:training_kind": kind, "atk:created": _now(),
             "atk:canonical_class": st.canonical,
             "atk:library_version": VERSION}
        g.update(extra or {})
        return self._write(self.kind_dir(name, kind), ex_id, x, rate, centre_hz,
                           g, description or f"{KIND_WORDS[kind]} of {st.words}")

    # -- adding real originals ----------------------------------------------
    def add_original_from_iq(self, x, fs: float, center_hz: float, box: dict,
                             type_name: str, *, source: dict | None = None,
                             who: str = "", note: str = "",
                             t0_utc: float | None = None,
                             recut: bool = True) -> Example:
        """Cut `box` = {t0_s, t1_s, f_lo_hz, f_hi_hz} (times relative to
        x[0], frequencies absolute) out of IQ at the profile's rate, re-cut
        around the signal's measured band, measure it, and file it as a REAL
        original of `type_name`. Refuses in words: a wrong rate, a box
        outside the IQ, a signal of another rate class than the type's, too
        little signal."""
        from atk_diffusion.dsp import measure as _m
        from atk_diffusion.dsp.resample import cut_to_canonical
        st = self.get_type(type_name)
        fs = float(fs)
        if abs(fs - self.fs) > 0.5:
            raise _profiles.ProfileMismatch(
                f"this IQ is at {fs:,.0f} S/s; the library is for "
                f"{_profiles.describe(self.pid)}. Examples are made at the "
                "receiver's own rate — pick the receiver this was recorded on.")
        xa = np.asarray(x)
        if xa.ndim > 1:
            xa = xa[0]
        n = xa.size
        t0, t1, f_lo, f_hi = _parse_box(box)
        dur = n / fs
        if t1 <= 0 or t0 >= dur:
            raise LibraryError(f"the box ({t0:.3f}–{t1:.3f} s) is outside the "
                               f"IQ given (0–{dur:.3f} s)")
        half = 0.5 * fs
        c = float(center_hz)
        if f_lo < c - half - 1 or f_hi > c + half + 1:
            raise LibraryError("the box reaches outside what the receiver saw "
                               f"({(c - half) / 1e6:.4f}–{(c + half) / 1e6:.4f} MHz)")
        m = max(MIN_MARGIN_S, 0.1 * (t1 - t0))
        s0 = max(0, int(math.floor((t0 - m) * fs)))
        s1 = min(n, int(math.ceil((t1 + m) * fs)))
        seg = np.asarray(xa[s0:s1], dtype=np.complex64)
        if not np.all(np.isfinite(seg)):
            raise LibraryError("the IQ holds samples that are not finite (a "
                               "damaged file or a failed read)")
        f_mid = 0.5 * (f_lo + f_hi)
        f_off = f_mid - c
        box_bw = max(1.0, f_hi - f_lo)
        # the floor of the SOURCE span (most of which is noise), per Hz — a
        # cut is low-passed to its signal, so its own median bin is signal
        # (the cut module's rule: atk:floor_per_hz is the source's)
        nf_src = _m.noise_floor(seg, fs, nfft=int(getattr(
            self.profile.stft, "fft_size", 1024)))
        floor = float(nf_src.get("floor_per_hz") or 0.0) or None
        y, fs_c, info = cut_to_canonical(seg, fs, f_off, box_bw)
        measured = _measure(y, fs_c, floor)
        notes = []
        rbw = self.profile.stft.rbw_hz(fs)
        ob = measured.get("bandwidth_hz")
        oc = measured.get("centre_hz")
        if recut and ob and oc is not None and ob >= MIN_BOX_BINS * rbw \
                and ob <= 1.05 * box_bw:
            f_off2 = f_off + float(oc)
            # stay inside what was drawn: the measured band, never wider
            y2, fs_c2, info2 = cut_to_canonical(seg, fs, f_off2, float(ob))
            measured2 = _measure(y2, fs_c2, floor)
            if measured2.get("bandwidth_hz"):
                y, fs_c, info, measured = y2, fs_c2, info2, measured2
                f_off = f_off2
                notes.append(f"re-cut around the measured 99 % band "
                             f"({ob:,.0f} Hz wide) — the box the detector draws")
        elif recut:
            notes.append("kept the drawn box: the signal's band could not be "
                         "measured inside it" if not ob else
                         "kept the drawn box: the measured band is narrower "
                         "than two FFT bins")
        cls = info["canonical_class"]
        if st.canonical and cls != st.canonical:
            words = {"voice": "voice-width (up to 30 kHz)",
                     "wideband": "wideband (30–400 kHz)",
                     "spread": "very wide (over 400 kHz)"}
            raise LibraryError(
                f"this signal measures {float(measured.get('bandwidth_hz') or box_bw):,.0f} Hz "
                f"wide — a {words.get(cls, cls)} signal; '{st.words}' holds "
                f"{words.get(st.canonical, st.canonical)} signals. It is either "
                "a different signal type, or the box missed part of it.")
        if y.size < MIN_SAMPLES:
            raise LibraryError(
                f"the cut holds only {y.size} samples at {fs_c:,.0f} S/s "
                f"({1e3 * y.size / fs_c:.1f} ms); draw the box over a longer "
                "stretch of the signal")
        truncated = False
        if y.size > MAX_SAMPLES:
            y = y[:MAX_SAMPLES]
            truncated = True
            notes.append(f"the box was longer than {MAX_SAMPLES:,} samples at "
                         f"this rate; the first {MAX_SAMPLES / fs_c:.1f} s were kept")
        if measured.get("snr_db") is None:
            notes.append("the SNR could not be measured: the signal is within "
                         "the floor's own uncertainty")
        dec = max(1, int(info["decimation"]))
        span, span_note = _active_span(
            y, int(round((t0 * fs - s0) / dec)), int(round((t1 * fs - s0) / dec)),
            fs_c)
        notes.append(span_note)
        src = dict(source or {})
        sample0 = int(src.get("sample_offset", 0) or 0)
        group = _group_key(src)
        box_abs = (sample0 + int(round(t0 * fs)), sample0 + int(round(t1 * fs)),
                   float(f_lo), float(f_hi))
        ex_id, idx = self._next_original_id(st)
        role, gidx = self._role_for_new(st, group or f"iq:{ex_id}", box_abs)
        hd = _holdout_for(span["start"], span["count"]) if role == "train" else None
        g = {"atk:receiver_profile": self.pid, "atk:tier": "record",
             "atk:method": KIND_METHOD["originals"],
             "atk:training_type": st.name, "atk:training_kind": "originals",
             "atk:role": role, "atk:holdout": hd, "atk:index": idx,
             "atk:group": group or f"iq:{ex_id}", "atk:group_index": gidx,
             "atk:signal_span": span,
             "atk:source_box_samples": [box_abs[0], box_abs[1]],
             "atk:canonical_class": cls,
             "atk:canonical_rate": float(fs_c),
             "atk:decimation": int(info["decimation"]),
             "atk:lowpass_hz": float(info["lowpass_hz"]),
             "atk:canonical_limited": bool(info["limited"]),
             "atk:f_offset_hz": float(f_off),
             "atk:source_center_hz": c, "atk:source_sample_rate": fs,
             "atk:source_capture": str(src.get("capture", "") or ""),
             "atk:source_sample_start": int(src.get("sample_offset", 0) or 0) + s0,
             "atk:source_sample_count": int(s1 - s0),
             "atk:source_kind": str(src.get("kind", "capture") or "capture"),
             "atk:box": {"t0_s": t0, "t1_s": t1, "f_lo_hz": f_lo, "f_hi_hz": f_hi},
             "atk:margin_s": m, "atk:measured": measured,
             "atk:snr_db": measured.get("snr_db"),
             "atk:truncated": truncated, "atk:created": _now(),
             "atk:cut_by": who or _prov.who(), "atk:notes": notes,
             "atk:library_version": VERSION}
        if note:
            g["atk:note"] = str(note)[:500]
        if src.get("cut_folder"):
            g["atk:source_cut"] = str(src["cut_folder"])
        centre_abs = c + f_off
        wall = None
        if src.get("epoch") is not None:
            wall = float(src["epoch"]) + (int(src.get("sample_offset", 0) or 0)
                                          + s0) / fs
        base = self._write(self.kind_dir(st.name, "originals"), ex_id, y, fs_c,
                           centre_abs, g,
                           f"real example of {st.words} ({self.pid})",
                           t0_utc=wall)
        if not st.canonical:
            st.canonical = cls
        st.next_index = idx + 1
        self._save_type(st)
        return _example_from_meta(base, "originals", st.name,
                                  _sigmf.read_meta(base))

    def add_original_from_capture(self, capture, box: dict, type_name: str,
                                  who: str = "", note: str = "") -> Example:
        """Cut a box out of a recorded SigMF capture of this receiver (box
        times relative to the capture's first sample)."""
        try:
            meta = _sigmf.read_meta(capture)
        except FileNotFoundError:
            raise LibraryError(f"there is no recording at "
                               f"{_sigmf.base_of(capture)} (no .sigmf-meta)") from None
        except json.JSONDecodeError as e:
            raise LibraryError(f"the recording's .sigmf-meta is not valid JSON "
                               f"({e})") from None
        fs = _sigmf.sample_rate_of(meta)
        try:
            pid = _profiles.profile_from_meta(meta)
        except ValueError:
            pid = ""
        if pid and pid != self.pid:
            raise _profiles.ProfileMismatch(
                f"this recording was made on {_profiles.describe(pid)} "
                f"({pid}); the library open is for "
                f"{_profiles.describe(self.pid)}. Pick that receiver first — "
                "every receiver has its own library.")
        if abs(fs - self.fs) > 0.5:
            raise _profiles.ProfileMismatch(
                f"this recording is at {fs:,.0f} S/s; the library is for "
                f"{_profiles.describe(self.pid)}.")
        total = _sigmf.num_samples(capture, meta)
        t0, t1, _flo, _fhi = _parse_box(box)
        m = max(MIN_MARGIN_S, 0.1 * (t1 - t0))
        s0 = max(0, int(math.floor((t0 - m) * fs)))
        s1 = min(total, int(math.ceil((t1 + m) * fs)))
        if s1 <= s0:
            raise LibraryError(f"the box ({t0:.3f}–{t1:.3f} s) is outside the "
                               f"recording (0–{total / fs:.3f} s)")
        centres = {_sigmf.center_of(meta, s) for s in (s0, s1 - 1)}
        for cap in meta.get("captures", []) or []:
            stt = int(cap.get("core:sample_start", 0))
            if s0 < stt < s1:
                centres.add(float(cap.get("core:frequency", 0.0)))
        if len(centres) > 1:
            raise LibraryError("the radio was retuned inside that box — draw "
                               "it on one side of the retune")
        centre = centres.pop()
        x = _sigmf.load(capture, start=s0, count=s1 - s0, meta=meta)
        if x.ndim > 1:
            x = x[0]
        epoch = None
        try:
            from datetime import datetime
            dt = (meta.get("captures") or [{}])[0].get("core:datetime")
            if dt:
                epoch = datetime.fromisoformat(str(dt).replace("Z", "+00:00")
                                               ).timestamp()
        except (ValueError, TypeError):
            epoch = None
        seg_box = {"t0_s": t0 - s0 / fs, "t1_s": t1 - s0 / fs,
                   "f_lo_hz": float(box["f_lo_hz"]),
                   "f_hi_hz": float(box["f_hi_hz"])}
        src = {"capture": str(_sigmf.base_of(capture)), "sample_offset": s0,
               "epoch": epoch, "kind": "capture"}
        ex = self.add_original_from_iq(x, fs, centre, seg_box, type_name,
                                       source=src, who=who, note=note)
        # the box as the operator drew it, in the capture's own times
        meta2 = _sigmf.read_meta(ex.base)
        meta2["global"]["atk:box"] = {"t0_s": t0, "t1_s": t1,
                                      "f_lo_hz": float(box["f_lo_hz"]),
                                      "f_hi_hz": float(box["f_hi_hz"])}
        mp = _sigmf.write_meta(ex.base, meta2)
        try:
            self.rf.record(mp, "training-example-meta", ex.id)
        except OSError:
            pass
        return ex

    def add_original_from_cut(self, cut_folder, type_name: str, who: str = "",
                              note: str = "") -> Example:
        """A signal cut (RF → Cuts) into the library. Its source capture is
        re-cut when it is still there (so the band is the measured one);
        otherwise the cut's own original is used as it stands."""
        d = Path(cut_folder)
        base = d / "original"
        if not _sigmf.meta_path(base).exists():
            raise LibraryError(f"{d} is not a signal cut folder (no "
                               "original.sigmf-meta)")
        for e in self.examples(type_name, "originals", include_removed=True):
            g0 = self.meta(e).get("global", {})
            if _same_path(g0.get("atk:source_cut"), d):
                raise LibraryError(f"that cut is already in '{type_name}' as "
                                   f"{e.id}" + (" (removed — bring it back "
                                                "instead)" if e.removed else ""))
        meta = _sigmf.read_meta(base)
        g = meta.get("global", {})
        pid = str(g.get("atk:receiver_profile", "") or "")
        if pid and pid != self.pid:
            raise _profiles.ProfileMismatch(
                f"this cut was made on {_profiles.describe(pid)}; the library "
                f"open is for {_profiles.describe(self.pid)}.")
        src_cap = str(g.get("atk:source_capture", "") or "")
        box = g.get("atk:box") or {}
        if src_cap and box and _sigmf.meta_path(src_cap).exists():
            try:
                ex = self.add_original_from_capture(src_cap, box, type_name,
                                                    who=who, note=note)
                meta2 = _sigmf.read_meta(ex.base)
                meta2["global"]["atk:source_cut"] = str(d)
                mp = _sigmf.write_meta(ex.base, meta2)
                try:
                    self.rf.record(mp, "training-example-meta", ex.id)
                except OSError:
                    pass
                return self.example(type_name, ex.id)
            except (LibraryError, _profiles.ProfileMismatch, ValueError):
                pass                    # fall back to the cut's own samples
        x = _sigmf.load(base, meta=meta)
        if x.ndim > 1:
            x = x[0]
        fs_c = _sigmf.sample_rate_of(meta)
        cls = str(g.get("atk:canonical_class", "") or "")
        dec = int(g.get("atk:decimation", 0) or 0)
        rates = self.canonical_rates()
        if cls not in rates or abs(rates[cls].rate - fs_c) > 0.5 \
                or (dec and dec != rates[cls].decimation):
            raise LibraryError(
                f"the cut is at {fs_c:,.0f} S/s, which is not a canonical rate "
                f"of {_profiles.describe(self.pid)}; it cannot be used as it "
                "stands (and its source recording is not there to re-cut)")
        st = self.get_type(type_name)
        if st.canonical and cls != st.canonical:
            raise LibraryError(f"the cut is a {cls}-width signal; '{st.words}' "
                               f"holds {st.canonical}-width signals")
        if x.size < MIN_SAMPLES:
            raise LibraryError("the cut is too short to train on")
        fl = g.get("atk:floor_per_hz")
        measured = _measure(x, fs_c, float(fl) if fl else None)
        span, span_note = _active_span(x, 0, int(x.size), fs_c)
        group = _group_key({"cut_folder": str(d)})
        ex_id, idx = self._next_original_id(st)
        role, gidx = self._role_for_new(st, group, None)
        hd = _holdout_for(span["start"], span["count"]) if role == "train" else None
        g2 = {"atk:receiver_profile": self.pid, "atk:tier": "record",
              "atk:method": KIND_METHOD["originals"],
              "atk:training_type": st.name, "atk:training_kind": "originals",
              "atk:role": role, "atk:holdout": hd, "atk:index": idx,
              "atk:group": group, "atk:group_index": gidx,
              "atk:signal_span": span,
              "atk:canonical_class": cls, "atk:canonical_rate": float(fs_c),
              "atk:decimation": int(rates[cls].decimation),
              "atk:lowpass_hz": float(g.get("atk:lowpass_hz", 0.45 * fs_c)),
              "atk:canonical_limited": bool(rates[cls].limited),
              "atk:source_capture": src_cap, "atk:source_cut": str(d),
              "atk:source_kind": "cut", "atk:box": box,
              "atk:measured": measured, "atk:snr_db": measured.get("snr_db"),
              "atk:created": _now(), "atk:cut_by": who or _prov.who(),
              "atk:notes": ["the cut's own samples: its source recording was "
                            "not there to re-cut around the measured band",
                            span_note],
              "atk:library_version": VERSION}
        if note:
            g2["atk:note"] = str(note)[:500]
        centre = _sigmf.center_of(meta)
        b = self._write(self.kind_dir(st.name, "originals"), ex_id, x, fs_c,
                        centre, g2, f"real example of {st.words} (from a cut)")
        if not st.canonical:
            st.canonical = cls
        st.next_index = idx + 1
        self._save_type(st)
        return _example_from_meta(b, "originals", st.name, _sigmf.read_meta(b))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _parse_box(box) -> tuple[float, float, float, float]:
    try:
        t0, t1 = float(box["t0_s"]), float(box["t1_s"])
        f_lo, f_hi = float(box["f_lo_hz"]), float(box["f_hi_hz"])
    except (KeyError, TypeError, ValueError):
        raise LibraryError("a box is {t0_s, t1_s, f_lo_hz, f_hi_hz}") from None
    if not (t1 > t0) or not (f_hi > f_lo):
        raise LibraryError("a box needs t1 after t0 and f_hi above f_lo")
    return t0, t1, f_lo, f_hi


def _holdout_for(start: int, n: int) -> dict | None:
    """The held-out tail of a TRAIN original's SIGNAL (its `signal_span`
    `start`, `n` samples) when it is long enough to spare one — taken from
    the signal, not from the noise margin after it (2026-10-26 review: a
    tail taken from the whole cut was mostly noise labelled as the type)."""
    if n < HOLDOUT_MIN_SAMPLES:
        return None
    count = max(WINDOW, int(round(HOLDOUT_FRACTION * n)))
    return {"start": int(start + n - count), "count": int(count),
            "why": "the last part of this signal is held back to grade the "
                   "model; copies are made only from the part before it"}


def signal_span(ex: Example) -> tuple[int, int]:
    """(start, count) of where the signal is inside an original."""
    sp = ex.signal_span
    if sp and int(sp.get("count", 0)) > 0:
        a = max(0, min(int(sp["start"]), int(ex.samples)))
        return a, max(0, min(int(sp["count"]), int(ex.samples) - a))
    return 0, int(ex.samples)


#: The block (seconds) and the level above the noise around it that the
#: signal span is trimmed with.
SPAN_BLOCK_S = 0.005
SPAN_RATIO = 1.5            # +1.8 dB in band


def _active_span(y: np.ndarray, a: int, b: int, fs_c: float
                 ) -> tuple[dict, str]:
    """Where the signal is in a cut: the drawn box [a, b) (canonical
    samples), trimmed at each end to the first and last block that stands
    1.8 dB above the noise around it (the margins outside the box, or the
    quietest tenth of the cut). A signal too weak to tell from the noise
    keeps the drawn box. -> ({start, count, how}, a sentence)."""
    n = int(y.size)
    a, b = max(0, min(int(a), n)), max(0, min(int(b), n))
    if b - a < MIN_SAMPLES:
        a, b = 0, n
    blk = max(32, int(round(SPAN_BLOCK_S * float(fs_c))))
    nb = n // blk
    how = "the drawn box"
    if nb >= 4:
        p = np.mean(np.abs(np.asarray(y[:nb * blk], np.complex64)
                           .reshape(nb, blk)) ** 2, axis=1)
        ia, ib = a // blk, max(a // blk + 1, -(-b // blk))
        outside = np.concatenate([p[:ia], p[ib:]])
        ref = float(np.median(outside)) if outside.size >= 2 else \
            float(np.percentile(p, 10))
        if ref > 0:
            on = np.nonzero(p[ia:ib] > SPAN_RATIO * ref)[0]
            if on.size:
                a2 = (ia + int(on[0])) * blk
                b2 = min(n, (ia + int(on[-1]) + 1) * blk)
                if b2 - a2 >= MIN_SAMPLES and (a2 > a or b2 < b):
                    a, b = max(a, a2), min(b, b2)
                    how = ("the drawn box, trimmed to where the signal stands "
                           "1.8 dB above the noise around it")
    span = {"start": int(a), "count": int(b - a), "how": how}
    return span, (f"the signal is samples {a:,}–{b:,} of the cut ({how}); the "
                  "noise either side is kept for the filters, never used as an "
                  "example of the type")


def _same_path(a, b) -> bool:
    if not a or not b:
        return False
    return os.path.normcase(os.path.normpath(str(a))) == \
        os.path.normcase(os.path.normpath(str(b)))


def _group_key(src: dict) -> str:
    """The recording an original comes from: its capture (the SigMF base),
    else its cut folder; '' when unknown (the original is its own group)."""
    for k in ("capture", "cut_folder"):
        v = str(src.get(k, "") or "").strip()
        if v:
            return os.path.normcase(os.path.normpath(v))
    return ""


def _measure(y: np.ndarray, fs_c: float, floor_per_hz: float | None = None
             ) -> dict:
    """The classical facts of a cut that the copies and the dataset need:
    occupied bandwidth and its centre, in-band SNR above the floor, the
    floor per Hz, the symbol rate when a cyclic line clears its threshold.
    Each with its method (MEASURED tier). `floor_per_hz` is the SOURCE
    span's floor when it is known — a tight cut's own median bin is signal."""
    from atk_diffusion.dsp import measure as _m
    out: dict = {"tier": "measured"}
    try:
        if floor_per_hz:
            fl = float(floor_per_hz)
            out["floor_method"] = ("the source span's median PSD bin (the cut "
                                   "itself is low-passed to its signal)")
        else:
            fl = float(_m.noise_floor(y, fs_c)["floor_per_hz"] or 0.0)
            out["floor_method"] = "the cut's own median PSD bin"
        out["floor_per_hz"] = fl
        ob = _m.occupied_bandwidth(y, fs_c, floor_per_hz=fl or None)
        out["bandwidth_hz"] = ob.get("value_hz")
        out["centre_hz"] = ob.get("centre_hz")
        out["lower_hz"], out["upper_hz"] = ob.get("lower_hz"), ob.get("upper_hz")
        out["bandwidth_caveats"] = list(ob.get("caveats") or [])
        band = (ob["lower_hz"], ob["upper_hz"]) if ob.get("value_hz") else None
        snr = _m.snr_above_floor(y, fs_c, band=band, floor_per_hz=fl or None)
        out["snr_db"] = snr.get("snr_db") if snr.get("measurable") else None
        out["snr_estimate_db"] = snr.get("snr_db")
        out["signal_power"] = float(snr.get("signal_power") or 0.0)
        out["snr_words"] = snr.get("words", "")
    except Exception as e:                                # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    try:
        if y.size >= 2048:
            sr = _m.symbol_rate(y, fs_c)
            out["symbol_rate_hz"] = sr.get("value_hz") if sr.get("known") else None
    except Exception:                                     # noqa: BLE001
        out["symbol_rate_hz"] = None
    return out


def measure(y, fs_c: float, floor_per_hz: float | None = None) -> dict:
    """Public: the same measurement the library files with an original."""
    return _measure(np.asarray(y, np.complex64), float(fs_c), floor_per_hz)


def pad_with_floor(x: np.ndarray, n: int, floor_per_hz: float, fs: float,
                   lowpass_hz: float | None, rng) -> np.ndarray:
    """`x` made `n` samples long with NOISE at the cut's own floor (never
    zeros: a classifier would learn that a silent stretch means a copy),
    centred, the noise through the cut's low-pass."""
    x = np.asarray(x, np.complex64)
    if x.size >= n:
        return x[:n]
    extra = n - x.size
    noise = floor_noise(extra, floor_per_hz, fs, lowpass_hz, rng)
    a = extra // 2
    return np.concatenate([noise[:a], x, noise[a:]]).astype(np.complex64)


def floor_noise(n: int, floor_per_hz: float, fs: float,
                lowpass_hz: float | None, rng) -> np.ndarray:
    """Complex noise whose density is `floor_per_hz` (power per Hz, full
    scale 1) inside the cut's low-pass, falling off outside it as the cut's
    filter makes it."""
    p = max(float(floor_per_hz or 0.0), 1e-16) * float(fs)
    w = (rng.standard_normal(int(n) + 512) + 1j * rng.standard_normal(int(n) + 512)
         ) * math.sqrt(p / 2.0)
    if lowpass_hz and lowpass_hz < 0.49 * fs:
        w = lowpass(w, lowpass_hz, fs)
    return w[256:256 + int(n)].astype(np.complex64)


def lowpass(x: np.ndarray, cutoff_hz: float, fs: float) -> np.ndarray:
    """The cut's own low-pass (dsp.resample.lowpass_taps at decimation 1),
    zero group delay."""
    from scipy.signal import fftconvolve
    from atk_diffusion.dsp.resample import lowpass_taps
    taps = lowpass_taps(float(cutoff_hz), float(fs), 1, per_decim=64)
    y = fftconvolve(np.asarray(x, np.complex128), taps, mode="same")
    return y.astype(np.complex64)
