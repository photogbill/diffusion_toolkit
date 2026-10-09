# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Teach-it-a-signal: mark a signal, name a class, and the detector knows it
in seconds (plan §4.B5; DETECTION_DESIGN §4 "Teach-it-a-signal is a
prototype, not a retrain", §4.2 Route → teach; decision D6).

Bill, 2026-10-08: *"we definitely have to do"*.

THE FLOW, each step visible:

1. **Mark** — one or more boxes (time × frequency) on a capture of THIS
   receiver profile, or IQ from ATK's ring buffer at the profile's rate.
   Profiles never mix: a mark from another receiver is refused in the plan's
   own sentence (`profiles.check_match`).
2. **Name** — a class, new or existing.
3. **Cut** — each mark is shifted, low-passed and integer-decimated to the
   profile's canonical rate for its bandwidth class
   (`dsp.resample.cut_to_canonical`), then split into the classifier's
   fixed-length windows. All marks of one teach go to ONE canonical rate
   (the class of the widest mark), because one class must be embedded at
   one rate.
4. **Store** — each cut is written as a SigMF pair under
   `rf.labeled(profile, class)` (`<profile>\\captures\\labeled\\<class>\\`)
   with `atk:source = "taught"` on its label and full provenance (source
   capture, sample range, decimation, canonical class, who cut it). The cut
   is the record (tier RECORD): nothing about it is reconstructed.
5. **Embed** — by a HOST-SUPPLIED classifier callable
   `embed(windows (n, L) complex64, fs) -> (n, D)` (ATK runs the profile's
   ONNX classifier on the CPU). One embedding per MARK — the mean of its
   windows, re-normalised — because windows of one burst are not independent
   examples, and counting them as such would let five marks pretend to be
   fifty. The window embeddings are kept too.
6. **Prototype** — `atk_diffusion.detect.prototypes.PrototypeBank.teach`
   (seconds, no GPU). No training run. The bank is saved beside the
   classifier's card (that module's convention) and writes its own counts
   into the card.
7. **Card** — the profile's classifier card gains the class as *taught*
   with its example count, so the language model's class list follows.

THE FLOOR. A class with fewer examples than the floor SAYS SO — in the
result's lines, in the card entry (`below_floor`), and in
`class_list_for_llm` — and its threshold is wider (the bank's job). It does
not pretend.

THE SCHEDULED FINE-TUNE. When taught classes accumulate, a fine-tune of the
embedding network on the whole labeled set folds them in properly
(DETECTION_DESIGN §4); `queue_finetune` writes the marker the training
environment's scheduler reads. Synthetic augmentation of a taught class
whose modulation TorchSig can make belongs to that fine-tune, not here.

LIMITS. A prototype from five marks is a sketch of a class, not a model of
it: the first experiment measures it (teach one class from five examples,
score it on the next day's captures). The bank and the classifier are other
parts of the toolkit; this module is the flow, and it works — stores,
embeds, records — when the bank is not there yet, saying what it could not
do.
"""

from __future__ import annotations

import inspect
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from atk_diffusion import cards as _cards
from atk_diffusion import profiles as _profiles
from atk_diffusion import provenance as _prov
from atk_diffusion import sigmf as _sigmf
from atk_diffusion.dsp import resample as _resample

#: Fewer examples (marked signals) than this and the class says so.
EXAMPLE_FLOOR = 10
#: The classifier's window length when the host does not say (samples at the
#: canonical rate).
DEFAULT_WINDOW = 1024
#: At most this many windows per mark go to the embedder.
MAX_WINDOWS_PER_MARK = 32

FINETUNE_MARKER = "finetune_requested.json"


@dataclass
class Mark:
    """One marked signal. Either a SigMF `capture` and the box in it, or `iq`
    at the profile's rate with its `fs` and `center_hz`.

    Times are seconds from the start of the capture (or of `iq`);
    frequencies are ABSOLUTE Hz, like a `detect.boxes.Detection`."""
    f_lo_hz: float
    f_hi_hz: float
    t0_s: float = 0.0
    t1_s: float | None = None
    capture: str | Path | None = None
    iq: np.ndarray | None = None
    fs: float | None = None
    center_hz: float | None = None
    profile: str = ""

    @classmethod
    def from_detection(cls, det, capture=None, iq=None, fs=None,
                       center_hz=None, profile: str = "") -> "Mark":
        return cls(f_lo_hz=float(det.f_lo), f_hi_hz=float(det.f_hi),
                   t0_s=float(det.t0), t1_s=float(det.t1), capture=capture,
                   iq=iq, fs=fs, center_hz=center_hz,
                   profile=profile or getattr(det, "profile", ""))

    @property
    def center(self) -> float:
        return 0.5 * (float(self.f_lo_hz) + float(self.f_hi_hz))

    @property
    def bw(self) -> float:
        return abs(float(self.f_hi_hz) - float(self.f_lo_hz))


@dataclass
class Cut:
    """A mark at the canonical rate, ready to store and embed."""
    mark: Mark
    x: np.ndarray
    fs: float
    info: dict
    windows: np.ndarray
    padded: bool
    source: dict = field(default_factory=dict)


@dataclass
class TeachResult:
    cls: str
    profile: str
    canonical_class: str
    canonical_rate: float
    examples: int
    total_examples: int
    windows: int
    floor: int
    below_floor: bool
    files: list = field(default_factory=list)
    embeddings_file: str = ""
    bank: str = ""
    bank_updated: bool = False
    card: str = ""
    card_updated: bool = False

    def lines(self) -> list[str]:
        out = [f"Taught '{self.cls}' on {_profiles.describe(self.profile)}: "
               f"{self.examples} new example{'s' if self.examples != 1 else ''}"
               f" ({self.total_examples} in all), cut to the "
               f"{self.canonical_class} rate of {self.canonical_rate:g} S/s, "
               f"{self.windows} windows embedded."]
        if self.below_floor:
            out.append(f"Only {self.total_examples} example"
                       f"{'s' if self.total_examples != 1 else ''} of "
                       f"'{self.cls}' — below the floor of {self.floor}. Its "
                       "threshold is wider and it will say UNKNOWN more "
                       "often; teach more examples before trusting it.")
        out.append(self.bank)
        out.append(self.card)
        return [s for s in out if s]

    def to_json(self) -> dict:
        d = dict(self.__dict__)
        d["lines"] = self.lines()
        return d


# ---------------------------------------------------------------------------
# Cutting
# ---------------------------------------------------------------------------
def _parse_iso(s: str) -> float | None:
    try:
        t = datetime.strptime(str(s).replace("Z", ""), "%Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        try:
            t = datetime.strptime(str(s).replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return None
    return t.replace(tzinfo=timezone.utc).timestamp()


def _source_iq(mark: Mark, profile: str):
    """(x at the profile's rate, fs, centre, source facts) for one mark."""
    if mark.capture is not None:
        meta = _sigmf.read_meta(mark.capture)
        cap_profile = _profiles.profile_from_meta(meta)
        _profiles.check_match(profile, cap_profile, what="this teach")
        fs = _sigmf.sample_rate_of(meta)
        if _sigmf.channels_of(meta) > 1:
            raise ValueError("teach takes one channel; this capture has "
                             f"{_sigmf.channels_of(meta)} — cut one first")
        start = max(0, int(round(float(mark.t0_s) * fs)))
        total = _sigmf.num_samples(mark.capture, meta)
        end = total if mark.t1_s is None else min(total,
                                                  int(round(float(mark.t1_s) * fs)))
        if end <= start:
            raise ValueError(f"the mark {mark.t0_s:g}–{mark.t1_s} s is outside "
                             f"the capture ({total / fs:.3g} s long)")
        x = _sigmf.load(mark.capture, start, end - start, meta=meta)
        centre = _sigmf.center_of(meta, start)
        caps = meta.get("captures") or [{}]
        t_epoch = _parse_iso(caps[0].get("core:datetime", ""))
        src = {"atk:source_capture": str(_sigmf.base_of(mark.capture).name),
               "atk:source_sample_start": int(start),
               "atk:source_sample_count": int(end - start),
               "epoch": (t_epoch + start / fs) if t_epoch is not None else None,
               "hw": str(meta.get("global", {}).get("core:hw", "") or "")}
        return np.asarray(x, dtype=np.complex64), fs, centre, src
    if mark.iq is None or mark.fs is None or mark.center_hz is None:
        raise ValueError("a mark needs a capture, or IQ with its sample rate "
                         "and centre frequency")
    if mark.profile:
        _profiles.check_match(profile, mark.profile, what="this teach")
    want = _profiles.parse_profile_id(profile).sample_rate
    if abs(float(mark.fs) - want) > 0.5:
        raise _profiles.ProfileMismatch(
            f"this IQ is at {float(mark.fs):g} S/s; {_profiles.describe(profile)}"
            f" runs at {want} S/s. Teach never resamples silently — use IQ "
            "from this receiver at its own rate.")
    fs = float(mark.fs)
    x = np.asarray(mark.iq, dtype=np.complex64)
    start = max(0, int(round(float(mark.t0_s) * fs)))
    end = x.size if mark.t1_s is None else min(x.size, int(round(float(mark.t1_s) * fs)))
    if end <= start:
        raise ValueError("the mark is outside the IQ it was given")
    src = {"atk:source_capture": "ATK IQ ring buffer",
           "atk:source_sample_start": int(start),
           "atk:source_sample_count": int(end - start), "epoch": None, "hw": ""}
    return x[start:end], fs, float(mark.center_hz), src


def windows_of(x: np.ndarray, length: int, max_windows: int
               ) -> tuple[np.ndarray, bool]:
    """Non-overlapping windows of `length`; a cut shorter than one window is
    zero-padded and says so (`padded`)."""
    x = np.asarray(x, dtype=np.complex64)
    L = int(length)
    if x.size < L:
        w = np.zeros((1, L), dtype=np.complex64)
        w[0, :x.size] = x
        return w, True
    n = min(int(max_windows), x.size // L)
    if n <= 0:
        n = 1
    # spread the windows across the cut rather than taking only the start
    starts = np.linspace(0, x.size - L, n).round().astype(int)
    return np.stack([x[s:s + L] for s in starts]).astype(np.complex64), False


def cut_examples(marks, profile: str, *, window: int = DEFAULT_WINDOW,
                 max_windows: int = MAX_WINDOWS_PER_MARK,
                 guard: float = 1.25) -> list[Cut]:
    """Each mark at the canonical rate of the WIDEST mark's bandwidth class,
    split into windows."""
    marks = list(marks)
    if not marks:
        raise ValueError("teach needs at least one marked signal")
    pid = _profiles.parse_profile_id(profile)
    widest = max(m.bw for m in marks)
    can = _profiles.canonical_for(pid.sample_rate, max(widest, 1.0))
    out = []
    for m in marks:
        if m.bw <= 0:
            raise ValueError("a mark needs a frequency extent (f_hi > f_lo)")
        x, fs, centre, src = _source_iq(m, profile)
        # the integer decimation of the class chosen for the whole teach
        mixed = _resample.shift(x, m.center - centre, fs)
        cutoff = min(0.5 * max(m.bw, 1.0) * guard, 0.49 * can.rate)
        y, fs_out = _resample.decimate(mixed, can.decimation, fs, cutoff_hz=cutoff)
        info = {"decimation": can.decimation, "canonical_rate": can.rate,
                "canonical_class": can.cls, "limited": can.limited,
                "lowpass_hz": cutoff, "f_offset_hz": m.center - centre,
                "bw_hz": m.bw}
        w, padded = windows_of(y, window, max_windows)
        out.append(Cut(m, y, fs_out, info, w, padded, src))
    return out


# ---------------------------------------------------------------------------
# The bank (another part of the toolkit; imported lazily)
# ---------------------------------------------------------------------------
def make_bank(profile: str, canonical_class: str):
    """A `detect.prototypes.PrototypeBank` for (profile, canonical class), or
    (None, why) when that module is not installed. The constructor is matched
    by parameter NAME, so this keeps working while that module settles."""
    try:
        from atk_diffusion.detect.prototypes import PrototypeBank
    except ImportError as exc:
        return None, ("the prototype bank (atk_diffusion.detect.prototypes) "
                      f"is not available here ({exc}) — the examples and "
                      "their embeddings are stored, and the class joins the "
                      "bank the next time it is built")
    try:
        params = list(inspect.signature(PrototypeBank).parameters.values())
    except (TypeError, ValueError):
        params = []
    kw = {}
    for p in params:
        n = p.name.lower()
        if n in ("profile", "profile_id", "receiver_profile"):
            kw[p.name] = profile
        elif n in ("canonical_class", "canonical", "bandwidth_class", "cls_rate",
                   "rate_class"):
            kw[p.name] = canonical_class
    try:
        return PrototypeBank(**kw), ""
    except TypeError:
        pass
    for args in ((profile, canonical_class), (profile,), ()):
        try:
            return PrototypeBank(*args), ""
        except TypeError:
            continue
    return None, ("the prototype bank could not be constructed for "
                  f"{profile} / {canonical_class}; the examples are stored")


def _bank_words(res, n: int, cls: str) -> str:
    if res is None:
        return f"The prototype bank took {n} example{'s' if n != 1 else ''} of '{cls}'."
    if isinstance(res, str):
        return res
    if isinstance(res, dict):
        for k in ("message", "why", "note", "words", "status"):
            if res.get(k):
                return str(res[k])
        return "The prototype bank answered: " + json.dumps(res, default=str)[:300]
    if isinstance(res, (list, tuple)):
        return " ".join(str(r) for r in res if r)
    if hasattr(res, "lines"):
        try:
            return " ".join(res.lines())
        except Exception:                                  # noqa: BLE001
            pass
    return str(res)


def _bank_floor(bank) -> int | None:
    for name in ("example_floor", "floor", "min_examples", "EXAMPLE_FLOOR"):
        v = getattr(bank, name, None)
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool) and v > 0:
            return int(v)
    return None


# ---------------------------------------------------------------------------
# The card
# ---------------------------------------------------------------------------
def update_card_classes(card, cls: str, examples: int, floor: int,
                        canonical_class: str) -> str:
    """Add or update the class in `card.classes`. A class the model was
    TRAINED on stays trained; its taught examples are counted beside it."""
    classes = [dict(c) if isinstance(c, dict) else {"name": str(c),
                                                    "source": "trained"}
               for c in (card.classes or [])]
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    thin = int(examples) < int(floor)
    for c in classes:
        if c.get("name") == cls:
            if c.get("source") == "taught":
                c.update(examples=int(examples), taught=int(examples),
                         below_floor=thin, thin=thin, floor=int(floor),
                         canonical_class=canonical_class, taught_at=now)
                words = "updated"
            else:
                c.update(taught_examples=int(examples), taught=int(examples),
                         taught_at=now)
                words = "already trained; taught examples counted beside it"
            card.classes = classes
            return words
    classes.append({"name": cls, "source": "taught", "examples": int(examples),
                    "taught": int(examples), "below_floor": thin, "thin": thin,
                    "floor": int(floor), "canonical_class": canonical_class,
                    "taught_at": now})
    card.classes = classes
    return "added as taught"


def class_list_for_llm(card) -> list[str]:
    """The class list the language model is given, with what each class IS:
    trained, or taught with its example count (and the floor when below it)."""
    out = []
    for c in card.classes or []:
        if not isinstance(c, dict):
            out.append(f"{c} (trained)")
            continue
        name = c.get("name", "?")
        if c.get("source") == "taught":
            n = int(c.get("examples", 0))
            s = f"{name} (taught, {n} example{'s' if n != 1 else ''}"
            if c.get("below_floor") or c.get("thin"):
                s += f" — below the floor of {c.get('floor', EXAMPLE_FLOOR)}, " \
                     "a sketch, not a model"
            out.append(s + ")")
        else:
            out.append(f"{name} (trained)")
    return out


def _count_examples(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for _ in folder.glob("*.sigmf-meta"))


# ---------------------------------------------------------------------------
# Teach
# ---------------------------------------------------------------------------
def teach(rf, profile: str, cls: str, marks, *,
          embed: Callable[[np.ndarray, float], np.ndarray],
          bank=None, card_dir=None, floor: int | None = None,
          window: int = DEFAULT_WINDOW, who: str = "",
          embedder_name: str = "",
          progress: Callable[[str], None] | None = None) -> TeachResult:
    """Teach `cls` from `marks` on `profile`. See the module docstring.

    `embed(windows, fs) -> (n, D)` is the host's classifier; `bank` the
    host's loaded `PrototypeBank` for this profile and canonical class (or
    None: the examples are stored and the result says the bank was not
    updated); `card_dir` the classifier's model folder (default: the newest
    `classifier1d` for the profile)."""
    def say(s):
        if progress:
            progress(s)
    name = str(cls).strip()
    if not name:
        raise ValueError("a taught class needs a name")
    if name.upper() == "UNKNOWN":
        raise ValueError("'UNKNOWN' is the detector's honest answer, not a "
                         "class that can be taught")
    profile = str(profile).strip().lower()
    _profiles.parse_profile_id(profile)
    marks = list(marks)
    say(f"cutting {len(marks)} marked signal(s)")
    cuts = cut_examples(marks, profile, window=window)
    can_cls, can_rate = cuts[0].info["canonical_class"], cuts[0].fs

    # -- store the examples (the record) -----------------------------------
    folder = Path(rf.labeled(profile, name))
    folder.mkdir(parents=True, exist_ok=True)
    stamp = (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_"
             + uuid.uuid4().hex[:6])
    by = who or _prov.who()
    files = []
    for i, c in enumerate(cuts):
        base = folder / f"{stamp}_{int(round(c.mark.center))}_{i:02d}"
        ann = _sigmf.Annotation(0, int(c.x.size), float(c.mark.f_lo_hz),
                                float(c.mark.f_hi_hz), label=name,
                                extra={"atk:source": "taught"})
        g = {"atk:receiver_profile": profile, "atk:tier": "record",
             "atk:source_capture": c.source["atk:source_capture"],
             "atk:source_sample_start": c.source["atk:source_sample_start"],
             "atk:source_sample_count": c.source["atk:source_sample_count"],
             "atk:decimation": int(c.info["decimation"]),
             "atk:canonical_class": can_cls, "atk:cut_by": by,
             "atk:taught_class": name,
             "atk:padded": bool(c.padded)}
        dp, mp = _sigmf.write_pair(base, c.x, c.fs, center_hz=c.mark.center,
                                   datatype="cf32", t0_utc=c.source.get("epoch"),
                                   annotations=[ann], extra_global=g,
                                   hw=c.source.get("hw", ""),
                                   description=f"taught example of {name}")
        rf.record(dp, "taught", f"{name} from {c.source['atk:source_capture']}")
        rf.record(mp, "taught-meta", name)
        files.append(str(dp))

    # -- embed ----------------------------------------------------------------
    say("embedding the examples")
    per_mark, all_w = [], []
    for c in cuts:
        e = np.asarray(embed(c.windows, c.fs), dtype=np.float32)
        if e.ndim != 2 or e.shape[0] != c.windows.shape[0]:
            raise ValueError(f"the classifier returned embeddings of shape "
                             f"{e.shape} for {c.windows.shape[0]} windows")
        if not np.all(np.isfinite(e)):
            raise ValueError("the classifier returned non-finite embeddings")
        all_w.append(e)
        m = e.mean(axis=0)
        nrm = float(np.linalg.norm(m))
        per_mark.append(m / nrm if nrm > 0 else m)
    mark_emb = np.stack(per_mark).astype(np.float32)
    emb_file = folder / f"taught_{stamp}.npz"
    np.savez(emb_file, mark_embeddings=mark_emb,
             window_embeddings=np.concatenate(all_w).astype(np.float32),
             canonical_rate=np.float64(can_rate),
             note=np.array(f"embedder={embedder_name or 'host classifier'}; "
                           f"tier=proposed (a model's embedding)"))
    rf.record(emb_file, "taught-embeddings", name)

    total = _count_examples(folder)
    fl = int(floor) if floor else (_bank_floor(bank) if bank is not None else None) \
        or EXAMPLE_FLOOR

    # -- the prototype --------------------------------------------------------
    bank_updated = False
    if bank is None:
        bank_words = ("No prototype bank was given, so the class is not live "
                      "yet: the examples and their embeddings are stored, and "
                      "ATK passes its loaded bank to make it live.")
    else:
        try:
            res = bank.teach(name, mark_emb)
            bank_words = _bank_words(res, len(cuts), name)
            bank_updated = True
        except Exception as exc:                           # noqa: BLE001
            bank_words = (f"The prototype bank refused the examples "
                          f"({type(exc).__name__}: {exc}); they are stored.")

    # -- the card -------------------------------------------------------------
    card_updated = False
    cdir = Path(card_dir) if card_dir else None
    if cdir is None:
        found = _cards.find(rf, profile, "classifier1d")
        cdir = found[0][0] if found else None
    if cdir is not None and bank_updated:
        # the bank lives beside the classifier's card (detect.prototypes'
        # convention): save it there, and let it write its own counts
        done = []
        for meth in ("save", "update_card"):
            fn = getattr(bank, meth, None)
            if not callable(fn):
                continue
            try:
                fn(cdir)
                done.append(meth)
            except Exception as exc:                       # noqa: BLE001
                bank_words += (f" (The bank could not {meth.replace('_', ' ')} "
                               f"beside the classifier: {exc}.)")
        if "save" in done:
            bank_words += " The bank is saved beside the classifier's card."
    if cdir is None:
        card_words = ("No classifier card exists for this profile yet; the "
                      "taught class joins its class list when one does.")
    else:
        try:
            card = _cards.load(cdir, expect_kind="classifier1d",
                               for_profile=profile, verify_weights=False)
            how = update_card_classes(card, name, total, fl, can_cls)
            _cards.save(cdir, card)
            card_updated = True
            card_words = f"The classifier card ({card.name}): '{name}' {how}."
        except (_cards.CardRefusal, _profiles.ProfileMismatch, ValueError) as exc:
            card_words = f"The classifier card was not updated: {exc}"

    result = TeachResult(cls=name, profile=profile, canonical_class=can_cls,
                         canonical_rate=float(can_rate), examples=len(cuts),
                         total_examples=total,
                         windows=int(sum(c.windows.shape[0] for c in cuts)),
                         floor=fl, below_floor=total < fl, files=files,
                         embeddings_file=str(emb_file), bank=bank_words,
                         bank_updated=bank_updated, card=card_words,
                         card_updated=card_updated)
    log = folder / "teach_log.jsonl"
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(_prov.stamp("teach", **result.to_json()),
                           default=str) + "\n")
    rf.record(log, "teach-log", name)
    say(result.lines()[0])
    return result


def queue_finetune(rf, profile: str, reason: str = "") -> Path:
    """Write (or extend) the marker the scheduled fine-tune reads: every
    taught class on disk for this profile with its example count. -> path."""
    profile = str(profile).strip().lower()
    _profiles.parse_profile_id(profile)
    base = Path(rf.captures(profile)) / "labeled"
    classes = {}
    if base.is_dir():
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            n = _count_examples(d)
            if n:
                classes[d.name] = n
    runs = Path(rf.runs(profile))
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / FINETUNE_MARKER
    prev = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prev = {}
    history = list(prev.get("requests", []))
    history.append({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "by": _prov.who(), "reason": str(reason)})
    marker = {"profile": profile, "classes": classes,
              "labeled_dir": str(base), "requests": history,
              "what": ("fine-tune the embedding network on the whole labeled "
                       "set; the prototype path stays the fallback until it "
                       "has run (DETECTION_DESIGN §4)")}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    tmp.replace(path)
    rf.record(path, "finetune-marker", reason)
    return path


def pending_finetune(rf, profile: str) -> dict | None:
    path = Path(rf.runs(str(profile).lower())) / FINETUNE_MARKER
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
