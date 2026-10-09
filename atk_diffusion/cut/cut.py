# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The signal cut — right-click to everything (DETECTION_DESIGN §4.2, D10;
ARCHITECTURE §5 "The signal cut").

Bill, 2026-10-08: *"with a right click from the waterfall, and then process
the result via any tools that can process that signal type, be it
demodulator, decoder, DF, or even just saving the original and the cleaned
up one in the same folder for offline analysis."*

    make_cut(x, fs, center_hz, box, profile, rf, source=None, who="", note="")
    cut_from_capture(path, box, rf, who="")
    CutFolder.open(path)
        .analyze(classifier=None, fingerprint=None)   step 2
        .clean(method, **params)                       step 3
        .routes_available() / .route(tool, input, runner)   step 4
        .report()                                      step 5

THE FOLDER (self-contained: readable without ATK, importable into a new
install, the unit the RF social graph and the watchlist point at):

    <rf_data>/<profile>/cuts/<UTC stamp>_<centre Hz>/
      original.sigmf-data   the box: shifted to 0 Hz, low-passed, integer-
      original.sigmf-meta   decimated to the profile's canonical rate for
                            the box's bandwidth class. cf32 little-endian
                            (numpy: np.fromfile(f, np.complex64)); a Kraken
                            cut keeps every coherent channel, sample-
                            interleaved (core:num_channels). Its meta says
                            atk:tier = record and carries the provenance:
                            atk:receiver_profile, atk:source_capture,
                            atk:source_sample_start/count (in the SOURCE's
                            samples), atk:decimation, atk:canonical_class,
                            atk:canonical_rate, atk:cut_by, atk:cut_at,
                            atk:box {t0_s, t1_s, f_lo_hz, f_hi_hz},
                            atk:margin_s, atk:lowpass_hz, atk:f_offset_hz,
                            atk:source_center_hz, atk:source_sample_rate,
                            atk:floor_per_hz (the source span's noise floor,
                            power per Hz, full scale = 1). core:frequency is
                            the box centre (absolute Hz); core:datetime is
                            the wall clock of the cut's first sample when
                            the source's was known; the box is also a SigMF
                            annotation.
      cleaned.sigmf-*       (optional; later cleans cleaned_2, cleaned_3 …)
                            atk:tier from the method (provenance.tier_for),
                            atk:method, atk:method_params (the parameters
                            ACTUALLY used — the α, the roll-off, the floor —
                            not only those asked for), atk:snr_before_db /
                            atk:snr_after_db (MEASURED, with atk:snr_method
                            saying how), atk:model_sha256 for learned methods;
                            the original's core:datetime (the signal's time,
                            not the time it was cleaned)
      separated_N.sigmf-*   (optional) FRESH separate's outputs, N = 1, 2 …
      analysis.json         every fact: the cut (with the original's sha256),
                            measurements (tier and method each), cyclic peaks
                            and class-table probe results, the SCF image's
                            axes and geometry, the class or UNKNOWN, the
                            fingerprint, cleans[], routes[], files{}
      scf.npy               float32 [64, 128]: rows = frequency over
                            [−fs/2, fs/2), columns = cycle frequency over
                            [0, fs/2); normalised 0..1 (cyclo.scf.scf_image)
      scf.png               the same, with axes in kHz
      cyclic_profile.npy    float64 [3, n]: α (Hz); max over f of |S(f, α)|
                            non-conjugate; the same conjugate — both
                            normalised by the non-conjugate value at α = 0
      cyclic_profile.png    the curves with the peaks that cleared their
                            derived threshold labelled
      report.md             written from analysis.json's facts only

THE ORIGINAL IS NEVER CHANGED. Every file written is hashed into the
rf_data write log at the moment it is written, and the original's own hash
is kept in analysis.json, so the folder can prove itself where no write log
knows it. Analyze, Clean and Route check the original first and refuse, in
words, if it changed; a cut copied into another rf_data (an import) is
checked against its own recorded hash and then entered in that write log.
`original` and every cleaned file sit side by side, always, and a decode
from anything but the original says so (`cut.route`).

THE CLEAN MENU (`METHODS`): matched (the parameters for the demodulator —
MEASURED — and, for a linear modulation, the matched-filtered signal —
CLEANED), fresh, fresh_separate, score (Kraken cuts), wiener,
rfi_mask_interp (INFERRED), diffusion (INVENTED; the denoiser is another
engineer's, passed in as a callable that names its weights' hash).
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from atk_diffusion import paths as _paths
from atk_diffusion import profiles as _profiles
from atk_diffusion import provenance as _prov
from atk_diffusion import sigmf as _sigmf
from atk_diffusion.detect import classes as _classes

ANALYSIS = "analysis.json"
VERSION = 1

#: The Clean menu → the provenance method whose tier the output carries.
#: Every one is in provenance.METHOD_TIERS. "matched" writes the matched
#: filter's output ("matched_filter", CLEANED); when the cut has no linear
#: modulation to match, only its parameters are recorded, tier MEASURED
#: ("matched_parameters", registered by cyclo.filters).
METHODS = {"matched": "matched_filter", "fresh": "fresh",
           "fresh_separate": "fresh_separate", "score": "score",
           "wiener": "wiener", "rfi_mask_interp": "rfi_mask_interp",
           "diffusion": "diffusion_denoise"}

#: Seconds of margin each side of the box: the larger of this and 10 % of
#: the box (so a burst's rise and fall are kept).
MIN_MARGIN_S = 0.05

#: The fewest samples a cut may hold at its canonical rate: the cyclic
#: profile, the SCF and every clean need at least this many.
MIN_CUT_SAMPLES = 256


class CutError(ValueError):
    """The cut cannot do what was asked. The message says why, in words."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _as_profile(profile) -> _profiles.ReceiverProfile:
    if isinstance(profile, _profiles.ReceiverProfile):
        return profile
    try:
        return _profiles.new_profile(str(profile))
    except ValueError as e:
        raise CutError(f"{profile!r} is not a receiver profile: {e}") from None


def _jsonable(obj):
    from atk_diffusion.cut.route import _default
    return json.loads(json.dumps(obj, default=_default))


def _parse_box(box) -> tuple:
    try:
        t0, t1 = float(box["t0_s"]), float(box["t1_s"])
        f_lo, f_hi = sorted((float(box["f_lo_hz"]), float(box["f_hi_hz"])))
    except (KeyError, TypeError, ValueError):
        raise CutError("a box is {t0_s, t1_s, f_lo_hz, f_hi_hz} — times in "
                       "seconds, frequencies in Hz") from None
    if not all(math.isfinite(v) for v in (t0, t1, f_lo, f_hi)):
        raise CutError("the box's times and frequencies must be finite numbers")
    if t1 <= t0:
        raise CutError("the box ends before it starts")
    if f_hi - f_lo <= 0:
        raise CutError("the box has no width in frequency")
    return t0, t1, f_lo, f_hi


def _margin(margin_s, t0: float, t1: float) -> float:
    if margin_s is None:
        return max(MIN_MARGIN_S, 0.1 * (t1 - t0))
    m = float(margin_s)
    if not (m >= 0 and math.isfinite(m)):
        raise CutError("the margin is a number of seconds, zero or more")
    return m


def _first_sample(t: float, fs: float) -> int:
    """floor(t·fs), robust to the float error of t: (0.3 − 0.1)·240000 is
    47999.999… — a box drawn at whole milliseconds must not start a sample
    early (nor differ between a cut from a stream and from its capture)."""
    return int(math.floor(t * fs + 1e-6))


def _end_sample(t: float, fs: float) -> int:
    return int(math.ceil(t * fs - 1e-6))


def _epoch_of(meta: dict) -> float | None:
    caps = meta.get("captures", []) or []
    dt = caps[0].get("core:datetime") if caps else None
    if not dt:
        return None
    s = str(dt).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Making a cut (step 1)
# ---------------------------------------------------------------------------
def _folder_name(center: float, wall: float | None) -> str:
    t = time.time() if wall is None else float(wall)
    stamp = datetime.fromtimestamp(t, tz=timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    return f"{stamp}_{int(round(center))}"


def _unique(parent: Path, name: str) -> Path:
    p = parent / name
    k = 2
    while p.exists():
        p = parent / f"{name}_{k}"
        k += 1
    return p


def make_cut(x, fs, center_hz, box, profile, rf, source=None, who="",
             note="", margin_s: float | None = None) -> "CutFolder":
    """Cut the box out of x and write the folder (step 1 of §4.2).

    x          IQ at the profile's rate, (n,) or (channels, n) — a Kraken
               cut keeps every channel
    fs         its rate (must be the profile's: the sample-rate law)
    center_hz  x's centre frequency (absolute)
    box        {t0_s, t1_s, f_lo_hz, f_hi_hz}: times relative to x[0],
               frequencies absolute
    profile    ReceiverProfile or id; rf an RfData
    source     where x came from: a capture path, a Detection (its class
               and candidates travel with the cut), or a dict
               {capture, sample_offset, epoch, detection, hw} — sample_offset
               is x[0]'s index in the capture and epoch the wall clock of
               the capture's sample 0
    who, note  who cut it and why (atk:cut_by, atk:note)
    """
    prof = _as_profile(profile)
    fs = float(fs)
    if abs(fs - float(prof.sample_rate)) > 0.5:
        raise _profiles.ProfileMismatch(
            f"this IQ is at {fs:,.0f} S/s; the profile {prof.id} is "
            f"{_profiles.describe(prof.id)}. A cut is made at the profile's "
            "own rate (the sample-rate law).")
    xa = np.asarray(x)
    if xa.ndim == 1:
        xa = xa[None, :]
    if xa.ndim != 2 or xa.shape[1] == 0:
        raise CutError("there is no IQ to cut")
    C, n = xa.shape
    t0, t1, f_lo, f_hi = _parse_box(box)
    dur = n / fs
    if t1 <= 0 or t0 >= dur:
        raise CutError(f"the box ({t0:.3f}–{t1:.3f} s) is outside the IQ "
                       f"given (0–{dur:.3f} s)")
    half = 0.5 * fs
    if f_lo < float(center_hz) - half or f_hi > float(center_hz) + half:
        raise CutError("the box reaches outside what the receiver saw "
                       f"({float(center_hz) - half:,.0f}–"
                       f"{float(center_hz) + half:,.0f} Hz)")
    # source details
    src: dict = {}
    det = None
    if source is not None:
        from atk_diffusion.detect.boxes import Detection
        if isinstance(source, Detection):
            det = source
        elif isinstance(source, dict):
            src = dict(source)
            det = src.pop("detection", None)
        else:
            src = {"capture": str(source)}
    if det is not None and not isinstance(det, dict):
        det = det.to_json()
    offset = int(src.get("sample_offset", 0) or 0)
    epoch = src.get("epoch")
    m = _margin(margin_s, t0, t1)
    s0 = max(0, _first_sample(t0 - m, fs))
    s1 = min(n, _end_sample(t1 + m, fs))
    f_mid = 0.5 * (f_lo + f_hi)
    f_off = f_mid - float(center_hz)
    from atk_diffusion.dsp.resample import cut_to_canonical
    outs, info = [], None
    for c in range(C):
        y, fs_c, info = cut_to_canonical(xa[c, s0:s1], fs, f_off, f_hi - f_lo)
        outs.append(y)
    y = np.stack(outs) if C > 1 else outs[0]
    n_c = int(np.asarray(y).shape[-1])
    if n_c < MIN_CUT_SAMPLES:
        raise CutError(
            f"the cut would hold only {n_c:,} samples at its canonical rate "
            f"of {float(info['canonical_rate']):,.0f} S/s "
            f"({n_c / float(info['canonical_rate']) * 1e3:,.1f} ms); the "
            f"analysis needs at least {MIN_CUT_SAMPLES} — give a longer "
            "stretch of IQ")
    if not np.all(np.isfinite(np.asarray(y))):
        raise CutError("the IQ holds samples that are not finite (NaN or "
                       "infinity) — a damaged file or a failed read; nothing "
                       "was cut")
    from atk_diffusion.dsp import measure as _measure
    nf = _measure.noise_floor(xa[0, s0:s1], fs,
                              nfft=int(getattr(prof.stft, "fft_size", 1024)))
    # the wall clock of the cut's first sample: the source's epoch is the
    # time of ITS sample 0, and x[0] is source sample `offset`
    wall = (float(epoch) + (offset + s0) / fs) if epoch is not None else None
    rf.ensure()
    parent = rf.cuts(prof.id)
    parent.mkdir(parents=True, exist_ok=True)
    folder = _unique(parent, _folder_name(f_mid, wall))
    folder.mkdir(parents=True)
    who = who or _prov.who()
    cut_at = _now()
    box_d = {"t0_s": t0, "t1_s": t1, "f_lo_hz": f_lo, "f_hi_hz": f_hi}
    g = {"atk:receiver_profile": prof.id, "atk:tier": "record",
         "atk:source_capture": src.get("capture", ""),
         "atk:source_sample_start": offset + s0,
         "atk:source_sample_count": s1 - s0,
         "atk:decimation": int(info["decimation"]),
         "atk:canonical_class": info["canonical_class"],
         "atk:canonical_rate": float(info["canonical_rate"]),
         "atk:canonical_limited": bool(info["limited"]),
         "atk:cut_by": who, "atk:cut_at": cut_at, "atk:box": box_d,
         "atk:margin_s": m, "atk:lowpass_hz": float(info["lowpass_hz"]),
         "atk:f_offset_hz": f_off, "atk:source_center_hz": float(center_hz),
         "atk:source_sample_rate": fs,
         "atk:floor_per_hz": float(nf["floor_per_hz"]),
         "atk:floor_method": nf["method"]}
    if note:
        g["atk:note"] = str(note)
    if det is not None:
        g["atk:source_detection"] = det
    fs_c = float(info["canonical_rate"])
    ann = _sigmf.Annotation(
        sample_start=max(0, int(round((t0 - s0 / fs) * fs_c))),
        sample_count=max(1, int(round((min(t1, dur) - max(t0, 0.0)) * fs_c))),
        freq_lower_edge=f_lo, freq_upper_edge=f_hi,
        label=(det or {}).get("cls") or "box", comment=str(note or ""))
    base = folder / "original"
    dp, mp = _sigmf.write_pair(
        base, y, fs_c, f_mid, datatype="cf32", t0_utc=wall,
        annotations=[ann], extra_global=g, hw=str(src.get("hw", "")),
        description=f"signal cut, {prof.id}, box {f_lo:,.0f}–{f_hi:,.0f} Hz",
        recorder="ATK Diffusion Toolkit — signal cut", channels=C)
    if wall is None:
        meta = _sigmf.read_meta(base)
        meta["captures"][0].pop("core:datetime", None)
        meta["global"]["atk:time_note"] = ("the wall-clock time of the source "
                                           "was not known; times are relative "
                                           "to the source's first sample")
        _sigmf.write_meta(base, meta)
    rf.record(dp, "cut-original", f"{prof.id} {f_lo:.0f}-{f_hi:.0f} Hz")
    rf.record(mp, "cut-original-meta")
    analysis = {
        "version": VERSION, "folder": folder.name, "profile": prof.id,
        "cut": {"center_hz": f_mid, "box": box_d, "margin_s": m,
                "channels": C, "samples": n_c,
                "duration_s": float(n_c / fs_c),
                "canonical_rate_hz": fs_c,
                "canonical_class": info["canonical_class"],
                "canonical_limited": bool(info["limited"]),
                "decimation": int(info["decimation"]),
                "lowpass_hz": float(info["lowpass_hz"]),
                "source_capture": src.get("capture", ""),
                "source_sample_start": offset + s0,
                "source_sample_count": s1 - s0,
                "source_sample_rate": fs, "source_center_hz": float(center_hz),
                "epoch": wall, "cut_by": who, "cut_at": cut_at,
                "note": str(note or ""),
                "profile_words": _profiles.describe(prof.id),
                "floor_per_hz": float(nf["floor_per_hz"]),
                "floor_db_per_hz": float(nf["floor_db_per_hz"]),
                "floor_method": nf["method"],
                "original_sha256": _paths.sha256_file(dp),
                "original_meta_sha256": _paths.sha256_file(mp)},
        "source": {"capture": src.get("capture", ""), "detection": det},
        "measurements": None, "cyclic": None, "scf": None, "class": None,
        "fingerprint": None, "cleans": [], "routes": [],
        "files": {"original.sigmf-data / .sigmf-meta":
                  "the box at the canonical rate — RECORD"},
        "history": [{"at": cut_at, "by": who, "what": "cut"}],
    }
    cf = CutFolder(folder, rf)
    cf._analysis = analysis
    cf._save()
    return cf


def cut_from_capture(path, box, rf, who: str = "", note: str = "",
                     margin_s: float | None = None) -> "CutFolder":
    """Cut a box out of a recorded SigMF capture. Box times are relative to
    the capture's first sample. The capture's profile comes from its
    metadata (profiles.profile_from_meta); a capture the write log says
    changed after it was recorded is named, not used; one the write log has
    never seen (an import) is cut and the cut says so."""
    try:
        meta = _sigmf.read_meta(path)
    except FileNotFoundError:
        raise CutError(f"there is no capture at {_sigmf.base_of(path)} (no "
                       ".sigmf-meta)") from None
    except json.JSONDecodeError as e:
        raise CutError(f"the capture's .sigmf-meta is not valid JSON ({e})"
                       ) from None
    fs = _sigmf.sample_rate_of(meta)
    if fs <= 0:
        raise CutError("the capture's metadata has no core:sample_rate")
    try:
        pid = _profiles.profile_from_meta(meta)
    except ValueError as e:
        raise CutError(f"the capture's receiver profile cannot be told: {e}"
                       ) from None
    data = _sigmf.data_path(path)
    if not data.exists():
        raise CutError(f"the capture's samples are missing ({data.name})")
    ok, why = rf.verify(data)
    imported = False
    if not ok:
        if "not in the write log" in why:
            imported = True
        else:
            raise CutError(why)
    total = _sigmf.num_samples(path, meta)
    t0, t1, _flo, _fhi = _parse_box(box)
    dur = total / fs
    if t1 <= 0 or t0 >= dur:
        raise CutError(f"the box ({t0:.3f}–{t1:.3f} s) is outside the capture "
                       f"(0–{dur:.3f} s)")
    m = _margin(margin_s, t0, t1)
    s0 = max(0, _first_sample(t0 - m, fs))
    s1 = min(total, _end_sample(t1 + m, fs))
    centres = {_sigmf.center_of(meta, s) for s in (s0, s1 - 1)}
    for c in meta.get("captures", []) or []:
        st = int(c.get("core:sample_start", 0))
        if s0 < st < s1:
            centres.add(float(c.get("core:frequency", 0.0)))
    if len(centres) > 1:
        raise CutError("the radio was retuned inside that box — draw it on "
                       "one side of the retune")
    center = centres.pop()
    x = _sigmf.load(path, start=s0, count=s1 - s0, meta=meta)
    epoch = _epoch_of(meta)
    seg_box = {"t0_s": t0 - s0 / fs, "t1_s": t1 - s0 / fs,
               "f_lo_hz": box["f_lo_hz"], "f_hi_hz": box["f_hi_hz"]}
    src = {"capture": str(_sigmf.base_of(path)), "sample_offset": s0,
           "epoch": epoch, "hw": meta.get("global", {}).get("core:hw", "")}
    note2 = note
    if imported:
        note2 = (note + "; " if note else "") + ("source not in the write log "
                                                 "(an import)")
    # the segment already holds the margins; keep them exactly
    return make_cut(x, fs, center, seg_box, pid, rf, source=src, who=who,
                    note=note2, margin_s=m)


# ---------------------------------------------------------------------------
# The folder
# ---------------------------------------------------------------------------
class CutFolder:
    """One cut folder. `open(path)` reads it; the steps write into it."""

    def __init__(self, path, rf=None):
        self.path = Path(path)
        if rf is None:
            # <root>/<profile>/cuts/<name>
            rf = _paths.RfData(self.path.parents[2])
        self.rf = rf
        self._analysis: dict | None = None

    # -- opening and bookkeeping ----------------------------------------------
    @classmethod
    def open(cls, path, rf=None) -> "CutFolder":
        p = Path(path)
        if not (p / ANALYSIS).exists() or not _sigmf.meta_path(
                p / "original").exists():
            raise CutError(f"{p} is not a cut folder (no original.sigmf-meta "
                           "and analysis.json)")
        cf = cls(p, rf)
        try:
            cf._analysis = json.loads((p / ANALYSIS).read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise CutError(f"{p.name}'s analysis.json is damaged ({e})") from None
        return cf

    @property
    def analysis(self) -> dict:
        if self._analysis is None:
            self._analysis = json.loads((self.path / ANALYSIS).read_text(
                encoding="utf-8"))
        return self._analysis

    @property
    def profile(self) -> str:
        return str(self.analysis.get("profile", ""))

    def _record(self, path, kind: str, note: str = "") -> None:
        self.rf.record(path, kind, note)

    def _save(self) -> None:
        p = self.path / ANALYSIS
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(_jsonable(self.analysis), indent=2),
                       encoding="utf-8")
        tmp.replace(p)
        self._record(p, "cut-analysis")

    def _verify_original(self) -> None:
        """The original is the record: it must be exactly what was cut. The
        write log is asked first; a folder it has never seen (copied in from
        another install) is checked against the hash the cut recorded in its
        own analysis.json, and on a match entered in this write log."""
        dp = _sigmf.data_path(self.path / "original")
        ok, why = self.rf.verify(dp)
        if ok:
            return
        want = (self.analysis.get("cut") or {}).get("original_sha256")
        if "not in the write log" in why and dp.exists():
            if want and _paths.sha256_file(dp) == want:
                self._record(dp, "cut-original-imported",
                             "matched the hash in the cut's own analysis.json")
                self._record(_sigmf.meta_path(self.path / "original"),
                             "cut-original-meta-imported")
                self.analysis.setdefault("history", []).append(
                    {"at": _now(), "by": _prov.who(),
                     "what": ("imported: the original matched the hash the cut "
                              "recorded; entered in this rf_data's write log")})
                return
            raise CutError(
                "the original is not in this rf_data's write log, and "
                + ("its contents do not match the hash the cut recorded when "
                   "it was made — it was changed. It is named, not used."
                   if want else
                   "the cut recorded no hash of it, so it cannot be checked — "
                   "it is named, not used."))
        raise CutError(f"the original was changed after it was cut — {why}")

    def load(self, name: str = "original") -> tuple:
        """(iq, fs, meta) of `original`, `cleaned`, `cleaned_2`, `separated_1`
        … — 1-D for one channel, [channels, n] for a Kraken cut."""
        base = self.path / name
        if not _sigmf.meta_path(base).exists():
            raise CutError(f"this cut has no '{name}'")
        meta = _sigmf.read_meta(base)
        return (_sigmf.load(base, meta=meta), _sigmf.sample_rate_of(meta),
                meta)

    def _floor(self, meta: dict) -> float | None:
        v = meta.get("global", {}).get("atk:floor_per_hz")
        return float(v) if v else None

    # -- step 2: Analyze --------------------------------------------------------
    def analyze(self, classifier=None, fingerprint=None) -> dict:
        """The SCF and cyclic profile (files + labelled peaks), the classical
        measurements, the class-table probes, and the classifier's class
        and embedding or UNKNOWN; the fingerprint if a matcher is given.
        Everything into analysis.json.

        classifier   callable(iq, fs) -> {cls, confidence, embedding, model}
        fingerprint  callable(iq, fs) -> dict
        """
        from atk_diffusion.cyclo import probes as _probes
        from atk_diffusion.cyclo import scf as _scf
        from atk_diffusion.dsp import measure as _measure
        self._verify_original()
        iq, fs, meta = self.load("original")
        x = iq if iq.ndim == 1 else iq[0]
        x = x.astype(np.complex128)
        a = self.analysis
        prof = _profiles.load_profile(self.rf, self.profile)
        # the SCF image (the classifier's second input) and the profile curve
        img, f_ax, a_ax = _scf.scf_image(x, fs, prof.fam)
        np.save(self.path / "scf.npy", img)
        self._record(self.path / "scf.npy", "cut-scf")
        # the display curve: FAM resolution twice the 16 k-point grid's
        # (more blocks only sharpen what the grid cannot show); which peaks
        # are real is decided by the lag-domain detector below
        alphas, prof_nc = _scf.cyclic_profile(x, fs, conj=False,
                                              n_alpha=1 << 14, max_blocks=2048)
        _a2, prof_c = _scf.cyclic_profile(x, fs, conj=True, n_alpha=1 << 14,
                                          max_blocks=2048)
        norm = float(prof_nc[np.argmin(np.abs(alphas))]) if alphas.size else 1.0
        norm = norm or 1.0
        cp = np.vstack([alphas, prof_nc / norm, prof_c / norm]) \
            if alphas.size else np.zeros((3, 0))
        np.save(self.path / "cyclic_profile.npy", cp)
        self._record(self.path / "cyclic_profile.npy", "cut-cyclic-profile")
        pk_nc = _measure.cyclic_peaks(x, fs, conj=False)
        pk_c = _measure.cyclic_peaks(x, fs, conj=True)
        peaks = ([{**p, "threshold": pk_nc["threshold"], "conj": False}
                  for p in pk_nc["peaks"]]
                 + [{**p, "threshold": pk_c["threshold"], "conj": True}
                    for p in pk_c["peaks"]])
        # conjugate peaks other than the strongest are sidebands of it at the
        # symbol rate (2·f_c ± R), not further carriers: say so
        conj = [p for p in peaks if p["conj"]]
        if len(conj) > 1:
            main = max(conj, key=lambda p: p["statistic"])
            rates = [p["alpha_hz"] for p in peaks if not p["conj"]]
            for p in conj:
                if p is main:
                    continue
                d = abs(p["alpha_hz"] - main["alpha_hz"])
                hit = next((r for r in rates if abs(d - abs(r)) <= 0.01 * d
                            + 2 * (pk_c.get("resolution_hz") or 0)), None)
                p["words"] = (f"twice the carrier ± the symbol rate "
                              f"({main['alpha_hz']:+,.1f} ± {d:,.1f} Hz)"
                              if hit else
                              f"a further conjugate feature, {d:,.1f} Hz from "
                              f"the carrier line at {main['alpha_hz']:+,.1f} Hz")
        meas = _measure.measure_all(x, fs, floor_per_hz=self._floor(meta))
        # the class table's own probes at this canonical rate
        rates = {}
        for name, r in _classes.cycle_frequencies(fs):
            c = _classes.get(name)
            if c.cp_lag_s > 0:
                continue
            rates.setdefault(r, []).append(name)
        probe_words, cands, probe_res = [], [], {}
        if rates:
            fams = {r: (_classes.get(ns[0]).family if len({_classes.get(n).family
                                                         for n in ns}) == 1
                        else "") for r, ns in rates.items()}
            res = _probes.symbol_rate_line(x, fs, sorted(rates), pfa=1e-3,
                                           family=fams)
            probe_res["symbol_rate_line"] = {k: res.get(k) for k in (
                "detected", "statistic", "threshold", "best_rate_hz",
                "alpha_hz", "integration_s", "pfa", "words")}
            hit_rates = sorted({row["rate_hz"] for row in res["per_rate"]
                                if row["detected"]})
            # the blind symbol rate (every lag, the comb's fundamental) is
            # the cut's own clock: the class-table probe looks only at the
            # lags its classes' family favours (FSK: about a symbol), where
            # e.g. a rectangular-pulse 2400 Bd signal shows no line at 2400
            # but a strong one at 4800 — measured: offered DSD as a P25
            sr = meas.get("symbol_rate") or {}
            blind = float(sr["value_hz"]) if sr.get("known") else None
            # a line at an integer multiple (2..6) of a detected rate or of
            # the blind rate is that signal's harmonic, not a second signal
            bases = hit_rates + ([blind] if blind else [])
            folded = {}
            for r in hit_rates:
                base = next((b for b in bases if b < 0.99 * r
                             and 2 <= round(r / b) <= 6
                             and abs(r / b - round(r / b)) < 0.01), None)
                if base is not None:
                    folded[r] = base
            for r in hit_rates:
                if r not in folded:
                    cands += rates[r]
            if blind:
                for r in rates:
                    if abs(r - blind) <= 0.005 * r:
                        cands += rates[r]
                        probe_res["symbol_rate_line"]["blind_rate_match"] = {
                            "listed_hz": r, "measured_hz": blind}
            if folded:
                probe_res["symbol_rate_line"]["harmonics_folded"] = {
                    f"{r:g}": f"{b:g}" for r, b in folded.items()}
            probe_words.append(res["words"])
        for name, lag in _classes.cp_lags(fs):
            c = _classes.get(name)
            res = _probes.cp_probe(x, fs, lag, pfa=1e-3,
                                   period_hint_hz=(c.symbol_rates[0]
                                                   if c.symbol_rates else None))
            probe_res[f"cp_probe {name}"] = {k: res.get(k) for k in (
                "detected", "statistic", "threshold", "rho", "cfo_hz",
                "symbol_period_s", "words")}
            if res["detected"]:
                cands.append(name)
                probe_words.append(res["words"])
        a["scf"] = {"file": "scf.npy", "shape": list(img.shape),
                    "f_axis_hz": [float(f_ax[0]), float(f_ax[-1]), len(f_ax)],
                    "alpha_axis_hz": [float(a_ax[0]), float(a_ax[-1]),
                                      len(a_ax)],
                    "geometry": {"channel_fft": prof.fam.channel_fft,
                                 "hop": prof.fam.hop,
                                 "window": prof.fam.window,
                                 "max_seconds": prof.fam.max_seconds},
                    "rows": "frequency", "columns": "cycle frequency",
                    "normalised": "0..1, largest |S| = 1"}
        cands = sorted(set(cands))
        consistency = []
        conj_present = (meas.get("carrier_offset") or {}).get(
            "conjugate_feature") == "present"
        fams = {_classes.get(c).family for c in cands if _classes.get(c)}
        if cands and conj_present and fams == {"fsk"}:
            consistency.append(
                "the rate matched " + ", ".join(cands) + ", which are FSK — but "
                "this cut has a conjugate carrier feature (BPSK / AM / ASK / "
                "MSK-class), which FSK does not make. Treat the candidates "
                "with suspicion: a linear modulation at the same rate fits "
                "better.")
        a["cyclic"] = {"peaks": peaks, "pfa": pk_nc.get("pfa"),
                       "resolution_hz": pk_nc.get("resolution_hz"),
                       "candidates": cands, "consistency": consistency,
                       "probes": probe_res, "probe_words": probe_words,
                       "profile_file": "cyclic_profile.npy",
                       "tier": "measured"}
        meas.pop("duration_s", None)
        a["measurements"] = meas
        a["class"] = self._classify(classifier, x, fs)
        if fingerprint is not None:
            fp = fingerprint(x, fs)
            a["fingerprint"] = fp if isinstance(fp, dict) else {"result": fp}
        else:
            a["fingerprint"] = None
        if iq.ndim == 2 and iq.shape[0] > 1:
            a["analysis_note"] = (f"analysed on channel 0 of {iq.shape[0]}; "
                                  "SCORE and DF use them all")
        a["analyzed_at"] = _now()
        a.setdefault("history", []).append({"at": a["analyzed_at"],
                                            "by": _prov.who(),
                                            "what": "analyze"})
        self._plots(img, f_ax, a_ax, cp, peaks)
        a["files"].update({
            "analysis.json": "every fact about this cut",
            "scf.npy": "SCF image float32 [64, 128], rows f, columns α",
            "scf.png": "the SCF image",
            "cyclic_profile.npy": "float64 [3, n]: α, non-conjugate, "
                                  "conjugate profile",
            "cyclic_profile.png": "the cyclic profile with labelled peaks"})
        self._save()
        return a

    def _classify(self, classifier, x, fs) -> dict:
        det = (self.analysis.get("source") or {}).get("detection") or {}
        if classifier is None:
            out = {"cls": _classes.UNKNOWN, "confidence": None,
                   "tier": "proposed", "source": "none",
                   "why": ("no classifier was given — the learned classifier "
                           "is loaded by its own card for this profile; the "
                           "cyclic probe candidates are listed above")}
        else:
            r = classifier(x, fs)
            if not isinstance(r, dict) or "cls" not in r:
                raise CutError("a classifier returns {cls, confidence, "
                               "embedding, model}")
            emb = r.get("embedding")
            out = {"cls": str(r["cls"]) or _classes.UNKNOWN,
                   "confidence": (float(r["confidence"])
                                  if r.get("confidence") is not None else None),
                   "embedding": (np.asarray(emb, dtype=float).tolist()
                                 if emb is not None else None),
                   "model": str(r.get("model", "")), "tier": "proposed",
                   "source": "classifier"}
        if det.get("cls"):
            out["detector_class"] = det["cls"]
        return out

    def _plots(self, img, f_ax, a_ax, cp, peaks) -> None:
        # The object-oriented API, never pyplot: inside ATK's Qt process
        # matplotlib.use("Agg") + pyplot (the first version) switched the
        # HOST's global backend and could close its figures.
        from matplotlib.figure import Figure
        from atk_diffusion.detect.boxes import CYCLIC_COLOUR
        fig = Figure(figsize=(7, 4), dpi=110)
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(img, origin="lower", aspect="auto", cmap="viridis",
                       extent=[a_ax[0] / 1e3, a_ax[-1] / 1e3, f_ax[0] / 1e3,
                               f_ax[-1] / 1e3])
        ax.set_xlabel("cycle frequency α (kHz)")
        ax.set_ylabel("frequency f (kHz)")
        ax.set_title(f"Spectral correlation |S(f, α)| — {self.path.name}",
                     fontsize=9)
        fig.colorbar(im, ax=ax, label="relative |S|")
        fig.tight_layout()
        fig.savefig(self.path / "scf.png")
        self._record(self.path / "scf.png", "cut-scf-png")
        fig = Figure(figsize=(7, 3.6), dpi=110)
        ax = fig.add_subplot(1, 1, 1)
        if cp.shape[1]:
            pos = cp[0] >= 0
            ax.plot(cp[0][pos] / 1e3, np.maximum(cp[1][pos], 1e-12), lw=0.8,
                    color="#3060a0", label="non-conjugate (symbol / chip rates)")
            ax.plot(cp[0] / 1e3, np.maximum(cp[2], 1e-12), lw=0.8,
                    color="#808080", ls="--",
                    label="conjugate (2 × carrier, BPSK/AM-class)")
            ax.set_yscale("log")
            for p in peaks:
                a = p["alpha_hz"]
                k = int(np.argmin(np.abs(cp[0] - a)))
                yv = max(float(cp[2 if p.get("conj") else 1][k]), 1e-12)
                ax.plot([a / 1e3], [yv], "o", color=CYCLIC_COLOUR, ms=5)
                ax.annotate(f"α = {a:,.0f} Hz\n{p['statistic'] / p['threshold']:.1f}"
                            "× threshold", (a / 1e3, yv), fontsize=7,
                            color=CYCLIC_COLOUR, xytext=(4, 4),
                            textcoords="offset points")
            ax.legend(fontsize=7, loc="upper right")
        else:
            ax.text(0.5, 0.5, "the cut is too short for a cyclic profile",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("cycle frequency α (kHz)")
        ax.set_ylabel("max over f of |S(f, α)|  (PSD peak = 1)")
        ax.set_title("Cyclic domain profile — peaks are clocks the signal "
                     "carries", fontsize=9)
        fig.tight_layout()
        fig.savefig(self.path / "cyclic_profile.png")
        self._record(self.path / "cyclic_profile.png", "cut-cyclic-png")

    # -- step 3: Clean ----------------------------------------------------------
    def clean(self, method: str, **params) -> dict:
        """Run one Clean method and write its output beside the original.

        matched         the parameters (rate, carrier, timing — MEASURED),
                        recorded and handed to the demodulator on Route; for
                        a linear modulation also the matched-filtered signal
                        (`cleaned`, CLEANED) with the SNR at the symbol
                        instants. params: rate_hint_hz, rolloff
        fresh           FRESH clean; α from params (alphas, conj_alphas) or
                        from the analysis (symbol rate, cyclic peaks)
        fresh_separate  params alpha_sets=[...] — one set per signal
        score           multichannel (Kraken) cuts; params alpha, conj, lag
        wiener          the time-invariant baseline
        rfi_mask_interp mask intermittent interference, interpolate (INFERRED)
        diffusion       params denoiser=callable(iq, fs, **kw) -> (y, info)
                        with info["model_sha256"] — another engineer's
                        learned denoiser, INVENTED tier
        Returns the clean record appended to analysis["cleans"]. A filter
        that cannot run on this cut refuses with a CutError in words."""
        from atk_diffusion.cyclo import filters as _filters
        if method not in METHODS:
            raise CutError(f"unknown clean '{method}' — one of "
                           + ", ".join(METHODS))
        tier = _prov.tier_for(METHODS[method])
        self._verify_original()
        iq, fs, meta = self.load("original")
        x = (iq if iq.ndim == 1 else iq[0]).astype(np.complex128)
        floor = self._floor(meta)
        rec = {"method": method, "tier": tier, "at": _now(),
               "by": _prov.who(), "files": []}
        safe = {k: v for k, v in params.items() if not callable(v)}
        rec["params"] = _jsonable(safe)
        if iq.ndim == 2 and iq.shape[0] > 1 and method != "score":
            rec["note"] = (f"applied to channel 0 of {iq.shape[0]}; only SCORE "
                           "uses the whole array")
        outputs = None
        try:
            if method == "matched":
                hint = params.get("rate_hint_hz") or self._rate_hint()
                mp = _filters.matched_parameters(x, fs, rate_hint_hz=hint)
                rec.update({"parameters": {k: mp[k] for k in (
                    "symbol_rate_hz", "carrier_offset_hz", "timing_offset_s",
                    "samples_per_symbol", "confidence")},
                    "parameters_tier": "measured",
                    "method_words": mp["method"]})
                try:
                    y, rep = _filters.matched_filter(
                        x, fs, params=mp, floor_per_hz=floor,
                        rolloff=params.get("rolloff"))
                    rep["parameters_words"] = mp["words"]
                except ValueError as e:
                    rec.update({"tier": _prov.tier_for("matched_parameters"),
                                "words": mp["words"],
                                "no_file": (f"no matched-filtered file: {e}. "
                                            "The parameters above are the "
                                            "result.")})
                    return self._finish_clean(rec, None, None, fs, meta, None)
            elif method == "fresh":
                al = (params["alphas"] if params.get("alphas") is not None
                      else self._alphas())
                cj = params.get("conj_alphas")
                if cj is None:
                    cj = self._conj_alphas()
                if not al and not cj:
                    raise CutError("FRESH needs cycle frequencies: run Analyze "
                                   "first (its peaks are the α), or pass "
                                   "alphas")
                y, rep = _filters.fresh_clean(x, fs, al, cj,
                                              floor_per_hz=floor,
                                              nperseg=params.get("nperseg"))
            elif method == "fresh_separate":
                sets = params.get("alpha_sets")
                if not sets:
                    raise CutError("separation needs alpha_sets — one set of "
                                   "cycle frequencies per signal")
                outputs, rep = _filters.fresh_separate(
                    x, fs, sets, floor_per_hz=floor,
                    nperseg=params.get("nperseg"))
                y = None
            elif method == "score":
                if iq.ndim != 2 or iq.shape[0] < 2:
                    raise CutError("SCORE needs the Kraken's coherent "
                                   "channels; this cut has one channel")
                alpha = params.get("alpha")
                if alpha is None:              # 0.0 is a real α (2·f_c = 0)
                    alpha = self._rate_hint()
                if alpha is None:
                    raise CutError("SCORE needs a cycle frequency: run Analyze "
                                   "or Matched first, or pass alpha")
                y, w, rep = _filters.score(iq, fs, float(alpha),
                                           conj=bool(params.get("conj", False)),
                                           lag=params.get("lag"))
            elif method == "wiener":
                y, rep = _filters.wiener_clean(x, fs, floor_per_hz=floor,
                                               nperseg=params.get("nperseg"))
            elif method == "rfi_mask_interp":
                kw = {k: v for k, v in params.items() if k in (
                    "pfa", "nperseg", "impulse_factor")}
                y, rep = _filters.rfi_mask_interp(x, fs, **kw)
                rep.update(self._blind_snr(x, y, fs, floor))
            else:   # diffusion
                fn = params.get("denoiser")
                if not callable(fn):
                    raise CutError("the diffusion clean needs denoiser="
                                   "callable(iq, fs, **kw) -> (y, info) — the "
                                   "learned denoiser is loaded by its card")
                kw = {k: v for k, v in params.items() if k != "denoiser"}
                y, info = fn(x, fs, **kw)
                info = dict(info or {})
                if not info.get("model_sha256"):
                    raise CutError("a learned clean must name the hash of the "
                                   "model weights that made it (model_sha256) "
                                   "— an INVENTED output that cannot be traced "
                                   "to its model is not written")
                y = np.asarray(y)
                if y.shape != x.shape:
                    raise CutError(f"the denoiser returned {y.shape} samples "
                                   f"for a cut of {x.shape} — a clean keeps "
                                   "the cut's length")
                rep = {"method": "diffusion", "words": info.get(
                    "words", "diffusion denoiser"), **info}
                rep.update(self._blind_snr(x, y, fs, floor))
                rep.setdefault("sizing", "a generative model can invent "
                                         "structure: a lead, never a reading")
        except CutError:
            raise
        except ValueError as e:
            raise CutError(f"{method} cannot run on this cut: {e}") from None
        return self._finish_clean(rec, y, rep, fs, meta, outputs)

    def _blind_snr(self, x, y, fs, floor) -> dict:
        from atk_diffusion.dsp import measure as _measure
        if not floor:
            floor = _measure.noise_floor(x, fs)["floor_per_hz"]
        ob = _measure.occupied_bandwidth(x, fs, floor_per_hz=floor)
        band = ((ob["lower_hz"], ob["upper_hz"]) if ob["value_hz"] else None)
        b = _measure.snr_above_floor(x, fs, band=band, floor_per_hz=floor)
        aft = _measure.snr_above_floor(np.asarray(y), fs, band=band,
                                       floor_per_hz=floor)
        return {"snr_before_db": b["snr_db"], "snr_after_db": aft["snr_db"],
                "snr_method": ("blind: in-band power over the SAME floor (the "
                               "source span's), in the original's occupied "
                               "band, before and after — interference removed "
                               "lowers it, so for an interference clean read "
                               "it with the masked fraction")}

    def _rate_hint(self):
        m = (self.analysis.get("measurements") or {}).get("symbol_rate") or {}
        if m.get("known"):
            return float(m["value_hz"])
        det = (self.analysis.get("source") or {}).get("detection") or {}
        return det.get("alpha_hz")

    def _alphas(self) -> list:
        r = self._rate_hint()
        if r:
            return [float(r)]
        return [p["alpha_hz"] for p in (self.analysis.get("cyclic") or {})
                .get("peaks", []) if not p.get("conj")][:2]

    def _conj_alphas(self) -> list:
        return [p["alpha_hz"] for p in (self.analysis.get("cyclic") or {})
                .get("peaks", []) if p.get("conj")][:3]

    def _next_name(self, stem: str) -> str:
        if not _sigmf.meta_path(self.path / stem).exists():
            return stem
        k = 2
        while _sigmf.meta_path(self.path / f"{stem}_{k}").exists():
            k += 1
        return f"{stem}_{k}"

    #: Report keys that are the parameters a method actually used.
    _EFFECTIVE = {
        "matched": ("symbol_rate_hz", "carrier_offset_hz", "timing_offset_s",
                    "rolloff", "rolloff_method", "samples_per_symbol", "taps",
                    "floor_per_hz", "instants"),
        "fresh": ("alphas_hz", "conj_alphas_hz", "nperseg", "floor_per_bin",
                  "floor_method", "branches"),
        "fresh_separate": ("nperseg", "floor_per_bin", "floor_method",
                           "branches", "gate"),
        "score": ("alpha_hz", "conj", "lag", "weights", "channels"),
        "wiener": ("nperseg", "floor_per_bin", "floor_method"),
        "rfi_mask_interp": ("pfa_per_cell", "nperseg", "impulse_factor"),
        "diffusion": ("model_sha256", "timestep", "steps", "model",
                      "snr_matched_timestep"),
    }

    def _finish_clean(self, rec, y, rep, fs, meta, outputs) -> dict:
        a = self.analysis
        g0 = meta.get("global", {})
        if rep is not None:
            rep = _jsonable({k: v for k, v in rep.items()})
            rec.update({"snr_before_db": rep.get("snr_before_db"),
                        "snr_after_db": rep.get("snr_after_db"),
                        "snr_method": rep.get("snr_method", ""),
                        "sizing": rep.get("sizing", ""),
                        "words": rep.get("words", ""),
                        "report": rep})
            if rep.get("model_sha256"):
                rec["model_sha256"] = rep["model_sha256"]
            if rec.get("snr_before_db") is not None and rec.get(
                    "snr_after_db") is not None:
                rec["gain_db"] = rec["snr_after_db"] - rec["snr_before_db"]
            used = {k: rep[k] for k in self._EFFECTIVE.get(rec["method"], ())
                    if k in rep}
            if rec["method"] == "fresh_separate":
                used["signals"] = [{k: s.get(k) for k in (
                    "alphas_hz", "conj_alphas_hz", "power_method")}
                    for s in rep.get("signals", [])]
            rec["method_params"] = {**rec["params"], **used}
        else:
            rec["method_params"] = dict(rec["params"])
        ids = [c.get("id") for c in a.get("cleans", [])]
        rec["id"] = f"clean_{len(ids) + 1}"
        cap0 = dict((meta.get("captures") or [{}])[0])
        centre = float(cap0.get("core:frequency", 0.0))
        out_method = (METHODS[rec["method"]] if rec["tier"] != "measured"
                      else "matched_parameters")

        def write(name: str, sig, extra: dict) -> None:
            g = {"atk:receiver_profile": g0.get("atk:receiver_profile"),
                 "atk:tier": rec["tier"], "atk:method": out_method,
                 "atk:method_params": rec["method_params"],
                 "atk:snr_before_db": rec.get("snr_before_db"),
                 "atk:snr_after_db": rec.get("snr_after_db"),
                 "atk:snr_method": rec.get("snr_method", ""),
                 "atk:cleaned_from": "original",
                 "atk:canonical_class": g0.get("atk:canonical_class"),
                 "atk:canonical_rate": g0.get("atk:canonical_rate"),
                 "atk:decimation": g0.get("atk:decimation"),
                 "atk:box": g0.get("atk:box"),
                 "atk:floor_per_hz": g0.get("atk:floor_per_hz"),
                 "atk:source_capture": g0.get("atk:source_capture", ""),
                 "atk:tier_words": _prov.TIER_WORDS.get(rec["tier"], "")}
            if rec.get("model_sha256"):
                g["atk:model_sha256"] = rec["model_sha256"]
            g.update(extra)
            base = self.path / name
            _sigmf.write_pair(
                base, np.asarray(sig, dtype=np.complex64), fs,
                centre, datatype="cf32", extra_global=g,
                description=(f"{rec['method']} clean of the original — "
                             f"{rec['tier'].upper()}, not the record"),
                recorder="ATK Diffusion Toolkit — signal cut")
            # the signal's own time, not the time it was cleaned
            m2 = _sigmf.read_meta(base)
            cap = {"core:sample_start": 0, "core:frequency": centre}
            if cap0.get("core:datetime"):
                cap["core:datetime"] = cap0["core:datetime"]
            m2["captures"] = [cap]
            _sigmf.write_meta(base, m2)
            self._record(_sigmf.data_path(base), f"cut-{rec['method']}",
                         rec["tier"])
            self._record(_sigmf.meta_path(base), f"cut-{rec['method']}-meta")
            rec["files"].append(name)
            a["files"][f"{name}.sigmf-data / .sigmf-meta"] = (
                f"{rec['method']} output — {rec['tier'].upper()}")

        if outputs is not None:
            for i, ys in enumerate(outputs, start=1):
                per = (rep.get("signals") or [{}])[i - 1] if rep else {}
                write(self._next_name(f"separated_{i}"), ys,
                      {"atk:separated_index": i,
                       "atk:snr_before_db": per.get("sinr_before_db"),
                       "atk:snr_after_db": per.get("sinr_after_db"),
                       "atk:snr_kind": "SINR — the other signals count as "
                                       "interference",
                       "atk:alphas_hz": per.get("alphas_hz"),
                       "atk:conj_alphas_hz": per.get("conj_alphas_hz"),
                       "atk:power_method": per.get("power_method")})
        elif y is not None:
            write(self._next_name("cleaned"), y, {})
        a.setdefault("cleans", []).append(rec)
        a.setdefault("history", []).append({"at": rec["at"], "by": rec["by"],
                                            "what": f"clean {rec['method']}"})
        self._save()
        return rec

    # -- step 4: Route ----------------------------------------------------------
    def routes_available(self) -> list[str]:
        """Every tool that accepts the cut's class (or any candidate class),
        then bench, df, ask, teach, fingerprint, save."""
        from atk_diffusion.cut import route as _route
        return _route.available(self.analysis)

    def route(self, tool: str, input: str = "original", runner=None) -> dict:
        """Record a route and, with a host `runner(iq, fs, meta) -> dict`,
        run it on `input` ("original", "cleaned" = the latest clean, or a
        file stem such as "cleaned_2" / "separated_1")."""
        from atk_diffusion.cut import route as _route
        self._verify_original()
        name = input
        if input == "cleaned":
            made = [f for c in self.analysis.get("cleans", [])
                    for f in c.get("files", [])]
            if not made:
                raise CutError("there is no cleaned file yet — run Clean "
                               "first, or route the original")
            name = made[-1]
        iq, fs, meta = self.load(name)
        try:
            rec = _route.perform(self.analysis, tool, name, iq, fs, meta,
                                 runner=runner)
        except CutError:
            raise
        except ValueError as e:
            raise CutError(str(e)) from None
        self.analysis.setdefault("routes", []).append(rec)
        self.analysis.setdefault("history", []).append(
            {"at": rec["at"], "by": rec["by"], "what": f"route {tool}"})
        self._save()
        return rec

    # -- step 5: Report ---------------------------------------------------------
    def report(self) -> Path:
        """report.md from analysis.json's facts only."""
        from atk_diffusion.cut import report as _report
        self.analysis["files"]["report.md"] = "this cut in words"
        self._save()
        return _report.write_report(self)
