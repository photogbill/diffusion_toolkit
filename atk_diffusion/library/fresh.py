# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Fresh copies: the signal type re-made by the native generator, with NEW
content, at the receiver's exact rate and through its measured front end —
then cut the way the field cut is (RFMLS Training, 2026-10-26).

A dirty copy is one transmission made worse; a fresh copy is another
transmission. One recording of a DMR radio holds one stretch of voice
frames; a classifier that has only ever seen those frames can learn THEM
rather than DMR. The native generator (`synth.native`) makes the class's real
physical layer — symbol rate, deviation, pulse shape, sync words, TDMA
timing — with random payload bits every time, so fresh copies give a type
the variety of content one recording cannot.

MATCHED TO THE TYPE. When the type is linked to a class-table class with a
native generator, the class's own parameters are used, and where the
generator accepts them and the type's REAL originals measured them, the
measured symbol rate and occupied bandwidth (the median over the originals)
take their place — so a "reference QPSK" type is re-made at the rate the
operator's signal actually runs at, not at a random one.

THE FIELD CUT, IMITATED (the dataset builder's own rule, `synth.datasets.
_example`): generated somewhere in the receiver's band, through the
profile's measured impairments (or a textbook receiver, said), quantised to
its datatype, the box its 99 % band (never narrower than two STFT bins) with
the centre missed by a little, shifted, low-passed with the 1.25 guard and
INTEGER-decimated to the type's canonical rate.

LIMITS, stated. Only types linked to a class the generator makes can have
fresh copies (`can_make`, with the reason when not). The generator models the
physical layer, not the voice codec or the FEC payload (`synth.native`'s own
fidelity note). A fresh copy is INVENTED tier and is never used to grade a
model.
"""

from __future__ import annotations

import math
import time

import numpy as np

from atk_diffusion import profiles as _profiles
from atk_diffusion.detect import classes as _classes
from atk_diffusion.dsp import impair as _impair
from atk_diffusion.dsp import resample as _resample
from atk_diffusion.library import store as _store
from atk_diffusion.synth import native as _native

CUT_MARGIN = 32                 # canonical samples each side for the filter
BOX_CENTRE_ERROR = 0.05
MAX_PER_TYPE = 20000
DEFAULT_LENGTH = 2 * _store.WINDOW


def _matched_params(lib: _store.TrainingLibrary, st) -> tuple[dict, list[str]]:
    """Parameters the generator takes that the originals measured."""
    c = _classes.get(st.link)
    kind = c.native if c else ""
    accept = _native._ACCEPT.get(kind, set())
    words: list[str] = []
    params: dict = {}
    # a fixed-protocol class (bandwidth known, a published symbol rate) is
    # generated as the standard says; only the references take measurements
    if c is None or c.bandwidth_hz > 0:
        return params, words
    rates, bws = [], []
    # TRAIN originals only: a kept-aside original grades the model, so
    # nothing measured on it may shape what the model trains on
    for ex in lib.examples(st.name, "originals"):
        if (ex.role or "train") != "train":
            continue
        m = (lib.meta(ex)["global"].get("atk:measured") or {})
        if m.get("symbol_rate_hz"):
            rates.append(float(m["symbol_rate_hz"]))
        if m.get("bandwidth_hz"):
            bws.append(float(m["bandwidth_hz"]))
    if rates and "symbol_rate" in accept:
        params["symbol_rate"] = float(np.median(rates))
        words.append(f"symbol rate {params['symbol_rate']:,.0f} Bd (measured "
                     f"on {len(rates)} original{'s' if len(rates) != 1 else ''})")
    elif bws and "bandwidth_hz" in accept:
        params["bandwidth_hz"] = float(np.median(bws))
        words.append(f"bandwidth {params['bandwidth_hz']:,.0f} Hz (measured on "
                     f"{len(bws)} original{'s' if len(bws) != 1 else ''})")
    return params, words


def can_make(lib: _store.TrainingLibrary, type_name: str) -> tuple[bool, str]:
    """(ok, why) — whether fresh copies can be made for this type here."""
    st = lib.get_type(type_name)
    if not st.link:
        return False, (f"'{st.words}' is not linked to a known kind of signal, "
                       "so there is no generator to re-make it. Link it (Edit "
                       "the type → \"It is a kind of…\") or use dirty and "
                       "AI-generated copies.")
    c = _classes.get(st.link)
    if c is None or not c.native:
        return False, (f"there is no generator for {c.label if c else st.link} "
                       "yet; use dirty and AI-generated copies")
    if not st.canonical:
        return False, ("add one real example first — the type's width (its "
                       "rate class) comes from it")
    params, _ = _matched_params(lib, st)
    ok, why = _native.can_generate(st.link, lib.fs, params or None)
    if not ok:
        return False, f"the generator cannot make {c.label} at this rate: {why}"
    can = lib.rate_of(st.canonical)
    if c.bandwidth_hz > 0.9 * can.rate:
        return False, (f"{c.label} is wider than the {st.canonical} cut rate "
                       f"({can.rate:,.0f} S/s)")
    return True, ""


def make_one(lib: _store.TrainingLibrary, st, params: dict, length: int,
             snr_range, rng) -> tuple[np.ndarray, dict]:
    """One fresh example at the type's canonical rate. -> (x, facts)."""
    prof = lib.profile
    fs = float(prof.sample_rate)
    can = lib.rate_of(st.canonical)
    d, rate = int(can.decimation), float(can.rate)
    L = int(length)
    n = (L + 2 * CUT_MARGIN) * d
    w0, w1 = CUT_MARGIN * d, (CUT_MARGIN + L) * d
    c = _classes.get(st.link)
    measured = _impair.is_measured(prof.impairments)
    floor_dbfs = 10.0 * math.log10(_impair.noise_power(
        prof.impairments if measured else None))
    lab = None
    for _attempt in range(8):
        snr = float(rng.uniform(*snr_range))
        width = max(c.bandwidth_hz, float(params.get("bandwidth_hz", 0.0)), 0.0)
        span = 0.45 * fs - 0.5 * max(width, rate)
        f_c = 0.0 if span <= 0 else float(rng.uniform(-span, span))
        x, lab = _native.generate(st.link, fs, n, snr, rng, carrier_offset_hz=f_c,
                                  noise_dbfs=floor_dbfs, params=params or None)
        if any(s < w1 and s + k > w0 for s, k in lab["bursts"]):
            break
    if measured:
        x = _impair.apply_impairments(x, fs, prof.impairments, rng,
                                      adc_bits=prof.adc_bits or None,
                                      datatype=prof.datatype)
    else:
        x = _impair.quantise(x, prof.datatype, prof.adc_bits or None)
    rbw = prof.stft.rbw_hz(fs)
    box_bw = max(float(lab["bandwidth_hz"]), 2.0 * rbw)
    centre = 0.5 * (lab["f_lo_hz"] + lab["f_hi_hz"])
    err = float(np.clip(rng.normal(0.0, BOX_CENTRE_ERROR * box_bw),
                        -3 * BOX_CENTRE_ERROR * box_bw, 3 * BOX_CENTRE_ERROR * box_bw))
    f_cut = centre + err
    cutoff = min(0.5 * box_bw * 1.25, 0.49 * rate)
    y, _fs_c = _resample.decimate(_resample.shift(x, f_cut, fs), d, fs,
                                  cutoff_hz=cutoff)
    win = np.ascontiguousarray(y[CUT_MARGIN:CUT_MARGIN + L], dtype=np.complex64)
    if win.size < L:
        win = np.concatenate([win, np.zeros(L - win.size, np.complex64)])
    facts = {"snr_db": snr, "symbol_rate_hz": float(lab.get("symbol_rate_hz", 0.0) or 0.0),
             "carrier_offset_hz": float(lab["carrier_offset_hz"]) - f_cut,
             "bandwidth_hz": float(lab.get("bandwidth_hz", 0.0) or 0.0),
             "lowpass_hz": cutoff, "rate": rate,
             "receiver": _impair.describe(prof.impairments),
             "receiver_measured": bool(measured),
             "params": _native._jsonable(lab.get("params", {}))}
    return win, facts


def make_fresh(lib: _store.TrainingLibrary, type_name: str, count: int,
               snr_range=(0.0, 25.0), seed: int = 0, length: int = DEFAULT_LENGTH,
               progress=None, should_stop=None) -> dict:
    """`count` fresh copies of a type, filed under <type>\\fresh\\."""
    st = lib.get_type(type_name)
    ok, why = can_make(lib, type_name)
    if not ok:
        raise _store.LibraryError(why)
    k = int(count)
    if k < 1 or k > MAX_PER_TYPE:
        raise _store.LibraryError(f"fresh copies per type is 1–{MAX_PER_TYPE}")
    lo, hi = (float(v) for v in snr_range)
    if hi < lo:
        raise _store.LibraryError("the SNR range is low, high")
    params, pwords = _matched_params(lib, st)
    rng = np.random.default_rng(int(seed))
    ids = lib.next_copy_ids(st.name, "fresh", "", k)
    t0 = time.time()
    made = 0
    for i, cid in enumerate(ids):
        if should_stop is not None and should_stop():
            break
        x, facts = make_one(lib, st, params, int(length), (lo, hi), rng)
        extra = {"atk:parent": "", "atk:generator": _native.GENERATOR,
                 "atk:generator_class": st.link, "atk:matched": pwords,
                 "atk:seed": [int(seed), i], "atk:snr_db": facts["snr_db"],
                 "atk:carrier_offset_hz": facts["carrier_offset_hz"],
                 "atk:lowpass_hz": facts["lowpass_hz"],
                 "atk:canonical_rate": facts["rate"],
                 "atk:measured": {"bandwidth_hz": facts["bandwidth_hz"],
                                  "symbol_rate_hz": facts["symbol_rate_hz"]
                                  or None, "snr_db": facts["snr_db"],
                                  "tier": "label (known exactly: generated)"},
                 "atk:receiver": facts["receiver"],
                 "atk:receiver_measured": facts["receiver_measured"],
                 "atk:generator_params": facts["params"],
                 "atk:snr_definition": _native.SNR_DEFINITION}
        lib.write_copy(st.name, "fresh", cid, x, facts["rate"], 0.0, extra,
                       f"fresh copy of {st.words} — generated, never received")
        made += 1
        if progress is not None and (made == k or made % max(1, k // 50) == 0):
            progress(f"[{100 * made // k}%] fresh copies of {st.words}: {made}/{k}")
    stopped = made < k
    return {"type": st.name, "kind": "fresh", "made": made, "asked": k,
            "seconds": round(time.time() - t0, 2), "matched": pwords,
            "receiver": _impair.describe(lib.profile.impairments),
            "stopped": stopped,
            "words": (f"{made} fresh cop{'ies' if made != 1 else 'y'} of "
                      f"{st.words}" + (" (stopped before the end)" if stopped else ""))}
