# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The training library's independent review (2026-10-26), one test per
finding that the toolkit owns: what keeps a grade honest when the operator
does the natural thing — two boxes on one recording, the same cut twice,
a role flipped after the generator learned — and what keeps a verdict from
counting noise as a signal."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from atk_diffusion import sigmf
from atk_diffusion.library import build, generative, store, train
from atk_diffusion.synth import native

PID = "rtlsdr_1024000_cu8"
FS = 1_024_000.0
F0 = 162.4e6


def _capture(rf, name="cap", seconds=3.4, bursts=((0.3, 1.0),), offset=150e3,
             seed=1, snr_db=25.0):
    """Noise with one or more bursts of the same NFM signal."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    pn = 10 ** (-35 / 10)
    x = native.noise(n, pn, rng)
    lab = None
    for t0, dur in bursts:
        s, lab = native.waveform("nfm_voice", FS, int(dur * FS), rng,
                                 carrier_offset_hz=offset)
        a = native.signal_amplitude(snr_db, lab["bandwidth_hz"], pn, FS)
        i0 = int(t0 * FS)
        x[i0:i0 + s.size] += (a * s).astype(np.complex64)
    base = rf.captures(PID) / name
    sigmf.write_pair(base, x, FS, F0, datatype="cu8",
                     extra_global={"atk:receiver_profile": PID})
    return base, lab


def _box(lab, t0, t1, pad_hz=6e3):
    return {"t0_s": t0, "t1_s": t1, "f_lo_hz": F0 + lab["f_lo_hz"] - pad_hz,
            "f_hi_hz": F0 + lab["f_hi_hz"] + pad_hz}


@pytest.fixture
def lib(rf):
    return store.TrainingLibrary(rf, PID)


# -- 1: one recording, one role; an overlap is refused -----------------------
def test_two_boxes_on_one_recording_share_its_role(rf, lib):
    lib.new_type("fire", link="nfm_voice")
    base, lab = _capture(rf, "one", bursts=((0.3, 1.0), (1.9, 1.0)))
    a = lib.add_original_from_capture(base, _box(lab, 0.25, 1.35), "fire")
    b = lib.add_original_from_capture(base, _box(lab, 1.85, 2.95), "fire")
    assert a.role == b.role == "train"           # the 2nd BOX is not the 2nd recording
    assert a.group == b.group
    base2, lab2 = _capture(rf, "two", seed=2)
    c = lib.add_original_from_capture(base2, _box(lab2, 0.25, 1.35), "fire")
    assert c.role == "test"                      # the 2nd RECORDING grades
    assert build._separate_test_recordings(lib, lib.get_type("fire")) == 1


def test_an_overlapping_box_is_refused_in_words(rf, lib):
    lib.new_type("fire", link="nfm_voice")
    base, lab = _capture(rf, "one")
    lib.add_original_from_capture(base, _box(lab, 0.25, 1.35), "fire")
    with pytest.raises(store.LibraryError, match="overlaps orig_0001"):
        lib.add_original_from_capture(base, _box(lab, 0.9, 1.3), "fire")


def test_a_role_set_on_one_box_moves_its_whole_recording(rf, lib):
    lib.new_type("fire", link="nfm_voice")
    base, lab = _capture(rf, "one", bursts=((0.3, 1.0), (1.9, 1.0)))
    a = lib.add_original_from_capture(base, _box(lab, 0.25, 1.35), "fire")
    b = lib.add_original_from_capture(base, _box(lab, 1.85, 2.95), "fire")
    lib.set_role("fire", a.id, "test")
    assert lib.example("fire", b.id).role == "test"
    hist = lib.meta(lib.example("fire", b.id))["global"]["atk:history"]
    assert "same recording" in hist[-1]["what"]


def test_a_cut_cannot_go_in_twice(rf, lib, tmp_path):
    lib.new_type("fire", link="nfm_voice")
    base, lab = _capture(rf, "one")
    ex = lib.add_original_from_capture(base, _box(lab, 0.25, 1.35), "fire")
    # a cut folder made from that original (its own samples, no capture)
    d = tmp_path / "cut1"
    d.mkdir()
    x = lib.load(ex)
    g = {k: v for k, v in lib.meta(ex)["global"].items()
         if k in ("atk:canonical_class", "atk:decimation", "atk:lowpass_hz")}
    g["atk:receiver_profile"] = PID
    sigmf.write_pair(d / "original", x, ex.rate, F0, datatype="cf32", extra_global=g)
    lib.add_original_from_cut(d, "fire")
    with pytest.raises(store.LibraryError, match="already in 'fire'"):
        lib.add_original_from_cut(d, "fire")


# -- 4: the held-back tail is signal, not the noise margin --------------------
def test_the_holdout_and_the_train_span_are_inside_the_signal(rf, lib):
    lib.new_type("fire", link="nfm_voice")
    base, lab = _capture(rf, "one", bursts=((0.5, 1.0),))
    ex = lib.add_original_from_capture(base, _box(lab, 0.45, 1.55), "fire")
    a, n = store.signal_span(ex)
    assert a > 0 and a + n < ex.samples
    s0, sn = lib.train_span(ex)
    assert s0 == a and s0 + sn == ex.holdout["start"]
    h0, hn = ex.holdout["start"], ex.holdout["count"]
    assert h0 + hn == a + n
    # the held-back tail is the signal: its power stands well above the margin's
    x = lib.load(ex)
    tail = np.mean(np.abs(x[h0:h0 + hn]) ** 2)
    margin = np.mean(np.abs(x[:max(1, a - 64)]) ** 2)
    assert tail > 10 * margin


# -- 2: the verdict grades on real windows only --------------------------------
def _card(per, acc, classes, metrics_extra=None):
    m = {"accuracy": acc, "per_class_accuracy": per}
    m.update(metrics_extra or {})
    return SimpleNamespace(metrics=m, class_names=lambda: list(classes),
                           calibration={})


def _man(test_counts):
    counts = {k: {"test": {"real": v}} for k, v in test_counts.items()}
    return {"library": {"counts": counts,
                        "test_sources": {k: 1 for k in test_counts}}}


def test_noise_windows_never_make_a_grade():
    # a type with nothing held out, noise perfect: NOT graded, not "100 %"
    v = train.recognizer_verdict(_card({"fire": None, "noise": 1.0}, 1.0,
                                       ["fire", "noise"]), _man({"fire": 0}))
    assert v["level"] == "unmeasured" and "100" not in v["headline"]
    # two types each 1 of 2 right, noise 8 of 8: 50 %, not 83 %
    v = train.recognizer_verdict(
        _card({"a": 0.5, "b": 0.5, "noise": 1.0}, 10 / 12, ["a", "b", "noise"]),
        _man({"a": 2, "b": 2}))
    assert v["accuracy"] == pytest.approx(0.5)
    assert "50 %" in v["headline"] and v["level"] == "not_ready"
    assert any("NOISE" in r for r in v["reasons"])


def test_real_accuracy_weights_by_real_windows():
    acc, n = train.real_accuracy({"a": 1.0, "b": 0.0, "noise": 1.0},
                                 _man({"a": 3, "b": 1})["library"]["counts"],
                                 ["a", "b", "noise"])
    assert acc == pytest.approx(0.75) and n == 4


# -- 6: the finder's held-out scenes hold kept-aside signals -------------------
def test_finder_scenes_grade_on_kept_aside_signals(rf, lib):
    lib.new_type("fire", link="nfm_voice")
    for i in range(2):
        base, lab = _capture(rf, f"r{i}", seed=5 + i)
        lib.add_original_from_capture(base, _box(lab, 0.25, 1.35), "fire")
    pools = build._library_pools(lib)
    assert pools["held_how"] == "kept_aside"
    held_ids = {e.id for _st, e, _s in pools["held"]}
    train_ids = {e.id for _st, e, _s in pools["train"]}
    assert held_ids and not (held_ids & train_ids)
    v = train.finder_verdict(SimpleNamespace(metrics={"ap50_any": 0.9}),
                             {"library": {"heldout_pool": "reused"}})
    assert v["level"] == "usable" and "SAME signals" in v["reasons"][0]


# -- 8: a generator taught on what now grades is refused ----------------------
def test_a_generator_taught_on_a_now_kept_aside_original_is_stale(rf, lib):
    lib.new_type("fire", link="nfm_voice")
    ids = []
    for i in range(3):
        base, lab = _capture(rf, f"r{i}", seed=7 + i)
        ids.append(lib.add_original_from_capture(base, _box(lab, 0.25, 1.35),
                                                 "fire").id)
    inp = {"originals": {"fire": [ids[0], ids[2]]}}     # the train ones
    assert generative.stale_originals(lib, inp) == []
    lib.set_role("fire", ids[2], "val")
    assert generative.stale_originals(lib, inp) == [f"fire/{ids[2]}"]


def test_fresh_copies_measure_train_originals_only(rf, lib):
    from atk_diffusion.library import fresh
    lib.new_type("ref", link="ref_bpsk")
    for i in range(2):
        base, lab = _capture(rf, f"r{i}", seed=11 + i)
        lib.add_original_from_capture(base, _box(lab, 0.25, 1.35), "ref")
    _p, words = fresh._matched_params(lib, lib.get_type("ref"))
    assert all("1 original" in w for w in words)          # the test one is not used


# -- 7, 10: every-type jobs and the write log ----------------------------------
def test_an_every_type_job_skips_an_empty_type(rf, lib, tmp_path):
    from atk_diffusion import cli
    lib.new_type("a_new", link="nfm_voice")
    lib.new_type("fire", link="nfm_voice")
    base, lab = _capture(rf, "one")
    lib.add_original_from_capture(base, _box(lab, 0.25, 1.35), "fire")
    out = tmp_path / "r.json"
    code = cli.main(["library", "dirty", "--profile", PID, "--type", "all",
                     "--per-original", "2", "--preset", "light",
                     "--rf-data", str(rf.root), "--result-file", str(out)])
    assert code == 0
    res = json.loads(out.read_text())["result"]["results"]
    by = {r["type"]: r for r in res}
    assert by["a_new"]["made"] == 0 and "skipped" in by["a_new"]["words"]
    assert by["fire"]["made"] == 2


def test_a_stop_file_ends_a_job_cleanly(rf, lib, tmp_path):
    from atk_diffusion import cli
    lib.new_type("fire", link="nfm_voice")
    base, lab = _capture(rf, "one")
    lib.add_original_from_capture(base, _box(lab, 0.25, 1.35), "fire")
    stop = tmp_path / "STOP"
    stop.write_text("stop")
    out = tmp_path / "r.json"
    cli.main(["library", "dirty", "--profile", PID, "--type", "all",
              "--per-original", "5", "--preset", "light", "--rf-data",
              str(rf.root), "--result-file", str(out), "--stop-file", str(stop)])
    res = json.loads(out.read_text())
    made = sum(r.get("made", 0) for r in (res.get("result") or {}).get("results", []))
    assert made == 0
    assert not list((rf.training(PID, "fire") / "dirty").glob(".*"))
