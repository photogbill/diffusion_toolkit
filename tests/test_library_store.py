# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The training library (RFMLS Training, 2026-10-26): one folder per signal
type per receiver, real originals cut the way the field cut is, roles that
keep the grade honest, removal that can be undone, and dirty copies that say
what was done to them — checked on synthetic recordings."""

from __future__ import annotations

import json

import numpy as np
import pytest

from atk_diffusion import profiles, sigmf
from atk_diffusion.library import dirty, store
from atk_diffusion.synth import native

PID = "rtlsdr_1024000_cu8"
FS = 1_024_000.0
F0 = 162.4e6


def _capture(rf, name="cap", seconds=1.6, offset=150e3, cls="nfm_voice",
             snr_db=25.0, seed=1, burst=(0.3, 1.0), pid=PID, fs=FS):
    """Noise with one signal from `burst[0]` for `burst[1]` seconds, `offset`
    Hz above F0, as the receiver recorded it (cu8)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    pn = 10 ** (-35 / 10)
    x = native.noise(n, pn, rng)
    s, lab = native.waveform(cls, fs, int(burst[1] * fs), rng,
                             carrier_offset_hz=offset)
    a = native.signal_amplitude(snr_db, lab["bandwidth_hz"], pn, fs)
    i0 = int(burst[0] * fs)
    x[i0:i0 + s.size] += (a * s).astype(np.complex64)
    base = rf.captures(pid) / name
    dp, _mp = sigmf.write_pair(base, x, fs, F0, datatype="cu8",
                               hw="RTL-SDR v3",
                               extra_global={"atk:receiver_profile": pid})
    rf.record(dp, "capture")
    return base, lab


def _box(lab, burst=(0.3, 1.0), pad_hz=6e3):
    return {"t0_s": burst[0], "t1_s": burst[0] + burst[1],
            "f_lo_hz": F0 + lab["f_lo_hz"] - pad_hz,
            "f_hi_hz": F0 + lab["f_hi_hz"] + pad_hz}


@pytest.fixture
def lib(rf):
    return store.TrainingLibrary(rf, PID)


def test_the_folder_is_one_per_signal_type_inside_the_receiver(rf, lib):
    st = lib.new_type("County Fire NFM", label="county fire dispatch",
                      link="nfm_voice")
    assert st.name == "county_fire_nfm"
    d = rf.training(PID, st.name)
    assert d == rf.root / PID / "training" / "county_fire_nfm"
    for k in store.KINDS:
        assert (d / k).is_dir()
    assert (rf.root / PID / "training" / "README.txt").exists()
    assert st.family == "fm" and st.canonical == "voice"
    with pytest.raises(store.LibraryError, match="already has"):
        lib.new_type("county_fire_nfm")
    with pytest.raises(store.LibraryError, match="reserved"):
        lib.new_type("noise")
    with pytest.raises(store.LibraryError, match="class table"):
        lib.new_type("x1", link="not_a_class")
    with pytest.raises(store.LibraryError, match="family"):
        lib.new_type("x2", family="psk-ish")


def test_a_real_original_is_cut_like_the_field_and_says_what_it_is(rf, lib):
    base, lab = _capture(rf)
    lib.new_type("fire", link="nfm_voice")
    ex = lib.add_original_from_capture(base, _box(lab), "fire", who="tester")
    assert ex.kind == "originals" and ex.role == "train"
    meta = sigmf.read_meta(ex.base)
    g = meta["global"]
    assert g["atk:tier"] == "record"
    assert g["atk:receiver_profile"] == PID
    assert g["atk:canonical_class"] == "voice"
    can = {c.cls: c for c in profiles.canonical_rates(FS)}["voice"]
    assert g["core:sample_rate"] == pytest.approx(can.rate)
    # re-cut around the MEASURED band: narrower than the generous box
    assert any("re-cut" in n for n in g["atk:notes"])
    assert g["atk:lowpass_hz"] < 0.5 * (_box(lab)["f_hi_hz"] - _box(lab)["f_lo_hz"]) * 1.25
    m = g["atk:measured"]
    assert m["bandwidth_hz"] == pytest.approx(lab["bandwidth_hz"], rel=0.35)
    assert m["snr_db"] is not None and m["snr_db"] > 10
    # the write log knows both files
    ok, _why = rf.verify(sigmf.data_path(ex.base))
    assert ok
    # it is long enough to hold a segment back — the END OF THE SIGNAL,
    # never the noise margin after it (2026-10-26 review)
    sp = ex.signal_span
    assert sp and 0 < sp["start"] and sp["start"] + sp["count"] < ex.samples
    assert ex.holdout and ex.holdout["start"] + ex.holdout["count"] == \
        sp["start"] + sp["count"]


def test_roles_keep_every_second_and_fourth_original_aside(rf, lib):
    lib.new_type("fire", link="nfm_voice")
    roles = []
    for i in range(5):
        base, lab = _capture(rf, name=f"c{i}", seed=10 + i)
        roles.append(lib.add_original_from_capture(base, _box(lab), "fire").role)
    assert roles == ["train", "test", "train", "val", "train"]
    c = lib.counts("fire")
    assert c["originals"] == 5 and c["roles"] == {"train": 3, "val": 1, "test": 1}
    test_ex = [e for e in lib.examples("fire", "originals") if e.role == "test"][0]
    with pytest.raises(store.LibraryError, match="grade the model"):
        lib.train_span(test_ex)
    # flipping a role is allowed and recorded
    lib.set_role("fire", test_ex.id, "train")
    assert lib.example("fire", test_ex.id).role == "train"


def test_a_signal_of_another_width_is_refused_for_a_type_in_words(rf, lib):
    lib.new_type("pagers", link="pocsag")
    base, lab = _capture(rf, cls="fm_broadcast", offset=-200e3)
    with pytest.raises(store.LibraryError, match="wide"):
        lib.add_original_from_capture(base, _box(lab, pad_hz=20e3), "pagers")


def test_another_receiver_is_refused(rf, lib):
    other = "rtlsdr_2048000_cu8"
    base, lab = _capture(rf, pid=other, fs=2_048_000.0)
    lib.new_type("fire", link="nfm_voice")
    with pytest.raises(profiles.ProfileMismatch):
        lib.add_original_from_capture(base, _box(lab), "fire")


def test_remove_and_restore_never_delete(rf, lib):
    base, lab = _capture(rf)
    lib.new_type("fire", link="nfm_voice")
    ex = lib.add_original_from_capture(base, _box(lab), "fire")
    lib.remove_example("fire", ex.id)
    assert lib.examples("fire") == []
    assert [e.id for e in lib.examples("fire", include_removed=True)] == [ex.id]
    lib.restore_example("fire", ex.id)
    assert [e.id for e in lib.examples("fire")] == [ex.id]
    gone = lib.remove_type("fire")
    assert lib.types() == [] and gone.exists()
    back = lib.restore_type(gone)
    assert back.name == "fire" and lib.counts("fire")["originals"] == 1


def test_dirty_copies_say_what_was_done_and_come_only_from_the_train_span(rf, lib):
    base, lab = _capture(rf)
    lib.new_type("fire", link="nfm_voice")
    ex = lib.add_original_from_capture(base, _box(lab), "fire")
    rec = dirty.recipe("harsh")
    res = dirty.make_dirty(lib, "fire", 6, rec, seed=3)
    assert res["made"] == 6
    copies = lib.examples("fire", "dirty")
    assert len(copies) == 6
    for c in copies:
        g = sigmf.read_meta(c.base)["global"]
        assert g["atk:tier"] == "invented" and g["atk:parent"] == ex.id
        crop = g["atk:crop"]
        assert crop["start"] + crop["count"] <= ex.holdout["start"]
        assert all("words" in e for e in g["atk:effects"])
        x = sigmf.load(c.base)
        assert x.size >= store.WINDOW and np.all(np.isfinite(x))
    # deterministic in the seed
    rng1, rng2 = np.random.default_rng(5), np.random.default_rng(5)
    xa = lib.load(ex, 0, 8192)
    facts = dirty._facts(lib.meta(ex))
    a, _ = dirty.make_one(xa, ex.rate, facts, rec, rng1)
    b, _ = dirty.make_one(xa, ex.rate, facts, rec, rng2)
    assert np.array_equal(a, b)


def test_noise_is_only_ever_added_to_the_snr_asked_for(rf, lib):
    from atk_diffusion.library.store import measure
    base, lab = _capture(rf, snr_db=30.0)
    lib.new_type("fire", link="nfm_voice")
    ex = lib.add_original_from_capture(base, _box(lab), "fire")
    meta = lib.meta(ex)
    facts = dirty._facts(meta)
    x = lib.load(ex, 0, 32768)
    only_noise = dirty.recipe("typical", {k: {"on": False} for k in
                                          dirty.EFFECT_ORDER if k != "noise"})
    only_noise["effects"]["noise"].update(p=1.0, snr_db=[5.0, 5.0])
    y, applied = dirty.make_one(x, ex.rate, facts, only_noise,
                                np.random.default_rng(0))
    # the added noise lifts the floor, so the copy is measured against its
    # own floor (its passband's median bin), not the original's
    m2 = measure(y, ex.rate)
    assert applied[0]["params"]["snr_db"] == 5.0
    est = m2["snr_estimate_db"]
    assert est is not None and 1.0 < est < 9.0
    # asking for MORE than the original has adds nothing and says so
    only_noise["effects"]["noise"].update(snr_db=[45.0, 45.0])
    y2, applied2 = dirty.make_one(x, ex.rate, facts, only_noise,
                                  np.random.default_rng(0))
    assert "no noise added" in applied2[0]["words"]


def test_recipes_refuse_nonsense_in_words():
    with pytest.raises(store.LibraryError, match="preset"):
        dirty.recipe("filthy")
    with pytest.raises(store.LibraryError, match="not an effect"):
        dirty.recipe("typical", {"rain": {"p": 1}})
    with pytest.raises(store.LibraryError, match="within"):
        dirty.recipe("typical", {"noise": {"snr_db": [-90, 10]}})
    lines = dirty.describe(dirty.recipe("light"))
    assert any(l.startswith("Noise") for l in lines)
    assert not any(l.startswith("DC spike") for l in lines)


def test_preview_writes_nothing(rf, lib):
    base, lab = _capture(rf)
    lib.new_type("fire", link="nfm_voice")
    lib.add_original_from_capture(base, _box(lab), "fire")
    before = sorted(p.name for p in rf.training(PID, "fire").rglob("*"))
    out = dirty.preview(lib, "fire", dirty.recipe("typical"), n=4, seed=1)
    assert len(out) == 4 and all(o["x"].size > 0 for o in out)
    assert sorted(p.name for p in rf.training(PID, "fire").rglob("*")) == before


# ---------------------------------------------------------------------------
# fresh copies
# ---------------------------------------------------------------------------
def test_fresh_copies_need_a_link_and_say_why_not(rf, lib):
    from atk_diffusion.library import fresh
    lib.new_type("mystery", family="fsk")
    ok, why = fresh.can_make(lib, "mystery")
    assert not ok and "linked" in why
    lib.new_type("pagers", link="pocsag")
    ok, why = fresh.can_make(lib, "pagers")
    assert ok, why


def test_fresh_copies_are_cut_at_the_types_rate_with_known_labels(rf, lib):
    from atk_diffusion.library import fresh
    base, lab = _capture(rf, cls="pocsag", offset=-120e3, burst=(0.2, 1.0))
    lib.new_type("pagers", link="pocsag")
    lib.add_original_from_capture(base, _box(lab, burst=(0.2, 1.0)), "pagers")
    res = fresh.make_fresh(lib, "pagers", 5, seed=2)
    assert res["made"] == 5
    ex = lib.examples("pagers", "fresh")
    assert len(ex) == 5
    can = {c.cls: c for c in profiles.canonical_rates(FS)}["voice"]
    for e in ex:
        g = sigmf.read_meta(e.base)["global"]
        assert g["atk:tier"] == "invented" and g["atk:generator_class"] == "pocsag"
        assert g["core:sample_rate"] == pytest.approx(can.rate)
        assert e.samples == fresh.DEFAULT_LENGTH
        assert g["atk:measured"]["symbol_rate_hz"] in (512.0, 1200.0, 2400.0)


def test_a_reference_type_is_remade_at_its_measured_rate(rf, lib):
    from atk_diffusion.library import fresh
    base, lab = _capture(rf, cls="ref_qpsk", offset=90e3, burst=(0.2, 1.2),
                         seconds=1.6)
    lib.new_type("mystery_qpsk", link="ref_qpsk")
    ex = lib.add_original_from_capture(base, _box(lab, burst=(0.2, 1.2), pad_hz=3e3),
                                       "mystery_qpsk")
    measured = sigmf.read_meta(ex.base)["global"]["atk:measured"]
    params, words = fresh._matched_params(lib, lib.get_type("mystery_qpsk"))
    if measured.get("symbol_rate_hz"):
        assert params["symbol_rate"] == pytest.approx(measured["symbol_rate_hz"])
        assert "measured" in words[0]
    else:
        assert "bandwidth_hz" in params


def test_a_write_stopped_part_way_is_never_listed(rf, lib):
    """Stop kills the job's process; a pair still under its temporary
    `.<id>.partial` name must not be read as an example."""
    lib.new_type("t1", link="nfm_voice")
    base, lab = _capture(rf)
    lib.add_original_from_capture(base, _box(lab), "t1")
    folder = lib.kind_dir("t1", "dirty")
    folder.mkdir(parents=True, exist_ok=True)
    sigmf.write_pair(folder / ".orig_0001_d00001.partial",
                     np.zeros(4096, np.complex64), 25600.0, F0, datatype="cf32")
    assert [e.kind for e in lib.examples("t1")] == ["originals"]
