# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The signal cut — right-click to everything (DETECTION_DESIGN §4.2;
ARCHITECTURE §5 "The signal cut"): make the folder, analyze, clean with
every method on the menu, route, report; the folder is self-contained, the
original is never changed, and every refusal is a sentence.

One analyzed cut is built per module (analysis costs seconds); every test
works on its own copy of the whole rf_data root, write log included."""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timezone

import numpy as np
import pytest

import helpers_cyclo as H
from atk_diffusion import paths, profiles, provenance, sigmf
from atk_diffusion.cut import cut as C
from atk_diffusion.cut import report as R
from atk_diffusion.cut import route as RT
from atk_diffusion.detect import classes
from atk_diffusion.dsp import resample

PID = "rtlsdr_240000_cu8"
KRAKEN = "krakensdr_240000_cu8"
FS = 240_000.0
CENTER = 162.4e6
F_SIG = 30e3
BOX = {"t0_s": 0.3, "t1_s": 1.3, "f_lo_hz": CENTER + F_SIG - 5e3,
       "f_hi_hz": CENTER + F_SIG + 5e3}
EPOCH = 1.7e9


def _bpsk(rng, n, snr=12.0, phase=0.0):
    s = H.linmod("bpsk", FS, 4800.0, n, rng, beta=0.35)
    return H.mix(s, F_SIG, FS) * H.inband_amplitude(snr, 6480.0, FS) \
        * np.exp(1j * phase)


def _scene(seed=1, seconds=1.6, snr=12.0):
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    return (_bpsk(rng, n, snr) + H.noise(n, rng)).astype(np.complex64)


def _sha(p):
    return paths.sha256_file(p)


@pytest.fixture(scope="module")
def analyzed(tmp_path_factory):
    root = tmp_path_factory.mktemp("cutroot") / "rf_data"
    rf = paths.RfData(root, create=True)
    cf = C.make_cut(_scene(), FS, CENTER, BOX, PID, rf,
                    source={"capture": "synthetic-scene", "epoch": EPOCH,
                            "sample_offset": 1000},
                    who="bill", note="test cut")
    cf.analyze()
    return root, cf.path.relative_to(root)


@pytest.fixture
def cut(analyzed, tmp_path):
    root, rel = analyzed
    new = tmp_path / "rf_data"
    shutil.copytree(root, new)
    return C.CutFolder.open(new / rel, paths.RfData(new))


def _original_hashes(cf):
    base = cf.path / "original"
    return _sha(sigmf.data_path(base)), _sha(sigmf.meta_path(base))


# ---------------------------------------------------------------------------
# step 1 — the cut and its provenance (ARCHITECTURE §5)
# ---------------------------------------------------------------------------
def test_the_original_carries_the_record_tier_and_its_provenance(cut):
    meta = sigmf.read_meta(cut.path / "original")
    g = meta["global"]
    assert sigmf.validate(meta) == []
    assert g["atk:tier"] == "record"
    assert g["atk:receiver_profile"] == PID
    assert g["atk:source_capture"] == "synthetic-scene"
    assert g["atk:cut_by"] == "bill" and g["atk:note"] == "test cut"
    # integer decimation to the profile's voice-class canonical rate
    can = profiles.canonical_for(FS, 10e3)
    assert g["atk:decimation"] == can.decimation == 5
    assert g["atk:canonical_class"] == "voice"
    assert g["core:sample_rate"] == g["atk:canonical_rate"] == 48_000.0
    assert FS / g["atk:decimation"] == g["core:sample_rate"]
    # the margin is 10 % of the box (0.1 s), so the cut starts at source
    # sample 0.2 s · FS, after the 1000 samples before x[0]
    assert g["atk:margin_s"] == pytest.approx(0.1)
    assert g["atk:source_sample_start"] == 1000 + 48_000
    assert g["atk:source_sample_count"] == 288_000
    assert g["atk:box"] == {k: pytest.approx(v) for k, v in BOX.items()}
    assert meta["captures"][0]["core:frequency"] == pytest.approx(CENTER + F_SIG)
    # the wall clock of the cut's FIRST sample: the source's epoch is its
    # sample 0, the cut starts 49,000 samples later
    t = datetime.fromisoformat(meta["captures"][0]["core:datetime"]
                               .replace("Z", "+00:00")).timestamp()
    assert t == pytest.approx(EPOCH + 49_000 / FS, abs=1e-3)
    stamp = datetime.fromtimestamp(EPOCH + 49_000 / FS, tz=timezone.utc
                                   ).strftime("%Y%m%dT%H%M%SZ")
    assert cut.path.name == f"{stamp}_{int(CENTER + F_SIG)}"
    ann = sigmf.annotations(meta)[0]
    assert ann.sample_start == 4_800 and ann.sample_count == 48_000
    assert ann.freq_lower_edge == BOX["f_lo_hz"]
    assert cut.analysis["cut"]["original_sha256"] == _sha(
        sigmf.data_path(cut.path / "original"))


def test_the_original_is_the_box_at_the_canonical_rate(cut):
    x = _scene()
    iq, fs, _meta = cut.load("original")
    want, fs_w, info = resample.cut_to_canonical(x[48_000:336_000], FS,
                                                 F_SIG, 10e3)
    assert fs == fs_w == 48_000.0 and iq.dtype == np.complex64
    assert np.allclose(iq, want, atol=1e-6)
    # the BPSK now sits at 0 Hz: its band holds far more than the rest
    P = np.abs(np.fft.fftshift(np.fft.fft(iq[:32_768]))) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(32_768, 1 / fs))
    assert P[np.abs(f) < 3_000].mean() > 50 * P[np.abs(f) > 9_000].mean()


def test_cut_from_capture_keeps_capture_samples_and_wall_clock(rf, tmp_path):
    x = _scene()
    base = rf.captures(PID) / "cap"
    dp, mp = sigmf.write_pair(base, x, FS, CENTER, datatype="cf32",
                              t0_utc=EPOCH, hw="RTL-SDR v3",
                              extra_global={"atk:receiver_profile": PID})
    rf.record(dp, "capture")
    rf.record(mp, "capture-meta")
    cf = C.cut_from_capture(base, BOX, rf, who="bill")
    g = sigmf.read_meta(cf.path / "original")["global"]
    assert g["atk:source_capture"] == str(base)
    assert g["atk:source_sample_start"] == 48_000         # capture samples
    assert g["atk:source_sample_count"] == 288_000
    assert "import" not in g.get("atk:note", "")
    t = datetime.fromisoformat(sigmf.read_meta(cf.path / "original")[
        "captures"][0]["core:datetime"].replace("Z", "+00:00")).timestamp()
    # the first version stamped the cut with the CAPTURE's start time
    assert t == pytest.approx(EPOCH + 0.2, abs=1e-3)
    iq, _fs, _m = cf.load()
    want, _f, _i = resample.cut_to_canonical(x[48_000:336_000], FS, F_SIG, 10e3)
    assert np.allclose(iq, want, atol=1e-6)
    # a capture the write log never saw is an import, and the cut says so
    other = tmp_path / "elsewhere" / "cap2"
    sigmf.write_pair(other, x, FS, CENTER, datatype="cf32",
                     extra_global={"atk:receiver_profile": PID})
    cf2 = C.cut_from_capture(other, BOX, rf)
    assert "an import" in sigmf.read_meta(cf2.path / "original")["global"][
        "atk:note"]
    # a recorded capture that changed afterwards is named, not used
    with open(dp, "ab") as f:
        f.write(b"\0" * 8)
    with pytest.raises(C.CutError, match="changed after it was recorded"):
        C.cut_from_capture(base, BOX, rf)


# ---------------------------------------------------------------------------
# step 2 — Analyze
# ---------------------------------------------------------------------------
def test_analyze_writes_measurements_cyclic_peaks_class_and_files(cut):
    a = cut.analysis
    m = a["measurements"]
    assert m["symbol_rate"]["value_hz"] == pytest.approx(4800.0, abs=1.0)
    assert m["symbol_rate"]["tier"] == "measured" and m["symbol_rate"]["method"]
    assert m["carrier_offset"]["conjugate_feature"] == "present"
    assert m["carrier_offset"]["value_hz"] == pytest.approx(0.0, abs=5.0)
    assert m["occupied_bandwidth"]["value_hz"] == pytest.approx(5_600, rel=0.2)
    assert m["snr"]["measurable"]
    nc = [p for p in a["cyclic"]["peaks"] if not p["conj"]]
    cj = [p for p in a["cyclic"]["peaks"] if p["conj"]]
    assert any(abs(p["alpha_hz"] - 4800.0) < 2.0 for p in nc)
    assert any(abs(p["alpha_hz"]) < 2.0 for p in cj)          # 2·f_c = 0
    assert all(p["statistic"] > p["threshold"] for p in a["cyclic"]["peaks"])
    # the class-table probe: 4800 sym/s is P25, DMR and NXDN96 alike; this
    # one has a conjugate feature, which FSK does not make — said
    assert a["cyclic"]["candidates"] == ["dmr", "nxdn96", "p25"]
    assert "suspicion" in a["cyclic"]["consistency"][0]
    assert a["class"]["cls"] == classes.UNKNOWN
    assert "no classifier" in a["class"]["why"]
    # the SCF image and the cyclic profile, as files
    img = np.load(cut.path / "scf.npy")
    assert img.shape == (64, 128) and img.dtype == np.float32
    assert 0.0 <= img.min() and img.max() == pytest.approx(1.0)
    cp = np.load(cut.path / "cyclic_profile.npy")
    assert cp.shape[0] == 3 and cp.shape[1] > 1000
    assert cp[1][np.argmin(np.abs(cp[0]))] == pytest.approx(1.0)
    for png in ("scf.png", "cyclic_profile.png"):
        assert (cut.path / png).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert a["scf"]["geometry"]["channel_fft"] == profiles.FamGeometry().channel_fft
    for f in ("scf.npy", "scf.png", "cyclic_profile.npy", "cyclic_profile.png"):
        assert cut.rf.verify(cut.path / f)[0]
    assert [h["what"] for h in a["history"]][:2] == ["cut", "analyze"]


def test_analyze_with_a_classifier_and_a_fingerprint(cut):
    calls = []

    def clf(iq, fs):
        calls.append((iq.size, fs))
        return {"cls": "ref_bpsk", "confidence": 0.81,
                "embedding": np.array([0.6, 0.8]), "model": "tiny"}

    a = cut.analyze(classifier=clf, fingerprint=lambda iq, fs: {"id": "tx-7"})
    assert calls == [(57_600, 48_000.0)]
    assert a["class"] == {"cls": "ref_bpsk", "confidence": 0.81,
                          "embedding": [0.6, 0.8], "model": "tiny",
                          "tier": "proposed", "source": "classifier"}
    assert a["fingerprint"] == {"id": "tx-7"}
    with pytest.raises(C.CutError, match="a classifier returns"):
        cut.analyze(classifier=lambda iq, fs: "bpsk")


def test_analysis_folds_a_harmonic_rate_into_its_fundamental(rf):
    """A rectangular-pulse 2400 Bd signal also makes a line at 4800: one
    signal, not a P25 / DMR / NXDN96 as well. Its own rate's classes
    (NXDN48 and POCSAG at 2400) remain the candidates."""
    rng = np.random.default_rng(9)
    n = int(1.6 * FS)
    s = H.mix(H.linmod("bpsk", FS, 2400.0, n, rng, rect=True), F_SIG, FS)
    x = (s * H.inband_amplitude(15.0, 4800.0, FS) + H.noise(n, rng)
         ).astype(np.complex64)
    cf = C.make_cut(x, FS, CENTER, BOX, PID, rf)
    a = cf.analyze()
    probe = a["cyclic"]["probes"]["symbol_rate_line"]
    assert probe["harmonics_folded"] == {"4800": "2400"}     # said, not hidden
    assert a["cyclic"]["candidates"] == ["nxdn48", "pocsag"]
    menu = cf.routes_available()
    assert menu[:3] == ["dsd", "pager", "multimon"]          # NXDN48, POCSAG


# ---------------------------------------------------------------------------
# step 3 — Clean, every method on the menu
# ---------------------------------------------------------------------------
def _denoiser(iq, fs, **kw):
    return 0.9 * iq, {"model_sha256": "ab" * 32, "words": "a stand-in",
                      "timestep": 120}


CLEANS = {
    "matched": {},
    "fresh": {},
    "fresh_separate": {"alpha_sets": [{"alphas": [4800.0], "conj": [0.0]},
                                      [6000.0]]},
    "wiener": {},
    "rfi_mask_interp": {"pfa": 1e-4},
    "diffusion": {"denoiser": _denoiser, "steps": 1},
}


@pytest.mark.parametrize("method", list(CLEANS))
def test_every_clean_writes_its_tier_params_and_measured_snr(cut, method):
    before = _original_hashes(cut)
    rec = cut.clean(method, **CLEANS[method])
    want_tier = provenance.tier_for(C.METHODS[method])
    assert rec["tier"] == want_tier
    assert rec["files"], rec.get("no_file")
    orig_cap = sigmf.read_meta(cut.path / "original")["captures"][0]
    for name in rec["files"]:
        meta = sigmf.read_meta(cut.path / name)
        g = meta["global"]
        assert sigmf.validate(meta) == []
        assert g["atk:tier"] == want_tier and g["atk:tier"] != "record"
        assert g["atk:method"] == C.METHODS[method]
        assert g["atk:cleaned_from"] == "original"
        assert isinstance(g["atk:method_params"], dict) and g["atk:method_params"]
        assert isinstance(g["atk:snr_before_db"], float)
        assert isinstance(g["atk:snr_after_db"], float)
        assert g["atk:snr_method"]
        # the signal's own time, not the time it was cleaned
        assert meta["captures"][0]["core:datetime"] == orig_cap["core:datetime"]
        assert cut.rf.verify(sigmf.data_path(cut.path / name))[0]
        iq, fs, _m = cut.load(name)
        assert fs == 48_000.0 and iq.shape == (57_600,)
    assert _original_hashes(cut) == before          # the record never changes
    assert cut.analysis["cleans"][-1] is rec and rec["id"] == "clean_1"
    assert cut.analysis["history"][-1]["what"] == f"clean {method}"


def test_matched_clean_records_parameters_and_filters_a_linear_signal(cut):
    rec = cut.clean("matched")
    assert rec["parameters_tier"] == "measured"
    p = rec["parameters"]
    assert p["symbol_rate_hz"] == pytest.approx(4800.0, abs=0.5)
    assert p["carrier_offset_hz"] == pytest.approx(0.0, abs=5.0)
    assert p["samples_per_symbol"] == pytest.approx(10.0, rel=1e-3)
    g = sigmf.read_meta(cut.path / "cleaned")["global"]
    assert g["atk:method_params"]["symbol_rate_hz"] == pytest.approx(4800.0,
                                                                     abs=0.5)
    assert 0.2 < g["atk:method_params"]["rolloff"] < 0.5
    assert rec["gain_db"] == pytest.approx(1.3, abs=0.6)


def test_matched_clean_of_an_fsk_gives_parameters_and_no_file(rf):
    rng = np.random.default_rng(4)
    n = int(1.6 * FS)
    s = H.mix(H.fsk4(FS, 4800.0, n, rng), F_SIG, FS)
    x = (s * H.inband_amplitude(15.0, 8000.0, FS) + H.noise(n, rng)
         ).astype(np.complex64)
    cf = C.make_cut(x, FS, CENTER, BOX, PID, rf)
    rec = cf.clean("matched")
    assert rec["tier"] == "measured" and rec["files"] == []
    assert "discriminator" in rec["no_file"]
    assert rec["parameters"]["timing_offset_s"] is None
    assert not (cf.path / "cleaned.sigmf-meta").exists()


def test_fresh_uses_the_analysis_alphas_and_writes_them(cut):
    rec = cut.clean("fresh")
    mp = sigmf.read_meta(cut.path / "cleaned")["global"]["atk:method_params"]
    # the first version wrote only the caller's (empty) kwargs
    assert mp["alphas_hz"] == [pytest.approx(4800.0, abs=1.0)]
    assert mp["conj_alphas_hz"] and mp["nperseg"] >= 32
    assert rec["gain_db"] > 0.5
    rec2 = cut.clean("fresh", alphas=[], conj_alphas=[0.0])
    assert rec2["files"] == ["cleaned_2"] and rec2["id"] == "clean_2"
    assert rec2["method_params"]["alphas_hz"] == []


def test_wiener_and_separate_files_and_their_numbers(cut):
    rec = cut.clean("wiener")
    # the 10 kHz box is about twice the BPSK's 6.5 kHz: ~2.6 dB of noise
    # outside the signal to remove, and no more
    assert 1.5 < rec["gain_db"] < 4.0
    sep = cut.clean("fresh_separate", **CLEANS["fresh_separate"])
    assert sep["files"] == ["separated_1", "separated_2"]
    g = sigmf.read_meta(cut.path / "separated_2")["global"]
    assert g["atk:separated_index"] == 2 and "SINR" in g["atk:snr_kind"]
    assert g["atk:alphas_hz"] == [6000.0]


def test_clean_refusals_are_sentences(cut):
    with pytest.raises(C.CutError, match="unknown clean 'magic'"):
        cut.clean("magic")
    with pytest.raises(C.CutError, match="this cut has one channel"):
        cut.clean("score", alpha=4800.0)
    with pytest.raises(C.CutError, match="needs denoiser"):
        cut.clean("diffusion")
    with pytest.raises(C.CutError, match="model_sha256"):
        cut.clean("diffusion", denoiser=lambda iq, fs: (iq, {}))
    with pytest.raises(C.CutError, match="keeps the cut's length"):
        cut.clean("diffusion", denoiser=lambda iq, fs: (
            iq[:10], {"model_sha256": "cd" * 32}))
    with pytest.raises(C.CutError, match="alpha_sets"):
        cut.clean("fresh_separate")
    with pytest.raises(C.CutError, match="fresh cannot run on this cut"):
        cut.clean("fresh", alphas=[1e6])
    assert cut.analysis["cleans"] == []


# ---------------------------------------------------------------------------
# step 4 — Route
# ---------------------------------------------------------------------------
def test_routes_offer_only_the_tools_of_the_cuts_class(cut):
    menu = cut.routes_available()
    assert menu == ["dsd", "bench", "df", "ask", "teach", "fingerprint", "save"]
    assert "pager" not in menu and "lte_search" not in menu
    # an UNKNOWN with no candidates gets only what every cut gets
    assert RT.available({"class": {"cls": classes.UNKNOWN}}) == list(RT.ALWAYS)
    # a detection's class travels with the cut
    assert RT.available({"source": {"detection": {"cls": "pocsag"}}})[:2] == [
        "pager", "multimon"]
    with pytest.raises(C.CutError, match="does not take this cut"):
        cut.route("pager")


def test_route_appends_to_the_story_and_labels_a_cleaned_input(cut):
    rec = cut.route("save")
    assert rec["result"]["saved"] and not rec["ran"]
    with pytest.raises(C.CutError, match="no cleaned file yet"):
        cut.route("dsd", input="cleaned")
    cut.clean("matched")
    seen = {}

    def dsd(iq, fs, meta):
        seen.update(meta["global"])
        return {"decoded": "NAC 293", "ok": True}

    r2 = cut.route("dsd", input="cleaned", runner=dsd)
    assert r2["ran"] and r2["input"] == "cleaned" and r2["input_tier"] == "cleaned"
    assert "CLEANED signal, not from the record" in r2["decoded_from_note"]
    assert r2["result"]["decoded"] == "NAC 293"
    assert "decoded_from_note" in r2["result"]
    # the matched-filter parameters travel to the tool
    assert r2["matched_parameters_passed"]
    assert seen["atk:matched_parameters"]["symbol_rate_hz"] == pytest.approx(
        4800.0, abs=0.5)
    r3 = cut.route("dsd", runner=lambda iq, fs, meta: "no sync")
    assert r3["input_tier"] == "record" and "decoded_from_note" not in r3
    assert r3["result"] == {"result": "no sync"}
    assert [r["tool"] for r in cut.analysis["routes"]] == ["save", "dsd", "dsd"]
    saved = json.loads((cut.path / "analysis.json").read_text("utf-8"))
    assert [r["tool"] for r in saved["routes"]] == ["save", "dsd", "dsd"]
    with pytest.raises(C.CutError, match="has no 'separated_9'"):
        cut.route("save", input="separated_9")


# ---------------------------------------------------------------------------
# step 5 — the report, from the facts
# ---------------------------------------------------------------------------
def test_report_is_written_from_the_facts(cut):
    cut.clean("matched")
    cut.clean("rfi_mask_interp")
    cut.route("save")
    p = cut.report()
    text = p.read_text("utf-8")
    a = cut.analysis
    assert text == R.render(a)
    assert cut.path.name in text and "RECORD" in text
    sr = a["measurements"]["symbol_rate"]["value_hz"]
    assert f"{sr:,.2f} Bd" in text
    snr = a["measurements"]["snr"]["snr_db"]
    assert f"{snr:+.1f} dB" in text
    assert "### clean_1 — matched (CLEANED)" in text
    assert "### clean_2 — rfi_mask_interp (INFERRED)" in text
    assert "Parameters for the demodulator (MEASURED)" in text
    assert "**save** on `original`" in text
    assert "`report.md` — this cut in words" in text
    assert cut.rf.verify(p)[0]


# ---------------------------------------------------------------------------
# the folder is self-contained; the original is never changed
# ---------------------------------------------------------------------------
def test_the_folder_reads_with_sigmf_alone(cut):
    cut.clean("wiener")
    base = cut.path / "original"
    meta = sigmf.read_meta(base)
    raw = np.fromfile(sigmf.data_path(base), dtype=np.complex64)
    assert np.array_equal(raw, sigmf.load(base))
    assert raw.size == meta["global"]["atk:source_sample_count"] // 5
    a = json.loads((cut.path / "analysis.json").read_text("utf-8"))
    assert a["cut"]["original_sha256"] == _sha(sigmf.data_path(base))
    assert sigmf.validate(sigmf.read_meta(cut.path / "cleaned")) == []


def test_a_cut_imported_into_a_new_install_proves_itself(cut, tmp_path):
    other = paths.RfData(tmp_path / "other_root", create=True)
    dst = other.cuts(PID) / cut.path.name
    shutil.copytree(cut.path, dst)
    moved = C.CutFolder.open(dst)                 # rf from the folder's path
    assert moved.rf.root == other.root
    rec = moved.clean("wiener")
    assert rec["files"] == ["cleaned"]
    assert "imported" in moved.analysis["history"][-2]["what"]
    assert other.verify(sigmf.data_path(dst / "original"))[0]
    # a copy whose original was altered is refused against its own hash
    bad = other.cuts(PID) / "tampered"
    shutil.copytree(cut.path, bad)
    raw = bytearray(sigmf.data_path(bad / "original").read_bytes())
    raw[100] ^= 0xFF
    sigmf.data_path(bad / "original").write_bytes(bytes(raw))
    with pytest.raises(C.CutError, match="do not match the hash"):
        C.CutFolder.open(bad).clean("wiener")


def test_an_original_changed_in_place_is_refused(cut):
    dp = sigmf.data_path(cut.path / "original")
    raw = bytearray(dp.read_bytes())
    raw[-1] ^= 0x01
    dp.write_bytes(bytes(raw))
    for step in (lambda: cut.analyze(), lambda: cut.clean("wiener"),
                 lambda: cut.route("save")):
        with pytest.raises(C.CutError, match="changed after it was cut"):
            step()


# ---------------------------------------------------------------------------
# refusals, in words
# ---------------------------------------------------------------------------
def test_refusals_are_plain_sentences(rf, tmp_path):
    x = _scene(seconds=1.0)
    with pytest.raises(C.CutError, match="outside the IQ given"):
        C.make_cut(x, FS, CENTER, dict(BOX, t0_s=5.0, t1_s=6.0), PID, rf)
    with pytest.raises(C.CutError, match="reaches outside what the receiver"):
        C.make_cut(x, FS, CENTER, dict(BOX, f_hi_hz=CENTER + 200e3), PID, rf)
    with pytest.raises(C.CutError, match="a box is"):
        C.make_cut(x, FS, CENTER, {"t0_s": 0.1}, PID, rf)
    with pytest.raises(C.CutError, match="ends before it starts"):
        C.make_cut(x, FS, CENTER, dict(BOX, t0_s=0.5, t1_s=0.4), PID, rf)
    with pytest.raises(profiles.ProfileMismatch, match="sample-rate law"):
        C.make_cut(x, 2.4e6, CENTER, BOX, PID, rf)
    with pytest.raises(C.CutError, match=r"only \d+ samples"):
        C.make_cut(x[:500], FS, CENTER, dict(BOX, t0_s=0.0, t1_s=0.002), PID,
                   rf, margin_s=0.0)
    bad = x.copy()
    bad[1000] = np.nan
    with pytest.raises(C.CutError, match="not finite"):
        C.make_cut(bad, FS, CENTER, dict(BOX, t0_s=0.0, t1_s=0.5), PID, rf)
    with pytest.raises(C.CutError, match="not a cut folder"):
        C.CutFolder.open(tmp_path)
    # a capture: the box outside it, a retune inside the box, no capture
    base = rf.captures(PID) / "retuned"
    sigmf.write_pair(base, x, FS, CENTER, datatype="cf32",
                     extra_global={"atk:receiver_profile": PID})
    meta = sigmf.read_meta(base)
    meta["captures"].append({"core:sample_start": int(0.8 * FS),
                             "core:frequency": CENTER + 1e6})
    sigmf.write_meta(base, meta)
    with pytest.raises(C.CutError, match="outside the capture"):
        C.cut_from_capture(base, dict(BOX, t0_s=3.0, t1_s=4.0), rf)
    with pytest.raises(C.CutError, match="retuned inside that box"):
        C.cut_from_capture(base, dict(BOX, t0_s=0.5, t1_s=0.95), rf)
    with pytest.raises(C.CutError, match="no capture at"):
        C.cut_from_capture(tmp_path / "nothing", BOX, rf)
    assert not list(rf.cuts(PID).glob("*")) or all(
        (p / "analysis.json").exists() for p in rf.cuts(PID).iterdir())


# ---------------------------------------------------------------------------
# the Kraken: five coherent channels stay together
# ---------------------------------------------------------------------------
def test_a_kraken_cut_keeps_all_five_channels_and_score_uses_them(rf):
    rng = np.random.default_rng(12)
    n = int(1.6 * FS)
    s = _bpsk(rng, n, snr=0.0)
    a = H.uca(5, 0.7)
    X = (np.outer(a, s) + np.stack([H.noise(n, rng) for _ in range(5)])
         ).astype(np.complex64)
    cf = C.make_cut(X, FS, CENTER, BOX, KRAKEN, rf, who="bill")
    meta = sigmf.read_meta(cf.path / "original")
    assert meta["global"]["core:num_channels"] == 5
    iq, fs, _m = cf.load()
    assert iq.shape == (5, 57_600) and cf.analysis["cut"]["channels"] == 5
    for c in (0, 3):
        want, _f, _i = resample.cut_to_canonical(X[c, 48_000:336_000], FS,
                                                 F_SIG, 10e3)
        assert np.allclose(iq[c], want, atol=1e-6)
    rec = cf.clean("score", alpha=4800.0)
    assert rec["tier"] == "cleaned" and rec["files"] == ["cleaned"]
    y, _fs, _m = cf.load("cleaned")
    assert y.ndim == 1 and y.size == 57_600
    # five channels of 0 dB each: SCORE's measured gain approaches 7 dB
    assert 4.0 < rec["gain_db"] < 8.0
    w = cf.clean("wiener")
    assert "channel 0 of 5" in w["note"]
    r = cf.route("df")
    assert "on the cut itself" in r["words"]
