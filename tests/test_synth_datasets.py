# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The dataset builder (synth.datasets): ARCHITECTURE §5's layout and
manifest, the sample-rate law, the refusals in words, overwrite, hashes and
the write log — with the native generator and with the real TorchSig 2.2.0.

Every build here is tiny (a handful of examples, short windows, short
scenes) so the file runs in seconds on a shared CPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from atk_diffusion import profiles, sigmf
from atk_diffusion.detect import classes
from atk_diffusion.paths import RfData, sha256_file
from atk_diffusion.synth import datasets as D
from atk_diffusion.synth import labels as LB
from atk_diffusion.synth import native

PID = "rtlsdr_2400000_cu8"

#: ARCHITECTURE §5, the manifest of a built dataset.
SECTION5_KEYS = ("name", "profile", "sample_rate", "kind", "generator", "stft",
                 "fam", "classes", "families", "splits", "label_sources",
                 "environment", "resampled", "created", "params", "files")


def _nb(rf, name="nb", cls=("ref_bpsk", "noise"), n=3, **kw):
    kw.setdefault("window", 256)
    kw.setdefault("compute_scf", False)
    return D.build_narrowband(rf, PID, name, list(cls), n, (5.0, 15.0),
                              "voice", **kw)


def _log(rf) -> dict:
    return rf.log.entries()


# ---------------------------------------------------------------------------
# narrowband, native
# ---------------------------------------------------------------------------
def test_narrowband_manifest_has_every_section5_field(rf):
    seen = []
    m = _nb(rf, compute_scf=True, progress=seen.append)
    for k in SECTION5_KEYS:
        assert k in m, k
    assert m["profile"] == PID and m["sample_rate"] == 2_400_000.0
    assert m["kind"] == "narrowband" and m["generator"] == "native"
    can = {c.cls: c for c in profiles.canonical_rates(2_400_000)}["voice"]
    assert m["canonical"] == {"class": "voice", "rate": can.rate,
                              "decimation": can.decimation, "limited": False}
    assert m["classes"] == ["ref_bpsk", "noise"]
    assert m["families"] == list(classes.FAMILIES)
    assert m["label_sources"] == ["synthetic"] and m["resampled"] is False
    assert m["environment"] == ""
    assert sum(m["splits"].values()) == 6
    assert m["tier"] == "invented" and m["method"] == "synthetic_native"
    assert m["quantised_to"] == "cu8"
    assert m["receiver_impairments_applied"] is False
    assert "not measured" in m["receiver"]
    assert m["stft"]["fft_size"] == 1024 and m["fam"]["channel_fft"] == 64
    assert m["scf"].startswith("computed")
    assert m["params"]["n_per_class"] == 3 and m["params"]["window"] == 256
    assert "÷50" in m["notes"][0]
    assert seen and seen[-1].startswith("nb: 6/6 examples (100 %)")
    # the manifest on disk is the one returned (less the path)
    disk = json.loads((Path(m["path"]) / "manifest.json").read_text("utf-8"))
    assert disk["files"] == m["files"] and "path" not in disk


def test_narrowband_shards_have_the_section5_arrays(rf):
    m = _nb(rf, compute_scf=True)
    shards = list(D.iter_shards(m["path"], "train"))
    assert shards
    z = shards[0]
    n = z["iq"].shape[0]
    assert z["iq"].dtype == np.complex64 and z["iq"].shape == (n, 256)
    assert z["label"].dtype == np.int32 and z["family"].dtype == np.int32
    for k in ("snr_db", "symbol_rate_hz", "carrier_offset_hz", "bandwidth_hz"):
        assert z[k].dtype == np.float32 and z[k].shape == (n,)
    assert z["scf"].dtype == np.float16 and z["scf"].shape == (n,) + D.SCF_SHAPE
    noise = z["label"] == 1
    assert noise.any() and (~noise).any()
    # the noise class has no SNR and no carrier
    assert np.isnan(z["snr_db"][noise]).all()
    assert (z["carrier_offset_hz"][noise] == 0).all()
    assert np.isfinite(z["snr_db"][~noise]).all()
    assert (z["family"][~noise] == classes.FAMILIES.index("psk_qam")).all()
    # the carrier is near the cut centre: the box-centre miss is small
    assert np.all(np.abs(z["carrier_offset_hz"][~noise]) < 0.25 * 48_000)


def test_files_are_hashed_into_the_manifest_and_the_write_log(rf):
    m = _nb(rf)
    d = Path(m["path"])
    log = _log(rf)
    assert m["files"]
    for rel, h in m["files"].items():
        p = d / rel
        assert sha256_file(p) == h
        key = p.resolve().relative_to(rf.root.resolve()).as_posix()
        assert log[key]["sha256"] == h and log[key]["kind"] == "dataset-shard"
        assert rf.verify(p) == (True, "")
    mkey = (d / "manifest.json").resolve().relative_to(rf.root.resolve()).as_posix()
    assert log[mkey]["kind"] == "dataset-manifest"
    assert D.verify(d) == (True, [])


def test_same_seed_same_data_other_seed_other_data(rf):
    a = _nb(rf, name="a", seed=4)
    b = _nb(rf, name="b", seed=4)
    c = _nb(rf, name="c", seed=5)
    za = next(D.iter_shards(a["path"], "train"))
    zb = next(D.iter_shards(b["path"], "train"))
    zc = next(D.iter_shards(c["path"], "train"))
    assert np.array_equal(za["iq"], zb["iq"])
    assert not np.array_equal(za["iq"], zc["iq"])


def test_small_shards_and_all_three_splits(rf):
    m = D.build_narrowband(rf, PID, "split", ["ref_qpsk"], 10, (0.0, 10.0),
                           "voice", window=128, compute_scf=False,
                           shard_size=2, splits=(0.6, 0.2, 0.2))
    assert m["splits"] == {"train": 6, "val": 2, "test": 2}
    trains = [r for r in m["files"] if r.startswith("train/")]
    assert len(trains) == 3                       # 6 examples, 2 per shard
    assert sum(z["iq"].shape[0] for z in D.iter_shards(m["path"], "val")) == 2
    assert sum(z["iq"].shape[0] for z in D.iter_shards(m["path"], "test")) == 2


def test_the_wideband_canonical_class_cuts_at_its_own_rate(rf):
    m = D.build_narrowband(rf, PID, "wide", ["fm_broadcast"], 2, (10.0, 20.0),
                           "wideband", window=128, compute_scf=False)
    assert m["canonical"]["class"] == "wideband"
    assert m["canonical"]["rate"] == 480_000.0 and m["canonical"]["decimation"] == 5


def test_measured_impairments_are_applied_and_said(rf, rng):
    from atk_diffusion.dsp import impair
    x = native.noise(1 << 16, 10 ** (-40 / 10), rng)
    x = impair.quantise(x, "cu8", 8)
    meas = impair.measure_impairments(x, 2_400_000.0, "cu8")
    impair.store(rf, PID, meas, device_serial="00000001")
    m = _nb(rf, name="measured", cls=("ref_bpsk",), n=2)
    assert m["receiver_impairments_applied"] is True
    assert "not measured" not in m["receiver"]
    w = D.build_wideband(rf, PID, "measured_wb", 1, scene_seconds=0.05)
    assert w["receiver_impairments_applied"] is True
    assert w["tiles"].startswith("computed")


def test_throughput_writes_nothing(rf):
    before = sorted(p for p in rf.root.rglob("*") if p.is_file())
    r = D.measure_throughput(PID, n=1, window=128, classes=["ref_bpsk"])
    assert r["examples"] == 1 and r["examples_per_s"] > 0 and r["scf"] is False
    r2 = D.measure_throughput(profiles.new_profile(PID), n=1, window=128)
    assert r2["classes"] and r2["canonical"] == "voice"
    r3 = D.measure_throughput(PID, n=1, window=128, rf=rf, classes=["ref_bpsk"],
                              compute_scf=True)
    assert r3["scf"] is True
    assert sorted(p for p in rf.root.rglob("*") if p.is_file()) == before


# ---------------------------------------------------------------------------
# refusals, in words
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kw, words", [
    ({"n": 0}, "n_per_class must be at least 1"),
    ({"window": 32}, "shorter than 64 samples"),
    ({"generator": "magic"}, "'native' or 'torchsig'"),
    ({"cls": ("ref_bpsk", "ref_bpsk")}, "listed twice"),
    ({"cls": ("not_a_class",)}, "not in the class table"),
    ({"cls": ("lte_dl",)}, "does not fit"),
    ({"cls": ()}, "at least one class"),
    ({"splits": (0.5, 0.5, 0.5)}, "add up to 1"),
])
def test_narrowband_refusals(rf, kw, words):
    with pytest.raises(D.DatasetRefusal, match=words):
        _nb(rf, **kw)
    assert not rf.datasets(PID, "nb").exists()


def test_more_refusals(rf):
    with pytest.raises(D.DatasetRefusal, match=r"\(low, high\)"):
        D.build_narrowband(rf, PID, "x", ["ref_bpsk"], 1, (10, 5), "voice")
    with pytest.raises(D.DatasetRefusal, match="canonical_class is one of"):
        D.build_narrowband(rf, PID, "x", ["ref_bpsk"], 1, (5, 10), "huge")
    with pytest.raises(D.DatasetRefusal, match="receiver profile"):
        D.build_narrowband(rf, 2400000, "x", ["ref_bpsk"], 1, (5, 10), "voice")
    # a 12 kHz receiver has only a (limited) voice class
    with pytest.raises(D.DatasetRefusal, match="has no wideband canonical rate"):
        D.build_narrowband(rf, "kiwisdr_12000_ci16", "x", ["ref_bpsk"], 1,
                           (5, 10), "wideband")


@pytest.mark.parametrize("name", ["a/b", "..", "my set", "a:b", "trailing.",
                                  "", "-lead", "ok.name"])
def test_a_dataset_name_is_plain_or_refused(rf, name):
    """Windows silently drops a trailing dot or space and refuses ':' — a
    name that is not the folder's name would be the first lie in a run."""
    with pytest.raises(D.DatasetRefusal, match="plain dataset name"):
        _nb(rf, name=name)


def test_torchsig_refused_in_words_when_it_is_not_usable(rf, monkeypatch):
    from atk_diffusion.synth import torchsig_backend as ts
    monkeypatch.setattr(ts, "available",
                        lambda refresh=False: (False, "TorchSig is not installed"))
    with pytest.raises(D.DatasetRefusal, match="TorchSig is not installed"):
        _nb(rf, generator="torchsig")


# ---------------------------------------------------------------------------
# overwrite: a night's work is not deleted by accident
# ---------------------------------------------------------------------------
def test_rebuild_is_refused_without_overwrite_and_replaces_with_it(rf):
    m1 = _nb(rf, seed=1)
    old = dict(m1["files"])
    with pytest.raises(D.DatasetRefusal, match="already exists"):
        _nb(rf, seed=2)
    m2 = _nb(rf, seed=2, overwrite=True)
    assert set(m2["files"]) == set(old)
    assert m2["files"] != old                       # new data, new hashes
    assert D.verify(m2["path"]) == (True, [])


def test_overwrite_refuses_a_folder_holding_foreign_files(rf):
    m = _nb(rf)
    stray = Path(m["path"]) / "train" / "mine.txt"
    stray.write_text("Bill's notes", encoding="utf-8")
    with pytest.raises(D.DatasetRefusal, match="not in its manifest"):
        _nb(rf, overwrite=True)
    assert stray.exists()                           # never deleted


def test_a_folder_without_a_manifest_is_not_ours(rf):
    d = rf.datasets(PID, "nb")
    d.mkdir(parents=True)
    (d / "something.bin").write_bytes(b"\x00")
    with pytest.raises(D.DatasetRefusal, match="no manifest"):
        _nb(rf)


def test_a_folder_of_another_profile_is_a_profile_mismatch(rf):
    m = _nb(rf)
    mp = Path(m["path"]) / "manifest.json"
    man = json.loads(mp.read_text("utf-8"))
    man["profile"] = "hackrf_8000000_ci8"
    mp.write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(profiles.ProfileMismatch, match="Profiles never mix"):
        _nb(rf, overwrite=True)


def test_a_failed_rebuild_leaves_the_old_dataset_intact(rf, monkeypatch):
    """overwrite=True must not delete the old dataset before the new one
    can be made: a generator that fails half-way leaves the old one whole."""
    m = _nb(rf, seed=1)
    calls = {"n": 0}
    real = native.generate

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("the generator fell over")
        return real(*a, **k)

    monkeypatch.setattr(native, "generate", flaky)
    with pytest.raises(RuntimeError, match="fell over"):
        _nb(rf, seed=2, overwrite=True)
    assert D.verify(m["path"]) == (True, [])
    assert D.load_manifest(m["path"])["params"]["seed"] == 1
    assert not (Path(m["path"]).parent / "nb.partial").exists()


def test_an_interrupted_build_is_cleared_and_said(rf):
    """A crash or Ctrl-C mid-build leaves `<name>.partial`; the next build of
    that name clears it (only files this builder writes) and says so."""
    work = rf.datasets(PID, "nb.partial")
    (work / "train").mkdir(parents=True)
    (work / "train" / "shard_000.npz").write_bytes(b"half")
    (work / "train" / "shard_001.npz.tmp").write_bytes(b"torn")
    m = _nb(rf)
    assert any("an interrupted build of nb (2 file(s) in nb.partial) was "
               "cleared" in n for n in m["notes"])
    assert not work.exists() and D.verify(m["path"]) == (True, [])


def test_a_partial_folder_with_foreign_files_is_left_alone(rf):
    work = rf.datasets(PID, "nb.partial")
    (work / "train").mkdir(parents=True)
    (work / "train" / "keep_me.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(D.DatasetRefusal, match="this builder did not write"):
        _nb(rf)
    assert (work / "train" / "keep_me.txt").exists()


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def test_load_manifest_refusals(rf, tmp_path):
    with pytest.raises(D.DatasetRefusal, match="not a dataset"):
        D.load_manifest(tmp_path)
    m = _nb(rf)
    assert D.load_manifest(Path(m["path"]) / "manifest.json",
                           expect_profile=PID)["name"] == "nb"
    with pytest.raises(profiles.ProfileMismatch, match="Profiles never mix"):
        D.load_manifest(m["path"], expect_profile="hackrf_8000000_ci8")
    mp = Path(m["path"]) / "manifest.json"
    man = json.loads(mp.read_text("utf-8"))
    man["resampled"] = True
    mp.write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(D.DatasetRefusal, match="resampled"):
        D.load_manifest(m["path"])
    with pytest.raises(D.DatasetRefusal, match="split is one of"):
        next(D.iter_shards(m["path"], "everything"))
    ok, problems = D.verify(m["path"])
    assert not ok and "resampled" in problems[0]


def test_verify_names_missing_changed_and_unlisted_files(rf):
    m = _nb(rf)
    d = Path(m["path"])
    rel = sorted(m["files"])[0]
    with open(d / rel, "ab") as f:
        f.write(b"\x00")
    (d / "val" / "extra.npz").write_bytes(b"x")
    ok, problems = D.verify(d)
    assert not ok
    assert any("changed since the dataset was built" in p for p in problems)
    assert any("extra.npz is not in the manifest" in p for p in problems)
    (d / rel).unlink()
    ok, problems = D.verify(d)
    assert any(p == f"{rel} is missing" for p in problems)


# ---------------------------------------------------------------------------
# wideband
# ---------------------------------------------------------------------------
def test_wideband_scenes_tiles_and_boxes(rf):
    seen = []
    m = D.build_wideband(rf, PID, "wb", 2, scene_seconds=0.1, progress=seen.append,
                         splits=(0.5, 0.5, 0.0))
    for k in SECTION5_KEYS:
        assert k in m, k
    assert m["kind"] == "wideband" and m["splits"] == {"train": 1, "val": 1, "test": 0}
    assert m["scene"]["samples"] == 240_000 and m["scene"]["datatype"] == "cu8"
    assert m["tiles"].startswith("computed")
    assert m["notes"] == ["no environment profile: random classes, offsets and times"]
    assert len(seen) == 2 and seen[0].startswith("wb: scene 1/2")
    n_ann = 0
    for split in ("train", "val"):
        for sc in D.iter_shards(m["path"], split):
            meta = sc["meta"]
            assert sigmf.validate(meta) == []
            g = meta["global"]
            assert g["atk:receiver_profile"] == PID
            assert g["atk:tier"] == "invented" and g["atk:dataset"] == "wb"
            assert profiles.profile_from_meta(meta) == PID
            for a in sc["annotations"]:
                assert a.source == "synthetic"
                assert a.extra["atk:family"] in classes.FAMILIES
            n_ann += len(sc["annotations"])
            t = sc["tiles"]
            assert t["spec"].dtype == np.float16 and t["spec"].ndim == 3
            assert t["boxes"].shape[1] == 6 and t["boxes"].dtype == np.float32
            assert t["box_class"].shape[0] == t["boxes"].shape[0]
            if len(t["boxes"]):
                assert (t["boxes"][:, 3] > t["boxes"][:, 1]).all()
                assert (t["boxes"][:, 4] > t["boxes"][:, 2]).all()
    assert n_ann == m["scene"]["annotations"] > 0
    assert D.verify(m["path"]) == (True, [])


def test_wideband_for_bills_region(rf):
    m = D.build_wideband(rf, PID, "va", 1, env="us-va-nokesville",
                         scene_seconds=0.05, compute_tiles=False)
    assert m["environment"] == "us-va-nokesville"
    assert m["tiles"] == "not computed: compute_tiles=False"
    assert "a prior, not the place" in m["notes"][0]
    sc = next(D.iter_shards(m["path"], "train"))
    assert sc["meta"]["global"]["atk:environment"] == "us-va-nokesville"
    assert sc["tiles"] is None


def test_wideband_class_list_and_refusals(rf):
    m = D.build_wideband(rf, PID, "few", 1, scene_seconds=0.05,
                         classes=["nfm_voice", "pocsag"], compute_tiles=False)
    assert m["classes"] == ["nfm_voice", "pocsag"]
    for sc in D.iter_shards(m["path"], "train"):
        assert {a.label for a in sc["annotations"]} <= {"nfm_voice", "pocsag"}
    with pytest.raises(D.DatasetRefusal, match="n_scenes must be at least 1"):
        D.build_wideband(rf, PID, "z", 0)
    with pytest.raises(D.DatasetRefusal, match="shorter than one"):
        D.build_wideband(rf, PID, "z", 1, scene_seconds=1e-5)
    with pytest.raises(D.DatasetRefusal, match="not in the class table: bogus"):
        D.build_wideband(rf, PID, "z", 1, classes=["bogus"])


def test_a_bladerf_scene_is_written_with_q11_scaling(rf):
    pid = "bladerf1_2000000_ci16"
    m = D.build_wideband(rf, pid, "q11", 1, scene_seconds=0.02, compute_tiles=False)
    assert m["scene"]["datatype"] == "ci16q11"
    sc = next(D.iter_shards(m["path"], "train"))
    assert sc["meta"]["global"]["atk:datatype"] == "ci16q11"
    assert profiles.profile_from_meta(sc["meta"]) == pid


def test_missing_helpers_are_said_not_hidden(rf, monkeypatch):
    """SCF, STFT tiles and the floor are other engineers' modules: when one
    cannot be imported the dataset is still built and the manifest says
    what was not computed and why."""
    monkeypatch.setitem(sys.modules, "atk_diffusion.cyclo.scf", None)
    m = _nb(rf, compute_scf=True)
    assert m["scf"].startswith("not computed: atk_diffusion.cyclo.scf")
    assert "scf" not in next(D.iter_shards(m["path"], "train"))
    from atk_diffusion.dsp import floor as _floor

    def no_floor(*a, **k):
        raise ValueError("the stand-in was too short")

    monkeypatch.setattr(_floor.NoiseFloor, "from_terminated", no_floor)
    w = D.build_wideband(rf, PID, "nofloor", 1, scene_seconds=0.05)
    assert "estimated from each scene (ValueError: the stand-in was too short)" \
        in w["tiles"]
    import atk_diffusion.dsp as dsp_pkg
    monkeypatch.setitem(sys.modules, "atk_diffusion.dsp.stft", None)
    monkeypatch.delattr(dsp_pkg, "stft", raising=False)
    w2 = D.build_wideband(rf, PID, "nostft", 1, scene_seconds=0.05)
    assert w2["tiles"].startswith("not computed: atk_diffusion.dsp.stft")
    assert not [r for r in w2["files"] if "tiles_" in r]


# ---------------------------------------------------------------------------
# TorchSig 2.2.0 — the real package
# ---------------------------------------------------------------------------
def test_narrowband_with_the_real_torchsig(rf):
    pytest.importorskip("torchsig", reason="TorchSig is only in the training "
                        "environment")
    from atk_diffusion.synth import torchsig_backend as ts
    ok, why = ts.available()
    if not ok:
        pytest.skip(f"TorchSig is not usable here: {why}")
    m = D.build_narrowband(rf, PID, "ts", ["ref_qpsk", "p25"], 2, (5.0, 15.0),
                           "voice", generator="torchsig", window=128,
                           compute_scf=False, impairment_level=1)
    assert m["generator"] == "torchsig 2.2.0" == LB.TORCHSIG_GENERATOR
    assert m["method"] == "synthetic_torchsig" and m["tier"] == "invented"
    assert m["params"]["impairment_level"] == 1
    z = next(D.iter_shards(m["path"], "train"))
    assert z["iq"].shape[1] == 128 and np.isfinite(z["snr_db"]).all()
    assert set(z["label"].tolist()) <= {0, 1}
    assert D.verify(m["path"]) == (True, [])


def test_wideband_with_the_real_torchsig(rf):
    pytest.importorskip("torchsig", reason="TorchSig is only in the training "
                        "environment")
    from atk_diffusion.synth import torchsig_backend as ts
    if not ts.available()[0]:
        pytest.skip("TorchSig is not usable here")
    m = D.build_wideband(rf, PID, "tswb", 1, generator="torchsig",
                         scene_seconds=0.03, compute_tiles=False)
    assert m["generator"] == "torchsig 2.2.0"
    sc = next(D.iter_shards(m["path"], "train"))
    assert sc["meta"]["global"]["atk:generator"] == "torchsig 2.2.0"
    assert all(a.source == "synthetic" for a in sc["annotations"])


# ---------------------------------------------------------------------------
# the cabled-set ingester
# ---------------------------------------------------------------------------
def _cabled_capture(rf, name="loop1", pid=PID, cls="ref_qpsk", seconds=0.05,
                    offset=200e3, center=915e6, extra_global=None, anns=None,
                    record=True, seed=3):
    fs = float(profiles.parse_profile_id(pid).sample_rate)
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    x, lab = native.generate(cls, fs, n, 20.0, rng, carrier_offset_hz=offset,
                             noise_dbfs=-40.0, params={"bandwidth_hz": 12e3}
                             if cls.startswith("ref_") else None)
    if anns is None:
        a = LB.label_to_annotation(lab, center)
        a.extra["atk:source"] = "cabled"
        anns = [a]
    g = {"atk:receiver_profile": pid, "atk:tx_power_dbm": -10.0,
         "atk:attenuation_db": 30.0, "atk:expected_input_dbm": -40.0}
    g.update(extra_global or {})
    dt = profiles.parse_profile_id(pid).datatype
    base = rf.cabled(pid) / name
    dp, mp = sigmf.write_pair(base, x, fs, center, datatype=dt,
                              annotations=anns, extra_global=g,
                              hw="RTL-SDR v3")
    if record:
        rf.record(dp, "cabled")
        rf.record(mp, "cabled-meta")
    return base, lab


def test_ingest_cabled_cuts_the_record_into_the_test_split(rf):
    base, lab = _cabled_capture(rf)
    said = []
    m = D.ingest_cabled(rf, PID, [base], "cab", window=256, compute_scf=True,
                        progress=said.append)
    for k in SECTION5_KEYS:
        assert k in m, k
    assert m["generator"] == "cabled" and m["label_sources"] == ["cabled"]
    assert m["tier"] == "record" and m["method"] == "cabled_ingest"
    assert m["classes"] == ["ref_qpsk"]
    assert m["splits"] == {"train": 0, "val": 0, "test": 1}
    assert m["canonical"]["class"] == "voice"
    src = m["sources"][0]
    assert src["capture"] == f"{PID}/cabled/loop1.sigmf-data"
    assert src["sha256"] == sha256_file(sigmf.data_path(base))
    assert src["loop"]["atk:attenuation_db"] == 30.0
    assert m["notes"] == ["every cabled annotation was used"]
    assert said == ["cab: cut 1/1 cabled signals"]
    z = next(D.iter_shards(m["path"], "test"))
    assert z["iq"].shape == (1, 256) and z["scf"].shape == (1,) + D.SCF_SHAPE
    assert abs(float(z["snr_db"][0]) - 20.0) < 1e-3
    assert abs(float(z["bandwidth_hz"][0]) - lab["bandwidth_hz"]) < 1.0
    assert D.verify(m["path"]) == (True, [])
    # the capture changing afterwards is caught by the dataset's verify
    with open(sigmf.data_path(base), "r+b") as f:
        f.write(b"\x00\x00")
    ok, problems = D.verify(m["path"])
    assert not ok and "changed since the dataset was built" in problems[0]
    # ... and by the write log on the next ingest
    with pytest.raises(D.DatasetRefusal, match="named, not used"):
        D.ingest_cabled(rf, PID, [base], "cab2", window=256)


def test_ingest_cabled_notes_what_it_left_out(rf):
    good = LB.label_to_annotation(
        native.generate("ref_qpsk", 2.4e6, 1000, 20.0, np.random.default_rng(0),
                        params={"bandwidth_hz": 12e3})[1], 915e6)
    good.extra["atk:source"] = "cabled"
    unknown = sigmf.Annotation(0, 1000, 915e6, 915.01e6, "martian",
                               extra={"atk:source": "cabled"})
    edgeless = sigmf.Annotation(0, 1000, None, None, "ref_qpsk",
                                extra={"atk:source": "cabled"})
    wide = sigmf.Annotation(0, 1000, 914e6, 916e6, "fm_broadcast",
                            extra={"atk:source": "cabled"})
    taught = sigmf.Annotation(0, 1000, 915e6, 915.01e6, "ref_qpsk",
                              extra={"atk:source": "taught"})
    base, _ = _cabled_capture(rf, anns=[good, unknown, edgeless, wide, taught],
                              record=False)
    m = D.ingest_cabled(rf, PID, [str(sigmf.meta_path(base))], "cab",
                        canonical_class="voice", window=128, compute_scf=False)
    notes = " | ".join(m["notes"])
    assert "not in the write log" in notes
    assert "1 annotation(s) not atk:source=cabled" in notes
    assert "'martian' is not in the class table" in notes
    assert "has no frequency edges" in notes
    assert "does not fit the voice rate" in notes
    assert m["classes"] == ["ref_qpsk"] and m["splits"]["test"] == 1


def test_ingest_cabled_refusals(rf):
    with pytest.raises(D.DatasetRefusal, match="no captures given"):
        D.ingest_cabled(rf, PID, [], "x")
    other, _ = _cabled_capture(rf, name="hack", pid="hackrf_8000000_ci8")
    with pytest.raises(profiles.ProfileMismatch, match="Profiles never mix"):
        D.ingest_cabled(rf, PID, [other], "x")
    res, _ = _cabled_capture(rf, name="res",
                             extra_global={"atk:resampled_from": "hackrf_8000000_ci8"})
    with pytest.raises(D.DatasetRefusal, match="was resampled from"):
        D.ingest_cabled(rf, PID, [res], "x")
    none, _ = _cabled_capture(rf, name="none", anns=[])
    with pytest.raises(D.DatasetRefusal, match="nothing to learn from"):
        D.ingest_cabled(rf, PID, [none], "x")
    wide = sigmf.Annotation(0, 1000, 914e6, 916e6, "fm_broadcast",
                            extra={"atk:source": "cabled"})
    w, _ = _cabled_capture(rf, name="wide", anns=[wide])
    with pytest.raises(D.DatasetRefusal, match="no cabled annotation fits"):
        D.ingest_cabled(rf, PID, [w], "x", canonical_class="voice")
    assert not rf.datasets(PID, "x").exists()


def test_ingest_cabled_refuses_when_nothing_can_be_cut(rf):
    """A capture shorter than one window yields no example: that is a
    refusal, not an empty dataset whose manifest claims a class."""
    short, _ = _cabled_capture(rf, name="short", seconds=0.002)
    with pytest.raises(D.DatasetRefusal, match="shorter than one window"):
        D.ingest_cabled(rf, PID, [short], "x", window=4096)
    assert not rf.datasets(PID, "x").exists()


def test_ingest_cabled_into_train_and_test(rf):
    bases = [_cabled_capture(rf, name=f"c{i}", seed=i)[0] for i in range(4)]
    m = D.ingest_cabled(rf, PID, bases, "mix", window=128, compute_scf=False,
                        splits=(0.5, 0.0, 0.5))
    assert m["splits"] == {"train": 2, "val": 0, "test": 2}
    assert len(m["sources"]) == 4
