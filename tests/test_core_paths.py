# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""rf_data: the root's rules, the layout and the write log (plan §3.2, D2)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from atk_diffusion import paths


def test_tests_never_resolve_to_a_real_root():
    root = paths.default_root()
    assert "atkdiff_rfdata_" in str(root)


def test_the_default_is_beside_the_code(monkeypatch, tmp_path):
    monkeypatch.delenv(paths.ROOT_ENV, raising=False)
    here = paths.toolkit_root()
    assert paths.default_root() == here.parent / "rf_data"


def test_vendored_inside_atk_shares_the_same_root(monkeypatch, tmp_path):
    monkeypatch.delenv(paths.ROOT_ENV, raising=False)
    atk = tmp_path / "Analyst_Toolkit" / "ATK"
    (atk / "atk").mkdir(parents=True)
    vendored = atk / "vendor" / "diffusion_toolkit"
    (vendored / "atk_diffusion").mkdir(parents=True)
    monkeypatch.setattr(paths, "toolkit_root", lambda: vendored)
    assert paths.default_root() == tmp_path / "Analyst_Toolkit" / "rf_data"


@pytest.mark.parametrize("bad", [
    "C:/Users/Bill/AppData/Local/rf_data", "D:/OneDrive/rf_data",
    "D:/OneDrive - Personal/rf_data", "E:/Dropbox/rf_data",
    "G:/Google Drive/rf_data"])
def test_appdata_and_sync_folders_are_refused(bad):
    ok, why = paths.check_root(bad)
    assert not ok and why
    with pytest.raises(paths.RootRefused):
        paths.RfData(bad)


def test_layout(rf):
    pid = "rtlsdr_2400000_cu8"
    assert rf.captures(pid) == rf.root / pid / "captures"
    assert rf.labeled(pid, "dmr") == rf.root / pid / "captures" / "labeled" / "dmr"
    assert rf.synthetic(pid, "nb1") == rf.root / pid / "synthetic" / "nb1"
    assert rf.models(pid, "m") == rf.root / pid / "models" / "m"
    assert rf.products("coverage") == rf.root / "products" / "coverage"
    assert (rf.root / "README.txt").read_text().startswith("rf_data")


def test_names_cannot_climb_out_of_the_root(rf):
    p = rf.synthetic("rtlsdr_2400000_cu8", "../../etc")
    assert rf.root in p.parents


def test_write_log_names_a_changed_file(rf):
    f = rf.captures("rtlsdr_2400000_cu8") / "a.sigmf-data"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"\x01" * 100)
    e = rf.record(f, "capture")
    assert e["size"] == 100 and len(e["sha256"]) == 64
    assert rf.verify(f) == (True, "")
    f.write_bytes(b"\x02" * 100)
    os.utime(f, ns=(e["mtime_ns"] + 10_000_000_000, e["mtime_ns"] + 10_000_000_000))
    ok, why = rf.verify(f)
    assert not ok and "changed after it was recorded" in why
    g = f.with_name("b.sigmf-data")
    g.write_bytes(b"x")
    ok, why = rf.verify(g)
    assert not ok and "not in the write log" in why


def test_write_log_survives_a_torn_line(rf):
    f = rf.root / "x.bin"
    f.write_bytes(b"abc")
    rf.record(f, "test")
    with open(rf.log.path, "a") as fh:
        fh.write('{"path": "torn')
    assert rf.verify(f)[0]


def test_known_profiles(rf):
    (rf.root / "hackrf_8000000_ci8").mkdir()
    (rf.profiles_dir() / "rtlsdr_2400000_cu8.json").write_text("{}")
    assert rf.known_profiles() == ["hackrf_8000000_ci8", "rtlsdr_2400000_cu8"]
