# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The toolkit's side of the contract with ATK's standard-library twin,
`atk/core/rf_data.py` (paths.py and profiles.py say: "ATK keeps its own copy,
held to this one by a contract test on both sides").

Two layers:

1. GOLDEN — the table below is a copy of ATK's `rf_data.GOLDEN`, value for
   value. The toolkit must give exactly these answers (profile ids, the
   canonical rates as (class, rate, decimation), the words). It runs always,
   ATK or not. Change a rule, change this table, in BOTH repositories.
2. THE TWO IMPLEMENTATIONS SIDE BY SIDE — when ATK is present (ATK_HOME,
   else /home/claude/atk_work, else ..\\ATK beside the toolkit) its file is
   loaded BY PATH with importlib (nothing of ATK is imported as a package)
   and both are swept over every family x the rates from 1.024 to 40 MS/s
   (the Kraken's among them) x every datatype tag either side knows. Absent,
   the comparison is skipped with the reason printed.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from atk_diffusion import paths, profiles

#: A copy of ATK's rf_data.GOLDEN (as of 2026-10-09). Identical values, on
#: purpose: test_the_golden_table_is_atks_own fails the day they differ.
GOLDEN = {
    "ids": [
        (("rtlsdr", 2_400_000, "cu8", ""), "rtlsdr_2400000_cu8"),
        (("bladerf1", 4_000_000, "ci16q11", ""), "bladerf1_4000000_ci16"),
        (("bladerf1", 4_000_000, "ci16_le", ""), "bladerf1_4000000_ci16"),
        (("krakensdr", 2_400_000, "cu8", "ch0"), "krakensdr_2400000_cu8_ch0"),
        (("hackrf", 8_000_000, "ci8", ""), "hackrf_8000000_ci8"),
        (("sigmf-import", 250_000, "cf32_le", ""), "sigmf-import_250000_cf32"),
    ],
    "canonical": [
        (2_400_000, [("voice", 48_000, 50), ("wideband", 480_000, 5),
                     ("spread", 2_400_000, 1)]),
        (4_000_000, [("voice", 50_000, 80), ("wideband", 500_000, 8),
                     ("spread", 2_000_000, 2)]),
        (20_000_000, [("voice", 50_000, 400), ("wideband", 500_000, 40),
                      ("spread", 2_000_000, 10)]),
        (12_000, [("voice", 12_000, 1)]),
    ],
    "describe": [
        ("rtlsdr_2400000_cu8", "the RTL-SDR at 2.4 MS/s"),
        ("krakensdr_2400000_cu8_ch3", "the KrakenSDR at 2.4 MS/s, channel 3"),
        ("bladerf1_4000000_ci16", "the bladeRF 1.0 (x40/x115) at 4 MS/s"),
        ("hackrf_20000000_ci8", "the HackRF One at 20 MS/s"),
        ("kiwisdr_12000_ci16", "the KiwiSDR (remote) at 12 kS/s"),
    ],
}

#: 1.024 to 40 MS/s: the RTL's and the Kraken's rates (1.024, 1.4, 1.8,
#: 1.92, 2.048, 2.4, 2.56, 2.88, 3.2 MS/s), the bladeRF's and the HackRF's,
#: LTE's multiples of 1.92 MS/s, a prime to make the decimation search work,
#: and the KiwiSDR's 12 kS/s and an import's 250 kS/s below the range.
RATES = (1_024_000, 1_200_000, 1_400_000, 1_536_000, 1_800_000, 1_920_000,
         2_000_000, 2_048_000, 2_400_000, 2_560_000, 2_880_000, 3_000_000,
         3_200_000, 3_840_000, 4_000_000, 5_000_000, 6_000_000, 7_680_000,
         8_000_000, 9_600_000, 10_000_000, 12_000_000, 12_500_000,
         15_360_000, 16_000_000, 20_000_000, 25_000_000, 30_720_000,
         38_400_000, 40_000_000, 1_999_993, 12_000, 250_000)


def _canon(rate) -> list[tuple]:
    return [(c.cls, c.rate, c.decimation) for c in profiles.canonical_rates(rate)]


# ---------------------------------------------------------------------------
# 1. the GOLDEN table, always
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("args, want", GOLDEN["ids"])
def test_golden_profile_ids(args, want):
    assert profiles.make_profile_id(*args) == want
    p = profiles.parse_profile_id(want)
    assert profiles.make_profile_id(p.family, p.sample_rate, p.datatype,
                                    p.variant) == want


@pytest.mark.parametrize("rate, want", GOLDEN["canonical"])
def test_golden_canonical_rates(rate, want):
    got = _canon(rate)
    assert got == want
    for (cls, r, d), (_c, wr, wd) in zip(got, want):
        assert isinstance(d, int) and r == wr and float(r) * d == float(rate)


@pytest.mark.parametrize("pid, words", GOLDEN["describe"])
def test_golden_words(pid, words):
    assert profiles.describe(pid) == words


def test_the_golden_table_is_atks_own_when_atk_is_here():
    atk = _atk_rf_data()
    if atk is None:
        pytest.skip(_ABSENT)
    assert GOLDEN == atk.GOLDEN, ("the toolkit's copy of GOLDEN differs from "
                                  "ATK's atk/core/rf_data.py — change both")


# ---------------------------------------------------------------------------
# 2. the two implementations side by side
# ---------------------------------------------------------------------------
_ABSENT = ""


def _atk_candidates() -> list[Path]:
    out = []
    if os.environ.get("ATK_HOME", "").strip():
        out.append(Path(os.environ["ATK_HOME"].strip()))
    out.append(Path("/home/claude/atk_work"))
    out.append(paths.toolkit_root().parent / "ATK")
    return out


def _atk_rf_data():
    """ATK's atk/core/rf_data.py loaded by path (stdlib only), or None."""
    global _ABSENT
    tried = []
    for root in _atk_candidates():
        f = root / "atk" / "core" / "rf_data.py"
        tried.append(str(f))
        if f.is_file():
            spec = importlib.util.spec_from_file_location("atk_rf_data_contract", f)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    _ABSENT = ("ATK is not present here, so its rf_data.py cannot be compared "
               "with the toolkit's (looked for " + "; ".join(tried) + "). The "
               "GOLDEN cases above still ran. Set ATK_HOME to an ATK folder to "
               "run the comparison.")
    print(_ABSENT)
    return None


def test_absent_atk_is_a_skip_with_its_reason(monkeypatch, tmp_path, capsys):
    """The skip path itself, exercised: no ATK -> None and a printed reason
    naming where it looked (so a sweep that compared nothing cannot pass
    silently as if it had)."""
    monkeypatch.setattr(__import__(__name__), "_atk_candidates",
                        lambda: [tmp_path / "no_atk_here"])
    assert _atk_rf_data() is None
    printed = capsys.readouterr().out
    assert "ATK is not present here" in printed and "no_atk_here" in printed
    assert "The GOLDEN cases above still ran" in _ABSENT


@pytest.fixture(scope="module")
def atk():
    mod = _atk_rf_data()
    if mod is None:
        pytest.skip(_ABSENT)
    return mod


def test_the_constants_agree(atk):
    assert atk.ROOT_ENV == paths.ROOT_ENV
    assert atk.ROOT_NAME == paths.ROOT_NAME
    assert atk.WRITE_LOG == paths.WRITE_LOG
    assert tuple(atk.FORBIDDEN_SEGMENTS) == tuple(paths.FORBIDDEN_SEGMENTS)
    assert atk._PID.pattern == profiles._PID.pattern
    assert atk._VARIANT.pattern == profiles._VARIANT.pattern
    assert atk.CLASS_MIN == tuple((c, int(profiles.BANDWIDTH_CLASSES[c]["min_rate"]))
                                  for c in profiles.CLASS_ORDER)


def test_the_families_agree(atk):
    assert set(atk.FAMILIES) == set(profiles.FAMILIES)
    for fam, d in atk.FAMILIES.items():
        mine = profiles.FAMILIES[fam]
        for k in ("label", "adc_bits", "datatype"):
            assert d[k] == mine[k], (fam, k)


def test_the_datatype_tags_agree(atk):
    tags = set(atk._DT) | set(profiles._DT_ALIASES) | {"", "CU8", " ci16q11 ",
                                                        "int8", "cf64"}
    for t in sorted(tags):
        assert atk.normalize_datatype(t) == profiles.normalize_datatype(t), t


def _both(fn_a, fn_b, *args):
    """Both raise ValueError, or both return the same thing."""
    try:
        a = fn_a(*args)
    except ValueError:
        a = ValueError
    try:
        b = fn_b(*args)
    except ValueError:
        b = ValueError
    return a, b


def test_the_sweep_ids_words_and_canonical_rates(atk):
    tags = sorted(set(atk._DT) | set(profiles._DT_ALIASES) | {"bogus"})
    families = sorted(set(atk.FAMILIES) | {"nosuchradio"})
    n = 0
    for fam in families:
        for rate in RATES + (2_400_000.5, 0, -1):
            for dt in tags:
                for var in ("", "ch0", "ch4", "Gain-High", "bad variant!"):
                    a, b = _both(atk.make_profile_id, profiles.make_profile_id,
                                 fam, rate, dt, var)
                    assert a == b, (fam, rate, dt, var, a, b)
                    n += 1
                    if a is ValueError:
                        continue
                    assert atk.describe(a) == profiles.describe(a), a
                    pa = atk.parse_profile_id(a)
                    pb = profiles.parse_profile_id(a)
                    assert pa == {"family": pb.family, "sample_rate": pb.sample_rate,
                                  "datatype": pb.datatype, "variant": pb.variant}
    assert n > 10_000
    for rate in RATES:
        assert atk.canonical_rates(rate) == _canon(rate), rate


def test_parsing_and_the_atk_sources_agree(atk):
    for pid in ("rtlsdr_2400000_cu8", "krakensdr_2048000_cu8_ch4",
                "RTLSDR_2400000_CU8", "rtlsdr_2400000", "rtlsdr_2.4e6_cu8",
                "foo_1_cu8", "hackrf_8000000_ci8_", "bladerf2_61440000_ci16_xb200"):
        a, b = _both(atk.parse_profile_id, profiles.parse_profile_id, pid)
        if a is ValueError or b is ValueError:
            assert a is b is ValueError, pid
        else:
            assert a == {"family": b.family, "sample_rate": b.sample_rate,
                         "datatype": b.datatype, "variant": b.variant}
    for src in ("rtl-sdr", "RTLSDR", "hackrf", "kraken", "krakensdr", "spyserver",
                "kiwi", "kiwisdr", "airspy", "bladerf", "nonsense", ""):
        for model in ("x115", "x40", "micro"):
            a, b = _both(atk.family_from_atk, profiles.family_from_atk, src, model)
            assert a == b, (src, model)


def test_root_rules_agree(atk, tmp_path):
    for p in (tmp_path / "rf_data", tmp_path / "Users" / "b" / "AppData" / "rf",
              tmp_path / "OneDrive - Work" / "rf", tmp_path / "Dropbox" / "x",
              tmp_path / "Google Drive" / "rf", tmp_path / "iCloud Drive",
              tmp_path / "Box Sync" / "rf", tmp_path / "appdatax" / "rf"):
        ok_a, why_a = atk.check_root(p)
        ok_b, why_b = paths.check_root(p)
        assert ok_a == ok_b, p
        if not ok_a:
            # the same refusal, said for ATK or for the toolkit
            assert why_a.split(".")[0] == why_b.split(".")[0], (why_a, why_b)


def test_the_write_log_is_one_format(atk, tmp_path):
    root = tmp_path / "rf_data"
    rf = paths.RfData(root, create=True)
    f1 = root / "rtlsdr_2400000_cu8" / "captures" / "a.sigmf-data"
    f1.parent.mkdir(parents=True)
    f1.write_bytes(b"\x01\x02" * 100)
    atk.record(root, f1, "capture", "written by ATK's recorder")
    assert rf.verify(f1) == (True, "")               # ATK writes, the toolkit reads
    f2 = f1.with_name("b.sigmf-data")
    f2.write_bytes(b"\x03" * 50)
    rf.record(f2, "capture")
    assert atk.verify(root, f2) == (True, "")        # and the other way round
    assert set(atk.entries(root)) == set(rf.log.entries())
    f1.write_bytes(b"\x09\x02" * 100)
    os.utime(f1, ns=(1, 1))
    assert not atk.verify(root, f1)[0] and not rf.verify(f1)[0]
    assert "changed after it was recorded" in atk.verify(root, f1)[1]
    assert "changed after it was recorded" in rf.verify(f1)[1]
