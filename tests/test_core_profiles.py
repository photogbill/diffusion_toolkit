# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Receiver profiles, canonical rates and the refusal (plan §3.1, §3.3, D2, D4)."""

from __future__ import annotations

import json

import pytest

from atk_diffusion import profiles as P


# -- naming --------------------------------------------------------------------
@pytest.mark.parametrize("args,pid", [
    (("rtlsdr", 2_400_000, "cu8"), "rtlsdr_2400000_cu8"),
    (("bladerf1", 4_000_000, "ci16_le"), "bladerf1_4000000_ci16"),
    (("bladerf1", 4_000_000, "ci16q11"), "bladerf1_4000000_ci16"),
    (("krakensdr", 2_400_000, "cu8", "ch0"), "krakensdr_2400000_cu8_ch0"),
    (("hackrf", 8_000_000, "ci8"), "hackrf_8000000_ci8"),
    (("sigmf-import", 250_000, "cf32_le"), "sigmf-import_250000_cf32"),
])
def test_profile_ids_are_built_and_parsed_back(args, pid):
    assert P.make_profile_id(*args) == pid
    back = P.parse_profile_id(pid)
    assert P.make_profile_id(back.family, back.sample_rate, back.datatype,
                             back.variant) == pid


def test_the_datatype_is_the_one_on_disk_not_the_plans_example():
    """The plan wrote rtlsdr_…_ci8; RTL samples are unsigned cu8."""
    assert P.FAMILIES["rtlsdr"]["datatype"] == "cu8"
    assert P.FAMILIES["krakensdr"]["datatype"] == "cu8"
    assert P.FAMILIES["hackrf"]["datatype"] == "ci8"


@pytest.mark.parametrize("bad", ["rtlsdr_2400000", "rtl_2400000_cu8",
                                 "rtlsdr_2.4e6_cu8", "rtlsdr_2400000_cs16",
                                 "", "RTLSDR 2400000 cu8"])
def test_bad_ids_are_refused(bad):
    with pytest.raises(ValueError):
        P.parse_profile_id(bad)


def test_fractional_rates_and_unknown_families_are_refused():
    with pytest.raises(ValueError):
        P.make_profile_id("rtlsdr", 2_400_000.5, "cu8")
    with pytest.raises(ValueError):
        P.make_profile_id("usrp", 2_400_000, "cu8")
    with pytest.raises(ValueError):
        P.make_profile_id("rtlsdr", 2_400_000, "cu8", "Bad Variant")


# -- canonical rates (D2) ------------------------------------------------------
def _rates(fs):
    return [(c.cls, c.rate, c.decimation) for c in P.canonical_rates(fs)]


def test_canonical_rates_reproduce_the_plans_bladerf_and_hackrf_examples():
    assert _rates(4_000_000) == [("voice", 50_000, 80), ("wideband", 500_000, 8),
                                 ("spread", 2_000_000, 2)]
    assert _rates(20_000_000) == [("voice", 50_000, 400),
                                  ("wideband", 500_000, 40),
                                  ("spread", 2_000_000, 10)]


def test_canonical_rates_for_the_rtl_hold_their_class():
    """48 k / 480 k / 2.4 M — not the plan's sketched 240 k / 1.2 M, which
    could not hold a 250 kHz LoRa chirp or a 2 MHz ADS-B signal."""
    assert _rates(2_400_000) == [("voice", 48_000, 50), ("wideband", 480_000, 5),
                                 ("spread", 2_400_000, 1)]


@pytest.mark.parametrize("fs", [1_024_000, 1_200_000, 1_440_000, 1_800_000,
                                1_920_000, 2_048_000, 2_400_000, 2_560_000,
                                4_000_000, 8_000_000, 10_000_000, 20_000_000,
                                30_720_000, 40_000_000])
def test_every_canonical_rate_is_an_exact_integer_decimation(fs):
    for c in P.canonical_rates(fs):
        assert c.decimation >= 1
        assert fs % c.decimation == 0
        assert c.rate == fs / c.decimation
        if not c.limited:
            assert c.rate >= P.BANDWIDTH_CLASSES[c.cls]["min_rate"]


def test_a_narrow_receiver_has_a_limited_voice_class_only():
    rates = P.canonical_rates(12_000)          # a KiwiSDR IQ stream
    assert [c.cls for c in rates] == ["voice"]
    assert rates[0].limited and rates[0].decimation == 1


def test_bandwidth_classes_choose_the_narrowest_that_holds():
    assert P.bandwidth_class(12_500) == "voice"
    assert P.bandwidth_class(125_000) == "wideband"
    assert P.bandwidth_class(2_000_000) == "spread"
    c = P.canonical_for(2_400_000, 200_000)
    assert (c.cls, c.decimation) == ("wideband", 5)


# -- the refusal ----------------------------------------------------------------
def test_the_refusal_is_the_plans_sentence():
    with pytest.raises(P.ProfileMismatch) as e:
        P.check_match("rtlsdr_2400000_cu8", "bladerf1_4000000_ci16",
                      what="this detector")
    msg = str(e.value)
    assert "this detector was trained for the RTL-SDR at 2.4 MS/s" in msg
    assert "this capture is the bladeRF 1.0 (x40/x115) at 4 MS/s" in msg


def test_matching_profiles_pass():
    P.check_match("rtlsdr_2400000_cu8", "RTLSDR_2400000_CU8")


def test_kraken_channels_are_described():
    assert P.describe("krakensdr_2400000_cu8_ch3").endswith("channel 3")


# -- deriving the profile of a capture -------------------------------------------
def test_profile_from_an_atk_recording_without_the_new_key():
    meta = {"global": {"core:datatype": "ci16_le", "atk:datatype": "ci16q11",
                       "core:sample_rate": 4_000_000, "core:hw": "Nuand bladeRF"}}
    assert P.profile_from_meta(meta) == "bladerf1_4000000_ci16"


def test_profile_from_a_kraken_recording_keeps_the_channel():
    meta = {"global": {"core:datatype": "cu8", "core:sample_rate": 2_400_000,
                       "core:hw": "KrakenSDR (channel 2)"}}
    assert P.profile_from_meta(meta) == "krakensdr_2400000_cu8_ch2"


def test_the_recorders_key_wins_over_derivation():
    meta = {"global": {"core:datatype": "cu8", "core:sample_rate": 2_400_000,
                       "core:hw": "something else",
                       "atk:receiver_profile": "rtlsdr_2400000_cu8"}}
    assert P.profile_from_meta(meta) == "rtlsdr_2400000_cu8"


def test_an_unknown_device_is_an_import_never_a_guess():
    meta = {"global": {"core:datatype": "cf32_le", "core:sample_rate": 1e6,
                       "core:hw": "Ettus B210"}}
    assert P.profile_from_meta(meta).startswith("sigmf-import_1000000_cf32")


def test_atk_source_keys_map_to_families():
    assert P.family_from_atk("rtl-sdr") == "rtlsdr"
    assert P.family_from_atk("bladerf", "x115") == "bladerf1"
    assert P.family_from_atk("bladerf", "micro") == "bladerf2"
    assert P.family_from_atk("kraken") == "krakensdr"


# -- the profile record ----------------------------------------------------------
def test_profile_round_trip_and_derived_fields_never_trusted(rf):
    prof = P.new_profile("rtlsdr_2400000_cu8")
    prof.safe_input = P.SafeInput(max_dbm=10.0, source="test sheet",
                                  entered="2026-10-08")
    path = P.save_profile(rf, prof)
    d = json.loads(path.read_text())
    assert d["description"] == "the RTL-SDR at 2.4 MS/s"
    d["canonical_rates"] = [{"cls": "voice", "rate": 1, "decimation": 1}]
    path.write_text(json.dumps(d))
    back = P.load_profile(rf, "rtlsdr_2400000_cu8")
    assert back.safe_input.max_dbm == 10.0
    assert [c.rate for c in back.canonical_rates()] == [48_000, 480_000, 2_400_000]


def test_an_unsaved_profile_says_its_impairments_are_unmeasured(rf):
    prof = P.load_profile(rf, "hackrf_20000000_ci8")
    assert any("unmeasured" in n for n in prof.notes)
    assert prof.stft.fft_size == 4096


def test_stft_geometry_scales_with_rate():
    assert P.default_stft(2_400_000).fft_size == 1024
    assert P.default_stft(10_000_000).fft_size == 2048
    assert abs(P.default_stft(2_400_000).rbw_hz(2_400_000) - 2343.75) < 1e-9
