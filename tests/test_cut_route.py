# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The Route rules (DETECTION_DESIGN §4.2 step 4) and the report's
formatting, without a cut folder: the toolkit owns the rules, ATK owns the
tools — every tool that accepts the cut's class, and only those."""

from __future__ import annotations

import json

import numpy as np
import pytest

from atk_diffusion.cut import report as R
from atk_diffusion.cut import route as RT
from atk_diffusion.detect import classes


def test_the_candidates_come_from_every_witness_and_only_real_classes():
    a = {"class": {"cls": "dmr"},
         "source": {"detection": {"cls": "p25", "measurements": {
             "candidates": ["nxdn96", "made_up"]}}},
         "cyclic": {"candidates": ["dmr", "pocsag"]}}
    assert RT.candidates_of(a) == ["dmr", "p25", "nxdn96", "pocsag"]
    menu = RT.available(a)
    assert menu == ["dsd", "pager", "multimon"] + list(RT.ALWAYS)
    assert RT.candidates_of({"class": {"cls": classes.UNKNOWN}}) == []
    assert RT.available({}) == list(RT.ALWAYS)


def test_the_words_of_each_tool():
    assert "CONFIRMS" in RT.words("dsd")
    assert "audio only; it confirms nothing" in RT.words("nfm")
    assert RT.words("df") == RT.ROUTE_WORDS["df"]
    assert "on the cut itself" in RT.words("df", multichannel=True)
    assert RT.words("a-tool-from-the-future") == "a-tool-from-the-future"


def test_perform_labels_what_a_tool_ran_on():
    a = {"cleans": [{"method": "matched", "parameters": {"symbol_rate_hz": 4800.0}}]}
    meta = {"global": {"atk:tier": "inferred", "big": np.zeros(5000),
                       "n": np.int64(3), "v": np.float32(0.5), "c": 1 + 2j}}
    rec = RT.perform(a, "bench", "cleaned_2", np.zeros(10, np.complex64),
                     48_000.0, meta, runner=lambda iq, fs, m: [1, 2],
                     who="bill")
    assert rec["by"] == "bill" and rec["ran"] and rec["result"] == {"result": [1, 2]}
    assert "INFERRED signal, not on the record" in rec["input_note"]
    assert "decoded_from_note" not in rec
    assert rec["matched_parameters_passed"]
    with pytest.raises(ValueError, match="offers only the tools"):
        RT.perform(a, "dsd", "original", np.zeros(4), 48_000.0, {})


def test_json_conversion_keeps_numbers_and_shortens_big_arrays():
    out = json.loads(json.dumps({
        "i": np.int32(7), "f": np.float64(2.5), "a": np.arange(3),
        "big": np.zeros(5000), "c": 3 - 4j, "p": object()}, default=RT._default))
    assert out["i"] == 7 and out["f"] == 2.5 and out["a"] == [0, 1, 2]
    assert out["big"] == "<array (5000,)>" and out["c"] == [3.0, -4.0]
    assert out["p"].startswith("<object")


def test_the_report_formats_what_it_is_given_and_nothing_else():
    assert R._f(None) == "—" and R._f("n/a", "{:.1f}") == "n/a"
    assert R._plain(True) == "yes" and R._plain(4800.0) == "4,800"
    assert R._plain([1.5, 2]) == "[1.5, 2]" and R._plain({"a": 1}) is None
    assert R._plain("x" * 100) is None
    text = R.render({"folder": "f", "cut": {}, "files": {},
                     "cyclic": {"peaks": [], "pfa": 1e-6, "probe_words": []},
                     "class": {"cls": "UNKNOWN", "tier": "proposed"},
                     "fingerprint": {"id": "tx-1"},
                     "cleans": [{"id": "clean_1", "method": "matched",
                                 "tier": "measured",
                                 "parameters": {"symbol_rate_hz": 4800.0,
                                                "carrier_offset_hz": 2.0,
                                                "timing_offset_s": None},
                                 "no_file": "no matched-filtered file: FSK.",
                                 "model_sha256": "ab" * 32,
                                 "note": "applied to channel 0 of 5"}],
                     "routes": [{"tool": "dsd", "input": "cleaned",
                                 "input_tier": "cleaned", "at": "t",
                                 "decoded_from_note": "decoded from a CLEANED",
                                 "input_note": "x", "result": {"ok": 1}}]})
    assert "none: no symbol clock or carrier feature" in text
    assert "### clean_1 — matched (MEASURED)" in text
    assert "first symbol centre — after" in text
    assert "no matched-filtered file: FSK." in text
    assert "Model weights sha256" in text and "channel 0 of 5" in text
    assert "⚠ decoded from a CLEANED" in text and "result: {'ok': 1}" in text
    assert "Fingerprint: {'id': 'tx-1'}" in text
