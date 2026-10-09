# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Model cards: a model without a card does not load (DETECTION_DESIGN §6.5)."""

from __future__ import annotations

import json

import pytest

from atk_diffusion import cards as C
from atk_diffusion import profiles as P


def _model(tmp_path, profile="rtlsdr_2400000_cu8", kind="proposer2d"):
    d = tmp_path / "m"
    d.mkdir()
    (d / "model.onnx").write_bytes(b"weights")
    card = C.new_card("det-v1", kind, profile,
                      classes=[{"name": "dmr", "source": "trained", "examples": 900},
                               {"name": "mine", "source": "taught", "examples": 5}],
                      metrics={"domain_gap": 0.12, "latency_ms": 85.0})
    C.save(d, card, weights_file="model.onnx")
    return d


def test_no_card_no_load(tmp_path):
    with pytest.raises(C.CardRefusal, match="does not load"):
        C.load(tmp_path)


def test_card_loads_for_its_own_profile(tmp_path):
    d = _model(tmp_path)
    card = C.load(d, expect_kind="proposer2d", for_profile="rtlsdr_2400000_cu8")
    assert card.class_names() == ["dmr", "mine"] and card.tier == "proposed"


def test_card_refuses_another_profile_in_words(tmp_path):
    d = _model(tmp_path)
    with pytest.raises(P.ProfileMismatch, match="trained for the RTL-SDR at 2.4 MS/s"):
        C.load(d, for_profile="bladerf1_4000000_ci16")


def test_card_refuses_another_kind(tmp_path):
    d = _model(tmp_path)
    with pytest.raises(C.CardRefusal, match="not the 1D classifier"):
        C.load(d, expect_kind="classifier1d")


def test_changed_weights_are_refused(tmp_path):
    d = _model(tmp_path)
    (d / "model.onnx").write_bytes(b"weightz")
    with pytest.raises(C.CardRefusal, match="not the file the card describes"):
        C.load(d)


def test_a_per_profile_kind_needs_a_profile(tmp_path):
    card = C.new_card("x", "denoiser", "")
    card.weights = {"file": "w.pt"}
    assert any("per receiver profile" in p for p in C.validate(card))


def test_summary_says_how_far_to_trust_it(tmp_path):
    d = _model(tmp_path)
    lines = C.summary(C.load(d))
    text = "\n".join(lines)
    assert "2 classes, 1 taught" in text and "domain gap: 0.12" in text
    assert "outputs are PROPOSED" in text
    card = C.new_card("y", "denoiser", "rtlsdr_2400000_cu8")
    assert any("not yet measured" in l for l in C.summary(card))


def test_find_lists_only_loadable_models(tmp_path, rf):
    pid = "rtlsdr_2400000_cu8"
    good = rf.models(pid, "good")
    good.mkdir(parents=True)
    (good / "model.onnx").write_bytes(b"w")
    C.save(good, C.new_card("good", "proposer2d", pid), "model.onnx")
    other = rf.models(pid, "wrong-profile")
    other.mkdir(parents=True)
    (other / "model.onnx").write_bytes(b"w")
    C.save(other, C.new_card("wp", "proposer2d", "hackrf_8000000_ci8"), "model.onnx")
    found = C.find(rf, pid, "proposer2d")
    assert [c.name for _d, c in found] == ["good"]
