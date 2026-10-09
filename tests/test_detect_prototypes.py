# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Open set by prototypes; teach = add a prototype (DETECTION_DESIGN §4,
decision D6; plan B5)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from atk_diffusion import cards, profiles
from atk_diffusion.detect.classes import UNKNOWN
from atk_diffusion.detect.prototypes import PrototypeBank

PID = "rtlsdr_2400000_cu8"
D = 16


def _centres(rng, k):
    c = rng.normal(size=(k, D))
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def _samples(rng, centre, n, spread=0.25):
    x = centre[None, :] + spread * rng.normal(size=(n, D)) / np.sqrt(D)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _bank(rng, n_train=60):
    cen = _centres(rng, 6)
    names = ["dmr", "p25", "pocsag", "nfm_voice"]
    bank = PrototypeBank(PID, "voice")
    for name, c in zip(names, cen[:4]):
        bank.add(name, _samples(rng, c, n_train))
    held = {name: _samples(rng, c, 200) for name, c in zip(names, cen[:4])}
    unknown = np.vstack([_samples(rng, cen[4], 150), _samples(rng, cen[5], 150)])
    return bank, held, unknown, cen


def test_calibrated_thresholds_accept_the_quantile_and_reject_the_unknown(rng):
    bank, held, unknown, _cen = _bank(rng)
    calib = {k: v[:100] for k, v in held.items()}
    test = {k: v[100:] for k, v in held.items()}
    thr = bank.calibrate_thresholds(calib)
    assert set(thr) == set(bank.classes) and all(bank.is_calibrated(c) for c in thr)
    acc = bank.accuracy(test)
    assert acc["correct"] > 0.88 and acc["confused"] < 0.02
    assert bank.unknown_rejection(unknown) > 0.95
    ev = bank.evaluate(test, unknown)
    assert ev["unknown_rejection"] == bank.unknown_rejection(unknown)


def test_classify_returns_the_class_or_unknown_with_the_nearest(rng):
    bank, held, unknown, _cen = _bank(rng)
    m = bank.classify(held["p25"][0])
    c, dist, thr = m                                   # unpacks as the contract says
    assert c == "p25" and dist <= thr and m.nearest == "p25" and not m.note
    u = bank.classify(unknown[0])
    assert u.cls == UNKNOWN and u.unknown and u.nearest in bank.classes
    assert "not close enough" in u.note and u.distance > u.threshold
    assert PrototypeBank(PID, "voice").classify(unknown[0]).cls == UNKNOWN
    with pytest.raises(ValueError, match="all zeros"):
        bank.classify(np.zeros(D))
    with pytest.raises(ValueError, match="different classifier"):
        bank.classify(np.ones(D + 1))


def test_teach_a_class_from_five_examples_in_seconds(rng):
    bank, held, _unknown, cen = _bank(rng)
    bank.calibrate_thresholds({k: v[:100] for k, v in held.items()})
    new = _samples(rng, cen[4], 5)
    words = bank.teach("lora_local", new)
    assert "taught lora_local from 5 example(s)" in words and "thin" not in words
    assert bank.is_taught("lora_local") and not bank.is_thin("lora_local")
    later = _samples(rng, cen[4], 200)                 # the "next day's" examples
    got = [m.cls for m in bank.classify_many(later)]
    assert np.mean([g == "lora_local" for g in got]) > 0.6
    # the old classes are unharmed
    assert bank.accuracy({"dmr": held["dmr"][100:]})["correct"] > 0.85


def test_a_thin_class_has_a_wider_threshold_and_says_so(rng):
    bank, _held, _unknown, cen = _bank(rng)
    words = bank.teach("beacon", _samples(rng, cen[5], 3))
    assert bank.is_thin("beacon") and "beacon is thin: 3 examples" in words
    base = bank._base["beacon"]
    assert bank.threshold("beacon") == pytest.approx(base * 1.5)
    m = bank.classify(_samples(rng, cen[5], 1)[0])
    if m.cls == "beacon":
        assert "thin" in m.note and "Teach more examples" in m.note
    # one example: nothing to measure the spread from — borrowed, widened
    bank.teach("single", _samples(rng, _centres(rng, 1)[0], 1))
    others = [bank._base[c] for c in bank.classes
              if c != "single" and bank.examples(c) >= 2]
    assert len(others) == 5
    assert bank.threshold("single") == pytest.approx(np.median(others) * 1.5)
    # extending it past the floor makes it an ordinary class
    bank.teach("beacon", _samples(rng, cen[5], 4))
    assert not bank.is_thin("beacon") and bank.examples("beacon") == 7


def test_sub_prototypes_cover_a_class_with_two_looks(rng):
    cen = _centres(rng, 3)
    two_looks = np.vstack([_samples(rng, cen[0], 40, 0.15), _samples(rng, cen[1], 40, 0.15)])
    one = PrototypeBank(PID, "voice").add("pocsag", two_looks)
    km = PrototypeBank(PID, "voice", k_sub=2).add("pocsag", two_looks)
    probe = _samples(rng, cen[1], 50, 0.15)
    d_one = np.mean([m.distance for m in one.classify_many(probe)])
    d_km = np.mean([m.distance for m in km.classify_many(probe)])
    assert km._proto["pocsag"].shape[0] == 2 and d_km < 0.5 * d_one


def test_save_load_round_trip_and_the_refusals(tmp_path, rng):
    bank, held, unknown, cen = _bank(rng)
    bank.calibrate_thresholds({k: v[:100] for k, v in held.items()})
    bank.teach("taught_one", _samples(rng, cen[4], 6))
    d = tmp_path / "clf"
    bank.save(d)
    back = PrototypeBank.load(d, PID, "voice")
    assert back.classes == bank.classes and back.dim == D
    for c in bank.classes:
        assert back.threshold(c) == pytest.approx(bank.threshold(c))
        assert back.is_calibrated(c) == bank.is_calibrated(c)
    probe = np.vstack([held["dmr"][150:160], unknown[:10]])
    assert [m.cls for m in back.classify_many(probe)] == \
        [m.cls for m in bank.classify_many(probe)]
    with pytest.raises(profiles.ProfileMismatch, match="made for the RTL-SDR"):
        PrototypeBank.load(d, "bladerf1_4000000_ci16")
    with pytest.raises(ValueError, match="voice cuts, not wideband"):
        PrototypeBank.load(d, PID, "wideband")
    with pytest.raises(FileNotFoundError):
        PrototypeBank.load(tmp_path / "nothing", PID)


def test_a_bank_is_bound_to_the_classifier_weights_and_updates_its_card(tmp_path, rng):
    d = tmp_path / "clf"
    d.mkdir()
    (d / "model.onnx").write_bytes(b"weights-v1")
    card = cards.new_card("clf-v1", "classifier1d", PID,
                          classes=[{"name": "dmr", "source": "trained", "examples": 900}])
    cards.save(d, card, "model.onnx")
    sha = cards.load(d).weights["sha256"]
    bank, _held, _unknown, cen = _bank(rng)
    bank.model_sha256 = sha
    bank.teach("lora_local", _samples(rng, cen[4], 3))
    bank.save(d)
    bank.update_card(d)
    c2 = cards.load(d, expect_kind="classifier1d", for_profile=PID)
    entry = {c["name"]: c for c in c2.classes}
    assert entry["lora_local"]["source"] == "taught" and entry["lora_local"]["examples"] == 3
    assert entry["lora_local"]["thin"] is True and entry["dmr"]["source"] == "trained"
    assert "1 taught" in "\n".join(cards.summary(c2))
    assert PrototypeBank.load(d, PID).classes == bank.classes          # same weights
    # new weights: the old embeddings mean nothing to them
    (d / "model.onnx").write_bytes(b"weights-v2")
    j = json.loads((d / "card.json").read_text())
    j["weights"]["sha256"] = "f" * 64
    (d / "card.json").write_text(json.dumps(j))
    with pytest.raises(ValueError, match="different classifier"):
        PrototypeBank.load(d, PID)


def test_describe_says_what_is_calibrated_and_what_is_thin(rng):
    bank, held, _u, cen = _bank(rng)
    bank.calibrate_thresholds({"dmr": held["dmr"]})
    bank.teach("thin_one", _samples(rng, cen[4], 2))
    lines = "\n".join(bank.describe())
    assert "the RTL-SDR at 2.4 MS/s" in lines
    assert "calibrated on held-out data" in lines and "UNCALIBRATED" in lines
    assert "THIN" in lines
    with pytest.raises(ValueError, match="UNKNOWN is not one"):
        bank.add(UNKNOWN, _samples(rng, cen[0], 3))
    with pytest.raises(ValueError, match="canonical class"):
        PrototypeBank(PID, "huge")
