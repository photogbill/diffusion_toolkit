# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Point-and-ask and teach-it-a-signal (plan §4.B5; DETECTION_DESIGN §4)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from atk_diffusion import cards, sigmf
from atk_diffusion.ask import point, teach
from atk_diffusion.detect.boxes import Detection
from atk_diffusion.profiles import ProfileMismatch

RTL = "rtlsdr_2400000_cu8"


# -- the colour ramp and the PNG ---------------------------------------------
def _atk_rgb_for(level_db, lo_db=0.0, hi_db=40.0):
    """ATK's own scalar `atk/core/levels.py::rgb_for`, verbatim logic."""
    span = float(hi_db) - float(lo_db)
    x = 0.0 if span <= 0 else (float(level_db) - float(lo_db)) / span
    x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
    prev_p, prev_c = point.WF_STOPS[0]
    for pos, col in point.WF_STOPS:
        if x <= pos:
            t = 0.0 if pos <= prev_p else (x - prev_p) / (pos - prev_p)
            return tuple(int(round(a + (b - a) * t)) for a, b in zip(prev_c, col))
        prev_p, prev_c = pos, col
    return point.WF_STOPS[-1][1]


def test_the_ramp_is_atks_waterfall_ramp():
    levels = np.linspace(-10, 50, 601)
    got = point.ramp_rgb(levels, 0.0, 40.0)
    want = np.array([_atk_rgb_for(v) for v in levels])
    assert np.max(np.abs(got.astype(int) - want)) <= 1
    assert tuple(point.ramp_rgb(-99.0)) == (8, 14, 40)          # below the floor
    assert tuple(point.ramp_rgb(99.0)) == (255, 250, 190)       # strong


def test_png_round_trips_and_carries_its_description():
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, size=(7, 11, 3), dtype=np.uint8)
    blob = point.encode_png(rgb, {"Description": "crop — test"})
    assert blob.startswith(b"\x89PNG\r\n\x1a\n")
    back, text = point.decode_png(blob)
    assert np.array_equal(back, rgb)
    assert text["Description"] == "crop - test"


def test_png_is_readable_by_pillow():
    Image = pytest.importorskip("PIL.Image", reason="Pillow is optional")
    import io
    rgb = np.zeros((3, 5, 3), dtype=np.uint8)
    rgb[1, 2] = (230, 220, 90)
    im = Image.open(io.BytesIO(point.encode_png(rgb)))
    assert im.size == (5, 3) and im.mode == "RGB"
    assert im.getpixel((2, 1)) == (230, 220, 90)


def test_a_one_row_burst_survives_shrinking():
    s = np.zeros((4000, 300))
    s[1234, 100:120] = 30.0                    # one frame, 20 bins
    out = point.fit_size(s, 256, 1024)
    assert out.shape[0] <= 1024 and out.max() == 30.0


# -- the payload ----------------------------------------------------------------
def _crop():
    s = np.full((64, 128), 1.0)
    s[20:30, 60:70] = 25.0
    axes = {"t0": 10.0, "t1": 11.0, "f_lo": 462.5e6, "f_hi": 462.6e6,
            "db_ref": "above_floor"}
    return s, axes


def _det():
    return Detection(t0=10.3, t1=10.5, f_lo=462.55e6, f_hi=462.5625e6,
                     sources=("energy", "learned"), family="fsk", cls="dmr",
                     confidence=0.62, snr_db=18.0, profile=RTL)


def test_payload_describes_the_axes_and_lists_the_facts_with_tiers():
    s, axes = _crop()
    p = point.build_payload(s, axes, _det(), {"symbol_rate_hz": 4800.0})
    blob = p["png"]
    assert blob.startswith(b"\x89PNG")
    rgb, _ = point.decode_png(blob)
    assert min(rgb.shape[:2]) >= 256                 # big enough to read
    f = p["facts"]
    assert f["centre_hz"] == pytest.approx(462.55625e6)
    assert f["bandwidth_hz"] == pytest.approx(12_500.0)
    assert f["symbol_rate_hz"] == 4800.0 and f["pri_s"] is None
    assert f["class"] == "dmr" and f["confidence"] == pytest.approx(0.62)
    assert f["tiers"]["class"] == "proposed"
    assert f["tiers"]["symbol_rate_hz"] == "measured"
    pr = p["prompt"]
    assert "462.5 MHz at the left edge" in pr and "462.6 MHz" in pr
    assert "top row is the earliest moment" in pr
    assert "symbol rate: 4800 symbols/s" in pr
    assert "pulse repetition interval (PRI): not measured" in pr
    assert "PROPOSED — no decoder has confirmed it" in pr
    assert "fingerprint: not known" in pr
    assert pr.rstrip().endswith('say "I don\'t know".')
    assert p["tier"] == "proposed"


def test_confirmed_class_and_fingerprint_are_worded_as_such():
    s, axes = _crop()
    d = _det()
    d.confirm("dsd", "TG 1234", "dmr")
    p = point.build_payload(s, axes, d.to_json(),
                            {"fingerprint": {"name": "handheld A",
                                             "distance": 0.21}})
    assert "CONFIRMED by the dsd decoder" in p["prompt"]
    assert "matches handheld A (distance 0.21)" in p["prompt"]


class _Model:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def __call__(self, prompt, png):
        self.calls.append((prompt, png))
        return self.reply(prompt) if callable(self.reply) else self.reply


def test_ask_returns_a_hypothesis_and_counts_the_way_out():
    s, axes = _crop()
    p = point.build_payload(s, axes, _det())
    m = _Model("It looks like DMR: two-slot TDMA bursts, 12.5 kHz wide.")
    r = point.ask(p, "What is this?", m)
    assert r["tier"] == "proposed" and "not an identification" in r["tier_words"]
    assert r["said_dont_know"] is False and "DMR" in r["answer"]
    prompt, png = m.calls[0]
    assert png == p["png"] and "THE ANALYST'S QUESTION: What is this?" in prompt
    assert prompt.rstrip().endswith('say "I don\'t know".')
    assert point.ask(p, "?", _Model("I don't know."))["said_dont_know"]
    assert point.ask(p, "?", _Model("<think>hmm</think> NSTR"))["said_dont_know"]
    long = ("I don't know the operator, but the 4800 symbols/s rate and the "
            "slot structure say this is DMR rather than P25 or NXDN.")
    assert point.said_dont_know(long) is False


def test_a_failing_model_is_a_sentence_not_an_exception():
    s, axes = _crop()
    p = point.build_payload(s, axes)

    def broken(prompt, png):
        raise RuntimeError("model not loaded")
    r = point.ask(p, "?", broken)
    assert "could not be asked" in r["error"] and r["answer"] == ""
    assert "returned nothing" in point.ask(p, "?", _Model(""))["error"]


def test_without_measurements_the_prompt_says_so_and_still_ends_with_the_way_out():
    s, axes = _crop()
    p = point.build_payload(s, axes, _det(), {"symbol_rate_hz": 4800.0})
    m = _Model("unsure")
    point.ask(p, "?", m, include_facts=False)
    prompt = m.calls[0][0]
    assert "symbol rate" not in prompt and "answer from the picture alone" in prompt
    assert prompt.rstrip().endswith('say "I don\'t know".')


def test_compare_with_without_scores_both_conditions():
    s, axes = _crop()
    payloads = [point.build_payload(s, axes, _det(), {"symbol_rate_hz": 4800.0})]

    def reply(prompt):
        return "This is DMR." if "4800" in prompt else "I don't know."
    res = point.compare_with_without(payloads, ["dmr"], _Model(reply))
    assert res["with"]["accuracy"] == 1.0 and res["without"]["accuracy"] == 0.0
    assert res["without"]["dont_know"] == 1


def test_message_for_is_atks_data_uri_shape():
    s, axes = _crop()
    msg = point.message_for(point.build_payload(s, axes), "what?")
    content = msg[0]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_crop_around_takes_the_box_with_margin():
    S = np.zeros((100, 200))
    t = np.arange(100) * 0.01
    f = 100e6 + np.arange(200) * 1e3
    crop, axes = point.crop_around(S, t, f, 0.40, 0.60, 100.05e6, 100.07e6)
    assert crop.shape[0] >= 16 and crop.shape[1] >= 16
    assert axes["f_lo"] < 100.05e6 < 100.07e6 < axes["f_hi"]
    assert axes["t0"] < 0.40 and axes["t1"] > 0.60


# -- teach ----------------------------------------------------------------------
FS = 2_400_000.0
CF = 462.0e6


def _capture(rf, profile=RTL, offset=120e3, seconds=0.06, seed=0,
             datatype="cu8", hw="RTL-SDR Blog V3"):
    rng = np.random.default_rng(seed)
    n = int(FS * seconds)
    k = np.arange(n)
    sym = rng.integers(0, 2, size=n // 500 + 1).repeat(500)[:n] * 2 - 1
    phase = 2 * np.pi * np.cumsum(offset + 2400.0 * sym) / FS
    x = 0.3 * np.exp(1j * phase) + 0.02 * (rng.standard_normal(n)
                                           + 1j * rng.standard_normal(n))
    base = rf.captures(profile) / f"cap{seed}"
    sigmf.write_pair(base, x.astype(np.complex64), FS, CF, datatype=datatype,
                     extra_global={"atk:receiver_profile": profile}, hw=hw)
    return base


def _embed(windows, fs):
    spec = np.abs(np.fft.fft(windows, axis=1))
    e = spec.reshape(spec.shape[0], 16, -1).sum(axis=2)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


class _Bank:
    example_floor = 5

    def __init__(self):
        self.calls = []

    def teach(self, cls, embeddings):
        self.calls.append((cls, np.asarray(embeddings)))
        return {"message": f"'{cls}' taught from {len(embeddings)} examples"}


def _classifier_card(rf, profile=RTL):
    d = rf.models(profile, "cls-v1")
    d.mkdir(parents=True)
    (d / "model.onnx").write_bytes(b"onnx")
    cards.save(d, cards.new_card("cls-v1", "classifier1d", profile,
                                 classes=[{"name": "dmr", "source": "trained",
                                           "examples": 900}]), "model.onnx")
    return d


def _marks(base, n=3):
    return [teach.Mark(f_lo_hz=CF + 120e3 - 6e3, f_hi_hz=CF + 120e3 + 6e3,
                       t0_s=0.01 * i, t1_s=0.01 * i + 0.012, capture=base)
            for i in range(n)]


def test_teach_stores_embeds_prototypes_and_updates_the_card(rf):
    base = _capture(rf)
    cdir = _classifier_card(rf)
    bank = _Bank()
    res = teach.teach(rf, RTL, "mynet", _marks(base), embed=_embed, bank=bank,
                      window=256, who="bill")
    # stored as taught SigMF at the voice-class canonical rate
    assert res.canonical_class == "voice" and res.canonical_rate == 48_000.0
    assert len(res.files) == 3 and res.examples == 3 and res.total_examples == 3
    meta = sigmf.read_meta(res.files[0])
    g = meta["global"]
    assert g["core:sample_rate"] == 48_000.0 and g["atk:decimation"] == 50
    assert g["atk:receiver_profile"] == RTL and g["atk:cut_by"] == "bill"
    assert g["atk:tier"] == "record" and g["atk:canonical_class"] == "voice"
    ann = meta["annotations"][0]
    assert ann["atk:source"] == "taught" and ann["core:label"] == "mynet"
    assert sigmf.validate(meta) == []
    for f in res.files:
        assert rf.verify(f)[0]
    # one embedding per MARK, not per window
    assert bank.calls[0][0] == "mynet" and bank.calls[0][1].shape == (3, 16)
    assert res.bank_updated and "taught from 3 examples" in res.bank
    # below the bank's floor of 5: it says so
    assert res.floor == 5 and res.below_floor
    assert any("below the floor of 5" in s for s in res.lines())
    # the card's class list follows
    card = cards.load(cdir)
    entry = [c for c in card.classes if c["name"] == "mynet"][0]
    assert entry["source"] == "taught" and entry["examples"] == 3
    assert entry["below_floor"] is True
    llm = teach.class_list_for_llm(card)
    assert "dmr (trained)" in llm
    assert any(s.startswith("mynet (taught, 3 examples — below the floor")
               for s in llm)
    # teaching more extends the count and clears the flag at the floor
    res2 = teach.teach(rf, RTL, "mynet", _marks(base, 2), embed=_embed,
                       bank=bank, window=256)
    assert res2.total_examples == 5 and not res2.below_floor
    entry = [c for c in cards.load(cdir).classes if c["name"] == "mynet"][0]
    assert entry["examples"] == 5 and entry["below_floor"] is False


def test_teach_without_a_bank_stores_and_says_the_class_is_not_live(rf):
    base = _capture(rf)
    res = teach.teach(rf, RTL, "mynet", _marks(base, 1), embed=_embed, window=256)
    assert not res.bank_updated and "not live yet" in res.bank
    assert "No classifier card exists" in res.card
    assert res.floor == teach.EXAMPLE_FLOOR and res.below_floor
    z = np.load(res.embeddings_file)
    assert z["mark_embeddings"].shape == (1, 16)


def test_teach_refuses_another_receivers_capture(rf):
    base = _capture(rf, profile="hackrf_2400000_ci8", datatype="ci8",
                    hw="HackRF One")
    with pytest.raises(ProfileMismatch, match="this teach was trained for"):
        teach.teach(rf, RTL, "x", _marks(base, 1), embed=_embed, window=256)


def test_teach_refuses_iq_at_another_rate_and_the_name_unknown(rf):
    iq = np.zeros(48_000, dtype=np.complex64)
    m = teach.Mark(f_lo_hz=CF - 5e3, f_hi_hz=CF + 5e3, iq=iq, fs=2_048_000.0,
                   center_hz=CF)
    with pytest.raises(ProfileMismatch, match="never resamples silently"):
        teach.teach(rf, RTL, "x", [m], embed=_embed)
    with pytest.raises(ValueError, match="honest answer"):
        teach.teach(rf, RTL, "unknown", [m], embed=_embed)


def test_a_short_mark_is_padded_and_says_so(rf):
    rng = np.random.default_rng(3)
    iq = (rng.standard_normal(24_000) + 1j * rng.standard_normal(24_000)
          ).astype(np.complex64)
    m = teach.Mark(f_lo_hz=CF - 5e3, f_hi_hz=CF + 5e3, iq=iq, fs=FS,
                   center_hz=CF, t0_s=0.0, t1_s=0.005)
    cuts = teach.cut_examples([m], RTL, window=1024)
    assert cuts[0].padded and cuts[0].windows.shape == (1, 1024)


def test_queue_finetune_writes_the_marker_with_class_counts(rf):
    base = _capture(rf)
    teach.teach(rf, RTL, "mynet", _marks(base, 2), embed=_embed, window=256)
    p = teach.queue_finetune(rf, RTL, reason="two taught classes")
    j = json.loads(p.read_text())
    assert j["classes"] == {"mynet": 2} and j["requests"][0]["reason"]
    teach.queue_finetune(rf, RTL, reason="again")
    assert len(teach.pending_finetune(rf, RTL)["requests"]) == 2
    assert rf.verify(p)[0]


def test_integration_with_the_real_prototype_bank(rf):
    pytest.importorskip("atk_diffusion.detect.prototypes",
                        reason="detect.prototypes (another engineer's module) "
                               "is not importable yet")
    bank, why = teach.make_bank(RTL, "voice")
    assert bank is not None, why
    assert type(bank).__name__ == "PrototypeBank"
    base = _capture(rf)
    cdir = _classifier_card(rf)
    res = teach.teach(rf, RTL, "mynet", _marks(base, 3), embed=_embed,
                      bank=bank, window=256, card_dir=cdir)
    assert res.bank_updated and "taught mynet from 3 example" in res.bank
    assert bank.examples("mynet") == 3 and bank.is_taught("mynet")
    assert res.floor == bank.example_floor and res.below_floor
    assert "saved beside the classifier's card" in res.bank
    from atk_diffusion.detect.prototypes import PrototypeBank
    again = PrototypeBank.load(cdir, RTL, "voice")
    assert again.examples("mynet") == 3
    entry = [c for c in cards.load(cdir).classes if c["name"] == "mynet"][0]
    assert entry["source"] == "taught" and entry["examples"] == 3
    assert entry["thin"] is True and entry["below_floor"] is True
