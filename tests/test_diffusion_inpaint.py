# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""D1: the learned IQ inpainter (learn.inpaint) and its experiment
(experiments.inpaint_eval). The modem tests are pure numpy; the model tests
train a tiny inpainter that proves the path (card, ONNX in the core
environment, only masked samples touched, the repair-track hook), not the
method."""

from __future__ import annotations

import json

import numpy as np
import pytest

from atk_diffusion import cards, profiles
from atk_diffusion.learn import inpaint as IP

PID = "rtlsdr_2400000_cu8"
FS = 48000.0


def test_the_sync_word_is_dmrs_dibit_mapping():
    s = IP.sync_symbols()
    assert s.size == 24
    # 0x7 0x5 = 0111 0101 -> dibits 01 11 01 01 -> +3 -3 +3 +3
    assert list(s[:4]) == [3, -3, 3, 3]


def test_the_modem_reads_clean_traffic_without_error(rng):
    x, bursts = IP.tdma_stream(3, FS, rng)
    for b in bursts:
        sym, v = IP.fsk4_demod(x, FS, b["start"], IP.BURST_SYMBOLS)
        assert np.array_equal(sym, b["symbols"])
        assert np.array_equal(sym[IP.SYNC_AT:IP.SYNC_AT + 24], IP.sync_symbols())
        # averaging the middle 70 % of a symbol reaches into its transitions:
        # the levels come out ~10 % inside nominal, in order, well apart
        lv = [np.median(v[b["symbols"] == s]) for s in (-3, -1, 1, 3)]
        assert all(np.diff(lv) > IP.DEV_HZ)
        assert lv[3] == pytest.approx(3 * IP.DEV_HZ, rel=0.2)
    y, _ = IP.add_noise(x, 20.0, rng, fs=FS)
    ser = np.mean([np.mean(IP.fsk4_demod(y, FS, b["start"], IP.BURST_SYMBOLS)[0]
                           != b["symbols"]) for b in bursts])
    assert ser < 0.05


def test_a_bad_fill_upsets_the_decoder_after_the_gap(rng):
    """The premise of D1: what goes into the gap changes errors AFTER it."""
    x, bursts = IP.tdma_stream(2, FS, rng)
    b = bursts[1]
    s, g = b["start"] + 70 * 10, 48
    good = x.copy()
    spike = x.copy()
    spike[s:s + g] = np.exp(2j * np.pi * 20000.0 * np.arange(g) / FS)   # a wild fill
    e_good = IP.fsk4_demod(good, FS, b["start"], IP.BURST_SYMBOLS)[0] != b["symbols"]
    e_bad = IP.fsk4_demod(spike, FS, b["start"], IP.BURST_SYMBOLS)[0] != b["symbols"]
    after = slice(70 + 6, 70 + 6 + 40)
    assert e_good[after].sum() == 0 and e_bad[after].sum() > 5


def test_the_hallucination_rule_is_judged_against_the_truth(rng):
    noise = (rng.normal(size=64) + 1j * rng.normal(size=64)) / np.sqrt(2)
    line = np.linspace(noise[0], noise[-1], 64)
    flag, ex = IP.hallucinated(line, noise)
    assert not flag                                      # noise-derived
    flag, ex = IP.hallucinated(4.0 * noise, noise)       # +12 dB of 'signal'
    assert flag and ex == pytest.approx(12.04, abs=0.01)


def test_noise_power_is_measured_from_the_quiet_cells(rng):
    x, _b = IP.tdma_stream(4, FS, rng)
    y, p_n = IP.add_noise(x, 15.0, rng, fs=FS)
    assert IP.noise_power(y) == pytest.approx(p_n, rel=0.15)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    torch.set_num_threads(1)
    from atk_diffusion.learn import unet as U
    from atk_diffusion.paths import RfData
    rf = RfData(tmp_path_factory.mktemp("inp") / "rf_data", create=True)
    d = IP.train_inpainter(rf, PID, window=64, synthetic=24, steps=15, batch=8, T=100,
                           unet=U.TINY_1D, repaint_steps=4, resample=2, validation=4,
                           name="tiny_inp")
    return rf, d


def test_training_writes_an_invented_card_that_runs_without_torch(tiny):
    rf, d = tiny
    card = cards.load(d, expect_kind="inpainter", for_profile=PID)
    assert card.tier == "invented" and card.weights["format"] == "onnx"
    assert card.input["rate"] == 48000.0 and card.input["decimation"] == 50
    assert card.input["canonical_class"] == "voice" and card.input["x0_clip"] > 0
    m = card.metrics
    assert m["hallucination"]["rule"] == IP.HALLUCINATION_RULE
    assert set(m["gap_snr_db"]) == {"diffusion", "linear"}
    inp = IP.Inpainter.load(d, for_profile=PID)
    assert inp.backend == "onnxruntime"
    with pytest.raises(profiles.ProfileMismatch):
        IP.Inpainter.load(d, for_profile="hackrf_8000000_ci8")


def test_inpaint_touches_only_the_gap_and_says_what_it_is(tiny, rng):
    _rf, d = tiny
    x, _b = IP.tdma_stream(2, FS, rng)
    y, p_n = IP.add_noise(10 * x, 0.0, rng, fs=FS, p_n=1.0)
    mask = np.zeros(y.size, dtype=bool)
    mask[1000:1020] = True
    mask[3000:3200] = True                       # longer than the 64-sample window
    out, info = IP.inpaint(y, mask, FS, model_dir=d, profile=PID, noise_power=1.0, seed=1)
    assert out.shape == y.shape and np.array_equal(out[~mask], y[~mask])
    assert np.all(np.isfinite(out)) and not np.array_equal(out[mask], y[mask])
    assert info["tier"] == "invented" and info["method"] == "diffusion_inpaint"
    assert info["chained"] and "invented on invented" in info["note"]
    with pytest.raises(ValueError, match="cut it to that rate first"):
        IP.inpaint(y, mask, 2.4e6, model_dir=d)
    with pytest.raises(RuntimeError, match="no inpainter model was given"):
        IP.inpaint(y, mask, FS)


def test_the_repair_tracks_learned_hook_calls_it(tiny, rng):
    iqd = pytest.importorskip("atk_diffusion.repair.iq_dropout",
                              reason="repair.iq_dropout is another engineer's module")
    _rf, d = tiny
    x, _b = IP.tdma_stream(2, FS, rng)
    y, _ = IP.add_noise(10 * x, 0.0, rng, fs=FS, p_n=1.0)
    y[2000:2030] = 0
    out, used = iqd.fill_spans(y, [(2000, 30)], method="learned", fs=FS,
                               learned_kw={"model_dir": str(d), "profile": PID,
                                           "noise_power": 1.0})
    assert used == ["diffusion_inpaint"]
    assert np.array_equal(out[:2000], y[:2000]) and np.array_equal(out[2030:], y[2030:])
    assert np.any(out[2000:2030] != 0)


def test_the_eval_with_classical_fills_only(tmp_path):
    from atk_diffusion.experiments import inpaint_eval as IE
    res = IE.run(None, PID, trials=6, gaps_ms=(1.0,), snr_db=22.0, silence_trials=6,
                 seed=2, out_dir=tmp_path / "d1")
    g = res["gaps"]["1"]
    assert res["methods"][0] == "zeros" and "linear" in res["methods"]
    assert res["method_tiers"]["zeros"] == "record"
    assert res["method_tiers"]["linear"] == "inferred"
    for m, r in g["methods"].items():
        for k in ("ser_gap", "ser_post", "ser_burst", "sync_found"):
            assert 0.0 <= r[k] <= 1.0
        if r["hallucination"] is not None:
            assert r["hallucination"]["rate"] == 0.0        # classical fills invent nothing
    assert res["reference_ser_no_dropout"] < 0.05
    md = (tmp_path / "d1" / "inpaint_eval.md").read_text(encoding="utf-8")
    assert "symbol errors after the gap" in md and "INVENTED" in md
    det = (tmp_path / "d1" / "inpaint_eval_detail.md").read_text(encoding="utf-8")
    assert "SER after gap" in det
    js = json.loads((tmp_path / "d1" / "inpaint_eval.json").read_text(encoding="utf-8"))
    assert js["tier"] == "measured"
    assert js["result"]["gaps"]["1"]["gap_samples"] == 48


def test_the_eval_scores_the_inpainter_beside_them(tiny):
    from atk_diffusion.experiments import inpaint_eval as IE
    rf, d = tiny
    res = IE.run(rf, PID, inpainter=d, trials=3, gaps_ms=(0.5,), silence_trials=3,
                 steps=3, resample=1, seed=4)
    assert res["methods"][-1] == "diffusion"
    assert res["method_tiers"]["diffusion"] == "invented"
    r = res["gaps"]["0.5"]["methods"]["diffusion"]
    assert r["used"] == ["diffusion_inpaint"] and r["hallucination"]["eligible"] == 3
    assert res["inpainter"]["backend"] == "onnxruntime"
    assert res["report_md"].startswith(str(rf.runs(PID)))
