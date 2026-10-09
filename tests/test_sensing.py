# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""ESP32 CSI sensing (plan §4.I): the console parser (both firmwares), the
serial reader and replay, the PulseFi pipeline with its research label,
breathing pauses, Fresnel placement, presence/motion, the small LSTM and the
synthetic experiment."""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from atk_diffusion.experiments import vitals_eval as VE
from atk_diffusion.sensing import csi as C
from atk_diffusion.sensing import presence as P
from atk_diffusion.sensing import vitals as V

MAC = "aa:bb:cc:dd:ee:ff"


def _espressif_line(data, ts=1234567, first_word=0, n_len=None, rssi=-38):
    meta = ["CSI_DATA", "0", MAC, str(rssi), "11", "1", "7", "0", "1", "1", "1",
            "0", "0", "0", "-96", "0", "6", "0", str(ts), "0", "47", "0",
            str(len(data) if n_len is None else n_len), str(first_word)]
    return ",".join(meta) + ',"[' + ",".join(str(v) for v in data) + ']"'


def _tool_line(data, ts=1234567):
    meta = ["CSI_DATA", "STA", MAC, "-41", "11", "1", "7", "0", "1", "1", "1",
            "0", "0", "0", "-95", "0", "6", "0", str(ts), "0", "128", "0", "1",
            "1696000000", str(len(data))]
    return ",".join(meta) + ",[" + " ".join(str(v) for v in data) + " ]"


# -- the parser ----------------------------------------------------------------------
def test_espressif_line_imaginary_then_real():
    data = [3, 4, -5, 12] + [0, 0] * 62
    fr = C.parse_line(_espressif_line(data))
    assert fr.variant == "esp-csi" and fr.mac == MAC and fr.rssi == -38
    assert fr.channel == 6 and fr.noise_floor == -96 and fr.t_us == 1234567
    assert fr.n_subcarriers == 64
    assert fr.csi[0] == 4 + 3j and fr.csi[1] == 12 - 5j        # [imag, real]
    assert fr.amplitude[0] == pytest.approx(5.0)
    assert fr.amplitude[1] == pytest.approx(13.0)


def test_csi_tool_line_with_a_role_and_spaces():
    data = [1, 2, 3, 4]
    fr = C.parse_line(_tool_line(data))
    assert fr.variant == "esp32-csi-tool" and fr.fields["role"] == "STA"
    assert fr.fields["real_timestamp"] == "1696000000"
    assert np.allclose(fr.csi, [2 + 1j, 4 + 3j])


def test_a_header_line_maps_another_firmwares_columns():
    p = C.CsiParser()
    assert p.parse("type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,"
                   "channel,local_timestamp,sig_len,rx_state,len,first_word,"
                   "data") is None
    fr = p.parse('CSI_DATA,7,' + MAC + ',-50,11,-97,3,20,11,999,60,0,4,0,"[1,2,3,4]"')
    assert fr.variant == "header" and fr.channel == 11 and fr.t_us == 999
    assert fr.noise_floor == -97 and fr.fields["agc_gain"] == "20"


def test_damaged_and_foreign_lines_are_skipped_and_counted():
    p = C.CsiParser()
    full = _espressif_line([1, 2] * 64)
    assert p.parse("I (312) wifi: mode : sta (aa:bb:cc:dd:ee:ff)") is None
    assert p.parse(full[:-40]) is None                         # torn
    assert "never closes" in p.last_error
    assert p.parse(_espressif_line([1, 2] * 64, n_len=384)) is None
    assert "length field says 384" in p.last_error
    assert p.parse("I (5) csi_recv: " + full) is not None     # a log prefix
    assert p.good == 1 and p.bad == 2 and p.skipped == 1
    assert "1 CSI frames read, 2 damaged lines skipped" in p.status()


def test_first_word_invalid_zeroes_the_first_two_subcarriers():
    fr = C.parse_line(_espressif_line([9, 9, 9, 9, 3, 4], first_word=1))
    assert fr.csi[0] == 0 and fr.csi[1] == 0 and fr.csi[2] == 4 + 3j


def test_serial_reader_is_a_plain_iterator_and_closes_the_port(tmp_path):
    lines = [b"ets Jun  8 2016 00:22:57\r\n"] + [
        (_espressif_line([1, 2] * 8, ts=1000 * i) + "\r\n").encode()
        for i in range(5)] + [b""]

    class FakePort:
        def __init__(self):
            self.i, self.closed = 0, False

        def readline(self):
            v = lines[min(self.i, len(lines) - 1)]
            self.i += 1
            return v

        def close(self):
            self.closed = True
    port = FakePort()
    got = list(C.serial_frames(opener=lambda: port, max_frames=3,
                               log_to=tmp_path / "raw.txt"))
    assert len(got) == 3 and port.closed and got[0].host_t is not None
    assert len(list(C.replay(tmp_path / "raw.txt"))) == 3


@pytest.mark.skipif(importlib.util.find_spec("serial") is not None,
                    reason="pyserial is installed here")
def test_missing_pyserial_is_a_sentence():
    with pytest.raises(RuntimeError, match="pyserial is not installed"):
        next(C.serial_frames("COM5"))


def test_to_matrix_unwraps_the_32_bit_clock_and_keeps_one_format():
    frames = []
    for i in range(6):
        ts = (C.TS_WRAP - 30_000 + 20_000 * i) % C.TS_WRAP     # wraps at i = 2
        frames.append(C.parse_line(_espressif_line([1, 2] * 8, ts=ts)))
    frames.append(C.parse_line(_espressif_line([1, 2] * 4, ts=500)))   # other format
    t, H, info = C.to_matrix(frames)
    assert np.allclose(np.diff(t), 0.02) and H.shape == (6, 8)
    assert info["dropped_other_format"] == 1 and info["clock"] == "esp32"
    tg, Ag, rinfo = C.resample_uniform(t, np.abs(H), 100.0)
    assert Ag.shape[1] == 8 and rinfo["gaps"] == []


def test_save_log_round_trips_through_the_parser(tmp_path):
    fr = C.parse_line(_espressif_line([3, 4, -5, 12] + [1, 1] * 10, ts=77))
    p = C.save_log([fr, fr], tmp_path / "s.txt")
    back = list(C.replay(p))
    assert len(back) == 2 and np.allclose(back[0].csi, fr.csi) and back[0].t_us == 77


# -- the vitals pipeline ----------------------------------------------------------------
def test_vitals_on_synthetic_csi_with_known_rates():
    t, H, truth = VE.synth_csi(60.0, 50.0, breath_bpm=15.0, heart_bpm=80.0,
                               rng=np.random.default_rng(4))
    rep = VE.classical(t, H, 50.0, apnea=False)
    assert rep.breath_bpm == pytest.approx(15.0, abs=1.0)
    assert rep.heart_bpm == pytest.approx(80.0, abs=5.0)
    assert rep.label == V.RESEARCH_LABEL and rep.tier == "measured"
    assert all(w["label"] == V.RESEARCH_LABEL for w in rep.windows)
    assert rep.lines()[0] == V.RESEARCH_LABEL
    assert "not a medical device and not a diagnosis" in rep.to_json()["label"]


def test_an_empty_room_mostly_reports_no_rate():
    t, H, _ = VE.synth_csi(60.0, 50.0, person=False, rng=np.random.default_rng(11))
    rep = VE.classical(t, H, 50.0, apnea=False)
    n = len(rep.windows)
    assert sum(w["breath_bpm"] is not None for w in rep.windows) <= 0.2 * n
    assert all(w["why"] for w in rep.windows if w["breath_bpm"] is None)
    assert "no clear rhythm" in rep.lines()[1] or rep.breath_bpm is None


def test_a_breathing_pause_is_found_and_labelled():
    t, H, _ = VE.synth_csi(100.0, 50.0, breath_bpm=14.0, apnea=(40.0, 58.0),
                           rng=np.random.default_rng(99))
    rep = VE.classical(t, H, 50.0)
    assert len(rep.apnea) == 1
    e = rep.apnea[0]
    assert 38.0 <= e["t0_s"] <= 48.0 and e["duration_s"] >= 10.0
    assert e["label"] == V.RESEARCH_LABEL
    t, H, _ = VE.synth_csi(100.0, 50.0, breath_bpm=14.0,
                           rng=np.random.default_rng(98))
    assert VE.classical(t, H, 50.0).apnea == []


def test_rates_come_from_a_line_that_repeats():
    fs = 20.0
    t = np.arange(int(40 * fs)) / fs
    x = np.sin(2 * np.pi * 0.25 * t) + 0.05 * np.random.default_rng(0).standard_normal(t.size)
    rate, prom = V.spectral_rate(x, fs, V.BREATH_BAND, min_prominence=20.0)
    ac, r = V.autocorr_rate(x, fs, V.BREATH_BAND, min_r=0.5)
    assert rate == pytest.approx(15.0, abs=0.3) and ac == pytest.approx(15.0, abs=0.5)
    assert r > 0.8
    # a weak sub-harmonic makes two periods correlate better than one: the
    # highest peak would read half the rate; the earliest strong one does not
    y = np.sin(2 * np.pi * 0.45 * t) + 0.2 * np.sin(np.pi * 0.45 * t)
    ac2, _ = V.autocorr_rate(y, fs, V.BREATH_BAND)
    assert ac2 == pytest.approx(27.0, abs=0.5)
    # a heart rate with a breathing harmonic beside it (the synthetic case
    # that read 47 for 95 before the rule)
    t2, H, _ = VE.synth_csi(60.0, 50.0, breath_bpm=22.0, heart_bpm=95.0,
                            rng=np.random.default_rng(1))
    rep = VE.classical(t2, H, 50.0, apnea=False)
    assert rep.heart_bpm == pytest.approx(95.0, abs=3.0)


def test_gaps_in_the_csi_are_not_measured():
    t, H, _ = VE.synth_csi(60.0, 50.0, rng=np.random.default_rng(5))
    cut = (t > 20) & (t < 23)
    tg, Ag, info = C.resample_uniform(t[~cut], np.abs(H[~cut]), 50.0)
    assert info["gaps"] and info["gap_fraction"] > 0.04
    rep = V.estimate(Ag, 50.0, gaps=info["gaps"], apnea=False)
    hit = [w for w in rep.windows if "gap of" in w["why"]]
    assert hit and all(w["breath_bpm"] is None for w in hit)


def test_too_short_a_recording_is_refused_in_words():
    with pytest.raises(ValueError, match="at least 10 s"):
        V.estimate(np.ones((100, 8)), 50.0)


# -- placement ------------------------------------------------------------------------
def test_fresnel_zones_and_the_placement_advice():
    lam = 299_792_458.0 / 2.437e9
    tx, rx = (0.0, 0.0), (4.0, 0.0)
    # on the bisector, the first-zone radius is sqrt(lam * d / 4)
    z = V.fresnel_zone(tx, rx, (2.0, 0.05))
    assert z["r1_m"] == pytest.approx(math.sqrt(lam * 4.0 / 4.0), rel=1e-3)
    assert z["zone"] == 1
    # put the chest exactly on the boundary of zone 3 (excess = 3 lam / 2)
    h = math.sqrt((2.0 + 0.75 * lam) ** 2 - 4.0)
    b = V.placement(tx, rx, (2.0, h))
    assert b["sensitivity"] < 0.05
    assert b["best_sensitivity"] > 0.95 and abs(b["move_m"]) > 0
    assert "Moving it" in b["words"] and b["label"] == V.RESEARCH_LABEL
    # the middle of a zone is already the best spot
    h2 = math.sqrt((2.0 + 0.625 * lam) ** 2 - 4.0)
    m = V.placement(tx, rx, (2.0, h2))
    assert m["sensitivity"] > 0.95 and "a good spot" in m["words"]


# -- presence ---------------------------------------------------------------------------
def test_presence_and_motion_against_an_empty_room():
    t, H0, _ = VE.synth_csi(60.0, 20.0, person=False, rng=np.random.default_rng(1))
    _tg, A0, _i = C.resample_uniform(t, np.abs(H0), 20.0)
    base = P.fit_baseline(A0[: 40 * 20], 20.0)
    quiet = P.detect(A0[40 * 20:], 20.0, base)
    assert quiet.motion_fraction <= 0.2
    assert quiet.tier == "proposed" and "not identification" in quiet.label
    t, H1, _ = VE.synth_csi(30.0, 20.0, person=False, walker=True,
                            rng=np.random.default_rng(1))
    _tg, A1, _i = C.resample_uniform(t, np.abs(H1), 20.0)
    busy = P.detect(A1, 20.0, base)
    assert busy.motion_fraction >= 0.8
    assert busy.windows[0]["label"] == P.LABEL
    with pytest.raises(ValueError, match="different frame format"):
        P.detect(A1[:, :10], 20.0, base)
    assert P.Baseline.from_json(base.to_json()).motion_threshold == base.motion_threshold


def test_a_short_baseline_is_refused():
    with pytest.raises(ValueError, match="nobody in the room"):
        P.fit_baseline(np.ones((20, 8)), 20.0)


# -- the small LSTM ----------------------------------------------------------------------
def test_vitals_lstm_trains_saves_and_loads_through_its_card(tmp_path):
    torch = pytest.importorskip("torch", reason="PyTorch is only in the training "
                                                "environment")
    torch.set_num_threads(1)
    from atk_diffusion import cards
    from atk_diffusion.learn import vitals as LV
    rng = np.random.default_rng(0)
    X, y = [], []
    for i in range(8):
        b, h = float(rng.uniform(9, 20)), float(rng.uniform(60, 100))
        t, H, _ = VE.synth_csi(32.0, 20.0, breath_bpm=b, heart_bpm=h,
                               rng=np.random.default_rng(i))
        _tg, A, _i = C.resample_uniform(t, np.abs(H), 20.0)
        X.append(LV.features(A, 20.0)[:300])
        y.append([b, h])
    card, metrics = LV.train(np.stack(X), np.array(y), tmp_path, epochs=3,
                             hidden=8, trained_on="synthetic test windows")
    assert card.kind == "vitals" and V.RESEARCH_LABEL in card.notes
    assert metrics["windows"] == 8
    model = LV.load(tmp_path)
    out = model.predict(np.stack(X)[:2])
    assert out.shape == (2, 2) and np.all(np.isfinite(out))
    rep = model.predict_report(np.stack(X)[:1])[0]
    assert rep["label"] == V.RESEARCH_LABEL
    (tmp_path / LV.WEIGHTS).write_bytes(b"tampered")
    with pytest.raises(cards.CardRefusal, match="not the file the card describes"):
        LV.load(tmp_path)


# -- the experiment --------------------------------------------------------------------------
def test_vitals_eval_reports_error_and_the_empty_room_rate(rf):
    r = VE.run(rf, conditions=({"breath_bpm": 14.0, "heart_bpm": 72.0},),
               duration_s=60.0, seeds=(1,), empty_seeds=(11,), lstm=False)
    c = r["classical"]
    assert c["breath_mae_bpm"] < 1.0 and r["label"] == V.RESEARCH_LABEL
    assert r["empty_room"]["breath_reported_fraction"] is not None
    assert "hallucination rate" in r["report_md"]
    assert r["pause"]["found"]
    assert len(r["files"]) == 2 and all(rf.verify(f)[0] for f in r["files"])
    assert "esp32csi_50_cf32" in r["files"][0]
