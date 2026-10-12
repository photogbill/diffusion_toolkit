# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Sifting (DETECTION_DESIGN §4.4): the deep scan for signals below the
noise floor — the frequency list, the plan and its words, the null the
threshold rests on, finding sub-floor signals blind, refusing on noise,
the ledger that keeps the environment from being recorded, confirmation,
the dwell files and the schedule.

Every search here has its negative control (ATK's rule: a search always
finds something, so a test that only shows it finding the signal shows
nothing). The scenes are small — 250 kS/s, 100 kHz stops, two channel
widths — so the suite runs in seconds a test, but every number the engine
publishes about itself (the null, κ, the sensitivity model) is checked
against the engine here."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import helpers_cyclo as H
from atk_diffusion import sigmf
from atk_diffusion.cyclo import sift as S

FS = 250_000.0
BW = 100_000.0
STOP = 433.0e6
K = 4
#: a QPSK/BPSK whose RRC 0.35 band (1.35 R = 24.1 kHz) fills the 25 kHz
#: channel centred at +25 kHz (channel centres are multiples of 12.5 kHz)
R = FS / 14
FC = 25_000.0
OCC = 1.35 * R


def _cfg(**kw) -> S.SiftConfig:
    base = dict(stops_hz=[STOP], bandwidth_hz=BW, dwell_s=K * 1.0 + S.SETTLE_S,
                coherent_s=1.0, min_width_hz=25_000.0, conjugate=True)
    base.update(kw)
    return S.SiftConfig(**base)


def _plan(**kw) -> S.SiftPlan:
    return S.make_plan(_cfg(**kw), FS, "ci16")


def _scene(kind: str, snr_db: float | None, seed: int, fc: float = FC,
           seconds: float = K * 1.0 + S.SETTLE_S) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(round(seconds * FS))
    x = H.noise(n, rng)
    if kind != "noise":
        x = x + H.inband_amplitude(snr_db, OCC, FS) * H.linmod(
            kind, FS, R, n, rng, fc=fc)
    return x.astype(np.complex64)


def _search(x, plan, pfa=1e-3):
    return S.search(x, FS, STOP, plan, pfa, threads=1)


# ---------------------------------------------------------------------------
# 1. The operator's list
# ---------------------------------------------------------------------------
def test_frequencies_ranges_steps_and_units():
    stops, probs = S.parse_stops("433.92, 446.0-446.2 step 0.1; 868300 kHz, "
                                 "1.2 GHz", 100e3)
    assert not probs
    assert stops == [433.92e6, 446.0e6, 446.1e6, 446.2e6, 868.3e6, 1.2e9]


def test_a_range_without_a_step_is_tiled_edge_to_edge_at_the_bandwidth():
    stops, probs = S.parse_stops("430-431", 250e3)
    assert not probs
    assert stops == [430.125e6, 430.375e6, 430.625e6, 430.875e6]
    # the bands touch and cover the range exactly
    assert stops[0] - 125e3 == pytest.approx(430e6)
    assert stops[-1] + 125e3 == pytest.approx(431e6)


def test_bad_tokens_are_said_and_good_ones_kept():
    stops, probs = S.parse_stops("433.92, banana, 440 parsecs, 400-401/-1",
                                 100e3)
    assert stops == [433.92e6]
    assert len(probs) == 3 and all(isinstance(p, str) for p in probs)


def test_duplicates_dropped_and_the_list_is_bounded():
    stops, _ = S.parse_stops("433.92, 433.92 MHz, 433920 kHz", 100e3)
    assert stops == [433.92e6]
    _s, probs = S.parse_stops("100-200 step 0.001", 100e3)
    assert any(str(S.MAX_STOPS) in p for p in probs)


def test_the_config_says_what_is_wrong_in_words():
    c = _cfg(bandwidth_hz=3e6, dwell_s=0.5, false_per_day=0)
    probs = c.problems(FS, usable_span_hz=0.75 * FS, f_min_hz=300e6,
                       f_max_hz=3.8e9)
    joined = " ".join(probs)
    assert "bandwidth" in joined and "dwell" in joined and "per day" in joined
    low = _cfg(stops_hz=[100e6]).problems(FS, f_min_hz=300e6)
    assert any("below what this radio tunes" in p for p in low)


# ---------------------------------------------------------------------------
# 2. The plan
# ---------------------------------------------------------------------------
def test_levels_are_a_factor_of_four_apart_and_alias_free():
    plan = _plan()
    widths = [lv.width_hz for lv in plan.levels]
    assert widths == [100e3, 25e3]
    for lv in plan.levels:
        assert lv.fs_c >= S.RATE_FACTOR * lv.width_hz * 0.999
        assert lv.cells_c > 0
        # half-overlapping channels cover the stop
        assert min(lv.offsets_hz) - lv.width_hz / 2 <= -BW / 2 + lv.spacing_hz
        assert max(lv.offsets_hz) + lv.width_hz / 2 >= BW / 2 - lv.spacing_hz


def test_a_bandwidth_close_to_the_sample_rate_drops_only_the_conjugate_search():
    plan = S.make_plan(_cfg(bandwidth_hz=0.75 * FS, min_width_hz=40_000.0),
                       FS, "ci16")
    assert plan.levels[0].cells_c == 0
    assert any("conjugately" in n for n in plan.notes)


def test_false_recordings_a_day_split_into_two_looks():
    p1, p2 = S.false_alarm_split(1.0, 2880.0, True)
    assert 2880.0 * p1 * p2 == pytest.approx(1.0, rel=1e-6)
    assert p1 <= 0.02
    q1, q2 = S.false_alarm_split(1.0, 2880.0, False)
    assert q2 == 1.0 and 2880.0 * q1 == pytest.approx(1.0)


def test_the_plan_is_said_before_the_run():
    words = _plan().words()
    for must in ("one pass every", "coherent segment", "Sensitivity",
                 "false recording", "written to disk", "memory"):
        assert must in words, must


# ---------------------------------------------------------------------------
# 3. The null — what the threshold rests on
# ---------------------------------------------------------------------------
def test_the_summed_statistic_is_gamma_k_under_the_null():
    """Summed over K segments, each α bin's statistic over its local median
    (×the Gamma median) is Gamma(K, 1): mean K, variance K, and the tail the
    threshold is computed from. Measured on noise through the real
    channelizer — every number the false-alarm rate claims rests on this."""
    from scipy.special import gammainccinv
    from atk_diffusion.cyclo.proposer import channelise
    plan = _plan()
    lv = plan.levels[1]
    x = _scene("noise", None, 7)
    acc = None
    seg = int(FS)
    for k in range(K):
        y0 = S.stage0(x[k * seg:(k + 1) * seg], 1)
        Y, *_ = channelise(y0, FS, lv.spacing_hz, lv.width_hz, lv.decim,
                           usable=0.5 * BW / FS + 1e-9)
        s = S._segment_stats(Y[:, :lv.seg_len], S.LAGS_NC, False, lv.nfft)
        acc = s if acc is None else acc + s
    G = S._gamma_ratio(acc, K)
    lo = int(S.ALPHA_MIN_BINS * lv.nfft / lv.seg_len)
    g = G[2][:, lo:].ravel()
    assert g.mean() == pytest.approx(K, rel=0.02)
    assert g.var() == pytest.approx(K, rel=0.08)
    for p in (1e-2, 1e-3):
        assert np.mean(g > gammainccinv(K, p)) == pytest.approx(p, rel=0.35)


def test_noise_alone_finds_nothing():
    plan = _plan()
    for seed in range(3):
        r = _search(_scene("noise", None, seed), plan, pfa=1e-2)
        assert r["features"] == [], [f.words() for f in r["features"]]


# ---------------------------------------------------------------------------
# 4. Finding signals below the floor, blind
# ---------------------------------------------------------------------------
def test_a_bpsk_carrier_is_found_under_the_floor_at_the_right_frequency():
    plan = _plan()
    r = _search(_scene("bpsk", -10.0, 3), plan)
    conj = [f for f in r["features"] if f.conj]
    assert conj, [f.words() for f in r["features"]]
    top = max(conj, key=lambda f: f.margin)
    assert top.line_hz == pytest.approx(STOP + FC, abs=3 * top.dalpha_hz)
    assert not top.energy_visible
    led = S.Ledger(None, class_rates=[])
    assert led.classify(top, STOP).klass == "new"


def test_a_proper_qpsk_has_a_symbol_line_and_no_carrier_line():
    """The discriminator, blind: QPSK's symbol rate is found and nothing
    conjugate is — a carrier line here would be invented."""
    plan = _plan()
    r = _search(_scene("qpsk", -3.0, 4), plan)
    nc = [f for f in r["features"] if not f.conj]
    assert any(f.alpha_hz == pytest.approx(R, abs=3 * f.dalpha_hz)
               for f in nc), [f.words() for f in r["features"]]
    assert not [f for f in r["features"] if f.conj]


def test_a_signal_above_the_floor_is_visible_not_new():
    plan = _plan()
    r = _search(_scene("qpsk", 10.0, 5), plan)
    assert r["features"]
    led = S.Ledger(None, class_rates=[])
    for f in r["features"]:
        assert f.energy_visible, f.words()
        assert led.classify(f, STOP).klass == "visible"


def test_the_sensitivity_model_matches_the_engine():
    """The plan's sentence is a measured κ scaled by the algebra; here the
    algebra meets the engine. 2.5 dB above the predicted SNR the QPSK line
    is found in at least 3 of 4 seeds; 3 dB below, in at most 1."""
    plan = _plan()
    pred = next(s["symbol_db"] for s in plan.sensitivity
                if s["width_hz"] == 25e3)

    def found(snr, seed):
        r = _search(_scene("qpsk", snr, 100 + seed), plan, pfa=plan.pfa_first)
        return any(not f.conj and abs(f.alpha_hz - R) <= 3 * f.dalpha_hz
                   for f in r["features"])
    above = sum(found(pred + 2.5, s) for s in range(4))
    below = sum(found(pred - 3.0, s) for s in range(4))
    assert above >= 3, (pred, above)
    assert below <= 1, (pred, below)


# ---------------------------------------------------------------------------
# 5. The ledger
# ---------------------------------------------------------------------------
def _f(alpha=1000.0, centre=STOP + 10e3, width=25e3, conj=False, **kw):
    f = S.Feature(conj=conj, alpha_hz=alpha, f_lo_hz=centre - width / 2,
                  f_hi_hz=centre + width / 2, width_hz=width, stop_hz=STOP,
                  statistic=50.0, threshold=30.0, p_family=1e-5, lag=1,
                  segments=K, dalpha_hz=1.0,
                  line_hz=(centre + alpha / 2) if conj else alpha)
    for k, v in kw.items():
        setattr(f, k, v)
    return f


def test_the_receivers_own_features_are_learned_and_named(tmp_path):
    led = S.Ledger(tmp_path / "rx.json", class_rates=[])
    led.learn_receiver([_f(alpha=4321.0)], STOP)
    led.save_receiver("test")
    again = S.Ledger(tmp_path / "rx.json", class_rates=[])
    # the radio's own follow the radio: same offset from the tuned centre,
    # another stop
    g = _f(alpha=4321.4, centre=STOP + 50e6 + 10e3)
    assert again.classify(g, STOP + 50e6).klass == "receiver"
    assert "calibrated" in again.receiver_words()
    again.forget_receiver()
    assert not (tmp_path / "rx.json").exists()


def test_the_same_alpha_at_three_separate_frequencies_is_local():
    led = S.Ledger(None, class_rates=[], apart_hz=BW)
    k = [led.classify(_f(alpha=100e3 / 3, centre=STOP + d), STOP + d).klass
         for d in (0.0, 200e3, 400e3)]
    assert k == ["new", "new", "local"]
    # the same place seen three times is one sighting, not three
    led2 = S.Ledger(None, class_rates=[])
    k2 = [led2.classify(_f(alpha=7777.0), STOP).klass for _ in range(3)]
    assert k2 == ["new", "new", "new"]


def test_background_explained_and_dc():
    led = S.Ledger(None, class_rates=[("p25", 4800.0)])
    led.learn_background([_f(alpha=2222.0)], STOP)
    assert led.classify(_f(alpha=2222.3), STOP).klass == "background"
    e = led.classify(_f(alpha=9600.2), STOP)            # 2 × 4800
    assert e.klass == "explained" and e.explained_by == ["p25"]
    dc = _f(alpha=0.0, centre=STOP, conj=True)
    assert led.classify(dc, STOP).klass == "dc"


def test_a_conjugate_line_is_matched_by_its_carrier_not_its_alpha():
    """The same carrier seen in two channels has two different α (α is
    twice the carrier's offset from the CHANNEL's centre)."""
    a = _f(alpha=2000.0, centre=STOP + 20e3, conj=True)       # carrier +21 k
    b = _f(alpha=-10000.0, centre=STOP + 26e3, conj=True)     # carrier +21 k
    assert S.same_line(a, b)
    assert S.confirms(a, [b]) is b


# ---------------------------------------------------------------------------
# 6. Deciding
# ---------------------------------------------------------------------------
def _res(features, kind="dwell"):
    return S.DwellResult(stop_hz=STOP, kind=kind, features=features,
                         near_miss=None, floor_db_per_hz=-100.0,
                         clipped_fraction=0.0, seconds_analysed=4.0,
                         analysis_s=1.0, segments=K, dwell_id="d1")


def test_new_is_held_then_kept_only_if_seen_again():
    cfg = _cfg()
    first = [_f(alpha=1234.0)]
    d = S.decide("dwell", _res(first), cfg, baseline=False)
    assert d["action"] == "hold"
    yes = S.decide("confirm", _res([_f(alpha=1234.3)], "confirm"), cfg, False,
                   held_features=first)
    assert yes["action"] == "keep" and yes["features"][0].confirmed
    no = S.decide("confirm", _res([], "confirm"), cfg, False,
                  held_features=[_f(alpha=1234.0)])
    assert no["action"] == "discard"


def test_baseline_calibration_and_the_environment_are_never_recorded():
    cfg = _cfg()
    for kind, base in (("dwell", True), ("calibrate", False)):
        assert S.decide(kind, _res([_f()]), cfg, base)["action"] == "discard"
    f = _f()
    f.klass = "local"
    assert S.decide("dwell", _res([f]), cfg, False)["action"] == "discard"
    f.klass = "visible"
    assert S.decide("dwell", _res([f]), _cfg(keep_visible=True), False
                    )["action"] == "hold"


def test_without_confirmation_new_is_kept_at_once():
    d = S.decide("dwell", _res([_f()]), _cfg(confirm=False), False)
    assert d["action"] == "keep"


# ---------------------------------------------------------------------------
# 7. The dwell files
# ---------------------------------------------------------------------------
def _store(tmp_path, budget=1 << 30, dt="ci16q11"):
    return S.DwellStore(tmp_path / "work", tmp_path / "keep", dt, FS,
                        "bladerf1_250000_ci16", budget, hw="bladeRF 1.0")


def test_a_kept_dwell_is_a_valid_sigmf_pair_with_its_features(tmp_path):
    st = _store(tmp_path)
    did, fh = st.open(STOP, "dwell")
    raw = H.noise(1000, np.random.default_rng(0)).astype(np.complex64)
    from atk_diffusion.dsp import iq as _iq
    data = _iq.from_complex(raw * 0.1, "ci16q11")
    fh.write(data)
    fh.close()
    st.closed(did, len(data))
    f = _f(alpha=1234.0)
    f.klass, f.confirmed = "new", True
    data_path, meta_path = st.keep(did, [f], "test keep")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert sigmf.validate(meta) == []
    assert meta["global"]["atk:datatype"] == "ci16q11"
    assert meta["captures"][0]["core:frequency"] == STOP
    a = meta["annotations"][0]
    assert a["atk:proposer"] == "sift" and a["atk:source"] == "confirmed"
    assert a["atk:symbol_rate"] == pytest.approx(1234.0)
    assert data_path.stat().st_size == len(data)
    assert not list((tmp_path / "work").glob("*.part"))


def test_discard_deletes_and_the_budget_refuses_to_keep(tmp_path):
    st = _store(tmp_path, budget=100)
    did, fh = st.open(STOP, "dwell")
    fh.write(b"\0" * 400)
    fh.close()
    p = st.closed(did, 400)
    assert st.keep(did, [_f()], "too big") is None
    assert p.exists()                       # refused, not truncated
    st.discard(did, "test")
    assert not p.exists()


def test_a_dwell_file_reads_back_by_segment(tmp_path):
    from atk_diffusion.dsp import iq as _iq
    x = (np.arange(500) + 1j * np.arange(500)).astype(np.complex64) / 1000
    p = tmp_path / "d.raw"
    p.write_bytes(_iq.from_complex(x, "ci16q11"))
    rd = S.DwellReader(p, "ci16q11")
    assert rd.samples == 500
    np.testing.assert_allclose(rd(100, 10), x[100:110], atol=1e-3)


def test_one_dwell_end_to_end_from_its_file(tmp_path):
    from atk_diffusion.dsp import iq as _iq
    x = _scene("bpsk", -6.0, 11)
    p = tmp_path / "dwell.part"
    p.write_bytes(_iq.from_complex(x * 0.05, "ci16"))
    plan = _plan()
    led = S.Ledger(None, class_rates=[])
    res = S.analyse_dwell(p, "ci16", FS, STOP, plan, "dwell", led)
    assert any(f.conj and f.klass == "new" for f in res.features)
    assert res.clipped_fraction == 0.0 and not res.warnings
    # a clipped front end is said
    p.write_bytes(_iq.from_complex(x * 2.0, "ci16"))   # past full scale
    res2 = S.analyse_dwell(p, "ci16", FS, STOP, plan, "dwell",
                           S.Ledger(None, class_rates=[]))
    assert any("full scale" in w for w in res2.warnings)


# ---------------------------------------------------------------------------
# 8. The schedule and the log
# ---------------------------------------------------------------------------
def test_the_schedule_confirms_first_relooks_once_and_waits_for_analysis():
    cfg = _cfg(stops_hz=[1e8, 2e8, 3e8], baseline_passes=0)
    sch = S.SiftSchedule(cfg)
    s1 = sch.next_step()
    assert (s1.kind, s1.stop_hz) == ("dwell", 1e8)
    sch.dwell_started()
    s2 = sch.next_step()
    sch.dwell_started()
    assert sch.next_step().kind == "wait"           # two waiting for analysis
    new = _f()
    new.klass = "new"
    said = sch.analysed(s1, _res([new]))
    assert "confirmation queued" in said
    c = sch.next_step()
    assert (c.kind, c.stop_hz, c.held_dwell) == ("confirm", 1e8, "d1")
    sch.dwell_started()
    miss = _res([])
    miss.near_miss = {"alpha_hz": 5.0, "p_family": 0.03}
    assert "relook queued" in sch.analysed(s2, miss)
    sch.analysed(c, _res([], "confirm"))
    r = sch.next_step()
    assert (r.kind, r.stop_hz) == ("relook", 2e8)
    sch.dwell_started()
    sch.analysed(r, miss)                           # a relook never relooks
    assert sch.next_step().stop_hz == 3e8


def test_baseline_passes_are_marked_and_calibration_ends():
    sch = S.SiftSchedule(_cfg(stops_hz=[1e8, 2e8], baseline_passes=1))
    a, b = sch.next_step(), sch.next_step()
    assert a.baseline and b.baseline
    c = sch.next_step()
    assert not c.baseline and sch.passes == 1
    cal = S.SiftSchedule(_cfg(stops_hz=[1e8, 2e8, 3e8, 4e8, 5e8]),
                         calibrate=True)
    kinds = [cal.next_step() for _ in range(4)]
    assert [k.kind for k in kinds] == ["calibrate"] * 3 + ["done"]
    assert [k.stop_hz for k in kinds[:3]] == [1e8, 3e8, 5e8]


def test_the_run_log_is_json_lines(tmp_path):
    log = S.SiftLog.for_run(tmp_path)
    log.put("retune", stop_hz=STOP, why="pass 1")
    log.put("analysed", features=[_f()])
    rows = log.entries()
    assert [r["event"] for r in rows] == ["retune", "analysed"]
    assert rows[1]["features"][0]["alpha_hz"] == 1000.0


# ---------------------------------------------------------------------------
# 9. What the independent review found (2026-10-12) — each pinned
# ---------------------------------------------------------------------------
def _held_carrier(plan):
    return _f(alpha=2 * 12.5e3, centre=STOP + 25e3, conj=True,
              dalpha_hz=plan.levels[-1].dalpha_hz, line_hz=STOP + FC)


def test_a_confirmation_looks_again_only_where_the_first_look_found_it():
    """The second look asks "is THAT line still here?": its threshold comes
    from the cells in the held lines' windows, not from the whole search.
    With the whole search's count it had to clear the first look's own
    threshold, and weak real finds were thrown away. Seed 44 at −16 dB
    puts the carrier between the two thresholds."""
    plan = _plan()
    held = _held_carrier(plan)
    x = _scene("bpsk", -16.0, 44)
    plain = S.search(x, FS, STOP, plan, plan.pfa_first, threads=1)
    assert S.confirms(held, plain["features"]) is None
    win = S.focus_windows([held])
    again = S.search(x, FS, STOP, plan, plan.pfa_first, threads=1,
                     focus=win, pfa_focus=plan.pfa_confirm)
    g = S.confirms(held, again["features"])
    assert g is not None and g.threshold == pytest.approx(
        again["threshold_focus"])
    assert again["threshold_focus"] < again["threshold"]
    # thousands of cells, not millions
    assert again["trials_focus"] == S.focus_trials(plan, win, STOP)
    assert again["trials_focus"] < plan.trials / 100


def test_the_focused_threshold_holds_its_false_alarm_rate_on_noise():
    """Negative control for the window: noise 'confirms' a held line no
    more often than the Pfa asked of the window (Bonferroni over its cells
    is conservative, so fewer)."""
    cfg = _cfg(dwell_s=2 + S.SETTLE_S)
    plan = S.make_plan(cfg, FS, "ci16")
    held = [_held_carrier(plan), _f(alpha=17857.0, centre=STOP - 25e3,
                                    dalpha_hz=plan.levels[-1].dalpha_hz)]
    win = S.focus_windows(held)
    hits, n = 0, 30
    for i in range(n):
        x = H.noise(int(round((2 + S.SETTLE_S) * FS)),
                    np.random.default_rng(1000 + i)).astype(np.complex64)
        r = S.search(x, FS, STOP, plan, 1e-9, threads=1, focus=win,
                     pfa_focus=0.3)
        hits += any(S.confirms(h, r["features"]) for h in held)
    assert hits / n <= 0.45, hits          # 0.3 asked; 14/60 measured


def test_a_carrier_that_drifted_between_looks_still_confirms():
    """Both oscillators drift between looks; an α bin is ~1 Hz, a few parts
    per BILLION at 433 MHz. The tolerance is CONFIRM_DRIFT_PPM of the
    carrier — and no wider."""
    first = _f(alpha=50e3, centre=STOP, conj=True)          # carrier +25 k
    tol = S.CONFIRM_DRIFT_PPM * 1e-6 * first.line_hz
    for moved, same in ((2.0, True), (50.0, True), (0.9 * tol, True),
                        (1.2 * tol, False), (1000.0, False)):
        g = _f(alpha=50e3 + 2 * moved, centre=STOP, conj=True)
        assert (S.confirms(first, [g]) is g) == same, moved


def test_receiver_and_background_carriers_survive_hours_of_drift():
    led = S.Ledger(None, class_rates=[])
    led.learn_background([_f(alpha=50e3, centre=STOP, conj=True)], STOP)
    # 500 Hz is 1.15 ppm at 433 MHz — hours of TCXO drift; 2 kHz (4.6 ppm)
    # is another carrier
    near = _f(alpha=50e3 + 2 * 500.0, centre=STOP, conj=True)
    far = _f(alpha=50e3 + 2 * 2000.0, centre=STOP, conj=True)
    assert led.classify(near, STOP).klass == "background"
    assert led.classify(far, STOP).klass == "new"
    rx = S.Ledger(None, class_rates=[])
    rx.learn_receiver([_f(alpha=50e3, centre=STOP, conj=True)], STOP)
    moved = _f(alpha=50e3 + 400.0, centre=STOP, conj=True)    # 200 Hz
    assert rx.classify(moved, STOP).klass == "receiver"


def _qpsk_pair(with_a: bool, with_b: bool, seed: int) -> np.ndarray:
    """A: a 10 kHz QPSK at −35 kHz. B: a separate 50 kHz QPSK (5 × A's
    rate) at +10 kHz, wide enough to be seen only in the 100 kHz channel."""
    rng = np.random.default_rng(seed)
    n = int(round((K + S.SETTLE_S) * FS))
    x = H.noise(n, rng)
    if with_a:
        x = x + H.inband_amplitude(0.0, 1.35 * FS / 25, FS) * H.linmod(
            "qpsk", FS, FS / 25, n, rng, fc=-35e3)
    if with_b:
        x = x + H.inband_amplitude(-3.0, 1.35 * FS / 5, FS) * H.linmod(
            "qpsk", FS, FS / 5, n, rng, fc=10e3)
    return x.astype(np.complex64)


def test_another_emitters_rate_is_never_folded_in_as_a_harmonic():
    """B's rate is 5 × A's, and B's wide place overlaps A's: once folded
    into A as a "harmonic", B was hidden as background with A."""
    plan = _plan(conjugate=False)
    both = S.search(_qpsk_pair(True, True, 8), FS, STOP, plan, 1e-3,
                    threads=1)["features"]
    rates = sorted(round(f.alpha_hz) for f in both)
    assert rates == [10000, 50000], [(f.alpha_hz, f.harmonics_hz)
                                     for f in both]
    assert all(not f.harmonics_hz for f in both)
    led = S.Ledger(None, class_rates=[])
    base = S.search(_qpsk_pair(True, False, 7), FS, STOP, plan, 1e-3,
                    threads=1)["features"]
    led.learn_background(base, STOP)
    for f in both:
        led.classify(f, STOP)
    k = {round(f.alpha_hz): f.klass for f in both}
    assert k == {10000: "background", 50000: "new"}
    assert S.decide("dwell", _res(both), _cfg(), False)["action"] == "hold"


def _gated(n, width_hz, gate_hz, fc, rng):
    from scipy.signal import firwin, lfilter
    h = firwin(255, width_hz / 2, fs=FS)
    s = lfilter(h, 1, rng.standard_normal(n) + 1j * rng.standard_normal(n))
    s = s * (((np.arange(n) / FS * gate_hz) % 1.0) < 0.5)
    s = s / np.sqrt(np.mean(np.abs(s) ** 2))
    return s * np.exp(2j * np.pi * fc * np.arange(n) / FS)


def test_one_wideband_emitter_is_one_feature_and_never_local_by_itself():
    """A 70 kHz noise-like emitter gated at 500 Hz fills many narrow
    channels of ONE stop: it was 11 features, 9 of them "local", and then
    every 500 Hz emitter in the run was local too."""
    plan = _plan(conjugate=False, min_width_hz=6250.0)
    n = int(round((K + S.SETTLE_S) * FS))
    rng = np.random.default_rng(3)
    x = H.noise(n, rng) + np.sqrt(10 ** (-0.6) * 70e3 / FS) * _gated(
        n, 70e3, 500.0, 0.0, rng)
    feats = S.search(x.astype(np.complex64), FS, STOP, plan, 1e-3,
                     threads=1)["features"]
    at_500 = [f for f in feats if abs(f.alpha_hz - 500.0) < 5]
    assert len(at_500) == 1, [(f.f_lo_hz - STOP, f.f_hi_hz - STOP)
                              for f in at_500]
    assert at_500[0].f_hi_hz - at_500[0].f_lo_hz >= 50e3   # its whole band
    led = S.Ledger(None, class_rates=[], apart_hz=BW)
    assert all(led.classify(f, STOP).klass != "local" for f in feats)
    # a narrow 500 Hz emitter at another stop is new, not local
    rng = np.random.default_rng(9)
    y = H.noise(n, rng) + np.sqrt(10 ** (-0.6) * 4e3 / FS) * _gated(
        n, 4e3, 500.0, 20e3, rng)
    other = S.search(y.astype(np.complex64), FS, 446e6, plan, 1e-3,
                     threads=1)["features"]
    near = [f for f in other if abs(f.alpha_hz - 500.0) < 5]
    assert near and led.classify(near[0], 446e6).klass == "new"


def test_local_needs_three_stops_further_apart_than_the_bandwidth():
    # three places in one stop: one emitter's pieces, never local — even
    # where the ledger was not told the bandwidth
    for apart in (BW, 0.0):
        one = S.Ledger(None, class_rates=[], apart_hz=apart)
        k = [one.classify(_f(alpha=3333.0, centre=STOP + d), STOP).klass
             for d in (-40e3, 0.0, 40e3)]
        assert "local" not in k, apart
    led = S.Ledger(None, class_rates=[], apart_hz=BW)
    led.classify(_f(alpha=3333.0, centre=STOP), STOP)
    # two more stops, closer than the bandwidth to each other: still not
    led.classify(_f(alpha=3333.0, centre=STOP + 1e6), STOP + 1e6)
    k2 = led.classify(_f(alpha=3333.0, centre=STOP + 1.05e6), STOP + 1.05e6)
    assert k2.klass != "local"
    # a third stop far enough from both: local
    k3 = led.classify(_f(alpha=3333.0, centre=STOP + 3e6), STOP + 3e6)
    assert k3.klass == "local"


def test_every_recorded_segment_is_searched():
    """The recorder skips the settling time BEFORE it writes; the plan used
    to take it off again, and one segment a dwell was never searched."""
    plan = S.make_plan(S.SiftConfig(stops_hz=[STOP], dwell_s=30.0,
                                    coherent_s=2.0), 2.4e6, "ci16")
    assert plan.segments == 15
    assert not _cfg(dwell_s=1.0, coherent_s=1.0).problems(FS)
    assert _cfg(dwell_s=0.5, coherent_s=1.0).problems(FS)


def test_the_plan_counts_the_memory_of_every_width_searched_at_once():
    """The search runs `level_threads` widths at once; the plan once stated
    the largest single width. Now an upper bound, checked here."""
    import tracemalloc
    cfg = _cfg(dwell_s=2.0, coherent_s=1.0)
    plan = S.make_plan(cfg, FS, "ci16")
    x = _scene("noise", None, 1, seconds=2.0 + S.SETTLE_S)
    n_lv = len(plan.levels)
    for th in (1, n_lv):
        est = S.analysis_memory(plan.levels, FS, 1.0, True, threads=th)
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        S.search(x, FS, STOP, plan, 1e-3, threads=th,
                 floor_per_hz=2 / FS, floor_spread_db=0.0)
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        assert peak <= est, (th, peak, est)
    assert S.analysis_memory(plan.levels, FS, 1.0, True, threads=n_lv) > \
        S.analysis_memory(plan.levels, FS, 1.0, True, threads=1)


def test_the_disk_budget_counts_every_earlier_sift(tmp_path):
    """Per run, a sift restarted daily filled the drive a budget at a
    time. A first look and its confirmation go together or not at all."""
    earlier = tmp_path / "keep_root" / "20261001-000000-abcdef"
    earlier.mkdir(parents=True)
    (earlier / "00001_433.000000MHz_dwell.sigmf-data").write_bytes(b"\0" * 600)
    st = S.DwellStore(tmp_path / "work", tmp_path / "keep_root" / "now",
                      "ci16", FS, "p", 1000)
    assert st.kept_bytes == 600
    a, fa = st.open(STOP, "dwell")
    fa.write(b"\0" * 300)
    fa.close()
    st.closed(a, 300)
    b, fb = st.open(STOP, "confirm")
    fb.write(b"\0" * 300)
    fb.close()
    st.closed(b, 300)
    assert st.can_keep(300) and not st.can_keep_all([a, b])
