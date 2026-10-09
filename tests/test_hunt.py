# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The self-hunting receiver (plan §4.B6): the goal in words, the bounded
action set with no transmit, the rule and model policies, the log, the
simulator and the first experiment."""

from __future__ import annotations

import json

import numpy as np
import pytest

from atk_diffusion.detect.boxes import Detection
from atk_diffusion.hunt import goal as G
from atk_diffusion.hunt import policy as P
from atk_diffusion.hunt import sim as S

BILL = "anything narrowband and bursty between 400 and 470"
RTL = "rtlsdr_2400000_cu8"


# -- the goal ---------------------------------------------------------------------
def test_bills_sentence_is_read_and_the_reading_is_said():
    g = G.parse_goal(BILL)
    assert (g.f_lo_hz, g.f_hi_hz) == (400e6, 470e6)
    assert g.bw_max_hz == 30_000.0 and g.burstiness == "bursty"
    assert g.problems() == [] and g.source == "rules"
    assert any("read as MHz" in n for n in g.notes)
    assert "between 400 MHz and 470 MHz" in g.describe()


@pytest.mark.parametrize("text,lo,hi", [
    ("between 2.4 and 2.5 GHz", 2.4e9, 2.5e9),
    ("400-470 MHz", 400e6, 470e6),
    ("from 144 MHz to 148 MHz", 144e6, 148e6),
    ("between 7000 and 7300", 7.0e6, 7.3e6),          # kHz: HF, said so
    ("between 162400000 and 162550000", 162.4e6, 162.55e6),
])
def test_units_and_ranges(text, lo, hi):
    g = G.parse_rules(text)
    assert g.f_lo_hz == pytest.approx(lo) and g.f_hi_hz == pytest.approx(hi)


def test_edges_around_and_class_bands():
    g = G.parse_rules("anything above 2.4 GHz")
    assert g.f_lo_hz == 2.4e9 and g.f_hi_hz == G.F_MAX_HZ
    assert any("a wide hunt is a slow one" in n for n in g.notes)
    g = G.parse_rules("around 433.92 MHz")
    assert g.f_lo_hz < 433.92e6 < g.f_hi_hz
    g = G.parse_rules("pagers")
    assert set(g.classes) == {"pocsag", "flex"}
    assert (g.f_lo_hz, g.f_hi_hz) == (929e6, 932e6)
    assert any("US band plan" in n for n in g.notes)


def test_bandwidth_words_families_and_traps():
    g = G.parse_rules("12.5 kHz wide bursts from 400 to 410 MHz at 9600 baud")
    assert (g.bw_min_hz, g.bw_max_hz) == (6_250.0, 25_000.0)
    assert (g.f_lo_hz, g.f_hi_hz) == (400e6, 410e6)       # 9600 baud is not a frequency
    g = G.parse_rules("I am looking for FSK signals narrower than 15 kHz "
                      "between 450 and 460 MHz")
    assert g.families == ("fsk",) and g.bw_max_hz == 15_000.0   # "am" is English
    g = G.parse_rules("continuous wideband between 600 and 700 MHz")
    assert g.burstiness == "continuous" and g.bw_min_hz == 30_000.0
    g = G.parse_rules("DMR or P25 between 450 MHz and 470 MHz")
    assert set(g.classes) == {"dmr", "p25"}
    assert G.parse_rules("anything unknown below 30 MHz").novel


def test_a_goal_without_a_band_says_why_it_cannot_start():
    g = G.parse_rules("anything bursty")
    assert any("no frequency range" in p for p in g.problems())


def test_the_model_reading_is_validated_and_typed_numbers_win():
    asked = []

    def llm_ok(prompt):
        asked.append(prompt)
        return ('thinking: {"f_lo_hz": 1} ... final: {"f_lo_hz": 929000000, '
                '"f_hi_hz": 932000000, "bw_min_hz": null, "bw_max_hz": 25000, '
                '"burstiness": "bursty", "families": ["fsk"], '
                '"classes": ["pocsag"]}')
    g = G.parse_goal("the alphanumeric message transmitters downtown", llm_ok)
    assert g.source == "llm+rules" and g.f_lo_hz == 929e6
    assert "pocsag" in g.classes and g.burstiness == "bursty"
    assert any("model's reading" in n for n in g.notes)
    assert asked[0].rstrip().endswith('say "I don\'t know".')

    def llm_khz(prompt):          # misreads the analyst's MHz as kHz
        return ('{"f_lo_hz": 400000, "f_hi_hz": 470000, "burstiness": "bursty",'
                ' "families": [], "classes": []}')
    g = G.parse_goal(BILL, llm_khz)
    assert (g.f_lo_hz, g.f_hi_hz) == (400e6, 470e6)

    def llm_bad(prompt):
        return '{"f_lo_hz": 5e8, "f_hi_hz": 4e8, "classes": ["death_ray"]}'
    g = G.parse_goal(BILL, llm_bad)
    assert g.source == "rules"
    assert any("was not used" in n for n in g.notes)

    g = G.parse_goal(BILL, lambda p: "I don't know.")
    assert g.source == "rules" and any("did not know" in n for n in g.notes)

    def llm_boom(prompt):
        raise RuntimeError("not loaded")
    assert G.parse_goal(BILL, llm_boom).source == "rules"


def _det(t0, t1, f, bw=12_500.0, cls="", family="fsk", **kw):
    return Detection(t0=t0, t1=t1, f_lo=f - bw / 2, f_hi=f + bw / 2,
                     sources=("energy",), cls=cls, family=family, **kw)


def test_matches_says_true_false_or_ambiguous():
    g = G.parse_goal(BILL)
    ok, why = g.matches(_det(10.2, 10.6, 450e6), (10.0, 11.0))
    assert ok is True and "matches the goal" in why
    ok, why = g.matches(_det(10.0, 11.0, 450e6), (10.0, 11.0))
    assert ok is None and "whole look" in why
    assert g.matches(_det(10.2, 10.6, 450e6, bw=200e3), (10, 11))[0] is False
    assert g.matches(_det(10.2, 10.6, 480e6), (10, 11))[0] is False
    gd = G.parse_goal("DMR between 450 and 470 MHz")
    assert gd.matches(_det(1.2, 1.4, 460e6, cls="p25"), (1, 2))[0] is False
    assert gd.matches(_det(1.2, 1.4, 460e6, cls=""), (1, 2))[0] is None


# -- the bounded action set --------------------------------------------------------
def _limits():
    return P.ReceiverLimits.for_profile(RTL)


def test_there_is_no_transmit_action_and_asking_for_one_is_refused():
    assert not set(P.ACTIONS) & {"transmit", "tx", "send", "jam", "beacon"}
    lim = _limits()
    for word in ("transmit", "tx", "jam", "beacon", "transmit_burst"):
        ok, why = P.validate_action(P.Action(word, {"center_hz": 450e6}), lim)
        assert not ok and "never transmits" in why
    ok, why = P.validate_action({"action": "reboot", "params": {}}, lim)
    assert not ok and "not one of the hunter's actions" in why


def test_the_validator_holds_the_receivers_limits():
    lim = _limits()
    V = P.validate_action
    assert V(P.Action("retune", {"center_hz": 450e6}), lim) == (True, "")
    ok, why = V(P.Action("retune", {"center_hz": 5e9}), lim)
    assert not ok and "outside the receiver's tuning range" in why
    ok, why = V(P.Action("retune", {"center_hz": 450e6, "power": 10}), lim)
    assert not ok and "does not take power" in why
    ok, why = V(P.Action("set_span", {"span_hz": 2.4e6}), lim)
    assert not ok and "never changes the sample rate" in why
    assert V(P.Action("set_span", {"span_hz": 1.8e6}), lim)[0]
    assert not V(P.Action("set_gain", {"gain_db": 80.0}), lim)[0]
    assert not V(P.Action("dwell", {"seconds": 0.0}), lim)[0]
    assert not V(P.Action("dwell", {"seconds": "long"}), lim)[0]
    st = P.HuntState(G.parse_goal(BILL), lim)
    ok, why = V(P.Action("cut", {"detection_id": "nope"}), lim, st)
    assert not ok and "no detection" in why
    assert not V(P.Action("skip", {"cell": 999}), lim, st)[0]
    assert V(P.Action("skip", {"cell": 0, "seconds": 30.0}), lim, st)[0]


def test_limits_come_from_atks_radio_table():
    lim = _limits()
    assert (lim.f_min_hz, lim.f_max_hz) == (24e6, 1_766e6)
    assert lim.usable_span_hz == pytest.approx(1.8e6)
    assert P.ReceiverLimits.for_profile("bladerf1_4000000_ci16").f_min_hz == 300e6
    with pytest.raises(ValueError, match="no tuning range is known"):
        P.ReceiverLimits.for_profile("airspy_2500000_ci16")


# -- the rule policy -----------------------------------------------------------------
def test_rule_policy_covers_the_band_looks_extends_and_marks():
    g = G.parse_goal(BILL)
    lim = _limits()
    st = P.HuntState(g, lim)
    pol = P.RulePolicy()
    a = pol.next_action(st)
    assert a.kind == "set_span" and "39 looks" in a.reason
    st.span_hz = a.params["span_hz"]
    st.last_action = a
    a = pol.next_action(st)
    assert a.kind == "retune" and a.reason.startswith("unexplored")
    st.center_hz = a.params["center_hz"]
    st.last_action = a
    a = pol.next_action(st)
    assert a.kind == "dwell"
    st.last_action = a
    # a signal that filled the look is ambiguous: the policy stays
    st.observe([_det(5.0, 6.0, st.center_hz)], 5.0, 6.0)
    a = pol.next_action(st)
    assert a.kind == "dwell" and a.reason.startswith("ambiguous")
    st.last_action = a
    # it starts inside the next look: a match, marked and cut — once, even
    # when the same burst runs on into the look after
    d = _det(6.5, 7.0, st.center_hz)
    obs = st.observe([d], 6.0, 7.0)
    assert obs["matches"] == 1
    assert [pol.next_action(st).kind for _ in range(2)] == ["mark", "cut"]
    st.observe([_det(7.0, 7.3, st.center_hz)], 7.0, 8.0)   # the same burst
    assert len(st.found) == 1 and st.found[0].t1 == 7.3
    st.observe([_det(9.2, 9.5, st.center_hz)], 9.0, 10.0)  # a new key-up
    assert len(st.found) == 2


def test_out_of_range_cells_are_skipped_once_with_the_reason():
    g = G.parse_goal("between 1700 and 1800 MHz")
    st = P.HuntState(g, _limits())
    assert any(not c.in_range for c in st.cells)
    st.span_hz = st.target_span
    a = P.RulePolicy().next_action(st)
    assert a.kind == "skip" and "outside the receiver's tuning range" in a.reason


def test_a_cell_with_only_other_signals_is_rested():
    g = G.parse_goal(BILL)
    st = P.HuntState(g, _limits())
    pol = P.RulePolicy(quiet_looks_to_skip=2)
    st.span_hz = st.target_span
    st.center_hz = st.cells[0].center_hz
    st.last_action = P.Action("dwell", {"seconds": 1.0})
    for k in range(2):
        st.observe([_det(k + 0.1, k + 0.5, st.center_hz, bw=1e6)], k, k + 1.0)
    a = pol.next_action(st)
    assert a.kind == "skip" and "not about" in a.reason


# -- the model policy ---------------------------------------------------------------
def _ready_state():
    st = P.HuntState(G.parse_goal(BILL), _limits())
    st.span_hz = st.target_span
    return st


def test_llm_policy_uses_valid_choices_and_falls_back_on_anything_else():
    prompts = []

    def good(prompt):
        prompts.append(prompt)
        return '{"action": "retune", "params": {"center_hz": 455000000}, ' \
               '"reason": "nobody has looked here"}'
    pol = P.LlmPolicy(good)
    a = pol.next_action(_ready_state())
    assert a.kind == "retune" and a.policy == "llm"
    assert "no action that transmits" in prompts[0]
    assert prompts[0].rstrip().endswith("the rules will choose.")
    assert "I don't know" in prompts[0]
    for reply, why in (('{"action": "transmit", "params": {}}', "never transmits"),
                       ('{"action": "retune", "params": {"center_hz": 9e9}}',
                        "tuning range"),
                       ("I don't know.", "did not know"),
                       ("no json here", "no JSON")):
        pol = P.LlmPolicy(lambda p, r=reply: r)
        a = pol.next_action(_ready_state())
        assert a.policy == "rules (fallback)" and why in a.reason
        assert pol.counts["fallback"] == 1

    def boom(prompt):
        raise RuntimeError("model unloaded")
    a = P.LlmPolicy(boom).next_action(_ready_state())
    assert a.policy == "rules (fallback)" and "could not be asked" in a.reason


# -- the loop, the log, the simulator -------------------------------------------------
def test_a_hunt_logs_every_retune_with_its_reason(tmp_path):
    g = G.parse_goal(BILL)
    lim = _limits()
    band = S.scripted_band(400e6, 470e6, 240.0, rng=np.random.default_rng(7))
    rx = S.SimReceiver(band, lim, rng=np.random.default_rng(8))
    log = P.HuntLog(tmp_path / "hunt.jsonl")
    res, st = P.run_hunt(g, rx, lim, P.RulePolicy(), log, duration_s=120.0)
    ents = log.entries()
    assert ents[0]["event"] == "start" and ents[-1]["event"] == "stop"
    retunes = log.retunes()
    assert retunes and res.retunes == len(retunes) == rx.retunes
    assert all(e["reason"] for e in retunes)
    assert all(e["action"] in P.ACTIONS for e in ents if "action" in e)
    dwells = [e for e in ents if e.get("action") == "dwell"]
    assert all("seen" in e for e in dwells)
    assert res.duration_s >= 120.0 and "time was up" in res.stopped_because
    assert "Every action and its reason" in res.lines()[-1]
    # a torn last line (a crash mid-write) is skipped, not fatal
    with open(log.path, "a") as f:
        f.write('{"action": "dwe')
    assert len(log.entries()) == len(ents)


def test_a_model_that_only_asks_to_transmit_never_reaches_the_receiver(tmp_path):
    g = G.parse_goal(BILL)
    lim = _limits()
    band = S.scripted_band(400e6, 470e6, 60.0, rng=np.random.default_rng(1))
    rx = S.SimReceiver(band, lim, rng=np.random.default_rng(2))
    pol = P.LlmPolicy(lambda p: '{"action": "transmit", "params": {"power_dbm": 30}}')
    res, _st = P.run_hunt(g, rx, lim, pol, P.HuntLog(tmp_path / "h.jsonl"),
                          duration_s=30.0)
    assert {c[0] for c in rx.calls} <= {"retune", "set_span", "set_gain", "dwell"}
    assert res.fallbacks > 0 and res.refusals == 0


def test_run_hunt_refuses_a_goal_without_a_band_and_a_hunt_without_a_log(tmp_path):
    lim = _limits()
    band = S.scripted_band(400e6, 470e6, 10.0)
    rx = S.SimReceiver(band, lim)
    with pytest.raises(ValueError, match="no frequency range"):
        P.run_hunt(G.parse_goal("anything bursty"), rx, lim,
                   log=P.HuntLog(tmp_path / "x.jsonl"))
    with pytest.raises(ValueError, match="needs a log"):
        P.run_hunt(G.parse_goal(BILL), rx, lim, log=None)


def test_hunt_log_for_run_lives_under_the_profiles_runs(rf):
    log = P.HuntLog.for_run(rf, RTL, run_id="r1")
    log.event("start")
    assert log.path == rf.runs(RTL) / "hunts" / "r1" / "hunt_log.jsonl"
    assert json.loads(log.path.read_text().splitlines()[0])["event"] == "start"


def test_the_simulated_receiver_is_deaf_while_retuning_and_sees_only_its_span():
    lim = _limits()
    sig = [S.SimSignal(0, 0.0, 10.0, 450.0e6, 12_500.0, 30.0),
           S.SimSignal(1, 0.0, 10.0, 460.0e6, 12_500.0, 30.0)]
    band = S.SimBand(400e6, 470e6, 10.0, sig)
    rx = S.SimReceiver(band, lim, rng=np.random.default_rng(0),
                       false_alarms_per_s_per_mhz=0.0, start_hz=450e6)
    dets = rx.dwell(1.0)
    assert [d.measurements["sim_signal"] for d in dets] == [0]
    t = rx.now()
    rx.retune(460e6)
    assert rx.now() == pytest.approx(t + lim.retune_latency_s)
    assert [d.measurements["sim_signal"] for d in rx.dwell(1.0)] == [1]
    truth = band.goal_truth(G.parse_goal(BILL))
    assert truth == []                      # 10 s carriers are not bursts


def test_hunter_versus_fixed_scan_experiment_writes_its_report(rf):
    from atk_diffusion.experiments import hunter_eval
    r = hunter_eval.run(rf, RTL, duration_s=900.0, seeds=(1, 2))
    h, s = r["policies"]["hunter"], r["policies"]["fixed_scan"]
    assert h["truth"] == s["truth"] > 0
    assert h["fraction_found"] > s["fraction_found"]
    assert "| hunter |" in r["report_md"] and "| fixed scan |" in r["report_md"]
    assert len(r["files"]) == 2
    for f in r["files"]:
        assert rf.verify(f)[0]
    assert "not a prediction of the field" in r["report_md"]


def test_a_real_run_is_scored_against_the_cabled_manifest():
    from atk_diffusion.experiments import hunter_eval
    manifest = {"signals": [
        {"start_s": 1.0, "duration_s": 0.5, "f_offset_hz": 100e3,
         "bandwidth_hz": 12.5e3},
        {"start_s": 4.0, "duration_s": 0.3, "f_offset_hz": -200e3,
         "bandwidth_hz": 12.5e3, "bursts_s": [[4.0, 0.1], [4.2, 0.1]]}]}
    F = P.Found
    found = [F("a", 450.1e6, 12e3, 11.0, 11.4, 11.6, 0, "m"),     # burst 1
             F("b", 449.8e6, 12e3, 14.0, 14.1, 14.5, 0, "m"),     # burst 2a
             F("c", 455.0e6, 12e3, 20.0, 20.2, 20.5, 0, "m")]     # nothing there
    sc = hunter_eval.score_manifest(found, manifest, tx_start_s=10.0,
                                    tx_center_hz=450e6, duration_s=60.0)
    assert sc["truth"] == 3 and sc["found"] == 2 and sc["false_marks"] == 1
    assert sc["median_time_to_find_s"] == pytest.approx(0.55)
