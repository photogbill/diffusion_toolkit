# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Pulse-train completion and deinterleaving (plan §4.D3)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from atk_diffusion import provenance
from atk_diffusion.repair import pulses as PU


def _train(rng, pri=1e-3, n=150, freq=0.0, width=5e-6, levels=None, t0=0.0,
           drop=0.15, jitter=0.0, tag=0, on=None):
    """ATK-shaped PDWs of one emitter; returns (received, dropped toas)."""
    seq = list(levels or [pri])
    t, k, rec, dropped = t0, 0, [], []
    for _ in range(n):
        tt = t + (rng.uniform(-jitter, jitter) * pri if jitter else 0.0)
        active = on is None or any(a <= t < b for a, b in on)
        if active:
            if rng.random() < drop:
                dropped.append(tt)
            else:
                rec.append({"start": 0, "end": 10, "toa_s": tt,
                            "width_s": width * (1 + 0.02 * rng.normal()),
                            "amplitude": 10.0, "amplitude_db": 20.0,
                            "freq_hz": freq + 2000 * rng.normal(),
                            "chirp_hz": 0.0, "true": tag})
        t += seq[k % len(seq)]
        k += 1
    return rec, np.array(dropped)


def test_pdws_in_atks_shape_or_bare_toas():
    rows = [{"toa_s": 0.002, "width_s": 1e-6, "freq_hz": 5.0, "start": 7,
             "end": 9, "amplitude": 1.0, "amplitude_db": 3.0, "chirp_hz": 0.0,
             "extra": "kept"}, {"toa_s": 0.001}]
    P = PU.as_pdws(rows)
    assert [p["toa_s"] for p in P] == [0.001, 0.002]
    assert P[1]["extra"] == "kept" and P[1]["start"] == 7
    assert math.isnan(P[0]["width_s"]) and P[0]["start"] == -1
    assert PU.as_pdws(np.array([0.3, 0.1]))[0]["toa_s"] == 0.1
    with pytest.raises(ValueError):
        PU.as_pdws([{"width_s": 1.0}])
    assert set(PU.PDW_FIELDS) <= set(P[1])


def test_sdif_cdif_and_the_pri_transform(rng):
    t = PU.toas(_train(rng, pri=1e-3, drop=0.1)[0])
    sd = PU.sdif(t)
    first = next(L for L in sd if L.peaks)
    assert first.level == 1 and abs(first.peaks[0] - 1e-3) < 2e-5
    thr = PU.mp_threshold(np.array([1e-3]), 100, 2, 0.1, x=0.2, k=0.3)
    assert math.isclose(thr[0], 0.2 * 98 * math.exp(-1e-3 / 0.03))
    cd = PU.cdif(t)
    assert any(abs(p - 1e-3) < 2e-5 for L in cd for p in L.peaks)
    taus, mag = PU.pri_transform(t, taus=np.array([1e-3, 2e-3]))
    assert mag[0] > 0.8 and mag[1] < 0.5        # 2×PRI is suppressed
    assert sd[0].to_json()["level"] == 1


def test_periodic_misses_tell_a_stagger_from_fading(rng):
    # a 1/2 ms stagger read on a 1 ms grid: slot 2 of every 3 is empty
    stag = [s for s in range(60) if s % 3 != 2]
    assert PU.periodic_misses(stag)[0]
    fading = [s for s in range(60) if rng.random() > 0.2]
    assert not PU.periodic_misses(fading)[0]


def test_completion_flags_every_inferred_pulse(rng):
    rec, dropped = _train(rng, pri=1e-3, n=150, drop=0.2)
    D = PU.deinterleave(rec)
    assert len(D.emitters) == 1
    e = D.emitters[0]
    assert e.kind == "constant" and abs(e.pri_s - 1e-3) < 1e-6
    assert len(e.received) == len(rec)
    assert len(e.inferred) > 0.8 * dropped.size
    for p in e.inferred:
        assert p["inferred"] is True and p["tier"] == "inferred"
        assert p["method"] == "pri_fill" and provenance.tier_for(p["method"]) == "inferred"
        assert p["start"] == -1 and p["end"] == -1
        assert math.isnan(p["amplitude"]) and math.isnan(p["amplitude_db"])
        assert p["sigma_toa_s"] > 0 and p["reason"] == "dropout"
        assert np.min(np.abs(dropped - p["toa_s"])) < 2e-5     # where one was sent
        # ATK's intra_pulse_summary slices x[start:end] and skips b <= a
        a, b = int(p["start"]), int(p["end"]) or 1
        assert b <= a
    for i in e.received:
        assert D.pdws[i]["tier"] == "measured" and D.pdws[i]["inferred"] is False
    train = D.train(0)
    assert len(train) == len(rec) + len(e.inferred)
    assert len(PU.received_only(train)) == len(rec)
    assert any("INFERRED" in ln for ln in D.lines())


def test_a_stagger_is_not_completed_into_a_constant_train(rng):
    """The trap: 1 ms / 2 ms is a 1 ms train missing every third pulse."""
    rec, dropped = _train(rng, levels=[1e-3, 2e-3], n=180, drop=0.15)
    D = PU.deinterleave(rec)
    assert len(D.emitters) == 1
    e = D.emitters[0]
    assert e.kind == "staggered (2-level)"
    assert abs(e.pri_s - 3e-3) < 1e-5
    assert sorted(round(v * 1e6) for v in e.levels_s) == [1000, 2000]
    assert len(e.inferred) <= dropped.size          # not one per frame


def test_three_level_stagger_and_jitter(rng):
    rec, _ = _train(rng, levels=[0.9e-3, 1.3e-3, 1.1e-3], n=210, drop=0.15)
    e = PU.deinterleave(rec).emitters
    assert len(e) == 1 and e[0].kind == "staggered (3-level)"
    assert [round(v * 1e6) for v in e[0].levels_s] in (
        [900, 1300, 1100], [1300, 1100, 900], [1100, 900, 1300])
    rec, _ = _train(np.random.default_rng(5), pri=1e-3, n=200, drop=0.1, jitter=0.06)
    e = PU.deinterleave(rec).emitters
    assert len(e) == 1 and e[0].kind == "jittered"
    assert 2.0 < e[0].jitter_pct < 5.0             # ±6 % uniform: σ 3.5 %


def test_two_emitters_on_one_carrier_are_separated_by_pri(rng):
    a, _ = _train(rng, pri=1.0e-3, n=200, tag=0)
    b, _ = _train(rng, pri=1.37e-3, n=150, t0=0.00031, tag=1)
    D = PU.deinterleave(a + b)
    assert len(D.emitters) == 2
    for e in D.emitters:
        tags = [D.pdws[i]["true"] for i in e.received]
        assert max(np.bincount(tags)) / len(tags) > 0.97


def test_noise_alone_invents_nothing(rng):
    noise = [{"toa_s": float(t), "width_s": float(rng.uniform(1e-6, 2e-5)),
              "freq_hz": float(rng.uniform(-1e6, 1e6))}
             for t in np.sort(rng.uniform(0, 0.3, 400))]
    D = PU.deinterleave(noise)
    assert D.emitters == [] and len(D.unassigned) == 400
    assert D.completed() == D.pdws


def test_nothing_is_inferred_between_scan_bursts(rng):
    rec, _ = _train(rng, pri=1e-3, n=600, drop=0.1,
                    on=[(0.0, 0.05), (0.2, 0.25), (0.4, 0.45)])
    D = PU.deinterleave(rec)
    assert len(D.emitters) == 1
    e = D.emitters[0]
    assert len(e.bursts) == 3
    for p in e.inferred:
        assert any(a <= p["toa_s"] <= b for a, b in e.bursts)
    assert any("separate stretches" in n for n in e.notes)


def test_collision_is_named(rng):
    a, _ = _train(rng, pri=1e-3, n=120, drop=0.0, freq=0.0)
    # remove one of A's pulses and put B's pulse on top of where it was due
    victim = a.pop(60)
    b, _ = _train(rng, pri=1.7e-3, n=70, drop=0.0, freq=400e3, width=20e-6,
                  t0=victim["toa_s"] - 1.7e-3 * 35)
    D = PU.deinterleave(a + b)
    ea = [e for e in D.emitters if abs(e.pri_s - 1e-3) < 1e-5][0]
    hit = [p for p in ea.inferred if abs(p["toa_s"] - victim["toa_s"]) < 2e-5]
    assert hit and hit[0]["reason"] == "collision" and hit[0]["masked_by"] >= 0


def test_analyse_train_received_and_completed(rng):
    rec, _ = _train(rng, pri=1e-3, n=150, drop=0.25)
    D = PU.deinterleave(rec)
    A = PU.analyse_train(D.train(0))
    assert A["n_inferred"] > 0
    err_rec = A["received"]["first_difference"]["mean_s"] / 1e-3 - 1
    err_comp = A["completed"]["first_difference"]["mean_s"] / 1e-3 - 1
    assert err_rec > 0.15                         # dropouts double intervals
    assert abs(err_comp) < err_rec / 3            # gaps over max_missing stay
    assert abs(A["completed"]["first_difference"]["median_s"] - 1e-3) < 1e-6
    assert A["received"]["kind"] == "constant"


def test_unknown_setting_is_refused():
    with pytest.raises(ValueError, match="unknown setting"):
        PU.deinterleave([0.0, 0.001, 0.002], magic=1)


def test_pulse_eval_numbers(tmp_path):
    from atk_diffusion.experiments.pulse_eval import pulse_eval
    res = pulse_eval(seeds=range(2), duration_s=0.2, out_dir=tmp_path)
    p = res["pooled"]
    assert p["accuracy"] > 0.9
    assert p["false_pulse_rate"] < 0.05 and p["false_inferred_off_time"] == 0
    assert p["pri_err_naive_completed"] < p["pri_err_naive_received"]
    assert p["fill_recall"] > 0.5
    assert (tmp_path / "pulse_eval.md").exists() and (tmp_path / "pulse_eval.json").exists()
    assert any("Hallucination" in ln for ln in res["lines"])
