# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""IQ dropout detection and repair before DSD (plan §4.D1)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from atk_diffusion import provenance, sigmf
from atk_diffusion.dsp import iq
from atk_diffusion.repair import iq_dropout as R

FS = 48_000.0


def _tones(n, rng, noise=0.02):
    t = np.arange(n) / FS
    x = (0.5 * np.exp(2j * np.pi * 1500 * t)
         + 0.3 * np.exp(2j * np.pi * -3200 * t + 0.7j)
         + noise * (rng.normal(size=n) + 1j * rng.normal(size=n)))
    return x.astype(np.complex64)


def _snr(truth, est):
    return 10 * np.log10(np.sum(np.abs(truth) ** 2)
                         / np.sum(np.abs(truth - est) ** 2))


def test_methods_are_declared_tiers():
    assert provenance.tier_for("janssen_fill") == "inferred"
    assert provenance.tier_for("blank") == "cleaned"
    for m in ("linear", "ar", "janssen", "learned"):
        assert provenance.tier_for(R.method_name(m)) in ("inferred", "invented")
    assert provenance.tier_for(R.method_name("learned")) == "invented"
    with pytest.raises(ValueError):
        R.method_name("magic")


def test_burg_recovers_a_complex_ar2(rng):
    from scipy.signal import lfilter
    a_true = np.poly([0.95 * np.exp(0.3j), 0.9 * np.exp(-1.1j)])
    e = (rng.normal(size=20000) + 1j * rng.normal(size=20000)) / np.sqrt(2)
    x = lfilter([1], a_true, e)
    a, err = R.burg(x, 2)
    assert np.max(np.abs(a - a_true)) < 0.02
    assert 0.9 < err < 1.1


def test_ar_and_janssen_beat_linear_on_a_predictable_signal(rng):
    x = _tones(20000, rng)
    s, L = 8000, 400
    left, right = x[s - 1024:s], x[s + L:s + L + 1024]
    truth = x[s:s + L]
    lin = R.linear_fill(left, right, L)
    fb = R.fb_fill(left, right, L, 32)
    blk = np.concatenate([left, fb, right])
    m = np.zeros(blk.size, bool)
    m[1024:1024 + L] = True
    jj = R.janssen_fill(blk, m, 32, init=fb)
    assert np.array_equal(jj[~m], blk[~m])          # the record is untouched
    assert _snr(truth, lin) < 3
    assert _snr(truth, fb) > 20
    assert _snr(truth, jj[m]) > 20


def test_a_fill_in_pure_noise_adds_no_energy(rng):
    """The hallucination check for the classical fills: minimum-phase AR
    extrapolation of noise decays, so a gap of noise is filled with LESS
    than the noise, never with a signal."""
    x = (rng.normal(size=8000) + 1j * rng.normal(size=8000)).astype(np.complex64)
    s, L = 4000, 300
    fb = R.fb_fill(x[s - 1024:s], x[s + L:s + L + 1024], L, 32)
    assert np.mean(np.abs(fb) ** 2) < 0.5 * np.mean(np.abs(x) ** 2)


@pytest.mark.parametrize("dt", ["cf32", "cu8", "ci8"])
def test_stuck_and_collapse_found_with_exact_edges(dt, rng):
    x = (0.1 * (rng.normal(size=120000) + 1j * rng.normal(size=120000))).astype(np.complex64)
    x[30000:31500] = 0
    x[60000:60400] = x[59999]
    x[90000:91000] *= 1e-3
    xq = iq.to_complex(iq.from_complex(x, dt), dt)
    if dt == "cu8":
        xq[30000:31500] = xq[30000]       # a zero-filled cu8 buffer
    found = {(d.start, d.count) for d in R.detect_dropouts(xq)}
    assert (30000, 1500) in found
    assert (59999, 401) in found
    assert (90000, 1000) in found
    assert len(found) == 3


def test_no_false_alarms_in_noise_or_on_tdma_slots(rng):
    total = 0
    for dt in ("cf32", "cu8"):
        y = (0.05 * (rng.normal(size=300000) + 1j * rng.normal(size=300000))).astype(np.complex64)
        total += len(R.detect_dropouts(iq.to_complex(iq.from_complex(y, dt), dt)))
    assert total == 0
    # a TDMA burst train falls TO the floor between slots, never below it
    n = 200000
    on = (np.arange(n) // 1440) % 2 == 0
    x = np.zeros(n, np.complex64)
    x[on] = np.exp(1j * rng.uniform(0, 2 * np.pi, on.sum()))
    x += (0.01 * (rng.normal(size=n) + 1j * rng.normal(size=n))).astype(np.complex64)
    assert R.detect_dropouts(iq.to_complex(iq.from_complex(x, "cu8"), "cu8")) == []


def _atk_style_capture(tmp_path, rng, n_true=48000, chunk=1000,
                       dropped=(5, 12, 13), retune_at=20):
    """A cu8 capture with the sidecar ATK's IqRecorder writes: the file is
    contiguous, each gap starts a captures segment whose datetime jumps by
    the missing time, the FIRST segment is to whole seconds, and the global
    block carries atk:gaps / atk:gap_samples."""
    truth = _tones(n_true, rng)
    created = 1_760_000_000.123456
    kept, caps, missing = [], [{"core:sample_start": 0,
                                "core:frequency": 162.4e6,
                                "core:datetime": "2025-10-09T08:53:20Z"}], 0
    pos = 0
    events = 0
    from datetime import datetime, timezone
    for k in range(n_true // chunk):
        if k in dropped:
            missing += chunk
            events += 1
            continue
        freq = 162.5e6 if k >= retune_at else 162.4e6
        new_seg = (k - 1 in dropped) or k == retune_at
        if new_seg:
            t = created + (pos + missing) / FS
            seg = {"core:sample_start": pos, "core:frequency": freq,
                   "core:datetime": datetime.fromtimestamp(t, timezone.utc)
                   .strftime("%Y-%m-%dT%H:%M:%S.%fZ")}
            if caps[-1]["core:sample_start"] == pos:
                caps[-1] = seg
            else:
                caps.append(seg)
        kept.append(truth[k * chunk:(k + 1) * chunk])
        pos += chunk
    data = np.concatenate(kept)
    base = tmp_path / "cap"
    (tmp_path / "cap.sigmf-data").write_bytes(iq.from_complex(data, "cu8"))
    meta = {"global": {"core:datatype": "cu8", "core:sample_rate": int(FS),
                       "core:version": "1.0.0", "core:hw": "RTL-SDR",
                       "core:recorder": "ATK (Analyst Toolkit)",
                       "atk:gaps": events, "atk:gap_samples": missing,
                       "atk:retunes": 1, "atk:settle_samples": 0},
            "captures": caps,
            "annotations": [{"core:sample_start": 15000, "core:sample_count": 2000,
                             "core:label": "dmr", "atk:source": "taught"}]}
    (tmp_path / "cap.sigmf-meta").write_text(json.dumps(meta), encoding="utf-8")
    return base, truth


def test_recorder_gaps_read_atk_sidecar(tmp_path, rng):
    base, _ = _atk_style_capture(tmp_path, rng)
    gaps = R.recorder_gaps(sigmf.read_meta(base))
    got = [(d.kind, d.start, d.count) for d in gaps]
    assert got == [("missing", 5000, 1000), ("missing", 11000, 2000),
                   ("retune", 17000, 0)]
    assert "whole seconds" in gaps[0].detail


def test_repair_capture_restores_the_timeline(tmp_path, rf, rng):
    base, truth = _atk_style_capture(tmp_path, rng)
    rep = R.repair_capture(base, tmp_path / "out" / "cap_rep", method="ar", rf=rf)
    y = sigmf.load(rep["data"])
    meta = sigmf.read_meta(rep["data"])
    assert y.size == truth.size                      # 3000 samples put back
    tq = iq.to_complex(iq.from_complex(truth, "cu8"), "cu8")
    ok = np.ones(y.size, bool)
    for s in rep["spans"]:
        ok[s["start"]:s["start"] + s["count"]] = False
        assert s["inserted"] and s["tier"] == "inferred"
        assert _snr(tq[s["start"]:s["start"] + s["count"]],
                    y[s["start"]:s["start"] + s["count"]]) > 15
    assert np.allclose(y[ok], tq[ok], atol=1e-6)     # record untouched, on time
    g = meta["global"]
    assert g["atk:tier"] == "inferred" and g["atk:method"] == "ar_fill"
    assert g["atk:receiver_profile"] == "rtlsdr_48000_cu8"
    assert "atk:gaps" not in g and g["atk:source_gaps"]["atk:gap_samples"] == 3000
    assert [c["core:sample_start"] for c in meta["captures"]] == [0, 20000]
    rep_anns = [a for a in meta["annotations"] if a.get("core:label") == "repaired"]
    assert [(a["core:sample_start"], a["core:sample_count"]) for a in rep_anns] \
        == [(5000, 1000), (12000, 2000)]
    assert all(a["atk:tier"] == "inferred" and a["atk:method"] == "ar_fill"
               for a in rep_anns)
    taught = [a for a in meta["annotations"] if a.get("core:label") == "dmr"][0]
    assert taught["core:sample_start"] == 18000       # moved past 3000 inserted
    assert sigmf.validate(meta) == []
    assert rf.verify(rep["data"])[0]
    assert any("INFERRED" in ln for ln in rep["lines"])


def test_long_spans_are_blanked_not_filled(rng):
    truth = _tones(48000, rng)
    x = truth.copy()
    x[10000:16000] = 0                               # 125 ms of zeros
    x[30000:30200] = 0
    drops = R.detect_dropouts(x)
    y, spans, _ = R.repair_array(x, drops, "ar", fs=FS, max_fill_s=0.05)
    by = {s.start: s for s in spans}
    assert by[10000].method == "blank" and by[10000].tier == "cleaned"
    assert "blanked" in by[10000].note
    assert np.all(y[10000:16000] == 0)
    assert by[30000].method == "ar_fill" and by[30000].tier == "inferred"
    assert _snr(truth[30000:30200], y[30000:30200]) > 20
    assert R.worst_tier(s.tier for s in spans) == "inferred"


def test_multichannel_spans_hit_every_channel(rng):
    x = np.stack([_tones(30000, rng), _tones(30000, rng)])
    x[:, 12000:12300] = 0
    drops = R.detect_dropouts(x)
    assert [(d.start, d.count) for d in drops] == [(12000, 300)]
    y, spans, _ = R.repair_array(x, drops, "janssen", fs=FS)
    assert y.shape == x.shape
    assert np.all(np.abs(y[:, 12000:12300]).mean(axis=1) > 0.3)


def test_learned_hook_only_touches_the_gap(rng):
    x = _tones(20000, rng)
    x[5000:5100] = 0
    drops = R.detect_dropouts(x)

    def vandal(z, mask, fs):           # rewrites EVERYTHING it is handed
        return np.full_like(z, 0.25 + 0.25j)

    y, spans, _ = R.repair_array(x, drops, "learned", fs=FS, inpainter=vandal)
    assert spans[0].tier == "invented" and spans[0].method == "diffusion_inpaint"
    keep = np.ones(x.size, bool)
    keep[5000:5100] = False
    assert np.array_equal(y[keep], x[keep])
    assert np.allclose(y[5000:5100], 0.25 + 0.25j)


def test_learned_without_the_module_says_so_in_words(rng, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_learn(name, *a, **k):
        if name.startswith("atk_diffusion.learn"):
            raise ImportError("not installed here")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_learn)
    x = _tones(5000, rng)
    x[2000:2050] = 0
    with pytest.raises(RuntimeError, match="classical repairs"):
        R.repair_array(x, R.detect_dropouts(x), "learned", fs=FS)


def test_a_capture_that_is_mostly_dropout_is_refused(tmp_path, rng):
    x = _tones(20000, rng)
    x[2000:16000] = 0
    sigmf.write_pair(tmp_path / "dead", x, FS)
    with pytest.raises(ValueError, match="not enough record"):
        R.repair_capture(tmp_path / "dead", tmp_path / "dead_rep")


def test_learned_inpainter_integration_when_present(rng):
    """Runs against atk_diffusion.learn.inpaint once that part exists."""
    try:
        from atk_diffusion.learn import inpaint  # noqa: F401
    except ImportError as e:
        pytest.skip(f"atk_diffusion.learn.inpaint is not importable yet ({e})")
    if not callable(getattr(inpaint, "inpaint", None)):
        pytest.skip("atk_diffusion.learn.inpaint has no inpaint() entry point yet")
    pytest.importorskip("torch")
    x = _tones(4096, rng)
    x[2000:2032] = 0
    try:
        y, spans, _ = R.repair_array(x, R.detect_dropouts(x), "learned", fs=FS)
    except RuntimeError as e:
        pytest.skip(f"the learned inpainter could not run here: {e}")
    keep = np.ones(x.size, bool)
    keep[2000:2032] = False
    assert np.array_equal(y[keep], x[keep])
    assert spans[0].tier == "invented"
