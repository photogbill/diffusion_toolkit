# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Low-SNR escalation (DETECTION_DESIGN §4.3): the IQ ring buffer, the
policy that says when to search it, `escalate` over it, the `Escalator`
that glues them to the detector pipeline — and the pipeline itself finding
a signal under the CFAR line only because the buffer integrated seconds."""

from __future__ import annotations

import math

import numpy as np
import pytest

import helpers_cyclo as H
from atk_diffusion import profiles, sigmf
from atk_diffusion.cyclo import escalate as E
from atk_diffusion.detect.boxes import Detection
from atk_diffusion.detect.pipeline import DetectorPipeline

FS48 = 48_000.0
PID48 = "rtlsdr_48000_cu8"
C48 = 155.0e6
FS96 = 960_000.0
PID96 = "rtlsdr_960000_cu8"
C96 = 739.0e6
LTE_BAND = (C96 - 270e3, C96 + 270e3)


def _lte(n, rng, cfo=700.0):
    """LTE-like at 0.96 MS/s: 15 kHz subcarriers (64-point), 36 used
    (540 kHz), a 0.5 ms slot of 7 symbols with prefixes 5,4,5,4,5,4,5 —
    the useful symbol 66.67 µs, the average period 1/14 kHz."""
    return H.lte_like(FS96, n, rng, nfft=64, used=36,
                      cps=(5, 4, 5, 4, 5, 4, 5), cfo=cfo)


def _fsk_scene(seed, seconds=3.0, snr=3.0, f=6_000.0):
    rng = np.random.default_rng(seed)
    n = int(seconds * FS48)
    s = H.mix(H.fsk4(FS48, 4800.0, n, rng), f, FS48)
    return (H.inband_amplitude(snr, 8000.0, FS48) * s + H.noise(n, rng)
            ).astype(np.complex64)


def _weak_box(t0=2.0, t1=3.0, snr=2.0, lo=4e3, hi=8e3, conf=None):
    return Detection(t0=t0, t1=t1, f_lo=C48 + lo, f_hi=C48 + hi,
                     sources=("energy",), snr_db=snr, confidence=conf)


# ---------------------------------------------------------------------------
# IqBuffer
# ---------------------------------------------------------------------------
def test_buffer_push_and_window_keep_samples_and_stream_times():
    b = E.IqBuffer(1.0, 100.0)                     # 100 samples
    ramp = np.arange(250, dtype=np.complex64)
    for k in range(5):                               # five blocks of 50
        b.push(ramp[k * 50:(k + 1) * 50], t_end=(k + 1) * 0.5, center_hz=1e6)
    assert b.seconds == pytest.approx(1.0) and b.capacity_seconds == 1.0
    assert b.span() == (pytest.approx(1.5), pytest.approx(2.5))
    x, t = b.window()
    assert np.array_equal(x, ramp[150:250]) and t == pytest.approx(1.5)
    x, t = b.window(1.7, 2.0)
    assert np.array_equal(x, ramp[170:200]) and t == pytest.approx(1.7)
    assert np.array_equal(b.get(0.0, 1.6), ramp[150:160])       # clipped
    assert b.window(3.0, 4.0)[0].size == 0
    assert b.nbytes == 800
    # a block longer than the ring keeps its last capacity samples
    b.push(np.arange(1000, 1300, dtype=np.complex64), t_end=5.5)
    assert np.array_equal(b.get(), np.arange(1200, 1300))
    b.push(np.zeros(0, np.complex64), t_end=5.5)                 # no-op
    assert b.seconds == pytest.approx(1.0)


def test_gaps_overlaps_and_retunes_restart_the_buffer_with_a_note():
    b = E.IqBuffer(2.0, 100.0)
    b.push(np.ones(100), t_end=1.0, center_hz=1e6)
    b.push(np.ones(50), t_end=2.0, center_hz=1e6)          # 0.5 s missing
    assert b.seconds == pytest.approx(0.5)
    assert "a gap of 500.000 ms" in b.notes[-1]
    b.push(np.ones(10), t_end=1.95, center_hz=1e6)         # goes backwards
    assert "an overlap" in b.notes[-1] and b.seconds == pytest.approx(0.1)
    b.push(np.ones(50), t_end=2.45, center_hz=2e6)
    assert "retuned from 1,000,000 Hz to 2,000,000 Hz" in b.notes[-1]
    assert b.seconds == pytest.approx(0.5) and b.center_hz == 2e6
    b.clear("asked")
    assert b.span() == (None, None) and b.window() [1] is None


def test_buffer_limits_are_refused_in_words():
    with pytest.raises(ValueError, match="over the 0.00 GB limit"):
        E.IqBuffer(10.0, 2.4e6, max_bytes=1000)
    with pytest.raises(ValueError, match="less than one sample"):
        E.IqBuffer(1e-9, 100.0)
    with pytest.raises(ValueError, match="positive length"):
        E.IqBuffer(-1.0, 100.0)
    b = E.IqBuffer(1.0, 100.0, channels=5)
    with pytest.raises(ValueError, match="holds 5 channel"):
        b.push(np.ones(10), t_end=0.1)
    b.push(np.ones((5, 30)), t_end=0.3)
    assert b.window()[0].shape == (5, 30) and b.nbytes == 5 * 100 * 8


# ---------------------------------------------------------------------------
# EscalationPolicy
# ---------------------------------------------------------------------------
def test_policy_triggers_in_words():
    p = E.EscalationPolicy.from_profile(PID48)
    assert p.escalate_snr_db == profiles.new_profile(PID48).escalate_snr_db
    ok, why = p.check(detection=_weak_box(snr=12.0, conf=0.5),
                      buffer_seconds=5, now=0)
    assert ok and "the energy proposer is unsure" in why
    ok, why = p.check(detection=_weak_box(snr=2.0, lo=20e3, hi=24e3),
                      buffer_seconds=5, now=0)
    assert ok and "under the profile's escalation line of 6.0 dB" in why
    ok, why = p.check(region=(1e6, 2e6), hunter_dwelling=True,
                      buffer_seconds=5, now=0)
    assert ok and "hunter is dwelling" in why
    ok, why = p.check(region=(3e6, 4e6), analyst=True, buffer_seconds=5, now=0)
    assert ok and "analyst asked" in why
    ok, why = p.check(detection=_weak_box(snr=15.0, conf=0.9, lo=50e3, hi=60e3),
                      buffer_seconds=5, now=0)
    assert not ok and why.startswith("no trigger") and "15.0 dB is above" in why
    ok, why = p.check(detection=_weak_box(snr=1.0, lo=70e3, hi=80e3),
                      buffer_seconds=1.0, now=0)
    assert not ok and "needs at least 2.0 s" in why
    # the pipeline's ambiguous band and the policy's are the same band
    from atk_diffusion.detect.pipeline import AMBIGUOUS
    assert tuple(p.ambiguous) == AMBIGUOUS


def test_policy_cooldown_and_the_analyst_override():
    p = E.EscalationPolicy(cooldown_s=10.0)
    d = _weak_box(snr=1.0)
    assert p.check(detection=d, buffer_seconds=5, now=100.0)[0]
    ok, why = p.check(detection=d, buffer_seconds=5, now=104.0)
    assert not ok and "searched 4.0 s ago" in why and "analyst can ask" in why
    assert p.check(region=(d.f_lo, d.f_hi), analyst=True, buffer_seconds=5,
                   now=105.0)[0]
    assert p.check(detection=d, buffer_seconds=5, now=116.0)[0]
    # the same signal's box a few hundred hertz over is the same region
    moved = _weak_box(snr=1.0, lo=4.3e3, hi=8.2e3)
    ok, why = p.check(detection=moved, buffer_seconds=5, now=118.0)
    assert not ok and "searched 2.0 s ago" in why
    # the whole span is one region with its own cooldown (the first version
    # had none: a dwelling hunter re-searched the span on every tile)
    assert p.check(hunter_dwelling=True, buffer_seconds=5, now=0.0)[0]
    assert not p.check(hunter_dwelling=True, buffer_seconds=5, now=3.0)[0]
    with pytest.raises(ValueError, match="a region is"):
        p.check(region="here", hunter_dwelling=True, buffer_seconds=5)


# ---------------------------------------------------------------------------
# escalate() over the buffer
# ---------------------------------------------------------------------------
def test_escalate_finds_a_cell_one_second_misses_and_energy_never_sees():
    """Bill's cell-tower case: an LTE-like downlink 13 dB under the floor
    over 60 % of the span. Energy (CFAR per frame) sees nothing; the cyclic
    look over ONE second misses it; over the 4-second buffer it is found,
    with the integration time on the box."""
    rng = np.random.default_rng(3)
    n = int(4 * FS96)
    x = (H.inband_amplitude(-13.0, 540e3, FS96) * _lte(n, rng)
         + H.noise(n, rng)).astype(np.complex64)
    mask, freqs = H.per_frame_cfar(x[:int(FS96)], FS96, pfa=1e-4)
    inband = mask[:, np.abs(freqs) < 250e3]
    assert inband.mean() < 5e-4                    # the false-alarm rate, no more
    short = E.IqBuffer(1.0, FS96)
    short.push(x[-int(FS96):], t_end=4.0, center_hz=C96)
    assert E.escalate(short, LTE_BAND, C96, PID96) == []
    buf = E.IqBuffer(4.0, FS96, epoch=1.7e9)
    buf.push(x, t_end=4.0, center_hz=C96)
    rep = {}
    found = E.escalate(buf, LTE_BAND, C96, PID96, report=rep)
    assert len(found) == 1
    d = found[0]
    assert "escalated" in d.flags and d.sources == ("cyclic",)
    assert d.cls == "lte_dl" and d.family == "ofdm"
    assert d.integration_s == pytest.approx(4.0)
    assert d.measurements["buffer_seconds"] == pytest.approx(4.0)
    assert d.t0 == pytest.approx(0.0) and d.t1 == pytest.approx(4.0)
    assert d.epoch == 1.7e9
    assert d.measurements["cp_lag_s"] == pytest.approx(1 / 15_000)
    # 13 dB under, the prefix correlation shows; the prefix train's period
    # (the 14 kHz symbol rate, alpha_hz) need not — said, not guessed
    assert d.alpha_hz is None or d.alpha_hz == pytest.approx(14_000.0, rel=1e-3)
    assert d.measurements["cfo_hz"] == pytest.approx(700.0, abs=300.0)
    assert d.measurements["cp_cells"] == 1
    assert rep["buffer_seconds"] == pytest.approx(4.0)
    assert rep["region"] == [LTE_BAND[0], LTE_BAND[1]]


def test_escalate_widens_a_fragment_box_and_refuses_another_band():
    x = _fsk_scene(1)
    buf = E.IqBuffer(3.0, FS48)
    buf.push(x, t_end=3.0, center_hz=C48)
    rep = {}
    found = E.escalate(buf, _weak_box(), C48, PID48, report=rep)
    assert "widened to 12,500 Hz" in rep["region_note"]
    assert rep["region"] == [pytest.approx(C48 - 250.0), pytest.approx(C48 + 12_250.0)]
    assert found and found[0].measurements["candidates"] == ["dmr", "nxdn96",
                                                             "p25"]
    assert "region_note" in found[0].measurements
    with pytest.raises(ValueError, match="holds another band"):
        E.escalate(buf, _weak_box(), C48 + 1e6, PID48)
    empty = E.IqBuffer(1.0, FS48)
    r2 = {}
    assert E.escalate(empty, None, C48, PID48, report=r2) == []
    assert "empty" in r2["words"]
    with pytest.raises(ValueError, match="outside the receiver's usable span"):
        E.escalate(buf, (C48 + 30e3, C48 + 40e3), C48, PID48)


# ---------------------------------------------------------------------------
# the Escalator — the pipeline's hook
# ---------------------------------------------------------------------------
def _fed(seed=1, seconds=3.0, block=12_000, **kw):
    esc = E.Escalator(PID48, seconds=3.0, **kw)
    x = _fsk_scene(seed, seconds)
    for k in range(int(x.size // block)):
        esc.feed(x[k * block:(k + 1) * block], (k + 1) * block / FS48, C48)
    return esc, x


def test_the_escalator_runs_and_logs_why():
    """The lead's glue: `buffer.seconds` is a property — the first version
    called it, and every escalation died as a TypeError."""
    esc, x = _fed()
    out = esc([_weak_box()], x[-48_000:], FS48, C48, 2.0, esc.profile)
    assert len(out) == 1 and "escalated" in out[0].flags
    assert out[0].integration_s == pytest.approx(3.0)
    assert out[0].measurements["candidates"] == ["dmr", "nxdn96", "p25"]
    line = esc.log[-1]
    assert line.startswith("t = 3.00 s")
    assert "under the profile's escalation line" in line and "1 found" in line
    assert esc.escalations == 1 and esc.found == 1
    st = esc.status()
    assert st["buffer_seconds"] == pytest.approx(3.0) and "1 found" in st["words"]
    # the same region again, at once: the cooldown (in stream time) holds
    assert esc([_weak_box()], x[-48_000:], FS48, C48, 2.0, esc.profile) == []
    assert "searched 0.0 s ago" in esc.log[-1]


def test_the_escalator_cooldown_counts_stream_seconds():
    esc, x = _fed()
    assert esc([_weak_box()], x, FS48, C48, 2.0, esc.profile)
    noise = H.noise(int(11 * FS48), np.random.default_rng(5)).astype(np.complex64)
    for k in range(11):                           # 11 s more of the stream
        esc.feed(noise[k * 48_000:(k + 1) * 48_000], 3.0 + (k + 1), C48)
    esc([_weak_box()], noise[-48_000:], FS48, C48, 13.0, esc.profile)
    assert esc.escalations == 2
    assert "nothing found over 3.0 s" in esc.log[-1]


def test_the_escalator_refusals_and_its_own_tile():
    esc = E.Escalator(PID48, seconds=3.0)
    out = esc([_weak_box()], np.zeros(1000, np.complex64), 2.4e6, C48, 0.0,
              esc.profile)
    assert out == [] and "skipped: the stream is at 2.4e+06 S/s" in esc.log[-1]
    # a host that never fed the ring: the tile itself goes in — too short
    tile = _fsk_scene(2, seconds=1.0)
    assert esc([_weak_box(t0=0, t1=1)], tile, FS48, C48, 0.0, esc.profile) == []
    assert esc.buffer.seconds == pytest.approx(1.0)
    assert "holds only 1.0 s" in esc.log[-1]
    # the retune note reaches the log
    esc.feed(tile[:4800], 1.1, C48 + 1e6)
    assert "retuned" in esc.log[-1]
    # an escalation refused by escalate() is logged, not raised
    esc2, x = _fed(seed=3)
    esc2.ask((C48 + 30e3, C48 + 40e3), C48)
    assert "the escalation was refused" in esc2.log[-1]


def test_dwelling_and_the_analyst_ask():
    esc, x = _fed(seed=2, seconds=3.0)
    assert not esc.wants_every_tile
    esc.dwell((C48 + 1e3, C48 + 11e3))
    assert esc.wants_every_tile and "dwelling on" in esc.log[-1]
    out = esc([], x[-48_000:], FS48, C48, 2.0, esc.profile)
    assert len(out) == 1 and "escalated" in out[0].flags
    assert "hunter is dwelling" in esc.log[-1] and "1 found" in esc.log[-1]
    n_log = len(esc.log)
    assert esc([], x[-48_000:], FS48, C48, 2.0, esc.profile) == []
    assert len(esc.log) == n_log           # cooldown refusals are not logged
    # the analyst's request ignores the cooldown
    again = esc.ask((C48 + 1e3, C48 + 11e3), C48)
    assert len(again) == 1 and "analyst asked" in esc.log[-1]
    esc.dwell((C48 + 1e3, C48 + 11e3), on=False)
    assert not esc.wants_every_tile and "stopped dwelling" in esc.log[-1]


# ---------------------------------------------------------------------------
# the detector pipeline, end to end
# ---------------------------------------------------------------------------
def test_a_cell_under_the_cfar_line_is_found_by_escalation_in_the_pipeline(
        rf):
    """A capture holding an LTE-like cell 10 dB under the floor, fed to the
    detector in quarter-second blocks: the energy proposer makes no box for
    it (it never can — it is part of the floor), so nothing it does could
    trigger a look; the hunter dwelling on the band does, the Escalator
    searches its 2.5-second ring, and the box comes back flagged
    'escalated' with 2.0 s or more of integration on it — and the log says
    why."""
    rng = np.random.default_rng(21)
    n = int(2.6 * FS96)
    x = (H.inband_amplitude(-10.0, 540e3, FS96) * _lte(n, rng, cfo=-1100.0)
         + H.noise(n, rng)).astype(np.complex64)
    base = rf.captures(PID96) / "cell"
    sigmf.write_pair(base, x, FS96, C96, datatype="cf32", t0_utc=1.7e9,
                     extra_global={"atk:receiver_profile": PID96})
    plain = DetectorPipeline(PID96).run_on_capture(base, chunk_seconds=0.25)
    assert not [d for d in plain["detections"]
                if d.f_hi > LTE_BAND[0] and d.f_lo < LTE_BAND[1]]
    esc = E.Escalator(PID96, seconds=2.5)
    esc.dwell(LTE_BAND)
    pipe = DetectorPipeline(PID96, escalation=esc)
    res = pipe.run_on_capture(base, chunk_seconds=0.25)
    found = [d for d in res["detections"] if "escalated" in d.flags]
    assert len(found) == 1
    d = found[0]
    assert d.integration_s >= 2.0 and "cyclic" in d.sources
    assert "energy" not in d.sources
    assert d.cls == "lte_dl" and d.measurements["cp_cells"] == 1
    assert d.measurements["cfo_hz"] == pytest.approx(-1100.0, abs=300.0)
    assert d.epoch == pytest.approx(1.7e9) and d.track_id
    assert any("hunter is dwelling" in line and "1 found" in line
               for line in esc.log)
    st = res["status"]
    assert st["counts_by_source"]["escalated"] == 1
    assert st["escalation"]["found"] == 1
    assert any(line.startswith("Low-SNR escalation: ON") for line in st["lines"])


def _small_48k():
    prof = profiles.new_profile(PID48)
    prof.stft = profiles.StftGeometry(fft_size=256, hop=256, tile_seconds=0.68,
                                      tile_rows=128, tile_overlap=0.25)
    return prof


def test_a_weak_energy_box_triggers_a_cyclic_look_over_the_buffer():
    """A P25-like 4FSK a few dB over the floor: the energy proposer makes a
    weak box in every tile (~8.5 dB, under this profile's line, set to
    10 dB — the line is the profile's to set); the escalation hook searches
    the ring over the box's channel once (the next tiles' boxes, a few
    hundred hertz apart, are the same region inside its cooldown) and the
    4800 sym/s line comes back."""
    prof = _small_48k()
    prof.escalate_snr_db = 10.0
    x = _fsk_scene(7, seconds=4.0, snr=6.0)
    esc = E.Escalator(prof, seconds=3.0)
    pipe = DetectorPipeline(prof, escalation=esc)
    dets = []
    for k in range(16):
        blk = x[k * 12_000:(k + 1) * 12_000]
        dets += pipe.feed(blk, C48, t_start=k * 12_000 / FS48)
    dets += pipe.finish()
    esc_d = [d for d in dets if "escalated" in d.flags]
    assert esc_d, list(esc.log)
    d = esc_d[0]
    assert d.integration_s >= 2.0 and d.alpha_hz == pytest.approx(4800.0,
                                                                  abs=1.0)
    assert set(d.measurements["candidates"]) == {"dmr", "nxdn96", "p25"}
    assert any("escalation line of 10.0 dB" in line and "found" in line
               for line in esc.log)
    assert esc.escalations == 1
    assert any("holds only" in line for line in esc.log)   # before 2 s
