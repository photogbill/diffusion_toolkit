# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Tracks from scripted boxes (DETECTION_DESIGN §5): PTT key-ups, a TDMA
pattern, a drifting carrier, a hopper."""

from __future__ import annotations

import numpy as np
import pytest

from atk_diffusion.detect.boxes import Detection
from atk_diffusion.detect.tracker import Tracker

F0 = 162.400e6


def _det(t0, t1, fc, bw, **kw):
    return Detection(t0=t0, t1=t1, f_lo=fc - bw / 2, f_hi=fc + bw / 2,
                     sources=kw.pop("sources", ("energy",)), **kw)


def test_ptt_key_ups_on_one_channel_are_one_track():
    tr = Tracker(max_gap_s=10.0)
    ups = [(0.0, 2.5), (5.0, 6.2), (9.0, 12.0)]
    a = [_det(t0, t1, F0, 11e3) for t0, t1 in ups]
    other = _det(1.0, 3.0, F0 + 25e3, 11e3)          # the next channel up
    for d in sorted(a + [other], key=lambda d: d.t0):
        tr.update([d])
    assert len({d.track_id for d in a}) == 1
    assert other.track_id and other.track_id != a[0].track_id
    t = tr.get(a[0].track_id)
    assert t.n_bursts == 3 and t.first_seen == 0.0 and t.last_seen == 12.0
    assert t.on_time_s == pytest.approx(2.5 + 1.2 + 3.0)
    assert t.duty_cycle == pytest.approx(6.7 / 12.0)
    assert t.pri_s == pytest.approx(4.5)              # median of 5.0 and 4.0
    assert not t.hop and t.center_hz == pytest.approx(F0)
    assert "3 bursts" in t.words() and "PROPOSED" in t.words()


def test_tdma_bursts_in_one_update_become_one_track_with_its_pri():
    """DMR: 27.5 ms bursts on a 30 ms frame — sixteen of them in one tile's
    update must all land on one track."""
    tr = Tracker(burst_gap_s=0.001)
    bursts = [_det(k * 0.030, k * 0.030 + 0.0275, F0, 7.6e3, family="fsk")
              for k in range(16)]
    touched = tr.update(bursts)
    assert len(touched) == 1 and len(tr.active()) == 1
    t = touched[0]
    assert t.n_bursts == 16 and t.pri_s == pytest.approx(0.030)
    assert t.duty_cycle == pytest.approx(16 * 0.0275 / (15 * 0.030 + 0.0275))
    assert t.family == "fsk"
    # the single-slot pattern (one burst every 60 ms) on another channel
    tr2 = Tracker(burst_gap_s=0.001)
    t2 = tr2.update([_det(k * 0.060, k * 0.060 + 0.0275, F0, 7.6e3)
                     for k in range(10)])[0]
    assert t2.pri_s == pytest.approx(0.060) and t2.duty_cycle == pytest.approx(
        10 * 0.0275 / (9 * 0.060 + 0.0275))


def test_a_drifting_carrier_is_one_track_with_its_drift():
    tr = Tracker()
    drift = 120.0                                      # Hz per second
    boxes = [_det(k * 0.25, (k + 1) * 0.25, F0 + drift * (k + 0.5) * 0.25, 1.5e3)
             for k in range(40)]
    for k in range(0, 40, 4):                          # in tile-sized updates
        tr.update(boxes[k:k + 4])
    assert len({b.track_id for b in boxes}) == 1
    t = tr.get(boxes[0].track_id)
    assert t.drift_hz_per_s == pytest.approx(drift, rel=0.02)
    assert t.n_bursts == 1 and t.duty_cycle == pytest.approx(1.0)
    assert not t.hop and "drifting +120.0 Hz/s" in t.words()


def test_a_hopper_is_one_track_flagged_hop_and_a_steady_carrier_is_not():
    rng = np.random.default_rng(3)
    channels = F0 + 50e3 * np.arange(8)
    hops, prev = [], -1
    for k in range(20):
        c = int(rng.integers(8))
        while c == prev:
            c = int(rng.integers(8))
        prev = c
        hops.append(_det(k * 0.020, (k + 1) * 0.020 - 0.0005, channels[c], 25e3))
    steady = [_det(k * 0.1, (k + 1) * 0.1, F0 - 200e3, 12e3) for k in range(4)]
    tr = Tracker()
    tr.update(hops + steady)
    assert len({h.track_id for h in hops}) == 1
    t = tr.get(hops[0].track_id)
    assert t.hop and t.n_hops == 19 and t.drift_hz_per_s is None
    assert "HOPPING (19 hops)" in t.words()
    s = tr.get(steady[0].track_id)
    assert s.id != t.id and not s.hop and len({d.track_id for d in steady}) == 1


def test_a_long_gap_or_a_family_conflict_starts_a_new_track():
    tr = Tracker(max_gap_s=5.0)
    a = _det(0.0, 1.0, F0, 12e3, family="fm")
    b = _det(1.2, 2.0, F0, 12e3, family="fsk")        # same channel, another family
    c = _det(10.0, 11.0, F0, 12e3, family="fm")        # after the gap
    for d in (a, b, c):
        tr.update([d])
    assert len({a.track_id, b.track_id, c.track_id}) == 3
    assert tr.get(a.track_id).closed                   # expired by the gap
    # an unknown family is compatible with anything
    d = _det(11.1, 12.0, F0, 12e3)
    tr.update([d])
    assert d.track_id == c.track_id


def test_fragments_of_one_signal_in_one_update_stay_one_track():
    tr = Tracker()
    tr.update([_det(0.0, 0.5, F0, 20e3)])
    frags = [_det(0.6, 0.7, F0 - 5e3, 10e3), _det(0.62, 0.8, F0 + 4e3, 10e3),
             _det(0.65, 0.9, F0, 18e3)]
    tr.update(frags)
    assert len(tr.active()) == 1
    assert len({f.track_id for f in frags}) == 1


def test_concurrent_channels_compete_through_the_assignment():
    tr = Tracker()
    first = [_det(0.0, 1.0, F0 + k * 12.5e3, 11e3) for k in range(5)]
    tr.update(first)
    nxt = [_det(1.0, 2.0, F0 + k * 12.5e3, 11e3) for k in range(5)]
    tr.update(list(reversed(nxt)))
    assert [d.track_id for d in nxt] == [d.track_id for d in first]
    assert len(tr.active()) == 5


def test_confirmation_upgrades_the_track_and_the_decoder_wins_the_label():
    tr = Tracker()
    d1 = _det(0.0, 0.5, F0, 12e3, cls="dmr", family="fsk")
    tr.update([d1])
    d2 = _det(0.6, 1.0, F0, 12e3, cls="dmr", family="fsk")
    d2.confirm("dsd", "TG 1234", decoder_class="p25")
    tr.update([d2])
    t = tr.get(d1.track_id)
    assert t.state == "confirmed" and t.confirmed_by == "dsd" and t.cls == "p25"
    d3 = _det(1.1, 1.5, F0, 12e3, cls="dmr", family="fsk")
    tr.update([d3])
    assert t.cls == "p25"                               # stays the decoder's
    # confirmed after it was tracked
    e = _det(0.0, 0.4, F0 + 1e6, 12e3, cls="pocsag")
    tr.update([e])
    e.confirm("pager", "1234567: TEST")
    assert tr.note_confirmed(e).state == "confirmed"


def test_retune_closes_everything_and_json_is_plain():
    tr = Tracker()
    tr.update([_det(0.0, 1.0, F0, 12e3), _det(0.0, 1.0, F0 + 1e6, 12e3)])
    closed = tr.close_all()
    assert len(closed) == 2 and not tr.active() and all(t.closed for t in closed)
    j = closed[0].to_json()
    assert not any(k.startswith("_") for k in j) and isinstance(j["sources"], list)
    with pytest.raises(ValueError):
        Tracker(max_gap_s=0)
