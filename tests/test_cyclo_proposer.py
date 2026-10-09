# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The cyclic proposer (DETECTION_DESIGN §3, §4.1; ARCHITECTURE §4.2): the
class table's cycle frequencies, in regions or across the span, a derived
false-alarm rate per call, OFDM by its prefix with the carrier offset
referred to the capture, harmonics folded, the class never guessed."""

from __future__ import annotations

import math

import numpy as np
import pytest

import helpers_cyclo as H
from atk_diffusion import profiles
from atk_diffusion.cyclo import proposer as P
from atk_diffusion.detect import classes

FS48 = 48_000.0
PID48 = "rtlsdr_48000_cu8"
C48 = 152.0e6
FS96 = 960_000.0
PID96 = "rtlsdr_960000_cu8"
C96 = 739.0e6
VOICE = ["p25", "dmr", "nxdn96"]


def _fsk4(seed, seconds, snr, f=4_000.0):
    rng = np.random.default_rng(seed)
    n = int(seconds * FS48)
    s = H.mix(H.fsk4(FS48, 4800.0, n, rng), f, FS48)
    return (H.inband_amplitude(snr, 8000.0, FS48) * s + H.noise(n, rng)
            ).astype(np.complex64)


def _fsk2(n, rng, rate, dev=4500.0):
    """POCSAG-like: NRZ 2FSK, continuous phase, ±dev Hz."""
    sps = int(FS48 // rate)
    f = np.repeat(rng.choice([-1.0, 1.0], n // sps + 2), sps)[:n] * dev
    return np.exp(2j * np.pi * np.cumsum(f) / FS48)


def _lte(seconds, snr, cfo, seed=4):
    rng = np.random.default_rng(seed)
    n = int(seconds * FS96)
    s = H.lte_like(FS96, n, rng, nfft=64, used=36, cps=(5, 4, 5, 4, 5, 4, 5),
                   cfo=cfo)
    return (H.inband_amplitude(snr, 540e3, FS96) * s + H.noise(n, rng)
            ).astype(np.complex64)


# ---------------------------------------------------------------------------
# the class table is what it scans
# ---------------------------------------------------------------------------
def test_the_plan_scans_the_class_table_for_the_receiver():
    cl = P._class_list(None, "rtlsdr")
    assert all(not c.negative for c in cl)
    groups = P._rate_groups(cl, 240_000.0)
    rates = sorted(r for g in groups for r in g["rates"])
    listed = sorted({r for name, r in classes.cycle_frequencies(240_000.0)
                     if classes.get(name).cp_lag_s == 0
                     and "rtlsdr" in (classes.get(name).profiles or ("rtlsdr",))})
    assert rates == listed == [512.0, 1200.0, 1600.0, 2400.0, 3200.0, 4800.0]
    g48 = next(g for g in groups if 4800.0 in g["rates"])
    assert g48["bw"] == 8_300.0                    # the widest of the three
    assert sorted(g48["classes"][4800.0]) == ["dmr", "nxdn96", "p25"]
    assert g48["families"][4800.0] == "fsk"
    rep = {}
    noise = H.noise(int(0.5 * 240_000), np.random.default_rng(0)).astype(
        np.complex64)
    P.cyclic_proposer(noise, 240_000.0, C48, "rtlsdr_240000_cu8", report=rep)
    labels = " | ".join(c["what"] for c in rep["calls"])
    assert "4800 sym/s" in labels and "2400, 512, 1200, 1600, 3200 sym/s" in labels
    assert rep["mode"] == "grid" and rep["probe_calls"] > 50
    assert rep["pfa_per_call"] == pytest.approx(1e-3 / rep["probe_calls"])


def test_lte_is_probed_at_its_prefix_lag_with_its_14_khz_symbol_rate():
    """The class table: LTE's cycle frequency is the 14 kHz symbol rate (7
    symbols a 0.5 ms slot); its cyclic prefix is seen at a LAG of 1/15 kHz
    = 66.67 µs (the useful symbol), not at a 15 kHz cycle frequency."""
    lte = classes.get("lte_dl")
    assert lte.symbol_rates == (14_000.0,)
    assert lte.cp_lag_s == pytest.approx(66.6667e-6, rel=1e-5)
    entries = P._cp_entries([lte, classes.get("nr_dl")])
    e = next(e for e in entries if e["classes"] == ["lte_dl"])
    assert e["lag_s"] == pytest.approx(1 / 15_000) and e["hint"] == 14_000.0
    # its symbol rate is NOT probed as a symbol-rate line (OFDM's is a train)
    assert all(14_000.0 not in g["rates"]
               for g in P._rate_groups([lte], 1.92e6))


# ---------------------------------------------------------------------------
# regions and span
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("regions", [None, [(C48 + 2e3, C48 + 6e3)]])
def test_a_4fsk_is_found_in_regions_and_across_the_span(regions):
    x = _fsk4(1, 2.0, 6.0)
    rep = {}
    dets = P.cyclic_proposer(x, FS48, C48, PID48, classes=VOICE,
                             regions=regions, t0=10.0, epoch=1.7e9, report=rep)
    assert rep["mode"] == ("grid" if regions is None else "regions")
    assert len(dets) == 1
    d = dets[0]
    assert d.sources == ("cyclic",) and d.family == "fsk"
    assert d.cls == ""                              # never guessed
    assert d.measurements["candidates"] == ["dmr", "nxdn96", "p25"]
    assert "the decoder says which" in d.measurements["class_words"]
    assert d.alpha_hz == pytest.approx(4800.0, abs=0.5)
    assert d.integration_s == pytest.approx(2.0)
    assert (d.t0, d.t1) == (pytest.approx(10.0), pytest.approx(12.0))
    assert d.epoch == 1.7e9 and d.profile == PID48
    assert d.f_lo < C48 + 4e3 < d.f_hi
    assert 0.45 <= d.confidence <= 0.95
    m = d.measurements
    for k in ("probe", "statistic", "threshold", "margin", "p_value_call",
              "channels_merged", "canonical_rate_hz", "lags", "symbol_rate_hz"):
        assert k in m, k
    assert m["statistic"] > m["threshold"] and m["margin"] > 1
    assert 0 not in m["lags"]                       # FSK has no |x|² line


def test_harmonics_are_folded_into_one_signal():
    """POCSAG-like 2FSK at 1200 Bd makes lines at 2400 and 4800 too: one
    detection, POCSAG, its harmonics listed — not a DMR as well."""
    rng = np.random.default_rng(3)
    n = int(2 * FS48)
    x = (H.inband_amplitude(10.0, 12_000.0, FS48) * _fsk2(n, rng, 1200.0)
         + H.noise(n, rng)).astype(np.complex64)
    dets = P.cyclic_proposer(x, FS48, C48, PID48,
                             regions=[(C48 - 6250, C48 + 6250)])
    assert len(dets) == 1
    d = dets[0]
    assert d.alpha_hz == pytest.approx(1200.0, abs=0.5) and d.cls == "pocsag"
    assert d.measurements["harmonics_hz"] == [2400.0, 4800.0]


def test_the_false_alarm_rate_per_call_on_noise():
    """`pfa` is the chance that one call says ANYTHING on noise alone —
    every rate, lag and offset of every probe (Bonferroni over the calls,
    each probe exact within itself)."""
    hits, trials, pfa = 0, 60, 0.2
    for seed in range(trials):
        x = H.noise(int(0.5 * FS48), np.random.default_rng(400 + seed)
                    ).astype(np.complex64)
        hits += bool(P.cyclic_proposer(x, FS48, C48, PID48, pfa=pfa,
                                       regions=[(C48 - 6250, C48 + 6250)]))
    mu = trials * pfa
    assert hits <= mu + 3.5 * math.sqrt(mu * (1 - pfa)), hits


def test_a_near_miss_is_reported_for_escalation():
    x = _fsk4(11, 1.0, -2.0)
    rep = {}
    dets = P.cyclic_proposer(x, FS48, C48, PID48, classes=VOICE,
                             regions=[(C48 - 2e3, C48 + 10e3)], report=rep)
    assert not dets
    nm = rep["near_misses"]
    assert nm and nm[0]["probe"] == "symbol_rate_line"
    assert nm[0]["statistic"] < nm[0]["threshold"] and nm[0]["p_call"] < 0.1


# ---------------------------------------------------------------------------
# OFDM by its prefix
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("regions", [
    None, [(C96 - 270e3, C96 + 270e3)], [(C96 - 150e3, C96 + 250e3)]])
def test_an_lte_like_cell_is_found_by_its_prefix(regions):
    """One second, 6 dB under the floor: the cell's prefix gives the class,
    its 14 kHz period, its carrier offset — referred to the CAPTURE's
    centre in every mode (the grid path once reported 7,066 Hz for a 900 Hz
    cell, the shift of its channel left in) — its SNR from the prefix, and
    how many cells share the channel."""
    x = _lte(1.0, -6.0, cfo=900.0)
    rep = {}
    dets = P.cyclic_proposer(x, FS96, C96, PID96, classes=["lte_dl"],
                             regions=regions, report=rep)
    assert len(dets) == 1
    d = dets[0]
    m = d.measurements
    assert d.cls == "lte_dl" and d.family == "ofdm" and d.sources == ("cyclic",)
    assert m["probe"] == "cp_probe"
    assert m["cp_lag_s"] == pytest.approx(1 / 15_000)
    assert d.alpha_hz == pytest.approx(14_000.0, rel=1e-3)
    assert m["period_detected"]
    # within the probe's own stated uncertainty (~300 Hz at −6 dB in 1 s)
    assert abs(m["cfo_hz"] - 900.0) < 3 * m["cfo_se_hz"] + 100.0
    assert m["cfo_se_hz"] < 500.0
    assert "capture's centre" in m["cfo_reference"]
    assert d.snr_db == pytest.approx(-6.0, abs=2.0)
    assert m["snr_method"].startswith("from the CP correlation")
    assert m["cp_cells"] == 1 and "time-synchronised" in m["cp_cells_limit"]
    assert abs(m["cp_cells_detail"][0]["cfo_hz"] - 900.0) < 3 * m["cfo_se_hz"] + 100
    assert "15 kHz subcarrier spacing" in m["class_words"]
    assert "PSS/SSS" in m["note"]
    assert d.integration_s == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# the polyphase channel grid
# ---------------------------------------------------------------------------
def test_the_channelizer_puts_a_tone_in_its_own_channel():
    fs = 48_000.0
    t = np.arange(int(0.5 * fs)) / fs
    x = np.exp(2j * np.pi * 6_000.0 * t).astype(np.complex64)
    Y, centres, fs_c, info = P.channelise(x, fs, 2_000.0, 4_000.0, 4)
    assert fs_c == 12_000.0 and info["M"] == 24
    p = np.mean(np.abs(Y) ** 2, axis=1)
    k = int(np.argmax(p))
    assert centres[k] == pytest.approx(6_000.0)
    others = np.delete(p, [k - 1, k, k + 1])
    assert 10 * np.log10(p[k] / others.max()) > 40.0
    assert np.all(np.abs(centres) + 2_000.0 <= P.USABLE * fs)
    assert P.probe_decimation(240_000.0, 8_300.0) == 11


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def test_inputs_are_checked_in_words():
    x = H.noise(24_000, np.random.default_rng(1)).astype(np.complex64)
    with pytest.raises(profiles.ProfileMismatch, match="sample-rate law"):
        P.cyclic_proposer(x, 96_000.0, C48, PID48)
    with pytest.raises(ValueError, match="not in the class table"):
        P.cyclic_proposer(x, FS48, C48, PID48, classes=["tetra"])
    rep = {}
    P.cyclic_proposer(np.stack([x, x]), FS48, C48, profiles.new_profile(PID48),
                      classes=VOICE, regions=[(C48 - 5e3, C48 + 5e3)],
                      report=rep)
    assert "multi-channel input (2 channels): channel 0 probed" in rep["notes"]
    assert rep["seconds"] > 0 and rep["detections"] == 0
