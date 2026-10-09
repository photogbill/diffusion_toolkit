# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The native generator (plan §3.3, §4.A; DETECTION_DESIGN §9): every class
with a native key, at any rate, with labels checked against the signal itself
— the 99 % occupied band by an independent estimator, the symbol rate by the
spectral line of a nonlinearity (|x|², the FM discriminator's derivative, the
cyclic-prefix lag product, the chirp slope), the SNR by its definition."""

from __future__ import annotations

import math

import numpy as np
import pytest

from atk_diffusion.detect import classes
from atk_diffusion.synth import native as N

RTL = 2_400_000.0
HACKRF = 20_000_000.0


# -- independent measurements (not the generator's own code) ----------------
def _line(feat, fs, lo, hi):
    """Frequency and prominence of the strongest spectral line in [lo, hi]."""
    feat = np.asarray(feat)
    feat = feat - feat.mean()
    n = feat.size
    nfft = 1 << int(math.ceil(math.log2(n)) + 1)
    F = np.abs(np.fft.fft(feat * np.hanning(n), nfft))
    f = np.fft.fftfreq(nfft, 1 / fs)
    idx = np.nonzero((f >= lo) & (f <= hi))[0]
    k = idx[np.argmax(F[idx])]
    a, b, c = (np.log(F[k + j] + 1e-30) for j in (-1, 0, 1))
    d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) else 0.0
    return f[k] + d * fs / nfft, F[k] / np.median(F[idx])


def _disc(x):
    return np.angle(x[1:] * np.conj(x[:-1]))


def _inband(x, fs, lo, hi):
    from scipy.signal import welch
    nper = 1 << int(math.log2(min(x.size, 1 << 16)))
    f, p = welch(x, fs=fs, nperseg=nper, return_onesided=False, detrend=False)
    return p[(f >= lo) & (f <= hi)].sum() / p.sum()


def _rate_for(cls):
    return RTL if N.can_generate(cls, RTL)[0] else HACKRF


# -- every class ---------------------------------------------------------------
@pytest.mark.parametrize("cls", N.supported())
def test_every_native_class_generates_with_a_complete_label(cls):
    fs = _rate_for(cls)
    n = int(fs * (0.02 if fs == RTL else 0.002))
    rng = np.random.default_rng(11)
    off = 0.0 if cls == "dc_spike" else 0.03 * fs
    ok, why = N.can_generate(cls, fs, carrier_offset_hz=off)
    assert ok, why
    x, lab = N.generate(cls, fs, n, 12.0, rng, carrier_offset_hz=off)
    assert x.dtype == np.complex64 and x.size == n and np.isfinite(x).all()
    want = {"cls", "family", "bandwidth_hz", "symbol_rate_hz",
            "carrier_offset_hz", "snr_db", "sample_start", "sample_count",
            "f_lo_hz", "f_hi_hz"}
    assert want <= set(lab)
    c = classes.get(cls)
    assert lab["cls"] == cls and lab["family"] == c.family
    assert lab["generator"] == "native" and lab["native"] == c.native
    assert 0 <= lab["sample_start"] and lab["sample_start"] + lab["sample_count"] <= n
    if cls == "noise":
        assert math.isnan(lab["snr_db"]) and lab["bandwidth_hz"] == 0.0
        return
    assert lab["snr_db"] == 12.0
    assert -fs / 2 <= lab["f_lo_hz"] < lab["f_hi_hz"] <= fs / 2
    assert lab["bandwidth_hz"] == pytest.approx(lab["f_hi_hz"] - lab["f_lo_hz"])
    assert lab["f_lo_hz"] <= lab["carrier_offset_hz"] + 1 / (n / fs) + 1
    assert lab["carrier_offset_hz"] <= lab["f_hi_hz"] + 1 / (n / fs) + 1 \
        or cls == "atsc"                       # VSB: carrier near the low edge


def test_supported_is_every_class_with_a_native_key():
    assert set(N.supported()) == {c.name for c in classes.CLASSES if c.native}
    assert {classes.get(c).native for c in N.supported()} == set(N.KINDS)


# -- the occupied band -----------------------------------------------------------
BW_CASES = [  # cls, fs, seconds, params, expected bandwidth range (Hz)
    ("p25", RTL, 0.2, {}, (5.5e3, 9.5e3)),
    ("dmr", RTL, 0.2, {}, (6.5e3, 9.8e3)),
    ("nxdn96", RTL, 0.2, {}, (7.0e3, 10.0e3)),
    ("nxdn48", RTL, 0.2, {}, (3.0e3, 5.0e3)),
    ("pocsag", RTL, 0.3, {"baud": 1200}, (8.0e3, 13.0e3)),
    ("flex", RTL, 0.2, {"symbol_rate": 1600, "levels": 2}, (9.0e3, 14.0e3)),
    ("nfm_voice", RTL, 0.6, {}, (3.0e3, 12.0e3)),
    ("noaa_wx", RTL, 0.6, {}, (5.0e3, 16.0e3)),
    ("fm_broadcast", RTL, 0.1, {}, (150e3, 260e3)),
    ("lora", RTL, 0.05, {"bw": 125e3, "sf": 7}, (118e3, 145e3)),
    ("lte_dl", RTL, 0.01, {"n_rb": 6}, (1.03e6, 1.10e6)),
    ("lte_dl", HACKRF, 0.003, {"n_rb": 50}, (8.7e6, 9.1e6)),
    ("nr_dl", HACKRF, 0.003, {"n_rb": 24}, (8.4e6, 8.7e6)),
    ("wifi_24", HACKRF, 0.001, {"data_symbols": 120}, (15.5e6, 17.5e6)),
    ("ble", HACKRF, 0.001, {"pdu_bytes": 37}, (0.9e6, 1.3e6)),
    ("atsc", HACKRF, 0.003, {}, (5.2e6, 6.0e6)),
    ("drone_fpv_analog", HACKRF, 0.003, {}, (13e6, 19.5e6)),
    ("drone_digital", HACKRF, 0.01, {}, (8.7e6, 9.1e6)),
    ("gnss_jamming", RTL, 0.01, {}, (1.9e6, 2.4e6)),
    ("ref_qpsk", RTL, 0.02, {"symbol_rate": 250e3, "rolloff": 0.35}, (250e3, 337e3)),
    ("ref_ofdm", RTL, 0.02, {"bandwidth_hz": 500e3}, (470e3, 520e3)),
]


@pytest.mark.parametrize("cls,fs,sec,params,rng_bw", BW_CASES,
                         ids=[f"{c[0]}@{c[1] / 1e6:g}M" for c in BW_CASES])
def test_occupied_band_holds_99_percent_and_is_physical(cls, fs, sec, params,
                                                         rng_bw):
    x, lab = N.waveform(cls, fs, int(fs * sec), np.random.default_rng(5),
                        carrier_offset_hz=0.05 * fs * (cls != "drone_fpv_analog"),
                        params=params)
    seg = np.concatenate([x[s:s + k] for s, k in lab["bursts"]])
    frac = _inband(seg, fs, lab["f_lo_hz"], lab["f_hi_hz"])
    assert 0.982 <= frac <= 0.997, frac
    assert rng_bw[0] <= lab["bandwidth_hz"] <= rng_bw[1], lab["bandwidth_hz"]


# -- the symbol rate, by its spectral line --------------------------------------
LINE_CASES = [  # cls, fs, seconds, params, feature
    ("p25", RTL, 1.0, {}, "disc"),          # Nyquist pulse: a weak line
    ("nxdn96", 2_048_000.0, 0.3, {}, "disc"),
    ("nxdn48", 3_200_000.0, 0.3, {}, "disc"),
    ("dmr", RTL, 0.3, {}, "disc_bursts"),
    ("dmr", HACKRF, 0.15, {}, "disc_bursts"),
    ("dmr", RTL, 0.5, {"slots": 1}, "disc_bursts"),
    ("pocsag", RTL, 0.5, {"baud": 512}, "disc"),
    ("pocsag", 1_000_000.0, 0.3, {"baud": 2400}, "disc"),
    ("flex", RTL, 0.3, {"symbol_rate": 3200, "levels": 4}, "disc"),
    ("ble", HACKRF, 0.0004, {"pdu_bytes": 37}, "disc_burst"),
    ("ble", RTL, 0.0004, {"pdu_bytes": 37}, "disc_burst"),
    ("adsb", RTL, 0.0002, {"frame": "long"}, "edges"),
    ("adsb", HACKRF, 0.0002, {"frame": "long"}, "edges"),
    ("lte_dl", RTL, 0.02, {"n_rb": 6}, "cp"),
    ("lte_dl", HACKRF, 0.004, {"n_rb": 50}, "cp"),
    ("nr_dl", HACKRF, 0.004, {"n_rb": 24}, "cp"),
    ("wifi_24", HACKRF, 0.0006, {"data_symbols": 100}, "cp_burst"),
    ("drone_digital", HACKRF, 0.04, {}, "cp"),     # gating keeps the timing
    ("ref_ofdm", RTL, 0.02, {"bandwidth_hz": 500e3}, "cp"),
    ("ref_bpsk", RTL, 0.01, {"symbol_rate": 100e3, "rolloff": 0.35}, "abs2"),
    ("ref_qpsk", RTL, 0.01, {"symbol_rate": 250e3, "rolloff": 0.35}, "abs2"),
    ("ref_8psk", 3_200_000.0, 0.01, {"symbol_rate": 100e3, "rolloff": 0.35}, "abs2"),
    ("ref_16qam", 2_048_000.0, 0.01, {"symbol_rate": 64e3, "rolloff": 0.35}, "abs2"),
    ("ref_64qam", RTL, 0.01, {"symbol_rate": 77777.7, "rolloff": 0.4}, "abs2"),
    ("ref_ask", RTL, 0.01, {"symbol_rate": 100e3, "rolloff": 0.35}, "abs2"),
    ("ref_2fsk", RTL, 0.01, {"symbol_rate": 50e3}, "disc"),
    ("ref_gfsk", RTL, 0.01, {"symbol_rate": 50e3}, "disc"),
]


def _feature(x, lab, fs, feat):
    """The nonlinearity whose spectrum has a line at the symbol rate, and
    the rate it is sampled at. FSK-family signals are first decimated to
    ~8 samples a symbol (the discriminator's derivative is then dominated by
    symbol transitions, not by the oversampling)."""
    rs = lab["symbol_rate_hz"]
    longest = max(lab["bursts"], key=lambda b: b[1])
    if feat.endswith("_burst"):
        x = x[longest[0]:longest[0] + longest[1]]
    if feat.startswith("disc"):
        from atk_diffusion.dsp import resample
        d = max(1, int(fs // (8 * rs)))
        y, f2 = resample.decimate(x, d, fs) if d > 1 else (x, fs)
        f = np.abs(np.diff(_disc(y))) ** 2
        if feat == "disc_bursts":         # burst edges are not symbol edges
            m = np.zeros(f.size, bool)
            guard = 3 * int(f2 / rs)
            for s, k in lab["bursts"]:
                a, b = int(s / d) + guard, int((s + k) / d) - guard
                m[max(a, 0):max(b, 0)] = True
            f = np.where(m, f, 0.0)
        return f, f2
    if feat == "abs2":
        return np.abs(x) ** 2, fs
    if feat == "edges":                   # PPM: a transition mid-bit, always
        s, k = longest
        return np.abs(np.diff(np.abs(x[s:s + k]))) ** 2, fs
    lag = int(round(lab["params"]["cp_lag_s"] * fs))   # the cyclic prefix
    return x[lag:] * np.conj(x[:-lag]), fs


@pytest.mark.parametrize("cls,fs,sec,params,feat", LINE_CASES,
                         ids=[f"{c[0]}@{c[1] / 1e6:g}M" for c in LINE_CASES])
def test_symbol_rate_line_matches_the_label(cls, fs, sec, params, feat):
    x, lab = N.waveform(cls, fs, int(fs * sec), np.random.default_rng(3),
                        params=params)
    rs = lab["symbol_rate_hz"]
    assert rs > 0
    f, f2 = _feature(x, lab, fs, feat)
    line, prom = _line(f, f2, 0.6 * rs, 1.4 * rs)
    tol = 0.002 if feat in ("abs2", "disc", "disc_bursts", "disc_burst") else 0.006
    assert abs(line - rs) / rs < tol, (line, rs)
    assert prom > 3.0


@pytest.mark.parametrize("cls,fs,params,sweep", [
    ("lora", RTL, {"bw": 125e3, "sf": 7}, 125e3),
    ("lora", RTL, {"bw": 250e3, "sf": 10}, 250e3),
    ("lora", HACKRF, {"bw": 500e3, "sf": 8}, 500e3),
    ("gnss_jamming", RTL, {}, 2e6),
    ("gnss_jamming", HACKRF, {"sweep_hz": 8e6}, 8e6)])
def test_chirp_slope_gives_the_lora_and_jammer_rates(cls, fs, params, sweep):
    """A chirp's frequency slope is sweep × rate: LoRa's symbol rate is
    BW / 2^SF; the jammer's 'symbol rate' is its sweep repetition rate."""
    sec = 0.3 if params.get("sf", 0) >= 10 else 0.02
    x, lab = N.waveform(cls, fs, int(fs * sec), np.random.default_rng(3),
                        params=params)
    s, k = max(lab["bursts"], key=lambda b: b[1])
    fi = _disc(x[s:s + k]) * fs / (2 * np.pi)
    sl = np.abs(np.diff(fi)) * fs
    sl = sl[sl < 3 * np.median(sl)]
    assert np.median(sl) / sweep == pytest.approx(lab["symbol_rate_hz"], rel=1e-3)


def test_broadcast_fm_carries_its_19_khz_pilot():
    x, lab = N.waveform("fm_broadcast", RTL, int(RTL * 0.1),
                        np.random.default_rng(2))
    f, prom = _line(_disc(x) * RTL / (2 * np.pi), RTL, 15e3, 23e3)
    assert f == pytest.approx(19_000.0, abs=20) and prom > 5
    assert lab["symbol_rate_hz"] == 0.0


def test_atsc_pilot_sits_at_the_lower_edge():
    fs = HACKRF
    x, lab = N.waveform("atsc", fs, int(fs * 0.003), np.random.default_rng(2),
                        carrier_offset_hz=1e6)
    spec = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(x.size, 1 / fs))
    pk = f[np.argmax(spec)]
    rs = float(N.ATSC_RS)
    assert pk == pytest.approx(1e6 - rs / 4, abs=2e3)
    assert lab["symbol_rate_hz"] == pytest.approx(rs)
    # the 99 % lower edge sits just under the pilot: the vestige below it
    # carries well under 0.5 % of the power
    assert 0 < pk - lab["f_lo_hz"] < 0.31e6


# -- the SNR definition -----------------------------------------------------------
@pytest.mark.parametrize("cls,fs,sec", [("p25", RTL, 0.2),
                                        ("lte_dl", RTL, 0.01),
                                        ("wifi_24", HACKRF, 0.0008)])
def test_snr_is_signal_over_noise_in_the_occupied_band(cls, fs, sec):
    snr = 10.0
    x, lab = N.generate(cls, fs, int(fs * sec), snr, np.random.default_rng(4),
                        noise_dbfs=-30.0, params={"data_symbols": 150}
                        if cls == "wifi_24" else None)
    pn = 1e-3
    on = np.concatenate([np.arange(s, s + k) for s, k in lab["bursts"]])
    seg = x[on]
    X = np.fft.fft(seg)
    f = np.fft.fftfreq(seg.size, 1 / fs)
    band = (f >= lab["f_lo_hz"]) & (f <= lab["f_hi_hz"])
    p_band = float(np.sum(np.abs(X[band]) ** 2)) / seg.size ** 2
    n0b = pn / fs * lab["bandwidth_hz"]
    est = 10 * math.log10((p_band - n0b) / 0.99 / n0b)   # 99 % of it is in band
    assert est == pytest.approx(snr, abs=0.25)


def test_noise_class_is_noise_at_the_stated_level():
    x, lab = N.generate("noise", RTL, 200_000, 99.0, np.random.default_rng(1),
                        noise_dbfs=-25.0)
    assert math.isnan(lab["snr_db"]) and lab["bandwidth_hz"] == 0.0
    assert 10 * math.log10(np.mean(np.abs(x) ** 2)) == pytest.approx(-25.0, abs=0.05)


def test_tone_and_dc_spike():
    fs, n = RTL, 120_000
    x, lab = N.waveform("spur", fs, n, np.random.default_rng(1),
                        carrier_offset_hz=123_456.0)
    f = np.fft.fftfreq(n, 1 / fs)[np.argmax(np.abs(np.fft.fft(x)))]
    assert f == pytest.approx(123_456.0, abs=fs / n)
    assert lab["bandwidth_hz"] == pytest.approx(fs / n)       # 1/T
    x, lab = N.waveform("dc_spike", fs, n, np.random.default_rng(1),
                        carrier_offset_hz=5e4)
    assert lab["carrier_offset_hz"] == 0.0 and "ignored" in lab["params"]["note"]
    assert abs(np.mean(x)) > 0.9


# -- refusals, the clip, determinism, any rate ------------------------------------
def test_signals_wider_than_the_rate_are_refused_in_words():
    with pytest.raises(N.SynthRefusal, match=r"2\.4 MS/s holds at most"):
        N.waveform("lte_dl", RTL, 1000, np.random.default_rng(0),
                   params={"n_rb": 50})
    ok, why = N.can_generate("wifi_24", RTL)
    assert not ok and "wide" in why and "2.4 MS/s" in why
    ok, why = N.can_generate("p25", RTL, carrier_offset_hz=1.198e6)
    assert not ok and "wrap" in why
    with pytest.raises(N.SynthRefusal, match="not in the class table"):
        N.waveform("smoke_signals", RTL, 100, np.random.default_rng(0))
    with pytest.raises(N.SynthRefusal, match="does not take 'baudrate'"):
        N.waveform("pocsag", RTL, 100, np.random.default_rng(0),
                   params={"baudrate": 1200})


def test_clip_shows_the_slice_a_receiver_would_hear():
    x, lab = N.waveform("lte_dl", RTL, 24_000, np.random.default_rng(0),
                        carrier_offset_hz=2.0e6, params={"n_rb": 50},
                        clip=True)
    assert lab["clipped"] and lab["carrier_offset_hz"] == 2.0e6
    assert lab["bandwidth_hz"] < RTL
    assert lab["f_hi_hz"] <= RTL / 2 and lab["f_lo_hz"] > -RTL / 2
    vis = lab["params"]["visible_power_fraction"]
    assert 0.05 < vis < 0.35            # ~1.2 MHz of a 9 MHz carrier is in view


def test_same_seed_same_signal():
    a = N.generate("dmr", RTL, 50_000, 5.0, np.random.default_rng(42))
    b = N.generate("dmr", RTL, 50_000, 5.0, np.random.default_rng(42))
    c = N.generate("dmr", RTL, 50_000, 5.0, np.random.default_rng(43))
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]
    assert not np.array_equal(a[0], c[0])


def test_an_awkward_rate_reports_the_rate_it_realised():
    """2 000 003 S/s cannot hold 4800 Bd exactly within the resampler's
    factors: the label says what was actually made, and the line agrees."""
    fs = 2_000_003.0
    x, lab = N.waveform("p25", fs, int(fs * 1.0), np.random.default_rng(3))
    rs = lab["symbol_rate_hz"]
    assert rs == pytest.approx(4800.0, rel=1e-4) and rs != 4800.0
    f, f2 = _feature(x, lab, fs, "disc")
    line, _ = _line(f, f2, 0.6 * rs, 1.4 * rs)
    assert abs(line - rs) < 1.0


# -- structure ----------------------------------------------------------------------
def test_dmr_bursts_are_27_5_ms_on_a_30_ms_grid():
    fs = RTL
    x, lab = N.waveform("dmr", fs, int(fs * 0.2), np.random.default_rng(9))
    full = [b for b in lab["bursts"] if 0 < b[0] and b[0] + b[1] < x.size]
    assert len(full) >= 4
    assert all(b[1] == pytest.approx(0.0275 * fs, abs=2) for b in full)
    gaps = np.diff([b[0] for b in full])
    assert np.allclose(gaps, 0.030 * fs, atol=2)
    env = np.abs(x)
    s, k = full[0]
    assert env[s + k + int(0.001 * fs)] < 1e-3          # off between bursts
    x1, lab1 = N.waveform("dmr", fs, int(fs * 0.25), np.random.default_rng(9),
                          params={"slots": 1})
    starts = [b[0] for b in lab1["bursts"] if b[1] > 0.02 * fs]
    assert np.allclose(np.diff(starts), 0.060 * fs, atol=2)


def test_adsb_squitter_decodes_with_a_valid_crc():
    fs = HACKRF
    x, lab = N.waveform("adsb", fs, int(fs * 0.0003), np.random.default_rng(8),
                        params={"frame": "long"})
    s, k = lab["bursts"][0]
    env = np.abs(x[s:s + k])
    us = fs / 1e6
    bits = []
    for i in range(112):
        a = env[int((8 + i) * us):int((8.5 + i) * us)].sum()
        b = env[int((8.5 + i) * us):int((9 + i) * us)].sum()
        bits.append(1 if a > b else 0)
    assert bits[:5] == [1, 0, 0, 0, 1]                   # DF17
    crc = N.crc24(bits[:88])
    assert crc == int("".join(map(str, bits[88:])), 2)


def test_pocsag_codewords_are_valid():
    bits = N._pocsag_bits(np.random.default_rng(1), 2)
    assert list(bits[:8]) == [1, 0, 1, 0, 1, 0, 1, 0] and bits.size == 576 + 2 * 544
    words = [int("".join(map(str, bits[576 + 32 * i:576 + 32 * (i + 1)])), 2)
             for i in range(34)]
    assert words[0] == N.POCSAG_SYNC and words[17] == N.POCSAG_SYNC
    for w in words:
        if w == N.POCSAG_SYNC:
            continue
        assert bin(w).count("1") % 2 == 0                  # even parity
        r = w >> 1
        for i in range(30, 9, -1):                          # BCH(31,21) syndrome
            if r & (1 << i):
                r ^= 0x769 << (i - 10)
        assert r == 0


def test_lte_carries_its_pss():
    """The PSS (Zadoff-Chu, root by N_ID2) is where the standard puts it:
    correlate the received 6-RB carrier against all three roots."""
    fs = 1_920_000.0                                     # LTE's own 1.4 MHz rate
    x, lab = N.waveform("lte_dl", fs, int(fs * 0.012), np.random.default_rng(6),
                        params={"n_rb": 6, "cell_id": 7})
    best = {}
    for u in (25, 29, 34):
        d = N._lte_pss(u)
        X = np.zeros(128, complex)
        X[np.r_[np.arange(-31, 0), np.arange(1, 32)] % 128] = d
        ref = np.fft.ifft(X)
        c = np.abs(np.correlate(x, ref, "valid"))
        best[u] = c.max() / np.median(c)
    assert max(best, key=best.get) == (25, 29, 34)[7 % 3]
    assert best[(25, 29, 34)[7 % 3]] > 8
