# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Transmit files for the cabled loop (cabled.txfiles, plan §3.5, §4.A6):
the exact bytes each transmitter plays (HackRF cs8, bladeRF SC16 Q11), the
markers a recording is lined up by, every local fallback generator, the
manifest that is the ground truth, and the refusals. Nothing here transmits.
The numbers are test fixtures, not data-sheet values."""

from __future__ import annotations

import builtins
import json

import numpy as np
import pytest

from atk_diffusion.cabled import txfiles as T
from atk_diffusion.dsp import iq

FS = 2_400_000.0


def _rng(seed=0):
    return np.random.default_rng(seed)


# -- the two transmitters' formats ----------------------------------------------
def test_hackrf_plays_signed_8_bit_pairs(rf):
    sig = [{"class": "ref_bpsk", "duration_s": 0.004, "f_offset_hz": 100e3,
            "symbol_rate_hz": 25e3}]
    tf = T.build(rf.cabled("rtlsdr_2400000_cu8") / "tx", "HackRF", FS, sig,
                 rx_rate=FS, name="h", rf=rf, seed=1, tx_center_hz=915e6)
    m = tf.manifest
    assert tf.path.name == "h.cs8" and tf.manifest_path.name == "h.manifest.json"
    raw = tf.path.read_bytes()
    assert len(raw) == 2 * m["n_samples"]
    v = np.frombuffer(raw, np.int8).astype(int)
    assert v.max() <= 90 and v.min() >= -90 and np.abs(v).max() >= 88   # 0.7 x 128
    assert m["format"] == "cs8" and m["datatype"] == "ci8"
    assert m["transmitter"] == "hackrf" and m["tx_rate"] == FS
    assert m["peak"] == pytest.approx(0.7, abs=0.01) and m["backoff"] == 0.7
    assert m["clipped_fraction"] == 0.0 and m["papr_db"] > 0
    assert m["tx_center_hz"] == 915e6 and m["rx_rate_planned"] == FS
    assert m["marker"]["kind"] == "chirp" and m["end_marker"] is not None
    import hashlib
    assert m["sha256"] == hashlib.sha256(raw).hexdigest()
    entries = rf.log.entries()
    assert any(k.endswith("tx/h.cs8") and e["kind"] == "cabled-tx"
               for k, e in entries.items())
    assert any(k.endswith("tx/h.manifest.json") for k in entries)
    lines = tf.lines()
    assert lines[0].startswith("h.cs8: ") and "for the hackrf, cs8" in lines[0]
    assert "1 signals between a chirp start marker and an end marker" == lines[1]


@pytest.mark.parametrize("fam", ["bladerf1", "bladerf2"])
def test_bladerf_plays_sc16_q11(tmp_path, fam):
    sig = [{"class": "nfm_voice", "duration_s": 0.004, "f_offset_hz": -50e3}]
    tf = T.build(tmp_path, fam, 4_000_000.0, sig, marker="pn", end_marker=False,
                 backoff=0.5, name="b")
    m = tf.manifest
    assert tf.path.suffix == ".bin" and m["format"] == "sc16q11"
    v = np.frombuffer(tf.path.read_bytes(), "<i2").astype(int)
    assert np.abs(v).max() <= 1024 and np.abs(v).max() >= 1020   # 0.5 x 2048
    back = iq.to_complex(tf.path.read_bytes(), "ci16q11")
    assert back.size == m["n_samples"]
    assert m["end_marker"] is None and m["marker"]["kind"] == "pn"
    assert m["marker"]["params"]["segments"] == 32
    assert "and no end marker" in tf.lines()[1]
    assert T.verify_file(T.load_manifest(tf.manifest_path)) == (True, "")


def test_the_layout_is_marker_gap_signals_gap_marker(tmp_path):
    sig = [{"class": "ref_qpsk", "duration_s": 0.003, "symbol_rate_hz": 20e3},
           {"native": "tone", "duration_s": 0.002, "f_offset_hz": 300e3,
            "label": "cw"}]
    tf = T.build(tmp_path, "hackrf", FS, sig, gap_s=0.001, name="l", seed=2)
    m = tf.manifest
    mk = m["marker"]["length"]
    gap = int(0.001 * FS)
    s0, s1 = m["signals"]
    assert s0["start_sample"] == mk + gap
    assert s1["start_sample"] == s0["start_sample"] + s0["count"] + gap
    assert m["end_marker"]["start_sample"] == s1["start_sample"] + s1["count"] + gap
    assert m["n_samples"] == m["end_marker"]["start_sample"] + mk + gap
    assert s1["label"] == "cw" and s1["generator"].startswith("local fallback")
    assert s0["start_s"] == pytest.approx(s0["start_sample"] / FS)
    # what is in the file at the signal's place is the signal, not silence
    x = iq.to_complex(tf.path.read_bytes(), "ci8")
    on = x[s0["start_sample"]:s0["start_sample"] + s0["count"]]
    off = x[mk + 10:mk + gap - 10]
    assert np.mean(np.abs(on) ** 2) > 100 * (np.mean(np.abs(off) ** 2) + 1e-6)


# -- refusals ----------------------------------------------------------------------
@pytest.mark.parametrize("kw, words", [
    ({"transmitter": "rtlsdr"}, "cannot be the loop's transmitter"),
    ({"transmitter": "hackrf", "tx_rate": 1e6}, "transmits at 2e\\+06"),
    ({"signals": []}, "at least one signal"),
    ({"backoff": 1.0}, "backoff is the peak"),
    ({"marker": "morse"}, "unknown marker 'morse'"),
    ({"signals": [{"class": "ref_qpsk", "duration_s": 0.001,
                   "f_offset_hz": 1.05e6, "symbol_rate_hz": 100e3}]},
     "does not fit in the transmitter's"),
])
def test_build_refusals(tmp_path, kw, words):
    args = {"transmitter": "hackrf", "tx_rate": FS,
            "signals": [{"class": "ref_bpsk", "duration_s": 0.001}]}
    args.update(kw)
    with pytest.raises(ValueError, match=words):
        T.build(tmp_path, args.pop("transmitter"), args.pop("tx_rate"),
                args.pop("signals"), **args)
    assert not list(tmp_path.glob("*.cs8"))


def test_a_signal_needs_a_class_or_a_generator():
    with pytest.raises(ValueError, match="needs a class"):
        T.make_signal({"duration_s": 0.001}, FS, _rng())


# -- markers --------------------------------------------------------------------------
def test_mseq_periods_and_unknown_degree():
    for deg in sorted(T.MSEQ_TAPS):
        s = T.mseq(deg)
        n = 2 ** deg - 1
        assert s.size == n and set(np.unique(s)) == {-1.0, 1.0}
        # maximal length: the cyclic autocorrelation is N at 0 and -1 elsewhere
        S = np.fft.fft(s.astype(np.float64))
        ac = np.real(np.fft.ifft(S * np.conj(S)))
        assert ac[0] == pytest.approx(n)
        assert np.allclose(ac[1:], -1.0, atol=1e-6)
    with pytest.raises(ValueError, match="no m-sequence taps for degree 3"):
        T.mseq(3)


def test_marker_parameters_fit_both_radios():
    c = T.default_marker_params("chirp", 8e6, rx_rate=2.4e6)
    assert c["bandwidth_hz"] == 1.2e6 and c["duration_s"] == 0.005
    p = T.default_marker_params("pn", 2.4e6)
    assert p["chip_rate_hz"] == 600e3 and p["degree"] == 10
    with pytest.raises(ValueError, match="unknown marker"):
        T.default_marker_params("morse", 2.4e6)
    with pytest.raises(ValueError, match="unknown marker"):
        T.marker_waveform("morse", 2.4e6, {})


def test_chirp_halves_sweep_opposite_ways():
    up, dn = T.chirp_parts(1e6, {"bandwidth_hz": 400e3, "duration_s": 0.002})
    assert up.size == dn.size == 2000
    fu = np.diff(np.unwrap(np.angle(up))) * 1e6 / (2 * np.pi)
    fd = np.diff(np.unwrap(np.angle(dn))) * 1e6 / (2 * np.pi)
    assert fu[0] == pytest.approx(-200e3, rel=0.01) and fu[-1] == pytest.approx(200e3, rel=0.01)
    assert fd[0] == pytest.approx(200e3, rel=0.01) and fd[-1] == pytest.approx(-200e3, rel=0.01)
    both = T.marker_waveform("chirp", 1e6, {"bandwidth_hz": 400e3,
                                            "duration_s": 0.002})
    assert both.size == 4000
    pn = T.pn_waveform(1e6, {"chip_rate_hz": 250e3, "degree": 5}, f_shift_hz=1e3)
    assert pn.size == 31 * 4 and np.allclose(np.abs(pn), 1.0)


# -- the generators -------------------------------------------------------------------
def test_rrc_is_unit_energy_and_finite_at_its_singular_points():
    h = T.rrc(0.25, 8)                       # t = ±1/(4β) = ±1 lands on a tap
    assert np.isfinite(h).all() and np.sum(h * h) == pytest.approx(1.0)
    assert np.allclose(h, h[::-1])
    assert np.sum(T.rrc(0.0, 4) ** 2) == pytest.approx(1.0)


LOCAL_KEYS = ["bpsk", "qpsk", "8psk", "16qam", "64qam", "ask4", "2fsk", "gfsk",
              "c4fm_4800", "c4fm_2400", "dmr_4800", "4fsk", "am", "nfm", "fm",
              "wfm", "tone", "noise", "ofdm", "ofdm_lte"]


@pytest.mark.parametrize("key", LOCAL_KEYS)
def test_every_local_generator_is_unit_power_and_says_its_facts(key):
    x, facts = T.local_signal(key, 240_000.0, 4096, _rng())
    assert x.dtype == np.complex64 and x.size == 4096
    assert np.mean(np.abs(x) ** 2) == pytest.approx(1.0, rel=1e-3)
    assert set(facts) == {"symbol_rate_hz", "bandwidth_hz"}
    assert facts["bandwidth_hz"] >= 0.0
    if key in ("bpsk", "qpsk", "8psk", "16qam", "64qam", "ask4", "2fsk", "gfsk",
               "c4fm_4800", "c4fm_2400", "dmr_4800", "4fsk", "ofdm", "ofdm_lte"):
        assert facts["symbol_rate_hz"] > 0
    else:
        assert facts["symbol_rate_hz"] is None


def test_local_generator_settings_and_refusal():
    _x, f = T.local_signal("qpsk", 240_000.0, 1024, _rng(), symbol_rate=30_000.0)
    assert f["symbol_rate_hz"] == pytest.approx(30_000.0)
    _x, f = T.local_signal("c4fm_2400", 48_000.0, 1024, _rng())
    assert f["symbol_rate_hz"] == pytest.approx(2400.0)
    _x, f = T.local_signal("noise", 240_000.0, 1024, _rng(), bandwidth_hz=50e3)
    assert f["bandwidth_hz"] == 50e3
    with pytest.raises(ValueError, match="cannot make 'theremin'"):
        T.local_signal("theremin", 48_000.0, 100, _rng())


def test_the_native_generator_is_used_and_said(tmp_path):
    x, lab = T.make_signal({"class": "dmr", "duration_s": 0.06,
                            "f_offset_hz": 50e3, "power_db": -6.0}, FS, _rng())
    assert lab["generator"] == "atk_diffusion.synth.native"
    assert lab["family"] == "fsk" and lab["power_rel_db"] == -6.0
    assert lab["bursts"]                       # DMR's TDMA slots
    assert lab["native_label"]["native"] == "dmr_4800"
    tf = T.build(tmp_path, "hackrf", FS, [{"class": "dmr", "duration_s": 0.06}],
                 name="d")
    s = tf.manifest["signals"][0]
    assert s["bursts_s"] and all(b[1] > 0 for b in s["bursts_s"])
    assert s["bursts_s"][0][0] >= s["start_s"]


def test_the_native_generator_ignores_a_setting_it_does_not_take():
    x, lab = T.make_signal({"class": "nfm_voice", "duration_s": 0.01,
                            "bandwidth_hz": 99e3}, FS, _rng())
    assert lab["generator"] == "atk_diffusion.synth.native"
    assert "does not take that setting" in lab["native_label"]["note"]


def test_the_local_fallback_is_used_and_said_when_native_cannot(monkeypatch):
    from atk_diffusion.synth import native

    def boom(*a, **k):
        raise RuntimeError("no such waveform today")

    monkeypatch.setattr(native, "waveform", boom)
    x, lab = T.make_signal({"class": "ref_qpsk", "duration_s": 0.002,
                            "f_offset_hz": 10e3, "bandwidth_hz": 40e3}, FS, _rng())
    assert lab["generator"].startswith("local fallback (synth.native could not "
                                       "make ref_qpsk: no such waveform today)")
    assert lab["bandwidth_hz"] == 40e3          # the spec's width wins locally
    f = np.fft.fftshift(np.fft.fftfreq(x.size, 1 / FS))
    peak = f[np.argmax(np.abs(np.fft.fftshift(np.fft.fft(x))))]
    assert abs(peak - 10e3) < 30e3              # the offset was applied

    monkeypatch.setattr(native, "waveform", lambda *a, **k: (np.zeros(3), {}))
    _x, lab = T.make_signal({"class": "ref_bpsk", "duration_s": 0.001}, FS, _rng())
    assert "wrong length or non-finite" in lab["generator"]

    monkeypatch.delattr(native, "waveform")
    _x, lab = T.make_signal({"class": "ref_bpsk", "duration_s": 0.001}, FS, _rng())
    assert "synth.native has no waveform()" in lab["generator"]


def test_the_local_fallback_when_native_cannot_be_imported(monkeypatch):
    real = builtins.__import__

    def no_native(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "atk_diffusion.synth" and fromlist and "native" in fromlist:
            raise ImportError("synth.native is absent")
        return real(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", no_native)
    _x, lab = T.make_signal({"class": "ref_bpsk", "duration_s": 0.001}, FS, _rng())
    assert "synth.native is not available (synth.native is absent)" in lab["generator"]


# -- the manifest ----------------------------------------------------------------------
def test_load_manifest_and_verify_file(tmp_path):
    tf = T.build(tmp_path, "hackrf", FS, [{"native": "tone", "duration_s": 0.001}],
                 name="v")
    d = T.load_manifest(tf.manifest_path)
    assert d["_dir"] == str(tmp_path) and T.load_manifest(d) is d
    assert json.loads(tf.manifest_path.read_text("utf-8"))["file"] == "v.cs8"
    raw = bytearray(tf.path.read_bytes())
    raw[0] ^= 0x01
    tf.path.write_bytes(bytes(raw))
    ok, why = T.verify_file(d)
    assert not ok and "hash changed" in why and "rebuild it" in why
    tf.path.unlink()
    assert T.verify_file(d) == (False, "the transmit file v.cs8 is missing")


def test_a_default_name_says_what_and_when(tmp_path):
    tf = T.build(tmp_path, "hackrf", FS, [{"native": "noise", "duration_s": 0.001,
                                            "bandwidth_hz": 100e3}])
    assert tf.path.name.startswith("loop_hackrf_2400000_") and tf.path.suffix == ".cs8"


def test_a_failing_write_log_does_not_lose_the_file(tmp_path):
    class BrokenLog:
        def record(self, *a, **k):
            raise OSError("the disk is read-only")

    tf = T.build(tmp_path, "hackrf", FS, [{"native": "tone", "duration_s": 0.001}],
                 name="w", rf=BrokenLog())
    assert tf.path.exists() and tf.manifest_path.exists()
    assert T.verify_file(T.load_manifest(tf.manifest_path)) == (True, "")
