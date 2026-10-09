# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The cabled loop end to end (plan §3.5, §4.A6): transmit files, the plan
and its exact commands, execution only on the cable, and labels aligned into
the recording. The receiver/transmitter numbers are TEST FIXTURES, not
data-sheet values."""

from __future__ import annotations

import json

import numpy as np
import pytest

from atk_diffusion import profiles, sigmf
from atk_diffusion.cabled import loop as L
from atk_diffusion.cabled import safety as S
from atk_diffusion.cabled import txfiles as T
from atk_diffusion.dsp import iq

RX = "rtlsdr_2400000_cu8"
FS = 2_400_000.0
F0 = 915e6
FIXTURE = "test fixture — not a data-sheet value"

SIGNALS = [{"class": "ref_qpsk", "duration_s": 0.02, "f_offset_hz": 200e3,
            "symbol_rate_hz": 50e3},
           {"class": "dmr", "duration_s": 0.03, "f_offset_hz": -300e3},
           {"class": "nfm_voice", "duration_s": 0.02, "f_offset_hz": 100e3,
            "power_db": -6.0}]


def _rx(pid=RX):
    p = profiles.new_profile(pid)
    p.safe_input = profiles.SafeInput(10.0, FIXTURE, "2026-10-08")
    return p


def _setup(**kw):
    d = dict(transmitter="hackrf", receiver_profile=RX, frequency_hz=F0,
             tx_gain=0.0, tx_power=S.TxPower(47.0, 10.0, FIXTURE, "2026-10-08"),
             attenuation_db=30.0, cable_loss_db=1.0, cabled=True, dc_block=True)
    d.update(kw)
    return S.LoopSetup(**d)


def _txfile(rf, marker="chirp", transmitter="hackrf", name="t"):
    return T.build(rf.cabled(RX) / "tx", transmitter, FS, SIGNALS, rx_rate=FS,
                   marker=marker, name=name, rf=rf, seed=3)


# -- transmit files ----------------------------------------------------------------
@pytest.mark.parametrize("degree", sorted(T.MSEQ_TAPS))
def test_m_sequences_are_maximal(degree):
    s = T.mseq(degree)
    n = 2 ** degree - 1
    assert s.size == n
    ac = np.array([np.dot(s, np.roll(s, k)) for k in range(1, min(n, 64))])
    assert np.all(ac == -1)                     # the two-valued autocorrelation


def test_hackrf_file_is_cs8_scaled_below_full_scale_with_its_manifest(rf):
    tf = _txfile(rf)
    m = tf.manifest
    raw = tf.path.read_bytes()
    assert tf.path.suffix == ".cs8" and len(raw) == 2 * m["n_samples"]
    v = np.frombuffer(raw, np.int8)
    assert np.abs(v).max() <= round(0.7 * 128) + 1 and m["clipped_fraction"] == 0.0
    assert m["format"] == "cs8" and m["transmitter"] == "hackrf"
    assert T.verify_file(T.load_manifest(tf.manifest_path)) == (True, "")
    sig = m["signals"]
    assert [s["class"] for s in sig] == ["ref_qpsk", "dmr", "nfm_voice"]
    assert sig[0]["start_sample"] == m["marker"]["length"] + int(0.02 * FS)
    assert sig[0]["symbol_rate_hz"] == pytest.approx(50e3)
    assert sig[1]["family"] == "fsk" and sig[0]["generator"]
    assert m["end_marker"]["start_sample"] > sig[-1]["start_sample"]
    assert rf.verify(tf.path)[0] and rf.verify(tf.manifest_path)[0]
    assert "the DAC does not clip" in tf.lines()[-1]


def test_bladerf_file_is_sc16_q11(rf):
    tf = T.build(rf.cabled(RX) / "tx", "bladerf1", FS, SIGNALS[:1], name="b")
    v = np.frombuffer(tf.path.read_bytes(), "<i2")
    assert tf.path.suffix == ".bin" and tf.manifest["format"] == "sc16q11"
    assert np.abs(v).max() <= 2047 and np.abs(v).max() > 1000
    back = iq.to_complex(tf.path.read_bytes(), "ci16q11")
    assert np.max(np.abs(back.real)) == pytest.approx(0.7, abs=0.01)


def test_transmit_files_refuse_what_cannot_be_played(rf):
    with pytest.raises(ValueError, match="cannot be the loop's transmitter"):
        T.build(rf.root, "rtlsdr", FS, SIGNALS)
    with pytest.raises(ValueError, match="transmits at 2e\\+06–2e\\+07"):
        T.build(rf.root, "hackrf", 1e6, SIGNALS)
    with pytest.raises(ValueError, match="does not fit"):
        T.build(rf.root, "hackrf", FS, [{"class": "ref_qpsk", "duration_s": 0.01,
                                         "f_offset_hz": 1.1e6,
                                         "symbol_rate_hz": 100e3}])


def test_the_native_generator_integration(rf):
    pytest.importorskip("atk_diffusion.synth.native",
                        reason="synth.native (another engineer's module) is not "
                               "importable yet")
    tf = T.build(rf.cabled(RX) / "tx", "hackrf", FS, SIGNALS, name="n")
    sig = tf.manifest["signals"]
    assert [s["generator"] for s in sig] == ["atk_diffusion.synth.native"] * 3
    assert sig[0]["symbol_rate_hz"] == pytest.approx(50e3)      # asked for
    assert sig[1]["symbol_rate_hz"] == pytest.approx(4800.0)    # DMR's own
    assert sig[1]["bursts_s"]                                   # TDMA on-time
    base, pre, drift = _record(rf, tf, name="rxn")
    res = L.align_labels(base, tf.manifest_path, rf=rf, setup=_setup())
    anns = sigmf.read_meta(res.capture)["annotations"]
    dmr = [a for a in anns if a["core:label"] == "dmr"]
    assert dmr and sum(a["core:sample_count"] for a in dmr) < \
        sig[1]["duration_s"] * FS                     # only the bursts


# -- the plan and its commands -------------------------------------------------------
def test_plan_builds_the_exact_hackrf_command_with_the_amp_off(rf):
    tf = _txfile(rf)
    plan = L.plan_run(_setup(), _rx(), tf.manifest_path)
    assert plan.runnable
    assert plan.args == ["hackrf_transfer", "-t", str(tf.path), "-f", "915000000",
                         "-s", "2400000", "-x", "0", "-a", "0"]
    assert "-R" not in plan.args                       # played once, never looped
    j = plan.to_json()
    assert j["runnable"] and j["tx_sha256"] == tf.manifest["sha256"]
    assert any("no antenna" in s for s in plan.steps())


def test_plan_builds_the_bladerf_script(rf):
    tf = T.build(rf.cabled(RX) / "tx", "bladerf1", FS, SIGNALS[:1], name="b")
    s = _setup(transmitter="bladerf1", tx_gain_min=-20.0, tx_gain_max=60.0,
               tx_gain=-20.0, tx_serial="abc123")
    plan = L.plan_run(s, _rx(), tf.manifest_path)
    assert plan.runnable, plan.lines()
    assert plan.args[:3] == ["bladeRF-cli", "-d", "*:serial=abc123"]
    script = plan.args[-1]
    assert plan.args[-2] == "-e"
    for part in ("set frequency tx 915000000", "set samplerate tx 2400000",
                 "set gain tx -20", f'tx config file="{tf.path}" format=bin',
                 "repeat=1", "tx start", "tx wait"):
        assert part in script


def test_a_refused_verdict_or_a_changed_file_is_not_runnable(rf):
    tf = _txfile(rf)
    plan = L.plan_run(_setup(cabled=False), _rx(), tf.manifest_path)
    assert not plan.runnable
    assert any(s.startswith("REFUSED: the cable is not confirmed")
               for s in plan.lines())
    raw = bytearray(tf.path.read_bytes())
    raw[100] ^= 1
    tf.path.write_bytes(bytes(raw))
    plan = L.plan_run(_setup(), _rx(), tf.manifest_path)
    assert not plan.runnable and "hash changed" in " ".join(plan.refusals)
    tf2 = T.build(rf.cabled(RX) / "tx", "bladerf1", FS, SIGNALS[:1], name="b2")
    plan = L.plan_run(_setup(), _rx(), tf2.manifest_path)
    assert any("made for the bladerf1" in r for r in plan.refusals)


# -- execution: only on the cable, only when asked --------------------------------------
def test_execute_refuses_without_confirmation_and_defaults_to_a_dry_run(rf):
    tf = _txfile(rf)
    plan = L.plan_run(_setup(), _rx(), tf.manifest_path)
    calls = []

    def runner(args):
        calls.append(args)
        return 0, "ok"
    r = L.execute(plan, runner=runner)
    assert not r["ran"] and "Nothing was transmitted" in r["why"]
    r = L.execute(plan, confirm_cabled="yes", runner=runner)
    assert not r["ran"]
    r = L.execute(plan, confirm_cabled=True, runner=runner)
    assert not r["ran"] and r["dry_run"] and "Dry run" in r["why"]
    r = L.execute(plan, confirm_cabled=True, dry_run=False)
    assert not r["ran"] and "never starts a transmitter on its own" in r["why"]
    assert calls == []
    ramp = S.Ramp.for_profile(rf, RX)
    r = L.execute(plan, confirm_cabled=True, dry_run=False, runner=runner,
                  ramp=ramp)
    assert r["ran"] and r["returncode"] == 0 and calls == [plan.args]
    assert len(ramp.steps(_setup())) == 1        # the run is on the record


def test_execute_rechecks_the_ramp_at_the_moment_of_running(rf):
    tf = _txfile(rf)
    plan = L.plan_run(_setup(tx_gain=20.0), _rx(), tf.manifest_path)
    assert plan.runnable                         # no ramp given at planning
    r = L.execute(plan, confirm_cabled=True, dry_run=False,
                  runner=lambda a: (0, ""), ramp=S.Ramp.for_profile(rf, RX))
    assert not r["ran"] and "minimum TX gain" in r["why"]


# -- the recording: measure and align ---------------------------------------------------
def _spans(manifest):
    """(class, start_s, duration_s) of every labelled on-time in a manifest:
    one per burst where the generator reports bursts, else one per signal."""
    out = []
    for sig in manifest["signals"]:
        for b0, bd in sig.get("bursts_s") or [[sig["start_s"], sig["duration_s"]]]:
            out.append((sig["class"], b0, bd))
    return sorted(out, key=lambda t: t[1])


def _record(rf, tf, *, cfo=2_000.0, drift=10e-6, amp=0.1, noise=0.01, pre_s=0.1,
            name="rx", profile=RX, datatype="cu8", hw="RTL-SDR", seed=0):
    y = iq.to_complex(tf.path.read_bytes(), tf.manifest["datatype"])
    n = y.size
    idx = np.arange(int(n / (1 + drift))) * (1 + drift)
    yd = np.interp(idx, np.arange(n), y.real) + 1j * np.interp(idx, np.arange(n),
                                                                y.imag)
    pre = int(pre_s * FS)
    x = np.concatenate([np.zeros(pre), yd, np.zeros(int(0.02 * FS))])
    t = np.arange(x.size) / FS
    rng = np.random.default_rng(seed)
    x = amp * x * np.exp(2j * np.pi * cfo * t) + noise * (
        rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size))
    base = rf.captures(profile) / name
    sigmf.write_pair(base, x.astype(np.complex64), FS, F0, datatype=datatype,
                     extra_global={"atk:receiver_profile": profile}, hw=hw)
    return base, pre, drift


def test_align_labels_writes_cabled_annotations_with_every_loop_number(rf):
    tf = _txfile(rf)
    base, pre, drift = _record(rf, tf)
    profiles.save_profile(rf, _rx())
    setup = _setup()
    res = L.align_labels(base, tf.manifest_path, rf=rf, setup=setup)
    spans = _spans(tf.manifest)
    assert res.found and res.labels == len(spans) >= 3, res.lines()
    assert res.cfo_hz == pytest.approx(2_000.0, abs=30.0)
    assert res.drift_ppm == pytest.approx(-10.0, abs=2.0)
    # the original is untouched; the labelled copy is under rf.cabled(profile)
    assert sigmf.read_meta(base)["annotations"] == []
    assert str(rf.cabled(RX)) in res.capture
    meta = sigmf.read_meta(res.capture)
    assert sigmf.validate(meta) == []
    anns = meta["annotations"]
    offsets = {s["class"]: s["f_offset_hz"] for s in tf.manifest["signals"]}
    for a, (cls, b0, _bd) in zip(anns, spans):
        truth = pre + b0 * FS / (1 + drift)
        assert abs(a["core:sample_start"] - truth) <= 2
        assert a["atk:source"] == "cabled" and a["core:label"] == cls
        assert a["core:freq_lower_edge"] < F0 + offsets[cls] + 2_000 \
            < a["core:freq_upper_edge"]
        assert a["atk:snr_db"] > 10.0
    g = meta["global"]
    assert g["atk:tx_power_dbm"] == pytest.approx(-37.0)
    assert g["atk:attenuation_db"] == 30.0 and g["atk:cable_loss_db"] == 1.0
    assert g["atk:splitter_loss_db"] == 0.0
    assert g["atk:expected_input_dbm"] == pytest.approx(-68.0)
    assert g["atk:tx_file_sha256"] == tf.manifest["sha256"]
    assert g["atk:alignment"]["end_marker_found"] is True
    assert rf.verify(sigmf.meta_path(res.capture))[0]
    assert rf.verify(sigmf.data_path(res.capture))[0]
    assert "cabled labels written" in res.lines()[1]


def test_the_pn_marker_aligns_too(rf):
    tf = _txfile(rf, marker="pn", name="p")
    base, pre, drift = _record(rf, tf, cfo=1_000.0, name="rxp")
    res = L.align_labels(base, tf.manifest_path, rf=rf, setup=_setup(),
                         verdict=S.check(_setup(), _rx()), search_s=0.12)
    spans = _spans(tf.manifest)
    assert res.found and res.labels == len(spans)
    assert res.cfo_hz == pytest.approx(1_000.0, abs=50.0)
    a = sigmf.read_meta(res.capture)["annotations"][0]
    assert abs(a["core:sample_start"] - (pre + spans[0][1] * FS / (1 + drift))) <= 2


def test_no_marker_no_labels(rf):
    tf = _txfile(rf)
    rng = np.random.default_rng(5)
    x = 0.01 * (rng.standard_normal(400_000) + 1j * rng.standard_normal(400_000))
    base = rf.captures(RX) / "empty"
    sigmf.write_pair(base, x.astype(np.complex64), FS, F0, datatype="cu8",
                     extra_global={"atk:receiver_profile": RX})
    res = L.align_labels(base, tf.manifest_path, rf=rf, setup=_setup())
    assert not res.found and res.labels == 0
    assert "worse than none" in res.lines()[0]
    assert sigmf.read_meta(base)["annotations"] == []


def test_alignment_refuses_another_receivers_recording(rf):
    tf = _txfile(rf)
    base, _pre, _d = _record(rf, tf, profile="hackrf_2400000_ci8",
                             datatype="ci8", hw="HackRF One", name="h")
    with pytest.raises(profiles.ProfileMismatch, match="this cabled run"):
        L.align_labels(base, tf.manifest_path, rf=rf, setup=_setup())


def test_measure_rx_feeds_the_ramp(rf):
    tf = _txfile(rf)
    base, _pre, _d = _record(rf, tf, name="m1")
    r = L.measure_rx(base, tf.manifest_path)
    assert r["found"] and r["clipped_fraction"] == 0.0
    assert r["above_noise_db"] > S.MEASURABLE_DB
    ramp = S.Ramp.for_profile(rf, RX)
    s0 = _setup()
    ramp.begin(s0, -68.0)
    ramp.record(s0, 0.0, r["level_dbfs"], r["clipped_fraction"],
                noise_dbfs=r["noise_dbfs"])
    assert ramp.permit(_setup(tx_gain=6.0))[0]
    # an overdriven recording clips, and the ramp will not climb from it
    base2, _p, _d = _record(rf, tf, amp=3.0, name="m2")
    r2 = L.measure_rx(base2, tf.manifest_path)
    assert r2["clipped_fraction"] > S.MAX_CLIP_FRACTION
    ramp2 = S.Ramp(rf.root / "r2.json")
    ramp2.begin(s0, -68.0)
    ramp2.record(s0, 0.0, r2["level_dbfs"] or 0.0, r2["clipped_fraction"],
                 noise_dbfs=r2["noise_dbfs"])
    ok, why = ramp2.permit(_setup(tx_gain=3.0))
    assert not ok and "clipping" in why


def test_the_manifest_is_plain_json(rf):
    tf = _txfile(rf)
    j = json.loads(tf.manifest_path.read_text())
    assert j["what"].startswith("cabled-loop transmit file")
    assert j["marker"]["kind"] == "chirp" and "bandwidth_hz" in j["marker"]["params"]
