# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Transmit safety on the cabled loop (plan §4.A6, D10): the arithmetic, the
refusals and the ramp, enforced by code.

The receiver and transmitter numbers below are TEST FIXTURES, chosen to make
the arithmetic easy to follow — they are not data-sheet values and must
never be copied into a profile."""

from __future__ import annotations

import inspect

import pytest

from atk_diffusion import profiles
from atk_diffusion.cabled import safety as S

RX = "rtlsdr_2400000_cu8"
FIXTURE = "test fixture — not a data-sheet value"


def _rx(pid=RX, max_dbm=10.0, source=FIXTURE, entered="2026-10-08"):
    p = profiles.new_profile(pid)
    p.safe_input = profiles.SafeInput(max_dbm=max_dbm, source=source,
                                      entered=entered)
    return p


def _setup(**kw):
    d = dict(transmitter="hackrf", receiver_profile=RX, frequency_hz=915e6,
             tx_gain=0.0,
             tx_power=S.TxPower(ref_gain=47.0, ref_dbm=10.0, source=FIXTURE,
                                entered="2026-10-08"),
             attenuation_db=30.0, cable_loss_db=1.0, cabled=True, dc_block=True)
    d.update(kw)
    return S.LoopSetup(**d)


def test_a_good_setup_passes_and_the_arithmetic_is_stated():
    v = S.check(_setup(), _rx())
    assert v.ok, v.lines()
    # TX at gain 0 = 10 - 47 = -37 dBm; minus 30 dB and 1 dB of cable
    assert v.tx_power_dbm == pytest.approx(-37.0)
    assert v.expected_input_dbm == pytest.approx(-68.0)
    assert v.ceiling_dbm == pytest.approx(-10.0) and v.margin_db == pytest.approx(58.0)
    text = "\n".join(v.lines())
    assert "Expected input at the receiver: -68.0 dBm" in text
    assert "less 20 dB of margin" in text and "Safe to run on the cable." in text
    # far below the -40 dBm target: a warning with the attenuation change
    assert any("remove about 28 dB of attenuation" in w for w in v.warnings)
    assert v.suggest_attenuation_change_db == pytest.approx(-28.0)


def test_expected_input_is_tx_less_attenuation_splitter_and_cable():
    assert S.expected_input_dbm(5.0, 40.0, 7.5, 1.5) == pytest.approx(-44.0)


def test_above_the_ceiling_is_refused_with_the_attenuation_needed():
    v = S.check(_setup(tx_gain=47.0, attenuation_db=10.0, cable_loss_db=0.0),
                _rx())
    assert not v.ok
    r = " ".join(v.refusals)
    assert "would get 0.0 dBm — above the ceiling of -10.0 dBm" in r
    assert "Add at least 10.0 dB of attenuation" in r


def test_far_above_the_target_but_under_the_ceiling_warns_to_add_attenuation():
    v = S.check(_setup(tx_gain=47.0, attenuation_db=30.0, cable_loss_db=0.0),
                _rx())
    assert v.ok and v.expected_input_dbm == pytest.approx(-20.0)
    assert any("add about 20 dB of attenuation" in w for w in v.warnings)


def test_the_top_gain_is_flagged_when_it_would_pass_the_ceiling():
    v = S.check(_setup(attenuation_db=15.0), _rx())
    assert v.ok
    assert any("the loop will refuse before the gain gets there" in w
               for w in v.warnings)


@pytest.mark.parametrize("tx", ["rtlsdr", "krakensdr", "kiwisdr"])
def test_only_a_bladerf_or_a_hackrf_may_transmit(tx):
    v = S.check(_setup(transmitter=tx), _rx())
    assert not v.ok and any("cannot transmit" in r for r in v.refusals)
    assert set(S.TRANSMITTERS) == {"hackrf", "bladerf1", "bladerf2"}


@pytest.mark.parametrize("field,value", [("cabled", False), ("cabled", "yes"),
                                         ("cabled", 1), ("dc_block", False),
                                         ("dc_block", "true")])
def test_cable_and_dc_block_must_be_confirmed_explicitly(field, value):
    v = S.check(_setup(**{field: value}), _rx())
    assert not v.ok
    word = "cable is not confirmed" if field == "cabled" else "DC block is not"
    assert any(word in r for r in v.refusals)
    if field == "cabled":
        assert any("never over the air" in r for r in v.refusals)


def test_the_receivers_safe_input_must_come_from_the_data_sheet():
    v = S.check(_setup(), _rx(max_dbm=None))
    assert any("not entered" in r and "data" in r and "will not guess" in r
               for r in v.refusals)
    v = S.check(_setup(), _rx(source=""))
    assert any("has no source" in r for r in v.refusals)
    v = S.check(_setup(), _rx(entered=""))
    assert any("has no date" in r for r in v.refusals)


def test_the_transmit_power_must_be_entered_with_its_source():
    v = S.check(_setup(tx_power=None), _rx())
    assert any("output power is not entered" in r for r in v.refusals)
    v = S.check(_setup(tx_power=S.TxPower(47.0, 10.0)), _rx())
    assert any("output power has no source" in r for r in v.refusals)


def test_the_kraken_needs_its_splitter_loss_and_physics_bounds_it():
    k = "krakensdr_2400000_cu8_ch0"
    v = S.check(_setup(receiver_profile=k), _rx(k))
    assert any("fed through a splitter" in r for r in v.refusals)
    v = S.check(_setup(receiver_profile=k, splitter_loss_db=3.0,
                       splitter_ways=5), _rx(k))
    assert any("5-way splitter loses at least 7.0 dB" in r for r in v.refusals)
    v = S.check(_setup(receiver_profile=k, splitter_loss_db=8.0,
                       splitter_ways=5), _rx(k))
    assert v.ok and v.expected_input_dbm == pytest.approx(-37 - 30 - 8 - 1)
    md = v.metadata()
    assert md["atk:splitter_loss_db"] == 8.0
    for key in ("atk:tx_power_dbm", "atk:attenuation_db", "atk:cable_loss_db",
                "atk:expected_input_dbm"):
        assert md[key] is not None


def test_gain_ranges_are_known_or_entered():
    v = S.check(_setup(tx_gain=50.0), _rx())
    assert any("outside the transmitter's range (0 to 47" in r for r in v.refusals)
    v = S.check(_setup(transmitter="bladerf1"), _rx())
    assert any("minimum TX gain is not known" in r for r in v.refusals)
    v = S.check(_setup(transmitter="bladerf1", tx_gain_min=-20.0,
                       tx_gain_max=60.0, tx_gain=-20.0), _rx())
    assert v.ok, v.refusals


def test_every_refusal_is_a_sentence_and_all_are_reported_at_once():
    v = S.check(S.LoopSetup("rtlsdr", RX, 915e6, 0.0), _rx(max_dbm=None))
    assert len(v.refusals) >= 5
    assert all(r.endswith(".") or r.endswith(")") for r in v.refusals)
    assert "Not run." in v.lines()[-1]


def test_ballpark_numbers_live_only_in_the_documentation():
    doc = S.__doc__
    assert "CONFIRM AGAINST THE DATA SHEETS, NEVER" in doc and "MEMORY" in doc
    src = inspect.getsource(S)
    for ballpark in ("50–60 dB", "40–50"):
        assert ballpark in doc
        assert src.count(ballpark) == 1          # nowhere but the docstring


# -- the ramp ---------------------------------------------------------------------
def test_the_first_run_of_a_setup_is_at_minimum_gain(tmp_path):
    ramp = S.Ramp(tmp_path / "ramp.json")
    v = S.check(_setup(tx_gain=10.0), _rx(), ramp)
    assert not v.ok
    assert any("first run of this setup — it runs at the minimum TX gain (0)"
               in r for r in v.refusals)
    assert S.check(_setup(tx_gain=0.0), _rx(), ramp).ok


def _measured(ramp, setup, gain, dbfs, noise=-60.0, clip=0.0):
    s = _setup(**{**setup, "tx_gain": gain})
    ramp.begin(s, S.check(s, _rx()).expected_input_dbm)
    ramp.record(s, gain, dbfs, clip, noise_dbfs=noise)


def test_the_ramp_climbs_only_as_the_receivers_reading_agrees(tmp_path):
    ramp = S.Ramp(tmp_path / "ramp.json")
    base = {}
    _measured(ramp, base, 0.0, -50.0)              # 10 dB above the noise
    # one measured point: cautious steps only
    ok, why = ramp.permit(_setup(tx_gain=10.0))
    assert not ok and "at most 6 dB per run" in why and "only one measured" in why
    assert ramp.permit(_setup(tx_gain=6.0))[0]
    _measured(ramp, base, 6.0, -44.2)              # +5.8 dB for a computed +6
    ok, why = ramp.permit(_setup(tx_gain=20.0))
    assert ok and "followed the arithmetic" in why
    # persisted, and read back
    again = S.Ramp(tmp_path / "ramp.json")
    assert len(again.steps(_setup())) == 2


def test_a_reading_that_disagrees_with_the_arithmetic_stops_the_climb(tmp_path):
    ramp = S.Ramp(tmp_path / "ramp.json")
    _measured(ramp, {}, 0.0, -50.0)
    _measured(ramp, {}, 6.0, -48.5)                # rose 1.5 dB for a computed 6
    ok, why = ramp.permit(_setup(tx_gain=12.0))
    assert not ok and "rose 1.5 dB where the arithmetic says 6.0 dB" in why
    assert "compressing" in why
    assert ramp.permit(_setup(tx_gain=3.0))[0]     # down is always allowed


def test_clipping_and_missing_readings_stop_the_climb(tmp_path):
    ramp = S.Ramp(tmp_path / "ramp.json")
    _measured(ramp, {}, 0.0, -3.0, clip=0.02)
    ok, why = ramp.permit(_setup(tx_gain=3.0))
    assert not ok and "was clipping at gain 0 (2% of samples" in why
    ramp2 = S.Ramp(tmp_path / "r2.json")
    s = _setup(tx_gain=0.0)
    ramp2.begin(s, -68.0)
    ok, why = ramp2.permit(_setup(tx_gain=3.0))
    assert not ok and "has no level reading yet" in why


def test_a_signal_lost_in_the_noise_is_climbed_in_small_steps(tmp_path):
    ramp = S.Ramp(tmp_path / "ramp.json")
    _measured(ramp, {}, 0.0, -58.0, noise=-60.0)   # only 2 dB above the noise
    ok, why = ramp.permit(_setup(tx_gain=12.0))
    assert not ok and "only 2.0 dB above the receiver's noise" in why
    assert ramp.permit(_setup(tx_gain=6.0))[0]


def test_an_absolute_reading_far_from_the_arithmetic_is_refused(tmp_path):
    ramp = S.Ramp(tmp_path / "ramp.json")
    s = _setup(tx_gain=0.0)
    ramp.begin(s, -68.0)
    # full scale known: -50 dBFS + (-5 dBm full scale) = -55 dBm, not -68
    ramp.record(s, 0.0, -50.0, 0.0, noise_dbfs=-70.0, rx_full_scale_dbm=-5.0)
    ok, why = ramp.permit(_setup(tx_gain=3.0))
    assert not ok and "the chain is not what the numbers say" in why


def test_a_changed_setup_starts_again_at_minimum(tmp_path):
    ramp = S.Ramp(tmp_path / "ramp.json")
    _measured(ramp, {}, 0.0, -50.0)
    _measured(ramp, {}, 6.0, -44.0)
    assert ramp.permit(_setup(tx_gain=12.0))[0]
    other = _setup(tx_gain=12.0, attenuation_db=20.0)   # an attenuator removed
    assert other.fingerprint() != _setup().fingerprint()
    ok, why = ramp.permit(other)
    assert not ok and "first run of this setup" in why
    moved = _setup(tx_gain=12.0, frequency_hz=433.92e6)  # TX power varies with f
    assert not ramp.permit(moved)[0]


def test_ramp_for_profile_lives_beside_the_cabled_captures(rf):
    ramp = S.Ramp.for_profile(rf, RX)
    ramp.begin(_setup(), -68.0)
    assert (rf.cabled(RX) / "loop_ramp.json").exists()
