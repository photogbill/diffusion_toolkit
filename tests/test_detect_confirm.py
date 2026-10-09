# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Only a decoder confirms; the decoder wins the label; disagreement is kept
(DETECTION_DESIGN §5; plan §2.1)."""

from __future__ import annotations

import numpy as np
import pytest

from atk_diffusion.detect.boxes import Detection
from atk_diffusion.detect.confirm import ConfirmResult, Confirmer
from atk_diffusion.detect.tracker import Tracker

IQ = np.zeros(4800, np.complex64)


def _det(cls="dmr", **kw):
    return Detection(t0=0.0, t1=0.5, f_lo=162.39e6, f_hi=162.41e6,
                     sources=("learned",), cls=cls, confidence=0.8, **kw)


def test_the_decoder_wins_the_label_and_the_disagreement_is_kept():
    calls = []

    def dsd(det, iq, fs):
        calls.append((det.cls, iq.size, fs))
        return ConfirmResult(ok=True, decoded="TG 1234 RID 5", decoder_class="p25")

    d = _det("dmr")
    res = Confirmer({"dsd": dsd}).confirm(d, IQ, 48_000.0)
    assert res.ok and res.decoder == "dsd" and res.decoder_class == "p25"
    assert calls == [("dmr", 4800, 48_000.0)]
    assert d.state == "confirmed" and d.confirmed_by == "dsd" and d.cls == "p25"
    assert d.measurements["classifier_said"] == "dmr" and "disagreement" in d.flags
    assert d.decoded == "TG 1234 RID 5"
    assert "the classifier said dmr; the decoder wins the label" in res.attempts[0]


def test_decoders_are_tried_in_the_class_tables_order_and_a_crash_is_words():
    order = []

    def pager(det, iq, fs):
        order.append("pager")
        raise RuntimeError("sync lost at bit 77")

    def multimon(det, iq, fs):
        order.append("multimon")
        return ConfirmResult(ok=True, decoded="POCSAG1200: Address: 1234567")

    d = _det("pocsag")
    res = Confirmer({"multimon": multimon, "pager": pager}).confirm(d, IQ, 48e3)
    assert order == ["pager", "multimon"]               # tools_for('pocsag') order
    assert res.ok and res.decoder == "multimon" and d.cls == "pocsag"
    assert "pager" in res.attempts[0] and "RuntimeError: sync lost at bit 77" in res.attempts[0]
    assert "disagreement" not in d.flags


def test_a_demodulator_never_confirms_even_when_registered():
    called = []

    def nfm(det, iq, fs):
        called.append(1)
        return ConfirmResult(ok=True, decoded="audio")

    d = _det("nfm_voice")
    res = Confirmer({"nfm": nfm}).confirm(d, IQ, 48e3)
    assert not res.ok and not called and d.state == "proposed"
    assert "no decoder that can confirm it" in res.why


def test_nothing_decoded_stays_proposed_and_says_why():
    d = _det("dmr")
    res = Confirmer({"dsd": lambda det, iq, fs: None}).confirm(d, IQ, 48e3)
    assert not res.ok and d.state == "proposed"
    assert "decoded nothing" in res.why
    res2 = Confirmer({}).confirm(_det("adsb"), IQ, 2e6)
    assert "not registered by ATK" in res2.why
    unknown = Confirmer({"dsd": lambda *a: None}).confirm(_det("UNKNOWN"), IQ, 48e3)
    assert "UNKNOWN" in unknown.why and not unknown.ok
    blank = Confirmer({}).confirm(_det(""), IQ, 48e3)
    assert "no class" in blank.why


def test_a_decode_from_a_reconstruction_says_so():
    d = _det("p25")
    res = Confirmer({"dsd": lambda det, iq, fs: ConfirmResult(ok=True, decoded="TG 9")}
                    ).confirm(d, IQ, 48e3, tier="invented")
    assert res.ok and "INVENTED" in res.note and "not from the record" in res.note
    assert d.state == "confirmed" and "INVENTED" in d.decoded
    assert d.measurements["confirmed_from_tier"] == "invented"
    rec = _det("p25")
    r2 = Confirmer({"dsd": lambda det, iq, fs: ConfirmResult(ok=True, decoded="TG 9")}
                   ).confirm(rec, IQ, 48e3)
    assert r2.note == "" and rec.decoded == "TG 9"


def test_the_track_is_upgraded_too():
    tr = Tracker()
    d = _det("dmr")
    tr.update([d])
    t = tr.get(d.track_id)
    Confirmer({"dsd": lambda det, iq, fs: ConfirmResult(ok=True, decoded="TG 1",
                                                        decoder_class="dmr")}
              ).confirm(d, IQ, 48e3, track=t)
    assert t.state == "confirmed" and t.confirmed_by == "dsd" and t.cls == "dmr"


def test_the_registry_is_checked_and_described():
    with pytest.raises(ValueError, match="not a decoder"):
        Confirmer({"dsd-plus": lambda *a: None})
    with pytest.raises(ValueError, match="not callable"):
        Confirmer({"dsd": "dsd.exe"})
    c = Confirmer({"pager": lambda *a: None})
    assert c.confirming_decoders("pocsag") == ["pager", "multimon"]
    assert c.available("pocsag") == ["pager"]
    words = c.describe("pocsag")
    assert "POCSAG pager can be confirmed by" in words and "available" in words
    assert "not registered by ATK" in words
