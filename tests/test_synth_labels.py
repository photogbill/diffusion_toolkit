# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Labels both ways (synth.labels): generator labels and TorchSig 2.2.0
per-signal metadata <-> SigMF annotations, every branch, plus one section
against the real TorchSig `Signal` (ARCHITECTURE §2 rule 10)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from atk_diffusion import sigmf
from atk_diffusion.detect import classes
from atk_diffusion.synth import labels as LB


# -- names and indices ----------------------------------------------------------
def test_family_index_and_its_refusal():
    for i, fam in enumerate(classes.FAMILIES):
        assert LB.family_index(fam) == i
    with pytest.raises(LB.LabelError, match="not a detector family"):
        LB.family_index("telepathy")


def test_torchsig_names_and_candidates():
    assert LB.torchsig_names("p25") == ("p25",)
    assert LB.torchsig_names("not_a_class") == ()
    assert LB.torchsig_name_for("ref_qpsk") == "qpsk"
    assert set(LB.classes_for_torchsig("fm")) >= {"nfm_voice", "fm_broadcast"}
    with pytest.raises(LB.LabelError, match="ATSC 1.0 .* has no TorchSig"):
        LB.torchsig_name_for("atsc")
    with pytest.raises(LB.LabelError, match="'martian' has no TorchSig"):
        LB.torchsig_name_for("martian")


def test_class_for_torchsig_shows_the_ambiguity():
    assert LB.class_for_torchsig("zigbee") == (classes.UNKNOWN, [])
    assert LB.class_for_torchsig("p25") == ("p25", [])
    # by bandwidth: nearest in ratio among classes with a width
    cls, alts = LB.class_for_torchsig("fm", 200e3)
    assert cls == "fm_broadcast" and "nfm_voice" in alts
    # no bandwidth: the TorchSig-family reference class when there is one
    cls, alts = LB.class_for_torchsig("2fsk")
    assert cls == "ref_2fsk" and {"pocsag", "flex"} <= set(alts)
    # no bandwidth and no reference class: the first, with the rest listed
    cls, alts = LB.class_for_torchsig("4fsk")
    assert cls == "nxdn96" and "nxdn48" in alts
    # a bandwidth but every candidate is width-less: falls back the same way
    assert LB.class_for_torchsig("bpsk", 10e3) == ("ref_bpsk", [])


# -- generator labels <-> annotations ----------------------------------------------
def _label(**kw):
    lab = {"cls": "dmr", "family": "fsk", "sample_start": 100, "sample_count": 5000,
           "f_lo_hz": 20e3 - 3.8e3, "f_hi_hz": 20e3 + 3.8e3, "snr_db": 12.34567,
           "symbol_rate_hz": 4800.0, "carrier_offset_hz": 20e3,
           "bursts": [[100, 2000], [3100, 2000]], "generator": "native"}
    lab.update(kw)
    return lab


def test_label_to_annotation_carries_every_number():
    lab = _label(clipped=True, torchsig_class="dmr", torchsig_snr_db=15.0,
                 torchsig_edges=(16e3, 24e3))
    a = LB.label_to_annotation(lab, 462e6, environment="us-va-nokesville",
                               sample_offset=50, extra={"atk:dataset": "x"})
    d = a.to_sigmf()
    assert d["core:sample_start"] == 150 and d["core:sample_count"] == 5000
    assert d["core:freq_lower_edge"] == pytest.approx(462e6 + 16.2e3)
    assert d["core:label"] == "dmr"
    assert d["atk:source"] == "synthetic" and d["atk:generator"] == "native"
    assert d["atk:snr_db"] == 12.346 and d["atk:symbol_rate"] == 4800.0
    assert d["atk:environment"] == "us-va-nokesville" and d["atk:clipped"] is True
    assert d["atk:torchsig_class"] == "dmr" and d["atk:torchsig_snr_db"] == 15.0
    assert d["atk:torchsig_edges"] == [462e6 + 16e3, 462e6 + 24e3]
    assert d["atk:bursts"] == [[0, 2000], [3000, 2000]]
    assert d["atk:dataset"] == "x"
    assert sigmf.validate({"global": {"core:datatype": "cf32_le",
                                      "core:sample_rate": 1.0,
                                      "core:version": "1.0.0"},
                           "annotations": [d]}) == []


def test_label_to_annotation_leaves_out_what_is_not_known():
    a = LB.label_to_annotation(_label(snr_db=float("nan"), symbol_rate_hz=0.0,
                                      bursts=[[100, 5000]], family=None,
                                      generator=None), 0.0, generator="torchsig 2.2.0")
    d = a.to_sigmf()
    assert "atk:snr_db" not in d and "atk:symbol_rate" not in d
    assert "atk:bursts" not in d and "atk:environment" not in d
    assert d["atk:family"] == "fsk"          # from the class table
    assert d["atk:generator"] == "torchsig 2.2.0"
    noise = LB.label_to_annotation({"cls": "noise", "sample_start": 0,
                                    "sample_count": 10, "f_lo_hz": 0.0,
                                    "f_hi_hz": 0.0}, 100e6)
    assert noise.freq_lower_edge is None and noise.freq_upper_edge is None
    odd = LB.label_to_annotation({"cls": "martian", "sample_start": 0,
                                  "sample_count": 10, "f_lo_hz": -1.0,
                                  "f_hi_hz": 1.0, "snr_db": "n/a"}, 0.0)
    assert odd.extra["atk:family"] == "unknown" and "atk:snr_db" not in odd.extra


def test_annotation_to_label_inverts_and_fills_gaps():
    lab = _label()
    back = LB.annotation_to_label(LB.label_to_annotation(lab, 462e6), 462e6)
    for k in ("cls", "family", "symbol_rate_hz", "carrier_offset_hz",
              "sample_start", "sample_count"):
        assert back[k] == pytest.approx(lab[k]) if isinstance(lab[k], float) \
            else back[k] == lab[k]
    assert back["bandwidth_hz"] == pytest.approx(7.6e3)
    assert back["snr_db"] == pytest.approx(12.346)
    assert back["source"] == "synthetic"
    # no carrier offset written: the centre of the edges; no edges: zero
    a = sigmf.Annotation(0, 10, 100e6 + 1e3, 100e6 + 3e3, "ref_bpsk", extra={})
    b = LB.annotation_to_label(a, 100e6)
    assert b["carrier_offset_hz"] == pytest.approx(2e3)
    assert math.isnan(b["snr_db"]) and b["family"] == "psk_qam"
    c = LB.annotation_to_label(sigmf.Annotation(0, 10, None, None, "martian"))
    assert c["bandwidth_hz"] == 0.0 and c["carrier_offset_hz"] == 0.0
    assert c["family"] == "unknown" and c["f_lo_hz"] is None


# -- TorchSig metadata <-> annotations ---------------------------------------------
def _ts_meta(**kw):
    m = {"class_name": "fm", "center_freq": 50e3, "bandwidth": 200e3,
         "start_in_samples": 10, "duration_in_samples": 1000, "snr_db": 18.0}
    m.update(kw)
    return m


def test_torchsig_to_annotation_infers_the_class_and_keeps_torchsigs_numbers():
    a = LB.torchsig_to_annotation(_ts_meta(), sample_rate=2.4e6, center_hz=98e6,
                                  environment="us-va-nokesville",
                                  extra={"atk:dataset": "d"})
    d = a.to_sigmf()
    assert d["core:label"] == "fm_broadcast"
    assert "nfm_voice" in d["atk:class_alternatives"]
    assert d["atk:torchsig_class"] == "fm" and d["atk:torchsig_snr_db"] == 18.0
    assert d["atk:torchsig_edges"] == [98e6 - 50e3, 98e6 + 150e3]
    assert d["core:freq_lower_edge"] == 98e6 - 50e3     # TorchSig's own box
    assert "atk:snr_db" not in d                         # a different quantity
    assert d["atk:environment"] == "us-va-nokesville" and d["atk:dataset"] == "d"


def test_torchsig_to_annotation_with_the_toolkits_own_measurements():
    a = LB.torchsig_to_annotation(_ts_meta(class_name="qpsk", snr_db=float("nan")),
                                  sample_rate=2.4e6, center_hz=0.0, cls="ref_qpsk",
                                  snr_db=7.5, symbol_rate_hz=25e3,
                                  box=(40e3, 60e3))
    d = a.to_sigmf()
    assert d["core:label"] == "ref_qpsk" and d["atk:family"] == "psk_qam"
    assert (d["core:freq_lower_edge"], d["core:freq_upper_edge"]) == (40e3, 60e3)
    assert d["atk:snr_db"] == 7.5 and d["atk:symbol_rate"] == 25e3
    assert "atk:torchsig_snr_db" not in d and "atk:class_alternatives" not in d


def test_torchsig_metadata_shapes_and_refusals():
    class WithToDict:
        def to_dict(self):
            return _ts_meta(class_name="p25", bandwidth=8e3)

    class WithFull:
        def get_full_metadata(self):
            return _ts_meta(class_name="lora", bandwidth=125e3)

    assert LB.torchsig_to_annotation(WithToDict(), sample_rate=1e6).label == "p25"
    assert LB.torchsig_to_annotation(WithFull(), sample_rate=1e6).label == "lora"
    with pytest.raises(LB.LabelError, match="not TorchSig metadata"):
        LB.torchsig_to_annotation(object(), sample_rate=1e6)
    with pytest.raises(LB.LabelError, match="no 'bandwidth'"):
        LB.torchsig_to_annotation({"center_freq": 0.0, "start_in_samples": 0,
                                   "duration_in_samples": 1}, sample_rate=1e6)
    unknown = LB.torchsig_to_annotation(_ts_meta(class_name="zigbee"),
                                        sample_rate=1e6)
    assert unknown.label == classes.UNKNOWN
    assert unknown.extra["atk:family"] == "unknown"


def test_annotation_to_torchsig_and_back():
    ann = sigmf.Annotation(500, 4000, 446e6 + 10e3, 446e6 + 22e3, "dmr",
                           extra={"atk:snr_db": 11.0, "atk:symbol_rate": 4800.0})
    m = LB.annotation_to_torchsig(ann, sample_rate=2.4e6, center_hz=446e6,
                                  num_iq_samples=1 << 16)
    assert m == {"class_name": "dmr", "center_freq": 16e3, "bandwidth": 12e3,
                 "start_in_samples": 500, "duration_in_samples": 4000,
                 "sample_rate": 2.4e6, "snr_db": 11.0,
                 "num_iq_samples_dataset": 1 << 16, "symbol_rate": 4800.0}
    back = LB.torchsig_to_annotation(m, sample_rate=2.4e6, center_hz=446e6,
                                     cls="dmr")
    assert back.freq_lower_edge == pytest.approx(ann.freq_lower_edge)
    assert back.freq_upper_edge == pytest.approx(ann.freq_upper_edge)
    # TorchSig's own name and SNR win when the label was made by TorchSig
    ts = sigmf.Annotation(0, 10, 1e3, 3e3, "nxdn96",
                          extra={"atk:torchsig_class": "4fsk",
                                 "atk:torchsig_snr_db": 6.0})
    m2 = LB.annotation_to_torchsig(ts, sample_rate=1e6)
    assert m2["class_name"] == "4fsk" and m2["snr_db"] == 6.0
    assert "symbol_rate" not in m2 and "num_iq_samples_dataset" not in m2
    with pytest.raises(LB.LabelError, match="no frequency edges"):
        LB.annotation_to_torchsig(sigmf.Annotation(0, 10), sample_rate=1e6)


def test_the_new_atk_keys_are_documented():
    for k in ("atk:torchsig_class", "atk:torchsig_snr_db", "atk:torchsig_edges",
              "atk:bursts", "atk:class_alternatives", "atk:dataset"):
        assert k in sigmf.ATK_KEYS


# -- the real TorchSig ------------------------------------------------------------
def test_round_trip_through_a_real_torchsig_signal():
    pytest.importorskip("torchsig", reason="TorchSig is only in the training "
                        "environment")
    from torchsig.signals.signal_types import Signal
    ann = sigmf.Annotation(100, 2048, 462e6 + 12e3, 462e6 + 24e3, "p25",
                           extra={"atk:snr_db": 14.0})
    meta = LB.annotation_to_torchsig(ann, sample_rate=2.4e6, center_hz=462e6)
    sig = Signal(data=np.zeros(2048, np.complex64), **meta)
    assert sig.class_name == "p25" and sig.snr_db == 14.0
    back = LB.torchsig_to_annotation(sig, sample_rate=2.4e6, center_hz=462e6)
    assert back.label == "p25"
    assert back.freq_lower_edge == pytest.approx(ann.freq_lower_edge)
    assert back.extra["atk:torchsig_snr_db"] == 14.0
