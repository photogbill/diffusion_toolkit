# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""SigMF pairs, datatypes and annotations (DETECTION_DESIGN §10)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from atk_diffusion import sigmf as S
from atk_diffusion.dsp import iq


@pytest.mark.parametrize("dt", ["cu8", "ci8", "ci16", "ci16q11", "cf32"])
def test_round_trip_every_datatype(tmp_path, dt, rng):
    x = (rng.normal(size=4096) + 1j * rng.normal(size=4096)).astype(np.complex64) * 0.2
    dp, mp = S.write_pair(tmp_path / "a", x, 2_400_000, 100e6, datatype=dt)
    y = S.load(dp)
    tol = {"cu8": 1 / 127, "ci8": 1 / 127, "ci16": 1e-4, "ci16q11": 1e-3,
           "cf32": 1e-7}[dt]
    assert y.dtype == np.complex64 and y.size == x.size
    assert np.max(np.abs(y - x)) <= tol + 1e-6


def test_q11_scaling_is_read_from_atk_datatype(tmp_path):
    """The trap: the bytes of a bladeRF file are any ci16_le file's bytes."""
    x = np.full(100, 0.5 + 0.0j, dtype=np.complex64)
    dp, mp = S.write_pair(tmp_path / "b", x, 4e6, datatype="ci16q11")
    meta = json.loads(mp.read_text())
    assert meta["global"]["core:datatype"] == "ci16_le"
    assert meta["global"]["atk:datatype"] == "ci16q11"
    assert abs(S.load(dp)[0].real - 0.5) < 1e-3
    del meta["global"]["atk:datatype"]
    mp.write_text(json.dumps(meta))
    assert abs(S.load(dp)[0].real - 0.5 / 16) < 1e-3     # 24 dB low, as warned


def test_rtl_bytes_are_offset_binary():
    raw = bytes([128, 127, 255, 0])
    x = iq.to_complex(raw, "cu8")
    assert abs(x[0].real - 0.5 / 127.5) < 1e-6
    assert abs(x[1].real - 1.0) < 1e-6 and abs(x[1].imag + 1.0) < 1e-6


def test_multichannel_interleave(tmp_path, rng):
    x = (rng.normal(size=(5, 1000)) + 1j * rng.normal(size=(5, 1000))).astype(np.complex64) * 0.1
    dp, mp = S.write_pair(tmp_path / "k", x, 2.4e6)
    y = S.load(dp)
    assert y.shape == (5, 1000)
    assert np.allclose(y, x, atol=1e-7)
    assert np.allclose(S.load(dp, channel=3), x[3], atol=1e-7)
    assert np.allclose(S.load(dp, start=10, count=20, channel=1), x[1, 10:30])


def test_annotations_round_trip_and_replace_by_source(tmp_path):
    x = np.zeros(1000, np.complex64)
    dp, mp = S.write_pair(tmp_path / "c", x, 1e6, annotations=[
        S.Annotation(100, 50, 1.0e6, 1.1e6, "dmr", extra={"atk:source": "taught"})])
    S.add_annotations(dp, [S.Annotation(10, 5, label="x",
                                        extra={"atk:source": "proposed"})])
    S.add_annotations(dp, [S.Annotation(20, 5, label="y",
                                        extra={"atk:source": "proposed"})],
                      replace_source="proposed")
    anns = S.annotations(dp)
    assert [a.label for a in anns] == ["y", "dmr"]        # sorted, x replaced
    assert anns[1].source == "taught" and anns[1].bandwidth == pytest.approx(1e5)
    assert S.validate(S.read_meta(dp)) == []


def test_validate_names_problems():
    bad = {"global": {"core:datatype": "cq3"}, "captures": [],
           "annotations": [{"core:freq_lower_edge": 2, "core:freq_upper_edge": 1,
                            "atk:source": "guess"}]}
    probs = " | ".join(S.validate(bad))
    for needle in ("core:sample_rate", "unsupported", "no core:sample_start",
                   "reversed", "unknown atk:source"):
        assert needle in probs


def test_centre_follows_capture_segments():
    meta = {"captures": [{"core:sample_start": 0, "core:frequency": 100.0},
                         {"core:sample_start": 500, "core:frequency": 200.0}]}
    assert S.center_of(meta, 499) == 100.0 and S.center_of(meta, 500) == 200.0


def test_clip_detector():
    x = np.array([1.0 + 0j, 0.1 + 0.1j] * 50, np.complex64)
    assert iq.clipped_fraction(x, "ci8") == pytest.approx(0.5)
