# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The cut's integer decimation and the logged resampler (plan §3.3)."""

from __future__ import annotations

import numpy as np
import pytest

from atk_diffusion import sigmf as S
from atk_diffusion.dsp import resample as R


def _tone(f, fs, n, a=0.5):
    t = np.arange(n) / fs
    return (a * np.exp(2j * np.pi * f * t)).astype(np.complex64)


def test_cut_to_canonical_moves_the_box_to_baseband_and_decimates_exactly():
    fs = 2_400_000
    x = _tone(300_000, fs, 240_000) + _tone(-500_000, fs, 240_000)
    y, fs_out, info = R.cut_to_canonical(x, fs, 300_000, 12_500)
    assert info["canonical_class"] == "voice" and info["decimation"] == 50
    assert fs_out == 48_000
    spec = np.abs(np.fft.fftshift(np.fft.fft(y[200:])))
    f = np.fft.fftshift(np.fft.fftfreq(y[200:].size, 1 / fs_out))
    assert abs(f[np.argmax(spec)]) < 100.0          # the box's tone is at 0 Hz
    assert np.max(np.abs(y[200:])) > 0.45            # kept
    # the other tone is far outside and gone
    assert np.mean(np.abs(y[200:]) ** 2) == pytest.approx(0.25, rel=0.05)


def test_decimation_keeps_time_alignment():
    fs = 1_000_000
    x = np.zeros(100_000, np.complex64)
    x[50_000] = 1.0
    y, fs_out = R.decimate(x, 10, fs)
    assert int(np.argmax(np.abs(y))) == 5_000


def test_resample_capture_is_logged_and_labelled(tmp_path, rf):
    fs = 2_000_000
    x = _tone(100_000, fs, 200_000)
    src_dp, src_mp = S.write_pair(tmp_path / "src", x, fs, 100e6,
                                  datatype="ci8", hw="HackRF One",
                                  annotations=[S.Annotation(20_000, 1_000, label="b")])
    out = R.resample_capture(src_dp, "rtlsdr_2400000_cu8", tmp_path / "out",
                             rf=rf, who="test", reason="unit test")
    assert out["exact"] and (out["up"], out["down"]) == (6, 5)
    meta = S.read_meta(out["meta"])
    g = meta["global"]
    assert g["atk:resampled_from"] == "hackrf_2000000_ci8"
    assert g["atk:receiver_profile"] == "rtlsdr_2400000_cu8"
    assert g["atk:tier"] == "cleaned"
    assert "resampled src from hackrf_2000000_ci8" in g["atk:resample_log"]
    assert S.annotations(meta)[0].sample_start == 24_000
    log = (rf.runs("rtlsdr_2400000_cu8") / "resample_log.txt").read_text()
    assert "polyphase 6/5" in log
    assert rf.verify(out["data"])[0]
