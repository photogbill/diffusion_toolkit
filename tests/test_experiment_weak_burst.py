# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""B3's first experiment — the weak-burst test (experiments.weak_burst).
Tiny runs: they prove the path and the bookkeeping (fixed false-alarm rate,
Pd per method, hallucination, the matched-filter line, the report), not the
method — that number comes from Bill's captures."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from atk_diffusion import profiles, sigmf
from atk_diffusion.experiments import weak_burst as WB

PID = "rtlsdr_256000_cu8"
GEOM = profiles.StftGeometry(fft_size=32, hop=32, window="hann",
                             tile_seconds=0.008, tile_rows=32)
TINY = dict(geometry=GEOM, rows=32, bins=32, cell_pfa=1e-3, pfa=0.1, seed=3)


def test_operating_point_is_the_loosest_rate_that_holds_the_target():
    grid = (0.1, 0.01, 0.001)
    pmin = [0.1, 0.1, 0.01, float("inf"), float("inf"), float("inf"), 0.001,
            float("inf"), float("inf"), float("inf")]
    g, rate = WB.operating_point(pmin, 0.2, grid)
    assert g == 0.01 and rate == 0.2
    # nothing on the grid holds 5 %: the strictest point is used and its
    # realised rate (over target) is reported, not hidden
    g, rate = WB.operating_point(pmin, 0.05, grid)
    assert g == 0.001 and rate == 0.1


def test_snr_at_pd_interpolates_and_flags():
    s = [-6.0, 0.0, 6.0]
    assert WB.snr_at_pd(s, [0.0, 0.4, 0.8], 0.5)["snr_db"] == pytest.approx(1.5)
    assert WB.snr_at_pd(s, [0.0, 0.4, 0.8], 0.9) is None
    first = WB.snr_at_pd(s, [0.95, 1.0, 1.0], 0.9)
    assert first == {"snr_db": -6.0, "at_or_below": True}


def test_matched_statistic_reads_one_on_noise_and_large_on_signal(rng):
    t = np.exp(2j * np.pi * 0.1 * np.arange(64)).astype(np.complex64)
    stats = [WB.matched_statistic((rng.normal(size=64) + 1j * rng.normal(size=64))
                                  / math.sqrt(2), t, 1.0) for _ in range(1200)]
    assert abs(np.mean(stats) - 1.0) < 0.12          # one lag: exponential, mean 1
    y = 0.5 * t + (rng.normal(size=64) + 1j * rng.normal(size=64)) / math.sqrt(2)
    assert WB.matched_statistic(y, t, 1.0) > 8.0     # E/N0 = 16 -> ~16


def test_classical_run_is_sane_and_writes_its_report(tmp_path):
    res = WB.run(None, PID, bursts=("pocsag", "adsb"), snrs_db=(-20.0, 20.0),
                 trials=10, noise_trials=24, detector="internal",
                 out_dir=tmp_path / "wb", **TINY)
    b = res["bursts"]["pocsag"]
    assert res["methods"] == ["raw", "wiener", "median", "wavelet"]
    assert res["method_tiers"] == {"raw": "record", "wiener": "cleaned",
                                   "median": "cleaned", "wavelet": "cleaned"}
    assert "internal" in res["detector"]
    for m in res["methods"]:
        assert b["pd"][m][1] >= 0.8 and b["pd"][m][0] <= 0.5, (m, b["pd"][m])
        assert b["operating_point"][m]["tile_pfa_realised"] <= 0.1 + 1e-9
    assert b["pd_matched_filter"][1] == 1.0
    for m in ("wiener", "median", "wavelet"):
        h = b["hallucination"][m]
        assert h["eligible"] > 0 and h["rate"] == 0.0       # cleaning adds nothing
    assert "adsb" in res["skipped"] and "2 MS/s" in res["skipped"]["adsb"]
    assert any("No diffusion denoiser" in v for v in res["verdict"])
    md = (tmp_path / "wb" / "weak_burst.md").read_text(encoding="utf-8")
    assert "INVENTED" in md and "No diffusion denoiser" in md
    det = (tmp_path / "wb" / "weak_burst_detail.md").read_text(encoding="utf-8")
    assert "| method | SNR at Pd 0.5 |" in det
    js = json.loads((tmp_path / "wb" / "weak_burst.json").read_text(encoding="utf-8"))
    assert js["tier"] == "measured"
    assert js["result"]["bursts"]["pocsag"]["pd"]["raw"] == b["pd"]["raw"]


def test_bills_capture_is_the_noise_and_another_profile_is_refused(tmp_path, rng):
    x = (rng.normal(size=1 << 15) + 1j * rng.normal(size=1 << 15)) * 0.03
    base = tmp_path / "quiet"
    sigmf.write_pair(base, x.astype(np.complex64), 256000.0, datatype="cu8",
                     extra_global={"atk:receiver_profile": PID})
    res = WB.run(None, PID, noise_capture=str(base) + ".sigmf-data",
                 bursts=("mfsk8",), snrs_db=(15.0,), trials=4, noise_trials=8,
                 detector="internal", **TINY)
    assert "Bill's own receiver noise" in res["noise"]
    other = tmp_path / "hack"
    sigmf.write_pair(other, x.astype(np.complex64), 8e6,
                     extra_global={"atk:receiver_profile": "hackrf_8000000_ci8"})
    with pytest.raises(profiles.ProfileMismatch, match="Profiles never mix"):
        WB.run(None, PID, noise_capture=str(other) + ".sigmf-data",
               bursts=("mfsk8",), snrs_db=(15.0,), trials=2, noise_trials=4, **TINY)


def test_the_pipelines_energy_proposer_is_the_detector_when_installed():
    pytest.importorskip("atk_diffusion.dsp.cfar", reason="dsp.cfar is another "
                        "engineer's module")
    pytest.importorskip("atk_diffusion.dsp.stft", reason="dsp.stft is another "
                        "engineer's module")
    res = WB.run(None, PID, bursts=("mfsk8",), snrs_db=(18.0,), trials=4,
                 noise_trials=8, detector="energy_proposer", **TINY)
    assert "energy_proposer" in res["detector"]
    assert res["bursts"]["mfsk8"]["pd"]["raw"][0] >= 0.75


@pytest.fixture(scope="module")
def tiny_denoiser(tmp_path_factory):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    torch.set_num_threads(1)
    from atk_diffusion.learn import denoiser as DN
    from atk_diffusion.learn import unet as U
    from atk_diffusion.paths import RfData
    rf = RfData(tmp_path_factory.mktemp("wbd") / "rf_data", create=True)
    d = DN.train_denoiser(rf, PID, domain="spectrogram", geometry=GEOM, patch=(16, 16),
                          synthetic=32, steps=20, batch=8, T=100, unet=U.TINY_2D,
                          hallucination_tiles=6, validation_tiles=3, pfa=1e-3,
                          name="wb_tiny")
    return rf, d


def test_with_the_diffusion_denoiser_every_number_is_there(tiny_denoiser):
    rf, d = tiny_denoiser
    res = WB.run(rf, PID, bursts=("mfsk8", "pocsag"), snrs_db=(-6.0, 18.0),
                 trials=6, noise_trials=12, denoiser=d, detector="internal", **TINY)
    assert res["methods"][-1] == "diffusion"
    assert res["method_tiers"]["diffusion"] == "invented"
    assert res["denoiser"]["name"] == "wb_tiny" and res["denoiser"]["t_star"] is not None
    for kind, seen in (("mfsk8", False), ("pocsag", True)):
        b = res["bursts"][kind]
        assert b["seen_by_denoiser"] is seen
        h = b["hallucination"]["diffusion"]
        assert h["eligible"] > 0 and 0 <= h["hallucinated"] <= h["eligible"]
        assert len(b["pd"]["diffusion"]) == 2
        assert set(b["snr_at_pd"]) == set(res["methods"]) | {"matched_filter"}
    assert res["report_md"].startswith(str(rf.runs(PID)))
    assert any(k in v for v in res["verdict"] for k in ("diffusion", "Pd 0.9"))


def test_a_denoiser_of_another_geometry_or_profile_is_refused(tiny_denoiser):
    from atk_diffusion.dsp import denoise_runtime as R
    _rf, d = tiny_denoiser
    other = profiles.StftGeometry(fft_size=32, hop=32, window="hann",
                                  tile_seconds=0.016, tile_rows=32)     # 4 frames/row
    with pytest.raises(R.DenoiserRefusal, match="never meets a spectrogram"):
        WB.run(None, PID, bursts=("mfsk8",), snrs_db=(10.0,), trials=2, noise_trials=4,
               denoiser=d, geometry=other, rows=32, bins=32)
    with pytest.raises(profiles.ProfileMismatch):
        WB.run(None, "rtlsdr_2400000_cu8", bursts=("mfsk8",), snrs_db=(10.0,),
               trials=2, noise_trials=4, denoiser=d)


def test_every_method_holds_the_false_alarm_rate_on_plain_noise(tmp_path):
    """2026-10-09: the experiment fed the detector's integrated layer the
    MAX-pooled tile as if it were mean-pooled power; pure noise then looked
    2.3x the floor, the raw detector fired on every noise-only tile (realised
    Pfa 1.000), its Pd column was meaningless and no tile was left to judge
    hallucination on. Every method must now hold the stated rate, and the
    hallucination check must have tiles to judge."""
    from atk_diffusion.experiments import weak_burst as W
    r = W.run(None, "rtlsdr_2400000_cu8", bursts=("mfsk8",), snrs_db=(-3.0, 3.0),
              trials=4, noise_trials=60, pfa=0.05, seed=3, out_dir=tmp_path)
    b = r["bursts"]["mfsk8"]
    for m, op in b["operating_point"].items():
        assert op["tile_pfa_realised"] <= 0.05, (m, op)
    assert b["hallucination"]["wiener"]["eligible"] > 0
    assert "integrated layer" in r["detector"]
