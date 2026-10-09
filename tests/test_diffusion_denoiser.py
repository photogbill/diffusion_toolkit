# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""B3: the denoiser (learn.denoiser) and its core-environment runtime
(dsp.denoise_runtime). The tiny model trained here learns nothing useful —
these tests prove the path: card, ONNX contract, profile refusal, the
measured hallucination rate, torch/ONNX parity, and that the core runtime
never imports PyTorch."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

from atk_diffusion import cards, profiles, sigmf
from atk_diffusion.dsp import denoise_runtime as R

PID = "rtlsdr_256000_cu8"
GEOM = dict(fft_size=32, hop=32, window="hann", tile_seconds=0.004, tile_rows=16)


@pytest.fixture(scope="module")
def spec_model(tmp_path_factory):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    torch.set_num_threads(1)
    from atk_diffusion.learn import denoiser as DN
    from atk_diffusion.learn import unet as U
    from atk_diffusion.paths import RfData
    rf = RfData(tmp_path_factory.mktemp("rfd") / "rf_data", create=True)
    d = DN.train_denoiser(rf, PID, domain="spectrogram",
                          geometry=profiles.StftGeometry(**GEOM), patch=(16, 16),
                          synthetic=40, steps=25, batch=8, T=100, unet=U.TINY_2D,
                          hallucination_tiles=8, validation_tiles=4, pfa=1e-3,
                          name="tiny_spec")
    return rf, d


def test_training_writes_a_card_with_measured_numbers(spec_model):
    rf, d = spec_model
    assert sorted(p.name for p in d.iterdir()) == ["card.json", "model.onnx", "model.pt"]
    card = cards.load(d, expect_kind="denoiser", for_profile=PID)
    assert card.tier == "invented" and card.profile == PID
    inp = card.input
    assert inp["domain"] == "spectrogram" and inp["schedule"]["T"] == 100
    assert inp["stft"]["pool"] == 2 and inp["stft"]["pool_mode"] == "max"
    assert inp["onnx"]["inputs"] == ["x", "t"] and inp["onnx"]["outputs"] == ["eps"]
    assert inp["normalize"]["noise_std_db"] > 0
    assert "mfsk8" not in inp["kinds"] and inp["unseen_kinds"] == ["mfsk8"]
    m = card.metrics
    h = m["hallucination"]
    assert h["tested"] == 8 and 0 <= h["hallucinated"] <= h["eligible"] <= 8
    assert m["hallucination_rate"] is None or 0.0 <= m["hallucination_rate"] <= 1.0
    assert set(m["val_rmse_db"]) == {"raw", "wiener", "median", "wavelet", "diffusion"}
    assert isinstance(m["beats_classical"], bool) and m["latency_ms"] > 0
    if not m["beats_classical"]:
        assert any("not to be shipped" in n for n in card.notes)
    text = "\n".join(cards.summary(card))
    assert "hallucination rate" in text and "INVENTED" in text
    logged = rf.log.entries()
    assert any(k.endswith("tiny_spec/model.onnx") for k in logged)


def test_runtime_and_torch_agree_and_say_invented(spec_model):
    from atk_diffusion.learn import denoiser as DN
    _rf, d = spec_model
    rt = R.DenoiserRuntime(d, PID)
    tm = DN.Denoiser.load(d, for_profile=PID)
    tile = np.random.default_rng(0).normal(0.0, 4.0, (16, 24)).astype(np.float32)
    a = rt.denoise(tile)
    b = tm.denoise(tile)
    assert a.shape == tile.shape and np.allclose(a, b, atol=1e-3)
    info = rt.last
    assert info["tier"] == "invented" and info["flag"] == "denoised"
    assert info["method"] == "diffusion_denoise" and "INVENTED" in info["words"]
    assert info["t"] == [rt.t_star] and info["model_sha256"] == rt.card.weights["sha256"]
    multi = rt.denoise(tile, steps=3)
    assert multi.shape == tile.shape and rt.last["steps"] == 3
    batch = rt.denoise(np.stack([tile, tile]))
    assert batch.shape == (2, 16, 24) and np.allclose(batch[0], a, atol=1e-4)


def test_runtime_refuses_another_profile_and_changed_weights(spec_model, tmp_path):
    _rf, d = spec_model
    with pytest.raises(profiles.ProfileMismatch, match="trained for the RTL-SDR"):
        R.DenoiserRuntime(d, "hackrf_8000000_ci8")
    bad = tmp_path / "copy"
    shutil.copytree(d, bad)
    raw = bytearray((bad / "model.onnx").read_bytes())
    raw[-10] ^= 0xFF
    (bad / "model.onnx").write_bytes(bytes(raw))
    with pytest.raises(cards.CardRefusal, match="changed"):
        R.DenoiserRuntime(bad, PID)


def test_a_tile_of_another_geometry_is_refused_in_words():
    with pytest.raises(R.DenoiserRefusal, match="never meets a spectrogram"):
        R.check_geometry({"fft_size": 1024, "hop": 1024, "window": "hann",
                          "pool": 5, "pool_mode": "max"}, 1024, 1024, "hann", 4)
    R.check_geometry({"fft_size": 32, "hop": 32, "window": "hann", "pool": 2,
                      "pool_mode": "max"}, 32, 32, "Hann", 2)


def test_denoise_tile_keeps_the_raw_measurements(spec_model):
    stft = pytest.importorskip("atk_diffusion.dsp.stft",
                               reason="dsp.stft (the pipeline's tiles) is "
                                      "another engineer's module")
    _rf, d = spec_model
    rt = R.DenoiserRuntime(d, PID)
    geom = profiles.StftGeometry(**GEOM)
    spec = np.random.default_rng(1).normal(2.0, 3.0, (16, 32)).astype(np.float32)
    tile = stft.Tile.from_spec(spec, 256000.0, 100e6, geom)
    out, info = rt.denoise_tile(tile)
    assert info["flag"] == "denoised" and info["tier"] == "invented"
    assert not np.array_equal(out.spec, tile.spec)
    assert np.array_equal(out.abs_db, tile.abs_db)          # never measured on
    assert np.array_equal(out.mean_above, tile.mean_above)  # a reconstruction


def test_the_core_runtime_never_imports_torch(spec_model, tmp_path):
    """ATK's core environment has no PyTorch: load the card, run the ONNX
    single step, and torch must still not be imported."""
    _rf, d = spec_model
    code = (
        "import sys, json, numpy as np\n"
        "from atk_diffusion.dsp import denoise_runtime as R\n"
        f"rt = R.DenoiserRuntime({str(d)!r}, {PID!r})\n"
        "out = rt.denoise(np.zeros((16, 16), np.float32))\n"
        "print(json.dumps({'torch': 'torch' in sys.modules, 'shape': list(out.shape),"
        " 'tier': rt.last['tier']}))\n")
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True, cwd=str(tmp_path),
                         env={**__import__("os").environ,
                              "PYTHONPATH": str(__import__("pathlib").Path(__file__).resolve().parents[1])})
    got = json.loads(res.stdout.strip().splitlines()[-1])
    assert got == {"torch": False, "shape": [16, 16], "tier": "invented"}


def test_hallucination_rate_counts_only_tiles_that_were_empty():
    from atk_diffusion.learn import denoiser as DN
    geom = {"fft_size": 32, "hop": 32, "window": "hann", "pool": 1,
            "pool_mode": "mean", "fs": 256000.0}
    rng = np.random.default_rng(5)
    tiles = 10 * np.log10(rng.exponential(size=(12, 16, 32)))

    class Identity:
        def denoise(self, x, **kw):
            return x

    class Blob:
        def denoise(self, x, **kw):
            y = np.array(x, copy=True)
            y[4:8, 10:13] += 25.0
            return y

    ident = DN.hallucination_rate(Identity(), tiles, 1e-3, geom)
    blob = DN.hallucination_rate(Blob(), tiles, 1e-3, geom)
    assert ident["rate"] == 0.0 and ident["eligible"] >= 8
    assert blob["rate"] == 1.0 and blob["eligible"] == ident["eligible"]


def test_spec_pairs_are_zero_where_there_is_no_signal(rng):
    from atk_diffusion.learn import denoiser as DN
    geom = DN.tile_geometry(PID, profiles.StftGeometry(**GEOM))
    noise = DN.NoiseSource(PID, 256000.0, (), rng)
    F = DN.floor_from_noise(noise, geom, frames=256)
    clean, noisy, bursts = DN.spec_pair(geom, F, 16, 16, noise, ["pocsag"], rng,
                                        p_empty=1.0)
    assert bursts == [] and np.all(clean == 0.0) and noisy.std() > 2.0
    clean, noisy, bursts = DN.spec_pair(geom, F, 16, 16, noise, ["pocsag"], rng,
                                        snr_db=(15.0, 15.0), p_empty=0.0, max_bursts=1)
    assert len(bursts) == 1 and clean.max() > 6.0


def test_tile_geometry_is_the_pipelines():
    from atk_diffusion.learn import denoiser as DN
    g = DN.tile_geometry("rtlsdr_2400000_cu8")
    assert (g["fft_size"], g["pool"], g["tile_rows"], g["pool_mode"]) == (1024, 5, 512, "max")


def test_noise_capture_of_another_profile_is_refused(tmp_path, rng):
    from atk_diffusion.learn import denoiser as DN
    x = (rng.normal(size=4096) + 1j * rng.normal(size=4096)).astype(np.complex64) * 0.01
    base = tmp_path / "hackrf_noise"
    sigmf.write_pair(base, x, 8e6, extra_global={"atk:receiver_profile": "hackrf_8000000_ci8"})
    with pytest.raises(profiles.ProfileMismatch, match="Profiles never mix"):
        DN.NoiseSource(PID, 256000.0, [str(base) + ".sigmf-data"], rng)
    ok = tmp_path / "rtl_noise"
    sigmf.write_pair(ok, x, 256000.0, extra_global={"atk:receiver_profile": PID})
    src = DN.NoiseSource(PID, 256000.0, [str(ok) + ".sigmf-data"], rng)
    assert src.draw(100).shape == (100,) and "Bill's own receiver noise" in src.words
    assert src.entries[0]["sha256"] and src.draw(10000).shape == (10000,)


def test_iq_denoiser_trains_and_runs(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    torch.set_num_threads(1)
    from atk_diffusion.learn import denoiser as DN
    from atk_diffusion.learn import unet as U
    from atk_diffusion.paths import RfData
    rf = RfData(tmp_path / "rf", create=True)
    d = DN.train_denoiser(rf, PID, domain="iq", window=64, synthetic=24, steps=12,
                          batch=8, T=100, unet=U.TINY_1D, hallucination_tiles=4,
                          pfa=1e-3, name="tiny_iq")
    card = cards.load(d, expect_kind="denoiser", for_profile=PID)
    assert card.input["domain"] == "iq" and card.input["window"] == 64
    assert set(card.metrics["val_snr_gain_db"]) == {"wiener", "diffusion"}
    assert card.metrics["hallucination_at_snr_db"] == card.input["snr_db_min"]
    rt = R.DenoiserRuntime(d, PID)
    z = (np.random.default_rng(2).normal(size=(3, 64))
         + 1j * np.random.default_rng(3).normal(size=(3, 64))).astype(np.complex64)
    out = rt.denoise(z, noise_power=1.0)
    assert out.shape == z.shape and np.iscomplexobj(out)
    assert "estimated" in " ".join(rt.last["notes"])
    out2 = rt.denoise(z[0], snr_db=-30.0, noise_power=1.0)
    assert rt.last["clamped"] and out2.shape == (64,)
    with pytest.raises(R.DenoiserRefusal, match="noise level"):
        rt.denoise(z[0])


def test_unknown_domain_is_refused(tmp_path):
    pytest.importorskip("torch")
    from atk_diffusion.learn import denoiser as DN
    from atk_diffusion.paths import RfData
    with pytest.raises(ValueError, match="unknown denoiser domain"):
        DN.train_denoiser(RfData(tmp_path / "rf", create=True), PID, domain="audio")
