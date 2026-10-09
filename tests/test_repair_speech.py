# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Speech enhancement before Whisper (plan §4.D2): the classical enhancers
and the learned one held at arm's length."""

from __future__ import annotations

import json
import struct
import sys
import wave

import numpy as np
import pytest

from atk_diffusion import provenance
from atk_diffusion.repair import speech as S

FS = 8000


@pytest.fixture
def speechy(rng):
    clean = S.speech_like(FS, 4.0, rng)
    return clean, S.add_noise(clean, 5.0, rng)


@pytest.mark.parametrize("bits", [8, 16, 24, 32])
def test_wav_round_trip_every_pcm_width(tmp_path, bits, rng):
    x = (0.4 * np.sin(np.linspace(0, 200, 4000))).astype(np.float32)
    p = tmp_path / f"a{bits}.wav"
    full = {8: 128, 16: 32768, 24: 1 << 23, 32: 1 << 31}[bits]
    v = np.round(x * (full - 1)).astype(np.int64)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(bits // 8)
        w.setframerate(FS)
        st = np.stack([v, v], axis=1).ravel()
        if bits == 8:
            raw = (st + 128).astype(np.uint8).tobytes()
        elif bits == 24:
            raw = b"".join(int(s).to_bytes(3, "little", signed=True) for s in st)
        else:
            raw = st.astype("<i2" if bits == 16 else "<i4").tobytes()
        w.writeframes(raw)
    y, fs, info = S.read_wav(p)
    assert fs == FS and info["channels"] == 2 and info["bits"] == bits
    assert np.max(np.abs(y - x)) < 2.0 / full * 2 + 1e-6


def test_write_wav_counts_clipping(tmp_path):
    out = S.write_wav(tmp_path / "c.wav", np.array([0.0, 0.5, 1.5, -2.0]), FS)
    assert out["clipped"] == 2
    y, fs, _ = S.read_wav(out["path"])
    assert fs == FS and y.size == 4


def test_float_wav_is_refused_in_words(tmp_path):
    data = np.zeros(10, "<f4").tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, FS, FS * 4, 4, 32)
    riff = (b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(data)) + b"WAVE"
            + b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + b"data" + struct.pack("<I", len(data)) + data)
    (tmp_path / "f.wav").write_bytes(riff)
    with pytest.raises(ValueError, match="16-bit PCM"):
        S.read_wav(tmp_path / "f.wav")


def test_stft_reconstructs_perfectly(rng):
    x = rng.normal(size=3001)
    for fs in (8000, 16000):
        n = S.frame_size(fs)
        assert np.allclose(S.istft(S.stft(x, n), n, x.size), x, atol=1e-12)


def test_mcra_tracks_noise_and_survives_digital_silence(rng):
    x = 0.05 * rng.normal(size=FS * 5)
    x[FS:FS + FS // 2] = 0.0                # DSD writes zeros between overs
    n = S.frame_size(FS)
    P = np.abs(S.stft(x, n)) ** 2
    lam, pres = S.mcra(P, (n // 2) / FS)
    true = np.mean(np.abs(S.stft(0.05 * rng.normal(size=FS * 5), n)) ** 2)
    late = lam[-60:-10, 2:-2] / true
    assert 0.75 < np.median(late) < 1.3
    assert np.percentile(late, 10) > 0.5     # no bin frozen near zero
    assert pres[-60:-10].mean() < 0.05


@pytest.mark.parametrize("method", S.METHODS)
def test_each_enhancer_measurably_helps(method, speechy):
    clean, noisy = speechy
    y, info = S.enhance(noisy, FS, method)
    assert info["tier"] == "cleaned" and provenance.tier_for(method) == "cleaned"
    gain = S.segmental_snr(clean, y, FS) - S.segmental_snr(clean, noisy, FS)
    assert gain > 2.5, gain
    pause = np.abs(clean) < 1e-9
    from scipy.ndimage import binary_erosion
    pz = binary_erosion(pause, iterations=int(0.04 * FS))
    att = 10 * np.log10(np.mean(y[pz] ** 2) / np.mean(noisy[pz] ** 2))
    assert att < -10.0, att
    assert info["attenuation_in_pauses_db"] < -6.0


def test_enhancers_add_nothing_to_pure_noise(rng):
    """A gain in [0, 1] cannot make speech out of noise: the output of
    noise alone is quieter than the input in every frame."""
    x = (0.05 * rng.normal(size=FS * 3)).astype(np.float32)
    for m in S.METHODS:
        y, _ = S.enhance(x, FS, m)
        fr = lambda z: np.sum(z[: z.size // 256 * 256].reshape(-1, 256) ** 2, axis=1)
        assert np.all(fr(y)[4:-4] <= fr(x)[4:-4] * 1.05)


def test_enhance_wav_writes_wav_and_sidecar_only(tmp_path, speechy):
    clean, noisy = speechy
    S.write_wav(tmp_path / "in.wav", noisy, FS)
    S.write_wav(tmp_path / "noise.wav",
                0.05 * np.random.default_rng(9).normal(size=FS), FS)
    out_dir = tmp_path / "out"
    rep = S.enhance_wav(tmp_path / "in.wav", out_dir / "in_mmse.wav", "mmse_lsa",
                        noise_path=tmp_path / "noise.wav")
    assert sorted(p.name for p in out_dir.iterdir()) == ["in_mmse.wav", "in_mmse.wav.json"]
    side = json.loads((out_dir / "in_mmse.wav.json").read_text())
    assert side["tier"] == "cleaned" and side["method"] == "mmse_lsa"
    assert side["source_sha256"] == provenance.sha256_path(tmp_path / "in.wav")
    assert "primed" in side["noise_tracker"]
    assert rep["clipped_samples"] == 0
    S.write_wav(tmp_path / "n16.wav", np.zeros(100), 16000)
    with pytest.raises(ValueError, match="must match"):
        S.enhance_wav(tmp_path / "in.wav", out_dir / "x.wav", "wiener",
                      noise_path=tmp_path / "n16.wav")


def test_classical_enhancers_are_callables(tmp_path, speechy):
    _, noisy = speechy
    S.write_wav(tmp_path / "in.wav", noisy, FS)
    encs = S.classical_enhancers()
    assert set(encs) == set(S.METHODS)
    r = encs["wiener"](tmp_path / "in.wav", tmp_path / "w.wav")
    assert r["tier"] == "cleaned" and (tmp_path / "w.wav").exists()
    with pytest.raises(ValueError, match="unknown enhancer"):
        S.enhance(noisy, FS, "deep_magic")


# ---------------------------------------------------------------------------
# The learned enhancer at arm's length (speech_learned)
# ---------------------------------------------------------------------------
FAKE_SGMSE = r'''
import argparse, glob, os, shutil, sys, time, wave
ap = argparse.ArgumentParser()
ap.add_argument("--test_dir", required=True)
ap.add_argument("--enhanced_dir", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--N", type=int, default=30)
ap.add_argument("--corrector", default="ald")
ap.add_argument("--corrector_steps", type=int, default=1)
ap.add_argument("--snr", type=float, default=0.5)
ap.add_argument("--device", default="cuda")
a = ap.parse_args()
if os.environ.get("FAKE_SLEEP"):
    time.sleep(float(os.environ["FAKE_SLEEP"]))
if os.environ.get("FAKE_FAIL"):
    sys.stderr.write("CUDA out of memory (fake)\n"); sys.exit(3)
assert open(a.ckpt, "rb").read().startswith(b"fake-checkpoint")
for f in sorted(glob.glob(os.path.join(a.test_dir, "*.wav"))):
    with wave.open(f, "rb") as r:
        p = r.getparams(); fr = r.readframes(p.nframes)
    os.makedirs(a.enhanced_dir, exist_ok=True)
    with wave.open(os.path.join(a.enhanced_dir, os.path.basename(f)), "wb") as w:
        w.setparams(p); w.writeframes(fr)
print("enhanced", a.N, a.device)
'''


@pytest.fixture
def fake_sgmse(tmp_path):
    repo = tmp_path / "sgmse"
    repo.mkdir()
    (repo / "enhancement.py").write_text(FAKE_SGMSE, encoding="utf-8")
    ck = tmp_path / "models" / "fake.ckpt"
    ck.parent.mkdir()
    ck.write_bytes(b"fake-checkpoint" + bytes(range(256)) * 8)
    return repo, ck, provenance.sha256_path(ck)


def test_sgmse_runner_plumbing(tmp_path, fake_sgmse, speechy):
    from atk_diffusion.repair import speech_learned as L
    repo, ck, sha = fake_sgmse
    _, noisy = speechy
    S.write_wav(tmp_path / "clip.wav", noisy, FS)
    run = L.SgmseRunner(sys.executable, repo, ck, sha, device="cpu", steps=5)
    ok, why = run.available()
    assert ok, why
    cmd = run.command(tmp_path / "i", tmp_path / "o")
    assert cmd[0] == sys.executable and cmd[1].endswith("enhancement.py")
    assert cmd[cmd.index("--ckpt") + 1] == str(ck)
    assert cmd[cmd.index("--N") + 1] == "5" and cmd[cmd.index("--device") + 1] == "cpu"
    out = tmp_path / "enh" / "clip_sgmse.wav"
    rep = run.enhance(tmp_path / "clip.wav", out)
    assert rep["tier"] == "invented" and rep["method"] == "speech_enhance"
    assert rep["model_sha256"] == sha
    y, fs, _ = S.read_wav(out)
    assert fs == FS and y.size == noisy.size
    side = json.loads(out.with_name(out.name + ".json").read_text())
    assert side["tier"] == "invented" and "INVENTED" in side["tier_words"]
    # the work folder is beside the output and is gone afterwards
    assert sorted(p.name for p in out.parent.iterdir()) == ["clip_sgmse.wav",
                                                            "clip_sgmse.wav.json"]


def test_sgmse_refuses_a_checkpoint_whose_hash_changed(tmp_path, fake_sgmse, speechy):
    from atk_diffusion.repair import speech_learned as L
    repo, ck, sha = fake_sgmse
    S.write_wav(tmp_path / "clip.wav", speechy[1], FS)
    run = L.SgmseRunner(sys.executable, repo, ck, "0" * 64)
    ok, why = run.available()
    assert not ok and "hash" in why
    with pytest.raises(L.EnhancerRefusal, match="hash"):
        run.enhance(tmp_path / "clip.wav", tmp_path / "o.wav")
    assert not (tmp_path / "o.wav").exists()
    ck.write_bytes(b"fake-checkpoint tampered")
    run2 = L.SgmseRunner(sys.executable, repo, ck, sha)
    assert not run2.available()[0]


def test_sgmse_unavailable_reasons_are_sentences(tmp_path, fake_sgmse):
    from atk_diffusion.repair import speech_learned as L
    repo, ck, sha = fake_sgmse
    cases = [L.SgmseRunner(tmp_path / "nopython.exe", repo, ck, sha),
             L.SgmseRunner(sys.executable, tmp_path / "nothere", ck, sha),
             L.SgmseRunner(sys.executable, repo, tmp_path / "none.ckpt", sha),
             L.SgmseRunner(sys.executable, repo, ck, "")]
    for run in cases:
        ok, why = run.available()
        assert not ok and why.endswith(".") and len(why.split()) > 5


def test_sgmse_failure_and_timeout_are_reported(tmp_path, fake_sgmse, speechy, monkeypatch):
    from atk_diffusion.repair import speech_learned as L
    repo, ck, sha = fake_sgmse
    S.write_wav(tmp_path / "clip.wav", speechy[1], FS)
    monkeypatch.setenv("FAKE_FAIL", "1")
    run = L.SgmseRunner(sys.executable, repo, ck, sha)
    with pytest.raises(L.EnhancerFailed, match="out of memory"):
        run.enhance(tmp_path / "clip.wav", tmp_path / "o1.wav")
    monkeypatch.delenv("FAKE_FAIL")
    monkeypatch.setenv("FAKE_SLEEP", "20")
    run = L.SgmseRunner(sys.executable, repo, ck, sha, timeout_s=1.0)
    with pytest.raises(L.EnhancerFailed, match="did not finish"):
        run.enhance(tmp_path / "clip.wav", tmp_path / "o2.wav")
    assert not (tmp_path / "o2.wav").exists()


def test_sgmse_from_card(tmp_path, fake_sgmse, speechy):
    from atk_diffusion import cards
    from atk_diffusion.repair import speech_learned as L
    repo, ck, sha = fake_sgmse
    card_dir = ck.parent
    L.write_card(card_dir, ck.name, name="sgmse-fake", license="MIT (test)")
    run = L.SgmseRunner.from_card(card_dir, sys.executable, repo)
    assert run.sha256 == sha and run.available()[0]
    assert cards.load(card_dir, expect_kind="speech_enhance").tier == "invented"
    S.write_wav(tmp_path / "clip.wav", speechy[1], FS)
    assert run.enhance(tmp_path / "clip.wav", tmp_path / "c.wav")["license"] == "MIT (test)"


def test_external_enhancer_generic_cli(tmp_path, speechy):
    from atk_diffusion.repair import speech_learned as L
    script = tmp_path / "df.py"
    script.write_text(
        "import sys, shutil, os\n"
        "src, out_dir = sys.argv[1], sys.argv[3]\n"
        "os.makedirs(out_dir, exist_ok=True)\n"
        "stem = os.path.splitext(os.path.basename(src))[0]\n"
        "shutil.copy(src, os.path.join(out_dir, stem + '_DeepFilterNet3.wav'))\n",
        encoding="utf-8")
    S.write_wav(tmp_path / "clip.wav", speechy[1], FS)
    ext = L.ExternalEnhancer("deepfilternet", [sys.executable, str(script), "{in}",
                                               "-o", "{out_dir}"],
                             output_glob="*_DeepFilterNet3.wav")
    assert ext.available()[0]
    rep = ext.enhance(tmp_path / "clip.wav", tmp_path / "e" / "clip_df.wav")
    assert rep["tier"] == "invented" and (tmp_path / "e" / "clip_df.wav").exists()
    bad = L.ExternalEnhancer("nothing", [str(tmp_path / "missing.exe"), "{in}"])
    ok, why = bad.available()
    assert not ok and "not found" in why
    ok, why = L.available()
    assert not ok and "not set up" in why and "classical" in why
