# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Audio gap filling for the Media Lab (repair.audio_inpaint, plan §4.D5):
what counts as a gap, the fill measured against leaving the zeros and
against a straight line, every filled span INFERRED, the audio outside the
gaps untouched, and the hallucination check — a fill must not put sound
where the truth is quiet."""

from __future__ import annotations

import json
import math
import wave
from pathlib import Path

import numpy as np
import pytest

from atk_diffusion import provenance
from atk_diffusion.repair import audio_inpaint as A

FS = 16_000


def voiced(seconds: float = 0.5, f0: float = 150.0, seed: int = 0) -> np.ndarray:
    """A vowel-like sound: harmonics of a slowly gliding pitch with a soft
    formant tilt, plus a little noise — the kind of audio a gap interrupts."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    t = np.arange(n) / FS
    f = f0 * (1 + 0.03 * np.sin(2 * np.pi * 2.0 * t))
    ph = 2 * np.pi * np.cumsum(f) / FS
    x = sum((0.5 / k) * np.sin(k * ph + k) for k in range(1, 9))
    x = 0.3 * x / np.max(np.abs(x))
    return (x + 1e-3 * rng.standard_normal(n)).astype(np.float64)


def err_db(fill, truth) -> float:
    """Error energy of a fill relative to the truth's energy, dB."""
    e = float(np.sum((np.asarray(fill) - truth) ** 2))
    return 10 * math.log10(max(e, 1e-30) / float(np.sum(truth ** 2)))


def write_wav(path, x, fs=FS):
    v = np.clip(np.round(np.asarray(x) * 32768.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(v.tobytes())


# -- what a gap is -------------------------------------------------------------
def test_a_short_zero_run_inside_audio_is_a_gap():
    x = voiced()
    x[4000:4080] = 0.0
    gaps = A.find_gaps(x, FS)
    assert len(gaps) == 1
    g = gaps[0]
    assert (g.start, g.count, g.kind) == (4000, 80, "zeros")
    assert "5.0 ms of identical samples inside active audio" == g.detail
    assert g.to_json()["count"] == 80


def test_a_stuck_value_is_a_gap_too():
    x = voiced()
    x[3000:3064] = 0.123
    g = A.find_gaps(x, FS)
    assert [(a.start, a.count, a.kind) for a in g] == [(3000, 64, "stuck")]


def test_silence_that_is_the_recording_is_left_alone():
    """DSD writes exact zeros between overs and a recorder pauses with
    zeros: long, at the edge, or beside quiet, it is not damage."""
    x = voiced(1.0)
    x[2000:2000 + int(0.4 * FS)] = 0.0             # 400 ms: too long
    assert A.find_gaps(x, FS) == []
    y = voiced()
    y[:100] = 0.0                                  # at the very start
    assert A.find_gaps(y, FS) == []
    z = voiced()
    z[1000:3000] *= 1e-4                           # quiet (-80 dBFS) ...
    z[2000:2050] = 0.0                             # ... with zeros inside it
    assert A.find_gaps(z, FS) == []
    assert A.find_gaps(voiced(), FS) == []         # nothing to find
    assert A.find_gaps(np.zeros(2), FS) == []


def test_marked_gaps_are_taken_as_marked():
    g = A.gaps_from_times([(0.25, 0.26), (0.3, 0.3), (0.1, 0.05)], FS)
    assert [(a.start, a.count, a.kind) for a in g] == [(4000, 160, "marked")]


# -- the fill, measured --------------------------------------------------------
@pytest.mark.parametrize("gap_ms, vs_zeros_db, vs_linear_db", [
    (2.0, 15.0, 5.0),        # measured: Janssen -21.7 dB, linear -13.3 dB
    (5.0, 30.0, 30.0),       # measured: Janssen -41.3 dB, linear +3.2 dB
])
def test_janssen_beats_zeros_and_a_straight_line(gap_ms, vs_zeros_db, vs_linear_db):
    """Error energy of the fill against the truth, relative to the truth's
    energy in the gap: leaving the zeros is 0 dB by definition. Across a
    5 ms gap a straight line is WORSE than the zeros; Janssen's AR fill is
    40 dB better than either."""
    truth = voiced(seed=1)
    s, c = 4000, int(gap_ms * 1e-3 * FS)
    x = truth.copy()
    x[s:s + c] = 0.0
    zeros = err_db(np.zeros(c), truth[s:s + c])              # 0 dB by definition
    out = {}
    for method in ("janssen", "lpc", "linear"):
        y, spans = A.fill(x, FS, A.find_gaps(x, FS), method)
        out[method] = err_db(y[s:s + c], truth[s:s + c])
        assert len(spans) == 1 and spans[0].tier == "inferred"
        assert spans[0].method == A.METHODS[method]
        assert spans[0].seconds == pytest.approx(c / FS)
        # the audio outside the gap is untouched, sample for sample
        keep = np.ones(truth.size, bool)
        keep[s:s + c] = False
        assert np.array_equal(y[keep], x[keep].astype(np.float32))
    assert zeros == pytest.approx(0.0)
    assert out["janssen"] < zeros - vs_zeros_db
    assert out["janssen"] < out["linear"] - vs_linear_db
    assert out["lpc"] < out["linear"]


def test_every_method_is_inferred_tier_in_the_provenance_table():
    for m in A.METHODS.values():
        assert provenance.tier_for(m) == "inferred"
    with pytest.raises(ValueError, match="unknown audio fill"):
        A.fill(np.zeros(10), FS, [], "magic")


def test_no_sound_is_invented_where_the_truth_is_quiet():
    """The hallucination check: a gap marked inside a quiet pause between
    two loud tones is filled at the pause's level — not with a tone."""
    rng = np.random.default_rng(4)
    tone = voiced(0.2, seed=2)
    pause = 3e-3 * rng.standard_normal(int(0.08 * FS))
    truth = np.concatenate([tone, pause, tone])
    s, c = tone.size + 480, 320                               # 20 ms, mid-pause
    x = truth.copy()
    x[s:s + c] = 0.0
    loud = 10 * math.log10(np.mean(tone ** 2))
    quiet = 10 * math.log10(np.mean(pause ** 2))
    for method in ("janssen", "lpc", "linear"):
        y, spans = A.fill(x, FS, [(s, c)], method)
        level = 10 * math.log10(np.mean(y[s:s + c].astype(np.float64) ** 2) + 1e-30)
        assert level < quiet + 6.0, (method, level, quiet)
        assert level < loud - 20.0
        assert spans[0].fill_to_context_db is not None
        assert spans[0].fill_to_context_db < 3.0
        assert "louder" not in spans[0].note


def test_a_fill_louder_than_its_context_is_flagged():
    x = 1e-3 * np.random.default_rng(0).standard_normal(4000)
    x[1999] = x[2100] = 0.9                       # two clicks at the edges
    y, spans = A.fill(x, FS, [(2000, 100)], "linear")
    assert spans[0].fill_to_context_db > 3.0
    assert "louder than the audio around it" in spans[0].note


def test_a_gap_too_long_to_solve_falls_back_and_says_so():
    x = np.tile(voiced(0.25, seed=3), 4)
    s, c = 6000, 5000                              # > 4096: no Janssen solve
    x[s:s + c] = 0.0
    y, spans = A.fill(x, FS, [A.AudioGap(s, c, "marked")], "janssen")
    assert spans[0].method == "lpc_fill" and spans[0].tier == "inferred"
    assert "janssen_fill could not run here" in spans[0].note
    assert np.isfinite(y).all()


def test_spans_are_clipped_to_the_audio_and_empty_ones_skipped():
    x = voiced(0.1)
    y, spans = A.fill(x, FS, [(x.size - 10, 50), (x.size + 5, 10), (0, 0)], "linear")
    assert [(s.start, s.count) for s in spans] == [(x.size - 10, 10)]
    assert y.dtype == np.float32 and y.size == x.size


# -- the WAV path, with its sidecar ---------------------------------------------
def test_inpaint_wav_writes_the_fill_and_lists_every_inferred_sample(tmp_path):
    x = voiced(seed=5)
    x[5000:5100] = 0.0
    src = tmp_path / "clip.wav"
    write_wav(src, x)
    out = tmp_path / "clip_filled.wav"
    side = A.inpaint_wav(src, out)
    assert out.exists() and Path(side["sidecar"]).exists()
    assert side["tier"] == "inferred" and side["method"] == "janssen_fill"
    assert side["tier_words"].startswith("INFERRED")
    assert side["source_sha256"] == provenance.sha256_path(src)
    assert [(g["start"], g["count"]) for g in side["gaps"]] == [(5000, 100)]
    assert side["filled"][0]["tier"] == "inferred"
    assert side["seconds_inferred"] == pytest.approx(100 / FS, abs=1e-4)
    assert "100 ms" not in side["lines"][0] and "6 ms of audio is INFERRED" in side["lines"][0]
    disk = json.loads(Path(side["sidecar"]).read_text("utf-8"))
    assert disk["filled"] == side["filled"]
    with wave.open(str(out), "rb") as w:
        y = np.frombuffer(w.readframes(w.getnframes()), "<i2") / 32768.0
    assert np.max(np.abs(y[5000:5100])) > 0.01           # the gap was bridged


def test_inpaint_wav_with_marked_gaps_and_with_none(tmp_path):
    x = voiced(seed=6)
    src = tmp_path / "a.wav"
    write_wav(src, x)
    side = A.inpaint_wav(src, tmp_path / "b.wav", method="lpc",
                         gaps=[(0.1, 0.105)])
    assert side["gaps"][0]["kind"] == "marked" and side["method"] == "lpc_fill"
    clean = A.inpaint_wav(src, tmp_path / "c.wav")
    assert clean["tier"] == "record" and clean["filled"] == []
    assert clean["lines"] == ["No gaps found; the output is the input."]
