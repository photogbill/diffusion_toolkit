# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Word / character error rate and the D2 with-and-without experiment."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from atk_diffusion.repair import wer as W


def test_normalisation_is_the_documented_one():
    assert W.normalize("“Don't” Push-to-Talk, O'Clock… 'quoted' — 10/12 Ça VA?") \
        == "don't push to talk o'clock quoted 10 12 ça va"
    assert W.normalize("  Multiple\t\nspaces ") == "multiple spaces"
    assert W.normalize("ten 10") == "ten 10"          # numbers are not spelled out


def test_counts_are_exact():
    s = W.wer("the cat sat on the mat", "the cat sat on mat mat now")
    assert (s.substitutions, s.deletions, s.insertions) == (1, 0, 1)
    assert s.ref_len == 6 and math.isclose(s.rate, 2 / 6)
    s = W.wer("alpha bravo charlie", "alpha charlie", keep_ops=True)
    assert (s.substitutions, s.deletions, s.insertions) == (0, 1, 0)
    assert [o[0] for o in s.ops] == ["=", "D", "="]
    c = W.cer("kitten", "sitting")
    assert c.errors == 3 and c.unit == "character"


def test_matches_a_plain_levenshtein(rng):
    def lev(a, b):
        d = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            prev, d[0] = d[0], i
            for j, y in enumerate(b, 1):
                prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
        return d[-1]
    for _ in range(400):
        a = list(rng.choice(list("abcd"), size=rng.integers(0, 10)))
        b = list(rng.choice(list("abcd"), size=rng.integers(0, 10)))
        s = W.align(a, b, keep_ops=True)
        assert s.errors == lev(a, b)
        assert s.hits + s.substitutions + s.deletions == len(a)
        assert s.hits + s.substitutions + s.insertions == len(b)
        assert [o[1] for o in s.ops if o[0] in "=SD"] == a
        assert [o[2] for o in s.ops if o[0] in "=SI"] == b


def test_empty_reference_is_undefined_but_insertions_count():
    s = W.wer("", "thank you for watching")
    assert math.isnan(s.rate) and s.insertions == 4
    assert W.wer("", "").rate == 0.0
    assert W.wer("hello there", "").deletions == 2
    assert "undefined" in s.words()


def test_corpus_is_total_errors_over_total_words():
    pairs = [("a b c d e f g h i j", "a b c d e f g h i j"), ("x", "y")]
    c = W.corpus(pairs)
    assert c.ref_len == 11 and c.errors == 1 and math.isclose(c.rate, 1 / 11)


def test_text_of_accepts_the_hosts_shapes():
    class R:
        text = "from an object"
    assert W.text_of("plain") == "plain"
    assert W.text_of({"text": "from a dict"}) == "from a dict"
    assert W.text_of(R()) == "from an object"
    assert W.text_of(None) == ""


def test_long_cer_is_fast(rng):
    import time
    a = "".join(rng.choice(list("abcdefgh "), size=3000))
    b = "".join(rng.choice(list("abcdefgh "), size=3100))
    t0 = time.time()
    W.cer(a, b)
    assert time.time() - t0 < 5.0


# ---------------------------------------------------------------------------
# experiments.wer_eval — the harness, with a stand-in transcriber
# ---------------------------------------------------------------------------
def _floor_transcriber(refs_by_stem, floor_db=-40.0):
    """A deterministic stand-in for Whisper: it returns the reference when the
    clip's noise floor is low, and otherwise drops the first word and
    'hears' two words that were never said — so enhancement measurably
    changes its output. It tests the harness, not speech recognition."""
    from atk_diffusion.repair.speech import read_wav

    def transcribe(path):
        x, fs, _ = read_wav(path)
        fr = x[: x.size // 160 * 160].reshape(-1, 160)
        rms = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-12)
        ref = refs_by_stem[Path(path).stem]
        if np.percentile(rms, 10) < floor_db:
            return {"text": ref}
        return {"text": " ".join(ref.split()[1:] + ["thank", "you"])}
    return transcribe


def test_wer_with_without_on_the_same_clips(tmp_path, rng):
    from atk_diffusion.experiments.wer_eval import wer_with_without
    from atk_diffusion.repair import speech as S
    from atk_diffusion.repair.speech_learned import EnhancerFailed
    refs = {"c1": "alpha bravo charlie delta", "c2": "echo foxtrot golf",
            "c3": "hotel india juliet kilo lima", "n1": "", "n2": ""}
    clips = []
    for stem in ("c1", "c2", "c3"):
        x = S.speech_like(8000, 2.0, rng) + 0.02 * rng.normal(size=16000)
        S.write_wav(tmp_path / f"{stem}.wav", x, 8000)
        clips.append(tmp_path / f"{stem}.wav")
    noise = []
    for stem in ("n1", "n2"):
        S.write_wav(tmp_path / f"{stem}.wav", 0.02 * rng.normal(size=16000), 8000)
        noise.append(tmp_path / f"{stem}.wav")

    def broken(src, dst):
        raise EnhancerFailed("the fake learned enhancer ran out of memory")

    encs = {"mmse_lsa": S.classical_enhancers(["mmse_lsa"])["mmse_lsa"],
            "learned_fake": broken}
    res = wer_with_without(clips, [refs[c.stem] for c in clips],
                           _floor_transcriber(refs), encs,
                           out_dir=tmp_path / "eval", noise_clips=noise)
    rows = {r["method"]: r for r in res["rows"]}
    assert [r["method"] for r in res["rows"]] == ["raw", "mmse_lsa", "learned_fake"]
    raw = rows["raw"]
    # each raw clip: 1 deletion + 2 insertions (+ shifted alignment) -> 3 errors
    assert raw["clips"] == 3 and raw["deletions"] == 3 and raw["insertions"] == 6
    assert math.isclose(raw["wer"], 9 / 12)
    m = rows["mmse_lsa"]
    assert m["tier"] == "cleaned" and m["wer"] == 0.0
    assert math.isclose(m["raw_wer_same_clips"], raw["wer"])
    assert m["better"] == 3 and m["worse"] == 0 and m["delta_wer"] < 0
    f = rows["learned_fake"]
    assert f["clips"] == 0 and f["failed"] == 3
    assert "out of memory" in f["failures"][0]["why"]
    h = {x["method"]: x for x in res["hallucination"]}
    assert h["raw"]["words"] == 4 and h["raw"]["clips_with_words"] == 2
    assert h["mmse_lsa"]["words"] == 0
    assert h["learned_fake"]["failed"] == 2
    assert any("lowered the WER" in ln for ln in res["lines"])
    md = Path(res["report_md"]).read_text(encoding="utf-8")
    assert "| raw | record | 3 |" in md and "Hallucination check" in md
    js = json.loads(Path(res["report_json"]).read_text(encoding="utf-8"))
    assert js["rows"][1]["method"] == "mmse_lsa"
    assert (tmp_path / "eval" / "mmse_lsa" / "c1.wav").exists()


def test_wer_eval_refuses_without_an_output_folder(tmp_path):
    from atk_diffusion.experiments.wer_eval import wer_with_without
    with pytest.raises(ValueError, match="output folder"):
        wer_with_without([tmp_path / "a.wav"], ["x"], lambda p: "x")
    with pytest.raises(ValueError, match="pair one to one"):
        wer_with_without([tmp_path / "a.wav"], ["x", "y"], lambda p: "x",
                         out_dir=tmp_path)
