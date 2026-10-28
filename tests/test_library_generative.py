# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""AI-generated variations (RFMLS Training): the generator is taught on the
library's real originals plus synthetic classes, a variation that does not
measure like its parent is thrown away, and every kept one is INVENTED and
names its model. Tiny networks on the CPU; the numbers mean nothing here."""

from __future__ import annotations

import numpy as np
import pytest

from atk_diffusion import sigmf
from atk_diffusion.library import generative as G
from atk_diffusion.library import store

from tests.test_library_store import PID, _box, _capture


def _nfm(fs=48_000.0, n=4096, off=0.0, seed=0):
    from atk_diffusion.synth import native
    rng = np.random.default_rng(seed)
    s, lab = native.waveform("nfm_voice", fs, n, rng, carrier_offset_hz=off)
    noise = native.noise(n, 1e-6, rng)
    return (s + noise).astype(np.complex64), lab


def test_the_gate_keeps_a_lookalike_and_throws_away_what_moved():
    x, lab = _nfm(seed=1)
    same, _ = _nfm(seed=2)
    q = G.quality(x, same, 48_000.0, None, 12_000.0)
    assert q["ok"], q
    moved, _ = _nfm(off=9_000.0, seed=3)
    q2 = G.quality(x, moved, 48_000.0, None, 20_000.0)
    assert not q2["ok"] and "thrown away" in q2["words"]
    rng = np.random.default_rng(4)
    noise = (rng.standard_normal(4096) + 1j * rng.standard_normal(4096)).astype(np.complex64) * 1e-3
    q3 = G.quality(x, noise, 48_000.0, None, 12_000.0)
    assert not q3["ok"]


def test_no_generator_is_said_in_words(rf):
    assert G.find(rf, PID, "voice") is None
    assert "No generator" in G.describe(rf, PID, "voice")


def test_teach_and_generate_end_to_end_tiny(rf):
    torch = pytest.importorskip("torch", reason="PyTorch is only in the "
                                "training environment")
    torch.set_num_threads(1)
    lib = store.TrainingLibrary(rf, PID)
    base, lab = _capture(rf)
    lib.new_type("fire", link="nfm_voice")
    ex = lib.add_original_from_capture(base, _box(lab), "fire")
    d = G.teach(rf, PID, "voice", preset="quick", steps=12, window=256,
                unet="tiny", synthetic_per_class=1, windows_per_original=4,
                device="cpu", include_dirty=False)
    found = G.find(rf, PID, "voice")
    assert found is not None and found[0] == d
    card = found[1]
    assert card.input["library"] and card.input["types"] == ["fire"]
    assert card.input["synthetic_classes"]
    assert "No generator" not in G.describe(rf, PID, "voice")
    res = G.generate(rf, PID, "fire", 3, strength=0.2, steps=3, device="cpu",
                     batch=4)
    assert res["made"] + res["thrown_away"] >= res["made"]
    assert "AI-generated" in res["words"]
    for c in lib.examples("fire", "generated"):
        g = sigmf.read_meta(c.base)["global"]
        assert g["atk:tier"] == "invented"
        assert g["atk:parent"] == ex.id
        assert g["atk:generator_model"] == card.name
        assert g["atk:quality"]["ok"]


def test_a_type_the_generator_never_saw_is_refused(rf):
    pytest.importorskip("torch", reason="PyTorch is only in the training "
                        "environment")
    lib = store.TrainingLibrary(rf, PID)
    base, lab = _capture(rf)
    lib.new_type("fire", link="nfm_voice")
    lib.add_original_from_capture(base, _box(lab), "fire")
    G.teach(rf, PID, "voice", preset="quick", steps=4, window=256, unet="tiny",
            synthetic_per_class=0, windows_per_original=2, device="cpu")
    base2, lab2 = _capture(rf, name="c2", seed=9)
    lib.new_type("ems", link="nfm_voice")
    lib.add_original_from_capture(base2, _box(lab2), "ems")
    with pytest.raises(store.LibraryError, match="teach it again"):
        G.generate(rf, PID, "ems", 2, device="cpu")
