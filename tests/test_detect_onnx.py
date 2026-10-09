# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""ONNX Runtime wrappers on the CPU (DETECTION_DESIGN §7; ARCHITECTURE §5).

Tiny models are exported here with exactly the contract's I/O names and
saved with cards, then run through the real onnxruntime — never a stub
written from the docs (ARCHITECTURE §2 rule 10). Skipped, with the reason,
where PyTorch or ONNX Runtime is absent."""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pytest

from atk_diffusion import cards, profiles
from atk_diffusion.detect import classes as C
from atk_diffusion.dsp.stft import Tile
from atk_diffusion.profiles import StftGeometry

PID = "rtlsdr_256000_cu8"
FS = 256_000.0
GEOM = StftGeometry(fft_size=256, hop=256, tile_seconds=0.32, tile_rows=64,
                    tile_overlap=0.25)
BOXES = [[8.0, 40.0, 24.0, 60.0], [30.0, 150.0, 50.0, 200.0]]
LABELS = [3, 6]
L, H, W = 256, 8, 8
CLASSES = ["dmr", "p25", "nfm_voice"]


def _torch():
    torch = pytest.importorskip("torch", reason="PyTorch is only in the training "
                                "environment")
    pytest.importorskip("onnxruntime", reason="ONNX Runtime is not installed")
    torch.set_num_threads(1)
    return torch


def _export(torch, model, args, path, inputs, outputs, dynamic=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(model, args, str(path), input_names=list(inputs),
                          output_names=list(outputs), dynamo=False,
                          opset_version=17, dynamic_axes=dynamic)


def _proposer_model(torch):
    class Tiny(torch.nn.Module):
        """Scores each of two fixed boxes by the mean of the tile inside it."""

        def __init__(self):
            super().__init__()
            self.register_buffer("b", torch.tensor(BOXES))
            self.register_buffer("lab", torch.tensor(LABELS, dtype=torch.int64))

        def forward(self, tile):
            zero = tile.sum() * 0.0
            m1 = tile[0, 0, 8:24, 40:60].mean()
            m2 = tile[0, 0, 30:50, 150:200].mean()
            scores = torch.sigmoid(20.0 * (torch.stack([m1, m2]) - 0.15))
            return self.b + zero, scores, self.lab + zero.to(torch.int64)
    return Tiny().eval()


def _classifier_model(torch):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(0)
            self.head = torch.nn.Linear(4, len(CLASSES))
            self.emb = torch.nn.Linear(4, 8)
            self.cyc = torch.nn.Linear(4, 2)
            for p in self.parameters():
                with torch.no_grad():
                    p.copy_(torch.randn(p.shape, generator=g))

        def forward(self, iq, scf):
            f = torch.cat([iq.mean(dim=2), (iq * iq).mean(dim=(1, 2))[:, None],
                           scf.mean(dim=(2, 3))], dim=1)
            e = self.emb(f)
            e = e / torch.sqrt((e * e).sum(dim=1, keepdim=True) + 1e-12)
            return self.head(f), e, self.cyc(f)
    return Tiny().eval()


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    torch = _torch()
    root = tmp_path_factory.mktemp("onnx_models")
    out = {}
    # the proposer, with a card
    d = root / "prop"
    d.mkdir()
    _export(torch, _proposer_model(torch), (torch.zeros(1, 1, 64, 256),),
            d / "model.onnx", ["tile"], ["boxes", "scores", "labels"])
    cards.save(d, cards.new_card(
        "tiny-proposer", "proposer2d", PID,
        input={"stft": {"fft_size": 256, "hop": 256, "window": "hann", "tile_rows": 64},
               "families": list(C.FAMILIES),
               "normalize": {"clip_db": [0, 40], "offset_db": 0, "scale_db": 40}},
        calibration={"min_score": 0.5}), "model.onnx")
    out["prop"] = d
    # the same graph with the wrong output names
    bad = root / "badnames"
    bad.mkdir()
    _export(torch, _proposer_model(torch), (torch.zeros(1, 1, 64, 256),),
            bad / "model.onnx", ["tile"], ["b", "s", "l"])
    cards.save(bad, cards.new_card("bad-names", "proposer2d", PID), "model.onnx")
    out["bad"] = bad
    # the classifier, dynamic batch
    cin = {"iq_len": L, "canonical": {"cls": "voice", "rate": 64000.0, "decimation": 4},
           "fam": {"channel_fft": 32, "hop": 8}, "cycle_scale": [4800.0, 1000.0],
           "iq_norm": "rms", "scf_norm": "max", "scf_shape": [H, W]}
    dyn = {k: {0: "batch"} for k in ("iq", "scf", "logits", "embedding", "cycle")}
    for name, dynamic in (("clf", dyn), ("clf_fixed", None)):
        d = root / name
        d.mkdir()
        _export(torch, _classifier_model(torch),
                (torch.zeros(1, 2, L), torch.zeros(1, 1, H, W)), d / "model.onnx",
                ["iq", "scf"], ["logits", "embedding", "cycle"], dynamic)
        cards.save(d, cards.new_card(
            f"tiny-{name}", "classifier1d", PID, input=dict(cin),
            classes=[{"name": c, "source": "trained", "examples": 100} for c in CLASSES],
            calibration={"temperature": 2.0}), "model.onnx")
        out[name] = d
    return out


def _tile(spec=None, t0=4.0):
    s = np.zeros((64, 256), np.float32) if spec is None else spec
    return Tile.from_spec(s, FS, 162.4e6, GEOM, t0=t0, profile=PID, epoch=1.7e9)


# ---------------------------------------------------------------------------
def test_normalize_tile_forms():
    from atk_diffusion.detect.onnx_models import normalize_tile
    x = np.array([[-5.0, 10.0, 50.0]], np.float32)
    assert np.allclose(normalize_tile(x, {"clip_db": [0, 40], "scale_db": 40}),
                       [[0.0, 0.25, 1.0]])
    assert np.allclose(normalize_tile(x, {"mean": 10.0, "std": 5.0}), [[-3, 0, 8]])
    assert normalize_tile(x, None) is not None
    with pytest.raises(ValueError, match="normalize"):
        normalize_tile(x, {"gamma": 2})


def test_proposer_boxes_are_absolute_and_familied(models):
    from atk_diffusion.detect.onnx_models import Proposer2D
    p = Proposer2D(models["prop"], profile=PID, threads=1)
    spec = np.zeros((64, 256), np.float32)
    spec[8:24, 40:60] = 20.0                       # something in box 1 only
    tile = _tile(spec)
    dets = p.run(tile)
    assert len(dets) == 1
    d = dets[0]
    t0, t1, f_lo, f_hi = tile.pixels_to_tf(*BOXES[0])
    assert (d.t0, d.t1, d.f_lo, d.f_hi) == pytest.approx((t0, t1, f_lo, f_hi))
    assert d.sources == ("learned",) and d.family == C.FAMILIES[LABELS[0]]
    assert d.cls == "" and d.state == "proposed"           # a family, never a modulation
    assert 0.5 <= d.confidence <= 1.0 and d.profile == PID and d.epoch == 1.7e9
    assert d.snr_db == pytest.approx(20.0, abs=0.1)         # measured from the tile beside
    assert len(p.run(_tile())) == 0                         # nothing there
    assert len(p.run(_tile(), min_score=0.0)) == 2
    lat = p.latency.summary()
    assert lat["calls"] == 3 and lat["mean_ms"] > 0 and lat["peak_ms"] >= lat["mean_ms"]


def test_proposer_refusals_are_sentences(models, tmp_path):
    from atk_diffusion.detect.onnx_models import Classifier1D, Proposer2D
    with pytest.raises(profiles.ProfileMismatch, match="trained for the RTL-SDR"):
        Proposer2D(models["prop"], profile="bladerf1_4000000_ci16")
    with pytest.raises(cards.CardRefusal, match="not the 1D classifier"):
        Classifier1D(models["prop"])
    with pytest.raises(cards.CardRefusal, match="Re-export it with those names"):
        Proposer2D(models["bad"], profile=PID)
    with pytest.raises(cards.CardRefusal, match="without a card does not load"):
        Proposer2D(tmp_path)
    p = Proposer2D(models["prop"], profile=PID, threads=1)
    other = Tile.from_spec(np.zeros((128, 256), np.float32), FS, 0.0,
                           StftGeometry(fft_size=256, hop=256, tile_seconds=0.64,
                                        tile_rows=128))
    with pytest.raises(ValueError, match="another spectrogram geometry"):
        p.run(other)


def test_classifier_calibrated_probs_embeddings_and_cycle(models, rng):
    import onnxruntime as ort
    from atk_diffusion.detect.onnx_models import Classifier1D
    c = Classifier1D(models["clf"], profile=PID, threads=1)
    assert c.canonical_class == "voice" and c.canonical_rate == 64000.0
    iq = (rng.normal(size=(5, L)) + 1j * rng.normal(size=(5, L))).astype(np.complex64)
    iq[2] *= 30.0                                       # rms-normalised away
    scf = rng.random((5, H, W)).astype(np.float32)
    out = c.run(iq, scf)
    assert out.probs.shape == (5, 3) and np.allclose(out.probs.sum(axis=1), 1.0)
    assert np.allclose(np.linalg.norm(out.embeddings, axis=1), 1.0)
    assert out.top == [CLASSES[i] for i in out.probs.argmax(axis=1)]
    assert np.allclose(out.confidence, out.probs.max(axis=1))
    # the calibration and the denormalisation are exactly the card's
    s = ort.InferenceSession(str(models["clf"] / "model.onnx"),
                             providers=["CPUExecutionProvider"])
    x = iq / np.sqrt(np.mean(np.abs(iq) ** 2, axis=1, keepdims=True))
    feed = {"iq": np.stack([x.real, x.imag], 1).astype(np.float32),
            "scf": (scf / scf.max(axis=(1, 2), keepdims=True))[:, None].astype(np.float32)}
    logits, _e, cyc = s.run(["logits", "embedding", "cycle"], feed)
    z = np.exp(logits / 2.0 - (logits / 2.0).max(1, keepdims=True))
    assert np.allclose(out.probs, z / z.sum(1, keepdims=True), atol=1e-5)
    assert np.allclose(out.cycle_hz, cyc * np.array([4800.0, 1000.0]), rtol=1e-5)
    assert c.latency.summary()["calls"] == 1
    with pytest.raises(ValueError, match="windows of 256 samples"):
        c.run(iq[:, :100], scf)
    with pytest.raises(ValueError, match="trained on 8 x 8"):
        c.run(iq, np.zeros((5, 4, 4), np.float32))


def test_a_fixed_batch_graph_is_run_window_by_window(models, rng):
    from atk_diffusion.detect.onnx_models import Classifier1D
    dyn = Classifier1D(models["clf"], profile=PID, threads=1)
    fixed = Classifier1D(models["clf_fixed"], profile=PID, threads=1)
    assert fixed.batch_fixed == 1 and dyn.batch_fixed is None
    iq = (rng.normal(size=(3, L)) + 1j * rng.normal(size=(3, L))).astype(np.complex64)
    scf = rng.random((3, H, W)).astype(np.float32)
    a, b = dyn.run(iq, scf), fixed.run(iq, scf)
    assert np.allclose(a.probs, b.probs, atol=1e-5)
    assert np.allclose(a.embeddings, b.embeddings, atol=1e-5)


def test_without_scf_images_the_cyclo_module_is_asked_for(models, rng, monkeypatch):
    from atk_diffusion.detect.onnx_models import Classifier1D
    c = Classifier1D(models["clf"], profile=PID, threads=1)
    monkeypatch.setitem(sys.modules, "atk_diffusion.cyclo.scf", None)
    iq = (rng.normal(size=(2, L)) + 1j * rng.normal(size=(2, L))).astype(np.complex64)
    with pytest.raises(RuntimeError, match="atk_diffusion.cyclo.scf is not available"):
        c.run(iq)


def test_integration_with_the_real_scf_module(models, tmp_path, rng):
    """Runs once atk_diffusion.cyclo.scf exists (written beside this module)."""
    try:
        from atk_diffusion.cyclo.scf import scf_image
    except ImportError as e:
        pytest.skip(f"atk_diffusion.cyclo.scf is not importable yet: {e}")
    torch = _torch()
    from atk_diffusion.detect.onnx_models import Classifier1D
    from atk_diffusion.profiles import FamGeometry
    iq = (rng.normal(size=(2, L)) + 1j * rng.normal(size=(2, L))).astype(np.complex64)
    res = scf_image(iq[0], 64000.0, FamGeometry(channel_fft=32, hop=8))
    img = np.asarray(res[0] if isinstance(res, tuple) else res)
    h, w = img.shape[-2:]
    d = tmp_path / "clf_scf"
    d.mkdir()
    _export(torch, _classifier_model(torch), (torch.zeros(1, 2, L), torch.zeros(1, 1, h, w)),
            d / "model.onnx", ["iq", "scf"], ["logits", "embedding", "cycle"],
            {k: {0: "batch"} for k in ("iq", "scf", "logits", "embedding", "cycle")})
    cards.save(d, cards.new_card(
        "tiny-scf", "classifier1d", PID,
        input={"iq_len": L, "canonical": {"cls": "voice", "rate": 64000.0, "decimation": 4},
               "fam": {"channel_fft": 32, "hop": 8}, "scf_shape": [int(h), int(w)]},
        classes=[{"name": c} for c in CLASSES]), "model.onnx")
    out = Classifier1D(d, profile=PID, threads=1).run(iq)
    assert out.probs.shape == (2, 3) and np.all(np.isfinite(out.embeddings))
