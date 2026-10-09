# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Research track (plan §4.R): the beacon co-designed with its detector (a
design study — nothing transmits) and CyberWolf anomaly context by
reconstruction error (context, never suppression)."""

from __future__ import annotations

import copy
import inspect

import numpy as np
import pytest

from atk_diffusion import cards, profiles
from atk_diffusion.learn import anomaly as A
from atk_diffusion.learn import beacon as B

RTL = "rtlsdr_2400000_cu8"


# -- the beacon: classical parts (numpy) ------------------------------------------
def test_the_bpsk_baseline_codebook():
    cb = B.bpsk_codebook(4, 8)
    assert cb.shape == (16, 8) and np.all(cb[:, 0] == 1)          # the pilot
    assert np.allclose(np.mean(np.abs(cb) ** 2, axis=1), 1.0)
    assert len({tuple(r) for r in cb.real}) == 16
    neg = {tuple(-r) for r in cb.real}
    assert not neg & {tuple(r) for r in cb.real}         # phase-ambiguity free


def test_noncoherent_ml_decodes_through_an_unknown_phase():
    rng = np.random.default_rng(0)
    cb = B.bpsk_codebook(4, 8)
    msg = rng.integers(0, 16, 200)
    imp = B.Impairments(adc_bits=0, iq_gain_db=0.0, iq_phase_deg=0.0, dc=0.0)
    y = B.channel(cb[msg], 30.0, imp, rng)
    d, _s = B.noncoherent_ml(y, cb)
    assert np.all(d == msg)


def test_the_channel_quantises_like_the_receiver():
    rng = np.random.default_rng(1)
    x = np.ones((4, 16), dtype=complex)
    imp = B.Impairments(adc_bits=3, full_scale=4.0, random_phase=False, cfo_max=0.0)
    y = B.channel(x, 40.0, imp, rng)
    step = 4.0 / 4                                             # fs / 2^(bits-1)
    assert np.allclose(np.round(y.real / step), y.real / step)
    z = B.channel(x, 0.0, imp, rng, signal=False)
    assert np.abs(z).mean() < 3.0                               # noise alone


def test_impairments_follow_the_receiver_profile():
    assert B.Impairments.for_profile(RTL).adc_bits == 8
    assert B.Impairments.for_profile("bladerf1_4000000_ci16").adc_bits == 12
    p = profiles.new_profile(RTL)
    p.impairments = {"iq_imbalance_db": 1.2, "dc_offset": 0.05}
    imp = B.Impairments.for_profile(p)
    assert imp.iq_gain_db == 1.2 and imp.dc == 0.05


def test_the_beacon_module_is_a_study_that_never_transmits():
    doc = B.__doc__
    assert "NOTHING IN THIS MODULE TRANSMITS" in doc and "authorisation" in doc
    src = inspect.getsource(B)
    assert "atk_diffusion.cabled" not in src and "hackrf_transfer" not in src
    public = [n for n in dir(B) if not n.startswith("_") and callable(getattr(B, n))]
    assert not [n for n in public if "transmit" in n.lower() or n.lower() == "tx"]


# -- the beacon: the autoencoder (PyTorch) -------------------------------------------
def test_beacon_autoencoder_trains_saves_loads_and_is_scored_against_bpsk(tmp_path):
    torch = pytest.importorskip("torch", reason="PyTorch is only in the training "
                                                "environment")
    torch.set_num_threads(1)
    m = B.train(RTL, tmp_path, steps=150, batch=128, seed=0)
    card = cards.load(tmp_path, expect_kind="beacon", for_profile=RTL)
    assert card.profile == RTL and B.NOT_A_TRANSMITTER in card.notes
    assert card.input["impairments"]["adc_bits"] == 8
    w = m.waveform(3)
    assert w.shape == (8,) and np.mean(np.abs(w) ** 2) == pytest.approx(1.0, rel=1e-4)
    m2 = B.load(tmp_path, for_profile=RTL)
    assert np.allclose(m2.waveform(3), w)
    with pytest.raises(profiles.ProfileMismatch):
        B.load(tmp_path, for_profile="hackrf_8000000_ci8")
    ev = B.evaluate(m2, snrs_db=(0.0, 8.0), blocks=500)
    for snr, r in ev["rows"].items():
        for key in ("ae_bler", "bpsk_bler", "ae_pd", "ae_pd_matched", "bpsk_pd"):
            assert 0.0 <= r[key] <= 1.0
    assert ev["rows"][8.0]["bpsk_bler"] <= ev["rows"][0.0]["bpsk_bler"]
    assert ev["label"] == B.NOT_A_TRANSMITTER
    md = B.report_md(ev)
    assert "nothing here transmits" in md and "| 8 |" in md


# -- anomaly: context, never suppression ------------------------------------------------
def _web(rng, n):
    return [{"bytes_out": float(rng.lognormal(7, 0.6)),
             "bytes_in": float(rng.lognormal(10, 0.8)),
             "packets_out": float(rng.integers(5, 40)),
             "packets_in": float(rng.integers(10, 80)),
             "duration_s": float(rng.lognormal(0.5, 0.7)),
             "mean_iat_s": float(rng.lognormal(-3, 0.5)),
             "std_iat_s": float(rng.lognormal(-3.5, 0.5)),
             "dst_port": int(rng.choice([80, 443, 443, 443, 53])),
             "connections": 1, "distinct_ports": 1} for _ in range(n)]


def _beacon(rng, n):
    return [{"bytes_out": 300.0, "bytes_in": 120.0, "packets_out": 2.0,
             "packets_in": 1.0, "duration_s": float(3600 + rng.normal(0, 5)),
             "mean_iat_s": 60.0, "std_iat_s": 0.4,
             "dst_port": int(rng.integers(50000, 60000)), "connections": 60,
             "distinct_ports": 1} for _ in range(n)]


def _model():
    rng = np.random.default_rng(0)
    X, names = A.featurize(_web(rng, 300))
    return A.PcaModel(3).fit(X), names


def test_featurize_and_the_classical_model():
    m, names = _model()
    assert "port_dynamic" in names and len(names) == len(A.NUMERIC) + 4
    rng = np.random.default_rng(9)
    normal = A.featurize(_web(rng, 20))[0]
    odd = A.featurize(_beacon(rng, 20))[0]
    assert np.median(m.errors(odd)) > np.percentile(m.errors(normal), 95)
    assert m.percentile(float(m.errors(odd)[0])) > 99.0


def test_annotate_only_adds_context_and_never_drops_hides_or_rerates():
    m, _ = _model()
    rng = np.random.default_rng(3)
    flows = _web(rng, 3) + _beacon(rng, 3) + [None]
    findings = [{"id": f"f{i}", "severity": sev, "kind": "beacon",
                 "summary": f"finding {i}", "context": ["prior context"]}
                for i, sev in enumerate(["low", "high", "critical", "low",
                                         "medium", "high", "low"])]
    before = copy.deepcopy(findings)
    out = A.annotate(findings, flows, m)
    assert findings == before                              # inputs untouched
    assert len(out) == len(findings)                       # nothing dropped
    assert [f["id"] for f in out] == [f["id"] for f in findings]   # nor re-ordered
    for f, g in zip(findings, out):
        for k, v in f.items():
            if k != "context":
                assert g[k] == v                           # severity unchanged
        assert g["context"][:-1] == f["context"]           # prior context kept
        c = g["context"][-1]
        assert c["kind"] == "reconstruction_error" and c["tier"] == "proposed"
        words = c["words"].lower()
        assert "the finding stands as raised" in words
        for banned in ("benign", "safe", "suppress", "false positive", "ignore",
                       "dismiss"):
            assert banned not in words
    hi = [g["context"][-1]["percentile"] for g in out[3:6]]
    lo = [g["context"][-1]["percentile"] for g in out[:3]]
    assert min(hi) > max(lo)
    assert "no flow features" in out[-1]["context"][-1]["words"]


def test_the_api_has_no_way_to_drop_or_hide_a_finding():
    public = {n: getattr(A, n) for n in dir(A) if not n.startswith("_")}
    for name, obj in public.items():
        low = name.lower()
        for bad in ("drop", "suppress", "filter", "hide", "remove", "dismiss",
                    "allowlist", "whitelist", "baseline_out", "prune"):
            assert bad not in low, name
    params = list(inspect.signature(A.annotate).parameters)
    assert params == ["findings", "flows", "model", "model_name"]
    for cls in (A.PcaModel, A.AutoencoderModel):
        for meth in dir(cls):
            for bad in ("drop", "suppress", "filter", "hide"):
                assert bad not in meth.lower()


def test_annotate_works_on_finding_objects_without_mutating_them():
    m, _ = _model()

    class Finding:
        def __init__(self):
            self.severity = "high"
            self.context = []
    f = Finding()
    out = A.annotate([f], _web(np.random.default_rng(1), 1), m)
    assert f.context == [] and out[0] is not f
    assert out[0].severity == "high" and len(out[0].context) == 1
    with pytest.raises(ValueError, match="one flow"):
        A.annotate([f, f], [None], m)


def test_models_save_and_load_through_cards(tmp_path):
    m, _ = _model()
    A.save(m, tmp_path / "pca")
    m2 = A.load(tmp_path / "pca")
    X = A.featurize(_beacon(np.random.default_rng(2), 4))[0]
    assert np.allclose(m.errors(X), m2.errors(X))
    card = cards.load(tmp_path / "pca", expect_kind="anomaly")
    assert card.tier == "proposed"
    assert any("never suppression" in n for n in card.notes)


def test_the_learned_comparator_beside_the_classical_one(tmp_path):
    torch = pytest.importorskip("torch", reason="PyTorch is only in the training "
                                                "environment")
    torch.set_num_threads(1)
    rng = np.random.default_rng(5)
    res = A.compare(_web(rng, 200), _web(rng, 40), _beacon(rng, 20))
    assert 0.9 <= res["pca_auc"] <= 1.0
    assert res["autoencoder_auc"] is not None and "keep_learned" in res
    ae = A.AutoencoderModel(epochs=50).fit(A.featurize(_web(rng, 100))[0])
    A.save(ae, tmp_path / "ae")
    ae2 = A.load(tmp_path / "ae")
    X = A.featurize(_beacon(rng, 3))[0]
    assert np.allclose(ae.errors(X), ae2.errors(X), atol=1e-5)
