# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The detector end to end (DETECTION_DESIGN §1–§5, §7; ARCHITECTURE §4.1)."""

from __future__ import annotations

import math
import sys
import warnings

import numpy as np
import pytest

from atk_diffusion import cards, profiles, sigmf
from atk_diffusion.detect import classes as C
from atk_diffusion.detect.pipeline import DetectorPipeline
from atk_diffusion.profiles import StftGeometry

RTL = "rtlsdr_2400000_cu8"
FS = 2_400_000.0
NOISE_P = 2e-4


def _noise(rng, n, power=NOISE_P):
    return ((rng.standard_normal(n) + 1j * rng.standard_normal(n))
            * math.sqrt(power / 2)).astype(np.complex64)


def _add(x, rng, fs, t0, t1, f, bw, snr_db, noise_p=NOISE_P):
    n0, n1 = int(round(t0 * fs)), int(round(t1 * fs))
    n = n1 - n0
    spec = np.fft.fft(rng.standard_normal(n) + 1j * rng.standard_normal(n))
    spec[np.abs(np.fft.fftfreq(n, 1 / fs)) > bw / 2] = 0
    s = np.fft.ifft(spec)
    s *= math.sqrt(10 ** (snr_db / 10) * noise_p * bw / fs / np.mean(np.abs(s) ** 2))
    x[n0:n1] += (s * np.exp(2j * np.pi * f * np.arange(n0, n1) / fs)).astype(np.complex64)


def _tone(x, fs, t0, t1, f, snr_db, fft=1024, noise_p=NOISE_P):
    """A carrier `snr_db` above the per-bin floor (Hann: σ²·1.5/N per bin)."""
    n0, n1 = int(round(t0 * fs)), int(round(t1 * fs))
    a = math.sqrt(noise_p * 1.5 / fft * 10 ** (snr_db / 10))
    x[n0:n1] += (a * np.exp(2j * np.pi * f * np.arange(n0, n1) / fs)).astype(np.complex64)


def _feed(p, x, center, rng=None, t_start=0.0, epoch=None):
    out, pos = [], 0
    while pos < x.size:
        n = int(rng.integers(20_000, 400_000)) if rng is not None else 240_000
        out += p.feed(x[pos:pos + n], center, t_start=t_start + pos / p.fs, epoch=epoch)
        pos += n
    return out


# bursts on and around tile boundaries (tiles step 0.8192 s, share 0.273 s)
BURSTS = [(0.30, 0.35, 200e3, 25e3, 12), (0.80, 0.86, -300e3, 25e3, 12),
          (0.95, 0.99, 500e3, 50e3, 15), (1.60, 1.70, -100e3, 12.5e3, 10),
          (1.62, 1.66, 650e3, 25e3, 12), (2.00, 2.02, 0.7e6, 100e3, 15),
          (2.44, 2.47, -0.5e6, 25e3, 12), (3.30, 3.31, -0.6e6, 25e3, 15)]


@pytest.fixture(scope="module")
def scene():
    """3.6 s at 2.4 MS/s: the bursts above and a carrier for 3.4 s (built
    once for the module; the tests only read it)."""
    rng = np.random.default_rng(2026)
    x = _noise(rng, int(3.6 * FS))
    for b in BURSTS:
        _add(x, rng, FS, *b)
    _tone(x, FS, 0.1, 3.5, 50e3, 15)
    x.setflags(write=False)
    return x


def test_each_burst_is_reported_once_and_a_carrier_is_one_track(scene, rng):
    x = scene
    p = DetectorPipeline(RTL)
    dets = _feed(p, x, 100e6, rng=rng, epoch=1.7e9) + p.finish()
    carrier = [d for d in dets if abs(d.center_hz - 100.05e6) < 10e3]
    bursts = [d for d in dets if d not in carrier]
    assert len(bursts) == len(BURSTS), [(round(d.t0, 3), d.center_hz) for d in bursts]
    for t0, t1, f, bw, _snr in BURSTS:
        hit = [d for d in bursts if abs(d.center_hz - (100e6 + f)) < bw / 2
               and d.t0 < t1 and d.t1 > t0]
        assert len(hit) == 1, (t0, f)
        d = hit[0]
        assert abs(d.t0 - t0) < 0.005 and abs(d.t1 - t1) < 0.005
        assert d.track_id and d.profile == RTL and d.wall_t0 == pytest.approx(1.7e9 + d.t0)
    # the carrier: abutting boxes across tiles, one track, nothing doubled
    carrier.sort(key=lambda d: d.t0)
    assert len({d.track_id for d in carrier}) == 1
    for a, b in zip(carrier, carrier[1:]):
        assert b.t0 == pytest.approx(a.t1, abs=1e-6)
    assert carrier[0].t0 == pytest.approx(0.1, abs=0.005)
    assert carrier[-1].t1 == pytest.approx(3.5, abs=0.005)
    st = p.status()
    assert st["counts_by_source"]["energy"] == len(dets) and st["refusal"] == ""
    assert st["latency_ms"]["tiles"] >= 4 and st["latency_ms"]["mean"] > 0
    assert any("Energy proposer: ON" in line for line in st["lines"])


def test_the_block_size_does_not_change_the_answer(scene):
    x = scene
    a = DetectorPipeline(RTL)
    da = _feed(a, x, 100e6, rng=np.random.default_rng(1)) + a.finish()
    b = DetectorPipeline(RTL)
    db = b.feed(x, 100e6, 0.0) + b.finish()
    key = lambda d: (round(d.t0, 6), round(d.t1, 6), round(d.f_lo), round(d.f_hi))
    assert sorted(map(key, da)) == sorted(map(key, db))


def test_a_retune_ends_the_stream_closes_tracks_and_no_box_spans_it(rng):
    x1 = _noise(rng, int(1.5 * FS))
    _tone(x1, FS, 0.0, 1.5, 100e3, 15)                  # on until the retune
    x2 = _noise(rng, int(1.5 * FS))
    _tone(x2, FS, 0.0, 1.5, 100e3, 15)
    p = DetectorPipeline(RTL)
    before = _feed(p, x1, 162.0e6)
    t1 = p.tracker.active()[0].id if p.tracker.active() else None
    after = p.feed(x2, 163.0e6, t_start=1.5) + p.finish()
    early = [d for d in before + after if d.center_hz < 162.5e6]
    late = [d for d in after if d.center_hz > 162.5e6]
    assert early and late
    assert all(d.t1 <= 1.5 + 1e-9 for d in early)
    assert all(d.t0 >= 1.5 - 1e-9 for d in late)
    assert max(d.t1 for d in early) == pytest.approx(1.5, abs=0.005)   # flushed, not lost
    assert {d.track_id for d in early}.isdisjoint({d.track_id for d in late})
    assert t1 is not None and p.tracker.get(t1).closed
    assert p.status()["retunes"] == 1


def test_a_gap_in_the_stream_restarts_tiles_but_keeps_the_floor(rng):
    p = DetectorPipeline(RTL)
    p.feed(_noise(rng, int(1.2 * FS)), 100e6, 0.0)
    fl = p.floor.level_db()
    p.feed(_noise(rng, int(1.2 * FS)), 100e6, 5.0)      # 3.8 s were lost
    assert p.status()["discontinuities"] == 1
    assert p.floor.level_db() == pytest.approx(fl, abs=0.2)
    assert any("jumped" in n for n in p.status()["notes"])


def test_a_refused_model_costs_one_feature_never_the_pipeline(tmp_path, rng):
    d = tmp_path / "prop"
    d.mkdir()
    (d / "model.onnx").write_bytes(b"not used: refused before it is read")
    cards.save(d, cards.new_card("hackrf-proposer", "proposer2d",
                                 "hackrf_8000000_ci8"), "model.onnx")
    p = DetectorPipeline(RTL, proposers={"learned": True}, proposer2d=d)
    st = p.status()
    assert "learned proposer refused" in st["refusal"]
    assert "trained for the HackRF One at 8 MS/s" in st["refusal"]
    assert not st["proposers"]["learned"] and st["proposers"]["energy"]
    assert any(line.startswith("REFUSED") for line in st["lines"])
    x = _noise(rng, int(1.2 * FS))
    _add(x, rng, FS, 0.2, 0.3, 300e3, 25e3, 12)
    dets = p.feed(x, 100e6, 0.0) + p.finish()
    assert len(dets) == 1 and dets[0].sources == ("energy",)


def test_energy_cannot_be_switched_off_and_modes_are_checked():
    p = DetectorPipeline(RTL, proposers={"energy": False})
    assert p.proposers["energy"] and any("cannot be switched off" in n for n in p.notes)
    with pytest.raises(ValueError, match="cyclic_mode"):
        DetectorPipeline(RTL, cyclic_mode="everything")


def test_an_unavailable_cyclic_proposer_is_said_in_words(rng, monkeypatch):
    monkeypatch.setitem(sys.modules, "atk_diffusion.cyclo.proposer", None)
    p = DetectorPipeline(RTL, proposers={"cyclic": True})
    x = _noise(rng, int(1.2 * FS))
    _add(x, rng, FS, 0.2, 0.3, 300e3, 25e3, 12)
    dets = p.feed(x, 100e6, 0.0) + p.finish()
    st = p.status()
    assert "cyclic proposer unavailable" in st["cyclic"] and not st["proposers"]["cyclic"]
    assert len(dets) == 1                                # energy carried on


def test_integration_with_the_real_cyclic_proposer(rng):
    """Runs once atk_diffusion.cyclo.proposer exists (written beside this)."""
    try:
        from atk_diffusion.cyclo.proposer import cyclic_proposer  # noqa: F401
    except ImportError as e:
        pytest.skip(f"atk_diffusion.cyclo.proposer is not importable yet: {e}")
    p = DetectorPipeline(RTL, proposers={"cyclic": True})
    x = _noise(rng, int(1.2 * FS))
    _add(x, rng, FS, 0.05, 1.15, 300e3, 12.5e3, 10)    # a 12.5 kHz noise-like signal
    dets = p.feed(x, 162.4e6, 0.0) + p.finish()
    st = p.status()
    assert st["cyclic"] == "" and st["proposers"]["cyclic"], st["notes"]
    assert st["cyclic_report"].get("mode") == "regions"
    assert any("characterising the energy boxes" in line for line in st["lines"])
    for d in dets:
        assert -0.01 <= d.t0 <= d.t1 <= 1.2 + 0.01 and d.profile == RTL
        if "cyclic" in d.sources:
            assert 161.2e6 <= d.f_lo <= d.f_hi <= 163.6e6


def test_denoised_boxes_are_beside_never_instead(rng):
    x = _noise(rng, int(1.2 * FS))
    _add(x, rng, FS, 0.2, 0.3, 300e3, 25e3, 12)
    raw = DetectorPipeline(RTL)
    plain = raw.feed(x, 100e6, 0.0) + raw.finish()

    def denoiser(spec):
        out = spec.copy()
        out[300:340, 100:110] = 25.0                     # it "finds" something
        return out

    p = DetectorPipeline(RTL, denoiser=denoiser)
    dets = p.feed(x, 100e6, 0.0) + p.finish()
    dn = [d for d in dets if "denoised" in d.flags]
    kept = [d for d in dets if "denoised" not in d.flags]
    assert [(d.t0, d.f_lo) for d in kept] == [(d.t0, d.f_lo) for d in plain]
    assert len(dn) == 1 and dn[0].snr_db < 3.0          # measured on the RAW tile
    assert p.status()["counts_by_source"]["denoised"] == 1

    def broken(spec):
        raise MemoryError("the GPU is the core's")

    q = DetectorPipeline(RTL, denoiser=broken)
    assert len(q.feed(x, 100e6, 0.0) + q.finish()) == len(plain)
    assert any("denoised path failed" in n for n in q.status()["notes"])


def test_run_on_capture_writes_proposed_annotations(tmp_path, rf, rng):
    x = _noise(rng, int(2.0 * FS))
    _add(x, rng, FS, 0.4, 0.5, 300e3, 25e3, 15)
    _add(x, rng, FS, 1.2, 1.25, -500e3, 50e3, 15)
    base = rf.captures(RTL) / "cap"
    sigmf.write_pair(base, x, FS, 162.4e6, datatype="cu8", hw="RTL-SDR",
                     t0_utc=1.7e9,
                     extra_global={"atk:receiver_profile": RTL},
                     annotations=[sigmf.Annotation(0, 100, label="taught one",
                                                   extra={"atk:source": "taught"})])
    p = DetectorPipeline(RTL, rf=rf)
    res = p.run_on_capture(base, write_annotations=True)
    assert res["annotations_written"] == 2 and res["seconds"] == pytest.approx(2.0)
    anns = sigmf.annotations(base)
    proposed = [a for a in anns if a.source == "proposed"]
    assert len(proposed) == 2 and any(a.source == "taught" for a in anns)
    a = min(proposed, key=lambda a: a.sample_start)
    assert a.sample_start == pytest.approx(0.4 * FS, abs=0.004 * FS)
    assert a.freq_lower_edge == pytest.approx(162.7e6 - 12.5e3, abs=3e3)
    assert a.label == "unknown" and a.extra["atk:proposer"] == "energy"
    assert a.extra["atk:tier"] == "proposed" and "atk:snr_db" in a.extra
    assert sigmf.validate(sigmf.read_meta(base)) == []
    assert rf.verify(sigmf.meta_path(base))[0]          # the write log knows
    assert res["detections"][0].epoch == pytest.approx(1.7e9)
    # a re-run replaces its own boxes and nothing else, and starts clean
    again = p.run_on_capture(base, write_annotations=True)
    assert len([a for a in sigmf.annotations(base) if a.source == "proposed"]) == 2
    assert len(again["tracks"]) == len(res["tracks"])
    # another receiver's capture is refused in the plan's words
    other = tmp_path / "blade"
    sigmf.write_pair(other, x[:400_000], 4e6, 100e6, datatype="ci16", hw="bladeRF x115")
    with pytest.raises(profiles.ProfileMismatch, match="set up for the RTL-SDR"):
        p.run_on_capture(other)


# ---------------------------------------------------------------------------
# cutout -> classifier -> prototypes -> class or UNKNOWN
# ---------------------------------------------------------------------------
PID_SMALL = "rtlsdr_256000_cu8"


def _small_profile():
    prof = profiles.new_profile(PID_SMALL)
    prof.stft = StftGeometry(fft_size=256, hop=256, tile_seconds=0.256, tile_rows=128,
                             tile_overlap=0.25)
    return prof


def _r1_classifier(tmp_path):
    """A tiny classifier whose embedding is the lag-1 autocorrelation of the
    cut: a narrow signal sits near (1, 0), a wide one well away from it."""
    torch = pytest.importorskip("torch", reason="PyTorch is only in the training "
                                "environment")
    pytest.importorskip("onnxruntime", reason="ONNX Runtime is not installed")
    torch.set_num_threads(1)

    class R1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(2, 2)
            with torch.no_grad():
                self.head.weight.copy_(torch.tensor([[10.0, -10.0], [-10.0, 10.0]]))
                self.head.bias.zero_()

        def forward(self, iq, scf):
            i, q = iq[:, 0, :], iq[:, 1, :]
            r1 = (i[:, :-1] * i[:, 1:] + q[:, :-1] * q[:, 1:]).mean(dim=1) / \
                ((i * i + q * q).mean(dim=1) + 1e-9)
            f = torch.stack([r1, 1.0 - r1], dim=1) + 0.0 * scf.mean()
            e = f / torch.sqrt((f * f).sum(dim=1, keepdim=True) + 1e-12)
            return self.head(f), e, f

    d = tmp_path / "clf"
    d.mkdir()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(R1().eval(), (torch.zeros(1, 2, 256), torch.zeros(1, 1, 1, 1)),
                          str(d / "model.onnx"), input_names=["iq", "scf"],
                          output_names=["logits", "embedding", "cycle"], dynamo=False,
                          opset_version=17,
                          dynamic_axes={k: {0: "b"} for k in
                                        ("iq", "scf", "logits", "embedding", "cycle")})
    can = profiles.canonical_for(256_000, 10e3)
    cards.save(d, cards.new_card(
        "r1-classifier", "classifier1d", PID_SMALL,
        input={"iq_len": 256, "canonical": {"class": can.cls, "rate": can.rate,
                                            "decimation": can.decimation},
               "scf_used": False, "scf_shape": [1, 1], "cycle_scale": [1.0, 1.0]},
        classes=[{"name": "nfm_voice"}, {"name": "p25"}]), "model.onnx")
    return d


def test_boxes_are_classified_through_the_cut_or_called_unknown(tmp_path, rng):
    from atk_diffusion.detect.onnx_models import Classifier1D
    from atk_diffusion.detect.prototypes import PrototypeBank
    from atk_diffusion.dsp import resample
    d = _r1_classifier(tmp_path)
    fs, prof = 256_000.0, _small_profile()
    clf = Classifier1D(d, profile=PID_SMALL, threads=1)
    assert clf.canonical_rate == 51_200.0 and clf.canonical_class == "voice"
    # teach "nfm_voice" from cuts of narrow signals made the way the pipeline cuts
    wins = []
    for _ in range(40):
        y = _noise(rng, 30_000)
        _add(y, rng, fs, 0.0, 30_000 / fs, 0.0, 2e3, 15)
        cut, _fs, _i = resample.cut_to_canonical(y, fs, 0.0, 4e3)
        wins.append(cut[100:356])
    emb = clf.run(np.array(wins)).embeddings
    bank = PrototypeBank(PID_SMALL, "voice")
    bank.teach("nfm_voice", emb[:20])
    bank.calibrate_thresholds({"nfm_voice": emb[20:]}, quantile=0.999)
    p = DetectorPipeline(prof, classifier=clf, prototypes=bank)
    x = _noise(rng, int(1.0 * fs))
    _add(x, rng, fs, 0.10, 0.30, -60e3, 2e3, 15)        # narrow: the taught class
    _add(x, rng, fs, 0.50, 0.70, 50e3, 20e3, 15)        # wide: nothing like it
    dets = p.feed(x, 162.4e6, 0.0) + p.finish()
    narrow = [d for d in dets if d.center_hz < 162.4e6]
    wide = [d for d in dets if d.center_hz > 162.4e6]
    assert narrow and wide
    for d in narrow:
        assert d.cls == "nfm_voice" and "taught" in d.flags
        assert d.family == C.get("nfm_voice").family
        assert d.confidence is not None and d.measurements["classifier_top"] == "nfm_voice"
    for d in wide:
        assert d.cls == C.UNKNOWN and d.measurements["nearest_class"] == "nfm_voice"
        assert d.measurements["prototype_distance"] > d.measurements["prototype_threshold"]
    st = p.status()
    assert st["classified"] == len(dets) and st["unknown"] == len(wide)
    assert any("open set" in line for line in st["lines"])
    # without a bank the classifier forces a class, and the boxes say so
    q = DetectorPipeline(prof, classifier=clf)
    forced = q.feed(x, 162.4e6, 0.0) + q.finish()
    assert all(d.cls in ("nfm_voice", "p25") and "open_set" in d.measurements
               for d in forced)
    assert any("cannot say UNKNOWN" in line for line in q.status()["lines"])


def test_the_escalation_hook_gets_weak_boxes_and_its_findings_are_flagged(rng):
    x = _noise(rng, int(1.2 * FS))
    _tone(x, FS, 0.0, 1.2, 200e3, 4.0)                 # weak: below escalate_snr_db
    _add(x, rng, FS, 0.4, 0.5, -300e3, 25e3, 15)        # strong: not escalated
    calls = []

    def escalate(dets, iq, fs, center_hz, t0, profile):
        calls.append((len(dets), iq.size, fs, profile.id))
        out = []
        for d in dets:
            e = type(d)(t0=d.t0, t1=d.t1, f_lo=d.f_lo, f_hi=d.f_hi,
                        sources=("cyclic",), alpha_hz=4800.0, integration_s=5.0)
            out.append(e)
        return out

    p = DetectorPipeline(RTL, escalation=escalate)
    dets = p.feed(x, 100e6, 0.0) + p.finish()
    assert calls and all(c[2] == FS and c[3] == RTL and c[1] > 0 for c in calls)
    weak = [d for d in dets if abs(d.center_hz - 100.2e6) < 5e3]
    strong = [d for d in dets if abs(d.center_hz - 99.7e6) < 20e3]
    assert weak and all("escalated" in d.flags and "cyclic" in d.sources
                        and d.alpha_hz == 4800.0 for d in weak)
    assert strong and all("escalated" not in d.flags for d in strong)
    assert p.status()["counts_by_source"]["escalated"] == len(weak)

    def broken(*a):
        raise TimeoutError("the buffer was still filling")

    q = DetectorPipeline(RTL, escalation=broken)
    assert len(q.feed(x, 100e6, 0.0) + q.finish()) == len(dets)
    assert any("escalation failed" in n and "TimeoutError" in n for n in q.status()["notes"])
