# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The evaluation harness (DETECTION_DESIGN §11, plan §7): reports, the
energy baseline, confirmation agreement, the domain gap, minutes to
acceptable, and the whole detector evaluation — on tiny synthetic stand-ins
in the exact dataset layout (ARCHITECTURE §5)."""

from __future__ import annotations

import importlib
import json

import numpy as np
import pytest

from atk_diffusion import cards
from atk_diffusion.detect.boxes import Detection
from atk_diffusion.experiments import detector_eval as DE
from atk_diffusion.experiments.report import experiment_dir, write_report
from atk_diffusion.learn import common as K
from atk_diffusion.paths import RfData

import helpers_learn as H


# ---------------------------------------------------------------------------
# Pure
# ---------------------------------------------------------------------------
def test_write_report_is_plain_json_and_markdown_with_its_tier(rf):
    out = experiment_dir(rf, H.PROFILE, "toy")
    assert out.parent == rf.runs(H.PROFILE) and out.name.endswith("_toy")
    assert experiment_dir(rf, H.PROFILE, "toy") != out           # never reused
    res = {"gap": np.float32(0.25), "nan": float("nan"), "n": np.int64(3),
           "arr": np.arange(3), "nested": {"ok": np.bool_(True), "inf": np.inf},
           "path": out}
    md, js = write_report(out, "toy", res, ["one line", "two"], rf=rf)
    d = json.loads(js.read_text(encoding="utf-8"))          # strict JSON parses
    assert d["tier"] == "measured" and d["provenance"]["tool"] == "experiments.toy"
    r = d["result"]
    assert r["gap"] == 0.25 and r["nan"] is None and r["n"] == 3
    assert r["arr"] == [0, 1, 2] and r["nested"] == {"ok": True, "inf": None}
    text = md.read_text(encoding="utf-8")
    assert "- one line" in text and "| gap | 0.25 |" in text
    assert "MEASURED" in text and "toy.json" in text
    assert rf.verify(md)[0] and rf.verify(js)[0]


def test_the_energy_baseline_finds_power_and_not_noise():
    rng = np.random.default_rng(0)
    lin = rng.exponential(1.0, (64, 128))
    noise = 10 * np.log10(lin)
    assert len(DE.energy_baseline(noise)["boxes"]) == 0
    lin[10:30, 40:60] += 30.0
    r = DE.energy_baseline(10 * np.log10(lin))
    assert len(r["boxes"]) >= 1
    iou = K.box_iou(r["boxes"][:1], np.array([[10, 40, 30, 60]]))[0, 0]
    assert iou > 0.7 and (r["labels"] == 0).all()


def test_false_alarm_arithmetic():
    preds = [{"scores": np.array([0.9, 0.2])}, {"scores": np.array([])}]
    fa = DE.false_alarms(preds, step_s=1800.0, score_thr=0.5)
    assert fa["count"] == 1 and fa["hours"] == 1.0 and fa["per_hour"] == 1.0


def test_confirmation_agreement_from_detections_records_and_files(tmp_path):
    agree = Detection(0, 1, 1e6, 1.1e6, cls="dmr").confirm("dsd", "TG 1", "dmr")
    disagree = Detection(0, 1, 1e6, 1.1e6, cls="dmr").confirm("dsd", "TG 2", "p25")
    proposed = Detection(0, 1, 1e6, 1.1e6, cls="dmr")         # not confirmed
    recs = [{"classifier_class": "pocsag", "decoder_class": "pocsag"},
            {"classifier_class": "pocsag", "decoder_class": "flex"}]
    f = tmp_path / "confirm.jsonl"
    f.write_text("\n".join(json.dumps(x) for x in
                           [agree.to_json(), disagree.to_json(), proposed.to_json()]
                           + recs) + "\n", encoding="utf-8")
    r = DE.confirmation_agreement([f])
    assert r["confirmed"] == 4 and r["agree"] == 2 and r["disagree"] == 2
    assert r["agreement"] == 0.5 and r["skipped"] == 1
    assert {"classifier": "dmr", "decoder": "p25"} in r["disagreements"]
    assert DE.confirmation_agreement([agree, disagree])["agreement"] == 0.5


# ---------------------------------------------------------------------------
# One small world: datasets and two trained models
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    torch.set_num_threads(1)
    from atk_diffusion.learn import classifier1d as C1
    from atk_diffusion.learn import proposer2d as P2
    rf = RfData(tmp_path_factory.mktemp("exp") / "rf_data", create=True)
    w = {"rf": rf}
    w["syn"] = H.make_wideband(rf, name="wb_syn", seed=1,
                               splits={"train": 24, "val": 8, "test": 8})
    w["cab"] = H.make_wideband(rf, name="wb_cab", seed=9, splits={"test": 8},
                               snr_range=(3.0, 9.0), generator="cabled",
                               label_sources=["cabled"])
    w["noise"] = H.make_wideband(rf, name="wb_noise", seed=11,
                                 splits={"test": 6}, noise_only=True)
    w["site"] = H.make_wideband(rf, name="wb_site", seed=5, with_t_s=True,
                                splits={"train": 16, "test": 6},
                                snr_range=(5.0, 12.0))
    four = ("ref_bpsk", "ref_qpsk", "ref_2fsk", "ref_am")
    w["nb"] = H.make_narrowband(rf, name="nb_syn", classes=four, seed=4,
                                per_class={"train": 32, "val": 12, "test": 12})
    w["nb_cab"] = H.make_narrowband(rf, name="nb_cab", classes=four[:3], seed=8,
                                    per_class={"test": 10}, snr_range=(-2.0, 8.0),
                                    generator="cabled", label_sources=["cabled"])
    w["nb_site"] = H.make_narrowband(rf, name="nb_site", classes=four[:3], seed=6,
                                     per_class={"train": 12, "test": 8},
                                     with_t_s=True)
    w["P"] = P2.train(rf, H.PROFILE, w["syn"], "det", epochs=6, batch_size=4,
                      width=8, fpn_channels=32, head_convs=1, threads=1,
                      latency_repeats=2, latency_threads=1)
    w["C"] = C1.train(rf, H.PROFILE, w["nb"], "amc", epochs=8,
                      rf_config={"name": "k5", "kernel": 5,
                                 "dilations": [1, 1, 1, 1]},
                      held_out_classes=("ref_am",), iq_width=8, scf_width=8,
                      batch_size=16, threads=1, latency_repeats=2,
                      latency_threads=1)
    return w


def test_domain_gap_for_the_proposer_is_folded_into_its_card(world):
    from atk_diffusion.experiments.domain_gap import domain_gap
    rf = world["rf"]
    r = domain_gap(rf, H.PROFILE, world["P"], world["syn"], world["cab"])
    s, c = r["synthetic"], r["cabled"]
    assert r["metric"] == "mAP@0.5" and s["tiles"] == 8 and c["tiles"] == 8
    assert r["gap"] == pytest.approx(s["map50"] - c["map50"])
    assert "note" not in r                               # the second set is cabled
    card = cards.load(world["P"])
    assert card.metrics["domain_gap"] == pytest.approx(r["gap"])
    assert card.metrics["map_cabled"] == pytest.approx(c["map50"])
    assert any(line.startswith("domain gap") for line in cards.summary(card))
    d = json.loads(open(r["report"]["json"], encoding="utf-8").read())
    assert d["result"]["gap"] == pytest.approx(r["gap"])


def test_domain_gap_for_the_classifier_and_the_honest_note(world):
    from atk_diffusion.experiments.domain_gap import domain_gap
    rf = world["rf"]
    r = domain_gap(rf, H.PROFILE, world["C"], world["nb"], world["nb_cab"])
    assert r["metric"] == "accuracy"
    assert r["cabled"]["known_cuts"] == 30 and r["synthetic"]["unknown_class_cuts"] == 12
    assert r["gap"] == pytest.approx(r["synthetic"]["accuracy"] - r["cabled"]["accuracy"])
    assert cards.load(world["C"]).metrics["accuracy_cabled"] == r["cabled"]["accuracy"]
    r2 = domain_gap(rf, H.PROFILE, world["C"], world["nb"], world["nb"],
                    update_card=False)
    assert "not the domain gap" in r2["note"]


def test_minutes_to_acceptable_for_the_proposer(world):
    from atk_diffusion.experiments.minutes_to_acceptable import minutes_to_acceptable
    rf = world["rf"]
    mins = [0, 0.05, 0.1, 0.25]
    r0 = minutes_to_acceptable(rf, H.PROFILE, world["P"], world["site"],
                               acceptance=1.01, minutes=mins, update_card=False)
    assert r0["minutes_to_acceptable"] is None and r0["metric"] == "f1"
    assert [c["minutes"] for c in r0["curve"]] == mins
    assert [c["items"] for c in r0["curve"]][0] == 0
    assert r0["time_source"].startswith("item times from the dataset's t_s")
    md = open(r0["report"]["markdown"], encoding="utf-8").read()
    assert "NOT REACHED" in md
    v0 = r0["curve"][0]["value"]
    print("on-site F1 curve:", [(c["minutes"], c["items"], c["value"])
                                for c in r0["curve"]])
    better = [c for c in r0["curve"][1:] if c["value"] is not None
              and v0 is not None and c["value"] > v0]
    if better:
        acc = better[0]["value"]
        r = minutes_to_acceptable(rf, H.PROFILE, world["P"], world["site"],
                                  acceptance=acc, minutes=mins, save_adapted=True)
        assert r["minutes_to_acceptable"] == better[0]["minutes"]
        new = cards.load(r["adapted_model"], expect_kind="proposer2d",
                         for_profile=H.PROFILE)
        assert any("adapted on site" in n for n in new.notes)
        assert new.calibration["min_score"] != cards.load(world["P"]).calibration["min_score"]
    else:
        r = minutes_to_acceptable(rf, H.PROFILE, world["P"], world["site"],
                                  acceptance=v0, minutes=mins)
        assert r["minutes_to_acceptable"] == 0
    assert cards.load(world["P"]).metrics["minutes_to_acceptable"]["mode"] == "thresholds"


def test_minutes_to_acceptable_for_the_classifier_three_ways(world):
    pytest.importorskip("torch")
    from atk_diffusion.experiments.minutes_to_acceptable import minutes_to_acceptable
    rf = world["rf"]
    mins = [0, 0.002, 0.005]
    for mode, metric in (("prototypes", "open_set_accuracy"),
                         ("temperature", "ece"), ("finetune", "accuracy")):
        r = minutes_to_acceptable(rf, H.PROFILE, world["C"], world["nb_site"],
                                  acceptance=0.0 if metric == "ece" else 2.0,
                                  mode=mode, minutes=mins, update_card=False,
                                  finetune_epochs=1, device="cpu")
        assert r["metric"] == metric and r["higher_is_better"] == (metric != "ece")
        vals = [c["value"] for c in r["curve"]]
        assert vals[0] is not None and all(v is None or v >= 0 for v in vals)
        assert r["minutes_to_acceptable"] is None            # lines never crossed
    with pytest.raises(ValueError, match="adaptation is one of"):
        minutes_to_acceptable(rf, H.PROFILE, world["C"], world["nb_site"],
                              acceptance=0.5, mode="thresholds")


def test_the_whole_detector_evaluation(world, tmp_path):
    rf = world["rf"]
    d1 = Detection(0, 1, 1e6, 1.1e6, cls="ref_bpsk").confirm("multimon", "", "ref_bpsk")
    d2 = Detection(0, 1, 1e6, 1.1e6, cls="ref_bpsk").confirm("multimon", "", "ref_qpsk")
    log = tmp_path / "c.jsonl"
    log.write_text("\n".join(json.dumps(d.to_json()) for d in (d1, d2)), "utf-8")
    lines = []
    r = DE.evaluate_detector(rf, H.PROFILE, proposer_dir=world["P"],
                             classifier_dir=world["C"],
                             synthetic_dataset=world["syn"],
                             cabled_dataset=world["cab"],
                             noise_dataset=world["noise"],
                             classifier_dataset=world["nb"],
                             confirm_logs=[log], threads=1, latency_repeats=2,
                             progress=lines.append)
    P = r["proposer"]
    assert {"synthetic", "cabled", "energy_baseline", "false_alarms",
            "detection_vs_snr", "latency"} <= set(P)
    fa = P["false_alarms"]["learned"]
    assert fa["tiles"] == 6 and fa["hours"] == pytest.approx(6 * 0.75 / 3600)
    assert P["false_alarms"]["energy"]["tiles"] == 6
    curve = P["detection_vs_snr"]["synthetic"]
    assert "_all" in curve and set(curve) - {"_all"} <= {"nfm_voice", "adsb",
                                                        "ref_ofdm"}
    assert all(0.0 <= b["value"] <= 1.0 for b in curve["_all"])
    assert P["domain_gap"] == pytest.approx(P["synthetic"]["map50"]
                                            - P["cabled"]["map50"])
    C = r["classifier"]
    assert C["unknown"]["n_unknown"] == 12 and C["unknown"]["held_out_classes"] == ["ref_am"]
    assert 0.0 <= C["unknown"]["unknown_rejection"] <= 1.0
    assert r["confirmation"]["agreement"] == 0.5
    pc = cards.load(world["P"])
    assert pc.metrics["false_alarms_per_hour"] == pytest.approx(fa["per_hour"])
    assert "detection_vs_snr" in pc.metrics and "energy_baseline" in pc.metrics
    cc = cards.load(world["C"])
    assert cc.metrics["unknown_rejection"] == C["unknown"]["unknown_rejection"]
    assert any("false alarms per hour" in x for x in lines)
    d = json.loads(open(r["report"]["json"], encoding="utf-8").read())
    assert d["tier"] == "measured" and d["result"]["confirmation"]["agree"] == 1


def test_the_harness_runs_without_pytorch(world):
    """ATK's core environment has numpy, scipy and onnxruntime but no
    PyTorch (ARCHITECTURE §1): the evaluation harness and calibration must
    run there. A subprocess with torch import blocked proves it."""
    import subprocess
    import sys
    code = f"""
import sys
class _NoTorch:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('torch', 'torchvision', 'torchaudio'):
            raise ImportError('PyTorch is not in this (core) environment')
        return None
sys.meta_path.insert(0, _NoTorch())
sys.path.insert(0, {str(K.Path(__file__).resolve().parents[1])!r})
from atk_diffusion.experiments import detector_eval as DE, domain_gap
from atk_diffusion.learn import calibrate, export as X
from atk_diffusion.paths import RfData
r = X.OnnxRunner({str(world['P'])!r}, 'proposer2d', for_profile={H.PROFILE!r}, threads=1)
s, _ = DE.evaluate_proposer(r, {str(world['syn'])!r}, {H.PROFILE!r}, 'test')
print('map50', s['map50'])
try:
    from atk_diffusion.learn import data
except RuntimeError as e:
    print('refused:', e)
assert 'torch' not in sys.modules
"""
    p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-2000:]
    assert "map50" in p.stdout
    assert "refused: PyTorch is not in this environment" in p.stdout


def test_nothing_to_evaluate_is_said_in_words(rf):
    with pytest.raises(ValueError, match="nothing to evaluate"):
        DE.evaluate_detector(rf, H.PROFILE)


def test_integration_with_the_dataset_builder(tmp_path):
    """The real builders (synth.datasets) -> the training readers
    (learn.common). Both formats of ARCHITECTURE §5, end to end: a builder
    that writes something the readers cannot open fails here, not on Bill's
    machine after an overnight generation."""
    sd = importlib.import_module("atk_diffusion.synth.datasets")
    rf = RfData(tmp_path / "rf_data", create=True)
    nb = sd.build_narrowband(rf, H.PROFILE, "builder_nb",
                             ["ref_bpsk", "ref_qpsk"], 4, (10.0, 20.0),
                             "voice", window=1024, compute_scf=True)
    wb = sd.build_wideband(rf, H.PROFILE, "builder_wb", 3, scene_seconds=0.25)
    for man in [m for m in (nb, wb) if m is not None]:
        d = man["path"]
        m = K.open_dataset(d, H.PROFILE)
        kind = m["kind"]
        for sp in K.SPLITS:
            if K.split_files(d, sp, kind):
                s = (K.ShardSet if kind == "narrowband" else K.TileSet)(d, sp, m)
                assert len(s) > 0
                break
        else:
            pytest.fail(f"the {kind} builder wrote a manifest but no split files")
