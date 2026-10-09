# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The command line (`python -m atk_diffusion …`, atk_diffusion.cli): every
cheap command run in-process with a temporary rf_data, checked for its WORDS
(plain sentences, `[ERR]` on a failure), its EXIT CODE (0 done, 1 refused or
failed, 2 a wrong command line) and the FILES it leaves. The transmit path is
checked with the runner replaced: nothing in this file can start a radio."""

from __future__ import annotations

import json
import os
import pkgutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from atk_diffusion import cli, profiles, sigmf
from atk_diffusion.synth import native

PID = "rtlsdr_1024000_cu8"
FS = 1_024_000.0
F0 = 162.4e6


@pytest.fixture
def run(rf, capsys):
    """run("status", ...) -> (exit code, stdout, stderr), against `rf`."""
    def go(*args, root=None):
        code = cli.main(["--rf-data", str(root or rf.root), *map(str, args)])
        out, err = capsys.readouterr()
        return code, out, err
    return go


def _burst_capture(rf, name="sig", seconds=1.3, offset=150e3, record=True):
    """Noise with one strong NFM burst, 0.3–0.7 s, `offset` Hz above F0, as
    an RTL-SDR at 1.024 MS/s would have recorded it (cu8)."""
    rng = np.random.default_rng(1)
    n = int(seconds * FS)
    pn = 10 ** (-35 / 10)
    x = native.noise(n, pn, rng)
    t0, dur = min(0.3, 0.25 * seconds), min(0.4, 0.5 * seconds)
    s, lab = native.waveform("nfm_voice", FS, int(dur * FS), rng,
                             carrier_offset_hz=offset)
    a = 10 ** (25 / 20) * np.sqrt(pn * 11e3 / FS)
    x[int(t0 * FS):int(t0 * FS) + s.size] += (a * s).astype(np.complex64)
    base = rf.captures(PID) / name
    dp, mp = sigmf.write_pair(base, x, FS, F0, datatype="cu8", hw="RTL-SDR v3",
                              extra_global={"atk:receiver_profile": PID})
    if record:
        rf.record(dp, "capture")
    return base, lab


def _terminated(rf, name="term"):
    from atk_diffusion.dsp import impair
    rng = np.random.default_rng(2)
    t = impair.quantise(native.noise(int(0.2 * FS), 10 ** (-35 / 10), rng), "cu8", 8)
    base = rf.captures(PID) / name
    sigmf.write_pair(base, t, FS, F0, datatype="cu8", hw="RTL-SDR v3",
                     extra_global={"atk:receiver_profile": PID})
    return base


# -- status, rfdata, usage ---------------------------------------------------------
def test_status_says_what_is_installed_and_where(run, tmp_path):
    missing = tmp_path / "not_yet"
    code, out, _ = run("status", root=missing)
    assert code == 0
    assert "ATK Diffusion Toolkit 0.1.0" in out and "[OK] numpy" in out
    assert f"rf_data: {missing} — does not exist yet" in out
    assert "rfdata init" in out and not missing.exists()     # status writes nothing
    from atk_diffusion import capabilities
    if capabilities.has("torch"):                          # the GPU, in words
        assert "\nGPU: " in out
    code, out, _ = run("profile", "new", PID)
    code, out, _ = run("status")
    assert code == 0 and "Receiver profiles (1):" in out
    assert f"{PID} — the RTL-SDR at 1.024 MS/s" in out
    assert "impairments NOT measured; safe input not entered" in out


def test_a_refused_root_is_said_and_exits_1(run, tmp_path):
    bad = tmp_path / "Users" / "bill" / "AppData" / "Local" / "rf_data"
    code, out, _ = run("status", root=bad)
    assert code == 1 and "[ERR] rf_data:" in out and "inside AppData" in out
    code, out, _ = run("rfdata", "init", root=bad)
    assert code == 1 and "the rf_data folder was refused" in out
    assert not bad.exists()


def test_json_mode_keeps_stdout_pure_json(run):
    code, out, err = run("--json", "status")
    d = json.loads(out)
    assert code == 0 and d["ok"] is True and d["command"] == "status"
    assert d["result"]["version"] == "0.1.0" and "numpy" in d["result"]["capabilities"]
    assert "ATK Diffusion Toolkit" in err                 # the words went to stderr
    code, out, err = run("profile", "show", "nonsense", "--json")
    d = json.loads(out)
    assert code == 1 and d["ok"] is False and "not a receiver profile" in d["error"]


def test_usage_errors_exit_2_with_words(run):
    code, _out, err = run("frobnicate")
    assert code == 2 and "[ERR] argument COMMAND: invalid choice" in err
    code, _out, err = run("synth", "narrowband", "--profile", PID)
    assert code == 2 and "the following arguments are required" in err
    assert cli.main([]) == 2


def test_an_unexpected_fault_is_said_not_raised(run, monkeypatch, capsys):
    def broken(ctx):
        raise TypeError("a fault deep inside")
    monkeypatch.setitem(cli._COMMANDS, ("status", None), broken)
    code, out, err = run("status")
    assert code == 1
    assert "[ERR] unexpected error (TypeError: a fault deep inside)" in out
    assert "--debug" in out and "Traceback" not in err
    code, out, err = run("status", "--debug")
    assert code == 1 and "Traceback" in err


def test_rfdata_init_and_verify(run, rf):
    code, out, _ = run("rfdata", "init")
    assert code == 0 and "rf_data is ready at" in out
    assert (rf.root / "README.txt").exists()
    base, _ = _burst_capture(rf)
    code, out, _ = run("rfdata", "verify")
    assert code == 0 and "1 unchanged, 0 changed since they were written, 0 missing" in out
    with open(sigmf.data_path(base), "r+b") as f:
        f.write(b"\xff\xff")
    os.utime(sigmf.data_path(base), ns=(1, 1))
    code, out, _ = run("rfdata", "verify", "--full")
    assert code == 1 and "[ERR] CHANGED:" in out and "named, not used" in out
    sigmf.data_path(base).unlink()
    code, out, _ = run("rfdata", "verify")
    assert code == 1 and "[ERR] MISSING:" in out


# -- profiles ------------------------------------------------------------------------
def test_profile_new_show_list(run, rf):
    code, out, _ = run("profile", "list")
    assert code == 0 and "No receiver profiles" in out
    code, out, _ = run("profile", "new", PID, "--serial", "00000001")
    assert code == 0 and f"Profile {PID} — the RTL-SDR at 1.024 MS/s — saved" in out
    assert "impair measure" in out                  # the next step, in words
    saved = json.loads(rf.profile_json(PID).read_text("utf-8"))
    assert saved["device_serial"] == "00000001"
    code, out, _ = run("profile", "new", PID)
    assert code == 1 and "already exists" in out and "--overwrite" in out
    code, out, _ = run("profile", "new", PID, "--overwrite")
    assert code == 0
    code, out, _ = run("profile", "new", "--family", "hackrf", "--rate", "8e6")
    assert code == 0 and "hackrf_8000000_ci8" in out
    code, out, _ = run("profile", "new", "--family", "kraken", "--rate", "2.4e6")
    assert code == 1 and "unknown receiver family 'kraken'" in out
    code, out, _ = run("profile", "new")
    assert code == 1 and "give --family and --rate" in out
    code, out, _ = run("profile", "show", PID)
    assert code == 0 and "canonical rates for cuts: voice 51.2 kS/s (÷20)" in out
    assert "maximum safe input: NOT entered" in out
    code, out, _ = run("profile", "list")
    assert code == 0 and out.count("\n") == 2 and "hackrf_8000000_ci8 —" in out


def test_safe_input_is_entered_with_its_source_and_date(run, rf):
    code, out, _ = run("profile", "set-safe-input", PID, "--max-dbm", "10",
                       "--source", "", "--date", "2026-10-09")
    assert code == 1 and "A number without its source is a number from memory" in out
    code, out, _ = run("profile", "set-safe-input", PID, "--max-dbm", "10",
                       "--source", "data sheet rev A", "--date", "09/10/2026")
    assert code == 1 and "YYYY-MM-DD" in out
    code, out, _ = run("profile", "set-safe-input", PID, "--max-dbm", "10",
                       "--source", "TEST FIXTURE, not a data sheet", "--date",
                       "2026-10-09")
    assert code == 0 and "ceiling for this receiver is -10 dBm" in out
    p = profiles.load_profile(rf, PID)
    assert p.safe_input.max_dbm == 10.0 and p.safe_input.entered == "2026-10-09"
    assert p.safe_input.source == "TEST FIXTURE, not a data sheet"


def test_geometry_shown_and_changed_with_a_note(run, rf):
    run("profile", "new", PID)
    code, out, _ = run("profile", "geometry", PID)
    assert code == 0 and "1024-point FFT, hop 1024" in out
    code, out, _ = run("profile", "geometry", PID, "--fft-size", "2048",
                       "--hop", "2048", "--fam-channel-fft", "128")
    assert code == 0 and "stft.fft_size changed from 1024 to 2048" in out
    assert "will be refused (in words) until they are retrained" in out
    p = profiles.load_profile(rf, PID)
    assert p.stft.fft_size == 2048 and p.fam.channel_fft == 128
    assert any("geometry changed" in n for n in p.notes)
    code, out, _ = run("profile", "geometry", PID, "--tile-overlap", "1.5")
    assert code == 1 and "overlap is a fraction" in out


def test_impair_measure_files_impairments_and_the_floor(run, rf):
    term = _terminated(rf)
    code, out, _ = run("impair", "measure", str(term), "--serial", "S1")
    assert code == 0, out
    assert "Impairments: measured" in out and "Detector floor:" in out
    p = profiles.load_profile(rf, PID)
    assert p.impairments["floor_mean_dbfs"] < -30 and p.device_serial == "S1"
    assert len(p.impairments["floor_db_per_bin"]) == p.stft.fft_size
    code, out, _ = run("impair", "measure", str(term), "--profile", "hackrf_8000000_ci8")
    assert code == 1 and "Profiles never mix" in out
    code, out, _ = run("impair", "measure", str(rf.root / "nothing"))
    assert code == 1 and "no SigMF capture" in out


# -- datasets ------------------------------------------------------------------------
def test_synth_narrowband_list_and_throughput(run, rf):
    code, out, _ = run("synth", "narrowband", "--profile", PID, "--name", "nb",
                       "--classes", "ref_bpsk,noise", "--n-per-class", "3",
                       "--snr", "5,15", "--window", "128", "--no-scf")
    assert code == 0, out
    assert "Dataset nb (narrowband) for the RTL-SDR at 1.024 MS/s: 6 train" in out
    assert "integer decimation ÷20" in out and "INVENTED" in out
    d = rf.datasets(PID, "nb")
    assert (d / "manifest.json").exists() and list((d / "train").glob("shard_*.npz"))
    code, out, _ = run("synth", "narrowband", "--profile", PID, "--name", "nb",
                       "--classes", "ref_bpsk", "--n-per-class", "1")
    assert code == 1 and "already exists" in out
    code, out, _ = run("synth", "narrowband", "--profile", PID, "--name", "x",
                       "--classes", "ref_bpsk", "--n-per-class", "1", "--snr", "5")
    assert code == 1 and "--snr is LOW,HIGH" in out
    code, out, _ = run("synth", "list", "--verify")
    assert code == 0 and f"{PID}/nb: narrowband, native, 6/0/0" in out
    assert "verified" in out
    code, out, _ = run("synth", "throughput", "--profile", PID, "--n", "1",
                       "--window", "128", "--classes", "ref_bpsk")
    assert code == 0 and "examples a second" in out and "Nothing was written" in out


def test_synth_wideband_for_bills_region(run, rf):
    code, out, _ = run("synth", "wideband", "--profile", PID, "--name", "wb",
                       "--scenes", "1", "--seconds", "0.1", "--env",
                       "us-va-nokesville", "--no-tiles")
    assert code == 0, out
    assert "Dataset wb (wideband)" in out and "a prior, not the place" in out
    from atk_diffusion.detect import classes
    assert f"classes: {len(classes.CLASSES)} — the whole class table" in out
    assert list((rf.datasets(PID, "wb") / "train").glob("scene_000.sigmf-meta"))


# -- detect, cut, resample -------------------------------------------------------------
def test_detect_finds_the_burst_and_writes_annotations(run, rf):
    base, lab = _burst_capture(rf)
    code, out, _ = run("detect", str(base), "--write-annotations")
    assert code == 0, out
    assert "Every one is PROPOSED until a decoder confirms it." in out
    anns = [a for a in sigmf.annotations(base) if a.source == "proposed"]
    lo, hi = F0 + lab["f_lo_hz"], F0 + lab["f_hi_hz"]
    hits = [a for a in anns if a.freq_lower_edge < hi and a.freq_upper_edge > lo
            and abs(a.sample_start / FS - 0.3) < 0.05]
    assert hits, out
    assert "annotations now in sig.sigmf-meta" in out
    assert rf.verify(sigmf.meta_path(base))[0]       # the log follows the meta
    code, out, _ = run("--json", "detect", str(base), "--max-seconds", "1.1")
    d = json.loads(out)
    assert d["ok"] and d["result"]["profile"] == PID and d["result"]["detections"]


def test_detect_refuses_another_profile_and_says_no_model(run, rf):
    base, _ = _burst_capture(rf, seconds=0.6)
    code, out, _ = run("detect", str(base), "--profile", "hackrf_8000000_ci8")
    assert code == 1 and "Profiles never mix" in out
    code, out, _ = run("detect", str(base), "--learned", "--list", "1")
    assert code == 0 and "there is no trained proposer for" in out
    code, out, _ = run("detect", str(base), "--proposer", "nope")
    assert code == 1 and "there is no model 'nope'" in out


def test_cut_writes_original_cleaned_analysis_and_report(run, rf):
    base, lab = _burst_capture(rf)
    lo, hi = F0 + lab["f_lo_hz"], F0 + lab["f_hi_hz"]
    code, out, _ = run("cut", str(base), "--t0", "0.3", "--t1", "0.7",
                       "--f-lo", f"{lo - 1e3:.0f}", "--f-hi", f"{hi + 1e3:.0f}",
                       "--clean", "wiener", "--who", "Bill")
    assert code == 0, out
    assert "RECORD tier — it is never changed" in out
    assert "occupied bandwidth (99 %)" in out and "SNR above the floor" in out
    assert "clean 'wiener': CLEANED" in out and "measured SNR" in out
    folders = list(rf.cuts(PID).iterdir())
    assert len(folders) == 1
    f = folders[0]
    for name in ("original.sigmf-data", "original.sigmf-meta", "cleaned.sigmf-meta",
                 "analysis.json", "report.md"):
        assert (f / name).exists(), name
    a = json.loads((f / "analysis.json").read_text("utf-8"))
    assert a["cut"]["cut_by"] == "Bill" and a["cut"]["decimation"] == 20
    code, out, err = run("cut", str(base), "--t0", "0.3", "--t1", "0.7",
                         "--f-lo", "1", "--f-hi", "2", "--clean", "diffusion")
    assert code == 2 and "invalid choice: 'diffusion'" in err


def test_resample_is_a_logged_step(run, rf):
    base, _ = _burst_capture(rf, seconds=0.2)
    code, out, _ = run("resample", str(base), "--to", PID)
    assert code == 1 and "already" in out
    code, out, _ = run("resample", str(base), "--to", "rtlsdr_2048000_cu8",
                       "--reason", "test")
    assert code == 0 and "polyphase 2/1" in out and "RESAMPLED" in out
    out_base = rf.captures("rtlsdr_2048000_cu8") / f"sig_from_{PID}"
    meta = sigmf.read_meta(out_base)
    assert meta["global"]["atk:resampled_from"] == PID
    assert (rf.runs("rtlsdr_2048000_cu8") / "resample_log.txt").read_text(
        "utf-8").count("resampled sig from") == 1
    code, out, _ = run("resample", str(base), "--to", "rtlsdr_2048000_cu8")
    assert code == 1 and "--overwrite replaces it" in out


# -- the cabled loop: words, and never a transmission without both flags ---------
RX = "rtlsdr_2400000_cu8"
LOOP = ["--tx", "hackrf", "--rx-profile", RX, "--freq", "915e6", "--tx-gain", "0",
        "--tx-power-dbm", "-40", "--tx-power-at-gain", "0", "--tx-power-source",
        "TEST FIXTURE, not a data sheet", "--tx-power-date", "2026-10-09",
        "--attenuation-db", "30", "--cabled", "--dc-block"]


def _safe(run):
    run("profile", "set-safe-input", RX, "--max-dbm", "10", "--source",
        "TEST FIXTURE, not a data sheet", "--date", "2026-10-09")


def test_cabled_check_refuses_and_says_how_to_clear_it(run):
    code, out, _ = run("cabled", "check", "--tx", "hackrf", "--rx-profile", RX,
                       "--freq", "915e6")
    assert code == 1
    assert "[ERR] REFUSED: the cable is not confirmed" in out
    assert "--cabled confirms the cable; --dc-block confirms the DC block" in out
    assert "profile set-safe-input" in out
    assert "Nothing was transmitted" in out
    _safe(run)
    code, out, _ = run("cabled", "check", *LOOP)
    assert code == 0 and "Safe to run on the cable." in out
    assert "Expected input at the receiver: -70.0 dBm" in out
    code, out, _ = run("cabled", "check", *LOOP[:-4], "--attenuation-db", "0",
                       "--tx-power-dbm", "15", "--cabled", "--dc-block")
    assert code == 1 and "above the ceiling" in out


def test_the_cli_never_transmits_without_both_flags_and_a_safe_verdict(run, rf,
                                                                      monkeypatch):
    from atk_diffusion.cabled import loop
    calls = []
    monkeypatch.setattr(loop, "subprocess_runner",
                        lambda args, timeout_s=None: calls.append(list(args)) or (0, "ok"))
    _safe(run)
    sig = ["--signal", "ref_qpsk,0.01,100e3", "--name", "t1"]
    code, out, _ = run("cabled", "plan", *LOOP, *sig)
    assert code == 0 and "NOTHING WAS TRANSMITTED" in out and calls == []
    assert "hackrf_transfer -t" in out and "-a 0" in out
    manifest = rf.cabled(RX) / "tx" / "t1.manifest.json"
    code, out, _ = run("cabled", "plan", *LOOP, "--tx-file", str(manifest),
                       "--execute")
    assert code == 1 and "needs --i-confirm-cabled-with-attenuators" in out
    assert calls == []
    # both flags, but the first run of a setup must be at the minimum gain
    bad = [v if v != "0" or LOOP[i - 1] != "--tx-gain" else "10"
           for i, v in enumerate(LOOP)]
    code, out, _ = run("cabled", "plan", *bad, "--tx-file", str(manifest),
                       "--execute", "--i-confirm-cabled-with-attenuators")
    assert code == 1 and calls == [] and "minimum TX gain" in out
    # both flags and a safe verdict: exactly the planned command, once
    code, out, _ = run("cabled", "plan", *LOOP, "--tx-file", str(manifest),
                       "--execute", "--i-confirm-cabled-with-attenuators")
    assert code == 0, out
    assert len(calls) == 1 and calls[0][:3] == ["hackrf_transfer", "-t",
                                               str(rf.cabled(RX) / "tx" / "t1.cs8")]
    assert "transmitted on the cable" in out
    ramp = json.loads((rf.cabled(RX) / "loop_ramp.json").read_text("utf-8"))
    assert len(next(iter(ramp["setups"].values()))["steps"]) == 1


def _record(rf, manifest_path, *, amp=0.1, noise=0.01, pre_s=0.1, name="rx"):
    from atk_diffusion.dsp import iq
    m = json.loads(Path(manifest_path).read_text("utf-8"))
    y = iq.to_complex((Path(manifest_path).parent / m["file"]).read_bytes(),
                      m["datatype"])
    fs = m["tx_rate"]
    x = np.concatenate([np.zeros(int(pre_s * fs)), y, np.zeros(int(0.02 * fs))])
    rng = np.random.default_rng(0)
    x = amp * x + noise * (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size))
    base = rf.captures(RX) / name
    sigmf.write_pair(base, x.astype(np.complex64), fs, 915e6, datatype="cu8",
                     extra_global={"atk:receiver_profile": RX}, hw="RTL-SDR")
    return base


def test_txfile_measure_align_and_the_cabled_dataset(run, rf):
    _safe(run)
    code, out, _ = run("cabled", "txfile", "--tx", "hackrf", "--rx-profile", RX,
                       "--preset", "--name", "ref", "--freq", "915e6")
    assert code == 0 and "8 signals between a chirp start marker" in out
    assert "Nothing was transmitted" in out
    manifest = rf.cabled(RX) / "tx" / "ref.manifest.json"
    code, out, _ = run("cabled", "txfile", "--tx", "hackrf", "--rx-profile", RX)
    assert code == 1 and "--signal CLASS" in out
    rec = _record(rf, manifest)
    code, out, _ = run("cabled", "measure", str(rec), "--tx-file", str(manifest),
                       *LOOP)
    assert code == 0, out
    assert "marker level" in out and "Recorded for the ramp at TX gain 0" in out
    code, out, _ = run("cabled", "align", str(rec), "--tx-file", str(manifest),
                       *LOOP)
    assert code == 0, out
    assert "synth cabled" in out
    labelled = [p for p in rf.cabled(RX).glob("*.sigmf-meta")]
    assert labelled
    code, out, _ = run("synth", "cabled", str(labelled[0]), "--profile", RX,
                       "--name", "cab", "--window", "256", "--no-scf")
    assert code == 0, out
    assert "Dataset cab (narrowband)" in out and "generator cabled" in out
    assert "RECORD" in out


# -- experiments ----------------------------------------------------------------------
def test_experiment_list_names_every_experiments_module(run):
    import atk_diffusion.experiments as E
    code, out, _ = run("experiment", "--list")
    assert code == 0
    mods = {m.name for m in pkgutil.iter_modules(E.__path__)} - {"report"}
    entries = " ".join(v[2] for v in cli.EXPERIMENTS.values())
    for m in sorted(mods):
        assert f"experiments.{m}." in entries, f"{m} has no `experiment` command"
    for name in cli.EXPERIMENTS:
        assert f"  {name} " in out
    code, out, _ = run("experiment")
    assert code == 0 and "weak-burst" in out


@pytest.mark.parametrize("name, args, where", [
    ("e1-coverage", ["--trials", "12", "--failure-trials", "12", "--max-cells",
                     "4000", "--seed", "1"], "krakensdr_2400000_cu8/runs/geo_eval"),
    ("pulse", ["--seeds", "1", "--duration", "0.1"], "shared/runs"),
    ("hunter", ["--duration", "60", "--seeds", "1"], "rtlsdr_2400000_cu8/runs"),
    ("vitals", ["--duration", "35", "--no-lstm"], "esp32csi_50_cf32/runs"),
    ("hf", [], "kiwisdr_12000_ci16/runs"),
    ("novelty", ["--n-docs", "3"], "shared/runs"),
    ("e5-reach", ["--no-learned"], "krakensdr_2400000_cu8/runs/geo_eval"),
    ("weak-burst", ["--profile", PID, "--bursts", "pocsag", "--snrs", "0,12",
                    "--trials", "3", "--noise-trials", "8"], f"{PID}/runs"),
    ("inpaint", ["--trials", "3", "--silence-trials", "3", "--gaps-ms", "1"],
     "rtlsdr_2400000_cu8/runs"),
    ("fingerprint", ["--enrol", "3", "--test", "3", "--no-learned"],
     "hackrf_8000000_ci8/runs/fingerprint_eval"),
])
def test_cheap_experiments_report_in_words_and_files(run, rf, name, args, where):
    code, out, _ = run("experiment", name, *args)
    assert code == 0, out
    assert f"Experiment {name} (plan " in out
    assert "The numbers are MEASURED" in out
    reports = [ln.split("report: ", 1)[1] for ln in out.splitlines()
               if ln.strip().startswith("report: ")]
    assert reports, out
    for r in reports:
        p = Path(r)
        assert p.is_file()
        assert (rf.root / where).resolve() in p.resolve().parents, (r, where)
    words = [ln for ln in out.splitlines()
             if ln and not ln.startswith(("  ", "Experiment ", "Done in"))]
    assert words, "an experiment must say its result in a sentence"


def test_experiment_e5_leaves_a_coverage_product(run, rf):
    code, out, _ = run("products", "list")
    assert code == 0 and "No products yet" in out
    run("experiment", "e5-reach", "--no-learned")
    code, out, _ = run("products", "list", "--verify")
    assert code == 0 and "coverage/e5-0:" in out and "verified" in out
    code, out, _ = run("products", "list", "--kind", "tea")
    assert code == 1 and "not a product kind" in out


def test_the_wer_experiment_runs_a_transcriber_command(run, rf, tmp_path):
    rng = np.random.default_rng(0)
    clips = tmp_path / "clips"
    clips.mkdir()
    for k in range(2):
        x = (0.2 * np.sin(2 * np.pi * 200 * np.arange(8000) / 8000.0)
             + 0.01 * rng.standard_normal(8000))
        with wave.open(str(clips / f"c{k}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes((x * 32767).astype("<i2").tobytes())
    refs = tmp_path / "refs.txt"
    refs.write_text("c0\thello world\nc1\thello there\n", encoding="utf-8")
    script = tmp_path / "fake_whisper.py"
    script.write_text("import sys\nprint('hello world')\n", encoding="utf-8")
    code, out, _ = run("experiment", "wer", "--clips", str(clips), "--references",
                       str(refs), "--transcriber",
                       f"{sys.executable} {script} {{wav}}")
    assert code == 0, out
    assert "report:" in out
    code, out, _ = run("experiment", "wer", "--clips", str(clips), "--references",
                       str(refs), "--transcriber", "whisper")
    assert code == 1 and "{wav}" in out


# -- hunt, vitals, repair --------------------------------------------------------------
def test_hunt_simulate_logs_every_retune_and_never_transmits(run, rf):
    code, out, _ = run("hunt", "simulate", "--duration", "60", "--seed", "2")
    assert code == 0, out
    assert "Goal: anything, no wider than 30 kHz, bursty, between 400 MHz and 470 MHz" in out
    assert "It never transmits." in out and "SIMULATION" in out
    logs = list((rf.runs("rtlsdr_2400000_cu8") / "hunts").rglob("hunt_log.jsonl"))
    assert len(logs) == 1 and logs[0].read_text("utf-8").count("\n") > 10
    code, out, _ = run("hunt", "simulate", "--goal", "everything everywhere")
    assert code == 1 and "cannot be hunted" in out


def test_vitals_replay_reads_a_saved_csi_log(run, rf, tmp_path):
    from atk_diffusion.experiments import vitals_eval as V
    from atk_diffusion.sensing import csi
    t, H, _truth = V.synth_csi(40.0, 50.0, breath_bpm=15.0, heart_bpm=72.0,
                               rng=np.random.default_rng(3))
    frames = [csi.CsiFrame(csi=H[i], mac="aa:bb:cc:dd:ee:ff", t_us=int(t[i] * 1e6))
              for i in range(len(t))]
    log = csi.save_log(frames, tmp_path / "csi.txt")
    code, out, _ = run("vitals", "replay", str(log))
    assert code == 0, out
    assert "RESEARCH-GRADE MEASUREMENT" in out and "not a medical device" in out
    line = next(ln for ln in out.splitlines() if ln.startswith("breathing:"))
    assert abs(float(line.split()[1]) - 15.0) < 1.5
    assert list(rf.runs("esp32csi_50_cf32").glob("vitals_replay_*/result.json"))
    (tmp_path / "empty.txt").write_text("boot log only\n", encoding="utf-8")
    code, out, _ = run("vitals", "replay", str(tmp_path / "empty.txt"))
    assert code == 1 and "0 CSI frame(s)" in out


def test_repair_audio_fills_and_lists(run, tmp_path):
    t = np.arange(8000) / 16000.0
    x = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 440 * t)
    x[4000:4080] = 0.0
    src = tmp_path / "a.wav"
    with wave.open(str(src), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    code, out, _ = run("repair", "audio", str(src))
    assert code == 0 and "1 gap(s) filled by janssen" in out and "INFERRED" in out
    assert (tmp_path / "a_filled.wav").exists() and (tmp_path / "a_filled.wav.json").exists()
    code, out, _ = run("repair", "audio", str(src), "--out", str(src))
    assert code == 1 and "the original is never changed" in out


# -- training ---------------------------------------------------------------------------
def test_training_without_pytorch_says_where_it_runs(run, monkeypatch):
    from atk_diffusion import capabilities
    monkeypatch.setattr(capabilities, "can_train",
                        lambda: (False, "PyTorch is not in this environment."))
    code, out, _ = run("train", "beacon", "--profile", PID)
    assert code == 1 and "training the beacon needs PyTorch" in out
    assert "install.bat builds the .venv" in out


def test_train_anomaly_without_pytorch_and_models_list(run, rf, tmp_path):
    rng = np.random.default_rng(0)
    flows = [{"bytes_out": float(rng.normal(1000, 50)), "bytes_in": 5000.0,
              "packets_out": 10, "packets_in": 20, "duration_s": 1.0,
              "mean_iat_s": 0.1, "std_iat_s": 0.01, "distinct_ports": 1,
              "connections": 1} for _ in range(30)]
    fp = tmp_path / "flows.jsonl"
    fp.write_text("\n".join(json.dumps(f) for f in flows), encoding="utf-8")
    code, out, _ = run("train", "anomaly", "--flows", str(fp), "--name", "an")
    assert code == 0, out
    assert "never a suppression" in out
    assert (rf.shared() / "models" / "an" / "card.json").exists()
    code, out, _ = run("models", "list", "--verify")
    assert code == 0 and "shared/an:" in out


def test_train_a_tiny_proposer_then_detect_with_it_and_evaluate(run, rf):
    torch = pytest.importorskip("torch", reason="PyTorch is only in the training "
                                "environment")
    torch.set_num_threads(1)
    code, out, _ = run("synth", "wideband", "--profile", PID, "--name", "wb",
                       "--scenes", "2", "--seconds", "1.1", "--splits", "0.5,0,0.5")
    assert code == 0, out
    code, out, _ = run("train", "proposer", "--profile", PID, "--dataset", "wb",
                       "--name", "prop", "--epochs", "1", "--max-tiles", "1",
                       "--width", "8", "--threads", "1", "--device", "cpu")
    assert code == 0, out
    assert f"Saved the 2D proposer to {rf.models(PID, 'prop')}" in out
    assert "outputs are PROPOSED" in out
    code, out, _ = run("export", "onnx", "prop", "--profile", PID)
    assert code == 0 and "loads in ONNX Runtime" in out
    base, _ = _burst_capture(rf)
    code, out, _ = run("detect", str(base), "--learned")
    assert code == 0 and "Learned proposer: prop (the newest" in out
    code, out, _ = run("experiment", "detector", "--profile", PID, "--proposer",
                       "prop", "--synthetic-dataset", "wb")
    assert code == 0, out
    assert "mAP@0.5" in out and "detector_eval.md" in out
    code, out, _ = run("train", "calibrate", "--profile", PID, "--model", "prop",
                       "--dataset", "wb", "--split", "test")
    assert code == 0 and "Calibrated prop on wb (test split)" in out
    code, out, _ = run("experiment", "domain-gap", "--profile", PID, "--model",
                       "prop", "--synthetic-dataset", "wb", "--cabled-dataset", "wb")
    assert code == 0 and "domain_gap.md" in out
    code, out, _ = run("experiment", "minutes", "--profile", PID, "--model", "prop",
                       "--onsite-dataset", "wb", "--acceptance", "0.5",
                       "--minutes", "0.5,1")
    assert code == 0 and "minutes_to_acceptable.md" in out
    code, out, _ = run("export", "onnx", "nope", "--profile", PID)
    assert code == 1 and "there is no model 'nope'" in out


@pytest.mark.parametrize("args, where, words", [
    (["beacon", "--profile", "hackrf_8000000_ci8", "--steps", "5",
      "--eval-blocks", "50"], "hackrf_8000000_ci8/models/beacon", "DESIGN STUDY"),
    (["radiomap", "--synthetic", "6", "--size", "16", "--steps", "2", "--batch", "3"],
     "shared/models/radiomap", "Synthetic fields: 6 patches"),
    (["position", "--synthetic", "40", "--epochs", "1"], "shared/models/position",
     "drive records from the synthetic drive world"),
    (["vitals", "--synthetic", "4", "--epochs", "1"], "esp32csi_50_cf32/models/vitals",
     "not a medical device"),
    (["fingerprint", "--profile", "hackrf_8000000_ci8", "--synthetic", "4",
      "--steps", "2"], "hackrf_8000000_ci8/models/fingerprint",
     "Synthetic bursts: 8 from 2 same-model radios"),
])
def test_quick_trainers_save_a_model_with_its_card(run, rf, args, where, words):
    pytest.importorskip("torch", reason="PyTorch is only in the training "
                        "environment").set_num_threads(1)
    code, out, _ = run("train", *args, "--threads", "1")
    assert code == 0, out
    assert words in out and "Saved the" in out
    d = rf.root / where
    assert (d / "card.json").exists()
    from atk_diffusion import cards
    assert cards.load(d).name                       # loads, weights verified
    if args[0] == "radiomap":
        code, out, _ = run("export", "onnx", str(d))
        assert code == 0 and "loads in ONNX Runtime" in out


# -- `python -m atk_diffusion` itself ------------------------------------------------
def test_python_dash_m_runs_the_command_line(tmp_path):
    env = dict(os.environ, ATK_RF_DATA=str(tmp_path / "rf"))
    env["PYTHONPATH"] = str(Path(cli.__file__).resolve().parents[1])
    p = subprocess.run([sys.executable, "-m", "atk_diffusion", "status"],
                       capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == 0, p.stderr
    assert "ATK Diffusion Toolkit 0.1.0" in p.stdout
    p = subprocess.run([sys.executable, "-m", "atk_diffusion", "nope"],
                       capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == 2 and "[ERR]" in p.stderr
