# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""`python -m atk_diffusion library …` — the door ATK's RFMLS Training tab
runs its long jobs through: plain words, exit codes, and the JSON result in
--result-file."""

from __future__ import annotations

import json

from atk_diffusion import cli
from tests.test_library_store import F0, PID, _box, _capture


def test_the_library_commands_and_the_result_file(rf, tmp_path, capsys):
    def run(*args):
        code = cli.main(["--rf-data", str(rf.root), *map(str, args)])
        out, err = capsys.readouterr()
        return code, out + err

    res = tmp_path / "r.json"
    code, out = run("library", "new-type", "--profile", PID, "--name", "fire",
                    "--link", "nfm_voice", "--result-file", res)
    assert code == 0 and "New signal type fire" in out
    assert json.loads(res.read_text())["result"]["name"] == "fire"
    base, lab = _capture(rf)
    b = _box(lab)
    code, out = run("library", "add", "--profile", PID, "--type", "fire",
                    "--capture", str(base) + ".sigmf-meta", "--t0", b["t0_s"],
                    "--t1", b["t1_s"], "--f-lo", b["f_lo_hz"], "--f-hi", b["f_hi_hz"],
                    "--result-file", res)
    assert code == 0, out
    assert json.loads(res.read_text())["result"]["role"] == "train"
    code, out = run("library", "dirty", "--profile", PID, "--type", "all",
                    "--per-original", 3, "--preset", "light", "--result-file", res)
    assert code == 0 and "3 dirty copies of" in out
    assert "[100%]" in out
    code, out = run("library", "list", "--profile", PID)
    assert code == 0 and "1 real (1 train" in out and "3 dirty" in out
    code, out = run("library", "plan", "--profile", PID, "--canonical", "voice")
    assert code == 0 and "weak test" in out
    code, out = run("library", "add", "--profile", PID, "--type", "nope",
                    "--capture", str(base) + ".sigmf-meta", "--t0", 0, "--t1", 1,
                    "--f-lo", F0, "--f-hi", F0 + 1e4, "--result-file", res)
    assert code == 1 and "[ERR]" in out
    r = json.loads(res.read_text())
    assert r["ok"] is False and "no signal type" in r["error"]
