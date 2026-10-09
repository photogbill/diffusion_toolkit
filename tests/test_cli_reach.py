# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""`atkdiff products reach` — plan E5 on real terrain from the command line.

The engine (geo.reach) had its own tests; what was missing was a way for an
operator to point it at DTED without writing Python. These run it on a
synthetic DTED level-1 tile written by geo.dted's own writer, and check the
product lands where the map reads it, with its tier said, and the refusals.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from atk_diffusion import cli


def _tile(root, lat0=38, lon0=-78):
    from atk_diffusion.geo import dted
    n = 121                                    # 30" posts: a small, valid tile
    y, x = np.mgrid[0:n, 0:n]
    ridge = 150.0 + 120.0 * np.exp(-((x - 60) ** 2) / 200.0)
    path = root / "w078" / "n38.dt1"
    dted.write_dted(path, ridge + 0 * y, lat0, lon0, 30.0, 30.0, level=1)
    return root


def _run(argv, capsys):
    code = cli.main(argv + ["--json"])
    out = capsys.readouterr()
    return code, json.loads(out.out) if out.out.strip() else {}, out.err


def test_reach_over_dted_writes_a_product_the_map_can_list(tmp_path, capsys):
    rf_root = tmp_path / "rf_data"
    _tile(rf_root / "shared" / "dted")
    code, res, err = _run(["--rf-data", str(rf_root), "products", "reach",
                           "--lat", "38.5", "--lon", "-77.6", "--freq", "150e6",
                           "--power-w", "5", "--radius-km", "6", "--grid-m", "500",
                           "--model", "deygout", "--label", "test radio"], capsys)
    assert code == 0, err
    r = res["result"]
    assert r["tier"] == "inferred" and r["model"] == "deygout"
    run = rf_root / "products" / "coverage"
    assert any(run.iterdir()), "nothing under products\\coverage"
    assert "never a promise of contact" in err
    code, res, err = _run(["--rf-data", str(rf_root), "products", "list",
                           "--kind", "coverage"], capsys)
    assert code == 0 and res["result"]["products"], err


def test_no_dted_is_a_sentence_and_flat_is_the_sanity_line(tmp_path, capsys):
    rf_root = tmp_path / "rf_data"
    code, res, err = _run(["--rf-data", str(rf_root), "products", "reach",
                           "--lat", "38.5", "--lon", "-77.6", "--freq", "150e6",
                           "--power-w", "5", "--model", "fspl"], capsys)
    assert code == 1 and "no DTED folder" in res["error"]
    code, res, err = _run(["--rf-data", str(rf_root), "products", "reach",
                           "--lat", "38.5", "--lon", "-77.6", "--freq", "150e6",
                           "--power-w", "5", "--model", "fspl", "--flat",
                           "--radius-km", "3", "--grid-m", "500"], capsys)
    assert code == 0, err


def test_outside_the_tiles_and_bad_power_are_refused(tmp_path, capsys):
    rf_root = tmp_path / "rf_data"
    _tile(rf_root / "shared" / "dted")
    code, res, _ = _run(["--rf-data", str(rf_root), "products", "reach",
                         "--lat", "40.5", "--lon", "-77.6", "--freq", "150e6",
                         "--power-w", "5", "--model", "fspl"], capsys)
    assert code == 1 and "does not cover" in res["error"]
    code, res, _ = _run(["--rf-data", str(rf_root), "products", "reach",
                         "--lat", "38.5", "--lon", "-77.6", "--freq", "150e6",
                         "--power-w", "5", "--power-dbm", "37"], capsys)
    assert code == 1 and "once" in res["error"]
    code, res, _ = _run(["--rf-data", str(rf_root), "products", "reach",
                         "--lat", "38.5", "--lon", "-77.6", "--freq", "150e6",
                         "--power-w", "5", "--antenna", "directional"], capsys)
    assert code == 1 and "--beamwidth" in res["error"]


def test_itm_runs_when_itmlogic_is_installed(tmp_path, capsys):
    pytest.importorskip("itmlogic", reason="itmlogic is not installed here")
    rf_root = tmp_path / "rf_data"
    _tile(rf_root / "shared" / "dted")
    code, res, err = _run(["--rf-data", str(rf_root), "products", "reach",
                           "--lat", "38.5", "--lon", "-77.6", "--freq", "150e6",
                           "--power-w", "5", "--radius-km", "4", "--grid-m", "1000",
                           "--model", "itm"], capsys)
    assert code == 0, err
    assert res["result"]["model"] == "itm"
