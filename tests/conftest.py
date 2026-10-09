# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""Test isolation: no test may write into a real rf_data folder.

Every test session points ATK_RF_DATA at a temporary folder before any test
runs, so `paths.default_root()` can never resolve to D:\\Analyst_Toolkit\\
rf_data on Bill's machine. Tests that want their own root use the `rf`
fixture.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SESSION_ROOT = Path(tempfile.mkdtemp(prefix="atkdiff_rfdata_"))
os.environ["ATK_RF_DATA"] = str(_SESSION_ROOT)


@pytest.fixture
def rf(tmp_path):
    from atk_diffusion.paths import RfData
    return RfData(tmp_path / "rf_data", create=True)


@pytest.fixture
def rng():
    import numpy as np
    return np.random.default_rng(1234)


def torch_or_skip():
    """Tests of the training environment skip, with a printed reason, when
    PyTorch is absent (ATK's core environment does not have it)."""
    return pytest.importorskip("torch", reason="PyTorch is only in the "
                               "training environment")
