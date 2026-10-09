# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""capabilities: what this installation can do, said in words, found with
importlib.util.find_spec (never by importing PyTorch to ask). The probe is
monkeypatched so both answers — present and absent — are tested whatever
this machine has."""

from __future__ import annotations

import importlib.util
import sys

import pytest

from atk_diffusion import capabilities as C


@pytest.fixture
def only(monkeypatch):
    """only({'numpy', ...}) — find_spec answers for exactly those modules."""
    def install(present: set):
        monkeypatch.setattr(importlib.util, "find_spec",
                            lambda name, *a, **k: object() if name in present else None)
    return install


def test_has_follows_find_spec_and_survives_its_errors(monkeypatch, only):
    only({"numpy"})
    assert C.has("numpy") is True and C.has("torch") is False

    def broken(name, *a, **k):
        raise ValueError("torch.__spec__ is None")

    monkeypatch.setattr(importlib.util, "find_spec", broken)
    assert C.has("torch") is False


def test_has_never_imports_the_module(monkeypatch):
    pytest.importorskip("tifffile")
    monkeypatch.delitem(sys.modules, "tifffile", raising=False)
    assert C.has("tifffile") is True
    assert "tifffile" not in sys.modules            # asked, not imported


def test_versions_come_from_package_metadata():
    import numpy
    assert C.version("numpy") == numpy.__version__
    assert C.version("pytest")                       # not in OPTIONAL: its own name
    assert C.version("no_such_module_anywhere") == ""


def test_report_and_lines_when_only_the_core_is_here(only):
    only({"numpy", "scipy", "onnxruntime"})
    rep = C.report()
    assert rep["python"] == sys.version.split()[0]
    assert set(rep) == {"python"} | set(C.OPTIONAL)
    assert rep["numpy"]["present"] and rep["numpy"]["env"] == "core"
    assert not rep["torch"]["present"] and rep["torch"]["version"] == ""
    lines = C.lines(rep)
    assert lines[0] == f"Python {rep['python']}"
    assert any(l.startswith("[OK] numpy") and "— everything" in l for l in lines)
    torch_line = next(l for l in lines if l.startswith("[--] torch"))
    assert "(expected in the train environment)" in torch_line
    tf = next(l for l in lines if l.startswith("[--] transformers"))
    assert "(expected in the text environment)" in tf
    assert len(lines) == 1 + len(C.OPTIONAL)


def test_lines_without_an_argument_probe_now(only):
    only(set(C.OPTIONAL))
    lines = C.lines()
    assert all(l.startswith("[OK]") for l in lines[1:])


def test_can_train_and_can_infer_say_why_not(only):
    only({"numpy"})
    ok, why = C.can_train()
    assert not ok and "envs\\atk_diffusion" in why and "get_diffusion.bat" in why
    ok, why = C.can_infer()
    assert not ok and "classical detectors" in why and "get_diffusion.bat" in why
    only({"torch", "onnxruntime"})
    assert C.can_train() == (True, "") and C.can_infer() == (True, "")


class _FakeTorch:
    """Stands in for PyTorch so every GPU answer is tested on any machine."""
    def __init__(self, version="2.4.1+cu126", cuda="12.6", available=True,
                 name="NVIDIA GeForce RTX 3080 Ti"):
        self.__version__ = version
        self.version = type("V", (), {"cuda": cuda})()
        self.cuda = type("C", (), {"is_available": staticmethod(lambda: available),
                                   "get_device_name": staticmethod(lambda i=0: name)})()


@pytest.mark.parametrize("fake, words", [
    (_FakeTorch(), ("GPU: NVIDIA GeForce RTX 3080 Ti (CUDA 12.6)", "use it")),
    (_FakeTorch("2.4.1+cpu", None, False), ("CPU-only build", "install.bat installs without /cpu")),
    (_FakeTorch("2.4.1+cu126", "12.6", False), ("built for CUDA 12.6", "NVIDIA driver")),
])
def test_gpu_says_what_pytorch_can_use(monkeypatch, fake, words):
    monkeypatch.setattr(C, "has", lambda m: m == "torch")
    monkeypatch.setitem(sys.modules, "torch", fake)
    g = C.gpu()
    assert g["available"] is fake.cuda.is_available()
    assert g["torch"] == fake.__version__
    for w in words:
        assert w in g["words"], g["words"]
    assert g["name"] == ("NVIDIA GeForce RTX 3080 Ti" if g["available"] else "")


def test_gpu_without_pytorch_or_when_it_will_not_load(monkeypatch):
    monkeypatch.setattr(C, "has", lambda m: False)
    g = C.gpu()
    assert g["available"] is False and "not checked" in g["words"]
    monkeypatch.setattr(C, "has", lambda m: m == "torch")
    monkeypatch.setitem(sys.modules, "torch", None)       # import torch -> ImportError
    g = C.gpu()
    assert g["available"] is False
    assert "installed but did not load (ModuleNotFoundError" in g["words"]
    assert "rebuilding the training environment" in g["words"]
