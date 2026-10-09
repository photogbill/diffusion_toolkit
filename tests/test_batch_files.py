# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The toolkit-root batch files (install.bat, atkdiff.bat), pinned by
structure: cmd.exe cannot run here, so what makes a Windows batch file go
wrong is checked as text — LF-only line endings (cmd misreads labels and
`goto` in them), a bare `pause` (a scripted run hangs on it), pip.exe or the
PATH python (ATK's rule: `<venv python> -m …` always), an unescaped `)` in
an echo inside a block (it closes the block early), a `call :label` with no
label, and the Python one-liners they run (compiled, and the version probe
run for real)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BATS = ("install.bat", "atkdiff.bat")


def _raw(name: str) -> bytes:
    return (ROOT / name).read_bytes()


def _lines(name: str) -> list[str]:
    return _raw(name).decode("ascii").split("\r\n")


def _code(name: str) -> list[str]:
    """Lines that are commands: not REM comments, not blank."""
    return [ln for ln in _lines(name)
            if ln.strip() and not ln.strip().upper().startswith("REM")]


@pytest.mark.parametrize("name", BATS)
def test_crlf_ascii_and_setlocal_first(name):
    raw = _raw(name)
    assert raw.count(b"\n") == raw.count(b"\r\n") > 10, "every line ends CRLF"
    assert b"\r\r" not in raw
    raw.decode("ascii")                                  # ASCII only
    code = _code(name)
    assert code[0].lower() == "@echo off"
    first_set = next(i for i, ln in enumerate(code) if ln.lower().startswith("set "))
    setloc = next(i for i, ln in enumerate(code) if ln.lower().startswith("setlocal"))
    assert setloc < first_set, "setlocal before the first set"


@pytest.mark.parametrize("name", BATS)
def test_no_bare_pause(name):
    for ln in _code(name):
        if re.search(r"\bpause\b", ln, re.I):
            assert re.match(r"\s*if\s+defined\s+\w+\s+pause\s*$", ln, re.I), ln


@pytest.mark.parametrize("name", BATS)
def test_never_pip_exe_or_the_path_python(name):
    for ln in _code(name):
        low = ln.lower()
        if low.lstrip().startswith("echo"):
            continue
        assert "pip.exe" not in low and "scripts\\pip" not in low, ln
        if "pip install" in low:
            assert "-m pip install" in low, ln
        assert not re.match(r"\s*python(\.exe)?\s", low), ln
        assert not re.search(r"(^|[\s&|(])python3?(\.exe)?\s+-", low), ln


@pytest.mark.parametrize("name", BATS)
def test_echo_text_never_closes_a_block(name):
    for ln in _code(name):
        s = ln.strip()
        if s.lower().startswith("echo") and s.lower() not in ("echo(", "echo off"):
            text = s[4:]
            assert not re.search(r"(?<!\^)[()]", text), f"escape the parenthesis: {ln}"


@pytest.mark.parametrize("name", BATS)
def test_blocks_balance_and_labels_exist(name):
    depth = 0
    for ln in _code(name):
        s = ln.strip()
        if s.lower().startswith("echo"):
            continue
        s = re.sub(r'"[^"]*"', '""', s)                  # quoted text is literal
        s = s.replace("^(", "").replace("^)", "")
        for ch in s:
            depth += {"(": 1, ")": -1}.get(ch, 0)
            assert depth >= 0, ln
    assert depth == 0
    text = "\r\n".join(_code(name))
    labels = set(re.findall(r"^:(\w+)", text, re.M))
    for target in re.findall(r"\b(?:call|goto)\s+:(\w+)", text, re.I):
        assert target.lower() == "eof" or target in labels, f":{target} is missing"


def test_atkdiff_runs_the_module_with_every_argument():
    code = "\r\n".join(_code("atkdiff.bat"))
    assert '"%PY%" -m atk_diffusion %*' in code
    assert "..\\ATK\\envs\\atk_diffusion\\Scripts\\python.exe" in code
    assert ".venv\\Scripts\\python.exe" in code
    i_atk = code.index("..\\ATK\\envs\\atk_diffusion")
    i_venv = code.index("%ROOT%.venv\\Scripts\\python.exe")
    assert i_atk < i_venv, "ATK's training environment first, then .venv"
    assert "PYTHONPATH=%ROOT%" in code                   # this folder's code runs
    assert "exit /b %RC%" in code                         # the exit code passes through
    said = " ".join(ln for ln in _code("atkdiff.bat") if ln.strip().startswith("echo"))
    assert "get_diffusion.bat" in said and "install.bat" in said and "[ERR]" in said


def test_install_bat_cuda_by_default_cpu_on_request():
    code = "\r\n".join(_code("install.bat"))
    assert 'set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"' in code
    assert re.search(r'if /i "%%~a"=="/cpu" set "WANT_CPU=1"', code)
    assert 'set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"' in code
    assert ('"%VPY%" -m pip install torch torchvision torchaudio --index-url '
            "%TORCH_INDEX%") in code
    assert '"%VPY%" -m pip install -e ".[train]"' in code
    assert code.index("torch torchvision torchaudio") < code.index('-e ".[train]"')
    # Python 3.11: ATK's own first, then the launcher; nothing from PATH
    assert code.index("..\\ATK\\.python\\python.exe") < code.index("call :try_launcher")
    assert "py -3.11 -c" in code
    # caches and scratch beside the code, never the user profile
    for v in ("PIP_CACHE_DIR=%ROOT%.pipcache", "TEMP=%ROOT%.tmp", "TMP=%ROOT%.tmp"):
        assert f'set "{v}"' in code
    said = " ".join(ln for ln in _code("install.bat") if ln.strip().startswith("echo"))
    assert "get_diffusion.bat" in said and "envs\\atk_diffusion" in said
    for mark in ("[OK]", "[--]", "[ERR]"):
        assert mark in said
    assert "CUDA is available" in said and "CUDA is NOT available" in said
    assert "exit /b 1" in code and "exit /b 0" in code


def _python_snippets(name: str) -> list[str]:
    out = []
    for ln in _code(name):
        m = re.search(r'-c "([^"]+)"', ln)
        if m:
            out.append(m.group(1).replace("%%", "%"))     # batch doubles a %
    return out


def test_the_embedded_python_compiles_and_the_probe_answers():
    snippets = _python_snippets("install.bat")
    assert len(snippets) >= 4
    for s in snippets:
        compile(s, "<install.bat>", "exec")
    probe = next(s for s in snippets if "ATKPY=" in s and "executable" in s)
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, timeout=60).stdout
    if sys.version_info[:2] == (3, 11):
        assert out.strip() == f"ATKPY={sys.executable}"
    else:
        assert out.strip() == ""            # anything but 3.11 says nothing
    venv = next(s for s in snippets if "'ok' if" in s)
    out = subprocess.run([sys.executable, "-c", venv], capture_output=True,
                         text=True, timeout=60).stdout.strip()
    assert out == ("ATKPY=ok" if sys.version_info[:2] == (3, 11)
                   else "ATKPY=Python %d.%d" % sys.version_info[:2])


def test_no_console_script_in_the_package():
    import tomllib
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    p = meta["project"]
    assert "scripts" not in p and "gui-scripts" not in p
    assert "entry-points" not in p
    import atk_diffusion
    assert p["name"] == "atk-diffusion" and p["version"] == atk_diffusion.__version__
    assert p["requires-python"] == ">=3.10"
    assert p["dependencies"] == ["numpy>=1.24", "scipy>=1.10"]
    extras = p["optional-dependencies"]
    assert extras["core"] == ["onnxruntime", "tifffile", "itmlogic"]
    assert "itmlogic" in extras["train"]          # the E5 reach map's ITM
    assert "torchsig==2.2.0" in extras["train"] and "torch" in extras["train"]
    assert extras["text"] == ["transformers"] and extras["serial"] == ["pyserial"]
    assert "License :: Other/Proprietary License" in p["classifiers"]
    assert meta["tool"]["setuptools"]["packages"]["find"]["include"] == ["atk_diffusion*"]
