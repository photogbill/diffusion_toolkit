@echo off
REM ============================================================================
REM  ATK Diffusion Toolkit - stand-alone setup: its own training environment
REM  (.venv in THIS folder) for using the toolkit WITHOUT ATK.
REM
REM  INSIDE ATK THIS IS NOT THE WAY. ATK's own get_diffusion.bat builds
REM  envs\atk_diffusion (PyTorch + TorchSig 2.2.0) and puts the toolkit's
REM  classical half into envs\atk_core. Run that from the ATK folder; this
REM  script is for the toolkit on its own (a second machine, a test bench).
REM
REM      install.bat          PyTorch for CUDA 12.6 (an NVIDIA GPU)
REM      install.bat /cpu     PyTorch for the CPU only (no NVIDIA GPU)
REM
REM  What it does, all of it inside this folder:
REM    1. finds Python 3.11 - ATK's own (..\ATK\.python\python.exe) first,
REM       then the py launcher (py -3.11); refuses, in words, if neither
REM    2. builds .venv here from it (nothing on PATH is used afterwards)
REM    3. .venv\Scripts\python.exe -m pip install torch torchvision torchaudio
REM       from the PyTorch index, then -e .[train] (TorchSig 2.2.0, ONNX,
REM       ONNX Runtime, tifffile, itmlogic) - always "python -m pip", never pip.exe
REM    4. checks that torch, torchsig and atk_diffusion import, and says
REM       whether CUDA (the GPU) is usable
REM
REM  pip's download cache and scratch space are kept beside the code
REM  (.pipcache\, .tmp\), never in the user profile or AppData. Your data
REM  is never here at all: it lives in rf_data\ beside this folder.
REM  Run it again at any time; it reuses what is already right.
REM  Afterwards:  atkdiff.bat status
REM ============================================================================
setlocal enableextensions enabledelayedexpansion

set "ROOT=%~dp0"
set "VENV=%ROOT%.venv"
set "VPY=%VENV%\Scripts\python.exe"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
set "TORCH_WORDS=CUDA 12.6"
set "WANT_CPU="
for %%a in (%*) do (
    if /i "%%~a"=="/cpu" set "WANT_CPU=1"
)
if defined WANT_CPU (
    set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
    set "TORCH_WORDS=the CPU only"
)

REM  A window opened by double-clicking closes when the script ends; keep it
REM  open then (and only then) so the result can be read.
set "ATKDIFF_DCLICK="
echo %cmdcmdline% | find /i "%~nx0" >nul 2>&1 && set "ATKDIFF_DCLICK=1"

REM  Scratch and pip's cache beside the code - never %TEMP% in the profile.
set "PIP_CACHE_DIR=%ROOT%.pipcache"
set "TEMP=%ROOT%.tmp"
set "TMP=%ROOT%.tmp"
if not exist "%PIP_CACHE_DIR%" mkdir "%PIP_CACHE_DIR%" >nul 2>&1
if not exist "%TEMP%" mkdir "%TEMP%" >nul 2>&1
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PYTHONUTF8=1"
set "PYTHONNOUSERSITE=1"

echo(
echo [atkdiff] ATK Diffusion Toolkit - stand-alone setup ^(.venv in this folder^)
echo [atkdiff] Inside ATK the normal path is ATK's own get_diffusion.bat, which
echo [atkdiff] builds envs\atk_diffusion. This script is for the toolkit on its own.
echo [atkdiff] PyTorch for %TORCH_WORDS% ^(install.bat /cpu for a machine without
echo [atkdiff] an NVIDIA GPU^).

if not exist "%ROOT%atk_diffusion\__init__.py" (
    echo [ERR] This folder has no atk_diffusion\ package beside install.bat - it is
    echo       not the toolkit's folder, or the copy is incomplete. Nothing was changed.
    goto :fail
)

REM ---- 1. a Python 3.11 ---------------------------------------------------------
set "BASEPY="
call :try_python "%ROOT%..\ATK\.python\python.exe" "ATK's own Python"
if not defined BASEPY call :try_launcher
if not defined BASEPY (
    echo [ERR] No Python 3.11 was found - not ATK's own ^(..\ATK\.python\python.exe^)
    echo       and not through the py launcher ^(py -3.11^). Install ATK first ^(its
    echo       install.bat fetches its own Python 3.11^), or install Python 3.11 from
    echo       python.org, then run install.bat again. Nothing was changed.
    goto :fail
)
echo [OK] Python 3.11: !BASEPY! ^(!BASEFROM!^)

REM ---- 2. the .venv ---------------------------------------------------------------
if exist "%VPY%" (
    call :venv_ok
    if not defined VENV_OK (
        echo [--] .venv is not a Python 3.11 environment ^(!VENV_WHY!^) - building it again.
        rmdir /s /q "%VENV%" 2>nul
    ) else (
        echo [OK] .venv already exists and is Python 3.11 - reusing it.
    )
)
if not exist "%VPY%" (
    echo [atkdiff] Building .venv from !BASEPY! ...
    "!BASEPY!" -m venv "%VENV%"
)
if not exist "%VPY%" (
    echo [ERR] Could not create .venv in %ROOT% - is the folder writable, and is
    echo       there disk space? Python's own message is above.
    goto :fail
)

REM ---- 3. PyTorch, then the toolkit ---------------------------------------------
echo [atkdiff] Updating pip inside .venv ...
"%VPY%" -m pip install --upgrade pip
echo [atkdiff] Installing PyTorch ^(torch, torchvision, torchaudio^) for %TORCH_WORDS%
echo [atkdiff] from %TORCH_INDEX% - a large download ^(about 3 GB for CUDA^).
"%VPY%" -m pip install torch torchvision torchaudio --index-url %TORCH_INDEX%
if errorlevel 1 (
    echo [ERR] PyTorch did not install - the pip messages above say why. A network
    echo       is needed for this step. Run install.bat again when it is fixed;
    echo       what already arrived is reused.
    goto :fail
)
echo [atkdiff] Installing the toolkit itself ^(editable^) with its training extras:
echo [atkdiff] TorchSig 2.2.0, ONNX, ONNX Runtime, tifffile, itmlogic ...
pushd "%ROOT%"
"%VPY%" -m pip install -e ".[train]"
set "PIPRC=!errorlevel!"
popd
if not "!PIPRC!"=="0" (
    echo [ERR] The toolkit's packages did not install - the pip messages above say
    echo       why. Run install.bat again when it is fixed.
    goto :fail
)

REM ---- 4. does it work? -----------------------------------------------------------
set "_OUT=%ROOT%.install-probe.out"
set "_ERR=%ROOT%.install-probe.err"
del /q "%_OUT%" "%_ERR%" 2>nul
"%VPY%" -c "import torch, torchsig, atk_diffusion; from importlib.metadata import version; c = torch.cuda.is_available(); print('ATKDIFF_TORCH=' + torch.__version__); print('ATKDIFF_TS=' + version('torchsig')); print('ATKDIFF_CUDA=' + ('yes ' + torch.cuda.get_device_name(0) if c else 'no'))" > "%_OUT%" 2> "%_ERR%"
set "T_VER="
set "TS_VER="
set "CUDA="
if exist "%_OUT%" for /f "usebackq tokens=1,* delims==" %%A in ("%_OUT%") do (
    if /i "%%A"=="ATKDIFF_TORCH" set "T_VER=%%B"
    if /i "%%A"=="ATKDIFF_TS" set "TS_VER=%%B"
    if /i "%%A"=="ATKDIFF_CUDA" set "CUDA=%%B"
)
if not defined T_VER (
    echo [ERR] torch, torchsig and atk_diffusion did not all import in .venv. It said:
    if exist "%_ERR%" type "%_ERR%"
    del /q "%_OUT%" "%_ERR%" 2>nul
    goto :fail
)
del /q "%_OUT%" "%_ERR%" 2>nul
echo [OK] PyTorch !T_VER!, TorchSig !TS_VER! and the toolkit import in .venv.
if /i "!CUDA:~0,3!"=="yes" (
    echo [OK] CUDA is available: training runs on the GPU ^(!CUDA:~4!^).
) else (
    if defined WANT_CPU (
        echo [--] CUDA is not used ^(you asked for /cpu^): training runs on the CPU -
        echo      correct, but slow for the larger models.
    ) else (
        echo [--] CUDA is NOT available: PyTorch will train on the CPU only - slow.
        echo      Check the NVIDIA driver ^(nvidia-smi should list the GPU^), then run
        echo      install.bat again.
    )
)
echo(
echo [atkdiff] Done. Next:  atkdiff.bat status
echo [atkdiff] The runbook is docs\FIRST_EXPERIMENTS.md.
if defined ATKDIFF_DCLICK pause
endlocal
exit /b 0

:fail
if defined ATKDIFF_DCLICK pause
endlocal
exit /b 1

REM ===========================================================================
REM  :try_python <exe> <words> - accept it only if it answers, with a
REM  sentinel, that it IS 3.11 (ATK's lesson: the py launcher prints its
REM  version list and exits 0 when the asked-for version is missing, and a
REM  quoted exe as the first token of a for /f command loses its quotes - so
REM  the interpreter is run directly, its answer written to a file here).
REM ===========================================================================
:try_python
if defined BASEPY exit /b 0
if not exist "%~1" exit /b 0
set "_PO=%ROOT%.python-probe.out"
del /q "%_PO%" 2>nul
"%~1" -c "import sys; print('ATKPY=' + sys.executable) if sys.version_info[:2] == (3, 11) else None" > "%_PO%" 2>nul
set "_CAND="
if exist "%_PO%" for /f "usebackq tokens=1,* delims==" %%A in ("%_PO%") do (
    if /i "%%A"=="ATKPY" set "_CAND=%%B"
)
del /q "%_PO%" 2>nul
if not defined _CAND exit /b 0
if not exist "!_CAND!" exit /b 0
set "BASEPY=!_CAND!"
set "BASEFROM=%~2"
exit /b 0

:try_launcher
if defined BASEPY exit /b 0
where py >nul 2>&1
if errorlevel 1 exit /b 0
set "_PO=%ROOT%.python-probe.out"
del /q "%_PO%" 2>nul
py -3.11 -c "import sys; print('ATKPY=' + sys.executable) if sys.version_info[:2] == (3, 11) else None" > "%_PO%" 2>nul
set "_CAND="
if exist "%_PO%" for /f "usebackq tokens=1,* delims==" %%A in ("%_PO%") do (
    if /i "%%A"=="ATKPY" set "_CAND=%%B"
)
del /q "%_PO%" 2>nul
if not defined _CAND exit /b 0
if not exist "!_CAND!" exit /b 0
set "BASEPY=!_CAND!"
set "BASEFROM=the py launcher, py -3.11"
exit /b 0

REM  :venv_ok - is .venv a 3.11 environment? Sets VENV_OK, or VENV_WHY.
:venv_ok
set "VENV_OK="
set "VENV_WHY=it did not answer"
set "_PO=%ROOT%.python-probe.out"
del /q "%_PO%" 2>nul
"%VPY%" -c "import sys; print('ATKPY=' + ('ok' if sys.version_info[:2] == (3, 11) else 'Python %%d.%%d' %% sys.version_info[:2]))" > "%_PO%" 2>nul
if exist "%_PO%" for /f "usebackq tokens=1,* delims==" %%A in ("%_PO%") do (
    if /i "%%A"=="ATKPY" (
        if /i "%%B"=="ok" (set "VENV_OK=1") else (set "VENV_WHY=it is %%B")
    )
)
del /q "%_PO%" 2>nul
exit /b 0
