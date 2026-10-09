@echo off
REM ============================================================================
REM  atkdiff.bat <command> ...  - the ATK Diffusion Toolkit's command line.
REM  It runs exactly:  <the toolkit's Python> -m atk_diffusion <command> ...
REM
REM      atkdiff.bat status
REM      atkdiff.bat profile new rtlsdr_2400000_cu8
REM      atkdiff.bat --help
REM
REM  WHICH PYTHON, in order (never the python on PATH):
REM    1. ..\ATK\envs\atk_diffusion\Scripts\python.exe  (built by ATK's
REM       get_diffusion.bat - the normal case beside ATK)
REM    2. .venv\Scripts\python.exe in this folder  (built by install.bat)
REM  The toolkit code that runs is the one in THIS folder.
REM  Exit code: 0 done, 1 refused or failed (it says why), 2 a wrong command.
REM ============================================================================
setlocal enableextensions

set "ROOT=%~dp0"
set "ATKDIFF_DCLICK="
echo %cmdcmdline% | find /i "%~nx0" >nul 2>&1 && set "ATKDIFF_DCLICK=1"

set "PY="
if exist "%ROOT%..\ATK\envs\atk_diffusion\Scripts\python.exe" set "PY=%ROOT%..\ATK\envs\atk_diffusion\Scripts\python.exe"
if not defined PY if exist "%ROOT%.venv\Scripts\python.exe" set "PY=%ROOT%.venv\Scripts\python.exe"
if not defined PY (
    echo [ERR] No toolkit environment was found, so nothing was run.
    echo       Beside ATK:  run ATK's get_diffusion.bat - it builds
    echo                    ..\ATK\envs\atk_diffusion.
    echo       On its own:  run install.bat in this folder - it builds .venv.
    if defined ATKDIFF_DCLICK pause
    endlocal
    exit /b 1
)

if not exist "%ROOT%atk_diffusion\__init__.py" (
    echo [ERR] %ROOT% has no atk_diffusion\ package - atkdiff.bat must stay in
    echo       the toolkit's folder.
    if defined ATKDIFF_DCLICK pause
    endlocal
    exit /b 1
)

REM  This folder's code first; UTF-8 so a sentence never stops a run.
if defined PYTHONPATH (
    set "PYTHONPATH=%ROOT%;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%ROOT%"
)
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONNOUSERSITE=1"

"%PY%" -m atk_diffusion %*
set "RC=%ERRORLEVEL%"
if defined ATKDIFF_DCLICK pause
endlocal & exit /b %RC%
