@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM Double-click launcher for dicom-multi-view on Windows.
REM First run: creates .venv and pip install -e . (needs PyPI).
REM Later runs: reuse existing venv (no reinstall).
REM Force reinstall: set REINSTALL=1 before launching, e.g.
REM   set REINSTALL=1 && Run-DicomMultiView.bat
REM Do NOT copy .venv/venv between machines — recreate with this script.

REM Prefer py launcher, then python on PATH
set "BOOTSTRAP="
where py >nul 2>&1
if %errorlevel% equ 0 (
  set "BOOTSTRAP=py -3"
  goto :have_bootstrap
)
where python >nul 2>&1
if %errorlevel% equ 0 (
  set "BOOTSTRAP=python"
  goto :have_bootstrap
)

echo ERROR: Python was not found on PATH.
echo Install Python 3.11+ from https://www.python.org/downloads/
echo and enable "Add python.exe to PATH", then try again.
pause
exit /b 1

:have_bootstrap
REM Prefer .venv; reuse existing venv\ if that is what is already present
set "VENV_DIR=.venv"
if exist "venv\Scripts\python.exe" if not exist ".venv\Scripts\python.exe" (
  set "VENV_DIR=venv"
)

if exist "%VENV_DIR%\Scripts\python.exe" goto :venv_ready

echo Creating virtual environment in %VENV_DIR% ...
%BOOTSTRAP% -m venv "%VENV_DIR%"
if errorlevel 1 (
  echo ERROR: Failed to create virtual environment.
  pause
  exit /b 1
)

echo Upgrading pip ...
"%VENV_DIR%\Scripts\python.exe" -m pip install -U pip
if errorlevel 1 (
  echo ERROR: pip upgrade failed.
  pause
  exit /b 1
)

echo Installing project dependencies ^(pip install -e .^) ...
"%VENV_DIR%\Scripts\python.exe" -m pip install -e .
if errorlevel 1 (
  echo ERROR: Dependency install failed. First run needs network access to PyPI.
  pause
  exit /b 1
)
goto :launch

:venv_ready
if /i not "%REINSTALL%"=="1" goto :launch
echo REINSTALL=1 set: reinstalling project dependencies ...
"%VENV_DIR%\Scripts\python.exe" -m pip install -U pip
"%VENV_DIR%\Scripts\python.exe" -m pip install -e .
if errorlevel 1 (
  echo ERROR: Reinstall failed.
  pause
  exit /b 1
)

:launch
echo Launching dicom-multi-view ...
"%VENV_DIR%\Scripts\python.exe" -m dicom_multi_view %*
if errorlevel 1 (
  echo.
  echo dicom-multi-view exited with an error.
  pause
  exit /b 1
)

endlocal
exit /b 0
