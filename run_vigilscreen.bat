@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if exist ".venv\Scripts\pythonw.exe" (
  start "VigilScreen" ".venv\Scripts\pythonw.exe" "vigil_app.py"
) else (
  py -3.11 vigil_app.py
  if errorlevel 1 pause
)
