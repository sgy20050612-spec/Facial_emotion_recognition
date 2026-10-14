@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=D:\fer_experiments\venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo Python environment not found: %PYTHON%
  pause
  exit /b 1
)
"%PYTHON%" -X utf8 -m realtime_fer.app
if errorlevel 1 (
  echo Application failed. Please copy the error above.
  pause
  exit /b 1
)
endlocal
