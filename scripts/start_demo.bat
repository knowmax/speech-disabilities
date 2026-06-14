@echo off
setlocal

set PROJECT_ROOT=%~dp0..
set PYTHON_EXE=c:/Users/GC/Downloads/speech-disabilities/.venv/Scripts/python.exe

if not exist "%PYTHON_EXE%" (
  echo Python executable not found at %PYTHON_EXE%
  exit /b 1
)

echo Starting backend on http://localhost:8000 ...
start "Speech Demo Backend" powershell -NoExit -Command "Set-Location '%PROJECT_ROOT%\backend'; %PYTHON_EXE% app.py"

echo Starting frontend on http://localhost:5173 ...
start "Speech Demo Frontend" powershell -NoExit -Command "Set-Location '%PROJECT_ROOT%'; %PYTHON_EXE% -m http.server 5173 -d frontend"

start "" http://localhost:5173

echo Demo stack launched.
endlocal
