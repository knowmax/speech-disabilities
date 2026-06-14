$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$backendDir = Join-Path $projectRoot "backend"
$pythonExe = "c:/Users/GC/Downloads/speech-disabilities/.venv/Scripts/python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Host "Python executable not found at $pythonExe" -ForegroundColor Red
    exit 1
}

Write-Host "Starting backend at http://localhost:8000 ..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$backendDir'; $pythonExe app.py"

Write-Host "Starting frontend static server at http://localhost:5173 ..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$projectRoot'; $pythonExe -m http.server 5173 -d frontend"

Start-Process "http://localhost:5173"

Write-Host "Demo stack launched. Use the new terminal windows to monitor logs." -ForegroundColor Green
