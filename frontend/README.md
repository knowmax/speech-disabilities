# Frontend Demo Console

Lightweight, zero-build frontend for competition demos.

## Features
- Backend health and runtime check
- Audio upload and `/transcribe` call
- Original vs corrected text display
- Stage timing bars from `stage_timings`
- Runtime telemetry inspector (`gpu_backend`, device, strategy)
- Demo Script Mode (auto health -> transcribe -> batch -> user stats)
- Judge scorecard + radar chart (accuracy/speed/personalization/adaptivity/infra)
- Corrected audio playback
- Batch process trigger
- User personalization stats view

## One-Click Launch (Windows)
From project root:
```powershell
scripts\start_demo.bat
```

or

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1
```

## Run
1. Start backend (from `backend/`):
```powershell
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

2. Start frontend (from project root):
```powershell
c:/Users/GC/Downloads/speech-disabilities/.venv/Scripts/python.exe -m http.server 5173 -d frontend
```

3. Open:
- http://localhost:5173

## Demo tip
Use `demo_user` with multiple runs and then click **Get User Stats** to show personalization accumulation.
