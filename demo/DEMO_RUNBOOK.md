# Competition Demo Runbook

## Generated Assets
- Video: `demo/competition_demo.mp4`
- Slides: `demo/competition_presentation.pptx`

## Demo Flow (3-5 min)
1. Open the PPT and briefly explain the problem and pipeline.
2. Show API runtime telemetry from `/health` and `/transcribe`.
3. Upload one impaired audio sample through `/transcribe`.
4. Compare original text vs corrected text.
5. Highlight `correction_metadata.strategy.mode` and `stage_timings.total_ms`.
6. Play corrected audio from `corrected_audio_url`.

## Talking Points
- The system is AMD-infra ready through GPU backend detection and runtime profiling.
- The correction engine adapts by confidence and speech characteristics.
- Judges can see both accessibility value and measurable system performance.

## Regenerate Assets
```powershell
c:/Users/GC/Downloads/speech-disabilities/.venv/Scripts/python.exe tools/generate_demo_video.py
c:/Users/GC/Downloads/speech-disabilities/.venv/Scripts/python.exe tools/generate_ppt.py
```
