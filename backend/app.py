"""
FastAPI backend for AI Speech Disability Interpreter
Handles audio upload, ASR, correction, and personalization
"""

from fastapi import FastAPI, File, UploadFile, WebSocket, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

# Import custom modules
from models.asr_engine import WhisperASR
from models.correction_model import CorrectionModel
from models.personalization import PersonalizationEngine
from models.tts_engine import TTSEngine
from utils.audio_preprocessing import AudioPreprocessor
from utils.feature_extraction import AudioFeatures

# Initialize FastAPI
app = FastAPI(
    title="AI Speech Disability Interpreter",
    description="Real-time speech interpretation for dysarthric and impaired speech",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models (lazy loading for faster startup)
asr_engine = None
correction_model = None
personalization_engine = None
tts_engine = None
audio_preprocessor = None

# Data directories
DATA_DIR = Path("data")
AUDIO_DIR = DATA_DIR / "audio_uploads"
CORRECTIONS_DIR = DATA_DIR / "corrections"
USER_PROFILES_DIR = DATA_DIR / "user_profiles"

# Create directories
for dir_path in [DATA_DIR, AUDIO_DIR, CORRECTIONS_DIR, USER_PROFILES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# Models
class TranscriptionRequest(BaseModel):
    user_id: Optional[str] = "default"
    language: str = "en"
    enable_correction: bool = True
    enable_personalization: bool = True


class CorrectionFeedback(BaseModel):
    user_id: str
    original_text: str
    corrected_text: str
    user_corrected_text: str
    timestamp: str


class TranscriptionResponse(BaseModel):
    original_text: str
    corrected_text: str
    confidence: float
    processing_time: float
    word_error_rate: Optional[float] = None
    features: Optional[dict] = None
    corrected_audio_url: Optional[str] = None  # URL to download corrected audio


# Startup event - initialize models
@app.on_event("startup")
async def startup_event():
    global asr_engine, correction_model, personalization_engine, tts_engine, audio_preprocessor
    
    print("🚀 Initializing AI Speech Interpreter...")
    print("📦 Loading models (this may take a minute)...")
    
    # Initialize components
    audio_preprocessor = AudioPreprocessor()
    asr_engine = WhisperASR(model_size="base")  # Can use "small", "medium" for better accuracy
    correction_model = CorrectionModel()
    personalization_engine = PersonalizationEngine(storage_path=USER_PROFILES_DIR)
    tts_engine = TTSEngine()
    
    print("✅ All models loaded successfully!")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "AI Speech Disability Interpreter",
        "version": "1.0.0",
        "models_loaded": asr_engine is not None
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "components": {
            "asr": asr_engine is not None,
            "correction": correction_model is not None,
            "personalization": personalization_engine is not None,
            "tts": tts_engine is not None
        }
    }


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    user_id: str = "default",
    enable_correction: bool = True,
    enable_personalization: bool = True
):
    """
    Upload audio file and get transcription with optional correction
    """
    import time
    start_time = time.time()
    
    try:
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = AUDIO_DIR / f"{user_id}_{timestamp}_{file.filename}"
        
        with open(audio_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Preprocess audio
        preprocessed_audio = audio_preprocessor.preprocess(str(audio_path))
        
        # Extract features
        audio_features = AudioFeatures.extract(preprocessed_audio)
        
        # Run ASR
        asr_result = asr_engine.transcribe(preprocessed_audio)
        original_text = asr_result["text"]
        confidence = asr_result.get("confidence", 0.0)
        
        # Apply correction if enabled
        corrected_text = original_text
        if enable_correction and original_text:
            correction_result = correction_model.correct(
                text=original_text,
                audio_features=audio_features,
                confidence=confidence
            )
            corrected_text = correction_result["corrected_text"]
            confidence = correction_result.get("confidence", confidence)
        
        # Apply personalization if enabled
        if enable_personalization and user_id != "default":
            corrected_text = personalization_engine.apply_user_corrections(
                user_id=user_id,
                text=corrected_text
            )
        
        # Generate corrected audio (normal speech from corrected text)
        corrected_audio_path = None
        if corrected_text and corrected_text != original_text:
            corrected_audio_filename = f"{user_id}_{timestamp}_corrected.wav"
            corrected_audio_path = AUDIO_DIR / corrected_audio_filename
            tts_engine.synthesize(
                text=corrected_text,
                output_path=str(corrected_audio_path)
            )
            corrected_audio_url = f"/audio/{corrected_audio_filename}"
        else:
            corrected_audio_url = None
        
        processing_time = time.time() - start_time
        
        # Calculate WER if we have ground truth (for demo purposes)
        wer = None
        
        return TranscriptionResponse(
            original_text=original_text,
            corrected_text=corrected_text,
            confidence=confidence,
            processing_time=processing_time,
            word_error_rate=wer,
            features={
                "speaking_rate": audio_features.get("speaking_rate"),
                "pitch_mean": audio_features.get("pitch_mean"),
                "energy": audio_features.get("energy")
            },
            corrected_audio_url=corrected_audio_url
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/feedback")
async def submit_feedback(feedback: CorrectionFeedback):
    """
    Submit user corrections to improve personalization
    """
    try:
        # Store feedback
        personalization_engine.add_correction(
            user_id=feedback.user_id,
            original=feedback.original_text,
            corrected_by_model=feedback.corrected_text,
            corrected_by_user=feedback.user_corrected_text
        )
        
        # Save to file for training
        feedback_path = CORRECTIONS_DIR / f"{feedback.user_id}_corrections.jsonl"
        with open(feedback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback.dict()) + "\n")
        
        return {"status": "success", "message": "Feedback recorded"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")


@app.get("/user/{user_id}/stats")
async def get_user_stats(user_id: str):
    """
    Get personalization statistics for a user
    """
    try:
        stats = personalization_engine.get_user_stats(user_id)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve stats: {str(e)}")


@app.get("/audio/{filename}")
async def download_audio(filename: str):
    """
    Download generated audio files
    """
    file_path = AUDIO_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        str(file_path),
        media_type="audio/wav",
        filename=filename
    )


@app.post("/synthesize")
async def synthesize_speech(
    text: str,
    user_id: str = "default",
    voice: str = "neutral"
):
    """
    Convert corrected text to speech
    """
    try:
        audio_path = tts_engine.synthesize(
            text=text,
            output_path=AUDIO_DIR / f"tts_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
            voice=voice
        )
        
        return FileResponse(
            audio_path,
            media_type="audio/wav",
            filename="synthesized_speech.wav"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")


@app.post("/batch-process")
async def batch_process():
    """
    Batch process all WAV files in audio_uploads folder
    Generates corrected speech in output folder (1-to-1 mapping)
    """
    import time
    
    try:
        # Output directory
        OUTPUT_DIR = DATA_DIR / "output"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Find all WAV files in audio_uploads
        wav_files = list(AUDIO_DIR.glob("*.wav"))
        
        if not wav_files:
            return {
                "status": "no_files",
                "message": "No WAV files found in audio_uploads folder",
                "processed": 0
            }
        
        results = []
        
        for input_path in wav_files:
            start_time = time.time()
            
            try:
                # 1. Preprocess audio
                preprocessed_audio = audio_preprocessor.preprocess(str(input_path))
                
                # 2. Extract features
                audio_features = AudioFeatures.extract(preprocessed_audio)
                
                # 3. Transcribe (ASR)
                asr_result = asr_engine.transcribe(preprocessed_audio)
                original_text = asr_result["text"]
                confidence = asr_result.get("confidence", 0.0)
                
                # 4. Correct text
                correction_result = correction_model.correct(
                    text=original_text,
                    audio_features=audio_features,
                    confidence=confidence
                )
                corrected_text = correction_result["corrected_text"]
                
                # 5. Generate corrected speech
                output_filename = input_path.stem + "_corrected.wav"
                output_path = OUTPUT_DIR / output_filename
                
                tts_engine.synthesize(
                    text=corrected_text,
                    output_path=str(output_path)
                )
                
                processing_time = time.time() - start_time
                
                results.append({
                    "status": "success",
                    "input_file": input_path.name,
                    "output_file": output_filename,
                    "original_text": original_text,
                    "corrected_text": corrected_text,
                    "confidence": confidence,
                    "processing_time": processing_time,
                    "output_url": f"/output/{output_filename}"
                })
                
            except Exception as e:
                results.append({
                    "status": "error",
                    "input_file": input_path.name,
                    "error": str(e)
                })
        
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "error"]
        
        return {
            "status": "completed",
            "total_files": len(wav_files),
            "successful": len(successful),
            "failed": len(failed),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


@app.get("/output/{filename}")
async def download_output(filename: str):
    """
    Download corrected audio files from output folder
    """
    OUTPUT_DIR = DATA_DIR / "output"
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        str(file_path),
        media_type="audio/wav",
        filename=filename
    )


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming transcription
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive audio chunk
            data = await websocket.receive_bytes()
            
            # Process in real-time (simplified - in production use streaming ASR)
            # For MVP, we'll buffer and process chunks
            # This is a placeholder for streaming implementation
            
            response = {
                "status": "processing",
                "message": "Streaming ASR coming in next iteration"
            }
            
            await websocket.send_json(response)
            
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()


if __name__ == "__main__":
    print("=" * 60)
    print("🎤 AI Speech Disability Interpreter - Backend Server")
    print("=" * 60)
    print("\n📍 Server starting on http://localhost:8000")
    print("📚 API docs available at http://localhost:8000/docs")
    print("\n⚠️  First run will download Whisper model (~150MB)\n")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
