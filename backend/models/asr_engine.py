"""
Whisper ASR Engine Wrapper
Handles speech-to-text conversion with confidence scoring
"""

import whisper
import torch
import numpy as np
import soundfile as sf
import time
from typing import Dict, Optional
import warnings
warnings.filterwarnings("ignore")


class WhisperASR:
    """
    OpenAI Whisper ASR engine with dysarthric speech support
    """
    
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        """
        Initialize Whisper model
        
        Args:
            model_size: Model size - "tiny", "base", "small", "medium", "large"
            device: "cuda" or "cpu" (auto-detect if None)
        """
        self.model_size = model_size
        self.runtime_profile = self._build_runtime_profile()

        # Auto-detect device. ROCm uses the "cuda" device string in PyTorch.
        if device is None:
            self.device = "cuda" if self.runtime_profile["gpu_available"] else "cpu"
        else:
            self.device = device

        self.use_fp16 = self.device == "cuda"
        
        print(f"Loading Whisper model '{model_size}' on {self.device}...")
        self.model = whisper.load_model(model_size, device=self.device)
        print(f"✅ Whisper {model_size} loaded successfully")

    def _build_runtime_profile(self) -> Dict:
        """
        Build hardware/runtime information for observability and demos.
        """
        gpu_available = torch.cuda.is_available()
        gpu_backend = "cpu"
        gpu_name = "cpu"

        if gpu_available:
            if getattr(torch.version, "hip", None):
                gpu_backend = "rocm"
            else:
                gpu_backend = "cuda"

            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = f"{gpu_backend}-gpu"

        return {
            "gpu_available": gpu_available,
            "gpu_backend": gpu_backend,
            "gpu_name": gpu_name,
            "torch_version": torch.__version__
        }

    def get_runtime_profile(self) -> Dict:
        return {
            "model": self.model_size,
            "device": self.device,
            "fp16_enabled": self.use_fp16,
            **self.runtime_profile
        }
        
    def transcribe(
        self,
        audio_path: str,
        language: str = "en",
        task: str = "transcribe",
        **kwargs
    ) -> Dict:
        """
        Transcribe audio file
        
        Args:
            audio_path: Path to audio file
            language: Language code (en, es, fr, etc.)
            task: "transcribe" or "translate"
            
        Returns:
            Dict with 'text', 'segments', 'language', 'confidence'
        """
        try:
            start = time.perf_counter()

            # Load audio with soundfile (no ffmpeg needed)
            audio, sr = sf.read(audio_path)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Resample to 16kHz if needed (Whisper expects 16kHz)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
            # Ensure float32
            audio = audio.astype(np.float32)

            decode_settings = {
                "language": language,
                "task": task,
                "fp16": self.use_fp16,
                "word_timestamps": True
            }
            decode_settings.update(kwargs)
            
            # Transcribe with Whisper (using numpy array instead of file path)
            result = self.model.transcribe(audio, **decode_settings)
            
            # Calculate average confidence from log probabilities
            confidence = self._calculate_confidence(result)
            
            return {
                "text": result["text"].strip(),
                "segments": result.get("segments", []),
                "language": result.get("language", language),
                "confidence": confidence,
                "word_timestamps": self._extract_word_timestamps(result),
                "runtime": {
                    "device": self.device,
                    "gpu_backend": self.runtime_profile["gpu_backend"],
                    "latency_ms": round((time.perf_counter() - start) * 1000, 2)
                }
            }
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "text": "",
                "segments": [],
                "language": language,
                "confidence": 0.0,
                "word_timestamps": []
            }
    
    def _calculate_confidence(self, result: Dict) -> float:
        """
        Calculate average confidence from segments
        """
        segments = result.get("segments", [])
        if not segments:
            return 0.0
        
        # Average of no_speech_prob (inverted) as proxy for confidence
        confidences = []
        for segment in segments:
            no_speech_prob = segment.get("no_speech_prob", 0.5)
            # Higher no_speech_prob = lower confidence
            confidence = 1.0 - no_speech_prob
            confidences.append(confidence)
        
        return np.mean(confidences) if confidences else 0.0
    
    def _extract_word_timestamps(self, result: Dict) -> list:
        """
        Extract word-level timestamps if available
        """
        word_timestamps = []
        
        for segment in result.get("segments", []):
            if "words" in segment:
                for word_info in segment["words"]:
                    word_timestamps.append({
                        "word": word_info.get("word", "").strip(),
                        "start": word_info.get("start", 0.0),
                        "end": word_info.get("end", 0.0),
                        "probability": word_info.get("probability", 0.0)
                    })
        
        return word_timestamps
    
    def transcribe_with_alternatives(self, audio_path: str, beam_size: int = 5) -> Dict:
        """
        Get multiple transcription hypotheses (useful for dysarthric speech)
        """
        # Whisper doesn't directly expose n-best, but we can use higher beam size
        result = self.transcribe(
            audio_path,
            beam_size=beam_size,
            best_of=beam_size
        )
        
        return result
