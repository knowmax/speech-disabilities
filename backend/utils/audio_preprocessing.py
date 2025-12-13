"""
Audio Preprocessing Pipeline
Noise reduction, normalization, VAD, etc.
"""

import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import noisereduce as nr
from typing import Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


class AudioPreprocessor:
    """
    Preprocess audio for better ASR accuracy
    """
    
    def __init__(
        self,
        target_sr: int = 16000,
        apply_noise_reduction: bool = True,
        normalize: bool = True
    ):
        """
        Initialize audio preprocessor
        
        Args:
            target_sr: Target sample rate (Whisper uses 16kHz)
            apply_noise_reduction: Apply noise reduction
            normalize: Normalize audio amplitude
        """
        self.target_sr = target_sr
        self.apply_noise_reduction = apply_noise_reduction
        self.normalize = normalize
        
        print("✅ Audio preprocessor initialized")
    
    def preprocess(
        self,
        audio_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Preprocess audio file
        
        Args:
            audio_path: Input audio file
            output_path: Where to save processed audio (optional)
            
        Returns:
            Path to processed audio
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Resample to target sample rate
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr
        
        # Apply noise reduction
        if self.apply_noise_reduction:
            audio = self._reduce_noise(audio, sr)
        
        # Normalize
        if self.normalize:
            audio = self._normalize_audio(audio)
        
        # Trim silence
        audio = self._trim_silence(audio, sr)
        
        # Save processed audio in temp directory
        if output_path is None:
            temp_dir = Path("data/temp")
            temp_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(temp_dir / f"{Path(audio_path).stem}_processed.wav")
        
        sf.write(output_path, audio, sr)
        
        return output_path
    
    def _reduce_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply noise reduction using spectral gating
        """
        try:
            # Use first 0.5 seconds as noise profile (or adapt based on VAD)
            noise_sample_length = min(int(0.5 * sr), len(audio) // 4)
            
            reduced = nr.reduce_noise(
                y=audio,
                sr=sr,
                stationary=True,
                prop_decrease=0.8
            )
            return reduced
        except Exception as e:
            print(f"⚠️  Noise reduction failed: {e}, using original audio")
            return audio
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range
        """
        max_val = np.abs(audio).max()
        if max_val > 0:
            return audio / max_val
        return audio
    
    def _trim_silence(
        self,
        audio: np.ndarray,
        sr: int,
        top_db: int = 30
    ) -> np.ndarray:
        """
        Trim leading and trailing silence
        """
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed
    
    def extract_segments(
        self,
        audio_path: str,
        min_segment_length: float = 0.5,
        max_segment_length: float = 30.0
    ) -> list:
        """
        Split audio into speech segments using VAD
        
        Args:
            audio_path: Input audio file
            min_segment_length: Minimum segment duration (seconds)
            max_segment_length: Maximum segment duration (seconds)
            
        Returns:
            List of (start_time, end_time, audio_segment) tuples
        """
        audio, sr = librosa.load(audio_path, sr=self.target_sr)
        
        # Simple energy-based VAD for MVP
        # In production, use webrtcvad or similar
        intervals = librosa.effects.split(audio, top_db=30)
        
        segments = []
        for start_idx, end_idx in intervals:
            start_time = start_idx / sr
            end_time = end_idx / sr
            duration = end_time - start_time
            
            # Filter by length
            if min_segment_length <= duration <= max_segment_length:
                segment_audio = audio[start_idx:end_idx]
                segments.append((start_time, end_time, segment_audio))
        
        return segments
    
    def get_audio_info(self, audio_path: str) -> dict:
        """
        Get audio file information
        """
        audio, sr = librosa.load(audio_path, sr=None)
        
        return {
            "duration": len(audio) / sr,
            "sample_rate": sr,
            "channels": 1,  # librosa loads as mono
            "samples": len(audio),
            "rms_energy": float(np.sqrt(np.mean(audio**2)))
        }
