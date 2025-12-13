"""
Audio Feature Extraction
Extract features relevant to dysarthric speech analysis
"""

import librosa
import numpy as np
from typing import Dict, Optional
import warnings
warnings.filterwarnings("ignore")


class AudioFeatures:
    """
    Extract acoustic features for speech impairment analysis
    """
    
    @staticmethod
    def extract(audio_path: str, sr: int = 16000) -> Dict:
        """
        Extract comprehensive audio features
        
        Args:
            audio_path: Path to audio file
            sr: Sample rate
            
        Returns:
            Dictionary of features
        """
        # Load audio
        if isinstance(audio_path, str):
            audio, sr = librosa.load(audio_path, sr=sr)
        else:
            # Already loaded audio array
            audio = audio_path
        
        features = {}
        
        # Duration
        features["duration"] = len(audio) / sr
        
        # Speaking rate (syllables per second - approximated by peaks)
        features["speaking_rate"] = AudioFeatures._estimate_speaking_rate(audio, sr)
        
        # Pitch features
        pitch_features = AudioFeatures._extract_pitch_features(audio, sr)
        features.update(pitch_features)
        
        # Energy features
        energy_features = AudioFeatures._extract_energy_features(audio)
        features.update(energy_features)
        
        # Spectral features
        spectral_features = AudioFeatures._extract_spectral_features(audio, sr)
        features.update(spectral_features)
        
        # Voice quality indicators
        quality_features = AudioFeatures._extract_voice_quality(audio, sr)
        features.update(quality_features)
        
        # Pause statistics
        pause_features = AudioFeatures._extract_pause_features(audio, sr)
        features.update(pause_features)
        
        return features
    
    @staticmethod
    def _estimate_speaking_rate(audio: np.ndarray, sr: int) -> float:
        """
        Estimate speaking rate (syllables/phonemes per second)
        Using onset detection as proxy
        """
        try:
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            onsets = librosa.onset.onset_detect(
                onset_envelope=onset_env,
                sr=sr,
                backtrack=True
            )
            
            duration = len(audio) / sr
            if duration > 0:
                return len(onsets) / duration
            return 0.0
        except:
            return 0.0
    
    @staticmethod
    def _extract_pitch_features(audio: np.ndarray, sr: int) -> Dict:
        """
        Extract pitch (F0) related features
        """
        try:
            # Extract pitch using piptrack
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            
            # Get pitch values
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if len(pitch_values) > 0:
                pitch_values = np.array(pitch_values)
                return {
                    "pitch_mean": float(np.mean(pitch_values)),
                    "pitch_std": float(np.std(pitch_values)),
                    "pitch_min": float(np.min(pitch_values)),
                    "pitch_max": float(np.max(pitch_values)),
                    "pitch_range": float(np.max(pitch_values) - np.min(pitch_values))
                }
            else:
                return {
                    "pitch_mean": 0.0,
                    "pitch_std": 0.0,
                    "pitch_min": 0.0,
                    "pitch_max": 0.0,
                    "pitch_range": 0.0
                }
        except:
            return {
                "pitch_mean": 0.0,
                "pitch_std": 0.0,
                "pitch_min": 0.0,
                "pitch_max": 0.0,
                "pitch_range": 0.0
            }
    
    @staticmethod
    def _extract_energy_features(audio: np.ndarray) -> Dict:
        """
        Extract energy/amplitude features
        """
        rms = librosa.feature.rms(y=audio)[0]
        
        return {
            "energy_mean": float(np.mean(rms)),
            "energy_std": float(np.std(rms)),
            "energy_max": float(np.max(rms)),
            "energy_min": float(np.min(rms))
        }
    
    @staticmethod
    def _extract_spectral_features(audio: np.ndarray, sr: int) -> Dict:
        """
        Extract spectral features
        """
        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        
        # Zero crossing rate (noise indicator)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        
        return {
            "spectral_centroid_mean": float(np.mean(centroid)),
            "spectral_rolloff_mean": float(np.mean(rolloff)),
            "zero_crossing_rate": float(np.mean(zcr))
        }
    
    @staticmethod
    def _extract_voice_quality(audio: np.ndarray, sr: int) -> Dict:
        """
        Extract voice quality indicators (jitter, shimmer approximations)
        """
        # Simplified approximations for MVP
        # In production, use specialized tools like Praat
        
        # Jitter (pitch variation) - approximated by pitch std
        pitches, _ = librosa.piptrack(y=audio, sr=sr)
        pitch_variation = float(np.std(pitches[pitches > 0])) if np.any(pitches > 0) else 0.0
        
        # Shimmer (amplitude variation)
        rms = librosa.feature.rms(y=audio)[0]
        amplitude_variation = float(np.std(rms)) / (float(np.mean(rms)) + 1e-8)
        
        return {
            "pitch_variation": pitch_variation,
            "amplitude_variation": amplitude_variation
        }
    
    @staticmethod
    def _extract_pause_features(audio: np.ndarray, sr: int, threshold_db: int = 30) -> Dict:
        """
        Extract pause/silence statistics
        """
        # Detect non-silent intervals
        intervals = librosa.effects.split(audio, top_db=threshold_db)
        
        # Calculate pause durations
        pause_durations = []
        for i in range(len(intervals) - 1):
            pause_start = intervals[i][1] / sr
            pause_end = intervals[i + 1][0] / sr
            pause_duration = pause_end - pause_start
            if pause_duration > 0.1:  # Only count pauses > 100ms
                pause_durations.append(pause_duration)
        
        if len(pause_durations) > 0:
            return {
                "pause_count": len(pause_durations),
                "pause_mean": float(np.mean(pause_durations)),
                "pause_total": float(np.sum(pause_durations))
            }
        else:
            return {
                "pause_count": 0,
                "pause_mean": 0.0,
                "pause_total": 0.0
            }
    
    @staticmethod
    def extract_mfcc(audio: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
        """
        Extract MFCC features (for ML models)
        """
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return mfcc
