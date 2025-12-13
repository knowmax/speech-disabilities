"""
Text-to-Speech Engine
Synthesizes clear speech from corrected text
"""

from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

try:
    from TTS.api import TTS
    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False
    print("⚠️  Coqui TTS not available, falling back to pyttsx3")

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("⚠️  pyttsx3 not available")


class TTSEngine:
    """
    Text-to-Speech synthesis for corrected text
    """
    
    def __init__(self, engine: str = "auto", model: str = "tts_models/en/ljspeech/tacotron2-DDC"):
        """
        Initialize TTS engine
        
        Args:
            engine: "coqui", "pyttsx3", or "auto" (auto-select best available)
            model: Coqui TTS model name
        """
        self.engine_type = engine
        
        if engine == "auto":
            if COQUI_AVAILABLE:
                self.engine_type = "coqui"
            elif PYTTSX3_AVAILABLE:
                self.engine_type = "pyttsx3"
            else:
                self.engine_type = "none"
                print("❌ No TTS engine available")
                return
        
        # Initialize engine
        if self.engine_type == "coqui" and COQUI_AVAILABLE:
            self._init_coqui(model)
        elif self.engine_type == "pyttsx3" and PYTTSX3_AVAILABLE:
            self._init_pyttsx3()
        else:
            print(f"❌ TTS engine '{engine}' not available")
            self.engine_type = "none"
    
    def _init_coqui(self, model: str):
        """
        Initialize Coqui TTS (high quality, slower)
        """
        try:
            print(f"Loading Coqui TTS model: {model}")
            self.tts = TTS(model_name=model, progress_bar=False)
            print("✅ Coqui TTS loaded")
        except Exception as e:
            print(f"❌ Failed to load Coqui TTS: {e}")
            # Fallback to pyttsx3
            if PYTTSX3_AVAILABLE:
                self.engine_type = "pyttsx3"
                self._init_pyttsx3()
            else:
                self.engine_type = "none"
    
    def _init_pyttsx3(self):
        """
        Initialize pyttsx3 (faster, offline, lower quality)
        """
        try:
            self.tts = pyttsx3.init()
            
            # Configure voice
            voices = self.tts.getProperty('voices')
            if voices:
                # Try to use a clear, neutral voice
                self.tts.setProperty('voice', voices[0].id)
            
            # Set speech rate and volume
            self.tts.setProperty('rate', 150)  # Speed (default ~200)
            self.tts.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
            
            print("✅ pyttsx3 TTS initialized")
        except Exception as e:
            print(f"❌ Failed to initialize pyttsx3: {e}")
            self.engine_type = "none"
    
    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice: str = "neutral",
        speed: float = 1.0
    ) -> Optional[Path]:
        """
        Synthesize speech from text
        
        Args:
            text: Text to synthesize
            output_path: Where to save audio
            voice: Voice selection (model-specific)
            speed: Speech rate multiplier
            
        Returns:
            Path to generated audio file, or None if failed
        """
        if not text or not text.strip():
            print("⚠️  Empty text, skipping TTS")
            return None
        
        if self.engine_type == "none":
            print("⚠️  No TTS engine available")
            return None
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.engine_type == "coqui":
                return self._synthesize_coqui(text, output_path, speed)
            elif self.engine_type == "pyttsx3":
                return self._synthesize_pyttsx3(text, output_path, speed)
            
        except Exception as e:
            print(f"❌ TTS synthesis failed: {e}")
            return None
    
    def _synthesize_coqui(self, text: str, output_path: Path, speed: float) -> Path:
        """
        Synthesize with Coqui TTS
        """
        self.tts.tts_to_file(
            text=text,
            file_path=str(output_path),
            speed=speed
        )
        return output_path
    
    def _synthesize_pyttsx3(self, text: str, output_path: Path, speed: float) -> Path:
        """
        Synthesize with pyttsx3
        """
        # Adjust rate based on speed
        base_rate = 150
        self.tts.setProperty('rate', int(base_rate * speed))
        
        # Save to file
        self.tts.save_to_file(text, str(output_path))
        self.tts.runAndWait()
        
        return output_path
    
    def get_available_voices(self) -> list:
        """
        Get list of available voices
        """
        if self.engine_type == "pyttsx3":
            voices = self.tts.getProperty('voices')
            return [{"id": v.id, "name": v.name, "languages": v.languages} for v in voices]
        elif self.engine_type == "coqui":
            return ["Default Coqui model voice"]
        else:
            return []
