"""
Text Correction Model
Uses transformer-based seq2seq to correct noisy ASR output from dysarthric speech
"""

from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from typing import Dict, Optional, List
import re


class CorrectionModel:
    """
    AI-powered text correction for dysarthric speech transcriptions
    Uses T5 or similar seq2seq model fine-tuned on error correction
    """
    
    def __init__(
        self,
        model_name: str = "t5-small",
        custom_model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize correction model
        
        Args:
            model_name: Hugging Face model name or path
            custom_model_path: Path to fine-tuned model (optional)
            device: cuda/cpu
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use custom model if provided, otherwise use base T5
        model_path = custom_model_path if custom_model_path else model_name
        
        print(f"Loading correction model: {model_path}...")
        
        try:
            # Try T5 first
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
            print(f"✅ T5 correction model loaded on {self.device}")
        except:
            # Fallback to Auto classes
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            print(f"✅ Correction model loaded on {self.device}")
        
        self.model.eval()
        
        # Common dysarthria-specific corrections (rule-based fallback)
        self.common_corrections = self._load_common_corrections()
        
    def correct(
        self,
        text: str,
        audio_features: Optional[Dict] = None,
        confidence: float = 1.0,
        max_length: int = 512
    ) -> Dict:
        """
        Correct transcribed text
        
        Args:
            text: Original ASR text
            audio_features: Optional audio features for context
            confidence: ASR confidence score
            max_length: Max output length
            
        Returns:
            Dict with corrected_text, confidence, corrections_made
        """
        if not text or not text.strip():
            return {
                "corrected_text": text,
                "confidence": 0.0,
                "corrections_made": []
            }
        
        # Apply rule-based corrections first (always apply, regardless of confidence)
        rule_corrected, rule_changes = self._apply_rule_based_corrections(text)
        
        # If confidence is very high and no rule changes, minimal correction
        if confidence > 0.95 and not rule_changes:
            corrected = self._apply_minimal_corrections(rule_corrected)
            return {
                "corrected_text": corrected,
                "confidence": confidence,
                "corrections_made": []
            }
        
        # Apply ML-based correction if confidence is lower
        if confidence < 0.95:
            ml_corrected = self._ml_correct(rule_corrected, max_length)
        else:
            ml_corrected = rule_corrected
        
        # Post-processing
        final_text = self._post_process(ml_corrected)
        
        return {
            "corrected_text": final_text,
            "confidence": min(confidence * 1.2, 1.0),  # Boost confidence after correction
            "corrections_made": rule_changes
        }
    
    def _ml_correct(self, text: str, max_length: int) -> str:
        """
        ML-based correction using T5
        """
        # Prepare input with task prefix
        input_text = f"correct dysarthric speech: {text}"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate correction
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                temperature=0.7
            )
        
        corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # If model output is empty or too different, keep original
        if not corrected or len(corrected) < 3:
            return text
        
        return corrected
    
    def _apply_rule_based_corrections(self, text: str) -> tuple:
        """
        Apply rule-based corrections for common dysarthric patterns
        """
        corrected = text
        changes = []
        
        for pattern, replacement, description in self.common_corrections:
            if re.search(pattern, corrected, re.IGNORECASE):
                new_text = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
                if new_text != corrected:
                    changes.append({
                        "type": "rule",
                        "pattern": pattern,
                        "description": description
                    })
                corrected = new_text
        
        return corrected, changes
    
    def _apply_minimal_corrections(self, text: str) -> str:
        """
        Light corrections for high-confidence transcriptions
        """
        # Fix spacing
        text = re.sub(r'\s+', ' ', text)
        # Capitalize first letter
        text = text.strip()
        if text:
            text = text[0].upper() + text[1:]
        return text
    
    def _post_process(self, text: str) -> str:
        """
        Post-processing cleanup
        """
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        # Fix punctuation spacing
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        # Capitalize sentences
        text = '. '.join(s.strip().capitalize() for s in text.split('.'))
        return text.strip()
    
    def _load_common_corrections(self) -> List[tuple]:
        """
        Common dysarthric speech error patterns
        Based on research literature on dysarthria
        """
        return [
            # Phoneme confusions
            (r'\bgog\b', 'go', 'g-cluster reduction'),
            (r'\btog\b', 'to', 'consonant confusion'),
            (r'\bwat\b', 'what', 'h-dropping'),
            (r'\bwif\b', 'with', 'th->f substitution'),
            (r'\bdis\b', 'this', 'th->d substitution'),
            (r'\bdat\b', 'that', 'th->d substitution'),
            (r'\bduh\b', 'the', 'th->d substitution'),
            (r'\bstow\b', 'store', 'r->w substitution'),
            (r'\bwedder\b', 'weather', 'th->d substitution'),
            
            # Common slurring patterns (always expand to formal)
            (r'\bgonna\b', 'going to', 'reduction expansion'),
            (r'\bwanna\b', 'want to', 'reduction expansion'),
            (r'\bkinda\b', 'kind of', 'reduction expansion'),
            (r'\bsorta\b', 'sort of', 'reduction expansion'),
            (r'\bgotta\b', 'got to', 'reduction expansion'),
            
            # Stuttering/repetition cleanup
            (r'\b(\w+)\s+\1\b', r'\1', 'repetition removal'),
            
            # Filler word cleanup (optional)
            (r'\buh+\b', '', 'filler removal'),
            (r'\bum+\b', '', 'filler removal'),
        ]
    
    def add_custom_correction(self, pattern: str, replacement: str, description: str = ""):
        """
        Add user-specific correction pattern
        """
        self.common_corrections.append((pattern, replacement, description))
