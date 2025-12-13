"""
Personalization Engine
Learns and applies user-specific corrections over time
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from datetime import datetime
import re


class PersonalizationEngine:
    """
    Per-user adaptation and learning from corrections
    """
    
    def __init__(self, storage_path: Path):
        """
        Initialize personalization engine
        
        Args:
            storage_path: Directory to store user profiles
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache of user corrections
        self.user_corrections: Dict[str, Dict] = defaultdict(lambda: {
            "word_mappings": {},  # incorrect_word -> correct_word
            "phrase_mappings": {},  # incorrect_phrase -> correct_phrase
            "correction_count": 0,
            "sessions": []
        })
        
        print("✅ Personalization engine initialized")
    
    def load_user_profile(self, user_id: str) -> Dict:
        """
        Load user's correction history
        """
        profile_path = self.storage_path / f"{user_id}_profile.json"
        
        if profile_path.exists():
            with open(profile_path, 'r', encoding='utf-8') as f:
                self.user_corrections[user_id] = json.load(f)
        
        return self.user_corrections[user_id]
    
    def save_user_profile(self, user_id: str):
        """
        Save user profile to disk
        """
        profile_path = self.storage_path / f"{user_id}_profile.json"
        
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(self.user_corrections[user_id], f, indent=2, ensure_ascii=False)
    
    def add_correction(
        self,
        user_id: str,
        original: str,
        corrected_by_model: str,
        corrected_by_user: str
    ):
        """
        Record a user correction for learning
        
        Args:
            user_id: User identifier
            original: Original ASR text
            corrected_by_model: Model's correction
            corrected_by_user: User's final correction
        """
        # Load user profile if not in memory
        if user_id not in self.user_corrections:
            self.load_user_profile(user_id)
        
        profile = self.user_corrections[user_id]
        
        # Extract word-level corrections
        self._extract_word_mappings(
            original,
            corrected_by_user,
            profile["word_mappings"]
        )
        
        # Extract phrase-level corrections
        self._extract_phrase_mappings(
            original,
            corrected_by_user,
            profile["phrase_mappings"]
        )
        
        # Record session
        profile["sessions"].append({
            "timestamp": datetime.now().isoformat(),
            "original": original,
            "model_correction": corrected_by_model,
            "user_correction": corrected_by_user
        })
        
        profile["correction_count"] += 1
        
        # Save to disk
        self.save_user_profile(user_id)
    
    def apply_user_corrections(self, user_id: str, text: str) -> str:
        """
        Apply learned user-specific corrections
        
        Args:
            user_id: User identifier
            text: Text to correct
            
        Returns:
            Corrected text
        """
        # Load user profile
        if user_id not in self.user_corrections:
            self.load_user_profile(user_id)
        
        profile = self.user_corrections[user_id]
        corrected = text
        
        # Apply phrase mappings first (more specific)
        for incorrect_phrase, correct_phrase in profile["phrase_mappings"].items():
            # Use word boundaries for whole-phrase matching
            pattern = r'\b' + re.escape(incorrect_phrase) + r'\b'
            corrected = re.sub(pattern, correct_phrase, corrected, flags=re.IGNORECASE)
        
        # Apply word mappings
        for incorrect_word, correct_word in profile["word_mappings"].items():
            # Word-level replacement with boundary matching
            pattern = r'\b' + re.escape(incorrect_word) + r'\b'
            corrected = re.sub(pattern, correct_word, corrected, flags=re.IGNORECASE)
        
        return corrected
    
    def _extract_word_mappings(
        self,
        original: str,
        corrected: str,
        mappings: Dict
    ):
        """
        Extract word-level corrections using simple alignment
        """
        original_words = original.lower().split()
        corrected_words = corrected.lower().split()
        
        # Simple alignment (for MVP; use proper alignment like edit distance in production)
        if len(original_words) == len(corrected_words):
            for orig, corr in zip(original_words, corrected_words):
                if orig != corr and len(orig) > 2:  # Skip very short words
                    # Track frequency of this correction
                    if orig in mappings:
                        if mappings[orig] != corr:
                            # Conflict - keep most frequent
                            pass
                    else:
                        mappings[orig] = corr
    
    def _extract_phrase_mappings(
        self,
        original: str,
        corrected: str,
        mappings: Dict
    ):
        """
        Extract phrase-level patterns
        """
        # For MVP, store exact phrase mappings
        # In production, use fuzzy matching and n-grams
        
        orig_clean = original.lower().strip()
        corr_clean = corrected.lower().strip()
        
        if orig_clean != corr_clean and len(orig_clean.split()) <= 5:
            # Store short phrase corrections
            mappings[orig_clean] = corr_clean
    
    def get_user_stats(self, user_id: str) -> Dict:
        """
        Get statistics for a user's corrections
        """
        if user_id not in self.user_corrections:
            self.load_user_profile(user_id)
        
        profile = self.user_corrections[user_id]
        
        return {
            "user_id": user_id,
            "total_corrections": profile["correction_count"],
            "unique_word_mappings": len(profile["word_mappings"]),
            "unique_phrase_mappings": len(profile["phrase_mappings"]),
            "sessions": len(profile["sessions"]),
            "last_session": profile["sessions"][-1]["timestamp"] if profile["sessions"] else None,
            "most_common_corrections": self._get_top_corrections(profile["word_mappings"], 10)
        }
    
    def _get_top_corrections(self, mappings: Dict, top_n: int = 10) -> List[Dict]:
        """
        Get most common corrections (for display)
        """
        # For MVP, just return all (in production, track frequency)
        return [
            {"from": k, "to": v}
            for k, v in list(mappings.items())[:top_n]
        ]
    
    def export_training_data(self, user_id: str) -> List[Dict]:
        """
        Export user corrections as training data for fine-tuning
        """
        if user_id not in self.user_corrections:
            self.load_user_profile(user_id)
        
        profile = self.user_corrections[user_id]
        
        training_pairs = []
        for session in profile["sessions"]:
            training_pairs.append({
                "input": session["original"],
                "target": session["user_correction"],
                "timestamp": session["timestamp"]
            })
        
        return training_pairs
