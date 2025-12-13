"""
Evaluation Metrics for Speech Disability Interpreter
WER, CER, and other speech quality metrics
"""

import jiwer
from typing import Dict, List, Tuple
import numpy as np


class EvaluationMetrics:
    """
    Calculate evaluation metrics for ASR and correction performance
    """
    
    @staticmethod
    def calculate_wer(reference: str, hypothesis: str) -> float:
        """
        Calculate Word Error Rate
        
        Args:
            reference: Ground truth text
            hypothesis: Predicted text
            
        Returns:
            WER as a percentage
        """
        if not reference or not hypothesis:
            return 100.0
        
        wer = jiwer.wer(reference, hypothesis)
        return wer * 100.0
    
    @staticmethod
    def calculate_cer(reference: str, hypothesis: str) -> float:
        """
        Calculate Character Error Rate
        
        Args:
            reference: Ground truth text
            hypothesis: Predicted text
            
        Returns:
            CER as a percentage
        """
        if not reference or not hypothesis:
            return 100.0
        
        cer = jiwer.cer(reference, hypothesis)
        return cer * 100.0
    
    @staticmethod
    def calculate_mer(reference: str, hypothesis: str) -> float:
        """
        Calculate Match Error Rate (word-level)
        """
        if not reference or not hypothesis:
            return 100.0
        
        mer = jiwer.mer(reference, hypothesis)
        return mer * 100.0
    
    @staticmethod
    def calculate_wil(reference: str, hypothesis: str) -> float:
        """
        Calculate Word Information Lost
        """
        if not reference or not hypothesis:
            return 100.0
        
        wil = jiwer.wil(reference, hypothesis)
        return wil * 100.0
    
    @staticmethod
    def calculate_improvement(
        reference: str,
        original_hypothesis: str,
        corrected_hypothesis: str
    ) -> Dict:
        """
        Calculate improvement metrics after correction
        
        Args:
            reference: Ground truth
            original_hypothesis: Original ASR output
            corrected_hypothesis: After correction
            
        Returns:
            Dict with WER/CER before, after, and improvement
        """
        wer_before = EvaluationMetrics.calculate_wer(reference, original_hypothesis)
        wer_after = EvaluationMetrics.calculate_wer(reference, corrected_hypothesis)
        
        cer_before = EvaluationMetrics.calculate_cer(reference, original_hypothesis)
        cer_after = EvaluationMetrics.calculate_cer(reference, corrected_hypothesis)
        
        return {
            "wer_before": wer_before,
            "wer_after": wer_after,
            "wer_improvement": wer_before - wer_after,
            "wer_improvement_percent": ((wer_before - wer_after) / wer_before * 100) if wer_before > 0 else 0.0,
            "cer_before": cer_before,
            "cer_after": cer_after,
            "cer_improvement": cer_before - cer_after,
            "cer_improvement_percent": ((cer_before - cer_after) / cer_before * 100) if cer_before > 0 else 0.0
        }
    
    @staticmethod
    def batch_evaluate(test_cases: List[Tuple[str, str, str]]) -> Dict:
        """
        Evaluate on multiple test cases
        
        Args:
            test_cases: List of (reference, original, corrected) tuples
            
        Returns:
            Average metrics across all test cases
        """
        results = []
        
        for reference, original, corrected in test_cases:
            improvement = EvaluationMetrics.calculate_improvement(
                reference, original, corrected
            )
            results.append(improvement)
        
        # Calculate averages
        avg_metrics = {
            "avg_wer_before": np.mean([r["wer_before"] for r in results]),
            "avg_wer_after": np.mean([r["wer_after"] for r in results]),
            "avg_wer_improvement": np.mean([r["wer_improvement"] for r in results]),
            "avg_cer_before": np.mean([r["cer_before"] for r in results]),
            "avg_cer_after": np.mean([r["cer_after"] for r in results]),
            "avg_cer_improvement": np.mean([r["cer_improvement"] for r in results]),
            "total_test_cases": len(test_cases),
            "individual_results": results
        }
        
        return avg_metrics
    
    @staticmethod
    def calculate_sentence_accuracy(reference: str, hypothesis: str) -> float:
        """
        Calculate exact sentence match accuracy (binary)
        """
        ref_clean = reference.strip().lower()
        hyp_clean = hypothesis.strip().lower()
        return 100.0 if ref_clean == hyp_clean else 0.0
    
    @staticmethod
    def get_detailed_errors(reference: str, hypothesis: str) -> Dict:
        """
        Get detailed error breakdown (insertions, deletions, substitutions)
        """
        measures = jiwer.compute_measures(reference, hypothesis)
        
        return {
            "insertions": measures["insertions"],
            "deletions": measures["deletions"],
            "substitutions": measures["substitutions"],
            "hits": measures["hits"],
            "total_words": len(reference.split())
        }


# Example usage and test cases
if __name__ == "__main__":
    # Test case: Dysarthric speech correction
    reference = "I want to go to the store"
    original = "I wan gog to te store"  # Simulated dysarthric output
    corrected = "I want to go to the store"
    
    print("=== Speech Disability Interpreter - Evaluation ===\n")
    
    print(f"Reference:  {reference}")
    print(f"Original:   {original}")
    print(f"Corrected:  {corrected}\n")
    
    improvement = EvaluationMetrics.calculate_improvement(reference, original, corrected)
    
    print(f"WER Before:  {improvement['wer_before']:.2f}%")
    print(f"WER After:   {improvement['wer_after']:.2f}%")
    print(f"WER Improvement: {improvement['wer_improvement']:.2f}% ({improvement['wer_improvement_percent']:.1f}% relative)\n")
    
    print(f"CER Before:  {improvement['cer_before']:.2f}%")
    print(f"CER After:   {improvement['cer_after']:.2f}%")
    print(f"CER Improvement: {improvement['cer_improvement']:.2f}% ({improvement['cer_improvement_percent']:.1f}% relative)\n")
    
    errors = EvaluationMetrics.get_detailed_errors(reference, original)
    print(f"Error Analysis (Original):")
    print(f"  Substitutions: {errors['substitutions']}")
    print(f"  Deletions: {errors['deletions']}")
    print(f"  Insertions: {errors['insertions']}")
    print(f"  Correct: {errors['hits']}")
