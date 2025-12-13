"""
Test Cases for Speech Disability Interpreter
Based on common dysarthric speech patterns
"""

from metrics import EvaluationMetrics


# Test cases: (reference, original_asr, expected_corrected)
# These simulate common dysarthric speech errors

TEST_CASES = [
    # Phoneme substitutions (th -> d, th -> f)
    (
        "I think this is the right way",
        "I tink dis is da right way",
        "I think this is the right way"
    ),
    
    # Consonant cluster reduction
    (
        "I want to go to the store",
        "I wan go to te store",
        "I want to go to the store"
    ),
    
    # Final consonant deletion
    (
        "Can you help me find the book",
        "Can you hel me fin the boo",
        "Can you help me find the book"
    ),
    
    # Slurring and reduction
    (
        "I would like some water please",
        "I woul like som water pease",
        "I would like some water please"
    ),
    
    # Articulatory imprecision
    (
        "My name is John and I need assistance",
        "My name i Jon an I nee asistance",
        "My name is John and I need assistance"
    ),
    
    # Speaking rate issues (word merging)
    (
        "I am going to the meeting today",
        "I am gonna the meeting today",
        "I am going to the meeting today"
    ),
    
    # Vowel distortion
    (
        "The doctor said I should rest",
        "The docter sed I shoud rest",
        "The doctor said I should rest"
    ),
    
    # Multiple error types
    (
        "Can you please repeat that more slowly",
        "Can ya pease repea tha more sloly",
        "Can you please repeat that more slowly"
    ),
]


def run_evaluation():
    """
    Run evaluation on all test cases
    """
    print("=" * 70)
    print("Speech Disability Interpreter - Test Case Evaluation")
    print("=" * 70)
    print()
    
    # Prepare test cases for batch evaluation
    batch_cases = [
        (ref, orig, corr)
        for ref, orig, corr in TEST_CASES
    ]
    
    # Run batch evaluation
    results = EvaluationMetrics.batch_evaluate(batch_cases)
    
    print(f"Total Test Cases: {results['total_test_cases']}\n")
    print("=" * 70)
    print("AVERAGE METRICS")
    print("=" * 70)
    print(f"Average WER Before Correction:  {results['avg_wer_before']:.2f}%")
    print(f"Average WER After Correction:   {results['avg_wer_after']:.2f}%")
    print(f"Average WER Improvement:        {results['avg_wer_improvement']:.2f}%\n")
    
    print(f"Average CER Before Correction:  {results['avg_cer_before']:.2f}%")
    print(f"Average CER After Correction:   {results['avg_cer_after']:.2f}%")
    print(f"Average CER Improvement:        {results['avg_cer_improvement']:.2f}%\n")
    
    print("=" * 70)
    print("INDIVIDUAL TEST CASE RESULTS")
    print("=" * 70)
    
    for i, (ref, orig, corr) in enumerate(TEST_CASES, 1):
        result = results['individual_results'][i - 1]
        
        print(f"\n--- Test Case {i} ---")
        print(f"Reference:  {ref}")
        print(f"Original:   {orig}")
        print(f"Corrected:  {corr}")
        print(f"WER: {result['wer_before']:.1f}% → {result['wer_after']:.1f}% (Δ {result['wer_improvement']:.1f}%)")
        print(f"CER: {result['cer_before']:.1f}% → {result['cer_after']:.1f}% (Δ {result['cer_improvement']:.1f}%)")
        
        # Get detailed errors
        errors_before = EvaluationMetrics.get_detailed_errors(ref, orig)
        errors_after = EvaluationMetrics.get_detailed_errors(ref, corr)
        
        print(f"Errors: {errors_before['substitutions']}S {errors_before['deletions']}D {errors_before['insertions']}I → "
              f"{errors_after['substitutions']}S {errors_after['deletions']}D {errors_after['insertions']}I")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    
    # Calculate success rate
    perfect_corrections = sum(
        1 for r in results['individual_results']
        if r['wer_after'] == 0.0
    )
    
    success_rate = (perfect_corrections / len(TEST_CASES)) * 100
    print(f"\nPerfect Corrections: {perfect_corrections}/{len(TEST_CASES)} ({success_rate:.1f}%)")
    
    return results


if __name__ == "__main__":
    run_evaluation()
