"""
Batch Process Dysarthric Audio Files
Processes all WAV files in audio_uploads/ and generates corrected speech in output/
1-to-1 relationship: input.wav -> output/input_corrected.wav
"""

import os
from pathlib import Path
import time
from models.asr_engine import WhisperASR
from models.correction_model import CorrectionModel
from models.tts_engine import TTSEngine
from utils.audio_preprocessing import AudioPreprocessor
from utils.feature_extraction import AudioFeatures

# Directories
INPUT_DIR = Path("data/audio_uploads")
OUTPUT_DIR = Path("data/output")

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize models
print("=" * 70)
print("🚀 Initializing AI Speech Interpreter for Batch Processing")
print("=" * 70)

print("\n📦 Loading models...")
audio_preprocessor = AudioPreprocessor()
asr_engine = WhisperASR(model_size="base")
correction_model = CorrectionModel()
tts_engine = TTSEngine()
print("✅ All models loaded!\n")


def process_audio_file(input_path: Path) -> dict:
    """
    Process a single audio file through the full pipeline
    
    Returns:
        dict with processing results
    """
    print(f"\n📤 Processing: {input_path.name}")
    start_time = time.time()
    
    try:
        # 1. Preprocess audio
        preprocessed_audio = audio_preprocessor.preprocess(str(input_path))
        
        # 2. Extract features
        audio_features = AudioFeatures.extract(preprocessed_audio)
        
        # 3. Run ASR (transcribe dysarthric speech)
        asr_result = asr_engine.transcribe(preprocessed_audio)
        original_text = asr_result["text"]
        confidence = asr_result.get("confidence", 0.0)
        
        print(f"   📝 Transcribed: '{original_text}'")
        print(f"   📊 Confidence: {confidence:.2%}")
        
        # 4. Apply correction
        correction_result = correction_model.correct(
            text=original_text,
            audio_features=audio_features,
            confidence=confidence
        )
        corrected_text = correction_result["corrected_text"]
        
        print(f"   ✅ Corrected: '{corrected_text}'")
        
        # 5. Generate corrected speech (normal speech)
        output_filename = input_path.stem + "_corrected.wav"
        output_path = OUTPUT_DIR / output_filename
        
        tts_engine.synthesize(
            text=corrected_text,
            output_path=str(output_path)
        )
        
        processing_time = time.time() - start_time
        
        print(f"   🔊 Generated: {output_filename}")
        print(f"   ⏱️  Processing time: {processing_time:.2f}s")
        
        return {
            "status": "success",
            "input_file": str(input_path),
            "output_file": str(output_path),
            "original_text": original_text,
            "corrected_text": corrected_text,
            "confidence": confidence,
            "processing_time": processing_time
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "input_file": str(input_path),
            "error": str(e)
        }


def main():
    """
    Process all WAV files in audio_uploads directory
    """
    print("=" * 70)
    print("📁 Scanning for audio files in:", INPUT_DIR)
    print("=" * 70)
    
    # Find all WAV files
    wav_files = list(INPUT_DIR.glob("*.wav"))
    
    if not wav_files:
        print("\n❌ No WAV files found in", INPUT_DIR)
        print("Please add dysarthric audio files to process.\n")
        return
    
    print(f"\n✅ Found {len(wav_files)} WAV file(s)\n")
    
    # Process each file
    results = []
    for wav_file in sorted(wav_files):
        result = process_audio_file(wav_file)
        results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 PROCESSING SUMMARY")
    print("=" * 70)
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]
    
    print(f"\n✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    
    if successful:
        print("\n" + "-" * 70)
        print("RESULTS:")
        print("-" * 70)
        for i, r in enumerate(successful, 1):
            input_name = Path(r["input_file"]).name
            output_name = Path(r["output_file"]).name
            print(f"\n{i}. {input_name}")
            print(f"   Original:  '{r['original_text']}'")
            print(f"   Corrected: '{r['corrected_text']}'")
            print(f"   Output:    {output_name}")
            print(f"   Time:      {r['processing_time']:.2f}s")
    
    if failed:
        print("\n" + "-" * 70)
        print("ERRORS:")
        print("-" * 70)
        for r in failed:
            print(f"❌ {Path(r['input_file']).name}: {r['error']}")
    
    print("\n" + "=" * 70)
    print(f"✅ Done! Corrected audio files saved to: {OUTPUT_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
