import pyttsx3
import numpy as np
import soundfile as sf
import os
import random

# -----------------------------
# Generate dysarthric-style speech by PHONETIC text transformation
# This simulates common dysarthric speech patterns
# -----------------------------

def dysarthric_transform(text):
    """
    Apply common dysarthric speech patterns to text
    These will be pronounced by TTS, creating dysarthric-like audio
    """
    # Common dysarthric substitutions:
    # - "th" → "d" or "f" 
    # - "r" → "w"
    # - consonant clusters simplified
    # - final consonants dropped
    
    text = text.lower()
    
    # TH sounds become D or F
    text = text.replace("the ", "duh ")
    text = text.replace("this ", "dis ")
    text = text.replace("that ", "dat ")
    text = text.replace("with ", "wif ")
    text = text.replace("weather", "wedder")
    text = text.replace("there", "dere")
    text = text.replace("they", "dey")
    text = text.replace("their", "der")
    text = text.replace("these", "dese")
    text = text.replace("those", "dose")
    
    # R becomes W (common in some dysarthria)
    text = text.replace("store", "stow")
    text = text.replace("are", "ah")
    text = text.replace("doctor", "doctah")
    text = text.replace("better", "bettah")
    text = text.replace("never", "nevah")
    
    # Consonant cluster reduction
    text = text.replace("str", "suh-tr")
    text = text.replace("spl", "suh-pl")
    
    # Final consonant deletion (casual speech pattern)
    text = text.replace("want to", "wanna")
    text = text.replace("going to", "gonna")
    text = text.replace("have to", "hafta")
    text = text.replace("got to", "gotta")
    text = text.replace(" and ", " an ")
    
    # Add slight pauses (dysfluency)
    words = text.split()
    result = []
    for i, word in enumerate(words):
        result.append(word)
        if random.random() < 0.15:  # 15% chance of pause
            result.append("...")
    
    return " ".join(result)


# Original paragraphs - longer, more natural speech
paragraphs = [
    "Hello, my name is John and I want to tell you about my day. This morning I woke up early and went to the store to buy some groceries. The weather was really nice, so I decided to walk there instead of driving. On my way back home, I met my friend Sarah and we talked for a while about the upcoming weekend.",
    
    "I have always wanted to visit the mountains and see the beautiful scenery there. My doctor told me that getting fresh air and exercise would be good for my health. I am planning to go hiking with my family next month. We are going to pack our bags with food, water, and warm clothes because the temperature can drop quickly in the mountains.",
    
    "Technology has changed our lives in so many ways. I remember when I was younger, we did not have smartphones or the internet. Now I can talk to my friends and family who live far away with just a few taps on my phone. It is amazing how much easier communication has become over the years."
]

print("Generating dysarthric-style speech samples...")
print("=" * 50)

tts = pyttsx3.init()
tts.setProperty('rate', 110)  # Slower rate

for i, original in enumerate(paragraphs, 1):
    # Transform to dysarthric pattern
    dysarthric_text = dysarthric_transform(original)
    
    filename = f"dysarthric_paragraph_{i}.wav"
    
    print(f"\nParagraph {i}:")
    print(f"  Original (first 80 chars): {original[:80]}...")
    print(f"  Dysarthric (first 80 chars): {dysarthric_text[:80]}...")
    
    tts.save_to_file(dysarthric_text, filename)
    tts.runAndWait()
    
    print(f"  ✅ Created {filename}")

print("\n" + "=" * 50)
print("✅ Done! Upload any dysarthric_paragraph_X.wav to test the system")
print("The AI should transcribe the dysarthric speech and correct it back to normal text.")