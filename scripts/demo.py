#!/usr/bin/env python3
"""Simple demo script for multilingual ASR system."""

import logging
import numpy as np
from pathlib import Path

from src.models.whisper_model import WhisperASRModel
from src.utils.device import get_device, set_seed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run a simple demo of the ASR system."""
    print("🎤 Multilingual Speech Recognition Demo")
    print("=" * 50)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device("auto")
    print(f"Using device: {device}")
    
    # Initialize model
    print("\nLoading Whisper model...")
    try:
        model = WhisperASRModel(
            model_name="openai/whisper-base",
            device=device,
            torch_dtype="float16" if device.type != "cpu" else "float32",
            enable_language_detection=True,
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Display model info
    model_info = model.get_model_info()
    print(f"\nModel Information:")
    print(f"  Model: {model_info['model_name']}")
    print(f"  Device: {model_info['device']}")
    print(f"  Parameters: {model_info['parameters']:,}")
    print(f"  Language Detection: {model_info['language_detection']}")
    
    # Test with synthetic audio
    print(f"\nTesting with synthetic audio...")
    
    # Generate synthetic audio (1 second of sine wave)
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    print(f"Generated {duration}s of synthetic audio at {sample_rate}Hz")
    
    # Transcribe audio
    try:
        result = model.transcribe(
            audio,
            language=None,  # Auto-detect
            return_timestamps=False
        )
        
        print(f"\nTranscription Result:")
        print(f"  Text: '{result}'")
        print(f"  Length: {len(result)} characters")
        
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return
    
    # Test language detection
    print(f"\nTesting language detection...")
    try:
        # Test with different language hints
        languages_to_test = ["en", "es", "fr", "de"]
        
        for lang in languages_to_test:
            result = model.transcribe(
                audio,
                language=lang,
                return_timestamps=False
            )
            print(f"  {lang}: '{result}'")
            
    except Exception as e:
        print(f"❌ Language detection test failed: {e}")
    
    # Test post-processing
    print(f"\nTesting post-processing...")
    test_text = "Hello, World! 123"
    
    # Test with different post-processing settings
    model_no_punct = WhisperASRModel(
        model_name="openai/whisper-base",
        device=device,
        torch_dtype="float16" if device.type != "cpu" else "float32",
        enable_punctuation=False,
        enable_casing=False,
        enable_numbers=False,
    )
    
    processed = model_no_punct._post_process(test_text)
    print(f"  Original: '{test_text}'")
    print(f"  Processed: '{processed}'")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"\nTo run the full evaluation:")
    print(f"  python scripts/main.py")
    print(f"\nTo launch the interactive demo:")
    print(f"  streamlit run demo/streamlit_demo.py")


if __name__ == "__main__":
    main()
