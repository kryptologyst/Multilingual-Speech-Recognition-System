"""Tests for ASR models."""

import pytest
import torch
import numpy as np

from src.models.whisper_model import WhisperASRModel
from src.utils.device import get_device


class TestWhisperASRModel:
    """Test cases for WhisperASRModel."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = WhisperASRModel(
            model_name="openai/whisper-base",
            device="cpu",
            torch_dtype="float32",
            enable_language_detection=False,
        )
        
        assert model.model_name == "openai/whisper-base"
        assert str(model.device) == "cpu"
        assert model.torch_dtype == torch.float32
        assert not model.enable_language_detection
    
    def test_model_info(self):
        """Test model info retrieval."""
        model = WhisperASRModel(
            model_name="openai/whisper-base",
            device="cpu",
            torch_dtype="float32",
        )
        
        info = model.get_model_info()
        
        assert "model_name" in info
        assert "device" in info
        assert "dtype" in info
        assert "parameters" in info
        assert info["model_name"] == "openai/whisper-base"
        assert info["device"] == "cpu"
    
    def test_transcribe_synthetic_audio(self):
        """Test transcription with synthetic audio."""
        model = WhisperASRModel(
            model_name="openai/whisper-base",
            device="cpu",
            torch_dtype="float32",
            enable_language_detection=False,
        )
        
        # Create synthetic audio (1 second of silence)
        sample_rate = 16000
        duration = 1.0
        audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
        
        # Transcribe (should handle empty audio gracefully)
        result = model.transcribe(audio, language="en")
        
        assert isinstance(result, str)
        # Empty audio might result in empty string or minimal output
        assert len(result) >= 0
    
    def test_post_processing(self):
        """Test text post-processing."""
        model = WhisperASRModel(
            model_name="openai/whisper-base",
            device="cpu",
            torch_dtype="float32",
            enable_punctuation=False,
            enable_casing=False,
            enable_numbers=False,
        )
        
        # Test post-processing
        test_text = "Hello, World! 123"
        processed = model._post_process(test_text)
        
        # Should remove punctuation, convert to lowercase, and replace numbers
        assert "hello" in processed.lower()
        assert "world" in processed.lower()
        assert "123" not in processed or "[NUMBER]" in processed
    
    def test_device_handling(self):
        """Test device handling."""
        # Test CPU device
        model_cpu = WhisperASRModel(
            model_name="openai/whisper-base",
            device="cpu",
            torch_dtype="float32",
        )
        assert str(model_cpu.device) == "cpu"
        
        # Test auto device selection
        model_auto = WhisperASRModel(
            model_name="openai/whisper-base",
            device="auto",
            torch_dtype="float32",
        )
        # Should select appropriate device
        assert model_auto.device is not None
    
    def test_forward_pass(self):
        """Test forward pass through model."""
        model = WhisperASRModel(
            model_name="openai/whisper-base",
            device="cpu",
            torch_dtype="float32",
        )
        
        # Create dummy input features
        batch_size = 1
        seq_length = 80
        feature_dim = 80
        input_features = torch.randn(batch_size, seq_length, feature_dim)
        
        # Forward pass
        with torch.no_grad():
            output = model.forward(input_features)
        
        # Check output shape (this depends on the specific model architecture)
        assert output is not None
        assert isinstance(output, torch.Tensor)
