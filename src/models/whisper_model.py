"""Whisper-based multilingual ASR model implementation."""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    WhisperFeatureExtractor,
)

from ..utils.device import get_device, get_torch_dtype
from ..utils.audio import load_audio, normalize_audio, trim_silence


logger = logging.getLogger(__name__)


class WhisperASRModel(nn.Module):
    """Whisper-based multilingual ASR model.
    
    This model wraps the Hugging Face Whisper implementation for multilingual
    speech recognition with additional features for research and evaluation.
    """
    
    def __init__(
        self,
        model_name: str = "openai/whisper-base",
        device: Union[str, torch.device] = "auto",
        torch_dtype: str = "float16",
        use_flash_attention: bool = False,
        enable_language_detection: bool = True,
        default_language: str = "en",
        enable_punctuation: bool = True,
        enable_casing: bool = True,
        enable_numbers: bool = True,
        **kwargs
    ):
        """Initialize Whisper ASR model.
        
        Args:
            model_name: Hugging Face model name or path.
            device: Device to run model on.
            torch_dtype: PyTorch data type for model.
            use_flash_attention: Whether to use flash attention.
            enable_language_detection: Whether to enable automatic language detection.
            default_language: Default language code.
            enable_punctuation: Whether to enable punctuation in output.
            enable_casing: Whether to enable proper casing in output.
            enable_numbers: Whether to enable number formatting in output.
            **kwargs: Additional model parameters.
        """
        super().__init__()
        
        self.model_name = model_name
        self.device = get_device(device)
        self.torch_dtype = get_torch_dtype(torch_dtype)
        self.use_flash_attention = use_flash_attention
        self.enable_language_detection = enable_language_detection
        self.default_language = default_language
        self.enable_punctuation = enable_punctuation
        self.enable_casing = enable_casing
        self.enable_numbers = enable_numbers
        
        # Suppress warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        # Load model and processor
        self._load_model()
        self._load_processor()
        
        # Move to device
        self.to(self.device)
        
        logger.info(f"Loaded Whisper model: {model_name}")
        logger.info(f"Model device: {self.device}")
        logger.info(f"Model dtype: {self.torch_dtype}")
    
    def _load_model(self) -> None:
        """Load the Whisper model."""
        try:
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                use_flash_attention_2=self.use_flash_attention,
            )
        except Exception as e:
            logger.warning(f"Failed to load with flash attention, falling back: {e}")
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
            )
    
    def _load_processor(self) -> None:
        """Load the Whisper processor."""
        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.tokenizer = WhisperTokenizer.from_pretrained(self.model_name)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.model_name)
    
    def transcribe(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        language: Optional[str] = None,
        task: str = "transcribe",
        return_timestamps: bool = False,
        chunk_length_s: int = 30,
        stride_length_s: int = 5,
        **generation_kwargs
    ) -> Union[str, Dict[str, Union[str, List[Dict]]]]:
        """Transcribe audio to text.
        
        Args:
            audio: Audio file path, numpy array, or torch tensor.
            language: Language code (e.g., 'en', 'es', 'fr'). If None, auto-detect.
            task: Task type ('transcribe' or 'translate').
            return_timestamps: Whether to return word-level timestamps.
            chunk_length_s: Length of audio chunks in seconds.
            stride_length_s: Stride length between chunks in seconds.
            **generation_kwargs: Additional generation parameters.
            
        Returns:
            Transcription result (string or dict with timestamps).
        """
        # Load and preprocess audio
        if isinstance(audio, str):
            waveform, sample_rate = load_audio(audio)
        elif isinstance(audio, np.ndarray):
            waveform = audio
            sample_rate = 16000  # Assume 16kHz
        elif isinstance(audio, torch.Tensor):
            waveform = audio.cpu().numpy()
            sample_rate = 16000
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")
        
        # Normalize and trim audio
        waveform = normalize_audio(waveform)
        waveform = trim_silence(waveform, sample_rate)
        
        # Prepare inputs
        inputs = self.feature_extractor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Set generation parameters
        generation_kwargs.setdefault("max_length", 448)
        generation_kwargs.setdefault("num_beams", 1)
        generation_kwargs.setdefault("early_stopping", True)
        generation_kwargs.setdefault("do_sample", False)
        
        # Prepare decoder inputs
        if language is None and self.enable_language_detection:
            # Auto-detect language
            language = self._detect_language(inputs["input_features"])
        
        if language:
            # Force language token
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language=language, task=task
            )
            generation_kwargs["forced_decoder_ids"] = forced_decoder_ids
        
        # Generate transcription
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs["input_features"],
                **generation_kwargs
            )
        
        # Decode transcription
        transcription = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        
        # Post-process transcription
        transcription = self._post_process(transcription)
        
        if return_timestamps:
            # Get word-level timestamps
            timestamps = self._extract_timestamps(
                waveform, sample_rate, generated_ids, chunk_length_s, stride_length_s
            )
            return {
                "text": transcription,
                "language": language or "auto",
                "timestamps": timestamps
            }
        
        return transcription
    
    def _detect_language(self, input_features: torch.Tensor) -> str:
        """Detect language from audio features.
        
        Args:
            input_features: Audio input features.
            
        Returns:
            Detected language code.
        """
        try:
            with torch.no_grad():
                # Get language logits
                logits = self.model.model.encoder(input_features)
                logits = self.model.proj_out(logits)
                
                # Get language probabilities
                language_probs = torch.softmax(logits[0, 0, :], dim=-1)
                
                # Get top language
                language_id = torch.argmax(language_probs).item()
                
                # Map to language code
                language_map = {
                    0: "en", 1: "zh", 2: "de", 3: "es", 4: "ru", 5: "ko", 6: "fr",
                    7: "ja", 8: "pt", 9: "tr", 10: "pl", 11: "ca", 12: "nl", 13: "ar",
                    14: "sv", 15: "it", 16: "id", 17: "hi", 18: "fi", 19: "vi",
                    20: "he", 21: "uk", 22: "el", 23: "ms", 24: "cs", 25: "ro",
                    26: "da", 27: "hu", 28: "ta", 29: "no", 30: "th", 31: "ur",
                    32: "hr", 33: "bg", 34: "lt", 35: "la", 36: "mi", 37: "ml",
                    38: "cy", 39: "sk", 40: "te", 41: "fa", 42: "lv", 43: "bn",
                    44: "sr", 45: "az", 46: "sl", 47: "kn", 48: "et", 49: "mk",
                    50: "br", 51: "eu", 52: "is", 53: "hy", 54: "ne", 55: "mn",
                    56: "bs", 57: "kk", 58: "sq", 59: "sw", 60: "gl", 61: "mr",
                    62: "pa", 63: "si", 64: "km", 65: "sn", 66: "yo", 67: "so",
                    68: "af", 69: "oc", 70: "ka", 71: "be", 72: "tg", 73: "sd",
                    74: "gu", 75: "am", 76: "yi", 77: "lo", 78: "uz", 79: "fo",
                    80: "ht", 81: "ps", 82: "tk", 83: "nn", 84: "mt", 85: "sa",
                    86: "lb", 87: "my", 88: "bo", 89: "tl", 90: "mg", 91: "as",
                    92: "tt", 93: "haw", 94: "ln", 95: "ha", 96: "ba", 97: "jw",
                    98: "su", 99: "yue"
                }
                
                return language_map.get(language_id, self.default_language)
                
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return self.default_language
    
    def _post_process(self, text: str) -> str:
        """Post-process transcription text.
        
        Args:
            text: Raw transcription text.
            
        Returns:
            Post-processed text.
        """
        if not self.enable_punctuation:
            # Remove punctuation
            import string
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        if not self.enable_casing:
            # Convert to lowercase
            text = text.lower()
        
        if not self.enable_numbers:
            # Convert numbers to words (simplified)
            import re
            text = re.sub(r'\d+', '[NUMBER]', text)
        
        return text.strip()
    
    def _extract_timestamps(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        generated_ids: torch.Tensor,
        chunk_length_s: int,
        stride_length_s: int
    ) -> List[Dict[str, Union[str, float]]]:
        """Extract word-level timestamps.
        
        Args:
            waveform: Audio waveform.
            sample_rate: Sample rate.
            generated_ids: Generated token IDs.
            chunk_length_s: Chunk length in seconds.
            stride_length_s: Stride length in seconds.
            
        Returns:
            List of word timestamps.
        """
        # This is a simplified implementation
        # In practice, you'd need more sophisticated timestamp extraction
        tokens = self.tokenizer.convert_ids_to_tokens(generated_ids[0])
        
        timestamps = []
        current_time = 0.0
        
        for token in tokens:
            if token.startswith("<|") and token.endswith("|>"):
                continue  # Skip special tokens
            
            # Estimate word duration (simplified)
            word_duration = 0.5  # Assume 0.5 seconds per word
            
            timestamps.append({
                "word": token,
                "start": current_time,
                "end": current_time + word_duration
            })
            
            current_time += word_duration
        
        return timestamps
    
    def get_model_info(self) -> Dict[str, Union[str, int, float]]:
        """Get model information.
        
        Returns:
            Dictionary with model information.
        """
        from ..utils.device import count_parameters
        
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "dtype": str(self.torch_dtype),
            "parameters": count_parameters(self.model),
            "language_detection": self.enable_language_detection,
            "default_language": self.default_language,
            "punctuation": self.enable_punctuation,
            "casing": self.enable_casing,
            "numbers": self.enable_numbers,
        }
    
    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            input_features: Input audio features.
            
        Returns:
            Model output logits.
        """
        return self.model(input_features)
