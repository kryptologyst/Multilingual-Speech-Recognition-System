"""Multilingual Speech Recognition System."""

__version__ = "0.1.0"
__author__ = "AI Research Team"
__email__ = "research@example.com"

from .models.whisper_model import WhisperASRModel
from .data.multilingual_dataset import MultilingualASRDataset, create_dataloader
from .eval.evaluator import ASREvaluator
from .metrics.asr_metrics import ASRMetrics, compute_wer, compute_cer, compute_token_accuracy
from .utils.device import get_device, set_seed, count_parameters
from .utils.audio import load_audio, normalize_audio, trim_silence

__all__ = [
    "WhisperASRModel",
    "MultilingualASRDataset",
    "create_dataloader",
    "ASREvaluator",
    "ASRMetrics",
    "compute_wer",
    "compute_cer",
    "compute_token_accuracy",
    "get_device",
    "set_seed",
    "count_parameters",
    "load_audio",
    "normalize_audio",
    "trim_silence",
]
