"""Evaluation metrics for ASR systems."""

import logging
from typing import Dict, List, Optional, Union

import jiwer
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logger = logging.getLogger(__name__)


class ASRMetrics:
    """ASR evaluation metrics calculator."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.reset()
    
    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self.predictions = []
        self.references = []
        self.latencies = []
        self.audio_lengths = []
    
    def add_batch(
        self,
        predictions: List[str],
        references: List[str],
        latencies: Optional[List[float]] = None,
        audio_lengths: Optional[List[float]] = None,
    ) -> None:
        """Add a batch of predictions and references.
        
        Args:
            predictions: List of predicted transcriptions.
            references: List of reference transcriptions.
            latencies: List of inference latencies in seconds.
            audio_lengths: List of audio durations in seconds.
        """
        self.predictions.extend(predictions)
        self.references.extend(references)
        
        if latencies is not None:
            self.latencies.extend(latencies)
        
        if audio_lengths is not None:
            self.audio_lengths.extend(audio_lengths)
    
    def compute_wer(self) -> float:
        """Compute Word Error Rate (WER).
        
        Returns:
            WER as a float between 0 and 1.
        """
        if not self.predictions or not self.references:
            return 0.0
        
        try:
            wer = jiwer.wer(self.references, self.predictions)
            return float(wer)
        except Exception as e:
            logger.warning(f"WER computation failed: {e}")
            return 0.0
    
    def compute_cer(self) -> float:
        """Compute Character Error Rate (CER).
        
        Returns:
            CER as a float between 0 and 1.
        """
        if not self.predictions or not self.references:
            return 0.0
        
        try:
            cer = jiwer.cer(self.references, self.predictions)
            return float(cer)
        except Exception as e:
            logger.warning(f"CER computation failed: {e}")
            return 0.0
    
    def compute_token_accuracy(self) -> float:
        """Compute token-level accuracy.
        
        Returns:
            Token accuracy as a float between 0 and 1.
        """
        if not self.predictions or not self.references:
            return 0.0
        
        try:
            # Tokenize predictions and references
            pred_tokens = []
            ref_tokens = []
            
            for pred, ref in zip(self.predictions, self.references):
                pred_tokens.extend(pred.lower().split())
                ref_tokens.extend(ref.lower().split())
            
            # Compute accuracy
            if len(ref_tokens) == 0:
                return 0.0
            
            # Pad or truncate to same length
            max_len = max(len(pred_tokens), len(ref_tokens))
            pred_tokens += ['<PAD>'] * (max_len - len(pred_tokens))
            ref_tokens += ['<PAD>'] * (max_len - len(ref_tokens))
            
            accuracy = accuracy_score(ref_tokens, pred_tokens)
            return float(accuracy)
            
        except Exception as e:
            logger.warning(f"Token accuracy computation failed: {e}")
            return 0.0
    
    def compute_latency_metrics(self) -> Dict[str, float]:
        """Compute latency-related metrics.
        
        Returns:
            Dictionary with latency metrics.
        """
        if not self.latencies:
            return {"mean_latency": 0.0, "std_latency": 0.0, "min_latency": 0.0, "max_latency": 0.0}
        
        latencies = np.array(self.latencies)
        
        return {
            "mean_latency": float(np.mean(latencies)),
            "std_latency": float(np.std(latencies)),
            "min_latency": float(np.min(latencies)),
            "max_latency": float(np.max(latencies)),
        }
    
    def compute_rtf(self) -> Dict[str, float]:
        """Compute Real-Time Factor (RTF).
        
        Returns:
            Dictionary with RTF metrics.
        """
        if not self.latencies or not self.audio_lengths:
            return {"mean_rtf": 0.0, "std_rtf": 0.0}
        
        if len(self.latencies) != len(self.audio_lengths):
            logger.warning("Mismatch between latencies and audio lengths")
            return {"mean_rtf": 0.0, "std_rtf": 0.0}
        
        rtf_values = []
        for latency, audio_length in zip(self.latencies, self.audio_lengths):
            if audio_length > 0:
                rtf_values.append(latency / audio_length)
        
        if not rtf_values:
            return {"mean_rtf": 0.0, "std_rtf": 0.0}
        
        rtf_values = np.array(rtf_values)
        
        return {
            "mean_rtf": float(np.mean(rtf_values)),
            "std_rtf": float(np.std(rtf_values)),
        }
    
    def compute_all_metrics(self) -> Dict[str, Union[float, Dict[str, float]]]:
        """Compute all available metrics.
        
        Returns:
            Dictionary with all computed metrics.
        """
        metrics = {
            "wer": self.compute_wer(),
            "cer": self.compute_cer(),
            "token_accuracy": self.compute_token_accuracy(),
            "latency": self.compute_latency_metrics(),
            "rtf": self.compute_rtf(),
        }
        
        return metrics
    
    def get_summary(self) -> str:
        """Get a formatted summary of all metrics.
        
        Returns:
            Formatted string with metrics summary.
        """
        metrics = self.compute_all_metrics()
        
        summary = "ASR Evaluation Results:\n"
        summary += "=" * 50 + "\n"
        summary += f"Word Error Rate (WER): {metrics['wer']:.4f}\n"
        summary += f"Character Error Rate (CER): {metrics['cer']:.4f}\n"
        summary += f"Token Accuracy: {metrics['token_accuracy']:.4f}\n"
        summary += f"Mean Latency: {metrics['latency']['mean_latency']:.4f}s\n"
        summary += f"Mean RTF: {metrics['rtf']['mean_rtf']:.4f}\n"
        
        return summary


def compute_wer(predictions: List[str], references: List[str]) -> float:
    """Compute WER for a list of predictions and references.
    
    Args:
        predictions: List of predicted transcriptions.
        references: List of reference transcriptions.
        
    Returns:
        WER as a float between 0 and 1.
    """
    metrics = ASRMetrics()
    metrics.add_batch(predictions, references)
    return metrics.compute_wer()


def compute_cer(predictions: List[str], references: List[str]) -> float:
    """Compute CER for a list of predictions and references.
    
    Args:
        predictions: List of predicted transcriptions.
        references: List of reference transcriptions.
        
    Returns:
        CER as a float between 0 and 1.
    """
    metrics = ASRMetrics()
    metrics.add_batch(predictions, references)
    return metrics.compute_cer()


def compute_token_accuracy(predictions: List[str], references: List[str]) -> float:
    """Compute token accuracy for a list of predictions and references.
    
    Args:
        predictions: List of predicted transcriptions.
        references: List of reference transcriptions.
        
    Returns:
        Token accuracy as a float between 0 and 1.
    """
    metrics = ASRMetrics()
    metrics.add_batch(predictions, references)
    return metrics.compute_token_accuracy()
