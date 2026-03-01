"""Evaluation module for ASR systems."""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..models.whisper_model import WhisperASRModel
from ..data.multilingual_dataset import MultilingualASRDataset, create_dataloader
from ..metrics.asr_metrics import ASRMetrics
from ..utils.device import get_device, set_seed

logger = logging.getLogger(__name__)


class ASREvaluator:
    """ASR system evaluator."""
    
    def __init__(
        self,
        model: WhisperASRModel,
        device: Union[str, torch.device] = "auto",
        seed: int = 42,
    ):
        """Initialize ASR evaluator.
        
        Args:
            model: ASR model to evaluate.
            device: Device to run evaluation on.
            seed: Random seed for reproducibility.
        """
        self.model = model
        self.device = get_device(device)
        self.seed = seed
        
        # Set seed for reproducibility
        set_seed(seed)
        
        logger.info(f"Initialized ASR evaluator on device: {self.device}")
    
    def evaluate_dataset(
        self,
        dataset: MultilingualASRDataset,
        batch_size: int = 1,
        return_predictions: bool = False,
        save_results: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Union[float, Dict, List]]:
        """Evaluate model on a dataset.
        
        Args:
            dataset: Dataset to evaluate on.
            batch_size: Batch size for evaluation.
            return_predictions: Whether to return individual predictions.
            save_results: Path to save results to.
            
        Returns:
            Dictionary with evaluation results.
        """
        logger.info(f"Starting evaluation on {len(dataset)} samples")
        
        # Create dataloader
        dataloader = create_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Use 0 for evaluation to avoid multiprocessing issues
        )
        
        # Initialize metrics
        metrics = ASRMetrics()
        predictions = []
        references = []
        latencies = []
        audio_lengths = []
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Evaluate
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch_predictions = []
                batch_latencies = []
                
                for i in range(len(batch['audio'])):
                    audio = batch['audio'][i].numpy()
                    reference = batch['transcripts'][i]
                    duration = batch['durations'][i].item()
                    
                    # Measure inference time
                    start_time = time.time()
                    
                    try:
                        # Transcribe audio
                        prediction = self.model.transcribe(
                            audio,
                            language=None,  # Auto-detect language
                            return_timestamps=False,
                        )
                        
                        inference_time = time.time() - start_time
                        
                        batch_predictions.append(prediction)
                        batch_latencies.append(inference_time)
                        
                    except Exception as e:
                        logger.warning(f"Transcription failed for sample {batch['ids'][i]}: {e}")
                        batch_predictions.append("")
                        batch_latencies.append(0.0)
                
                # Add to metrics
                metrics.add_batch(
                    predictions=batch_predictions,
                    references=batch['transcripts'],
                    latencies=batch_latencies,
                    audio_lengths=batch['durations'].tolist(),
                )
                
                # Store individual results if requested
                if return_predictions:
                    for i in range(len(batch['audio'])):
                        predictions.append({
                            'id': batch['ids'][i],
                            'prediction': batch_predictions[i],
                            'reference': batch['transcripts'][i],
                            'language': batch['languages'][i],
                            'duration': batch['durations'][i].item(),
                            'latency': batch_latencies[i],
                            'speaker': batch['speakers'][i],
                        })
        
        # Compute metrics
        results = metrics.compute_all_metrics()
        
        # Add dataset statistics
        results['dataset_stats'] = dataset.get_statistics()
        
        # Add individual predictions if requested
        if return_predictions:
            results['predictions'] = predictions
        
        # Save results if requested
        if save_results:
            self._save_results(results, save_results)
        
        logger.info("Evaluation completed")
        logger.info(f"WER: {results['wer']:.4f}")
        logger.info(f"CER: {results['cer']:.4f}")
        logger.info(f"Token Accuracy: {results['token_accuracy']:.4f}")
        logger.info(f"Mean Latency: {results['latency']['mean_latency']:.4f}s")
        logger.info(f"Mean RTF: {results['rtf']['mean_rtf']:.4f}")
        
        return results
    
    def evaluate_file(
        self,
        audio_path: Union[str, Path],
        reference_text: Optional[str] = None,
        language: Optional[str] = None,
        return_timestamps: bool = False,
    ) -> Dict[str, Union[str, float, List]]:
        """Evaluate model on a single audio file.
        
        Args:
            audio_path: Path to audio file.
            reference_text: Reference transcription (optional).
            language: Language code (optional).
            return_timestamps: Whether to return timestamps.
            
        Returns:
            Dictionary with evaluation results.
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Evaluating single file: {audio_path}")
        
        # Measure inference time
        start_time = time.time()
        
        try:
            # Transcribe audio
            result = self.model.transcribe(
                str(audio_path),
                language=language,
                return_timestamps=return_timestamps,
            )
            
            inference_time = time.time() - start_time
            
            # Prepare results
            if isinstance(result, dict):
                prediction = result['text']
                timestamps = result.get('timestamps', [])
                detected_language = result.get('language', 'unknown')
            else:
                prediction = result
                timestamps = []
                detected_language = 'unknown'
            
            results = {
                'prediction': prediction,
                'detected_language': detected_language,
                'latency': inference_time,
                'audio_path': str(audio_path),
            }
            
            if return_timestamps:
                results['timestamps'] = timestamps
            
            # Compute metrics if reference is provided
            if reference_text:
                metrics = ASRMetrics()
                metrics.add_batch([prediction], [reference_text], [inference_time])
                
                results.update({
                    'reference': reference_text,
                    'wer': metrics.compute_wer(),
                    'cer': metrics.compute_cer(),
                    'token_accuracy': metrics.compute_token_accuracy(),
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed for {audio_path}: {e}")
            return {
                'prediction': '',
                'error': str(e),
                'latency': inference_time,
                'audio_path': str(audio_path),
            }
    
    def _save_results(self, results: Dict, save_path: Union[str, Path]) -> None:
        """Save evaluation results to file.
        
        Args:
            results: Evaluation results.
            save_path: Path to save results to.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        import json
        
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_types(results)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {save_path}")
    
    def create_leaderboard(
        self,
        results: Dict,
        save_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """Create a leaderboard from evaluation results.
        
        Args:
            results: Evaluation results.
            save_path: Path to save leaderboard to.
            
        Returns:
            DataFrame with leaderboard.
        """
        # Create leaderboard data
        leaderboard_data = {
            'Metric': ['WER', 'CER', 'Token Accuracy', 'Mean Latency (s)', 'Mean RTF'],
            'Value': [
                results['wer'],
                results['cer'],
                results['token_accuracy'],
                results['latency']['mean_latency'],
                results['rtf']['mean_rtf'],
            ],
            'Description': [
                'Word Error Rate (lower is better)',
                'Character Error Rate (lower is better)',
                'Token-level Accuracy (higher is better)',
                'Average inference latency (lower is better)',
                'Real-Time Factor (lower is better)',
            ]
        }
        
        leaderboard = pd.DataFrame(leaderboard_data)
        
        # Save if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            leaderboard.to_csv(save_path, index=False)
            logger.info(f"Leaderboard saved to: {save_path}")
        
        return leaderboard
    
    def get_model_performance_summary(self) -> str:
        """Get a summary of model performance.
        
        Returns:
            Formatted performance summary.
        """
        model_info = self.model.get_model_info()
        
        summary = "Model Performance Summary\n"
        summary += "=" * 50 + "\n"
        summary += f"Model: {model_info['model_name']}\n"
        summary += f"Device: {model_info['device']}\n"
        summary += f"Parameters: {model_info['parameters']:,}\n"
        summary += f"Language Detection: {model_info['language_detection']}\n"
        summary += f"Default Language: {model_info['default_language']}\n"
        summary += f"Punctuation: {model_info['punctuation']}\n"
        summary += f"Casing: {model_info['casing']}\n"
        summary += f"Numbers: {model_info['numbers']}\n"
        
        return summary
