#!/usr/bin/env python3
"""Main script for multilingual ASR training and evaluation."""

import argparse
import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.models.whisper_model import WhisperASRModel
from src.data.multilingual_dataset import MultilingualASRDataset, create_dataloader
from src.eval.evaluator import ASREvaluator
from src.utils.device import get_device, set_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function for ASR training and evaluation.
    
    Args:
        cfg: Hydra configuration object.
    """
    # Set seed for reproducibility
    set_seed(cfg.seed)
    
    # Set up device
    device = get_device(cfg.device)
    logger.info(f"Using device: {device}")
    
    # Print configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Initialize model
    logger.info("Initializing model...")
    model = WhisperASRModel(
        model_name=cfg.model.model_name,
        device=device,
        torch_dtype=cfg.model.torch_dtype,
        use_flash_attention=cfg.model.use_flash_attention,
        enable_language_detection=cfg.model.enable_language_detection,
        default_language=cfg.model.default_language,
        enable_punctuation=cfg.model.enable_punctuation,
        enable_casing=cfg.model.enable_casing,
        enable_numbers=cfg.model.enable_numbers,
    )
    
    # Print model info
    model_info = model.get_model_info()
    logger.info("Model Information:")
    for key, value in model_info.items():
        logger.info(f"  {key}: {value}")
    
    # Create datasets
    logger.info("Creating datasets...")
    
    train_dataset = MultilingualASRDataset(
        data_dir=cfg.data.data_dir,
        meta_file=cfg.data.meta_file,
        audio_dir=cfg.data.audio_dir,
        sample_rate=cfg.data.sample_rate,
        max_duration=cfg.data.max_duration,
        min_duration=cfg.data.min_duration,
        augmentation=cfg.data.augmentation,
        split="train",
        privacy_mode=cfg.privacy_mode,
    )
    
    val_dataset = MultilingualASRDataset(
        data_dir=cfg.data.data_dir,
        meta_file=cfg.data.meta_file,
        audio_dir=cfg.data.audio_dir,
        sample_rate=cfg.data.sample_rate,
        max_duration=cfg.data.max_duration,
        min_duration=cfg.data.min_duration,
        augmentation=None,  # No augmentation for validation
        split="val",
        privacy_mode=cfg.privacy_mode,
    )
    
    test_dataset = MultilingualASRDataset(
        data_dir=cfg.data.data_dir,
        meta_file=cfg.data.meta_file,
        audio_dir=cfg.data.audio_dir,
        sample_rate=cfg.data.sample_rate,
        max_duration=cfg.data.max_duration,
        min_duration=cfg.data.min_duration,
        augmentation=None,  # No augmentation for testing
        split="test",
        privacy_mode=cfg.privacy_mode,
    )
    
    # Print dataset statistics
    logger.info("Dataset Statistics:")
    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(f"  Validation samples: {len(val_dataset)}")
    logger.info(f"  Test samples: {len(test_dataset)}")
    
    # Create evaluator
    evaluator = ASREvaluator(model, device=device, seed=cfg.seed)
    
    # Run evaluation on test set
    logger.info("Running evaluation on test set...")
    
    test_results = evaluator.evaluate_dataset(
        test_dataset,
        batch_size=1,
        return_predictions=True,
        save_results=f"{cfg.output_dir}/test_results.json"
    )
    
    # Print results
    logger.info("Test Results:")
    logger.info(f"  WER: {test_results['wer']:.4f}")
    logger.info(f"  CER: {test_results['cer']:.4f}")
    logger.info(f"  Token Accuracy: {test_results['token_accuracy']:.4f}")
    logger.info(f"  Mean Latency: {test_results['latency']['mean_latency']:.4f}s")
    logger.info(f"  Mean RTF: {test_results['rtf']['mean_rtf']:.4f}")
    
    # Create leaderboard
    leaderboard = evaluator.create_leaderboard(
        test_results,
        save_path=f"{cfg.output_dir}/leaderboard.csv"
    )
    
    logger.info("Leaderboard:")
    logger.info(leaderboard.to_string(index=False))
    
    # Save model info
    import json
    with open(f"{cfg.output_dir}/model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)
    
    logger.info("Evaluation completed successfully!")


def evaluate_single_file(
    audio_path: str,
    model_name: str = "openai/whisper-base",
    device: str = "auto",
    language: str = None,
    return_timestamps: bool = False,
) -> None:
    """Evaluate a single audio file.
    
    Args:
        audio_path: Path to audio file.
        model_name: Model name to use.
        device: Device to run on.
        language: Language code (optional).
        return_timestamps: Whether to return timestamps.
    """
    # Set up device
    device = get_device(device)
    
    # Initialize model
    model = WhisperASRModel(
        model_name=model_name,
        device=device,
        enable_language_detection=True,
    )
    
    # Create evaluator
    evaluator = ASREvaluator(model, device=device)
    
    # Evaluate file
    result = evaluator.evaluate_file(
        audio_path,
        language=language,
        return_timestamps=return_timestamps
    )
    
    # Print results
    print("Evaluation Results:")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Detected Language: {result['detected_language']}")
    print(f"  Latency: {result['latency']:.4f}s")
    
    if return_timestamps and 'timestamps' in result:
        print("  Timestamps:")
        for ts in result['timestamps']:
            print(f"    {ts['start']:.2f}s - {ts['end']:.2f}s: {ts['word']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilingual ASR Training and Evaluation")
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "single"],
        default="eval",
        help="Mode to run in"
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        help="Path to audio file for single file evaluation"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/whisper-base",
        help="Model name to use"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on"
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Language code for single file evaluation"
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Return timestamps for single file evaluation"
    )
    
    args = parser.parse_args()
    
    if args.mode == "single":
        if not args.audio_path:
            print("Error: --audio_path is required for single file evaluation")
            sys.exit(1)
        
        evaluate_single_file(
            args.audio_path,
            args.model_name,
            args.device,
            args.language,
            args.timestamps
        )
    else:
        # Run main function with Hydra
        main()
