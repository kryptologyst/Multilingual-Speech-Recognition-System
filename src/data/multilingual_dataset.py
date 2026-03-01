"""Data loading and preprocessing for multilingual ASR."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from ..utils.audio import load_audio, normalize_audio, trim_silence, apply_speed_perturbation, add_noise
from ..utils.device import set_seed

logger = logging.getLogger(__name__)


class MultilingualASRDataset(Dataset):
    """Dataset for multilingual ASR training and evaluation."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        meta_file: str = "meta.csv",
        audio_dir: str = "wav",
        sample_rate: int = 16000,
        max_duration: float = 30.0,
        min_duration: float = 0.5,
        augmentation: Optional[Dict] = None,
        split: Optional[str] = None,
        languages: Optional[List[str]] = None,
        privacy_mode: bool = True,
    ):
        """Initialize multilingual ASR dataset.
        
        Args:
            data_dir: Directory containing the dataset.
            meta_file: Name of the metadata CSV file.
            audio_dir: Name of the audio directory.
            sample_rate: Target sample rate for audio.
            max_duration: Maximum audio duration in seconds.
            min_duration: Minimum audio duration in seconds.
            augmentation: Data augmentation configuration.
            split: Dataset split ('train', 'val', 'test').
            languages: List of languages to include.
            privacy_mode: Whether to enable privacy-preserving features.
        """
        self.data_dir = Path(data_dir)
        self.meta_file = meta_file
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.augmentation = augmentation or {}
        self.split = split
        self.languages = languages
        self.privacy_mode = privacy_mode
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Filter by split
        if split:
            self.metadata = self.metadata[self.metadata['split'] == split]
        
        # Filter by languages
        if languages:
            self.metadata = self.metadata[self.metadata['language'].isin(languages)]
        
        # Filter by duration
        self.metadata = self.metadata[
            (self.metadata['duration'] >= min_duration) &
            (self.metadata['duration'] <= max_duration)
        ]
        
        logger.info(f"Loaded {len(self.metadata)} samples for split '{split}'")
        if languages:
            logger.info(f"Languages: {languages}")
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load dataset metadata.
        
        Returns:
            DataFrame with metadata.
        """
        meta_path = self.data_dir / self.meta_file
        
        if not meta_path.exists():
            # Create synthetic metadata if it doesn't exist
            logger.warning(f"Metadata file not found: {meta_path}")
            logger.info("Creating synthetic metadata for demo purposes")
            return self._create_synthetic_metadata()
        
        try:
            metadata = pd.read_csv(meta_path)
            
            # Validate required columns
            required_columns = ['id', 'path', 'transcript', 'language']
            missing_columns = [col for col in required_columns if col not in metadata.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Add duration if not present
            if 'duration' not in metadata.columns:
                metadata['duration'] = metadata['path'].apply(
                    lambda x: self._get_audio_duration(x)
                )
            
            # Add split if not present
            if 'split' not in metadata.columns:
                metadata['split'] = self._assign_splits(metadata)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            logger.info("Creating synthetic metadata for demo purposes")
            return self._create_synthetic_metadata()
    
    def _create_synthetic_metadata(self) -> pd.DataFrame:
        """Create synthetic metadata for demo purposes.
        
        Returns:
            DataFrame with synthetic metadata.
        """
        import random
        
        # Create synthetic samples
        samples = []
        languages = ['en', 'es', 'fr', 'de', 'zh', 'ja', 'ko']
        
        for i in range(100):  # Create 100 synthetic samples
            language = random.choice(languages)
            duration = random.uniform(1.0, 10.0)
            
            # Generate synthetic transcript
            transcript = self._generate_synthetic_transcript(language)
            
            sample = {
                'id': f"synthetic_{i:03d}",
                'path': f"synthetic_{i:03d}.wav",
                'transcript': transcript,
                'language': language,
                'duration': duration,
                'split': random.choice(['train', 'val', 'test']),
                'speaker': f"speaker_{i % 10}",
            }
            samples.append(sample)
        
        return pd.DataFrame(samples)
    
    def _generate_synthetic_transcript(self, language: str) -> str:
        """Generate synthetic transcript for a language.
        
        Args:
            language: Language code.
            
        Returns:
            Synthetic transcript.
        """
        transcripts = {
            'en': [
                "Hello, how are you today?",
                "The weather is nice today.",
                "I love learning new languages.",
                "Technology is advancing rapidly.",
                "Music brings people together."
            ],
            'es': [
                "Hola, ¿cómo estás hoy?",
                "El clima está bonito hoy.",
                "Me encanta aprender idiomas nuevos.",
                "La tecnología avanza rápidamente.",
                "La música une a las personas."
            ],
            'fr': [
                "Bonjour, comment allez-vous aujourd'hui?",
                "Le temps est beau aujourd'hui.",
                "J'aime apprendre de nouvelles langues.",
                "La technologie progresse rapidement.",
                "La musique rassemble les gens."
            ],
            'de': [
                "Hallo, wie geht es dir heute?",
                "Das Wetter ist heute schön.",
                "Ich liebe es, neue Sprachen zu lernen.",
                "Die Technologie entwickelt sich schnell.",
                "Musik bringt Menschen zusammen."
            ],
            'zh': [
                "你好，你今天怎么样？",
                "今天天气很好。",
                "我喜欢学习新语言。",
                "技术发展很快。",
                "音乐把人们聚集在一起。"
            ],
            'ja': [
                "こんにちは、今日はどうですか？",
                "今日は天気がいいです。",
                "新しい言語を学ぶのが好きです。",
                "技術は急速に進歩しています。",
                "音楽は人々を結びつけます。"
            ],
            'ko': [
                "안녕하세요, 오늘 어떠세요?",
                "오늘 날씨가 좋습니다.",
                "새로운 언어를 배우는 것을 좋아합니다.",
                "기술이 빠르게 발전하고 있습니다.",
                "음악은 사람들을 하나로 만듭니다."
            ]
        }
        
        import random
        return random.choice(transcripts.get(language, transcripts['en']))
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file.
        
        Args:
            audio_path: Path to audio file.
            
        Returns:
            Duration in seconds.
        """
        try:
            full_path = self.data_dir / self.audio_dir / audio_path
            if full_path.exists():
                from ..utils.audio import get_audio_info
                info = get_audio_info(full_path)
                return info['duration']
        except Exception:
            pass
        
        # Return random duration for synthetic data
        import random
        return random.uniform(1.0, 10.0)
    
    def _assign_splits(self, metadata: pd.DataFrame) -> List[str]:
        """Assign dataset splits.
        
        Args:
            metadata: Dataset metadata.
            
        Returns:
            List of split assignments.
        """
        splits = []
        for _, row in metadata.iterrows():
            # Simple split assignment based on ID
            if hash(row['id']) % 10 < 8:
                splits.append('train')
            elif hash(row['id']) % 10 < 9:
                splits.append('val')
            else:
                splits.append('test')
        
        return splits
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[str, np.ndarray, float]]:
        """Get dataset item.
        
        Args:
            idx: Item index.
            
        Returns:
            Dictionary with audio, transcript, and metadata.
        """
        row = self.metadata.iloc[idx]
        
        # Load audio
        audio_path = self.data_dir / self.audio_dir / row['path']
        
        if audio_path.exists():
            waveform, sr = load_audio(audio_path, self.sample_rate)
        else:
            # Generate synthetic audio for demo
            waveform = self._generate_synthetic_audio(row['duration'])
        
        # Apply augmentation if enabled and in training mode
        if self.split == 'train' and self.augmentation.get('enabled', False):
            waveform = self._apply_augmentation(waveform)
        
        # Normalize audio
        waveform = normalize_audio(waveform)
        
        return {
            'audio': waveform,
            'transcript': row['transcript'],
            'language': row['language'],
            'duration': row['duration'],
            'id': row['id'],
            'speaker': row.get('speaker', 'unknown'),
        }
    
    def _generate_synthetic_audio(self, duration: float) -> np.ndarray:
        """Generate synthetic audio for demo purposes.
        
        Args:
            duration: Audio duration in seconds.
            
        Returns:
            Synthetic audio waveform.
        """
        # Generate a simple sine wave with some noise
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        frequency = 440.0  # A4 note
        waveform = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        # Add some noise
        noise = 0.1 * np.random.randn(len(waveform))
        waveform += noise
        
        return waveform.astype(np.float32)
    
    def _apply_augmentation(self, waveform: np.ndarray) -> np.ndarray:
        """Apply data augmentation to waveform.
        
        Args:
            waveform: Input waveform.
            
        Returns:
            Augmented waveform.
        """
        # Speed perturbation
        if self.augmentation.get('speed_perturbation', {}).get('enabled', False):
            speed_config = self.augmentation['speed_perturbation']
            speed_factor = np.random.uniform(
                speed_config.get('min_speed', 0.9),
                speed_config.get('max_speed', 1.1)
            )
            waveform = apply_speed_perturbation(waveform, self.sample_rate, speed_factor)
        
        # Noise injection
        if self.augmentation.get('noise_injection', {}).get('enabled', False):
            noise_config = self.augmentation['noise_injection']
            noise_factor = noise_config.get('noise_factor', 0.01)
            waveform = add_noise(waveform, noise_factor)
        
        return waveform
    
    def get_statistics(self) -> Dict[str, Union[int, float, Dict]]:
        """Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics.
        """
        stats = {
            'total_samples': len(self.metadata),
            'languages': self.metadata['language'].value_counts().to_dict(),
            'speakers': self.metadata['speaker'].value_counts().to_dict() if 'speaker' in self.metadata.columns else {},
            'duration_stats': {
                'mean': float(self.metadata['duration'].mean()),
                'std': float(self.metadata['duration'].std()),
                'min': float(self.metadata['duration'].min()),
                'max': float(self.metadata['duration'].max()),
            },
            'split_distribution': self.metadata['split'].value_counts().to_dict() if 'split' in self.metadata.columns else {},
        }
        
        return stats


def create_dataloader(
    dataset: MultilingualASRDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader for the dataset.
    
    Args:
        dataset: Dataset to create loader for.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data.
        num_workers: Number of worker processes.
        pin_memory: Whether to pin memory.
        
    Returns:
        DataLoader instance.
    """
    def collate_fn(batch):
        """Custom collate function for ASR data."""
        # Pad audio sequences to the same length
        max_length = max(len(item['audio']) for item in batch)
        
        padded_audio = []
        transcripts = []
        languages = []
        durations = []
        ids = []
        speakers = []
        
        for item in batch:
            # Pad audio
            audio = item['audio']
            if len(audio) < max_length:
                padding = np.zeros(max_length - len(audio))
                audio = np.concatenate([audio, padding])
            padded_audio.append(audio)
            
            transcripts.append(item['transcript'])
            languages.append(item['language'])
            durations.append(item['duration'])
            ids.append(item['id'])
            speakers.append(item['speaker'])
        
        return {
            'audio': torch.tensor(np.array(padded_audio), dtype=torch.float32),
            'transcripts': transcripts,
            'languages': languages,
            'durations': torch.tensor(durations, dtype=torch.float32),
            'ids': ids,
            'speakers': speakers,
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
