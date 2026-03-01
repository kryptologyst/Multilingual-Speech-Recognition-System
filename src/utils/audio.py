"""Audio processing utilities for speech recognition."""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from pydub import AudioSegment


def load_audio(
    file_path: Union[str, Path],
    sample_rate: int = 16000,
    mono: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, int]:
    """Load audio file and return waveform and sample rate.
    
    Args:
        file_path: Path to audio file.
        sample_rate: Target sample rate for resampling.
        mono: Whether to convert to mono.
        normalize: Whether to normalize audio.
        
    Returns:
        Tuple of (waveform, sample_rate).
        
    Raises:
        FileNotFoundError: If audio file doesn't exist.
        ValueError: If audio file format is not supported.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    try:
        # Use librosa for robust audio loading
        waveform, sr = librosa.load(
            str(file_path),
            sr=sample_rate,
            mono=mono,
            res_type="kaiser_fast"
        )
        
        if normalize:
            waveform = normalize_audio(waveform)
            
        return waveform, sr
        
    except Exception as e:
        # Fallback to pydub for unsupported formats
        try:
            audio = AudioSegment.from_file(str(file_path))
            
            if mono:
                audio = audio.set_channels(1)
            
            # Convert to target sample rate
            audio = audio.set_frame_rate(sample_rate)
            
            # Convert to numpy array
            waveform = np.array(audio.get_array_of_samples(), dtype=np.float32)
            
            if audio.channels == 2:
                waveform = waveform.reshape((-1, 2)).mean(axis=1)
            
            # Normalize
            if normalize:
                waveform = normalize_audio(waveform)
            
            return waveform, sample_rate
            
        except Exception as fallback_error:
            raise ValueError(f"Failed to load audio file {file_path}: {e}, {fallback_error}")


def normalize_audio(waveform: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Normalize audio to target dB level.
    
    Args:
        waveform: Input audio waveform.
        target_db: Target dB level for normalization.
        
    Returns:
        Normalized waveform.
    """
    if len(waveform) == 0:
        return waveform
    
    # Calculate RMS and convert to dB
    rms = np.sqrt(np.mean(waveform ** 2))
    if rms == 0:
        return waveform
    
    current_db = 20 * np.log10(rms)
    
    # Calculate gain factor
    gain_factor = 10 ** ((target_db - current_db) / 20)
    
    return waveform * gain_factor


def resample_audio(
    waveform: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample audio to target sample rate.
    
    Args:
        waveform: Input waveform.
        orig_sr: Original sample rate.
        target_sr: Target sample rate.
        
    Returns:
        Resampled waveform.
    """
    if orig_sr == target_sr:
        return waveform
    
    return librosa.resample(
        waveform,
        orig_sr=orig_sr,
        target_sr=target_sr,
        res_type="kaiser_fast"
    )


def trim_silence(
    waveform: np.ndarray,
    sample_rate: int,
    top_db: float = 20.0,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Trim silence from the beginning and end of audio.
    
    Args:
        waveform: Input waveform.
        sample_rate: Sample rate of the audio.
        top_db: Silence threshold in dB.
        frame_length: Frame length for analysis.
        hop_length: Hop length for analysis.
        
    Returns:
        Trimmed waveform.
    """
    if len(waveform) == 0:
        return waveform
    
    # Use librosa's trim function
    trimmed, _ = librosa.effects.trim(
        waveform,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length
    )
    
    return trimmed


def pad_audio(
    waveform: np.ndarray,
    target_length: int,
    pad_value: float = 0.0,
    pad_mode: str = "constant",
) -> np.ndarray:
    """Pad audio to target length.
    
    Args:
        waveform: Input waveform.
        target_length: Target length in samples.
        pad_value: Value to pad with.
        pad_mode: Padding mode ('constant', 'reflect', 'edge').
        
    Returns:
        Padded waveform.
    """
    if len(waveform) >= target_length:
        return waveform[:target_length]
    
    pad_length = target_length - len(waveform)
    
    if pad_mode == "constant":
        return np.pad(waveform, (0, pad_length), mode="constant", constant_values=pad_value)
    else:
        return np.pad(waveform, (0, pad_length), mode=pad_mode)


def extract_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """Extract mel spectrogram from waveform.
    
    Args:
        waveform: Input waveform.
        sample_rate: Sample rate of the audio.
        n_fft: FFT window size.
        hop_length: Hop length between windows.
        n_mels: Number of mel filter banks.
        fmin: Minimum frequency.
        fmax: Maximum frequency.
        
    Returns:
        Mel spectrogram.
    """
    if fmax is None:
        fmax = sample_rate // 2
    
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    
    # Convert to log scale
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return log_mel_spec


def apply_speed_perturbation(
    waveform: np.ndarray,
    sample_rate: int,
    speed_factor: float,
) -> np.ndarray:
    """Apply speed perturbation to audio.
    
    Args:
        waveform: Input waveform.
        sample_rate: Sample rate of the audio.
        speed_factor: Speed factor (>1.0 for faster, <1.0 for slower).
        
    Returns:
        Speed-perturbed waveform.
    """
    if speed_factor == 1.0:
        return waveform
    
    return librosa.effects.time_stretch(waveform, rate=speed_factor)


def apply_pitch_shift(
    waveform: np.ndarray,
    sample_rate: int,
    n_steps: float,
) -> np.ndarray:
    """Apply pitch shift to audio.
    
    Args:
        waveform: Input waveform.
        sample_rate: Sample rate of the audio.
        n_steps: Number of semitones to shift (positive for higher pitch).
        
    Returns:
        Pitch-shifted waveform.
    """
    if n_steps == 0:
        return waveform
    
    return librosa.effects.pitch_shift(waveform, sr=sample_rate, n_steps=n_steps)


def add_noise(
    waveform: np.ndarray,
    noise_factor: float = 0.01,
    noise_type: str = "gaussian",
) -> np.ndarray:
    """Add noise to audio waveform.
    
    Args:
        waveform: Input waveform.
        noise_factor: Noise level factor.
        noise_type: Type of noise ('gaussian', 'uniform').
        
    Returns:
        Noisy waveform.
    """
    if noise_factor == 0:
        return waveform
    
    if noise_type == "gaussian":
        noise = np.random.normal(0, noise_factor, len(waveform))
    elif noise_type == "uniform":
        noise = np.random.uniform(-noise_factor, noise_factor, len(waveform))
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")
    
    return waveform + noise


def save_audio(
    waveform: np.ndarray,
    file_path: Union[str, Path],
    sample_rate: int = 16000,
    format: str = "wav",
) -> None:
    """Save audio waveform to file.
    
    Args:
        waveform: Audio waveform to save.
        file_path: Output file path.
        sample_rate: Sample rate of the audio.
        format: Audio format ('wav', 'flac', 'mp3').
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    sf.write(
        str(file_path),
        waveform,
        sample_rate,
        format=format,
        subtype="PCM_16" if format == "wav" else None
    )


def get_audio_info(file_path: Union[str, Path]) -> Dict[str, Union[int, float]]:
    """Get audio file information.
    
    Args:
        file_path: Path to audio file.
        
    Returns:
        Dictionary with audio information.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    try:
        info = sf.info(str(file_path))
        return {
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "duration": info.duration,
            "frames": info.frames,
            "format": info.format,
            "subtype": info.subtype,
            "file_size": file_path.stat().st_size,
        }
    except Exception as e:
        # Fallback to pydub
        try:
            audio = AudioSegment.from_file(str(file_path))
            return {
                "sample_rate": audio.frame_rate,
                "channels": audio.channels,
                "duration": len(audio) / 1000.0,  # Convert ms to seconds
                "frames": len(audio.raw_data) // audio.frame_width,
                "format": file_path.suffix[1:].upper(),
                "subtype": "Unknown",
                "file_size": file_path.stat().st_size,
            }
        except Exception as fallback_error:
            raise ValueError(f"Failed to get audio info for {file_path}: {e}, {fallback_error}")


def validate_audio_file(file_path: Union[str, Path]) -> bool:
    """Validate if file is a valid audio file.
    
    Args:
        file_path: Path to audio file.
        
    Returns:
        True if valid audio file, False otherwise.
    """
    try:
        get_audio_info(file_path)
        return True
    except Exception:
        return False
