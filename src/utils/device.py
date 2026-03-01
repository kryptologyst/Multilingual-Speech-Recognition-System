"""Utility functions for device management and deterministic behavior."""

import os
import random
from typing import Optional, Union

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional determinism
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(device: Union[str, torch.device] = "auto") -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device specification. If "auto", automatically select best available.
        
    Returns:
        PyTorch device object.
        
    Raises:
        RuntimeError: If specified device is not available.
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    if isinstance(device, str):
        device = torch.device(device)
    
    # Verify device availability
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    elif device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS is not available")
    
    return device


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to PyTorch dtype.
    
    Args:
        dtype_str: String representation of dtype (e.g., "float16", "float32").
        
    Returns:
        PyTorch dtype object.
        
    Raises:
        ValueError: If dtype string is not recognized.
    """
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "bool": torch.bool,
    }
    
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    
    return dtype_map[dtype_str]


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format time duration in a human-readable format.
    
    Args:
        seconds: Time duration in seconds.
        
    Returns:
        Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


def format_size(bytes_size: int) -> str:
    """Format file size in a human-readable format.
    
    Args:
        bytes_size: Size in bytes.
        
    Returns:
        Formatted size string.
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"
