"""
Taï National Park - Logging Utilities

This module provides comprehensive logging utilities for training and evaluation.
Includes structured logging, model information logging, and training configuration logging.
"""

import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn
from datetime import datetime
import argparse


def setup_logging(
    log_file: Optional[Union[str, Path]] = None,
    level: str = 'INFO',
    format_string: Optional[str] = None,
    include_timestamp: bool = True
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level
        format_string: Custom format string
        include_timestamp: Whether to include timestamp in logs
        
    Returns:
        Configured logger
    """
    
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # Create logger
    logger = logging.getLogger('tai_park_training')
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Default format
    if format_string is None:
        if include_timestamp:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        else:
            format_string = '%(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger


def log_model_info(logger: logging.Logger, model: nn.Module):
    """
    Log comprehensive model information.
    
    Args:
        logger: Logger instance
        model: PyTorch model
    """
    
    logger.info("🧠 MODEL INFORMATION")
    logger.info("=" * 50)
    
    # Model name
    model_name = getattr(model, 'model_name', model.__class__.__name__)
    logger.info(f"📝 Model: {model_name}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    logger.info(f"📊 Parameters:")
    logger.info(f"   Total: {total_params:,}")
    logger.info(f"   Trainable: {trainable_params:,}")
    logger.info(f"   Frozen: {frozen_params:,}")
    
    # Model size estimation
    param_size = total_params * 4  # Assuming float32
    buffer_size = sum(buf.numel() for buf in model.buffers()) * 4
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)
    
    logger.info(f"💾 Estimated model size: {model_size_mb:.2f} MB")
    
    # Model architecture summary
    logger.info(f"🏗️  Architecture:")
    for name, module in model.named_children():
        logger.info(f"   {name}: {module.__class__.__name__}")
    
    # Special attributes
    special_attrs = ['use_site_embedding', 'use_attention', 'num_classes', 'feature_dim']
    for attr in special_attrs:
        if hasattr(model, attr):
            logger.info(f"🔧 {attr}: {getattr(model, attr)}")


def log_training_config(logger: logging.Logger, config: Any, args: argparse.Namespace):
    """
    Log training configuration.
    
    Args:
        logger: Logger instance
        config: Configuration object
        args: Command line arguments
    """
    
    logger.info("⚙️  TRAINING CONFIGURATION")
    logger.info("=" * 50)
    
    # Training parameters
    logger.info(f"🎯 Training:")
    logger.info(f"   Epochs: {config.get('training.num_epochs', 'N/A')}")
    logger.info(f"   Batch size: {config.get('training.batch_size', 'N/A')}")
    logger.info(f"   Learning rate: {config.get('training.learning_rate', 'N/A')}")
    logger.info(f"   Weight decay: {config.get('training.weight_decay', 'N/A')}")
    logger.info(f"   Early stopping patience: {config.get('training.early_stopping_patience', 'N/A')}")
    
    # Data parameters
    logger.info(f"📚 Data:")
    logger.info(f"   Data directory: {config.get('data.raw_dir', 'N/A')}")
    logger.info(f"   Image size: {config.get('image.size', 'N/A')}")
    logger.info(f"   Number of workers: {config.get('training.num_workers', 'N/A')}")
    logger.info(f"   Sampler type: {config.get('training.sampler_type', 'N/A')}")
    logger.info(f"   Aggressive augmentation: {args.aggressive_aug}")
    
    # Model parameters
    logger.info(f"🧠 Model:")
    logger.info(f"   Architecture: {config.get('model.name', 'N/A')}")
    logger.info(f"   Pretrained: {config.get('model.pretrained', 'N/A')}")
    logger.info(f"   Dropout: {config.get('model.dropout', 'N/A')}")
    logger.info(f"   Freeze backbone: {args.freeze_backbone}")
    
    # Hardware settings
    logger.info(f"💻 Hardware:")
    logger.info(f"   Device: {args.device}")
    logger.info(f"   Mixed precision: {args.mixed_precision}")
    logger.info(f"   Compile: {args.compile}")
    
    # Logging settings
    logger.info(f"📝 Logging:")
    logger.info(f"   W&B enabled: {args.wandb}")
    logger.info(f"   Experiment name: {args.experiment_name}")
    logger.info(f"   Log level: {args.log_level}")