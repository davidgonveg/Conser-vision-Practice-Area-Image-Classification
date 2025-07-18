"""
Taï National Park - Logging Utilities

This module provides comprehensive logging utilities for training and evaluation.
Includes structured logging, model information logging, and training configuration logging.
"""

import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
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
    
    logger.info(" MODEL INFORMATION")
    logger.info("=" * 50)
    
    # Model name
    model_name = getattr(model, 'model_name', model.__class__.__name__)
    logger.info(f" Model: {model_name}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    logger.info(f" Parameters:")
    logger.info(f"   Total: {total_params:,}")
    logger.info(f"   Trainable: {trainable_params:,}")
    logger.info(f"   Frozen: {frozen_params:,}")
    
    # Model size estimation
    param_size = total_params * 4  # Assuming float32
    buffer_size = sum(buf.numel() for buf in model.buffers()) * 4
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)
    
    logger.info(f" Estimated model size: {model_size_mb:.2f} MB")
    
    # Model architecture summary
    logger.info(f"  Architecture:")
    for name, module in model.named_children():
        logger.info(f"   {name}: {module.__class__.__name__}")
    
    # Special attributes
    special_attrs = ['use_site_embedding', 'use_attention', 'num_classes', 'feature_dim']
    for attr in special_attrs:
        if hasattr(model, attr):
            logger.info(f" {attr}: {getattr(model, attr)}")


def log_training_config(logger: logging.Logger, config: Any, args: argparse.Namespace):
    """
    Log training configuration.
    
    Args:
        logger: Logger instance
        config: Configuration object
        args: Command line arguments
    """
    
    logger.info("  TRAINING CONFIGURATION")
    logger.info("=" * 50)
    
    # Training parameters
    logger.info(f" Training:")
    logger.info(f"   Epochs: {config.get('training.num_epochs', 'N/A')}")
    logger.info(f"   Batch size: {config.get('training.batch_size', 'N/A')}")
    logger.info(f"   Learning rate: {config.get('training.learning_rate', 'N/A')}")
    logger.info(f"   Weight decay: {config.get('training.weight_decay', 'N/A')}")
    logger.info(f"   Early stopping patience: {config.get('training.early_stopping_patience', 'N/A')}")
    
    # Data parameters
    logger.info(f" Data:")
    logger.info(f"   Data directory: {config.get('data.raw_dir', 'N/A')}")
    logger.info(f"   Image size: {config.get('image.size', 'N/A')}")
    logger.info(f"   Number of workers: {config.get('training.num_workers', 'N/A')}")
    logger.info(f"   Sampler type: {config.get('training.sampler_type', 'N/A')}")
    logger.info(f"   Aggressive augmentation: {args.aggressive_aug}")
    
    # Model parameters
    logger.info(f" Model:")
    logger.info(f"   Architecture: {config.get('model.name', 'N/A')}")
    logger.info(f"   Pretrained: {config.get('model.pretrained', 'N/A')}")
    logger.info(f"   Dropout: {config.get('model.dropout', 'N/A')}")
    logger.info(f"   Freeze backbone: {args.freeze_backbone}")
    
    # Hardware settings
    logger.info(f" Hardware:")
    logger.info(f"   Device: {args.device}")
    logger.info(f"   Mixed precision: {args.mixed_precision}")
    logger.info(f"   Compile: {args.compile}")
    
    # Logging settings
    logger.info(f" Logging:")
    logger.info(f"   W&B enabled: {args.wandb}")
    logger.info(f"   Experiment name: {args.experiment_name}")
    logger.info(f"   Log level: {args.log_level}")


def log_dataset_info(logger: logging.Logger, data_manager: Any):
    """
    Log dataset information.
    
    Args:
        logger: Logger instance
        data_manager: DataLoaderManager instance
    """
    
    logger.info(" DATASET INFORMATION")
    logger.info("=" * 50)
    
    # Dataset sizes
    train_size = len(data_manager.train_dataset)
    val_size = len(data_manager.val_dataset)
    total_size = train_size + val_size
    
    logger.info(f" Dataset sizes:")
    logger.info(f"   Training: {train_size:,} samples")
    logger.info(f"   Validation: {val_size:,} samples")
    logger.info(f"   Total: {total_size:,} samples")
    logger.info(f"   Split ratio: {train_size/total_size:.1%} / {val_size/total_size:.1%}")
    
    # Batch information
    logger.info(f" Batch information:")
    logger.info(f"   Training batches: {len(data_manager.train_loader):,}")
    logger.info(f"   Validation batches: {len(data_manager.val_loader):,}")
    
    # Class distribution
    train_dist = data_manager.train_dataset.get_class_distribution()
    logger.info(f"  Training class distribution:")
    for class_name, count in train_dist.items():
        percentage = (count / train_size) * 100
        logger.info(f"   {class_name}: {count:,} ({percentage:.1f}%)")
    
    # Site information
    train_sites = data_manager.train_dataset.get_site_distribution()
    val_sites = data_manager.val_dataset.get_site_distribution()
    
    logger.info(f" Site information:")
    logger.info(f"   Training sites: {len(train_sites)}")
    logger.info(f"   Validation sites: {len(val_sites)}")
    logger.info(f"   No site overlap: {len(set(train_sites.keys()) & set(val_sites.keys())) == 0}")


def log_experiment_start(logger: logging.Logger, args: argparse.Namespace):
    """
    Log experiment start information.
    
    Args:
        logger: Logger instance
        args: Command line arguments
    """
    
    logger.info(" EXPERIMENT START")
    logger.info("=" * 70)
    logger.info(f" Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f" Experiment name: {args.experiment_name or 'Unnamed'}")
    logger.info(f" Config file: {args.config}")
    logger.info(f" Quick test mode: {args.quick_test}")
    logger.info(f" Dry run mode: {args.dry_run}")
    
    # System information
    logger.info(f" System:")
    logger.info(f"   Python version: {sys.version.split()[0]}")
    logger.info(f"   PyTorch version: {torch.__version__}")
    logger.info(f"   CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"   CUDA version: {torch.version.cuda}")
        logger.info(f"   GPU: {torch.cuda.get_device_name()}")
        logger.info(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def log_experiment_end(
    logger: logging.Logger, 
    training_time: float,
    best_val_loss: float,
    best_val_acc: float,
    final_epoch: int
):
    """
    Log experiment end information.
    
    Args:
        logger: Logger instance
        training_time: Total training time in seconds
        best_val_loss: Best validation loss achieved
        best_val_acc: Best validation accuracy achieved
        final_epoch: Final epoch number
    """
    
    logger.info(" EXPERIMENT COMPLETE")
    logger.info("=" * 70)
    logger.info(f" End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Total training time: {training_time:.2f} seconds ({training_time/3600:.2f} hours)")
    logger.info(f" Best validation loss: {best_val_loss:.6f}")
    logger.info(f" Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f" Final epoch: {final_epoch}")
    logger.info(f" Average time per epoch: {training_time/final_epoch:.2f} seconds")


def log_checkpoint_save(logger: logging.Logger, path: Path, is_best: bool = False):
    """
    Log checkpoint save information.
    
    Args:
        logger: Logger instance
        path: Path to saved checkpoint
        is_best: Whether this is the best model
    """
    
    checkpoint_size = path.stat().st_size / (1024 * 1024)  # MB
    
    if is_best:
        logger.info(f" Best model checkpoint saved: {path}")
    else:
        logger.info(f" Checkpoint saved: {path}")
    
    logger.info(f"   Size: {checkpoint_size:.2f} MB")


def log_epoch_summary(
    logger: logging.Logger,
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    val_log_loss: float,
    epoch_time: float,
    lr: float,
    is_best: bool = False
):
    """
    Log epoch summary.
    
    Args:
        logger: Logger instance
        epoch: Current epoch
        train_loss: Training loss
        train_acc: Training accuracy
        val_loss: Validation loss
        val_acc: Validation accuracy
        val_log_loss: Validation log loss
        epoch_time: Time taken for epoch
        lr: Current learning rate
        is_best: Whether this is the best epoch
    """
    
    status = " BEST" if is_best else "Emoji de gráficas"
    
    logger.info(f"{status} Epoch {epoch:3d} | "
                f"Train: {train_loss:.4f}/{train_acc:.4f} | "
                f"Val: {val_loss:.4f}/{val_acc:.4f} | "
                f"LogLoss: {val_log_loss:.4f} | "
                f"Time: {epoch_time:.1f}s | "
                f"LR: {lr:.2e}")


def log_class_performance(
    logger: logging.Logger,
    class_metrics: Dict[str, Dict[str, float]],
    class_names: Optional[List[str]] = None
):
    """
    Log class-wise performance metrics.
    
    Args:
        logger: Logger instance
        class_metrics: Dictionary with class-wise metrics
        class_names: List of class names (optional)
    """
    
    if class_names is None:
        class_names = [
            'antelope_duiker', 'bird', 'blank', 'civet_genet',
            'hog', 'leopard', 'monkey_prosimian', 'rodent'
        ]
    
    logger.info("  CLASS-WISE PERFORMANCE")
    logger.info("-" * 50)
    logger.info(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    logger.info("-" * 50)
    
    for class_name in class_names:
        if class_name in class_metrics:
            metrics = class_metrics[class_name]
            logger.info(f"{class_name:<15} "
                       f"{metrics['precision']:<10.3f} "
                       f"{metrics['recall']:<10.3f} "
                       f"{metrics['f1_score']:<10.3f}")


def log_site_performance(
    logger: logging.Logger,
    site_metrics: Dict[str, float],
    top_n: int = 5
):
    """
    Log site-wise performance metrics.
    
    Args:
        logger: Logger instance
        site_metrics: Dictionary with site-wise accuracy
        top_n: Number of top/bottom sites to show
    """
    
    if not site_metrics:
        return
    
    # Sort sites by accuracy
    sorted_sites = sorted(site_metrics.items(), key=lambda x: x[1], reverse=True)
    
    logger.info(" SITE-WISE PERFORMANCE")
    logger.info("-" * 40)
    
    # Top performing sites
    logger.info(f" Top {top_n} sites:")
    for site, acc in sorted_sites[:top_n]:
        logger.info(f"   {site}: {acc:.3f}")
    
    # Bottom performing sites
    logger.info(f"  Bottom {top_n} sites:")
    for site, acc in sorted_sites[-top_n:]:
        logger.info(f"   {site}: {acc:.3f}")
    
    # Overall site statistics
    accuracies = list(site_metrics.values())
    logger.info(f" Site accuracy statistics:")
    logger.info(f"   Mean: {sum(accuracies)/len(accuracies):.3f}")
    logger.info(f"   Std: {torch.std(torch.tensor(accuracies)):.3f}")
    logger.info(f"   Min: {min(accuracies):.3f}")
    logger.info(f"   Max: {max(accuracies):.3f}")


def create_experiment_log(
    output_dir: Path,
    config: Dict[str, Any],
    args: argparse.Namespace,
    model_info: Dict[str, Any]
) -> Path:
    """
    Create a comprehensive experiment log file.
    
    Args:
        output_dir: Output directory
        config: Configuration dictionary
        args: Command line arguments
        model_info: Model information
        
    Returns:
        Path to the created log file
    """
    
    log_data = {
        'experiment_info': {
            'name': args.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'config_file': args.config,
            'quick_test': args.quick_test,
            'dry_run': args.dry_run
        },
        'configuration': config,
        'arguments': vars(args),
        'model_info': model_info,
        'system_info': {
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None
        }
    }
    
    log_file = output_dir / 'experiment_log.json'
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2, default=str)
    
    return log_file


def log_gpu_memory_usage(logger: logging.Logger):
    """
    Log GPU memory usage information.
    
    Args:
        logger: Logger instance
    """
    
    if not torch.cuda.is_available():
        return
    
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3   # GB
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    logger.info(f"  GPU Memory Usage:")
    logger.info(f"   Allocated: {allocated:.2f} GB")
    logger.info(f"   Reserved: {reserved:.2f} GB")
    logger.info(f"   Max allocated: {max_allocated:.2f} GB")


def setup_file_logging(log_dir: Path, experiment_name: str) -> Dict[str, logging.Logger]:
    """
    Setup multiple file loggers for different purposes.
    
    Args:
        log_dir: Directory for log files
        experiment_name: Name of the experiment
        
    Returns:
        Dictionary of loggers
    """
    
    loggers = {}
    
    # Main training logger
    main_logger = setup_logging(
        log_file=log_dir / f"{experiment_name}_training.log",
        level='INFO'
    )
    loggers['main'] = main_logger
    
    # Debug logger
    debug_logger = setup_logging(
        log_file=log_dir / f"{experiment_name}_debug.log",
        level='DEBUG'
    )
    loggers['debug'] = debug_logger
    
    # Metrics logger
    metrics_logger = setup_logging(
        log_file=log_dir / f"{experiment_name}_metrics.log",
        level='INFO'
    )
    loggers['metrics'] = metrics_logger
    
    return loggers