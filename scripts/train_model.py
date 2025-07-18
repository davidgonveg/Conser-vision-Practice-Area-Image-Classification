#!/usr/bin/env python3
"""
Taï National Park Species Classification - Training Script

This script handles the complete training pipeline for camera trap species classification.
It includes site-aware validation, class balancing, and comprehensive logging.

Usage:
    python scripts/train_model.py --config configs/base_config.yaml
    python scripts/train_model.py --model efficientnet_b3 --epochs 50 --batch-size 32
"""

import os
import sys
import argparse
import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config
from src.data import DataLoaderManager, get_quick_setup
from src.models.model import create_model, get_model_config
from src.training.trainer import Trainer
from src.evaluation.metrics import MetricsCalculator
from src.utils.logging_utils import setup_logging, log_model_info, log_training_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Taï Park Species Classification Model')
    
    # Configuration
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                       help='Path to configuration file')
    
    # Model settings
    parser.add_argument('--model', type=str, default=None,
                       help='Model architecture (overrides config)')
    parser.add_argument('--model-config', type=str, default=None,
                       help='Predefined model config (baseline, efficient, site_aware, etc.)')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained weights')
    parser.add_argument('--freeze-backbone', action='store_true',
                       help='Freeze backbone weights')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--weight-decay', type=float, default=None,
                       help='Weight decay (overrides config)')
    
    # Data settings
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Data directory path (overrides config)')
    parser.add_argument('--image-size', type=int, default=None,
                       help='Image size (overrides config)')
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Number of data loading workers')
    parser.add_argument('--aggressive-aug', action='store_true',
                       help='Use aggressive data augmentation')
    
    # Training strategy
    parser.add_argument('--sampler', type=str, default=None,
                       choices=['weighted', 'site_aware', 'balanced_batch'],
                       help='Sampling strategy (overrides config)')
    parser.add_argument('--class-weights', action='store_true',
                       help='Use class weights in loss function')
    parser.add_argument('--focal-loss', action='store_true',
                       help='Use focal loss instead of cross entropy')
    
    # Validation and monitoring
    parser.add_argument('--validation-sites', type=str, default=None,
                       help='Path to validation sites file')
    parser.add_argument('--early-stopping', type=int, default=None,
                       help='Early stopping patience (overrides config)')
    parser.add_argument('--save-frequency', type=int, default=None,
                       help='Model save frequency in epochs')
    
    # Logging and monitoring
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='tai-park-species',
                       help='W&B project name')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name for logging')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    # Hardware settings
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, cuda:0, etc.)')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Use mixed precision training')
    parser.add_argument('--compile', action='store_true',
                       help='Use torch.compile for faster training')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for models and logs')
    
    # Quick configs
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test with minimal epochs and small dataset')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run - setup everything but don\'t train')
    
    return parser.parse_args()


def setup_device(device_str: str = 'auto') -> torch.device:
    """Setup compute device."""
    if device_str == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f" Using GPU: {torch.cuda.get_device_name()}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            device = torch.device('cpu')
            print(" Using CPU")
    else:
        device = torch.device(device_str)
        print(f" Using specified device: {device}")
    
    return device


def create_output_directories(config: Config, experiment_name: str) -> Dict[str, Path]:
    """Create output directories for models, logs, etc."""
    
    base_output_dir = Path(config.get('output.models_dir', 'results'))
    if experiment_name:
        output_dir = base_output_dir / experiment_name
    else:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_dir = base_output_dir / f"experiment_{timestamp}"
    
    directories = {
        'models': output_dir / 'models',
        'logs': output_dir / 'logs',
        'plots': output_dir / 'plots',
        'metrics': output_dir / 'metrics'
    }
    
    # Create all directories
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return directories


def setup_logging_and_monitoring(
    args: argparse.Namespace, 
    directories: Dict[str, Path]
) -> Tuple[logging.Logger, Optional[SummaryWriter], Optional[Any]]:
    """Setup logging and monitoring tools."""
    
    # Setup basic logging
    log_file = directories['logs'] / 'training.log'
    logger = setup_logging(log_file, level=args.log_level)
    
    # Setup TensorBoard
    tb_writer = SummaryWriter(log_dir=directories['logs'] / 'tensorboard')
    
    # Setup Weights & Biases
    wandb_run = None
    if args.wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.experiment_name,
            config=vars(args),
            dir=directories['logs']
        )
        logger.info(" W&B logging enabled")
    
    return logger, tb_writer, wandb_run


def load_and_merge_config(args: argparse.Namespace) -> Config:
    """Load config file and merge with command line arguments."""
    
    # Load base config
    config = Config(args.config)
    
    # Override with command line arguments
    overrides = {}
    
    # Model settings
    if args.model:
        overrides['model.name'] = args.model
    if args.pretrained is not None:
        overrides['model.pretrained'] = args.pretrained
    
    # Training settings
    if args.epochs:
        overrides['training.num_epochs'] = args.epochs
    if args.batch_size:
        overrides['training.batch_size'] = args.batch_size
    if args.learning_rate:
        overrides['training.learning_rate'] = args.learning_rate
    if args.weight_decay:
        overrides['training.weight_decay'] = args.weight_decay
    if args.early_stopping:
        overrides['training.early_stopping_patience'] = args.early_stopping
    
    # Data settings
    if args.data_dir:
        overrides['data.raw_dir'] = args.data_dir
    if args.image_size:
        overrides['image.size'] = [args.image_size, args.image_size]
    if args.num_workers:
        overrides['training.num_workers'] = args.num_workers
    if args.sampler:
        overrides['training.sampler_type'] = args.sampler
    
    # Apply overrides
    for key, value in overrides.items():
        keys = key.split('.')
        current = config.config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    
    return config


def create_model_and_optimizer(
    config: Config, 
    args: argparse.Namespace, 
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None
) -> Tuple[nn.Module, optim.Optimizer, nn.Module]:
    """Create model, optimizer, and loss function."""
    
    # Model configuration
    if args.model_config:
        model_config = get_model_config(args.model_config)
    else:
        model_config = {
            'model_name': config.get('model.name', 'efficientnet_b3'),
            'num_classes': config.get('model.num_classes', 8),
            'pretrained': config.get('model.pretrained', True),
            'dropout': config.get('model.dropout', 0.2),
            'freeze_backbone': args.freeze_backbone
        }
    
    # Create model
    model = create_model(**model_config)
    model = model.to(device)
    
    # Compile model if requested (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        model = torch.compile(model)
        print(" Model compiled for faster training")
    
    # Create optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.909431, weight_decay=0.005)
    
    # Create loss function
    if args.focal_loss:
        from src.training.losses import FocalLoss
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        print(" Using Focal Loss")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        if class_weights is not None:
            print("  Using weighted CrossEntropyLoss")
        else:
            print(" Using standard CrossEntropyLoss")
    
    return model, optimizer, criterion


def create_data_loaders(
    config: Config, 
    args: argparse.Namespace
) -> Tuple[DataLoaderManager, torch.Tensor]:
    """Create data loaders with proper configuration."""
    
    # Data loader configuration
    data_config = {
        'data_dir': config.get('data.raw_dir', 'data/raw'),
        'batch_size': config.get('training.batch_size', 32),
        'num_workers': config.get('training.num_workers', 4),
        'image_size': config.get('image.size', [224, 224])[0],
        'validation_sites_file': args.validation_sites or 'data/processed/validation_sites.csv',
        'train_sampler_type': config.get('training.sampler_type', 'weighted'),
        'aggressive_augmentation': args.aggressive_aug,
        'pin_memory': True
    }
    
    # Quick test configuration
    if args.quick_test:
        data_config['batch_size'] = 8
        data_config['num_workers'] = 2
        print(" Quick test mode - using smaller batch size")
    
    # Create data manager
    data_manager = DataLoaderManager(**data_config)
    
    # Get class weights
    class_weights = None
    if args.class_weights:
        class_weights = data_manager.get_class_weights()
        print(f" Class weights: {class_weights}")
    
    return data_manager, class_weights


def save_training_config(
    config: Config, 
    args: argparse.Namespace, 
    directories: Dict[str, Path],
    model: nn.Module
):
    """Save training configuration and model info."""
    
    # Save merged config
    config_path = directories['models'] / 'config.yaml'
    with open(config_path, 'w') as f:
        import yaml
        yaml.dump(config.config, f, default_flow_style=False)
    
    # Save training arguments
    args_path = directories['models'] / 'args.json'
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Save model info
    model_info = {
        'model_name': getattr(model, 'model_name', 'unknown'),
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    info_path = directories['models'] / 'model_info.json'
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)


def main():
    """Main training function."""
    
    # Parse arguments
    args = parse_args()
    
    # Setup device
    device = setup_device(args.device)
    
    # Load and merge configuration
    config = load_and_merge_config(args)
    
    # Create output directories
    directories = create_output_directories(config, args.experiment_name)
    
    # Setup logging and monitoring
    logger, tb_writer, wandb_run = setup_logging_and_monitoring(args, directories)
    
    logger.info(" Starting Taï Park Species Classification Training")
    logger.info(f" Output directory: {directories['models']}")
    logger.info(f" Configuration: {args.config}")
    
    try:
        # Create data loaders
        logger.info(" Creating data loaders...")
        data_manager, class_weights = create_data_loaders(config, args)
        
        # Move class weights to device
        if class_weights is not None:
            class_weights = class_weights.to(device)
        
        # Create model, optimizer, and loss function
        logger.info(" Creating model...")
        model, optimizer, criterion = create_model_and_optimizer(
            config, args, device, class_weights
        )
        
        # Log model information
        log_model_info(logger, model)
        log_training_config(logger, config, args)
        
        # Save configuration
        save_training_config(config, args, directories, model)
        
        # Create scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.72)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
            output_dir=directories['models'],
            logger=logger,
            tb_writer=tb_writer,
            wandb_run=wandb_run,
            mixed_precision=args.mixed_precision
        )
        
        # Congelar capas excepto layer4
        for name, param in model.named_parameters():
            if 'layer4' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        
        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume:
            start_epoch = trainer.load_checkpoint(args.resume)
            logger.info(f" Resumed training from epoch {start_epoch}")
        
        # Quick test mode
        if args.quick_test:
            logger.info(" Quick test mode - training for 2 epochs")
            config.config['training']['num_epochs'] = 2
        
        # Dry run mode
        if args.dry_run:
            logger.info(" Dry run mode - setup completed successfully")
            logger.info(f" Training samples: {len(data_manager.train_dataset)}")
            logger.info(f" Validation samples: {len(data_manager.val_dataset)}")
            logger.info(f" Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            return
        
        # Start training
        logger.info(" Starting training...")
        trainer.train(
            train_loader=data_manager.train_loader,
            val_loader=data_manager.val_loader,
            num_epochs=config.get('training.num_epochs', 50),
            start_epoch=start_epoch
        )
        
        # Training completed
        logger.info(" Training completed successfully!")
        
        # Save final model
        final_model_path = directories['models'] / 'final_model.pth'
        trainer.save_model(final_model_path)
        logger.info(f" Final model saved to: {final_model_path}")
        
    except Exception as e:
        logger.error(f" Training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    finally:
        # Cleanup
        if tb_writer:
            tb_writer.close()
        if wandb_run:
            wandb_run.finish()


if __name__ == '__main__':
    main()