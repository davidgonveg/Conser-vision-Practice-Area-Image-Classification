#!/usr/bin/env python3
"""
Taï National Park Species Classification - Advanced Training Script

This script provides comprehensive training capabilities with full configurability
for model architecture, training parameters, data loading, and monitoring.

Usage Examples:
    # Basic training
    python scripts/train_model.py
    
    # Custom model and parameters
    python scripts/train_model.py --model efficientnet_b3 --epochs 50 --batch-size 32
    
    # Notebook replica with exact parameters
    python scripts/train_model.py --model resnet152 --optimizer sgd --learning-rate 0.01 \
        --momentum 0.909431 --weight-decay 0.005 --scheduler plateau --freeze-backbone
    
    # Advanced training with all features
    python scripts/train_model.py --model efficientnet_b4 --aggressive-aug --class-weights \
        --focal-loss --mixed-precision --wandb --experiment-name "advanced_training"
"""

import os
import sys
import argparse
import logging
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Core imports with error handling
try:
    from src.data import DataLoaderManager
    from src.models.model import create_model
    from src.training.trainer import Trainer
    from src.utils.logging_utils import setup_logging
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory and src/ modules are available")
    sys.exit(1)

# Optional imports
try:
    from src.utils.config import Config
except ImportError:
    Config = None
    
try:
    from src.evaluation.metrics import MetricsCalculator
except ImportError:
    MetricsCalculator = None

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

warnings.filterwarnings("ignore", category=UserWarning)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments with comprehensive options."""
    
    parser = argparse.ArgumentParser(
        description="Taï National Park Species Classification - Advanced Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # === DATA ARGUMENTS ===
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--data-dir', type=str, default='data/raw',
                           help='Path to data directory')
    data_group.add_argument('--validation-sites', type=str,
                           help='Path to validation sites CSV file')
    data_group.add_argument('--fraction', type=float, default=1.0,
                           help='Fraction of dataset to use (0.0-1.0)')
    data_group.add_argument('--random-state', type=int, default=42,
                           help='Random seed for reproducibility')
    data_group.add_argument('--cache-data', action='store_true',
                           help='Cache preprocessed data for faster loading')
    
    # === MODEL ARGUMENTS ===
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--model', type=str, default='resnet50',
                            choices=['resnet50', 'resnet101', 'resnet152', 
                                   'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                                   'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                                   'efficientnet_b6', 'efficientnet_b7', 'convnext_base',
                                   'convnext_large', 'vit_base_patch16_224'],
                            help='Model architecture to use')
    model_group.add_argument('--pretrained', action='store_true', default=True,
                            help='Use pretrained weights')
    model_group.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                            help='Do not use pretrained weights')
    model_group.add_argument('--dropout', type=float, default=0.5,
                            help='Dropout rate')
    model_group.add_argument('--freeze-backbone', action='store_true',
                            help='Freeze backbone and only train classifier')
    model_group.add_argument('--unfreeze-layers', nargs='+', 
                            help='Specific layers to unfreeze (e.g., layer4 fc)')
    model_group.add_argument('--use-attention', action='store_true',
                            help='Use attention mechanism')
    model_group.add_argument('--use-site-embedding', action='store_true',
                            help='Use site embedding for domain adaptation')
    
    # === TRAINING ARGUMENTS ===
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--epochs', type=int, default=50,
                            help='Number of training epochs')
    train_group.add_argument('--batch-size', type=int, default=32,
                            help='Batch size for training')
    train_group.add_argument('--learning-rate', '--lr', type=float, default=0.001,
                            help='Learning rate')
    train_group.add_argument('--weight-decay', type=float, default=0.0001,
                            help='Weight decay (L2 regularization)')
    train_group.add_argument('--momentum', type=float, default=0.9,
                            help='SGD momentum')
    train_group.add_argument('--optimizer', type=str, default='adam',
                            choices=['adam', 'adamw', 'sgd', 'rmsprop'],
                            help='Optimizer to use')
    train_group.add_argument('--early-stopping', type=int, default=10,
                            help='Early stopping patience')
    train_group.add_argument('--gradient-clip', type=float, default=1.0,
                            help='Gradient clipping value')
    
    # === SCHEDULER ARGUMENTS ===
    sched_group = parser.add_argument_group('Learning Rate Scheduler')
    sched_group.add_argument('--scheduler', type=str, default='plateau',
                            choices=['plateau', 'cosine', 'step', 'exponential', 'none'],
                            help='Learning rate scheduler')
    sched_group.add_argument('--scheduler-patience', type=int, default=5,
                            help='Scheduler patience (for ReduceLROnPlateau)')
    sched_group.add_argument('--scheduler-factor', type=float, default=0.5,
                            help='Factor to reduce LR (for ReduceLROnPlateau)')
    sched_group.add_argument('--scheduler-step-size', type=int, default=10,
                            help='Step size for StepLR scheduler')
    sched_group.add_argument('--scheduler-gamma', type=float, default=0.1,
                            help='Gamma for StepLR and ExponentialLR')
    
    # === DATA LOADING ARGUMENTS ===
    loader_group = parser.add_argument_group('Data Loading')
    loader_group.add_argument('--num-workers', type=int, default=4,
                             help='Number of data loading workers')
    loader_group.add_argument('--pin-memory', action='store_true', default=True,
                             help='Pin memory for faster GPU transfer')
    loader_group.add_argument('--sampler', type=str, default='weighted',
                             choices=['random', 'weighted', 'site_aware', 'balanced_batch'],
                             help='Sampling strategy')
    loader_group.add_argument('--image-size', type=int, default=224,
                             help='Input image size')
    
    # === AUGMENTATION ARGUMENTS ===
    aug_group = parser.add_argument_group('Data Augmentation')
    aug_group.add_argument('--aggressive-aug', action='store_true',
                          help='Use aggressive data augmentation')
    aug_group.add_argument('--horizontal-flip', type=float, default=0.5,
                          help='Horizontal flip probability')
    aug_group.add_argument('--rotation', type=int, default=15,
                          help='Max rotation degrees')
    aug_group.add_argument('--brightness', type=float, default=0.2,
                          help='Brightness adjustment factor')
    aug_group.add_argument('--contrast', type=float, default=0.2,
                          help='Contrast adjustment factor')
    aug_group.add_argument('--color-jitter', type=float, default=0.2,
                          help='Color jitter intensity')
    
    # === LOSS FUNCTION ARGUMENTS ===
    loss_group = parser.add_argument_group('Loss Function')
    loss_group.add_argument('--loss', type=str, default='cross_entropy',
                           choices=['cross_entropy', 'focal', 'label_smoothing'],
                           help='Loss function to use')
    loss_group.add_argument('--focal-alpha', type=float, default=1.0,
                           help='Focal loss alpha parameter')
    loss_group.add_argument('--focal-gamma', type=float, default=2.0,
                           help='Focal loss gamma parameter')
    loss_group.add_argument('--label-smoothing', type=float, default=0.1,
                           help='Label smoothing factor')
    loss_group.add_argument('--class-weights', action='store_true',
                           help='Use class weights for imbalanced data')
    
    # === HARDWARE ARGUMENTS ===
    hw_group = parser.add_argument_group('Hardware Configuration')
    hw_group.add_argument('--device', type=str, default='auto',
                         help='Device to use (cuda, cpu, auto)')
    hw_group.add_argument('--mixed-precision', action='store_true',
                         help='Use mixed precision training')
    hw_group.add_argument('--compile', action='store_true',
                         help='Compile model with torch.compile (PyTorch 2.0+)')
    hw_group.add_argument('--deterministic', action='store_true',
                         help='Use deterministic algorithms')
    
    # === LOGGING ARGUMENTS ===
    log_group = parser.add_argument_group('Logging and Monitoring')
    log_group.add_argument('--experiment-name', type=str,
                          help='Experiment name for logging')
    log_group.add_argument('--log-level', type=str, default='INFO',
                          choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                          help='Logging level')
    log_group.add_argument('--save-frequency', type=int, default=5,
                          help='Model checkpoint save frequency (epochs)')
    log_group.add_argument('--log-frequency', type=int, default=100,
                          help='Training log frequency (batches)')
    
    # === WANDB ARGUMENTS ===
    if WANDB_AVAILABLE:
        wandb_group = parser.add_argument_group('Weights & Biases')
        wandb_group.add_argument('--wandb', action='store_true',
                                help='Enable Weights & Biases logging')
        wandb_group.add_argument('--wandb-project', type=str, default='tai-park-species',
                                help='W&B project name')
        wandb_group.add_argument('--wandb-entity', type=str,
                                help='W&B entity name')
        wandb_group.add_argument('--wandb-tags', nargs='+',
                                help='W&B tags for the run')
    
    # === UTILITY ARGUMENTS ===
    util_group = parser.add_argument_group('Utility Options')
    util_group.add_argument('--config', type=str,
                           help='Path to configuration file')
    util_group.add_argument('--resume', type=str,
                           help='Path to checkpoint to resume from')
    util_group.add_argument('--quick-test', action='store_true',
                           help='Quick test with small dataset')
    util_group.add_argument('--dry-run', action='store_true',
                           help='Dry run without actual training')
    util_group.add_argument('--profile', action='store_true',
                           help='Profile training performance')
    util_group.add_argument('--output-dir', type=str, default='results',
                           help='Output directory for results')
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup and configure the compute device."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = torch.device('cpu')
            print("💻 Using CPU")
    else:
        device = torch.device(device_arg)
        print(f"🎯 Using device: {device}")
    
    return device


def set_seeds(seed: int, deterministic: bool = False):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"🔒 Deterministic mode enabled with seed {seed}")
    else:
        torch.backends.cudnn.benchmark = True
        print(f"🎲 Seeds set to {seed}")


def create_output_directories(base_dir: str, experiment_name: Optional[str] = None) -> Dict[str, Path]:
    """Create output directory structure."""
    
    # Create experiment name if not provided
    if experiment_name is None:
        from datetime import datetime
        experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create directories
    base_path = Path(base_dir)
    directories = {
        'base': base_path,
        'models': base_path / 'models' / experiment_name,
        'logs': base_path / 'logs' / experiment_name,
        'plots': base_path / 'plots' / experiment_name,
        'predictions': base_path / 'predictions' / experiment_name
    }
    
    # Create all directories
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return directories


def load_config(config_path: Optional[str]) -> Config:
    """Load configuration from file or create default."""
    if config_path and Path(config_path).exists():
        return Config(config_path)
    else:
        # Create default config
        return Config()


def create_data_loaders(args: argparse.Namespace) -> Tuple[DataLoaderManager, Optional[torch.Tensor]]:
    """Create data loaders with specified configuration."""
    
    print("📚 Setting up data loaders...")
    
    # Data loader configuration
    data_config = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'image_size': args.image_size,
        'validation_sites_file': args.validation_sites,
        'train_sampler_type': args.sampler,
        'aggressive_augmentation': args.aggressive_aug,
        'pin_memory': args.pin_memory,
        'cache_data': args.cache_data
    }
    
    # Quick test configuration
    if args.quick_test:
        data_config['batch_size'] = min(8, args.batch_size)
        data_config['num_workers'] = 2
        print("⚡ Quick test mode - using smaller batch size")
    
    # Create data manager
    data_manager = DataLoaderManager(**data_config)
    
    # Get class weights if requested
    class_weights = None
    if args.class_weights:
        class_weights = data_manager.get_class_weights()
        print(f"⚖️  Using class weights: {class_weights}")
    
    return data_manager, class_weights


def create_model_and_components(
    args: argparse.Namespace, 
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None
) -> Tuple[nn.Module, optim.Optimizer, nn.Module, Optional[optim.lr_scheduler._LRScheduler]]:
    """Create model, optimizer, loss function, and scheduler."""
    
    print("🧠 Creating model...")
    
    # Create model
    model_kwargs = {
        'num_classes': 8,
        'pretrained': args.pretrained,
        'dropout': args.dropout,
        'use_attention': args.use_attention,
        'use_site_embedding': args.use_site_embedding
    }
    
    model = create_model(args.model, **model_kwargs)
    
    # Handle layer freezing
    if args.freeze_backbone:
        print("🧊 Freezing backbone layers...")
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze specific layers if specified
        if args.unfreeze_layers:
            for layer_name in args.unfreeze_layers:
                if hasattr(model, layer_name):
                    for param in getattr(model, layer_name).parameters():
                        param.requires_grad = True
                    print(f"🔥 Unfroze layer: {layer_name}")
                else:
                    print(f"⚠️  Layer '{layer_name}' not found in model")
        else:
            # Default: unfreeze classifier
            for name, param in model.named_parameters():
                if any(cls_name in name for cls_name in ['classifier', 'fc', 'head']):
                    param.requires_grad = True
    
    # Move model to device
    model = model.to(device)
    
    # Compile model if requested (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        print("⚡ Compiling model...")
        model = torch.compile(model)
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📊 Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(trainable_params, lr=args.learning_rate, 
                              weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(trainable_params, lr=args.learning_rate, 
                               weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(trainable_params, lr=args.learning_rate, 
                             momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'rmsprop':
        optimizer = optim.RMSprop(trainable_params, lr=args.learning_rate, 
                                 weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    print(f"🎯 Using optimizer: {args.optimizer.upper()}")
    
    # Create loss function
    try:
        from src.training.losses import FocalLoss, LabelSmoothingCrossEntropy
    except ImportError:
        print("⚠️  Custom loss functions not available, using standard losses")
        FocalLoss = None
        LabelSmoothingCrossEntropy = None
    
    if args.loss == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif args.loss == 'focal':
        if FocalLoss is not None:
            criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, weight=class_weights)
        else:
            print("⚠️  Focal loss not available, using CrossEntropy")
            criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif args.loss == 'label_smoothing':
        if hasattr(nn.CrossEntropyLoss, 'label_smoothing'):
            # PyTorch 1.10+ has native label smoothing
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
        elif LabelSmoothingCrossEntropy is not None:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing, weight=class_weights)
        else:
            print("⚠️  Label smoothing not available, using CrossEntropy")
            criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")
    
    print(f"📉 Using loss function: {args.loss}")
    
    # Create scheduler
    scheduler = None
    if args.scheduler != 'none':
        if args.scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=args.scheduler_patience, factor=args.scheduler_factor
            )
        elif args.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        elif args.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma
            )
        elif args.scheduler == 'exponential':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)
        
        print(f"📈 Using scheduler: {args.scheduler}")
    
    return model, optimizer, criterion, scheduler


def setup_logging_and_monitoring(
    args: argparse.Namespace, 
    directories: Dict[str, Path]
) -> Tuple[logging.Logger, SummaryWriter, Optional[Any]]:
    """Setup logging and monitoring systems."""
    
    # Setup basic logging
    log_file = directories['logs'] / 'training.log'
    logger = setup_logging(log_file, level=args.log_level)
    
    # Setup TensorBoard
    tb_writer = SummaryWriter(log_dir=directories['logs'] / 'tensorboard')
    
    # Setup Weights & Biases
    wandb_run = None
    if args.wandb and WANDB_AVAILABLE:
        wandb_config = vars(args)
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.experiment_name,
            tags=args.wandb_tags,
            config=wandb_config,
            dir=str(directories['logs'])
        )
        logger.info("📊 W&B logging enabled")
    elif args.wandb and not WANDB_AVAILABLE:
        logger.warning("⚠️  W&B requested but not available. Install with: pip install wandb")
    
    return logger, tb_writer, wandb_run


def log_configuration(logger: logging.Logger, args: argparse.Namespace, model: nn.Module):
    """Log comprehensive configuration information."""
    
    logger.info("🚀 TRAINING CONFIGURATION")
    logger.info("=" * 60)
    
    # Model information
    logger.info("🧠 MODEL:")
    logger.info(f"   Architecture: {args.model}")
    logger.info(f"   Pretrained: {args.pretrained}")
    logger.info(f"   Dropout: {args.dropout}")
    logger.info(f"   Freeze backbone: {args.freeze_backbone}")
    if args.unfreeze_layers:
        logger.info(f"   Unfreeze layers: {args.unfreeze_layers}")
    
    # Training configuration
    logger.info("🎯 TRAINING:")
    logger.info(f"   Epochs: {args.epochs}")
    logger.info(f"   Batch size: {args.batch_size}")
    logger.info(f"   Learning rate: {args.learning_rate}")
    logger.info(f"   Optimizer: {args.optimizer}")
    logger.info(f"   Weight decay: {args.weight_decay}")
    logger.info(f"   Early stopping: {args.early_stopping}")
    
    # Data configuration
    logger.info("📚 DATA:")
    logger.info(f"   Data directory: {args.data_dir}")
    logger.info(f"   Image size: {args.image_size}")
    logger.info(f"   Sampler: {args.sampler}")
    logger.info(f"   Fraction: {args.fraction}")
    logger.info(f"   Aggressive augmentation: {args.aggressive_aug}")
    
    # Hardware configuration
    logger.info("💻 HARDWARE:")
    logger.info(f"   Device: {args.device}")
    logger.info(f"   Mixed precision: {args.mixed_precision}")
    logger.info(f"   Compile: {args.compile}")
    logger.info(f"   Workers: {args.num_workers}")
    
    # Model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("📊 MODEL STATS:")
    logger.info(f"   Total parameters: {total_params:,}")
    logger.info(f"   Trainable parameters: {trainable_params:,}")
    logger.info(f"   Frozen parameters: {total_params - trainable_params:,}")


def save_configuration(args: argparse.Namespace, directories: Dict[str, Path]):
    """Save training configuration for reproducibility."""
    
    config_data = {
        'arguments': vars(args),
        'model_info': {
            'architecture': args.model,
            'pretrained': args.pretrained,
            'dropout': args.dropout
        },
        'training_info': {
            'optimizer': args.optimizer,
            'learning_rate': args.learning_rate,
            'scheduler': args.scheduler,
            'loss_function': args.loss
        },
        'data_info': {
            'data_dir': args.data_dir,
            'image_size': args.image_size,
            'batch_size': args.batch_size,
            'sampler': args.sampler
        }
    }
    
    # Save as JSON
    config_path = directories['models'] / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2, default=str)


def main():
    """Main training function."""
    
    print("🦁 Taï National Park Species Classification - Advanced Training")
    print("=" * 60)
    
    # Parse arguments
    args = parse_args()
    
    # Quick validation
    if args.wandb and not WANDB_AVAILABLE:
        print("⚠️  Warning: W&B requested but not installed. Install with: pip install wandb")
    
    # Setup device and seeds
    device = setup_device(args.device)
    set_seeds(args.random_state, args.deterministic)
    
    # Create output directories
    directories = create_output_directories(args.output_dir, args.experiment_name)
    print(f"📁 Output directory: {directories['models']}")
    
    # Setup logging and monitoring
    logger, tb_writer, wandb_run = setup_logging_and_monitoring(args, directories)
    
    try:
        # Log configuration
        logger.info("🚀 Starting Taï Park Species Classification Training")
        
        # Create data loaders
        data_manager, class_weights = create_data_loaders(args)
        
        # Move class weights to device
        if class_weights is not None:
            class_weights = class_weights.to(device)
        
        # Create model and components
        model, optimizer, criterion, scheduler = create_model_and_components(
            args, device, class_weights
        )
        
        # Log configuration
        log_configuration(logger, args, model)
        
        # Save configuration
        save_configuration(args, directories)
        
        # Dry run check
        if args.dry_run:
            logger.info("🏃‍♂️ Dry run completed successfully!")
            return
        
        # Create trainer
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            output_dir=directories['models'],
            logger=logger,
            tb_writer=tb_writer,
            wandb_run=wandb_run,
            mixed_precision=args.mixed_precision,
            gradient_clip=args.gradient_clip,
            save_frequency=args.save_frequency,
            log_frequency=args.log_frequency
        )
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume:
            start_epoch = trainer.load_checkpoint(Path(args.resume))
        
        # Training
        logger.info("🎯 Starting training...")
        training_start = time.time()
        
        # Quick test mode
        if args.quick_test:
            logger.info("⚡ Quick test mode - using subset of data")
            # Limit to fewer epochs for quick test
            args.epochs = min(2, args.epochs)
        
        # Train the model
        history = trainer.train(
            train_loader=data_manager.train_loader,
            val_loader=data_manager.val_loader,
            num_epochs=args.epochs,
            start_epoch=start_epoch,
            early_stopping_patience=args.early_stopping
        )
        
        training_time = time.time() - training_start
        logger.info(f"✅ Training completed in {training_time/3600:.2f} hours")
        
        # Save final model
        final_model_path = directories['models'] / 'final_model.pth'
        trainer.save_model(final_model_path)
        
        # Save training history
        history_path = directories['models'] / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Model summary
        model_summary = trainer.get_model_summary()
        logger.info("📊 FINAL RESULTS:")
        logger.info(f"   Best validation accuracy: {model_summary['best_val_acc']:.4f}")
        logger.info(f"   Best validation loss: {model_summary['best_val_loss']:.4f}")
        logger.info(f"   Total epochs trained: {model_summary['current_epoch']}")
        
        logger.info("🎉 Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("⏹️  Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training failed with error: {str(e)}")
        raise
    finally:
        # Clean up
        if tb_writer:
            tb_writer.close()
        if wandb_run:
            wandb_run.finish()


if __name__ == "__main__":
    main()