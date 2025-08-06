"""
Taï National Park - Training Module

This module provides comprehensive training infrastructure for wildlife classification:
- Advanced trainer with mixed precision, gradient clipping, and monitoring
- Multiple loss functions including Focal Loss for imbalanced data
- Learning rate schedulers and optimization strategies
- Checkpoint management and experiment tracking

Key Components:
- Trainer: Main training orchestrator with full feature support
- Loss functions: Cross-entropy, Focal, Label Smoothing, Balanced losses
- Optimization utilities: Learning rate scheduling, gradient management
- Monitoring: TensorBoard, Weights & Biases integration

Usage Example:
    >>> from src.training import Trainer
    >>> from src.training.losses import FocalLoss
    >>> 
    >>> # Create trainer
    >>> trainer = Trainer(
    ...     model=model,
    ...     criterion=FocalLoss(alpha=1.0, gamma=2.0),
    ...     optimizer=optimizer,
    ...     device=device,
    ...     mixed_precision=True
    ... )
    >>> 
    >>> # Train model
    >>> history = trainer.train(train_loader, val_loader, num_epochs=50)
"""

# Core training functionality
from .trainer import Trainer

# Loss functions
try:
    from .losses import (
        FocalLoss,
        LabelSmoothingCrossEntropy,
        WeightedFocalLoss,
        BalancedCrossEntropy,
        create_loss_function,
        get_loss_config,
        LOSS_CONFIGS
    )
    _LOSSES_AVAILABLE = True
except ImportError:
    _LOSSES_AVAILABLE = False
    FocalLoss = None
    LabelSmoothingCrossEntropy = None

# Version and metadata
__version__ = "1.0.0"
__author__ = "Taï Park Species Classification Project"

# Training configurations for different scenarios
TRAINING_CONFIGS = {
    "notebook_replica": {
        "description": "Exact replica of solution.ipynb parameters",
        "model": "resnet152",
        "optimizer": "sgd",
        "learning_rate": 0.01,
        "momentum": 0.909431,
        "weight_decay": 0.005,
        "scheduler": "plateau",
        "scheduler_patience": 2,
        "scheduler_factor": 0.72,
        "batch_size": 64,
        "epochs": 5,
        "freeze_backbone": True,
        "unfreeze_layers": ["layer4", "fc"]
    },
    "competition_standard": {
        "description": "Standard configuration for competition",
        "model": "efficientnet_b4",
        "optimizer": "adamw",
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "scheduler": "cosine",
        "loss": "focal",
        "focal_gamma": 2.0,
        "class_weights": True,
        "aggressive_aug": True,
        "mixed_precision": True,
        "batch_size": 32,
        "epochs": 75,
        "sampler": "site_aware"
    },
    "rare_species_focus": {
        "description": "Optimized for rare species classification",
        "model": "efficientnet_b5",
        "optimizer": "adamw",
        "learning_rate": 0.0008,
        "weight_decay": 0.01,
        "scheduler": "cosine",
        "loss": "weighted_focal",
        "focal_gamma": 3.0,
        "class_weights": True,
        "aggressive_aug": True,
        "mixed_precision": True,
        "batch_size": 24,
        "epochs": 100,
        "sampler": "balanced_batch"
    },
    "fast_development": {
        "description": "Fast configuration for development and testing",
        "model": "efficientnet_b0",
        "optimizer": "adam",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "scheduler": "plateau",
        "loss": "cross_entropy",
        "batch_size": 32,
        "epochs": 20,
        "mixed_precision": True,
        "quick_test": True,
        "fraction": 0.2
    },
    "scientific_baseline": {
        "description": "Reproducible baseline for scientific experiments",
        "model": "resnet101",
        "optimizer": "sgd",
        "learning_rate": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "scheduler": "step",
        "scheduler_step_size": 30,
        "scheduler_gamma": 0.1,
        "loss": "cross_entropy",
        "class_weights": True,
        "deterministic": True,
        "batch_size": 32,
        "epochs": 100
    }
}


def get_training_config(config_name: str) -> dict:
    """
    Get predefined training configuration.
    
    Args:
        config_name: Name of the configuration
        
    Returns:
        Training configuration dictionary
        
    Raises:
        ValueError: If config_name is not found
    """
    if config_name not in TRAINING_CONFIGS:
        available_configs = list(TRAINING_CONFIGS.keys())
        raise ValueError(f"Unknown training config: {config_name}. Available: {available_configs}")
    
    return TRAINING_CONFIGS[config_name].copy()


def list_training_configs() -> list:
    """
    List available training configurations.
    
    Returns:
        List of available configuration names
    """
    return list(TRAINING_CONFIGS.keys())


def print_training_config(config_name: str) -> None:
    """
    Print details of a training configuration.
    
    Args:
        config_name: Name of the configuration
    """
    if config_name not in TRAINING_CONFIGS:
        print(f"❌ Unknown config: {config_name}")
        print(f"Available configs: {list_training_configs()}")
        return
    
    config = TRAINING_CONFIGS[config_name]
    print(f"🎯 Training Configuration: {config_name}")
    print(f"📝 Description: {config['description']}")
    print("-" * 50)
    
    for key, value in config.items():
        if key != 'description':
            print(f"  {key:20}: {value}")


# Helper functions for training setup
def create_trainer_from_config(
    model,
    train_config: dict,
    device: str = "cuda",
    **kwargs
) -> Trainer:
    """
    Create trainer from configuration dictionary.
    
    Args:
        model: PyTorch model
        train_config: Training configuration dictionary
        device: Device to use for training
        **kwargs: Additional trainer arguments
        
    Returns:
        Configured Trainer instance
    """
    import torch.optim as optim
    import torch.nn as nn
    
    # Create optimizer
    optimizer_name = train_config.get('optimizer', 'adam').lower()
    learning_rate = train_config.get('learning_rate', 0.001)
    weight_decay = train_config.get('weight_decay', 0.0001)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = train_config.get('momentum', 0.9)
        optimizer = optim.SGD(trainable_params, lr=learning_rate, 
                             momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Create loss function
    loss_type = train_config.get('loss', 'cross_entropy')
    class_weights = kwargs.get('class_weights')
    
    if _LOSSES_AVAILABLE:
        if loss_type == 'focal':
            gamma = train_config.get('focal_gamma', 2.0)
            criterion = FocalLoss(gamma=gamma, weight=class_weights)
        elif loss_type == 'label_smoothing':
            smoothing = train_config.get('label_smoothing', 0.1)
            criterion = LabelSmoothingCrossEntropy(smoothing=smoothing, weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Create scheduler
    scheduler = None
    scheduler_type = train_config.get('scheduler')
    if scheduler_type:
        if scheduler_type == 'plateau':
            patience = train_config.get('scheduler_patience', 5)
            factor = train_config.get('scheduler_factor', 0.5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=patience, factor=factor
            )
        elif scheduler_type == 'cosine':
            epochs = train_config.get('epochs', 50)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler_type == 'step':
            step_size = train_config.get('scheduler_step_size', 30)
            gamma = train_config.get('scheduler_gamma', 0.1)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    # Create trainer
    trainer_kwargs = {
        'mixed_precision': train_config.get('mixed_precision', False),
        'gradient_clip': train_config.get('gradient_clip', 1.0),
        **kwargs
    }
    
    return Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        **trainer_kwargs
    )


# Export key components
__all__ = [
    'Trainer',
    'FocalLoss',
    'LabelSmoothingCrossEntropy',
    'WeightedFocalLoss',
    'BalancedCrossEntropy',
    'create_loss_function',
    'get_loss_config',
    'get_training_config',
    'list_training_configs',
    'print_training_config',
    'create_trainer_from_config',
    'TRAINING_CONFIGS',
    'LOSS_CONFIGS'
]

# Remove unavailable items from __all__ if losses module is not available
if not _LOSSES_AVAILABLE:
    __all__ = [item for item in __all__ if 'Loss' not in item and 'loss' not in item.lower()]