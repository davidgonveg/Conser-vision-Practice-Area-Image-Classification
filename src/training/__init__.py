"""
Taï National Park - Training Module

This module provides comprehensive training functionality for camera trap species classification.
"""

from .trainer import Trainer
from .losses import (
    FocalLoss, LabelSmoothingCrossEntropy, ClassBalancedLoss,
    AsymmetricLoss, DiceLoss, CombinedLoss, OnlineHardExampleMining,
    DistillationLoss, SupConLoss, get_loss_function, compute_class_weights
)

__all__ = [
    'Trainer',
    'FocalLoss', 
    'LabelSmoothingCrossEntropy', 
    'ClassBalancedLoss',
    'AsymmetricLoss', 
    'DiceLoss', 
    'CombinedLoss', 
    'OnlineHardExampleMining',
    'DistillationLoss', 
    'SupConLoss', 
    'get_loss_function', 
    'compute_class_weights'
]