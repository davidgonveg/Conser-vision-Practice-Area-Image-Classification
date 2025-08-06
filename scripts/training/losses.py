"""
Taï National Park Species Classification - Loss Functions

This module provides various loss functions for wildlife classification,
including Focal Loss for handling class imbalance and Label Smoothing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in wildlife classification.
    
    Focal Loss is particularly useful for datasets with extreme class imbalance,
    which is common in camera trap data where some species are much more rare.
    
    Reference: Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal loss for dense object detection. ICCV, 2017.
    """
    
    def __init__(
        self, 
        alpha: float = 1.0, 
        gamma: float = 2.0, 
        weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (default: 1.0)
            gamma: Focusing parameter (default: 2.0)
            weight: Manual rescaling weight given to each class
            reduction: Specifies the reduction to apply to the output
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Focal Loss.
        
        Args:
            inputs: Predictions from model (logits) [N, C]
            targets: Ground truth labels [N]
            
        Returns:
            Computed focal loss
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        
        # Compute probabilities
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing.
    
    Label smoothing helps prevent overfitting and overconfidence in predictions,
    which can be beneficial for camera trap classification where images might
    contain multiple animals or be ambiguous.
    """
    
    def __init__(self, smoothing: float = 0.1, weight: Optional[torch.Tensor] = None):
        """
        Initialize Label Smoothing Cross Entropy Loss.
        
        Args:
            smoothing: Label smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing)
            weight: Manual rescaling weight given to each class
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.weight = weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Label Smoothing Cross Entropy.
        
        Args:
            inputs: Predictions from model (logits) [N, C]
            targets: Ground truth labels [N]
            
        Returns:
            Computed label smoothing cross entropy loss
        """
        num_classes = inputs.size(-1)
        log_preds = F.log_softmax(inputs, dim=-1)
        
        # Create smoothed targets
        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # Apply class weights if provided
        if self.weight is not None:
            weight_expanded = self.weight.unsqueeze(0).expand(targets.size(0), -1)
            true_dist = true_dist * weight_expanded
        
        return torch.mean(torch.sum(-true_dist * log_preds, dim=-1))


class WeightedFocalLoss(nn.Module):
    """
    Weighted Focal Loss combining class weighting with focal loss.
    
    This is particularly useful for wildlife classification where we have
    both class imbalance (some species are rare) and hard examples
    (some images are difficult to classify).
    """
    
    def __init__(
        self, 
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Weighted Focal Loss.
        
        Args:
            alpha: Weighting factor for each class [C]
            gamma: Focusing parameter (default: 2.0)
            reduction: Specifies the reduction to apply to the output
        """
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Weighted Focal Loss.
        
        Args:
            inputs: Predictions from model (logits) [N, C]
            targets: Ground truth labels [N]
            
        Returns:
            Computed weighted focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Get alpha for current targets
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha.gather(0, targets)
        else:
            alpha_t = 1.0
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BalancedCrossEntropy(nn.Module):
    """
    Balanced Cross Entropy Loss for handling class imbalance.
    
    Automatically computes class weights based on inverse frequency
    and applies them to standard cross entropy loss.
    """
    
    def __init__(self, beta: float = 0.9999):
        """
        Initialize Balanced Cross Entropy Loss.
        
        Args:
            beta: Hyperparameter for re-weighting (default: 0.9999)
        """
        super(BalancedCrossEntropy, self).__init__()
        self.beta = beta
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, samples_per_class: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Balanced Cross Entropy.
        
        Args:
            inputs: Predictions from model (logits) [N, C]
            targets: Ground truth labels [N]
            samples_per_class: Number of samples per class [C]
            
        Returns:
            Computed balanced cross entropy loss
        """
        effective_num = 1.0 - torch.pow(self.beta, samples_per_class)
        weights = (1.0 - self.beta) / effective_num
        weights = weights / weights.sum() * len(weights)
        
        return F.cross_entropy(inputs, targets, weight=weights.to(inputs.device))


def create_loss_function(
    loss_type: str,
    num_classes: int = 8,
    class_weights: Optional[torch.Tensor] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: Type of loss function to create
        num_classes: Number of classes
        class_weights: Optional class weights
        **kwargs: Additional arguments for specific loss functions
        
    Returns:
        Initialized loss function
    """
    
    if loss_type.lower() == 'cross_entropy':
        return nn.CrossEntropyLoss(weight=class_weights)
    
    elif loss_type.lower() == 'focal':
        alpha = kwargs.get('alpha', 1.0)
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma, weight=class_weights)
    
    elif loss_type.lower() == 'weighted_focal':
        gamma = kwargs.get('gamma', 2.0)
        return WeightedFocalLoss(alpha=class_weights, gamma=gamma)
    
    elif loss_type.lower() == 'label_smoothing':
        smoothing = kwargs.get('smoothing', 0.1)
        return LabelSmoothingCrossEntropy(smoothing=smoothing, weight=class_weights)
    
    elif loss_type.lower() == 'balanced_ce':
        beta = kwargs.get('beta', 0.9999)
        return BalancedCrossEntropy(beta=beta)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Loss function configurations for different scenarios
LOSS_CONFIGS = {
    'balanced': {
        'type': 'focal',
        'alpha': 1.0,
        'gamma': 2.0,
        'use_class_weights': True
    },
    'rare_species': {
        'type': 'weighted_focal',
        'gamma': 3.0,
        'use_class_weights': True
    },
    'smooth': {
        'type': 'label_smoothing',
        'smoothing': 0.1,
        'use_class_weights': False
    },
    'standard': {
        'type': 'cross_entropy',
        'use_class_weights': True
    }
}


def get_loss_config(config_name: str) -> dict:
    """Get predefined loss configuration."""
    if config_name not in LOSS_CONFIGS:
        raise ValueError(f"Unknown loss config: {config_name}. Available: {list(LOSS_CONFIGS.keys())}")
    return LOSS_CONFIGS[config_name]


# Example usage and testing
if __name__ == "__main__":
    # Test loss functions
    import torch
    
    # Sample data
    batch_size = 32
    num_classes = 8
    
    # Create sample inputs and targets
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test different loss functions
    losses_to_test = [
        ('CrossEntropy', nn.CrossEntropyLoss()),
        ('Focal', FocalLoss(alpha=1.0, gamma=2.0)),
        ('LabelSmoothing', LabelSmoothingCrossEntropy(smoothing=0.1)),
        ('WeightedFocal', WeightedFocalLoss(gamma=2.0))
    ]
    
    print("Testing loss functions:")
    print("-" * 40)
    
    for name, loss_fn in losses_to_test:
        loss_value = loss_fn(inputs, targets)
        print(f"{name:15}: {loss_value.item():.4f}")
    
    print("\nAll loss functions working correctly!")