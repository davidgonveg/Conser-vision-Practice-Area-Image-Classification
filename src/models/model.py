"""
Tai Park Wildlife Model Architecture

This module replicates the exact model architecture and setup from the notebook
with ResNet152 backbone and custom classification head.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

logger = logging.getLogger(__name__)


class WildlifeClassifier(nn.Module):
    """
    Wildlife classifier with ResNet backbone exactly like notebook.
    
    Uses ResNet152 with selective fine-tuning (only layer4 unfrozen)
    and custom multi-layer classification head.
    """
    
    def __init__(
        self,
        model_name: str = 'resnet152',
        num_classes: int = 8,
        pretrained: bool = True,
        freeze_layers: bool = True,
        unfreeze_layers: Optional[List[str]] = None,
        dropout_rates: Tuple[float, float] = (0.5, 0.3),
        hidden_sizes: Tuple[int, int] = (1024, 256)
    ):
        """
        Initialize wildlife classifier exactly like notebook.
        
        Args:
            model_name: Backbone model name ('resnet152', 'resnet50', etc.)
            num_classes: Number of output classes (8 for wildlife)
            pretrained: Whether to use pretrained weights
            freeze_layers: Whether to freeze backbone layers
            unfreeze_layers: List of layer names to unfreeze (notebook uses ['layer4'])
            dropout_rates: Dropout rates for classification head layers
            hidden_sizes: Hidden layer sizes for classification head
        """
        
        super(WildlifeClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        
        # Create backbone exactly like notebook
        self.backbone = self._create_backbone()
        
        # Freeze/unfreeze layers exactly like notebook
        if freeze_layers:
            self._freeze_layers(unfreeze_layers or ['layer4'])
        
        # Create custom classification head exactly like notebook
        self.backbone.fc = self._create_classification_head(
            dropout_rates, hidden_sizes
        )
        
        logger.info(f"Created {model_name} with {num_classes} classes")
        self._log_trainable_parameters()

    def _create_backbone(self) -> nn.Module:
        """Create backbone model exactly like notebook."""
        
        if self.model_name == 'resnet152':
            if self.pretrained:
                weights = models.ResNet152_Weights.DEFAULT
                backbone = models.resnet152(weights=weights)
            else:
                backbone = models.resnet152(weights=None)
        elif self.model_name == 'resnet50':
            if self.pretrained:
                weights = models.ResNet50_Weights.DEFAULT
                backbone = models.resnet50(weights=weights)
            else:
                backbone = models.resnet50(weights=None)
        elif self.model_name == 'resnet101':
            if self.pretrained:
                weights = models.ResNet101_Weights.DEFAULT
                backbone = models.resnet101(weights=weights)
            else:
                backbone = models.resnet101(weights=None)
        elif self.model_name.startswith('efficientnet'):
            # Support for EfficientNet models
            if self.model_name == 'efficientnet_b0':
                if self.pretrained:
                    weights = models.EfficientNet_B0_Weights.DEFAULT
                    backbone = models.efficientnet_b0(weights=weights)
                else:
                    backbone = models.efficientnet_b0(weights=None)
            elif self.model_name == 'efficientnet_b3':
                if self.pretrained:
                    weights = models.EfficientNet_B3_Weights.DEFAULT
                    backbone = models.efficientnet_b3(weights=weights)
                else:
                    backbone = models.efficientnet_b3(weights=None)
            elif self.model_name == 'efficientnet_b4':
                if self.pretrained:
                    weights = models.EfficientNet_B4_Weights.DEFAULT
                    backbone = models.efficientnet_b4(weights=weights)
                else:
                    backbone = models.efficientnet_b4(weights=None)
            else:
                raise ValueError(f"Unsupported EfficientNet model: {self.model_name}")
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        return backbone

    def _freeze_layers(self, unfreeze_layers: List[str]):
        """
        Freeze layers exactly like notebook.
        
        Freezes all layers except those specified in unfreeze_layers.
        Notebook example: only 'layer4' is unfrozen.
        """
        
        # First freeze all parameters
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        
        # Then unfreeze specified layers
        for name, param in self.backbone.named_parameters():
            for unfreeze_layer in unfreeze_layers:
                if unfreeze_layer in name:
                    param.requires_grad = True
                    break
        
        logger.info(f"Unfrozen layers: {unfreeze_layers}")

    def _create_classification_head(
        self, 
        dropout_rates: Tuple[float, float],
        hidden_sizes: Tuple[int, int]
    ) -> nn.Module:
        """
        Create classification head exactly like notebook.
        
        Notebook architecture:
        - Linear(2048, 1024) + BatchNorm + ReLU + Dropout(0.5)
        - Linear(1024, 256) + BatchNorm + ReLU + Dropout(0.3)  
        - Linear(256, 8)
        """
        
        # Get input features from backbone
        if self.model_name.startswith('resnet'):
            in_features = self.backbone.fc.in_features
        elif self.model_name.startswith('efficientnet'):
            in_features = self.backbone.classifier.in_features
        else:
            raise ValueError(f"Cannot determine input features for {self.model_name}")
        
        # Create exact notebook architecture
        classification_head = nn.Sequential(
            # First layer: 2048 -> 1024 (notebook)
            nn.Linear(in_features, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rates[0]),
            
            # Second layer: 1024 -> 256 (notebook)  
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rates[1]),
            
            # Final layer: 256 -> 8 (notebook)
            nn.Linear(hidden_sizes[1], self.num_classes)
        )
        
        return classification_head

    def _log_trainable_parameters(self):
        """Log trainable parameters like notebook verification."""
        
        trainable_params = []
        frozen_params = []
        
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
            else:
                frozen_params.append(name)
        
        logger.info(f"Trainable parameters: {len(trainable_params)}")
        logger.info(f"Frozen parameters: {len(frozen_params)}")
        
        # Log first few trainable and frozen for verification
        logger.debug("First 5 trainable parameters:")
        for name in trainable_params[:5]:
            logger.debug(f"  {name}: requires_grad=True")
        
        logger.debug("First 5 frozen parameters:")
        for name in frozen_params[:5]:
            logger.debug(f"  {name}: requires_grad=False")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.backbone(x)

    def get_feature_extractor(self) -> nn.Module:
        """Get feature extractor (backbone without classification head)."""
        
        if self.model_name.startswith('resnet'):
            # Remove the final fc layer
            modules = list(self.backbone.children())[:-1]
            feature_extractor = nn.Sequential(*modules)
        elif self.model_name.startswith('efficientnet'):
            # Remove the final classifier
            feature_extractor = nn.Sequential(
                self.backbone.features,
                self.backbone.avgpool,
                nn.Flatten()
            )
        else:
            raise ValueError(f"Feature extraction not implemented for {self.model_name}")
        
        return feature_extractor

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature embeddings (before final classification layer)."""
        
        feature_extractor = self.get_feature_extractor()
        with torch.no_grad():
            embeddings = feature_extractor(x)
        
        return embeddings


def create_model(
    model_name: str = 'resnet152',
    num_classes: int = 8,
    pretrained: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    **model_kwargs
) -> WildlifeClassifier:
    """
    Create and initialize wildlife classifier exactly like notebook.
    
    Args:
        model_name: Model architecture name
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        device: Device to move model to
        **model_kwargs: Additional model arguments
        
    Returns:
        Initialized WildlifeClassifier model
    """
    
    model = WildlifeClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        **model_kwargs
    )
    
    # Move to device if specified
    if device is not None:
        if isinstance(device, str):
            device = torch.device(device)
        model = model.to(device)
        logger.info(f"Model moved to device: {device}")
    
    return model


def get_model_info(model: WildlifeClassifier) -> Dict[str, any]:
    """Get detailed model information."""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'model_name': model.model_name,
        'num_classes': model.num_classes,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': frozen_params,
        'trainable_percentage': (trainable_params / total_params) * 100,
        'pretrained': model.pretrained
    }


# Notebook-style model creation function
def create_notebook_model(device: Optional[str] = None) -> WildlifeClassifier:
    """
    Create model exactly like the notebook.
    
    Returns the exact ResNet152 setup from the notebook:
    - ResNet152 with ImageNet pretrained weights
    - Only layer4 unfrozen for fine-tuning
    - Custom 3-layer classification head
    """
    
    # Auto-detect device like notebook
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
    
    # Create model exactly like notebook
    model = create_model(
        model_name='resnet152',
        num_classes=8,
        pretrained=True,
        freeze_layers=True,
        unfreeze_layers=['layer4'],  # Notebook setting
        dropout_rates=(0.5, 0.3),    # Notebook settings
        hidden_sizes=(1024, 256),    # Notebook settings
        device=device
    )
    
    return model


# Example usage
if __name__ == "__main__":
    
    # Create model exactly like notebook
    model = create_notebook_model()
    
    # Print model info
    info = get_model_info(model)
    print(f"Model: {info['model_name']}")
    print(f"Total parameters: {info['total_parameters']:,}")
    print(f"Trainable parameters: {info['trainable_parameters']:,}")
    print(f"Trainable percentage: {info['trainable_percentage']:.1f}%")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    if torch.cuda.is_available():
        x = x.cuda()
    
    with torch.no_grad():
        output = model(x)
        print(f"Output shape: {output.shape}")  # Should be [2, 8]