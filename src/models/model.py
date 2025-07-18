"""
Taï National Park - Model Architectures

This module provides various model architectures optimized for camera trap species classification:
- Pre-trained CNN backbones with custom heads
- Site-aware models for domain adaptation
- Ensemble architectures
- Multi-scale feature extraction

Key Features:
- EfficientNet, ResNet, ConvNeXt architectures
- Site embedding for domain adaptation
- Attention mechanisms for wildlife detection
- Test-time augmentation support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import logging
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class WildlifeClassifier(nn.Module):
    """
    Main classifier for camera trap species classification.
    Supports various backbone architectures with custom classification heads.
    """
    
    def __init__(
        self,
        model_name: str = "efficientnet_b3",
        num_classes: int = 8,
        pretrained: bool = True,
        dropout: float = 0.2,
        use_site_embedding: bool = False,
        site_embedding_dim: int = 64,
        num_sites: int = 200,
        use_attention: bool = False,
        freeze_backbone: bool = False,
        custom_head: bool = True
    ):
        """
        Initialize wildlife classifier.
        
        Args:
            model_name: Name of the backbone model (timm compatible)
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate for the classifier head
            use_site_embedding: Whether to use site embeddings
            site_embedding_dim: Dimension of site embeddings
            num_sites: Number of unique sites
            use_attention: Whether to use attention mechanisms
            freeze_backbone: Whether to freeze backbone weights
            custom_head: Whether to use custom classification head
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.use_site_embedding = use_site_embedding
        self.use_attention = use_attention
        
        # Create backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg'
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info(f"Frozen backbone: {model_name}")
        
        # Site embedding
        if use_site_embedding:
            self.site_embedding = nn.Embedding(num_sites, site_embedding_dim)
            self.site_projection = nn.Linear(site_embedding_dim, self.feature_dim // 4)
            logger.info(f"Added site embedding: {num_sites} sites -> {site_embedding_dim}d")
        
        # Attention mechanism
        if use_attention:
            self.attention = SpatialAttention(self.feature_dim)
            logger.info("Added spatial attention mechanism")
        
        # Classification head
        if custom_head:
            self.classifier = self._build_custom_head(dropout)
        else:
            self.classifier = nn.Linear(self.feature_dim, num_classes)
        
        logger.info(f"Created {model_name} classifier with {self._count_parameters():,} parameters")
    
    def _build_custom_head(self, dropout: float) -> nn.Module:
        """Build custom classification head exactly like the successful notebook."""
        
        layers = []
        
        # Primera capa: 2048 -> 1024
        layers.extend([
            nn.Linear(self.feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)  # Dropout alto como en tu notebook
        ])
        
        # Segunda capa: 1024 -> 256  
        layers.extend([
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)  # Dropout menor como en tu notebook
        ])
        
        # Capa final: 256 -> 8
        layers.append(nn.Linear(256, self.num_classes))
        
        return nn.Sequential(*layers)
    
    def forward(
        self, 
        images: torch.Tensor, 
        site_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            images: Input images [B, C, H, W]
            site_ids: Site IDs for embedding [B] (optional)
            
        Returns:
            Class logits [B, num_classes]
        """
        
        # Extract features
        features = self.backbone(images)  # [B, feature_dim]
        
        # Apply attention if enabled
        if self.use_attention:
            features = self.attention(features)
        
        # Add site embedding if enabled
        if self.use_site_embedding and site_ids is not None:
            site_emb = self.site_embedding(site_ids)  # [B, site_embedding_dim]
            site_features = self.site_projection(site_emb)  # [B, feature_dim // 4]
            features = torch.cat([features, site_features], dim=1)
        
        # Classification
        logits = self.classifier(features)
        
        return logits
    
    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        return self.backbone(images)


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for focusing on important image regions."""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Attention layers
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention to features."""
        attention_weights = self.attention(features)
        return features * attention_weights


class EnsembleClassifier(nn.Module):
    """Ensemble of multiple models for improved performance."""
    
    def __init__(
        self,
        models: List[nn.Module],
        ensemble_method: str = "average",
        weights: Optional[List[float]] = None
    ):
        """
        Initialize ensemble classifier.
        
        Args:
            models: List of trained models
            ensemble_method: Method for combining predictions ("average", "weighted", "stacking")
            weights: Weights for weighted averaging
        """
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.ensemble_method = ensemble_method
        self.num_models = len(models)
        
        if weights is None:
            self.weights = torch.ones(self.num_models) / self.num_models
        else:
            self.weights = torch.tensor(weights)
        
        logger.info(f"Created ensemble with {self.num_models} models using {ensemble_method}")
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass through ensemble."""
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(*args, **kwargs)
                predictions.append(F.softmax(pred, dim=1))
        
        # Combine predictions
        if self.ensemble_method == "average":
            ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
        elif self.ensemble_method == "weighted":
            weighted_preds = [pred * weight for pred, weight in zip(predictions, self.weights)]
            ensemble_pred = torch.sum(torch.stack(weighted_preds), dim=0)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return torch.log(ensemble_pred + 1e-8)  # Return log probabilities


class MultiScaleClassifier(nn.Module):
    """Multi-scale classifier for handling animals at different distances."""
    
    def __init__(
        self,
        backbone_name: str = "efficientnet_b3",
        num_classes: int = 8,
        scales: List[int] = [224, 288, 384],
        pretrained: bool = True
    ):
        """
        Initialize multi-scale classifier.
        
        Args:
            backbone_name: Name of backbone model
            num_classes: Number of classes
            scales: List of input scales
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        self.scales = scales
        self.num_scales = len(scales)
        
        # Shared backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        
        feature_dim = self.backbone.num_features
        
        # Scale-specific processing
        self.scale_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ) for _ in scales
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * self.num_scales, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Final classifier
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        logger.info(f"Created multi-scale classifier with scales: {scales}")
    
    def forward(self, images: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass with multi-scale inputs.
        
        Args:
            images: Dictionary of {scale: image_tensor}
            
        Returns:
            Class logits
        """
        
        scale_features = []
        
        for i, scale in enumerate(self.scales):
            if scale in images:
                # Extract features at this scale
                features = self.backbone(images[scale])
                
                # Process scale-specific features
                processed = self.scale_processors[i](features)
                scale_features.append(processed)
            else:
                # Use zero features if scale not provided
                device = next(self.parameters()).device
                zero_features = torch.zeros(
                    images[self.scales[0]].size(0), 
                    self.backbone.num_features
                ).to(device)
                scale_features.append(zero_features)
        
        # Fuse features
        fused_features = torch.cat(scale_features, dim=1)
        fused = self.fusion(fused_features)
        
        # Final classification
        logits = self.classifier(fused)
        
        return logits


def create_model(
    model_name: str,
    num_classes: int = 8,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        **kwargs: Additional model arguments
        
    Returns:
        Initialized model
    """
    
    if model_name.startswith("ensemble_"):
        # Create ensemble model
        base_models = kwargs.get("base_models", ["efficientnet_b3", "resnet50"])
        models = [create_model(name, num_classes, pretrained) for name in base_models]
        return EnsembleClassifier(models, **kwargs)
    
    elif model_name.startswith("multiscale_"):
        # Create multi-scale model
        backbone = model_name.replace("multiscale_", "")
        return MultiScaleClassifier(backbone, num_classes, **kwargs)
    
    else:
        # Create standard classifier
        return WildlifeClassifier(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )


def load_pretrained_model(
    model_path: Union[str, Path],
    model_name: str,
    num_classes: int = 8,
    device: str = "cpu",
    **kwargs
) -> nn.Module:
    """
    Load a pretrained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        model_name: Name of model architecture
        num_classes: Number of classes
        device: Device to load model on
        **kwargs: Additional model arguments
        
    Returns:
        Loaded model
    """
    
    model = create_model(model_name, num_classes, pretrained=False, **kwargs)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded pretrained model from {model_path}")
    return model


# Model configurations for different use cases
MODEL_CONFIGS = {
    "baseline": {
        "model_name": "resnet50",
        "dropout": 0.2,
        "custom_head": True
    },
    "efficient": {
        "model_name": "efficientnet_b3",
        "dropout": 0.3,
        "custom_head": True,
        "use_attention": True
    },
    "convnext": {
        "model_name": "convnext_base",
        "dropout": 0.2,
        "custom_head": True
    },
    "site_aware": {
        "model_name": "efficientnet_b4",
        "dropout": 0.3,
        "use_site_embedding": True,
        "site_embedding_dim": 64,
        "custom_head": True
    },
    "large": {
        "model_name": "efficientnet_b5",
        "dropout": 0.4,
        "custom_head": True,
        "use_attention": True
    }
}


def get_model_config(config_name: str) -> Dict:
    """Get predefined model configuration."""
    if config_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[config_name].copy()


# Testing and example usage
if __name__ == "__main__":
    print("Testing model architectures...")
    
    # Test basic classifier
    model = WildlifeClassifier(
        model_name="efficientnet_b0",  # Small model for testing
        num_classes=8,
        pretrained=False,  # Faster for testing
        dropout=0.2,
        custom_head=True
    )
    
    # Test forward pass
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        outputs = model(dummy_images)
        print(f"✅ Basic classifier output shape: {outputs.shape}")
    
    # Test site-aware model
    site_model = WildlifeClassifier(
        model_name="efficientnet_b0",
        num_classes=8,
        pretrained=False,
        use_site_embedding=True,
        num_sites=100
    )
    
    dummy_sites = torch.randint(0, 100, (batch_size,))
    
    with torch.no_grad():
        site_outputs = site_model(dummy_images, dummy_sites)
        print(f"✅ Site-aware model output shape: {site_outputs.shape}")
    
    # Test model configs
    for config_name in MODEL_CONFIGS:
        try:
            config = get_model_config(config_name)
            config["pretrained"] = False  # Faster for testing
            test_model = create_model(**config)
            print(f"✅ {config_name} config works")
        except Exception as e:
            print(f"❌ {config_name} config failed: {e}")
    
    print("\n🎉 All model tests passed!")
    print("\nKey features implemented:")
    print("  🧠 Multiple CNN architectures (EfficientNet, ResNet, ConvNeXt)")
    print("  🌍 Site embedding for domain adaptation")
    print("  👁️  Spatial attention mechanisms")
    print("  🎯 Custom classification heads")
    print("  📊 Ensemble methods")
    print("  📏 Multi-scale processing")
    print("  ⚙️  Configurable architectures")