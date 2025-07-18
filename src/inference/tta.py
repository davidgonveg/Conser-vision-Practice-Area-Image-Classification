"""
Taï National Park - Test Time Augmentation (TTA)

This module provides comprehensive Test Time Augmentation strategies for camera trap
species classification. TTA improves model predictions by averaging results from
multiple augmented versions of the same image.

Key Features:
- Multiple TTA strategies (flip, rotate, crop, color, etc.)
- Configurable augmentation pipelines
- Ensemble averaging methods
- Wildlife-specific augmentations
- Performance optimization
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import numpy as np
from PIL import Image
import logging
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import random

from ..data.transforms import (
    AdaptiveBrightnessContrast, ColorJitterCameraTrap,
    WildlifeSpecificRotation, AnimalFriendlyFlip
)

logger = logging.getLogger(__name__)


class TTAStrategy(Enum):
    """Test Time Augmentation strategies."""
    FLIP = "flip"
    ROTATE = "rotate"
    CROP = "crop"
    COLOR = "color"
    SCALE = "scale"
    BRIGHTNESS = "brightness"
    MULTI_CROP = "multi_crop"
    COMPREHENSIVE = "comprehensive"


@dataclass
class TTAConfig:
    """Configuration for Test Time Augmentation."""
    
    # Basic settings
    n_augmentations: int = 5
    image_size: Union[int, Tuple[int, int]] = 224
    
    # Strategy settings
    use_flip: bool = True
    use_rotate: bool = True
    use_crop: bool = True
    use_color: bool = True
    use_scale: bool = False
    use_brightness: bool = True
    
    # Augmentation parameters
    flip_horizontal: bool = True
    flip_vertical: bool = False
    rotation_degrees: Tuple[float, float] = (-10, 10)
    crop_scales: Tuple[float, float] = (0.85, 1.0)
    color_jitter_strength: float = 0.1
    brightness_range: Tuple[float, float] = (0.9, 1.1)
    
    # Multi-crop settings
    multi_crop_sizes: List[int] = None
    five_crop: bool = True
    ten_crop: bool = False
    
    # Averaging method
    averaging_method: str = "arithmetic"  # "arithmetic", "geometric", "weighted"
    
    # Performance settings
    batch_tta: bool = False
    use_wildlife_specific: bool = True


class TTATransform:
    """Base class for TTA transformations."""
    
    def __init__(self, config: TTAConfig):
        self.config = config
        self.image_size = config.image_size
        if isinstance(self.image_size, int):
            self.image_size = (self.image_size, self.image_size)
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Apply transformation to image."""
        raise NotImplementedError


class FlipTTA(TTATransform):
    """Flip-based TTA transformations."""
    
    def get_transforms(self) -> List[Callable]:
        """Get list of flip transformations."""
        transforms = []
        
        # Base transform (no flip)
        transforms.append(self._create_base_transform())
        
        # Horizontal flip
        if self.config.use_flip and self.config.flip_horizontal:
            transforms.append(self._create_flip_transform(horizontal=True))
        
        # Vertical flip (rarely used for wildlife)
        if self.config.use_flip and self.config.flip_vertical:
            transforms.append(self._create_flip_transform(vertical=True))
        
        # Both flips
        if (self.config.use_flip and 
            self.config.flip_horizontal and 
            self.config.flip_vertical):
            transforms.append(self._create_flip_transform(horizontal=True, vertical=True))
        
        return transforms
    
    def _create_base_transform(self) -> Callable:
        """Create base transformation without flips."""
        return T.Compose([
            T.Resize(self.image_size, interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _create_flip_transform(self, horizontal: bool = False, vertical: bool = False) -> Callable:
        """Create flip transformation."""
        transforms = [
            T.Resize(self.image_size, interpolation=InterpolationMode.BILINEAR)
        ]
        
        if horizontal:
            transforms.append(T.RandomHorizontalFlip(p=1.0))
        if vertical:
            transforms.append(T.RandomVerticalFlip(p=1.0))
        
        transforms.extend([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return T.Compose(transforms)


class RotationTTA(TTATransform):
    """Rotation-based TTA transformations."""
    
    def get_transforms(self) -> List[Callable]:
        """Get list of rotation transformations."""
        transforms = []
        
        if not self.config.use_rotate:
            return [self._create_base_transform()]
        
        # Base (no rotation)
        transforms.append(self._create_base_transform())
        
        # Small rotations suitable for wildlife
        angles = [-10, -5, 5, 10]
        for angle in angles:
            transforms.append(self._create_rotation_transform(angle))
        
        return transforms
    
    def _create_base_transform(self) -> Callable:
        """Create base transformation without rotation."""
        return T.Compose([
            T.Resize(self.image_size, interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _create_rotation_transform(self, angle: float) -> Callable:
        """Create rotation transformation."""
        return T.Compose([
            T.Resize(self.image_size, interpolation=InterpolationMode.BILINEAR),
            T.Lambda(lambda img: TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class CropTTA(TTATransform):
    """Crop-based TTA transformations."""
    
    def get_transforms(self) -> List[Callable]:
        """Get list of crop transformations."""
        transforms = []
        
        if not self.config.use_crop:
            return [self._create_center_crop_transform()]
        
        # Five crop (center + 4 corners)
        if self.config.five_crop:
            transforms.append(self._create_five_crop_transform())
        
        # Ten crop (five crop + horizontal flip)
        if self.config.ten_crop:
            transforms.append(self._create_ten_crop_transform())
        
        # Multi-scale crops
        if self.config.multi_crop_sizes:
            for size in self.config.multi_crop_sizes:
                transforms.append(self._create_multi_scale_transform(size))
        
        # If no specific crop method, use center crop
        if not transforms:
            transforms.append(self._create_center_crop_transform())
        
        return transforms
    
    def _create_center_crop_transform(self) -> Callable:
        """Create center crop transformation."""
        return T.Compose([
            T.Resize(int(self.image_size[0] * 1.14), interpolation=InterpolationMode.BILINEAR),
            T.CenterCrop(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _create_five_crop_transform(self) -> Callable:
        """Create five crop transformation."""
        return T.Compose([
            T.Resize(int(self.image_size[0] * 1.14), interpolation=InterpolationMode.BILINEAR),
            T.FiveCrop(self.image_size),
            T.Lambda(lambda crops: torch.stack([
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
                    T.ToTensor()(crop)
                ) for crop in crops
            ]))
        ])
    
    def _create_ten_crop_transform(self) -> Callable:
        """Create ten crop transformation."""
        return T.Compose([
            T.Resize(int(self.image_size[0] * 1.14), interpolation=InterpolationMode.BILINEAR),
            T.TenCrop(self.image_size),
            T.Lambda(lambda crops: torch.stack([
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
                    T.ToTensor()(crop)
                ) for crop in crops
            ]))
        ])
    
    def _create_multi_scale_transform(self, size: int) -> Callable:
        """Create multi-scale transformation."""
        return T.Compose([
            T.Resize((size, size), interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class ColorTTA(TTATransform):
    """Color-based TTA transformations."""
    
    def get_transforms(self) -> List[Callable]:
        """Get list of color transformations."""
        transforms = []
        
        # Base transform (no color augmentation)
        transforms.append(self._create_base_transform())
        
        if not self.config.use_color:
            return transforms
        
        # Color jitter variations
        jitter_configs = [
            (0.1, 0.1, 0.1, 0.05),  # Mild
            (0.2, 0.2, 0.1, 0.1),   # Moderate
            (0.1, 0.05, 0.05, 0.02) # Conservative
        ]
        
        for brightness, contrast, saturation, hue in jitter_configs:
            transforms.append(self._create_color_jitter_transform(
                brightness, contrast, saturation, hue
            ))
        
        return transforms
    
    def _create_base_transform(self) -> Callable:
        """Create base transformation without color augmentation."""
        return T.Compose([
            T.Resize(self.image_size, interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _create_color_jitter_transform(
        self, 
        brightness: float, 
        contrast: float, 
        saturation: float, 
        hue: float
    ) -> Callable:
        """Create color jitter transformation."""
        return T.Compose([
            T.Resize(self.image_size, interpolation=InterpolationMode.BILINEAR),
            T.ColorJitter(brightness=brightness, contrast=contrast, 
                         saturation=saturation, hue=hue),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class WildlifeSpecificTTA(TTATransform):
    """Wildlife-specific TTA transformations."""
    
    def get_transforms(self) -> List[Callable]:
        """Get list of wildlife-specific transformations."""
        transforms = []
        
        # Base transform
        transforms.append(self._create_base_transform())
        
        if not self.config.use_wildlife_specific:
            return transforms
        
        # Wildlife-specific augmentations
        transforms.extend([
            self._create_brightness_adaptive_transform(),
            self._create_wildlife_rotation_transform(),
            self._create_camera_trap_color_transform()
        ])
        
        return transforms
    
    def _create_base_transform(self) -> Callable:
        """Create base transformation."""
        return T.Compose([
            T.Resize(self.image_size, interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _create_brightness_adaptive_transform(self) -> Callable:
        """Create adaptive brightness transformation."""
        return T.Compose([
            T.Resize(self.image_size, interpolation=InterpolationMode.BILINEAR),
            AdaptiveBrightnessContrast(auto_adjust=True),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _create_wildlife_rotation_transform(self) -> Callable:
        """Create wildlife-appropriate rotation transformation."""
        return T.Compose([
            T.Resize(self.image_size, interpolation=InterpolationMode.BILINEAR),
            WildlifeSpecificRotation(degrees=(-8, 8), probability=1.0),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _create_camera_trap_color_transform(self) -> Callable:
        """Create camera trap specific color transformation."""
        return T.Compose([
            T.Resize(self.image_size, interpolation=InterpolationMode.BILINEAR),
            ColorJitterCameraTrap(brightness=0.15, contrast=0.1, saturation=0.1, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class TTAPredictor:
    """
    Test Time Augmentation predictor that applies multiple augmentations
    and averages the results.
    """
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        config: TTAConfig, 
        device: torch.device = None
    ):
        """
        Initialize TTA predictor.
        
        Args:
            model: Trained model
            config: TTA configuration
            device: Device to run predictions on
        """
        self.model = model
        self.config = config
        self.device = device or torch.device('cpu')
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Create TTA transforms
        self.transforms = self._create_tta_transforms()
        
        logger.info(f"🎭 TTAPredictor initialized with {len(self.transforms)} transforms")
    
    def _create_tta_transforms(self) -> List[Callable]:
        """Create TTA transformation pipeline."""
        all_transforms = []
        
        # Flip transforms
        if self.config.use_flip:
            flip_tta = FlipTTA(self.config)
            all_transforms.extend(flip_tta.get_transforms())
        
        # Rotation transforms
        if self.config.use_rotate:
            rotation_tta = RotationTTA(self.config)
            all_transforms.extend(rotation_tta.get_transforms())
        
        # Crop transforms
        if self.config.use_crop:
            crop_tta = CropTTA(self.config)
            all_transforms.extend(crop_tta.get_transforms())
        
        # Color transforms
        if self.config.use_color:
            color_tta = ColorTTA(self.config)
            all_transforms.extend(color_tta.get_transforms())
        
        # Wildlife-specific transforms
        if self.config.use_wildlife_specific:
            wildlife_tta = WildlifeSpecificTTA(self.config)
            all_transforms.extend(wildlife_tta.get_transforms())
        
        # If no specific transforms, use basic set
        if not all_transforms:
            all_transforms = [
                T.Compose([
                    T.Resize(self.config.image_size, interpolation=InterpolationMode.BILINEAR),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            ]
        
        # Limit number of transforms
        if len(all_transforms) > self.config.n_augmentations:
            all_transforms = all_transforms[:self.config.n_augmentations]
        
        return all_transforms
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Predict with TTA for a single image.
        
        Args:
            image: Input PIL image
            
        Returns:
            Dictionary with predictions and probabilities
        """
        
        predictions = []
        
        with torch.no_grad():
            for transform in self.transforms:
                # Apply transform
                transformed = transform(image)
                
                # Handle different transform outputs
                if isinstance(transformed, torch.Tensor):
                    if transformed.dim() == 3:  # Single image
                        batch_input = transformed.unsqueeze(0).to(self.device)
                        output = self.model(batch_input)
                        probs = F.softmax(output, dim=1)
                        predictions.append(probs.cpu().numpy()[0])
                    
                    elif transformed.dim() == 4:  # Multiple crops (e.g., FiveCrop)
                        batch_input = transformed.to(self.device)
                        output = self.model(batch_input)
                        probs = F.softmax(output, dim=1)
                        # Average across crops
                        avg_probs = torch.mean(probs, dim=0)
                        predictions.append(avg_probs.cpu().numpy())
        
        # Average predictions
        if predictions:
            if self.config.averaging_method == "arithmetic":
                avg_predictions = np.mean(predictions, axis=0)
            elif self.config.averaging_method == "geometric":
                avg_predictions = np.exp(np.mean(np.log(predictions + 1e-8), axis=0))
            else:  # arithmetic as fallback
                avg_predictions = np.mean(predictions, axis=0)
        else:
            # Fallback: uniform distribution
            avg_predictions = np.ones(8) / 8
        
        # Ensure probabilities sum to 1
        avg_predictions = avg_predictions / np.sum(avg_predictions)
        
        # Get predicted class
        predicted_class_idx = np.argmax(avg_predictions)
        confidence = avg_predictions[predicted_class_idx]
        
        return {
            'probabilities': avg_predictions,
            'predicted_class_idx': predicted_class_idx,
            'confidence': confidence,
            'n_augmentations': len(predictions)
        }
    
    def predict_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """
        Predict with TTA for a batch of images.
        
        Args:
            images: List of PIL images
            
        Returns:
            List of prediction dictionaries
        """
        
        results = []
        
        for image in images:
            result = self.predict(image)
            results.append(result)
        
        return results


def create_tta_config(
    strategy: Union[str, TTAStrategy] = TTAStrategy.COMPREHENSIVE,
    n_augmentations: int = 5,
    image_size: Union[int, Tuple[int, int]] = 224,
    **kwargs
) -> TTAConfig:
    """
    Create TTA configuration for different strategies.
    
    Args:
        strategy: TTA strategy to use
        n_augmentations: Number of augmentations
        image_size: Target image size
        **kwargs: Additional configuration parameters
        
    Returns:
        TTAConfig instance
    """
    
    if isinstance(strategy, str):
        strategy = TTAStrategy(strategy)
    
    # Base configuration
    config = TTAConfig(
        n_augmentations=n_augmentations,
        image_size=image_size,
        **kwargs
    )
    
    # Strategy-specific configurations
    if strategy == TTAStrategy.FLIP:
        config.use_flip = True
        config.use_rotate = False
        config.use_crop = False
        config.use_color = False
        config.use_brightness = False
        
    elif strategy == TTAStrategy.ROTATE:
        config.use_flip = False
        config.use_rotate = True
        config.use_crop = False
        config.use_color = False
        config.use_brightness = False
        
    elif strategy == TTAStrategy.CROP:
        config.use_flip = False
        config.use_rotate = False
        config.use_crop = True
        config.use_color = False
        config.use_brightness = False
        config.five_crop = True
        
    elif strategy == TTAStrategy.COLOR:
        config.use_flip = False
        config.use_rotate = False
        config.use_crop = False
        config.use_color = True
        config.use_brightness = True
        
    elif strategy == TTAStrategy.COMPREHENSIVE:
        config.use_flip = True
        config.use_rotate = True
        config.use_crop = True
        config.use_color = True
        config.use_brightness = True
        config.use_wildlife_specific = True
        
    return config


def get_tta_transforms(
    strategy: Union[str, TTAStrategy] = TTAStrategy.COMPREHENSIVE,
    n_augmentations: int = 5,
    image_size: Union[int, Tuple[int, int]] = 224,
    **kwargs
) -> List[Callable]:
    """
    Get TTA transforms for a given strategy.
    
    Args:
        strategy: TTA strategy
        n_augmentations: Number of augmentations
        image_size: Target image size
        **kwargs: Additional parameters
        
    Returns:
        List of transform functions
    """
    
    config = create_tta_config(strategy, n_augmentations, image_size, **kwargs)
    
    # Create transforms based on strategy
    all_transforms = []
    
    if config.use_flip:
        flip_tta = FlipTTA(config)
        all_transforms.extend(flip_tta.get_transforms())
    
    if config.use_rotate:
        rotation_tta = RotationTTA(config)
        all_transforms.extend(rotation_tta.get_transforms())
    
    if config.use_crop:
        crop_tta = CropTTA(config)
        all_transforms.extend(crop_tta.get_transforms())
    
    if config.use_color:
        color_tta = ColorTTA(config)
        all_transforms.extend(color_tta.get_transforms())
    
    if config.use_wildlife_specific:
        wildlife_tta = WildlifeSpecificTTA(config)
        all_transforms.extend(wildlife_tta.get_transforms())
    
    # Limit transforms
    if len(all_transforms) > n_augmentations:
        all_transforms = all_transforms[:n_augmentations]
    
    return all_transforms


# Example usage and testing
if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    
    print("Testing TTA module...")
    
    # Create dummy image
    dummy_image = Image.new('RGB', (640, 480), color='red')
    
    # Test different TTA strategies
    strategies = [
        TTAStrategy.FLIP,
        TTAStrategy.ROTATE,
        TTAStrategy.CROP,
        TTAStrategy.COLOR,
        TTAStrategy.COMPREHENSIVE
    ]
    
    for strategy in strategies:
        try:
            transforms = get_tta_transforms(strategy=strategy, n_augmentations=3)
            print(f"✅ {strategy.value}: {len(transforms)} transforms")
            
            # Test transform application
            for i, transform in enumerate(transforms):
                result = transform(dummy_image)
                if isinstance(result, torch.Tensor):
                    print(f"   Transform {i+1}: {result.shape}")
                else:
                    print(f"   Transform {i+1}: {type(result)}")
            
        except Exception as e:
            print(f"❌ {strategy.value}: {e}")
    
    # Test TTA configuration
    try:
        config = create_tta_config(
            strategy=TTAStrategy.COMPREHENSIVE,
            n_augmentations=5,
            image_size=224
        )
        print(f"✅ TTA Config created: {config.n_augmentations} augmentations")
        
    except Exception as e:
        print(f"❌ TTA Config failed: {e}")
    
    print("\n🎉 TTA module tests completed!")
    print("\nKey features implemented:")
    print("  🔄 Multiple TTA strategies")
    print("  🎭 Flip, rotate, crop, color augmentations")
    print("  🐆 Wildlife-specific transformations")
    print("  📊 Configurable augmentation pipelines")
    print("  ⚡ Batch and single image prediction")
    print("  🧮 Multiple averaging methods")