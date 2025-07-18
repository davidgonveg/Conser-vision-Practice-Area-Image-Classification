"""
Taï National Park - Data Module

This module provides comprehensive data handling for camera trap species classification:
- Custom PyTorch datasets with site-based validation splits
- Specialized image transformations for wildlife photography
- Balanced data loaders with multiple sampling strategies
- Preprocessing utilities and data quality validation

Key Components:
- TaiParkDataset: Site-aware dataset with robust image loading
- Specialized transforms: Day/night adaptive, wildlife-friendly augmentations
- DataLoaderManager: Easy configuration for train/val/test data loading
- Preprocessing utilities: Validation, cleaning, and analysis tools

Usage Example:
    >>> from src.data import DataLoaderManager, get_train_transforms, get_val_transforms
    >>> 
    >>> # Quick setup for training
    >>> manager = DataLoaderManager(
    ...     data_dir="data/raw",
    ...     batch_size=32,
    ...     train_sampler_type="site_aware"
    ... )
    >>> train_loader = manager.train_loader
    >>> val_loader = manager.val_loader
    >>> 
    >>> # Or individual components
    >>> from src.data import TaiParkDataset, create_datasets
    >>> train_transform = get_train_transforms(image_size=224, aggressive=False)
    >>> train_ds, val_ds = create_datasets("data/raw", train_transform=train_transform)
"""

# Core dataset functionality
from .dataset import (
    TaiParkDataset,
    create_datasets,
    create_test_dataset
)

# Image transformations
from .transforms import (
    get_train_transforms,
    get_val_transforms,
    get_test_time_augmentation_transforms,
    visualize_transforms,
    
    # Individual transform classes for custom pipelines
    AdaptiveBrightnessContrast,
    HistogramEqualization,
    WildlifeSpecificRotation,
    CameraTrapCrop,
    AnimalFriendlyFlip,
    EnvironmentalNoise,
    ColorJitterCameraTrap,
    NightVisionSimulation
)

# Data loaders and sampling
from .data_loader import (
    DataLoaderManager,
    create_balanced_dataloader,
    create_standard_dataloader,
    wildlife_collate_fn,
    analyze_dataloader_performance,
    
    # Specialized samplers
    SiteAwareBatchSampler,
    BalancedBatchSampler
)

# Preprocessing and utilities
from .preprocessing import (
    DatasetAnalyzer,
    ImageValidator,
    CacheManager,
    preprocess_dataset,
    validate_and_clean_dataset,
    create_balanced_subset
)

# Version and metadata
__version__ = "1.0.0"
__author__ = "Taï Park Species Classification Project"

# Class names for reference
CLASS_NAMES = [
    'antelope_duiker', 'bird', 'blank', 'civet_genet',
    'hog', 'leopard', 'monkey_prosimian', 'rodent'
]

# Default configurations
DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 4

# ImageNet normalization constants (commonly used baseline)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

__all__ = [
    # Core dataset
    'TaiParkDataset',
    'create_datasets', 
    'create_test_dataset',
    
    # Transforms
    'get_train_transforms',
    'get_val_transforms',
    'get_test_time_augmentation_transforms',
    'visualize_transforms',
    'AdaptiveBrightnessContrast',
    'HistogramEqualization',
    'WildlifeSpecificRotation',
    'CameraTrapCrop',
    'AnimalFriendlyFlip',
    'EnvironmentalNoise',
    'ColorJitterCameraTrap',
    'NightVisionSimulation',
    
    # Data loaders
    'DataLoaderManager',
    'create_balanced_dataloader',
    'create_standard_dataloader',
    'wildlife_collate_fn',
    'analyze_dataloader_performance',
    'SiteAwareBatchSampler',
    'BalancedBatchSampler',
    
    # Preprocessing
    'DatasetAnalyzer',
    'ImageValidator', 
    'CacheManager',
    'preprocess_dataset',
    'validate_and_clean_dataset',
    'create_balanced_subset',
    
    # Constants and metadata
    'CLASS_NAMES',
    'DEFAULT_IMAGE_SIZE',
    'DEFAULT_BATCH_SIZE',
    'DEFAULT_NUM_WORKERS',
    'IMAGENET_MEAN',
    'IMAGENET_STD'
]


def get_quick_setup(
    data_dir: str = "data/raw",
    batch_size: int = DEFAULT_BATCH_SIZE,
    image_size: int = DEFAULT_IMAGE_SIZE,
    sampler_type: str = "weighted",
    aggressive_augmentation: bool = False,
    num_workers: int = DEFAULT_NUM_WORKERS,
    **kwargs
):
    """
    Quick setup function for common use cases.
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size for training
        image_size: Target image size
        sampler_type: Type of sampler ("weighted", "site_aware", "balanced_batch")
        aggressive_augmentation: Whether to use aggressive augmentations
        num_workers: Number of data loading workers
        **kwargs: Additional arguments for DataLoaderManager
        
    Returns:
        Configured DataLoaderManager instance
        
    Example:
        >>> from src.data import get_quick_setup
        >>> manager = get_quick_setup(
        ...     data_dir="data/raw",
        ...     batch_size=32,
        ...     sampler_type="site_aware"
        ... )
        >>> train_loader = manager.train_loader
        >>> val_loader = manager.val_loader
    """
    
    return DataLoaderManager(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        train_sampler_type=sampler_type,
        aggressive_augmentation=aggressive_augmentation,
        num_workers=num_workers,
        **kwargs
    )


def analyze_dataset_quick(data_dir: str = "data/raw", sample_size: int = 100):
    """
    Quick dataset analysis for initial exploration.
    
    Args:
        data_dir: Path to data directory
        sample_size: Number of images to sample for analysis
        
    Returns:
        Analysis report dictionary
        
    Example:
        >>> from src.data import analyze_dataset_quick
        >>> report = analyze_dataset_quick("data/raw", sample_size=50)
        >>> print(f"Dataset health: {report['summary']['dataset_health']}")
    """
    
    analyzer = DatasetAnalyzer(data_dir)
    return analyzer.generate_report()


# Convenience functions for common workflows
def create_training_pipeline(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    validation_split: float = 0.2,
    **kwargs
):
    """
    Create a complete training pipeline with sensible defaults.
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size
        image_size: Image size for training
        validation_split: Validation split ratio
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with train_loader, val_loader, and metadata
    """
    
    manager = DataLoaderManager(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        train_sampler_type="site_aware",  # Best for camera trap data
        aggressive_augmentation=False,    # Start conservative
        **kwargs
    )
    
    return {
        'train_loader': manager.train_loader,
        'val_loader': manager.val_loader,
        'class_weights': manager.get_class_weights(),
        'train_dataset': manager.train_dataset,
        'val_dataset': manager.val_dataset,
        'class_names': CLASS_NAMES,
        'num_classes': len(CLASS_NAMES)
    }


def validate_setup():
    """
    Validate that the data module is properly set up.
    
    Returns:
        Dictionary with validation results
    """
    
    import importlib
    import sys
    
    validation = {
        'imports_successful': True,
        'missing_dependencies': [],
        'warnings': []
    }
    
    # Check required dependencies
    required_packages = [
        'torch', 'torchvision', 'PIL', 'cv2', 'numpy', 
        'pandas', 'tqdm', 'pathlib'
    ]
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            else:
                importlib.import_module(package)
        except ImportError:
            validation['missing_dependencies'].append(package)
            validation['imports_successful'] = False
    
    # Check if running in expected environment
    if 'torch' in sys.modules:
        import torch
        if not torch.cuda.is_available():
            validation['warnings'].append("CUDA not available - training will be slower")
    
    return validation


# Module initialization
def _check_environment():
    """Check if the environment is properly configured."""
    validation = validate_setup()
    
    if not validation['imports_successful']:
        missing = ', '.join(validation['missing_dependencies'])
        print(f"⚠️  Warning: Missing dependencies: {missing}")
        print("Please install required packages with: pip install -r requirements.txt")
    
    if validation['warnings']:
        for warning in validation['warnings']:
            print(f"⚠️  {warning}")


# Run environment check on import
_check_environment()

# Module-level convenience instances
def get_default_transforms(image_size: int = DEFAULT_IMAGE_SIZE):
    """Get default train and validation transforms."""
    return {
        'train': get_train_transforms(image_size=image_size),
        'val': get_val_transforms(image_size=image_size)
    }