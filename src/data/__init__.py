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

import logging

logger = logging.getLogger(__name__)

# Core dataset functionality
try:
    from .dataset import (
        TaiParkDataset,
        create_datasets,
        create_test_dataset
    )
    _DATASET_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Dataset imports failed: {e}")
    _DATASET_AVAILABLE = False
    TaiParkDataset = None
    create_datasets = None
    create_test_dataset = None

# Image transformations
try:
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
    _TRANSFORMS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Transforms imports failed: {e}")
    _TRANSFORMS_AVAILABLE = False
    # Set to None for missing imports
    get_train_transforms = None
    get_val_transforms = None

# Data loaders and sampling
try:
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
    _DATALOADER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"DataLoader imports failed: {e}")
    _DATALOADER_AVAILABLE = False
    DataLoaderManager = None
    create_balanced_dataloader = None

# Preprocessing and utilities
try:
    from .preprocessing import (
        DatasetAnalyzer,
        ImageValidator,
        CacheManager,
        preprocess_dataset,
        validate_and_clean_dataset,
        create_balanced_subset
    )
    _PREPROCESSING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Preprocessing imports failed: {e}")
    _PREPROCESSING_AVAILABLE = False
    DatasetAnalyzer = None

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

# Export list - only include available components
__all__ = [
    # Constants that are always available
    'CLASS_NAMES',
    'DEFAULT_IMAGE_SIZE', 
    'DEFAULT_BATCH_SIZE',
    'DEFAULT_NUM_WORKERS',
    'IMAGENET_MEAN',
    'IMAGENET_STD'
]

# Add available components to __all__
if _DATASET_AVAILABLE:
    __all__.extend([
        'TaiParkDataset',
        'create_datasets',
        'create_test_dataset'
    ])

if _TRANSFORMS_AVAILABLE:
    __all__.extend([
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
        'NightVisionSimulation'
    ])

if _DATALOADER_AVAILABLE:
    __all__.extend([
        'DataLoaderManager',
        'create_balanced_dataloader',
        'create_standard_dataloader',
        'wildlife_collate_fn',
        'analyze_dataloader_performance',
        'SiteAwareBatchSampler',
        'BalancedBatchSampler'
    ])

if _PREPROCESSING_AVAILABLE:
    __all__.extend([
        'DatasetAnalyzer',
        'ImageValidator',
        'CacheManager',
        'preprocess_dataset',
        'validate_and_clean_dataset',
        'create_balanced_subset'
    ])

# Convenience functions (only create if dependencies are available)
if _DATALOADER_AVAILABLE and DataLoaderManager is not None:
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
    
    __all__.append('get_quick_setup')

if _PREPROCESSING_AVAILABLE and DatasetAnalyzer is not None:
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
    
    __all__.append('analyze_dataset_quick')

# Validation function to check what's available
def validate_data_module():
    """
    Validate which components of the data module are available.
    
    Returns:
        Dictionary with availability status of each component
    """
    
    return {
        'dataset': _DATASET_AVAILABLE,
        'transforms': _TRANSFORMS_AVAILABLE, 
        'dataloader': _DATALOADER_AVAILABLE,
        'preprocessing': _PREPROCESSING_AVAILABLE,
        'core_functionality': _DATASET_AVAILABLE and _DATALOADER_AVAILABLE
    }

__all__.append('validate_data_module')

# Print warnings if core components are missing
if not _DATALOADER_AVAILABLE:
    logger.error("❌ DataLoaderManager not available - core functionality will be limited")

if not _DATASET_AVAILABLE:
    logger.error("❌ TaiParkDataset not available - core functionality will be limited")

# Summary message
_available_count = sum([_DATASET_AVAILABLE, _TRANSFORMS_AVAILABLE, _DATALOADER_AVAILABLE, _PREPROCESSING_AVAILABLE])
logger.info(f"📚 Data module loaded - {_available_count}/4 components available")