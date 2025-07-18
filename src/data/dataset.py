"""
Taï National Park Species Dataset

This module implements a PyTorch Dataset for the camera trap species classification
challenge. It handles efficient loading of images with site-based validation splits.

Key Features:
- Site-based train/validation splits (no site overlap)
- Robust image loading with error handling
- Class balancing support
- Efficient caching for faster training
- Support for both PIL and OpenCV backends
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import logging
from typing import Optional, Tuple, List, Dict, Union, Callable
import warnings
from collections import Counter
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaiParkDataset(Dataset):
    """
    PyTorch Dataset for Taï National Park camera trap images.
    
    This dataset implements site-based validation splits to ensure models
    generalize to new camera trap locations.
    """
    
    # Class names mapping
    CLASS_NAMES = [
        'antelope_duiker', 'bird', 'blank', 'civet_genet',
        'hog', 'leopard', 'monkey_prosimian', 'rodent'
    ]
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        features_file: str = "train_features.csv",
        labels_file: Optional[str] = "train_labels.csv",
        validation_sites_file: Optional[str] = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        backend: str = "PIL",
        cache_images: bool = False,
        max_cache_size: int = 1000,
        class_weights: Optional[Dict[str, float]] = None,
        return_site_info: bool = False
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Path to the data directory containing raw files
            features_file: Name of the features CSV file
            labels_file: Name of the labels CSV file (None for test set)
            validation_sites_file: Path to CSV with validation sites
            split: 'train', 'val', or 'test'
            transform: Transform function to apply to images
            backend: 'PIL' or 'opencv' for image loading
            cache_images: Whether to cache loaded images in memory
            max_cache_size: Maximum number of images to cache
            class_weights: Dictionary with class weights for balancing
            return_site_info: Whether to return site information with samples
        """
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.backend = backend.lower()
        self.cache_images = cache_images
        self.max_cache_size = max_cache_size
        self.class_weights = class_weights or {}
        self.return_site_info = return_site_info
        
        # Image cache
        self._image_cache = {}
        self._cache_access_order = []
        
        # Load data
        self._load_data(features_file, labels_file, validation_sites_file)
        
        # Setup class information
        self._setup_class_info()
        
        logger.info(f"Dataset initialized: {len(self)} samples in '{split}' split")
        if hasattr(self, 'labels'):
            self._log_class_distribution()
    
    def _load_data(
        self, 
        features_file: str, 
        labels_file: Optional[str],
        validation_sites_file: Optional[str]
    ):
        """Load and filter data based on split."""
        
        # Load features
        features_path = self.data_dir / features_file
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        self.features = pd.read_csv(features_path)
        logger.info(f"Loaded {len(self.features)} samples from {features_file}")
        
        # Load labels if available (training/validation)
        if labels_file:
            labels_path = self.data_dir / labels_file
            if not labels_path.exists():
                raise FileNotFoundError(f"Labels file not found: {labels_path}")
            
            self.labels = pd.read_csv(labels_path)
            
            # Ensure features and labels match
            if not set(self.features['id']) == set(self.labels['id']):
                logger.warning("Features and labels IDs don't match perfectly")
            
            # Merge features and labels
            self.data = self.features.merge(self.labels, on='id', how='inner')
        else:
            # Test set - no labels
            self.data = self.features.copy()
            self.labels = None
        
        # Apply site-based split
        if self.split in ['train', 'val']:
            self._apply_site_split(validation_sites_file)
        
        # Reset index
        self.data = self.data.reset_index(drop=True)
        
        logger.info(f"Final dataset size for '{self.split}': {len(self.data)}")
    
    def _apply_site_split(self, validation_sites_file: Optional[str]):
        """Apply site-based train/validation split."""
        
        if validation_sites_file and Path(validation_sites_file).exists():
            # Load predefined validation sites
            val_sites_df = pd.read_csv(validation_sites_file)
            validation_sites = set(val_sites_df['site'].tolist())
            logger.info(f"Loaded {len(validation_sites)} validation sites from file")
        else:
            # Create random site split
            all_sites = list(self.data['site'].unique())
            np.random.seed(42)  # For reproducibility
            np.random.shuffle(all_sites)
            
            val_ratio = 0.2
            n_val_sites = int(len(all_sites) * val_ratio)
            validation_sites = set(all_sites[:n_val_sites])
            
            logger.info(f"Created random validation split: {len(validation_sites)} validation sites")
        
        # Filter data based on split
        if self.split == 'train':
            self.data = self.data[~self.data['site'].isin(validation_sites)]
        elif self.split == 'val':
            self.data = self.data[self.data['site'].isin(validation_sites)]
        
        logger.info(f"Site split applied - {self.split}: {len(self.data)} samples")
    
    def _setup_class_info(self):
        """Setup class-related information."""
        self.num_classes = len(self.CLASS_NAMES)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.CLASS_NAMES)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        if hasattr(self, 'labels') and self.labels is not None:
            # Calculate class weights for balancing
            class_counts = self.data[self.CLASS_NAMES].sum()
            total_samples = len(self.data)
            
            self.default_class_weights = {}
            for cls in self.CLASS_NAMES:
                if class_counts[cls] > 0:
                    self.default_class_weights[cls] = total_samples / (self.num_classes * class_counts[cls])
                else:
                    self.default_class_weights[cls] = 1.0
    
    def _log_class_distribution(self):
        """Log class distribution for monitoring."""
        if hasattr(self, 'labels') and self.labels is not None:
            class_counts = self.data[self.CLASS_NAMES].sum()
            logger.info(f"Class distribution in '{self.split}' split:")
            for cls, count in class_counts.items():
                percentage = (count / len(self.data)) * 100
                logger.info(f"  {cls}: {int(count)} ({percentage:.1f}%)")
    
    def _load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Load image using specified backend with error handling.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Loaded image as numpy array or None if loading fails
        """
        
        # Check cache first
        if self.cache_images and str(image_path) in self._image_cache:
            self._update_cache_access(str(image_path))
            return self._image_cache[str(image_path)]
        
        try:
            if self.backend == "pil":
                image = Image.open(image_path).convert('RGB')
                image = np.array(image)
            elif self.backend == "opencv":
                image = cv2.imread(str(image_path))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError(f"Unknown backend: {self.backend}")
            
            if image is None:
                logger.warning(f"Failed to load image: {image_path}")
                return None
            
            # Cache image if enabled
            if self.cache_images:
                self._add_to_cache(str(image_path), image)
            
            return image
            
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            return None
    
    def _add_to_cache(self, image_path: str, image: np.ndarray):
        """Add image to cache with LRU eviction."""
        if len(self._image_cache) >= self.max_cache_size:
            # Remove least recently used image
            oldest_path = self._cache_access_order.pop(0)
            del self._image_cache[oldest_path]
        
        self._image_cache[image_path] = image.copy()
        self._cache_access_order.append(image_path)
    
    def _update_cache_access(self, image_path: str):
        """Update cache access order for LRU."""
        if image_path in self._cache_access_order:
            self._cache_access_order.remove(image_path)
            self._cache_access_order.append(image_path)
    
    def _create_fallback_image(self, height: int = 224, width: int = 224) -> np.ndarray:
        """Create a fallback image when loading fails."""
        # Create a simple noise pattern
        np.random.seed(42)
        return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
            - image: Transformed image tensor
            - label: Target tensor (if labels available)
            - id: Sample ID
            - site: Site information (if return_site_info=True)
            - image_path: Path to the image file
        """
        
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        
        # Get sample info
        sample = self.data.iloc[idx]
        image_path = self.data_dir / sample['filepath']
        
        # Load image
        image = self._load_image(image_path)
        
        # Fallback if image loading failed
        if image is None:
            image = self._create_fallback_image()
            logger.warning(f"Using fallback image for {image_path}")
        
        # Convert to PIL Image for transforms compatibility
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Prepare return dictionary
        result = {
            'image': image,
            'id': sample['id'],
            'image_path': str(image_path)
        }
        
        # Add labels if available
        if hasattr(self, 'labels') and self.labels is not None:
            # Get one-hot encoded labels
            label_values = sample[self.CLASS_NAMES].values.astype(np.float32)
            result['label'] = torch.from_numpy(label_values)
            
            # Get class index (for categorical)
            class_idx = np.argmax(label_values)
            result['class_idx'] = torch.tensor(class_idx, dtype=torch.long)
            
            # Add class weight if specified
            class_name = self.CLASS_NAMES[class_idx]
            if class_name in self.class_weights:
                result['weight'] = torch.tensor(self.class_weights[class_name], dtype=torch.float32)
            elif hasattr(self, 'default_class_weights'):
                result['weight'] = torch.tensor(self.default_class_weights[class_name], dtype=torch.float32)
        
        # Add site info if requested
        if self.return_site_info:
            result['site'] = sample['site']
        
        return result
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution for the current split."""
        if hasattr(self, 'labels') and self.labels is not None:
            return self.data[self.CLASS_NAMES].sum().to_dict()
        return {}
    
    def get_site_distribution(self) -> Dict[str, int]:
        """Get site distribution for the current split."""
        return self.data['site'].value_counts().to_dict()
    
    def get_sample_weights(self) -> torch.Tensor:
        """
        Get sample weights for balanced sampling.
        
        Returns:
            Tensor of weights for each sample
        """
        if not hasattr(self, 'labels') or self.labels is None:
            return torch.ones(len(self))
        
        weights = []
        for idx in range(len(self)):
            sample = self.data.iloc[idx]
            class_idx = np.argmax(sample[self.CLASS_NAMES].values)
            class_name = self.CLASS_NAMES[class_idx]
            
            if class_name in self.class_weights:
                weight = self.class_weights[class_name]
            elif hasattr(self, 'default_class_weights'):
                weight = self.default_class_weights[class_name]
            else:
                weight = 1.0
            
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)


def create_datasets(
    data_dir: Union[str, Path],
    validation_sites_file: Optional[str] = None,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    **dataset_kwargs
) -> Tuple[TaiParkDataset, TaiParkDataset]:
    """
    Create train and validation datasets with proper site splits.
    
    Args:
        data_dir: Path to the data directory
        validation_sites_file: Path to validation sites CSV
        train_transform: Transform for training data
        val_transform: Transform for validation data
        **dataset_kwargs: Additional arguments for dataset
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    
    train_dataset = TaiParkDataset(
        data_dir=data_dir,
        split='train',
        transform=train_transform,
        validation_sites_file=validation_sites_file,
        **dataset_kwargs
    )
    
    val_dataset = TaiParkDataset(
        data_dir=data_dir,
        split='val',
        transform=val_transform,
        validation_sites_file=validation_sites_file,
        **dataset_kwargs
    )
    
    return train_dataset, val_dataset


def create_test_dataset(
    data_dir: Union[str, Path],
    transform: Optional[Callable] = None,
    **dataset_kwargs
) -> TaiParkDataset:
    """
    Create test dataset.
    
    Args:
        data_dir: Path to the data directory
        transform: Transform for test data
        **dataset_kwargs: Additional arguments for dataset
        
    Returns:
        Test dataset
    """
    
    return TaiParkDataset(
        data_dir=data_dir,
        features_file="test_features.csv",
        labels_file=None,
        split='test',
        transform=transform,
        **dataset_kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    from pathlib import Path
    
    # Example usage
    data_dir = Path("data/raw")
    
    if data_dir.exists():
        print("Testing TaiParkDataset...")
        
        # Create datasets
        train_dataset, val_dataset = create_datasets(
            data_dir=data_dir,
            cache_images=False,
            return_site_info=True
        )
        
        print(f"Train dataset: {len(train_dataset)} samples")
        print(f"Validation dataset: {len(val_dataset)} samples")
        
        # Test loading a sample
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"Sample keys: {sample.keys()}")
            print(f"Image shape: {sample['image'].shape}")
            if 'label' in sample:
                print(f"Label shape: {sample['label'].shape}")
                print(f"Class: {train_dataset.CLASS_NAMES[sample['class_idx']]}")
        
        # Print class distributions
        print("\nTrain class distribution:")
        for cls, count in train_dataset.get_class_distribution().items():
            print(f"  {cls}: {count}")
        
        print("\nValidation class distribution:")
        for cls, count in val_dataset.get_class_distribution().items():
            print(f"  {cls}: {count}")
            
    else:
        print(f"Data directory not found: {data_dir}")
        print("This is normal if running outside the project directory.")