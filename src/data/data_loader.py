"""
Taï National Park - DataLoader Module

This module provides specialized DataLoaders for camera trap images with:
- Site-based validation splits
- Balanced sampling strategies
- Efficient memory management
- Multi-worker support
- Custom collate functions for wildlife data

Key Features:
- Weighted sampling for class balance
- Site-aware batching
- Memory-efficient loading
- Robust error handling
- Performance monitoring
"""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, BatchSampler, RandomSampler
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from collections import Counter, defaultdict
import warnings
from pathlib import Path

from .dataset import TaiParkDataset, create_datasets, create_test_dataset
from .transforms import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)


class SiteAwareBatchSampler(BatchSampler):
    """
    Batch sampler that tries to include samples from different sites in each batch.
    This helps the model learn site-invariant features.
    """
    
    def __init__(
        self,
        dataset: TaiParkDataset,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        site_diversity_ratio: float = 0.7
    ):
        """
        Args:
            dataset: The dataset to sample from
            batch_size: Size of each batch
            drop_last: Whether to drop the last incomplete batch
            shuffle: Whether to shuffle samples
            site_diversity_ratio: Target ratio of unique sites per batch
        """
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.site_diversity_ratio = site_diversity_ratio
        
        # Group samples by site
        self.site_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            sample = dataset.data.iloc[idx]
            site = sample['site']
            self.site_to_indices[site].append(idx)
        
        self.sites = list(self.site_to_indices.keys())
        self.total_samples = len(dataset)
        
        logger.info(f"SiteAwareBatchSampler: {len(self.sites)} sites, {self.total_samples} samples")
    
    def __iter__(self):
        """Generate batches with site diversity."""
        
        # Create list of all indices
        all_indices = list(range(self.total_samples))
        
        if self.shuffle:
            np.random.shuffle(all_indices)
            # Also shuffle sites
            sites_shuffled = self.sites.copy()
            np.random.shuffle(sites_shuffled)
        else:
            sites_shuffled = self.sites
        
        batch = []
        site_counter = 0
        used_indices = set()
        
        while len(used_indices) < self.total_samples:
            # Try to add samples from different sites
            target_sites_in_batch = max(1, int(self.batch_size * self.site_diversity_ratio))
            sites_for_batch = sites_shuffled[site_counter:site_counter + target_sites_in_batch]
            
            # If we don't have enough sites, wrap around
            if len(sites_for_batch) < target_sites_in_batch:
                remaining = target_sites_in_batch - len(sites_for_batch)
                sites_for_batch.extend(sites_shuffled[:remaining])
            
            # Sample from selected sites
            for site in sites_for_batch:
                available_indices = [idx for idx in self.site_to_indices[site] 
                                   if idx not in used_indices]
                
                if available_indices:
                    # Add samples from this site
                    samples_per_site = max(1, self.batch_size // len(sites_for_batch))
                    selected = np.random.choice(
                        available_indices, 
                        size=min(samples_per_site, len(available_indices)),
                        replace=False
                    )
                    
                    for idx in selected:
                        if len(batch) < self.batch_size:
                            batch.append(idx)
                            used_indices.add(idx)
            
            # Fill remaining batch slots with any available samples
            while len(batch) < self.batch_size and len(used_indices) < self.total_samples:
                remaining_indices = [idx for idx in all_indices if idx not in used_indices]
                if remaining_indices:
                    idx = np.random.choice(remaining_indices)
                    batch.append(idx)
                    used_indices.add(idx)
                else:
                    break
            
            # Yield batch when full
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                site_counter = (site_counter + target_sites_in_batch) % len(sites_shuffled)
            elif len(used_indices) >= self.total_samples:
                # Last batch
                if not self.drop_last and batch:
                    yield batch
                break
    
    def __len__(self):
        """Return number of batches."""
        if self.drop_last:
            return self.total_samples // self.batch_size
        else:
            return (self.total_samples + self.batch_size - 1) // self.batch_size


class BalancedBatchSampler(BatchSampler):
    """
    Batch sampler that ensures balanced class representation in each batch.
    Useful for handling class imbalance.
    """
    
    def __init__(
        self,
        dataset: TaiParkDataset,
        batch_size: int,
        drop_last: bool = False,
        samples_per_class: Optional[int] = None
    ):
        """
        Args:
            dataset: The dataset to sample from
            batch_size: Size of each batch
            drop_last: Whether to drop the last incomplete batch
            samples_per_class: Fixed number of samples per class per batch
        """
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.samples_per_class = samples_per_class or max(1, batch_size // dataset.num_classes)
        
        # Group samples by class
        self.class_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            sample = dataset.data.iloc[idx]
            class_idx = np.argmax(sample[dataset.CLASS_NAMES].values)
            self.class_to_indices[class_idx].append(idx)
        
        self.classes = list(self.class_to_indices.keys())
        self.total_samples = len(dataset)
        
        logger.info(f"BalancedBatchSampler: {len(self.classes)} classes")
        for cls_idx, indices in self.class_to_indices.items():
            cls_name = dataset.CLASS_NAMES[cls_idx]
            logger.info(f"  Class {cls_name}: {len(indices)} samples")
    
    def __iter__(self):
        """Generate balanced batches."""
        
        # Create iterators for each class
        class_iterators = {}
        for cls_idx, indices in self.class_to_indices.items():
            indices_shuffled = indices.copy()
            np.random.shuffle(indices_shuffled)
            class_iterators[cls_idx] = iter(indices_shuffled * 1000)  # Repeat many times
        
        batch = []
        total_yielded = 0
        
        while total_yielded < self.total_samples:
            # Sample from each class
            for cls_idx in self.classes:
                for _ in range(self.samples_per_class):
                    if len(batch) < self.batch_size:
                        try:
                            idx = next(class_iterators[cls_idx])
                            batch.append(idx)
                        except StopIteration:
                            # This class is exhausted, skip
                            break
            
            # Yield batch when full
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                total_yielded += len(batch[:self.batch_size])
                batch = batch[self.batch_size:]
            else:
                # Not enough samples to continue
                if not self.drop_last and batch:
                    yield batch
                    total_yielded += len(batch)
                break
    
    def __len__(self):
        """Return number of batches."""
        if self.drop_last:
            return self.total_samples // self.batch_size
        else:
            return (self.total_samples + self.batch_size - 1) // self.batch_size


def wildlife_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for wildlife camera trap data.
    Handles variable-sized data and provides useful batch statistics.
    """
    
    # Separate different data types
    images = []
    labels = []
    class_indices = []
    weights = []
    ids = []
    sites = []
    image_paths = []
    
    for sample in batch:
        images.append(sample['image'])
        ids.append(sample['id'])
        image_paths.append(sample['image_path'])
        
        if 'label' in sample:
            labels.append(sample['label'])
            class_indices.append(sample['class_idx'])
        
        if 'weight' in sample:
            weights.append(sample['weight'])
        
        if 'site' in sample:
            sites.append(sample['site'])
    
    # Stack tensors
    result = {
        'image': torch.stack(images),
        'id': ids,
        'image_path': image_paths
    }
    
    # Add labels if available
    if labels:
        result['label'] = torch.stack(labels)
        result['class_idx'] = torch.stack(class_indices)
    
    # Add weights if available
    if weights:
        result['weight'] = torch.stack(weights)
    
    # Add sites if available
    if sites:
        result['site'] = sites
        # Add batch site diversity statistics
        unique_sites = len(set(sites))
        result['batch_site_diversity'] = unique_sites / len(sites)
    
    # Add batch statistics
    result['batch_size'] = len(batch)
    
    if labels:
        # Class distribution in this batch
        class_counts = torch.bincount(torch.stack(class_indices), minlength=8)
        result['batch_class_distribution'] = class_counts.float() / len(batch)
    
    return result


def create_balanced_dataloader(
    dataset: TaiParkDataset,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    sampler_type: str = "weighted",
    **kwargs
) -> DataLoader:
    """
    Create a balanced DataLoader for training.
    
    Args:
        dataset: The dataset to load from
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        sampler_type: Type of sampler ("weighted", "balanced_batch", "site_aware")
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        Configured DataLoader
    """
    
    if sampler_type == "weighted":
        # Use weighted random sampler
        sample_weights = dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=True
        )
        batch_sampler = None
        shuffle = False
        
    elif sampler_type == "balanced_batch":
        # Use balanced batch sampler
        sampler = None
        batch_sampler = BalancedBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=kwargs.get('drop_last', True)
        )
        shuffle = False
        
    elif sampler_type == "site_aware":
        # Use site-aware batch sampler
        sampler = None
        batch_sampler = SiteAwareBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=kwargs.get('drop_last', True),
            shuffle=True
        )
        shuffle = False
        
    else:
        # Default random sampling
        sampler = None
        batch_sampler = None
        shuffle = kwargs.get('shuffle', True)
    
    # Remove conflicting arguments
    dataloader_kwargs = kwargs.copy()
    dataloader_kwargs.pop('shuffle', None)
    dataloader_kwargs.pop('drop_last', None)
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size if batch_sampler is None else 1,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=wildlife_collate_fn,
        **dataloader_kwargs
    )


def create_standard_dataloader(
    dataset: TaiParkDataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    **kwargs
) -> DataLoader:
    """
    Create a standard DataLoader for validation/testing.
    
    Args:
        dataset: The dataset to load from
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        Configured DataLoader
    """
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=wildlife_collate_fn,
        **kwargs
    )


class DataLoaderManager:
    """
    Manager class for handling multiple DataLoaders with consistent configuration.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        batch_size: int = 32,
        val_batch_size: Optional[int] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        image_size: Union[int, Tuple[int, int]] = 224,
        validation_sites_file: Optional[str] = None,
        train_sampler_type: str = "weighted",
        aggressive_augmentation: bool = False,
        cache_images: bool = False,
        **dataset_kwargs
    ):
        """
        Initialize DataLoader manager.
        
        Args:
            data_dir: Path to data directory
            batch_size: Training batch size
            val_batch_size: Validation batch size (defaults to batch_size)
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            image_size: Target image size
            validation_sites_file: Path to validation sites file
            train_sampler_type: Type of training sampler
            aggressive_augmentation: Whether to use aggressive augmentation
            cache_images: Whether to cache images in memory
            **dataset_kwargs: Additional dataset arguments
        """
        
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.image_size = image_size
        self.validation_sites_file = validation_sites_file
        self.train_sampler_type = train_sampler_type
        self.aggressive_augmentation = aggressive_augmentation
        self.cache_images = cache_images
        self.dataset_kwargs = dataset_kwargs
        
        # Create transforms
        self.train_transform = get_train_transforms(
            image_size=image_size,
            aggressive=aggressive_augmentation
        )
        self.val_transform = get_val_transforms(image_size=image_size)
        
        # Initialize datasets and loaders
        self._create_datasets()
        self._create_dataloaders()
        
        logger.info("DataLoaderManager initialized successfully")
        self._log_summary()
    
    def _create_datasets(self):
        """Create train and validation datasets."""
        
        self.train_dataset, self.val_dataset = create_datasets(
            data_dir=self.data_dir,
            validation_sites_file=self.validation_sites_file,
            train_transform=self.train_transform,
            val_transform=self.val_transform,
            cache_images=self.cache_images,
            **self.dataset_kwargs
        )
        
        logger.info(f"Created datasets - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")
    
    def _create_dataloaders(self):
        """Create train and validation dataloaders."""
        
        # Training dataloader with balancing
        self.train_loader = create_balanced_dataloader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler_type=self.train_sampler_type,
            drop_last=True
        )
        
        # Validation dataloader (standard)
        self.val_loader = create_standard_dataloader(
            dataset=self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
        
        logger.info(f"Created dataloaders - Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
    
    def create_test_loader(
        self,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> DataLoader:
        """
        Create test dataloader.
        
        Args:
            batch_size: Test batch size (defaults to val_batch_size)
            **kwargs: Additional arguments for dataset
            
        Returns:
            Test DataLoader
        """
        
        test_batch_size = batch_size or self.val_batch_size
        
        test_dataset = create_test_dataset(
            data_dir=self.data_dir,
            transform=self.val_transform,
            cache_images=self.cache_images,
            **{**self.dataset_kwargs, **kwargs}
        )
        
        test_loader = create_standard_dataloader(
            dataset=test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
        
        logger.info(f"Created test dataloader - {len(test_dataset)} samples, {len(test_loader)} batches")
        return test_loader
    
    def get_class_weights(self) -> torch.Tensor:
        """Get class weights for loss functions."""
        class_distribution = self.train_dataset.get_class_distribution()
        total_samples = sum(class_distribution.values())
        num_classes = len(self.train_dataset.CLASS_NAMES)
        
        weights = []
        for class_name in self.train_dataset.CLASS_NAMES:
            count = class_distribution.get(class_name, 1)
            weight = total_samples / (num_classes * count)
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def _log_summary(self):
        """Log summary of the data loading setup."""
        logger.info("\n" + "="*50)
        logger.info("DATALOADER SUMMARY")
        logger.info("="*50)
        
        logger.info(f"📂 Data directory: {self.data_dir}")
        logger.info(f"🖼️  Image size: {self.image_size}")
        logger.info(f"🎯 Train sampler: {self.train_sampler_type}")
        logger.info(f"📦 Batch sizes - Train: {self.batch_size}, Val: {self.val_batch_size}")
        logger.info(f"👥 Workers: {self.num_workers}")
        logger.info(f"🚀 Aggressive augmentation: {self.aggressive_augmentation}")
        logger.info(f"💾 Cache images: {self.cache_images}")
        
        # Dataset sizes
        logger.info(f"\n📊 Dataset sizes:")
        logger.info(f"  Train: {len(self.train_dataset):,} samples")
        logger.info(f"  Val: {len(self.val_dataset):,} samples")
        logger.info(f"  Total: {len(self.train_dataset) + len(self.val_dataset):,} samples")
        
        # Batch counts
        logger.info(f"\n🔄 Batches per epoch:")
        logger.info(f"  Train: {len(self.train_loader):,} batches")
        logger.info(f"  Val: {len(self.val_loader):,} batches")
        
        # Class distribution
        train_dist = self.train_dataset.get_class_distribution()
        logger.info(f"\n🏷️  Train class distribution:")
        for class_name, count in train_dist.items():
            percentage = (count / len(self.train_dataset)) * 100
            logger.info(f"  {class_name}: {count:,} ({percentage:.1f}%)")


# Utility functions
def analyze_dataloader_performance(dataloader: DataLoader, num_batches: int = 10) -> Dict[str, float]:
    """
    Analyze DataLoader performance metrics.
    
    Args:
        dataloader: DataLoader to analyze
        num_batches: Number of batches to test
        
    Returns:
        Performance metrics dictionary
    """
    
    import time
    
    times = []
    batch_sizes = []
    site_diversities = []
    
    logger.info(f"Analyzing DataLoader performance over {num_batches} batches...")
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        start_time = time.time()
        
        # Access all data to measure actual loading time
        _ = batch['image']
        if 'label' in batch:
            _ = batch['label']
        
        end_time = time.time()
        
        times.append(end_time - start_time)
        batch_sizes.append(batch['batch_size'])
        
        if 'batch_site_diversity' in batch:
            site_diversities.append(batch['batch_site_diversity'])
    
    metrics = {
        'avg_batch_time': np.mean(times),
        'std_batch_time': np.std(times),
        'avg_batch_size': np.mean(batch_sizes),
        'samples_per_second': np.sum(batch_sizes) / np.sum(times),
    }
    
    if site_diversities:
        metrics['avg_site_diversity'] = np.mean(site_diversities)
    
    logger.info("Performance metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.3f}")
    
    return metrics


# Example usage and testing
if __name__ == "__main__":
    from pathlib import Path
    
    # Test DataLoader creation
    data_dir = Path("data/raw")
    
    if data_dir.exists():
        print("Testing DataLoader creation...")
        
        try:
            # Create manager
            manager = DataLoaderManager(
                data_dir=data_dir,
                batch_size=16,
                num_workers=2,  # Reduced for testing
                image_size=224,
                train_sampler_type="weighted",
                aggressive_augmentation=False,
                cache_images=False
            )
            
            print("✅ DataLoaderManager created successfully")
            
            # Test loading a batch
            train_batch = next(iter(manager.train_loader))
            val_batch = next(iter(manager.val_loader))
            
            print(f"✅ Train batch loaded - shape: {train_batch['image'].shape}")
            print(f"✅ Val batch loaded - shape: {val_batch['image'].shape}")
            
            # Test different samplers
            samplers_to_test = ["weighted", "site_aware", "balanced_batch"]
            
            for sampler_type in samplers_to_test:
                try:
                    test_loader = create_balanced_dataloader(
                        dataset=manager.train_dataset,
                        batch_size=8,
                        num_workers=0,
                        sampler_type=sampler_type
                    )
                    test_batch = next(iter(test_loader))
                    print(f"✅ {sampler_type} sampler works")
                except Exception as e:
                    print(f"❌ {sampler_type} sampler failed: {e}")
            
            # Test performance
            if len(manager.train_loader) > 0:
                metrics = analyze_dataloader_performance(manager.train_loader, num_batches=3)
                print("✅ Performance analysis completed")
            
        except Exception as e:
            print(f"❌ DataLoader test failed: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print(f"Data directory not found: {data_dir}")
        print("This is normal if running outside the project directory.")
    
    print("\n🎉 DataLoader module tests completed!")
    print("\nKey features implemented:")
    print("  ⚖️  Weighted sampling for class balance")
    print("  🌍 Site-aware batching")
    print("  🎯 Balanced batch sampling")
    print("  📊 Custom collate functions")
    print("  ⚡ Performance monitoring")
    print("  🔧 Easy configuration management")