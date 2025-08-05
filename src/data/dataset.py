"""
Tai Park Wildlife Dataset - Notebook Style Implementation

This module replicates the exact dataset logic from the successful notebook
with proper site-aware splitting to prevent data leakage.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile, ImageFilter, ImageEnhance
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.transforms import v2 as transforms

# Allow truncated images (common in camera traps)
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


def custom_preprocessing(image):
    """
    Custom preprocessing function exactly like the notebook.
    
    Applies color enhancement, brightness, contrast adjustments and sharpening
    to help wildlife stand out against complex backgrounds.
    """
    # Convertir a escala de grises puede ayudar a enfocarse en la textura y forma
    image = ImageEnhance.Color(image).enhance(0.8)  # Ligeramente menos saturación
    image = ImageEnhance.Brightness(image).enhance(1.1)  # Aumentar ligeramente el brillo
    image = ImageEnhance.Contrast(image).enhance(1.2)  # Aumentar el contraste para resaltar características
    
    # Aplicar un suave enfoque puede ayudar a resaltar los animales contra fondos complejos
    image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=50))
    
    return image


def data_augmentation(image):
    """
    Data augmentation function exactly like the notebook.
    
    Applies random transformations to increase dataset diversity.
    """
    # Transformaciones aleatorias
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
        transforms.RandomRotation(degrees=12),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomEqualize(p=0.33),
    ])
    
    # Aplicar transformaciones aleatorias
    augmented_image = transform(image)
    
    return augmented_image


class ImagesDataset(Dataset):
    """
    Exact replica of notebook's ImagesDataset class.
    
    Reads in an image, transforms pixel values, and serves
    a dictionary containing the image id, image tensors, and label.
    """

    def __init__(self, x_df, y_df=None, preprocessing=None, augmentation=None, data_dir=None):
        self.data = x_df
        self.label = y_df
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.data_dir = data_dir or ""
        
        # Exact transform pipeline from notebook
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ])

    def __getitem__(self, index):
        # Load image exactly like notebook
        image_path = os.path.join(self.data_dir, self.data.iloc[index]["filepath"])
        image = Image.open(image_path).convert("RGB")

        # Preprocesamiento de la imagen
        if self.preprocessing is not None:
            image = self.preprocessing(image)
        
        # Aumento de datos
        if self.augmentation is not None:
            image = self.augmentation(image)
        
        # Apply transforms
        image = self.transform(image)
        image_id = self.data.index[index]
        
        # if we don't have labels (e.g. for test set) just return the image and image id
        if self.label is None:
            sample = {"image_id": image_id, "image": image}
        else:
            label = torch.tensor(self.label.iloc[index].values, dtype=torch.float)
            sample = {"image_id": image_id, "image": image, "label": label}
        
        return sample

    def __len__(self):
        return len(self.data)


def create_combined_dataset(original_dataset, num_augmentations, augmentation_functions, x_train, y_train, data_dir):
    """
    Creates a combined dataset with the original dataset and multiple augmented versions.
    Exact replica of notebook function.

    Args:
        original_dataset: The original dataset without augmentation.
        num_augmentations: The number of augmented datasets to create.
        augmentation_functions: A list of augmentation functions (e.g., transforms).
        x_train: Training features DataFrame
        y_train: Training labels DataFrame

    Returns:
        combined_dataset: A ConcatDataset with the original and augmented datasets.
    """
    datasets = [original_dataset]
    for i in range(num_augmentations):
        augmented_dataset = ImagesDataset(
            x_train, y_train, 
            preprocessing=custom_preprocessing,
            augmentation=augmentation_functions,
            data_dir=data_dir
        )
        datasets.append(augmented_dataset)

    combined_dataset = ConcatDataset(datasets)
    return combined_dataset


class TaiParkDatasetNotebookStyle:
    """
    Main dataset manager that replicates notebook workflow with site-aware splits.
    
    This combines the notebook's exact data loading logic with proper site-based
    train/validation splits to prevent data leakage.
    """
    
    # Wildlife species classes
    CLASS_NAMES = [
        'antelope_duiker',
        'bird', 
        'blank',
        'civet_genet',
        'hog',
        'leopard',
        'monkey_prosimian',
        'rodent'
    ]
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        fraction: float = 1.0,
        random_state: int = 1,  # notebook uses random_state=1
        validation_sites_file: Optional[str] = None,
        test_size: float = 0.25,
        use_preprocessing: bool = True,
        use_augmentation: bool = True,
        num_augmentations: int = 2
    ):
        """
        Initialize dataset manager exactly like notebook but with site-aware splits.
        
        Args:
            data_dir: Path to data directory containing CSV files
            fraction: Fraction of data to use (notebook's frac parameter)
            random_state: Random seed (notebook uses 1, you used 42)
            validation_sites_file: Path to validation sites CSV
            test_size: Validation split size (notebook uses 0.25)
            use_preprocessing: Whether to apply custom preprocessing
            use_augmentation: Whether to apply data augmentation
            num_augmentations: Number of augmented datasets to create
        """
        
        self.data_dir = Path(data_dir)
        self.fraction = fraction
        self.random_state = random_state
        self.test_size = test_size
        self.use_preprocessing = use_preprocessing
        self.use_augmentation = use_augmentation
        self.num_augmentations = num_augmentations
        
        # Load data exactly like notebook
        self._load_notebook_data()
        
        # Create site-aware splits (better than notebook's stratified split)
        self._create_site_aware_splits(validation_sites_file)
        
        # Store species labels like notebook
        self.species_labels = sorted(self.train_labels.columns.unique())
        
        logger.info(f"Dataset loaded: {len(self.y_train)} train, {len(self.y_eval)} val samples")
        self._log_split_distribution()

    def _load_notebook_data(self):
        """Load data exactly like the notebook."""
        
        # Exact notebook loading
        self.train_features = pd.read_csv(self.data_dir / "train_features.csv", index_col="id")
        self.test_features = pd.read_csv(self.data_dir / "test_features.csv", index_col="id")
        self.train_labels = pd.read_csv(self.data_dir / "train_labels.csv", index_col="id")
        
        # Apply fraction sampling like notebook
        self.y = self.train_labels.sample(frac=self.fraction, random_state=self.random_state)
        self.x = self.train_features.loc[self.y.index].filepath.to_frame()
        
        logger.info(f"Loaded {len(self.y)} samples with fraction={self.fraction}")

    def _create_site_aware_splits(self, validation_sites_file: Optional[str]):
        """Create site-aware splits instead of notebook's stratified split."""
        
        # Get site information for proper splitting
        x_with_sites = self.train_features.loc[self.y.index][['filepath', 'site']]
        
        if validation_sites_file and Path(validation_sites_file).exists():
            # Load predefined validation sites
            val_sites_df = pd.read_csv(validation_sites_file)
            validation_sites = set(val_sites_df['site'].tolist())
            logger.info(f"Loaded {len(validation_sites)} validation sites from file")
        else:
            # Create random site split (better than stratified for this problem)
            all_sites = list(x_with_sites['site'].unique())
            np.random.seed(self.random_state)
            np.random.shuffle(all_sites)
            
            n_val_sites = int(len(all_sites) * self.test_size)
            validation_sites = set(all_sites[:n_val_sites])
            
            logger.info(f"Created site-aware validation split: {len(validation_sites)} validation sites")
        
        # Split by sites instead of stratified sampling
        train_mask = ~x_with_sites['site'].isin(validation_sites)
        val_mask = x_with_sites['site'].isin(validation_sites)
        
        # Create x_train, x_eval, y_train, y_eval like notebook
        self.x_train = x_with_sites[train_mask][['filepath']]
        self.x_eval = x_with_sites[val_mask][['filepath']]
        self.y_train = self.y.loc[self.x_train.index]
        self.y_eval = self.y.loc[self.x_eval.index]
        
        # Store site info for debugging
        self.train_sites = x_with_sites.loc[self.x_train.index, 'site']
        self.val_sites = x_with_sites.loc[self.x_eval.index, 'site']

    def _log_split_distribution(self):
        """Log class distribution exactly like notebook."""
        
        # Create split percentages DataFrame like notebook
        split_pcts = pd.DataFrame({
            "train": self.y_train.idxmax(axis=1).value_counts(normalize=True),
            "eval": self.y_eval.idxmax(axis=1).value_counts(normalize=True),
        })
        
        logger.info("Species percentages by split:")
        logger.info(f"\n{(split_pcts.fillna(0) * 100).astype(int)}")
        
        # Also log absolute counts
        train_counts = self.y_train.sum().sort_values(ascending=False)
        eval_counts = self.y_eval.sum().sort_values(ascending=False)
        
        logger.info(f"Train counts: {train_counts.to_dict()}")
        logger.info(f"Eval counts: {eval_counts.to_dict()}")

    def create_datasets(self):
        """
        Create train and eval datasets exactly like notebook.
        
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        
        # Choose preprocessing and augmentation based on settings
        preprocessing = custom_preprocessing if self.use_preprocessing else None
        augmentation = data_augmentation if self.use_augmentation else None
        
        # Create original train dataset exactly like notebook
        train_dataset_original = ImagesDataset(
            self.x_train, self.y_train, 
            preprocessing=preprocessing, 
            augmentation=None,
            data_dir=data_dir 
        )
        
        # Create combined dataset with augmentations like notebook
        if self.use_augmentation and self.num_augmentations > 0:
            train_dataset = create_combined_dataset(
                train_dataset_original, 
                self.num_augmentations, 
                augmentation,
                self.x_train, 
                self.y_train,
                self.data_dir
            )
        else:
            train_dataset = train_dataset_original
        
        # Create eval dataset exactly like notebook (no augmentation)
        eval_dataset = ImagesDataset(self.x_eval, self.y_eval, data_dir=self.data_dir)
        
        return train_dataset, eval_dataset

    def create_dataloaders(self, batch_size: int = 64, shuffle_train: bool = True):
        """
        Create dataloaders exactly like notebook.
        
        Args:
            batch_size: Batch size for both train and eval
            shuffle_train: Whether to shuffle training data
            
        Returns:
            Tuple of (train_dataloader, eval_dataloader)
        """
        
        train_dataset, eval_dataset = self.create_datasets()
        
        # Create dataloaders exactly like notebook
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=4, pin_memory=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
        
        return train_dataloader, eval_dataloader

    def create_test_dataset(self):
        """Create test dataset for inference."""
        
        x_test = self.test_features.filepath.to_frame()
        test_dataset = ImagesDataset(x_test, y_df=None, data_dir=self.data_dir)  # No labels for test
        
        return test_dataset

    def create_test_dataloader(self, batch_size: int = 64):
        """Create test dataloader for inference."""
        
        test_dataset = self.create_test_dataset()
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
        
        return test_dataloader

    def get_class_distribution(self, split: str = 'train') -> Dict[str, int]:
        """Get class distribution for a specific split."""
        
        if split == 'train':
            return self.y_train.sum().to_dict()
        elif split == 'eval':
            return self.y_eval.sum().to_dict()
        else:
            raise ValueError(f"Unknown split: {split}")

    def get_site_distribution(self, split: str = 'train') -> Dict[str, int]:
        """Get site distribution for a specific split."""
        
        if split == 'train':
            return self.train_sites.value_counts().to_dict()
        elif split == 'eval':
            return self.val_sites.value_counts().to_dict()
        else:
            raise ValueError(f"Unknown split: {split}")


# Convenience functions for easy usage
def create_notebook_style_datasets(
    data_dir: Union[str, Path],
    fraction: float = 1.0,
    random_state: int = 1,
    validation_sites_file: Optional[str] = None,
    batch_size: int = 64,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and eval dataloaders exactly like the notebook.
    
    This is the main function that replicates your notebook workflow
    but with proper site-aware splits.
    
    Args:
        data_dir: Path to data directory
        fraction: Fraction of data to use (notebook's frac parameter)
        random_state: Random seed (notebook uses 1)
        validation_sites_file: Path to validation sites CSV
        batch_size: Batch size for dataloaders
        **kwargs: Additional arguments for TaiParkDatasetNotebookStyle
        
    Returns:
        Tuple of (train_dataloader, eval_dataloader)
    """
    
    dataset_manager = TaiParkDatasetNotebookStyle(
        data_dir=data_dir,
        fraction=fraction,
        random_state=random_state,
        validation_sites_file=validation_sites_file,
        **kwargs
    )
    
    return dataset_manager.create_dataloaders(batch_size=batch_size)


def create_test_dataloader_notebook_style(
    data_dir: Union[str, Path],
    batch_size: int = 64
) -> DataLoader:
    """Create test dataloader for inference."""
    
    dataset_manager = TaiParkDatasetNotebookStyle(data_dir=data_dir, fraction=1.0)
    return dataset_manager.create_test_dataloader(batch_size=batch_size)


# Example usage
if __name__ == "__main__":
    
    # Replicate notebook workflow exactly
    print("Creating datasets exactly like notebook...")
    
    train_dataloader, eval_dataloader = create_notebook_style_datasets(
        data_dir="data/raw",
        fraction=1.0,  # frac parameter from notebook
        random_state=1,  # notebook's random_state
        batch_size=64,  # notebook's batch_size
        use_preprocessing=True,
        use_augmentation=True,
        num_augmentations=2
    )
    
    print(f"Train dataloader: {len(train_dataloader)} batches")
    print(f"Eval dataloader: {len(eval_dataloader)} batches")
    
    # Test loading a batch
    train_batch = next(iter(train_dataloader))
    print(f"Batch keys: {train_batch.keys()}")
    print(f"Image batch shape: {train_batch['image'].shape}")
    print(f"Label batch shape: {train_batch['label'].shape}")