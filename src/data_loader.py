import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedGroupKFold
from pathlib import Path
from .config import TRAIN_FEATURES, TRAIN_LABELS, BATCH_SIZE, SEED, IMAGE_DIR, USE_AUGMENTATION
from .preprocessing import get_base_transforms, custom_preprocessing

class ImagesDataset(Dataset):
    """
    Reads in an image, transforms pixel values, and serves
    a dictionary containing the image id, image tensors, and label.
    """
    def __init__(self, x_df, y_df=None, preprocessing=None, augmentation=None, image_dir=None):
        self.data = x_df
        # Ensure 'filepath' column exists if x_df is not exactly the structure we expect
        if "filepath" not in self.data.columns and isinstance(self.data, pd.DataFrame):
             # If passed a dataframe with just the filepath as index or similar, adjust here. 
             # In notebook x was `filepath.to_frame()`.
             pass 
        
        self.label = y_df
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.transform = get_base_transforms()
        self.image_dir = Path(image_dir) if image_dir else IMAGE_DIR

    def __getitem__(self, index):
        # The CSV contains paths like 'train_features/ZJ000000.jpg' or 'test_features/...'
        # But actual images are in 'data/raw/images'.
        # We take the filename and join with IMAGE_DIR.
        csv_filepath = self.data.iloc[index]["filepath"]
        filename = Path(csv_filepath).name
        
        # Defensive check for multiprocessing attribute access issues
        if not hasattr(self, 'image_dir'):
            from .config import IMAGE_DIR
            self.image_dir = IMAGE_DIR
            
        image_path = self.image_dir / filename
        image = Image.open(image_path).convert("RGB")

        # Custom Preprocessing (PIL level)
        if self.preprocessing is not None:
            image = self.preprocessing(image)
        
        # Augmentation (PIL/Tensor level depending on v2)
        if self.augmentation is not None:
            image = self.augmentation(image)
        
        # Base Transform (Resize, ToTensor, Normalize)
        image = self.transform(image)
        
        image_id = self.data.index[index]
        
        if self.label is None:
            sample = {"image_id": image_id, "image": image}
        else:
            label_values = self.label.iloc[index].values
            label = torch.tensor(label_values, dtype=torch.float)
            sample = {"image_id": image_id, "image": image, "label": label}
            
        return sample

    def __len__(self):
        return len(self.data)

def load_data():
    """
    Loads train features and labels CSVs.
    Returns:
        train_features (pd.DataFrame)
        train_labels (pd.DataFrame)
    """
    if not TRAIN_FEATURES.exists():
        raise FileNotFoundError(f"Train features file not found at {TRAIN_FEATURES}")
        
    train_features = pd.read_csv(TRAIN_FEATURES, index_col="id")
    train_labels = pd.read_csv(TRAIN_LABELS, index_col="id")
    return train_features, train_labels

def get_data_splits(train_features, train_labels, groups=None, test_size=0.25, random_state=SEED):
    """
    Splits data into training and evaluation sets.
    If groups are provided, uses GroupShuffleSplit to ensure no group overlap.
    """
    # Align X and y
    y = train_labels
    x = train_features.loc[y.index].filepath.to_frame()
    
    if groups is not None:
        # Align groups with y
        groups = groups.loc[y.index]
        
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        # generated indices are relative to x/y, which are aligned
        train_idx, eval_idx = next(gss.split(x, y, groups))
        
        x_train = x.iloc[train_idx]
        x_eval = x.iloc[eval_idx]
        y_train = y.iloc[train_idx]
        y_eval = y.iloc[eval_idx]
    else:
        # Fallback to stratified shuffle split
        x_train, x_eval, y_train, y_eval = train_test_split(
            x, y, stratify=y, test_size=test_size, random_state=random_state
        )

    return x_train, x_eval, y_train, y_eval

def get_kfold_splits(train_features, train_labels, params_n_folds=5, random_state=SEED):
    """
    Returns a generator for K-Fold splits (StratifiedGroupKFold).
    Yields: (fold_index, x_train, x_eval, y_train, y_eval)
    """
    y = train_labels
    x = train_features.loc[y.index].filepath.to_frame()
    groups = train_features.loc[y.index, 'site']
    
    sgkf = StratifiedGroupKFold(n_splits=params_n_folds, shuffle=True, random_state=random_state)
    
    # StratifiedGroupKFold expects 1D array of class labels for 'y', not One-Hot Encoded
    # We convert OHE to class indices for the split logic
    y_split = y.idxmax(axis=1) # Convert to series of class names/indices
    
    # We pass y_split just for stratification purposes
    for fold, (train_idx, eval_idx) in enumerate(sgkf.split(x, y_split, groups)):
        x_train = x.iloc[train_idx]
        x_eval = x.iloc[eval_idx]
        y_train = y.iloc[train_idx]
        y_eval = y.iloc[eval_idx]
        
        yield fold, x_train, x_eval, y_train, y_eval

def get_dataloaders(x_train, y_train, x_eval, y_eval, augmentation_functions=None, batch_size=BATCH_SIZE, image_dir=None):
    """
    Creates DataLoaders for training and evaluation.
    """
    # Augmentation Control
    if not USE_AUGMENTATION:
        augmentation_functions = None
        
    # Training set
    # Training set
    # Using On-the-Fly augmentation:
    # The dataset wraps individual images. When __getitem__ is called, it applies random transforms.
    # Num_workers in DataLoader ensures these transforms happen in parallel on CPU while GPU trains.
    train_dataset = ImagesDataset(
        x_train, y_train, 
        preprocessing=custom_preprocessing, 
        augmentation=augmentation_functions, 
        image_dir=image_dir
    )
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True) # Optimized for speed
    
    # Evaluation set (No augmentation)
    eval_dataset = ImagesDataset(x_eval, y_eval, preprocessing=custom_preprocessing, image_dir=image_dir)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, eval_loader
