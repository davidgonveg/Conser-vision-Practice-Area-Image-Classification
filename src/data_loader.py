import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
from sklearn.model_selection import train_test_split
from pathlib import Path
from .config import TRAIN_FEATURES, TRAIN_LABELS, BATCH_SIZE, SEED, IMAGE_DIR
from .preprocessing import get_base_transforms, custom_preprocessing

class ImagesDataset(Dataset):
    """
    Reads in an image, transforms pixel values, and serves
    a dictionary containing the image id, image tensors, and label.
    """
    def __init__(self, x_df, y_df=None, preprocessing=None, augmentation=None):
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

    def __getitem__(self, index):
        # The CSV contains paths like 'train_features/ZJ000000.jpg' or 'test_features/...'
        # But actual images are in 'data/raw/images'.
        # We take the filename and join with IMAGE_DIR.
        csv_filepath = self.data.iloc[index]["filepath"]
        filename = Path(csv_filepath).name
        image_path = IMAGE_DIR / filename
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

def get_data_splits(train_features, train_labels, test_size=0.25, random_state=SEED):
    """
    Splits data into training and evaluation sets.
    """
    # Align X and y
    y = train_labels
    x = train_features.loc[y.index].filepath.to_frame()
    
    x_train, x_eval, y_train, y_eval = train_test_split(
        x, y, stratify=y, test_size=test_size, random_state=random_state
    )
    return x_train, x_eval, y_train, y_eval

def create_combined_dataset(x_train, y_train, num_augmentations, augmentation_functions):
    """
    Creates a combined dataset with the original dataset and multiple augmented versions.
    """
    # Original dataset with custom preprocessing but NO augmentation
    original_dataset = ImagesDataset(
        x_train, y_train, preprocessing=custom_preprocessing, augmentation=None
    )
    
    datasets = [original_dataset]
    for i in range(num_augmentations):
        # Augmented datasets: apply preprocessing AND augmentation
        augmented_dataset = ImagesDataset(
            x_train, y_train, 
            preprocessing=custom_preprocessing,
            augmentation=augmentation_functions
        )
        datasets.append(augmented_dataset)

    combined_dataset = ConcatDataset(datasets)
    return combined_dataset

def get_dataloaders(x_train, y_train, x_eval, y_eval, augmentation_functions=None, batch_size=BATCH_SIZE):
    """
    Creates DataLoaders for training and evaluation.
    """
    # Training set
    if augmentation_functions:
        # Create augmented dataset (2 augmentations as per notebook default)
        train_dataset = create_combined_dataset(x_train, y_train, 2, augmentation_functions)
    else:
        train_dataset = ImagesDataset(x_train, y_train, preprocessing=custom_preprocessing)
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) # num_workers=0 for Windows safety
    
    # Evaluation set (No augmentation)
    eval_dataset = ImagesDataset(x_eval, y_eval, preprocessing=custom_preprocessing)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, eval_loader
