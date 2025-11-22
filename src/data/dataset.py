import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

from src.data.processing import get_train_transforms, get_val_test_transforms

# ==============================================================================
# CLASES CUSTOM DATASET
# ==============================================================================

class ImagesDataset(Dataset):
    """Clase PyTorch Dataset para cargar imágenes de entrenamiento y validación."""
    def __init__(self, data_frame, transforms=None):
        self.data_frame = data_frame
        self.transforms = transforms
        self.image_dir = 'data/raw/images'
        
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        image_id = self.data_frame.iloc[idx]['image_id']
        label = self.data_frame.iloc[idx]['target']
        
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert('RGB')
        
        if self.transforms:
            image = self.transforms(image)
            
        return image, label


class TestDataset(Dataset):
    """Clase PyTorch Dataset para cargar imágenes de prueba (sin etiquetas)."""
    def __init__(self, data_frame, transforms=None):
        self.data_frame = data_frame
        self.transforms = transforms
        self.image_dir = 'data/raw/images'
        
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        image_id = self.data_frame.iloc[idx]['id'] 
        
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert('RGB')
        
        if self.transforms:
            image = self.transforms(image)
            
        return image, image_id 

# ==============================================================================
# FUNCIONES DE ORQUESTACIÓN DE DATOS
# ==============================================================================

def get_dataloaders(config):
    """Carga, divide y crea los DataLoaders de entrenamiento y validación."""
    
    features = pd.read_csv(config['paths']['train_features'])
    labels = pd.read_csv(config['paths']['train_labels'])
    
    full_data = features.merge(labels, on='id') 
    full_data = full_data.rename(columns={'id': 'image_id'})
    
    if config['data']['frac'] < 1.0:
        full_data = full_data.sample(frac=config['data']['frac'], 
                                     random_state=config['data']['random_state']).reset_index(drop=True)
    
    # Mapear one-hot a etiqueta entera (target)
    species_cols = [col for col in full_data.columns if col not in ['image_id', 'filepath', 'site']]
    full_data['target'] = full_data[species_cols].values.argmax(axis=1)

    # Split del dataset (Train / Validation)
    train_df, val_df = train_test_split(
        full_data, 
        test_size=config['data']['test_size'], 
        random_state=config['data']['random_state'],
        stratify=full_data['target'] 
    )

    # Crear DataLoaders
    train_transforms = get_train_transforms()
    val_transforms = get_val_test_transforms()

    train_dataset = ImagesDataset(train_df.reset_index(drop=True), transforms=train_transforms)
    val_dataset = ImagesDataset(val_df.reset_index(drop=True), transforms=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=4)
    
    print(f"Datos cargados: Train Samples={len(train_dataset)}, Val Samples={len(val_dataset)}")
    
    return train_loader, val_loader, species_cols

def get_test_dataloader(config):
    """Carga y crea el DataLoader del conjunto de prueba."""
    
    test_df = pd.read_csv(config['paths']['test_features'])
    
    test_transforms = get_val_test_transforms()
    test_dataset = TestDataset(test_df.reset_index(drop=True), transforms=test_transforms)

    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['data']['batch_size'], 
        shuffle=False, 
        num_workers=4
    )
    
    print(f"Test Samples={len(test_dataset)}")
    return test_loader