import torch
from torchvision import transforms

# Media y Desviación Estándar de ImageNet 
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transforms():
    """Define las transformaciones de datos para entrenamiento (con aumento de datos)."""
    return transforms.Compose([
        transforms.Resize(256), 
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def get_val_test_transforms():
    """Define las transformaciones para validación y prueba (sin aumento de datos)."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])