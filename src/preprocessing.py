from PIL import Image, ImageEnhance, ImageFilter
from torchvision.transforms import v2 as transforms
import torch

def custom_preprocessing(image):
    """
    Applies custom preprocessing to a PIL image:
    - Color enhancement (lower saturation)
    - Brightness enhancement
    - Contrast enhancement
    - Unsharp mask
    """
    # Convert to grayscale might help focus on texture and shape (Note: notebook comment said this but code does enhancements)
    # The code in notebook:
    image = ImageEnhance.Color(image).enhance(0.8)  # Slightly less saturation
    image = ImageEnhance.Brightness(image).enhance(1.1)  # Increase brightness slightly
    image = ImageEnhance.Contrast(image).enhance(1.2)  # Increase contrast
    
    # Apply soft focus/unsharp mask to highlight animals against complex backgrounds
    image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=50))
    
    return image

def get_augmentation_transforms():
    """
    Returns the composition of augmentation transforms used in training.
    """
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
        transforms.RandomRotation(degrees=12),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomEqualize(p=0.33),
    ])

def get_base_transforms():
    """
    Returns the basic transforms (Resize, ToTensor, Normalize) applied to all images.
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),
    ])
