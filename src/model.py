import torch.nn as nn
import torchvision.models as models
from .config import NUM_CLASSES

def get_model(num_classes=NUM_CLASSES):
    """
    Initializes the model architecture:
    - Pretrained ResNet152 backbone
    - Frozen early layers (only layer4 and fc trainable)
    - Custom classification head
    """
    weights = models.ResNet152_Weights.DEFAULT
    model = models.resnet152(weights=weights)

    # Freeze all layers except layer4
    for name, param in model.named_parameters():
        if 'layer4' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Replace the final fully connected layer
    # Input features for ResNet152 fc layer is 2048
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5), # First dropout layer
        nn.Linear(1024, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3), # Second dropout layer
        nn.Linear(256, num_classes)
    )
    
    return model
