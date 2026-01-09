import torch.nn as nn
import torchvision.models as models
from .config import NUM_CLASSES, MODEL_ARCH

def build_custom_head(input_features, num_classes):
    """
    Creates a custom classification head
    """
    return nn.Sequential(
        nn.Linear(input_features, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(1024, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

def get_resnet152(num_classes):
    weights = models.ResNet152_Weights.DEFAULT
    model = models.resnet152(weights=weights)

    # Freeze all layers except layer4
    for name, param in model.named_parameters():
        if 'layer4' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Replace fc
    model.fc = build_custom_head(model.fc.in_features, num_classes)
    return model

def get_efficientnet_v2_s(num_classes):
    weights = models.EfficientNet_V2_S_Weights.DEFAULT
    model = models.efficientnet_v2_s(weights=weights)
    
    # Freeze early layers
    # EfficientNet features: features.0 ... features.7
    # unfreeze last blocks (features.7) and classifier
    for name, param in model.features.named_parameters():
        if 'features.7' in name: # Last block
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    # Replace classifier
    # Original classifier is Sequential(Dropout, Linear)
    # in_features for V2-S is 1280
    in_features = model.classifier[1].in_features
    model.classifier = build_custom_head(in_features, num_classes)
    
    return model


def get_convnext_large(num_classes):
    # ConvNeXt Large weights (ImageNet-1k)
    # Using default weights from torchvision
    weights = models.ConvNeXt_Large_Weights.DEFAULT
    model = models.convnext_large(weights=weights)
    
    # Freeze implementation details:
    # convnext features are in model.features
    # We freeze everything except the last few blocks if desired,
    # or just keep it simple and freeze all features for now, unfreezing the last stage?
    # For simplicity and robust transfer, let's freeze all `features` and only train classifier, 
    # OR unfreeze the last stage (stage 7 equivalent).
    # ConvNeXt structure: features[0]..features[7]
    
    # Let's freeze all features first
    # for param in model.features.parameters():
    #      param.requires_grad = False
    
    # Actually, for "Modern" models, fine-tuning more layers usually helps.
    # Let's verify what the user usually does. The user used "layer4" in ResNet.
    # In ConvNeXt, features[7] is the last block.
    
    for name, param in model.features.named_parameters():
        if 'features.7' in name:
             param.requires_grad = True
        else:
             param.requires_grad = False

    # Replace classifier
    # ConvNeXt classifier is Sequential(LayerNorm2d?, Flatten, Linear)
    # Actually checking torchvision source:
    # classifier = Sequential(LayerNorm2d, Flatten, Linear)
    # But input features to classifier[2] (Linear) is 1536 for Large.
    
    # We will replace the last Linear layer but keep LayerNorm/Flatten if possible
    # Or just replace the whole block with our Custom Head.
    # Our Custom Head starts with Linear, so we need Flatten first.
    
    # Let's respect the original Norm which is important for ConvNeXt
    # model.classifier[2] is the Linear layer.
    
    in_features = model.classifier[2].in_features # Should be 1536
    
    # We replace the single Linear layer with our heavier head
    model.classifier[2] = build_custom_head(in_features, num_classes)
    
    return model

def get_model(num_classes=NUM_CLASSES, model_name=MODEL_ARCH):
    """
    Initializes the model architecture based on config
    """
    if model_name == 'resnet152':
        return get_resnet152(num_classes)
    elif model_name == 'efficientnet_v2_s':
        return get_efficientnet_v2_s(num_classes)
    elif model_name == 'convnext_large':
        return get_convnext_large(num_classes)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")
