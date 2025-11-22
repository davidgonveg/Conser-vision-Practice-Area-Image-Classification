import torch
import torch.nn as nn
from torchvision import models

def build_model(config):
    """
    Carga un modelo preentrenado basado en el nombre de la configuración, 
    modifica el clasificador final y opcionalmente congela el backbone.

    Args:
        config (dict): Diccionario de configuración.
                       Debe contener 'model'['name'] y 'model'['num_classes'].

    Returns:
        torch.nn.Module: El modelo PyTorch listo.
    """
    model_name = config['model']['name'].lower()
    
    # 1. Cargar el modelo base y sus pesos preentrenados
    if model_name == 'resnet152':
        model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
        num_ftrs = model.fc.in_features
        model.classifier_layer = model.fc # Guardamos la referencia para el reemplazo
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        num_ftrs = model.fc.in_features
        model.classifier_layer = model.fc
    elif model_name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # El clasificador de VGG es diferente (una secuencia de capas)
        num_ftrs = model.classifier[6].in_features
        model.classifier_layer = model.classifier
    else:
        raise ValueError(f"Modelo '{model_name}' no soportado para tuning.")
    
    # 2. Congelar el backbone si la configuración lo indica
    if config['model'].get('freeze_backbone', True):
        for param in model.parameters():
            param.requires_grad = False
    
    # 3. Reemplazar la capa de clasificación (Head)
    new_head = nn.Sequential(
        nn.Linear(num_ftrs, config['model']['num_classes'])
    )
    
    if 'resnet' in model_name:
        model.fc = new_head
    elif 'vgg' in model_name:
        # Reemplazar la última capa de la secuencia del clasificador en VGG
        model.classifier[-1] = new_head[0] # [0] porque nn.Sequential tiene un solo Linear

    # 4. Mover el modelo al dispositivo (GPU/CPU)
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"Modelo {model_name} cargado. Clasificador configurado para {config['model']['num_classes']} clases.")
        
    return model