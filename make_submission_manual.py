import torch
import os
import sys
import copy

# Añadimos la ruta de la carpeta raíz al PATH 
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 

# Importar funciones necesarias
from src.main import load_config
from src.models.model import build_model
from src.data.dataset import get_dataloaders, get_test_dataloader
from src.models.evaluation import generate_predictions

# ==============================================================================
# ⚠️ CONFIGURACIÓN MANUAL DEL MODELO A EVALUAR ⚠️
# ==============================================================================
MODEL_FILE_NAME = "trial_27_resnet50_val_loss_0_4537.pth" # <-- MEJOR GLOBAL
MODEL_PATH = os.path.join("outputs", "tuning_models", MODEL_FILE_NAME)
MODEL_NAME = "resnet50" # <-- EL MODELO USADO EN ESE TRIAL

# Ruta donde se guardará la submission. 
SUBMISSION_OUTPUT = f"submission_{MODEL_NAME}_BEST_GLOBAL_0_4537.csv"


def create_manual_submission():
    """Carga un modelo específico de tuning_models y genera un submission.csv."""
    
    print(f"--- Generando Submission para: {MODEL_FILE_NAME} ---")
    
    # 1. Cargar Configuración base
    config = load_config()

    # 2. Obtener Columnas de Especies
    # Necesario para el formato final de la submission.csv
    _, _, species_cols = get_dataloaders(config)

    # 3. Ajustar Configuración para el modelo específico
    temp_config = copy.deepcopy(config)
    temp_config['model']['name'] = MODEL_NAME
    # Revisa si este modelo fue entrenado con freeze_backbone: False (el valor por defecto es True)
    # Usaremos el valor de config.yaml, pero si falla puedes probar a cambiarlo aquí
    # temp_config['model']['freeze_backbone'] = True # o False, dependiendo del trial

    # 4. Construir Arquitectura del Modelo
    model = build_model(temp_config)
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    
    # 5. Cargar Pesos del Modelo
    print(f"Cargando pesos desde {MODEL_PATH} en el dispositivo {device}...")
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
    except FileNotFoundError:
        print(f"ERROR: Archivo de modelo no encontrado en {MODEL_PATH}")
        return

    # 6. Generar DataLoader de Prueba
    test_loader = get_test_dataloader(config)

    # 7. Generar Submission
    temp_config['paths']['submission_output'] = SUBMISSION_OUTPUT # Sobreescribimos la ruta temporal
    generate_predictions(model, test_loader, temp_config, species_cols)
    
    print(f"✨ Submission final guardada como: {SUBMISSION_OUTPUT}")
    print("¡El proceso ha finalizado!")


if __name__ == "__main__":
    create_manual_submission()