import yaml
import argparse
import torch.nn as nn
import torch.optim as optim
import os
import sys

# Añadimos la ruta de la carpeta raíz al PATH para Optuna si es necesario
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

# Importar módulos del proyecto
from src.data.dataset import get_dataloaders, get_test_dataloader
from src.models.model import build_model
from src.training.trainer import Trainer
from src.models.evaluation import generate_predictions
from src.training.tuning import run_tuning, finalize_best_model_and_submission


def load_config(config_path="config.yaml"):
    """Carga el archivo de configuración YAML, forzando la codificación UTF-8 y verificando contenido."""
    
    # 1. Manejo de la ruta
    if not os.path.exists(config_path):
        abs_path = os.path.join(os.getcwd(), config_path)
        if os.path.exists(abs_path):
            config_path = abs_path
        else:
            # Si el archivo no existe, lanzamos un error claro.
            raise FileNotFoundError(f"Archivo de configuración no encontrado en: {config_path}")
        
    # 2. Carga del archivo con codificación explícita
    # La codificación UTF-8 previene el error 'NoneType' causado por el Byte Order Mark (BOM).
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        raise IOError(f"Error al cargar el YAML: {e}")

    # 3. Verificación de contenido
    if config is None:
        raise ValueError(f"El archivo '{config_path}' está vacío o solo contiene comentarios. Asegúrate de que tenga contenido YAML válido.")
        
    return config


def main_run(config, species_cols):
    """Ejecuta un solo entrenamiento (el flujo original del notebook)."""
    print("\n--- MODO: ENTRENAMIENTO ÚNICO ---")

    # 1. Preparar los datos 
    # Ya cargamos species_cols antes, solo necesitamos los DataLoaders
    train_loader, val_loader, _ = get_dataloaders(config) 
    test_loader = get_test_dataloader(config)
    
    # 2. Construir el modelo (con la configuración Fija)
    model = build_model(config)
    
    # 3. Definir optimizador y pérdida
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config['training']['lr'], 
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 4. Ejecutar Trainer
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, config)
    best_model = trainer.run()
    
    # 5. Generar submission
    generate_predictions(best_model, test_loader, config, species_cols)
    
    print("\n--- PROCESO COMPLETO FINALIZADO (RUN) ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entrenamiento/Tuning de clasificador de vida silvestre.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Ruta al archivo de configuración YAML.')
    
    parser.add_argument('--mode', type=str, default='run', choices=['run', 'tune'], 
                        help='Modo de operación: "run" (entrenamiento único) o "tune" (búsqueda de hyperparámetros).')
    
    args = parser.parse_args()
    
    config = load_config(args.config)

    # 1. CARGA INICIAL DE DATOS PARA OBTENER LOS NOMBRES DE LAS COLUMNAS DE ESPECIE
    # Necesitamos esto tanto para 'run' como para 'tune'
    _, _, species_cols = get_dataloaders(config) 

    if args.mode == 'run':
        os.makedirs(os.path.dirname(config['paths']['model_output']), exist_ok=True)
        main_run(config, species_cols)
        
    elif args.mode == 'tune':
        # 2. Configuración de Optuna (inyección de parámetros)
        config['tuning'] = {
            'n_trials': 50, 
            'timeout_seconds': None,
            'study_name': 'wildlife_classification_study',
        }
        config['training']['max_tuning_epochs'] = 15

        # 3. Creación de directorios para guardado
        tuning_dir = os.path.join("outputs", "tuning_models")
        final_models_dir = os.path.join("outputs", "models")
        os.makedirs(tuning_dir, exist_ok=True) 
        os.makedirs(final_models_dir, exist_ok=True) 

        # 4. Ejecución del Tuning
        study = run_tuning(config)
        
        # 5. Finalización y generación de submission del mejor modelo
        finalize_best_model_and_submission(study, config, species_cols)