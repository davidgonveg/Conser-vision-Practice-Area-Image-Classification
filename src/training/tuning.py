import optuna
import yaml
import torch 
import torch.nn as nn
import torch.optim as optim
import os
import copy 
import glob 

# Importaciones necesarias para la finalización
from src.data.dataset import get_dataloaders, get_test_dataloader
from src.models.model import build_model
from src.training.trainer import Trainer
from src.models.evaluation import generate_predictions 


# ==============================================================================
# LÓGICA DE OPTUNA
# ==============================================================================

def objective(trial, base_config):
    """
    Función objetivo de Optuna: Define el espacio de búsqueda de hyperparámetros,
    ejecuta el entrenamiento y devuelve el valor a minimizar (val_loss).
    """
    print(f"\n--- TRIAL {trial.number}: Buscando hyperparámetros... ---")
    
    config = copy.deepcopy(base_config)

    # 1. Definición del Espacio de Búsqueda (ESTÁTICO)
    
    # Modelo
    model_name = trial.suggest_categorical('model_name', ['resnet50', 'resnet152'])
    config['model']['name'] = model_name
    
    # Optimizador y LR/Momentum (se mantiene igual)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    config['training']['lr'] = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    if optimizer_name == 'SGD':
        config['training']['momentum'] = trial.suggest_float('momentum', 0.8, 0.99)
        config['training']['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    
    # Batch Size: ESPACIO DE BÚSQUEDA MÁXIMO Y ESTÁTICO (se resuelve el error)
    config['data']['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64])

    # Congelamiento
    config['model']['freeze_backbone'] = trial.suggest_categorical('freeze_backbone', [True, False])
    
    config['training']['num_epochs'] = config['training']['max_tuning_epochs']

    # *****************************************************************
    # NUEVA LÓGICA DE PRUNING POR RESTRICCIÓN DE MEMORIA (OOM)
    # *****************************************************************
    batch_size = config['data']['batch_size']
    
    # Si el modelo es ResNet152 y el batch size es 64, sabemos que fallará por VRAM (OOM).
    # Lo podamos manualmente antes de empezar el entrenamiento.
    if model_name == 'resnet152' and batch_size == 64:
        print(f"ATENCIÓN: Poda por restricción de memoria (ResNet152 con Batch Size {batch_size}).")
        # Devolvemos una pérdida alta para penalizar este trial en el estudio
        return 1000.0 

    try:
        # 2. Preparar DataLoaders (el resto del código se mantiene)
        train_loader, val_loader, _ = get_dataloaders(config) 
        model = build_model(config)
        
        # 3. Definir Optimizador (se mantiene)
        criterion = nn.CrossEntropyLoss()
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(trainable_params, lr=config['training']['lr'])
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(trainable_params, 
                                  lr=config['training']['lr'], 
                                  momentum=config['training']['momentum'],
                                  weight_decay=config['training']['weight_decay'])
        
        # 4. Ejecutar Trainer en modo tuning (se mantiene)
        trainer = Trainer(
            model=model, train_loader=train_loader, val_loader=val_loader, 
            criterion=criterion, optimizer=optimizer, config=config, 
            trial_number=trial.number
        )

        final_val_loss, final_val_acc = trainer.run_tuning_mode()
        
        # 5. Corrección del error 'Trial.report() got an unexpected keyword argument 'epoch''
        # Optuna 3.x ya no acepta el argumento 'epoch' aquí, simplemente lo quitamos.
        trial.report(final_val_loss) # <--- CORREGIDO
        
        if trial.should_prune():
            # Limpieza antes de podar
            del model
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()
        
        # Limpieza estándar
        del model
        torch.cuda.empty_cache()

        return final_val_loss

    except Exception as e:
        # Captura errores de OOM u otros fallos
        print(f"Trial {trial.number} falló debido a un error: {e}")
        torch.cuda.empty_cache()
        raise optuna.exceptions.TrialPruned()


def run_tuning(config):
    """Define y ejecuta el estudio de Optuna."""
    
    # ... (código se mantiene igual)
    study = optuna.create_study(
        study_name=config['tuning']['study_name'],
        direction='minimize',  
        storage=config['paths']['tuning_db'],
        load_if_exists=True
    )
    
    study.optimize(lambda trial: objective(trial, config), 
                   n_trials=config['tuning']['n_trials'], 
                   timeout=config['tuning']['timeout_seconds'])
    
    # ... (código de impresión de resultados)
    
    # La limpieza se realiza en el main
    return study # Devolvemos el estudio para la post-procesamiento


# ==============================================================================
# LÓGICA DE FINALIZACIÓN Y SUBMISSION
# ==============================================================================

def finalize_best_model_and_submission(study, config, species_cols):
    """
    Identifica el mejor modelo global, elimina los demás, y genera el submission.csv.
    """
    if study.best_trial is None:
        print("No se puede generar la submission final: No hay trials exitosos.")
        return

    best_trial = study.best_trial
    best_trial_num = best_trial.number
    best_loss = best_trial.value
    
    best_params = best_trial.params
    best_model_name = best_params['model_name']
    
    base_tuning_dir = os.path.join("outputs", "tuning_models")
    final_output_dir = os.path.dirname(config['paths']['model_output']) 
    
    # 1. Buscar y renombrar el archivo del mejor trial
    search_pattern = os.path.join(base_tuning_dir, f"trial_{best_trial_num}_{best_model_name}_val_loss_*.pth")
    best_trial_files = glob.glob(search_pattern)
    
    if not best_trial_files:
        print(f"ADVERTENCIA: Archivo del mejor trial ({best_trial_num}) no encontrado. No se puede finalizar.")
        return

    best_trial_path = best_trial_files[0]
    
    final_model_name = f"BEST_OVERALL_{best_model_name}_val_loss_{best_loss:.4f}".replace('.', '_')
    final_path = os.path.join(final_output_dir, f"{final_model_name}.pth")
    
    # 2. Limpieza de modelos intermedios
    print("\nLimpiando modelos intermedios de los trials...")
    for f in glob.glob(os.path.join(base_tuning_dir, "*.pth")):
        if f != best_trial_path:
            os.remove(f)
    print(f"Se eliminaron los modelos que no eran el mejor de {base_tuning_dir}.")
    
    # 3. Renombrar y mover el mejor modelo
    if os.path.exists(final_path):
        os.remove(final_path) 
    
    os.rename(best_trial_path, final_path)
    print(f"✅ Mejor modelo (Trial {best_trial_num}) renombrado y movido a: {final_path}")
    
    # 4. Generar Submission Final con el mejor modelo
    
    # Recargamos el modelo con los parámetros del mejor trial
    config_for_loading = copy.deepcopy(config)
    config_for_loading['model']['name'] = best_model_name
    config_for_loading['model']['freeze_backbone'] = best_params.get('freeze_backbone', True)
    
    best_model_arch = build_model(config_for_loading) 

    # Cargar el estado del modelo
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    best_model_arch.load_state_dict(torch.load(final_path, map_location=device))
    
    # Generar DataLoader de Prueba
    test_loader = get_test_dataloader(config)

    # Generar el submission file (guarda el submission.csv)
    generate_predictions(best_model_arch, test_loader, config, species_cols)
    
    print("✨ Submission.csv generada exitosamente con el MEJOR MODELO GLOBAL.")

    # Intentar eliminar la carpeta temporal (si está vacía)
    try:
        if not os.listdir(base_tuning_dir):
            os.rmdir(base_tuning_dir)
            print(f"Directorio de tuning ({base_tuning_dir}) eliminado.")
    except OSError:
        pass