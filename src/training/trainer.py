import torch
import torch.nn as nn
from tqdm import tqdm
import os
import pandas as pd

from src.training.utils import EarlyStopping

class Trainer:
    """Clase para encapsular la lógica principal de entrenamiento y validación del modelo."""
    
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, config, trial_number=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        
        self.device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.log_path = config['paths']['logs_output']
        
        # Configuración del guardado (ÚNICA para cada trial/run)
        model_name = config['model']['name']
        base_dir = os.path.dirname(config['paths']['model_output'])
        
        prefix = f"{model_name}"
        if trial_number is not None:
            # Prefijo único para el tuning: trial_XX_modelname
            base_dir = os.path.join("outputs", "tuning_models") # Guardar en carpeta temporal
            prefix = f"trial_{trial_number}_{model_name}" 
        
        # Inicializar Early Stopping
        self.early_stopper = EarlyStopping(
            patience=config['training']['early_stopping_patience'], 
            verbose=True, 
            base_dir=base_dir, 
            filename_prefix=prefix
        )
        
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    def _train_epoch(self):
        """Ejecuta un pase completo sobre el dataset de entrenamiento."""
        self.model.train()
        running_loss = 0.0
        
        train_pbar = tqdm(self.train_loader, desc="Training")
        
        for inputs, labels in train_pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            train_pbar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def _validate_epoch(self):
        """Ejecuta un pase completo sobre el dataset de validación."""
        self.model.eval()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        with torch.no_grad():
            val_pbar = tqdm(self.val_loader, desc="Validation")
            for inputs, labels in val_pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total_preds += labels.size(0)
                correct_preds += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_acc = correct_preds / total_preds
        return epoch_loss, epoch_acc

    def run(self):
        """Bucle principal de entrenamiento (Modo RUN único)."""
        print(f"Comenzando entrenamiento en dispositivo: {self.device}")
        
        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            print(f"\n--- Época {epoch}/{self.config['training']['num_epochs']} ---")
            
            train_loss = self._train_epoch()
            val_loss, val_acc = self._validate_epoch()
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            self.early_stopper(val_loss, self.model)
            
            if self.early_stopper.early_stop:
                print("¡Early stopping activado!")
                break

        print("\nEntrenamiento finalizado. Cargando el mejor modelo.")
        
        # Cargar el mejor modelo (guardado por EarlyStopping)
        self.model.load_state_dict(torch.load(self.early_stopper.last_best_path, map_location=self.device))
        
        self._save_history()
        
        return self.model

    def run_tuning_mode(self):
        """Bucle principal de entrenamiento simplificado para Optuna."""
        
        # Usamos las épocas máximas de tuning
        max_epochs = self.config['training']['max_tuning_epochs']
        
        for epoch in range(1, max_epochs + 1):
            
            # --- FASE DE ENTRENAMIENTO CON BARRA DE PROGRESO ---
            self.model.train()
            
            # Barra de progreso para el entrenamiento: muestra el número de Trial y la Época
            train_pbar = tqdm(
                self.train_loader, 
                desc=f"Trial {self.early_stopper.filename_prefix} - Epoch {epoch}/{max_epochs} (Train)", 
                leave=False # Importante para que las barras se borren y no saturen la terminal
            ) 
            
            for inputs, labels in train_pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                # Actualizar la descripción de la barra de progreso con la pérdida actual
                train_pbar.set_postfix({'loss': loss.item()})


            # --- VALIDACIÓN (EXISTENTE) ---
            # La validación sigue usando la barra definida en _validate_epoch
            val_loss, val_acc = self._validate_epoch()
            
            self.early_stopper(val_loss, self.model)
            
            if self.early_stopper.early_stop:
                break
                
        # Cargar el mejor modelo (el último guardado, que es el mejor de este trial)
        self.model.load_state_dict(torch.load(self.early_stopper.last_best_path, map_location=self.device))
        
        # Devolver la mejor pérdida de validación encontrada para Optuna
        return self.early_stopper.val_loss_min, val_acc


    def _save_history(self):
        """Guarda el historial de entrenamiento en un archivo CSV."""
        df_history = pd.DataFrame(self.history)
        
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
        df_history.to_csv(self.log_path, index=False)
        print(f"Historial de entrenamiento guardado en: {self.log_path}")