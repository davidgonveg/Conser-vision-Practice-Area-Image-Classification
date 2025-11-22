import numpy as np
import torch
import os 

class EarlyStopping:
    """Detiene el entrenamiento si la pérdida de validación no mejora, guarda el mejor modelo 
       con la métrica en el nombre, y elimina el modelo anterior para ahorrar espacio."""
       
    def __init__(self, patience=7, verbose=False, delta=0, base_dir='outputs/models', filename_prefix='model'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf # Corregido a minúsculas
        self.delta = delta
        
        # Variables para el guardado inteligente
        self.base_dir = base_dir
        self.filename_prefix = filename_prefix
        self.last_best_path = None 
        # Asegurarse de que el directorio base exista
        os.makedirs(base_dir, exist_ok=True)

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                 print(f'EarlyStopping counter: {self.counter} de {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Guarda el modelo cuando la pérdida de validación disminuye y elimina el anterior."""
        
        # 1. Formatear la pérdida de validación para el nombre del archivo 
        metric_str = f"{val_loss:.4f}".replace('.', '_')
        
        # 2. Generar el nuevo nombre de archivo
        new_filename = f"{self.filename_prefix}_val_loss_{metric_str}.pth"
        new_path = os.path.join(self.base_dir, new_filename)
        
        if self.verbose:
            print(f'Pérdida de validación disminuida ({self.val_loss_min:.4f} --> {val_loss:.4f}). Guardando modelo en {new_filename}...')
        
        # 3. Guardar el nuevo mejor modelo
        torch.save(model.state_dict(), new_path) 
        
        # 4. Eliminar el archivo del mejor modelo anterior (si existe y no es el mismo)
        if self.last_best_path and os.path.exists(self.last_best_path) and self.last_best_path != new_path:
            os.remove(self.last_best_path)
            
        # 5. Actualizar estado
        self.last_best_path = new_path
        self.val_loss_min = val_loss