"""
Tai Park Wildlife Trainer - Notebook Style Implementation

This module replicates the exact training logic from the successful notebook
including early stopping, learning rate scheduling, loss tracking, and model saving.
"""

import copy
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import os
import numpy as np

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import (
    TaiParkDatasetNotebookStyle, 
    ImagesDataset, 
    create_combined_dataset,
    custom_preprocessing,
    data_augmentation
)
from ..models.model import WildlifeClassifier

logger = logging.getLogger(__name__)


class NotebookStyleTrainer:
    """
    Trainer that replicates the exact training logic from the notebook.
    
    Includes all notebook features:
    - Dynamic dataset recreation each epoch (with augmentation)
    - Evaluation at quarter-steps within epochs
    - Early stopping with custom logic
    - Learning rate scheduling
    - Detailed loss tracking
    - Best model state saving
    """
    
    def __init__(
        self,
        model: WildlifeClassifier,
        dataset_manager: TaiParkDatasetNotebookStyle,
        device: Optional[Union[str, torch.device]] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        early_stopping_config: Optional[Dict] = None
    ):
        """
        Initialize trainer exactly like notebook setup.
        
        Args:
            model: WildlifeClassifier model
            dataset_manager: TaiParkDatasetNotebookStyle instance
            device: Device for training
            criterion: Loss function (defaults to CrossEntropyLoss)
            optimizer: Optimizer (defaults to notebook's SGD setup)
            scheduler: Learning rate scheduler (defaults to ReduceLROnPlateau)
            early_stopping_config: Early stopping configuration
        """
        
        self.model = model
        self.dataset_manager = dataset_manager
        
        # Setup device exactly like notebook
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")
        else:
            self.device = torch.device(device) if isinstance(device, str) else device
        
        self.model.to(self.device)
        
        # Setup loss function exactly like notebook
        self.criterion = criterion or nn.CrossEntropyLoss()
        
        # Setup optimizer exactly like notebook
        if optimizer is None:
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=0.01, 
                momentum=0.909431,  # From notebook's Optuna optimization
                weight_decay=0.005
            )
        else:
            self.optimizer = optimizer
        
        # Setup scheduler exactly like notebook
        if scheduler is None:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 
                patience=2, 
                factor=0.72
            )
        else:
            self.scheduler = scheduler
        
        # Early stopping configuration exactly like notebook
        self.early_stopping_config = early_stopping_config or {
            'min_delta': 0.0001,
            'tolerance': 5
        }
        
        # Training state tracking like notebook
        self.tracking_loss = pd.DataFrame(columns=['Epoch', 'Loss'])
        self.tracking_loss_ev = pd.DataFrame(columns=['Epoch', 'Loss'])
        self.best_eval_loss = float('inf')
        self.best_model_state = None
        self.early_stopping_counter = 0

    def early_stopping(
        self, 
        current_val_loss: float, 
        previous_val_loss: float, 
        min_delta: float, 
        tolerance: int, 
        counter: int
    ) -> Union[str, int]:
        """
        Early stopping logic exactly like notebook.
        
        Returns:
            "True" if should stop, "False" if should reset counter, or new counter value
        """
        if (current_val_loss - previous_val_loss) > min_delta:
            counter += 1
            if counter >= tolerance:
                return "True"
            return counter
        return "False"

    def train_epoch(
        self, 
        epoch: int, 
        num_epochs: int,
        batch_size: int = 64
    ) -> bool:
        """
        Train one epoch exactly like notebook.
        
        Recreates dataset each epoch and evaluates at quarter-steps.
        
        Returns:
            True if should continue training, False if early stopping triggered
        """
        
        print(f"Starting epoch {epoch}")
        
        # Recreate dataset each epoch exactly like notebook
        train_dataset_original = ImagesDataset(
            self.dataset_manager.x_train, 
            self.dataset_manager.y_train, 
            preprocessing=custom_preprocessing, 
            augmentation=None,
            data_dir=self.dataset_manager.data_dir
        )
        
        train_dataset = create_combined_dataset(
            train_dataset_original, 
            2,  # num_augmentations from notebook
            data_augmentation,
            self.dataset_manager.x_train, 
            self.dataset_manager.y_train,
            self.dataset_manager.data_dir
        )
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        # Create eval dataloader
        eval_dataset = ImagesDataset(self.dataset_manager.x_eval, self.dataset_manager.y_eval, data_dir=self.dataset_manager.data_dir)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
        
        steps_per_epoch = len(train_dataloader)
        quarter_step = steps_per_epoch // 10  # Evaluation frequency from notebook
        
        self.model.train()
        
        for batch_n, batch in tqdm(
            enumerate(train_dataloader), 
            total=len(train_dataloader), 
            colour="cyan"  # Notebook color
        ):
            
            # Standard training step
            self.optimizer.zero_grad()
            outputs = self.model(batch["image"].to(self.device))
            loss = self.criterion(outputs, batch["label"].to(self.device))
            
            # Evaluation at quarter-steps exactly like notebook
            if (batch_n == 0) or (batch_n % quarter_step == 0) or (batch_n == steps_per_epoch - 1):
                
                epoch_progress = (batch_n + 1) / steps_per_epoch
                
                # Track training loss
                self.tracking_loss.loc[len(self.tracking_loss)] = [
                    epoch + epoch_progress - 1, 
                    float(loss)
                ]
                
                # Evaluate model exactly like notebook
                eval_loss = self._evaluate_model(eval_dataloader)
                avg_eval_loss = eval_loss
                
                # Track evaluation loss
                self.tracking_loss_ev.loc[len(self.tracking_loss_ev)] = [
                    epoch + epoch_progress - 1, 
                    avg_eval_loss
                ]
                
                # Learning rate scheduling
                self.scheduler.step(avg_eval_loss)
                
                # Logging exactly like notebook
                print(f"Epoch {epoch} - Train Loss: {float(loss)}")
                print(f"Epoch {epoch} - Eval Loss: {avg_eval_loss}")
                print(f"Counter {self.early_stopping_counter}")
                
                # Update best model exactly like notebook
                if avg_eval_loss < self.best_eval_loss:
                    self.best_eval_loss = avg_eval_loss
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                
                # Early stopping check exactly like notebook
                if (epoch > 1) and ((epoch_progress < 0.97) or (epoch_progress > 0.03) or epoch_progress == 0):
                    
                    if len(self.tracking_loss_ev) >= 2:
                        ultimo_valor_loss = self.tracking_loss_ev.iloc[-1]['Loss']
                        penultimo_valor_loss = self.tracking_loss_ev.iloc[-2]['Loss']
                        
                        aux = self.early_stopping(
                            ultimo_valor_loss, 
                            penultimo_valor_loss,
                            min_delta=self.early_stopping_config['min_delta'],
                            tolerance=self.early_stopping_config['tolerance'],
                            counter=self.early_stopping_counter
                        )
                        
                        if aux == "True":
                            print("Early Stopping triggered")
                            return False  # Stop training
                        elif aux == "False":
                            self.early_stopping_counter = 0
                        else:
                            self.early_stopping_counter = aux
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
        
        return True  # Continue training

    def _evaluate_model(self, eval_dataloader: DataLoader) -> float:
        """Evaluate model exactly like notebook."""
        
        eval_loss = 0.0
        total_batches = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, disable=True):
                logits = self.model.forward(batch["image"].to(self.device))
                eval_loss += float(self.criterion(logits, batch["label"].to(self.device)))
                total_batches += 1
        
        avg_eval_loss = eval_loss / total_batches
        self.model.train()  # Switch back to training mode
        
        return avg_eval_loss

    def train(
        self, 
        num_epochs: int = 5,
        batch_size: int = 64,
        save_best_model: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Main training loop exactly like notebook.
        
        Args:
            num_epochs: Number of epochs to train
            batch_size: Batch size for training
            save_best_model: Whether to load best model state at end
            
        Returns:
            Dictionary with training and evaluation loss tracking
        """
        
        print(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(1, num_epochs + 1):
            
            # Train epoch (may trigger early stopping)
            should_continue = self.train_epoch(epoch, num_epochs, batch_size)
            
            if not should_continue:
                print(f"Training stopped early at epoch {epoch}")
                break
        
        # Load best model state exactly like notebook
        if save_best_model and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model with eval loss: {self.best_eval_loss:.4f}")
        
        return {
            'training_loss': self.tracking_loss,
            'evaluation_loss': self.tracking_loss_ev
        }

    def get_loss_history(self) -> Dict[str, pd.DataFrame]:
        """Get loss tracking history."""
        return {
            'training_loss': self.tracking_loss.copy(),
            'evaluation_loss': self.tracking_loss_ev.copy()
        }

    def save_model(self, filepath: Union[str, Path], save_optimizer: bool = True):
        """Save model checkpoint."""
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'model_name': self.model.model_name,
                'num_classes': self.model.num_classes,
                'pretrained': self.model.pretrained
            },
            'training_loss': self.tracking_loss,
            'evaluation_loss': self.tracking_loss_ev,
            'best_eval_loss': self.best_eval_loss
        }
        
        if save_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: Union[str, Path], load_optimizer: bool = True):
        """Load model checkpoint."""
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.tracking_loss = checkpoint.get('training_loss', pd.DataFrame())
        self.tracking_loss_ev = checkpoint.get('evaluation_loss', pd.DataFrame())
        self.best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))
        
        logger.info(f"Model loaded from {filepath}")


def create_notebook_trainer(
    model: WildlifeClassifier,
    dataset_manager: TaiParkDatasetNotebookStyle,
    **trainer_kwargs
) -> NotebookStyleTrainer:
    """
    Create trainer with exact notebook configuration.
    
    Args:
        model: WildlifeClassifier model
        dataset_manager: Dataset manager instance
        **trainer_kwargs: Additional trainer arguments
        
    Returns:
        Configured NotebookStyleTrainer
    """
    
    return NotebookStyleTrainer(
        model=model,
        dataset_manager=dataset_manager,
        **trainer_kwargs
    )


# Example usage
if __name__ == "__main__":
    from ..models.model import create_notebook_model
    from ..data.dataset import TaiParkDatasetNotebookStyle
    
    # Create model exactly like notebook
    model = create_notebook_model()
    
    # Create dataset manager
    dataset_manager = TaiParkDatasetNotebookStyle(
        data_dir="data/raw",
        fraction=1.0,
        random_state=1
    )
    
    # Create trainer
    trainer = create_notebook_trainer(model, dataset_manager)
    
    # Train exactly like notebook
    loss_history = trainer.train(
        num_epochs=5,
        batch_size=64
    )
    
    print("Training completed!")
    print(f"Final evaluation loss: {trainer.best_eval_loss:.4f}")
    
    # Display loss tracking like notebook
    pd.set_option('display.max_rows', None)
    print("Evaluation Loss History:")
    print(loss_history['evaluation_loss'])