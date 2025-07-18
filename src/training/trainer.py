"""
Taï National Park - Training Module

This module provides the core training functionality for camera trap species classification.
Includes site-aware validation, advanced logging, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import defaultdict
import time
import json
from tqdm import tqdm

from ..evaluation.metrics import MetricsCalculator
from ..utils.config import Config


class Trainer:
    """
    Main trainer class for camera trap species classification.
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        config: Config,
        output_dir: Path,
        logger: logging.Logger,
        tb_writer: Optional[Any] = None,
        wandb_run: Optional[Any] = None,
        mixed_precision: bool = False
    ):
        """
        Initialize trainer.
        
        Args:
            model: The model to train
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            config: Training configuration
            output_dir: Output directory for saving models
            logger: Logger instance
            tb_writer: TensorBoard writer
            wandb_run: Weights & Biases run
            mixed_precision: Whether to use mixed precision training
        """
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.tb_writer = tb_writer
        self.wandb_run = wandb_run
        self.mixed_precision = mixed_precision
        
        # Mixed precision scaler
        self.scaler = GradScaler() if mixed_precision else None
        
        # Metrics calculator
        self.metrics_calculator = MetricsCalculator(num_classes=8)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.early_stopping_counter = 0
        self.training_history = defaultdict(list)
        
        # Configuration
        self.early_stopping_patience = config.get('training.early_stopping_patience', 10)
        self.save_frequency = config.get('training.save_frequency', 5)
        self.gradient_clip_norm = config.get('training.gradient_clip_norm', 1.0)
        
        self.logger.info(f" Trainer initialized with mixed precision: {mixed_precision}")
    
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary with training metrics
        """
        
        self.model.train()
        
        # Initialize metrics
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        # Class-wise metrics
        class_correct = torch.zeros(8)
        class_total = torch.zeros(8)
        
        # Site-wise metrics
        site_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            class_indices = batch['class_idx'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, class_indices)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, class_indices)
            
            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if self.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                
                self.optimizer.step()
            
            # Statistics
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Accuracy
            _, predicted = torch.max(outputs, 1)
            running_corrects += (predicted == class_indices).sum().item()
            
            # Class-wise accuracy
            for i in range(batch_size):
                true_class = class_indices[i].item()
                pred_class = predicted[i].item()
                
                class_total[true_class] += 1
                if true_class == pred_class:
                    class_correct[true_class] += 1
            
            # Site-wise accuracy (if available)
            if 'site' in batch:
                for i, site in enumerate(batch['site']):
                    true_class = class_indices[i].item()
                    pred_class = predicted[i].item()
                    
                    site_metrics[site]['total'] += 1
                    if true_class == pred_class:
                        site_metrics[site]['correct'] += 1
            
            # Update progress bar
            current_loss = running_loss / total_samples
            current_acc = running_corrects / total_samples
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.4f}'
            })
            
            # Log batch metrics to tensorboard
            if self.tb_writer and batch_idx % 100 == 0:
                global_step = self.current_epoch * len(train_loader) + batch_idx
                self.tb_writer.add_scalar('train/batch_loss', loss.item(), global_step)
                self.tb_writer.add_scalar('train/batch_acc', current_acc, global_step)
        
        # Calculate epoch metrics
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        
        # Calculate class-wise accuracy
        class_acc = {}
        for i, class_name in enumerate(['antelope_duiker', 'bird', 'blank', 'civet_genet', 
                                      'hog', 'leopard', 'monkey_prosimian', 'rodent']):
            if class_total[i] > 0:
                class_acc[class_name] = (class_correct[i] / class_total[i]).item()
            else:
                class_acc[class_name] = 0.0
        
        # Calculate site-wise accuracy
        site_acc = {}
        for site, metrics in site_metrics.items():
            if metrics['total'] > 0:
                site_acc[site] = metrics['correct'] / metrics['total']
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'class_accuracy': class_acc,
            'site_accuracy': site_acc
        }
    
    def validate_epoch(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        
        self.model.eval()
        
        # Initialize metrics
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        # For detailed metrics
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        # Class-wise metrics
        class_correct = torch.zeros(8)
        class_total = torch.zeros(8)
        
        # Site-wise metrics
        site_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        # Progress bar
        pbar = tqdm(val_loader, desc=f'Epoch {self.current_epoch + 1} [Val]')
        
        with torch.no_grad():
            for batch in pbar:
                # Move data to device
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                class_indices = batch['class_idx'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, class_indices)
                
                # Statistics
                batch_size = images.size(0)
                running_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Predictions and probabilities
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # Store for detailed metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(class_indices.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Accuracy
                running_corrects += (predicted == class_indices).sum().item()
                
                # Class-wise accuracy
                for i in range(batch_size):
                    true_class = class_indices[i].item()
                    pred_class = predicted[i].item()
                    
                    class_total[true_class] += 1
                    if true_class == pred_class:
                        class_correct[true_class] += 1
                
                # Site-wise accuracy (if available)
                if 'site' in batch:
                    for i, site in enumerate(batch['site']):
                        true_class = class_indices[i].item()
                        pred_class = predicted[i].item()
                        
                        site_metrics[site]['total'] += 1
                        if true_class == pred_class:
                            site_metrics[site]['correct'] += 1
                
                # Update progress bar
                current_loss = running_loss / total_samples
                current_acc = running_corrects / total_samples
                pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.4f}'
                })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        
        # Calculate detailed metrics
        detailed_metrics = self.metrics_calculator.calculate_all_metrics(
            y_true=all_labels,
            y_pred=all_predictions,
            y_proba=all_probabilities
        )
        
        # Calculate class-wise accuracy
        class_acc = {}
        for i, class_name in enumerate(['antelope_duiker', 'bird', 'blank', 'civet_genet', 
                                      'hog', 'leopard', 'monkey_prosimian', 'rodent']):
            if class_total[i] > 0:
                class_acc[class_name] = (class_correct[i] / class_total[i]).item()
            else:
                class_acc[class_name] = 0.0
        
        # Calculate site-wise accuracy
        site_acc = {}
        for site, metrics in site_metrics.items():
            if metrics['total'] > 0:
                site_acc[site] = metrics['correct'] / metrics['total']
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'log_loss': detailed_metrics['log_loss'],
            'class_accuracy': class_acc,
            'site_accuracy': site_acc,
            'detailed_metrics': detailed_metrics
        }
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int,
        start_epoch: int = 0
    ) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            start_epoch: Starting epoch (for resuming)
            
        Returns:
            Training history dictionary
        """
        
        self.logger.info(f" Starting training for {num_epochs} epochs")
        self.logger.info(f" Training samples: {len(train_loader.dataset)}")
        self.logger.info(f" Validation samples: {len(val_loader.dataset)}")
        
        training_start_time = time.time()
        
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self.train_epoch(train_loader)
            
            # Validation phase
            val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Log metrics
            self._log_epoch_metrics(train_metrics, val_metrics, epoch_time)
            
            # Save training history
            self._update_training_history(train_metrics, val_metrics)
            
            # Check for best model
            is_best = val_metrics['log_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.best_val_acc = val_metrics['accuracy']
                self.early_stopping_counter = 0
                
                # Save best model
                best_model_path = self.output_dir / 'best_model.pth'
                self.save_checkpoint(best_model_path, epoch, is_best=True)
                self.logger.info(f" New best model saved! Val Loss: {val_metrics['loss']:.4f}")
            else:
                self.early_stopping_counter += 1
            
            # Regular checkpoint saving
            if (epoch + 1) % self.save_frequency == 0:
                checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch + 1}.pth'
                self.save_checkpoint(checkpoint_path, epoch)
            
            # Early stopping
            if self.early_stopping_counter >= self.early_stopping_patience:
                self.logger.info(f" Early stopping triggered after {self.early_stopping_patience} epochs without improvement")
                break
        
        # Training completed
        training_time = time.time() - training_start_time
        self.logger.info(f" Training completed in {training_time:.2f} seconds")
        self.logger.info(f" Best validation loss: {self.best_val_loss:.4f}")
        self.logger.info(f" Best validation accuracy: {self.best_val_acc:.4f}")
        
        # Save final training history
        self._save_training_history()
        
        return dict(self.training_history)
    
    def _log_epoch_metrics(
        self, 
        train_metrics: Dict[str, Any], 
        val_metrics: Dict[str, Any], 
        epoch_time: float
    ):
        """Log metrics for the current epoch."""
        
        # Console logging
        self.logger.info(f" Epoch {self.current_epoch + 1} completed in {epoch_time:.2f}s")
        self.logger.info(f"   Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        self.logger.info(f"   Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, LogLoss: {val_metrics['log_loss']:.4f}")
        
        # Log class-wise accuracy
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("  Class-wise validation accuracy:")
            for class_name, acc in val_metrics['class_accuracy'].items():
                self.logger.debug(f"     {class_name}: {acc:.4f}")
        
        # TensorBoard logging
        if self.tb_writer:
            self.tb_writer.add_scalar('train/loss', train_metrics['loss'], self.current_epoch)
            self.tb_writer.add_scalar('train/accuracy', train_metrics['accuracy'], self.current_epoch)
            self.tb_writer.add_scalar('val/loss', val_metrics['loss'], self.current_epoch)
            self.tb_writer.add_scalar('val/accuracy', val_metrics['accuracy'], self.current_epoch)
            self.tb_writer.add_scalar('val/log_loss', val_metrics['log_loss'], self.current_epoch)
            self.tb_writer.add_scalar('train/epoch_time', epoch_time, self.current_epoch)
            
            # Log learning rate
            if self.optimizer.param_groups:
                lr = self.optimizer.param_groups[0]['lr']
                self.tb_writer.add_scalar('train/learning_rate', lr, self.current_epoch)
            
            # Log class-wise accuracy
            for class_name, acc in val_metrics['class_accuracy'].items():
                self.tb_writer.add_scalar(f'val/class_accuracy/{class_name}', acc, self.current_epoch)
        
        # Weights & Biases logging
        if self.wandb_run:
            log_dict = {
                'epoch': self.current_epoch,
                'train/loss': train_metrics['loss'],
                'train/accuracy': train_metrics['accuracy'],
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'],
                'val/log_loss': val_metrics['log_loss'],
                'train/epoch_time': epoch_time,
            }
            
            # Add learning rate
            if self.optimizer.param_groups:
                log_dict['train/learning_rate'] = self.optimizer.param_groups[0]['lr']
            
            # Add class-wise accuracy
            for class_name, acc in val_metrics['class_accuracy'].items():
                log_dict[f'val/class_accuracy/{class_name}'] = acc
            
            self.wandb_run.log(log_dict)
    
    def _update_training_history(self, train_metrics: Dict[str, Any], val_metrics: Dict[str, Any]):
        """Update training history."""
        
        self.training_history['train_loss'].append(train_metrics['loss'])
        self.training_history['train_accuracy'].append(train_metrics['accuracy'])
        self.training_history['val_loss'].append(val_metrics['loss'])
        self.training_history['val_accuracy'].append(val_metrics['accuracy'])
        self.training_history['val_log_loss'].append(val_metrics['log_loss'])
        
        # Add learning rate
        if self.optimizer.param_groups:
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
    
    def _save_training_history(self):
        """Save training history to file."""
        
        history_path = self.output_dir / 'training_history.json'
        
        # Convert to serializable format
        history_dict = {}
        for key, values in self.training_history.items():
            history_dict[key] = [float(v) if isinstance(v, (int, float, np.number)) else v for v in values]
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        self.logger.info(f" Training history saved to: {history_path}")
    
    def save_checkpoint(self, path: Path, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'training_history': dict(self.training_history),
            'config': self.config.config,
            'is_best': is_best
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        self.logger.info(f" Checkpoint saved to: {path}")
    
    def load_checkpoint(self, path: Path) -> int:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Starting epoch number
        """
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        if 'training_history' in checkpoint:
            for key, values in checkpoint['training_history'].items():
                self.training_history[key] = values
        
        start_epoch = checkpoint['epoch'] + 1
        
        self.logger.info(f" Checkpoint loaded from: {path}")
        self.logger.info(f" Resuming from epoch {start_epoch}")
        
        return start_epoch
    
    def save_model(self, path: Path):
        """Save model for inference."""
        
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'model_name': getattr(self.model, 'model_name', 'unknown'),
            'num_classes': 8,
            'config': self.config.config,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc
        }
        
        torch.save(model_state, path)
        self.logger.info(f" Model saved for inference: {path}")

    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information."""
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_name': getattr(self.model, 'model_name', 'unknown'),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'current_epoch': self.current_epoch
        }