import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import time
import copy

from .config import DEVICE, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, NUM_EPOCHS, EARLY_STOPPING_PATIENCE
from .utils import EarlyStopping, save_checkpoint
from .tracker import ExperimentTracker

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    # Progress bar for the epoch
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch in pbar:
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass with Mixed Precision
        with torch.amp.autocast('cuda',):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        # Backward pass and optimize with Scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Statistics
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        
        # Accuracy calculation
        # Labels are one-hot encoded floats, verify with argmax
        _, preds = torch.max(outputs, 1)
        _, targets = torch.max(labels, 1)
        
        running_corrects += torch.sum(preds == targets).item()
        total_samples += batch_size
        
        # Update progress bar
        current_loss = running_loss / total_samples
        current_acc = running_corrects / total_samples
        pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.4f}'})
        
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for batch in pbar:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            
            _, preds = torch.max(outputs, 1)
            _, targets = torch.max(labels, 1)
            
            running_corrects += torch.sum(preds == targets).item()
            total_samples += batch_size
            
            current_loss = running_loss / total_samples
            current_acc = running_corrects / total_samples
            pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.4f}'})
            
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc

def train_model(model, train_loader, val_loader):
    model = model.to(DEVICE)
    
    # Loss function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    
    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.72, patience=2)
    
    # Early Stopping
    early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    
    # Best model tracking
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    best_acc = 0.0
    
    print(f"Starting training on {DEVICE}...")
    
    # Mixed Precision Scaler
    # Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda',)

    # Initialize Experiment Tracker
    tracker = ExperimentTracker(experiment_name="resnet152_custom")
    
    # Log Configuration
    tracker.log_config({
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": 64, # Need closer coupling with config.BATCH_SIZE if imported
            "num_epochs": NUM_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE
        },
        "model": {
            "name": "ResNet152",
            "num_classes": 8,
            "frozen_layers": "all except layer4 and fc"
        },
        "hardware": {
            "device": str(DEVICE),
            "mixed_precision": True
        }
    })

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 10)
        
        start_time = time.time()
        
        # Train and Validate
        # We need to pass scaler to train_one_epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        # Scheduler Step
        scheduler.step(val_loss)
        
        elapsed = time.time() - start_time
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} [{elapsed:.0f}s]")
        
        # Log metrics to CSV
        current_lr = optimizer.param_groups[0]['lr']
        tracker.log_metric(epoch+1, train_loss, train_acc, val_loss, val_acc, current_lr)
        
        # Checkpointing
        # Save if validation loss improves 
        # (Alternatively could track accuracy, but loss is standard and safer for stability)
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'val_acc': val_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
        
        # Early Stopping
        early_stopper(val_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered")
            break
            
    print(f"Training complete. Best Validation Loss: {best_loss:.4f}, Best Validation Acc: {best_acc:.4f}")
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model
