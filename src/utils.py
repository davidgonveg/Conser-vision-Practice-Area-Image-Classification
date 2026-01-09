import torch
import shutil
from pathlib import Path
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # Import config here to avoid circular import if config imports utils (unlikely but safe)
    # Or assuming config handles paths. Ideally pass path.
    # But since we want to enforce organization, let's just use the path from config if available or just save to proper location.
    from .config import MODELS_DIR
    filepath = MODELS_DIR / filename
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, MODELS_DIR / 'model_best.pth.tar')

def load_checkpoint(filename, model, optimizer=None):
    if Path(filename).exists():
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded checkpoint '{filename}' (epoch {checkpoint.get('epoch', 0)})")
        return checkpoint
    else:
        print(f"No checkpoint found at '{filename}'")
        return None

def calculate_class_weights(labels_df):
    """
    Calculates class weights for imbalanced datasets.
    Args:
        labels_df: DataFrame containing one-hot encoded labels.
    Returns:
        torch.Tensor: Float tensor of weights for each class.
    """
    # Convert one-hot to indices
    # labels_df has columns: [id, class1, class2, ...]
    # we need to find which class is 1 for each row
    y_indices = labels_df.idxmax(axis=1).values
    
    # Get unique classes from columns (excluding non-class cols if any, but passed df usually is just targets)
    classes = labels_df.columns.values
    
    # We map string class names to integer indices 0..7
    # Ideally we should match the order in config.CLASS_NAMES
    # For computation, we just need the array of all labels as indices (0, 0, 1, 7, ...)
    
    # Map class strings to integers based on column order
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    y_integers = np.array([class_to_idx[cls] for cls in y_indices])
    
    # Compute weights: w_j = n_samples / (n_classes * n_samples_j)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_integers),
        y=y_integers
    )
    
    # Check if we missed any classes in this split (unlikely with big data but possible)
    # If a class is missing, its weight won't be in the output of compute_class_weight if not handled carefully
    # But compute_class_weight with 'balanced' and keys provided usually works. 
    # Let's ensure output array size matches number of classes.
    
    if len(class_weights) != len(classes):
        # Fallback or mapping
        full_weights = np.ones(len(classes))
        unique_classes = np.unique(y_integers)
        for i, cls_idx in enumerate(unique_classes):
            full_weights[cls_idx] = class_weights[i]
        class_weights = full_weights

    return torch.tensor(class_weights, dtype=torch.float)
