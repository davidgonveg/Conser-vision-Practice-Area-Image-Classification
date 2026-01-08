import argparse
import sys
from pathlib import Path

# Add src to python path to avoid import errors if run from root
sys.path.append(str(Path(__file__).parent / "src"))

from src import config
from src.data_loader import load_data, get_data_splits, get_dataloaders
from src.preprocessing import get_augmentation_transforms
from src.model import get_model
from src.train import train_model
from src.predict import generate_submission
from src.dataset_cache import prepare_cached_dataset

def main():
    parser = argparse.ArgumentParser(description="Image Classification Pipeline")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "predict", "test_data", "dry_run"], help="Mode to run: train, predict, test_data, or dry_run")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of epochs for training")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        print("Loading data...")
        train_features, train_labels = load_data()
        
        print("Splitting data...")
        # Note: In a real scenario, you might want to save/load splits to ensure consistency
        x_train, x_eval, y_train, y_eval = get_data_splits(train_features, train_labels)
        
        print("Preparing cached dataset (offline resizing)...")
        cache_dir = prepare_cached_dataset()
        
        print("Creating dataloaders...")
        augmentation = get_augmentation_transforms()
        train_loader, val_loader = get_dataloaders(x_train, y_train, x_eval, y_eval, augmentation_functions=augmentation, image_dir=cache_dir)
        
        print("Initializing model...")
        model = get_model(config.NUM_CLASSES)
        
        # Override config epochs if passed via CLI
        # (This implies train_model should accept num_epochs or modify config dynamically)
        # For simplicity, modifying the global config variable before calling train_model
        config.NUM_EPOCHS = args.epochs
        
        print(f"Starting training for {config.NUM_EPOCHS} epochs...")
        train_model(model, train_loader, val_loader)
        
    elif args.mode == "predict":
        print("Generating submission...")
        generate_submission()
        
    elif args.mode == "test_data":
        print("Testing data loading...")
        train_features, train_labels = load_data()
        x_train, x_eval, y_train, y_eval = get_data_splits(train_features, train_labels)
        train_loader, _ = get_dataloaders(x_train, y_train, x_eval, y_eval)
        
        batch = next(iter(train_loader))
        print(f"Batch keys: {batch.keys()}")
        print(f"Image shape: {batch['image'].shape}")
        print(f"Label shape: {batch['label'].shape}")
        print("Data loading test passed!")

    elif args.mode == "dry_run":
        print("Running dry run...")
        # Load a small subset of data
        train_features, train_labels = load_data()
        # Mock split
        x_train, _, y_train, _ = get_data_splits(train_features, train_labels)
        # Create loader (tiny batch)
        train_loader, _ = get_dataloaders(x_train, y_train, x_train, y_train, batch_size=2)
        
        # Initialize model
        model = get_model(config.NUM_CLASSES)
        model.to(config.DEVICE)
        
        # Get one batch
        batch = next(iter(train_loader))
        inputs = batch["image"].to(config.DEVICE)
        
        # Forward pass
        print("Performing forward pass...")
        outputs = model(inputs)
        print(f"Output shape: {outputs.shape}")
        
        print("Dry run passed!")

if __name__ == "__main__":
    main()
