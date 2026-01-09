import argparse
import sys
from pathlib import Path

# Add src to python path to avoid import errors if run from root
sys.path.append(str(Path(__file__).parent / "src"))

from src import config
from src.data_loader import load_data, get_data_splits, get_dataloaders, get_kfold_splits
from src.preprocessing import get_augmentation_transforms
from src.model import get_model
from src.train import train_model
from src.predict import generate_submission
from src.dataset_cache import prepare_cached_dataset
from src.utils import calculate_class_weights

def main():
    parser = argparse.ArgumentParser(description="Image Classification Pipeline")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "predict", "test_data", "dry_run"], help="Mode to run: train, predict, test_data, or dry_run")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of epochs for training")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        print("Loading data...")
        train_features, train_labels = load_data()
        
        print(f"Starting K-Fold Cross Validation ({config.N_FOLDS} folds)...")
        
        # Prepare cache once for all folds
        print("Preparing cached dataset (offline resizing)...")
        cache_dir = prepare_cached_dataset()
        
        # K-Fold Loop
        kfold_generator = get_kfold_splits(train_features, train_labels, params_n_folds=config.N_FOLDS)
        
        for fold, x_train, x_eval, y_train, y_eval in kfold_generator:
            print(f"\n{'='*20} FOLD {fold+1}/{config.N_FOLDS} {'='*20}")
            
            # Validation check
            train_sites = train_features.loc[x_train.index, 'site'].unique()
            eval_sites = train_features.loc[x_eval.index, 'site'].unique()
            intersection = set(train_sites) & set(eval_sites)
            if intersection:
                 print(f"WARNING: Site leakage in Fold {fold+1}! {intersection}")
            
            print("Creating dataloaders...")
            augmentation = get_augmentation_transforms()
            train_loader, val_loader = get_dataloaders(x_train, y_train, x_eval, y_eval, augmentation_functions=augmentation, image_dir=cache_dir)
            
            print("Initializing model...")
            # Re-init model for each fold
            model = get_model(config.NUM_CLASSES)
            
            print("Calculating class weights...")
            class_weights = calculate_class_weights(y_train)
            
            print(f"Starting training for Fold {fold+1}...")
            # We need to modify train_model or save_checkpoint to handle fold names?
            # Or simpler: train_model saves "model_best.pth.tar" to utils logic.
            # We should rename it after training to "model_fold_X.pth.tar"
            
            best_model = train_model(model, train_loader, val_loader, class_weights=class_weights)
            
            # Rename best model artifact
            import shutil
            src_path = config.MODELS_DIR / "model_best.pth.tar"
            dst_path = config.MODELS_DIR / f"model_fold_{fold}.pth.tar"
            if src_path.exists():
                shutil.copy(src_path, dst_path)
                print(f"Saved best model for Fold {fold+1} to {dst_path}")
                
        print("\nCross Validation Complete!")
        
    elif args.mode == "predict":
        print("Generating submission...")
        generate_submission()
        
    elif args.mode == "test_data":
        print("Testing data loading...")
        train_features, train_labels = load_data()
        groups = train_features["site"]
        x_train, x_eval, y_train, y_eval = get_data_splits(train_features, train_labels, groups=groups)
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
        groups = train_features["site"]
        x_train, _, y_train, _ = get_data_splits(train_features, train_labels, groups=groups)
        # Test KFold logic
        print("Testing K-Fold split logic...")
        kfold_gen = get_kfold_splits(train_features, train_labels, params_n_folds=2)
        for f, xt, xe, yt, ye in kfold_gen:
            print(f"Fold {f}: Train size {len(xt)}, Eval size {len(xe)}")
            break # Just test one fold
            
        # Create loader (tiny batch)
        train_loader, _ = get_dataloaders(x_train, y_train, x_train, y_train, batch_size=2)
        
        # Initialize model
        model = get_model(config.NUM_CLASSES)
        model.to(config.DEVICE)
        
        # Get one batch
        batch = next(iter(train_loader))
        inputs = batch["image"].to(config.DEVICE)
        
        # Test class weights calculation
        print("Testing class weights calculation...")
        weights = calculate_class_weights(y_train)
        print(f"Class weights shape: {weights.shape}")
        
        # Forward pass
        print("Performing forward pass...")
        outputs = model(inputs)
        print(f"Output shape: {outputs.shape}")
        
        print("Dry run passed!")

if __name__ == "__main__":
    main()
