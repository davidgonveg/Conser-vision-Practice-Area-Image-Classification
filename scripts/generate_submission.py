#!/usr/bin/env python3
"""
Taï National Park Species Classification - Submission Generation Script

This script generates competition submissions from trained models with support for
Test Time Augmentation (TTA), model ensembling, and automatic validation.

Usage Examples:
    # Single model submission
    python scripts/generate_submission.py --model results/models/best_model.pth
    
    # TTA submission for better accuracy
    python scripts/generate_submission.py --model results/models/best_model.pth --use-tta
    
    # Ensemble multiple models
    python scripts/generate_submission.py \
        --ensemble results/models/model1.pth results/models/model2.pth results/models/model3.pth \
        --ensemble-weights 0.4 0.4 0.2
    
    # Complete competition submission with TTA and ensemble
    python scripts/generate_submission.py \
        --ensemble results/models/efficientnet_b4.pth results/models/resnet152.pth \
        --use-tta --tta-transforms 8 \
        --output submissions/final_submission.csv
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Core imports
try:
    from src.data import DataLoaderManager
    from src.models.model import create_model
    from src.utils.logging_utils import setup_logging
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Optional imports
try:
    from src.inference.predictor import WildlifePredictor
except ImportError:
    WildlifePredictor = None

# Class names for Taï Park competition
CLASS_NAMES = [
    'antelope_duiker', 'bird', 'blank', 'civet_genet',
    'hog', 'leopard', 'monkey_prosimian', 'rodent'
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(
        description="Generate competition submission for Taï Park Species Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--model', type=str,
                           help='Path to single trained model')
    model_group.add_argument('--ensemble', nargs='+',
                           help='Paths to multiple models for ensemble')
    model_group.add_argument('--ensemble-weights', nargs='+', type=float,
                           help='Weights for ensemble models (must sum to 1.0)')
    model_group.add_argument('--ensemble-method', type=str, default='weighted',
                           choices=['simple', 'weighted', 'rank'],
                           help='Ensemble method to use')
    
    # Data arguments
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--data-dir', type=str, default='data/raw',
                           help='Path to data directory')
    data_group.add_argument('--test-metadata', type=str,
                           help='Path to test metadata CSV file')
    data_group.add_argument('--batch-size', type=int, default=64,
                           help='Batch size for inference')
    data_group.add_argument('--num-workers', type=int, default=4,
                           help='Number of data loading workers')
    data_group.add_argument('--image-size', type=int, default=224,
                           help='Input image size')
    
    # Test Time Augmentation
    tta_group = parser.add_argument_group('Test Time Augmentation')
    tta_group.add_argument('--use-tta', action='store_true',
                          help='Use Test Time Augmentation')
    tta_group.add_argument('--tta-transforms', type=int, default=5,
                          help='Number of TTA transforms')
    tta_group.add_argument('--tta-horizontal-flip', action='store_true',
                          help='Include horizontal flip in TTA')
    tta_group.add_argument('--tta-rotations', nargs='+', type=int,
                          default=[0, 90, 180, 270],
                          help='Rotations for TTA')
    tta_group.add_argument('--tta-scales', nargs='+', type=float,
                          default=[1.0, 1.1],
                          help='Scales for TTA')
    
    # Output arguments
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument('--output', type=str,
                             default='data/submissions/submission.csv',
                             help='Output submission file path')
    output_group.add_argument('--save-probabilities', action='store_true',
                             help='Save detailed probabilities')
    output_group.add_argument('--submission-format', type=str,
                             help='Path to submission format file for validation')
    
    # Hardware arguments
    hw_group = parser.add_argument_group('Hardware Configuration')
    hw_group.add_argument('--device', type=str, default='auto',
                         help='Device to use (cuda, cpu, auto)')
    hw_group.add_argument('--mixed-precision', action='store_true',
                         help='Use mixed precision for faster inference')
    
    # Utility arguments
    util_group = parser.add_argument_group('Utility Options')
    util_group.add_argument('--verbose', '-v', action='store_true',
                           help='Verbose output')
    util_group.add_argument('--validate-format', action='store_true',
                           help='Validate submission format')
    util_group.add_argument('--dry-run', action='store_true',
                           help='Dry run without saving submission')
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup compute device."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            print("💻 Using CPU")
    else:
        device = torch.device(device_arg)
    
    return device


def load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    
    try:
        print(f"📂 Loading model from: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=device)
        
        # Determine model architecture
        if 'config' in checkpoint and 'model' in checkpoint['config']:
            model_config = checkpoint['config']['model']
            model_name = model_config.get('name', 'efficientnet_b3')
            num_classes = model_config.get('num_classes', 8)
        else:
            model_name = checkpoint.get('model_name', 'efficientnet_b3')
            num_classes = 8
        
        # Create model
        model = create_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False
        )
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded: {model_name}")
        return model
        
    except Exception as e:
        print(f"❌ Failed to load model from {model_path}: {e}")
        raise


def load_ensemble_models(model_paths: List[Path], device: torch.device) -> List[torch.nn.Module]:
    """Load multiple models for ensemble."""
    
    print(f"🔄 Loading {len(model_paths)} models for ensemble...")
    models = []
    
    for path in model_paths:
        model = load_model(path, device)
        models.append(model)
    
    print(f"✅ Loaded {len(models)} models for ensemble")
    return models


def create_test_dataloader(args: argparse.Namespace) -> torch.utils.data.DataLoader:
    """Create test data loader."""
    
    print("📚 Setting up test data loader...")
    
    data_config = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'image_size': args.image_size,
        'pin_memory': True
    }
    
    data_manager = DataLoaderManager(**data_config)
    
    # Get test loader
    if hasattr(data_manager, 'test_loader'):
        return data_manager.test_loader
    else:
        # Create test dataset manually if not available
        from src.data import create_test_dataset, get_val_transforms
        
        test_transform = get_val_transforms(args.image_size)
        test_dataset = create_test_dataset(args.data_dir, transform=test_transform)
        
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )


def apply_tta_transforms(image: torch.Tensor, args: argparse.Namespace) -> List[torch.Tensor]:
    """Apply Test Time Augmentation transforms."""
    
    transforms = [image]  # Original image
    
    # Horizontal flip
    if args.tta_horizontal_flip:
        transforms.append(torch.flip(image, dims=[3]))
    
    # Rotations
    for rotation in args.tta_rotations:
        if rotation == 0:
            continue  # Skip 0 degrees (already have original)
        elif rotation == 90:
            transforms.append(torch.rot90(image, k=1, dims=[2, 3]))
        elif rotation == 180:
            transforms.append(torch.rot90(image, k=2, dims=[2, 3]))
        elif rotation == 270:
            transforms.append(torch.rot90(image, k=3, dims=[2, 3]))
    
    return transforms


def predict_single_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_tta: bool = False,
    tta_args: Optional[argparse.Namespace] = None,
    mixed_precision: bool = False
) -> Tuple[np.ndarray, List[str]]:
    """Generate predictions from single model."""
    
    model.eval()
    all_probabilities = []
    all_ids = []
    
    print("🔮 Generating predictions...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Predicting")):
            images = batch['image'].to(device)
            batch_ids = batch.get('id', [f"test_{batch_idx}_{i}" for i in range(len(images))])
            
            batch_probs = []
            
            if use_tta and tta_args:
                # Apply TTA transforms
                for img_idx in range(images.size(0)):
                    img = images[img_idx].unsqueeze(0)
                    tta_transforms = apply_tta_transforms(img, tta_args)
                    
                    img_probs = []
                    for transformed_img in tta_transforms:
                        if mixed_precision:
                            with torch.cuda.amp.autocast():
                                outputs = model(transformed_img)
                        else:
                            outputs = model(transformed_img)
                        
                        probs = F.softmax(outputs, dim=1)
                        img_probs.append(probs.cpu().numpy())
                    
                    # Average TTA predictions
                    avg_probs = np.mean(img_probs, axis=0)
                    batch_probs.append(avg_probs)
                
                batch_probs = np.vstack(batch_probs)
            else:
                # Standard prediction
                if mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                else:
                    outputs = model(images)
                
                probabilities = F.softmax(outputs, dim=1)
                batch_probs = probabilities.cpu().numpy()
            
            all_probabilities.append(batch_probs)
            all_ids.extend(batch_ids)
    
    return np.vstack(all_probabilities), all_ids


def ensemble_predictions(
    predictions_list: List[np.ndarray],
    method: str = 'weighted',
    weights: Optional[List[float]] = None
) -> np.ndarray:
    """Ensemble multiple model predictions."""
    
    print(f"🤝 Ensembling {len(predictions_list)} models using {method} method...")
    
    if method == 'simple':
        # Simple average
        return np.mean(predictions_list, axis=0)
    
    elif method == 'weighted':
        # Weighted average
        if weights is None:
            weights = [1.0 / len(predictions_list)] * len(predictions_list)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        ensemble_pred = np.zeros_like(predictions_list[0])
        for pred, weight in zip(predictions_list, weights):
            ensemble_pred += pred * weight
        
        return ensemble_pred
    
    elif method == 'rank':
        # Rank-based ensemble
        rank_sums = np.zeros_like(predictions_list[0])
        
        for pred in predictions_list:
            ranks = np.argsort(np.argsort(-pred, axis=1), axis=1)  # Higher prob = lower rank
            rank_sums += ranks
        
        # Convert back to probabilities (inverse ranking)
        num_classes = rank_sums.shape[1]
        ensemble_pred = (num_classes - 1 - rank_sums) / (num_classes - 1)
        
        # Normalize to proper probabilities
        ensemble_pred = ensemble_pred / ensemble_pred.sum(axis=1, keepdims=True)
        
        return ensemble_pred
    
    else:
        raise ValueError(f"Unknown ensemble method: {method}")


def create_submission(
    probabilities: np.ndarray,
    ids: List[str],
    class_names: List[str]
) -> pd.DataFrame:
    """Create submission DataFrame."""
    
    print("📝 Creating submission DataFrame...")
    
    submission_data = {'id': ids}
    
    for i, class_name in enumerate(class_names):
        submission_data[class_name] = probabilities[:, i]
    
    return pd.DataFrame(submission_data)


def validate_submission(
    submission_df: pd.DataFrame,
    submission_format_path: Optional[Path] = None
) -> bool:
    """Validate submission format."""
    
    print("✅ Validating submission format...")
    
    # Check required columns
    expected_columns = ['id'] + CLASS_NAMES
    missing_columns = set(expected_columns) - set(submission_df.columns)
    
    if missing_columns:
        print(f"❌ Missing columns: {missing_columns}")
        return False
    
    # Check for NaN values
    if submission_df.isnull().any().any():
        print("❌ Submission contains NaN values")
        return False
    
    # Check probability ranges
    prob_columns = CLASS_NAMES
    prob_data = submission_df[prob_columns]
    
    if (prob_data < 0).any().any() or (prob_data > 1).any().any():
        print("❌ Probabilities out of range [0, 1]")
        return False
    
    # Check if probabilities sum to ~1
    row_sums = prob_data.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        print("❌ Probabilities don't sum to 1")
        print(f"Row sum range: {row_sums.min():.6f} - {row_sums.max():.6f}")
        return False
    
    # Check against format file if provided
    if submission_format_path and submission_format_path.exists():
        format_df = pd.read_csv(submission_format_path)
        
        if len(submission_df) != len(format_df):
            print(f"❌ Wrong number of rows: {len(submission_df)} vs {len(format_df)}")
            return False
        
        if not set(submission_df['id']).issubset(set(format_df['id'])):
            print("❌ Submission IDs don't match format file")
            return False
    
    print("✅ Submission format is valid")
    return True


def save_submission(
    submission_df: pd.DataFrame,
    output_path: Path,
    save_probabilities: bool = False
) -> None:
    """Save submission to file."""
    
    print(f"💾 Saving submission to: {output_path}")
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save main submission
    submission_df.to_csv(output_path, index=False)
    
    # Save detailed probabilities if requested
    if save_probabilities:
        prob_path = output_path.parent / f"{output_path.stem}_probabilities.csv"
        submission_df.to_csv(prob_path, index=False)
        print(f"📊 Detailed probabilities saved to: {prob_path}")
    
    print(f"✅ Submission saved successfully!")


def print_submission_summary(submission_df: pd.DataFrame, class_names: List[str]) -> None:
    """Print submission summary statistics."""
    
    print("\n" + "="*50)
    print("📈 SUBMISSION SUMMARY")
    print("="*50)
    
    print(f"📊 Total predictions: {len(submission_df)}")
    
    # Most confident predictions per class
    print(f"\n🎯 Most confident predictions by class:")
    for class_name in class_names:
        max_prob = submission_df[class_name].max()
        max_idx = submission_df[class_name].idxmax()
        max_id = submission_df.loc[max_idx, 'id']
        print(f"  {class_name:18}: {max_prob:.4f} (ID: {max_id})")
    
    # Average confidence per class
    print(f"\n📊 Average confidence by class:")
    for class_name in class_names:
        avg_prob = submission_df[class_name].mean()
        print(f"  {class_name:18}: {avg_prob:.4f}")
    
    # Prediction distribution
    print(f"\n🏆 Predicted class distribution:")
    predictions = submission_df[class_names].idxmax(axis=1)
    pred_counts = predictions.value_counts()
    
    for class_name in class_names:
        count = pred_counts.get(class_name, 0)
        percentage = (count / len(submission_df)) * 100
        print(f"  {class_name:18}: {count:6} ({percentage:5.1f}%)")


def main():
    """Main submission generation function."""
    
    print("🦁 Taï National Park Species Classification - Submission Generation")
    print("=" * 75)
    
    # Parse arguments
    args = parse_args()
    
    # Validate arguments
    if not args.model and not args.ensemble:
        print("❌ Error: Please specify either --model or --ensemble")
        return
    
    if args.ensemble and args.ensemble_weights:
        if len(args.ensemble_weights) != len(args.ensemble):
            print("❌ Error: Number of ensemble weights must match number of models")
            return
        
        if abs(sum(args.ensemble_weights) - 1.0) > 1e-6:
            print("❌ Error: Ensemble weights must sum to 1.0")
            return
    
    # Setup device
    device = setup_device(args.device)
    
    try:
        # Create test data loader
        test_loader = create_test_dataloader(args)
        print(f"📊 Test samples: {len(test_loader.dataset)}")
        
        start_time = time.time()
        
        if args.model:
            # Single model prediction
            model = load_model(Path(args.model), device)
            
            probabilities, ids = predict_single_model(
                model, test_loader, device, args.use_tta, args, args.mixed_precision
            )
        
        else:
            # Ensemble prediction
            model_paths = [Path(p) for p in args.ensemble]
            models = load_ensemble_models(model_paths, device)
            
            # Get predictions from all models
            all_predictions = []
            ids = None
            
            for i, model in enumerate(models):
                print(f"🔮 Generating predictions from model {i+1}/{len(models)}...")
                pred, batch_ids = predict_single_model(
                    model, test_loader, device, args.use_tta, args, args.mixed_precision
                )
                all_predictions.append(pred)
                
                if ids is None:
                    ids = batch_ids
            
            # Ensemble predictions
            probabilities = ensemble_predictions(
                all_predictions, args.ensemble_method, args.ensemble_weights
            )
        
        inference_time = time.time() - start_time
        print(f"⏱️  Inference completed in {inference_time:.2f} seconds")
        
        # Create submission
        submission_df = create_submission(probabilities, ids, CLASS_NAMES)
        
        # Validate submission
        format_path = Path(args.submission_format) if args.submission_format else None
        is_valid = validate_submission(submission_df, format_path)
        
        if not is_valid:
            print("❌ Submission validation failed!")
            return
        
        # Print summary
        if args.verbose:
            print_submission_summary(submission_df, CLASS_NAMES)
        
        # Save submission
        if not args.dry_run:
            output_path = Path(args.output)
            save_submission(submission_df, output_path, args.save_probabilities)
            
            print(f"\n🎉 Submission generated successfully!")
            print(f"📁 File: {output_path}")
            print(f"📊 Shape: {submission_df.shape}")
        else:
            print("🏃‍♂️ Dry run completed - submission not saved")
        
    except KeyboardInterrupt:
        print("\n⏹️  Submission generation interrupted by user")
    except Exception as e:
        print(f"❌ Submission generation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()