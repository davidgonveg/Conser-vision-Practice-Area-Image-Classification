#!/usr/bin/env python3
"""
Taï National Park Species Classification - Submission Generation Script

This script generates submission files for the competition using trained models.
Supports single models, ensembles, and Test Time Augmentation.

Usage:
    python scripts/generate_submission.py --model results/models/best_model.pth --data-dir data/raw
    python scripts/generate_submission.py --ensemble results/models/model1.pth results/models/model2.pth --use-tta
    python scripts/generate_submission.py --model results/models/best_model.pth --output submissions/my_submission.csv
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.predictor import WildlifePredictor
from src.utils.config import Config
from src.utils.logging_utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate Taï Park Species Classification Submission')
    
    # Model settings
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model file')
    parser.add_argument('--ensemble', type=str, nargs='+', default=None,
                       help='Paths to multiple models for ensemble')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to model configuration file')
    parser.add_argument('--ensemble-weights', type=float, nargs='+', default=None,
                       help='Weights for ensemble models (default: equal weights)')
    
    # Data settings
    parser.add_argument('--data-dir', type=str, default='data/raw',
                       help='Path to data directory')
    parser.add_argument('--test-features', type=str, default=None,
                       help='Path to test_features.csv (default: data-dir/test_features.csv)')
    parser.add_argument('--test-images', type=str, default=None,
                       help='Path to test images directory (default: data-dir/test_features)')
    
    # Prediction settings
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for predictions')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--use-tta', action='store_true',
                       help='Use Test Time Augmentation')
    parser.add_argument('--tta-n-augmentations', type=int, default=5,
                       help='Number of TTA augmentations')
    
    # Output settings
    parser.add_argument('--output', type=str, default=None,
                       help='Output submission file path')
    parser.add_argument('--output-dir', type=str, default='data/submissions',
                       help='Output directory for submission files')
    parser.add_argument('--submission-name', type=str, default=None,
                       help='Name for the submission file')
    
    # Validation and safety
    parser.add_argument('--validate-format', action='store_true', default=True,
                       help='Validate submission format')
    parser.add_argument('--save-probabilities', action='store_true',
                       help='Save detailed probabilities for analysis')
    parser.add_argument('--dry-run', action='store_true',
                       help='Perform dry run without generating predictions')
    
    # Logging
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress bars')
    
    return parser.parse_args()


def setup_submission_environment(args: argparse.Namespace) -> Tuple[logging.Logger, Path]:
    """Setup submission environment and logging."""
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / 'submission_generation.log'
    logger = setup_logging(log_file, level=args.log_level)
    
    logger.info("🎯 Starting Submission Generation")
    logger.info(f"📂 Data directory: {args.data_dir}")
    logger.info(f"📂 Output directory: {output_dir}")
    logger.info(f"🔧 Device: {args.device}")
    logger.info(f"🎭 Use TTA: {args.use_tta}")
    logger.info(f"📊 Batch size: {args.batch_size}")
    
    # Log model information
    if args.model:
        logger.info(f"🧠 Single model: {args.model}")
    elif args.ensemble:
        logger.info(f"🎭 Ensemble models: {len(args.ensemble)} models")
        for i, model_path in enumerate(args.ensemble):
            logger.info(f"   Model {i+1}: {model_path}")
        if args.ensemble_weights:
            logger.info(f"   Weights: {args.ensemble_weights}")
    
    return logger, output_dir


def validate_inputs(args: argparse.Namespace, logger: logging.Logger) -> Dict[str, Path]:
    """Validate input paths and return resolved paths."""
    
    logger.info("✅ Validating inputs...")
    
    paths = {}
    
    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    paths['data_dir'] = data_dir
    
    # Validate test features file
    if args.test_features:
        test_features = Path(args.test_features)
    else:
        test_features = data_dir / 'test_features.csv'
    
    if not test_features.exists():
        raise FileNotFoundError(f"Test features file not found: {test_features}")
    paths['test_features'] = test_features
    
    # Validate test images directory
    if args.test_images:
        test_images = Path(args.test_images)
    else:
        test_images = data_dir / 'test_features'
    
    if not test_images.exists():
        raise FileNotFoundError(f"Test images directory not found: {test_images}")
    paths['test_images'] = test_images
    
    # Validate model files
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        paths['models'] = [model_path]
    elif args.ensemble:
        model_paths = []
        for model_path in args.ensemble:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            model_paths.append(model_path)
        paths['models'] = model_paths
    else:
        raise ValueError("Either --model or --ensemble must be specified")
    
    # Validate ensemble weights
    if args.ensemble and args.ensemble_weights:
        if len(args.ensemble_weights) != len(args.ensemble):
            raise ValueError("Number of ensemble weights must match number of models")
        
        # Normalize weights
        total_weight = sum(args.ensemble_weights)
        args.ensemble_weights = [w / total_weight for w in args.ensemble_weights]
    
    logger.info("✅ Input validation passed")
    return paths


def load_predictors(
    model_paths: List[Path],
    args: argparse.Namespace,
    logger: logging.Logger
) -> List[WildlifePredictor]:
    """Load predictor models."""
    
    logger.info(f"🔮 Loading {len(model_paths)} predictor(s)...")
    
    predictors = []
    
    for i, model_path in enumerate(model_paths):
        try:
            predictor = WildlifePredictor(
                model_path=model_path,
                config_path=args.config,
                device=args.device,
                batch_size=args.batch_size,
                use_tta=args.use_tta,
                tta_n_augmentations=args.tta_n_augmentations
            )
            predictors.append(predictor)
            logger.info(f"✅ Model {i+1} loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model {i+1}: {e}")
            raise
    
    return predictors


def generate_ensemble_predictions(
    predictors: List[WildlifePredictor],
    test_features_path: Path,
    test_images_dir: Path,
    ensemble_weights: Optional[List[float]],
    logger: logging.Logger,
    show_progress: bool = True
) -> pd.DataFrame:
    """Generate ensemble predictions."""
    
    logger.info(f"🎭 Generating ensemble predictions with {len(predictors)} models...")
    
    # Load test features
    test_df = pd.read_csv(test_features_path)
    logger.info(f"📊 Test samples: {len(test_df)}")
    
    # Get predictions from each model
    all_predictions = []
    
    for i, predictor in enumerate(predictors):
        logger.info(f"🔮 Generating predictions with model {i+1}/{len(predictors)}...")
        
        # Generate predictions
        predictions_df = predictor.predict_from_dataframe(
            df=test_df,
            image_dir=test_images_dir,
            show_progress=show_progress
        )
        
        all_predictions.append(predictions_df)
        logger.info(f"✅ Model {i+1} predictions complete")
    
    # Ensemble predictions
    logger.info("🎯 Combining ensemble predictions...")
    
    # Initialize ensemble DataFrame
    ensemble_df = all_predictions[0][['id']].copy()
    
    # Get class names
    class_names = [
        'antelope_duiker', 'bird', 'blank', 'civet_genet',
        'hog', 'leopard', 'monkey_prosimian', 'rodent'
    ]
    
    # Combine predictions
    for class_name in class_names:
        # Extract probabilities for this class from all models
        class_probs = np.array([df[class_name].values for df in all_predictions])
        
        # Apply ensemble weights
        if ensemble_weights:
            weighted_probs = np.average(class_probs, axis=0, weights=ensemble_weights)
        else:
            weighted_probs = np.mean(class_probs, axis=0)
        
        ensemble_df[class_name] = weighted_probs
    
    # Normalize probabilities (ensure they sum to 1)
    prob_columns = class_names
    prob_sums = ensemble_df[prob_columns].sum(axis=1)
    ensemble_df[prob_columns] = ensemble_df[prob_columns].div(prob_sums, axis=0)
    
    logger.info("✅ Ensemble predictions generated")
    return ensemble_df


def generate_single_predictions(
    predictor: WildlifePredictor,
    test_features_path: Path,
    test_images_dir: Path,
    logger: logging.Logger,
    show_progress: bool = True
) -> pd.DataFrame:
    """Generate predictions with a single model."""
    
    logger.info("🔮 Generating single model predictions...")
    
    # Generate predictions
    predictions_df = predictor.generate_submission(
        test_features_path=test_features_path,
        test_images_dir=test_images_dir,
        output_path=None,  # We'll handle output ourselves
        format_check=False  # We'll validate separately
    )
    
    logger.info("✅ Single model predictions generated")
    return predictions_df


def validate_submission_format(
    submission_df: pd.DataFrame,
    logger: logging.Logger
) -> bool:
    """Validate submission format according to competition requirements."""
    
    logger.info("🔍 Validating submission format...")
    
    # Required columns
    required_columns = [
        'id', 'antelope_duiker', 'bird', 'blank', 'civet_genet',
        'hog', 'leopard', 'monkey_prosimian', 'rodent'
    ]
    
    # Check columns
    missing_columns = set(required_columns) - set(submission_df.columns)
    if missing_columns:
        logger.error(f"❌ Missing required columns: {missing_columns}")
        return False
    
    # Check data types
    prob_columns = required_columns[1:]  # Exclude 'id'
    
    for col in prob_columns:
        if not pd.api.types.is_numeric_dtype(submission_df[col]):
            logger.error(f"❌ Column '{col}' must be numeric")
            return False
    
    # Check probability ranges
    for col in prob_columns:
        if (submission_df[col] < 0).any() or (submission_df[col] > 1).any():
            logger.error(f"❌ Column '{col}' contains values outside [0, 1] range")
            return False
    
    # Check probability sums
    prob_sums = submission_df[prob_columns].sum(axis=1)
    invalid_sums = abs(prob_sums - 1.0) > 1e-6
    
    if invalid_sums.any():
        n_invalid = invalid_sums.sum()
        logger.warning(f"⚠️ {n_invalid} rows have probabilities that don't sum to 1.0")
        
        # Fix by normalization
        submission_df.loc[:, prob_columns] = submission_df[prob_columns].div(prob_sums, axis=0)
        logger.info("✅ Probabilities normalized")
    
    # Check for NaN values
    if submission_df[prob_columns].isnull().any().any():
        logger.error("❌ Submission contains NaN values")
        return False
    
    # Check expected number of rows
    expected_rows = 4464  # From the competition description
    if len(submission_df) != expected_rows:
        logger.warning(f"⚠️ Expected {expected_rows} rows, got {len(submission_df)}")
    
    logger.info("✅ Submission format validation passed")
    return True


def generate_submission_filename(
    args: argparse.Namespace,
    output_dir: Path
) -> Path:
    """Generate submission filename."""
    
    if args.output:
        return Path(args.output)
    
    # Generate automatic filename
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    if args.submission_name:
        filename = f"{args.submission_name}_{timestamp}.csv"
    elif args.ensemble:
        filename = f"ensemble_{len(args.ensemble)}models_{timestamp}.csv"
    else:
        model_name = Path(args.model).stem
        filename = f"{model_name}_{timestamp}.csv"
    
    if args.use_tta:
        filename = filename.replace('.csv', '_tta.csv')
    
    return output_dir / filename


def save_submission(
    submission_df: pd.DataFrame,
    output_path: Path,
    logger: logging.Logger
) -> None:
    """Save submission to file."""
    
    logger.info(f"💾 Saving submission to: {output_path}")
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    
    # Log file info
    file_size = output_path.stat().st_size / 1024  # KB
    logger.info(f"✅ Submission saved successfully")
    logger.info(f"   File size: {file_size:.1f} KB")
    logger.info(f"   Rows: {len(submission_df):,}")
    logger.info(f"   Columns: {len(submission_df.columns)}")


def save_detailed_probabilities(
    submission_df: pd.DataFrame,
    output_path: Path,
    logger: logging.Logger
) -> None:
    """Save detailed probabilities for analysis."""
    
    detailed_path = output_path.parent / f"{output_path.stem}_detailed{output_path.suffix}"
    
    # Add additional analysis columns
    class_names = [
        'antelope_duiker', 'bird', 'blank', 'civet_genet',
        'hog', 'leopard', 'monkey_prosimian', 'rodent'
    ]
    
    detailed_df = submission_df.copy()
    
    # Add predicted class and confidence
    prob_values = detailed_df[class_names].values
    detailed_df['predicted_class_idx'] = np.argmax(prob_values, axis=1)
    detailed_df['predicted_class'] = [class_names[i] for i in detailed_df['predicted_class_idx']]
    detailed_df['confidence'] = np.max(prob_values, axis=1)
    
    # Add entropy (measure of uncertainty)
    detailed_df['entropy'] = -np.sum(prob_values * np.log(prob_values + 1e-15), axis=1)
    
    # Save detailed file
    detailed_df.to_csv(detailed_path, index=False)
    logger.info(f"📊 Detailed probabilities saved to: {detailed_path}")


def print_submission_summary(
    submission_df: pd.DataFrame,
    args: argparse.Namespace,
    logger: logging.Logger
) -> None:
    """Print submission summary."""
    
    logger.info("\n" + "="*60)
    logger.info("SUBMISSION SUMMARY")
    logger.info("="*60)
    
    # Basic info
    logger.info(f"📊 Total predictions: {len(submission_df):,}")
    logger.info(f"📁 Output format: CSV")
    
    # Model info
    if args.ensemble:
        logger.info(f"🎭 Ensemble: {len(args.ensemble)} models")
        if args.ensemble_weights:
            logger.info(f"   Weights: {args.ensemble_weights}")
    else:
        logger.info(f"🧠 Single model: {Path(args.model).name}")
    
    # Prediction settings
    logger.info(f"🎯 TTA enabled: {args.use_tta}")
    if args.use_tta:
        logger.info(f"   TTA augmentations: {args.tta_n_augmentations}")
    
    # Prediction statistics
    class_names = [
        'antelope_duiker', 'bird', 'blank', 'civet_genet',
        'hog', 'leopard', 'monkey_prosimian', 'rodent'
    ]
    
    # Most confident predictions
    prob_values = submission_df[class_names].values
    predicted_classes = np.argmax(prob_values, axis=1)
    confidences = np.max(prob_values, axis=1)
    
    logger.info(f"\n📈 Prediction Statistics:")
    logger.info(f"   Mean confidence: {np.mean(confidences):.3f}")
    logger.info(f"   Min confidence: {np.min(confidences):.3f}")
    logger.info(f"   Max confidence: {np.max(confidences):.3f}")
    
    # Class predictions distribution
    logger.info(f"\n🏷️  Predicted Class Distribution:")
    for i, class_name in enumerate(class_names):
        count = np.sum(predicted_classes == i)
        percentage = (count / len(submission_df)) * 100
        logger.info(f"   {class_name}: {count:,} ({percentage:.1f}%)")
    
    logger.info("\n" + "="*60)


def main():
    """Main submission generation function."""
    
    # Parse arguments
    args = parse_args()
    
    # Setup environment
    logger, output_dir = setup_submission_environment(args)
    
    try:
        # Validate inputs
        paths = validate_inputs(args, logger)
        
        # Dry run check
        if args.dry_run:
            logger.info("🔍 Dry run mode - validation complete, exiting")
            return
        
        # Load predictors
        predictors = load_predictors(paths['models'], args, logger)
        
        # Generate predictions
        if len(predictors) > 1:
            # Ensemble prediction
            submission_df = generate_ensemble_predictions(
                predictors=predictors,
                test_features_path=paths['test_features'],
                test_images_dir=paths['test_images'],
                ensemble_weights=args.ensemble_weights,
                logger=logger,
                show_progress=not args.quiet
            )
        else:
            # Single model prediction
            submission_df = generate_single_predictions(
                predictor=predictors[0],
                test_features_path=paths['test_features'],
                test_images_dir=paths['test_images'],
                logger=logger,
                show_progress=not args.quiet
            )
        
        # Validate submission format
        if args.validate_format:
            if not validate_submission_format(submission_df, logger):
                raise ValueError("Submission format validation failed")
        
        # Generate output filename
        output_path = generate_submission_filename(args, output_dir)
        
        # Save submission
        save_submission(submission_df, output_path, logger)
        
        # Save detailed probabilities if requested
        if args.save_probabilities:
            save_detailed_probabilities(submission_df, output_path, logger)
        
        # Print summary
        print_submission_summary(submission_df, args, logger)
        
        logger.info("🎉 Submission generation completed successfully!")
        logger.info(f"📁 Final submission file: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Submission generation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()