#!/usr/bin/env python3
"""
Taï National Park Species Classification - Model Evaluation Script

This script evaluates trained models on validation data and generates comprehensive
performance reports including site-based analysis, confusion matrices, and detailed metrics.

Usage Examples:
    # Basic evaluation
    python scripts/evaluate_model.py --model results/models/best_model.pth
    
    # Detailed evaluation with all visualizations
    python scripts/evaluate_model.py --model results/models/best_model.pth \
        --save-plots --detailed-analysis --save-predictions
    
    # Compare multiple models
    python scripts/evaluate_model.py \
        --models results/models/model1.pth results/models/model2.pth \
        --model-names "EfficientNet-B3" "ResNet152" \
        --compare-models
    
    # Evaluation with Test Time Augmentation
    python scripts/evaluate_model.py --model results/models/best_model.pth --use-tta
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, log_loss, roc_auc_score
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Core imports with error handling
try:
    from src.data import DataLoaderManager, create_datasets
    from src.models.model import create_model, load_pretrained_model
    from src.utils.config import Config
    from src.utils.logging_utils import setup_logging
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Optional imports
try:
    from src.evaluation.metrics import MetricsCalculator
    from src.inference.predictor import WildlifePredictor
except ImportError:
    MetricsCalculator = None
    WildlifePredictor = None

warnings.filterwarnings("ignore", category=UserWarning)

# Class names
CLASS_NAMES = [
    'antelope_duiker', 'bird', 'blank', 'civet_genet',
    'hog', 'leopard', 'monkey_prosimian', 'rodent'
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(
        description="Taï National Park Species Classification - Model Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--model', type=str, 
                           help='Path to trained model file')
    model_group.add_argument('--models', nargs='+', 
                           help='Paths to multiple models for comparison')
    model_group.add_argument('--model-names', nargs='+',
                           help='Names for models (for comparison)')
    model_group.add_argument('--config', type=str,
                           help='Path to model configuration file')
    
    # Data arguments
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--data-dir', type=str, default='data/raw',
                           help='Path to data directory')
    data_group.add_argument('--validation-sites', type=str,
                           help='Path to validation sites CSV file')
    data_group.add_argument('--batch-size', type=int, default=64,
                           help='Batch size for evaluation')
    data_group.add_argument('--num-workers', type=int, default=4,
                           help='Number of data loading workers')
    data_group.add_argument('--image-size', type=int, default=224,
                           help='Input image size')
    
    # Evaluation arguments
    eval_group = parser.add_argument_group('Evaluation Options')
    eval_group.add_argument('--use-tta', action='store_true',
                           help='Use Test Time Augmentation')
    eval_group.add_argument('--tta-transforms', type=int, default=5,
                           help='Number of TTA transforms')
    eval_group.add_argument('--detailed-analysis', action='store_true',
                           help='Generate detailed per-class analysis')
    eval_group.add_argument('--compare-models', action='store_true',
                           help='Compare multiple models')
    eval_group.add_argument('--calibration-analysis', action='store_true',
                           help='Analyze model calibration')
    
    # Output arguments
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output-dir', type=str, default='results/evaluation',
                             help='Output directory for results')
    output_group.add_argument('--save-plots', action='store_true',
                             help='Save evaluation plots')
    output_group.add_argument('--save-predictions', action='store_true',
                             help='Save detailed predictions')
    output_group.add_argument('--save-embeddings', action='store_true',
                             help='Save model embeddings for visualization')
    output_group.add_argument('--experiment-name', type=str,
                             help='Experiment name for output files')
    
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
    util_group.add_argument('--quiet', '-q', action='store_true',
                           help='Quiet mode (minimal output)')
    
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
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Determine model architecture
        if 'config' in checkpoint and 'model' in checkpoint['config']:
            model_config = checkpoint['config']['model']
            model_name = model_config.get('name', 'efficientnet_b3')
            num_classes = model_config.get('num_classes', 8)
        else:
            # Try to infer from checkpoint
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
        
        print(f"✅ Model loaded successfully: {model_name}")
        return model
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise


def create_data_loader(args: argparse.Namespace) -> DataLoaderManager:
    """Create data loader for evaluation."""
    
    print("📚 Setting up data loader...")
    
    data_config = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'image_size': args.image_size,
        'validation_sites_file': args.validation_sites,
        'train_sampler_type': 'sequential',  # For evaluation
        'aggressive_augmentation': False,  # No augmentation for eval
        'pin_memory': True
    }
    
    data_manager = DataLoaderManager(**data_config)
    return data_manager


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_tta: bool = False,
    mixed_precision: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on validation data.
    
    Returns:
        Tuple of (predictions, probabilities, true_labels)
    """
    
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    print("🔍 Evaluating model...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
            images = batch['image'].to(device)
            labels = batch['label'] if 'label' in batch else None
            
            # Forward pass
            if mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
            else:
                outputs = model(images)
            
            # Apply softmax to get probabilities
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            # Store results
            all_predictions.append(predictions.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            
            if labels is not None:
                if len(labels.shape) > 1:  # One-hot encoded
                    labels = torch.argmax(labels, dim=1)
                all_labels.append(labels.cpu().numpy())
    
    # Concatenate all results
    predictions = np.concatenate(all_predictions)
    probabilities = np.concatenate(all_probabilities)
    
    if all_labels:
        true_labels = np.concatenate(all_labels)
    else:
        true_labels = None
    
    return predictions, probabilities, true_labels


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str]
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics."""
    
    print("📊 Calculating metrics...")
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names))
    )
    
    # Averaged metrics
    precision_macro = precision_recall_fscore_support(y_true, y_pred, average='macro')[0]
    recall_macro = precision_recall_fscore_support(y_true, y_pred, average='macro')[1]
    f1_macro = precision_recall_fscore_support(y_true, y_pred, average='macro')[2]
    
    precision_weighted = precision_recall_fscore_support(y_true, y_pred, average='weighted')[0]
    recall_weighted = precision_recall_fscore_support(y_true, y_pred, average='weighted')[1]
    f1_weighted = precision_recall_fscore_support(y_true, y_pred, average='weighted')[2]
    
    # Log loss
    try:
        logloss = log_loss(y_true, y_prob)
    except:
        logloss = None
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    class_report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    
    return {
        'accuracy': accuracy,
        'log_loss': logloss,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'per_class': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist()
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    normalize: bool = True
) -> plt.Figure:
    """Create confusion matrix plot."""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
        title = 'Normalized Confusion Matrix'
    else:
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        title = 'Confusion Matrix'
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Set ticks and labels
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Add text annotations
    if normalize:
        thresh = cm_norm.max() / 2.
        for i, j in np.ndindex(cm_norm.shape):
            ax.text(j, i, f'{cm_norm[i, j]:.2f}\n({cm[i, j]})',
                   ha="center", va="center",
                   color="white" if cm_norm[i, j] > thresh else "black",
                   fontsize=8)
    else:
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            ax.text(j, i, cm[i, j],
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_per_class_metrics(
    metrics: Dict[str, Any],
    class_names: List[str],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Create per-class metrics visualization."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    precision = metrics['per_class']['precision']
    recall = metrics['per_class']['recall']
    f1 = metrics['per_class']['f1']
    support = metrics['per_class']['support']
    
    # Precision plot
    axes[0, 0].bar(class_names, precision, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Precision by Class', fontweight='bold')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_ylim(0, 1)
    
    # Recall plot
    axes[0, 1].bar(class_names, recall, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('Recall by Class', fontweight='bold')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].set_ylim(0, 1)
    
    # F1 plot
    axes[1, 0].bar(class_names, f1, color='lightcoral', alpha=0.7)
    axes[1, 0].set_title('F1-Score by Class', fontweight='bold')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].set_ylim(0, 1)
    
    # Support plot
    axes[1, 1].bar(class_names, support, color='gold', alpha=0.7)
    axes[1, 1].set_title('Support by Class', fontweight='bold')
    axes[1, 1].set_ylabel('Number of Samples')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def save_predictions(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    true_labels: Optional[np.ndarray],
    class_names: List[str],
    save_path: Path
) -> None:
    """Save detailed predictions to CSV."""
    
    print(f"💾 Saving predictions to: {save_path}")
    
    # Create DataFrame
    data = {
        'predicted_class_idx': predictions,
        'predicted_class_name': [class_names[idx] for idx in predictions],
    }
    
    # Add probability columns
    for i, class_name in enumerate(class_names):
        data[f'prob_{class_name}'] = probabilities[:, i]
    
    # Add true labels if available
    if true_labels is not None:
        data['true_class_idx'] = true_labels
        data['true_class_name'] = [class_names[idx] for idx in true_labels]
        data['correct'] = (predictions == true_labels)
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)


def print_summary(metrics: Dict[str, Any], class_names: List[str]) -> None:
    """Print evaluation summary."""
    
    print("\n" + "="*60)
    print("🎯 EVALUATION RESULTS SUMMARY")
    print("="*60)
    
    # Overall metrics
    print(f"🎯 Overall Accuracy: {metrics['accuracy']:.4f}")
    if metrics['log_loss'] is not None:
        print(f"📉 Log Loss: {metrics['log_loss']:.4f}")
    
    print(f"\n📊 Macro-averaged Metrics:")
    print(f"   Precision: {metrics['precision_macro']:.4f}")
    print(f"   Recall:    {metrics['recall_macro']:.4f}")
    print(f"   F1-Score:  {metrics['f1_macro']:.4f}")
    
    print(f"\n⚖️  Weighted-averaged Metrics:")
    print(f"   Precision: {metrics['precision_weighted']:.4f}")
    print(f"   Recall:    {metrics['recall_weighted']:.4f}")
    print(f"   F1-Score:  {metrics['f1_weighted']:.4f}")
    
    # Per-class summary
    print(f"\n🦎 Per-class Performance:")
    print("-" * 60)
    print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
    print("-" * 60)
    
    for i, class_name in enumerate(class_names):
        precision = metrics['per_class']['precision'][i]
        recall = metrics['per_class']['recall'][i]
        f1 = metrics['per_class']['f1'][i]
        support = metrics['per_class']['support'][i]
        
        print(f"{class_name:<20} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {support:<10}")
    
    print("-" * 60)


def main():
    """Main evaluation function."""
    
    print("🦁 Taï National Park Species Classification - Model Evaluation")
    print("=" * 70)
    
    # Parse arguments
    args = parse_args()
    
    # Validate arguments
    if not args.model and not args.models:
        print("❌ Error: Please specify either --model or --models")
        return
    
    # Setup device
    device = setup_device(args.device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    if args.experiment_name:
        output_dir = output_dir / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / 'evaluation.log'
    logger = setup_logging(log_file, level='INFO')
    
    try:
        # Create data loader
        data_manager = create_data_loader(args)
        val_loader = data_manager.val_loader
        
        print(f"📊 Validation samples: {len(val_loader.dataset)}")
        
        if args.model:
            # Single model evaluation
            model = load_model(Path(args.model), device)
            
            # Evaluate model
            start_time = time.time()
            predictions, probabilities, true_labels = evaluate_model(
                model, val_loader, device, args.use_tta, args.mixed_precision
            )
            eval_time = time.time() - start_time
            
            print(f"⏱️  Evaluation completed in {eval_time:.2f} seconds")
            
            if true_labels is not None:
                # Calculate metrics
                metrics = calculate_metrics(true_labels, predictions, probabilities, CLASS_NAMES)
                
                # Print summary
                print_summary(metrics, CLASS_NAMES)
                
                # Save metrics
                metrics_path = output_dir / 'metrics.json'
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                print(f"💾 Metrics saved to: {metrics_path}")
                
                # Generate plots
                if args.save_plots:
                    print("📈 Generating plots...")
                    
                    # Confusion matrix
                    cm_path = output_dir / 'confusion_matrix.png'
                    cm_fig = plot_confusion_matrix(
                        np.array(metrics['confusion_matrix']), 
                        CLASS_NAMES, 
                        cm_path
                    )
                    plt.close(cm_fig)
                    
                    # Per-class metrics
                    metrics_path = output_dir / 'per_class_metrics.png'
                    metrics_fig = plot_per_class_metrics(metrics, CLASS_NAMES, metrics_path)
                    plt.close(metrics_fig)
                    
                    print(f"📊 Plots saved to: {output_dir}")
            
            # Save predictions
            if args.save_predictions:
                pred_path = output_dir / 'predictions.csv'
                save_predictions(predictions, probabilities, true_labels, CLASS_NAMES, pred_path)
        
        else:
            # Multiple model comparison
            print("🔄 Comparing multiple models...")
            # TODO: Implement model comparison
            print("⚠️  Model comparison not yet implemented")
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📁 Results saved to: {output_dir}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Evaluation interrupted by user")
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()