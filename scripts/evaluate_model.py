#!/usr/bin/env python3
"""
Taï National Park Species Classification - Model Evaluation Script

This script evaluates trained models on validation data and generates comprehensive
performance reports including site-based analysis and detailed metrics.

Usage:
    python scripts/evaluate_model.py --model results/models/best_model.pth --data-dir data/raw
    python scripts/evaluate_model.py --model results/models/best_model.pth --validation-sites data/processed/validation_sites.csv
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
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.predictor import WildlifePredictor
from src.data import create_datasets, DataLoaderManager
from src.evaluation.metrics import MetricsCalculator
from src.utils.config import Config
from src.utils.logging_utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Taï Park Species Classification Model')
    
    # Model and data settings
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to model configuration file')
    parser.add_argument('--data-dir', type=str, default='data/raw',
                       help='Path to data directory')
    parser.add_argument('--validation-sites', type=str, default='data/processed/validation_sites.csv',
                       help='Path to validation sites CSV file')
    
    # Evaluation settings
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--use-tta', action='store_true',
                       help='Use Test Time Augmentation')
    parser.add_argument('--tta-n-augmentations', type=int, default=5,
                       help='Number of TTA augmentations')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default='results/evaluation',
                       help='Output directory for evaluation results')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save detailed predictions to CSV')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save visualization plots')
    parser.add_argument('--detailed-analysis', action='store_true',
                       help='Generate detailed analysis including per-site metrics')
    
    # Logging
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress bars')
    
    return parser.parse_args()


def setup_evaluation_environment(args: argparse.Namespace) -> Tuple[logging.Logger, Path]:
    """Setup evaluation environment and logging."""
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / 'evaluation.log'
    logger = setup_logging(log_file, level=args.log_level)
    
    logger.info("🔍 Starting Model Evaluation")
    logger.info(f"📂 Model: {args.model}")
    logger.info(f"📂 Data directory: {args.data_dir}")
    logger.info(f"📂 Output directory: {output_dir}")
    logger.info(f"🔧 Device: {args.device}")
    logger.info(f"🎭 Use TTA: {args.use_tta}")
    
    return logger, output_dir


def load_validation_data(args: argparse.Namespace, logger: logging.Logger) -> Tuple[torch.utils.data.DataLoader, List[str]]:
    """Load validation dataset."""
    
    logger.info("📚 Loading validation data...")
    
    # Create data manager
    data_manager = DataLoaderManager(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation_sites_file=args.validation_sites,
        pin_memory=True
    )
    
    val_loader = data_manager.val_loader
    class_names = data_manager.val_dataset.CLASS_NAMES
    
    logger.info(f"📊 Validation samples: {len(data_manager.val_dataset)}")
    logger.info(f"📦 Validation batches: {len(val_loader)}")
    logger.info(f"🌍 Validation sites: {len(data_manager.val_dataset.get_site_distribution())}")
    
    return val_loader, class_names


def evaluate_with_dataloader(
    predictor: WildlifePredictor,
    val_loader: torch.utils.data.DataLoader,
    logger: logging.Logger,
    show_progress: bool = True
) -> Dict[str, Any]:
    """Evaluate model using DataLoader."""
    
    logger.info("🎯 Running evaluation...")
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    all_ids = []
    all_sites = []
    all_image_paths = []
    
    # Progress bar
    iterator = tqdm(val_loader, desc="Evaluating", disable=not show_progress)
    
    for batch in iterator:
        # Get batch data
        images = batch['image']
        labels = batch['class_idx']
        ids = batch['id']
        image_paths = batch['image_path']
        
        # Get site info if available
        sites = batch.get('site', ['unknown'] * len(ids))
        
        # Predict on batch
        with torch.no_grad():
            if predictor.use_tta:
                # Handle TTA - predict each image individually
                batch_predictions = []
                batch_probabilities = []
                
                for i in range(len(images)):
                    # Save image temporarily and predict
                    temp_path = image_paths[i]
                    result = predictor.predict_single(temp_path, return_probabilities=True)
                    
                    batch_predictions.append(result['predicted_class_idx'])
                    batch_probabilities.append(result['probabilities'])
                
                predictions = np.array(batch_predictions)
                probabilities = np.array(batch_probabilities)
            else:
                # Standard batch prediction
                outputs = predictor.model(images.to(predictor.device))
                probabilities = F.softmax(outputs, dim=1).cpu().numpy()
                predictions = np.argmax(probabilities, axis=1)
        
        # Store results
        all_predictions.extend(predictions)
        all_probabilities.extend(probabilities)
        all_labels.extend(labels.cpu().numpy())
        all_ids.extend(ids)
        all_sites.extend(sites)
        all_image_paths.extend(image_paths)
        
        # Update progress bar
        if show_progress:
            current_acc = np.mean(np.array(all_predictions) == np.array(all_labels))
            iterator.set_postfix({'Accuracy': f'{current_acc:.3f}'})
    
    logger.info(f"✅ Evaluation completed on {len(all_predictions)} samples")
    
    return {
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'labels': all_labels,
        'ids': all_ids,
        'sites': all_sites,
        'image_paths': all_image_paths
    }


def calculate_metrics(
    results: Dict[str, Any],
    class_names: List[str],
    logger: logging.Logger
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics."""
    
    logger.info("📊 Calculating metrics...")
    
    # Initialize metrics calculator
    metrics_calc = MetricsCalculator(num_classes=len(class_names))
    
    # Calculate all metrics
    metrics = metrics_calc.calculate_all_metrics(
        y_true=results['labels'],
        y_pred=results['predictions'],
        y_proba=results['probabilities']
    )
    
    # Add class names to metrics
    metrics['class_names'] = class_names
    
    # Calculate site-wise metrics if available
    if 'sites' in results and results['sites']:
        site_metrics = calculate_site_metrics(results, class_names, logger)
        metrics['site_metrics'] = site_metrics
    
    # Log key metrics
    logger.info(f"🎯 Overall Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"🎯 Log Loss: {metrics['log_loss']:.4f}")
    logger.info(f"🎯 Macro F1: {metrics['macro_metrics']['macro_f1']:.4f}")
    logger.info(f"🎯 Top-3 Accuracy: {metrics['top_3_accuracy']:.4f}")
    
    return metrics


def calculate_site_metrics(
    results: Dict[str, Any],
    class_names: List[str],
    logger: logging.Logger
) -> Dict[str, Any]:
    """Calculate site-wise performance metrics."""
    
    logger.info("🌍 Calculating site-wise metrics...")
    
    # Group by site
    site_data = {}
    for i, site in enumerate(results['sites']):
        if site not in site_data:
            site_data[site] = {
                'predictions': [],
                'probabilities': [],
                'labels': [],
                'ids': []
            }
        
        site_data[site]['predictions'].append(results['predictions'][i])
        site_data[site]['probabilities'].append(results['probabilities'][i])
        site_data[site]['labels'].append(results['labels'][i])
        site_data[site]['ids'].append(results['ids'][i])
    
    # Calculate metrics for each site
    metrics_calc = MetricsCalculator(num_classes=len(class_names))
    site_metrics = {}
    
    for site, data in site_data.items():
        if len(data['predictions']) > 0:
            site_metrics[site] = metrics_calc.calculate_all_metrics(
                y_true=data['labels'],
                y_pred=data['predictions'],
                y_proba=data['probabilities']
            )
            site_metrics[site]['n_samples'] = len(data['predictions'])
    
    # Calculate aggregated site statistics
    site_accuracies = [metrics['accuracy'] for metrics in site_metrics.values()]
    site_log_losses = [metrics['log_loss'] for metrics in site_metrics.values()]
    
    aggregated_stats = {
        'n_sites': len(site_metrics),
        'accuracy_mean': np.mean(site_accuracies),
        'accuracy_std': np.std(site_accuracies),
        'accuracy_min': np.min(site_accuracies),
        'accuracy_max': np.max(site_accuracies),
        'log_loss_mean': np.mean(site_log_losses),
        'log_loss_std': np.std(site_log_losses),
        'log_loss_min': np.min(site_log_losses),
        'log_loss_max': np.max(site_log_losses),
    }
    
    logger.info(f"🌍 Site metrics calculated for {len(site_metrics)} sites")
    logger.info(f"   Accuracy range: {aggregated_stats['accuracy_min']:.3f} - {aggregated_stats['accuracy_max']:.3f}")
    logger.info(f"   Mean accuracy: {aggregated_stats['accuracy_mean']:.3f} ± {aggregated_stats['accuracy_std']:.3f}")
    
    return {
        'per_site': site_metrics,
        'aggregated': aggregated_stats
    }


def generate_detailed_report(
    metrics: Dict[str, Any],
    results: Dict[str, Any],
    class_names: List[str],
    output_dir: Path,
    logger: logging.Logger
) -> Dict[str, Any]:
    """Generate detailed evaluation report."""
    
    logger.info("📋 Generating detailed report...")
    
    report = {
        'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'summary': {
            'total_samples': len(results['predictions']),
            'overall_accuracy': metrics['accuracy'],
            'log_loss': metrics['log_loss'],
            'macro_f1': metrics['macro_metrics']['macro_f1'],
            'weighted_f1': metrics['weighted_metrics']['weighted_f1'],
            'top_3_accuracy': metrics['top_3_accuracy']
        },
        'class_performance': metrics['class_wise_metrics'],
        'confusion_matrix': metrics['confusion_matrix'],
        'class_distribution': metrics['class_distribution']
    }
    
    # Add site analysis if available
    if 'site_metrics' in metrics:
        report['site_analysis'] = metrics['site_metrics']['aggregated']
        
        # Find best and worst performing sites
        site_perf = metrics['site_metrics']['per_site']
        site_accuracies = [(site, data['accuracy']) for site, data in site_perf.items()]
        site_accuracies.sort(key=lambda x: x[1], reverse=True)
        
        report['best_sites'] = site_accuracies[:5]
        report['worst_sites'] = site_accuracies[-5:]
    
    # Identify problematic classes
    class_f1_scores = [(name, data['f1_score']) for name, data in metrics['class_wise_metrics'].items()]
    class_f1_scores.sort(key=lambda x: x[1])
    
    report['problematic_classes'] = class_f1_scores[:3]
    report['best_performing_classes'] = class_f1_scores[-3:]
    
    # Save report
    report_path = output_dir / 'evaluation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"📋 Detailed report saved to: {report_path}")
    
    return report


def save_predictions(
    results: Dict[str, Any],
    class_names: List[str],
    output_dir: Path,
    logger: logging.Logger
):
    """Save detailed predictions to CSV."""
    
    logger.info("💾 Saving predictions...")
    
    # Create predictions DataFrame
    predictions_data = []
    
    for i in range(len(results['predictions'])):
        row = {
            'id': results['ids'][i],
            'image_path': results['image_paths'][i],
            'true_class': class_names[results['labels'][i]],
            'predicted_class': class_names[results['predictions'][i]],
            'correct': results['predictions'][i] == results['labels'][i],
            'confidence': results['probabilities'][i][results['predictions'][i]],
            'site': results['sites'][i] if 'sites' in results else 'unknown'
        }
        
        # Add class probabilities
        for j, class_name in enumerate(class_names):
            row[f'prob_{class_name}'] = results['probabilities'][i][j]
        
        predictions_data.append(row)
    
    predictions_df = pd.DataFrame(predictions_data)
    
    # Save to CSV
    predictions_path = output_dir / 'predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    
    logger.info(f"💾 Predictions saved to: {predictions_path}")
    
    # Save errors only
    errors_df = predictions_df[~predictions_df['correct']]
    errors_path = output_dir / 'errors.csv'
    errors_df.to_csv(errors_path, index=False)
    
    logger.info(f"💾 Errors saved to: {errors_path} ({len(errors_df)} samples)")


def create_visualization_plots(
    metrics: Dict[str, Any],
    results: Dict[str, Any],
    class_names: List[str],
    output_dir: Path,
    logger: logging.Logger
):
    """Create visualization plots."""
    
    logger.info("📊 Creating visualization plots...")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = np.array(metrics['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Class Performance
    class_metrics = metrics['class_wise_metrics']
    classes = list(class_metrics.keys())
    precision = [class_metrics[cls]['precision'] for cls in classes]
    recall = [class_metrics[cls]['recall'] for cls in classes]
    f1 = [class_metrics[cls]['f1_score'] for cls in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    plt.figure(figsize=(14, 6))
    plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
    plt.bar(x, recall, width, label='Recall', alpha=0.8)
    plt.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Per-Class Performance Metrics')
    plt.xticks(x, classes, rotation=45)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'class_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Class Distribution
    class_dist = metrics['class_distribution']
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_dist.keys(), class_dist.values(), alpha=0.7)
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Validation Set')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Site Performance (if available)
    if 'site_metrics' in metrics:
        site_metrics = metrics['site_metrics']['per_site']
        sites = list(site_metrics.keys())
        site_accuracies = [site_metrics[site]['accuracy'] for site in sites]
        
        plt.figure(figsize=(15, 6))
        bars = plt.bar(range(len(sites)), site_accuracies, alpha=0.7)
        plt.xlabel('Sites')
        plt.ylabel('Accuracy')
        plt.title('Per-Site Performance')
        plt.xticks(range(len(sites)), sites, rotation=90)
        
        # Add horizontal line for mean accuracy
        mean_acc = np.mean(site_accuracies)
        plt.axhline(y=mean_acc, color='red', linestyle='--', 
                   label=f'Mean: {mean_acc:.3f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'site_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info("📊 Visualization plots saved")


def print_summary(
    metrics: Dict[str, Any],
    report: Dict[str, Any],
    logger: logging.Logger
):
    """Print evaluation summary."""
    
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    
    summary = report['summary']
    logger.info(f"📊 Total Samples: {summary['total_samples']:,}")
    logger.info(f"🎯 Overall Accuracy: {summary['overall_accuracy']:.4f}")
    logger.info(f"🎯 Log Loss: {summary['log_loss']:.4f}")
    logger.info(f"🎯 Macro F1: {summary['macro_f1']:.4f}")
    logger.info(f"🎯 Weighted F1: {summary['weighted_f1']:.4f}")
    logger.info(f"🎯 Top-3 Accuracy: {summary['top_3_accuracy']:.4f}")
    
    if 'site_analysis' in report:
        site_analysis = report['site_analysis']
        logger.info(f"\n🌍 Site Analysis:")
        logger.info(f"   Sites evaluated: {site_analysis['n_sites']}")
        logger.info(f"   Accuracy range: {site_analysis['accuracy_min']:.3f} - {site_analysis['accuracy_max']:.3f}")
        logger.info(f"   Mean accuracy: {site_analysis['accuracy_mean']:.3f} ± {site_analysis['accuracy_std']:.3f}")
    
    logger.info(f"\n🏆 Best Performing Classes:")
    for class_name, f1_score in report['best_performing_classes']:
        logger.info(f"   {class_name}: {f1_score:.3f}")
    
    logger.info(f"\n⚠️  Problematic Classes:")
    for class_name, f1_score in report['problematic_classes']:
        logger.info(f"   {class_name}: {f1_score:.3f}")
    
    logger.info("\n" + "="*60)


def main():
    """Main evaluation function."""
    
    # Parse arguments
    args = parse_args()
    
    # Setup environment
    logger, output_dir = setup_evaluation_environment(args)
    
    try:
        # Load predictor
        logger.info("🔮 Loading predictor...")
        predictor = WildlifePredictor(
            model_path=args.model,
            config_path=args.config,
            device=args.device,
            batch_size=args.batch_size,
            use_tta=args.use_tta,
            tta_n_augmentations=args.tta_n_augmentations
        )
        
        # Load validation data
        val_loader, class_names = load_validation_data(args, logger)
        
        # Run evaluation
        results = evaluate_with_dataloader(
            predictor=predictor,
            val_loader=val_loader,
            logger=logger,
            show_progress=not args.quiet
        )
        
        # Calculate metrics
        metrics = calculate_metrics(results, class_names, logger)
        
        # Generate detailed report
        report = generate_detailed_report(
            metrics=metrics,
            results=results,
            class_names=class_names,
            output_dir=output_dir,
            logger=logger
        )
        
        # Save predictions if requested
        if args.save_predictions:
            save_predictions(results, class_names, output_dir, logger)
        
        # Create plots if requested
        if args.save_plots:
            create_visualization_plots(metrics, results, class_names, output_dir, logger)
        
        # Print summary
        print_summary(metrics, report, logger)
        
        logger.info("✅ Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()