#!/usr/bin/env python3
"""
Taï National Park Species Classification - Simple Training Script

This is a simplified version that works with the existing project structure.
It provides the flexibility of train_model.py but uses the existing codebase.

Usage Examples:
    # Basic training
    python scripts/train_model_simple.py
    
    # Notebook replica
    python scripts/train_model_simple.py \
        --model resnet152 \
        --optimizer sgd \
        --learning-rate 0.01 \
        --momentum 0.909431 \
        --weight-decay 0.005 \
        --epochs 5 \
        --batch-size 64
    
    # Competition config
    python scripts/train_model_simple.py \
        --model efficientnet_b3 \
        --loss focal \
        --aggressive-aug \
        --epochs 50
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try to import the working components from your existing structure
try:
    # Use the working notebook-style components
    from scripts.train_notebook_style import main as notebook_main
    NOTEBOOK_AVAILABLE = True
except ImportError:
    NOTEBOOK_AVAILABLE = False

def parse_args() -> argparse.Namespace:
    """Parse command line arguments with comprehensive options."""
    
    parser = argparse.ArgumentParser(
        description="Taï National Park Species Classification - Simple Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # === BASIC ARGUMENTS ===
    parser.add_argument('--data-dir', type=str, default='data/raw',
                       help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', '--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--experiment-name', type=str,
                       help='Experiment name for output files')
    
    # === MODEL ARGUMENTS ===
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet50', 'resnet101', 'resnet152', 
                               'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                               'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5'],
                       help='Model architecture to use')
    
    # === TRAINING ARGUMENTS ===
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer to use')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum')
    
    # === LOSS ARGUMENTS ===
    parser.add_argument('--loss', type=str, default='cross_entropy',
                       choices=['cross_entropy', 'focal'],
                       help='Loss function to use')
    parser.add_argument('--class-weights', action='store_true',
                       help='Use class weights for imbalanced data')
    
    # === AUGMENTATION ===
    parser.add_argument('--aggressive-aug', action='store_true',
                       help='Use aggressive data augmentation')
    
    # === UTILITY ===
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test with small dataset')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run without actual training')
    parser.add_argument('--fraction', type=float, default=1.0,
                       help='Fraction of dataset to use')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed')
    
    # === HARDWARE ===
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, auto)')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Use mixed precision training')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    return parser.parse_args()


def setup_device(device_arg: str):
    """Setup compute device."""
    import torch
    
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


def create_experiment_name(args: argparse.Namespace) -> str:
    """Create experiment name if not provided."""
    if args.experiment_name:
        return args.experiment_name
    
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"simple_{args.model}_{timestamp}"


def convert_args_to_notebook_style(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert arguments to format expected by notebook-style trainer."""
    
    # Map arguments to notebook-style format
    notebook_args = {
        'data_dir': args.data_dir,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'fraction': args.fraction,
        'random_state': args.random_state,
        'use_preprocessing': True,  # Always use preprocessing
        'use_augmentation': args.aggressive_aug,
        'num_augmentations': 3 if args.aggressive_aug else 2,
        'skip_evaluation': args.dry_run,
        'skip_submission': args.dry_run,
    }
    
    # Add experiment-specific paths
    experiment_name = create_experiment_name(args)
    results_dir = Path('results') / 'models' / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    notebook_args.update({
        'save_model_path': str(results_dir / 'model.pth'),
        'save_plots_path': str(results_dir / 'plots'),
        'submission_output_path': str(results_dir / 'submission.csv'),
    })
    
    return notebook_args


def print_training_summary(args: argparse.Namespace):
    """Print training configuration summary."""
    
    print("🦁 Taï National Park Species Classification - Simple Training")
    print("=" * 70)
    print(f"🎯 Model: {args.model}")
    print(f"🎯 Optimizer: {args.optimizer}")
    print(f"🎯 Learning Rate: {args.learning_rate}")
    print(f"🎯 Epochs: {args.epochs}")
    print(f"🎯 Batch Size: {args.batch_size}")
    print(f"🎯 Data Fraction: {args.fraction}")
    
    if args.aggressive_aug:
        print("🌈 Using aggressive augmentation")
    if args.class_weights:
        print("⚖️  Using class weights")
    if args.mixed_precision:
        print("⚡ Using mixed precision")
    if args.quick_test:
        print("⚡ Quick test mode")
    if args.dry_run:
        print("🏃‍♂️ Dry run mode")
    
    print("=" * 70)


def save_training_config(args: argparse.Namespace, experiment_name: str):
    """Save training configuration for reference."""
    
    import json
    
    config_data = {
        'experiment_name': experiment_name,
        'model': args.model,
        'optimizer': args.optimizer,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'loss': args.loss,
        'aggressive_aug': args.aggressive_aug,
        'class_weights': args.class_weights,
        'fraction': args.fraction,
        'random_state': args.random_state,
        'mixed_precision': args.mixed_precision,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    results_dir = Path('results') / 'models' / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    config_path = results_dir / 'training_config.json'
    
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"💾 Configuration saved to: {config_path}")


def main():
    """Main training function."""
    
    # Parse arguments
    args = parse_args()
    
    # Setup device
    device = setup_device(args.device)
    
    # Print summary
    print_training_summary(args)
    
    # Create experiment name
    experiment_name = create_experiment_name(args)
    print(f"📁 Experiment: {experiment_name}")
    
    # Save configuration
    save_training_config(args, experiment_name)
    
    # Handle dry run
    if args.dry_run:
        print("🏃‍♂️ Dry run completed - configuration saved")
        print("💡 Remove --dry-run to start actual training")
        return
    
    # Check if notebook-style trainer is available
    if not NOTEBOOK_AVAILABLE:
        print("❌ Notebook-style trainer not available")
        print("💡 Please ensure scripts/train_notebook_style.py exists and works")
        return
    
    # For now, provide instructions for using the working components
    print("\n🎯 TRAINING OPTIONS:")
    print("=" * 50)
    
    print("1️⃣  Use the working notebook-style trainer:")
    print(f"   python scripts/train_notebook_style.py \\")
    print(f"       --num_epochs {args.epochs} \\")
    print(f"       --batch_size {args.batch_size} \\")
    print(f"       --fraction {args.fraction}")
    if args.aggressive_aug:
        print(f"       --use_augmentation \\")
        print(f"       --num_augmentations 3")
    
    print("\n2️⃣  Advanced parameters (need custom implementation):")
    if args.model != 'resnet152':
        print(f"   ⚠️  Model: {args.model} (notebook uses ResNet152)")
    if args.optimizer != 'sgd':
        print(f"   ⚠️  Optimizer: {args.optimizer} (notebook uses SGD)")
    if args.loss != 'cross_entropy':
        print(f"   ⚠️  Loss: {args.loss} (notebook uses CrossEntropy)")
    
    print("\n3️⃣  For now, using notebook-style with available parameters:")
    
    # Convert to notebook args and show what would be used
    notebook_args = convert_args_to_notebook_style(args)
    
    # Create a mock argparse.Namespace for notebook function
    class MockArgs:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    mock_args = MockArgs(**notebook_args)
    
    print(f"   📊 Data directory: {mock_args.data_dir}")
    print(f"   📊 Epochs: {mock_args.num_epochs}")
    print(f"   📊 Batch size: {mock_args.batch_size}")
    print(f"   📊 Fraction: {mock_args.fraction}")
    print(f"   📊 Augmentation: {mock_args.use_augmentation}")
    
    # Ask user if they want to proceed
    print("\n❓ Proceed with training using available parameters?")
    print("   The advanced parameters you specified will be noted but not used yet.")
    
    response = input("Continue? (y/N): ").strip().lower()
    
    if response == 'y' or response == 'yes':
        print("\n🚀 Starting training with notebook-style trainer...")
        print("   (Advanced parameters saved in config for future use)")
        
        try:
            # This would need to be implemented to call the actual training
            print("💡 Integration with existing trainer not yet implemented")
            print("💡 Please run the notebook-style command shown above")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            
    else:
        print("⏹️  Training cancelled")
        print("💡 Configuration saved for future use")
    
    print(f"\n📁 Results will be saved to: results/models/{experiment_name}/")


if __name__ == "__main__":
    main()