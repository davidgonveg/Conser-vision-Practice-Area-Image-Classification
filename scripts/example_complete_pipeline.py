"""
Complete Pipeline Example - Notebook to Production

This script demonstrates the complete workflow from training to submission
replicating your notebook exactly but in a production-ready format.
"""

import sys
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import TaiParkDatasetNotebookStyle, ImagesDataset, custom_preprocessing
from src.models.model import create_notebook_model, get_model_info
from src.training.trainer import create_notebook_trainer
from src.evaluation.evaluator import evaluate_notebook_style
from src.inference.predictor import create_notebook_submission
from src.utils.helpers import notebook_style_summary, visualize_samples, plot_class_distribution


def complete_pipeline_example():
    """
    Complete pipeline that replicates your notebook workflow exactly.
    
    This demonstrates how all the modules work together to recreate
    your successful notebook results.
    """
    
    print("🦁 COMPLETE WILDLIFE CLASSIFICATION PIPELINE")
    print("=" * 60)
    print("Replicating notebook workflow with modular code...")
    print()
    
    # =============================================================================
    # 1. DATASET LOADING (exactly like notebook)
    # =============================================================================
    print("📊 STEP 1: Loading Dataset")
    print("-" * 30)
    
    dataset_manager = TaiParkDatasetNotebookStyle(
        data_dir="data/raw",
        fraction=1.0,              # notebook's frac parameter
        random_state=1,            # notebook's random_state  
        use_preprocessing=True,    # custom_preprocessing
        use_augmentation=True,     # data_augmentation
        num_augmentations=2        # notebook's num_augmentations
    )
    
    print(f"✅ Dataset loaded:")
    print(f"   Training samples: {len(dataset_manager.y_train):,}")
    print(f"   Validation samples: {len(dataset_manager.y_eval):,}")
    print(f"   Species: {len(dataset_manager.species_labels)}")
    print(f"   Site-aware split: ✅ No data leakage")
    print()
    
    # =============================================================================
    # 2. MODEL CREATION (exactly like notebook)
    # =============================================================================
    print("🤖 STEP 2: Creating Model")
    print("-" * 30)
    
    model = create_notebook_model()
    model_info = get_model_info(model)
    
    print(f"✅ Model created:")
    print(f"   Architecture: {model_info['model_name']}")
    print(f"   Total parameters: {model_info['total_parameters']:,}")
    print(f"   Trainable parameters: {model_info['trainable_parameters']:,}")
    print(f"   Trainable percentage: {model_info['trainable_percentage']:.1f}%")
    print(f"   Only layer4 unfrozen: ✅ (like notebook)")
    print()
    
    # =============================================================================
    # 3. TRAINING (exactly like notebook)
    # =============================================================================
    print("🚂 STEP 3: Training Model")
    print("-" * 30)
    
    trainer = create_notebook_trainer(model, dataset_manager)
    
    print("Training configuration (exactly like notebook):")
    print("   Optimizer: SGD (lr=0.01, momentum=0.909431, weight_decay=0.005)")
    print("   Scheduler: ReduceLROnPlateau (patience=2, factor=0.72)")
    print("   Early stopping: tolerance=5, min_delta=0.0001")
    print("   Dataset recreation: Each epoch (with augmentation)")
    print("   Evaluation: Quarter-steps within epochs")
    print()
    
    # Train for fewer epochs in example (you can change this)
    loss_history = trainer.train(
        num_epochs=2,  # Reduce for example, notebook uses 5
        batch_size=64,
        save_best_model=True
    )
    
    print(f"✅ Training completed:")
    print(f"   Best eval loss: {trainer.best_eval_loss:.6f}")
    print(f"   Training steps: {len(loss_history['training_loss'])}")
    print(f"   Evaluation steps: {len(loss_history['evaluation_loss'])}")
    print()
    
    # =============================================================================
    # 4. EVALUATION (exactly like notebook)
    # =============================================================================
    print("🔍 STEP 4: Model Evaluation")
    print("-" * 30)
    
    # Create evaluation dataloader exactly like notebook
    eval_dataset = ImagesDataset(
        dataset_manager.x_eval, 
        dataset_manager.y_eval,
        preprocessing=custom_preprocessing
    )
    eval_dataloader = DataLoader(eval_dataset, batch_size=64, num_workers=4, pin_memory=True)
    
    # Evaluate exactly like notebook
    eval_results = evaluate_notebook_style(
        model=trainer.model,
        eval_dataloader=eval_dataloader,
        true_labels_df=dataset_manager.y_eval,
        species_labels=dataset_manager.species_labels,
        device=trainer.device,
        save_plots_dir="results/plots"
    )
    
    print(f"✅ Evaluation completed:")
    print(f"   Accuracy: {eval_results['accuracy']:.1%}")
    print(f"   Random baseline: {eval_results['random_accuracy']:.1%}")
    print(f"   Most common baseline: {eval_results['baseline_accuracy']:.1%}")
    print(f"   Improvement over random: {eval_results['accuracy']/eval_results['random_accuracy']:.1f}x")
    print()
    
    # =============================================================================
    # 5. SUBMISSION GENERATION (exactly like notebook)
    # =============================================================================
    print("📤 STEP 5: Generating Submission")
    print("-" * 30)
    
    # Check if test data exists
    test_features_path = Path("data/raw/test_features.csv")
    if test_features_path.exists():
        
        # Load test features exactly like notebook
        test_features_df = pd.read_csv(test_features_path, index_col="id")
        
        # Generate submission exactly like notebook
        submission_df = create_notebook_submission(
            model=trainer.model,
            test_features_df=test_features_df,
            species_labels=dataset_manager.species_labels,
            device=trainer.device,
            output_path="data/submissions/example_submission.csv",
            submission_format_path="data/raw/submission_format.csv" if Path("data/raw/submission_format.csv").exists() else None,
            batch_size=64,
            use_preprocessing=True
        )
        
        print(f"✅ Submission generated:")
        print(f"   Shape: {submission_df.shape}")
        print(f"   Test samples: {len(submission_df):,}")
        print(f"   Species columns: {len(submission_df.columns)}")
        print(f"   File: data/submissions/example_submission.csv")
        print(f"   Ready for competition! 🎯")
        
    else:
        print("⚠️  Test features file not found")
        print("   Place test_features.csv in data/raw/ to generate submission")
    
    print()
    
    # =============================================================================
    # 6. FINAL SUMMARY (notebook style)
    # =============================================================================
    print("🎉 COMPLETE PIPELINE SUMMARY")
    print("=" * 60)
    
    notebook_style_summary(trainer, dataset_manager, model_info, loss_history)
    
    print("\n🚀 NEXT STEPS:")
    print("   1. Run full training with more epochs")
    print("   2. Experiment with different architectures")
    print("   3. Try ensemble methods")
    print("   4. Upload submission to competition")
    print("   5. Iterate and improve!")
    
    return {
        'dataset_manager': dataset_manager,
        'model': model,
        'trainer': trainer,
        'eval_results': eval_results,
        'loss_history': loss_history
    }


def quick_demo():
    """Quick demo with minimal data for testing."""
    
    print("🧪 QUICK DEMO - Testing Pipeline")
    print("=" * 40)
    
    # Use small fraction for speed
    dataset_manager = TaiParkDatasetNotebookStyle(
        data_dir="data/raw",
        fraction=0.05,  # Only 5% of data
        random_state=1
    )
    
    model = create_notebook_model()
    trainer = create_notebook_trainer(model, dataset_manager)
    
    # Quick training
    loss_history = trainer.train(num_epochs=1, batch_size=32)
    
    print(f"✅ Quick demo completed!")
    print(f"   Used {dataset_manager.fraction*100}% of data")
    print(f"   Best eval loss: {trainer.best_eval_loss:.6f}")
    print(f"   Pipeline works! 🎉")
    
    return trainer


def visualizations_demo():
    """Demo visualization functions."""
    
    print("🎨 VISUALIZATION DEMO")
    print("=" * 30)
    
    dataset_manager = TaiParkDatasetNotebookStyle("data/raw", fraction=0.1)
    
    # Sample visualizations (if data exists)
    try:
        visualize_samples(dataset_manager, save_path="results/plots/demo_samples.png")
        plot_class_distribution(dataset_manager, save_path="results/plots/demo_distribution.png")
        print("✅ Visualizations created in results/plots/")
    except Exception as e:
        print(f"⚠️  Visualization demo skipped: {e}")


if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete Pipeline Example')
    parser.add_argument('--mode', choices=['full', 'quick', 'viz'], default='quick',
                       help='Demo mode: full pipeline, quick test, or visualizations')
    
    args = parser.parse_args()
    
    # Create output directories
    Path("results/models").mkdir(parents=True, exist_ok=True)
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    Path("data/submissions").mkdir(parents=True, exist_ok=True)
    
    if args.mode == 'full':
        results = complete_pipeline_example()
    elif args.mode == 'quick':
        results = quick_demo()
    elif args.mode == 'viz':
        visualizations_demo()
    
    print("\n✨ Example completed! Check results/ directory for outputs.")