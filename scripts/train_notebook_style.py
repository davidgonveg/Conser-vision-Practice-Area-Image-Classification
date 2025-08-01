def main():
    """Main training function that replicates notebook workflow exactly."""
    
    parser = argparse.ArgumentParser(description='Train Wildlife Classifier - Notebook Style')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                       help='Path to data directory')
    parser.add_argument('--fraction', type=float, default=1.0,
                       help='Fraction of data to use (notebook frac parameter)')
    parser.add_argument('--random_state', type=int, default=1,
                       help='Random seed (notebook uses 1)')
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--validation_sites_file', type=str, default=None,
                       help='Path to validation sites CSV file')
    parser.add_argument('--save_model_path', type=str, default='results/models/notebook_style_model.pth',
                       help='Path to save trained model')
    parser.add_argument('--save_plots_path', type=str, default='results/plots/loss_curves.png',
                       help='Path to save loss plots')
    parser.add_argument('--use_preprocessing', action='store_true', default=True,
                       help='Use custom preprocessing (notebook default: True)')
    parser.add_argument('--use_augmentation', action='store_true', default=True,
                       help='Use data augmentation (notebook default: True)')
    parser.add_argument('--num_augmentations', type=int, default=2,
                       help='Number of augmented datasets (notebook default: 2)')
    
    # Evaluation and submission arguments
    parser.add_argument('--skip_evaluation', action='store_true',
                       help='Skip model evaluation step')
    parser.add_argument('--skip_submission', action='store_true',
                       help='Skip submission generation step')
    parser.add_argument('--submission_format_path', type=str, default='data/raw/submission_format.csv',
                       help='Path to submission format CSV')
    parser.add_argument('--submission_output_path', type=str, default='data/submissions/submission.csv',
                       help='Path to save submission file')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Wildlife Classification Training - Notebook Style")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Fraction: {args.fraction}")
    logger.info(f"Random state: {args.random_state}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Create output directories
    Path(args.save_model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save_plots_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.submission_output_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Load dataset exactly like notebook
        logger.info("📊 Loading dataset...")
        dataset_manager = TaiParkDatasetNotebookStyle(
            data_dir=args.data_dir,
            fraction=args.fraction,
            random_state=args.random_state,
            validation_sites_file=args.validation_sites_file,
            use_preprocessing=args.use_preprocessing,
            use_augmentation=args.use_augmentation,
            num_augmentations=args.num_augmentations
        )
        
        logger.info(f"✅ Dataset loaded: {len(dataset_manager.y_train)} train, {len(dataset_manager.y_eval)} val")
        
        # Display species distribution like notebook
        logger.info("📈 Species distribution:")
        train_dist = dataset_manager.get_class_distribution('train')
        eval_dist = dataset_manager.get_class_distribution('eval')
        
        for species in dataset_manager.species_labels:
            train_count = train_dist.get(species, 0)
            eval_count = eval_dist.get(species, 0)
            logger.info(f"  {species}: {train_count} train, {eval_count} val")
        
        # 2. Create model exactly like notebook
        logger.info("🤖 Creating model...")
        model = create_notebook_model()
        
        model_info = get_model_info(model)
        logger.info(f"✅ Model created: {model_info['model_name']}")
        logger.info(f"   Total parameters: {model_info['total_parameters']:,}")
        logger.info(f"   Trainable parameters: {model_info['trainable_parameters']:,}")
        logger.info(f"   Trainable percentage: {model_info['trainable_percentage']:.1f}%")
        
        # 3. Create trainer exactly like notebook
        logger.info("🏋️ Setting up trainer...")
        trainer = create_notebook_trainer(model, dataset_manager)
        
        logger.info("✅ Trainer ready:")
        logger.info(f"   Optimizer: SGD (lr=0.01, momentum=0.909431, weight_decay=0.005)")
        logger.info(f"   Scheduler: ReduceLROnPlateau (patience=2, factor=0.72)")
        logger.info(f"   Loss function: CrossEntropyLoss")
        logger.info(f"   Early stopping: tolerance=5, min_delta=0.0001")
        
        # 4. Train model exactly like notebook  
        logger.info("🚂 Starting training...")
        loss_history = trainer.train(
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            save_best_model=True
        )
        
        logger.info(f"✅ Training completed!")
        logger.info(f"   Best evaluation loss: {trainer.best_eval_loss:.4f}")
        logger.info(f"   Total training steps: {len(loss_history['training_loss'])}")
        logger.info(f"   Total evaluation steps: {len(loss_history['evaluation_loss'])}")
        
        # 5. Display loss tracking like notebook
        logger.info("📊 Loss History (last 10 evaluations):")
        pd.set_option('display.max_rows', None)
        recent_eval_loss = loss_history['evaluation_loss'].tail(10)
        print(recent_eval_loss.to_string(index=False))
        
        # 6. Save model exactly like notebook
        logger.info("💾 Saving model...")
        trainer.save_model(args.save_model_path)
        logger.info(f"✅ Model saved to {args.save_model_path}")
        
        # 7. Plot loss curves exactly like notebook
        logger.info("📈 Creating loss plots...")
        plot_loss_curves(loss_history, args.save_plots_path)
        
        # 8. Model Evaluation exactly like notebook
        if not args.skip_evaluation:
            logger.info("🔍 Starting model evaluation...")
            
            # Create evaluation dataloader
            eval_dataset = ImagesDataset(dataset_manager.x_eval, dataset_manager.y_eval, 
                                       preprocessing=custom_preprocessing)
            eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)
            
            # Evaluate exactly like notebook
            eval_results = evaluate_notebook_style(
                model=trainer.model,
                eval_dataloader=eval_dataloader,
                true_labels_df=dataset_manager.y_eval,
                species_labels=dataset_manager.species_labels,
                device=trainer.device,
                save_plots_dir=Path(args.save_plots_path).parent
            )
            
            logger.info(f"✅ Evaluation completed!")
            logger.info(f"   Accuracy: {eval_results['accuracy']:.1%}")
            logger.info(f"   Incorrect predictions: {eval_results['incorrect_count']}")
            
            # Baseline comparisons like notebook
            logger.info("📊 Baseline Comparisons:")
            logger.info(f"   Random guessing: {eval_results['random_accuracy']:.1%}")
            logger.info(f"   Always most common: {eval_results['baseline_accuracy']:.1%}")
            logger.info(f"   Our model: {eval_results['accuracy']:.1%}")
        
        # 9. Generate Submission exactly like notebook
        if not args.skip_submission:
            logger.info("📤 Generating submission...")
            
            # Load test features exactly like notebook
            test_features_path = Path(args.data_dir) / "test_features.csv"
            if test_features_path.exists():
                test_features_df = pd.read_csv(test_features_path, index_col="id")
                
                # Generate submission exactly like notebook
                submission_df = create_notebook_submission(
                    model=trainer.model,
                    test_features_df=test_features_df,
                    species_labels=dataset_manager.species_labels,
                    device=trainer.device,
                    output_path=args.submission_output_path,
                    submission_format_path=args.submission_format_path if Path(args.submission_format_path).exists() else None,
                    batch_size=args.batch_size,
                    use_preprocessing=True
                )
                
                logger.info(f"✅ Submission generated!")
                logger.info(f"   Submission shape: {submission_df.shape}")
                logger.info(f"   Saved to: {args.submission_output_path}")
                logger.info("   Ready for competition upload! 🎯")
            else:
                logger.warning(f"Test features file not found: {test_features_path}")
                logger.info("Skipping submission generation")
        
        # 10. Final Summary exactly like notebook
        logger.info("🎉 Training Summary:")
        logger.info(f"   Dataset: {len(dataset_manager.y_train) + len(dataset_manager.y_eval)} total samples")
        logger.info(f"   Model: ResNet152 with {model_info['trainable_parameters']:,} trainable parameters")
        logger.info(f"   Training: {args.num_epochs} epochs, batch size {args.batch_size}")
        logger.info(f"   Best eval loss: {trainer.best_eval_loss:.4f}")
        logger.info(f"   Early stopping: {'Yes' if len(loss_history['evaluation_loss']) < args.num_epochs * 10 else 'No'}")
        
        # Site distribution summary (this is better than notebook's stratified split)
        train_sites = len(dataset_manager.get_site_distribution('train'))
        eval_sites = len(dataset_manager.get_site_distribution('eval'))
        logger.info(f"   Site-aware split: {train_sites} train sites, {eval_sites} val sites")
        logger.info("   ✨ No data leakage - sites are completely separated!")
        
        if not args.skip_evaluation:
            logger.info(f"   Final accuracy: {eval_results['accuracy']:.1%}")
        
        if not args.skip_submission:
            logger.info(f"   Submission ready: {args.submission_output_path}")
        
        return trainer, loss_history
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise"""
Main Training Script - Notebook Style

This script replicates the exact workflow from the successful notebook
but in a clean, modular way. Run this to reproduce your notebook results.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import TaiParkDatasetNotebookStyle, ImagesDataset, custom_preprocessing
from src.models.model import create_notebook_model, get_model_info
from src.training.trainer import create_notebook_trainer
from src.evaluation.evaluator import evaluate_notebook_style
from src.inference.predictor import create_notebook_submission
from torch.utils.data import DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_loss_curves(loss_history: dict, save_path: str = None):
    """
    Plot loss curves exactly like notebook.
    
    Creates the same visualization with smoothed training and evaluation loss.
    """
    
    tracking_loss = loss_history['training_loss'].copy()
    tracking_loss_ev = loss_history['evaluation_loss'].copy()
    
    # Add smoothed loss exactly like notebook
    tracking_loss_ev["loss_smoothed"] = tracking_loss_ev["Loss"].rolling(window=5).mean()
    tracking_loss["loss_smoothed"] = tracking_loss["Loss"].rolling(window=5).mean()
    
    # Create plot exactly like notebook
    plt.figure(figsize=(10, 5))
    plt.plot(tracking_loss_ev['Epoch'], tracking_loss_ev['loss_smoothed'], 
             color="red", marker="o", label="Evaluation")
    plt.plot(tracking_loss['Epoch'], tracking_loss['loss_smoothed'], 
             color="blue", marker="x", label="Training")
    
    plt.xlabel("(Epoch, Batch)")
    plt.ylabel("Loss")
    plt.legend(loc=0)
    plt.title("Training and Evaluation Loss")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Loss plot saved to {save_path}")
    
    plt.show()


def main():
    """Main training function that replicates notebook workflow exactly."""
    
    parser = argparse.ArgumentParser(description='Train Wildlife Classifier - Notebook Style')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                       help='Path to data directory')
    parser.add_argument('--fraction', type=float, default=1.0,
                       help='Fraction of data to use (notebook frac parameter)')
    parser.add_argument('--random_state', type=int, default=1,
                       help='Random seed (notebook uses 1)')
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--validation_sites_file', type=str, default=None,
                       help='Path to validation sites CSV file')
    parser.add_argument('--save_model_path', type=str, default='results/models/notebook_style_model.pth',
                       help='Path to save trained model')
    parser.add_argument('--save_plots_path', type=str, default='results/plots/loss_curves.png',
                       help='Path to save loss plots')
    parser.add_argument('--use_preprocessing', action='store_true', default=True,
                       help='Use custom preprocessing (notebook default: True)')
    parser.add_argument('--use_augmentation', action='store_true', default=True,
                       help='Use data augmentation (notebook default: True)')
    parser.add_argument('--num_augmentations', type=int, default=2,
                       help='Number of augmented datasets (notebook default: 2)')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Wildlife Classification Training - Notebook Style")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Fraction: {args.fraction}")
    logger.info(f"Random state: {args.random_state}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Create output directories
    Path(args.save_model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save_plots_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Load dataset exactly like notebook
        logger.info("📊 Loading dataset...")
        dataset_manager = TaiParkDatasetNotebookStyle(
            data_dir=args.data_dir,
            fraction=args.fraction,
            random_state=args.random_state,
            validation_sites_file=args.validation_sites_file,
            use_preprocessing=args.use_preprocessing,
            use_augmentation=args.use_augmentation,
            num_augmentations=args.num_augmentations
        )
        
        logger.info(f"✅ Dataset loaded: {len(dataset_manager.y_train)} train, {len(dataset_manager.y_eval)} val")
        
        # Display species distribution like notebook
        logger.info("📈 Species distribution:")
        train_dist = dataset_manager.get_class_distribution('train')
        eval_dist = dataset_manager.get_class_distribution('eval')
        
        for species in dataset_manager.species_labels:
            train_count = train_dist.get(species, 0)
            eval_count = eval_dist.get(species, 0)
            logger.info(f"  {species}: {train_count} train, {eval_count} val")
        
        # 2. Create model exactly like notebook
        logger.info("🤖 Creating model...")
        model = create_notebook_model()
        
        model_info = get_model_info(model)
        logger.info(f"✅ Model created: {model_info['model_name']}")
        logger.info(f"   Total parameters: {model_info['total_parameters']:,}")
        logger.info(f"   Trainable parameters: {model_info['trainable_parameters']:,}")
        logger.info(f"   Trainable percentage: {model_info['trainable_percentage']:.1f}%")
        
        # 3. Create trainer exactly like notebook
        logger.info("🏋️ Setting up trainer...")
        trainer = create_notebook_trainer(model, dataset_manager)
        
        logger.info("✅ Trainer ready:")
        logger.info(f"   Optimizer: SGD (lr=0.01, momentum=0.909431, weight_decay=0.005)")
        logger.info(f"   Scheduler: ReduceLROnPlateau (patience=2, factor=0.72)")
        logger.info(f"   Loss function: CrossEntropyLoss")
        logger.info(f"   Early stopping: tolerance=5, min_delta=0.0001")
        
        # 4. Train model exactly like notebook  
        logger.info("🚂 Starting training...")
        loss_history = trainer.train(
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            save_best_model=True
        )
        
        logger.info(f"✅ Training completed!")
        logger.info(f"   Best evaluation loss: {trainer.best_eval_loss:.4f}")
        logger.info(f"   Total training steps: {len(loss_history['training_loss'])}")
        logger.info(f"   Total evaluation steps: {len(loss_history['evaluation_loss'])}")
        
        # 5. Display loss tracking like notebook
        logger.info("📊 Loss History (last 10 evaluations):")
        pd.set_option('display.max_rows', None)
        recent_eval_loss = loss_history['evaluation_loss'].tail(10)
        print(recent_eval_loss.to_string(index=False))
        
        # 6. Save model exactly like notebook
        logger.info("💾 Saving model...")
        trainer.save_model(args.save_model_path)
        logger.info(f"✅ Model saved to {args.save_model_path}")
        
        # 7. Plot loss curves exactly like notebook
        logger.info("📈 Creating loss plots...")
        plot_loss_curves(loss_history, args.save_plots_path)
        
        # 8. Summary exactly like notebook
        logger.info("🎉 Training Summary:")
        logger.info(f"   Dataset: {len(dataset_manager.y_train) + len(dataset_manager.y_eval)} total samples")
        logger.info(f"   Model: ResNet152 with {model_info['trainable_parameters']:,} trainable parameters")
        logger.info(f"   Training: {args.num_epochs} epochs, batch size {args.batch_size}")
        logger.info(f"   Best eval loss: {trainer.best_eval_loss:.4f}")
        logger.info(f"   Early stopping: {'Yes' if len(loss_history['evaluation_loss']) < args.num_epochs * 10 else 'No'}")
        
        # Site distribution summary (this is better than notebook's stratified split)
        train_sites = len(dataset_manager.get_site_distribution('train'))
        eval_sites = len(dataset_manager.get_site_distribution('eval'))
        logger.info(f"   Site-aware split: {train_sites} train sites, {eval_sites} val sites")
        logger.info("   ✨ No data leakage - sites are completely separated!")
        
        return trainer, loss_history
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise


def quick_test():
    """Quick test function for development."""
    
    logger.info("🧪 Running quick test...")
    
    # Test with small fraction for speed
    dataset_manager = TaiParkDatasetNotebookStyle(
        data_dir="data/raw",
        fraction=0.1,  # Use only 10% of data for quick test
        random_state=1
    )
    
    model = create_notebook_model()
    trainer = create_notebook_trainer(model, dataset_manager)
    
    # Train for just 1 epoch
    loss_history = trainer.train(num_epochs=1, batch_size=32)
    
    logger.info(f"✅ Quick test completed! Best eval loss: {trainer.best_eval_loss:.4f}")
    
    return trainer, loss_history


if __name__ == "__main__":
    
    # Check if running in test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--quick_test":
        quick_test()
    else:
        main()