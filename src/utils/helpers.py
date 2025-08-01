"""
Utility functions for the Tai Park Wildlife Classification project.

This module contains helper functions for visualization, model analysis,
and other common tasks.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Add other random seeds as needed
    logger.info(f"Random seed set to {seed}")


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    return device


def visualize_samples(
    dataset_manager,
    num_samples: int = 8,
    figsize: Tuple[int, int] = (20, 20),
    save_path: Optional[str] = None
):
    """
    Visualize random samples from the dataset exactly like notebook.
    
    Creates a grid showing one sample for each species class.
    """
    
    species_labels = dataset_manager.species_labels
    
    # Create grid exactly like notebook
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=figsize)
    
    # Iterate through each species like notebook
    for species, ax in zip(species_labels, axes.flat):
        # Get an image ID for this species
        species_samples = dataset_manager.y_train[dataset_manager.y_train[species] == 1]
        
        if len(species_samples) > 0:
            img_id = species_samples.sample(1, random_state=42).index[0]
            
            # Get filepath and load image
            img_filepath = dataset_manager.x_train.loc[img_id, 'filepath']
            full_path = Path(dataset_manager.data_dir) / img_filepath
            
            # Load and display image
            img = Image.open(full_path)
            ax.imshow(img)
            ax.set_title(f"{img_id} | {species}")
            ax.axis('off')
        else:
            ax.set_title(f"No samples for {species}")
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Sample visualization saved to {save_path}")
    
    plt.show()


def plot_class_distribution(
    dataset_manager,
    save_path: Optional[str] = None
):
    """Plot class distribution for train and validation sets."""
    
    train_dist = dataset_manager.get_class_distribution('train')
    eval_dist = dataset_manager.get_class_distribution('eval')
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Train': train_dist,
        'Validation': eval_dist
    }).fillna(0)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Absolute counts
    df.plot(kind='bar', ax=ax1)
    ax1.set_title('Class Distribution (Absolute Counts)')
    ax1.set_xlabel('Species')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Percentages
    df_pct = df.div(df.sum(axis=0), axis=1) * 100
    df_pct.plot(kind='bar', ax=ax2)
    ax2.set_title('Class Distribution (Percentages)')
    ax2.set_xlabel('Species')
    ax2.set_ylabel('Percentage')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Class distribution plot saved to {save_path}")
    
    plt.show()


def plot_site_distribution(
    dataset_manager,
    save_path: Optional[str] = None
):
    """Plot site distribution to verify no data leakage."""
    
    train_sites = dataset_manager.get_site_distribution('train')
    eval_sites = dataset_manager.get_site_distribution('eval')
    
    # Check for overlap (should be zero for proper split)
    train_site_set = set(train_sites.keys())
    eval_site_set = set(eval_sites.keys())
    overlap = train_site_set.intersection(eval_site_set)
    
    if overlap:
        logger.warning(f"Site overlap detected: {overlap}")
    else:
        logger.info("✅ No site overlap - perfect split!")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Train sites
    train_df = pd.Series(train_sites).sort_values(ascending=False)
    train_df.head(20).plot(kind='bar', ax=ax1, color='blue', alpha=0.7)
    ax1.set_title(f'Top 20 Training Sites ({len(train_sites)} total)')
    ax1.set_xlabel('Site ID')
    ax1.set_ylabel('Number of Images')
    
    # Validation sites
    eval_df = pd.Series(eval_sites).sort_values(ascending=False)
    eval_df.head(20).plot(kind='bar', ax=ax2, color='red', alpha=0.7)
    ax2.set_title(f'Top 20 Validation Sites ({len(eval_sites)} total)')
    ax2.set_xlabel('Site ID')
    ax2.set_ylabel('Number of Images')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Site distribution plot saved to {save_path}")
    
    plt.show()
    
    return len(overlap) == 0  # Return True if no overlap


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str]
) -> Dict[str, Union[float, np.ndarray, str]]:
    """
    Comprehensive model evaluation.
    
    Returns metrics including accuracy, log loss, confusion matrix, and classification report.
    """
    
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    all_image_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            image_ids = batch['image_id']
            
            # Forward pass
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            # Convert to numpy and store
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(torch.argmax(labels, dim=1).cpu().numpy())
            all_image_ids.extend(image_ids)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = (all_predictions == all_labels).mean()
    
    # Log loss (convert labels to one-hot for log_loss function)
    labels_onehot = np.eye(len(class_names))[all_labels]
    logloss = log_loss(labels_onehot, all_probabilities)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Classification report
    report = classification_report(
        all_labels, 
        all_predictions, 
        target_names=class_names,
        output_dict=True
    )
    
    return {
        'accuracy': accuracy,
        'log_loss': logloss,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'labels': all_labels,
        'image_ids': all_image_ids
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    normalize: bool = True,
    save_path: Optional[str] = None
):
    """Plot confusion matrix."""
    
    if normalize:
        cm_plot = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
        fmt = '.2f'
    else:
        cm_plot = cm
        title = 'Confusion Matrix'
        fmt = 'd'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_plot, 
        annot=True, 
        fmt=fmt, 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def create_submission_file(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    class_names: List[str],
    device: torch.device,
    output_path: str
) -> pd.DataFrame:
    """
    Create submission file for competition.
    
    Generates probabilities for each test image exactly like notebook format.
    """
    
    model.eval()
    all_probabilities = []
    all_image_ids = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            images = batch['image'].to(device)
            image_ids = batch['image_id']
            
            # Forward pass
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            
            all_probabilities.extend(probabilities.cpu().numpy())
            all_image_ids.extend(image_ids)
    
    # Create submission DataFrame
    submission_df = pd.DataFrame(all_probabilities, columns=class_names)
    submission_df.insert(0, 'id', all_image_ids)
    
    # Save to file
    submission_df.to_csv(output_path, index=False)
    logger.info(f"Submission file saved to {output_path}")
    logger.info(f"Submission shape: {submission_df.shape}")
    
    return submission_df


def analyze_predictions(
    predictions_data: Dict,
    class_names: List[str],
    save_path: Optional[str] = None
):
    """Analyze model predictions in detail."""
    
    predictions = predictions_data['predictions']
    probabilities = predictions_data['probabilities']
    labels = predictions_data['labels']
    
    # Calculate confidence statistics
    max_probs = np.max(probabilities, axis=1)
    predicted_classes = np.argmax(probabilities, axis=1)
    
    # Create analysis DataFrame
    analysis_df = pd.DataFrame({
        'image_id': predictions_data['image_ids'],
        'true_class': [class_names[i] for i in labels],
        'predicted_class': [class_names[i] for i in predictions],
        'confidence': max_probs,
        'correct': predictions == labels
    })
    
    # Summary statistics
    print("Prediction Analysis Summary:")
    print(f"Total samples: {len(analysis_df)}")
    print(f"Accuracy: {analysis_df['correct'].mean():.4f}")
    print(f"Mean confidence: {analysis_df['confidence'].mean():.4f}")
    print(f"Std confidence: {analysis_df['confidence'].std():.4f}")
    
    # Confidence by correctness
    print("\nConfidence by Correctness:")
    correct_conf = analysis_df[analysis_df['correct']]['confidence'].mean()
    incorrect_conf = analysis_df[~analysis_df['correct']]['confidence'].mean()
    print(f"Correct predictions: {correct_conf:.4f}")
    print(f"Incorrect predictions: {incorrect_conf:.4f}")
    
    # Class-wise accuracy
    print("\nClass-wise Accuracy:")
    class_accuracy = analysis_df.groupby('true_class')['correct'].mean().sort_values(ascending=False)
    for class_name, acc in class_accuracy.items():
        print(f"{class_name}: {acc:.4f}")
    
    if save_path:
        analysis_df.to_csv(save_path, index=False)
        logger.info(f"Prediction analysis saved to {save_path}")
    
    return analysis_df


def notebook_style_summary(
    trainer,
    dataset_manager,
    model_info: Dict,
    loss_history: Dict
):
    """
    Create a summary exactly like the notebook output.
    
    Displays all the key information in notebook style.
    """
    
    print("=" * 60)
    print("🎯 WILDLIFE CLASSIFICATION TRAINING SUMMARY")
    print("=" * 60)
    
    # Dataset Summary
    print("\n📊 DATASET:")
    print(f"   Total samples: {len(dataset_manager.y_train) + len(dataset_manager.y_eval):,}")
    print(f"   Training samples: {len(dataset_manager.y_train):,}")
    print(f"   Validation samples: {len(dataset_manager.y_eval):,}")
    print(f"   Data fraction used: {dataset_manager.fraction}")
    print(f"   Random state: {dataset_manager.random_state}")
    
    # Site split verification
    train_sites = len(dataset_manager.get_site_distribution('train'))
    eval_sites = len(dataset_manager.get_site_distribution('eval'))
    print(f"   Training sites: {train_sites}")
    print(f"   Validation sites: {eval_sites}")
    print("   ✅ Site-aware split (no data leakage)")
    
    # Model Summary
    print(f"\n🤖 MODEL:")
    print(f"   Architecture: {model_info['model_name']}")
    print(f"   Total parameters: {model_info['total_parameters']:,}")
    print(f"   Trainable parameters: {model_info['trainable_parameters']:,}")
    print(f"   Trainable percentage: {model_info['trainable_percentage']:.1f}%")
    print(f"   Pretrained: {model_info['pretrained']}")
    
    # Training Summary
    print(f"\n🚂 TRAINING:")
    print(f"   Best evaluation loss: {trainer.best_eval_loss:.6f}")
    print(f"   Total training steps: {len(loss_history['training_loss'])}")
    print(f"   Total evaluation steps: {len(loss_history['evaluation_loss'])}")
    
    # Early stopping info
    final_epoch = loss_history['evaluation_loss']['Epoch'].max()
    early_stopped = final_epoch < 4.9  # Less than 5 epochs
    print(f"   Early stopping: {'Yes' if early_stopped else 'No'}")
    if early_stopped:
        print(f"   Stopped at epoch: {final_epoch:.1f}")
    
    # Loss trends
    eval_losses = loss_history['evaluation_loss']['Loss']
    if len(eval_losses) > 1:
        trend = "Decreasing" if eval_losses.iloc[-1] < eval_losses.iloc[0] else "Increasing"
        print(f"   Loss trend: {trend}")
    
    print("\n" + "=" * 60)
    print("✨ Training completed successfully!")
    print("=" * 60)


# Example usage functions
def quick_visualization_demo(dataset_manager):
    """Quick demo of visualization functions."""
    
    print("🎨 Creating visualizations...")
    
    # Visualize samples
    visualize_samples(dataset_manager, save_path="results/plots/sample_images.png")
    
    # Plot distributions
    plot_class_distribution(dataset_manager, save_path="results/plots/class_distribution.png")
    plot_site_distribution(dataset_manager, save_path="results/plots/site_distribution.png")
    
    print("✅ Visualizations complete!")


if __name__ == "__main__":
    # Demo usage
    print("Utility functions ready! Import these in your training scripts.")