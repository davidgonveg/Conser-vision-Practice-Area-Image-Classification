"""
Model Evaluation Module

This module replicates the exact evaluation logic from the notebook
for analyzing model performance on validation data.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, log_loss
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class NotebookStyleEvaluator:
    """
    Model evaluator that replicates exact notebook evaluation logic.
    
    Includes prediction generation, accuracy calculation, confusion matrix,
    and detailed analysis exactly like the notebook.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        species_labels: List[str]
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model for evaluation
            device: Device for inference
            species_labels: List of species class names
        """
        
        self.model = model
        self.device = device
        self.species_labels = species_labels
        
        logger.info(f"Evaluator initialized with {len(species_labels)} classes")

    def make_predictions(
        self, 
        dataloader: DataLoader,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Make predictions exactly like notebook.
        
        Generates softmax probabilities for each image in the dataloader.
        
        Args:
            dataloader: DataLoader with evaluation data
            show_progress: Whether to show progress bar
            
        Returns:
            DataFrame with predictions (rows=images, cols=species)
        """
        
        preds_collector = []
        
        # Put model in eval mode exactly like notebook
        self.model.eval()
        
        # No gradients needed for evaluation
        with torch.no_grad():
            
            progress_bar = tqdm(dataloader, total=len(dataloader)) if show_progress else dataloader
            
            for batch in progress_bar:
                
                # 1) Run forward pass exactly like notebook
                logits = self.model.forward(batch["image"].to(self.device))
                
                # 2) Apply softmax exactly like notebook
                preds = F.softmax(logits, dim=1)
                
                # 3) Store batch predictions exactly like notebook
                preds_df = pd.DataFrame(
                    preds.cpu().detach().numpy(),
                    index=batch["image_id"],
                    columns=self.species_labels,
                )
                preds_collector.append(preds_df)
        
        # Concatenate all predictions exactly like notebook
        eval_preds_df = pd.concat(preds_collector)
        
        logger.info(f"Generated predictions for {len(eval_preds_df)} samples")
        
        return eval_preds_df

    def analyze_predictions(
        self, 
        predictions_df: pd.DataFrame,
        true_labels_df: pd.DataFrame,
        show_incorrect: bool = True
    ) -> Dict[str, any]:
        """
        Analyze predictions exactly like notebook.
        
        Args:
            predictions_df: Model predictions DataFrame
            true_labels_df: True labels DataFrame  
            show_incorrect: Whether to print incorrect predictions
            
        Returns:
            Dictionary with analysis results
        """
        
        # Get predicted and true labels exactly like notebook
        predicted_labels = predictions_df.idxmax(axis=1)
        true_labels = true_labels_df.idxmax(axis=1)
        
        # Calculate accuracy exactly like notebook
        correct_predictions = (true_labels == predicted_labels)
        accuracy = correct_predictions.sum() / len(predicted_labels)
        
        # Show incorrect predictions exactly like notebook
        if show_incorrect:
            print("Incorrect Predictions Analysis:")
            print("=" * 50)
            
            incorrect_rows = predictions_df[~correct_predictions]
            
            for idx, row in incorrect_rows.iterrows():
                correct_label = true_labels_df.loc[idx][true_labels_df.loc[idx] == 1].index[0]
                predicted_label = predicted_labels.loc[idx]
                confidence = row[predicted_label]
                
                print(f"ID: {idx}")
                print(f"Predicted: {predicted_label} (confidence: {confidence:.3f})")
                print(f"Correct: {correct_label}")
                print(f"All predictions: {row.round(3).to_dict()}")
                print()
        
        # Distribution analysis exactly like notebook
        print("Label Distribution Analysis:")
        print("=" * 40)
        
        print("Predicted labels distribution:")
        pred_dist = predicted_labels.value_counts()
        print(pred_dist)
        print()
        
        print("True labels distribution:")
        true_dist = true_labels.value_counts()
        print(true_dist)
        print()
        
        # Baseline comparisons exactly like notebook
        most_common_class = true_dist.index[0]
        baseline_accuracy = (true_labels == most_common_class).sum() / len(predicted_labels)
        random_accuracy = 1.0 / len(self.species_labels)
        
        print("Baseline Comparisons:")
        print(f"Random guessing accuracy: {random_accuracy:.1%}")
        print(f"Always predict '{most_common_class}' accuracy: {baseline_accuracy:.1%}")
        print(f"Model accuracy: {accuracy:.1%}")
        print()
        
        return {
            'predictions_df': predictions_df,
            'predicted_labels': predicted_labels,
            'true_labels': true_labels,
            'accuracy': accuracy,
            'baseline_accuracy': baseline_accuracy,
            'random_accuracy': random_accuracy,
            'correct_predictions': correct_predictions,
            'incorrect_count': (~correct_predictions).sum(),
            'predicted_distribution': pred_dist,
            'true_distribution': true_dist
        }

    def plot_confusion_matrix(
        self,
        true_labels: pd.Series,
        predicted_labels: pd.Series,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 10)
    ):
        """
        Plot confusion matrix exactly like notebook.
        
        Args:
            true_labels: True labels series
            predicted_labels: Predicted labels series
            save_path: Path to save plot
            figsize: Figure size
        """
        
        # Create confusion matrix exactly like notebook
        fig, ax = plt.subplots(figsize=figsize)
        
        cm = ConfusionMatrixDisplay.from_predictions(
            true_labels,
            predicted_labels,
            ax=ax,
            xticks_rotation=90,
            colorbar=True,
        )
        
        ax.set_title("Confusion Matrix - Model Performance")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()

    def evaluate_model(
        self,
        eval_dataloader: DataLoader,
        true_labels_df: pd.DataFrame,
        save_plots_dir: Optional[str] = None,
        show_analysis: bool = True
    ) -> Dict[str, any]:
        """
        Complete model evaluation exactly like notebook.
        
        Args:
            eval_dataloader: DataLoader with evaluation data
            true_labels_df: True labels DataFrame
            save_plots_dir: Directory to save plots
            show_analysis: Whether to show detailed analysis
            
        Returns:
            Complete evaluation results
        """
        
        logger.info("🔍 Starting model evaluation...")
        
        # Make predictions exactly like notebook
        predictions_df = self.make_predictions(eval_dataloader)
        
        # Analyze predictions exactly like notebook
        if show_analysis:
            analysis_results = self.analyze_predictions(
                predictions_df, 
                true_labels_df,
                show_incorrect=True
            )
        else:
            # Quick analysis without printing
            predicted_labels = predictions_df.idxmax(axis=1)
            true_labels = true_labels_df.idxmax(axis=1)
            correct_predictions = (true_labels == predicted_labels)
            accuracy = correct_predictions.sum() / len(predicted_labels)
            
            analysis_results = {
                'predictions_df': predictions_df,
                'predicted_labels': predicted_labels,
                'true_labels': true_labels,
                'accuracy': accuracy,
                'correct_predictions': correct_predictions
            }
        
        # Plot confusion matrix exactly like notebook
        if save_plots_dir:
            Path(save_plots_dir).mkdir(parents=True, exist_ok=True)
            cm_path = Path(save_plots_dir) / "confusion_matrix.png"
        else:
            cm_path = None
        
        self.plot_confusion_matrix(
            analysis_results['true_labels'],
            analysis_results['predicted_labels'],
            save_path=cm_path
        )
        
        logger.info(f"✅ Evaluation completed! Accuracy: {analysis_results['accuracy']:.1%}")
        
        return analysis_results


def evaluate_notebook_style(
    model: torch.nn.Module,
    eval_dataloader: DataLoader,
    true_labels_df: pd.DataFrame,
    species_labels: List[str],
    device: torch.device,
    save_plots_dir: Optional[str] = None
) -> Dict[str, any]:
    """
    Complete evaluation exactly like notebook workflow.
    
    This is the main function that replicates your notebook evaluation.
    
    Args:
        model: Trained model
        eval_dataloader: Evaluation DataLoader
        true_labels_df: True labels DataFrame (y_eval from notebook)
        species_labels: List of species names
        device: Device for inference
        save_plots_dir: Directory to save plots
        
    Returns:
        Complete evaluation results
    """
    
    evaluator = NotebookStyleEvaluator(model, device, species_labels)
    
    return evaluator.evaluate_model(
        eval_dataloader=eval_dataloader,
        true_labels_df=true_labels_df,
        save_plots_dir=save_plots_dir,
        show_analysis=True
    )


# Example usage
if __name__ == "__main__":
    
    # Example of how to use the evaluator
    print("Model Evaluator ready!")
    print("Use evaluate_notebook_style() to replicate notebook evaluation exactly.")