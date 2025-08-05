"""
Inference and Submission Module

This module replicates the exact submission generation logic from the notebook
for creating competition submissions.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import ImagesDataset, custom_preprocessing

logger = logging.getLogger(__name__)


class NotebookStylePredictor:
    """
    Predictor that generates submissions exactly like notebook.
    
    Handles test data loading, prediction generation, and submission formatting
    exactly as done in the successful notebook.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        species_labels: List[str]
    ):
        """
        Initialize predictor.
        
        Args:
            model: Trained model for predictions
            device: Device for inference
            species_labels: List of species class names (must be in correct order)
        """
        
        self.model = model
        self.device = device
        self.species_labels = species_labels
        
        # Put model in evaluation mode
        self.model.eval()
        
        logger.info(f"Predictor initialized with {len(species_labels)} classes")

    def create_test_dataset_and_loader(
        self,
        test_features_df: pd.DataFrame,
        batch_size: int = 64,
        use_preprocessing: bool = True
    ) -> DataLoader:
        """
        Create test dataset and dataloader exactly like notebook.
        
        Args:
            test_features_df: Test features DataFrame (from test_features.csv)
            batch_size: Batch size for predictions
            use_preprocessing: Whether to use custom preprocessing
            
        Returns:
            DataLoader for test data
        """
        
        # Create test dataset exactly like notebook
        preprocessing = custom_preprocessing if use_preprocessing else None
        
        test_dataset = ImagesDataset(
            test_features_df.filepath.to_frame(),
            y_df=None,  # No labels for test set
            preprocessing=preprocessing,
            augmentation=None  # No augmentation for test
        )
        
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        
        logger.info(f"Created test dataloader with {len(test_dataset)} samples")
        
        return test_dataloader

    def generate_predictions(
        self,
        test_dataloader: DataLoader,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Generate predictions exactly like notebook.
        
        Args:
            test_dataloader: DataLoader with test data
            show_progress: Whether to show progress bar
            
        Returns:
            DataFrame with predictions (rows=test_images, cols=species)
        """
        
        preds_collector = []
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # No gradients needed for inference
        with torch.no_grad():
            
            progress_bar = tqdm(test_dataloader, total=len(test_dataloader), 
                              desc="Generating predictions") if show_progress else test_dataloader
            
            for batch in progress_bar:
                
                # Run forward pass exactly like notebook
                logits = self.model.forward(batch["image"].to(self.device))
                
                # Apply softmax exactly like notebook
                preds = F.softmax(logits, dim=1)
                
                # Store batch predictions exactly like notebook
                preds_df = pd.DataFrame(
                    preds.cpu().detach().numpy(),
                    index=batch["image_id"],
                    columns=self.species_labels,
                )
                preds_collector.append(preds_df)
        
        # Concatenate all predictions exactly like notebook
        submission_df = pd.concat(preds_collector)
        
        logger.info(f"Generated predictions for {len(submission_df)} test samples")
        
        return submission_df

    def validate_submission(
        self,
        submission_df: pd.DataFrame,
        submission_format_path: str
    ) -> bool:
        """
        Validate submission exactly like notebook.
        
        Args:
            submission_df: Generated submission DataFrame
            submission_format_path: Path to submission_format.csv
            
        Returns:
            True if validation passes
        """
        
        try:
            # Load submission format exactly like notebook
            submission_format = pd.read_csv(submission_format_path, index_col="id")
            
            # Validate index exactly like notebook
            assert all(submission_df.index == submission_format.index), \
                "Submission index doesn't match format"
            
            # Validate columns exactly like notebook
            assert all(submission_df.columns == submission_format.columns), \
                "Submission columns don't match format"
            
            # Additional validations
            assert not submission_df.isnull().any().any(), \
                "Submission contains null values"
            
            assert (submission_df >= 0).all().all(), \
                "Submission contains negative probabilities"
            
            # Check that probabilities sum to ~1 for each row
            row_sums = submission_df.sum(axis=1)
            assert (row_sums >= 0.99).all() and (row_sums <= 1.01).all(), \
                "Probabilities don't sum to 1 for all rows"
            
            logger.info("✅ Submission validation passed!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Submission validation failed: {str(e)}")
            return False

    def create_submission(
        self,
        test_features_df: pd.DataFrame,
        output_path: str,
        submission_format_path: Optional[str] = None,
        batch_size: int = 64,
        use_preprocessing: bool = True,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Create complete submission exactly like notebook.
        
        This is the main function that replicates your notebook submission workflow.
        
        Args:
            test_features_df: Test features DataFrame
            output_path: Path to save submission CSV
            submission_format_path: Path to submission format for validation
            batch_size: Batch size for predictions
            use_preprocessing: Whether to use custom preprocessing
            validate: Whether to validate submission format
            
        Returns:
            Submission DataFrame
        """
        
        logger.info("🚀 Creating submission exactly like notebook...")
        
        # 1. Create test dataloader exactly like notebook
        test_dataloader = self.create_test_dataset_and_loader(
            test_features_df, 
            batch_size=batch_size,
            use_preprocessing=use_preprocessing
        )
        
        # 2. Generate predictions exactly like notebook
        submission_df = self.generate_predictions(test_dataloader)
        
        # 3. Validate submission if requested
        if validate and submission_format_path:
            is_valid = self.validate_submission(submission_df, submission_format_path)
            if not is_valid:
                raise ValueError("Submission validation failed!")
        
        # 4. Save submission exactly like notebook
        submission_df.to_csv(output_path)
        logger.info(f"✅ Submission saved to {output_path}")
        
        # 5. Show submission summary exactly like notebook
        print(f"Submission shape: {submission_df.shape}")
        print(f"Sample predictions:")
        print(submission_df.head())
        
        return submission_df


def create_notebook_submission(
    model: torch.nn.Module,
    test_features_df: pd.DataFrame,
    species_labels: List[str],
    device: torch.device,
    output_path: str,
    submission_format_path: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Create submission exactly like notebook workflow.
    
    This is the main convenience function that replicates your entire
    notebook submission process.
    
    Args:
        model: Trained model
        test_features_df: Test features DataFrame (from test_features.csv)
        species_labels: List of species names in correct order
        device: Device for inference
        output_path: Path to save submission
        submission_format_path: Path to submission format for validation
        **kwargs: Additional arguments for predictor
        
    Returns:
        Submission DataFrame
    """
    
    predictor = NotebookStylePredictor(model, device, species_labels)
    
    return predictor.create_submission(
        test_features_df=test_features_df,
        output_path=output_path,
        submission_format_path=submission_format_path,
        **kwargs
    )


# Convenience function for loading model and creating submission
def load_model_and_create_submission(
    model_path: str,
    model_class: torch.nn.Module,
    test_features_path: str,
    species_labels: List[str],
    output_path: str,
    submission_format_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> pd.DataFrame:
    """
    Load trained model and create submission in one step.
    
    Args:
        model_path: Path to saved model checkpoint
        model_class: Model class to instantiate
        test_features_path: Path to test_features.csv
        species_labels: List of species names
        output_path: Path to save submission
        submission_format_path: Path to submission format
        device: Device for inference
        
    Returns:
        Submission DataFrame
    """
    
    # Auto-detect device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = model_class
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Load test features
    test_features_df = pd.read_csv(test_features_path, index_col="id")
    
    # Create submission
    return create_notebook_submission(
        model=model,
        test_features_df=test_features_df,
        species_labels=species_labels,
        device=device,
        output_path=output_path,
        submission_format_path=submission_format_path
    )


# Example usage
if __name__ == "__main__":
    
    print("Submission predictor ready!")
    print("Use create_notebook_submission() to replicate notebook submission exactly.")