"""
Taï National Park - Evaluation Metrics

This module provides comprehensive evaluation metrics for camera trap species classification.
Includes log loss, accuracy, precision, recall, F1-score, and confusion matrix analysis.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    log_loss, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from typing import Dict, List, Optional, Union, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Comprehensive metrics calculator for wildlife classification.
    """
    
    def __init__(self, num_classes: int = 8):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes: Number of classification classes
        """
        self.num_classes = num_classes
        self.class_names = [
            'antelope_duiker', 'bird', 'blank', 'civet_genet',
            'hog', 'leopard', 'monkey_prosimian', 'rodent'
        ]
    
    def calculate_log_loss(
        self, 
        y_true: Union[List, np.ndarray], 
        y_proba: Union[List, np.ndarray]
    ) -> float:
        """
        Calculate log loss (primary competition metric).
        
        Args:
            y_true: True class labels
            y_proba: Predicted probabilities
            
        Returns:
            Log loss value
        """
        
        y_true = np.array(y_true)
        y_proba = np.array(y_proba)
        
        # Ensure probabilities are valid
        y_proba = np.clip(y_proba, 1e-15, 1 - 1e-15)
        
        # Calculate log loss
        return log_loss(y_true, y_proba)
    
    def calculate_accuracy(
        self, 
        y_true: Union[List, np.ndarray], 
        y_pred: Union[List, np.ndarray]
    ) -> float:
        """
        Calculate overall accuracy.
        
        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            
        Returns:
            Accuracy score
        """
        
        return accuracy_score(y_true, y_pred)
    
    def calculate_class_wise_metrics(
        self, 
        y_true: Union[List, np.ndarray], 
        y_pred: Union[List, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate precision, recall, and F1-score for each class.
        
        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            
        Returns:
            Dictionary with class-wise metrics
        """
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate metrics for each class
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Organize by class
        class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            class_metrics[class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i])
            }
        
        return class_metrics
    
    def calculate_macro_metrics(
        self, 
        y_true: Union[List, np.ndarray], 
        y_pred: Union[List, np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate macro-averaged metrics.
        
        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            
        Returns:
            Dictionary with macro-averaged metrics
        """
        
        return {
            'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0)
        }
    
    def calculate_weighted_metrics(
        self, 
        y_true: Union[List, np.ndarray], 
        y_pred: Union[List, np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate weighted-averaged metrics.
        
        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            
        Returns:
            Dictionary with weighted-averaged metrics
        """
        
        return {
            'weighted_precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'weighted_recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    
    def calculate_confusion_matrix(
        self, 
        y_true: Union[List, np.ndarray], 
        y_pred: Union[List, np.ndarray]
    ) -> np.ndarray:
        """
        Calculate confusion matrix.
        
        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            
        Returns:
            Confusion matrix
        """
        
        return confusion_matrix(y_true, y_pred)
    
    def calculate_class_distribution(
        self, 
        y_true: Union[List, np.ndarray]
    ) -> Dict[str, int]:
        """
        Calculate class distribution in the dataset.
        
        Args:
            y_true: True class labels
            
        Returns:
            Dictionary with class counts
        """
        
        y_true = np.array(y_true)
        unique, counts = np.unique(y_true, return_counts=True)
        
        distribution = {}
        for i, class_name in enumerate(self.class_names):
            if i in unique:
                distribution[class_name] = int(counts[np.where(unique == i)[0][0]])
            else:
                distribution[class_name] = 0
        
        return distribution
    
    def calculate_top_k_accuracy(
        self, 
        y_true: Union[List, np.ndarray], 
        y_proba: Union[List, np.ndarray],
        k: int = 3
    ) -> float:
        """
        Calculate top-k accuracy.
        
        Args:
            y_true: True class labels
            y_proba: Predicted probabilities
            k: Number of top predictions to consider
            
        Returns:
            Top-k accuracy
        """
        
        y_true = np.array(y_true)
        y_proba = np.array(y_proba)
        
        # Get top k predictions
        top_k_pred = np.argsort(y_proba, axis=1)[:, -k:]
        
        # Check if true label is in top k
        correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top_k_pred[i]:
                correct += 1
        
        return correct / len(y_true)
    
    def calculate_class_imbalance_metrics(
        self, 
        y_true: Union[List, np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate metrics related to class imbalance.
        
        Args:
            y_true: True class labels
            
        Returns:
            Dictionary with imbalance metrics
        """
        
        distribution = self.calculate_class_distribution(y_true)
        counts = list(distribution.values())
        
        if len(counts) == 0:
            return {}
        
        max_count = max(counts)
        min_count = min([c for c in counts if c > 0])
        
        return {
            'imbalance_ratio': max_count / min_count if min_count > 0 else float('inf'),
            'gini_coefficient': self._calculate_gini_coefficient(counts),
            'entropy': self._calculate_entropy(counts)
        }
    
    def _calculate_gini_coefficient(self, counts: List[int]) -> float:
        """Calculate Gini coefficient for class distribution."""
        
        if len(counts) == 0:
            return 0.0
        
        counts = np.array(counts)
        n = len(counts)
        
        if n == 1:
            return 0.0
        
        # Sort counts
        sorted_counts = np.sort(counts)
        
        # Calculate Gini coefficient
        cumsum = np.cumsum(sorted_counts)
        gini = (2 * np.sum((np.arange(n) + 1) * sorted_counts)) / (n * cumsum[-1]) - (n + 1) / n
        
        return gini
    
    def _calculate_entropy(self, counts: List[int]) -> float:
        """Calculate entropy of class distribution."""
        
        counts = np.array(counts)
        total = np.sum(counts)
        
        if total == 0:
            return 0.0
        
        probabilities = counts / total
        probabilities = probabilities[probabilities > 0]  # Remove zero probabilities
        
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    def calculate_all_metrics(
        self, 
        y_true: Union[List, np.ndarray], 
        y_pred: Union[List, np.ndarray],
        y_proba: Optional[Union[List, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            y_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary with all metrics
        """
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = self.calculate_accuracy(y_true, y_pred)
        metrics['class_wise_metrics'] = self.calculate_class_wise_metrics(y_true, y_pred)
        metrics['macro_metrics'] = self.calculate_macro_metrics(y_true, y_pred)
        metrics['weighted_metrics'] = self.calculate_weighted_metrics(y_true, y_pred)
        metrics['confusion_matrix'] = self.calculate_confusion_matrix(y_true, y_pred).tolist()
        metrics['class_distribution'] = self.calculate_class_distribution(y_true)
        metrics['imbalance_metrics'] = self.calculate_class_imbalance_metrics(y_true)
        
        # Probability-based metrics
        if y_proba is not None:
            metrics['log_loss'] = self.calculate_log_loss(y_true, y_proba)
            metrics['top_3_accuracy'] = self.calculate_top_k_accuracy(y_true, y_proba, k=3)
            metrics['top_2_accuracy'] = self.calculate_top_k_accuracy(y_true, y_proba, k=2)
        
        return metrics
    
    def print_classification_report(
        self, 
        y_true: Union[List, np.ndarray], 
        y_pred: Union[List, np.ndarray]
    ) -> str:
        """
        Generate detailed classification report.
        
        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            
        Returns:
            Classification report string
        """
        
        return classification_report(
            y_true, 
            y_pred, 
            target_names=self.class_names,
            zero_division=0
        )
    
    def get_worst_performing_classes(
        self, 
        y_true: Union[List, np.ndarray], 
        y_pred: Union[List, np.ndarray],
        metric: str = 'f1_score',
        n_worst: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Get the worst performing classes based on a specific metric.
        
        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            metric: Metric to use ('precision', 'recall', 'f1_score')
            n_worst: Number of worst classes to return
            
        Returns:
            List of tuples (class_name, metric_value)
        """
        
        class_metrics = self.calculate_class_wise_metrics(y_true, y_pred)
        
        # Extract the specified metric for each class
        class_scores = [(class_name, metrics[metric]) 
                       for class_name, metrics in class_metrics.items()]
        
        # Sort by metric value (ascending for worst)
        class_scores.sort(key=lambda x: x[1])
        
        return class_scores[:n_worst]
    
    def get_confusion_matrix_analysis(
        self, 
        y_true: Union[List, np.ndarray], 
        y_pred: Union[List, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Analyze confusion matrix to identify common misclassifications.
        
        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            
        Returns:
            Dictionary with confusion matrix analysis
        """
        
        cm = self.calculate_confusion_matrix(y_true, y_pred)
        
        # Find most common misclassifications
        misclassifications = []
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j and cm[i, j] > 0:
                    misclassifications.append({
                        'true_class': self.class_names[i],
                        'predicted_class': self.class_names[j],
                        'count': int(cm[i, j]),
                        'percentage': float(cm[i, j] / cm[i].sum() * 100)
                    })
        
        # Sort by count
        misclassifications.sort(key=lambda x: x['count'], reverse=True)
        
        return {
            'confusion_matrix': cm.tolist(),
            'most_common_misclassifications': misclassifications[:10],
            'per_class_accuracy': [
                float(cm[i, i] / cm[i].sum()) if cm[i].sum() > 0 else 0.0
                for i in range(len(self.class_names))
            ]
        }


# Utility functions for batch processing
def calculate_batch_metrics(
    outputs: torch.Tensor, 
    targets: torch.Tensor,
    metrics_calculator: MetricsCalculator
) -> Dict[str, float]:
    """
    Calculate metrics for a batch of predictions.
    
    Args:
        outputs: Model outputs [batch_size, num_classes]
        targets: Target labels [batch_size]
        metrics_calculator: MetricsCalculator instance
        
    Returns:
        Dictionary with batch metrics
    """
    
    # Convert to numpy
    probabilities = F.softmax(outputs, dim=1).cpu().numpy()
    predictions = torch.argmax(outputs, dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    
    # Calculate metrics
    metrics = {
        'accuracy': metrics_calculator.calculate_accuracy(targets, predictions),
        'log_loss': metrics_calculator.calculate_log_loss(targets, probabilities),
        'top_3_accuracy': metrics_calculator.calculate_top_k_accuracy(targets, probabilities, k=3)
    }
    
    return metrics


def aggregate_site_metrics(
    predictions: Dict[str, List[int]],
    targets: Dict[str, List[int]],
    probabilities: Dict[str, List[List[float]]],
    metrics_calculator: MetricsCalculator
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics aggregated by site.
    
    Args:
        predictions: Dictionary of site -> predictions
        targets: Dictionary of site -> targets
        probabilities: Dictionary of site -> probabilities
        metrics_calculator: MetricsCalculator instance
        
    Returns:
        Dictionary with site-wise metrics
    """
    
    site_metrics = {}
    
    for site in predictions.keys():
        if site in targets and site in probabilities:
            site_preds = predictions[site]
            site_targets = targets[site]
            site_probs = probabilities[site]
            
            if len(site_preds) > 0:
                site_metrics[site] = metrics_calculator.calculate_all_metrics(
                    y_true=site_targets,
                    y_pred=site_preds,
                    y_proba=site_probs
                )
    
    return site_metrics