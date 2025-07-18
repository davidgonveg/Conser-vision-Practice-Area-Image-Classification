"""
Taï National Park - Evaluation Module

This module provides comprehensive evaluation functionality for camera trap species classification.
"""

from .metrics import (
    MetricsCalculator, calculate_batch_metrics, aggregate_site_metrics
)

__all__ = [
    'MetricsCalculator',
    'calculate_batch_metrics', 
    'aggregate_site_metrics'
]