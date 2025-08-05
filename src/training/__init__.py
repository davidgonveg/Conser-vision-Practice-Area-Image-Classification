"""
Training utilities for wildlife classification models.
"""

from .trainer import (
    NotebookStyleTrainer,
    create_notebook_trainer
)

__all__ = [
    'NotebookStyleTrainer',
    'create_notebook_trainer'
]
