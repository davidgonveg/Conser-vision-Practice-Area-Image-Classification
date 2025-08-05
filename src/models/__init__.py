"""
Model architectures for wildlife classification.
"""

# Solo importar lo que realmente existe
from .model import (
    WildlifeClassifier,
    create_notebook_model
)

__all__ = [
    'WildlifeClassifier',
    'create_notebook_model'
]
