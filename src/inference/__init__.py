"""
Inference and prediction utilities.
"""

try:
    from .predictor import create_notebook_submission
    __all__ = ['create_notebook_submission']
except ImportError:
    __all__ = []
