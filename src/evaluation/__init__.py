"""
Model evaluation and metrics calculation.
"""

# Solo importar lo que existe
try:
    from .evaluator import evaluate_notebook_style
    __all__ = ['evaluate_notebook_style']
except ImportError:
    __all__ = []
