"""Reusable helpers for a tabular ML workflow.

    eda            -> initial data exploration
    preprocessing  -> cleaning, feature typing, splitting, ColumnTransformer
    modeling       -> candidate models, pipelines, cross-validated comparison
    evaluation     -> hold-out metrics, error analysis, feature importance, run comparison
    improvement    -> hyperparameter search, feature selection, feature engineering
"""

from . import eda, evaluation, improvement, modeling, preprocessing

__all__ = ["eda", "preprocessing", "modeling", "evaluation", "improvement"]
