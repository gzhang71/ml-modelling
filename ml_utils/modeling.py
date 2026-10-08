"""Candidate models, pipeline construction and cross-validated model comparison."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from .preprocessing import CLASSIFICATION, REGRESSION


def default_scoring(task: str, y: pd.Series | None = None) -> str:
    """Primary metric used for model selection and tuning."""
    if task == REGRESSION:
        return "neg_root_mean_squared_error"
    if y is not None and pd.Series(y).nunique() == 2:
        return "roc_auc"
    return "f1_macro"


def default_metrics(task: str, y: pd.Series | None = None) -> list[str]:
    if task == REGRESSION:
        return ["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"]
    labels = set(pd.Series(y).dropna().unique()) if y is not None else set()
    if len(labels) == 2:
        # f1 / average_precision need a positive label of 1; fall back to f1_macro otherwise.
        if labels <= {0, 1}:
            return ["roc_auc", "accuracy", "f1", "average_precision"]
        return ["roc_auc", "accuracy", "f1_macro"]
    return ["accuracy", "f1_macro", "balanced_accuracy"]


def candidate_models(task: str, random_state: int = 42) -> dict[str, BaseEstimator]:
    """A dummy baseline plus a linear model, a random forest and gradient boosting."""
    if task == CLASSIFICATION:
        return {
            "baseline": DummyClassifier(strategy="most_frequent"),
            "logistic_regression": LogisticRegression(max_iter=2000),
            "random_forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=random_state),
            "gradient_boosting": HistGradientBoostingClassifier(random_state=random_state),
        }
    if task == REGRESSION:
        return {
            "baseline": DummyRegressor(strategy="mean"),
            "ridge": Ridge(),
            "random_forest": RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=random_state),
            "gradient_boosting": HistGradientBoostingRegressor(random_state=random_state),
        }
    raise ValueError(f"Unknown task: {task!r}")


def make_pipeline(preprocessor: ColumnTransformer, model: BaseEstimator) -> Pipeline:
    """Bundle preprocessing and model so CV/tuning never leaks test-fold statistics."""
    return Pipeline([("preprocess", clone(preprocessor)), ("model", clone(model))])


def get_cv(task: str, n_splits: int = 5, random_state: int = 42):
    cls = StratifiedKFold if task == CLASSIFICATION else KFold
    return cls(n_splits=n_splits, shuffle=True, random_state=random_state)


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    task: str,
    models: dict[str, BaseEstimator] | None = None,
    scoring: list[str] | None = None,
    cv=5,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Cross-validate each candidate; returns mean/std per metric, sorted by the first metric."""
    models = models or candidate_models(task)
    scoring = scoring or default_metrics(task, y)
    if isinstance(cv, int):
        cv = get_cv(task, cv)

    rows = []
    for name, model in models.items():
        start = time.perf_counter()
        res = cross_validate(make_pipeline(preprocessor, model), X, y, cv=cv, scoring=scoring,
                             n_jobs=n_jobs, error_score=np.nan)
        row = {"model": name}
        for metric in scoring:
            scores = res[f"test_{metric}"]
            row[f"{metric}_mean"] = np.nanmean(scores)
            row[f"{metric}_std"] = np.nanstd(scores)
        row["fit_seconds"] = round(time.perf_counter() - start, 2)
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(f"{scoring[0]}_mean", ascending=False)
        .reset_index(drop=True)
    )


def fit_model(
    X: pd.DataFrame, y: pd.Series, preprocessor: ColumnTransformer, model: BaseEstimator
) -> Pipeline:
    pipe = make_pipeline(preprocessor, model)
    return pipe.fit(X, y)
