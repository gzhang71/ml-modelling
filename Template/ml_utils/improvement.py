"""Model improvement: hyperparameter search, feature selection and simple feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV

from .modeling import default_scoring, get_cv, make_pipeline


def default_param_space(model: BaseEstimator) -> dict:
    """Reasonable random-search space for the models returned by ``candidate_models``.

    Keys are prefixed with ``model__`` so they apply to the pipeline's model step.
    """
    name = type(model).__name__
    spaces = {
        "LogisticRegression": {
            "C": loguniform(1e-3, 1e2),
            "class_weight": [None, "balanced"],
        },
        "Ridge": {"alpha": loguniform(1e-3, 1e3)},
        "RandomForestClassifier": {
            "n_estimators": randint(200, 800),
            "max_depth": [None, 5, 10, 20, 30],
            "min_samples_leaf": randint(1, 20),
            "max_features": ["sqrt", "log2", 0.3, 0.5, 1.0],
            "class_weight": [None, "balanced", "balanced_subsample"],
        },
        "RandomForestRegressor": {
            "n_estimators": randint(200, 800),
            "max_depth": [None, 5, 10, 20, 30],
            "min_samples_leaf": randint(1, 20),
            "max_features": ["sqrt", "log2", 0.3, 0.5, 1.0],
        },
        "HistGradientBoostingClassifier": {
            "learning_rate": loguniform(0.01, 0.3),
            "max_iter": randint(100, 1000),
            "max_leaf_nodes": randint(15, 128),
            "min_samples_leaf": randint(5, 100),
            "l2_regularization": loguniform(1e-4, 10),
            "class_weight": [None, "balanced"],
        },
        "HistGradientBoostingRegressor": {
            "learning_rate": loguniform(0.01, 0.3),
            "max_iter": randint(100, 1000),
            "max_leaf_nodes": randint(15, 128),
            "min_samples_leaf": randint(5, 100),
            "l2_regularization": loguniform(1e-4, 10),
        },
    }
    if name not in spaces:
        raise ValueError(f"No default search space for {name}; pass param_distributions explicitly.")
    return {f"model__{k}": v for k, v in spaces[name].items()}


def tune_model(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    model: BaseEstimator,
    task: str,
    param_distributions: dict | None = None,
    scoring: str | None = None,
    n_iter: int = 30,
    cv: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 0,
) -> RandomizedSearchCV:
    """Randomized search over the full pipeline; the result is refit on all of X, y.

    Use ``.best_estimator_`` as the improved model and ``.best_params_`` / ``.best_score_``
    to report what changed. Preprocessing params can be tuned too (``preprocess__...``).
    """
    search = RandomizedSearchCV(
        make_pipeline(preprocessor, model),
        param_distributions=param_distributions or default_param_space(model),
        n_iter=n_iter,
        scoring=scoring or default_scoring(task, y),
        cv=get_cv(task, cv, random_state),
        random_state=random_state,
        n_jobs=n_jobs,
        refit=True,
        verbose=verbose,
        error_score=np.nan,
    )
    return search.fit(X, y)


def search_results(search: RandomizedSearchCV, top: int = 10) -> pd.DataFrame:
    """Top-N parameter combinations from a fitted search."""
    res = pd.DataFrame(search.cv_results_)
    cols = ["rank_test_score", "mean_test_score", "std_test_score", "mean_fit_time"]
    params = [c for c in res.columns if c.startswith("param_")]
    return res.sort_values("rank_test_score")[cols + params].head(top).reset_index(drop=True)


def select_features(
    importance: pd.DataFrame,
    min_importance: float = 0.0,
    top_k: int | None = None,
) -> list[str]:
    """Keep features from ``evaluation.feature_importance`` output above a threshold / in top-k.

    Permutation importance <= 0 means shuffling the feature did not hurt: likely noise.
    """
    kept = importance[importance["importance_mean"] > min_importance]
    if top_k is not None:
        kept = kept.head(top_k)
    return kept["feature"].tolist()


def add_log_features(df: pd.DataFrame, columns: list[str] | None = None, skew_threshold: float = 1.0,
                     exclude: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Add ``log1p`` versions of heavily right-skewed, non-negative numeric columns.

    Pass the returned column list back in as ``columns`` when transforming test data,
    so train and test get identical features.
    """
    df = df.copy()
    exclude = set(exclude or [])
    if columns is None:
        numeric = df.select_dtypes(include="number").drop(columns=list(exclude & set(df.columns)))
        columns = [c for c in numeric.columns if numeric[c].min() >= 0 and numeric[c].skew() > skew_threshold]
    for c in columns:
        df[f"log_{c}"] = np.log1p(df[c].clip(lower=0))
    return df, columns


def add_datetime_features(df: pd.DataFrame, columns: list[str], drop_original: bool = True) -> pd.DataFrame:
    """Expand datetime columns into year / month / day-of-week / hour parts."""
    df = df.copy()
    for c in columns:
        dt = pd.to_datetime(df[c], errors="coerce")
        df[f"{c}_year"] = dt.dt.year
        df[f"{c}_month"] = dt.dt.month
        df[f"{c}_dayofweek"] = dt.dt.dayofweek
        if (dt.dt.hour.fillna(0) != 0).any():
            df[f"{c}_hour"] = dt.dt.hour
    return df.drop(columns=columns) if drop_original else df
