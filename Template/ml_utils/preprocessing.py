"""Cleaning, feature typing, train/test split and the sklearn preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

CLASSIFICATION = "classification"
REGRESSION = "regression"


@dataclass
class FeatureTypes:
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)

    @property
    def all(self) -> list[str]:
        return self.numeric + self.categorical


def infer_task(y: pd.Series, max_classes: int = 20) -> str:
    """Guess 'classification' vs 'regression' from the target column."""
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return CLASSIFICATION
    if pd.api.types.is_integer_dtype(y) and y.nunique() <= max_classes:
        return CLASSIFICATION
    return REGRESSION


def clean(
    df: pd.DataFrame,
    target: str | None = None,
    max_missing_pct: float = 60.0,
    drop_duplicates: bool = True,
    drop_constant: bool = True,
    drop_id_like: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Drop rows with missing target, duplicates, mostly-empty, constant and ID-like columns."""
    df = df.copy()
    dropped: dict[str, list[str]] = {}

    if target is not None:
        df = df[df[target].notna()]
    if drop_duplicates:
        df = df.drop_duplicates()

    features = [c for c in df.columns if c != target]
    missing = [c for c in features if df[c].isna().mean() * 100 > max_missing_pct]
    dropped["mostly_missing"] = missing
    if drop_constant:
        dropped["constant"] = [c for c in features if c not in missing and df[c].nunique(dropna=True) <= 1]
    if drop_id_like:
        # Every value unique and non-float: almost certainly an identifier, not a feature.
        dropped["id_like"] = [
            c for c in features
            if c not in missing
            and not pd.api.types.is_float_dtype(df[c])
            and df[c].nunique(dropna=True) == len(df)
            and len(df) > 1
        ]

    to_drop = sorted({c for cols in dropped.values() for c in cols})
    if verbose:
        for reason, cols in dropped.items():
            if cols:
                print(f"Dropping {reason}: {cols}")
    return df.drop(columns=to_drop).reset_index(drop=True)


def infer_feature_types(
    df: pd.DataFrame,
    target: str | None = None,
    max_categorical_cardinality: int = 50,
) -> FeatureTypes:
    """Split columns into numeric and categorical; high-cardinality text columns are skipped."""
    types = FeatureTypes()
    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            types.categorical.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            types.numeric.append(col)
        elif s.nunique(dropna=True) <= max_categorical_cardinality:
            types.categorical.append(col)
    return types


def split(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train/test split. Stratifies by default for classification targets."""
    X, y = df.drop(columns=[target]), df[target]
    if stratify is None:
        stratify = infer_task(y) == CLASSIFICATION
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if stratify else None
    )


def build_preprocessor(
    feature_types: FeatureTypes,
    scale_numeric: bool = True,
    numeric_impute: str = "median",
    min_category_frequency: int | float | None = None,
) -> ColumnTransformer:
    """Impute + scale numeric columns, impute + one-hot categorical columns."""
    numeric_steps = [("impute", SimpleImputer(strategy=numeric_impute, add_indicator=True))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    categorical = Pipeline([
        # Casting to str makes mixed/bool columns safe and turns NaN into its own category.
        ("to_str", FunctionTransformer(_as_string_categories, feature_names_out="one-to-one")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist" if min_category_frequency else "ignore",
            min_frequency=min_category_frequency,
            sparse_output=False,
        )),
    ])

    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), feature_types.numeric),
            ("cat", categorical, feature_types.categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _as_string_categories(X):
    return pd.DataFrame(X).astype(object).where(pd.notna(X), "__missing__").astype(str)
