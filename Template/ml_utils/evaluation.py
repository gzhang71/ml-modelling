"""Hold-out evaluation, error analysis and feature importance."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.inspection import permutation_importance

from .preprocessing import CLASSIFICATION, REGRESSION


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series, task: str) -> dict[str, float]:
    """Compute the standard metric set for a fitted model on held-out data."""
    y_pred = model.predict(X_test)
    if task == REGRESSION:
        return {
            "rmse": float(np.sqrt(metrics.mean_squared_error(y_test, y_pred))),
            "mae": float(metrics.mean_absolute_error(y_test, y_pred)),
            "r2": float(metrics.r2_score(y_test, y_pred)),
            "mape": float(metrics.mean_absolute_percentage_error(y_test, y_pred)),
        }
    if task != CLASSIFICATION:
        raise ValueError(f"Unknown task: {task!r}")

    result = {
        "accuracy": float(metrics.accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(metrics.balanced_accuracy_score(y_test, y_pred)),
        "f1_macro": float(metrics.f1_score(y_test, y_pred, average="macro")),
        "precision_macro": float(metrics.precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(metrics.recall_score(y_test, y_pred, average="macro", zero_division=0)),
    }
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        classes = model.classes_
        if len(classes) == 2:
            pos = proba[:, 1]
            y_bin = (np.asarray(y_test) == classes[1]).astype(int)
            result["roc_auc"] = float(metrics.roc_auc_score(y_bin, pos))
            result["average_precision"] = float(metrics.average_precision_score(y_bin, pos))
            result["log_loss"] = float(metrics.log_loss(y_bin, pos))
        elif set(np.unique(y_test)) <= set(classes):
            result["roc_auc_ovr"] = float(metrics.roc_auc_score(y_test, proba, multi_class="ovr", labels=classes))
            result["log_loss"] = float(metrics.log_loss(y_test, proba, labels=classes))
    return result


def classification_details(model, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-class report and confusion matrix as DataFrames."""
    y_pred = model.predict(X_test)
    report = pd.DataFrame(metrics.classification_report(y_test, y_pred, output_dict=True, zero_division=0)).T
    labels = np.unique(np.concatenate([np.asarray(y_test), y_pred]))
    cm = pd.DataFrame(
        metrics.confusion_matrix(y_test, y_pred, labels=labels),
        index=pd.Index(labels, name="actual"),
        columns=pd.Index(labels, name="predicted"),
    )
    return report, cm


def threshold_analysis(model, X_test: pd.DataFrame, y_test: pd.Series,
                       thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """Precision/recall/F1 across decision thresholds for a binary classifier."""
    thresholds = np.linspace(0.05, 0.95, 19) if thresholds is None else thresholds
    pos_label = model.classes_[1]
    proba = model.predict_proba(X_test)[:, 1]
    y_true = (np.asarray(y_test) == pos_label).astype(int)
    rows = []
    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        rows.append({
            "threshold": round(float(t), 3),
            "precision": metrics.precision_score(y_true, y_pred, zero_division=0),
            "recall": metrics.recall_score(y_true, y_pred, zero_division=0),
            "f1": metrics.f1_score(y_true, y_pred, zero_division=0),
            "positive_rate": y_pred.mean(),
        })
    return pd.DataFrame(rows)


def residual_summary(model, X_test: pd.DataFrame, y_test: pd.Series, n_bins: int = 5) -> pd.DataFrame:
    """Regression error broken down by quantile of the true target, to spot systematic bias."""
    y_pred = model.predict(X_test)
    df = pd.DataFrame({"y_true": np.asarray(y_test), "y_pred": y_pred})
    df["residual"] = df["y_true"] - df["y_pred"]
    df["bin"] = pd.qcut(df["y_true"], q=n_bins, duplicates="drop")
    return df.groupby("bin", observed=True).agg(
        n=("residual", "size"),
        mean_residual=("residual", "mean"),
        mae=("residual", lambda r: r.abs().mean()),
    )


def feature_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str | None = None,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Permutation importance on the *raw* input columns (works for any fitted pipeline)."""
    result = permutation_importance(
        model, X, y, scoring=scoring, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )
    return (
        pd.DataFrame({
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        })
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def compare_runs(runs: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Side-by-side table of metric dicts, e.g. {'v1 baseline': {...}, 'v2 tuned': {...}}.

    Adds a delta column for each run relative to the first one.
    """
    table = pd.DataFrame(runs)
    first = table.columns[0]
    for col in table.columns[1:]:
        table[f"Δ {col} vs {first}"] = table[col] - table[first]
    return table


def plot_confusion_matrix(model, X_test: pd.DataFrame, y_test: pd.Series, normalize: str | None = "true"):
    """Requires matplotlib."""
    from sklearn.metrics import ConfusionMatrixDisplay

    return ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, normalize=normalize).figure_


def plot_roc_pr(model, X_test: pd.DataFrame, y_test: pd.Series):
    """ROC and precision-recall curves side by side (binary). Requires matplotlib."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax1)
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax2)
    fig.tight_layout()
    return fig


def plot_predictions(model, X_test: pd.DataFrame, y_test: pd.Series):
    """Predicted vs actual and residuals vs predicted (regression). Requires matplotlib."""
    import matplotlib.pyplot as plt

    y_pred = model.predict(X_test)
    y_true = np.asarray(y_test)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.scatter(y_true, y_pred, s=8, alpha=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax1.plot(lims, lims, "k--", lw=1)
    ax1.set(xlabel="actual", ylabel="predicted", title="Predicted vs actual")
    ax2.scatter(y_pred, y_true - y_pred, s=8, alpha=0.5)
    ax2.axhline(0, color="k", lw=1, ls="--")
    ax2.set(xlabel="predicted", ylabel="residual", title="Residuals")
    fig.tight_layout()
    return fig
