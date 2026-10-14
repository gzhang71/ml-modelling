"""Initial data exploration helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def overview(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, missingness, cardinality and an example value."""
    n = len(df)
    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "n_missing": df.isna().sum(),
            "pct_missing": (df.isna().mean() * 100).round(2),
            "n_unique": df.nunique(dropna=True),
            "pct_unique": (df.nunique(dropna=True) / max(n, 1) * 100).round(2),
            "example": [df[c].dropna().iloc[0] if df[c].notna().any() else None for c in df.columns],
        }
    )
    return summary.sort_values("pct_missing", ascending=False)


def print_report(df: pd.DataFrame, target: str | None = None) -> None:
    """Print a quick text report: shape, duplicates, column overview, target summary."""
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Duplicate rows: {df.duplicated().sum():,}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB\n")
    with pd.option_context("display.max_rows", 200, "display.width", 160):
        print(overview(df), "\n")
        numeric = df.select_dtypes(include="number")
        if not numeric.empty:
            print("Numeric summary:")
            print(numeric.describe().T, "\n")
        if target is not None:
            print(f"Target '{target}':")
            print(target_summary(df, target), "\n")
            corr = target_correlations(df, target)
            if not corr.empty:
                print("Top numeric correlations with target:")
                print(corr.head(15), "\n")


def target_summary(df: pd.DataFrame, target: str) -> pd.DataFrame | pd.Series:
    """Class balance for categorical-like targets, distribution stats for continuous ones."""
    y = df[target]
    if _looks_categorical(y):
        counts = y.value_counts(dropna=False)
        return pd.DataFrame({"count": counts, "pct": (counts / counts.sum() * 100).round(2)})
    return y.describe()


def target_correlations(df: pd.DataFrame, target: str, method: str = "spearman") -> pd.Series:
    """Correlation of each numeric feature with the target (target label-encoded if needed)."""
    y = df[target]
    if not pd.api.types.is_numeric_dtype(y):
        y = pd.Series(pd.factorize(y)[0], index=y.index).where(y.notna())
    numeric = df.drop(columns=[target]).select_dtypes(include="number")
    numeric = numeric.loc[:, numeric.nunique() > 1]  # correlation is undefined for constants
    if numeric.empty:
        return pd.Series(dtype=float)
    corr = numeric.corrwith(y, method=method)
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


def high_correlation_pairs(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Pairs of numeric features whose absolute correlation exceeds ``threshold``."""
    corr = df.select_dtypes(include="number").corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().reset_index()
    pairs.columns = ["feature_a", "feature_b", "abs_corr"]
    return pairs[pairs["abs_corr"] >= threshold].sort_values("abs_corr", ascending=False).reset_index(drop=True)


def outlier_summary(df: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    """Count values outside the [Q1 - k*IQR, Q3 + k*IQR] range for each numeric column."""
    rows = []
    for col in df.select_dtypes(include="number").columns:
        s = df[col].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        rows.append({"column": col, "lower": lo, "upper": hi, "n_outliers": n_out,
                     "pct_outliers": round(n_out / max(len(s), 1) * 100, 2)})
    return pd.DataFrame(rows).sort_values("n_outliers", ascending=False).reset_index(drop=True)


def plot_distributions(df: pd.DataFrame, columns: list[str] | None = None, bins: int = 30, ncols: int = 3):
    """Histogram per numeric column / bar chart per categorical column. Requires matplotlib."""
    import matplotlib.pyplot as plt

    columns = columns or list(df.columns)
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    for ax, col in zip(axes.flat, columns):
        s = df[col].dropna()
        if pd.api.types.is_numeric_dtype(s) and not _looks_categorical(s):
            ax.hist(s, bins=bins)
        else:
            s.astype(str).value_counts().head(20).plot.bar(ax=ax)
        ax.set_title(col)
    for ax in list(axes.flat)[len(columns):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_time_series(
    df: pd.DataFrame,
    time_col: str | None = None,
    columns: list[str] | None = None,
    freq: str | None = None,
    rolling: int | None = None,
):
    """Line chart per numeric column over time, to eyeball trend and seasonality. Requires matplotlib.

    ``time_col`` is parsed as datetime; if omitted the index is used. ``freq`` (e.g. "D", "W", "MS")
    resamples each series to its mean, which smooths noise so seasonal cycles stand out. ``rolling``
    overlays a centred rolling mean over that many periods to separate trend from the seasonal swing.
    """
    import matplotlib.pyplot as plt

    data = df.set_index(pd.to_datetime(df[time_col])).drop(columns=[time_col]) if time_col else df.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    columns = columns or [c for c in data.select_dtypes(include="number").columns
                          if not pd.api.types.is_bool_dtype(data[c])]
    data = data[columns]
    if freq is not None:
        data = data.resample(freq).mean()

    fig, axes = plt.subplots(len(columns), 1, figsize=(14, 3 * len(columns)), sharex=True, squeeze=False)
    for ax, col in zip(axes.flat, columns):
        s = data[col].dropna()
        ax.plot(s.index, s.values, linewidth=1, label=col)
        if rolling is not None:
            trend = s.rolling(rolling, center=True, min_periods=1).mean()
            ax.plot(trend.index, trend.values, linewidth=2, label=f"rolling mean ({rolling})")
            ax.legend(loc="upper left")
        ax.set_title(col)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _looks_categorical(s: pd.Series, max_unique: int = 20) -> bool:
    return (
        not pd.api.types.is_numeric_dtype(s)
        or pd.api.types.is_bool_dtype(s)
        or (pd.api.types.is_integer_dtype(s) and s.nunique() <= max_unique)
    )
