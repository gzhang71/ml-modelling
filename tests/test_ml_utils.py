import numpy as np
import pandas as pd
import pytest

from ml_utils import eda, evaluation, improvement, modeling, preprocessing as prep


def make_df(task: str, n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "id": np.arange(n),
        "x1": rng.normal(size=n),
        "x2": rng.exponential(size=n),
        "flag": rng.random(n) > 0.5,
        "cat": rng.choice(["a", "b", "c", None], n),
        "mixed": rng.choice([1, "one", None], n),
        "const": "same",
        "mostly_nan": np.where(rng.random(n) < 0.9, np.nan, 1.0),
    })
    signal = 2 * df.x1 + df.flag + (df.cat == "a")
    if task == "regression":
        df["y"] = signal + rng.normal(scale=0.3, size=n)
    elif task == "binary_str":
        df["y"] = np.where(signal > signal.median(), "yes", "no")
    else:
        df["y"] = pd.qcut(signal, 3, labels=False)
    return df


@pytest.mark.parametrize("kind,expected_task", [
    ("regression", "regression"), ("binary_str", "classification"), ("multiclass", "classification"),
])
def test_full_workflow(kind, expected_task):
    df = make_df(kind)
    eda.overview(df)
    eda.target_summary(df, "y")
    eda.target_correlations(df, "y")
    eda.high_correlation_pairs(df)
    eda.outlier_summary(df)

    df = prep.clean(df, target="y", verbose=False)
    assert not {"id", "const", "mostly_nan"} & set(df.columns)
    task = prep.infer_task(df["y"])
    assert task == expected_task

    X_train, X_test, y_train, y_test = prep.split(df, "y")
    ftypes = prep.infer_feature_types(X_train)
    pre = prep.build_preprocessor(ftypes)

    comparison = modeling.compare_models(X_train, y_train, pre, task, cv=3)
    assert comparison.drop(columns="model").notna().all().all()
    assert comparison.iloc[0]["model"] != "baseline"

    model = modeling.candidate_models(task)["gradient_boosting"]
    fitted = modeling.fit_model(X_train, y_train, pre, model)
    metrics = evaluation.evaluate(fitted, X_test, y_test, task)
    assert all(np.isfinite(v) for v in metrics.values())

    imp = evaluation.feature_importance(fitted, X_test, y_test, n_repeats=2)
    assert imp.iloc[0]["feature"] == "x1"
    assert "x1" in improvement.select_features(imp)

    search = improvement.tune_model(X_train, y_train, pre, model, task, n_iter=3, cv=3)
    assert np.isfinite(search.best_score_)
    improvement.search_results(search)

    table = evaluation.compare_runs({
        "v1": metrics, "v2": evaluation.evaluate(search.best_estimator_, X_test, y_test, task),
    })
    assert table.shape[1] == 3

    if task == "regression":
        evaluation.residual_summary(fitted, X_test, y_test)
    else:
        evaluation.classification_details(fitted, X_test, y_test)
        if y_train.nunique() == 2:
            evaluation.threshold_analysis(fitted, X_test, y_test)


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest", "gradient_boosting"])
def test_default_param_spaces_are_valid(name):
    df = prep.clean(make_df("multiclass"), target="y", verbose=False)
    X_train, _, y_train, _ = prep.split(df, "y")
    pre = prep.build_preprocessor(prep.infer_feature_types(X_train))
    model = modeling.candidate_models("classification")[name]
    search = improvement.tune_model(X_train, y_train, pre, model, "classification", n_iter=2, cv=2)
    assert np.isfinite(search.best_score_)


def test_feature_engineering_helpers():
    df = pd.DataFrame({"skewed": np.r_[np.ones(50), [100.0]], "when": pd.date_range("2024-01-01", periods=51, freq="h")})
    out, cols = improvement.add_log_features(df)
    assert cols == ["skewed"] and "log_skewed" in out
    out = improvement.add_datetime_features(df, ["when"])
    assert {"when_year", "when_month", "when_dayofweek", "when_hour"} <= set(out.columns)
    assert "when" not in out
