"""End-to-end walkthrough of ml_utils on a synthetic mixed-type classification dataset.

Run:  python example.py
Swap ``load_data()`` for your own DataFrame and set TARGET.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_utils import eda, evaluation, improvement, modeling, preprocessing as prep

TARGET = "churned"


def load_data(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customer_id": np.arange(n),
        "tenure_months": rng.integers(1, 72, n),
        "monthly_spend": rng.lognormal(3.5, 0.6, n).round(2),
        "support_tickets": rng.poisson(1.5, n),
        "plan": rng.choice(["basic", "plus", "pro"], n, p=[0.5, 0.3, 0.2]),
        "region": rng.choice(["north", "south", "east", "west", None], n),
        "noise": rng.normal(size=n),
        "constant": 1,
    })
    logit = -1.0 - 0.04 * df.tenure_months + 0.5 * df.support_tickets + 0.6 * (df.plan == "basic")
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df.loc[rng.choice(n, 150, replace=False), "monthly_spend"] = np.nan
    return df


def main() -> None:
    # 1. Initial data exploration and preprocessing
    df = load_data()
    eda.print_report(df, target=TARGET)
    print(eda.outlier_summary(df.drop(columns=[TARGET])), "\n")

    df = prep.clean(df, target=TARGET)
    task = prep.infer_task(df[TARGET])
    X_train, X_test, y_train, y_test = prep.split(df, TARGET)
    ftypes = prep.infer_feature_types(X_train)
    preprocessor = prep.build_preprocessor(ftypes)
    print(f"\nTask: {task} | numeric: {ftypes.numeric} | categorical: {ftypes.categorical}\n")

    # 2. Model development: compare candidates with CV on the training set only
    comparison = modeling.compare_models(X_train, y_train, preprocessor, task)
    print(comparison.round(4), "\n")
    best_name = comparison.loc[comparison["model"] != "baseline", "model"].iloc[0]
    best_model = modeling.candidate_models(task)[best_name]
    v1 = modeling.fit_model(X_train, y_train, preprocessor, best_model)

    # 3. Model evaluation on the held-out test set
    runs = {f"v1 {best_name}": evaluation.evaluate(v1, X_test, y_test, task)}
    report, cm = evaluation.classification_details(v1, X_test, y_test)
    print(report.round(3), "\n", cm, "\n")
    importance = evaluation.feature_importance(v1, X_test, y_test, scoring=modeling.default_scoring(task, y_train))
    print(importance.round(4), "\n")

    # 4a. Improvement #1: hyperparameter tuning
    search = improvement.tune_model(X_train, y_train, preprocessor, best_model, task, n_iter=20)
    print(improvement.search_results(search, top=5), "\n")
    v2 = search.best_estimator_
    runs["v2 tuned"] = evaluation.evaluate(v2, X_test, y_test, task)

    # 4b. Improvement #2: drop features that don't help. Importance is measured on a
    # validation slice of the training data so the test set stays untouched.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.25, stratify=y_train, random_state=0
    )
    val_model = modeling.fit_model(X_fit, y_fit, preprocessor, v2.named_steps["model"])
    val_importance = evaluation.feature_importance(
        val_model, X_val, y_val, scoring=modeling.default_scoring(task, y_train)
    )
    keep = improvement.select_features(val_importance)
    print(f"Keeping features: {keep}\n")
    ftypes_v3 = prep.FeatureTypes(
        numeric=[c for c in ftypes.numeric if c in keep],
        categorical=[c for c in ftypes.categorical if c in keep],
    )
    v3_model = v2.named_steps["model"]
    v3 = modeling.fit_model(X_train, y_train, prep.build_preprocessor(ftypes_v3), v3_model)
    runs["v3 tuned + selected"] = evaluation.evaluate(v3, X_test, y_test, task)

    print(evaluation.compare_runs(runs).round(4))


if __name__ == "__main__":
    main()
