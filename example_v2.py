import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold, train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

SEED = 42
def load_data() -> tuple[pd.DataFrame, pd.Series]:
    # In the interview: df = pd.read_csv(path); y = df.pop("target"); return df, y
    X, y = make_regression(n_samples=5000, n_features=10, n_informative=5, noise=10, random_state=SEED)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    X["cat"] = np.random.default_rng(SEED).choice(["a", "b", "c"], len(X))
    y = y + X["f0"] * X["f1"] * 20  # a non-linear term so the GBM has something to win
    return X, pd.Series(y)

def build_pipeline(model, df: pd.DataFrame):
    num_columns = df.select_dtypes("number").columns
    str_columns = df.select_dtypes(exclude="number").columns
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_columns),
        ("str", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(drop="first", handle_unkonw="ignore")), str_columns)
    ])
    return Pipeline([('pre', pre), ('model', model)])

if __name__ == '__main__':
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=SEED)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    models = {
        'dummy': DummyRegressor(),
        'ridge': Ridge(),
        'gbm': HistGradientBoostingRegressor(),
    }
    for name, model in models.items():
        scores = cross_val_score(build_pipeline(model, X), X_train, y_train, cv=cv, scoring='r2')
        print('name: ', name, 'scores: ', scores, 'mean: ', scores.mean(), 'std: ', scores.std(), '')

    param_dist = {
        "model__learning_rate": [0.01, 0.03, 0.1, 0.3],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__min_samples_leaf": [10, 20, 50, 100],
        "model__l2_regularization": [0.0, 0.1, 1.0],
    }
    search = RandomizedSearchCV(build_pipeline(HistGradientBoostingRegressor(random_state=SEED), X), param_dist,
                                n_iter=20, scoring="neg_root_mean_squared_error", cv=cv, random_state=SEED, n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"tuned    rmse = {-search.best_score_:.3f}  {search.best_params_}")

    # 5. evaluate once on test
    pred = search.best_estimator_.predict(X_test)
    print(f"\ntest rmse = {root_mean_squared_error(y_test, pred):.3f}")
    print(f"test mae  = {mean_absolute_error(y_test, pred):.3f}")
    print(f"test r2   = {r2_score(y_test, pred):.4f}")