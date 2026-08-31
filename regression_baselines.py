"""Compare five conventional regressions with model.py on the same split."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import model


ROOT = Path(__file__).resolve().parent


def metrics(y_true, predictions):
    predictions = np.clip(predictions, 0.0, 1.0)
    return {
        "R2": r2_score(y_true, predictions),
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_true, predictions)),
    }


def main():
    _, X, y = model.load_training(model.TRAINING_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=model.TEST_SIZE, random_state=model.RANDOM_STATE
    )
    candidates = {
        "Ordinary least squares": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge (alpha=1.0)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Lasso (alpha=0.001)": make_pipeline(
            StandardScaler(), Lasso(alpha=0.001, max_iter=20000)
        ),
        "Elastic Net (alpha=0.001, l1_ratio=0.15)": make_pipeline(
            StandardScaler(), ElasticNet(alpha=0.001, l1_ratio=0.15, max_iter=20000)
        ),
        "Huber robust regression": make_pipeline(
            StandardScaler(), HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=2000)
        ),
    }
    rows = []
    for name, candidate in candidates.items():
        candidate.fit(X_train, y_train)
        for dataset, features, target in (
            ("Training", X_train, y_train),
            ("Testing", X_test, y_test),
        ):
            rows.append(
                {"Model": name, "Dataset": dataset, "Rows": len(target), **metrics(target, candidate.predict(features))}
            )
        candidate.fit(X, y)
        for dataset, path in model.EXTERNAL_TEST_SETS.items():
            _, features, target = model.load_external(path)
            rows.append(
                {"Model": name, "Dataset": dataset, "Rows": len(target), **metrics(target, candidate.predict(features))}
            )
    results = pd.DataFrame(rows)
    results.to_csv(ROOT / "regression_baseline_metrics.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
