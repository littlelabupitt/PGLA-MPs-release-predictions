from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model import (
    AllowedTrainingBlend,
    FEATURE_COLS,
    FORMULATION_IDS,
    RANDOM_STATE,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

REVISED_TRAINING_PATH = DATA_DIR / "training_data.xlsx"

EXTERNAL_TEST_SETS = {
    "IBMECA Trial 1": DATA_DIR / "ibmeca_trial_1.xlsx",
    "IBMECA Trial 3": DATA_DIR / "ibmeca_trial_3.xlsx",
}

TARGET_COL = "Release"
EXTERNAL_TARGET_COL = "Actual"
CLIP_BOUNDS = (0.0, 1.0)


def clip_release(pred):
    return np.clip(pred, *CLIP_BOUNDS)


def load_training(path):
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    return df, df[FEATURE_COLS], df[TARGET_COL]


def load_external(path):
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=FEATURE_COLS + [EXTERNAL_TARGET_COL]).reset_index(drop=True)
    return df, df[FEATURE_COLS], df[EXTERNAL_TARGET_COL]


def evaluate(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def make_model_py_equivalent():
    hgb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.03, random_state=42)
    rf = RandomForestRegressor(
        n_estimators=250,
        max_depth=12,
        random_state=42,
        n_jobs=1,
    )
    stacked = StackingRegressor(
        estimators=[("hgb", hgb), ("rf", rf)],
        final_estimator=Ridge(alpha=0.5),
        passthrough=True,
        n_jobs=1,
    )
    return make_pipeline(StandardScaler(), stacked)


def make_latest_model():
    return AllowedTrainingBlend()


def split_train_model(model, path, test_size=0.2, random_state=RANDOM_STATE):
    df, X, y = load_training(path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return {
        "df": df,
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def full_train_model(model, path):
    df, X, y = load_training(path)
    model.fit(X, y)
    return {"df": df, "model": model, "X": X, "y": y}


def prediction_metrics_row(label, dataset, y_true, y_pred, extra=None):
    row = {"Model": label, "Dataset": dataset, "Rows": len(y_true)}
    if extra:
        row.update(extra)
    row.update(evaluate(y_true, y_pred))
    return row


def external_metric_rows(label, model):
    rows = []
    for dataset_name, path in EXTERNAL_TEST_SETS.items():
        _, X_external, y_external = load_external(path)
        rows.append(
            prediction_metrics_row(
                label,
                dataset_name,
                y_external,
                clip_release(model.predict(X_external)),
            )
        )
    return rows


def formulation_metric_rows(label, model, df, formulation_ids=FORMULATION_IDS):
    rows = []
    all_pred = clip_release(model.predict(df[FEATURE_COLS]))
    for formulation_id in formulation_ids:
        subset = df[df["Formulation Index"].eq(formulation_id)]
        if subset.empty:
            rows.append(
                {
                    "Model": label,
                    "Formulation Index": formulation_id,
                    "Rows": 0,
                    "R2": np.nan,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                }
            )
            continue
        subset_pred = all_pred[subset.index]
        row = {"Model": label, "Formulation Index": formulation_id, "Rows": len(subset)}
        row.update(evaluate(subset[TARGET_COL], subset_pred))
        rows.append(row)
    return rows
