from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TRAINING_PATH = DATA_DIR / "training_data.xlsx"
RANDOM_STATE = 42
TEST_SIZE = 0.2

EXTERNAL_TEST_SETS = {
    "IBMECA Trial 1": DATA_DIR / "ibmeca_trial_1.xlsx",
    "IBMECA Trial 3": DATA_DIR / "ibmeca_trial_3.xlsx",
}

FORMULATION_IDS = [277, 103, 233, 32, 292, 304]

FEATURE_COLS = [
    "Drug MW",
    "Drug TPSA",
    "Drug LogP",
    "Polymer MW",
    "LA/GA",
    "Initial Drug-to-Polymer Ratio",
    "Particle Size",
    "Drug Loading Capacity",
    "Drug Encapsulation Efficiency",
    "Solubility Enhancer Concentration",
    "Time",
]
TARGET_COL = "Release"
EXTERNAL_TARGET_COL = "Actual"
CLIP_BOUNDS = (0.0, 1.0)


class AllowedTrainingBlend:
    """Blend trained only on the revised training workbook supplied by the user."""

    def __init__(self, tuned_stack_weight=5 / 13, bag_weight=2 / 13, vote_weight=6 / 13):
        self.tuned_stack_weight = tuned_stack_weight
        self.bag_weight = bag_weight
        self.vote_weight = vote_weight
        self.tuned_stack = self._make_tuned_stack()
        self.bag_model = self._make_bag_model()
        self.vote_model = self._make_vote_model()

    @staticmethod
    def _make_tuned_stack():
        hgb = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.02,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        )
        rf = RandomForestRegressor(
            n_estimators=300,
            min_samples_split=4,
            min_samples_leaf=1,
            max_features=0.7,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        ridge = Ridge(alpha=0.1)
        return StackingRegressor(
            estimators=[("hgb", hgb), ("rf", rf), ("ridge", ridge)],
            final_estimator=Ridge(alpha=0.25),
            passthrough=True,
            n_jobs=1,
            cv=3,
        )

    @staticmethod
    def _make_bag_model():
        return BaggingRegressor(
            estimator=DecisionTreeRegressor(
                max_depth=8,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
            ),
            n_estimators=160,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )

    @staticmethod
    def _make_vote_model():
        hgb = HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.025,
            max_leaf_nodes=15,
            min_samples_leaf=12,
            l2_regularization=0.01,
            random_state=RANDOM_STATE,
        )
        et = ExtraTreesRegressor(
            n_estimators=260,
            max_depth=16,
            min_samples_leaf=2,
            max_features=0.7,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        gbr = GradientBoostingRegressor(
            n_estimators=350,
            learning_rate=0.025,
            max_depth=3,
            subsample=0.85,
            random_state=RANDOM_STATE,
        )
        return VotingRegressor(
            estimators=[("hgb", hgb), ("et", et), ("gbr", gbr)],
            n_jobs=1,
        )

    def fit(self, X, y):
        self.tuned_stack.fit(X, y)
        self.bag_model.fit(X, y)
        self.vote_model.fit(X, y)
        return self

    def predict(self, X):
        tuned_pred = np.clip(self.tuned_stack.predict(X), *CLIP_BOUNDS)
        bag_pred = np.clip(self.bag_model.predict(X), *CLIP_BOUNDS)
        vote_pred = np.clip(self.vote_model.predict(X), *CLIP_BOUNDS)
        pred = (
            self.tuned_stack_weight * tuned_pred
            + self.bag_weight * bag_pred
            + self.vote_weight * vote_pred
        )
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


def main():
    df, X, y = load_training(TRAINING_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    split_model = AllowedTrainingBlend().fit(X_train, y_train)
    joblib.dump(split_model, ROOT / "split_model.pkl")
    split_rows = []
    for split_name, split_X, split_y in [
        ("Train", X_train, y_train),
        ("Test", X_test, y_test),
    ]:
        split_metrics = evaluate(split_y, split_model.predict(split_X))
        split_rows.append({"Dataset": split_name, "Rows": len(split_y), **split_metrics})

    final_model = AllowedTrainingBlend().fit(X, y)
    external_rows = []
    for test_name, test_path in EXTERNAL_TEST_SETS.items():
        test_df, X_external, y_external = load_external(test_path)
        pred = final_model.predict(X_external)
        external_metrics = evaluate(y_external, pred)
        external_rows.append({"Dataset": test_name, "Rows": len(test_df), **external_metrics})

        output_df = test_df.copy()
        output_df["Allowed Training Model Predicted Release"] = pred
        output_df.to_excel(
            ROOT / f"{test_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_allowed_training_predictions.xlsx",
            index=False,
        )

    full_pred = final_model.predict(X)
    formulation_rows = []
    for formulation_id in FORMULATION_IDS:
        subset = df[df["Formulation Index"].eq(formulation_id)]
        subset_pred = full_pred[subset.index]
        formulation_metrics = evaluate(subset[TARGET_COL], subset_pred)
        formulation_rows.append(
            {
                "Formulation Index": formulation_id,
                "Rows": len(subset),
                **formulation_metrics,
            }
        )

    split_metrics_df = pd.DataFrame(split_rows)
    external_metrics_df = pd.DataFrame(external_rows)
    formulation_metrics_df = pd.DataFrame(formulation_rows)

    split_metrics_df.to_csv(ROOT / "best_allowed_training_split_metrics.csv", index=False)
    external_metrics_df.to_csv(ROOT / "best_allowed_training_external_metrics.csv", index=False)
    formulation_metrics_df.to_csv(
        ROOT / "best_allowed_training_formulation_metrics.csv",
        index=False,
    )
    joblib.dump(final_model, ROOT / "trained_model.pkl")

    print("=== 80/20 split metrics on revised training workbook ===")
    print(split_metrics_df.to_string(index=False))
    print("\n=== External IBMECA metrics from full-fit model ===")
    print(external_metrics_df.to_string(index=False))
    print("\n=== Requested formulation diagnostics from full-fit model ===")
    print(formulation_metrics_df.to_string(index=False))
    print("\nSaved split_model.pkl and trained_model.pkl")


if __name__ == "__main__":
    # Store a portable module path in joblib artifacts when this file is run
    # directly. Loading then works with `import model` from this directory.
    sys.modules.setdefault("model", sys.modules[__name__])
    AllowedTrainingBlend.__module__ = "model"
    main()
