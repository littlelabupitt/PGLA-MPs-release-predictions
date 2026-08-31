"""Create a 600-dpi SHAP beeswarm and CSV data behind Figure 7.

SHAP and residual values use the fixed held-out split (random_state=42).
Per-formulation values use the full-data refitted model and therefore describe
profile consistency, not leave-one-formulation-out external generalization.
"""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "paper2_matplotlib"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from model import (
    AllowedTrainingBlend,
    FEATURE_COLS,
    RANDOM_STATE,
    TARGET_COL,
    TRAINING_PATH,
)


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "publication_figures"
OUTDIR.mkdir(exist_ok=True)
FULL_MODEL_PATH = ROOT / "trained_model.pkl"
SPLIT_MODEL_PATH = ROOT / "split_model.pkl"
FORMULATION_ID = "Formulation Index"


def load_data():
    data = pd.read_excel(TRAINING_PATH, engine="openpyxl")
    data.columns = data.columns.str.strip()
    required = FEATURE_COLS + [TARGET_COL, FORMULATION_ID]
    return data.dropna(subset=required).reset_index(drop=True)


def predict(model, frame):
    return np.clip(model.predict(frame[FEATURE_COLS]), 0.0, 1.0)


def save_figure(fig, stem):
    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 600},
        ".tiff": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
    }.items():
        path = OUTDIR / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        print(f"Saved {path}")


def export_formulation_values(data, full_model):
    all_pred = predict(full_model, data)
    scored = data[[FORMULATION_ID, TARGET_COL]].copy()
    scored["Predicted Release"] = all_pred
    rows = []
    for formulation_id, group in scored.groupby(FORMULATION_ID):
        if len(group) < 2:
            continue
        observed = group[TARGET_COL].to_numpy()
        predicted = group["Predicted Release"].to_numpy()
        rows.append(
            {
                "Formulation Index": formulation_id,
                "Number of observations": len(group),
                "R2": r2_score(observed, predicted),
                "MAE": mean_absolute_error(observed, predicted),
                "RMSE": np.sqrt(mean_squared_error(observed, predicted)),
                "Evaluation Type": "Full-fit formulation profile consistency",
            }
        )
    values = pd.DataFrame(rows).sort_values("Formulation Index")
    path = OUTDIR / "formulation_wise_generalization_graph_values.csv"
    values.to_csv(path, index=False)
    print(f"Saved {path}")
    return values


def export_residual_values(data, split_model):
    indices = data.index.to_series()
    _, test_indices = train_test_split(indices, test_size=0.2, random_state=RANDOM_STATE)
    test = data.loc[test_indices].copy()
    predicted = predict(split_model, test)
    values = pd.DataFrame(
        {
            "Source Row Index": test.index,
            "Formulation Index": test[FORMULATION_ID].to_numpy(),
            "Observed Release": test[TARGET_COL].to_numpy(),
            "Predicted Release": predicted,
            "Residual (Observed - Predicted)": test[TARGET_COL].to_numpy() - predicted,
            "Dataset": "Fixed 20% held-out test split",
        }
    ).sort_values("Source Row Index")
    path = OUTDIR / "residual_graph_values.csv"
    values.to_csv(path, index=False)
    print(f"Saved {path}")
    return test, values


def create_shap_beeswarm(data, split_model, test):
    train = data.drop(index=test.index)
    background = train[FEATURE_COLS].sample(min(40, len(train)), random_state=RANDOM_STATE)
    explain = test[FEATURE_COLS].sample(min(120, len(test)), random_state=RANDOM_STATE)

    def predict_function(values):
        frame = pd.DataFrame(values, columns=FEATURE_COLS)
        return np.clip(split_model.predict(frame), 0.0, 1.0)

    explainer = shap.KernelExplainer(predict_function, background)
    raw_values = np.asarray(explainer.shap_values(explain, nsamples=150))
    if raw_values.ndim == 3 and raw_values.shape[-1] == 1:
        raw_values = raw_values[..., 0]

    long_rows = []
    for sample_position, (source_index, feature_row) in enumerate(explain.iterrows()):
        for feature_position, feature in enumerate(FEATURE_COLS):
            long_rows.append(
                {
                    "Explained Sample": sample_position,
                    "Source Row Index": source_index,
                    "Formulation Index": data.loc[source_index, FORMULATION_ID],
                    "Feature": feature,
                    "Feature Value": feature_row[feature],
                    "SHAP Value": raw_values[sample_position, feature_position],
                }
            )
    shap_csv = OUTDIR / "shap_beeswarm_values.csv"
    pd.DataFrame(long_rows).to_csv(shap_csv, index=False)
    print(f"Saved {shap_csv}")

    importance = pd.DataFrame(
        {"Feature": FEATURE_COLS, "Mean Absolute SHAP Value": np.abs(raw_values).mean(axis=0)}
    ).sort_values("Mean Absolute SHAP Value", ascending=False)
    importance_path = OUTDIR / "shap_beeswarm_feature_importance.csv"
    importance.to_csv(importance_path, index=False)
    print(f"Saved {importance_path}")

    plt.figure(figsize=(9.2, 6.3))
    shap.summary_plot(
        raw_values,
        explain,
        feature_names=FEATURE_COLS,
        plot_type="dot",
        max_display=len(FEATURE_COLS),
        plot_size=None,
        show=False,
        color_bar=True,
    )
    ax = plt.gca()
    ax.set_xlabel("SHAP value (impact on predicted release)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Features", fontsize=12, fontweight="bold")
    ax.tick_params(axis="both", labelsize=10)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.axvline(0, color="#555555", linewidth=0.9, zorder=0)
    plt.tight_layout()
    save_figure(plt.gcf(), "Figure7b_SHAP_beeswarm_real_data")
    plt.close()


def main():
    data = load_data()
    full_model = joblib.load(FULL_MODEL_PATH)
    split_model = joblib.load(SPLIT_MODEL_PATH)
    export_formulation_values(data, full_model)
    test, _ = export_residual_values(data, split_model)
    create_shap_beeswarm(data, split_model, test)


if __name__ == "__main__":
    main()
