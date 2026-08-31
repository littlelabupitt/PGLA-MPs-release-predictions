"""Rebuild manuscript Figures 6--8 from the current weighted ensemble.

Outputs are written to ``publication_figures_6_8`` in vector (SVG/PDF) and
high-resolution raster (600-dpi PNG/TIFF) formats.  Figure 6A and Figure 7C
use the fixed 80/20 held-out split; profile-consistency panels use the model
refit on the complete curated dataset; Figure 8 uses the same full-fit model
used for external IBMECA evaluation.
"""

import os
import tempfile
from pathlib import Path

import joblib
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "paper2_matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from model import (
    AllowedTrainingBlend,
    EXTERNAL_TARGET_COL,
    EXTERNAL_TEST_SETS,
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
SHAP_IMPORTANCE_PATH = ROOT / "expected_results" / "shap_beeswarm_feature_importance.csv"
FORMULATION_ID = "Formulation Index"

PROFILE_LABELS = {
    336: "RG755S",
    346: "RG858S",
    277: "Low drug MW",
    103: "High drug MW",
    233: "Low drug logP",
    32: "High drug logP",
    292: "Low polymer MW",
    304: "High polymer MW",
}

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#7B61A8"
GRAY = "#4D4D4D"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "lines.linewidth": 1.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
)


def load_training_data():
    data = pd.read_excel(TRAINING_PATH, engine="openpyxl")
    data.columns = data.columns.str.strip()
    required = FEATURE_COLS + [TARGET_COL, FORMULATION_ID]
    return data.dropna(subset=required).reset_index(drop=True)


def load_models():
    if not FULL_MODEL_PATH.exists() or not SPLIT_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Expected current full-fit and split-fit model files are missing. "
            "Run model.py first to create trained_model.pkl and split_model.pkl."
        )
    return joblib.load(FULL_MODEL_PATH), joblib.load(SPLIT_MODEL_PATH)


def predict(model, frame):
    return np.clip(model.predict(frame[FEATURE_COLS]), 0.0, 1.0)


def save_all(fig, stem):
    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 600},
        ".tiff": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
    }.items():
        path = OUTDIR / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        print(f"Saved {path}")


def panel_label(ax, label):
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.5, width=0.8)


def metric_text(y_true, y_pred):
    return (
        f"$R^2$ = {r2_score(y_true, y_pred):.3f}\n"
        f"MAE = {mean_absolute_error(y_true, y_pred):.3f}\n"
        f"RMSE = {np.sqrt(mean_squared_error(y_true, y_pred)):.3f}"
    )


def plot_profile(ax, group, predictions, title, show_legend=False):
    order = np.argsort(group["Time"].to_numpy())
    time = group["Time"].to_numpy()[order]
    observed = group[TARGET_COL].to_numpy()[order]
    predicted = np.asarray(predictions)[order]
    ax.plot(time, observed, color=BLUE, marker="o", markersize=3.2, label="Observed")
    ax.plot(
        time,
        predicted,
        color=ORANGE,
        marker="s",
        markersize=3.0,
        linestyle="--",
        label="Predicted",
    )
    ax.set_title(title, pad=5)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Cumulative fractional release")
    ax.set_ylim(-0.04, 1.04)
    ax.text(
        0.04,
        0.94,
        f"$R^2$ = {r2_score(observed, predicted):.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    if show_legend:
        ax.legend(frameon=False, loc="lower right")
    clean_axis(ax)


def make_figure6(data, full_model, split_model):
    indices = data.index.to_series()
    _, test_indices = train_test_split(indices, test_size=0.2, random_state=RANDOM_STATE)
    test = data.loc[test_indices]
    test_pred = predict(split_model, test)

    fig = plt.figure(figsize=(11.4, 12.2), constrained_layout=True)
    grid = fig.add_gridspec(4, 2, height_ratios=[1.05, 1.0, 1.0, 1.0])

    ax = fig.add_subplot(grid[0, 0])
    observed = test[TARGET_COL].to_numpy()
    ax.scatter(observed, test_pred, s=11, color=BLUE, alpha=0.45, edgecolors="none", rasterized=True)
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1.2)
    ax.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), xlabel="Observed CFR", ylabel="Predicted CFR")
    ax.set_aspect("equal", adjustable="box")
    ax.text(0.05, 0.95, metric_text(observed, test_pred), transform=ax.transAxes, va="top")
    panel_label(ax, "A")
    clean_axis(ax)

    ax = fig.add_subplot(grid[0, 1])
    for fid, color, marker in [(336, BLUE, "o"), (346, ORANGE, "s")]:
        group = data[data[FORMULATION_ID].eq(fid)].sort_values("Time")
        pred = predict(full_model, group)
        label = PROFILE_LABELS[fid]
        ax.plot(group["Time"], group[TARGET_COL], color=color, marker=marker, markersize=2.8, label=f"{label}, observed")
        ax.plot(group["Time"], pred, color=color, linestyle="--", linewidth=1.7, label=f"{label}, predicted")
    ax.set(xlabel="Time (days)", ylabel="Cumulative fractional release", ylim=(-0.04, 1.04))
    ax.legend(frameon=False, ncol=2, loc="upper left", columnspacing=0.9, handlelength=2.0)
    panel_label(ax, "B")
    clean_axis(ax)

    pairs = [
        (277, 103, "Drug molecular weight"),
        (233, 32, "Drug logP"),
        (292, 304, "Polymer molecular weight"),
    ]
    for row, (low_id, high_id, group_title) in enumerate(pairs, start=1):
        for col, fid in enumerate((low_id, high_id)):
            ax = fig.add_subplot(grid[row, col])
            group = data[data[FORMULATION_ID].eq(fid)].copy()
            plot_profile(
                ax,
                group,
                predict(full_model, group),
                f"{group_title}: {PROFILE_LABELS[fid].split(' ', 1)[0].lower()}\nFormulation {fid}",
                show_legend=(row == 1 and col == 0),
            )
            if row == 1 and col == 0:
                panel_label(ax, "C")

    save_all(fig, "Figure6_release_prediction_performance")
    plt.close(fig)


def formulation_r2(data, full_model):
    rows = []
    for fid, group in data.groupby(FORMULATION_ID):
        if len(group) < 2:
            continue
        pred = predict(full_model, group)
        rows.append({"Formulation Index": fid, "Count": len(group), "R2": r2_score(group[TARGET_COL], pred)})
    return pd.DataFrame(rows)


def make_figure7(data, full_model, split_model):
    indices = data.index.to_series()
    _, test_indices = train_test_split(indices, test_size=0.2, random_state=RANDOM_STATE)
    test = data.loc[test_indices]
    test_pred = predict(split_model, test)
    residuals = test[TARGET_COL].to_numpy() - test_pred
    profiles = formulation_r2(data, full_model)
    profiles.to_csv(OUTDIR / "Figure7_formulation_r2_values.csv", index=False)

    if not SHAP_IMPORTANCE_PATH.exists():
        raise FileNotFoundError(f"Missing saved SHAP importance data: {SHAP_IMPORTANCE_PATH}")
    importance = pd.read_csv(SHAP_IMPORTANCE_PATH)
    if "Mean Absolute SHAP Value" in importance.columns:
        importance = importance.rename(columns={"Mean Absolute SHAP Value": "Mean_Abs_SHAP"})
    importance = importance.sort_values("Mean_Abs_SHAP")

    fig = plt.figure(figsize=(13.2, 4.2), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 1.0])

    ax = fig.add_subplot(grid[0, 0])
    finite_r2 = profiles.loc[np.isfinite(profiles["R2"]), "R2"]
    bins = np.linspace(max(-0.5, finite_r2.min()), 1.0, 26)
    ax.hist(finite_r2, bins=bins, color=BLUE, edgecolor="white", linewidth=0.6)
    median = finite_r2.median()
    ax.axvline(median, color=GRAY, linestyle="--", linewidth=1.3)
    ax.text(0.04, 0.94, f"n = {len(finite_r2)}\nMedian $R^2$ = {median:.2f}", transform=ax.transAxes, va="top")
    ax.set(xlabel="Per-formulation $R^2$", ylabel="Number of formulations")
    panel_label(ax, "A")
    clean_axis(ax)

    ax = fig.add_subplot(grid[0, 1])
    ax.barh(importance["Feature"], importance["Mean_Abs_SHAP"], color=PURPLE)
    ax.set(xlabel="Mean absolute SHAP value", ylabel="")
    panel_label(ax, "B")
    clean_axis(ax)

    ax = fig.add_subplot(grid[0, 2])
    ax.hist(residuals, bins=30, color=GREEN, edgecolor="white", linewidth=0.6)
    ax.axvline(0, color=GRAY, linewidth=1.2)
    ax.axvline(residuals.mean(), color=ORANGE, linestyle="--", linewidth=1.2)
    ax.text(0.04, 0.94, f"Mean = {residuals.mean():.3f}\nSD = {residuals.std(ddof=1):.3f}", transform=ax.transAxes, va="top")
    ax.set(xlabel="Residual (observed − predicted)", ylabel="Count")
    panel_label(ax, "C")
    clean_axis(ax)

    save_all(fig, "Figure7_interpretability_and_diagnostics")
    plt.close(fig)


def make_figure8(full_model):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), constrained_layout=True, sharey=True)
    rows = []
    trial_titles = ["RG502H PLGA", "RG504H PLGA"]
    for label, path, title, ax, letter in zip(EXTERNAL_TEST_SETS, EXTERNAL_TEST_SETS.values(), trial_titles, axes, ["A", "B"]):
        frame = pd.read_excel(path, engine="openpyxl")
        frame.columns = frame.columns.str.strip()
        frame = frame.dropna(subset=FEATURE_COLS + [EXTERNAL_TARGET_COL]).sort_values("Time")
        pred = predict(full_model, frame)
        observed = frame[EXTERNAL_TARGET_COL].to_numpy()
        ax.plot(frame["Time"], observed, color=BLUE, marker="o", markersize=4.2, label="Experimental")
        ax.plot(frame["Time"], pred, color=ORANGE, marker="s", markersize=3.8, linestyle="--", label="Predicted")
        ax.set(xlabel="Time (days)", ylim=(-0.04, 1.04), title=title)
        ax.text(0.05, 0.94, metric_text(observed, pred), transform=ax.transAxes, va="top", fontsize=8)
        panel_label(ax, letter)
        clean_axis(ax)
        rows.append(
            {
                "Dataset": label,
                "Rows": len(frame),
                "R2": r2_score(observed, pred),
                "MAE": mean_absolute_error(observed, pred),
                "RMSE": np.sqrt(mean_squared_error(observed, pred)),
            }
        )
    axes[0].set_ylabel("Cumulative fractional release")
    handles = [
        Line2D([0], [0], color=BLUE, marker="o", markersize=4, label="Experimental"),
        Line2D([0], [0], color=ORANGE, marker="s", linestyle="--", markersize=4, label="Predicted"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    pd.DataFrame(rows).to_csv(OUTDIR / "Figure8_external_metrics.csv", index=False)
    save_all(fig, "Figure8_IBMECA_external_validation")
    plt.close(fig)


def main():
    data = load_training_data()
    full_model, split_model = load_models()
    make_figure6(data, full_model, split_model)
    make_figure7(data, full_model, split_model)
    make_figure8(full_model)


if __name__ == "__main__":
    main()
