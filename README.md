# Reproducible weighted-ensemble release model

This directory contains the data, code, fitted artifact, and reference results needed to reproduce the reported cumulative fractional release (CFR) model. The definitive implementation is **`model.py`**.

## Model

`model.py` defines a weighted ensemble with three branches:

1. A tuned stack with HistGradientBoosting, Random Forest, and Ridge base learners and a Ridge meta-learner with original-feature passthrough (`cv=3`).
2. A bagging regressor containing bootstrap-aggregated decision trees.
3. A voting regressor that averages HistGradientBoosting, ExtraTrees, and GradientBoosting.

The branch weights are 5/13, 2/13, and 6/13, respectively. Each branch prediction is clipped to [0,1], the weighted predictions are summed, and the final CFR prediction is clipped to [0,1] again.

The model uses these eleven predictors:

- Drug MW
- Drug TPSA
- Drug LogP
- Polymer MW
- LA/GA
- Initial Drug-to-Polymer Ratio
- Particle Size
- Drug Loading Capacity
- Drug Encapsulation Efficiency
- Solubility Enhancer Concentration
- Time

## Data

All required workbooks are in `data/`:

- `training_data.xlsx`: exact curated training workbook used by `model.py`; 6,130 complete observations representing 352 formulations.
- `ibmeca_trial_1.xlsx`: five-point IBMECA/RG502H release profile used for external evaluation.
- `ibmeca_trial_3.xlsx`: nine-point IBMECA/RG504H release profile used for external evaluation.

The training target is `Release`; the external-workbook target is `Actual`. Units and preprocessing are already encoded in the supplied workbooks. Do not change column names.

## Environment setup

Python 3.11 is recommended. From this directory on Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Reproduce the primary results

```powershell
python model.py
```

This command performs the fixed row-wise 80/20 split (`random_state=42`), trains and evaluates the split model, refits the deployable model on all complete training rows, evaluates both IBMECA workbooks, and writes:

- `split_model.pkl`
- `trained_model.pkl`
- `best_allowed_training_split_metrics.csv`
- `best_allowed_training_external_metrics.csv`
- `best_allowed_training_formulation_metrics.csv`
- prediction workbooks for both IBMECA trials

Expected primary metrics are:

| Evaluation | Rows | R2 | MAE | RMSE |
|---|---:|---:|---:|---:|
| Training split | 4,904 | 0.9503 | 0.0538 | 0.0742 |
| Held-out test split | 1,226 | 0.9019 | 0.0724 | 0.1049 |
| IBMECA Trial 1 | 5 | 0.9666 | 0.0321 | 0.0367 |
| IBMECA Trial 3 | 9 | 0.9556 | 0.0328 | 0.0400 |

Small last-digit differences can occur across numerical-library or platform versions. The pinned environment minimizes this variation.

To load the deployable artifact from this directory:

```python
import joblib
import model  # required so the custom ensemble class is available

fitted_model = joblib.load("trained_model.pkl")
predicted_cfr = fitted_model.predict(new_data[model.FEATURE_COLS])
```

## Supporting analyses

Run the following after `python model.py`:

```powershell
python regression_baselines.py
python ablation.py
python explain_model.py
python reproduce_figures.py
```

- `regression_baselines.py` compares OLS, Ridge, Lasso, Elastic Net, and Huber regression under the same split and external-evaluation procedure.
- `ablation.py` evaluates individual ensemble branches, branch-removal scenarios, and estimator-order permutations.
- `explain_model.py` creates the model-agnostic Kernel-SHAP beeswarm and exports the values underlying the formulation-profile and residual graphs.
- `reproduce_figures.py` creates publication-quality Figures 6–8 as PDF/SVG and 600-dpi PNG/TIFF files in `publication_figures/`.
- `evaluation.py` provides shared loading, metric, external-evaluation, and formulation-evaluation helpers for `ablation.py`.

Reference CSVs from the verified run are in `expected_results/`. `SHA256SUMS.txt` records file hashes for provenance checks.

## Interpretation and validation caveats

- The reported 80/20 evaluation is a **row-wise random split**, not a formulation-grouped split. Because a formulation has multiple time points, the train and test subsets can contain observations from the same formulation. The test score therefore measures interpolation across held-out observations and must not be described as unseen-formulation generalization.
- Formulation-wise R2 values generated from the full-fit model are profile-consistency diagnostics, not independent external validation.
- IBMECA observations are not used to fit estimator parameters. However, the final architecture and fixed branch weights were chosen during development partly with reference to IBMECA behavior. The IBMECA results should therefore be described as external predictive checks rather than an untouched confirmatory validation set.
- The two IBMECA datasets contain only five and nine observations. Their R2 values are informative but statistically fragile.
- The serialized model is included for convenience, but rerunning `model.py` is the authoritative reproducibility path.

## Directory layout

```text
reproducible_workspace/
├── README.md
├── requirements.txt
├── model.py
├── trained_model.pkl
├── regression_baselines.py
├── ablation.py
├── evaluation.py
├── explain_model.py
├── reproduce_figures.py
├── data/
│   ├── training_data.xlsx
│   ├── ibmeca_trial_1.xlsx
│   └── ibmeca_trial_3.xlsx
└── expected_results/
    └── verified CSV outputs
```
