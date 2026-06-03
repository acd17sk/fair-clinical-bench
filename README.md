# fair-clinical-bench

A reproducible benchmark for **clinical risk prediction** that ranks models on **accuracy, calibration, fairness, and explainability** — not just AUC. CPU-only; no GPU required.

## Why

Most clinical ML papers report a single AUC and call it done. In deployment, three things bite:

1. **Calibration drifts** across subgroups — a model with strong global AUC can still be miscalibrated for the population that matters.
2. **Disparate impact** appears at the metric level — equal AUC ≠ equal access.
3. **Black-box predictions** don't survive contact with clinicians.

`fair-clinical-bench` runs a flight of models against a clinical dataset and emits a single comparison report covering all four axes.

## What it does

Given a tabular clinical dataset with a binary outcome and at least one demographic attribute (sex, age band, race, etc.), the framework:

1. Trains a panel of models (logistic regression, random forest, gradient boosting, calibrated MLP).
2. Reports global metrics (AUC, AUPRC, Brier, log-loss).
3. Computes calibration curves + ECE per demographic slice.
4. Computes group-fairness metrics (demographic parity, equalized odds, calibration-by-group).
5. Produces SHAP-based local explanations for the top-K riskiest predictions.
6. Renders a single Markdown report with tables + figures.

## Datasets supported

- **Pima Indians Diabetes** (UCI, 768 rows) — quick smoke test.
- **Heart Disease (Cleveland)** (UCI, 303 rows) — small, classical.
- **Diabetes 130-Hospitals (Strack et al.)** (UCI, 100K rows) — full benchmark.
- Bring-your-own: pass a CSV with `target` + at least one declared demographic column.

## Quick start

```bash
pip install -e .
fair-bench run --dataset pima --output reports/pima.md
```

## CLI

The `fair-bench` command-line tool runs the full benchmark pipeline: load a dataset, train all registered models, compute metrics, generate figures, and render a Markdown report.

### `fair-bench run`

```bash
fair-bench run --dataset <name-or-csv> --output <report.md> [options]
```

| Option | Required | Description |
|---|---|---|
| `--dataset` | Yes | Built-in dataset name (`pima`, `heart`, `diabetes130`) or path to a CSV file. |
| `--output` | Yes | Output Markdown report file path. Figures are written to a `figures/` subdirectory alongside the report. |
| `--target` | No | Target column name. Required when `--dataset` is a CSV path. |
| `--groups` | No | Comma-separated group column names for fairness analysis (e.g. `race,gender`). Only used with CSV datasets. |

#### Built-in dataset

```bash
fair-bench run --dataset heart --output reports/heart.md
```

#### Bring-your-own CSV

```bash
fair-bench run \
  --dataset data/my_cohort.csv \
  --target readmission_30d \
  --groups race,gender \
  --output reports/cohort.md
```

If `--dataset` is not a built-in name and not an existing file path, the CLI exits with an error. If `--target` is omitted for a CSV dataset, the CLI exits with an error.

## Report output

The `--output` flag produces a Markdown report with the following sections:

### Global Metrics

A table comparing all models on discrimination and calibration metrics:

| Metric | Description |
|---|---|
| AUC | Area Under the ROC Curve |
| AUPRC | Area Under the Precision-Recall Curve |
| Brier | Brier score (lower is better) |
| Log-Loss | Cross-entropy loss (lower is better) |
| ECE | Expected Calibration Error (lower is better) |

### Calibration Curves

Per-model calibration (reliability) curves, plotted as PNG images. When group columns are defined, separate curves are drawn for each group value alongside the "perfectly calibrated" diagonal.

### Fairness Metrics

Per-model, per-group fairness metrics:

| Metric | Description |
|---|---|
| Demographic Parity | Max absolute difference in mean predicted probability between groups |
| Equalized Odds | Max of TPR difference and FPR difference across groups (threshold=0.5) |
| Calibration by Group | ECE computed separately for each group value |

A comparison table figure is also generated showing all fairness metrics side-by-side.

### SHAP Explanations

Per-model SHAP-based explanations:

- **Global Feature Importance**: Top-10 features ranked by mean absolute SHAP value, shown as a horizontal bar plot.
- **Local Explanations**: Top-5 highest-risk instances with their top contributing features.

## Figures

All figures are written to a `figures/<model_name>/` subdirectory relative to the output report:

```
reports/
├── cohort.md
└── figures/
    ├── logistic_regression/
    │   ├── calibration.png
    │   ├── fairness.png
    │   └── shap.png
    └── random_forest/
        ├── calibration.png
        ├── fairness.png
        └── shap.png
```

Figure generation uses matplotlib with the `Agg` backend (no display required). All figures are saved at 150 DPI.

## Data module

The `fair_clinical_bench.data` package provides a unified interface for loading clinical datasets, managing fairness-relevant group columns, and creating reproducible train/val/test splits.

### ClinicalDataset

All datasets are returned as a `ClinicalDataset` dataclass:

```python
from fair_clinical_bench.data import ClinicalDataset

ds: ClinicalDataset  # returned by any loader

ds.X          # pd.DataFrame — feature matrix
ds.y          # pd.Series     — target variable
ds.group_cols # list[str]     — protected / demographic group columns in X
ds.name       # str           — short dataset identifier
ds.description  # str | None  — longer description

# Convenience properties
ds.features   # alias for ds.X
ds.target     # alias for ds.y
ds.groups     # ds.X[ds.group_cols] or None if no groups defined
```

`ClinicalDataset` validates on construction: it checks that `X` and `y` have matching row counts, that `group_cols` contains no duplicates, and that every group column exists in `X`.

### Built-in datasets

Three datasets ship with the package. Each loader downloads the canonical UCI version on first call and caches it under `~/.fair-clinical-bench/datasets/`. If the download fails (e.g. offline), a minimal synthetic dataset (50 rows) is generated so demos and tests do not break.

```python
from fair_clinical_bench.data import load_pima, load_heart, load_diabetes130, list_builtin_datasets

# List available built-in datasets
print(list_builtin_datasets())  # ['pima', 'heart', 'diabetes130']

# Load a built-in dataset
pima = load_pima()
print(pima.name)         # 'pima'
print(pima.group_cols)   # ['Age']
print(pima.X.shape)      # (768, 8)  (or 50 rows if synthetic fallback)

heart = load_heart()
print(heart.group_cols)  # ['sex']

diabetes130 = load_diabetes130()
print(diabetes130.group_cols)  # ['race', 'gender', 'age']
```

Each loader handles dataset-specific preprocessing:
- **Pima**: treats zeros in biologically-impossible columns (Glucose, BloodPressure, etc.) as missing and imputes with the median.
- **Heart Disease**: parses `'?'` sentinel values and coerces all columns to numeric.
- **Diabetes 130-Hospitals**: drops columns with >50% missing values, normalises `'?'` and `'Unknown/Invalid'` sentinels, and imputes the remainder.

### Bring-your-own CSV

Load any CSV file as a `ClinicalDataset` with `load_csv`:

```python
from fair_clinical_bench.data import load_csv

ds = load_csv(
    path="data/my_cohort.csv",
    target_col="readmission_30d",
    group_cols=["race", "gender"],
    encoding="utf-8",  # default
)
```

`load_csv` validates that the target column exists and is not also listed as a group column. Missing group columns are warned about and silently dropped. Missing feature values are imputed (median for numeric, mode for categorical).

### Stratified splits

Create reproducible train/val/test splits with optional group-aware stratification:

```python
from fair_clinical_bench.data import load_pima, stratified_split

pima = load_pima()
splits = stratified_split(
    X=pima.X,
    y=pima.y,
    group_col="Age",        # optional — stratify by group + target
    train_frac=0.6,
    val_frac=0.2,
    test_frac=0.2,
    random_seed=42,
)

X_train = pima.X.iloc[splits.train]
y_train = pima.y.iloc[splits.train]
```

The split function:
- Validates that fractions sum to 1.0 (within tolerance).
- Falls back to target-only stratification if group+target combinations are too sparse (< 2 samples).
- Warns when only one unique group value is present.
- Uses a fixed `random_seed` for full reproducibility.

## Models

The `fair_clinical_bench.models` package provides a uniform interface for training and evaluating clinical risk prediction models.

### BenchmarkModel protocol

All model wrappers implement the `BenchmarkModel` protocol, which requires two methods:

```python
from fair_clinical_bench.models import BenchmarkModel

class MyModel:
    def fit(self, X, y) -> BenchmarkModel:
        """Train on feature matrix X and binary target y."""
        ...

    def predict_proba(self, X) -> np.ndarray:
        """Return (n_samples, 2) array of class probabilities."""
        ...
```

Both `fit` and `predict_proba` accept `pd.DataFrame` or `np.ndarray` inputs.

### Model registry

Models are registered by string name and can be retrieved at runtime:

```python
from fair_clinical_bench.models import register, get_model, list_models

# List available models
print(list_models())
# ['calibrated_mlp', 'logistic_regression', 'random_forest', 'xgboost']

# Get a model class by name and instantiate
ModelClass = get_model("logistic_regression")
model = ModelClass(random_state=42)
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)

# Register a custom model
from fair_clinical_bench.models import BenchmarkModel, register

class MyCustomModel:
    def fit(self, X, y) -> BenchmarkModel: ...
    def predict_proba(self, X) -> np.ndarray: ...

register("my_custom", MyCustomModel)
```

### Built-in models

Four models ship with the benchmark, each handling preprocessing automatically:

| Model | Preprocessing | Key features |
|---|---|---|
| `LogisticRegressionModel` | StandardScaler (numeric) + OneHotEncoder (categorical) | sklearn Pipeline; `max_iter`, `random_state` |
| `RandomForestModel` | passthrough (numeric) + OneHotEncoder (categorical) | sklearn Pipeline; `n_estimators`, `random_state` |
| `XGBoostModel` | `pd.Categorical` conversion for object/string columns | Native XGBoost categorical support; `n_estimators`, `max_depth`, `learning_rate` |
| `CalibratedMLPModel` | StandardScaler (numeric) + OneHotEncoder (categorical) | `CalibratedClassifierCV` with sigmoid method; automatic fallback to 2-fold or uncalibrated MLP for small datasets |

```python
from fair_clinical_bench.models import (
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
    CalibratedMLPModel,
)

# Logistic regression with custom hyperparameters
lr = LogisticRegressionModel(max_iter=2000, C=0.5)
lr.fit(X_train, y_train)

# Random forest with more trees
rf = RandomForestModel(n_estimators=200)
rf.fit(X_train, y_train)

# XGBoost with tuned depth and learning rate
xgb = XGBoostModel(n_estimators=200, max_depth=4, learning_rate=0.05)
xgb.fit(X_train, y_train)

# Calibrated MLP with custom architecture
mlp = CalibratedMLPModel(hidden_layer_sizes=(128, 64, 32), max_iter=1000)
mlp.fit(X_train, y_train)

# All models return (n_samples, 2) probability arrays
proba = lr.predict_proba(X_test)
```

Each model auto-detects numeric vs. categorical columns when given a `pd.DataFrame`. When given an `np.ndarray`, all columns are treated as numeric. Calling `predict_proba` before `fit` raises a `RuntimeError`.

## Metrics

The `fair_clinical_bench.metrics` package provides three families of evaluation metrics, all accepting `np.ndarray` or `pd.Series` inputs.

### Discrimination metrics

Measure the model's ability to separate positive from negative cases:

```python
from fair_clinical_bench.metrics import auc_score, auprc_score

auc = auc_score(y_true, y_prob)      # Area Under the ROC Curve [0, 1]
auprc = auprc_score(y_true, y_prob)  # Area Under the Precision-Recall Curve [0, 1]
```

Both functions return `float("nan")` when `y_true` contains fewer than two unique classes.

### Calibration metrics

Measure how well predicted probabilities match observed outcomes:

```python
from fair_clinical_bench.metrics import (
    brier_score,
    log_loss_score,
    expected_calibration_error,
    reliability_curve,
)

brier = brier_score(y_true, y_prob)          # [0, 1]; lower is better
ll = log_loss_score(y_true, y_prob)          # cross-entropy; lower is better
ece = expected_calibration_error(y_true, y_prob, n_bins=10)  # [0, 1]; lower is better

# Reliability curve: returns (prob_true, prob_pred, bin_counts)
prob_true, prob_pred, counts = reliability_curve(y_true, y_prob, n_bins=10)
```

`expected_calibration_error` and `reliability_curve` use equal-width probability bins (default 10). The reliability curve returns three arrays: the observed positive fraction per bin, the mean predicted probability per bin, and the sample count per bin. Empty bins have `NaN` in `prob_true` and `prob_pred`.

### Fairness metrics

Measure disparities across demographic groups:

```python
from fair_clinical_bench.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    calibration_by_group,
)

# Maximum absolute difference in mean predicted probability between any two groups
dp_diff = demographic_parity_difference(y_true, y_prob, group=groups)

# Maximum of TPR difference and FPR difference across groups (threshold=0.5 by default)
eo_diff = equalized_odds_difference(y_true, y_prob, group=groups, threshold=0.5)

# ECE computed separately for each group
ece_by_group = calibration_by_group(y_true, y_prob, group=groups, n_bins=10)
# Returns: {'group_a': 0.03, 'group_b': 0.07, ...}
```

All fairness metrics return `0.0` (or an empty dict) when fewer than two unique groups are present. Lower values indicate better fairness.

## Explainability

The `fair_clinical_bench.explain` package provides SHAP-based model explanations. Both functions require a fitted model with a `predict_proba` method and a `pd.DataFrame` feature matrix.

### Local explanations

Generate SHAP explanations for the highest-risk predictions:

```python
from fair_clinical_bench.explain import shap_local_explanations

explanations = shap_local_explanations(
    model=fitted_model,
    X=X_test,
    background_samples=100,  # size of background dataset for SHAP explainer
    top_k=5,                 # number of top features per explanation
)

# Returns DataFrame with columns:
# instance_index, predicted_risk, feature_name, shap_value
print(explanations.head())
```

The function selects the `top_k` instances with the highest predicted risk and computes SHAP values for each. Results include the top-K contributing features per instance, sorted by absolute SHAP value. If the primary `shap.Explainer` fails, it falls back to `PermutationExplainer`, and then to an empty DataFrame if both fail (with warnings).

### Global feature importance

Compute mean absolute SHAP values across all samples:

```python
from fair_clinical_bench.explain import shap_global_importance

importance = shap_global_importance(
    model=fitted_model,
    X=X_test,
    background_samples=100,
)

# Returns DataFrame sorted by descending importance:
# feature_name, mean_abs_shap
print(importance.head())
```

The function uses a random sample of `background_samples` rows as the SHAP background dataset. Results are sorted by `mean_abs_shap` in descending order. The same fallback chain (Explainer → PermutationExplainer → empty DataFrame) applies.

## License

MIT.

## Status

This repository is being built as a demonstration of **Epic Mode** — an autonomous coupling-aware execution layer for [opencode-swarm](https://github.com/zaxbysauce/opencode-swarm). Epic Mode decides per phase whether tasks can be safely parallelized, calibrates against observed scope discipline, and persists per-phase decisions to `.swarm/evidence/epic-promotions.jsonl`. See `docs/epic-mode-tracking.md` for the live log of decisions Epic made during the build.


