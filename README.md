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

## License

MIT.

## Status

This repository is being built as a demonstration of **Epic Mode** — an autonomous coupling-aware execution layer for [opencode-swarm](https://github.com/zaxbysauce/opencode-swarm). Epic Mode decides per phase whether tasks can be safely parallelized, calibrates against observed scope discipline, and persists per-phase decisions to `.swarm/evidence/epic-promotions.jsonl`. See `docs/epic-mode-tracking.md` for the live log of decisions Epic made during the build.


