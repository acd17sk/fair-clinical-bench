# Architecture

`fair-clinical-bench` is organised as decoupled modules so that no two tasks within a build phase need to touch the same file. This is intentional — it makes the project a natural fit for Epic Mode's parallel-promotion path.

## Module map

```
fair_clinical_bench/
├── __init__.py
├── cli.py                  # argparse / click entry point
├── data/
│   ├── __init__.py
│   ├── loader.py           # dataset adapters (Pima, Heart, Diabetes-130, BYO-CSV)
│   ├── schema.py           # dataclass for `ClinicalDataset` (X, y, group_cols, ...)
│   └── splits.py           # stratified train/val/test
├── models/
│   ├── __init__.py
│   ├── protocol.py           # `BenchmarkModel` protocol (fit, predict_proba)
│   ├── registry.py           # register / get_model / list_models
│   ├── logistic_regression.py
│   ├── random_forest.py
│   ├── xgboost.py
│   └── calibrated_mlp.py
├── metrics/
│   ├── __init__.py
│   ├── discrimination.py   # AUC, AUPRC
│   ├── calibration.py      # Brier, log-loss, ECE, reliability curve
│   └── fairness.py         # demographic parity, equalized odds, calibration-by-group
├── explain/
│   ├── __init__.py
│   ├── local.py            # SHAP local explanations (top-K highest-risk)
│   └── global_.py          # SHAP global feature importance (mean |SHAP|)
└── report/
    ├── __init__.py
    ├── renderer.py         # Markdown report assembly
    └── figures.py          # calibration curves, fairness tables, SHAP plots
```

## Phases (build plan)

The swarm will build this in four phases, each with multiple parallel-friendly tasks:

**Phase 1 — Data foundation**
- 1.1 dataset loaders (Pima, Heart, Diabetes-130, BYO-CSV)
- 1.2 `ClinicalDataset` schema + group-column contract
- 1.3 stratified splits + reproducibility seeding

**Phase 2 — Models**
- 2.1 `BenchmarkModel` protocol + registry
- 2.2 logistic regression wrapper
- 2.3 random forest wrapper
- 2.4 xgboost wrapper
- 2.5 calibrated MLP wrapper

**Phase 3 — Metrics & explanations**
- 3.1 discrimination metrics
- 3.2 calibration metrics + reliability curve
- 3.3 fairness metrics (demographic parity, equalized odds, calibration-by-group)
- 3.4 SHAP local explanations
- 3.5 SHAP global explanations

**Phase 4 — Report & CLI**
- 4.1 Markdown report renderer
- 4.2 figure generation (calibration, fairness, SHAP)
- 4.3 click-based CLI (`fair-bench run`)
- 4.4 end-to-end smoke test against Pima

Each phase has 3–5 independent tasks → Epic Mode should promote nearly every phase to parallel execution, and the calibration loop has multiple opportunities to learn from any divergence.

## Decoupling contract

Every module in `models/`, `metrics/`, and `explain/` MUST be self-contained — no module under those directories may import another sibling at runtime. Cross-module composition happens only in `report/renderer.py` and `cli.py`. This is what keeps Epic Mode's `p` low.





