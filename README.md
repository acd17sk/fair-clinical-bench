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

## License

MIT.

## Status

This repository is being built as a demonstration of **Epic Mode** — an autonomous coupling-aware execution layer for [opencode-swarm](https://github.com/zaxbysauce/opencode-swarm). Epic Mode decides per phase whether tasks can be safely parallelized, calibrates against observed scope discipline, and persists per-phase decisions to `.swarm/evidence/epic-promotions.jsonl`. See `docs/epic-mode-tracking.md` for the live log of decisions Epic made during the build.


