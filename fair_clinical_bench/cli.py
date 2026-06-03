"""Command-line entry point for fair-clinical-bench.

Provides ``fair-bench run`` to load a dataset, train all registered models,
compute metrics, generate figures, and render a Markdown report.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd

from fair_clinical_bench.data.loader import (
    list_builtin_datasets,
    load_csv,
    load_diabetes130,
    load_heart,
    load_pima,
)
from fair_clinical_bench.data.schema import ClinicalDataset
from fair_clinical_bench.data.splits import stratified_split
from fair_clinical_bench.explain.global_ import shap_global_importance
from fair_clinical_bench.explain.local import shap_local_explanations
from fair_clinical_bench.metrics.calibration import (
    brier_score,
    expected_calibration_error,
)
from fair_clinical_bench.metrics.discrimination import auc_score, auprc_score
from fair_clinical_bench.metrics.fairness import (
    calibration_by_group,
    demographic_parity_difference,
    equalized_odds_difference,
)
from fair_clinical_bench.models.registry import list_models
from fair_clinical_bench.report.figures import (
    plot_calibration_curves,
    plot_fairness_comparison,
    plot_shap_summary,
)
from fair_clinical_bench.report.renderer import render_report


@click.group()
def cli() -> None:
    """fair-clinical-bench — clinical risk prediction benchmark."""


@cli.command()
@click.option(
    "--dataset",
    required=True,
    help="Built-in dataset name (pima, heart, diabetes130) or path to a CSV file.",
)
@click.option(
    "--output",
    required=True,
    help="Output Markdown report file path.",
)
@click.option(
    "--target",
    default=None,
    help="Target column name (required for CSV datasets).",
)
@click.option(
    "--groups",
    default=None,
    help="Comma-separated group column names (optional for CSV datasets).",
)
def run(
    dataset: str,
    output: str,
    target: str | None,
    groups: str | None,
) -> None:
    """Run the full benchmark pipeline on a dataset."""
    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    builtin_names = list_builtin_datasets()
    if dataset in builtin_names:
        clinical_ds = _load_builtin(dataset)
    else:
        csv_path = Path(dataset)
        if not csv_path.exists():
            click.echo(
                f"Error: dataset '{dataset}' is not a built-in name and not a file.",
                err=True,
            )
            sys.exit(1)
        if target is None:
            click.echo("Error: --target is required for CSV datasets.", err=True)
            sys.exit(1)
        group_cols = [g.strip() for g in groups.split(",")] if groups else []
        clinical_ds = load_csv(csv_path, target_col=target, group_cols=group_cols or None)

    # ------------------------------------------------------------------
    # 2. Split data
    # ------------------------------------------------------------------
    group_col_for_split = clinical_ds.group_cols[0] if clinical_ds.group_cols else None
    split = stratified_split(
        clinical_ds.X,
        clinical_ds.y,
        group_col=group_col_for_split,
        random_seed=42,
    )

    x_train = clinical_ds.X.iloc[split.train]
    y_train = clinical_ds.y.iloc[split.train]
    x_test = clinical_ds.X.iloc[split.test]
    y_test = clinical_ds.y.iloc[split.test]

    # ------------------------------------------------------------------
    # 3. Train models and collect results
    # ------------------------------------------------------------------
    model_names = list_models()
    if not model_names:
        click.echo("Warning: no models registered.", err=True)

    output_dir = Path(output).parent.resolve()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    models_results: dict[str, dict[str, Any]] = {}

    for model_name in model_names:
        click.echo(f"Training {model_name} ...")
        try:
            model_cls = _get_model_class(model_name)
            model = model_cls()
            model.fit(x_train, y_train)
            y_prob = model.predict_proba(x_test)[:, 1]
        except Exception as exc:
            warnings.warn(
                f"Model '{model_name}' failed during training or prediction: {exc}",
                stacklevel=2,
            )
            continue

        # Global metrics
        metrics = {
            "auc": auc_score(y_test, y_prob),
            "auprc": auprc_score(y_test, y_prob),
            "brier": brier_score(y_test, y_prob),
            "log_loss": _safe_log_loss(y_test, y_prob),
            "ece": expected_calibration_error(y_test, y_prob),
        }

        # Fairness metrics per group column
        fairness: dict[str, dict[str, Any]] = {}
        for group_col in clinical_ds.group_cols:
            group_values = x_test[group_col].values
            fairness[group_col] = {
                "demographic_parity": demographic_parity_difference(
                    y_test, y_prob, group_values
                ),
                "equalized_odds": equalized_odds_difference(
                    y_test, y_prob, group_values
                ),
                "calibration_by_group": calibration_by_group(
                    y_test, y_prob, group_values
                ),
            }

        # SHAP explanations (on a subset to keep runtime reasonable)
        shap_global: pd.DataFrame | None = None
        shap_local: pd.DataFrame | None = None
        try:
            shap_subset = x_test.head(min(100, len(x_test)))
            shap_global = shap_global_importance(model, shap_subset, background_samples=50)
            shap_local = shap_local_explanations(
                model, shap_subset, background_samples=50, top_k=5
            )
        except Exception as exc:
            warnings.warn(
                f"SHAP explanations failed for '{model_name}': {exc}",
                stacklevel=2,
            )

        # Figures
        model_fig_dir = figures_dir / model_name
        model_fig_dir.mkdir(parents=True, exist_ok=True)

        calib_path = model_fig_dir / "calibration.png"
        group_for_plot = (
            x_test[clinical_ds.group_cols[0]].values if clinical_ds.group_cols else None
        )
        plot_calibration_curves(y_test, y_prob, group_for_plot, calib_path)

        fairness_path = model_fig_dir / "fairness.png"
        plot_fairness_comparison({model_name: _flatten_fairness(fairness)}, fairness_path)

        shap_path = model_fig_dir / "shap.png"
        if shap_global is not None and not shap_global.empty:
            plot_shap_summary(shap_global, shap_path)
        else:
            shap_path = None  # type: ignore[assignment]

        models_results[model_name] = {
            "metrics": metrics,
            "fairness": fairness,
            "shap_global": shap_global,
            "shap_local": shap_local,
            "figures": {
                "calibration": str(calib_path),
                "fairness": str(fairness_path),
                "shap": str(shap_path) if shap_path else None,
            },
        }

    # ------------------------------------------------------------------
    # 4. Render report
    # ------------------------------------------------------------------
    results = {
        "dataset": {
            "name": clinical_ds.name,
            "description": clinical_ds.description,
            "n_samples": len(clinical_ds.X),
            "n_features": len(clinical_ds.X.columns),
            "group_cols": clinical_ds.group_cols,
        },
        "models": models_results,
    }

    render_report(results, output)
    click.echo(f"Report written to: {output}")


def _load_builtin(name: str) -> ClinicalDataset:
    """Load a built-in dataset by name."""
    if name == "pima":
        return load_pima()
    if name == "heart":
        return load_heart()
    if name == "diabetes130":
        return load_diabetes130()
    raise ValueError(f"Unknown built-in dataset: {name}")


def _get_model_class(model_name: str) -> Any:
    """Import and return the model class for *model_name*.

    We import here rather than at module top-level so that the CLI file
    does not force-load every model implementation on simple ``--help``.
    """
    from fair_clinical_bench.models.registry import get_model

    return get_model(model_name)


def _safe_log_loss(y_true: np.ndarray | pd.Series, y_prob: np.ndarray) -> float:
    """Compute log-loss, guarding against all-identical labels."""
    from fair_clinical_bench.metrics.calibration import log_loss_score

    try:
        return log_loss_score(y_true, y_prob)
    except ValueError:
        return float("nan")


def _flatten_fairness(fairness: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Flatten nested fairness dict for figure generation."""
    flat: dict[str, Any] = {}
    for group_col, metrics in fairness.items():
        for metric_name, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    flat[f"{group_col}/{metric_name}/{sub_key}"] = sub_val
            else:
                flat[f"{group_col}/{metric_name}"] = value
    return flat


def main(argv: list[str] | None = None) -> int:
    """Entry point used by the ``fair-bench`` console script."""
    try:
        cli(args=argv)
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, int):
            return code
        return 0 if code is None else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
