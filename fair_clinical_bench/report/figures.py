"""Figure generation for fair-clinical-bench reports.

Produces matplotlib visualisations for calibration curves, fairness tables,
and SHAP summary plots.  Every function accepts raw results, writes a figure
to disk, and returns the file path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fair_clinical_bench.metrics.calibration import reliability_curve

matplotlib.use("Agg")


def plot_calibration_curves(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    group: np.ndarray | pd.Series | None,
    output_path: str | Path,
    n_bins: int = 10,
) -> str:
    """Plot calibration (reliability) curves per demographic group.

    Parameters
    ----------
    y_true :
        Ground-truth binary labels.
    y_prob :
        Predicted probabilities for the positive class.
    group :
        Group membership for each sample, or *None* for a single curve.
    output_path :
        Destination file path (``.png`` recommended).
    n_bins :
        Number of equal-width probability bins.

    Returns
    -------
    str
        Absolute path to the saved figure.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

    if group is None:
        prob_true, prob_pred, _bin_counts = reliability_curve(
            y_true, y_prob, n_bins=n_bins
        )
        valid = ~np.isnan(prob_true)
        ax.plot(
            prob_pred[valid],
            prob_true[valid],
            "o-",
            label="All data",
        )
    else:
        group_arr = np.asarray(group)
        group_values = np.unique(group_arr)
        for g in group_values:
            mask = group_arr == g
            prob_true, prob_pred, _bin_counts = reliability_curve(
                y_true[mask], y_prob[mask], n_bins=n_bins
            )
            valid = ~np.isnan(prob_true)
            ax.plot(
                prob_pred[valid],
                prob_true[valid],
                "o-",
                label=str(g),
            )

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def plot_fairness_comparison(
    fairness_results: dict[str, dict[str, Any]],
    output_path: str | Path,
) -> str:
    """Render a fairness-metric comparison table as a matplotlib figure.

    Parameters
    ----------
    fairness_results :
        Mapping ``model_name -> {metric_name -> value}``.  Values may be
        floats or nested dicts (e.g. calibration-by-group); nested dicts are
        flattened with ``/`` separators.
    output_path :
        Destination file path.

    Returns
    -------
    str
        Absolute path to the saved figure.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Guard against empty fairness data
    if not any(metrics for metrics in fairness_results.values()):
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "No fairness metrics available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return str(output_path)

    # Flatten nested dicts so every cell is scalar
    rows: list[dict[str, Any]] = []
    for model_name, metrics in fairness_results.items():
        row: dict[str, Any] = {"Model": model_name}
        for metric_name, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    row[f"{metric_name}/{sub_key}"] = sub_val
            else:
                row[metric_name] = value
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index("Model")

    fig, ax = plt.subplots(figsize=(max(6, len(df.columns) * 1.5), max(3, len(df) * 0.5)))
    ax.axis("off")

    table = ax.table(
        cellText=df.round(4).values,
        rowLabels=df.index,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    ax.set_title("Fairness Comparison", fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def plot_shap_summary(
    shap_df: pd.DataFrame,
    output_path: str | Path,
    top_k: int = 10,
) -> str:
    """Render a horizontal bar plot of mean absolute SHAP values.

    Parameters
    ----------
    shap_df :
        DataFrame with at least columns ``feature_name`` and ``mean_abs_shap``.
    output_path :
        Destination file path.
    top_k :
        Number of top features to display.

    Returns
    -------
    str
        Absolute path to the saved figure.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = shap_df.copy()
    df = df.sort_values("mean_abs_shap", ascending=True).tail(top_k)

    fig, ax = plt.subplots(figsize=(6, max(3, top_k * 0.4)))
    ax.barh(df["feature_name"], df["mean_abs_shap"])
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(f"Top-{top_k} SHAP Feature Importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)
