"""Markdown report renderer for fair-clinical-bench.

Composes model results, metrics, and figure paths into a single
comprehensive Markdown document.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def render_report(
    results: dict[str, Any],
    output_path: str | Path,
) -> str:
    """Render a comprehensive Markdown report from benchmark results.

    Parameters
    ----------
    results :
        Structured results dictionary.  Expected top-level keys:
        ``dataset`` and ``models``.
    output_path :
        Destination file path (``.md`` recommended).

    Returns
    -------
    str
        Absolute path to the saved report.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    _render_header(lines, results.get("dataset", {}))
    _render_global_metrics(lines, results.get("models", {}))
    _render_calibration(lines, results.get("models", {}))
    _render_fairness(lines, results.get("models", {}))
    _render_shap(lines, results.get("models", {}))

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return str(output_path)


def _render_header(lines: list[str], dataset_info: dict[str, Any]) -> None:
    lines.append("# Fair Clinical Bench Report")
    lines.append("")
    lines.append(f"**Dataset:** {dataset_info.get('name', 'Unknown')}")
    lines.append("")
    if dataset_info.get("description"):
        lines.append(f"{dataset_info['description']}")
        lines.append("")
    lines.append(f"- Samples: {dataset_info.get('n_samples', 'N/A')}")
    lines.append(f"- Features: {dataset_info.get('n_features', 'N/A')}")
    group_cols = dataset_info.get("group_cols", [])
    if group_cols:
        lines.append(f"- Group columns: {', '.join(group_cols)}")
    else:
        lines.append("- Group columns: (none)")
    lines.append("")


def _render_global_metrics(lines: list[str], models: dict[str, Any]) -> None:
    lines.append("## Global Metrics")
    lines.append("")

    rows: list[dict[str, Any]] = []
    for model_name, model_data in models.items():
        metrics = model_data.get("metrics", {})
        rows.append(
            {
                "Model": model_name,
                "AUC": metrics.get("auc"),
                "AUPRC": metrics.get("auprc"),
                "Brier": metrics.get("brier"),
                "Log-Loss": metrics.get("log_loss"),
                "ECE": metrics.get("ece"),
            }
        )

    if rows:
        df = pd.DataFrame(rows).set_index("Model")
        lines.append(_dataframe_to_markdown(df))
    else:
        lines.append("*No models evaluated.*")
    lines.append("")


def _render_calibration(lines: list[str], models: dict[str, Any]) -> None:
    lines.append("## Calibration Curves")
    lines.append("")

    for model_name, model_data in models.items():
        fig_path = model_data.get("figures", {}).get("calibration")
        if fig_path:
            lines.append(f"### {model_name}")
            lines.append("")
            rel_path = Path(fig_path).name
            lines.append(f"![Calibration curve for {model_name}]({rel_path})")
            lines.append("")

    if not any(
        model_data.get("figures", {}).get("calibration")
        for model_data in models.values()
    ):
        lines.append("*No calibration curves available.*")
        lines.append("")


def _render_fairness(lines: list[str], models: dict[str, Any]) -> None:
    lines.append("## Fairness Metrics")
    lines.append("")

    for model_name, model_data in models.items():
        fairness = model_data.get("fairness", {})
        if not fairness:
            continue

        lines.append(f"### {model_name}")
        lines.append("")

        for group_col, group_metrics in fairness.items():
            lines.append(f"#### Group: {group_col}")
            lines.append("")

            dp = group_metrics.get("demographic_parity")
            eo = group_metrics.get("equalized_odds")
            cbp = group_metrics.get("calibration_by_group", {})

            rows: list[dict[str, Any]] = []
            if dp is not None:
                rows.append({"Metric": "Demographic Parity", "Value": dp})
            if eo is not None:
                rows.append({"Metric": "Equalized Odds", "Value": eo})
            for group_val, ece_val in cbp.items():
                rows.append({"Metric": f"ECE ({group_val})", "Value": ece_val})

            if rows:
                df = pd.DataFrame(rows).set_index("Metric")
                lines.append(_dataframe_to_markdown(df))
            else:
                lines.append("*No fairness metrics available for this group.*")
            lines.append("")

        fig_path = model_data.get("figures", {}).get("fairness")
        if fig_path:
            rel_path = Path(fig_path).name
            lines.append(f"![Fairness comparison for {model_name}]({rel_path})")
            lines.append("")

    if not any(model_data.get("fairness") for model_data in models.values()):
        lines.append("*No fairness metrics available (no group columns defined).*")
        lines.append("")


def _render_shap(lines: list[str], models: dict[str, Any]) -> None:
    lines.append("## SHAP Explanations")
    lines.append("")

    for model_name, model_data in models.items():
        lines.append(f"### {model_name}")
        lines.append("")

        shap_global = model_data.get("shap_global")
        if shap_global is not None and not shap_global.empty:
            lines.append("#### Global Feature Importance")
            lines.append("")
            top_k = shap_global.head(10)
            lines.append(_dataframe_to_markdown(top_k.set_index("feature_name")))
            lines.append("")

        fig_path = model_data.get("figures", {}).get("shap")
        if fig_path:
            rel_path = Path(fig_path).name
            lines.append(f"![SHAP summary for {model_name}]({rel_path})")
            lines.append("")

        shap_local = model_data.get("shap_local")
        if shap_local is not None and not shap_local.empty:
            lines.append("#### Local Explanations (Top Risk Instances)")
            lines.append("")
            lines.append(_dataframe_to_markdown(shap_local.head(20)))
            lines.append("")

    if not any(
        model_data.get("shap_global") is not None
        or model_data.get("shap_local") is not None
        for model_data in models.values()
    ):
        lines.append("*No SHAP explanations available.*")
        lines.append("")


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a simple Markdown table."""
    if df.empty:
        return "*Empty table.*"

    lines: list[str] = []
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    lines.append(header)
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines.append(separator)
    for idx, row in df.iterrows():
        vals = [str(idx)] + [str(v) for v in row.values]
        row_str = "| " + " | ".join(vals) + " |"
        lines.append(row_str)
    return "\n".join(lines)
