"""Report layer for fair-clinical-bench."""

from fair_clinical_bench.report.figures import (
    plot_calibration_curves,
    plot_fairness_comparison,
    plot_shap_summary,
)
from fair_clinical_bench.report.renderer import render_report

__all__ = [
    "plot_calibration_curves",
    "plot_fairness_comparison",
    "plot_shap_summary",
    "render_report",
]
