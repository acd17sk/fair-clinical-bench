"""Metrics layer for fair-clinical-bench."""

from fair_clinical_bench.metrics.calibration import (
    brier_score,
    expected_calibration_error,
    log_loss_score,
    reliability_curve,
)
from fair_clinical_bench.metrics.discrimination import auc_score, auprc_score
from fair_clinical_bench.metrics.fairness import (
    calibration_by_group,
    demographic_parity_difference,
    equalized_odds_difference,
)

__all__ = [
    "auc_score",
    "auprc_score",
    "brier_score",
    "log_loss_score",
    "expected_calibration_error",
    "reliability_curve",
    "demographic_parity_difference",
    "equalized_odds_difference",
    "calibration_by_group",
]
