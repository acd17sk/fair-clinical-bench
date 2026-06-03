"""Tests for fair_clinical_bench.metrics — discrimination, calibration, and fairness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_perfect_predictions(n: int = 100, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Create labels and perfectly calibrated predictions."""
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    y_prob = y_true.astype(float) * 0.9 + (1 - y_true) * 0.1
    return y_true, y_prob


def _make_random_predictions(n: int = 100, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Create labels and random predictions (AUC ~ 0.5)."""
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    y_prob = rng.uniform(0, 1, size=n)
    return y_true, y_prob


# ---------------------------------------------------------------------------
# Discrimination
# ---------------------------------------------------------------------------


class TestDiscriminationMetrics:
    """Tests for AUC and AUPRC."""

    def test_auc_perfect_predictions(self) -> None:
        y_true, y_prob = _make_perfect_predictions(n=100)
        score = auc_score(y_true, y_prob)
        assert score > 0.95

    def test_auc_random_predictions(self) -> None:
        y_true, y_prob = _make_random_predictions(n=1000)
        score = auc_score(y_true, y_prob)
        assert 0.45 <= score <= 0.55

    def test_auc_single_class_returns_nan(self) -> None:
        y_true = np.zeros(10, dtype=int)
        y_prob = np.ones(10) * 0.5
        score = auc_score(y_true, y_prob)
        assert np.isnan(score)

    def test_auprc_perfect_predictions(self) -> None:
        y_true, y_prob = _make_perfect_predictions(n=100)
        score = auprc_score(y_true, y_prob)
        assert score > 0.95

    def test_auprc_random_predictions(self) -> None:
        y_true, y_prob = _make_random_predictions(n=1000)
        score = auprc_score(y_true, y_prob)
        # AUPRC for random predictions should be close to prevalence
        prevalence = y_true.mean()
        assert 0.0 <= score <= max(prevalence * 2, 0.5)

    def test_auprc_single_class_returns_nan(self) -> None:
        y_true = np.ones(10, dtype=int)
        y_prob = np.ones(10) * 0.9
        score = auprc_score(y_true, y_prob)
        assert np.isnan(score)

    def test_auc_with_pandas_series(self) -> None:
        y_true = pd.Series([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        score = auc_score(y_true, y_prob)
        assert score == 1.0

    def test_auprc_with_pandas_series(self) -> None:
        y_true = pd.Series([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        score = auprc_score(y_true, y_prob)
        assert score == 1.0


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


class TestCalibrationMetrics:
    """Tests for Brier score, log-loss, ECE, and reliability curve."""

    def test_brier_score_perfect(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        score = brier_score(y_true, y_prob)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_brier_score_random(self) -> None:
        y_true, y_prob = _make_random_predictions(n=1000)
        score = brier_score(y_true, y_prob)
        assert 0.15 <= score <= 0.4

    def test_log_loss_perfect(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        score = log_loss_score(y_true, y_prob)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_log_loss_clips_extremes(self) -> None:
        y_true = np.array([0, 1])
        y_prob = np.array([0.0, 1.0])
        # Should not raise or return inf because of clipping
        score = log_loss_score(y_true, y_prob)
        assert np.isfinite(score)

    def test_ece_perfectly_calibrated(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        score = expected_calibration_error(y_true, y_prob, n_bins=2)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_ece_random(self) -> None:
        y_true, y_prob = _make_random_predictions(n=1000)
        score = expected_calibration_error(y_true, y_prob, n_bins=10)
        assert 0.0 <= score <= 0.5

    def test_reliability_curve_shape(self) -> None:
        y_true, y_prob = _make_random_predictions(n=100)
        prob_true, prob_pred, bin_counts = reliability_curve(
            y_true, y_prob, n_bins=10
        )
        assert prob_true.shape == (10,)
        assert prob_pred.shape == (10,)
        assert bin_counts.shape == (10,)
        assert bin_counts.sum() == 100

    def test_reliability_curve_empty_bins_are_nan(self) -> None:
        y_true = np.array([0, 1])
        y_prob = np.array([0.1, 0.9])
        prob_true, prob_pred, bin_counts = reliability_curve(
            y_true, y_prob, n_bins=10
        )
        # Most bins should be empty
        empty_bins = bin_counts == 0
        assert np.all(np.isnan(prob_true[empty_bins]))
        assert np.all(np.isnan(prob_pred[empty_bins]))

    def test_reliability_curve_with_pandas(self) -> None:
        y_true = pd.Series([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        prob_true, prob_pred, bin_counts = reliability_curve(
            y_true, y_prob, n_bins=2
        )
        assert bin_counts.sum() == 4


# ---------------------------------------------------------------------------
# Fairness
# ---------------------------------------------------------------------------


class TestFairnessMetrics:
    """Tests for demographic parity, equalized odds, and calibration-by-group."""

    def test_demographic_parity_difference_zero_when_equal(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])
        group = np.array(["A", "A", "B", "B"])
        diff = demographic_parity_difference(y_true, y_prob, group)
        assert diff == pytest.approx(0.0, abs=1e-6)

    def test_demographic_parity_difference_nonzero(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.3, 0.3, 0.7, 0.7])
        group = np.array(["A", "A", "B", "B"])
        diff = demographic_parity_difference(y_true, y_prob, group)
        assert diff == pytest.approx(0.4, abs=1e-6)

    def test_demographic_parity_single_group_returns_zero(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.3, 0.7, 0.5, 0.5])
        group = np.array(["A", "A", "A", "A"])
        diff = demographic_parity_difference(y_true, y_prob, group)
        assert diff == 0.0

    def test_equalized_odds_difference_zero_when_equal(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.1, 0.9, 0.9])
        group = np.array(["A", "B", "A", "B"])
        diff = equalized_odds_difference(y_true, y_prob, group)
        assert diff == pytest.approx(0.0, abs=1e-6)

    def test_equalized_odds_difference_nonzero(self) -> None:
        # Group A: TPR=1.0, FPR=0.0  (perfect for both classes)
        # Group B: TPR=0.0, FPR=1.0  (inverted for both classes)
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_prob = np.array([0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1])
        group = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
        diff = equalized_odds_difference(y_true, y_prob, group)
        assert diff == pytest.approx(1.0, abs=1e-6)

    def test_equalized_odds_single_group_returns_zero(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        group = np.array(["A", "A", "A", "A"])
        diff = equalized_odds_difference(y_true, y_prob, group)
        assert diff == 0.0

    def test_calibration_by_group(self) -> None:
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
        group = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
        result = calibration_by_group(y_true, y_prob, group, n_bins=2)
        assert "A" in result
        assert "B" in result
        assert all(np.isfinite(v) for v in result.values())

    def test_calibration_by_group_single_group(self) -> None:
        y_true = np.array([0, 1])
        y_prob = np.array([0.1, 0.9])
        group = np.array(["A", "A"])
        result = calibration_by_group(y_true, y_prob, group, n_bins=2)
        assert "A" in result

    def test_fairness_with_pandas(self) -> None:
        y_true = pd.Series([0, 1, 0, 1])
        y_prob = np.array([0.3, 0.3, 0.7, 0.7])
        group = pd.Series(["A", "A", "B", "B"])
        diff = demographic_parity_difference(y_true, y_prob, group)
        assert diff == pytest.approx(0.4, abs=1e-6)
