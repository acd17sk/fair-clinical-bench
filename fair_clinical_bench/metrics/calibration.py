"""Calibration metrics for fair-clinical-bench."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


def brier_score(y_true: np.ndarray | pd.Series, y_prob: np.ndarray) -> float:
    """Compute the Brier score.

    Parameters
    ----------
    y_true : np.ndarray | pd.Series
        Ground-truth binary labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        Brier score in the range [0, 1]; lower is better.
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    return float(brier_score_loss(y_true_arr, y_prob_arr))


def log_loss_score(y_true: np.ndarray | pd.Series, y_prob: np.ndarray) -> float:
    """Compute the logistic loss (cross-entropy).

    Parameters
    ----------
    y_true : np.ndarray | pd.Series
        Ground-truth binary labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        Log-loss value; lower is better.
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    # Clip probabilities to avoid log(0)
    y_prob_clipped = np.clip(y_prob_arr, 1e-15, 1 - 1e-15)
    return float(log_loss(y_true_arr, y_prob_clipped))


def expected_calibration_error(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) with equal-width bins.

    Parameters
    ----------
    y_true : np.ndarray | pd.Series
        Ground-truth binary labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    n_bins : int, optional
        Number of equal-width probability bins, by default 10.

    Returns
    -------
    float
        ECE value in the range [0, 1]; lower is better.
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:], strict=False):
        in_bin = (y_prob_arr > lower) & (y_prob_arr <= upper)
        if lower == 0.0:
            in_bin = (y_prob_arr >= lower) & (y_prob_arr <= upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0.0:
            avg_confidence = y_prob_arr[in_bin].mean()
            avg_accuracy = y_true_arr[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
    return float(ece)


def reliability_curve(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a reliability curve (calibration curve) with equal-width bins.

    Parameters
    ----------
    y_true : np.ndarray | pd.Series
        Ground-truth binary labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    n_bins : int, optional
        Number of equal-width probability bins, by default 10.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - prob_true : fraction of positives in each bin.
        - prob_pred : mean predicted probability in each bin.
        - bin_counts : number of samples in each bin.
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    prob_true = np.zeros(n_bins, dtype=float)
    prob_pred = np.zeros(n_bins, dtype=float)
    bin_counts = np.zeros(n_bins, dtype=int)

    for idx, (lower, upper) in enumerate(
        zip(bin_boundaries[:-1], bin_boundaries[1:], strict=False)
    ):
        if lower == 0.0:
            in_bin = (y_prob_arr >= lower) & (y_prob_arr <= upper)
        else:
            in_bin = (y_prob_arr > lower) & (y_prob_arr <= upper)
        count = int(in_bin.sum())
        bin_counts[idx] = count
        if count > 0:
            prob_true[idx] = float(y_true_arr[in_bin].mean())
            prob_pred[idx] = float(y_prob_arr[in_bin].mean())
        else:
            prob_true[idx] = float("nan")
            prob_pred[idx] = float("nan")

    return prob_true, prob_pred, bin_counts
