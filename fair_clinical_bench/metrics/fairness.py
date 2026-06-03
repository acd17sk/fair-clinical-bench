"""Fairness metrics for fair-clinical-bench."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Internal ECE implementation (duplicated to keep module self-contained)."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:], strict=False):
        in_bin = (y_prob > lower) & (y_prob <= upper)
        if lower == 0.0:
            in_bin = (y_prob >= lower) & (y_prob <= upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0.0:
            avg_confidence = y_prob[in_bin].mean()
            avg_accuracy = y_true[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
    return float(ece)


def demographic_parity_difference(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    group: np.ndarray | pd.Series,
) -> float:
    """Compute demographic parity difference across groups.

    Demographic parity requires that the mean predicted probability is equal
    across all groups. This metric returns the maximum absolute difference
    between any two group means.

    Parameters
    ----------
    y_true : np.ndarray | pd.Series
        Ground-truth binary labels (not used in computation, kept for API
        consistency).
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    group : np.ndarray | pd.Series
        Group membership for each sample.

    Returns
    -------
    float
        Maximum absolute difference in mean predicted probability between any
        two groups. Range [0, 1]; lower is better.
    """
    y_prob_arr = np.asarray(y_prob)
    group_arr = np.asarray(group)
    group_values = np.unique(group_arr)
    if len(group_values) < 2:
        return 0.0
    means = [y_prob_arr[group_arr == g].mean() for g in group_values]
    return float(max(means) - min(means))


def equalized_odds_difference(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    group: np.ndarray | pd.Series,
    threshold: float = 0.5,
) -> float:
    """Compute equalized odds difference across groups.

    Equalized odds requires that TPR and FPR are equal across groups. This
    metric returns the maximum of the TPR difference and FPR difference
    between any two groups.

    Parameters
    ----------
    y_true : np.ndarray | pd.Series
        Ground-truth binary labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    group : np.ndarray | pd.Series
        Group membership for each sample.
    threshold : float, optional
        Decision threshold for converting probabilities to binary predictions,
        by default 0.5.

    Returns
    -------
    float
        Maximum of TPR difference and FPR difference across groups.
        Range [0, 1]; lower is better.
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    group_arr = np.asarray(group)
    group_values = np.unique(group_arr)
    if len(group_values) < 2:
        return 0.0

    y_pred = (y_prob_arr >= threshold).astype(int)

    tprs: list[float] = []
    fprs: list[float] = []
    for g in group_values:
        mask = group_arr == g
        yt = y_true_arr[mask]
        yp = y_pred[mask]
        pos = yt.sum()
        neg = len(yt) - pos
        tpr = float((yp[yt == 1] == 1).sum() / pos) if pos > 0 else 0.0
        fpr = float((yp[yt == 0] == 1).sum() / neg) if neg > 0 else 0.0
        tprs.append(tpr)
        fprs.append(fpr)

    tpr_diff = max(tprs) - min(tprs)
    fpr_diff = max(fprs) - min(fprs)
    return float(max(tpr_diff, fpr_diff))


def calibration_by_group(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    group: np.ndarray | pd.Series,
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute Expected Calibration Error (ECE) per group.

    Parameters
    ----------
    y_true : np.ndarray | pd.Series
        Ground-truth binary labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    group : np.ndarray | pd.Series
        Group membership for each sample.
    n_bins : int, optional
        Number of equal-width probability bins, by default 10.

    Returns
    -------
    dict[str, float]
        Mapping from group value to ECE for that group.
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    group_arr = np.asarray(group)
    group_values = np.unique(group_arr)

    result: dict[str, float] = {}
    for g in group_values:
        mask = group_arr == g
        result[str(g)] = _expected_calibration_error(
            y_true_arr[mask], y_prob_arr[mask], n_bins=n_bins
        )
    return result
