"""Discrimination metrics for fair-clinical-bench."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def auc_score(y_true: np.ndarray | pd.Series, y_prob: np.ndarray) -> float:
    """Compute the Area Under the ROC Curve (AUC).

    Parameters
    ----------
    y_true : np.ndarray | pd.Series
        Ground-truth binary labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        AUC score in the range [0, 1].
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    if len(np.unique(y_true_arr)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true_arr, y_prob_arr))


def auprc_score(y_true: np.ndarray | pd.Series, y_prob: np.ndarray) -> float:
    """Compute the Area Under the Precision-Recall Curve (AUPRC).

    Parameters
    ----------
    y_true : np.ndarray | pd.Series
        Ground-truth binary labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        AUPRC score in the range [0, 1].
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    if len(np.unique(y_true_arr)) < 2:
        return float("nan")
    return float(average_precision_score(y_true_arr, y_prob_arr))
