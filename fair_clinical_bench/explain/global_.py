"""Global SHAP explanations for fair-clinical-bench."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import shap


def shap_global_importance(
    model: Any,
    X: pd.DataFrame,  # noqa: N803
    background_samples: int = 100,
) -> pd.DataFrame:
    """Compute global feature importance using mean absolute SHAP values.

    Parameters
    ----------
    model : Any
        A fitted model with a ``predict_proba`` method.
    X : pd.DataFrame
        Feature matrix for which to compute global importance.
    background_samples : int, optional
        Number of background samples for the SHAP explainer, by default 100.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``feature_name`` and ``mean_abs_shap``,
        sorted by descending importance.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame for SHAP global explanations.")

    # Build background data
    n_bg = min(background_samples, len(X))
    bg_indices = np.random.choice(len(X), size=n_bg, replace=False)
    X_background = X.iloc[bg_indices]  # noqa: N806

    # Compute SHAP values
    try:
        explainer = shap.Explainer(model.predict_proba, X_background)
        shap_values = explainer(X)
    except Exception as exc:
        warnings.warn(
            f"shap.Explainer failed ({exc}); falling back to PermutationExplainer.",
            stacklevel=2,
        )
        try:
            explainer = shap.PermutationExplainer(model.predict_proba, X_background)
            shap_values = explainer(X)
        except Exception as fallback_exc:
            warnings.warn(
                f"PermutationExplainer also failed ({fallback_exc}); "
                "returning empty importance.",
                stacklevel=2,
            )
            return pd.DataFrame(columns=["feature_name", "mean_abs_shap"])

    # Extract SHAP values for the positive class
    if isinstance(shap_values, shap.Explanation):
        values = shap_values.values
        if values.ndim == 3:
            values = values[:, :, 1]
        feature_names = list(shap_values.feature_names or X.columns)
    else:
        values = np.asarray(shap_values)
        if values.ndim == 3:
            values = values[:, :, 1]
        feature_names = list(X.columns)

    mean_abs = np.abs(values).mean(axis=0)
    result = pd.DataFrame(
        {
            "feature_name": feature_names,
            "mean_abs_shap": mean_abs,
        }
    )
    return result.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
