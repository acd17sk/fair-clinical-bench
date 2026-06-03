"""Local SHAP explanations for fair-clinical-bench."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import shap


def shap_local_explanations(
    model: Any,
    X: pd.DataFrame,  # noqa: N803
    background_samples: int = 100,
    top_k: int = 5,
) -> pd.DataFrame:
    """Generate SHAP-based local explanations for the highest-risk predictions.

    Parameters
    ----------
    model : Any
        A fitted model with a ``predict_proba`` method.
    X : pd.DataFrame
        Feature matrix for which to compute explanations.
    background_samples : int, optional
        Number of background samples for the SHAP explainer, by default 100.
    top_k : int, optional
        Number of top features to report per explanation, by default 5.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per highest-risk prediction and columns:
        ``instance_index``, ``predicted_risk``, ``feature_name``,
        ``shap_value``.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame for SHAP local explanations.")

    # Compute predicted risk (positive-class probability)
    proba = model.predict_proba(X)
    risk = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.ravel()

    # Select top-K highest-risk instances
    n_top = min(top_k, len(X))
    top_indices = np.argsort(risk)[-n_top:][::-1]

    # Build background data
    n_bg = min(background_samples, len(X))
    bg_indices = np.random.choice(len(X), size=n_bg, replace=False)
    X_background = X.iloc[bg_indices]  # noqa: N806

    # Compute SHAP values
    try:
        explainer = shap.Explainer(model.predict_proba, X_background)
        shap_values = explainer(X.iloc[top_indices])
    except Exception as exc:
        warnings.warn(
            f"shap.Explainer failed ({exc}); falling back to PermutationExplainer.",
            stacklevel=2,
        )
        try:
            explainer = shap.PermutationExplainer(model.predict_proba, X_background)
            shap_values = explainer(X.iloc[top_indices])
        except Exception as fallback_exc:
            warnings.warn(
                f"PermutationExplainer also failed ({fallback_exc}); "
                "returning empty explanations.",
                stacklevel=2,
            )
            return pd.DataFrame(
                columns=["instance_index", "predicted_risk", "feature_name", "shap_value"]
            )

    # Extract SHAP values for the positive class
    if isinstance(shap_values, shap.Explanation):
        values = shap_values.values
        if values.ndim == 3:
            # (samples, features, classes) — take positive class
            values = values[:, :, 1]
        feature_names = list(shap_values.feature_names or X.columns)
    else:
        values = np.asarray(shap_values)
        if values.ndim == 3:
            values = values[:, :, 1]
        feature_names = list(X.columns)

    rows: list[dict[str, Any]] = []
    for rank, idx in enumerate(top_indices):
        instance_values = values[rank]
        abs_values = np.abs(instance_values)
        top_feature_indices = np.argsort(abs_values)[-top_k:][::-1]
        for fidx in top_feature_indices:
            rows.append(
                {
                    "instance_index": int(idx),
                    "predicted_risk": float(risk[idx]),
                    "feature_name": str(feature_names[fidx]),
                    "shap_value": float(instance_values[fidx]),
                }
            )

    return pd.DataFrame(rows)
