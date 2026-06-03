"""Tests for fair_clinical_bench.explain — SHAP local and global explanations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fair_clinical_bench.explain.global_ import shap_global_importance
from fair_clinical_bench.explain.local import shap_local_explanations
from fair_clinical_bench.models.logistic_regression import LogisticRegressionModel
from fair_clinical_bench.models.random_forest import RandomForestModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_numeric_data(n: int = 50, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Create a synthetic all-numeric dataset."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(  # noqa: N806
        {
            "age": rng.integers(20, 90, size=n),
            "bmi": rng.uniform(18.0, 40.0, size=n).round(1),
            "systolic": rng.integers(90, 180, size=n),
        }
    )
    y = pd.Series(rng.integers(0, 2, size=n), name="target")
    return X, y


# ---------------------------------------------------------------------------
# Local explanations
# ---------------------------------------------------------------------------


class TestLocalExplanations:
    """Tests for shap_local_explanations."""

    def test_local_returns_dataframe(self) -> None:
        X, y = _make_numeric_data(n=30)  # noqa: N806
        model = LogisticRegressionModel(random_state=42)
        model.fit(X, y)
        result = shap_local_explanations(
            model, X, background_samples=10, top_k=2
        )
        assert isinstance(result, pd.DataFrame)
        assert "instance_index" in result.columns
        assert "predicted_risk" in result.columns
        assert "feature_name" in result.columns
        assert "shap_value" in result.columns

    def test_local_top_k_rows(self) -> None:
        X, y = _make_numeric_data(n=30)  # noqa: N806
        model = LogisticRegressionModel(random_state=42)
        model.fit(X, y)
        top_k = 2
        result = shap_local_explanations(
            model, X, background_samples=10, top_k=top_k
        )
        # Each of the top_k instances should have top_k feature rows
        assert len(result) == top_k * top_k

    def test_local_highest_risk_first(self) -> None:
        X, y = _make_numeric_data(n=30)  # noqa: N806
        model = LogisticRegressionModel(random_state=42)
        model.fit(X, y)
        result = shap_local_explanations(
            model, X, background_samples=10, top_k=2
        )
        risks = result.groupby("instance_index", sort=False)["predicted_risk"].first()
        assert risks.iloc[0] >= risks.iloc[1]

    def test_local_with_random_forest(self) -> None:
        X, y = _make_numeric_data(n=30)  # noqa: N806
        model = RandomForestModel(random_state=42)
        model.fit(X, y)
        result = shap_local_explanations(
            model, X, background_samples=10, top_k=2
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_local_non_dataframe_raises(self) -> None:
        model = LogisticRegressionModel(random_state=42)
        with pytest.raises(TypeError, match="pandas DataFrame"):
            shap_local_explanations(model, np.zeros((5, 3)))

    def test_local_empty_fallback(self) -> None:
        """Very small datasets should still produce a result."""
        X, y = _make_numeric_data(n=6)  # noqa: N806
        model = LogisticRegressionModel(random_state=42)
        model.fit(X, y)
        result = shap_local_explanations(
            model, X, background_samples=5, top_k=2
        )
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Global explanations
# ---------------------------------------------------------------------------


class TestGlobalExplanations:
    """Tests for shap_global_importance."""

    def test_global_returns_dataframe(self) -> None:
        X, y = _make_numeric_data(n=30)  # noqa: N806
        model = LogisticRegressionModel(random_state=42)
        model.fit(X, y)
        result = shap_global_importance(model, X, background_samples=10)
        assert isinstance(result, pd.DataFrame)
        assert "feature_name" in result.columns
        assert "mean_abs_shap" in result.columns

    def test_global_sorted_descending(self) -> None:
        X, y = _make_numeric_data(n=30)  # noqa: N806
        model = LogisticRegressionModel(random_state=42)
        model.fit(X, y)
        result = shap_global_importance(model, X, background_samples=10)
        mean_abs = result["mean_abs_shap"].values
        assert np.all(mean_abs[:-1] >= mean_abs[1:])

    def test_global_all_features_present(self) -> None:
        X, y = _make_numeric_data(n=30)  # noqa: N806
        model = LogisticRegressionModel(random_state=42)
        model.fit(X, y)
        result = shap_global_importance(model, X, background_samples=10)
        assert set(result["feature_name"]) == set(X.columns)

    def test_global_with_random_forest(self) -> None:
        X, y = _make_numeric_data(n=30)  # noqa: N806
        model = RandomForestModel(random_state=42)
        model.fit(X, y)
        result = shap_global_importance(model, X, background_samples=10)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(X.columns)

    def test_global_non_dataframe_raises(self) -> None:
        model = LogisticRegressionModel(random_state=42)
        with pytest.raises(TypeError, match="pandas DataFrame"):
            shap_global_importance(model, np.zeros((5, 3)))

    def test_global_empty_fallback(self) -> None:
        X, y = _make_numeric_data(n=6)  # noqa: N806
        model = LogisticRegressionModel(random_state=42)
        model.fit(X, y)
        result = shap_global_importance(model, X, background_samples=5)
        assert isinstance(result, pd.DataFrame)
