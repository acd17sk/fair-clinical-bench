"""Tests for fair_clinical_bench.models — model wrappers, protocol, and registry."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from fair_clinical_bench.models.calibrated_mlp import CalibratedMLPModel
from fair_clinical_bench.models.logistic_regression import LogisticRegressionModel
from fair_clinical_bench.models.protocol import BenchmarkModel
from fair_clinical_bench.models.random_forest import RandomForestModel
from fair_clinical_bench.models.registry import get_model, list_models, register, unregister

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_numeric_data(n: int = 100, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
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


def _make_mixed_data(n: int = 100, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Create a synthetic dataset with numeric and categorical columns."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(  # noqa: N806
        {
            "age": rng.integers(20, 90, size=n),
            "sex": rng.choice(["M", "F"], size=n),
            "bmi": rng.uniform(18.0, 40.0, size=n).round(1),
            "race": rng.choice(["A", "B", "C"], size=n),
        }
    )
    y = pd.Series(rng.integers(0, 2, size=n), name="target")
    return X, y


def _make_all_categorical_data(n: int = 100, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Create a synthetic all-categorical dataset."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(  # noqa: N806
        {
            "sex": rng.choice(["M", "F"], size=n),
            "race": rng.choice(["A", "B", "C"], size=n),
            "smoker": rng.choice(["Y", "N"], size=n),
        }
    )
    y = pd.Series(rng.integers(0, 2, size=n), name="target")
    return X, y


def _assert_proba_valid(proba: np.ndarray, n_samples: int) -> None:
    """Assert that probability array has correct shape and valid values."""
    assert proba.shape == (n_samples, 2)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)
    np.testing.assert_array_almost_equal(proba.sum(axis=1), np.ones(n_samples), decimal=5)


# ---------------------------------------------------------------------------
# XGBoost availability guard
# ---------------------------------------------------------------------------


def _xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except Exception:
        return False


XGBOOST_AVAILABLE = _xgboost_available()

if XGBOOST_AVAILABLE:
    from fair_clinical_bench.models.xgboost import XGBoostModel
else:
    XGBoostModel = None  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    """Verify that concrete classes satisfy the BenchmarkModel protocol."""

    def test_logistic_regression_is_protocol(self) -> None:
        model = LogisticRegressionModel()
        assert isinstance(model, BenchmarkModel)

    def test_random_forest_is_protocol(self) -> None:
        model = RandomForestModel()
        assert isinstance(model, BenchmarkModel)

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="xgboost not available")
    def test_xgboost_is_protocol(self) -> None:
        model = XGBoostModel()
        assert isinstance(model, BenchmarkModel)

    def test_calibrated_mlp_is_protocol(self) -> None:
        model = CalibratedMLPModel()
        assert isinstance(model, BenchmarkModel)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    """Tests for the model registry."""

    def test_list_models_contains_expected(self) -> None:
        names = list_models()
        assert "logistic_regression" in names
        assert "random_forest" in names
        assert "calibrated_mlp" in names
        if XGBOOST_AVAILABLE:
            assert "xgboost" in names

    def test_list_models_is_sorted(self) -> None:
        names = list_models()
        assert names == sorted(names)

    def test_get_model_returns_class(self) -> None:
        cls = get_model("logistic_regression")
        assert cls is LogisticRegressionModel

    def test_get_model_instantiable(self) -> None:
        cls = get_model("random_forest")
        instance = cls()
        assert isinstance(instance, RandomForestModel)

    def test_get_model_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError) as exc_info:
            get_model("nonexistent_model")
        assert "nonexistent_model" in str(exc_info.value)
        assert "Available models" in str(exc_info.value)

    def test_register_new_model(self) -> None:
        class DummyModel:
            def fit(self, X: Any, y: Any) -> DummyModel:  # noqa: N803
                return self

            def predict_proba(self, X: Any) -> np.ndarray:  # noqa: N803
                return np.array([[0.5, 0.5]])

        register("dummy", DummyModel)  # type: ignore[arg-type]
        try:
            assert "dummy" in list_models()
            assert get_model("dummy") is DummyModel
        finally:
            unregister("dummy")

    def test_register_duplicate_raises(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            register("logistic_regression", LogisticRegressionModel)


# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------


class TestLogisticRegressionModel:
    """Tests for LogisticRegressionModel."""

    def test_fit_and_predict_proba_numeric(self) -> None:
        X, y = _make_numeric_data(n=100)  # noqa: N806
        model = LogisticRegressionModel(random_state=42)
        fitted = model.fit(X, y)
        assert fitted is model
        proba = model.predict_proba(X)
        _assert_proba_valid(proba, n_samples=100)

    def test_fit_and_predict_proba_mixed(self) -> None:
        X, y = _make_mixed_data(n=100)  # noqa: N806
        model = LogisticRegressionModel(random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)
        _assert_proba_valid(proba, n_samples=100)

    def test_fit_and_predict_proba_all_categorical(self) -> None:
        X, y = _make_all_categorical_data(n=100)  # noqa: N806
        model = LogisticRegressionModel(random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)
        _assert_proba_valid(proba, n_samples=100)

    def test_predict_proba_before_fit_raises(self) -> None:
        X, _ = _make_numeric_data(n=10)  # noqa: N806
        model = LogisticRegressionModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_proba(X)

    def test_fit_returns_self(self) -> None:
        X, y = _make_numeric_data(n=20)  # noqa: N806
        model = LogisticRegressionModel()
        result = model.fit(X, y)
        assert result is model

    def test_small_dataset(self) -> None:
        X, y = _make_numeric_data(n=10)  # noqa: N806
        model = LogisticRegressionModel(random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)
        _assert_proba_valid(proba, n_samples=10)

    def test_numpy_input(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 3))  # noqa: N806
        y = rng.integers(0, 2, size=50)
        model = LogisticRegressionModel(random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)
        _assert_proba_valid(proba, n_samples=50)


# ---------------------------------------------------------------------------
# Random forest
# ---------------------------------------------------------------------------


class TestRandomForestModel:
    """Tests for RandomForestModel."""

    def test_fit_and_predict_proba_numeric(self) -> None:
        X, y = _make_numeric_data(n=100)  # noqa: N806
        model = RandomForestModel(random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)
        _assert_proba_valid(proba, n_samples=100)

    def test_fit_and_predict_proba_mixed(self) -> None:
        X, y = _make_mixed_data(n=100)  # noqa: N806
        model = RandomForestModel(random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)
        _assert_proba_valid(proba, n_samples=100)

    def test_fit_and_predict_proba_all_categorical(self) -> None:
        X, y = _make_all_categorical_data(n=100)  # noqa: N806
        model = RandomForestModel(random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)
        _assert_proba_valid(proba, n_samples=100)

    def test_predict_proba_before_fit_raises(self) -> None:
        X, _ = _make_numeric_data(n=10)  # noqa: N806
        model = RandomForestModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_proba(X)

    def test_fit_returns_self(self) -> None:
        X, y = _make_numeric_data(n=20)  # noqa: N806
        model = RandomForestModel()
        result = model.fit(X, y)
        assert result is model

    def test_small_dataset(self) -> None:
        X, y = _make_numeric_data(n=10)  # noqa: N806
        model = RandomForestModel(random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)
        _assert_proba_valid(proba, n_samples=10)

    def test_numpy_input(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 3))  # noqa: N806
        y = rng.integers(0, 2, size=50)
        model = RandomForestModel(random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)
        _assert_proba_valid(proba, n_samples=50)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="xgboost not available")
class TestXGBoostModel:
    """Tests for XGBoostModel."""

    def test_fit_and_predict_proba_numeric(self) -> None:
        X, y = _make_numeric_data(n=100)  # noqa: N806
        model = XGBoostModel(random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)
        _assert_proba_valid(proba, n_samples=100)

    def test_fit_and_predict_proba_mixed(self) -> None:
        X, y = _make_mixed_data(n=100)  # noqa: N806
        model = XGBoostModel(random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)
        _assert_proba_valid(proba, n_samples=100)

    def test_fit_and_predict_proba_all_categorical(self) -> None:
        X, y = _make_all_categorical_data(n=100)  # noqa: N806
        model = XGBoostModel(random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)
        _assert_proba_valid(proba, n_samples=100)

    def test_predict_proba_before_fit_raises(self) -> None:
        X, _ = _make_numeric_data(n=10)  # noqa: N806
        model = XGBoostModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_proba(X)

    def test_fit_returns_self(self) -> None:
        X, y = _make_numeric_data(n=20)  # noqa: N806
        model = XGBoostModel()
        result = model.fit(X, y)
        assert result is model

    def test_small_dataset(self) -> None:
        X, y = _make_numeric_data(n=10)  # noqa: N806
        model = XGBoostModel(random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)
        _assert_proba_valid(proba, n_samples=10)

    def test_numpy_input(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 3))  # noqa: N806
        y = rng.integers(0, 2, size=50)
        model = XGBoostModel(random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)
        _assert_proba_valid(proba, n_samples=50)


# ---------------------------------------------------------------------------
# Calibrated MLP
# ---------------------------------------------------------------------------


class TestCalibratedMLPModel:
    """Tests for CalibratedMLPModel."""

    def test_fit_and_predict_proba_numeric(self) -> None:
        X, y = _make_numeric_data(n=100)  # noqa: N806
        model = CalibratedMLPModel(random_state=42, max_iter=200)
        model.fit(X, y)
        proba = model.predict_proba(X)
        _assert_proba_valid(proba, n_samples=100)

    def test_fit_and_predict_proba_mixed(self) -> None:
        X, y = _make_mixed_data(n=100)  # noqa: N806
        model = CalibratedMLPModel(random_state=42, max_iter=200)
        model.fit(X, y)
        proba = model.predict_proba(X)
        _assert_proba_valid(proba, n_samples=100)

    def test_fit_and_predict_proba_all_categorical(self) -> None:
        X, y = _make_all_categorical_data(n=100)  # noqa: N806
        model = CalibratedMLPModel(random_state=42, max_iter=200)
        model.fit(X, y)
        proba = model.predict_proba(X)
        _assert_proba_valid(proba, n_samples=100)

    def test_predict_proba_before_fit_raises(self) -> None:
        X, _ = _make_numeric_data(n=10)  # noqa: N806
        model = CalibratedMLPModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_proba(X)

    def test_fit_returns_self(self) -> None:
        X, y = _make_numeric_data(n=20)  # noqa: N806
        model = CalibratedMLPModel()
        result = model.fit(X, y)
        assert result is model

    def test_small_dataset_fallback(self) -> None:
        """Very small datasets should trigger the calibration fallback."""
        X, y = _make_numeric_data(n=6)  # noqa: N806
        model = CalibratedMLPModel(random_state=42, max_iter=200)
        with warnings.catch_warnings(record=True) as _:
            warnings.simplefilter("always")
            model.fit(X, y)
            # May or may not warn depending on exact class distribution
        proba = model.predict_proba(X)
        _assert_proba_valid(proba, n_samples=6)

    def test_numpy_input(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 3))  # noqa: N806
        y = rng.integers(0, 2, size=50)
        model = CalibratedMLPModel(random_state=42, max_iter=200)
        model.fit(X, y)
        proba = model.predict_proba(X)
        _assert_proba_valid(proba, n_samples=50)


# ---------------------------------------------------------------------------
# Edge cases shared across models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_cls",
    [
        LogisticRegressionModel,
        RandomForestModel,
        CalibratedMLPModel,
    ],
)
class TestModelEdgeCases:
    """Edge-case tests applied to all sklearn-based models."""

    def test_single_class_raises_or_degenerate(self, model_cls: type) -> None:
        """A single-class target is not a valid binary classification problem."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((20, 3)), columns=["a", "b", "c"])  # noqa: N806
        y = pd.Series(np.zeros(20, dtype=int), name="target")
        model = model_cls(random_state=42)
        try:
            model.fit(X, y)
            proba = model.predict_proba(X)
            assert proba.shape == (20, 1)
        except ValueError:
            pass  # Expected — sklearn rejects single-class problems

    def test_imbalanced_classes(self, model_cls: type) -> None:
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((50, 3)), columns=["a", "b", "c"])  # noqa: N806
        y = pd.Series([0] * 45 + [1] * 5, name="target")
        model = model_cls(random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)
        _assert_proba_valid(proba, n_samples=50)

    def test_predict_on_unseen_categorical_value(self, model_cls: type) -> None:
        """One-hot encoder with handle_unknown='ignore' should tolerate new categories."""
        rng = np.random.default_rng(0)
        X_train = pd.DataFrame(  # noqa: N806
            {
                "num": rng.standard_normal(50),
                "cat": rng.choice(["A", "B"], size=50),
            }
        )
        y_train = pd.Series(rng.integers(0, 2, size=50), name="target")
        model = model_cls(random_state=42)
        model.fit(X_train, y_train)

        X_test = pd.DataFrame(  # noqa: N806
            {
                "num": rng.standard_normal(10),
                "cat": rng.choice(["C", "D"], size=10),
            }
        )
        proba = model.predict_proba(X_test)
        _assert_proba_valid(proba, n_samples=10)


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="xgboost not available")
@pytest.mark.parametrize("model_cls", [XGBoostModel])
class TestXGBoostEdgeCases:
    """Edge-case tests specific to XGBoost."""

    def test_imbalanced_classes(self, model_cls: type) -> None:
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((50, 3)), columns=["a", "b", "c"])  # noqa: N806
        y = pd.Series([0] * 45 + [1] * 5, name="target")
        model = model_cls(random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)
        _assert_proba_valid(proba, n_samples=50)

    def test_predict_on_unseen_categorical_value(self, model_cls: type) -> None:
        rng = np.random.default_rng(0)
        X_train = pd.DataFrame(  # noqa: N806
            {
                "num": rng.standard_normal(50),
                "cat": pd.Categorical(rng.choice(["A", "B"], size=50)),
            }
        )
        y_train = pd.Series(rng.integers(0, 2, size=50), name="target")
        model = model_cls(random_state=42)
        model.fit(X_train, y_train)

        X_test = pd.DataFrame(  # noqa: N806
            {
                "num": rng.standard_normal(10),
                "cat": pd.Categorical(rng.choice(["C", "D"], size=10)),
            }
        )
        proba = model.predict_proba(X_test)
        _assert_proba_valid(proba, n_samples=10)
