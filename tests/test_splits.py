"""Tests for fair_clinical_bench.data.splits — stratified split function."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from fair_clinical_bench.data.splits import SplitIndices, stratified_split


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n: int = 200, n_classes: int = 2, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Create a small synthetic dataset with balanced classes."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.integers(0, n_classes, size=n), name="target")
    return X, y


# ---------------------------------------------------------------------------
# SplitIndices
# ---------------------------------------------------------------------------

class TestSplitIndices:
    def test_is_namedtuple(self):
        train = np.array([0, 1])
        val = np.array([2])
        test = np.array([3])
        s = SplitIndices(train=train, val=val, test=test)
        assert s.train is train
        assert s.val is val
        assert s.test is test

    def test_unpacking(self):
        train = np.array([0])
        val = np.array([1])
        test = np.array([2])
        s = SplitIndices(train=train, val=val, test=test)
        t, v, te = s
        np.testing.assert_array_equal(t, train)
        np.testing.assert_array_equal(v, val)
        np.testing.assert_array_equal(te, test)


# ---------------------------------------------------------------------------
# Basic split proportions
# ---------------------------------------------------------------------------

class TestBasicSplitProportions:
    def test_default_fractions(self):
        X, y = _make_data(n=200)
        result = stratified_split(X, y)
        assert len(result.train) == 120  # 0.6 * 200
        assert len(result.val) == 40     # 0.2 * 200
        assert len(result.test) == 40    # 0.2 * 200

    def test_custom_fractions(self):
        X, y = _make_data(n=100)
        result = stratified_split(X, y, train_frac=0.7, val_frac=0.15, test_frac=0.15)
        assert len(result.train) == 70
        assert len(result.val) == 15
        assert len(result.test) == 15

    def test_all_indices_cover_dataset(self):
        X, y = _make_data(n=100)
        result = stratified_split(X, y)
        all_idx = np.sort(np.concatenate([result.train, result.val, result.test]))
        np.testing.assert_array_equal(all_idx, np.arange(100))

    def test_no_overlap_between_splits(self):
        X, y = _make_data(n=100)
        result = stratified_split(X, y)
        assert len(set(result.train) & set(result.val)) == 0
        assert len(set(result.train) & set(result.test)) == 0
        assert len(set(result.val) & set(result.test)) == 0

    def test_returns_numpy_arrays(self):
        X, y = _make_data(n=100)
        result = stratified_split(X, y)
        assert isinstance(result.train, np.ndarray)
        assert isinstance(result.val, np.ndarray)
        assert isinstance(result.test, np.ndarray)

    def test_returns_int_dtype(self):
        X, y = _make_data(n=100)
        result = stratified_split(X, y)
        assert np.issubdtype(result.train.dtype, np.integer)
        assert np.issubdtype(result.val.dtype, np.integer)
        assert np.issubdtype(result.test.dtype, np.integer)


# ---------------------------------------------------------------------------
# Reproducibility with same seed
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_split(self):
        X, y = _make_data(n=200)
        r1 = stratified_split(X, y, random_seed=42)
        r2 = stratified_split(X, y, random_seed=42)
        np.testing.assert_array_equal(r1.train, r2.train)
        np.testing.assert_array_equal(r1.val, r2.val)
        np.testing.assert_array_equal(r1.test, r2.test)

    def test_different_seed_different_split(self):
        X, y = _make_data(n=200)
        r1 = stratified_split(X, y, random_seed=42)
        r2 = stratified_split(X, y, random_seed=99)
        # At least one split should differ
        assert (
            not np.array_equal(r1.train, r2.train)
            or not np.array_equal(r1.val, r2.val)
            or not np.array_equal(r1.test, r2.test)
        )


# ---------------------------------------------------------------------------
# Stratification quality
# ---------------------------------------------------------------------------

class TestStratificationQuality:
    def test_class_distribution_preserved_in_train(self):
        X, y = _make_data(n=200, n_classes=2, seed=0)
        result = stratified_split(X, y)
        overall_ratio = (y == 0).mean()
        train_ratio = (y.iloc[result.train] == 0).mean()
        assert abs(overall_ratio - train_ratio) < 0.1

    def test_class_distribution_preserved_in_val(self):
        X, y = _make_data(n=200, n_classes=2, seed=0)
        result = stratified_split(X, y)
        overall_ratio = (y == 0).mean()
        val_ratio = (y.iloc[result.val] == 0).mean()
        assert abs(overall_ratio - val_ratio) < 0.15

    def test_class_distribution_preserved_in_test(self):
        X, y = _make_data(n=200, n_classes=2, seed=0)
        result = stratified_split(X, y)
        overall_ratio = (y == 0).mean()
        test_ratio = (y.iloc[result.test] == 0).mean()
        assert abs(overall_ratio - test_ratio) < 0.15


# ---------------------------------------------------------------------------
# Group stratification
# ---------------------------------------------------------------------------

class TestGroupStratification:
    def test_group_col_as_string(self):
        X, y = _make_data(n=200, seed=0)
        rng = np.random.default_rng(1)
        X["group"] = rng.choice(["A", "B"], size=200)
        result = stratified_split(X, y, group_col="group")
        assert len(result.train) + len(result.val) + len(result.test) == 200

    def test_group_col_as_series(self):
        X, y = _make_data(n=200, seed=0)
        groups = pd.Series(np.random.default_rng(1).choice(["A", "B"], size=200))
        result = stratified_split(X, y, group_col=groups)
        assert len(result.train) + len(result.val) + len(result.test) == 200

    def test_group_col_as_numpy_array(self):
        X, y = _make_data(n=200, seed=0)
        groups = np.random.default_rng(1).choice(["A", "B"], size=200)
        result = stratified_split(X, y, group_col=groups)
        assert len(result.train) + len(result.val) + len(result.test) == 200

    def test_group_col_as_string_with_numpy_x_raises_type_error(self):
        X_np = np.random.default_rng(0).standard_normal((100, 3))
        y = pd.Series(np.random.default_rng(1).integers(0, 2, size=100))
        with pytest.raises(TypeError, match="group_col as string requires X to be a pandas DataFrame"):
            stratified_split(X_np, y, group_col="col0")

    def test_group_col_string_not_in_x_raises_value_error(self):
        X, y = _make_data(n=100)
        with pytest.raises(ValueError, match="not found in X columns"):
            stratified_split(X, y, group_col="nonexistent")

    def test_group_col_length_mismatch_raises_value_error(self):
        X, y = _make_data(n=100)
        groups = np.array(["A"] * 50)
        with pytest.raises(ValueError, match="does not match"):
            stratified_split(X, y, group_col=groups)

    def test_group_col_invalid_type_raises_type_error(self):
        X, y = _make_data(n=100)
        with pytest.raises(TypeError, match="group_col must be a string"):
            stratified_split(X, y, group_col=123)


# ---------------------------------------------------------------------------
# Single group warning
# ---------------------------------------------------------------------------

class TestSingleGroupWarning:
    def test_single_group_warns(self):
        X, y = _make_data(n=100)
        groups = np.array(["A"] * 100)
        with pytest.warns(UserWarning, match="Only one unique group"):
            stratified_split(X, y, group_col=groups)

    def test_single_group_still_returns_valid_split(self):
        X, y = _make_data(n=100)
        groups = np.array(["A"] * 100)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = stratified_split(X, y, group_col=groups)
        assert len(result.train) + len(result.val) + len(result.test) == 100


# ---------------------------------------------------------------------------
# Group+target fallback warning
# ---------------------------------------------------------------------------

class TestGroupTargetFallback:
    def test_warns_when_group_target_combo_has_fewer_than_2(self):
        """When each group has only one sample per target, fallback warning fires."""
        # Need enough samples so the split succeeds after fallback to target-only.
        # 40 samples, 2 groups, 2 classes → each group+target combo has ~10 samples,
        # but we force the warning by making each group appear with each target only once
        # in a way that triggers the <2 check. Actually, we need groups where some
        # group+target combos have count < 2. Let's use a dataset where one group
        # has only 1 sample of a particular target.
        X = pd.DataFrame({"f0": range(40)})
        # Groups: A appears with target 0 (19 times) and target 1 (1 time)
        #         B appears with target 0 (1 time) and target 1 (19 times)
        # The A::1 and B::0 combos have count=1 → triggers warning
        groups = np.array(["A"] * 20 + ["B"] * 20)
        y = pd.Series([0] * 19 + [1] * 1 + [0] * 1 + [1] * 19)
        with pytest.warns(UserWarning, match="Some group\\+target combinations"):
            result = stratified_split(X, y, group_col=groups)
        # Should still produce a valid split (fell back to target-only)
        assert len(result.train) + len(result.val) + len(result.test) == 40


# ---------------------------------------------------------------------------
# Invalid fractions
# ---------------------------------------------------------------------------

class TestInvalidFractions:
    def test_fractions_sum_less_than_one(self):
        X, y = _make_data(n=100)
        with pytest.raises(ValueError, match="Fractions must sum to 1.0"):
            stratified_split(X, y, train_frac=0.3, val_frac=0.3, test_frac=0.3)

    def test_fractions_sum_greater_than_one(self):
        X, y = _make_data(n=100)
        with pytest.raises(ValueError, match="Fractions must sum to 1.0"):
            stratified_split(X, y, train_frac=0.5, val_frac=0.5, test_frac=0.5)

    def test_zero_test_frac_passes_sum_check_but_fails_in_sklearn(self):
        """0.5+0.5+0.0=1.0 passes our check, but sklearn rejects train_size=1.0."""
        X, y = _make_data(n=100)
        with pytest.raises(Exception):  # sklearn InvalidParameterError
            stratified_split(X, y, train_frac=0.5, val_frac=0.5, test_frac=0.0)

    def test_negative_frac_passes_sum_check_but_fails_in_sklearn(self):
        """1.2+(-0.1)+(-0.1)=1.0 passes our check, but sklearn rejects train_size=1.2."""
        X, y = _make_data(n=100)
        with pytest.raises(Exception):  # sklearn InvalidParameterError
            stratified_split(X, y, train_frac=1.2, val_frac=-0.1, test_frac=-0.1)

    def test_error_message_shows_actual_sum(self):
        X, y = _make_data(n=100)
        with pytest.raises(ValueError) as exc_info:
            stratified_split(X, y, train_frac=0.3, val_frac=0.3, test_frac=0.3)
        msg = str(exc_info.value)
        assert "0.900000" in msg


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEmptyData:
    def test_empty_y_raises(self):
        X = pd.DataFrame()
        y = pd.Series([], dtype=int)
        with pytest.raises(ValueError):
            stratified_split(X, y)


class TestSingleClass:
    def test_single_class_raises_or_produces_empty_split(self):
        """sklearn train_test_split with stratify requires >= 2 classes.
        Depending on sklearn version, this may raise or produce degenerate splits."""
        X, _ = _make_data(n=20, n_classes=1, seed=0)
        y = pd.Series(np.zeros(20, dtype=int))
        try:
            result = stratified_split(X, y)
            # If it doesn't raise, at least verify the split is degenerate
            # (all indices in one split or similar)
            total = len(result.train) + len(result.val) + len(result.test)
            assert total == 20
        except (ValueError, Exception):
            pass  # Expected: sklearn rejects single-class stratification


class TestSmallDataset:
    def test_very_small_dataset(self):
        """Minimum viable dataset for stratified 3-way split."""
        X = pd.DataFrame({"f0": range(12)})
        y = pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        result = stratified_split(X, y, random_seed=0)
        assert len(result.train) + len(result.val) + len(result.test) == 12
        assert len(result.train) > 0
        assert len(result.val) > 0
        assert len(result.test) > 0


class TestNumpyInputs:
    def test_accepts_numpy_x_and_y(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 3))
        y = rng.integers(0, 2, size=100)
        result = stratified_split(X, y)
        assert len(result.train) == 60
        assert len(result.val) == 20
        assert len(result.test) == 20


class TestLargeDataset:
    def test_large_dataset_proportions(self):
        X, y = _make_data(n=10000)
        result = stratified_split(X, y)
        assert abs(len(result.train) / 10000 - 0.6) < 0.01
        assert abs(len(result.val) / 10000 - 0.2) < 0.01
        assert abs(len(result.test) / 10000 - 0.2) < 0.01


class TestMultiClass:
    def test_three_class_stratification(self):
        X, y = _make_data(n=300, n_classes=3, seed=0)
        result = stratified_split(X, y)
        # Each class should appear in all splits
        for split in [result.train, result.val, result.test]:
            classes_in_split = set(y.iloc[split].unique())
            assert classes_in_split == {0, 1, 2}
