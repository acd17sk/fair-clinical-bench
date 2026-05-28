"""Tests for fair_clinical_bench.data.schema.ClinicalDataset."""

from __future__ import annotations

import pandas as pd
import pytest

from fair_clinical_bench.data.schema import ClinicalDataset


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Return a minimal feature DataFrame with a group column."""
    return pd.DataFrame(
        {
            "age": [45, 62, 33, 71],
            "bmi": [28.1, 31.4, 22.0, 26.7],
            "race": ["White", "Black", "Asian", "Hispanic"],
        }
    )


@pytest.fixture
def sample_target() -> pd.Series:
    """Return a minimal target Series."""
    return pd.Series([0, 1, 0, 1], name="outcome")


# ── Happy path ──────────────────────────────────────────────────────────────


class TestClinicalDatasetCreation:
    """Test basic dataclass instantiation."""

    def test_create_with_all_fields(
        self, sample_dataframe: pd.DataFrame, sample_target: pd.Series
    ) -> None:
        ds = ClinicalDataset(
            X=sample_dataframe,
            y=sample_target,
            group_cols=["race"],
            name="test_dataset",
            description="A test dataset for validation",
        )
        assert ds.X is sample_dataframe
        assert ds.y is sample_target
        assert ds.group_cols == ["race"]
        assert ds.name == "test_dataset"
        assert ds.description == "A test dataset for validation"

    def test_create_with_defaults(
        self, sample_dataframe: pd.DataFrame, sample_target: pd.Series
    ) -> None:
        ds = ClinicalDataset(X=sample_dataframe, y=sample_target)
        assert ds.group_cols == []
        assert ds.name == ""
        assert ds.description is None

    def test_create_with_empty_group_cols(
        self, sample_dataframe: pd.DataFrame, sample_target: pd.Series
    ) -> None:
        ds = ClinicalDataset(
            X=sample_dataframe,
            y=sample_target,
            group_cols=[],
        )
        assert ds.group_cols == []

    def test_create_with_none_description(
        self, sample_dataframe: pd.DataFrame, sample_target: pd.Series
    ) -> None:
        ds = ClinicalDataset(
            X=sample_dataframe,
            y=sample_target,
            description=None,
        )
        assert ds.description is None


# ── Validation (post_init) ──────────────────────────────────────────────────


class TestClinicalDatasetValidation:
    """Test __post_init__ validation of group_cols."""

    def test_missing_group_col_raises(
        self, sample_dataframe: pd.DataFrame, sample_target: pd.Series
    ) -> None:
        with pytest.raises(ValueError) as exc_info:
            ClinicalDataset(
                X=sample_dataframe,
                y=sample_target,
                group_cols=["nonexistent_col"],
            )
        assert "nonexistent_col" in str(exc_info.value)
        assert "Available columns" in str(exc_info.value)

    def test_multiple_missing_group_cols_raise(
        self, sample_dataframe: pd.DataFrame, sample_target: pd.Series
    ) -> None:
        with pytest.raises(ValueError) as exc_info:
            ClinicalDataset(
                X=sample_dataframe,
                y=sample_target,
                group_cols=["missing_a", "missing_b"],
            )
        msg = str(exc_info.value)
        assert "missing_a" in msg
        assert "missing_b" in msg

    def test_valid_group_col_passes(
        self, sample_dataframe: pd.DataFrame, sample_target: pd.Series
    ) -> None:
        ds = ClinicalDataset(
            X=sample_dataframe,
            y=sample_target,
            group_cols=["race"],
        )
        assert ds.group_cols == ["race"]

    def test_x_y_length_mismatch_raises(
        self, sample_dataframe: pd.DataFrame, sample_target: pd.Series
    ) -> None:
        short_y = sample_target.iloc[:-1]
        with pytest.raises(ValueError) as exc_info:
            ClinicalDataset(
                X=sample_dataframe,
                y=short_y,
                group_cols=["race"],
            )
        msg = str(exc_info.value)
        assert "4" in msg
        assert "3" in msg

    def test_duplicate_group_cols_raises(
        self, sample_dataframe: pd.DataFrame, sample_target: pd.Series
    ) -> None:
        with pytest.raises(ValueError) as exc_info:
            ClinicalDataset(
                X=sample_dataframe,
                y=sample_target,
                group_cols=["race", "race"],
            )
        msg = str(exc_info.value)
        assert "race" in msg
        assert "Duplicate" in msg

    def test_empty_group_cols_skips_validation(self) -> None:
        """Empty group_cols should not raise even with an empty DataFrame."""
        empty_df = pd.DataFrame()
        empty_y = pd.Series([], dtype="float64")
        ds = ClinicalDataset(
            X=empty_df,
            y=empty_y,
            group_cols=[],
        )
        assert ds.group_cols == []


# ── Property accessors ──────────────────────────────────────────────────────


class TestClinicalDatasetProperties:
    """Test features, target, and groups properties."""

    def test_features_returns_x(
        self, sample_dataframe: pd.DataFrame, sample_target: pd.Series
    ) -> None:
        ds = ClinicalDataset(X=sample_dataframe, y=sample_target)
        assert ds.features is sample_dataframe
        assert isinstance(ds.features, pd.DataFrame)

    def test_target_returns_y(
        self, sample_dataframe: pd.DataFrame, sample_target: pd.Series
    ) -> None:
        ds = ClinicalDataset(X=sample_dataframe, y=sample_target)
        assert ds.target is sample_target
        assert isinstance(ds.target, pd.Series)

    def test_groups_returns_subset_when_defined(
        self, sample_dataframe: pd.DataFrame, sample_target: pd.Series
    ) -> None:
        ds = ClinicalDataset(
            X=sample_dataframe,
            y=sample_target,
            group_cols=["race"],
        )
        groups = ds.groups
        assert groups is not None
        assert isinstance(groups, pd.DataFrame)
        assert list(groups.columns) == ["race"]
        assert list(groups["race"]) == ["White", "Black", "Asian", "Hispanic"]

    def test_groups_returns_none_when_empty(
        self, sample_dataframe: pd.DataFrame, sample_target: pd.Series
    ) -> None:
        ds = ClinicalDataset(
            X=sample_dataframe,
            y=sample_target,
            group_cols=[],
        )
        assert ds.groups is None

    def test_groups_returns_multiple_columns(
        self, sample_dataframe: pd.DataFrame, sample_target: pd.Series
    ) -> None:
        ds = ClinicalDataset(
            X=sample_dataframe,
            y=sample_target,
            group_cols=["race", "age"],
        )
        groups = ds.groups
        assert groups is not None
        assert list(groups.columns) == ["race", "age"]
        assert groups.shape == (4, 2)
