"""Tests for fair_clinical_bench.data.loader — built-in dataset loaders."""

from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from fair_clinical_bench.data.loader import (
    _generate_synthetic_diabetes130,
    _generate_synthetic_heart,
    _generate_synthetic_pima,
    _impute_missing,
    list_builtin_datasets,
    load_csv,
    load_diabetes130,
    load_heart,
    load_pima,
)
from fair_clinical_bench.data.schema import ClinicalDataset

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_cache_dir(tmp_path: Path):
    """Provide an isolated cache directory so tests don't pollute ~/.fair-clinical-bench."""
    with patch("fair_clinical_bench.data.loader._CACHE_DIR", tmp_path):
        yield tmp_path


# ---------------------------------------------------------------------------
# list_builtin_datasets
# ---------------------------------------------------------------------------

class TestListBuiltinDatasets:
    def test_returns_three_datasets(self):
        result = list_builtin_datasets()
        assert isinstance(result, list)
        assert len(result) == 3

    def test_contains_expected_names(self):
        result = list_builtin_datasets()
        assert set(result) == {"pima", "heart", "diabetes130"}


# ---------------------------------------------------------------------------
# load_pima — happy path (offline fallback via mocked download)
# ---------------------------------------------------------------------------

class TestLoadPima:
    def test_returns_clinical_dataset(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_pima()
        assert isinstance(ds, ClinicalDataset)

    def test_correct_name(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_pima()
        assert ds.name == "pima"

    def test_has_description(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_pima()
        assert ds.description is not None
        assert "Pima" in ds.description

    def test_feature_shape_matches_target(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_pima()
        assert len(ds.X) == len(ds.y)

    def test_expected_row_count_synthetic(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_pima()
        assert len(ds.X) == 50

    def test_expected_feature_columns(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_pima()
        expected_cols = {
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
        }
        assert set(ds.X.columns) == expected_cols

    def test_target_column_values(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_pima()
        assert set(ds.y.unique()).issubset({0, 1})

    def test_group_columns(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_pima()
        assert ds.group_cols == ["Age"]

    def test_no_missing_values_after_imputation(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_pima()
        assert ds.X.isna().sum().sum() == 0

    def test_groups_property_returns_dataframe(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_pima()
        groups = ds.groups
        assert isinstance(groups, pd.DataFrame)
        assert list(groups.columns) == ["Age"]

    def test_features_property(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_pima()
        assert ds.features is ds.X

    def test_target_property(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_pima()
        assert ds.target is ds.y

    def test_caches_file_after_first_load(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False) as mock_dl:
            load_pima()
            load_pima()
        # _download_file is called only on the first load (csv doesn't exist yet)
        assert mock_dl.call_count == 1

    def test_load_with_pre_existing_csv(self, tmp_cache_dir):
        """If a CSV already exists in cache, loader reads it directly without download."""
        csv_path = tmp_cache_dir / "pima-indians-diabetes.csv"
        df = pd.DataFrame({
            "Pregnancies": [1, 2],
            "Glucose": [100, 120],
            "BloodPressure": [70, 80],
            "SkinThickness": [20, 25],
            "Insulin": [80, 90],
            "BMI": [25.0, 30.0],
            "DiabetesPedigreeFunction": [0.5, 0.6],
            "Age": [30, 40],
            "Outcome": [0, 1],
        })
        df.to_csv(csv_path, index=False)

        with patch("fair_clinical_bench.data.loader._download_file", return_value=False) as mock_dl:
            ds = load_pima()

        assert mock_dl.call_count == 0
        assert len(ds.X) == 2
        assert ds.name == "pima"


# ---------------------------------------------------------------------------
# load_heart — happy path (offline fallback)
# ---------------------------------------------------------------------------

class TestLoadHeart:
    def test_returns_clinical_dataset(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_heart()
        assert isinstance(ds, ClinicalDataset)

    def test_correct_name(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_heart()
        assert ds.name == "heart"

    def test_has_description(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_heart()
        assert ds.description is not None
        assert "Heart" in ds.description

    def test_feature_shape_matches_target(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_heart()
        assert len(ds.X) == len(ds.y)

    def test_expected_row_count_synthetic(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_heart()
        assert len(ds.X) == 50

    def test_expected_feature_columns(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_heart()
        expected_cols = {
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal",
        }
        assert set(ds.X.columns) == expected_cols

    def test_target_column_values(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_heart()
        assert set(ds.y.unique()).issubset({0, 1})

    def test_group_columns(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_heart()
        assert ds.group_cols == ["sex"]

    def test_no_missing_values_after_imputation(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_heart()
        assert ds.X.isna().sum().sum() == 0

    def test_groups_property_returns_dataframe(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_heart()
        groups = ds.groups
        assert isinstance(groups, pd.DataFrame)
        assert list(groups.columns) == ["sex"]


# ---------------------------------------------------------------------------
# load_diabetes130 — happy path (offline fallback)
# ---------------------------------------------------------------------------

class TestLoadDiabetes130:
    def test_returns_clinical_dataset(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_diabetes130()
        assert isinstance(ds, ClinicalDataset)

    def test_correct_name(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_diabetes130()
        assert ds.name == "diabetes130"

    def test_has_description(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_diabetes130()
        assert ds.description is not None
        assert "Diabetes 130" in ds.description

    def test_feature_shape_matches_target(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_diabetes130()
        assert len(ds.X) == len(ds.y)

    def test_expected_row_count_synthetic(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_diabetes130()
        assert len(ds.X) == 50

    def test_expected_feature_columns(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_diabetes130()
        expected_cols = {
            "race", "gender", "age", "time_in_hospital", "num_lab_procedures",
            "num_procedures", "num_medications", "number_outpatient",
            "number_emergency", "number_inpatient", "number_diagnoses",
        }
        assert set(ds.X.columns) == expected_cols

    def test_target_column_values(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_diabetes130()
        assert set(ds.y.unique()).issubset({"NO", ">30", "<30"})

    def test_group_columns(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_diabetes130()
        assert ds.group_cols == ["race", "gender", "age"]

    def test_no_missing_values_after_imputation(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_diabetes130()
        assert ds.X.isna().sum().sum() == 0

    def test_groups_property_returns_dataframe(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_diabetes130()
        groups = ds.groups
        assert isinstance(groups, pd.DataFrame)
        assert list(groups.columns) == ["race", "gender", "age"]

    def test_fallback_on_zip_or_csv_exception(self, tmp_cache_dir):
        """If zip extraction or CSV read raises, loader falls back to synthetic data."""
        with (
            patch("fair_clinical_bench.data.loader._download_file", return_value=True),
            patch(
                "fair_clinical_bench.data.loader.zipfile.ZipFile",
                side_effect=zipfile.BadZipFile("bad"),
            ),
        ):
            ds = load_diabetes130()
        assert isinstance(ds, ClinicalDataset)
        assert len(ds.X) == 50

    def test_target_not_dropped_even_when_mostly_missing(self, tmp_cache_dir):
        """The target column must survive even if >50 % of its values are missing."""
        csv_path = tmp_cache_dir / "diabetes-130.csv"
        df = pd.DataFrame({
            "race": ["Caucasian", "AfricanAmerican", None, "Hispanic", "Asian"],
            "gender": ["Male", "Female", "Male", "Female", "Male"],
            "age": ["[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)"],
            "time_in_hospital": [1, 2, 3, 4, 5],
            "num_lab_procedures": [10, 20, 30, 40, 50],
            "num_procedures": [0, 1, 2, 3, 4],
            "num_medications": [5, 10, 15, 20, 25],
            "number_outpatient": [0, 1, 0, 1, 0],
            "number_emergency": [0, 0, 1, 0, 1],
            "number_inpatient": [0, 1, 0, 1, 0],
            "number_diagnoses": [3, 5, 7, 9, 11],
            "readmitted": [np.nan, np.nan, np.nan, np.nan, "NO"],
        })
        df.to_csv(csv_path, index=False)

        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_diabetes130()

        assert "readmitted" not in ds.X.columns
        assert len(ds.y) == 5
        assert ds.y.isna().sum() == 0


# ---------------------------------------------------------------------------
# Offline fallback — network failure simulation
# ---------------------------------------------------------------------------

class TestOfflineFallback:
    def test_pima_fallback_generates_data(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_pima()
        assert len(ds.X) > 0
        assert len(ds.y) > 0

    def test_heart_fallback_generates_data(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_heart()
        assert len(ds.X) > 0
        assert len(ds.y) > 0

    def test_diabetes130_fallback_generates_data(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_diabetes130()
        assert len(ds.X) > 0
        assert len(ds.y) > 0

    def test_fallback_writes_csv_to_cache(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            load_pima()
        csv_path = tmp_cache_dir / "pima-indians-diabetes.csv"
        assert csv_path.exists()

    def test_fallback_heart_writes_csv_to_cache(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            load_heart()
        csv_path = tmp_cache_dir / "heart-disease-cleveland.csv"
        assert csv_path.exists()

    def test_fallback_diabetes130_writes_csv_to_cache(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            load_diabetes130()
        csv_path = tmp_cache_dir / "diabetes-130.csv"
        assert csv_path.exists()


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

class TestSyntheticGenerators:
    def test_synthetic_pima_shape(self):
        df = _generate_synthetic_pima(n_rows=100)
        assert len(df) == 100
        assert "Outcome" in df.columns
        assert "Glucose" in df.columns

    def test_synthetic_pima_deterministic(self):
        df1 = _generate_synthetic_pima(n_rows=50)
        df2 = _generate_synthetic_pima(n_rows=50)
        pd.testing.assert_frame_equal(df1, df2)

    def test_synthetic_heart_shape(self):
        df = _generate_synthetic_heart(n_rows=100)
        assert len(df) == 100
        assert "target" in df.columns
        assert "age" in df.columns

    def test_synthetic_heart_deterministic(self):
        df1 = _generate_synthetic_heart(n_rows=50)
        df2 = _generate_synthetic_heart(n_rows=50)
        pd.testing.assert_frame_equal(df1, df2)

    def test_synthetic_diabetes130_shape(self):
        df = _generate_synthetic_diabetes130(n_rows=100)
        assert len(df) == 100
        assert "readmitted" in df.columns
        assert "race" in df.columns

    def test_synthetic_diabetes130_deterministic(self):
        df1 = _generate_synthetic_diabetes130(n_rows=50)
        df2 = _generate_synthetic_diabetes130(n_rows=50)
        pd.testing.assert_frame_equal(df1, df2)

    def test_synthetic_pima_value_ranges(self):
        df = _generate_synthetic_pima()
        assert df["Glucose"].min() >= 50
        assert df["Glucose"].max() <= 199
        assert df["BMI"].min() >= 15.0
        assert df["BMI"].max() <= 50.0

    def test_synthetic_heart_value_ranges(self):
        df = _generate_synthetic_heart()
        assert df["age"].min() >= 29
        assert df["age"].max() <= 80
        assert df["chol"].min() >= 120
        assert df["chol"].max() <= 569

    def test_synthetic_diabetes130_categorical_values(self):
        df = _generate_synthetic_diabetes130()
        valid_races = {"Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"}
        valid_genders = {"Male", "Female"}
        valid_readmitted = {"NO", ">30", "<30"}
        assert set(df["race"].unique()).issubset(valid_races)
        assert set(df["gender"].unique()).issubset(valid_genders)
        assert set(df["readmitted"].unique()).issubset(valid_readmitted)


# ---------------------------------------------------------------------------
# _impute_missing
# ---------------------------------------------------------------------------

class TestImputeMissing:
    def test_imputes_numeric_with_median(self):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
        result = _impute_missing(df)
        assert result["a"].isna().sum() == 0
        # Median of [1, 2, 4] = 2.0
        assert result.loc[2, "a"] == 2.0

    def test_imputes_categorical_with_mode(self):
        df = pd.DataFrame({"b": ["X", "X", None, "Y"]}).astype({"b": "object"})
        result = _impute_missing(df)
        assert result["b"].isna().sum() == 0
        # Mode is "X"
        assert result.loc[2, "b"] == "X"

    def test_no_op_when_no_missing(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["X", "Y", "Z"]})
        result = _impute_missing(df)
        pd.testing.assert_frame_equal(result, df)

    def test_handles_all_nan_column(self):
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
        result = _impute_missing(df)
        # All NaN numeric → median is NaN, but fillna with median of all-NAN is NaN
        # The function should not crash
        assert "a" in result.columns

    def test_preserves_non_missing_values(self):
        df = pd.DataFrame({"a": [10.0, 20.0, np.nan], "b": ["A", "B", "A"]})
        result = _impute_missing(df)
        assert result.loc[0, "a"] == 10.0
        assert result.loc[1, "a"] == 20.0
        assert result.loc[0, "b"] == "A"
        assert result.loc[1, "b"] == "B"

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan]})
        original = df.copy()
        _impute_missing(df)
        pd.testing.assert_frame_equal(df, original)

    def test_uses_is_numeric_dtype_for_string_array(self):
        """StringArray (pandas >=1.0) should be treated as categorical, not numeric."""
        df = pd.DataFrame({"s": pd.array(["X", "X", None, "Y"], dtype="string")})
        result = _impute_missing(df)
        assert result["s"].isna().sum() == 0
        assert result.loc[2, "s"] == "X"


# ---------------------------------------------------------------------------
# _download_file
# ---------------------------------------------------------------------------

class TestDownloadFile:
    def test_returns_false_for_invalid_url(self, tmp_path):
        from fair_clinical_bench.data.loader import _download_file
        dest = tmp_path / "test.txt"
        result = _download_file(
            "http://invalid-host-that-does-not-exist.example/test",
            dest,
            timeout=1.0,
        )
        assert result is False

    def test_returns_false_for_malformed_url(self, tmp_path):
        from fair_clinical_bench.data.loader import _download_file
        dest = tmp_path / "test.txt"
        result = _download_file("not-a-url", dest, timeout=1.0)
        assert result is False


# ---------------------------------------------------------------------------
# Group column validation via ClinicalDataset
# ---------------------------------------------------------------------------

class TestGroupColumnValidation:
    def test_pima_group_col_in_features(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_pima()
        for col in ds.group_cols:
            assert col in ds.X.columns

    def test_heart_group_col_in_features(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_heart()
        for col in ds.group_cols:
            assert col in ds.X.columns

    def test_diabetes130_group_cols_in_features(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_diabetes130()
        for col in ds.group_cols:
            assert col in ds.X.columns

    def test_pima_groups_not_empty(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_pima()
        assert len(ds.group_cols) > 0

    def test_heart_groups_not_empty(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_heart()
        assert len(ds.group_cols) > 0

    def test_diabetes130_groups_not_empty(self, tmp_cache_dir):
        with patch("fair_clinical_bench.data.loader._download_file", return_value=False):
            ds = load_diabetes130()
        assert len(ds.group_cols) > 0


# ---------------------------------------------------------------------------
# load_csv — BYO-CSV dataset loader
# ---------------------------------------------------------------------------

class TestLoadCsvHappyPath:
    """Test load_csv with valid CSV inputs."""

    def test_returns_clinical_dataset(self, tmp_path: Path):
        csv = tmp_path / "valid.csv"
        csv.write_text("age,bmi,outcome\n45,28.1,0\n62,31.4,1\n")
        ds = load_csv(csv, target_col="outcome")
        assert isinstance(ds, ClinicalDataset)

    def test_correct_feature_count(self, tmp_path: Path):
        csv = tmp_path / "valid.csv"
        csv.write_text("age,bmi,outcome\n45,28.1,0\n62,31.4,1\n33,22.0,0\n")
        ds = load_csv(csv, target_col="outcome")
        assert set(ds.X.columns) == {"age", "bmi"}

    def test_target_values_match_csv(self, tmp_path: Path):
        csv = tmp_path / "valid.csv"
        csv.write_text("age,bmi,outcome\n45,28.1,0\n62,31.4,1\n33,22.0,0\n71,26.7,1\n")
        ds = load_csv(csv, target_col="outcome")
        assert list(ds.y) == [0, 1, 0, 1]

    def test_row_count_matches(self, tmp_path: Path):
        csv = tmp_path / "valid.csv"
        csv.write_text("a,b,target\n1,2,0\n3,4,1\n5,6,0\n")
        ds = load_csv(csv, target_col="target")
        assert len(ds.X) == 3
        assert len(ds.y) == 3

    def test_name_derived_from_filename_stem(self, tmp_path: Path):
        csv = tmp_path / "my_custom_dataset.csv"
        csv.write_text("x,y\n1,0\n2,1\n")
        ds = load_csv(csv, target_col="y")
        assert ds.name == "my_custom_dataset"

    def test_description_mentions_filename(self, tmp_path: Path):
        csv = tmp_path / "special_data.csv"
        csv.write_text("x,y\n1,0\n2,1\n")
        ds = load_csv(csv, target_col="y")
        assert ds.description is not None
        assert "special_data.csv" in ds.description

    def test_group_cols_preserved_when_present(self, tmp_path: Path):
        csv = tmp_path / "valid.csv"
        csv.write_text("age,race,outcome\n45,White,0\n62,Black,1\n")
        ds = load_csv(csv, target_col="outcome", group_cols=["race"])
        assert ds.group_cols == ["race"]

    def test_multiple_group_cols_preserved(self, tmp_path: Path):
        csv = tmp_path / "valid.csv"
        csv.write_text("age,race,gender,outcome\n45,White,M,0\n62,Black,F,1\n")
        ds = load_csv(csv, target_col="outcome", group_cols=["race", "gender"])
        assert ds.group_cols == ["race", "gender"]

    def test_group_cols_in_features(self, tmp_path: Path):
        csv = tmp_path / "valid.csv"
        csv.write_text("age,race,outcome\n45,White,0\n62,Black,1\n")
        ds = load_csv(csv, target_col="outcome", group_cols=["race"])
        assert "race" in ds.X.columns

    def test_accepts_pathlib_path(self, tmp_path: Path):
        csv = tmp_path / "valid.csv"
        csv.write_text("x,y\n1,0\n2,1\n")
        ds = load_csv(csv, target_col="y")
        assert len(ds.X) == 2

    def test_accepts_string_path(self, tmp_path: Path):
        csv = tmp_path / "valid.csv"
        csv.write_text("x,y\n1,0\n2,1\n")
        ds = load_csv(str(csv), target_col="y")
        assert len(ds.X) == 2

    def test_no_group_cols_when_none_provided(self, tmp_path: Path):
        csv = tmp_path / "valid.csv"
        csv.write_text("age,bmi,outcome\n45,28.1,0\n62,31.4,1\n")
        ds = load_csv(csv, target_col="outcome")
        assert ds.group_cols == []

    def test_no_group_cols_when_empty_list(self, tmp_path: Path):
        csv = tmp_path / "valid.csv"
        csv.write_text("age,bmi,outcome\n45,28.1,0\n62,31.4,1\n")
        ds = load_csv(csv, target_col="outcome", group_cols=[])
        assert ds.group_cols == []

    def test_no_missing_values_after_imputation(self, tmp_path: Path):
        csv = tmp_path / "with_nan.csv"
        csv.write_text("a,b,target\n1.0,2.0,0\n3.0,,1\n,5.0,0\n")
        ds = load_csv(csv, target_col="target")
        assert ds.X.isna().sum().sum() == 0


class TestLoadCsvMissingTarget:
    """Test load_csv raises ValueError when target column is absent."""

    def test_raises_value_error_for_missing_target(self, tmp_path: Path):
        csv = tmp_path / "valid.csv"
        csv.write_text("age,bmi,label\n45,28.1,0\n62,31.4,1\n")
        with pytest.raises(ValueError) as exc_info:
            load_csv(csv, target_col="outcome")
        assert "outcome" in str(exc_info.value)

    def test_error_message_lists_available_columns(self, tmp_path: Path):
        csv = tmp_path / "valid.csv"
        csv.write_text("age,bmi,label\n45,28.1,0\n62,31.4,1\n")
        with pytest.raises(ValueError) as exc_info:
            load_csv(csv, target_col="outcome")
        msg = str(exc_info.value)
        assert "age" in msg
        assert "bmi" in msg
        assert "label" in msg

    def test_error_message_mentions_target_not_found(self, tmp_path: Path):
        csv = tmp_path / "valid.csv"
        csv.write_text("a,b,c\n1,2,3\n")
        with pytest.raises(ValueError) as exc_info:
            load_csv(csv, target_col="missing")
        assert "missing" in str(exc_info.value)
        assert "not found" in str(exc_info.value).lower()


class TestLoadCsvMissingGroupCols:
    """Test load_csv warns when group columns are absent."""

    def test_warns_for_missing_group_col(self, tmp_path: Path):
        csv = tmp_path / "valid.csv"
        csv.write_text("age,bmi,outcome\n45,28.1,0\n62,31.4,1\n")
        with pytest.warns(UserWarning) as record:
            ds = load_csv(csv, target_col="outcome", group_cols=["race"])
        assert len(record) == 1
        assert "race" in str(record[0].message)
        assert ds.group_cols == []

    def test_warns_for_multiple_missing_group_cols(self, tmp_path: Path):
        csv = tmp_path / "valid.csv"
        csv.write_text("age,bmi,outcome\n45,28.1,0\n62,31.4,1\n")
        with pytest.warns(UserWarning) as record:
            ds = load_csv(csv, target_col="outcome", group_cols=["race", "gender"])
        assert len(record) == 1
        msg = str(record[0].message)
        assert "race" in msg
        assert "gender" in msg
        assert ds.group_cols == []

    def test_warns_for_mixed_present_and_missing_group_cols(self, tmp_path: Path):
        csv = tmp_path / "valid.csv"
        csv.write_text("age,race,outcome\n45,White,0\n62,Black,1\n")
        with pytest.warns(UserWarning) as record:
            ds = load_csv(csv, target_col="outcome", group_cols=["race", "gender"])
        assert len(record) == 1
        msg = str(record[0].message)
        assert "gender" in msg
        # race should be kept
        assert ds.group_cols == ["race"]

    def test_no_warning_when_all_group_cols_present(self, tmp_path: Path):
        csv = tmp_path / "valid.csv"
        csv.write_text("age,race,outcome\n45,White,0\n62,Black,1\n")
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ds = load_csv(csv, target_col="outcome", group_cols=["race"])
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0
        assert ds.group_cols == ["race"]

    def test_warning_message_lists_available_columns(self, tmp_path: Path):
        csv = tmp_path / "valid.csv"
        csv.write_text("age,bmi,outcome\n45,28.1,0\n62,31.4,1\n")
        with pytest.warns(UserWarning) as record:
            load_csv(csv, target_col="outcome", group_cols=["race"])
        msg = str(record[0].message)
        assert "Available columns" in msg
        assert "age" in msg


class TestLoadCsvEmpty:
    """Test load_csv with empty or near-empty CSV files."""

    def test_empty_csv_raises(self, tmp_path: Path):
        csv = tmp_path / "empty.csv"
        csv.write_text("")
        with pytest.raises((pd.errors.EmptyDataError, ValueError)):
            load_csv(csv, target_col="target")

    def test_header_only_csv_raises(self, tmp_path: Path):
        csv = tmp_path / "header_only.csv"
        csv.write_text("a,b,target\n")
        ds = load_csv(csv, target_col="target")
        # pandas reads header-only as 0-row DataFrame; load_csv should handle it
        assert len(ds.X) == 0
        assert len(ds.y) == 0

    def test_single_row_csv(self, tmp_path: Path):
        csv = tmp_path / "single.csv"
        csv.write_text("a,b,target\n1,2,0\n")
        ds = load_csv(csv, target_col="target")
        assert len(ds.X) == 1
        assert len(ds.y) == 1
        assert ds.y.iloc[0] == 0


class TestLoadCsvEdgeCases:
    """Test load_csv with edge-case inputs."""

    def test_unicode_column_names(self, tmp_path: Path):
        csv = tmp_path / "unicode.csv"
        csv.write_text("αβ,γδ,outcome\n1,2,0\n3,4,1\n")
        ds = load_csv(csv, target_col="outcome")
        assert "αβ" in ds.X.columns
        assert "γδ" in ds.X.columns

    def test_unicode_values(self, tmp_path: Path):
        csv = tmp_path / "unicode_vals.csv"
        csv.write_text("name,target\nJosé,0\nMüller,1\n田中,0\n")
        ds = load_csv(csv, target_col="target")
        assert list(ds.X["name"]) == ["José", "Müller", "田中"]

    def test_special_characters_in_values(self, tmp_path: Path):
        csv = tmp_path / "special.csv"
        csv.write_text("desc,target\n\"hello, world\",0\n\"line1\nline2\",1\n")
        ds = load_csv(csv, target_col="target")
        assert ds.X.iloc[0, 0] == "hello, world"

    def test_whitespace_in_column_names_stripped_by_pandas(self, tmp_path: Path):
        csv = tmp_path / "ws.csv"
        csv.write_text(" col_a , col_b ,target\n1,2,0\n3,4,1\n")
        ds = load_csv(csv, target_col="target")
        # pandas preserves whitespace in column names by default
        assert " col_a " in ds.X.columns or "col_a" in ds.X.columns

    def test_all_nan_feature_column_imputed(self, tmp_path: Path):
        csv = tmp_path / "all_nan.csv"
        csv.write_text("a,b,target\n,,0\n,,1\n")
        ds = load_csv(csv, target_col="target")
        # Should not crash; imputation handles all-NaN columns
        assert "a" in ds.X.columns
        assert "b" in ds.X.columns

    def test_mixed_numeric_and_categorical(self, tmp_path: Path):
        csv = tmp_path / "mixed.csv"
        csv.write_text("age,category,target\n25,A,0\n30,B,1\n35,A,0\n")
        ds = load_csv(csv, target_col="target")
        assert ds.X["age"].dtype in (np.float64, np.int64, "int64", "float64")
        assert ds.X["category"].dtype == "object" or ds.X["category"].dtype == "string"

    def test_large_numeric_values(self, tmp_path: Path):
        csv = tmp_path / "large.csv"
        csv.write_text("big,target\n9999999999,0\n10000000000,1\n")
        ds = load_csv(csv, target_col="target")
        assert ds.X["big"].iloc[0] == 9999999999
        assert ds.X["big"].iloc[1] == 10000000000

    def test_negative_numeric_values(self, tmp_path: Path):
        csv = tmp_path / "negative.csv"
        csv.write_text("val,target\n-5.5,0\n-100.0,1\n0.0,0\n")
        ds = load_csv(csv, target_col="target")
        assert ds.X["val"].iloc[0] == -5.5
        assert ds.X["val"].iloc[1] == -100.0

    def test_boolean_like_target_values(self, tmp_path: Path):
        csv = tmp_path / "bool.csv"
        csv.write_text("x,y\n1,True\n0,False\n")
        ds = load_csv(csv, target_col="y")
        # pandas may parse True/False as booleans
        assert len(ds.y) == 2

    def test_target_column_is_also_used_as_feature_before_drop(self, tmp_path: Path):
        """Target column must NOT appear in features after loading."""
        csv = tmp_path / "valid.csv"
        csv.write_text("a,b,target\n1,2,0\n3,4,1\n")
        ds = load_csv(csv, target_col="target")
        assert "target" not in ds.X.columns

    def test_many_rows(self, tmp_path: Path):
        csv = tmp_path / "many.csv"
        lines = ["a,b,target"]
        for i in range(1000):
            lines.append(f"{i},{i*2},{i % 2}")
        csv.write_text("\n".join(lines) + "\n")
        ds = load_csv(csv, target_col="target")
        assert len(ds.X) == 1000

    def test_many_columns(self, tmp_path: Path):
        csv = tmp_path / "wide.csv"
        headers = [f"f{i}" for i in range(50)] + ["target"]
        values = [str(i) for i in range(50)] + ["1"]
        csv.write_text(",".join(headers) + "\n" + ",".join(values) + "\n")
        ds = load_csv(csv, target_col="target")
        assert len(ds.X.columns) == 50

    def test_path_with_spaces(self, tmp_path: Path):
        nested = tmp_path / "my data folder"
        nested.mkdir()
        csv = nested / "my dataset.csv"
        csv.write_text("x,y\n1,0\n2,1\n")
        ds = load_csv(csv, target_col="y")
        assert len(ds.X) == 2

    def test_null_bytes_in_string_values(self, tmp_path: Path):
        csv = tmp_path / "nullbyte.csv"
        csv.write_text("name,target\nfoo\x00bar,0\nbaz,1\n")
        ds = load_csv(csv, target_col="target")
        assert len(ds.X) == 2

    def test_imputation_preserves_non_nan_values(self, tmp_path: Path):
        csv = tmp_path / "partial_nan.csv"
        csv.write_text("a,b,target\n10.0,20.0,0\n,30.0,1\n40.0,,0\n")
        ds = load_csv(csv, target_col="target")
        assert ds.X.loc[0, "a"] == 10.0
        assert ds.X.loc[0, "b"] == 20.0
        assert ds.X.loc[1, "b"] == 30.0
        assert ds.X.loc[2, "a"] == 40.0

    def test_groups_property_with_valid_group_cols(self, tmp_path: Path):
        csv = tmp_path / "groups.csv"
        csv.write_text("age,race,target\n45,White,0\n62,Black,1\n")
        ds = load_csv(csv, target_col="target", group_cols=["race"])
        groups = ds.groups
        assert groups is not None
        assert list(groups.columns) == ["race"]
        assert list(groups["race"]) == ["White", "Black"]

    def test_groups_property_none_without_group_cols(self, tmp_path: Path):
        csv = tmp_path / "no_groups.csv"
        csv.write_text("a,b,target\n1,2,0\n3,4,1\n")
        ds = load_csv(csv, target_col="target")
        assert ds.groups is None

    def test_features_property(self, tmp_path: Path):
        csv = tmp_path / "valid.csv"
        csv.write_text("a,b,target\n1,2,0\n3,4,1\n")
        ds = load_csv(csv, target_col="target")
        assert ds.features is ds.X

    def test_target_property(self, tmp_path: Path):
        csv = tmp_path / "valid.csv"
        csv.write_text("a,b,target\n1,2,0\n3,4,1\n")
        ds = load_csv(csv, target_col="target")
        assert ds.target is ds.y


class TestLoadCsvTargetInGroupCols:
    """Test load_csv raises ValueError when target column is also a group column."""

    def test_raises_value_error_when_target_in_group_cols(self, tmp_path: Path):
        csv = tmp_path / "valid.csv"
        csv.write_text("age,bmi,outcome\n45,28.1,0\n62,31.4,1\n")
        with pytest.raises(ValueError) as exc_info:
            load_csv(csv, target_col="outcome", group_cols=["outcome"])
        assert "cannot also be a group column" in str(exc_info.value)
        assert "outcome" in str(exc_info.value)

    def test_raises_value_error_when_target_in_multiple_group_cols(self, tmp_path: Path):
        csv = tmp_path / "valid.csv"
        csv.write_text("age,race,outcome\n45,White,0\n62,Black,1\n")
        with pytest.raises(ValueError) as exc_info:
            load_csv(csv, target_col="outcome", group_cols=["race", "outcome"])
        assert "cannot also be a group column" in str(exc_info.value)
        assert "outcome" in str(exc_info.value)

    def test_no_error_when_target_not_in_group_cols(self, tmp_path: Path):
        csv = tmp_path / "valid.csv"
        csv.write_text("age,race,outcome\n45,White,0\n62,Black,1\n")
        ds = load_csv(csv, target_col="outcome", group_cols=["race"])
        assert ds.group_cols == ["race"]


class TestLoadCsvEncoding:
    """Test load_csv encoding parameter."""

    def test_loads_latin1_encoded_file(self, tmp_path: Path):
        csv = tmp_path / "latin1.csv"
        csv.write_bytes("name,target\nJos\xe9,0\nM\xfcller,1\n".encode("latin-1"))
        ds = load_csv(csv, target_col="target", encoding="latin-1")
        assert list(ds.X["name"]) == ["José", "Müller"]

    def test_defaults_to_utf8(self, tmp_path: Path):
        csv = tmp_path / "utf8.csv"
        csv.write_text("name,target\n田中,0\n山田,1\n", encoding="utf-8")
        ds = load_csv(csv, target_col="target")
        assert list(ds.X["name"]) == ["田中", "山田"]


class TestLoadCsvTargetNotImputed:
    """Test that target column values are never imputed."""

    def test_target_with_nan_not_imputed(self, tmp_path: Path):
        csv = tmp_path / "target_nan.csv"
        csv.write_text("a,b,target\n1.0,2.0,0\n3.0,4.0,\n5.0,6.0,1\n")
        ds = load_csv(csv, target_col="target")
        # The target should retain NaN, not be imputed
        assert ds.y.isna().sum() == 1
        assert ds.y.iloc[0] == 0
        assert ds.y.iloc[2] == 1
        # Features should still be imputed
        assert ds.X.isna().sum().sum() == 0

    def test_target_categorical_not_imputed(self, tmp_path: Path):
        csv = tmp_path / "target_cat_nan.csv"
        csv.write_text("a,b,target\n1.0,2.0,Yes\n3.0,4.0,\n5.0,6.0,No\n")
        ds = load_csv(csv, target_col="target")
        assert ds.y.isna().sum() == 1
        assert ds.y.iloc[0] == "Yes"
        assert ds.y.iloc[2] == "No"
        assert ds.X.isna().sum().sum() == 0
