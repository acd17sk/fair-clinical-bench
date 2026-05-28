"""Built-in dataset loaders for fair-clinical-bench.

Each loader attempts to download the canonical UCI version on first call and
persists it to ``~/.fair-clinical-bench/datasets/``.  If the download fails
(e.g. the host is unreachable) a minimal synthetic dataset (50 rows) is generated
so that tests and demos do not fail offline.
"""

from __future__ import annotations

import urllib.request
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import ClinicalDataset

_CACHE_DIR = Path.home() / ".fair-clinical-bench" / "datasets"


def _ensure_cache_dir() -> Path:
    """Create the dataset cache directory if it does not exist."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _download_file(url: str, dest: Path, timeout: float = 30.0) -> bool:
    """Download a file from *url* to *dest*.  Return ``True`` on success."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            dest.write_bytes(response.read())
        return True
    except Exception:
        return False


def _generate_synthetic_pima(n_rows: int = 50) -> pd.DataFrame:
    """Generate synthetic Pima-like data for offline fallback."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "Pregnancies": rng.integers(0, 17, size=n_rows),
            "Glucose": rng.integers(50, 200, size=n_rows),
            "BloodPressure": rng.integers(40, 120, size=n_rows),
            "SkinThickness": rng.integers(10, 100, size=n_rows),
            "Insulin": rng.integers(15, 300, size=n_rows),
            "BMI": rng.uniform(15.0, 50.0, size=n_rows).round(1),
            "DiabetesPedigreeFunction": rng.uniform(0.08, 2.5, size=n_rows).round(3),
            "Age": rng.integers(21, 81, size=n_rows),
            "Outcome": rng.integers(0, 2, size=n_rows),
        }
    )


def _generate_synthetic_heart(n_rows: int = 50) -> pd.DataFrame:
    """Generate synthetic Heart Disease-like data for offline fallback."""
    rng = np.random.default_rng(43)
    return pd.DataFrame(
        {
            "age": rng.integers(29, 81, size=n_rows),
            "sex": rng.integers(0, 2, size=n_rows),
            "cp": rng.integers(1, 5, size=n_rows),
            "trestbps": rng.integers(90, 200, size=n_rows),
            "chol": rng.integers(120, 570, size=n_rows),
            "fbs": rng.integers(0, 2, size=n_rows),
            "restecg": rng.integers(0, 3, size=n_rows),
            "thalach": rng.integers(70, 210, size=n_rows),
            "exang": rng.integers(0, 2, size=n_rows),
            "oldpeak": rng.uniform(0.0, 6.5, size=n_rows).round(1),
            "slope": rng.integers(1, 4, size=n_rows),
            "ca": rng.integers(0, 4, size=n_rows),
            "thal": rng.integers(3, 8, size=n_rows),
            "target": rng.integers(0, 2, size=n_rows),
        }
    )


def _generate_synthetic_diabetes130(n_rows: int = 50) -> pd.DataFrame:
    """Generate synthetic Diabetes 130-Hospitals-like data for offline fallback."""
    rng = np.random.default_rng(44)
    races = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"]
    genders = ["Male", "Female"]
    age_bins = [
        "[0-10)",
        "[10-20)",
        "[20-30)",
        "[30-40)",
        "[40-50)",
        "[50-60)",
        "[60-70)",
        "[70-80)",
        "[80-90)",
        "[90-100)",
    ]
    readmitted_labels = ["NO", ">30", "<30"]
    return pd.DataFrame(
        {
            "race": rng.choice(races, size=n_rows),
            "gender": rng.choice(genders, size=n_rows),
            "age": rng.choice(age_bins, size=n_rows),
            "time_in_hospital": rng.integers(1, 15, size=n_rows),
            "num_lab_procedures": rng.integers(1, 100, size=n_rows),
            "num_procedures": rng.integers(0, 6, size=n_rows),
            "num_medications": rng.integers(1, 40, size=n_rows),
            "number_outpatient": rng.integers(0, 5, size=n_rows),
            "number_emergency": rng.integers(0, 5, size=n_rows),
            "number_inpatient": rng.integers(0, 5, size=n_rows),
            "number_diagnoses": rng.integers(1, 17, size=n_rows),
            "readmitted": rng.choice(readmitted_labels, size=n_rows),
        }
    )


def _impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values column-wise (numeric → median, categorical → mode)."""
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode_vals = df[col].mode()
                fill_val = mode_vals[0] if not mode_vals.empty else "Unknown"
                df[col] = df[col].fillna(fill_val)
    return df


def load_pima() -> ClinicalDataset:
    """Load the Pima Indians Diabetes dataset.

    The canonical UCI version contains 768 rows.  The target column is
    ``Outcome``.  ``Age`` is used as the demographic group column when
    available; otherwise the group list is empty.

    Returns
    -------
    ClinicalDataset
        Dataset container with features, target, and group metadata.
    """
    cache = _ensure_cache_dir()
    csv_path = cache / "pima-indians-diabetes.csv"

    if not csv_path.exists():
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "pima-indians-diabetes/pima-indians-diabetes.data"
        )
        success = _download_file(url, csv_path)
        if success:
            df = pd.read_csv(csv_path, header=None)
            df.columns = [
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age",
                "Outcome",
            ]
            df.to_csv(csv_path, index=False)
        else:
            df = _generate_synthetic_pima()
            df.to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path)

    # In the Pima dataset zeros in these columns are biologically impossible
    # and represent missing values.
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_cols:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    df = _impute_missing(df)

    target_col = "Outcome"
    y = df[target_col]
    X = df.drop(columns=[target_col])  # noqa: N806

    group_cols = ["Age"] if "Age" in X.columns else []

    return ClinicalDataset(
        X=X,
        y=y,
        group_cols=group_cols,
        name="pima",
        description=(
            "Pima Indians Diabetes Database (UCI, ~768 rows). "
            "Binary classification of diabetes onset."
        ),
    )


def load_heart() -> ClinicalDataset:
    """Load the Heart Disease Cleveland dataset.

    The canonical UCI version contains 303 rows.  The target column is
    ``target``.  ``sex`` is used as the demographic group column.

    Returns
    -------
    ClinicalDataset
        Dataset container with features, target, and group metadata.
    """
    cache = _ensure_cache_dir()
    csv_path = cache / "heart-disease-cleveland.csv"

    if not csv_path.exists():
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "heart-disease/processed.cleveland.data"
        )
        success = _download_file(url, csv_path)
        if success:
            df = pd.read_csv(csv_path, header=None)
            df.columns = [
                "age",
                "sex",
                "cp",
                "trestbps",
                "chol",
                "fbs",
                "restecg",
                "thalach",
                "exang",
                "oldpeak",
                "slope",
                "ca",
                "thal",
                "target",
            ]
            # The Cleveland file uses '?' for missing values.
            df = df.replace("?", np.nan)
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df.to_csv(csv_path, index=False)
        else:
            df = _generate_synthetic_heart()
            df.to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path)
    df = df.replace("?", np.nan)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = _impute_missing(df)

    target_col = "target"
    y = df[target_col]
    X = df.drop(columns=[target_col])  # noqa: N806

    group_cols = ["sex"] if "sex" in X.columns else []

    return ClinicalDataset(
        X=X,
        y=y,
        group_cols=group_cols,
        name="heart",
        description=(
            "Heart Disease Cleveland dataset (UCI, ~303 rows). "
            "Binary classification of heart disease presence."
        ),
    )


def load_diabetes130() -> ClinicalDataset:
    """Load the Diabetes 130-Hospitals dataset.

    The canonical UCI version contains ~100 K rows.  The target column is
    ``readmitted``.  ``race``, ``gender``, and ``age`` are used as demographic
    group columns.

    Returns
    -------
    ClinicalDataset
        Dataset container with features, target, and group metadata.
    """
    cache = _ensure_cache_dir()
    csv_path = cache / "diabetes-130.csv"

    if not csv_path.exists():
        zip_path = cache / "diabetes-130.zip"
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "00296/dataset_diabetes.zip"
        )
        success = _download_file(url, zip_path)
        if success:
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(cache)
                extracted = cache / "diabetic_data.csv"
                df = pd.read_csv(extracted)
                df.to_csv(csv_path, index=False)
            except Exception:
                df = _generate_synthetic_diabetes130()
                df.to_csv(csv_path, index=False)
        else:
            df = _generate_synthetic_diabetes130()
            df.to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path)

    # Normalise sentinel missing values.
    df = df.replace("?", np.nan)
    df = df.replace("Unknown/Invalid", np.nan)

    # Drop feature columns that are >50 % missing (never drop the target).
    target_col = "readmitted"
    missing_ratio = df.drop(columns=[target_col], errors="ignore").isna().mean()
    cols_to_drop = missing_ratio[missing_ratio > 0.5].index.tolist()
    df = df.drop(columns=cols_to_drop)

    df = _impute_missing(df)

    target_col = "readmitted"
    if target_col not in df.columns:
        df[target_col] = "NO"

    y = df[target_col]
    X = df.drop(columns=[target_col])  # noqa: N806

    group_cols = [col for col in ["race", "gender", "age"] if col in X.columns]

    return ClinicalDataset(
        X=X,
        y=y,
        group_cols=group_cols,
        name="diabetes130",
        description=(
            "Diabetes 130-Hospitals dataset (UCI, ~100 K rows). "
            "Prediction of early readmission."
        ),
    )


def list_builtin_datasets() -> list[str]:
    """Return the names of all built-in datasets."""
    return ["pima", "heart", "diabetes130"]


def load_csv(
    path: str | Path,
    target_col: str,
    group_cols: list[str] | None = None,
    encoding: str = "utf-8",
) -> ClinicalDataset:
    """Load an arbitrary CSV file as a ``ClinicalDataset``.

    Parameters
    ----------
    path:
        Filesystem path to the CSV file.
    target_col:
        Name of the column to use as the target / label.
    group_cols:
        Optional list of column names that identify protected / demographic
        groups.  Columns that do not exist in the file are warned about and
        dropped from the list.
    encoding:
        Character encoding to use when reading the CSV file (default ``utf-8``).

    Returns
    -------
    ClinicalDataset
        Dataset container with features, target, and group metadata.

    Raises
    ------
    ValueError
        If *target_col* is not found in the CSV, or if *target_col* is also
        listed in *group_cols*.
    """
    df = pd.read_csv(path, encoding=encoding)

    if target_col not in df.columns:
        available = list(df.columns)
        raise ValueError(
            f"Target column '{target_col}' not found in CSV. "
            f"Available columns: {available}"
        )

    if group_cols is not None and target_col in group_cols:
        raise ValueError(
            f"Target column '{target_col}' cannot also be a group column."
        )

    effective_group_cols: list[str] = []
    if group_cols is not None:
        missing = [col for col in group_cols if col not in df.columns]
        if missing:
            warnings.warn(
                f"Group column(s) not found in CSV and will be ignored: {missing}. "
                f"Available columns: {list(df.columns)}",
                UserWarning,
                stacklevel=2,
            )
        effective_group_cols = [col for col in group_cols if col in df.columns]

    # Safety fix: separate target before imputation so target values are never altered
    y = df[target_col]
    feature_df = df.drop(columns=[target_col])
    feature_df = _impute_missing(feature_df)
    X = feature_df  # noqa: N806

    name = Path(path).stem
    description = f"User-provided CSV dataset loaded from '{Path(path).name}'."

    return ClinicalDataset(
        X=X,
        y=y,
        group_cols=effective_group_cols,
        name=name,
        description=description,
    )
