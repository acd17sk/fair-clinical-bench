"""Reproducible stratified train/val/test splits."""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class SplitIndices(NamedTuple):
    """Integer indices for a train/val/test split.

    Attributes
    ----------
    train : np.ndarray
        Integer indices for the training set.
    val : np.ndarray
        Integer indices for the validation set.
    test : np.ndarray
        Integer indices for the test set.
    """

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def stratified_split(
    X: pd.DataFrame | np.ndarray,  # noqa: N803
    y: pd.Series | np.ndarray,
    group_col: str | pd.Series | np.ndarray | None = None,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    random_seed: int = 42,
) -> SplitIndices:
    """Create stratified train/val/test splits with reproducible seeding.

    Parameters
    ----------
    X : pd.DataFrame | np.ndarray
        Feature matrix (n_samples, n_features).
    y : pd.Series | np.ndarray
        Target variable (n_samples,).
    group_col : str | pd.Series | np.ndarray | None, optional
        Group column for additional stratification. If a string, it must be a
        column name in *X* (requires *X* to be a DataFrame). If a Series or
        array, it must have the same length as *y*.
    train_frac : float, optional
        Fraction of data for the training set (default 0.6).
    val_frac : float, optional
        Fraction of data for the validation set (default 0.2).
    test_frac : float, optional
        Fraction of data for the test set (default 0.2).
    random_seed : int, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    SplitIndices
        Named tuple with *train*, *val*, and *test* integer index arrays.

    Raises
    ------
    ValueError
        If the fractions do not sum to approximately 1.0, or if *group_col* is a
        string not found in *X* columns.
    TypeError
        If *group_col* is a string but *X* is not a DataFrame.
    """
    total_frac = train_frac + val_frac + test_frac
    if not np.isclose(total_frac, 1.0, atol=1e-6):
        raise ValueError(
            f"Fractions must sum to 1.0 (within tolerance), got {total_frac:.6f} "
            f"(train={train_frac}, val={val_frac}, test={test_frac})"
        )

    y_arr: np.ndarray = y.values if isinstance(y, pd.Series) else np.asarray(y)
    n_samples = len(y_arr)
    indices = np.arange(n_samples)

    stratify: np.ndarray = y_arr

    if group_col is not None:
        groups: np.ndarray
        if isinstance(group_col, str):
            if not isinstance(X, pd.DataFrame):
                raise TypeError(
                    "group_col as string requires X to be a pandas DataFrame"
                )
            if group_col not in X.columns:
                available = list(X.columns)
                raise ValueError(
                    f"group_col '{group_col}' not found in X columns. "
                    f"Available columns: {available}"
                )
            groups = X[group_col].values
        elif isinstance(group_col, pd.Series):
            groups = group_col.values
        elif isinstance(group_col, np.ndarray):
            groups = group_col
        else:
            raise TypeError(
                "group_col must be a string, pandas Series, or numpy array"
            )

        if len(groups) != n_samples:
            raise ValueError(
                f"group_col length ({len(groups)}) does not match "
                f"y length ({n_samples})"
            )

        unique_groups = np.unique(groups)
        if len(unique_groups) == 1:
            warnings.warn(
                "Only one unique group found in group_col. "
                "Proceeding with target-only stratification.",
                UserWarning,
                stacklevel=2,
            )
        else:
            combined = np.array([f"{g}::{t}" for g, t in zip(groups, y_arr, strict=False)])
            _, counts = np.unique(combined, return_counts=True)
            if np.any(counts < 2):
                warnings.warn(
                    "Some group+target combinations have fewer than 2 samples. "
                    "Falling back to target-only stratification.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                stratify = combined

    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_frac,
        stratify=stratify,
        random_state=random_seed,
    )

    # Second split: val vs test from the temp set
    temp_stratify = stratify[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_frac / (val_frac + test_frac),
        stratify=temp_stratify,
        random_state=random_seed,
    )

    return SplitIndices(
        train=np.asarray(train_idx, dtype=int),
        val=np.asarray(val_idx, dtype=int),
        test=np.asarray(test_idx, dtype=int),
    )
