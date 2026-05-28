"""Core data schema for fair-clinical-bench."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ClinicalDataset:
    """Container for a clinical dataset with fairness-relevant group columns.

    Attributes
    ----------
    X : pd.DataFrame
        Feature matrix (n_samples, n_features).
    y : pd.Series
        Target variable (n_samples,).
    group_cols : list[str]
        Column names in *X* that identify protected / demographic groups.
    name : str
        Human-readable dataset identifier.
    description : str | None
        Optional longer description of the dataset.
    """

    X: pd.DataFrame
    y: pd.Series
    group_cols: list[str] = field(default_factory=list)
    name: str = ""
    description: str | None = None

    def __post_init__(self) -> None:
        """Validate dataset invariants."""
        if len(self.X) != len(self.y):
            raise ValueError(
                f"X and y row counts do not align: X has {len(self.X)} rows, "
                f"y has {len(self.y)} rows"
            )

        if not self.group_cols:
            return

        if len(self.group_cols) != len(set(self.group_cols)):
            seen = set()
            duplicates = []
            for col in self.group_cols:
                if col in seen:
                    duplicates.append(col)
                seen.add(col)
            raise ValueError(f"Duplicate group columns detected: {duplicates}")

        missing = [col for col in self.group_cols if col not in self.X.columns]
        if missing:
            available = list(self.X.columns)
            raise ValueError(
                f"Group column(s) not found in X: {missing}. "
                f"Available columns: {available}"
            )

    @property
    def features(self) -> pd.DataFrame:
        """Return the feature matrix *X*."""
        return self.X

    @property
    def target(self) -> pd.Series:
        """Return the target variable *y*."""
        return self.y

    @property
    def groups(self) -> pd.DataFrame | None:
        """Return the group columns from *X*, or *None* if no groups are defined."""
        if not self.group_cols:
            return None
        return self.X[self.group_cols]
