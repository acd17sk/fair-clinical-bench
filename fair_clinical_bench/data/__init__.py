"""Data layer for fair-clinical-bench."""

from .loader import list_builtin_datasets, load_csv, load_diabetes130, load_heart, load_pima
from .schema import ClinicalDataset
from .splits import SplitIndices, stratified_split

__all__ = [
    "ClinicalDataset",
    "SplitIndices",
    "list_builtin_datasets",
    "load_csv",
    "load_diabetes130",
    "load_heart",
    "load_pima",
    "stratified_split",
]
