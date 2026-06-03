"""Explainability layer for fair-clinical-bench."""

from fair_clinical_bench.explain.global_ import shap_global_importance
from fair_clinical_bench.explain.local import shap_local_explanations

__all__ = [
    "shap_local_explanations",
    "shap_global_importance",
]
