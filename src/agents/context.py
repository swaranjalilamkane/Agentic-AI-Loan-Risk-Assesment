"""
Shared context for agents — models, SHAP explainers, test data.

Loaded once per Python process via lru_cache. All three agents call
`get_context()` and get back the same artifacts object, so the orchestrator
doesn't re-load models on every invocation.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def get_context():
    """Load models + SHAP values + test data exactly once per process."""
    from src.models.explain_borrower import _load_pipeline_artifacts
    return _load_pipeline_artifacts()
