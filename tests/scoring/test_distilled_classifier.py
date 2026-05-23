"""Unit tests for the distilled classifier module."""

from __future__ import annotations

import pytest

from humanize_rl.scoring.distilled_classifier import HAS_ML_DEPS, DistilledScoreResult


def test_has_ml_deps_constant() -> None:
    """Verify that HAS_ML_DEPS constant is boolean."""
    assert isinstance(HAS_ML_DEPS, bool)


def test_distilled_score_result_structure() -> None:
    """Verify that DistilledScoreResult can be instantiated."""
    result = DistilledScoreResult(
        overall=0.8, ai_probability=0.2, per_dim={"specificity": 0.9}, latency_ms=1.5
    )
    assert result.overall == 0.8
    assert result.ai_probability == 0.2
    assert result.per_dim == {"specificity": 0.9}
    assert result.latency_ms == 1.5


def test_model_graceful_error_without_deps() -> None:
    """Verify that model raising correct error if instantiated without torch/transformers."""
    if not HAS_ML_DEPS:
        from humanize_rl.scoring.distilled_classifier import (
            DistilledScorer,
            StylisticDeBERTa,
        )

        with pytest.raises(
            RuntimeError, match="PyTorch and Hugging Face Transformers are required"
        ):
            StylisticDeBERTa()

        with pytest.raises(
            ImportError, match="PyTorch and Hugging Face Transformers are required"
        ):
            DistilledScorer("dummy-path")
