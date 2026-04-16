"""Tests for the pipeline orchestrator.

Tests the end-to-end flow: load AIify output → score pairs → verify discrimination.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from humanize_rl.pipeline import (
    ScoredPair,
    load_aiify_output,
    score_pairs,
)

AIIFY_OUTPUT = Path("output/01-aiify-dataset.jsonl")


@pytest.fixture
def scored_pairs() -> list[ScoredPair]:
    if not AIIFY_OUTPUT.exists():
        pytest.skip("AIify output not found — run `just aiify` first")
    pairs = load_aiify_output(AIIFY_OUTPUT)
    return score_pairs(pairs)


class TestPipelineEndToEnd:
    """End-to-end pipeline tests on real LLM-generated data."""

    def test_all_pairs_loaded(self, scored_pairs: list[ScoredPair]) -> None:
        assert len(scored_pairs) >= 40, f"Only {len(scored_pairs)} pairs loaded"

    def test_originals_score_higher_than_aiified(
        self, scored_pairs: list[ScoredPair]
    ) -> None:
        """Every original should score higher than its AI-ified version."""
        failures = [
            (p.id, p.original_score.overall, p.aiified_score.overall)
            for p in scored_pairs
            if p.delta <= 0
        ]
        assert len(failures) == 0, (
            f"{len(failures)} pairs where AI-ified scored >= original: {failures[:3]}"
        )

    def test_mean_delta_above_threshold(
        self, scored_pairs: list[ScoredPair]
    ) -> None:
        """Mean score delta should be substantial (>0.2)."""
        mean_delta = sum(p.delta for p in scored_pairs) / len(scored_pairs)
        assert mean_delta > 0.2, f"Mean delta {mean_delta:.3f} too low"

    def test_aiified_mean_below_half(
        self, scored_pairs: list[ScoredPair]
    ) -> None:
        """AI-ified text should average below 0.5 humanness."""
        mean_ai = sum(
            p.aiified_score.overall for p in scored_pairs
        ) / len(scored_pairs)
        assert mean_ai < 0.55, f"AI-ified mean {mean_ai:.3f} too high"

    def test_original_mean_above_threshold(
        self, scored_pairs: list[ScoredPair]
    ) -> None:
        """Original human text should average above 0.7 humanness."""
        mean_human = sum(
            p.original_score.overall for p in scored_pairs
        ) / len(scored_pairs)
        assert mean_human > 0.7, f"Original mean {mean_human:.3f} too low"
