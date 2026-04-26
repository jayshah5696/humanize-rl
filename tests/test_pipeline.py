"""Tests for the pipeline orchestrator.

Tests the end-to-end flow:
- Pairs: load AIify output → score → verify discrimination
- Triples: load AIify + humanize → 3-class scoring → SFT pairs
"""

from __future__ import annotations

from pathlib import Path

import pytest

from humanize_rl.pipeline import (
    ScoredPair,
    ScoredTriple,
    build_triples,
    load_aiify_output,
    score_pairs,
    score_triples,
)

AIIFY_OUTPUT = Path("output/01-aiify-dataset.jsonl")
HUMANIZE_OUTPUT = Path("output/02-humanize-dataset.jsonl")


@pytest.fixture
def scored_pairs() -> list[ScoredPair]:
    if not AIIFY_OUTPUT.exists():
        pytest.skip("AIify output not found — run `just aiify` first")
    pairs = load_aiify_output(AIIFY_OUTPUT)
    return score_pairs(pairs)


@pytest.fixture
def scored_triples() -> list[ScoredTriple]:
    if not AIIFY_OUTPUT.exists() or not HUMANIZE_OUTPUT.exists():
        pytest.skip("Pipeline outputs not found — run `just aiify` then humanize")
    triples = build_triples(AIIFY_OUTPUT, HUMANIZE_OUTPUT)
    return score_triples(triples)


class TestPairScoring:
    """Tests on AIify pairs."""

    def test_all_pairs_loaded(self, scored_pairs: list[ScoredPair]) -> None:
        assert len(scored_pairs) >= 40

    def test_originals_score_higher_than_aiified(
        self, scored_pairs: list[ScoredPair]
    ) -> None:
        failures = [
            (p.id, p.original_score.overall, p.aiified_score.overall)
            for p in scored_pairs
            if p.delta <= 0
        ]
        assert len(failures) == 0, f"Pairs where AI scored >= original: {failures[:3]}"

    def test_mean_delta_above_threshold(self, scored_pairs: list[ScoredPair]) -> None:
        mean_delta = sum(p.delta for p in scored_pairs) / len(scored_pairs)
        assert mean_delta > 0.2, f"Mean delta {mean_delta:.3f} too low"


class TestTripleScoring:
    """Tests on full triples (original → AIified → humanized)."""

    def test_triples_loaded(self, scored_triples: list[ScoredTriple]) -> None:
        assert len(scored_triples) >= 40

    def test_humanized_scores_higher_than_aiified(
        self, scored_triples: list[ScoredTriple]
    ) -> None:
        """Humanization should improve score over AIified version."""
        improvements = sum(1 for t in scored_triples if t.humanize_delta > 0)
        ratio = improvements / len(scored_triples)
        assert ratio >= 0.8, f"Only {ratio:.0%} of triples improved after humanization"

    def test_humanized_mean_above_original_or_close(
        self, scored_triples: list[ScoredTriple]
    ) -> None:
        """Humanized text should score close to (or above) original human."""
        orig_mean = sum(t.original_score.overall for t in scored_triples) / len(
            scored_triples
        )
        hum_mean = sum(t.humanized_score.overall for t in scored_triples) / len(
            scored_triples
        )
        # Allow humanized to be up to 0.15 below original
        assert hum_mean >= orig_mean - 0.15, (
            f"Humanized mean {hum_mean:.3f} too far below original {orig_mean:.3f}"
        )

    def test_sft_pairs_available(self, scored_triples: list[ScoredTriple]) -> None:
        """Most triples should produce valid SFT pairs."""
        sft_ready = sum(1 for t in scored_triples if t.humanize_delta > 0.15)
        ratio = sft_ready / len(scored_triples)
        assert ratio >= 0.6, f"Only {ratio:.0%} of triples are SFT-ready"

    def test_recovery_ratio_reasonable(
        self, scored_triples: list[ScoredTriple]
    ) -> None:
        """Mean recovery ratio should be substantial."""
        mean_recovery = sum(t.recovery_ratio for t in scored_triples) / len(
            scored_triples
        )
        assert mean_recovery >= 0.5, f"Mean recovery ratio {mean_recovery:.1%} too low"
