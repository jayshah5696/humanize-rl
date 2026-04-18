"""Tests for Layer 2 gate thresholds."""

from __future__ import annotations

from humanize_rl.scoring.gate import needs_layer2


class TestNeedsLayer2:
    def test_source_scoring_skips_clearly_human(self) -> None:
        assert needs_layer2(0.86, "source_scoring") is False

    def test_source_scoring_skips_clearly_ai(self) -> None:
        assert needs_layer2(0.29, "source_scoring") is False

    def test_source_scoring_keeps_boundary_scores(self) -> None:
        assert needs_layer2(0.85, "source_scoring") is True
        assert needs_layer2(0.30, "source_scoring") is True

    def test_aiified_scoring_skips_only_obvious_ai(self) -> None:
        assert needs_layer2(0.24, "aiified_scoring") is False
        assert needs_layer2(0.25, "aiified_scoring") is True

    def test_humanized_scoring_always_runs(self) -> None:
        assert needs_layer2(0.01, "humanized_scoring") is True
        assert needs_layer2(0.99, "humanized_scoring") is True

    def test_unknown_context_defaults_to_run(self) -> None:
        assert needs_layer2(0.10, "other") is True
