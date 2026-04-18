"""Tests for benchmark evaluator.

These tests verify:
1. AUROC computation is correct
2. Layer 1 actually discriminates human vs AI on real samples
3. Export works
"""

from __future__ import annotations

from pathlib import Path

import pytest

from humanize_rl.benchmark.evaluator import (
    BenchmarkReport,
    _auroc,
    _best_threshold,
    evaluate,
    export_scored,
    summarize_by_domain,
    summarize_by_label,
    summarize_by_source,
)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "benchmark"


# ---------------------------------------------------------------------------
# Unit tests for AUROC
# ---------------------------------------------------------------------------


class TestAUROC:
    def test_perfect_separation(self) -> None:
        scores = [0.9, 0.8, 0.7, 0.1, 0.2, 0.3]
        labels = [1, 1, 1, 0, 0, 0]
        assert _auroc(scores, labels) == 1.0

    def test_random_gives_half(self) -> None:
        # All same score → 0.5 (tied)
        scores = [0.5, 0.5, 0.5, 0.5]
        labels = [1, 1, 0, 0]
        assert _auroc(scores, labels) == 0.5

    def test_inverse_gives_zero(self) -> None:
        scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        labels = [1, 1, 1, 0, 0, 0]
        assert _auroc(scores, labels) == 0.0

    def test_empty_positives(self) -> None:
        assert _auroc([0.5], [0]) == 0.5


class TestBestThreshold:
    def test_finds_optimal(self) -> None:
        scores = [0.9, 0.8, 0.2, 0.1]
        labels = [1, 1, 0, 0]
        threshold, accuracy = _best_threshold(scores, labels)
        assert accuracy == 1.0


# ---------------------------------------------------------------------------
# Integration: real dataset benchmark
# ---------------------------------------------------------------------------


class TestBenchmarkOnRealData:
    """These are the actual discrimination tests — the tangible result."""

    @pytest.fixture
    def report(self) -> BenchmarkReport:
        human_path = DATA_DIR / "human_samples.jsonl"
        ai_path = DATA_DIR / "ai_samples.jsonl"
        if not human_path.exists() or not ai_path.exists():
            pytest.skip("Benchmark data not found")
        return evaluate(human_path, ai_path)

    def test_auroc_above_minimum(self, report: BenchmarkReport) -> None:
        """Layer 1 alone should achieve AUROC ≥ 0.75 (design target)."""
        assert report.auroc >= 0.75, (
            f"AUROC {report.auroc:.4f} below minimum 0.75"
        )

    def test_accuracy_reasonable(self, report: BenchmarkReport) -> None:
        """At best threshold, accuracy should be ≥ 70%."""
        assert report.accuracy >= 0.70, (
            f"Accuracy {report.accuracy:.4f} below 70%"
        )

    def test_sentence_variance_above_chance(self, report: BenchmarkReport) -> None:
        """sentence_variance should be above chance (0.5).

        Note: rival.tips found sentence_variance as #1 discriminator across
        178 models. On our curated dataset, structural features (list_overuse,
        opener_pattern) dominate. sentence_variance may become stronger on
        more diverse / longer-form data where AI text has uniform sentence rhythm.
        """
        assert report.per_dim_auroc["sentence_variance"] > 0.5, (
            f"sentence_variance AUROC={report.per_dim_auroc['sentence_variance']:.4f}"
        )

    def test_no_dimension_is_useless(self, report: BenchmarkReport) -> None:
        """Every dimension should have AUROC > 0.5 (better than random)."""
        for dim, auc in report.per_dim_auroc.items():
            assert auc >= 0.5, f"{dim} AUROC={auc:.4f} — worse than random"

    def test_print_report(self, report: BenchmarkReport) -> None:
        """Not a real assertion — prints the report for visibility."""
        print("\n" + str(report))

    def test_export_scored_output(
        self, report: BenchmarkReport, tmp_path: Path
    ) -> None:
        """Scored output can be exported to JSONL."""
        out = tmp_path / "scored.jsonl"
        export_scored(report, out)
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        assert len(lines) == report.n_human + report.n_ai

    def test_summarize_by_label(self, report: BenchmarkReport) -> None:
        summary = summarize_by_label(report)
        assert set(summary) == {"human", "ai"}
        assert summary["human"]["count"] == report.n_human
        assert summary["ai"]["count"] == report.n_ai

    def test_summarize_by_source(self, report: BenchmarkReport) -> None:
        summary = summarize_by_source(report)
        assert summary
        total = sum(bucket["count"] for bucket in summary.values())
        assert total == report.n_human + report.n_ai

    def test_summarize_by_domain(self, report: BenchmarkReport) -> None:
        summary = summarize_by_domain(report)
        assert summary
        total = sum(bucket["count"] for bucket in summary.values())
        assert total == report.n_human + report.n_ai
