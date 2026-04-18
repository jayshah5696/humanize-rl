"""Benchmark evaluator — score dataset and compute discrimination metrics.

Computes AUROC, accuracy, and per-dimension analysis for
Layer 1 scorer on labeled human vs AI text.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from humanize_rl.scoring.aggregator import HumannessResult, score_text


@dataclass
class ScoredSample:
    """A text sample with its label and scores."""

    id: str
    label: str  # "human" or "ai"
    source: str
    text: str
    result: HumannessResult


@dataclass
class BenchmarkReport:
    """Full benchmark results."""

    auroc: float
    accuracy: float
    threshold: float
    n_human: int
    n_ai: int
    per_dim_auroc: dict[str, float] = field(default_factory=dict)
    confusion: dict[str, int] = field(default_factory=dict)
    scored_samples: list[ScoredSample] = field(default_factory=list)
    by_label: dict[str, dict[str, float | int]] = field(default_factory=dict)
    by_source: dict[str, dict[str, float | int]] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "LAYER 1 BENCHMARK REPORT",
            "=" * 60,
            f"  Samples:    {self.n_human} human, {self.n_ai} AI",
            f"  AUROC:      {self.auroc:.4f}",
            f"  Accuracy:   {self.accuracy:.4f} (threshold={self.threshold:.2f})",
            f"  TP={self.confusion.get('tp', 0)}  FP={self.confusion.get('fp', 0)}  "
            f"TN={self.confusion.get('tn', 0)}  FN={self.confusion.get('fn', 0)}",
            "",
            "Per-dimension AUROC:",
        ]
        for dim, auc in sorted(
            self.per_dim_auroc.items(), key=lambda x: -x[1]
        ):
            bar = "█" * int(auc * 20) + "░" * (20 - int(auc * 20))
            lines.append(f"  {dim:<22s} {bar} {auc:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)


def _auroc(scores: list[float], labels: list[int]) -> float:
    """Compute AUROC via the Wilcoxon-Mann-Whitney statistic.

    No sklearn dependency — pure Python.
    labels: 1 = human (positive), 0 = AI (negative).
    Higher score should correspond to label=1 for AUROC > 0.5.
    """
    positives = [s for s, lab in zip(scores, labels, strict=True) if lab == 1]
    negatives = [s for s, lab in zip(scores, labels, strict=True) if lab == 0]

    if not positives or not negatives:
        return 0.5

    n_pos = len(positives)
    n_neg = len(negatives)

    # Count concordant pairs
    concordant = 0
    tied = 0
    for p in positives:
        for n in negatives:
            if p > n:
                concordant += 1
            elif math.isclose(p, n, abs_tol=1e-9):
                tied += 1

    return (concordant + 0.5 * tied) / (n_pos * n_neg)


def _best_threshold(
    scores: list[float], labels: list[int]
) -> tuple[float, float]:
    """Find threshold that maximizes accuracy.

    Returns (threshold, accuracy).
    """
    thresholds = sorted(set(scores))
    best_acc = 0.0
    best_t = 0.5

    for t in thresholds:
        correct = sum(
            1
            for s, lab in zip(scores, labels, strict=True)
            if (s >= t and lab == 1) or (s < t and lab == 0)
        )
        acc = correct / len(labels)
        if acc > best_acc:
            best_acc = acc
            best_t = t

    return best_t, best_acc


def load_samples(path: Path) -> list[dict]:
    """Load JSONL samples."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def summarize_by_label(
    report: BenchmarkReport,
) -> dict[str, dict[str, float | int]]:
    """Summarize scored samples by label."""
    summary: dict[str, dict[str, float | int]] = {}
    buckets: dict[str, list[ScoredSample]] = {}
    for sample in report.scored_samples:
        buckets.setdefault(sample.label, []).append(sample)

    for label, samples in buckets.items():
        mean_score = sum(sample.result.overall for sample in samples) / len(samples)
        summary[label] = {
            "count": len(samples),
            "mean_overall": round(mean_score, 4),
        }
    return summary


def summarize_by_source(
    report: BenchmarkReport,
) -> dict[str, dict[str, float | int]]:
    """Summarize scored samples by source."""
    summary: dict[str, dict[str, float | int]] = {}
    buckets: dict[str, list[ScoredSample]] = {}
    for sample in report.scored_samples:
        buckets.setdefault(sample.source, []).append(sample)

    for source, samples in buckets.items():
        mean_score = sum(sample.result.overall for sample in samples) / len(samples)
        summary[source] = {
            "count": len(samples),
            "mean_overall": round(mean_score, 4),
        }
    return summary


def evaluate(
    human_path: Path,
    ai_path: Path,
) -> BenchmarkReport:
    """Run full benchmark evaluation.

    1. Load human and AI samples
    2. Score each with Layer 1
    3. Compute AUROC, accuracy, per-dim AUROC
    """
    human_samples = load_samples(human_path)
    ai_samples = load_samples(ai_path)

    scored: list[ScoredSample] = []

    for sample in human_samples + ai_samples:
        result = score_text(sample["text"])
        scored.append(
            ScoredSample(
                id=sample["id"],
                label=sample["label"],
                source=sample.get("source", "unknown"),
                text=sample["text"],
                result=result,
            )
        )

    # Overall AUROC
    overall_scores = [s.result.overall for s in scored]
    labels = [1 if s.label == "human" else 0 for s in scored]

    auroc = _auroc(overall_scores, labels)
    threshold, accuracy = _best_threshold(overall_scores, labels)

    # Confusion matrix at best threshold
    tp = sum(1 for s, lab in zip(overall_scores, labels, strict=True) if s >= threshold and lab == 1)
    fp = sum(1 for s, lab in zip(overall_scores, labels, strict=True) if s >= threshold and lab == 0)
    tn = sum(1 for s, lab in zip(overall_scores, labels, strict=True) if s < threshold and lab == 0)
    fn = sum(1 for s, lab in zip(overall_scores, labels, strict=True) if s < threshold and lab == 1)

    # Per-dimension AUROC
    dim_names = list(scored[0].result.per_dim.keys())
    per_dim_auroc: dict[str, float] = {}
    for dim in dim_names:
        dim_scores = [s.result.per_dim[dim] for s in scored]
        per_dim_auroc[dim] = _auroc(dim_scores, labels)

    report = BenchmarkReport(
        auroc=auroc,
        accuracy=accuracy,
        threshold=threshold,
        n_human=len(human_samples),
        n_ai=len(ai_samples),
        per_dim_auroc=per_dim_auroc,
        confusion={"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        scored_samples=scored,
    )
    report.by_label = summarize_by_label(report)
    report.by_source = summarize_by_source(report)
    return report


def export_scored(
    report: BenchmarkReport,
    output_path: Path,
) -> None:
    """Export scored samples to JSONL for inspection."""
    with open(output_path, "w") as f:
        for sample in report.scored_samples:
            record = {
                "id": sample.id,
                "label": sample.label,
                "source": sample.source,
                "overall_score": round(sample.result.overall, 4),
                "per_dim": {
                    k: round(v, 4) for k, v in sample.result.per_dim.items()
                },
                "text_preview": sample.text[:100] + "..."
                if len(sample.text) > 100
                else sample.text,
            }
            f.write(json.dumps(record) + "\n")
