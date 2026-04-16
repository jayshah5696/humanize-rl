"""Aggregate Layer 1 (and optionally Layer 2) dimension scores.

Layer 1: 8 deterministic dimensions, free, <10ms.
Layer 2: 8 LLM judge dimensions, ~$0.01/sample, 2-5s.
Combined: 0.4 * L1 + 0.6 * L2 (per 08-two-layer-scoring.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from humanize_rl.scoring.layer1 import (
    score_closing,
    score_contractions,
    score_em_dash,
    score_hedging,
    score_list_overuse,
    score_opener,
    score_sentence_variance,
    score_transitions,
)

# Default weights — equal until benchmark-tuned
DEFAULT_WEIGHTS: dict[str, float] = {
    "opener_pattern": 1.0,
    "hedging_density": 1.0,
    "list_overuse": 1.0,
    "sentence_variance": 1.0,
    "contractions": 1.0,
    "closing_pattern": 1.0,
    "em_dash_density": 1.0,
    "transition_overuse": 1.0,
}

# Mapping from dimension name to scoring function
_SCORERS: dict[str, callable] = {
    "opener_pattern": score_opener,
    "hedging_density": score_hedging,
    "list_overuse": score_list_overuse,
    "sentence_variance": score_sentence_variance,
    "contractions": score_contractions,
    "closing_pattern": score_closing,
    "em_dash_density": score_em_dash,
    "transition_overuse": score_transitions,
}


@dataclass(frozen=True)
class HumannessResult:
    """Result of Layer 1 humanness scoring."""

    overall: float
    per_dim: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [f"Overall: {self.overall:.3f}"]
        for dim, score in sorted(self.per_dim.items()):
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            lines.append(f"  {dim:<22s} {bar} {score:.2f}")
        return "\n".join(lines)


def score_text(
    text: str,
    weights: dict[str, float] | None = None,
) -> HumannessResult:
    """Score text across all 8 Layer 1 dimensions.

    Args:
        text: Input text to score.
        weights: Optional dimension weights. Defaults to equal weighting.

    Returns:
        HumannessResult with overall score and per-dimension breakdown.
    """
    w = weights or DEFAULT_WEIGHTS

    per_dim: dict[str, float] = {}
    for dim_name, scorer in _SCORERS.items():
        per_dim[dim_name] = scorer(text)

    # Weighted average
    total_weight = sum(w.get(d, 1.0) for d in per_dim)
    weighted_sum = sum(per_dim[d] * w.get(d, 1.0) for d in per_dim)
    overall = weighted_sum / total_weight if total_weight > 0 else 0.0

    return HumannessResult(overall=overall, per_dim=per_dim)


@dataclass(frozen=True)
class CombinedResult:
    """Combined Layer 1 + Layer 2 humanness scoring."""

    overall: float
    layer1: HumannessResult
    layer2_overall: float
    layer2_per_dim: dict[str, float]
    layer2_raw: dict[str, int]
    layer2_reasoning: str = ""

    def __str__(self) -> str:
        lines = [
            f"Combined: {self.overall:.3f}  "
            f"(L1={self.layer1.overall:.3f}, L2={self.layer2_overall:.3f})",
            "",
            "Layer 1 (deterministic):",
        ]
        for dim, score in sorted(self.layer1.per_dim.items()):
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            lines.append(f"  {dim:<22s} {bar} {score:.2f}")
        lines.append("")
        lines.append("Layer 2 (LLM judge):")
        for dim, score in sorted(self.layer2_per_dim.items()):
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            raw = self.layer2_raw.get(dim, 0)
            lines.append(f"  {dim:<28s} {bar} {score:.2f} (raw={raw})")
        return "\n".join(lines)


def combine_scores(
    layer1: HumannessResult,
    layer2_overall: float,
    layer2_per_dim: dict[str, float],
    layer2_raw: dict[str, int],
    layer2_reasoning: str = "",
    layer1_weight: float = 0.4,
    layer2_weight: float = 0.6,
) -> CombinedResult:
    """Combine Layer 1 and Layer 2 scores.

    Default weights: 0.4 * L1 + 0.6 * L2 (per design doc).
    Layer 2 gets more weight because it captures what regex cannot.
    """
    overall = (
        layer1_weight * layer1.overall + layer2_weight * layer2_overall
    )
    return CombinedResult(
        overall=overall,
        layer1=layer1,
        layer2_overall=layer2_overall,
        layer2_per_dim=layer2_per_dim,
        layer2_raw=layer2_raw,
        layer2_reasoning=layer2_reasoning,
    )
