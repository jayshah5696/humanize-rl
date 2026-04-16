"""Aggregate Layer 1 dimension scores into an overall humanness result.

All 8 dimensions weighted equally for now. Weights can be tuned
once we have benchmark AUROC data.
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
