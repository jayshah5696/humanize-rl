"""V-Slice 0 pair-acceptance gate.

Implements the minimum subset of v03 spec section 10:
- score thresholds (aiified <= 0.45, humanized >= 0.75, deltas)
- length ratio checks
- "humanized must beat aiified on >= 2 dimensions"
- recovery-ratio suspicion flag

V-Slice 3 will extend this with entity / number / discourse-role preservation
checks. Until then, we only enforce what we can compute without an extra LLM
call — keeps the walking skeleton free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from humanize_rl.data.preservation import (
    PreservationResult,
    evaluate_preservation,
)

_WORD_RE = re.compile(r"\b\w+\b")


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


@dataclass(frozen=True)
class GateThresholds:
    max_aiified_score: float = 0.45
    min_humanized_score: float = 0.75
    min_humanize_delta: float = 0.30
    min_aiify_delta: float = 0.25
    min_dim_improvements: int = 2

    # Length ratios — see spec §10.2 (preferred bands; we only enforce hard maxes)
    max_aiify_to_original: float = 1.25
    max_humanized_to_aiified: float = 1.20
    max_humanized_to_original: float = 1.20
    min_length_ratio: float = 0.75  # symmetric floor for any of the three ratios

    # Suspicion (don't reject, just flag)
    max_recovery_ratio: float = 1.25
    max_humanized_minus_original: float = 0.05  # preferred; flag if exceeded


@dataclass(frozen=True)
class GateResult:
    accepted: bool
    rejected_reasons: tuple[str, ...]
    suspicion_flags: tuple[str, ...]
    metrics: dict[str, float]
    aiify_preservation: PreservationResult | None = None
    humanize_preservation: PreservationResult | None = None


def _length_ratio(num: str, den: str) -> float:
    d = max(_word_count(den), 1)
    return _word_count(num) / d


def _dimension_improvements(
    aiified_per_dim: dict[str, float],
    humanized_per_dim: dict[str, float],
    epsilon: float = 0.05,
) -> int:
    n = 0
    for dim, ai_score in aiified_per_dim.items():
        h_score = humanized_per_dim.get(dim, ai_score)
        if h_score - ai_score >= epsilon:
            n += 1
    return n


def evaluate_triple(
    *,
    original_text: str,
    aiified_text: str,
    humanized_text: str,
    original_score: float,
    aiified_score: float,
    humanized_score: float,
    aiified_per_dim: dict[str, float],
    humanized_per_dim: dict[str, float],
    thresholds: GateThresholds | None = None,
    enforce_preservation: bool = True,
) -> GateResult:
    """Evaluate one triple against the v03 (V-Slice 0 subset) gate.

    Returns the pass/fail decision plus the metrics we computed, so the
    walking-skeleton report can show *why* something was rejected without
    rerunning the math.
    """
    t = thresholds or GateThresholds()
    rejected: list[str] = []
    flags: list[str] = []

    # --- score thresholds ---------------------------------------------------
    if aiified_score > t.max_aiified_score:
        rejected.append(f"aiified_score>{t.max_aiified_score}")
    if humanized_score < t.min_humanized_score:
        rejected.append(f"humanized_score<{t.min_humanized_score}")

    aiify_delta = original_score - aiified_score
    humanize_delta = humanized_score - aiified_score
    recovery_ratio = humanize_delta / aiify_delta if aiify_delta > 0 else 0.0

    if aiify_delta < t.min_aiify_delta:
        rejected.append(f"aiify_delta<{t.min_aiify_delta}")
    if humanize_delta < t.min_humanize_delta:
        rejected.append(f"humanize_delta<{t.min_humanize_delta}")

    # --- dimension improvements --------------------------------------------
    n_improved = _dimension_improvements(aiified_per_dim, humanized_per_dim)
    if n_improved < t.min_dim_improvements:
        rejected.append(f"dim_improvements<{t.min_dim_improvements}")

    # --- length ratios ------------------------------------------------------
    r_ai = _length_ratio(aiified_text, original_text)
    r_hum_ai = _length_ratio(humanized_text, aiified_text)
    r_hum_orig = _length_ratio(humanized_text, original_text)

    for name, value, max_v in (
        ("aiify/original", r_ai, t.max_aiify_to_original),
        ("humanized/aiified", r_hum_ai, t.max_humanized_to_aiified),
        ("humanized/original", r_hum_orig, t.max_humanized_to_original),
    ):
        if value > max_v:
            rejected.append(f"length_ratio_{name}>{max_v}")
        if value < t.min_length_ratio:
            rejected.append(f"length_ratio_{name}<{t.min_length_ratio}")

    # --- suspicion flags (do NOT reject) -----------------------------------
    if recovery_ratio > t.max_recovery_ratio:
        flags.append(f"recovery_ratio>{t.max_recovery_ratio}")
    if humanized_score - original_score > t.max_humanized_minus_original:
        flags.append("humanized_beats_original")

    # --- preservation checks (V-Slice 3) -----------------------------------
    aiify_preservation: PreservationResult | None = None
    humanize_preservation: PreservationResult | None = None
    if enforce_preservation:
        aiify_preservation = evaluate_preservation(
            original=original_text, rewrite=aiified_text
        )
        humanize_preservation = evaluate_preservation(
            original=original_text, rewrite=humanized_text
        )
        for stage_label, pres in (
            ("aiify", aiify_preservation),
            ("humanize", humanize_preservation),
        ):
            if pres.numbers_dropped:
                rejected.append(
                    f"{stage_label}_dropped_numbers:{','.join(pres.numbers_dropped[:3])}"
                )
            if pres.entities_dropped:
                rejected.append(
                    f"{stage_label}_dropped_entities:{','.join(pres.entities_dropped[:3])}"
                )
            for drift in pres.role_drift:
                rejected.append(f"{stage_label}_role_drift:{drift}")

    metrics = {
        "aiify_delta": aiify_delta,
        "humanize_delta": humanize_delta,
        "recovery_ratio": recovery_ratio,
        "n_dim_improvements": float(n_improved),
        "length_ratio_aiify_over_original": r_ai,
        "length_ratio_humanized_over_aiified": r_hum_ai,
        "length_ratio_humanized_over_original": r_hum_orig,
    }
    return GateResult(
        accepted=not rejected,
        rejected_reasons=tuple(rejected),
        suspicion_flags=tuple(flags),
        metrics=metrics,
        aiify_preservation=aiify_preservation,
        humanize_preservation=humanize_preservation,
    )
