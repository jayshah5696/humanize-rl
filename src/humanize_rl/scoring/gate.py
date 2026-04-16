"""Gate logic — decide whether Layer 2 is needed.

Saves 30-50% of LLM judge calls by skipping obvious cases.
"""

from __future__ import annotations


def needs_layer2(layer1_score: float, context: str = "source_scoring") -> bool:
    """Determine if the expensive Layer 2 LLM judge is needed.

    Args:
        layer1_score: Layer 1 overall score (0-1).
        context: Scoring context affects thresholds.

    Returns:
        True if Layer 2 should be run.
    """
    if context == "source_scoring":
        # Checking if human-written text
        if layer1_score > 0.85:
            return False  # clearly human, skip
        if layer1_score < 0.30:
            return False  # clearly AI, skip
        return True  # ambiguous, need judge

    if context == "aiified_scoring":
        # Checking if AI-ification worked
        if layer1_score < 0.25:
            return False  # obviously AI, confirmed
        return True

    if context == "humanized_scoring":
        # Always run Layer 2 on final output
        return True

    return True  # default: run Layer 2
