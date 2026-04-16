"""Layer 1 deterministic scoring — 8 heuristic dimensions.

Each function takes text and returns a float in [0.0, 1.0].
  0.0 = clearly AI-generated pattern
  1.0 = clearly human-written pattern

No API keys. No LLM calls. Deterministic. <10ms per text.
"""

from __future__ import annotations

import re
import statistics

from humanize_rl.scoring.patterns import (
    BULLET_LINE_RE,
    CLOSING_PATTERNS,
    CONTRACTION_PATTERNS,
    HEDGE_PATTERNS,
    OPENER_PATTERNS,
    TRANSITION_PATTERNS,
)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences. Simple but effective for scoring."""
    # Split on sentence-ending punctuation followed by space or end
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in raw if len(s.split()) >= 2]


# ---------------------------------------------------------------------------
# Dimension 1: opener_pattern
# ---------------------------------------------------------------------------


def score_opener(text: str) -> float:
    """Detect sycophantic/chatbot openers.

    Returns 0.0 if a known AI opener is found, 1.0 otherwise.
    """
    first_line = text.strip().split("\n")[0]
    for pattern in OPENER_PATTERNS:
        if pattern.search(first_line):
            return 0.0
    return 1.0


# ---------------------------------------------------------------------------
# Dimension 2: hedging_density
# ---------------------------------------------------------------------------


def score_hedging(text: str) -> float:
    """Score based on density of hedge phrases.

    0 hedges → 1.0, scaling down with more hedges per 200 words.
    """
    total_words = len(text.split())
    if total_words == 0:
        return 0.5

    matches = sum(1 for p in HEDGE_PATTERNS if p.search(text))
    # Normalize to per-200-word density
    chunks = max(total_words / 200, 1.0)
    density = matches / chunks

    if density >= 3:
        return 0.1
    if density >= 2:
        return 0.3
    if density >= 1:
        return 0.5
    if density >= 0.5:
        return 0.8
    return 1.0


# ---------------------------------------------------------------------------
# Dimension 3: list_overuse
# ---------------------------------------------------------------------------


def score_list_overuse(text: str) -> float:
    """Score based on ratio of bullet/numbered list lines to total lines.

    High bullet ratio → AI-like → low score.
    """
    lines = [line for line in text.strip().split("\n") if line.strip()]
    total = len(lines)
    if total == 0:
        return 0.5

    bullet_lines = sum(1 for line in lines if BULLET_LINE_RE.match(line))
    ratio = bullet_lines / total

    if ratio > 0.50:
        return 0.1
    if ratio > 0.25:
        return 0.3
    if ratio > 0.10:
        return 0.5
    if ratio > 0.05:
        return 0.7
    return 1.0


# ---------------------------------------------------------------------------
# Dimension 4: sentence_variance
# ---------------------------------------------------------------------------


def score_sentence_variance(text: str) -> float:
    """Score based on coefficient of variation of sentence lengths.

    Low CV = uniform (AI-like). High CV = varied (human-like).
    #1 discriminator per rival.tips (CV 2.78 across 178 models).
    """
    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return 0.5  # insufficient data

    lengths = [len(s.split()) for s in sentences]
    mean = statistics.mean(lengths)
    if mean == 0:
        return 0.5

    stdev = statistics.stdev(lengths)
    cv = stdev / mean

    if cv < 0.15:
        return 0.1  # extremely uniform
    if cv < 0.25:
        return 0.3
    if cv < 0.35:
        return 0.5
    if cv < 0.50:
        return 0.7
    return 0.9


# ---------------------------------------------------------------------------
# Dimension 5: contractions
# ---------------------------------------------------------------------------


def score_contractions(text: str) -> float:
    """Score based on contraction rate.

    Zero contractions in informal text = strong AI tell.
    """
    total_words = len(text.split())
    if total_words == 0:
        return 0.5

    contraction_count = sum(
        len(p.findall(text)) for p in CONTRACTION_PATTERNS
    )
    rate = contraction_count / total_words

    if rate < 0.005:
        return 0.2  # suspiciously few
    if rate < 0.015:
        return 0.5
    if rate < 0.035:
        return 0.8
    return 0.9


# ---------------------------------------------------------------------------
# Dimension 6: closing_pattern
# ---------------------------------------------------------------------------


def score_closing(text: str) -> float:
    """Detect AI sign-off phrases in the last paragraph.

    Returns 0.0 if a known AI closing is found, 1.0 otherwise.
    """
    # Get last meaningful paragraph
    paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    if not paragraphs:
        return 1.0

    last_para = paragraphs[-1]
    # Also check last line within last paragraph
    last_line = last_para.strip().split("\n")[-1]
    check_text = last_para + " " + last_line

    for pattern in CLOSING_PATTERNS:
        if pattern.search(check_text):
            return 0.0
    return 1.0


# ---------------------------------------------------------------------------
# Dimension 7: em_dash_density
# ---------------------------------------------------------------------------


def score_em_dash(text: str) -> float:
    """Score based on em-dash rate per 500 words.

    Heavy em-dash use is a strong AI writing tell (rival.tips CV 1.11).
    """
    words = len(text.split())
    if words == 0:
        return 0.5

    em_dashes = text.count("\u2014") + text.count("--")
    rate_per_500 = (em_dashes / words) * 500

    if rate_per_500 > 3.0:
        return 0.1  # extremely heavy
    if rate_per_500 > 1.5:
        return 0.3
    if rate_per_500 > 0.5:
        return 0.5
    if rate_per_500 > 0.2:
        return 0.7
    return 0.9


# ---------------------------------------------------------------------------
# Dimension 8: transition_overuse
# ---------------------------------------------------------------------------


def score_transitions(text: str) -> float:
    """Score based on density of formal transition words.

    "Furthermore", "Moreover", "Additionally" etc. per 200 words.
    """
    words = len(text.split())
    if words == 0:
        return 0.5

    matches = sum(len(p.findall(text)) for p in TRANSITION_PATTERNS)
    chunks = max(words / 200, 1.0)
    density = matches / chunks

    if density >= 3:
        return 0.1
    if density >= 2:
        return 0.3
    if density >= 1:
        return 0.5
    return 0.9
