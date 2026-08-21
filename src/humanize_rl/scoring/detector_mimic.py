"""Frozen Pangram-style detector mimic for offline model gates.

This intentionally does not call an external detector. It mirrors the report
shape we care about for SFT/RL gates: document label, AI fraction, segment
scores, and marker evidence.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, Field

from humanize_rl.reward.checks import (
    AI_TELL_PHRASES,
    ALL_CAPS_PHRASE_RE,
    BROKEN_INFORMAL_GRAMMAR_RE,
    EMOJI_RE,
    FAKE_CASUAL_RE,
    HASHTAG_RE,
    MARKDOWN_LIST_RE,
    OPTION_MENU_RE,
    REGISTER_MISMATCH_RE,
    SENTENCE_RE,
    THANKS_PADDING_RE,
    VAGUE_SUBSTITUTION_RE,
    WRAPPER_RE,
    word_count,
)

ExpectedLabel = Literal["human", "mixed", "ai"]
PredictionShort = Literal["Human", "Mixed", "AI"]

CORPORATE_FILLER_RE = re.compile(
    r"\b(?:circle back|touch base|leverage|synergy|robust|seamless|unlock|"
    r"empower|mission-critical|at scale|operational excellence|"
    r"transformative|delve|elevate|game-changer)\b",
    re.IGNORECASE,
)
POLITE_WRAPPER_START_RE = re.compile(r"^\s*(?:certainly|of course)[,.!]?\s+", re.I)
BALANCED_TEMPLATE_RE = re.compile(
    r"\b(?:not only|whether you're|from .{1,40} to .{1,40}|"
    r"in today's (?:fast-paced|ever-changing)|"
    r"by (?:leveraging|utilizing|embracing))\b",
    re.IGNORECASE,
)
LOW_INFORMATION_RE = re.compile(
    r"\b(?:stuff|things|thing|something)\b(?:\W+\w+){0,8}"
    r"\b(?:stuff|things|thing|something)\b",
    re.IGNORECASE,
)
CONTEXTUAL_AI_TELL_PHRASES = frozenset(
    {
        "seamless",
        "unlock",
        "empower",
        "mission-critical",
    }
)

MARKER_WEIGHTS: dict[str, float] = {
    "wrapper_phrase": 0.30,
    "ai_tell_phrase": 0.28,
    "corporate_filler": 0.20,
    "balanced_template": 0.18,
    "option_menu": 0.18,
    "markdown_list": 0.12,
    "fake_casual_phrase": 0.34,
    "low_specificity_substitution": 0.22,
    "broken_informal_grammar": 0.30,
    "register_mismatch": 0.25,
    "thanks_padding": 0.12,
    "all_caps": 0.22,
    "emoji": 0.10,
    "hashtag": 0.23,
}


class DetectorMimicRow(BaseModel):
    """One frozen row in the detector-mimic gate set."""

    id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    expected_label: ExpectedLabel
    group: str = "default"
    source: str = "project_frozen_v01"
    notes: str = ""


class DetectorMimicWindow(BaseModel):
    """Segment-level detector mimic score."""

    index: int
    text: str
    fraction_ai: float
    markers: list[str]


class DetectorMimicResult(BaseModel):
    """Document-level detector mimic result."""

    id: str = ""
    group: str = ""
    expected_label: ExpectedLabel | None = None
    prediction: str
    prediction_short: PredictionShort
    fraction_ai: float
    fraction_ai_assisted: float
    fraction_human: float
    max_window_ai_probability: float
    markers: list[str]
    windows: list[DetectorMimicWindow]
    gate_error: str = ""


class DetectorMimicSummary(BaseModel):
    """Aggregate gate metrics for a detector-mimic run."""

    total_rows: int
    human_rows: int
    nonhuman_rows: int
    false_positive_rows: int
    false_negative_rows: int
    mean_fraction_ai: float
    max_fraction_ai: float
    gate_passed: bool
    false_positive_ids: list[str]
    false_negative_ids: list[str]


class DetectorMimicReport(BaseModel):
    """Full detector-mimic report."""

    source: str = ""
    summary: DetectorMimicSummary
    rows: list[DetectorMimicResult]


def _unique_markers(markers: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for marker in markers:
        if marker not in seen:
            seen.add(marker)
            out.append(marker)
    return out


def _contains_ai_tell(text: str) -> bool:
    lowered = text.lower()
    if any(
        phrase in lowered
        for phrase in AI_TELL_PHRASES
        if phrase not in CONTEXTUAL_AI_TELL_PHRASES
    ):
        return True
    contextual_hits = sum(
        1 for phrase in CONTEXTUAL_AI_TELL_PHRASES if phrase in lowered
    )
    return contextual_hits >= 3


def _markers_for_text(text: str) -> list[str]:
    markers: list[str] = []
    if WRAPPER_RE.search(text) or POLITE_WRAPPER_START_RE.search(text):
        markers.append("wrapper_phrase")
    if _contains_ai_tell(text):
        markers.append("ai_tell_phrase")
    if CORPORATE_FILLER_RE.search(text):
        markers.append("corporate_filler")
    if BALANCED_TEMPLATE_RE.search(text):
        markers.append("balanced_template")
    if OPTION_MENU_RE.search(text):
        markers.append("option_menu")
    if MARKDOWN_LIST_RE.search(text):
        markers.append("markdown_list")
    if FAKE_CASUAL_RE.search(text):
        markers.append("fake_casual_phrase")
    if VAGUE_SUBSTITUTION_RE.search(text) or LOW_INFORMATION_RE.search(text):
        markers.append("low_specificity_substitution")
    if BROKEN_INFORMAL_GRAMMAR_RE.search(text):
        markers.append("broken_informal_grammar")
    if REGISTER_MISMATCH_RE.search(text):
        markers.append("register_mismatch")
    if THANKS_PADDING_RE.search(text):
        markers.append("thanks_padding")
    if ALL_CAPS_PHRASE_RE.search(text):
        markers.append("all_caps")
    if EMOJI_RE.search(text):
        markers.append("emoji")
    if HASHTAG_RE.search(text):
        markers.append("hashtag")
    return _unique_markers(markers)


def _raw_ai_fraction(markers: list[str], text: str) -> float:
    score = sum(MARKER_WEIGHTS[marker] for marker in markers)
    words = word_count(text)
    if markers and words > 0:
        marker_density = min(len(markers) / max(words / 18.0, 1.0), 1.0)
        score += marker_density * 0.12
    if not markers and words >= 8:
        score = 0.06
    return min(score, 0.98)


def _prediction_from_fraction(fraction_ai: float) -> tuple[str, PredictionShort]:
    if fraction_ai >= 0.65:
        return "AI Detected", "AI"
    if fraction_ai >= 0.35:
        return "Mixed AI and Human", "Mixed"
    return "Human Written", "Human"


def _split_windows(text: str) -> list[str]:
    windows = [
        match.group(0).strip()
        for match in SENTENCE_RE.finditer(text.strip())
        if match.group(0).strip()
    ]
    return windows or [text.strip()]


def score_detector_mimic_text(text: str) -> DetectorMimicResult:
    """Score one text with a deterministic detector mimic."""
    markers = _markers_for_text(text)
    fraction_ai = _raw_ai_fraction(markers, text)
    prediction, prediction_short = _prediction_from_fraction(fraction_ai)

    windows: list[DetectorMimicWindow] = []
    for index, window in enumerate(_split_windows(text)):
        window_markers = _markers_for_text(window)
        windows.append(
            DetectorMimicWindow(
                index=index,
                text=window,
                fraction_ai=_raw_ai_fraction(window_markers, window),
                markers=window_markers,
            )
        )

    max_window = max((window.fraction_ai for window in windows), default=fraction_ai)
    fraction_ai_assisted = 0.0
    if 0.25 <= fraction_ai < 0.65:
        fraction_ai_assisted = min(0.60, 1.0 - abs(fraction_ai - 0.45))

    return DetectorMimicResult(
        prediction=prediction,
        prediction_short=prediction_short,
        fraction_ai=round(fraction_ai, 6),
        fraction_ai_assisted=round(fraction_ai_assisted, 6),
        fraction_human=round(max(0.0, 1.0 - fraction_ai), 6),
        max_window_ai_probability=round(max_window, 6),
        markers=markers,
        windows=windows,
    )


def _gate_error(row: DetectorMimicRow, result: DetectorMimicResult) -> str:
    if row.expected_label == "human" and result.prediction_short != "Human":
        return "false_positive"
    if row.expected_label in {"mixed", "ai"} and result.prediction_short == "Human":
        return "false_negative"
    return ""


def evaluate_detector_mimic_rows(
    rows: Iterable[DetectorMimicRow], source: str = ""
) -> DetectorMimicReport:
    """Score frozen detector rows and compute a strict pass/fail gate."""
    scored: list[DetectorMimicResult] = []
    for row in rows:
        result = score_detector_mimic_text(row.text)
        error = _gate_error(row, result)
        scored.append(
            result.model_copy(
                update={
                    "id": row.id,
                    "group": row.group,
                    "expected_label": row.expected_label,
                    "gate_error": error,
                }
            )
        )

    false_positive_ids = [
        result.id for result in scored if result.gate_error == "false_positive"
    ]
    false_negative_ids = [
        result.id for result in scored if result.gate_error == "false_negative"
    ]
    fractions = [result.fraction_ai for result in scored]
    human_rows = sum(1 for result in scored if result.expected_label == "human")
    total = len(scored)
    summary = DetectorMimicSummary(
        total_rows=total,
        human_rows=human_rows,
        nonhuman_rows=total - human_rows,
        false_positive_rows=len(false_positive_ids),
        false_negative_rows=len(false_negative_ids),
        mean_fraction_ai=round(sum(fractions) / total, 6) if total else 0.0,
        max_fraction_ai=max(fractions) if fractions else 0.0,
        gate_passed=not false_positive_ids and not false_negative_ids,
        false_positive_ids=false_positive_ids,
        false_negative_ids=false_negative_ids,
    )
    return DetectorMimicReport(source=source, summary=summary, rows=scored)
