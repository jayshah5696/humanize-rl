"""Offline alignment checks between Pangram exports and the detector mimic."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from humanize_rl.scoring.detector_mimic import (
    DetectorMimicRow,
    score_detector_mimic_text,
)

NormalizedDetectorLabel = Literal["human", "mixed", "ai"]
SourceMatch = Literal["id", "text"]


class PangramDetection(BaseModel):
    """Normalized Pangram-style detector output for one document."""

    id: str = ""
    text: str = ""
    prediction_short: str = ""
    normalized_label: NormalizedDetectorLabel
    fraction_ai: float
    fraction_ai_assisted: float | None = None
    fraction_human: float | None = None
    fraction_nonhuman: float
    window_count: int = 0


class PangramAlignmentRow(BaseModel):
    """One row comparing external Pangram output against the local mimic."""

    id: str
    expected_label: str
    source_match: SourceMatch
    mimic_prediction_short: str
    mimic_label: NormalizedDetectorLabel
    mimic_fraction_ai: float
    pangram_prediction_short: str
    pangram_label: NormalizedDetectorLabel
    pangram_fraction_ai: float
    pangram_fraction_ai_assisted: float | None = None
    pangram_fraction_nonhuman: float
    fraction_ai_abs_delta: float
    label_agreement: bool
    gate_error: str = ""


class PangramAlignmentSummary(BaseModel):
    """Aggregate alignment metrics for a Pangram export comparison."""

    total_rows: int
    matched_rows: int
    missing_rows: int
    coverage: float
    label_agreement_rows: int
    label_disagreement_rows: int
    mean_abs_fraction_delta: float
    max_abs_fraction_delta: float
    gate_passed: bool
    missing_ids: list[str]
    label_disagreement_ids: list[str]
    failures: list[str]


class PangramAlignmentReport(BaseModel):
    """Full offline Pangram alignment report."""

    source: str = ""
    min_coverage: float
    max_label_disagreements: int
    max_mean_abs_fraction_delta: float
    summary: PangramAlignmentSummary
    rows: list[PangramAlignmentRow]


def _get_path(mapping: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _first(mapping: Mapping[str, Any], paths: Iterable[tuple[str, ...]]) -> Any:
    for path in paths:
        value = _get_path(mapping, path)
        if value is not None:
            return value
    return None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_fraction(value: Any) -> float | None:
    if value is None:
        return None
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        return None
    if fraction > 1.0 and fraction <= 100.0:
        fraction = fraction / 100.0
    return min(max(fraction, 0.0), 1.0)


def _label_from_prediction(
    prediction: str, fraction_ai: float | None
) -> NormalizedDetectorLabel | None:
    lowered = prediction.strip().lower()
    if lowered:
        if "mixed" in lowered or "assist" in lowered or (
            "ai" in lowered and "human" in lowered
        ):
            return "mixed"
        if "human" in lowered and "ai" not in lowered:
            return "human"
        if "ai" in lowered or "generated" in lowered:
            return "ai"
    if fraction_ai is None:
        return None
    if fraction_ai >= 0.65:
        return "ai"
    if fraction_ai >= 0.35:
        return "mixed"
    return "human"


def _merge_nested_result(mapping: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    views: list[Mapping[str, Any]] = [mapping]
    for key in ("result", "output", "analysis", "detection", "response"):
        nested = mapping.get(key)
        if isinstance(nested, Mapping):
            views.append({**mapping, **nested})
    return views


def _has_detection_signal(mapping: Mapping[str, Any]) -> bool:
    return any(
        _first(mapping, paths) is not None
        for paths in (
            (("prediction_short",), ("prediction",), ("label",), ("verdict",)),
            (
                ("fraction_ai",),
                ("ai_fraction",),
                ("score_ai",),
                ("ai_score",),
                ("probability_ai",),
                ("percent_ai",),
            ),
        )
    )


def _candidate_mappings(data: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(data, list):
        for item in data:
            yield from _candidate_mappings(item)
        return
    if not isinstance(data, Mapping):
        return

    for view in _merge_nested_result(data):
        if _has_detection_signal(view):
            yield view

    for key in (
        "results",
        "detections",
        "documents",
        "items",
        "data",
        "outputs",
        "responses",
        "rows",
    ):
        value = data.get(key)
        if value is not None:
            yield from _candidate_mappings(value)


def _detection_from_mapping(mapping: Mapping[str, Any]) -> PangramDetection | None:
    fraction_ai = _as_fraction(
        _first(
            mapping,
            (
                ("fraction_ai",),
                ("ai_fraction",),
                ("score_ai",),
                ("ai_score",),
                ("probability_ai",),
                ("percent_ai",),
            ),
        )
    )
    prediction_short = _as_text(
        _first(
            mapping,
            (
                ("prediction_short",),
                ("prediction",),
                ("label",),
                ("verdict",),
                ("class",),
            ),
        )
    )
    normalized_label = _label_from_prediction(prediction_short, fraction_ai)
    if normalized_label is None or fraction_ai is None:
        return None

    fraction_ai_assisted = _as_fraction(
        _first(
            mapping,
            (
                ("fraction_ai_assisted",),
                ("ai_assisted_fraction",),
                ("fraction_assisted",),
            ),
        )
    )
    windows = mapping.get("windows")
    return PangramDetection(
        id=_as_text(
            _first(
                mapping,
                (
                    ("id",),
                    ("task_id",),
                    ("customer_id",),
                    ("external_id",),
                    ("item_id",),
                    ("request_id",),
                    ("metadata", "id"),
                    ("metadata", "row_id"),
                ),
            )
        ),
        text=_as_text(
            _first(
                mapping,
                (
                    ("text",),
                    ("input_text",),
                    ("document",),
                    ("content",),
                    ("input", "text"),
                    ("request", "text"),
                    ("metadata", "text"),
                ),
            )
        ),
        prediction_short=prediction_short,
        normalized_label=normalized_label,
        fraction_ai=round(fraction_ai, 6),
        fraction_ai_assisted=fraction_ai_assisted,
        fraction_human=_as_fraction(
            _first(mapping, (("fraction_human",), ("human_fraction",)))
        ),
        fraction_nonhuman=round(min(1.0, fraction_ai + (fraction_ai_assisted or 0.0)), 6),
        window_count=len(windows) if isinstance(windows, list) else 0,
    )


def load_pangram_detections(data: Any) -> list[PangramDetection]:
    """Normalize a Pangram JSON payload into document detections."""
    detections: list[PangramDetection] = []
    seen: set[tuple[str, str, str, float]] = set()
    for mapping in _candidate_mappings(data):
        detection = _detection_from_mapping(mapping)
        if detection is None:
            continue
        key = (
            detection.id,
            detection.text,
            detection.prediction_short,
            detection.fraction_ai,
        )
        if key in seen:
            continue
        seen.add(key)
        detections.append(detection)
    return detections


def load_pangram_export_path(path: Path) -> Any:
    """Read Pangram JSON or JSONL export data."""
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line]
    return json.loads(path.read_text())


def _mimic_label(prediction_short: str) -> NormalizedDetectorLabel:
    if prediction_short == "Human":
        return "human"
    if prediction_short == "Mixed":
        return "mixed"
    return "ai"


def _is_nonhuman(label: NormalizedDetectorLabel) -> bool:
    return label in {"mixed", "ai"}


def _labels_agree(
    mimic_label: NormalizedDetectorLabel, pangram_label: NormalizedDetectorLabel
) -> bool:
    if mimic_label == pangram_label:
        return True
    return _is_nonhuman(mimic_label) and _is_nonhuman(pangram_label)


def _build_detection_indexes(
    detections: Iterable[PangramDetection],
) -> tuple[dict[str, PangramDetection], dict[str, PangramDetection]]:
    by_id: dict[str, PangramDetection] = {}
    by_text: dict[str, PangramDetection] = {}
    for detection in detections:
        if detection.id and detection.id not in by_id:
            by_id[detection.id] = detection
        if detection.text and detection.text not in by_text:
            by_text[detection.text] = detection
    return by_id, by_text


def build_pangram_alignment_report(
    rows: Iterable[DetectorMimicRow],
    pangram_export: Any,
    *,
    source: str = "",
    min_coverage: float = 1.0,
    max_label_disagreements: int = 0,
    max_mean_abs_fraction_delta: float = 0.30,
) -> PangramAlignmentReport:
    """Compare a Pangram export to the local detector mimic for the same rows."""
    row_list = list(rows)
    detections = load_pangram_detections(pangram_export)
    by_id, by_text = _build_detection_indexes(detections)

    alignment_rows: list[PangramAlignmentRow] = []
    missing_ids: list[str] = []
    for row in row_list:
        source_match: SourceMatch = "id"
        detection = by_id.get(row.id)
        if detection is None:
            detection = by_text.get(row.text)
            source_match = "text"
        if detection is None:
            missing_ids.append(row.id)
            continue

        mimic = score_detector_mimic_text(row.text)
        mimic_label = _mimic_label(mimic.prediction_short)
        label_agreement = _labels_agree(mimic_label, detection.normalized_label)
        abs_delta = abs(mimic.fraction_ai - detection.fraction_nonhuman)
        alignment_rows.append(
            PangramAlignmentRow(
                id=row.id,
                expected_label=row.expected_label,
                source_match=source_match,
                mimic_prediction_short=mimic.prediction_short,
                mimic_label=mimic_label,
                mimic_fraction_ai=mimic.fraction_ai,
                pangram_prediction_short=detection.prediction_short,
                pangram_label=detection.normalized_label,
                pangram_fraction_ai=detection.fraction_ai,
                pangram_fraction_ai_assisted=detection.fraction_ai_assisted,
                pangram_fraction_nonhuman=detection.fraction_nonhuman,
                fraction_ai_abs_delta=round(abs_delta, 6),
                label_agreement=label_agreement,
                gate_error="" if label_agreement else "label_disagreement",
            )
        )

    label_disagreement_ids = [
        row.id for row in alignment_rows if row.gate_error == "label_disagreement"
    ]
    deltas = [row.fraction_ai_abs_delta for row in alignment_rows]
    total = len(row_list)
    matched = len(alignment_rows)
    coverage = matched / total if total else 0.0
    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
    failures: list[str] = []
    if coverage < min_coverage:
        failures.append("missing_coverage")
    if len(label_disagreement_ids) > max_label_disagreements:
        failures.append("label_disagreement")
    if mean_delta > max_mean_abs_fraction_delta:
        failures.append("fraction_delta")

    summary = PangramAlignmentSummary(
        total_rows=total,
        matched_rows=matched,
        missing_rows=len(missing_ids),
        coverage=round(coverage, 6),
        label_agreement_rows=matched - len(label_disagreement_ids),
        label_disagreement_rows=len(label_disagreement_ids),
        mean_abs_fraction_delta=round(mean_delta, 6),
        max_abs_fraction_delta=round(max(deltas), 6) if deltas else 0.0,
        gate_passed=not failures,
        missing_ids=missing_ids,
        label_disagreement_ids=label_disagreement_ids,
        failures=failures,
    )
    return PangramAlignmentReport(
        source=source,
        min_coverage=min_coverage,
        max_label_disagreements=max_label_disagreements,
        max_mean_abs_fraction_delta=max_mean_abs_fraction_delta,
        summary=summary,
        rows=alignment_rows,
    )
