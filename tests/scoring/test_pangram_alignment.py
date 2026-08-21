from __future__ import annotations

from humanize_rl.scoring.detector_mimic import DetectorMimicRow
from humanize_rl.scoring.pangram_alignment import (
    build_pangram_alignment_report,
    load_pangram_detections,
)


def test_load_pangram_detections_accepts_v3_task_payload() -> None:
    detections = load_pangram_detections(
        {
            "task_id": "task_123",
            "stage": "STAGE_SUCCESS",
            "text": "Certainly, here's a more natural version that unlocks value.",
            "prediction_short": "AI",
            "fraction_ai": 0.8,
            "fraction_ai_assisted": 0.1,
            "fraction_human": 0.1,
            "windows": [
                {
                    "text": "Certainly, here's a more natural version",
                    "label": "AI-Generated",
                    "ai_assistance_score": 0.9,
                }
            ],
        }
    )

    assert len(detections) == 1
    detection = detections[0]
    assert detection.id == "task_123"
    assert detection.prediction_short == "AI"
    assert detection.normalized_label == "ai"
    assert detection.fraction_ai == 0.8
    assert detection.fraction_nonhuman == 0.9
    assert detection.window_count == 1


def test_pangram_alignment_passes_when_external_labels_match_mimic() -> None:
    rows = [
        DetectorMimicRow(
            id="human_1",
            text="I deployed the hotfix and checked the logs. Everything is stable now.",
            expected_label="human",
        ),
        DetectorMimicRow(
            id="ai_1",
            text="Certainly, here's a more natural version that unlocks value.",
            expected_label="ai",
        ),
    ]
    pangram_export = {
        "results": [
            {
                "id": "human_1",
                "prediction_short": "Human",
                "fraction_ai": 0.04,
                "fraction_ai_assisted": 0.0,
                "fraction_human": 0.96,
            },
            {
                "id": "ai_1",
                "prediction_short": "AI",
                "fraction_ai": 0.86,
                "fraction_ai_assisted": 0.08,
                "fraction_human": 0.06,
            },
        ]
    }

    report = build_pangram_alignment_report(
        rows,
        pangram_export,
        source="fixture",
        max_mean_abs_fraction_delta=0.40,
    )

    assert report.summary.total_rows == 2
    assert report.summary.matched_rows == 2
    assert report.summary.label_disagreement_rows == 0
    assert report.summary.gate_passed
    assert report.rows[0].source_match == "id"


def test_pangram_alignment_counts_ai_assisted_fraction_as_detector_risk() -> None:
    rows = [
        DetectorMimicRow(
            id="assisted_1",
            text="Whether you're updating docs or sending notes, this version helps"
            " unlock value at scale.",
            expected_label="mixed",
        )
    ]
    pangram_export = {
        "results": [
            {
                "id": "assisted_1",
                "prediction_short": "AI-Assisted",
                "fraction_ai": 0.0,
                "fraction_ai_assisted": 0.64,
                "fraction_human": 0.36,
            }
        ]
    }

    report = build_pangram_alignment_report(
        rows,
        pangram_export,
        source="fixture",
        max_mean_abs_fraction_delta=0.40,
    )

    assert report.rows[0].pangram_label == "mixed"
    assert report.rows[0].pangram_fraction_ai == 0.0
    assert report.rows[0].pangram_fraction_ai_assisted == 0.64
    assert report.rows[0].pangram_fraction_nonhuman == 0.64
    assert report.summary.mean_abs_fraction_delta < 0.40
    assert report.summary.gate_passed is True


def test_pangram_alignment_fails_on_missing_and_disagreement() -> None:
    rows = [
        DetectorMimicRow(
            id="human_1",
            text="I deployed the hotfix and checked the logs. Everything is stable now.",
            expected_label="human",
        ),
        DetectorMimicRow(
            id="ai_1",
            text="Certainly, here's a more natural version that unlocks value.",
            expected_label="ai",
        ),
    ]
    pangram_export = {
        "results": [
            {
                "id": "ai_1",
                "prediction_short": "Human",
                "fraction_ai": 0.02,
                "fraction_ai_assisted": 0.0,
                "fraction_human": 0.98,
            }
        ]
    }

    report = build_pangram_alignment_report(rows, pangram_export, source="fixture")

    assert report.summary.gate_passed is False
    assert report.summary.missing_ids == ["human_1"]
    assert report.summary.label_disagreement_ids == ["ai_1"]
    assert "missing_coverage" in report.summary.failures
    assert "label_disagreement" in report.summary.failures
