from __future__ import annotations

from pathlib import Path

from humanize_rl.scoring.detector_mimic import (
    DetectorMimicRow,
    evaluate_detector_mimic_rows,
    score_detector_mimic_text,
)


def test_detector_mimic_marks_plain_direct_text_as_human() -> None:
    result = score_detector_mimic_text(
        "I deployed the hotfix and checked the logs. Everything is stable now."
    )

    assert result.prediction_short == "Human"
    assert result.fraction_human >= 0.70
    assert result.markers == []


def test_detector_mimic_flags_template_ai_tells() -> None:
    result = score_detector_mimic_text(
        "Certainly, here's a more natural version. It is worth noting that this "
        "solution unlocks a seamless experience and empowers the team."
    )

    assert result.prediction_short in {"AI", "Mixed"}
    assert result.fraction_ai >= 0.45
    assert {"wrapper_phrase", "ai_tell_phrase"} <= set(result.markers)


def test_detector_mimic_allows_specific_human_use_of_corporate_words() -> None:
    result = score_detector_mimic_text(
        "The onboarding handoff was seamless because Nate moved the DNS notes "
        "into the setup ticket."
    )

    assert result.prediction_short == "Human"
    assert result.fraction_ai < 0.35
    assert "corporate_filler" in result.markers
    assert "ai_tell_phrase" not in result.markers


def test_detector_mimic_flags_fake_casual_reward_hacks() -> None:
    result = score_detector_mimic_text(
        "We got us the project stuff working before day ends. Thanks for looking."
    )

    assert result.prediction_short in {"AI", "Mixed"}
    assert result.fraction_ai >= 0.45
    assert {"fake_casual_phrase", "broken_informal_grammar"} <= set(result.markers)


def test_evaluate_detector_mimic_rows_summarizes_gate() -> None:
    rows = [
        DetectorMimicRow(
            id="human_1",
            text="I deployed the hotfix and checked the logs. Everything is stable now.",
            expected_label="human",
            group="plain_control",
        ),
        DetectorMimicRow(
            id="ai_1",
            text=(
                "Certainly, here's a more natural version that unlocks a "
                "seamless experience."
            ),
            expected_label="ai",
            group="template_ai",
        ),
        DetectorMimicRow(
            id="hack_1",
            text="We got us the project stuff working before day ends.",
            expected_label="ai",
            group="reward_hack",
        ),
    ]

    report = evaluate_detector_mimic_rows(rows)

    assert report.summary.total_rows == 3
    assert report.summary.false_positive_rows == 0
    assert report.summary.false_negative_rows == 0
    assert report.summary.gate_passed


def test_detector_mimic_frozen_v01_set_passes() -> None:
    rows = [
        DetectorMimicRow.model_validate_json(line)
        for line in Path("data/eval/detector_mimic_v01.jsonl").read_text().splitlines()
        if line.strip()
    ]

    report = evaluate_detector_mimic_rows(rows)

    assert report.summary.total_rows == 22
    assert report.summary.human_rows == 8
    assert report.summary.false_positive_rows == 0
    assert report.summary.false_negative_rows == 0
    assert report.summary.gate_passed
