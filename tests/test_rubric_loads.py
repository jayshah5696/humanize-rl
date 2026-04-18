"""Tests for the Layer 2 rubric configuration."""

from __future__ import annotations

from pathlib import Path

from arka.labeling.rubric import RubricLoader

RUBRIC_PATH = Path("rubrics/humanness_v01.yaml")


class TestHumannessRubric:
    def test_rubric_loads(self) -> None:
        rubric = RubricLoader().load(RUBRIC_PATH)
        assert rubric.version == "1.0"
        assert len(rubric.dimensions) == 8
        assert len(rubric.overall_weights) == 8
        assert len(rubric.few_shot) == 2

    def test_weights_match_dimensions(self) -> None:
        rubric = RubricLoader().load(RUBRIC_PATH)
        dimension_names = {dimension.name for dimension in rubric.dimensions}
        assert set(rubric.overall_weights) == dimension_names
        assert sum(rubric.overall_weights.values()) == 1.0

    def test_canary_examples_cover_pass_and_fail(self) -> None:
        rubric = RubricLoader().load(RUBRIC_PATH)
        verdicts = {example.expected_verdict for example in rubric.few_shot}
        assert verdicts == {"pass", "fail"}

    def test_few_shot_scores_reference_known_dimensions(self) -> None:
        rubric = RubricLoader().load(RUBRIC_PATH)
        dimension_names = {dimension.name for dimension in rubric.dimensions}
        for example in rubric.few_shot:
            assert set(example.scores) == dimension_names
