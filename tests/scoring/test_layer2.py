"""Tests for Layer 2 normalization and failure handling."""

from __future__ import annotations

from pathlib import Path

import pytest
from arka.labeling.models import LabelResult
from arka.labeling.rubric import RubricLoader

from humanize_rl.scoring.layer2 import (
    DEFAULT_RUBRIC_PATH,
    _build_llm_config,
    _default_layer2,
    _label_result_to_layer2,
    _normalize_score,
    score_layer2_batch,
)


@pytest.fixture
def rubric_path() -> Path:
    return DEFAULT_RUBRIC_PATH


@pytest.fixture
def rubric(rubric_path: Path):
    return RubricLoader().load(rubric_path)


def _make_label_result(scores: dict[str, int]) -> LabelResult:
    return LabelResult(
        scores=scores,
        overall=3.7,
        reasoning="clear reasoning",
        rubric_hash="hash",
        rubric_version="1.0",
        judge_model="judge-model",
        judge_prompt_hash="prompt-hash",
        provider="openai",
        latency_ms=123,
    )


class TestNormalizeScore:
    def test_scale_endpoints(self) -> None:
        assert _normalize_score(1) == 0.0
        assert _normalize_score(5) == 1.0

    def test_scale_midpoint(self) -> None:
        assert _normalize_score(3) == 0.5


class TestBuildLLMConfig:
    def test_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY not set"):
            _build_llm_config()

    def test_builds_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        config = _build_llm_config(model="google/gemini-3-flash-preview")
        assert config.provider == "openai"
        assert config.model == "google/gemini-3-flash-preview"
        assert str(config.base_url) == "https://openrouter.ai/api/v1"
        assert config.timeout_seconds == 120
        assert config.max_retries == 5
        assert config.api_key.get_secret_value() == "test-key"


class TestLabelResultToLayer2:
    def test_normalizes_all_dimensions_and_overall(self, rubric) -> None:
        scores = {
            "structural_symmetry": 5,
            "specificity": 1,
            "formality_gradient": 3,
            "voice_consistency": 4,
            "rhetorical_sophistication": 2,
            "padding_density": 5,
            "personality_presence": 1,
            "copula_avoidance": 3,
        }
        result = _label_result_to_layer2(_make_label_result(scores), rubric)

        assert result.per_dim["structural_symmetry"] == 1.0
        assert result.per_dim["specificity"] == 0.0
        assert result.per_dim["formality_gradient"] == 0.5
        assert result.raw_scores == scores
        assert result.reasoning == "clear reasoning"
        assert result.judge_model == "judge-model"
        assert result.latency_ms == 123

        expected_overall = (
            1.0 * 0.15
            + 0.0 * 0.15
            + 0.5 * 0.15
            + 0.75 * 0.15
            + 0.25 * 0.10
            + 1.0 * 0.10
            + 0.0 * 0.10
            + 0.5 * 0.10
        ) / 1.0
        assert result.overall == pytest.approx(expected_overall)

    def test_missing_dimension_uses_scale_min(self, rubric) -> None:
        result = _label_result_to_layer2(
            _make_label_result({"structural_symmetry": 5}),
            rubric,
        )
        assert result.per_dim["structural_symmetry"] == 1.0
        assert result.per_dim["specificity"] == 0.0
        assert result.per_dim["padding_density"] == 0.0


class TestDefaultLayer2:
    def test_returns_neutral_default(self) -> None:
        result = _default_layer2()
        assert result.overall == 0.5
        assert result.per_dim == {}
        assert result.raw_scores == {}
        assert "neutral default" in result.reasoning.lower()


class TestScoreLayer2Batch:
    def test_uses_default_instructions_when_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        rubric,
    ) -> None:
        calls: list[tuple[str, str, object]] = []

        class FakeEngine:
            def __init__(self, llm_client: object) -> None:
                self.llm_client = llm_client

            def label(self, instruction: str, response: str, rubric: object):
                calls.append((instruction, response, rubric))
                return _make_label_result(
                    {
                        dimension.name: dimension.scale_max
                        for dimension in rubric.dimensions
                    }
                )

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setattr(
            "humanize_rl.scoring.layer2.RubricLoader.load",
            lambda self, path: rubric,
        )
        monkeypatch.setattr(
            "humanize_rl.scoring.layer2.LLMClient",
            lambda config: object(),
        )
        monkeypatch.setattr(
            "humanize_rl.scoring.layer2.LabelingEngine",
            FakeEngine,
        )

        results = score_layer2_batch(["alpha", "beta"])

        assert len(results) == 2
        assert [call[0] for call in calls] == [
            "Evaluate this text for AI writing patterns.",
            "Evaluate this text for AI writing patterns.",
        ]
        assert [call[1] for call in calls] == ["alpha", "beta"]
        assert all(result.overall == 1.0 for result in results)

    def test_falls_back_to_neutral_default_on_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        rubric,
    ) -> None:
        class FakeEngine:
            def __init__(self, llm_client: object) -> None:
                self.llm_client = llm_client
                self.calls = 0

            def label(self, instruction: str, response: str, rubric: object):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("boom")
                return _make_label_result(
                    {
                        dimension.name: dimension.scale_max
                        for dimension in rubric.dimensions
                    }
                )

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setattr(
            "humanize_rl.scoring.layer2.RubricLoader.load",
            lambda self, path: rubric,
        )
        monkeypatch.setattr(
            "humanize_rl.scoring.layer2.LLMClient",
            lambda config: object(),
        )
        monkeypatch.setattr(
            "humanize_rl.scoring.layer2.LabelingEngine",
            FakeEngine,
        )

        results = score_layer2_batch(
            texts=["ok", "bad", "ok-again"],
            instructions=["i1", "i2", "i3"],
        )

        assert [result.overall for result in results] == [1.0, 0.5, 1.0]
        output = capsys.readouterr().out
        assert "[WARN] L2 scoring failed for text 1: boom" in output
        assert "Layer 2: 1/3 failed, used neutral defaults" in output
