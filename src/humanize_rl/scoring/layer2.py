"""Layer 2 LLM judge scoring — 8 dimensions via Arka LabelingEngine.

Uses the humanness_v01.yaml rubric with Gemini 3.1 Pro as judge.
Each dimension scored 1-5, normalized to 0-1 for combination with Layer 1.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from arka.config.models import LLMConfig
from arka.labeling.engine import LabelingEngine
from arka.labeling.models import LabelResult
from arka.labeling.rubric import Rubric, RubricLoader
from arka.llm.client import LLMClient

DEFAULT_RUBRIC_PATH = Path("rubrics/humanness_v01.yaml")


@dataclass(frozen=True)
class Layer2Result:
    """Result from LLM judge scoring."""

    overall: float  # 0-1 normalized
    per_dim: dict[str, float] = field(default_factory=dict)  # 0-1 normalized
    raw_scores: dict[str, int] = field(default_factory=dict)  # 1-5 raw
    reasoning: str = ""
    judge_model: str = ""
    latency_ms: int = 0


def _normalize_score(raw: int, scale_min: int = 1, scale_max: int = 5) -> float:
    """Normalize 1-5 scale to 0-1."""
    return (raw - scale_min) / (scale_max - scale_min)


def _build_llm_config(
    model: str = "google/gemini-3.1-pro-preview",
) -> LLMConfig:
    """Build LLM config for the judge."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set. Layer 2 requires an API key.")
    return LLMConfig(
        provider="openai",
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout_seconds=120,
        max_retries=5,
    )


def _label_result_to_layer2(
    result: LabelResult,
    rubric: Rubric,
) -> Layer2Result:
    """Convert Arka LabelResult to Layer2Result with normalization.

    Arka returns overall as weighted sum in [1.0, 5.0] (raw scale).
    We normalize per-dim to [0, 1] and recompute overall as weighted
    mean of normalized scores.
    """
    per_dim_normalized: dict[str, float] = {}
    for dim in rubric.dimensions:
        raw = result.scores.get(dim.name, dim.scale_min)
        per_dim_normalized[dim.name] = _normalize_score(
            raw, dim.scale_min, dim.scale_max
        )

    # Recompute overall as weighted mean of normalized scores
    total_weight = sum(rubric.overall_weights.values())
    overall_normalized = (
        sum(
            per_dim_normalized[name] * rubric.overall_weights[name]
            for name in rubric.overall_weights
        )
        / total_weight
        if total_weight > 0
        else 0.0
    )

    return Layer2Result(
        overall=overall_normalized,
        per_dim=per_dim_normalized,
        raw_scores=result.scores,
        reasoning=result.reasoning,
        judge_model=result.judge_model,
        latency_ms=result.latency_ms,
    )


def score_layer2(
    text: str,
    instruction: str = "Evaluate this text for AI writing patterns.",
    rubric_path: Path = DEFAULT_RUBRIC_PATH,
    model: str = "google/gemini-3.1-pro-preview",
) -> Layer2Result:
    """Score a single text with the Layer 2 LLM judge.

    Args:
        text: The text to score.
        instruction: Context for the judge.
        rubric_path: Path to the rubric YAML.
        model: Model to use as judge.

    Returns:
        Layer2Result with normalized scores.
    """
    rubric = RubricLoader().load(rubric_path)
    llm_config = _build_llm_config(model=model)
    llm_client = LLMClient(config=llm_config)
    engine = LabelingEngine(llm_client=llm_client)

    result = engine.label(
        instruction=instruction,
        response=text,
        rubric=rubric,
    )

    return _label_result_to_layer2(result, rubric)


def _default_layer2() -> Layer2Result:
    """Return a neutral Layer2Result for failed scoring."""
    return Layer2Result(
        overall=0.5,
        per_dim={},
        raw_scores={},
        reasoning="Layer 2 scoring failed — using neutral default.",
    )


def score_layer2_batch(
    texts: list[str],
    instructions: list[str] | None = None,
    rubric_path: Path = DEFAULT_RUBRIC_PATH,
    model: str = "google/gemini-3.1-pro-preview",
    max_workers: int = 3,
) -> list[Layer2Result]:
    """Score multiple texts with the Layer 2 LLM judge.

    Scores individually for fault tolerance — one failure doesn't
    kill the batch.

    Args:
        texts: List of texts to score.
        instructions: Optional per-text instructions.
        rubric_path: Path to the rubric YAML.
        model: Model to use as judge.
        max_workers: Concurrent API calls.

    Returns:
        List of Layer2Results (one per text).
    """
    rubric = RubricLoader().load(rubric_path)
    llm_config = _build_llm_config(model=model)
    llm_client = LLMClient(config=llm_config)
    engine = LabelingEngine(llm_client=llm_client)

    default_instruction = "Evaluate this text for AI writing patterns."
    if instructions is None:
        instructions = [default_instruction] * len(texts)

    results: list[Layer2Result] = []
    failed = 0
    for i, (inst, text) in enumerate(zip(instructions, texts, strict=True)):
        try:
            result = engine.label(
                instruction=inst,
                response=text,
                rubric=rubric,
            )
            results.append(_label_result_to_layer2(result, rubric))
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN] L2 scoring failed for text {i}: {exc}")
            results.append(_default_layer2())
            failed += 1

        if (i + 1) % 10 == 0:
            print(f"  ... scored {i + 1}/{len(texts)}")

    if failed:
        print(f"  Layer 2: {failed}/{len(texts)} failed, used neutral defaults")

    return results
