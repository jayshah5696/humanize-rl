"""Composite reward function for humanize RL tasks."""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from humanize_rl_env.reward.checks import (
    CheckReport,
    run_deterministic_checks,
    sentence_count,
    word_bounds,
    word_count,
)
from humanize_rl_env.reward.profiles import RewardProfile
from humanize_rl_env.reward.tasks import RLTask, load_tasks
from humanize_rl_env.scoring.aggregator import score_text

TRACK_A_STYLE_CAP = 0.20

# 50/50 split: ridge rubric dims vs deterministic checks
RIDGE_WEIGHT = 0.50
DETERMINISTIC_WEIGHT = 0.50
DETERMINISTIC_REPETITION_FAIL_CAP = 0.20
DETERMINISTIC_LENGTH_FAIL_CAP = 0.20
DETERMINISTIC_SEMANTIC_FAIL_CAP = 0.0
DETERMINISTIC_SURFACE_FAIL_CAP = 0.40
DETERMINISTIC_FORMAT_FAIL_CAP = 0.40
TARGET_LENGTH_RATIO_WEIGHT = 2.0

HARD_FORMAT_FAILURES = {
    "subject_line",
    "signoff",
    "salutation",
    "instruction_leak",
    "placeholder_disallowed",
    "placeholder_required",
    "wrong_format_markdown",
    "wrong_format_bullets",
    "wrong_format_heading",
    "missing_contraction",
    "paragraph_count",
    "sentence_window",
}

SEMANTIC_FAILURES = {
    "invented_number",
    "invented_temporal_detail",
    "invented_detail",
    "low_specificity_substitution",
    "low_source_overlap",
    "missing_number",
    "missing_entity",
    "missing_required_fact",
    "missing_must_include_phrase",
    "forbidden_phrase",
    "forbidden_fact",
    "placeholder_required",
    "unsupported_negation",
}

RewardMode = Literal["strict", "scalar_softened", "p50_50_no_penalty"]

DEFAULT_PENALTY_CAP = 1.0
DEFAULT_RIDGE_WEIGHT_SOFT = 0.45
DEFAULT_DET_WEIGHT_SOFT = 0.35
DEFAULT_RISK_WEIGHT_SOFT = 0.20

# Rubric dim names from track_a scorer (order matches predict_rubric columns)
RIDGE_RUBRIC_DIMS = [
    "structural_symmetry",
    "specificity",
    "formality_gradient",
    "voice_consistency",
    "rhetorical_sophistication",
    "padding_density",
    "personality_presence",
    "copula_avoidance",
]
CORPORATE_FILLER_RE = re.compile(
    r"\b(?:circle back|touch base|leverage|synergy|robust|seamless|unlock|empower|"
    r"mission-critical|at scale|operational excellence|transformative)\b",
    re.IGNORECASE,
)


class TrackAScorer(Protocol):
    """Optional Track A scorer protocol."""

    def predict_proba(self, rows: list[str]) -> Any: ...


class RidgeScorerAdapter:
    """Wraps RidgeScorer (predict_binary) into the TrackAScorer protocol.

    predict_proba returns [[P(human), P(AI)]] per row so that
    _track_a_human_probability picks up the last element as P(AI)
    and 1 - P(AI) = P(human) is used to push style score up.
    We invert: TrackAScorer convention is P(human), so we return
    1 - predict_binary to give the human probability.
    """

    def __init__(self, ridge_scorer: Any) -> None:
        self._scorer = ridge_scorer

    def predict_proba(self, rows: list[str]) -> list[list[float]]:
        """Returns [[P(AI), P(human)]] — last element is P(human)."""
        ai_probs = self._scorer.predict_binary(rows)
        return [[float(p), 1.0 - float(p)] for p in ai_probs]

    def predict_rubric(self, rows: list[str]) -> list[list[float]]:
        """Returns 8 rubric dim scores per row via the ridge regression heads."""
        raw = self._scorer.predict_rubric(rows)  # shape (N, 8), already clipped [0,1]
        return [list(map(float, row)) for row in raw]


class _StateRidgeScorer:
    """Minimal sklearn scorer reconstructed from a portable state-dict pkl."""

    def __init__(self, state: dict) -> None:
        self.vectorizer = state["vectorizer"]
        self.classifier = state["classifier"]
        self.regressors = state.get("regressors", [])

    def predict_binary(self, texts: list[str]) -> Any:
        feats = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(feats)[:, 1]

    def predict_rubric(self, texts: list[str]) -> Any:
        import numpy as np

        feats = self.vectorizer.transform(texts)
        preds = np.zeros((len(texts), len(self.regressors)), dtype="float32")
        for i, reg in enumerate(self.regressors):
            preds[:, i] = reg.predict(feats)
        return np.clip(preds, 0.0, 1.0)


def _load_ridge_pickle(candidate: Path) -> Any | None:
    try:
        with candidate.open("rb") as fh:
            obj = pickle.load(fh)
    except (ImportError, ModuleNotFoundError, AttributeError):
        return None
    if isinstance(obj, dict) and "vectorizer" in obj:
        return _StateRidgeScorer(obj)
    if hasattr(obj, "predict_binary"):
        return obj
    return None


def load_ridge_scorer(path: Path | None = None) -> TrackAScorer | None:
    """Load the best available ridge pkl and wrap in RidgeScorerAdapter.

    Searches DEFAULT_RIDGE_PATHS in order; returns None if none found.
    Falls back gracefully so the reward scorer runs without it.
    """
    package_root = Path(__file__).resolve().parents[1]
    default_paths = [
        package_root / "ridge_state.pkl",
        Path("environments/humanize_rl_env/humanize_rl_env/ridge_state.pkl"),
        Path("models/track_a_10k/ridge.pkl"),
        Path("models/distilled/baseline_ridge.pkl"),
    ]
    candidates = [path] if path else default_paths
    for candidate in candidates:
        if candidate and candidate.exists():
            raw = _load_ridge_pickle(candidate)
            if raw is None:
                continue
            return RidgeScorerAdapter(raw)
    return None


@dataclass(frozen=True)
class RewardResult:
    """Scalar reward plus inspectable component diagnostics."""

    reward: float
    raw_reward: float
    components: dict[str, float]
    weighted_components: dict[str, float]
    penalties: dict[str, float]
    diagnostics: list[dict[str, object]] = field(default_factory=list)
    profile: str = ""


def clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def risk_compliance(
    penalty_sum: float, penalty_cap: float = DEFAULT_PENALTY_CAP
) -> float:
    """Map a negative penalty total into a [0, 1] compliance diagnostic."""
    if penalty_cap <= 0:
        raise ValueError("penalty_cap must be > 0")
    return clip(1.0 + penalty_sum / penalty_cap, 0.0, 1.0)


def _score_from_failures(report: CheckReport, names: set[str]) -> float:
    failures = sum(
        1
        for diagnostic in report.diagnostics
        if diagnostic.name in names and not diagnostic.passed
    )
    return max(0.0, 1.0 - (failures / max(len(names), 1)))


def _has_failure(report: CheckReport, names: set[str]) -> bool:
    return any(
        diagnostic.name in names and not diagnostic.passed
        for diagnostic in report.diagnostics
    )


def _length_score(task: RLTask, response: str) -> float:
    words = word_count(response)
    constraints = task.constraints
    score = 1.0

    if constraints.target_words is not None:
        lower_words, upper_words = word_bounds(task)
        if lower_words is not None and words < lower_words:
            under_ratio = (lower_words - words) / lower_words
            score -= min(1.0, under_ratio * TARGET_LENGTH_RATIO_WEIGHT)
        if upper_words is not None and words > upper_words:
            over_ratio = (words - upper_words) / upper_words
            score -= min(1.0, over_ratio * TARGET_LENGTH_RATIO_WEIGHT)
    if constraints.max_words is not None and words > constraints.max_words:
        over_ratio = (words - constraints.max_words) / constraints.max_words
        score -= min(1.0, over_ratio)
    if constraints.min_words is not None and words < constraints.min_words:
        under_ratio = (constraints.min_words - words) / constraints.min_words
        score -= min(1.0, under_ratio)
    if constraints.exact_sentences is not None:
        distance = abs(sentence_count(response) - constraints.exact_sentences)
        score -= min(1.0, distance * 0.35)

    return clip(score, 0.0, 1.0)


def _style_score(response: str, track_a_score: float | None) -> float:
    """Layer 1 heuristic score, optionally blended with ridge P(human) at 20% cap."""
    layer1_score = score_text(response).overall
    if track_a_score is None:
        return layer1_score
    capped = min(max(track_a_score, 0.0), 1.0) * TRACK_A_STYLE_CAP
    return clip((layer1_score * (1.0 - TRACK_A_STYLE_CAP)) + capped, 0.0, 1.0)


def _ridge_rubric_score(
    track_a_scorer: TrackAScorer | None, response: str
) -> float | None:
    """Mean of 8 rubric dim predictions from the ridge regression heads.

    Returns None when no scorer is loaded (graceful degradation).
    """
    if track_a_scorer is None or not hasattr(track_a_scorer, "predict_rubric"):
        return None
    dims = track_a_scorer.predict_rubric([response])[0]  # list of 8 floats
    return sum(dims) / len(dims)


def _deterministic_score(report: CheckReport, task: RLTask, response: str) -> float:
    """Equal-weight mean of all deterministic constraint-satisfaction components.

    Covers: faithfulness, task_following, length, format, placeholder, clarity,
    repetition, task-specific semantic suitability, and surface naturalness.
    These are the checks the model must pass regardless of writing style.
    """
    repetition = _repetition_score(report)
    length = _length_score(task, response)
    semantic_faithfulness = _semantic_faithfulness_score(report)
    recommendation_suitability = _recommendation_suitability_score(report)
    surface_naturalness = _surface_naturalness_score(report)
    hard_format = _hard_format_score(report)
    length_failed = _has_failure(report, {"too_long", "too_short", "sentence_count"})
    components = [
        _faithfulness_score(report),
        _task_following_score(report),
        length,
        _format_score(report),
        _placeholder_score(report),
        _clarity_score(response),
        repetition,
        semantic_faithfulness,
        recommendation_suitability,
        surface_naturalness,
        hard_format,
    ]
    score = sum(components) / len(components)
    if semantic_faithfulness == 0.0:
        return min(score, DETERMINISTIC_SEMANTIC_FAIL_CAP)
    if recommendation_suitability == 0.0:
        return min(score, DETERMINISTIC_SEMANTIC_FAIL_CAP)
    if repetition == 0.0:
        return min(score, DETERMINISTIC_REPETITION_FAIL_CAP)
    if length == 0.0 or length_failed:
        return min(score, DETERMINISTIC_LENGTH_FAIL_CAP)
    if surface_naturalness < 1.0:
        return min(score, DETERMINISTIC_SURFACE_FAIL_CAP)
    if hard_format < 1.0:
        return min(score, DETERMINISTIC_FORMAT_FAIL_CAP)
    return score


def _track_a_human_probability(
    track_a_scorer: TrackAScorer | None, response: str
) -> float | None:
    if track_a_scorer is None:
        return None
    probabilities = track_a_scorer.predict_proba([response])
    first = probabilities[0]
    if isinstance(first, list | tuple):
        return float(first[-1])
    if hasattr(first, "tolist"):
        values = first.tolist()
        if isinstance(values, list):
            return float(values[-1])
    return float(first)


def _task_following_score(report: CheckReport) -> float:
    return _score_from_failures(
        report,
        {
            "option_menu",
            "wrapper_phrase",
            "instruction_leak",
            "refusal",
            "repetition",
            "sentence_count",
            "too_long",
            "too_short",
            "missing_must_include_phrase",
            "forbidden_phrase",
            "forbidden_opener",
            "em_dash",
            "paragraph_count",
            "sentence_window",
            "missing_contraction",
            "unsuitable_recommendation",
            "emoji",
            "all_caps",
            "hashtag",
            "salutation",
            "fake_casual_phrase",
            "low_specificity_substitution",
            "broken_informal_grammar",
            "register_mismatch",
            "thanks_padding",
            "placeholder_required",
        },
    )


def _faithfulness_score(report: CheckReport) -> float:
    return _score_from_failures(
        report,
        {
            "invented_detail",
            "missing_number",
            "missing_entity",
            "missing_required_fact",
            "forbidden_fact",
            "missing_must_include_phrase",
            "low_specificity_substitution",
            "unsuitable_recommendation",
        },
    )


def _format_score(report: CheckReport) -> float:
    return _score_from_failures(
        report,
        {
            "subject_line",
            "signoff",
            "wrong_format_markdown",
            "wrong_format_bullets",
            "wrong_format_heading",
            "option_menu",
            "wrapper_phrase",
            "em_dash",
            "paragraph_count",
            "emoji",
            "all_caps",
            "hashtag",
            "salutation",
        },
    )


def _placeholder_score(report: CheckReport) -> float:
    return _score_from_failures(
        report, {"placeholder_disallowed", "placeholder_required"}
    )


def _repetition_score(report: CheckReport) -> float:
    diagnostic = report.by_name().get("repetition")
    return 1.0 if diagnostic is None or diagnostic.passed else 0.0


def _recommendation_suitability_score(report: CheckReport) -> float:
    diagnostic = report.by_name().get("unsuitable_recommendation")
    return 1.0 if diagnostic is None or diagnostic.passed else 0.0


def _semantic_faithfulness_score(report: CheckReport) -> float:
    failed = [
        diagnostic
        for diagnostic in report.diagnostics
        if diagnostic.name in SEMANTIC_FAILURES and not diagnostic.passed
    ]
    return 0.0 if failed else 1.0


def _surface_naturalness_score(report: CheckReport) -> float:
    return _score_from_failures(
        report,
        {
            "emoji",
            "all_caps",
            "hashtag",
            "em_dash",
            "ai_tell_phrase",
            "fake_casual_phrase",
            "broken_informal_grammar",
            "register_mismatch",
            "thanks_padding",
        },
    )


def _hard_format_score(report: CheckReport) -> float:
    return _score_from_failures(report, HARD_FORMAT_FAILURES)


def _clarity_score(response: str) -> float:
    words = word_count(response)
    if words == 0:
        return 0.0
    sentences = max(sentence_count(response), 1)
    avg_sentence_len = words / sentences
    if avg_sentence_len <= 24:
        return 1.0
    if avg_sentence_len <= 35:
        return 0.7
    return 0.4


def _corporate_filler_score(response: str) -> float:
    matches = len(CORPORATE_FILLER_RE.findall(response))
    return max(0.0, 1.0 - matches * 0.25)


def _base_components(
    task: RLTask, response: str, report: CheckReport, track_a_score: float | None
) -> dict[str, float]:
    style = _style_score(response, track_a_score)
    faithfulness = _faithfulness_score(report)
    task_following = _task_following_score(report)
    length = _length_score(task, response)
    format_score = _format_score(report)
    placeholder = _placeholder_score(report)
    clarity = _clarity_score(response)
    repetition = _repetition_score(report)
    semantic_faithfulness = _semantic_faithfulness_score(report)
    recommendation_suitability = _recommendation_suitability_score(report)
    surface_naturalness = _surface_naturalness_score(report)
    hard_format = _hard_format_score(report)
    no_corporate_filler = _corporate_filler_score(response)

    return {
        "style": style,
        "naturalness": style,
        "tone_appropriateness": min(style, no_corporate_filler),
        "faithfulness": faithfulness,
        "fact_preservation": faithfulness,
        "correctness_adherence": min(faithfulness, task_following),
        "task_following": task_following,
        "length": length,
        "concision": length,
        "format": format_score,
        "structure_restraint": format_score,
        "placeholder": placeholder,
        "clarity": clarity,
        "repetition": repetition,
        "semantic_faithfulness": semantic_faithfulness,
        "recommendation_suitability": recommendation_suitability,
        "surface_naturalness": surface_naturalness,
        "hard_format": hard_format,
        "no_corporate_filler": no_corporate_filler,
    }


def _weighted_components(
    profile: RewardProfile, components: dict[str, float]
) -> dict[str, float]:
    total_weight = profile.total_weight
    if total_weight <= 0:
        return {}
    return {
        name: components.get(name, 0.0) * (weight / total_weight)
        for name, weight in profile.weights.items()
    }


def score_response(
    task: RLTask,
    response: str,
    track_a_scorer: TrackAScorer | None = None,
    reward_mode: RewardMode = "strict",
) -> RewardResult:
    """Score one task/response pair.

    strict: 0.50 × ridge_rubric + 0.50 × deterministic + penalties
    p50_50_no_penalty: 0.50 × ridge_rubric + 0.50 × deterministic

    When no ridge scorer is loaded, falls back to:
      Final reward = deterministic + penalties
    (i.e. ridge_weight collapses to 0 and deterministic fills 100%).
    p50_50_no_penalty fails loudly instead because the intended blend is impossible.

    All individual components are preserved in RewardResult for diagnostics.
    """
    if reward_mode not in ("strict", "scalar_softened", "p50_50_no_penalty"):
        raise ValueError(f"unknown reward_mode: {reward_mode!r}")

    report = run_deterministic_checks(task, response)
    track_a_score = _track_a_human_probability(track_a_scorer, response)
    components = _base_components(task, response, report, track_a_score)

    # -- ridge rubric (50%) --
    ridge_rubric = _ridge_rubric_score(track_a_scorer, response)
    ridge_dims: dict[str, float] = {}
    if track_a_scorer is not None and hasattr(track_a_scorer, "predict_rubric"):
        raw_dims = track_a_scorer.predict_rubric([response])[0]
        ridge_dims = dict(zip(RIDGE_RUBRIC_DIMS, raw_dims, strict=False))

    if reward_mode == "p50_50_no_penalty" and ridge_rubric is None:
        raise RuntimeError("p50_50_no_penalty requires an available ridge scorer")

    # -- deterministic (50%) --
    det_score = _deterministic_score(report, task, response)

    # -- combine --
    if ridge_rubric is not None:
        raw_reward_base = RIDGE_WEIGHT * ridge_rubric + DETERMINISTIC_WEIGHT * det_score
        profile_name = "50_50_ridge_deterministic"
    else:
        raw_reward_base = det_score
        profile_name = "deterministic_only"

    penalties = {
        diagnostic.name: diagnostic.penalty
        for diagnostic in report.diagnostics
        if diagnostic.penalty != 0.0
    }

    if reward_mode == "p50_50_no_penalty":
        raw_reward = raw_reward_base
        reward = clip(raw_reward)
        profile_name = "p50_50_no_penalty"
    elif reward_mode == "scalar_softened":
        penalty_sum = sum(penalties.values())
        ridge_raw = ridge_rubric if ridge_rubric is not None else 0.0
        raw_reward = (
            DEFAULT_RIDGE_WEIGHT_SOFT * ridge_raw
            + DEFAULT_DET_WEIGHT_SOFT * det_score
            + DEFAULT_RISK_WEIGHT_SOFT * risk_compliance(penalty_sum)
        )
        reward = clip(raw_reward, 0.0, 1.0)
        profile_name = "scalar_softened"
    else:
        raw_reward = raw_reward_base + sum(penalties.values())
        reward = clip(raw_reward)

    # weighted_components reflects the actual contribution to raw_reward_base
    weighted = {
        "ridge_rubric": (RIDGE_WEIGHT * ridge_rubric)
        if ridge_rubric is not None
        else 0.0,
        "deterministic": (DETERMINISTIC_WEIGHT if ridge_rubric is not None else 1.0)
        * det_score,
        **{f"ridge_{k}": v for k, v in ridge_dims.items()},
    }

    return RewardResult(
        reward=reward,
        raw_reward=raw_reward,
        components=components,
        weighted_components=weighted,
        penalties=penalties,
        diagnostics=[
            {
                "name": diagnostic.name,
                "passed": diagnostic.passed,
                "penalty": diagnostic.penalty,
                "message": diagnostic.message,
                "matches": diagnostic.matches,
            }
            for diagnostic in report.diagnostics
        ],
        profile=profile_name,
    )


def load_track_a_scorer(path: Path | None) -> TrackAScorer | None:
    """Load an optional pickle Track A scorer."""
    if path is None:
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def score_jsonl(
    task_path: Path,
    response_path: Path,
    output_path: Path,
    track_a_scorer: TrackAScorer | None = None,
    reward_mode: RewardMode = "strict",
) -> list[dict[str, object]]:
    """Score JSONL responses with rows shaped as {task_id, response}."""
    tasks = {task.id: task for task in load_tasks(task_path)}
    results: list[dict[str, object]] = []
    for line_number, line in enumerate(response_path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        row = json.loads(stripped)
        task_id = str(row.get("task_id", ""))
        if task_id not in tasks:
            raise ValueError(
                f"Unknown task_id at {response_path}:{line_number}: {task_id}"
            )
        response = str(row.get("response", ""))
        result = score_response(
            tasks[task_id], response, track_a_scorer, reward_mode=reward_mode
        )
        results.append(
            {
                "task_id": task_id,
                "reward": result.reward,
                "raw_reward": result.raw_reward,
                "profile": result.profile,
                "components": result.components,
                "weighted_components": result.weighted_components,
                "penalties": result.penalties,
                "diagnostics": result.diagnostics,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results)
    )
    return results
