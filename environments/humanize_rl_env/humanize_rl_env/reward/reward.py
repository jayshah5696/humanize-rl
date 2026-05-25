"""Composite reward function for humanize RL tasks."""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from humanize_rl_env.reward.checks import (
    CheckReport,
    run_deterministic_checks,
    sentence_count,
    word_count,
)
from humanize_rl_env.reward.profiles import RewardProfile, profile_for_task
from humanize_rl_env.reward.tasks import RLTask, load_tasks
from humanize_rl_env.scoring.aggregator import score_text

TRACK_A_STYLE_CAP = 0.20
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
        """Returns [[P(AI), P(human)]] — last element is P(human), used as style boost."""
        ai_probs = self._scorer.predict_binary(rows)
        return [[float(p), 1.0 - float(p)] for p in ai_probs]


def load_ridge_scorer(path: Path | None = None) -> "TrackAScorer | None":
    """Load the best available ridge pkl and wrap in RidgeScorerAdapter.

    Searches DEFAULT_RIDGE_PATHS in order; returns None if none found.
    Falls back gracefully so the reward scorer runs without it.
    """
    import pickle

    default_paths = [
        Path(__file__).resolve().parents[1] / "ridge_state.pkl",  # bundled state dict
        Path(__file__).resolve().parents[1] / "ridge.pkl",        # bundled legacy
        Path("models/track_a_10k/ridge.pkl"),
        Path("models/distilled/baseline_ridge.pkl"),
    ]
    candidates = [path] if path else default_paths
    for candidate in candidates:
        if candidate and candidate.exists():
            raw = _load_state_pkl(candidate)
            if raw is not None:
                return RidgeScorerAdapter(raw)
    return None


class _BundledRidgeScorer:
    """Minimal sklearn-only scorer reconstructed from a state-dict pkl.

    Avoids any dependency on the original humanize_rl package class path.
    Exposes predict_binary(texts) -> np.ndarray[P(AI)].
    """

    def __init__(self, state: dict) -> None:
        self.vectorizer = state["vectorizer"]
        self.classifier = state["classifier"]
        self.regressors = state.get("regressors", [])

    def predict_binary(self, texts: list[str]) -> Any:
        feats = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(feats)[:, 1]  # P(AI)


def _load_state_pkl(candidate: Path) -> Any:
    """Load either a state-dict pkl or a legacy RidgeScorer pkl."""
    import pickle
    import sys
    import types

    with candidate.open("rb") as fh:
        data = fh.read()

    import io
    # First try: state dict (keys vectorizer/classifier/regressors)
    try:
        obj = pickle.load(io.BytesIO(data))
        if isinstance(obj, dict) and "vectorizer" in obj:
            return _BundledRidgeScorer(obj)
        # Legacy full RidgeScorer object — try wrapping directly
        if hasattr(obj, "predict_binary"):
            return obj
    except ModuleNotFoundError:
        pass

    # Second try: inject fake module so pickle resolves the old class path
    for old_path in [
        "humanize_rl.scoring.distilled.baselines",
        "humanize_rl.scoring.distilled.base",
    ]:
        if old_path not in sys.modules:
            fake = types.ModuleType(old_path)
            fake.BaseDistilledScorer = object  # type: ignore[attr-defined]

            class _Shim:
                def __setstate__(self, s): self.__dict__.update(s)
                def predict_binary(self, texts):
                    feats = self.vectorizer.transform(texts)
                    return self.classifier.predict_proba(feats)[:, 1]

            fake.RidgeScorer = _Shim  # type: ignore[attr-defined]
            sys.modules[old_path] = fake

    try:
        return pickle.load(io.BytesIO(data))
    except Exception:
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


def _score_from_failures(report: CheckReport, names: set[str]) -> float:
    failures = sum(
        1
        for diagnostic in report.diagnostics
        if diagnostic.name in names and not diagnostic.passed
    )
    return max(0.0, 1.0 - (failures / max(len(names), 1)))


def _length_score(task: RLTask, response: str) -> float:
    words = word_count(response)
    constraints = task.constraints
    score = 1.0

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
    layer1_score = score_text(response).overall
    if track_a_score is None:
        return layer1_score
    capped = min(max(track_a_score, 0.0), 1.0) * TRACK_A_STYLE_CAP
    return clip((layer1_score * (1.0 - TRACK_A_STYLE_CAP)) + capped, 0.0, 1.0)


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
            "refusal",
            "sentence_count",
            "too_long",
            "too_short",
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
        },
    )


def _format_score(report: CheckReport) -> float:
    return _score_from_failures(
        report,
        {
            "subject_line",
            "signoff",
            "wrong_format_markdown",
            "option_menu",
            "wrapper_phrase",
        },
    )


def _placeholder_score(report: CheckReport) -> float:
    return _score_from_failures(
        report, {"placeholder_disallowed", "placeholder_required"}
    )


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
) -> RewardResult:
    """Score one task/response pair with component breakdown and penalties."""
    report = run_deterministic_checks(task, response)
    track_a_score = _track_a_human_probability(track_a_scorer, response)
    components = _base_components(task, response, report, track_a_score)
    profile = profile_for_task(task)
    weighted = _weighted_components(profile, components)
    penalties = {
        diagnostic.name: diagnostic.penalty
        for diagnostic in report.diagnostics
        if diagnostic.penalty != 0.0
    }
    raw_reward = sum(weighted.values()) + sum(penalties.values())
    reward = clip(raw_reward)

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
        profile=profile.name,
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
        result = score_response(tasks[task_id], response, track_a_scorer)
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
