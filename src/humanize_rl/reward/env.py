"""Single-turn RL environment wrapper for humanize reward tasks.

This module is intentionally framework-light. It exposes the same core shape that
Prime Verifiers `SingleTurnEnv` uses (prompt -> completion -> scalar reward),
without requiring `verifiers` as an install-time dependency.
"""

from __future__ import annotations

import json
import re
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from humanize_rl.reward.reward import RewardResult, TrackAScorer, score_response
from humanize_rl.reward.tasks import RLTask, load_tasks
from humanize_rl.reward.verifiers_adapter import build_verifiers_rubric

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TASK_SET = "mix_v2_p5050_filtered"
TASK_SET_PATHS = {
    "v02_smoke": REPO_ROOT
    / "environments/humanize_rl_env/humanize_rl_env/humanize_tasks_v02_smoke.jsonl",
    "v03": REPO_ROOT / "data/rl/humanize_tasks_v03_filtered.jsonl",
    "mix_v2": REPO_ROOT / "data/rl/humanize_tasks_rl_mix_v2.jsonl",
    "mix_v2_p5050": REPO_ROOT / "data/rl/humanize_tasks_rl_mix_v2_p5050_filtered.jsonl",
    "mix_v2_p5050_filtered": REPO_ROOT
    / "data/rl/humanize_tasks_rl_mix_v2_p5050_filtered.jsonl",
}


@dataclass(frozen=True)
class EnvironmentObservation:
    """Rendered single-turn observation for one task."""

    task_id: str
    prompt: list[dict[str, str]]
    task: dict[str, object]


@dataclass(frozen=True)
class EnvironmentStep:
    """Result of taking one action in the stateless environment."""

    observation: EnvironmentObservation
    response: str
    reward: float
    done: bool
    info: dict[str, object]


@dataclass
class HumanizeRLEnv:
    """One-episode-per-task environment for reward debugging and adapters."""

    tasks: list[RLTask]
    track_a_scorer: TrackAScorer | None = None
    reward_mode: str = "strict"
    _index: int = 0
    _active_task: RLTask | None = None
    rollout_log: list[EnvironmentStep] = field(default_factory=list)

    def reset(self, task: RLTask | None = None) -> EnvironmentObservation:
        """Start a new one-step episode and return its observation."""
        if task is None:
            if not self.tasks:
                raise ValueError("HumanizeRLEnv requires at least one task")
            task = self.tasks[self._index % len(self.tasks)]
            self._index += 1

        self._active_task = task
        return observation_for_task(task)

    def step(self, response: str) -> EnvironmentStep:
        """Score the response and terminate the episode."""
        if self._active_task is None:
            raise ValueError("reset() must be called before step()")

        task = self._active_task
        result = score_response(
            task,
            response,
            self.track_a_scorer,
            reward_mode=self.reward_mode,  # type: ignore[arg-type]
        )
        observation = observation_for_task(task)
        step = EnvironmentStep(
            observation=observation,
            response=response,
            reward=result.reward,
            done=True,
            info=info_from_reward_result(result),
        )
        self.rollout_log.append(step)
        self._active_task = None
        return step

    def run_response_map(self, responses: dict[str, str]) -> list[EnvironmentStep]:
        """Run one logged episode for each task with a response keyed by task id."""
        steps: list[EnvironmentStep] = []
        for task in self.tasks:
            if task.id not in responses:
                continue
            self.reset(task)
            steps.append(self.step(responses[task.id]))
        return steps

    def write_log_jsonl(self, path: Path) -> None:
        """Write all rollout diagnostics as JSONL."""
        write_rollout_log(path, self.rollout_log)


@dataclass(frozen=True)
class PrimeDatasetRow:
    """Framework-neutral row that mirrors Prime Verifiers input fields."""

    prompt: list[dict[str, str]]
    info: str
    answer: str
    example_id: int


def render_prompt(task: RLTask) -> str:
    """Render a task into the exact user-facing prompt."""
    if task.input_text:
        return f"{task.instruction}\n\nSource:\n{task.input_text}"
    return task.instruction


def observation_for_task(task: RLTask) -> EnvironmentObservation:
    """Build an observation from a task."""
    prompt = [{"role": "user", "content": render_prompt(task)}]
    return EnvironmentObservation(
        task_id=task.id,
        prompt=prompt,
        task=task.model_dump(by_alias=True, exclude_none=True),
    )


def prime_dataset_row(task: RLTask, example_id: int = 0) -> PrimeDatasetRow:
    """Build the row shape consumed by Prime Verifiers `SingleTurnEnv`."""
    task_payload = task.model_dump(by_alias=True, exclude_none=True)
    prompt_text = render_prompt(task)
    return PrimeDatasetRow(
        prompt=[{"role": "user", "content": prompt_text}],
        info=json.dumps({"task_id": task.id, "task": task_payload}, ensure_ascii=False),
        answer="",
        example_id=example_id,
    )


def _parse_json_mapping(value: object) -> dict[str, object] | None:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return None


def _task_payload_from_state(state: Any) -> dict[str, object] | None:
    task_payload = _parse_json_mapping(state.get("task"))
    if task_payload is not None:
        return task_payload

    info = _parse_json_mapping(state.get("info"))
    if info is not None:
        info_task = _parse_json_mapping(info.get("task"))
        if info_task is not None:
            return info_task

    input_obj = state.get("input")
    if isinstance(input_obj, Mapping):
        input_task = _parse_json_mapping(input_obj.get("task"))
        if input_task is not None:
            return input_task
        input_info = _parse_json_mapping(input_obj.get("info"))
        if input_info is not None:
            return _parse_json_mapping(input_info.get("task"))
    return None


def _question_from_state(state: Any) -> str | None:
    input_obj = state.get("input")
    if isinstance(input_obj, Mapping):
        question = input_obj.get("question")
        if isinstance(question, str) and question.strip():
            return question
    question = state.get("question")
    if isinstance(question, str) and question.strip():
        return question
    return None


def _task_id_from_state(state: Any) -> str | None:
    input_obj = state.get("input")
    if isinstance(input_obj, Mapping):
        task_id = input_obj.get("task_id")
        if isinstance(task_id, str) and task_id:
            return task_id

    info = _parse_json_mapping(state.get("info"))
    if info is not None:
        task_id = info.get("task_id")
        if isinstance(task_id, str) and task_id:
            return task_id

    task_payload = _task_payload_from_state(state)
    if task_payload is not None:
        task_id = task_payload.get("id")
        if isinstance(task_id, str) and task_id:
            return task_id

    task_id = state.get("task_id")
    if isinstance(task_id, str) and task_id:
        return task_id
    return None


def _stable_example_id(task_id: str | None) -> int:
    if task_id is None:
        return 0
    suffix = re.search(r"(\d+)$", task_id)
    if suffix is not None:
        return int(suffix.group(1))
    return zlib.crc32(task_id.encode("utf-8")) & 0x7FFFFFFF


def ensure_example_id_in_state(state: Any) -> Any:
    """Restore example_id after hosted runners reshape rollout inputs."""
    if state.get("example_id") is not None:
        return state

    example_id = _stable_example_id(_task_id_from_state(state))
    state["example_id"] = example_id
    input_obj = state.get("input")
    if isinstance(input_obj, dict):
        input_obj["example_id"] = example_id
    return state


def ensure_prompt_in_state(state: Any) -> Any:
    """Restore prompt after hosted runners reshape rollout inputs."""
    if state.get("prompt") is not None:
        return state

    question = _question_from_state(state)
    if question is not None:
        prompt = [{"role": "user", "content": question}]
    else:
        task_payload = _task_payload_from_state(state)
        if task_payload is None:
            raise ValueError("rollout input is missing prompt, question, and task")
        prompt = [
            {
                "role": "user",
                "content": render_prompt(RLTask.model_validate(task_payload)),
            }
        ]

    state["prompt"] = prompt
    input_obj = state.get("input")
    if isinstance(input_obj, dict):
        input_obj["prompt"] = prompt
    return state


def build_prime_single_turn_env(vf: Any, dataset: Any, rubric: Any) -> Any:
    """Create a SingleTurnEnv with prompt-recovery for Prime Hosted Training."""

    class HumanizePrimeSingleTurnEnv(vf.SingleTurnEnv):  # type: ignore[misc]
        async def setup_state(self, state: Any) -> Any:
            parent_state = await super().setup_state(state)
            if parent_state is not None:
                state = parent_state
            ensure_example_id_in_state(state)
            return ensure_prompt_in_state(state)

    return HumanizePrimeSingleTurnEnv(
        dataset=dataset,
        eval_dataset=dataset,
        rubric=rubric,
        message_type="chat",
    )


def info_from_reward_result(result: RewardResult) -> dict[str, object]:
    """Convert a reward result to rollout info."""
    return {
        "raw_reward": result.raw_reward,
        "profile": result.profile,
        "components": result.components,
        "weighted_components": result.weighted_components,
        "penalties": result.penalties,
        "diagnostics": result.diagnostics,
    }


def step_to_json(step: EnvironmentStep) -> dict[str, object]:
    """Serialize an environment step for JSONL logs."""
    return {
        "task_id": step.observation.task_id,
        "prompt": step.observation.prompt,
        "task": step.observation.task,
        "response": step.response,
        "reward": step.reward,
        "done": step.done,
        "info": step.info,
    }


def write_rollout_log(path: Path, steps: list[EnvironmentStep]) -> None:
    """Write rollout steps as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(step_to_json(step), ensure_ascii=False) + "\n" for step in steps
        )
    )


def resolve_task_path(task_path: Path | None, task_set: str = DEFAULT_TASK_SET) -> Path:
    """Resolve explicit task_path or a named bundled/local task set."""
    if task_path is not None:
        return task_path
    if task_set not in TASK_SET_PATHS:
        supported = ", ".join(sorted(TASK_SET_PATHS))
        raise ValueError(f"unknown task_set: {task_set!r}; supported: {supported}")
    return TASK_SET_PATHS[task_set]


def load_env_from_jsonl(
    path: Path,
    split: str = "all",
    reward_mode: str = "strict",
) -> HumanizeRLEnv:
    """Load a native environment from task JSONL."""
    tasks = load_tasks(path)
    if split != "all":
        tasks = [task for task in tasks if task.split == split]
    return HumanizeRLEnv(tasks=tasks, reward_mode=reward_mode)


def load_prime_environment(
    task_path: Path | None = None,
    split: str = "train",
    task_set: str = DEFAULT_TASK_SET,
    reward_mode: str = "p50_50_no_penalty",
) -> Any:
    """Load a Prime Verifiers SingleTurnEnv lazily.

    Latest Verifiers docs show the simple path as:
    `Dataset.from_list([{prompt: [...], ...}])`, async reward function receiving
    `completion` plus dataset fields, `vf.Rubric(funcs=[...])`, then
    `vf.SingleTurnEnv(dataset=dataset, rubric=rubric)`.
    """
    try:
        import verifiers as vf
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError(
            "Prime Verifiers adapter requires `verifiers` and `datasets`. "
            "Install with `uv add verifiers` when ready to run Prime eval/training."
        ) from exc

    resolved_task_path = resolve_task_path(task_path, task_set)
    tasks = load_tasks(resolved_task_path)
    if split != "all":
        tasks = [task for task in tasks if task.split == split]
    rows = [
        prime_dataset_row(task, example_id=i).__dict__ for i, task in enumerate(tasks)
    ]
    dataset = Dataset.from_list(rows)

    rubric = build_verifiers_rubric(vf, reward_mode=reward_mode)
    return build_prime_single_turn_env(vf, dataset, rubric)
