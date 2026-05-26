"""Single-turn RL environment wrapper for humanize reward tasks.

This module is intentionally framework-light. It exposes the same core shape that
Prime Verifiers `SingleTurnEnv` uses (prompt -> completion -> scalar reward),
without requiring `verifiers` as an install-time dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from humanize_rl_env.reward.reward import RewardResult, TrackAScorer, score_response
from humanize_rl_env.reward.tasks import RLTask, load_tasks
from humanize_rl_env.reward.verifiers_adapter import build_verifiers_rubric


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
        result = score_response(task, response, self.track_a_scorer)
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
    """Framework-neutral row that mirrors Prime Verifiers dataset fields.

    task is serialised as a JSON string so Prime's display code can hash it
    (set([o["task"] ...]) crashes on unhashable dicts).
    Reward functions parse it back with json.loads().
    """

    prompt: list[dict[str, str]]
    task_id: str
    task: str  # JSON string — parse with json.loads() in reward functions
    info: str


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


def prime_dataset_row(task: RLTask) -> PrimeDatasetRow:
    """Build the row shape consumed by Prime Verifiers `SingleTurnEnv`."""
    task_payload = task.model_dump(by_alias=True, exclude_none=True)
    return PrimeDatasetRow(
        prompt=[{"role": "user", "content": render_prompt(task)}],
        task_id=task.id,
        task=json.dumps(task_payload, ensure_ascii=False),  # string — hashable
        info=json.dumps({"task_id": task.id, "task": task_payload}, ensure_ascii=False),
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


def load_env_from_jsonl(path: Path, split: str = "all") -> HumanizeRLEnv:
    """Load a native environment from task JSONL."""
    tasks = load_tasks(path)
    if split != "all":
        tasks = [task for task in tasks if task.split == split]
    return HumanizeRLEnv(tasks=tasks)


def load_prime_environment(task_path: Path, split: str = "train") -> Any:
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

    tasks = load_tasks(task_path)
    if split != "all":
        tasks = [task for task in tasks if task.split == split]
    rows = [prime_dataset_row(task).__dict__ for task in tasks]
    dataset = Dataset.from_list(rows)

    rubric = build_verifiers_rubric(vf)
    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric)
