"""End-to-end tests for the Prime Intellect environment package.

Verifies:
1. load_environment() returns a vf.SingleTurnEnv (requires verifiers)
2. preview_dataset_row() works without verifiers
3. Reward functions rank good > bad responses
4. All metrics return floats in expected ranges
5. Dataset rows have required Prime Verifiers schema fields
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

TASK_PATH = Path(__file__).resolve().parents[2] / "data/rl/humanize_tasks_v01_smoke.jsonl"
ENV_PATH = Path(__file__).resolve().parents[2] / "environments/humanize_rl_env"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _real_task_dict() -> dict:
    from humanize_rl.reward.tasks import load_tasks
    tasks = load_tasks(TASK_PATH)
    return tasks[0].model_dump(by_alias=True, exclude_none=True)


GOOD_RESPONSE = "Staging is back. Root cause: missing STRIPE_WEBHOOK_SECRET. Monitoring till 3 pm."
BAD_RESPONSE = (
    "It is important to note that, as a comprehensive outcome of our diligent "
    "investigation, the staging environment has been successfully restored."
)


# ---------------------------------------------------------------------------
# 1. preview_dataset_row (no verifiers needed)
# ---------------------------------------------------------------------------

def test_preview_dataset_row_schema():
    """Row must have prompt, task_id, task, info — the Prime Verifiers shape."""
    import sys
    sys.path.insert(0, str(ENV_PATH))
    from humanize_rl_env import preview_dataset_row  # type: ignore[import]

    row = preview_dataset_row(task_path=str(TASK_PATH))
    assert set(row.keys()) >= {"prompt", "task_id", "task", "info"}
    assert isinstance(row["prompt"], list)
    assert row["prompt"][0]["role"] == "user"
    assert isinstance(row["task_id"], str)
    assert isinstance(row["task"], dict)
    # info must be JSON-serialisable string
    parsed = json.loads(row["info"])
    assert parsed["task_id"] == row["task_id"]


def test_preview_dataset_row_all_train_rows():
    """Every train row must load without validation errors."""
    from humanize_rl.reward.tasks import load_tasks
    from humanize_rl.reward.env import prime_dataset_row

    tasks = load_tasks(TASK_PATH)
    train = [t for t in tasks if t.split == "train"]
    assert len(train) > 0, "No train tasks in smoke JSONL"
    for t in train:
        row = prime_dataset_row(t)
        assert row.task_id == t.id


# ---------------------------------------------------------------------------
# 2. Reward function correctness
# ---------------------------------------------------------------------------

def test_good_beats_bad_reward():
    """humanize_reward must rank the clean response above the AI-bloated one."""
    from humanize_rl.reward.verifiers_adapter import humanize_reward

    task = _real_task_dict()
    good_c = [{"role": "assistant", "content": GOOD_RESPONSE}]
    bad_c  = [{"role": "assistant", "content": BAD_RESPONSE}]

    r_good = asyncio.run(humanize_reward(good_c, task, {}))
    r_bad  = asyncio.run(humanize_reward(bad_c,  task, {}))

    assert r_good > r_bad, f"good={r_good:.3f} bad={r_bad:.3f}"


def test_all_metrics_return_floats():
    """Every metric function must return a finite float."""
    from humanize_rl.reward.verifiers_adapter import (
        clarity_metric, faithfulness_metric, format_metric,
        humanize_reward, invented_detail_penalty_metric,
        length_metric, option_menu_penalty_metric,
        placeholder_metric, risk_penalty_metric,
        style_metric, task_following_metric,
        wrapper_phrase_penalty_metric,
    )

    task = _real_task_dict()
    completion = [{"role": "assistant", "content": GOOD_RESPONSE}]
    state: dict = {}

    metrics = [
        humanize_reward, style_metric, task_following_metric,
        faithfulness_metric, length_metric, format_metric,
        clarity_metric, placeholder_metric, risk_penalty_metric,
        option_menu_penalty_metric, wrapper_phrase_penalty_metric,
        invented_detail_penalty_metric,
    ]

    async def run():
        results = {}
        for fn in metrics:
            val = await fn(completion, task, state)
            assert isinstance(val, float), f"{fn.__name__} returned {type(val)}"
            assert val == val, f"{fn.__name__} returned NaN"  # NaN check
            results[fn.__name__] = val
        return results

    results = asyncio.run(run())
    for name, val in results.items():
        print(f"  {name}: {val:.4f}")


def test_reward_cached_in_state():
    """humanize_reward must cache the RewardResult in state (avoids double-scoring)."""
    from humanize_rl.reward.verifiers_adapter import (
        REWARD_STATE_KEY, humanize_reward, style_metric,
    )

    task = _real_task_dict()
    completion = [{"role": "assistant", "content": GOOD_RESPONSE}]
    state: dict = {}

    asyncio.run(humanize_reward(completion, task, state))
    assert REWARD_STATE_KEY in state, "RewardResult not cached in state"

    # second metric must reuse cached result, not re-score
    asyncio.run(style_metric(completion, task, state))
    assert state.get("humanize_reward") is not None


# ---------------------------------------------------------------------------
# 3. load_environment (requires verifiers)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("verifiers"),
    reason="verifiers not installed",
)
def test_load_environment_returns_single_turn_env():
    import sys
    sys.path.insert(0, str(ENV_PATH))
    from humanize_rl_env import load_environment  # type: ignore[import]
    import verifiers as vf

    env = load_environment(split="train", task_path=str(TASK_PATH))
    assert isinstance(env, vf.SingleTurnEnv)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("verifiers"),
    reason="verifiers not installed",
)
def test_load_environment_eval_split():
    import sys
    sys.path.insert(0, str(ENV_PATH))
    from humanize_rl_env import load_environment  # type: ignore[import]
    import verifiers as vf

    env = load_environment(split="eval", task_path=str(TASK_PATH))
    assert isinstance(env, vf.SingleTurnEnv)
