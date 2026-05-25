from __future__ import annotations

from humanize_rl.reward.reward import score_response
from humanize_rl.reward.tasks import RLTask
from scripts.evaluate_reward_env import summarize_scores
from scripts.run_rl_rollouts import build_rollouts


def _tasks() -> list[RLTask]:
    return [
        RLTask.model_validate(
            {
                "id": "rl_v01_000001",
                "family": "rewrite_repair",
                "domain": "slack",
                "mode": "rewrite",
                "register": "casual",
                "instruction": "Clean up this Slack update.",
                "input_text": "Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix.",
                "constraints": {
                    "max_words": 20,
                    "preserve_numbers": True,
                    "preserve_entities": True,
                },
                "reward_profile": "rewrite_faithful_concise",
                "trap_tags": ["option_menu", "wrapper_phrase"],
                "split": "test",
                "required_facts": [
                    "Staging recovered",
                    "3 pm",
                    "STRIPE_WEBHOOK_SECRET",
                ],
            }
        ),
        RLTask.model_validate(
            {
                "id": "rl_v01_000002",
                "family": "direct_email",
                "domain": "email",
                "mode": "direct_generation",
                "register": "warm_professional",
                "instruction": "Write a short email saying the invoice was paid today and receipt is attached.",
                "constraints": {
                    "max_words": 55,
                    "no_subject_line": True,
                    "no_signoff": True,
                },
                "reward_profile": "direct_workplace_message",
                "trap_tags": ["email_overformat"],
                "split": "test",
            }
        ),
    ]


def test_heuristic_sft_rollouts_beat_heuristic_base_rollouts() -> None:
    tasks = _tasks()
    base = build_rollouts(tasks, "heuristic_base")
    sft = build_rollouts(tasks, "heuristic_sft")
    task_by_id = {task.id: task for task in tasks}

    base_scores = [
        score_response(task_by_id[str(row["task_id"])], str(row["response"])).reward
        for row in base
    ]
    sft_scores = [
        score_response(task_by_id[str(row["task_id"])], str(row["response"])).reward
        for row in sft
    ]

    assert sum(sft_scores) / len(sft_scores) > sum(base_scores) / len(base_scores)


def test_summary_counts_option_menu_and_wrapper_penalties() -> None:
    task = _tasks()[0]
    result = score_response(task, "Here's a version:\n\nOption 1: Sarah fixed staging.")
    summary = summarize_scores(
        [
            {
                "reward": result.reward,
                "profile": result.profile,
                "penalties": result.penalties,
            }
        ]
    )

    assert summary["penalty_counts"]["option_menu"] == 1
    assert summary["penalty_counts"]["wrapper_phrase"] == 1
