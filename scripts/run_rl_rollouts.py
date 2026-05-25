from __future__ import annotations

import json
from pathlib import Path

import click

from humanize_rl.reward.tasks import RLTask, load_tasks


def render_prompt(task: RLTask) -> str:
    """Render one task as a model prompt."""
    if task.input_text:
        return f"{task.instruction}\n\nSource:\n{task.input_text}"
    return task.instruction


def heuristic_sft_response(task: RLTask) -> str:
    """Small deterministic baseline that imitates a cleaner SFT-style answer."""
    required = " ".join(task.required_facts)
    if task.family == "rewrite_repair":
        return "Staging is back after the missing STRIPE_WEBHOOK_SECRET fix; we'll monitor it until 3 pm."
    if task.family == "direct_email":
        return "The invoice was paid today, and the receipt is attached."
    if task.family == "slack_chat":
        return "Please hold deploys until the cache issue is fixed."
    if task.family == "compression":
        return "Checkout errors affected 180 customers after the payment adapter timed out. The rollback finished at 10:42, and the queue should clear by 11:15."
    if task.family == "tone_shift":
        return "We're improving onboarding by Friday."
    if task.family == "technical_explain":
        return "Idempotency keys let payment retries be safe: if the first request succeeded but the response was lost, the retry returns the same result instead of charging twice."
    if task.family == "product_copy_cleanup":
        return "Help teams find the user and product signals they need without digging through reports."
    if task.family == "candidate_customer_comms":
        return "Thank you for taking the time to meet with us. We decided not to move forward, but we appreciated the conversation and the effort you put into the process."
    if task.family == "adversarial_ai_tell_removal":
        return "Please send your budget notes by Thursday."
    if task.family == "placeholder_discipline":
        return "Please send the account email, [account email], so we can investigate."
    return required or "Done."


def heuristic_base_response(task: RLTask) -> str:
    """Small deterministic baseline that preserves known pre-RL failure modes."""
    good = heuristic_sft_response(task)
    if "option_menu" in task.trap_tags or "wrapper_phrase" in task.trap_tags:
        return (
            f"Here's a version:\n\nOption 1: {good}\n\nOption 2: {good} Thanks, Sarah"
        )
    if "email_overformat" in task.trap_tags:
        return f"Subject: Quick update\n\n{good}\n\nBest,"
    if "ai_tell_phrase" in task.trap_tags:
        return f"Certainly. It is worth noting that {good} Please do not hesitate to reach out."
    if "placeholder_required" in task.trap_tags:
        return "Please send the account email so we can investigate for Sarah."
    if "over_markdown" in task.trap_tags:
        return f"- {good}\n- This is important to understand."
    return f"Here's a polished version: {good}"


def build_rollouts(tasks: list[RLTask], policy: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task in tasks:
        if policy == "heuristic_base":
            response = heuristic_base_response(task)
        elif policy == "heuristic_sft":
            response = heuristic_sft_response(task)
        else:
            raise ValueError(f"Unsupported policy: {policy}")
        rows.append(
            {
                "task_id": task.id,
                "policy": policy,
                "prompt": render_prompt(task),
                "response": response,
            }
        )
    return rows


@click.command()
@click.option(
    "--task-path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("data/rl/humanize_tasks_v01_smoke.jsonl"),
    show_default=True,
)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("outputs/reward_env/rollouts.jsonl"),
    show_default=True,
)
@click.option(
    "--policy",
    type=click.Choice(["heuristic_base", "heuristic_sft"]),
    default="heuristic_sft",
    show_default=True,
)
@click.option(
    "--split", type=click.Choice(["train", "validation", "test", "all"]), default="all"
)
def main(task_path: Path, output_path: Path, policy: str, split: str) -> None:
    """Create deterministic offline rollout fixtures for reward validation."""
    tasks = load_tasks(task_path)
    if split != "all":
        tasks = [task for task in tasks if task.split == split]
    rows = build_rollouts(tasks, policy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    click.echo(f"rows={len(rows)}")
    click.echo(f"policy={policy}")
    click.echo(f"output_path={output_path}")


if __name__ == "__main__":
    main()
