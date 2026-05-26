"""Reference-rollout filter for v03 tasks (slice 5).

Generate N reference completions per task with the *primary strong* model,
optionally one rollout each from a *weak* model (for B2 gap) and a
*second-strong* model on an audit subset (for C2 cross-model rho), score
every rollout with the humanize-rl reward env, then drop:

  * BROKEN  — primary mean reward < --reward-low.
  * TRIVIAL — primary mean reward > --reward-high AND std < --reward-std-min.

Parallel: uses ThreadPoolExecutor across tasks (one worker per task; rollouts
for a given task still run sequentially to preserve per-task error grouping
and to keep rate-limit pressure bounded).

See plan §7.1 and §8.

Usage:

    uv run scripts/eval/reference_rollout_filter.py \\
        --input data/rl/humanize_tasks_v03_judged_kept.jsonl \\
        --output data/rl/humanize_tasks_v03_filtered.jsonl \\
        --rollouts-out data/rl/v03_ref_rollouts.jsonl \\
        --model openai/gpt-5.4-mini \\
        --rollouts-per-task 3 \\
        --weak-model google/gemini-3.1-flash-lite-preview \\
        --second-strong-model google/gemini-3-flash-preview \\
        --audit-subset 200
"""

from __future__ import annotations

import json
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click

from humanize_rl.reward.reward import load_ridge_scorer, score_response
from humanize_rl.reward.tasks import RLTask, load_tasks, write_tasks


def _render_prompt(task: RLTask) -> str:
    if task.input_text:
        return f"{task.instruction}\n\nSource:\n{task.input_text}"
    return task.instruction


def _call_model(
    client, model: str, prompt: str, temperature: float, max_tokens: int
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


def _rollout_record(
    task_id: str, rollout: int, model: str, completion: str, scorer, task: RLTask
) -> dict:
    result = score_response(task, completion, scorer)
    return {
        "task_id": task_id,
        "rollout": rollout,
        "model": model,
        "reward": result.reward,
        "components": result.components,
        "penalties": result.penalties,
        "completion": completion,
    }


@click.command(context_settings={"show_default": True})
@click.option(
    "--input", "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
)
@click.option(
    "--output", type=click.Path(dir_okay=False, path_type=Path), required=True,
    help="Filtered tasks JSONL (broken+trivial dropped).",
)
@click.option(
    "--rollouts-out", type=click.Path(dir_okay=False, path_type=Path),
    required=True, help="Per-rollout reward + diagnostics JSONL.",
)
@click.option("--model", default="openai/gpt-5.4-mini",
              help="Primary strong model (B1, filtering, std).")
@click.option("--rollouts-per-task", type=int, default=3)
@click.option(
    "--weak-model", default=None,
    help="Optional weak model — one rollout per task for B2 strong-weak gap.",
)
@click.option(
    "--second-strong-model", default=None,
    help="Optional second strong model — one rollout per audit-subset task for C2.",
)
@click.option(
    "--audit-subset", type=int, default=200,
    help="Number of tasks (front of file) to also rollout with --second-strong-model.",
)
@click.option("--temperature", type=float, default=0.7)
@click.option("--max-tokens", type=int, default=1200)
@click.option(
    "--reward-low", type=float, default=0.4,
    help="Drop tasks with primary mean reward < this (BROKEN).",
)
@click.option(
    "--reward-high", type=float, default=0.9, help="Trivial threshold.",
)
@click.option(
    "--reward-std-min", type=float, default=0.05,
    help="Trivial AND std < this → drop.",
)
@click.option(
    "--limit", type=int, default=None,
    help="Only process first N tasks (smoke).",
)
@click.option("--max-workers", type=int, default=16,
              help="ThreadPoolExecutor size; one task per worker.")
@click.option("--api-key-var", default="OPENROUTER_API_KEY")
@click.option("--api-base-url", default="https://openrouter.ai/api/v1")
@click.option("--sleep", type=float, default=0.0,
              help="Sleep between rollouts within a single task (rate-limit cushion).")
def main(
    input_path: Path, output: Path, rollouts_out: Path, model: str,
    rollouts_per_task: int, weak_model: str | None,
    second_strong_model: str | None, audit_subset: int,
    temperature: float, max_tokens: int,
    reward_low: float, reward_high: float, reward_std_min: float,
    limit: int | None, max_workers: int,
    api_key_var: str, api_base_url: str, sleep: float,
) -> None:
    """Generate reference rollouts and drop broken / trivial tasks."""
    from openai import OpenAI

    api_key = os.environ.get(api_key_var)
    if not api_key:
        raise click.ClickException(f"{api_key_var} not set.")

    client = OpenAI(api_key=api_key, base_url=api_base_url)
    scorer = load_ridge_scorer()
    if scorer is None:
        click.secho(
            "  ! no ridge scorer found; reward collapses to deterministic-only.",
            fg="yellow",
        )
    else:
        click.secho("  ✓ ridge scorer loaded", fg="green")

    tasks = load_tasks(input_path)
    if limit:
        tasks = tasks[:limit]
    click.echo(
        f"loaded {len(tasks)} tasks; primary={model} x{rollouts_per_task}"
        + (f"; weak={weak_model} x1" if weak_model else "")
        + (f"; second_strong={second_strong_model} x1 (first {audit_subset})"
           if second_strong_model else "")
    )

    audit_ids = {t.id for t in tasks[:audit_subset]} if second_strong_model else set()

    rollouts_out.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    rf = rollouts_out.open("w")
    counters = {"primary_done": 0, "weak_done": 0, "second_done": 0,
                "errors": 0, "broken": 0, "trivial": 0, "kept": 0}

    def _process(task: RLTask) -> tuple[RLTask | None, str]:
        """Return (kept_task_or_None, status_str)."""
        prompt = _render_prompt(task)
        rewards: list[float] = []
        records: list[dict] = []
        # Primary
        for r in range(rollouts_per_task):
            try:
                completion = _call_model(
                    client, model, prompt, temperature, max_tokens
                )
            except Exception as exc:  # noqa: BLE001
                return None, f"error:primary:{type(exc).__name__}:{str(exc)[:120]}"
            rec = _rollout_record(task.id, r, model, completion, scorer, task)
            rewards.append(rec["reward"])
            records.append(rec)
            if sleep:
                time.sleep(sleep)
        # Weak (one rollout)
        if weak_model:
            try:
                completion = _call_model(
                    client, weak_model, prompt, temperature, max_tokens
                )
                records.append(_rollout_record(
                    task.id, 0, weak_model, completion, scorer, task
                ))
            except Exception as exc:  # noqa: BLE001
                # Non-fatal; just skip the weak rollout.
                records.append({
                    "task_id": task.id, "rollout": 0, "model": weak_model,
                    "error": f"{type(exc).__name__}:{str(exc)[:120]}",
                })
        # Second strong (audit subset only)
        if second_strong_model and task.id in audit_ids:
            try:
                completion = _call_model(
                    client, second_strong_model, prompt, temperature, max_tokens
                )
                records.append(_rollout_record(
                    task.id, 0, second_strong_model, completion, scorer, task
                ))
            except Exception as exc:  # noqa: BLE001
                records.append({
                    "task_id": task.id, "rollout": 0,
                    "model": second_strong_model,
                    "error": f"{type(exc).__name__}:{str(exc)[:120]}",
                })

        # Persist rollouts atomically.
        with write_lock:
            for rec in records:
                rf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rf.flush()
            counters["primary_done"] += sum(1 for r in records if r.get("model") == model)
            if weak_model:
                counters["weak_done"] += sum(
                    1 for r in records if r.get("model") == weak_model and "error" not in r
                )
            if second_strong_model:
                counters["second_done"] += sum(
                    1 for r in records
                    if r.get("model") == second_strong_model and "error" not in r
                )

        if not rewards:
            return None, "error:no_rewards"
        mean_r = statistics.mean(rewards)
        std_r = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        if mean_r < reward_low:
            return None, f"broken:mean={mean_r:.3f}"
        if mean_r > reward_high and std_r < reward_std_min:
            return None, f"trivial:mean={mean_r:.3f},std={std_r:.3f}"
        return task, f"kept:mean={mean_r:.3f},std={std_r:.3f}"

    kept: list[RLTask] = []
    total = len(tasks)
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_process, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futures), start=1):
                task_obj, status = fut.result()
                if task_obj is not None:
                    kept.append(task_obj)
                    counters["kept"] += 1
                elif status.startswith("broken"):
                    counters["broken"] += 1
                elif status.startswith("trivial"):
                    counters["trivial"] += 1
                else:
                    counters["errors"] += 1
                    if counters["errors"] <= 5:
                        click.secho(f"  ! {status}", fg="red")
                if i % 25 == 0 or i == total:
                    elapsed = time.time() - t0
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (total - i) / rate if rate > 0 else 0
                    click.echo(
                        f"  {i}/{total} kept={counters['kept']} "
                        f"broken={counters['broken']} trivial={counters['trivial']} "
                        f"err={counters['errors']} "
                        f"primary_rollouts={counters['primary_done']} "
                        f"weak={counters['weak_done']} second={counters['second_done']} "
                        f"| {rate:.2f} task/s ETA {eta/60:.1f}m"
                    )
    finally:
        rf.close()

    write_tasks(output, kept)
    click.secho(
        f"wrote {len(kept)} / {len(tasks)} kept → {output} "
        f"(broken={counters['broken']} trivial={counters['trivial']} "
        f"err={counters['errors']})",
        fg="green",
    )


if __name__ == "__main__":
    main()
