from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import click

DEFAULT_FAILURE_PATH = Path("runs/prime_eval_smoke/llama32_3b_failure_set_env0314.jsonl")
DEFAULT_OUTPUT_PATH = Path(
    "data/processed/sft/reference_targets/llama32_3b_failure_refs_env0314.jsonl"
)
DEFAULT_TASK_PATHS = (
    Path("data/rl/humanize_tasks_rl_mix_v2_p5050_filtered.jsonl"),
    Path("data/rl/humanize_tasks_v03_filtered.jsonl"),
    Path("environments/humanize_rl_env/humanize_rl_env/humanize_tasks_v02_smoke.jsonl"),
)
DEFAULT_MODEL = "google/gemini-3.1-pro-preview"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    )


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def existing_task_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    task_ids: set[str] = set()
    for row in load_jsonl(path):
        metadata = row.get("metadata")
        if isinstance(metadata, dict):
            task_id = metadata.get("task_id")
            if task_id:
                task_ids.add(str(task_id))
    return task_ids


def existing_failure_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    for row in load_jsonl(path):
        metadata = row.get("metadata")
        if isinstance(metadata, dict):
            task_id = str(metadata.get("task_id") or "")
            comparison = str(metadata.get("failure_comparison") or "")
            if task_id:
                keys.add((task_id, comparison))
    return keys


def render_task_prompt(task: dict[str, Any]) -> str:
    instruction = str(task.get("instruction") or "").strip()
    input_text = str(task.get("input_text") or "").strip()
    if input_text:
        return f"{instruction}\n\nSource:\n{input_text}"
    return instruction


def load_task_index(task_paths: list[Path] | tuple[Path, ...]) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for path in task_paths:
        for row in load_jsonl(path):
            task_id = str(row.get("id") or "")
            if task_id:
                tasks[task_id] = row
    return tasks


def build_reference_messages(
    task: dict[str, Any], failure: dict[str, Any]
) -> list[dict[str, str]]:
    task_prompt = render_task_prompt(task)
    constraints = task.get("constraints") or {}
    required_facts = task.get("required_facts") or []
    forbidden_facts = task.get("forbidden_facts") or []
    forbidden_phrases = task.get("forbidden_phrases") or []
    reasons = failure.get("reasons") or []
    phrase_hits = failure.get("phrase_hits") or []

    user = f"""Create one supervised fine-tuning target for this humanize-RL task.

Return only the final answer. Do not explain options. Do not include a subject line, signoff, wrapper, markdown, emoji, or all-caps emphasis unless the task explicitly requires it.

Write plain natural prose. Prefer short concrete words. Avoid corporate filler, fake warmth, tutorial padding, and these phrases when possible: hope you're doing well, touch base, So, I was thinking, operational game, vital, crucial, leverage, enhance, navigate the complexities.

Preserve every required fact. Do not add forbidden facts. Respect the length and format constraints. If the source is vague, keep the answer equally vague.

Task id: {task.get("id")}
Family: {task.get("family")}
Mode: {task.get("mode")}
Register: {task.get("register")}
Constraints: {json.dumps(constraints, ensure_ascii=False, sort_keys=True)}
Required facts: {json.dumps(required_facts, ensure_ascii=False)}
Forbidden facts: {json.dumps(forbidden_facts, ensure_ascii=False)}
Forbidden phrases: {json.dumps(forbidden_phrases, ensure_ascii=False)}

Failure reasons to avoid: {json.dumps(reasons, ensure_ascii=False)}
Humanizer phrase hits to avoid: {json.dumps(phrase_hits, ensure_ascii=False)}
Bad base sample preview: {failure.get("base_completion_preview", "")}
Bad RL sample preview: {failure.get("rl_completion_preview", "")}

Task prompt:
{task_prompt}
"""
    return [
        {
            "role": "system",
            "content": (
                "You write concise, natural human prose for supervised fine-tuning. "
                "You obey task constraints exactly and return only the target answer."
            ),
        },
        {"role": "user", "content": user},
    ]


def call_openrouter(
    *,
    messages: list[dict[str, str]],
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/jayshah5696/humanize-rl",
            "X-Title": "humanize-rl",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter request failed: HTTP {exc.code}: {body}") from exc

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected OpenRouter response shape: {data}") from exc
    return str(content).strip()


def build_sft_row(
    task: dict[str, Any],
    failure: dict[str, Any],
    *,
    response: str | None,
    model: str = DEFAULT_MODEL,
    dry_run: bool = False,
) -> dict[str, Any]:
    return {
        "instruction": render_task_prompt(task),
        "response": response or "",
        "metadata": {
            "task_id": task.get("id"),
            "family": task.get("family"),
            "mode": task.get("mode"),
            "domain": task.get("domain"),
            "register": task.get("register"),
            "reward_profile": task.get("reward_profile"),
            "failure_comparison": failure.get("comparison"),
            "failure_reasons": failure.get("reasons") or [],
            "phrase_hits": failure.get("phrase_hits") or [],
            "generator_model": model,
            "dry_run": dry_run,
            "source": "prime_failure_reference_generation",
        },
    }


@click.command()
@click.option(
    "--failure-path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=DEFAULT_FAILURE_PATH,
    show_default=True,
)
@click.option(
    "--task-path",
    "task_paths",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    multiple=True,
    default=DEFAULT_TASK_PATHS,
    show_default=True,
)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=DEFAULT_OUTPUT_PATH,
    show_default=True,
)
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
@click.option("--api-key-var", default="OPENROUTER_API_KEY", show_default=True)
@click.option("--temperature", type=float, default=0.2, show_default=True)
@click.option("--max-tokens", type=int, default=4096, show_default=True)
@click.option("--limit", type=int, default=None)
@click.option("--sleep-seconds", type=float, default=0.0, show_default=True)
@click.option("--dry-run", is_flag=True, help="Write prompts without calling OpenRouter.")
@click.option("--resume", is_flag=True, help="Skip task ids already present in output.")
@click.option("--overwrite", is_flag=True, help="Remove existing output before writing.")
def main(
    failure_path: Path,
    task_paths: tuple[Path, ...],
    output_path: Path,
    model: str,
    api_key_var: str,
    temperature: float,
    max_tokens: int,
    limit: int | None,
    sleep_seconds: float,
    dry_run: bool,
    resume: bool,
    overwrite: bool,
) -> None:
    """Generate SFT target responses for eval failure rows using a Google model."""
    if not model.startswith("google/"):
        raise click.ClickException("Only Google OpenRouter models are allowed here.")

    api_key = os.environ.get(api_key_var)
    if not dry_run and not api_key:
        raise click.ClickException(f"Missing {api_key_var}")
    if resume and overwrite:
        raise click.ClickException("--resume and --overwrite cannot be used together.")
    if output_path.exists() and overwrite:
        output_path.unlink()
    if output_path.exists() and not resume:
        output_path.unlink()

    failures = load_jsonl(failure_path)
    if limit is not None:
        failures = failures[:limit]
    tasks = load_task_index(list(task_paths))

    missing: list[str] = []
    skipped_existing = 0
    written_rows = 0
    existing_keys = existing_failure_keys(output_path) if resume else set()
    for i, failure in enumerate(failures, start=1):
        task_id = str(failure.get("task_id") or "")
        failure_key = (task_id, str(failure.get("comparison") or ""))
        if failure_key in existing_keys:
            skipped_existing += 1
            click.echo(f"skip_existing {i}/{len(failures)} task_id={task_id}")
            continue
        task = tasks.get(task_id)
        if task is None:
            missing.append(task_id)
            continue
        messages = build_reference_messages(task, failure)
        response = None
        if dry_run:
            row = build_sft_row(task, failure, response=None, model=model, dry_run=True)
            row["metadata"]["messages"] = messages
        else:
            response = call_openrouter(
                messages=messages,
                model=model,
                api_key=str(api_key),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            row = build_sft_row(task, failure, response=response, model=model)
            if sleep_seconds and i < len(failures):
                time.sleep(sleep_seconds)
        append_jsonl_row(output_path, row)
        written_rows += 1
        click.echo(f"wrote {i}/{len(failures)} task_id={task_id}")

    click.echo(f"wrote_rows={written_rows}")
    click.echo(f"skipped_existing={skipped_existing}")
    click.echo(f"missing_tasks={len(missing)}")
    if missing:
        click.echo("missing_task_ids=" + ",".join(missing))
    click.echo(f"output_path={output_path}")


if __name__ == "__main__":
    main()
