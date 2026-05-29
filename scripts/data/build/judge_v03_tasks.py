"""Post-assemble judge for v03 RL tasks.

Replaces the per-mode `labeling_engine` arka stage so we can use a cheap model
(Flash-Lite) for judging while authors run on more expensive models. See plan
§4.3 and slice 4 decisions.

Reads assembled tasks (output of `rl_task_assemble.py`), scores each against
`rubrics/v03_task_quality.yaml`, and writes:

  * <out>            \u2014 tasks with `quality_judge` field appended, all kept
  * <kept-out>       \u2014 only tasks with weighted score \u2265 threshold

Usage:

    uv run scripts/data/build/judge_v03_tasks.py \\
        --input data/rl/humanize_tasks_v03.jsonl \\
        --output data/rl/humanize_tasks_v03.judged.jsonl \\
        --kept-output data/rl/humanize_tasks_v03_kept.jsonl \\
        --model google/gemini-3.1-flash-lite-preview \\
        --rubric rubrics/v03_task_quality.yaml \\
        --threshold 3.8
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import click
import yaml

_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json(text: str) -> dict | None:
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None


def _build_prompt(rubric: dict, task: dict) -> str:
    dims = rubric["dimensions"]
    weights = rubric["overall_weights"]
    dim_lines = []
    for d in dims:
        crit = "; ".join(f"{k}={v}" for k, v in d["criteria"].items())
        dim_lines.append(
            f"- {d['name']} ({d['scale_min']}\u2013{d['scale_max']}): "
            f"{d['description'].strip()}\n  criteria: {crit}"
        )
    schema = (
        "{ \"scores\": { "
        + ", ".join(f'"{d["name"]}": int' for d in dims)
        + " }, \"reasoning\": str }"
    )
    return (
        "You are a strict rubric judge for an LLM-task quality benchmark.\n"
        "Score the task spec below on each dimension (integer 1\u20135).\n\n"
        "Dimensions:\n" + "\n".join(dim_lines) + "\n\n"
        f"Weights: {json.dumps(weights)}\n"
        f"Output JSON only, matching: {schema}\n\n"
        "TASK SPEC:\n"
        f"  mode: {task['mode']}\n"
        f"  instruction:\n  {task['instruction']}\n\n"
        f"  input_text (first 600 chars):\n  "
        f"{task['input_text'][:600] if task.get('input_text') else '(none)'}\n\n"
        f"  constraints: {json.dumps(task.get('constraints', {}), indent=2)[:800]}\n"
    )


def _weighted_score(scores: dict, weights: dict) -> float:
    total = 0.0
    wsum = 0.0
    for dim, w in weights.items():
        if dim in scores and isinstance(scores[dim], (int, float)):
            total += float(scores[dim]) * float(w)
            wsum += float(w)
    return total / wsum if wsum > 0 else 0.0


@click.command(context_settings={"show_default": True})
@click.option("--input", "input_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              required=True)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path),
              required=True, help="All tasks + quality_judge field.")
@click.option("--kept-output", type=click.Path(dir_okay=False, path_type=Path),
              required=True, help="Only kept tasks (\u2265 threshold).")
@click.option("--model", default="google/gemini-3.1-flash-lite-preview")
@click.option("--rubric", type=click.Path(exists=True, dir_okay=False,
                                          path_type=Path),
              default=Path("rubrics/v03_task_quality.yaml"))
@click.option("--threshold", type=float, default=3.8)
@click.option("--max-tasks", type=int, default=None,
              help="Smoke / partial-run cap.")
@click.option("--api-key-var", default="OPENROUTER_API_KEY")
@click.option("--api-base-url", default="https://openrouter.ai/api/v1")
@click.option("--sleep", type=float, default=0.0)
@click.option("--reasoning-max-tokens", type=int, default=256,
              help="Cap OpenRouter reasoning budget on Gemini models.")
def main(
    input_path: Path, output: Path, kept_output: Path, model: str,
    rubric: Path, threshold: float, max_tasks: int | None,
    api_key_var: str, api_base_url: str, sleep: float,
    reasoning_max_tokens: int,
) -> None:
    """Judge v03 tasks against the v03_task_quality rubric."""
    from openai import OpenAI

    api_key = os.environ.get(api_key_var)
    if not api_key:
        raise click.ClickException(f"{api_key_var} not set.")

    rubric_obj = yaml.safe_load(rubric.read_text())
    weights = rubric_obj["overall_weights"]
    client = OpenAI(api_key=api_key, base_url=api_base_url)

    tasks: list[dict] = []
    with input_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    if max_tasks:
        tasks = tasks[:max_tasks]
    click.echo(f"judging {len(tasks)} tasks with {model}")

    output.parent.mkdir(parents=True, exist_ok=True)
    kept_output.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    errors = 0
    with output.open("w") as out_fh, kept_output.open("w") as kept_fh:
        for i, task in enumerate(tasks, start=1):
            prompt = _build_prompt(rubric_obj, task)
            extra = {"reasoning": {"max_tokens": reasoning_max_tokens}}
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.0,
                    extra_body=extra,
                )
                text = (resp.choices[0].message.content or "").strip()
                parsed = _extract_json(text)
            except Exception as exc:  # noqa: BLE001
                errors += 1
                parsed = None
                text = f"judge_error: {type(exc).__name__}: {exc}"
            if not parsed or "scores" not in parsed:
                verdict = {
                    "scores": {},
                    "weighted_score": 0.0,
                    "keep": False,
                    "reason": "unparseable_judge_output",
                    "raw": text[:200],
                }
            else:
                ws = _weighted_score(parsed["scores"], weights)
                verdict = {
                    "scores": parsed["scores"],
                    "weighted_score": round(ws, 3),
                    "keep": ws >= threshold,
                    "reasoning": parsed.get("reasoning", ""),
                }
            judged = {**task, "quality_judge": verdict}
            out_fh.write(json.dumps(judged, ensure_ascii=False) + "\n")
            if verdict.get("keep"):
                kept += 1
                kept_fh.write(json.dumps(judged, ensure_ascii=False) + "\n")
            if i % 50 == 0 or i == len(tasks):
                click.echo(
                    f"  {i}/{len(tasks)} kept={kept} errors={errors} "
                    f"last_score={verdict.get('weighted_score')}"
                )
            if sleep:
                time.sleep(sleep)

    rate = kept / len(tasks) if tasks else 0
    click.secho(
        f"done: kept={kept}/{len(tasks)} ({rate:.1%}) errors={errors}",
        fg="green",
    )
    click.echo(f"all judged   \u2192 {output}")
    click.echo(f"kept only    \u2192 {kept_output}")


if __name__ == "__main__":
    main()
