"""Generic RL-task assembler.

Reads arka outputs (one or more JSONL files), parses the embedded v03 task
payload from `response`, validates against `RLTask`, assigns canonical
`rl_v03_NNNNNN` ids + train/val/test split, and writes a single JSONL.

See plan §7.1.

Usage:

    uv run scripts/data/build/rl_task_assemble.py \\
        --inputs "data/processed/v03_*.jsonl" \\
        --out data/rl/humanize_tasks_v03.jsonl \\
        --split-seed 99 \\
        --split 0.8 0.1 0.1 \\
        --summary-out data/rl/humanize_tasks_v03_summary.json
"""

from __future__ import annotations

import glob
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

import click

from humanize_rl.reward.tasks import RLTask, summarize_tasks, write_tasks

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _strip_fence(text: str) -> str:
    m = _JSON_FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _extract_payload(response: str) -> dict[str, Any] | None:
    """Pull the v03 task payload out of an arka `response` value.

    Arka returns `{"text": "<stringified JSON>"}` from our mode prompts.
    Some authors will emit the payload directly; handle both.
    """
    if not response:
        return None
    text = _strip_fence(response)
    try:
        outer = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(outer, dict) and "text" in outer and isinstance(outer["text"], str):
        inner = _strip_fence(outer["text"])
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            return None
    if isinstance(outer, dict):
        return outer
    return None


def _infer_mode_from_path(path: Path) -> str | None:
    name = path.stem
    for mode in [
        "long_form_generate",
        "rewrite_humanize",
        "multi_constraint_compose",
        "compression",
        "tone_shift",
        "expansion",
    ]:
        if mode in name:
            return mode
    return None


def _infer_author_from_path(path: Path) -> str | None:
    """v03_<mode>_<author-slug>.jsonl where slug uses `_` for `/`."""
    name = path.stem
    for raw in [
        "google/gemini-3.1-pro-preview",
        "openai/gpt-5.4-mini",
        "google/gemini-3.1-flash-lite-preview",
        "google/gemini-3-flash-preview",
        "google/gemini-3-pro-preview",
    ]:
        slug = raw.replace("/", "_")
        if slug in name:
            return raw
    return None


def _profile_for_mode(mode: str) -> str:
    return {
        "rewrite_humanize": "rewrite_faithful_concise",
        "compression": "compression_update",
        "tone_shift": "rewrite_faithful_concise",
        "expansion": "rewrite_faithful_concise",
        "long_form_generate": "direct_workplace_message",
        "multi_constraint_compose": "direct_workplace_message",
    }.get(mode, "rewrite_faithful_concise")


def _family_for_mode(mode: str) -> str:
    return {
        "rewrite_humanize": "rewrite_repair",
        "compression": "compression",
        "tone_shift": "tone_shift",
        "expansion": "rewrite_repair",
        "long_form_generate": "direct_email",
        "multi_constraint_compose": "direct_email",
    }.get(mode, "rewrite_repair")


def _normalize_constraints(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not raw:
        return {}
    # Strip None values; arka authors sometimes emit nulls.
    return {k: v for k, v in raw.items() if v is not None and v != ""}


_VALID_DOMAINS = {
    "email", "slack", "technical", "product", "leadership",
    "support", "hiring", "creative_general",
}
_DOMAIN_ALIASES = {
    "business": "email", "corporate": "email", "workplace": "email",
    "chat": "slack", "messaging": "slack",
    "engineering": "technical", "computer vision": "technical",
    "academic": "technical", "science": "technical", "research": "technical",
    "marketing": "product", "copy": "product",
    "management": "leadership", "exec": "leadership",
    "customer": "support", "customer_support": "support",
    "recruiting": "hiring", "candidate": "hiring",
    "creative": "creative_general", "writing": "creative_general",
    "blog": "creative_general", "journalism": "creative_general",
}
_VALID_REGISTERS = {
    "casual", "neutral", "warm_professional", "candid", "terse",
    "technical", "founder_like",
    "warm", "direct", "formal", "academic", "journalistic", "literary",
}


def _normalize_domain(raw: str | None) -> str:
    if not raw:
        return "creative_general"
    key = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if key in _VALID_DOMAINS:
        return key
    if key in _DOMAIN_ALIASES:
        return _DOMAIN_ALIASES[key]
    key2 = raw.strip().lower()
    if key2 in _DOMAIN_ALIASES:
        return _DOMAIN_ALIASES[key2]
    return "creative_general"


def _normalize_register(raw: str | None) -> str:
    if not raw:
        return "direct"
    key = raw.strip().lower()
    if key in _VALID_REGISTERS:
        return key
    return {"professional": "warm_professional", "friendly": "warm",
            "informal": "casual", "conversational": "casual"}.get(key, "direct")


def _build_task(
    payload: dict[str, Any],
    mode: str,
    author: str | None,
    source_dataset: str,
    source_row_id: str,
    index: int,
) -> RLTask:
    constraints = _normalize_constraints(payload.get("constraints"))
    # Drop verifier-incompatible keys silently rather than fail validation.
    instruction = (payload.get("instruction") or "").strip()
    input_text = (payload.get("input_text") or "").strip()
    domain = _normalize_domain(payload.get("domain"))
    register = _normalize_register(
        payload.get("register") or constraints.get("register_target")
    )
    trap_tags = payload.get("trap_tags") or ["wrapper_phrase"]
    reward_profile = payload.get("reward_profile") or _profile_for_mode(mode)
    family = payload.get("family") or _family_for_mode(mode)

    return RLTask.model_validate(
        {
            "id": f"rl_v03_{index:06d}",
            "family": family,
            "domain": domain,
            "mode": mode,
            "register": register,
            "instruction": instruction,
            "input_text": input_text,
            "constraints": constraints,
            "reward_profile": reward_profile,
            "trap_tags": list(trap_tags),
            "source": source_dataset,
            "source_group": source_dataset,
            "split": "train",  # overwritten below
            "task_author_model": author,
        }
    )


def _assign_splits(
    tasks: list[RLTask], split: tuple[float, float, float], seed: int
) -> list[RLTask]:
    train, val, test = split
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError(f"split must sum to 1.0, got {train + val + test}")
    rng = random.Random(seed)
    idx = list(range(len(tasks)))
    rng.shuffle(idx)
    n = len(tasks)
    n_tr = int(round(train * n))
    n_va = int(round(val * n))
    splits_map: dict[int, str] = {}
    for k, i in enumerate(idx):
        if k < n_tr:
            splits_map[i] = "train"
        elif k < n_tr + n_va:
            splits_map[i] = "validation"
        else:
            splits_map[i] = "test"
    out: list[RLTask] = []
    for i, task in enumerate(tasks):
        # RLTask is frozen-ish (BaseModel); rebuild via dict.
        data = task.model_dump(by_alias=True, exclude_none=True)
        data["split"] = splits_map[i]
        out.append(RLTask.model_validate(data))
    return out


def _expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pat in patterns:
        matches = [Path(p) for p in glob.glob(pat)]
        if not matches and Path(pat).exists():
            matches = [Path(pat)]
        paths.extend(sorted(matches))
    return paths


def _stable_dedupe_key(task: RLTask) -> str:
    payload = f"{task.instruction}\n---\n{task.input_text}"
    return hashlib.sha1(payload.encode()).hexdigest()


@click.command(context_settings={"show_default": True})
@click.option(
    "--inputs", multiple=True, required=True,
    help="Glob or path to arka output JSONL(s). Repeatable.",
)
@click.option(
    "--out", type=click.Path(dir_okay=False, path_type=Path), required=True,
    help="Output validated-task JSONL.",
)
@click.option(
    "--split", type=(float, float, float), default=(0.8, 0.1, 0.1),
    help="train val test fractions.",
)
@click.option("--split-seed", type=int, default=99)
@click.option(
    "--summary-out", type=click.Path(dir_okay=False, path_type=Path), default=None,
    help="Write summary stats JSON.",
)
@click.option(
    "--max-tasks", type=int, default=None,
    help="Cap output (for smoke runs).",
)
def main(
    inputs: tuple[str, ...],
    out: Path,
    split: tuple[float, float, float],
    split_seed: int,
    summary_out: Path | None,
    max_tasks: int | None,
) -> None:
    """Assemble v03 RL tasks from arka outputs."""
    paths = _expand_inputs(list(inputs))
    if not paths:
        raise click.ClickException(f"no inputs matched: {inputs}")
    click.echo(f"reading {len(paths)} input files:")
    for p in paths:
        click.echo(f"  - {p}")

    tasks: list[RLTask] = []
    skipped = {"empty_response": 0, "no_payload": 0, "validation_error": 0}
    seen_keys: set[str] = set()
    index = 1

    for path in paths:
        mode = _infer_mode_from_path(path)
        if not mode:
            click.secho(f"  ! cannot infer mode from {path.name}; skipping", fg="yellow")
            continue
        author = _infer_author_from_path(path)
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            response = row.get("response") or ""
            if not response:
                skipped["empty_response"] += 1
                continue
            payload = _extract_payload(response)
            if not payload:
                skipped["no_payload"] += 1
                continue
            source_dataset = row.get("source_dataset") or path.stem
            source_row_id = row.get("source_row_id") or "unknown"
            try:
                task = _build_task(
                    payload, mode, author, source_dataset, source_row_id, index
                )
            except ValueError as exc:
                skipped["validation_error"] += 1
                click.secho(f"  ! {path.name}: {exc}", fg="yellow")
                continue
            key = _stable_dedupe_key(task)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            tasks.append(task)
            index += 1
            if max_tasks and len(tasks) >= max_tasks:
                break
        if max_tasks and len(tasks) >= max_tasks:
            break

    click.echo(f"kept tasks: {len(tasks)}")
    click.echo(f"skipped:    {skipped}")

    tasks = _assign_splits(tasks, split, split_seed)
    write_tasks(out, tasks)
    click.secho(f"wrote {len(tasks)} tasks → {out}", fg="green")

    if summary_out:
        summary = {
            "count": len(tasks),
            "skipped": skipped,
            "split": {"split": list(split), "seed": split_seed},
            **summarize_tasks(tasks),
        }
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(json.dumps(summary, indent=2))
        click.echo(f"summary → {summary_out}")


if __name__ == "__main__":
    main()
