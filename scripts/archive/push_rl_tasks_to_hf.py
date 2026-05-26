"""Push combined RL task dataset to HuggingFace.

Combines v01 (100 template tasks, 10 families) + v02 (512 real-source tasks, 3 families)
into jayshah5696/humanize-rl-tasks with train/validation/test splits.

Usage:
    uv run scripts/push_rl_tasks_to_hf.py
"""
from __future__ import annotations

import json
from pathlib import Path

import click
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

REPO_ID = "jayshah5696/humanize-rl-tasks"

SOURCES = [
    ("data/rl/humanize_tasks_v01_smoke.jsonl", "v01_template"),
    ("data/rl/humanize_tasks_v02.jsonl", "v02_real_source"),
]

README = """\
---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
tags:
  - rl
  - reward
  - humanize
  - writing
  - rewrite
  - verifiers
  - prime-intellect
size_categories:
  - n<1K
---

# humanize-rl-tasks

Single-turn RL task dataset for the [humanize-rl](https://github.com/jayshah5696/humanize-rl) project.

Each row is one writing task. A model receives the `prompt` (instruction + source text),
produces a completion, and the environment scores it with the 50/50 reward formula:

```
reward = 0.50 × ridge_rubric_mean + 0.50 × deterministic_mean + penalties
```

## Dataset composition

| source | tasks | families | input length |
|---|---|---|---|
| v01_template | 100 | 10 (template-generated) | ~20 words |
| v02_real_source | 512 | 3 (real email/creative/general) | ~103 words avg, up to 221 |
| **total** | **612** | **13** | |

## Splits

| split | rows |
|---|---|
| train | 489 |
| validation | 62 |
| test | 61 |

## Schema

| field | type | description |
|---|---|---|
| `id` | string | Unique task ID (`rl_v01_XXXXXX`) |
| `family` | string | Task family (rewrite_repair, compression, tone_shift, …) |
| `domain` | string | Domain (email, slack, technical, …) |
| `mode` | string | Task mode (rewrite, direct_generation) |
| `register` | string | Target register (casual, terse, warm_professional, …) |
| `instruction` | string | Full user-facing instruction |
| `input_text` | string | Source text to rewrite (empty for direct_generation) |
| `constraints` | dict | Hard constraints (max_words, preserve_numbers, …) |
| `reward_profile` | string | Which reward profile to use |
| `required_facts` | list[str] | Facts that must appear in a correct response |
| `forbidden_facts` | list[str] | Facts that must NOT appear |
| `forbidden_phrases` | list[str] | Phrases that must NOT appear |
| `trap_tags` | list[str] | Expected failure modes for this task |
| `split` | string | train / validation / test |
| `source` | string | Dataset version/source |

## Reward formula

See [reward README](https://github.com/jayshah5696/humanize-rl/blob/main/src/humanize_rl/reward/README.md).

## Prime Intellect environment

```bash
prime env install jayshah5696/humanize-rl-env
prime eval run jayshah5696/humanize-rl-env --model openrouter/free \\
  --api-base-url https://openrouter.ai/api/v1 \\
  --api-key-var OPENROUTER_API_KEY \\
  --num-examples 20 --rollouts-per-example 4 --max-tokens 1024
```
"""


def flatten_constraints(row: dict) -> dict:
    """Flatten nested constraints dict and serialise lists as JSON strings."""
    flat = dict(row)
    constraints = flat.pop("constraints", {}) or {}
    if isinstance(constraints, str):
        try:
            constraints = json.loads(constraints)
        except Exception:
            constraints = {}
    for k, v in constraints.items():
        flat[f"constraint_{k}"] = v
    # Serialise list fields so HF datasets handles them uniformly
    for field in ("required_facts", "forbidden_facts", "forbidden_phrases", "trap_tags"):
        val = flat.get(field, [])
        if isinstance(val, list):
            flat[field] = json.dumps(val)
    return flat


@click.command()
@click.option("--dry-run", is_flag=True, help="Print stats without pushing")
def main(dry_run: bool) -> None:
    api = HfApi()
    user = api.whoami()["name"]
    click.echo(f"HF user: {user}")

    # Load + combine
    all_rows: list[dict] = []
    for path_str, source_tag in SOURCES:
        path = Path(path_str)
        if not path.exists():
            click.echo(f"  SKIP (not found): {path}", err=True)
            continue
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        for r in rows:
            r["dataset_source"] = source_tag
        all_rows.extend(rows)
        click.echo(f"  loaded {len(rows):4d} rows from {path.name}")

    click.echo(f"Total: {len(all_rows)} tasks")

    # Split
    by_split: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for row in all_rows:
        split = row.get("split", "train")
        if split not in by_split:
            split = "train"
        by_split[split].append(flatten_constraints(row))

    for split, rows in by_split.items():
        click.echo(f"  {split}: {len(rows)}")

    if dry_run:
        click.echo("Dry run — not pushing.")
        return

    # Build DatasetDict
    ds_dict = DatasetDict({
        split: Dataset.from_list(rows)
        for split, rows in by_split.items()
        if rows
    })
    click.echo(f"\nDatasetDict: {ds_dict}")

    # Create repo
    api.create_repo(REPO_ID, repo_type="dataset", exist_ok=True, private=False)
    click.echo(f"Repo ready: https://huggingface.co/datasets/{REPO_ID}")

    # Push dataset
    click.echo("Pushing dataset…")
    ds_dict.push_to_hub(REPO_ID, commit_message="Add humanize-rl-tasks v01+v02 combined (612 tasks)")
    click.echo("✓ Dataset pushed")

    # Push README
    api.upload_file(
        path_or_fileobj=README.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Add dataset card",
    )
    click.echo(f"✓ README pushed\n→ https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
