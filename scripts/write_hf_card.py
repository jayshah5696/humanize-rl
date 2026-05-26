#!/usr/bin/env python3
"""Generate Hugging Face dataset/model cards from prepared artifacts."""

from __future__ import annotations

import collections
import json
from pathlib import Path

import click


def infer_domain(row: dict) -> str:
    d = row.get("domain") or row.get("origin_domain") or ""
    if d:
        return d
    inst = row.get("instruction", "").lower()
    if "slack" in inst or "channel" in inst or " dm " in inst:
        return "chat"
    if "email" in inst or "dear " in inst:
        return "email"
    if "shorten" in inst or "compress" in inst or "tighten" in inst:
        return "compress"
    if "grammar" in inst or "fix the" in inst or "typo" in inst:
        return "grammar"
    if "story" in inst or "paragraph" in inst or "vivid" in inst or "creative" in inst:
        return "creative"
    return "general"


@click.group()
def cli() -> None:
    """Generate Hugging Face model/dataset card READMEs."""
    pass


@cli.command()
@click.option(
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Input SFT dataset JSONL file.",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(),
    help="Output path for SFT card README.",
)
@click.option("--version", default="v2", help="Version of the SFT dataset.")
def sft(input_file: str, output: str, version: str) -> None:
    """Generate Hugging Face card for SFT dataset."""
    raw_lines = Path(input_file).read_text().splitlines()
    rows = [json.loads(line) for line in raw_lines if line.strip()]

    domain_counts = collections.Counter(infer_domain(r) for r in rows)
    source_counts = collections.Counter(
        r.get("source") or r.get("origin_source") or "curated" for r in rows
    )
    ph_count = sum("[" in r.get("response", "") for r in rows)

    naturalness_scores = []
    for r in rows:
        nat = None
        if "naturalness_judge" in r:
            nat = r["naturalness_judge"]
        elif "quality_judge" in r and isinstance(r["quality_judge"], dict):
            nat = r["quality_judge"].get("naturalness")
        if nat is not None:
            naturalness_scores.append(float(nat))

    avg_naturalness = (
        sum(naturalness_scores) / len(naturalness_scores) if naturalness_scores else 0.0
    )

    domain_rows_md = "\n".join(
        f"| {k} | {v} | {v / len(rows):.1%} |" for k, v in domain_counts.most_common()
    )

    source_rows_md = "\n".join(f"| {k} | {v} |" for k, v in source_counts.most_common())

    card = f"""---
configs:
  - config_name: {version}
    data_files:
      - split: train
        path: data/{version}/train-00000-of-00001.parquet
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
size_categories:
  - 1K<n<10K
---

# humanize-rl-sft-dataset ({version})

**{len(rows):,}** high-quality SFT pairs for training a model to write natural, direct prose.

Part of the [humanize-rl](https://github.com/jayshah5696/humanize-rl) project — a two-layer scoring and alignment pipeline for training small models to generate natural, human-sounding text.

## What this trains

A model that can:
- Write natural Slack messages and emails from scratch.
- Rewrite stiff/formal/corporate text into direct, human-sounding prose.
- Fix grammar without making text formal.
- Shorten and compress without losing meaning.

## Domain breakdown

| Domain | Rows | % |
|--------|------|---|
{domain_rows_md}

## Quality

- All rows passed a deterministic heuristic quality check.
- All rows passed a Flash Lite LLM judge (naturalness ≥ 4, fact preservation, no AI tells).
- Average naturalness score: **{avg_naturalness:.2f} / 5.0** (based on {len(naturalness_scores)} graded rows)
- Bad-phrase rate (Certainly, Furthermore, etc.): < 0.1%
- Placeholder rows (containing '['): **{ph_count}** ({ph_count / len(rows):.1%})

## Sources

| Source | Rows |
|--------|------|
{source_rows_md}

- `safe_expand_*`: generated from curated seeds using a safe prompt-based expansion.
- `stream_b`: instruction-response pairs generated from real human-written text (wardacoder, corbt/enron-emails, liamdugan/raid human rows, euclaise/writingprompts).
- `chat_expanded`: Slack/chat focused pairs generated from hand-crafted direct seeds.
- `curated`: hand-verified base seeds.

## Schema

| Field | Type | Description |
|-------|------|-------------|
| id | str | Row ID |
| instruction | str | User task or rewrite request |
| response | str | Natural human-sounding response |
| messages | list | ShareGPT format: user/assistant turns |
| domain | str | chat / email / general / creative / grammar / compress |
| source | str | Data stream origin |
| mode | str | direct_generation or rewrite_humanize |
| naturalness_judge | int | Flash Lite judge score (1-5), null for curated rows |
| version | str | "{version}" |

## Project

This dataset is built and maintained as part of the **humanize-rl** project.

- **GitHub:** [https://github.com/jayshah5696/humanize-rl](https://github.com/jayshah5696/humanize-rl)
- **Goal:** Train small open models (Gemma 4 E2B) to natively produce natural, human-sounding text — both from scratch and by rewriting stiff/formal drafts.
- **Architecture:** Two-layer scoring pipeline (Layer 1 deterministic heuristics + Layer 2 LLM judge), SFT on this dataset, optional RL post-training with DAPO.

## License

Apache-2.0. Source datasets have individual licenses — see [`data/source_manifest_v04.json`](https://github.com/jayshah5696/humanize-rl/blob/main/data/source_manifest_v04.json) in the training repo for attribution details.
"""
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(card)
    click.echo(f"Wrote SFT dataset card ({version}) to {output}")


@cli.command()
@click.option(
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Input RL tasks JSONL file.",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(),
    help="Output path for RL tasks card README.",
)
def rl_tasks(input_file: str, output: str) -> None:
    """Generate Hugging Face card for RL tasks dataset."""
    raw_lines = Path(input_file).read_text().splitlines()
    rows = [json.loads(line) for line in raw_lines if line.strip()]

    source_counts = collections.Counter(
        r.get("dataset_source") or r.get("source") or "unknown" for r in rows
    )
    family_counts = collections.Counter(r.get("family") or "unknown" for r in rows)
    split_counts = collections.Counter(r.get("split") or "train" for r in rows)

    total_tasks = len(rows)
    total_families = len(family_counts)

    source_rows_md = "\n".join(f"| {k} | {v} |" for k, v in source_counts.most_common())

    split_rows_md = "\n".join(f"| {k} | {v} |" for k, v in split_counts.most_common())

    card = f"""---
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

| source | tasks |
|---|---|
{source_rows_md}
| **total** | **{total_tasks}** |

- Total task families: {total_families}

## Splits

| split | rows |
|---|---|
{split_rows_md}

## Schema

| field | type | description |
|---|---|---|
| `id` | string | Unique task ID |
| `family` | string | Task family (rewrite_repair, compression, tone_shift, ...) |
| `domain` | string | Domain (email, slack, technical, ...) |
| `mode` | string | Task mode (rewrite, direct_generation) |
| `register` | string | Target register (casual, terse, warm_professional, ...) |
| `instruction` | string | Full user-facing instruction |
| `input_text` | string | Source text to rewrite (empty for direct_generation) |
| `constraints` | dict | Hard constraints (max_words, preserve_numbers, ...) |
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
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(card)
    click.echo(f"Wrote RL tasks card to {output}")


@cli.command()
@click.option(
    "--metadata",
    required=True,
    type=click.Path(exists=True),
    help="Path to model training metadata.json.",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(),
    help="Output path for scorer card README.",
)
def scorer(metadata: str, output: str) -> None:
    """Generate Hugging Face card for surrogate scorer model."""
    meta = json.loads(Path(metadata).read_text())

    version = meta.get("version", "unknown")
    data_path = meta.get("data_path", "unknown")
    rubric_path = meta.get("rubric_path", "unknown")
    rubric_dims = meta.get("rubric_dimensions", [])
    summary = meta.get("summary", {})
    models = meta.get("models", {})

    dims_md = "\n".join(f"- `{d}`" for d in rubric_dims)

    models_md = ""
    for model_name, info in models.items():
        models_md += f"- **{model_name}**:\n"
        models_md += f"  - Path: `{info.get('path')}`\n"
        models_md += f"  - Train Time: {info.get('train_seconds', 0.0):.2f}s\n"

    card = f"""# Humanize-RL Distilled Scorer ({version})

Recommended local distilled humanness surrogate scorers trained on L2 rubric-labeled rows.

## Overview

Surrogate scorers trained on Gemini Layer-2 rubric-labeled rows. Used to score humanized completions locally with low latency and high alignment to the full LLM judge.

- **Training data:** `{data_path}` ({summary.get("samples", 0):,} total samples)
  - **AI samples:** {summary.get("ai_samples", 0):,}
  - **Human-ish samples:** {summary.get("humanish_samples", 0):,}
- **Rubric path:** `{rubric_path}`
- **Default AI probability threshold:** {meta.get("default_ai_probability_threshold", 0.5)}
- **Sklearn version:** {meta.get("sklearn_version", "unknown")}

## Rubric Dimensions Scored

{dims_md}

## Trained Surrogate Models

{models_md}

## Why Ridge?

Ridge/TF-IDF has the best practical tradeoff: ~0.997-0.999 AUROC, lowest rubric MSE among practical models, ~1ms/row latency, and 0.0% false positives on human-authored challenge sets.
"""
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(card)
    click.echo(f"Wrote scorer card ({version}) to {output}")


if __name__ == "__main__":
    cli()
