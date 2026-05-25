# humanize-rl

Two-layer AI writing detection, scoring, and humanization training pipeline.

## Overview

- **Layer 1** (deterministic, free): 8 regex/heuristic dimensions — detects AI writing patterns in microseconds
- **Layer 2** (LLM judge, paid): 8 dimensions via [arka](https://github.com/jayshah5696/arka) LabelingEngine + rubric YAML
- **Training**: SFT on Gemma 4 E2B (bf16 LoRA), conditional RL via DAPO

## Quick Start

```bash
uv sync
just test
just lint
just score "Your text here"
```

## Setup

### Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- [`just`](https://github.com/casey/just)

### Dependency model

This repo depends on [`arka`](https://github.com/jayshah5696/arka).

For reproducibility, `pyproject.toml` pins `arka` as a GitHub source through `uv`, so fresh clones and CI do not rely on a machine-specific local path.

### Environment

Layer 2 scoring and Arka pipeline runs require an OpenRouter API key:

```bash
export OPENROUTER_API_KEY=...
```

## Common commands

```bash
just test
just lint
just format
just check
just score "Your text here"
just score-file path/to/file.txt
just benchmark
just aiify
just humanize
just pipeline
just score-all
```

## Generated outputs

Common generated artifacts:

- `output/01-aiify-dataset.jsonl`
- `output/02-humanize-dataset.jsonl`
- `data/benchmark/scored_output.jsonl`
- `data/benchmark/scored_3class_v01.jsonl`
- `data/benchmark/scored_combined_v01.jsonl`
- `data/processed/`

## Architecture

```
Input Text
    │
    ▼
Layer 1: Deterministic Scoring (free, <1ms)
    8 regex/heuristic dims: opener, hedging, lists,
    sentence variance, contractions, closing, em-dash, transitions
    │
    ▼ gate: skip Layer 2 if clearly human/AI
Layer 2: LLM Judge Scoring (paid, 2-5s)
    8 dims: structure, specificity, formality, voice,
    rhetoric, padding, personality, copula
    │
    ▼
Track A Ridge Scorer (distilled from Layer 2 labels, 1ms)
    TF-IDF Logistic+Ridge → binary P(AI) + 8 rubric dims
    AUROC 0.9988, rubric MSE 0.041 on 10k labelled rows
    │
    ▼
RL Reward (50/50 formula)
    0.50 × ridge_rubric_mean  ← 8 humanness dims
    0.50 × deterministic_mean ← faithfulness, task_following,
                                 length, format, placeholder, clarity
    + additive penalties      ← invented_detail, option_menu, etc.
    → clipped to [-1, 1]
    │
    ▼
SFT/RL Training → Gemma 4 E2B (bf16 LoRA)
```

## Reward rubric

See [`src/humanize_rl/reward/README.md`](src/humanize_rl/reward/README.md) for the full formula, all 8 ridge dims, 6 deterministic components, and penalty table.

See [`environments/humanize_rl_env/README.md`](environments/humanize_rl_env/README.md) for the Prime Intellect environment spec.

## Repository status

See `STATUS.md` for current progress and next steps.

## License

Apache 2.0
