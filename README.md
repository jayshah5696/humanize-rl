# humanize-rl

Two-layer AI writing detection, scoring, and humanization training pipeline.

## Overview

- **Layer 1** (deterministic, free): 8 regex/heuristic dimensions — detects AI writing patterns in microseconds
- **Layer 2** (LLM judge, paid): 8 dimensions via [arka](https://github.com/jayshah5696/arka) LabelingEngine + rubric YAML
- **Training**: SFT on Gemma 4 26B A4B (bf16 LoRA), conditional RL via DAPO

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
Layer 1: Deterministic Scoring (free, <10ms, deterministic)
    8 dims: opener, hedging, lists, sentence variance,
    contractions, closing, em-dash, transitions
    │
    ▼ gate: skip Layer 2 if clearly human/AI
Layer 2: LLM Judge Scoring (paid, 2-5s, nuanced)
    8 dims: structure, specificity, formality, voice,
    rhetoric, padding, personality, copula
    │
    ▼
Combined Score → Training Data → SFT/RL
```

## Repository status

See `STATUS.md` for current progress and next steps.

## License

Apache 2.0
