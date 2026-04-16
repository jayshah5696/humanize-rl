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

## License

Apache 2.0
