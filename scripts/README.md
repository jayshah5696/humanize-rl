# Scripts

Project scripts are organized by workflow subdirectories to avoid clutter, using small reusable python tools instead of a single massive app-wide command CLI.

Run scripts via:

```bash
uv run scripts/<subfolder>/<script>.py --option value
```

---

## Shared Utility Tools (Root Level)

These three Click-based utility scripts reside at the root of `scripts/`:

- **`scripts/publish_to_hf.py`**: Pushes datasets, model folders, and extra artifacts to Hugging Face Hub. Resolves API tokens using `HF_TOKEN`, `HUGGINGFACE_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, or cached Hub login tokens.
- **`scripts/write_hf_card.py`**: Generates README cards (`README.md`) for SFT datasets, RL task datasets, and surrogate models based on metadata/statistics from local files.
- **`scripts/jsonl_tool.py`**: Reusable data operations tool for JSONL files.
  - `count`: count rows.
  - `sample`: random sample (with seed) or slice first N lines.
  - `dedupe`: remove duplicates based on a single field or composite comma-separated keys.
  - `merge`: sequentially merge multiple files.
  - `split`: split into train/validation/test sets.
  - `duplicate`: replicate rows N times and tag indices.

---

## Directory Structure

All specialized scripts are organized into these workflow subdirectories:

### Data Preparation
- **`scripts/data/seeds/`**: Seed builders and seed sampling (e.g. chat seeds, expansion seeds, walking skeleton seeds).
- **`scripts/data/fetch/`**: Fetching datasets from stream sources.
- **`scripts/data/build/`**: Compiling SFT datasets, extracting Prime audit failure sets, packaging Prime SFT messages datasets, labeling pools, false positive challenge sets, and RL tasks.

### Workflows
- **`scripts/label/`**: Labeling engines, quality judges, and verification gates.
- **`scripts/eval/`**: Audit scripts, scorer/reward evaluations, Prime ablation matrices, detector-mimic runs, Pangram bulk exports/runs and alignment comparisons, SFT eval manifests, SFT/RL promotion gates, and MLX output evaluations.
- **`scripts/train/`**: Training pipelines, local MLX tuning scripts, smoke test wrappers, and Prime SFT launch/output/checkpoint/config handoff helpers, including `verify_prime_sft_launch_readiness.py` for the final pre-spend SFT launch artifact check.

### Frameworks & Infrastructure
- **`scripts/mlx/`**: MLX weights conversion, quantization, and merge utilities.
- **`scripts/rl/`**: Reward environment validation and RL rollouts. Includes
  `build_p5050_filtered_mix.py` for rebuilding the Prime p50 task set from saved
  Modal rollouts and `report_taskset.py` for writing Prime taskset audit reports.
- **`scripts/figures/`**: Figure generation, plotting loss curves, and architectural diagrams.

### History
- **`scripts/archive/`**: Superseded/legacy scripts kept for traceability.

---

## General Rules

- **Click CLI**: All new or rewritten Python scripts must use Click for CLI argument parsing.
- **TDD**: Write a failing test first in `tests/` before implementing new script features.
- **No Duplication**: Check if a subcommand in `jsonl_tool.py` or `publish_to_hf.py` can solve your task before writing a new one-off script.
- **Archive First**: Move superseded scripts to `scripts/archive/` instead of deleting them.
