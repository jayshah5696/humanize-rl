# humanize-rl justfile

# Run tests
test *args:
    uv run pytest {{args}}

# Lint
lint:
    uv run ruff check .

# Format
format:
    uv run ruff format .

# Check (lint + format check)
check:
    uv run ruff check .
    uv run ruff format --check .

# Score text for humanness (Layer 1)
score text:
    uv run python -m humanize_rl.scoring.cli "{{text}}"

# Score a file
score-file path:
    uv run python -m humanize_rl.scoring.cli --file {{path}}

# Run benchmark on human vs AI samples
benchmark:
    uv run python -m humanize_rl.benchmark.cli

# Run benchmark and export scored output
benchmark-export:
    uv run python -m humanize_rl.benchmark.cli --output data/benchmark/scored_output.jsonl

# Run AIify pipeline (requires OPENROUTER_API_KEY)
aiify:
    uv run arka --config configs/01-aiify.yaml --run-id aiify-v01

# Humanize pipeline (requires OPENROUTER_API_KEY + AIify output)
humanize:
    uv run arka --config configs/02-humanize.yaml --run-id humanize-v04

# Score all outputs + compute 3-class AUROC + export SFT pairs
pipeline:
    uv run python -m humanize_rl.pipeline_cli

# Score all with Layer 1 + Layer 2 (LLM judge) — the real benchmark
score-all:
    uv run python -m humanize_rl.score_all

# Score all with cheaper model (dev/testing)
score-all-cheap:
    uv run python -m humanize_rl.score_all --model google/gemini-3-flash-preview

# Full end-to-end: aiify → humanize → score L1+L2 → export
full: aiify humanize score-all
