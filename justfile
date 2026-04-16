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

# Score AIified output + compute AUROC on real LLM data
pipeline:
    uv run python -m humanize_rl.pipeline_cli
