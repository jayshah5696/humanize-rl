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

# ---------------------------------------------------------------------------
# V03 walking skeleton (V-Slice 0): 10 instruction_technical seeds end-to-end.
# Cost: ~$0.05 (Flash Lite) + ~$0.50 (Pro) = ~$0.55 per full run.
# ---------------------------------------------------------------------------

# Build the 10 hand-pasted walking-skeleton seeds.
v03-ws-seeds:
    uv run python scripts/build_walking_skeleton_seeds.py
    uv run python scripts/duplicate_seeds.py \
        --input seeds/v03/walking_skeleton.jsonl \
        --output seeds/v03/walking_skeleton_x2.jsonl \
        --copies 4

# AIify: 20 candidates (2 per seed).
v03-ws-aiify:
    uv run arka --config configs/v03/01-aiify-walking-skeleton.yaml --run-id v03-ws-aiify

# AIify selector: keep the best of the 2 candidates per seed (→ 10).
v03-ws-aiify-select:
    uv run python -m humanize_rl.data.selector \
        --mode aiify \
        --input output/v03/ws-aiify-candidates.jsonl \
        --output output/v03/ws-aiify.jsonl \
        --report runs/v03/ws-aiify-selection.json
    uv run python scripts/duplicate_seeds.py \
        --input output/v03/ws-aiify.jsonl \
        --output output/v03/ws-aiify-x2.jsonl \
        --copies 3

# Humanize: 20 candidates (2 per AIified input).
v03-ws-humanize:
    uv run arka --config configs/v03/02-humanize-walking-skeleton.yaml --run-id v03-ws-humanize

# Humanize selector: keep the best humanize candidate per input (→ 10).
v03-ws-humanize-select:
    uv run python -m humanize_rl.data.selector \
        --mode humanize \
        --input output/v03/ws-humanize-candidates.jsonl \
        --output output/v03/ws-humanize.jsonl \
        --originals seeds/v03/walking_skeleton.jsonl \
        --aiify-selected output/v03/ws-aiify.jsonl \
        --report runs/v03/ws-humanize-selection.json

# Score (L1 only), gate, export tiny benchmark + SFT pairs.
v03-ws-score:
    uv run python -m humanize_rl.data.walking_skeleton

# V-Slice 4: emit v03_core / v03_ood_ai / v03_diagnostics + full report.
v03-ws-report:
    uv run python -m humanize_rl.data.report_v03

# Full v03 walking skeleton in one shot.
v03-ws: v03-ws-seeds v03-ws-aiify v03-ws-aiify-select v03-ws-humanize v03-ws-humanize-select v03-ws-score v03-ws-report
