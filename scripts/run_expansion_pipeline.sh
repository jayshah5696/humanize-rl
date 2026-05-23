#!/bin/bash
set -e

echo "1. Duplicating seeds..."
rtk uv run python scripts/duplicate_seeds.py \
    --input seeds/v03/expansion_seeds.jsonl \
    --output seeds/v03/expansion_seeds_x1.jsonl \
    --copies 1

echo "2. AIify..."
rtk uv run arka --config configs/v03/01-aiify-expansion.yaml --run-id expansion-aiify

echo "3. AIify select..."
rtk uv run python -m humanize_rl.data.selector \
    --mode aiify \
    --input output/v03/expansion-aiify-candidates.jsonl \
    --output output/v03/expansion-aiify.jsonl \
    --report runs/v03/expansion-aiify-selection.json

echo "4. Duplicate AIify..."
rtk uv run python scripts/duplicate_seeds.py \
    --input output/v03/expansion-aiify.jsonl \
    --output output/v03/expansion-aiify-x1.jsonl \
    --copies 1

echo "5. Humanize..."
rtk uv run arka --config configs/v03/02-humanize-expansion.yaml --run-id expansion-humanize

echo "6. Humanize select..."
rtk uv run python -m humanize_rl.data.selector \
    --mode humanize \
    --input output/v03/expansion-humanize-candidates.jsonl \
    --output output/v03/expansion-humanize.jsonl \
    --originals seeds/v03/expansion_seeds.jsonl \
    --aiify-selected output/v03/expansion-aiify.jsonl \
    --report runs/v03/expansion-humanize-selection.json

echo "7. Score & Export..."
rtk uv run python -m humanize_rl.data.walking_skeleton \
    --aiify-output output/v03/expansion-aiify.jsonl \
    --humanize-output output/v03/expansion-humanize.jsonl \
    --benchmark-out data/benchmark/expansion_matched.jsonl \
    --sft-out data/processed/expansion_sft.jsonl \
    --report-out runs/v03/expansion_report.json \
    --seeds seeds/v03/expansion_seeds.jsonl

echo "Pipeline complete! Outputs saved to data/processed/expansion_sft.jsonl"
