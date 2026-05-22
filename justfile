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

# ---------------------------------------------------------------------------
# V03 1,000-seed scaleup
# ---------------------------------------------------------------------------

# Build the 8,000-seed dataset via HF streaming loaders
seeds-scaleup:
	uv run python -m humanize_rl.data.loaders --num-seeds 8000 --output seeds/v03/corpus_seeds.jsonl

# Generate duplicated seeds for 1-to-1 AIify selection
v03-scaleup-seeds:
	uv run python scripts/duplicate_seeds.py \
		--input seeds/v03/corpus_seeds.jsonl \
		--output seeds/v03/corpus_seeds_x1.jsonl \
		--copies 1

# AIify 8,000 seeds (8,000 candidates)
v03-corpus-aiify:
	uv run arka --config configs/v03/01-aiify-corpus.yaml --run-id v03-corpus-aiify

# AIify selection and duplication (copies 1)
v03-corpus-aiify-select:
	uv run python -m humanize_rl.data.selector \
		--mode aiify \
		--input output/v03/corpus-aiify-candidates.jsonl \
		--output output/v03/corpus-aiify.jsonl \
		--report runs/v03/corpus-aiify-selection.json
	uv run python scripts/duplicate_seeds.py \
		--input output/v03/corpus-aiify.jsonl \
		--output output/v03/corpus-aiify-x1.jsonl \
		--copies 1

# Humanize (8,000 candidates)
v03-corpus-humanize:
	uv run arka --config configs/v03/02-humanize-corpus.yaml --run-id v03-corpus-humanize

# Humanize selection
v03-corpus-humanize-select:
	uv run python -m humanize_rl.data.selector \
		--mode humanize \
		--input output/v03/corpus-humanize-candidates.jsonl \
		--output output/v03/corpus-humanize.jsonl \
		--originals seeds/v03/corpus_seeds.jsonl \
		--aiify-selected output/v03/corpus-aiify.jsonl \
		--report runs/v03/corpus-humanize-selection.json

# Score with L1, gate, export matched corpus benchmark + SFT pairs
v03-corpus-score:
	uv run python -m humanize_rl.data.walking_skeleton \
		--aiify-output output/v03/corpus-aiify.jsonl \
		--humanize-output output/v03/corpus-humanize.jsonl \
		--benchmark-out data/benchmark/v03_corpus_matched.jsonl \
		--sft-out data/processed/v03_corpus_sft.jsonl \
		--report-out runs/v03/corpus_skeleton_report.json \
		--seeds seeds/v03/corpus_seeds.jsonl

# Report generation
v03-corpus-report:
	uv run python -m humanize_rl.data.report_v03 \
		--matched data/benchmark/v03_corpus_matched.jsonl \
		--sft data/processed/v03_corpus_sft.jsonl \
		--core-out data/benchmark/v03_corpus_core.jsonl \
		--ood-out data/benchmark/v03_corpus_ood_ai.jsonl \
		--diagnostics-out data/benchmark/v03_corpus_diagnostics.jsonl \
		--report-json runs/v03/v03_corpus_report.json \
		--report-md runs/v03/v03_corpus_report.md

# E2E corpus scale-up pipeline
v03-corpus: v03-scaleup-seeds v03-corpus-aiify v03-corpus-aiify-select v03-corpus-humanize v03-corpus-humanize-select v03-corpus-score v03-corpus-report
