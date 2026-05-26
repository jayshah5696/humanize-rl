# justfile for humanize-rl

test:
    PYTHONPATH=src uv run pytest tests/data/test_scorer_dataset.py tests/scoring/test_distilled_scorers.py tests/benchmark/test_evaluate.py tests/test_jsonl_tool.py


train-scorer model_type:
    uv run python -m humanize_rl.training.train_scorer --model-type {{model_type}}


eval-scorer model_type:
    uv run python -m humanize_rl.benchmark.evaluate_scorer {{model_type}}

ui:
    uv run mlflow ui

lint:
    uv run ruff check src/ tests/
    uv run ruff format --check src/ tests/

format:
    uv run ruff format src/ tests/

prep-data:
    uv run python scripts/data/build/prep_dataset.py

merge-data:
    uv run python scripts/archive/publish_dataset.py --dry-run
    uv run arka --config configs/v03/03_sft_data_prep.yaml --run-id v03-sft-prep

publish-dataset repo_id="jayshah5696/humanize-rl-sft-dataset" flags="":
    uv run python scripts/publish_to_hf.py dataset --repo-id {{repo_id}} --path data/processed/v04_sft_final.jsonl --config-name v2 --readme runs/cards/sft_v2.md --artifact runs/v03/domain_distribution_v2.png:domain_distribution_v2.png {{flags}}

build-v2-seeds:
    uv run python scripts/data/seeds/build_v04_seeds.py

fetch-v2-sources flags="":
    uv run python scripts/data/fetch/fetch_v04_sources.py {{flags}}

stream-a-pilot:
    uv run arka --config configs/v04/stream_a_evol_pilot.yaml --run-id v04-stream-a-pilot

# ---- v03 RL tasks pipeline (docs/plans/v03-rl-tasks-dataset.md) ----

v03-fetch-sources:
    uv run scripts/data/fetch/fetch_hf_sources.py --config configs/v03/sources.yaml --out data/raw/v03_writing_seeds.jsonl

v03-run-mode mode author="google/gemini-3.1-flash-lite-preview":
    uv run scripts/data/build/run_arka_pipeline.py --config configs/v03/v03_{{mode}}.yaml --author {{author}} --out data/processed/v03_{{mode}}_$(echo {{author}} | tr '/' '_').jsonl --seed-path data/raw/v03_writing_seeds.jsonl --run-id v03-{{mode}}-$(echo {{author}} | tr '/' '_')

v03-assemble:
    PYTHONPATH=src uv run scripts/data/build/rl_task_assemble.py --inputs "data/processed/v03_*.jsonl" --out data/rl/humanize_tasks_v03.jsonl --split 0.8 0.1 0.1 --split-seed 99 --summary-out data/rl/humanize_tasks_v03_summary.json

v03-rollout-filter model="openai/gpt-5.4-mini" weak="google/gemini-3.1-flash-lite-preview" second="google/gemini-3-flash-preview":
    PYTHONPATH=src uv run scripts/eval/reference_rollout_filter.py --input data/rl/humanize_tasks_v03_judged_kept.jsonl --output data/rl/humanize_tasks_v03_filtered.jsonl --rollouts-out data/rl/v03_ref_rollouts.jsonl --model {{model}} --rollouts-per-task 3 --weak-model {{weak}} --second-strong-model {{second}} --audit-subset 200 --max-workers 16

v03-verify strict="--no-strict":
    PYTHONPATH=src uv run scripts/data/build/verify_rl_task_quality.py --input data/rl/humanize_tasks_v03_filtered.jsonl --rollouts data/rl/v03_ref_rollouts.jsonl --thresholds configs/v03/verification_thresholds.yaml --report-out runs/v03/slice5_verification.md --json-out runs/v03/slice5_verification.json {{strict}}

v03-publish repo_id="jayshah5696/humanize-rl-tasks-v03" flags="":
    uv run python scripts/publish_to_hf.py dataset --repo-id {{repo_id}} --path data/rl/humanize_tasks_v03_filtered.jsonl --readme runs/cards/rl_tasks_v03.md --artifact runs/v03/slice5_verification.md:verification_report.md --artifact runs/v03/slice5_verification.json:verification_report.json --artifact data/rl/humanize_tasks_v03_summary.json:dataset_summary.json {{flags}}

