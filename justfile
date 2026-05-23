# justfile for humanize-rl

test:
    PYTHONPATH=src uv run pytest tests/data/test_scorer_dataset.py tests/scoring/test_distilled_scorers.py tests/benchmark/test_evaluate.py


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
    uv run python scripts/prep_dataset.py

merge-data:
    uv run python scripts/publish_dataset.py --dry-run
    uv run arka --config configs/v03/03_sft_data_prep.yaml --run-id v03-sft-prep

publish-dataset repo_id="jayshah5696/humanize-rl-sft-dataset" flags="":
    uv run python scripts/publish_dataset.py --repo-id {{repo_id}} {{flags}}

build-v2-seeds:
    uv run python scripts/build_v04_seeds.py

fetch-v2-sources flags="":
    uv run python scripts/fetch_v04_sources.py {{flags}}

stream-a-pilot:
    uv run arka --config configs/v04/stream_a_evol_pilot.yaml --run-id v04-stream-a-pilot

