# Scripts Consolidation and Folder Cleanup Plan

## Goal

Reduce duplicate one-off scripts while keeping the current workflow style:

```bash
uv run scripts/something.py --param value
```

No project-wide command router is needed. Prefer small reusable scripts over many near-duplicate scripts.

## Rules

- Keep individual scripts runnable with `uv run scripts/...`.
- Use Click for new/rewritten script arguments.
- Before adding a new script, check whether a common script can support the task.
- If a task repeats across scripts, extract it into a common reusable script or helper.
- Do not delete historical scripts immediately; move superseded scripts to `scripts/archive/` first.
- Keep behavior-preserving folder moves separate from consolidation changes.

## Phase 1: Add common scripts

### `scripts/publish_to_hf.py`

Generic Hugging Face uploader for datasets, model folders, README files, and extra artifacts.

Examples:

```bash
uv run scripts/publish_to_hf.py dataset \
  --repo-id jayshah5696/humanize-rl-sft-dataset \
  --path data/processed/v04_sft_final.jsonl \
  --config-name v2 \
  --readme runs/cards/sft_v2.md \
  --artifact runs/v03/domain_distribution_v2.png:domain_distribution_v2.png

uv run scripts/publish_to_hf.py dataset \
  --repo-id jayshah5696/humanize-rl-tasks \
  --path data/rl/humanize_tasks_v02.jsonl \
  --readme runs/cards/rl_tasks.md

uv run scripts/publish_to_hf.py model \
  --repo-id jayshah5696/humanize-rl-track-a-ridge-scorer \
  --folder models/track_a_10k \
  --readme runs/cards/track_a_ridge.md \
  --artifact runs/track_a_capped/REPORT.md:reports/track_a_report.md
```

Token lookup order:

1. `HF_TOKEN`
2. `HUGGINGFACE_TOKEN`
3. `HUGGING_FACE_HUB_TOKEN`
4. cached `huggingface-cli login` token

Can replace or archive later:

```text
scripts/archive/publish_dataset.py
scripts/archive/publish_v04_dataset.py
scripts/archive/push_datasets.py
scripts/archive/push_rl_tasks_to_hf.py
scripts/archive/publish_track_a_to_hf.py
scripts/archive/publish_track_a_ridge_to_hf.py
```

Keep dataset-specific formatting/build logic outside this script. This script should upload prepared artifacts.

### `scripts/write_hf_card.py`

Generate Hugging Face dataset/model cards from prepared artifacts.

Examples:

```bash
uv run scripts/write_hf_card.py sft \
  --input data/processed/v04_sft_final.jsonl \
  --output runs/cards/sft_v2.md \
  --version v2

uv run scripts/write_hf_card.py rl-tasks \
  --input data/rl/humanize_tasks_v02.jsonl \
  --output runs/cards/rl_tasks.md

uv run scripts/write_hf_card.py scorer \
  --metadata models/track_a_10k/metadata.json \
  --output runs/cards/track_a_ridge.md
```

This removes large hardcoded README strings from publishing scripts.

### `scripts/jsonl_tool.py`

Reusable JSONL utility for common data operations.

Examples:

```bash
uv run scripts/jsonl_tool.py count data/processed/v04_sft_final.jsonl

uv run scripts/jsonl_tool.py sample \
  --input data/processed/v04_sft_final.jsonl \
  --output data/processed/v04_sample100.jsonl \
  --n 100 \
  --seed 42

uv run scripts/jsonl_tool.py dedupe \
  --input data/l2/raw_labels.jsonl \
  --output data/l2/deduped_labels.jsonl \
  --key id

uv run scripts/jsonl_tool.py merge \
  --output data/processed/combined.jsonl \
  data/a.jsonl data/b.jsonl data/c.jsonl

uv run scripts/jsonl_tool.py split \
  --input data/rl/tasks.jsonl \
  --out-dir data/rl/splits \
  --train 0.8 --valid 0.1 --test 0.1
```

Can replace or reduce later:

```text
scripts/data/build/dedupe_l2_labels.py
scripts/data/seeds/duplicate_seeds.py
scripts/data/build/merge_v04_kept.py
scripts/data/seeds/make_v04_base_sample.py
scripts/data/seeds/make_v04_base100.py
```

## Phase 2: Optional common figure script

Only add this if figure duplication keeps growing.

```bash
uv run scripts/create_figure.py domain-distribution \
  --input data/processed/v04_sft_final.jsonl \
  --field domain \
  --output runs/v03/domain_distribution_v2.png
```

Potentially reduces:

```text
scripts/figures/create_sft_figures.py
scripts/figures/create_sft_v2_figures.py
scripts/figures/create_public_scorer_figures.py
scripts/figures/create_track_a_summary_figures.py
scripts/figures/generate_plots.py
```

## Phase 3: Folder cleanup

After common scripts exist, group remaining scripts by workflow:

```text
scripts/
  README.md
  publish_to_hf.py
  write_hf_card.py
  jsonl_tool.py

  data/
    seeds/
    fetch/
    build/

  label/
  eval/
  train/
  figures/
  mlx/
  rl/
  archive/
```

Suggested placement:

```text
scripts/data/seeds/
  build_expansion_seeds.py
  build_v04_seeds.py
  build_v04_chat_seeds.py
  build_v04_synthetic_seeds.py
  build_walking_skeleton_seeds.py
  duplicate_seeds.py
  make_v04_base_sample.py
  make_v04_base100.py

scripts/data/fetch/
  fetch_v04_sources.py
  fetch_v04_stream_b.py

scripts/data/build/
  prep_dataset.py
  build_gemma4_sft_dataset.py
  build_l2_labeling_pool.py
  build_false_positive_set.py
  build_rl_tasks_v01.py
  build_rl_tasks_v02.py
  merge_v04_kept.py
  merge_l2_labels_into_scorer_data.py
  dedupe_l2_labels.py

scripts/label/
  label_l2_pool.py
  judge_v04_quality.py
  verify_v04_quality.py

scripts/eval/
  audit_track_a_cross_source.py
  audit_track_a_scorer.py
  audit_track_a_scorer_fast.py
  diagnose_scorer_shortcuts.py
  evaluate_false_positive_set.py
  evaluate_reward_env.py
  evaluate_mlx_gemma4_outputs.py
  check_rl_run_summary.py
  run_track_a_capped_report.py
  run_track_a_full_report.py
  run_vf_eval_modal.py

scripts/train/
  train_final_track_a_models.py
  train_mlx_gemma4_local.py
  train_mlx_gemma4_local_logged.py
  smoke_mlx_tune_gemma4_lora.py
  smoke_mlx_gemma4_lora.sh
  run_expansion_pipeline.sh

scripts/mlx/
  save_mlx_gemma4_loaded_model.py
  save_mlx_gemma4_merged.py
  save_mlx_gemma4_merged_quant.py
  quantize_mlx_merged_linears.py
  resave_mlx_format_metadata.py
  strip_mlx_unloadable_biases.py

scripts/rl/
  run_reward_env.py
  run_rl_rollouts.py
  project_full_run_cost.py

scripts/figures/
  create_modal_sft_report_figures.py
  create_model_internals_diagram.py
  create_public_scorer_figures.py
  create_scorer_architecture_diagram.py
  create_sft_figures.py
  create_sft_loss_curve_from_wandb.py
  create_sft_v2_figures.py
  create_track_a_summary_figures.py
  generate_plots.py
```

## Phase 4: Update references

After moving files:

- update `justfile`
- update docs with script paths
- update README examples
- check shell scripts that call other scripts
- run `rtk grep "scripts/" .` and fix stale paths

## Archive candidates

Move to `scripts/archive/` only after a common replacement exists:

```text
publish_dataset.py
publish_v04_dataset.py
push_datasets.py
push_rl_tasks_to_hf.py
publish_track_a_to_hf.py
publish_track_a_ridge_to_hf.py
train_mlx_gemma4_local.py.tmp
```

Potentially archive later after confirmation:

```text
build_rl_tasks_v01.py
audit_track_a_scorer.py
```
