# Prime `prime-rl` SFT configs

These configs are for Prime's open `prime-rl` runtime, not Hosted Training
`prime train` env-RL TOMLs in `configs/prime/`.

Use this path first for dataset SFT:

```bash
uv run sft @ configs/prime_rl/qwen35_08b_sft_smoke_env0314.toml \
  --wandb.project humanize-rl \
  --wandb.name qwen35-08b-prime-sft-smoke-env0314
```

Current Qwen 2B target config for the env0315 ablation gate:

```bash
uv run sft @ configs/prime_rl/qwen35_2b_sft_target_messages_env0314_gate_env0315.toml
```

Current S2 repair-data target config:

```bash
uv run sft @ configs/prime_rl/qwen35_2b_sft_target_messages_env0315_clean50_gate_env0315.toml
```

Use S2 for the next quality-oriented Qwen 2B SFT run unless intentionally
spending budget on the S1 no-new-repairs baseline.

Prime auth is separate from W&B tracking. Before launching the Qwen 2B target
run from a Prime sandbox or pod, verify secrets without printing values:

```bash
uv run scripts/train/prime_sft_preflight.py \
  --check-hf-viewer \
  --output runs/prime_sft_preflight/qwen35_2b_env0315.json
```

For the S2 clean50 config:

```bash
uv run scripts/train/prime_sft_preflight.py \
  --config configs/prime_rl/qwen35_2b_sft_target_messages_env0315_clean50_gate_env0315.toml \
  --check-hf-viewer \
  --output runs/prime_sft_preflight/qwen35_2b_env0315_clean50.json
```

The lightweight train helpers declare inline `uv` script dependencies, so this
preflight does not need to build the full local project environment.

If W&B is missing, add it to Prime's secret store from a shell where the env var
is already set:

```bash
prime --plain secret create \
  --name WANDB_API_KEY \
  --value "$WANDB_API_KEY" \
  --description "Weights and Biases tracking token"
```

The tracked launch path is a Linux/CUDA Prime sandbox or pod running the open
`prime-rl` package, not `prime train` Hosted Training. A sandbox launch should
use a CUDA image, upload this repo state, install Prime `prime-rl`, then run:

```bash
uv run sft @ /workspace/humanize-rl/configs/prime_rl/qwen35_2b_sft_target_messages_env0314_gate_env0315.toml
```

For S2:

```bash
uv run sft @ /workspace/humanize-rl/configs/prime_rl/qwen35_2b_sft_target_messages_env0315_clean50_gate_env0315.toml
```

To generate a smaller secret-free sandbox launch kit for the current Qwen 2B
config:

```bash
uv run scripts/train/prepare_prime_sft_launch_kit.py
```

For S2:

```bash
uv run scripts/train/prepare_prime_sft_launch_kit.py \
  --config configs/prime_rl/qwen35_2b_sft_target_messages_env0315_clean50_gate_env0315.toml \
  --output-dir runs/prime_sft_launch_kit/qwen35_2b_env0315_clean50 \
  --eval-manifest runs/prime_sft_promotion/TEMPLATE_QWEN35_2B_ENV0315_CLEAN50/sft_eval_manifest.json
```

S2 post-SFT eval/promotion manifest:

```bash
uv run scripts/eval/build_sft_eval_manifest.py \
  --config configs/prime_rl/qwen35_2b_sft_target_messages_env0315_clean50_gate_env0315.toml \
  --promotion-root runs/prime_sft_promotion/TEMPLATE_QWEN35_2B_ENV0315_CLEAN50 \
  --after-sft-template configs/prime/qwen35_2b_p5050_after_sft_env0315_full200_template.toml \
  --after-sft-config 'configs/prime/qwen35_2b_p5050_after_sft_env0315_clean50_<checkpoint_slug>.toml' \
  --rl-run-name 'humanize-p5050-qwen35-2b-after-sft-env0315-clean50-<checkpoint_slug>' \
  --output runs/prime_sft_promotion/TEMPLATE_QWEN35_2B_ENV0315_CLEAN50/sft_eval_manifest.json
```

When `--checkpoint-id` is a real checkpoint instead of
`READY_SFT_CHECKPOINT_ID`, `<checkpoint_slug>` in the after-SFT config path and
RL run name is replaced automatically with a filesystem-safe slug.

Before spending the S2 launch, verify the config, preflight report, launch kit,
launch archive, runner script, README, and eval manifest still match:

```bash
uv run scripts/train/verify_prime_sft_launch_readiness.py \
  --config configs/prime_rl/qwen35_2b_sft_target_messages_env0315_clean50_gate_env0315.toml \
  --preflight-report runs/prime_sft_preflight/qwen35_2b_env0315_clean50.json \
  --launch-manifest runs/prime_sft_launch_kit/qwen35_2b_env0315_clean50/manifest.json \
  --launch-archive runs/prime_sft_launch_kit/qwen35_2b_env0315_clean50.tar.gz \
  --eval-manifest runs/prime_sft_promotion/TEMPLATE_QWEN35_2B_ENV0315_CLEAN50/sft_eval_manifest.json \
  --output runs/prime_sft_preflight/qwen35_2b_env0315_clean50_launch_readiness.json
```

Current generated paths:

```text
runs/prime_sft_launch_kit/qwen35_2b_env0315/
runs/prime_sft_launch_kit/qwen35_2b_env0315.tar.gz
runs/prime_sft_launch_kit/qwen35_2b_env0315_clean50/
runs/prime_sft_launch_kit/qwen35_2b_env0315_clean50.tar.gz
runs/prime_sft_launch_kit/qwen35_2b_env0315_clean50/sft_eval_manifest.json
runs/prime_sft_preflight/qwen35_2b_env0315_clean50_launch_readiness.json
runs/prime_sft_promotion/TEMPLATE_QWEN35_2B_ENV0315_CLEAN50/sft_eval_manifest.json
```

This is still dataset SFT on:

```text
jayshah5696/humanize-rl-prime-sft-messages-env0314
jayshah5696/humanize-rl-prime-sft-messages-env0315-clean50
```

The `gate_env0315` suffix means the resulting checkpoint must be evaluated
against env `0.3.15`, the rollout audit, and the detector-mimic gate before it
is used as the checkpoint for hosted RL.

If a real Pangram export is collected for the frozen detector rows, compare it
against the local mimic offline:

```bash
uv run scripts/eval/export_detector_mimic_for_pangram.py \
  --input data/eval/detector_mimic_v01.jsonl \
  --output runs/detector_mimic/pangram_bulk_items.json

uv run scripts/eval/compare_detector_mimic_to_pangram.py \
  --input data/eval/detector_mimic_v01.jsonl \
  --pangram-output runs/detector_mimic/pangram_export.json \
  --output runs/detector_mimic/pangram_alignment_report.json
```

After the SFT run finishes and the output directory is available locally or in
the sandbox, verify that the expected `weights/step_N` snapshot and adapter
artifacts exist:

```bash
uv run scripts/train/verify_prime_sft_output.py \
  --config configs/prime_rl/qwen35_2b_sft_target_messages_env0314_gate_env0315.toml \
  --step 200 \
  --output runs/prime_sft_promotion/<READY_SFT_CHECKPOINT_ID>/sft_output_verification.json
```

For S2:

```bash
uv run scripts/train/verify_prime_sft_output.py \
  --config configs/prime_rl/qwen35_2b_sft_target_messages_env0315_clean50_gate_env0315.toml \
  --step 200 \
  --output runs/prime_sft_promotion/<READY_SFT_CHECKPOINT_ID>/sft_output_verification.json
```

The SFT dataset is a dedicated one-file-per-split HF dataset so Prime does not
accidentally load older parquet/JSONL artifacts from the broader SFT repo:

```text
jayshah5696/humanize-rl-prime-sft-messages-env0314
```

Current split counts:

- train: `4313`
- validation: `239`
- test: `241`
- total accepted: `4793`
- upstream rejected by builder: `62`
- duplicate ids: `0`
- repair-reference rows: `20`

S2 clean50 split counts:

- train: `4358`
- validation: `242`
- test: `243`
- total accepted: `4843`
- upstream rejected by builder: `62`
- accepted repair-reference rows: `70`
- HF commit: `e1e6b1e839c0dc0450313e5f558c90a6ac925557`

Important:

- Qwen3.5 configs set `[renderer] name = "qwen3.5"` based on the current
  `renderers` model map.
- These configs are schema-checked locally with `prime-rl-configs`; full
  `uv run sft --dry-run` needs Linux/CUDA because full `prime-rl` depends on
  CUDA torch wheels.
- Keep W&B enabled by TOML or CLI. Never commit W&B keys.
