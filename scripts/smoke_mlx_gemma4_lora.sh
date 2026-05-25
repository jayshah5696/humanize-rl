#!/usr/bin/env bash
set -euo pipefail

# Local-only MLX smoke test for Gemma 4 E2B LoRA.
# This validates dataset format + adapter plumbing; it is not a quality training run.

DATA_DIR="${DATA_DIR:-data/processed/sft/gemma4_e2b_v04_mlx_smoke}"
ADAPTER_PATH="${ADAPTER_PATH:-adapters/gemma4_e2b_v04_mlx_smoke}"
MODEL="${MODEL:-unsloth/gemma-4-E2B-it}"
ITERS="${ITERS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "MLX smoke run requires Apple Silicon arm64. Detected: $(uname -m)" >&2
  exit 1
fi

if [[ ! -f "${DATA_DIR}/train.jsonl" ]]; then
  echo "Missing ${DATA_DIR}/train.jsonl. Run scripts/build_gemma4_sft_dataset.py first." >&2
  exit 1
fi

MEM_BYTES="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
MEM_GB="$((MEM_BYTES / 1024 / 1024 / 1024))"
echo "Detected Apple Silicon local hardware: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown), ${MEM_GB}GB unified memory"
echo "MLX config: model=${MODEL} data=${DATA_DIR} iters=${ITERS} batch=${BATCH_SIZE} grad_accum=${GRAD_ACCUM} effective_batch=$((BATCH_SIZE * GRAD_ACCUM))"

if [[ "${MEM_GB}" -lt 24 ]]; then
  echo "Note: <24GB unified memory detected. Keeping this to a tiny smoke run is intentional." >&2
fi

uvx --from 'mlx-lm[train]' --python 3.12 mlx_lm.lora \
  --model "${MODEL}" \
  --train \
  --data "${DATA_DIR}" \
  --iters "${ITERS}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accumulation-steps "${GRAD_ACCUM}" \
  --learning-rate "${LEARNING_RATE:-2e-4}" \
  --max-seq-length "${MAX_SEQ_LENGTH:-2048}" \
  --num-layers "${NUM_LAYERS:-8}" \
  --steps-per-report "${STEPS_PER_REPORT:-1}" \
  --steps-per-eval "${STEPS_PER_EVAL:-25}" \
  --val-batches "${VAL_BATCHES:-5}" \
  --grad-checkpoint \
  --mask-prompt \
  --adapter-path "${ADAPTER_PATH}"

echo "MLX smoke adapter written to ${ADAPTER_PATH}"
