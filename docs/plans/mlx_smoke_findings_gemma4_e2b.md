# MLX Smoke Findings: Gemma 4 E2B-it

**Date:** 2026-05-24  
**Local hardware:** Apple M3 Pro, 18GB unified memory, macOS 26.5, arm64  
**Dataset:** `data/processed/sft/gemma4_e2b_v04_mlx_smoke`

## Summary

The local MLX smoke slice now succeeds with the correct `mlx-tune` path:

1. The smoke dataset is valid MLX chat JSONL.
2. `mlx-vlm` can load and run `unsloth/gemma-4-E2B-it-UD-MLX-4bit` locally.
3. Local inference fits comfortably, peaking around **4.6GB** memory.
4. Direct `mlx-lm` LoRA training on HF/Unsloth Gemma 4 checkpoints fails on shared-KV loader issues.
5. `mlx-tune` works when we follow its Gemma 4 examples: use `FastVisionModel`, `VLMSFTTrainer`, `UnslothVisionDataCollator`, VLM-style text messages, and the `mlx-community/gemma-4-e2b-it-4bit` checkpoint.

Recommendation: use MLX-Tune for local tiny smoke/pilot runs, but keep Modal + Unsloth as the canonical publishable training path until we have longer MLX runs and export parity verified.

## Commands tried

### 1. Official `mlx-lm` LoRA smoke

```bash
MAX_SEQ_LENGTH=512 NUM_LAYERS=4 ITERS=2 VAL_BATCHES=1 STEPS_PER_EVAL=1 ./scripts/smoke_mlx_gemma4_lora.sh
```

Result: failed during model load with unexpected parameters for shared-KV layers:

```text
ValueError: Received 60 parameters not in model:
language_model.model.layers.15.self_attn.k_norm.weight,
language_model.model.layers.15.self_attn.k_proj.weight,
language_model.model.layers.15.self_attn.v_proj.weight,
...
language_model.model.layers.34.self_attn.v_proj.weight
```

This matches active upstream Gemma 4 MLX issues/PRs around shared KV projections and unused `k_proj`/`v_proj`/`k_norm` parameters.

### 2. `mlx-vlm` inference smoke

```bash
uvx --from mlx-vlm --python 3.12 mlx_vlm.generate \
  --model unsloth/gemma-4-E2B-it-UD-MLX-4bit \
  --prompt 'Write a one-sentence Slack update saying staging is fixed.' \
  --max-tokens 60 \
  --temperature 0.0 \
  --skip-special-tokens
```

Result: success.

Output:

```text
Staging is now fixed.
```

Runtime:

```text
Prompt: 21 tokens, 10.377 tokens/sec
Generation: 7 tokens, 68.083 tokens/sec
Peak memory: 4.593 GB
```

### 3. `mlx-tune` LoRA smoke — corrected path

The issue was our code. We used the generic `SFTTrainer` / text path, but Gemma 4 models are VLMs in `mlx-tune` and must use the VLM training path even for text-only data.

Correct command:

```bash
uvx --from mlx-tune --with datasets --python 3.12 python scripts/smoke_mlx_tune_gemma4_lora.py
```

Correct implementation details:

- model: `mlx-community/gemma-4-e2b-it-4bit`
- loader: `FastVisionModel.from_pretrained(...)`
- adapter: `FastVisionModel.get_peft_model(...)`
- dataset: `messages` with content blocks like `[{"type": "text", "text": "..."}]`
- trainer: `VLMSFTTrainer`
- config: `VLMSFTConfig`
- collator: `UnslothVisionDataCollator(model, processor)`

Result: success on tiny run.

```text
#trainable params: 6.334464 M || all params: 4647.449856 M || trainable%: 0.136%
Training: 2/2
loss=5.2812 -> loss=4.0273
Training complete! Average loss: 4.6543
Adapters saved to: outputs_gemma4_humanize_mlx_smoke/adapters
Adapters saved to adapters/gemma4_e2b_v04_mlx_tune_smoke
```

This proves LoRA plumbing works locally. It does not prove quality yet.

## Interpretation

MLX is viable for:

- local Gemma 4 E2B-it inference;
- checking prompt/chat-template rendering;
- checking that the smoke dataset is loadable;
- tiny LoRA smoke/pilot runs using `mlx-tune` VLM path;
- sanity-checking base vs local adapter behavior before Modal training.

MLX is still not yet the canonical path for:

- publishable training artifacts;
- reliable reproduction of Unsloth/CUDA behavior;
- longer runs without more monitoring and export validation.

## Impact on vertical slice

Slice 2 should be considered **complete as a working local MLX-Tune smoke run**, with this conclusion:

> Local MLX inference works, and local MLX-Tune LoRA training works for a tiny Gemma 4 E2B-it text-only VLM-path smoke run. Proceed to either a larger local MLX pilot or Slice 3 Modal + Unsloth pilot.

## Current local config recommendation

Keep `configs/training/gemma4_e2b_sft_v04.yaml` MLX settings conservative:

```yaml
mlx_smoke:
  inference_model: unsloth/gemma-4-E2B-it-UD-MLX-4bit
  training_model: mlx-community/gemma-4-e2b-it-4bit
  backend: mlx-tune-vlm
  max_steps: 2-50
  batch_size: 1
  grad_accumulation_steps: 4
  effective_batch_size: 4
  max_seq_length: 512
  lora_r: 4 for smoke, 8 or 16 for pilot
```

Use `mlx-tune` for local training smoke. Avoid direct `mlx_lm.lora` for Gemma 4 E2B-it until the shared-KV loader path is fixed for the relevant checkpoints.
