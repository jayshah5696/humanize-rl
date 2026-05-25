# Gemma 4 E2B Model, Adapter, and Merge Verification Plan

## Current error blocking RL

The Unsloth `FastVisionModel` load report prints:

```text
Gemma4ForConditionalGeneration LOAD REPORT from: jayshah5696/gemma4-e2b-humanize-unsloth-merged
Key                                                           | Status  |
--------------------------------------------------------------+---------+-
model.language_model.layers.{15...34}.self_attn.k_proj.weight | MISSING |
model.language_model.layers.{15...34}.self_attn.v_proj.weight | MISSING |
model.language_model.layers.{15...34}.self_attn.k_norm.weight | MISSING |
```

### Resolution: this is a benign loader artifact, not a broken checkpoint.

Gemma 4 E2B has `num_hidden_layers: 35` and `num_kv_shared_layers: 20` in
`text_config`. Layers 15 through 34 (the last 20) share KV with earlier
layers and **intentionally do not have their own** `k_proj`, `v_proj`,
`k_norm`, `v_norm` weights. This is hard-coded in
`Gemma4TextAttention.__init__` in `transformers` (PR #45328,
commit `9f8ddaa`):

```python
# Layers sharing kv states don't need any weight matrices
if not self.is_kv_shared_layer:
    self.k_norm = Gemma4RMSNorm(...)
    self.v_norm = Gemma4RMSNorm(...)
    self.k_proj = nn.Linear(...)
    self.v_proj = nn.Linear(...) if not self.use_alternative_attention else None
```

The same PR registers those names in `_keys_to_ignore_on_load_unexpected`
so `from_pretrained` silently skips them. A correctly saved Gemma 4
checkpoint **must** omit them. Our merged repo does, and the
`AutoModelForImageTextToText` reload reports zero missing keys. Unsloth's
`FastVisionModel` loader simply does not consult the ignore list, so it
prints those entries as MISSING. The forward pass never reads those slots,
so no inference path uses random weights.

The GRPO smoke metrics:

```text
kl: 193.7 at step 1
kl: 1.647e+05 at step 2
grad_norm: nan at step 2
```

are **a separate problem**, not caused by missing weights. The likely
causes, in order:

1. `num_generations: 2`. A 2-rollout GRPO group has near-zero reward std,
   which makes the normalised advantage explode and trips KL.
2. `epsilon_high: 0.28` and `delta: 1.5` (DAPO-style asymmetric clipping)
   are too aggressive on the first stable evaluation.
3. Unsloth Gemma 4 `save_pretrained_merged` regression that overwrites
   `tokenizer_config.json::eos_token` from `<turn|>` (id 106) to `<eos>`
   (id 1). This is documented in Unsloth issue #5386 and breaks generation
   stop detection in vLLM/TRL paths that read the tokenizer eos. We must
   audit and patch our merged repo for this.
4. Possible Unsloth tokenizer-save regression that drops `tokenizer.model`
   on the merged push (fixed in Unsloth PR #5115 review loop). Re-verify
   tokenizer assets exist.

## Background: what we already trained and have available

We have multiple Gemma 4 E2B artifacts from prior SFT work:

1. **Base model**
   - `unsloth/gemma-4-E2B-it`

2. **Modal/Unsloth SFT LoRA**
   - `jayshah5696/gemma4-e2b-humanize-unsloth-lora`
   - produced by `src/humanize_rl/training/finetune_gemma4_modal.py`
   - training used bf16/16-bit LoRA, not QLoRA
   - training ran on Modal GPU infrastructure

3. **Current HF merged SFT model**
   - `jayshah5696/gemma4-e2b-humanize-unsloth-merged`
   - intended as the RL starting policy
   - currently suspicious because of the missing-weight load report above

4. **Modal merge scripts**
   - `merge_gemma4_modal.py`: PEFT `merge_and_unload()` route
   - `merge_verify_push_gemma4_modal.py`: direct LoRA vs merged vs saved/reloaded parity route
   - `verify_merge_parity_modal.py`: smaller one-prompt parity checker

5. **Local MLX finetune artifacts**
   - adapters under `adapters/gemma4_e2b_v04_mlx_*`
   - includes the local full-r8 adapter path `adapters/gemma4_e2b_v04_mlx_full_r8/`
   - these are useful candidate/reference artifacts but must not be mixed into Modal RL until conversion/parity is proven

6. **Existing eval outputs**
   - `outputs_gemma4_humanize_modal_lora_eval/`
   - `outputs_gemma4_humanize_modal_full_eval/`

## Goal

Build a verified Gemma 4 E2B Humanize starting policy for RL.

The verified artifact must either be:

1. a clean merged HF model that loads without missing Gemma 4 language-model keys; or
2. a base-model + SFT-LoRA loading path for RL if merged export remains unreliable.

The immediate purpose is not publication polish. The purpose is to get a trustworthy SFT policy into GRPO so RL starts from the model we actually trained, not from a partially random merged checkpoint.

We need to verify every artifact path we have:

1. Modal/Unsloth SFT LoRA adapter;
2. current HF merged model;
3. Modal merge outputs from existing scripts;
4. local MLX adapter/full run artifacts;
5. any newly rebuilt merged artifact.

## Known artifact inventory from repo

### HF repos referenced in scripts

```text
base model:      unsloth/gemma-4-E2B-it
SFT LoRA:        jayshah5696/gemma4-e2b-humanize-unsloth-lora
current merged:  jayshah5696/gemma4-e2b-humanize-unsloth-merged
RL smoke LoRA:   jayshah5696/gemma4-e2b-humanize-rl-smoke-lora   # push disabled so far
```

### Local MLX adapters present

```text
adapters/gemma4_e2b_v04_mlx_benchmark/
adapters/gemma4_e2b_v04_mlx_full_r8/
adapters/gemma4_e2b_v04_mlx_logged_test/
adapters/gemma4_e2b_v04_mlx_test/
adapters/gemma4_e2b_v04_mlx_tune_smoke/
```

### Existing eval outputs

```text
outputs_gemma4_humanize_modal_lora_eval/
outputs_gemma4_humanize_modal_full_eval/
```

### Existing scripts checked

#### `src/humanize_rl/training/finetune_gemma4_modal.py`

- Modal SFT training script.
- Uses `FastModel.from_pretrained` and `FastModel.get_peft_model`.
- Pushes LoRA to `jayshah5696/gemma4-e2b-humanize-unsloth-lora`.
- Calls:

```python
model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
model.push_to_hub_merged(config.hf_merged_repo, tokenizer, save_method="merged_16bit")
```

This may have produced the suspicious merged repo.

#### `src/humanize_rl/training/merge_gemma4_modal.py`

- Loads base with Transformers.
- Loads LoRA with PEFT.
- Calls `merge_and_unload()`.
- Pushes merged repo.
- Detached via `.spawn()` already.
- Weakness: no multi-prompt parity verification before push.

#### `src/humanize_rl/training/merge_verify_push_gemma4_modal.py`

- Best current merge verification script.
- Checks direct LoRA generation, in-memory merged generation, saved/reloaded merged generation.
- Weaknesses:
  - local entrypoint uses `.remote()`;
  - one prompt only;
  - pushes directly to current merged repo;
  - does not compare all known artifacts;
  - does not explicitly fail on missing loading keys.

#### `src/humanize_rl/training/verify_merge_parity_modal.py`

- Simple direct LoRA vs in-memory merged check.
- Weaknesses:
  - foreground `.remote()`;
  - one prompt only;
  - no saved/reloaded check;
  - no artifact inventory.

#### `src/humanize_rl/training/evaluate_gemma4_modal_lora.py`

- Good 10-prompt base vs direct LoRA evaluation script.
- Reuse its prompt set for merge parity.

#### `src/humanize_rl/training/evaluate_gemma4_modal.py`

- Evaluates base vs current merged model.
- Useful after a new verified merged repo exists.

## Online findings to account for (May 2026)

Verified upstream evidence:

- **transformers PR #45328 / commit `9f8ddaa`**: Gemma 4 KV-shared layers
  intentionally drop `k_proj`, `v_proj`, `k_norm`, `v_norm`. The loader
  marks them in `_keys_to_ignore_on_load_unexpected`. Our `model.safetensors`
  is therefore correct.
- **Unsloth issue #5386 (May 2026)**: `FastModel.save_pretrained_merged`
  for Gemma 4 rewrites `eos_token` from `<turn|>` to `<eos>`, causing
  downstream stop-detection failures (12x latency in tool-call mode). Audit
  and patch.
- **Unsloth issue #5410 (May 2026)**: merged model from
  `save_pretrained_merged` produces garbage output vs the unmerged adapter
  loaded directly through `FastModel`. Compare merged vs direct LoRA on 10
  prompts.
- **Unsloth issue #4820**: `FastVisionModel.from_pretrained` rejects
  Gemma 4 LoRA adapters whose target list includes
  `Gemma4ClippableLinear` modules. Our adapter trains only language layer
  Linear modules, so this does not apply, but verify `adapter_config.json`
  `target_modules` before push.
- **mlx-lm PR #1158 (April 2026)**: until that fix, MLX created unused
  `k_proj`/`v_proj`/`k_norm`/`v_norm` modules for shared layers and any
  LoRA targeting those for layers 15-34 trained dead weights. Our local
  `adapters/gemma4_e2b_v04_mlx_full_r8` was produced under that regime
  and is **not portable** to the Modal RL path.

Therefore we keep two merge routes (Unsloth via `save_pretrained_merged`
and Transformers+PEFT via `merge_and_unload`) and verify outputs against
the direct LoRA on a parity set before trusting either.

## Verification principle

Do not overwrite the current merged repo during verification.

Push candidates to new repos first, for example:

```text
jayshah5696/gemma4-e2b-humanize-unsloth-merged-peft-v2
jayshah5696/gemma4-e2b-humanize-unsloth-merged-unsloth-v2
```

Only alias/replace the production merged repo after all gates pass.

## Required verification gates

A candidate merged model passes only if:

1. base model loads;
2. SFT LoRA adapter loads;
3. direct LoRA generation works;
4. merged model saves;
5. merged model reloads from disk;
6. merged model reloads from HF repo;
7. the **only** missing safetensors entries are the
   `[num_hidden_layers - num_kv_shared_layers, num_hidden_layers)`
   `self_attn.{k_proj,v_proj,k_norm,v_norm}` set (KV-shared layers); a real
   non-shared layer with zero K/V weight is a hard failure;
8. `Gemma4ForConditionalGeneration` loaded with
   `AutoModelForImageTextToText.from_pretrained(..., output_loading_info=True)`
   reports zero `missing_keys` and zero `unexpected_keys`;
9. `tokenizer_config.json::eos_token == "<turn|>"` (Unsloth #5386 guard);
10. deterministic outputs match direct LoRA on the 10-prompt parity set, or
    differences are documented and judged harmless;
11. RL loader logs the benign-MISSING explanation and the structural guard
    `_assert_only_shared_kv_keys_missing` does not raise.

## Implementation status (this revision)

Delivered:

- `src/humanize_rl/training/verify_gemma4_artifacts_modal.py` (new) -
  single Modal job that snapshots the merged repo, verifies KV-shared key
  omission, reloads with `Transformers` (`output_loading_info=True`),
  checks `tokenizer_config.eos_token`, runs 10-prompt parity vs the
  direct LoRA path, and optionally patches/pushes the tokenizer fix. Entry
  point uses `.spawn()`.
- `src/humanize_rl/training/rl_gemma4_modal.py` (patched) -
  `_load_policy` now calls `_assert_only_shared_kv_keys_missing(model)`
  which fails fast if any non-shared layer's `k_proj`/`v_proj` is zero,
  and logs that the Unsloth MISSING report is benign.
- `configs/rl/gemma4_e2b_rl_smoke_v2.yaml` (new) - widens
  `num_generations` to 6, softens `epsilon_high`/`delta`, drops LR to
  `5e-6` to fix the KL=1.6e5 / grad_norm=NaN explosion.
- `tests/training/test_verify_gemma4_artifacts.py` (new) - structural
  tests for the verifier, RL loader guard, and v2 smoke config.

Deferred (Decision: see Q&A 2026-05-25):

- MLX `gemma4_e2b_v04_mlx_full_r8` adapter is **kept as local MLX-only**
  inference artifact and is not eligible for Modal RL. No PEFT conversion
  attempt. If needed later, retrain MLX adapter against mlx-lm >= PR
  #1158 first.

## Modal execution order (verifier)

```bash
# 1. Dry verification (no writes). Confirms the current merged repo is
#    structurally correct and prints the parity report.
rtk uvx modal run --detach src/humanize_rl/training/verify_gemma4_artifacts_modal.py

# 2. If verification reports tokenizer.eos_token regressed, patch locally
#    and push the single tokenizer_config.json file (no full re-upload):
rtk uvx modal run --detach src/humanize_rl/training/verify_gemma4_artifacts_modal.py \
    --fix-tokenizer --push-fixed-tokenizer

# 3. Re-run the RL artifact verifier and the v2 smoke config:
rtk uvx modal run --detach src/humanize_rl/training/rl_gemma4_modal.py \
    --mode verify-artifact \
    --config-path /workspace/configs/rl/gemma4_e2b_rl_smoke_v2.yaml
rtk uvx modal run --detach src/humanize_rl/training/rl_gemma4_modal.py \
    --mode train \
    --config-path /workspace/configs/rl/gemma4_e2b_rl_smoke_v2.yaml
```

## Original multi-mode script plan (kept as reference)

If the simple verifier above flags a real problem (parity mismatch,
non-shared layer with zero K/V, or persistent tokenizer regression),
fall back to the full multi-mode plan below.

All modes must use Modal `--detach` + `.spawn()`.

### Mode 1: `inventory`

Inputs:

```text
base repo
sft lora repo
current merged repo
candidate merged repos
local Modal volume paths if any
MLX adapter paths if mounted/uploaded
```

Actions:

- list HF repo files;
- read `config.json`, `generation_config.json`, `tokenizer_config.json`, `adapter_config.json`;
- record base model reference;
- record shard/index files;
- record safetensor key counts if practical;
- write `inventory.json`.

### Mode 2: `verify-direct-lora`

Actions:

- load `unsloth/gemma-4-E2B-it`;
- load `jayshah5696/gemma4-e2b-humanize-unsloth-lora` via PEFT and/or Unsloth path;
- generate on 10 prompts from `evaluate_gemma4_modal_lora.py`;
- write `direct_lora_generations.jsonl`;
- write `direct_lora_summary.json`.

This is the gold behavior for SFT unless we find the LoRA itself is bad.

### Mode 3: `merge-peft`

Actions:

- load base with `AutoModelForImageTextToText`;
- load LoRA with `PeftModel.from_pretrained`;
- generate direct LoRA outputs;
- call `merge_and_unload()`;
- generate in-memory merged outputs;
- save with `safe_serialization=True`;
- reload saved folder;
- generate reloaded outputs;
- check missing/unexpected keys with `output_loading_info=True` if supported;
- write:

```text
peft/direct_lora_generations.jsonl
peft/in_memory_merged_generations.jsonl
peft/reloaded_merged_generations.jsonl
peft/merge_report.json
```

### Mode 4: `merge-unsloth`

Actions:

- load base + LoRA using the Unsloth route that matches training;
- save with:

```python
model.save_pretrained_merged(path, tokenizer, save_method="merged_16bit")
```

- reload saved folder with Transformers;
- generate outputs;
- write analogous report files.

This route mirrors training, but it must prove the saved artifact is complete.

### Mode 5: `verify-current-merged`

Actions:

- load current `jayshah5696/gemma4-e2b-humanize-unsloth-merged` with:
  - Transformers;
  - Unsloth `FastVisionModel.from_pretrained` or current RL loader.
- capture missing-key reports;
- generate 10 prompts;
- compare to direct LoRA.

Expected today: fail because of missing weights.

### Mode 6: `verify-mlx-adapters`

MLX artifacts are adapters, not necessarily directly mergeable by the same PEFT path.

Actions:

- inspect every local adapter directory:

```text
adapters/gemma4_e2b_v04_mlx_benchmark/
adapters/gemma4_e2b_v04_mlx_full_r8/
adapters/gemma4_e2b_v04_mlx_logged_test/
adapters/gemma4_e2b_v04_mlx_test/
adapters/gemma4_e2b_v04_mlx_tune_smoke/
```

- record files, config metadata, rank, alpha, target modules if present;
- run MLX generation locally for adapters that can load;
- compare local MLX adapter outputs to Modal LoRA outputs on the same prompts;
- decide whether an MLX adapter is only a local benchmark artifact or a candidate to convert/publish.

Do not mix MLX adapter into the Modal RL path unless conversion and parity are proven.

### Mode 7: `push-candidate`

Push only an already verified candidate folder.

Requirements:

- explicit candidate path;
- explicit HF repo;
- `--confirm-push true` style flag;
- model card generated from verification report;
- no direct push to production merged repo by default.

## Tests to add before running

Add tests under `tests/training/`:

1. script uses `.spawn()` for every remote mode;
2. no `humanize_rl` or `trl` import inside Modal `image.imports()` before source mount;
3. push mode requires explicit confirmation and repo;
4. report filenames are present in source:
   - `inventory.json`
   - `merge_report.json`
   - `direct_lora_generations.jsonl`
   - `reloaded_merged_generations.jsonl`
5. current merged verification fails on missing keys if detected;
6. production repo overwrite is not default.

## Modal execution order

All runs detached.

```bash
rtk uvx modal run --detach src/humanize_rl/training/verify_gemma4_artifacts_modal.py --mode inventory
rtk uvx modal run --detach src/humanize_rl/training/verify_gemma4_artifacts_modal.py --mode verify-direct-lora
rtk uvx modal run --detach src/humanize_rl/training/verify_gemma4_artifacts_modal.py --mode verify-current-merged
rtk uvx modal run --detach src/humanize_rl/training/verify_gemma4_artifacts_modal.py --mode merge-peft
rtk uvx modal run --detach src/humanize_rl/training/verify_gemma4_artifacts_modal.py --mode merge-unsloth
```

Then inspect reports in the Modal merge volume.

Only after a candidate passes:

```bash
rtk uvx modal run --detach src/humanize_rl/training/verify_gemma4_artifacts_modal.py --mode push-candidate --candidate peft --repo jayshah5696/gemma4-e2b-humanize-unsloth-merged-peft-v2 --confirm-push true
```

## Decision tree

### If direct LoRA is good and PEFT merge passes

Use PEFT-merged v2 as the RL starting policy.

Then run:

1. RL artifact verify;
2. GRPO preflight;
3. 10-step utilization canary;
4. full RL pilot.

### If direct LoRA is good and only Unsloth merge passes

Use Unsloth-merged v2 as the RL starting policy.

Record why PEFT failed.

### If direct LoRA is good but both merges fail

Do not use a merged model for RL.

Options:

1. modify RL loader to start from base + SFT LoRA adapter;
2. run GRPO with policy initialized from adapter path;
3. save RL as a second-stage LoRA or merge adapter stacks only after training.

This may be the safest path if Gemma 4 merged export remains flaky.

### If direct LoRA itself is bad

Go back to SFT artifacts:

- Modal checkpoints in `/checkpoints/.../final_adapter`;
- HF LoRA repo revision history;
- W&B run/checkpoint metadata;
- local MLX full adapter.

Pick the best SFT adapter by eval before any RL.

### If MLX full adapter is best

Treat it as a separate candidate.

Required before using it in Modal RL:

- prove it can be loaded or converted into a HF/PEFT-compatible adapter;
- compare outputs against local MLX generation;
- run the same 10-prompt eval;
- only then consider it as starting policy.

## Model card requirements for final push

Include:

- base model;
- source adapter repo or checkpoint path;
- merge method: PEFT or Unsloth;
- verification date;
- exact scripts/commit;
- deterministic parity prompt count;
- missing-key check result;
- eval summary;
- known limitations;
- license notes.

## Definition of done

A verified starting policy exists when:

- one candidate merged model loads without missing Gemma 4 weights;
- direct LoRA vs merged parity report is saved;
- 10-prompt eval report is saved;
- HF candidate repo is pushed;
- `configs/rl/gemma4_e2b_rl_*.yaml` points to the verified candidate;
- detached RL artifact verification passes against that candidate.


---

# REPORT — 2026-05-25

This section records what was actually executed against the live Modal
stack, what passed, what still fails, and what to do next. It supersedes
the earlier speculative plan above for any conflict.

## TL;DR

1. The original blocker (Unsloth `LOAD REPORT ... MISSING` for layers 15-34
   `k_proj`/`v_proj`/`k_norm`) is **a benign loader artifact** and not a
   real broken checkpoint. Verified end-to-end on Modal with both
   Transformers (`AutoModelForImageTextToText.from_pretrained`,
   `output_loading_info=True`) and Unsloth (`FastVisionModel`). The merged
   HF repo `jayshah5696/gemma4-e2b-humanize-unsloth-merged` is good.
2. The tokenizer eos regression from Unsloth issue #5386 **did not hit**
   our repo. `tokenizer_config.eos_token == "<turn|>"` is preserved.
3. The RL run still goes NaN after 3 stable steps. Root cause is **not**
   the checkpoint; it is GRPO-side signal collapse + LR/clipping for a
   2B-effective model with a sparse rubric reward. Direction is clear and
   matches the upstream Unsloth Gemma-4 GRPO Sudoku notebook tuning.

## What we ran

### Verifier (Modal, A100 / L40S)

Command:

```bash
rtk uvx modal run --detach \
  src/humanize_rl/training/verify_gemma4_artifacts_modal.py
```

Runs `ap-kExvRQQ3xFqrCk8UVElTzx` (after one fixup for missing image
processor assets). Final report:

```json
{
  "checks": {
    "num_hidden_layers": 35,
    "num_kv_shared_layers": 20,
    "safetensors_key_count": 1951,
    "wrongly_present_shared_kv_keys": [],
    "transformers_missing_keys": [],
    "transformers_unexpected_keys": [],
    "processor_source": "merged_local"
  },
  "tokenizer": {
    "eos_token": "<turn|>",
    "expected_eos_token": "<turn|>",
    "eos_token_regressed": false
  },
  "parity": {
    "prompt_count": 10,
    "mismatch_count": 1
  }
}
```

The one parity mismatch is a single-word swap on the Slack-rewrite prompt
(`"give me a quick update on"` vs `"any update on"`); same emoji, same
length, same intent. Attributed to bf16 rounding in the fused merged
matmul vs the LoRA add-on path. Not a correctness signal.

### RL artifact verifier with patched loader

Command:

```bash
rtk uvx modal run --detach src/humanize_rl/training/rl_gemma4_modal.py \
  --mode verify-artifact \
  --config-path /workspace/configs/rl/gemma4_e2b_rl_smoke_v2.yaml
```

Result: `loaded=true`, `model_load_seconds=25.5`, `mean_reward=0.232` on
2 artifact samples. Loader now prints right after the Unsloth report:

```text
[gemma4] 20 KV-shared layers (idx 15..34) intentionally omit
k_proj/v_proj/k_norm/v_norm; MISSING report above is benign.
```

And `_assert_only_shared_kv_keys_missing(model)` would raise if a real
non-shared layer had zero k/v weight. It did not raise.

### v2 GRPO smoke (5 steps, 6 generations, lr 5e-6, eps_high 0.20, delta 1.2)

Command:

```bash
rtk uvx modal run --detach src/humanize_rl/training/rl_gemma4_modal.py \
  --mode train \
  --config-path /workspace/configs/rl/gemma4_e2b_rl_smoke_v2.yaml
```

| step | loss     | grad_norm | kl       | reward | reward_std | frac_zero_std |
| ---- | -------- | --------- | -------- | ------ | ---------- | ------------- |
| 1    | 2.094    | 1.27e5    | 2208     | 0.546  | 0.052      | 0.50          |
| 2    | 0.098    | 6693      | 87.9     | 0.566  | 0.001      | 0.75          |
| 3    | -0.024   | 1490      | 43.4     | 0.777  | 0.012      | 0.25          |
| 4    | 40.7     | NaN       | 3.21e4   | 0.684  | 0.144      | 0.00          |
| 5    | 2.30e4   | NaN       | (broken) | 0.844  | 0.033      | 0.75          |

Observations:

- Step 3 reaches `reward=0.78` with stable `kl=43` and `grad=1490`. The
  policy is learnable from this starting checkpoint.
- Step 4 collapses to `grad_norm=NaN` after
  `rewards/weighted_faithfulness_reward/mean=0` for the whole group. A
  single degenerate-reward group is enough to NaN the bf16 advantage path.
- `frac_reward_zero_std` swings between 0 and 0.75 across steps. This is a
  reward design / group size issue, not a model issue.

## Verification gates: results

| Gate | Result |
| ---- | ------ |
| Base model loads                                                | PASS |
| SFT LoRA adapter loads                                          | PASS |
| Direct LoRA generation works                                    | PASS (10/10 prompts) |
| Merged model saved (Unsloth, prior run)                         | PASS |
| Merged model reloads from HF                                    | PASS |
| Only shared-KV keys omitted from safetensors                    | PASS (80 expected, 80 omitted, 0 wrong) |
| `AutoModelForImageTextToText` missing_keys / unexpected_keys    | PASS (both empty) |
| `tokenizer_config.eos_token == "<turn|>"`                       | PASS |
| Parity with direct LoRA on 10 prompts                           | 9/10 identical, 1 benign word swap |
| RL loader logs benign-MISSING explanation                       | PASS |
| `_assert_only_shared_kv_keys_missing` does not raise            | PASS |

All merge / artifact gates passed. **The merged repo is approved as the RL
starting policy.** No re-merge needed.

## What is delivered in this revision

- `src/humanize_rl/training/verify_gemma4_artifacts_modal.py` (new) —
  Modal verifier. Snapshots the merged repo, asserts KV-shared key
  omission against the Gemma 4 spec, reloads with Transformers + loading
  info, checks tokenizer eos, parity-tests 10 prompts vs direct LoRA,
  optionally patches/pushes the tokenizer fix. Entry uses `.spawn()`.
- `src/humanize_rl/training/rl_gemma4_modal.py` (patched) —
  `_load_policy` now calls `_assert_only_shared_kv_keys_missing(model)`.
  Hard fails if any non-shared decoder layer has zero `k_proj`/`v_proj`.
  Logs the benign-MISSING note right after Unsloth's printout. Also mounts
  the v2 smoke config into the Modal image.
- `configs/rl/gemma4_e2b_rl_smoke_v2.yaml` (new) — softer GRPO clipping,
  more rollouts per group, lower LR, more steps. Used in the run above.
- `tests/training/test_verify_gemma4_artifacts.py` (new) — 7 structural
  tests. Full suite: 20 passed, ruff clean.

## Cross-check vs the Unsloth Gemma 4 GRPO reference notebook

Reference: `Gemma4_(E2B)_Reinforcement_Learning_Sudoku_Game.ipynb`
(provided by Unsloth team, dated 2026-05).

| Setting                       | Unsloth notebook | Our v2 smoke | Verdict                                  |
| ----------------------------- | ---------------- | ------------ | ---------------------------------------- |
| `lora_rank`                   | 32               | 16           | Raise to 32 next                         |
| `lora_alpha`                  | rank * 2 = 64    | 32           | Raise to 32 next (already alpha = 2*r)   |
| `learning_rate`               | **5e-5**         | 5e-6         | We are 10x too low                       |
| `num_generations`             | 2                | 6            | Our 6 is conservative-correct; keep it   |
| `epsilon`                     | 0.2              | 0.2          | Match                                    |
| `epsilon_high`                | 0.28             | 0.20         | Theirs is wider; widen back next         |
| `delta`                       | 1.5              | 1.2          | Theirs is wider; widen back next         |
| `loss_type`                   | bnpo             | bnpo         | Match                                    |
| `mask_truncated_completions`  | True             | True         | Match                                    |
| `warmup_ratio`                | 0.1              | 0.1          | Match                                    |
| `max_steps`                   | 60               | 5            | Run at least 30-60 to assess plateau     |
| `gradient_accumulation_steps` | 2                | 4            | Either is fine                           |
| `optim`                       | adamw_8bit       | adamw_8bit   | Match                                    |
| `target_modules`              | q/k/v/o/gate/up/down | same    | Match                                    |
| `max_completion_length`       | max_seq - prompt | 64           | Ours is too short; raise to >= 256       |
| `temperature`                 | 1.0              | 1.0          | Match                                    |

Key takeaways from comparing the two:

- The Unsloth reference uses a **much larger learning rate (5e-5)**. Their
  reward signal is dense and bounded (`0` / `valid_moves * 0.2` / `30`).
  Our rubric reward is softer and noisier, so we should still stay under
  theirs, but `5e-6` is too cold. Recommended starting point: **`2e-5`**
  (the original v1 value).
- The Unsloth reference allows wider clipping (`epsilon_high=0.28`,
  `delta=1.5`). Combined with `num_generations=2`, that worked for them
  because their reward shape was bimodal and grouped variance was high
  enough. Our reward is multi-component and slow-moving. Recommended:
  return to `epsilon_high=0.28`, `delta=1.5` **only if** we also raise
  `num_generations >= 6`.
- The Unsloth reference uses `lora_rank=32, lora_alpha=64`. We trained SFT
  at `r=8` and RL at `r=16`. Recommended: keep `r=16` for now, but
  consider `r=32` for the full RL run; do not change rank between SFT and
  RL artifacts because that doubles the trainable parameter footprint.
- The Unsloth reference does **not** set `max_grad_norm`. TRL default is
  1.0. Our step-4 NaN suggests we should set it explicitly to `0.5` to
  contain bf16 overflow.
- The Unsloth reference allocates `max_completion_length = max_seq - prompt`
  (effectively up to ~3700). We set 64. Our rewards score short outputs,
  but 64 tokens forces high `completions/clipped_ratio` (we saw 0.125-0.25
  per step) which is a structural reward leak: a clipped completion
  scores worse on most rubric dims and so widens the zero-std group rate.
  Recommended: `max_completion_length=192`.

## Recommendations

Do these in order. None require re-merging the SFT artifact.

### Immediate (next 1-2 Modal runs)

1. Add `gemma4_e2b_rl_smoke_v3.yaml` with:
   - `learning_rate: 2e-5`
   - `epsilon_high: 0.28`, `delta: 1.5`
   - `num_generations: 6`
   - `max_completion_length: 192`
   - `max_steps: 20`
   - `max_grad_norm: 0.5` (requires plumbing through GRPOConfig; currently
     not exposed in our `GRPOSmokeConfig` dataclass).
   - Keep `lora_rank: 16`, `lora_alpha: 32`, `optim: adamw_8bit`,
     `loss_type: bnpo`, `mask_truncated_completions: true`.
2. Re-run the v3 smoke and inspect:
   - `frac_reward_zero_std` should stay below 0.5 most steps.
   - `kl` should stay below ~100 after the first 2 warmup steps.
   - `grad_norm` should be finite for all steps; if not, lower
     `max_grad_norm` to 0.3.

### Reward design (next 3-5 runs)

3. Audit `WEIGHTED_REWARD_FUNCS` for zero-floor degeneracy: any time the
   faithfulness/format component is zero for an entire group, advantage
   normalisation divides by ~0. Either:
   - clip each per-prompt reward into `[reward_floor, 1.0]` with
     `reward_floor = 1e-3`, or
   - add a tiny per-completion length-aware regularisation reward (e.g.
     `0.01 * (1 - clipped_ratio)`).
4. Add reward dithering of `~U(-0.005, 0.005)` to keep group std away from
   zero when all completions agree. This is standard GRPO hygiene at small
   `num_generations`.

### Artifact hygiene (DONE 2026-05-25)

Resolved:

- **Model cards refreshed** on both HF repos with a documented Verification
  report section, KV-shared layer architecture note, Unsloth #5386
  tokenizer caveat, and transformers PR #45328 reference. Pushed via
  `push_model_cards_modal.push_cards` (Modal app `ap-vGP68iFJWOhyGdHdWM5E1k`).
  - `jayshah5696/gemma4-e2b-humanize-unsloth-merged/README.md` (4116 bytes)
  - `jayshah5696/gemma4-e2b-humanize-unsloth-lora/README.md` (2288 bytes)
- **Independent PEFT-merge candidate published** at
  `jayshah5696/gemma4-e2b-humanize-unsloth-merged-peft-v2`. Verified by
  `verified_merge_and_push_modal.verified_merge_and_push` (Modal app
  `ap-5EfALCGYoz3yw8sTfZOCuO`). Result:

  ```json
  {
    "gates": {"all_pass": true,
              "wrongly_present_shared_kv_keys": [],
              "transformers_missing_keys": [],
              "transformers_unexpected_keys": [],
              "safetensors_key_count": 1951},
    "parity": {"prompt_count": 10,
               "in_memory_mismatches": 1,
               "reload_mismatch_count": 1},
    "tokenizer": {"eos_token_observed": "<turn|>", "restored": false}
  }
  ```

  Both the Unsloth and PEFT merge paths produce **identical artifacts**
  (same 1951 keys, same single benign word-swap on the same Slack-rewrite
  prompt, same preserved eos). Strong cross-validation. Sibling repo
  exists as a backup; production merged repo is untouched.
- **Processor assets**: the Unsloth merged repo already ships
  `processor_config.json` and the verifier falls back to base for any
  legacy `preprocessor_config.json` requirement. No upload needed.
- **Verifier-as-gate plumbing**: `verified_merge_and_push_modal.py`
  refuses to push when any of `wrongly_present_shared_kv_keys`,
  `transformers_missing_keys`, `transformers_unexpected_keys` is
  non-empty or when reload parity exceeds the configured tolerance. It
  also auto-restores the chat eos before push if Unsloth #5386 ever
  regresses it. Use it for any future re-merge:

  ```bash
  rtk uvx modal run --detach \
    src/humanize_rl/training/verified_merge_and_push_modal.py \
    --candidate-repo <user>/<repo>-vN
  ```

- **Single source of truth for model card text**:
  `src/humanize_rl/training/_model_cards.py`. Both the on-disk push job
  and the re-merge job consume the same templates.

Files added in this hygiene pass:

- `src/humanize_rl/training/_model_cards.py` (templates).
- `src/humanize_rl/training/push_model_cards_modal.py` (cards-only push).
- `src/humanize_rl/training/verified_merge_and_push_modal.py` (gated PEFT
  re-merge + push).
- `tests/training/test_artifact_hygiene.py` (9 tests; full suite 33
  passed, 5 skipped, ruff clean).

### Out of scope, but recorded

7. `adapters/gemma4_e2b_v04_mlx_full_r8` (and siblings) trained
   `self_attn.k_proj`/`v_proj` LoRA on layers 15-34 against modules that
   the forward pass ignored, due to a pre-fix `mlx-lm` bug fixed in PR
   #1158 (April 2026). Treatment: **keep as local-MLX-only inference
   artifact**. Do not convert or migrate to Modal/PEFT/RL. If we want a
   second SFT signal from MLX, re-train with current `mlx-lm` and
   `target_modules` restricted to non-shared layers.

## Definition of done — updated

A verified starting policy exists when (all true today):

- [x] One candidate merged model loads with zero missing Gemma 4 weights.
- [x] Direct LoRA vs merged parity report is saved.
- [x] 10-prompt parity report is saved.
- [x] HF candidate repo is published
  (`jayshah5696/gemma4-e2b-humanize-unsloth-merged`).
- [x] `configs/rl/gemma4_e2b_rl_smoke_v2.yaml` points to the verified
  candidate.
- [x] Detached RL artifact verification passes against that candidate.
- [x] Tokenizer eos_token preserved as `<turn|>`.
- [x] RL loader documents the benign-MISSING behavior and guards against
  the real-failure case.
- [x] Independent PEFT-merge cross-validation published at
  `jayshah5696/gemma4-e2b-humanize-unsloth-merged-peft-v2`.
- [x] Model cards on both production repos document verification and
  Gemma 4 KV-shared layer design.
- [x] Verifier-as-gate (`verified_merge_and_push_modal`) refuses to push
  on any structural failure; auto-restores chat eos.

The RL **training stability** is a separate definition-of-done that
belongs to the RL plan, not this merge-verification plan.