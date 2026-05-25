# Gemma 4 E2B Humanize-RL Fine-Tuning Implementation Plan

**Date:** 2026-05-24  
**Status:** Implementation plan / approval checklist  
**Target:** `unsloth/gemma-4-E2B-it` or official equivalent `google/gemma-4-E2B-it`  
**Training stage covered:** SFT first; preference/RL only after SFT passes diagnostics  
**Recommended production backend:** Modal + Unsloth on NVIDIA GPU  
**Recommended local backend:** MLX only for smoke tests and small ablations, not the canonical run

---

## 0. Executive decision

Use the **instruction-tuned model** first: **`gemma-4-E2B-it`**.

Do **not** start from the base model for the first production SFT run. Our goal is not to teach general instruction following from scratch; it is to preserve Gemma 4's instruction-following behavior while shifting style toward direct, natural, human-sounding writing. Starting from `-it` gives us that behavior for free and lowers the amount of data required.

Base-model SFT is an ablation only if we later have a much larger, cleaner, release-eligible direct-instruction dataset and can prove that base SFT beats `-it` on instruction following, formatting, factuality, and humanness.

---

## 1. Current project state

From `docs/plans/gemma_4_e2b_alignment_strategy.md`, the strategy is sound:

1. direct generative humanness, not only rewrite/humanize;
2. SFT before preference tuning or RL;
3. two-layer scoring with a distilled local scorer;
4. governance gates before full training;
5. RL only if SFT + preference tuning plateau.

Given the note that **Track A is done** and **Track B is done**, the next milestone should be:

> Build a frozen, auditable SFT dataset, train `gemma-4-E2B-it` with bf16 LoRA, run checkpoint diagnostics, and only then consider preference tuning.

Repository observation: I see Track A artifacts in `data/scorer/`, `models/track_a_10k`, and `paper/track_a_scorer_report.md`. I only see a small v04 Arka pilot output at `configs/v04/runs/v04-base-pilot-reviewed/report/samples.jsonl`; if the completed Track B artifacts exist outside this checkout or are generated but not committed, the implementation should point to those finalized files before training.

---

## 2. Honest recommendation: Modal + Unsloth vs MLX

### 2.1 Production recommendation: Modal + Unsloth

Use **Modal + Unsloth** for the canonical run.

Reasons:

- Unsloth now documents Gemma 4 E2B/E4B/26B/31B fine-tuning support.
- Unsloth's Gemma 4 docs include Gemma 4-specific bug fixes around shared KV/cache behavior and training-loss quirks.
- Unsloth reports Gemma 4 E2B LoRA fitting in roughly 8-10GB VRAM, with faster/lower-memory training than standard FA2 setups.
- Modal gives reproducible remote containers, persistent Volumes for checkpoints, secret management, retries, and clean GPU selection.
- This project wants a publishable, reproducible training run. Cloud GPU logs + fixed container image + saved dataset manifest is easier to audit than a local laptop run.

Recommended Modal GPU:

- **Pilot:** L40S, A100-40GB, or A10G if we keep sequence length small. E2B should fit on modest GPUs, but L40S/A100 gives headroom for batch size, eval generation, and fewer OOM surprises.
- **Canonical SFT:** L40S or A100-40GB.
- **Avoid choosing H100 first** unless time matters more than cost; E2B LoRA does not need it.

### 2.2 Local recommendation: MLX as a smoke-test path only

Use **MLX** for:

- local data-format validation;
- 50-200 step smoke runs;
- checking loss moves in the right direction;
- tiny ablations on a sample split;
- interactive inspection before spending cloud GPU money.

Do **not** make MLX the canonical training path yet.

Reasons:

- `mlx-lm` is official and supports LoRA/QLoRA for Gemma-family models, but its documented support is more general than the current Gemma 4-specific Unsloth support.
- `mlx-tune` looks useful and explicitly aims to bring an Unsloth-like API to Mac users, including Gemma 4 examples, but it is unofficial and should be treated as a workflow bridge, not the production baseline.
- Apple Silicon unified memory is nice, but training throughput and ecosystem maturity still lag CUDA for repeatable production runs.
- Export/runtime parity can be a hidden time sink. For this project, the actual deliverable is a reliable adapter/checkpoint, not proving MLX is production-ready.

Practical stance:

> Prototype locally with MLX if convenient. Train and publish from Modal + Unsloth.

---

## 3. Base vs instruct decision

### Chosen model

Use:

```text
unsloth/gemma-4-E2B-it
```

or the official HF repo if Unsloth's wrapper is not needed for a given experiment.

### Why `-it`

Our SFT dataset is instruction-response data. We want the model to answer user tasks naturally, not learn all chat behavior from scratch. The `-it` model already knows:

- chat roles;
- instruction following;
- refusal/safety priors;
- concise assistant behavior;
- formatting expectations.

The fine-tune should adjust style and reduce AI-writing tells while preserving the above.

### When to test base

Run a base-model ablation only if all are true:

1. final SFT set has at least ~10k genuinely high-quality rows, preferably much more;
2. coverage includes direct generation, rewrite, email, Slack/chat, technical explanation, support, product copy, and document-style tasks;
3. the held-out eval confirms base does not regress on instruction following;
4. training budget allows a real comparison, not a one-off curiosity run.

---

## 4. Gemma 4 internals that matter for this project

Add these to project internals / run notes:

1. **Chat format:** Gemma 4 uses standard `system`, `user`, and `assistant` roles, rendered by the Gemma 4 chat template.
2. **Thinking mode:** Thinking is explicit. For our SFT, train on **final visible answers only**. Do not include thought blocks.
3. **Small models:** E2B/E4B are multimodal and have shared KV/cache behavior. Use the current Unsloth/Transformers stack with Gemma 4 fixes.
4. **Loss scale:** Unsloth notes E2B/E4B may show losses around 13-15 because of multimodal-model quirks. Do not judge the run only by raw absolute loss; rely on validation loss trend and generation evals.
5. **Generation defaults for eval:** use Gemma 4-recommended defaults unless a task requires otherwise: temperature around `1.0`, `top_p=0.95`, `top_k=64`; also test deterministic low-temperature behavior for regression evals.
6. **Inference template parity:** The same chat template used in training must be used during evaluation/export. Wrong template/EOS can make a good adapter look broken.

---

## 5. Dataset quality gates before training

The most important work now is not writing the trainer. It is freezing a dataset that will not teach the model bad habits.

### 5.1 Required final SFT schema

Canonical JSONL row:

```json
{
  "id": "sft_v04_000001",
  "messages": [
    {"role": "user", "content": "Draft a short Slack update..."},
    {"role": "assistant", "content": "Quick update: ..."}
  ],
  "domain": "chat",
  "task_type": "slack_chat",
  "mode": "direct_generation",
  "source": "v04_stream_b_or_synthetic",
  "license": "MIT|CC-BY-4.0|ODC-By-1.0|project_synthetic",
  "release_eligible": true,
  "quality": {
    "l1_score": 0.0,
    "l2_score": 0.0,
    "local_scorer_ai_probability": 0.0,
    "judge_keep": true
  }
}
```

Training can consume only `messages`, but the metadata must stay in the frozen dataset manifest.

### 5.2 Hard rejects

Reject a row if any condition is true:

- missing instruction or response;
- response is too short to train meaningful behavior, unless task explicitly asks for ultra-short output;
- response has obvious AI tells:
  - `Certainly`
  - `Of course`
  - `it is worth noting`
  - `furthermore`
  - `moreover`
  - `in conclusion`
  - `please don't hesitate`
  - `I hope this email finds you well`
- rewrite task adds names, dates, tools, project names, metrics, root causes, companies, or locations not in the input;
- direct-generation task invents specifics where placeholders should be used;
- instruction mentions AI, ChatGPT, AI-generated, humanize, detector evasion, or anything that teaches meta-behavior rather than writing behavior;
- PII is present and not intentionally public/release-safe;
- source license is not approved for the intended release.

### 5.3 Important quality issue seen in current pilot sample

In `configs/v04/runs/v04-base-pilot-reviewed/report/samples.jsonl`, one sample asks:

> Draft an email to my manager requesting PTO for the first week of July, outlining who will cover my projects.

The generated response says Sarah will cover the projects, but the instruction did not provide Sarah. That is an unsupported invented name. This exact failure mode must be blocked before training.

The existing `scripts/verify_v04_quality.py` already has logic for fake names and missing placeholders. Keep this gate, strengthen it, and run it over the final Track B set.

### 5.4 Scorer use

Use the Track A scorer as a **filtering signal**, not as the only truth.

The Track A report shows the selected TF-IDF Logistic+Ridge scorer is fast and strong on the labeled split, but it wins partly because obvious AI phrases are lexical shortcuts. That is fine for data filtering. It is not enough for final model selection or RL reward by itself.

Recommended use:

- reject high AI-probability outputs;
- rank borderline rows for manual/LLM review;
- compute diagnostic aggregate scores by domain;
- never use the same scorer as the only final evaluator.

### 5.5 Final split design

Create frozen splits by source group / near-duplicate cluster, not random row only:

```text
data/processed/sft/gemma4_e2b_v04/
  train.jsonl
  valid.jsonl
  test.jsonl
  manifest.json
  dataset_card.md
  quality_report.md
```

Suggested split:

- train: 90%
- validation: 5%
- test: 5%

But hold out entire near-duplicate clusters and source groups. Do not let a rewrite seed and its synthetic variants leak across train/test.

---

## 6. Target dataset mixture

Because Track A and Track B are done, build the SFT set from the highest-quality rows only.

Recommended first canonical mix:

| bucket | target share | notes |
|---|---:|---|
| human/gold direct pairs | 20-35% | strongest anchor; use release-eligible only |
| high-quality synthetic direct pairs | 35-50% | must pass judge + scorer + unsupported-detail checks |
| rewrite/humanize pairs | 15-30% | useful, but do not let rewrite dominate |
| legacy hand-vetted pairs | 5-10% | high-signal gold anchors |

Avoid a dataset that is mostly rewrite tasks. The project strategy correctly says the model should natively generate natural responses from scratch. If rewrite examples dominate, the model may learn editing behavior rather than direct writing behavior.

Minimum practical target:

- **Pilot SFT:** 500-1,000 clean rows.
- **First real SFT:** 3,000-10,000 clean rows.
- **Base-model ablation:** only after 10,000+ strong rows; preferably more.

---

## 7. Training recipe: Modal + Unsloth

### 7.1 LoRA settings

Start with conservative bf16 LoRA:

```yaml
model_name: unsloth/gemma-4-E2B-it
precision: bf16
load_in_4bit: false
full_finetuning: false
max_seq_length: 2048
lora:
  r: 16
  alpha: 16
  dropout: 0.0
  bias: none
  target: all language attention + MLP linear layers
  finetune_vision_layers: false
  finetune_language_layers: true
  finetune_attention_modules: true
  finetune_mlp_modules: true
```

If memory is tighter than expected, use QLoRA as a fallback, not the first choice. Project guidance says bf16 LoRA for Gemma 4 E2B.

### 7.2 Trainer settings

Recommended starting settings:

```yaml
per_device_train_batch_size: 1-2
gradient_accumulation_steps: 8-16
effective_batch_size: 16
learning_rate: 2.0e-4
num_train_epochs: 1-3
warmup_ratio: 0.03
weight_decay: 0.001
lr_scheduler_type: cosine
optim: adamw_8bit
max_grad_norm: 0.3
seed: 3407
train_on_responses_only: true
save_strategy: steps
eval_strategy: steps
```

Run ablations:

1. `r=8, alpha=8`, 1 epoch;
2. `r=16, alpha=16`, 1-2 epochs;
3. `r=32, alpha=32`, only if underfitting is visible and validation improves.

Do not chase training loss below ~0.2. For style SFT, that is likely memorization/overfitting.

### 7.3 Response-only loss

Use response-only training. The prompt/instruction tokens should be masked out. This matters because we want the model to learn the assistant output style, not memorize user prompts.

For Gemma 4, Unsloth examples use parts like:

```python
instruction_part = "<|turn>user\n"
response_part = "<|turn>model\n"
```

Verify this against the exact tokenizer/chat template in the training script.

### 7.4 Modal infrastructure

Use:

- one Modal App;
- one persistent Volume for HF/model cache;
- one persistent Volume for datasets;
- one persistent Volume for checkpoints/results;
- HF token secret;
- optional W&B secret;
- explicit timeout and retries.

Modal run artifacts:

```text
/checkpoints/gemma4-e2b-humanize-sft/
  run_config.yaml
  dataset_manifest.json
  adapter checkpoints
  final_adapter/
  eval_generations.jsonl
  metrics.json
```

---

## 8. MLX local path

### 8.1 Official MLX-LM path

`mlx-lm` supports LoRA/QLoRA with local JSONL datasets. Use this for smoke testing:

```bash
mlx_lm.lora \
  --model unsloth/gemma-4-E2B-it \
  --train \
  --data data/processed/sft/gemma4_e2b_v04_mlx \
  --iters 100 \
  --batch-size 1 \
  --grad-accumulation-steps 8 \
  --mask-prompt \
  --adapter-path adapters/gemma4_e2b_v04_smoke
```

This requires `train.jsonl` and optional `valid.jsonl` in the MLX data directory. Use `chat` format:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

### 8.2 MLX-Tune path

`mlx-tune` is attractive because it mimics Unsloth's API and now advertises SFT/DPO/GRPO plus Gemma 4 examples. It is best used for local prototyping when you want a script that is easy to port to Unsloth later.

Caveat: keep it out of the canonical reproducibility story unless we verify:

- exact Gemma 4 chat template parity;
- response-only masking parity;
- adapter export compatibility;
- generation parity against the Modal/Unsloth adapter;
- stable install on the target Mac.

---

## 9. Evaluation before accepting a checkpoint

Do not select by validation loss only.

### 9.1 Automatic evals

Evaluate each checkpoint on:

1. frozen SFT validation loss;
2. held-out generation prompts by domain;
3. Track A local scorer;
4. Layer 1 heuristics;
5. external LLM judge split using Google model via OpenRouter;
6. instruction-following checks;
7. unsupported-detail checks;
8. length/verbosity metrics.

### 9.2 Required diagnostic prompt buckets

- Slack/team updates;
- professional email;
- customer/support replies;
- technical explanations;
- product copy;
- rewrite/humanize;
- shorten/compress;
- grammar/clarity;
- adversarial prompts that invite AI tells;
- prompts with missing details where placeholders are required.

### 9.3 Acceptance gates

Accept SFT checkpoint only if it:

- improves humanness score over base `gemma-4-E2B-it` on held-out prompts;
- does not increase unsupported details;
- does not inflate average response length by more than ~10-15%;
- does not regress instruction following;
- does not overuse placeholders when details are actually provided;
- does not collapse into terse Slack-like style for all domains;
- passes sampled human review.

---

## 10. Vertical slice implementation plan

Your proposed sequence is the right one:

1. use `-it`;
2. check model/data quality;
3. run a small MLX smoke test locally;
4. run a small Modal pilot with W&B;
5. run the full Modal training job;
6. publish LoRA artifacts after evaluation passes.

My only pushback: **do not publish merged/full weights automatically at the end of the first full run.** Publish the LoRA adapter privately or as a draft first, run evals, confirm license/release eligibility, then publish public adapter + optional merged weights. The LoRA adapter is the canonical artifact; merged weights are a convenience artifact after approval.

### Slice 0 — Lock model choice and run config

**Goal:** make the model decision explicit and reusable by every script.

Create:

```text
configs/training/gemma4_e2b_sft_v04.yaml
```

Minimum config:

```yaml
model:
  name: unsloth/gemma-4-E2B-it
  family: gemma-4
  base_type: instruct
  chat_template: gemma-4
  train_on_responses_only: true
  enable_thinking: false

lora:
  precision: bf16
  load_in_4bit: false
  r: 16
  alpha: 16
  dropout: 0.0
  max_seq_length: 2048

training:
  learning_rate: 2.0e-4
  epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  warmup_ratio: 0.03
  weight_decay: 0.001
  max_grad_norm: 0.3
  lr_scheduler_type: cosine
  seed: 3407

tracking:
  wandb_project: humanize-rl
  wandb_entity: null
  run_group: gemma4-e2b-sft-v04
```

Definition of done:

- config exists;
- all later scripts can load it;
- `gemma-4-E2B-it` is the default and base model is not used unless explicitly overridden.

### Slice 1 — Dataset quality and frozen small subset

**Goal:** prove we can build an auditable SFT dataset before training anything.

Create:

```text
scripts/build_gemma4_sft_dataset.py
scripts/report_sft_dataset_quality.py
```

Inputs:

- final Track B rows;
- legacy gold rows;
- quality-judged rows if present;
- Track A scorer artifact from `models/track_a_10k`.

Outputs:

```text
data/processed/sft/gemma4_e2b_v04/
  train.jsonl
  valid.jsonl
  test.jsonl
  smoke_train.jsonl
  smoke_valid.jsonl
  manifest.json
  quality_report.md
  rejected.jsonl
```

`smoke_train.jsonl` should be tiny, e.g. 50-100 rows. `smoke_valid.jsonl` should be 10-20 rows. This is for MLX and Modal plumbing checks, not quality claims.

Dataset builder responsibilities:

- normalize to `messages` format;
- preserve metadata: domain, task type, mode, source, license, release eligibility;
- reject hard failures;
- flag unsupported details and fake names;
- reject obvious AI-tell phrases;
- dedupe exact rows;
- near-dedupe by instruction/response similarity or existing Arka near-dedupe output;
- score with Track A local scorer if artifact is available;
- split by source group / near-duplicate cluster;
- write a manifest with counts and input file hashes.

Quality report must include:

- row counts by split;
- domain/task/mode/source/license distribution;
- direct-generation vs rewrite ratio;
- response length percentiles;
- rejected row counts by reason;
- phrase rates for known AI tells;
- unsupported-name/detail flag counts;
- local scorer distribution;
- 5-10 sample rows per major domain;
- release eligibility summary.

Definition of done:

- hard-reject checks catch the known invented-name PTO failure;
- `smoke_train.jsonl` and `smoke_valid.jsonl` are valid chat JSONL;
- `quality_report.md` is readable enough to decide whether training is worth running;
- full train/valid/test files are frozen and reproducible from `manifest.json`.

### Slice 2 — MLX local smoke run

**Goal:** verify local data formatting, tokenizer/chat template compatibility, LoRA plumbing, and loss movement on a tiny subset.

Create:

```text
scripts/smoke_mlx_gemma4_lora.sh
```

Use a separate local MLX environment rather than adding MLX packages to the main project dependencies immediately. This avoids polluting the repo's CUDA/Modal dependency story.

Recommended command shape:

```bash
uvx --python 3.12 --with 'mlx-lm[train]' mlx_lm.lora \
  --model unsloth/gemma-4-E2B-it \
  --train \
  --data data/processed/sft/gemma4_e2b_v04_mlx_smoke \
  --iters 50 \
  --batch-size 1 \
  --grad-accumulation-steps 8 \
  --mask-prompt \
  --adapter-path adapters/gemma4_e2b_v04_mlx_smoke
```

If we use `mlx-tune`, keep it as an optional second script:

```text
scripts/smoke_mlx_tune_gemma4_lora.py
```

MLX smoke run checks:

- package installs on the local Mac;
- model loads;
- data loads;
- prompt masking works;
- LoRA adapter is written;
- generation from adapter does not crash;
- loss is finite and generally moves down over 50-100 iterations.

Definition of done:

- a local adapter appears under `adapters/gemma4_e2b_v04_mlx_smoke/`;
- one base-vs-adapter generation comparison is saved;
- no claim is made that this is a good model.

### Slice 3 — Modal + Unsloth pilot with W&B

**Goal:** prove the production training stack works before the full run.

Create:

```text
src/humanize_rl/training/finetune_gemma4_modal.py
```

Modal requirements:

- `modal.App`;
- image built with `uv_pip_install`;
- install current Unsloth, Transformers, TRL, PEFT, Accelerate, Datasets, W&B;
- persistent HF cache Volume;
- persistent dataset Volume or upload/mount from local path;
- persistent checkpoint Volume;
- `huggingface` secret with `HF_TOKEN`;
- `wandb` secret with `WANDB_API_KEY`;
- timeout high enough for pilot and full runs;
- retries enabled.

Pilot dataset:

- use `smoke_train.jsonl` / `smoke_valid.jsonl`, or a 200-500 row `pilot_train.jsonl`;
- run 50-200 optimizer steps;
- log to W&B project `humanize-rl`;
- save adapter checkpoint to Modal Volume;
- save run config and dataset manifest with checkpoint.

Pilot acceptance checks:

- Modal image builds;
- GPU function starts;
- W&B run appears with config, loss, LR, step metrics;
- Unsloth loads `gemma-4-E2B-it` with the expected chat template;
- response-only masking is verified in logs;
- adapter saves cleanly;
- base-vs-adapter generation script runs on 10 prompts;
- no obvious template/EOS breakage.

Definition of done:

```text
/checkpoints/gemma4-e2b-humanize-sft/pilot-*/
  run_config.yaml
  dataset_manifest.json
  final_adapter/
  eval_generations.jsonl
  metrics.json
```

### Slice 4 — Full Modal SFT run

**Goal:** train the first real SFT checkpoint from the frozen dataset.

Run matrix:

1. `r=8, alpha=8`, 1 epoch, cheap baseline;
2. `r=16, alpha=16`, 1 epoch, default candidate;
3. `r=16, alpha=16`, 2 epochs only if validation/generation suggests underfitting.

Do not start with `r=32`. E2B is small and the dataset is stylistic; overfitting is more likely than under-capacity.

Full-run requirements:

- W&B tracking enabled;
- eval every fixed number of steps;
- checkpoint every fixed number of steps;
- final adapter saved;
- tokenizer/template files saved;
- run config saved;
- dataset manifest saved;
- training logs and metrics saved;
- final eval runner executed automatically or immediately after.

Definition of done:

- at least one full candidate run completes;
- eval artifacts exist;
- checkpoint can be reloaded locally or on Modal;
- adapter generates normal Gemma 4 responses with the correct template;
- candidate is compared against base `gemma-4-E2B-it`.

### Slice 5 — Publish artifacts after approval

**Goal:** publish only artifacts that passed evaluation and data-governance checks.

Publishing order:

1. private/draft LoRA adapter repo;
2. dataset card or reconstruction manifest;
3. eval report;
4. public LoRA adapter after approval;
5. optional merged weights after approval;
6. optional GGUF/MLX export later.

Recommended HF repos:

```text
humanize-rl/gemma-4-e2b-it-humanize-sft-lora
humanize-rl/gemma-4-e2b-it-humanize-sft-merged   # optional after approval
humanize-rl/humanize-sft-v04                     # only if release-eligible
```

Pushback on full weights:

- publish **LoRA adapter first**;
- publish merged weights only after eval passes and license/release eligibility is confirmed;
- if any training data is internal-only, publish model artifacts only if the model-release policy allows it and dataset rows are not redistributed.

Definition of done:

- HF model card includes model base, training data summary, known limitations, eval metrics, intended use, and license notes;
- W&B run URL is linked in the internal report;
- adapter hash and dataset manifest hash are recorded.

### Slice 6 — Evaluation gate before next phase

**Goal:** decide whether SFT is good enough or whether we need preference tuning.

Create:

```text
scripts/evaluate_gemma4_sft_checkpoint.py
```

Responsibilities:

- generate outputs from base and fine-tuned adapter on the same prompt set;
- score both with Layer 1 + Track A scorer;
- run LLM judge on a sampled subset;
- compare length, unsupported details, and AI-tell phrase rates;
- write `metrics.json` and `eval_generations.jsonl`.

Decision:

- If SFT passes gates: publish adapter and stop.
- If SFT improves style but has specific failures: patch dataset and rerun SFT.
- If SFT plateaus after dataset fixes: then build preference pairs and consider DPO/IPO/SimPO.
- Do not jump directly to RL.

---

## 11. Suggested implementation order in this repo

1. Add `configs/training/gemma4_e2b_sft_v04.yaml`.
2. Add tests for dataset row normalization and rejection rules.
3. Implement `scripts/build_gemma4_sft_dataset.py`.
4. Implement `scripts/report_sft_dataset_quality.py`.
5. Generate `smoke_train.jsonl` / `smoke_valid.jsonl`.
6. Add `scripts/smoke_mlx_gemma4_lora.sh`.
7. Run MLX smoke locally.
8. Implement Modal trainer.
9. Run Modal pilot with W&B.
10. Implement eval runner.
11. Run full Modal candidate matrix.
12. Publish LoRA adapter only after eval approval.

Recommended TDD tests:

```text
tests/training/test_gemma4_sft_dataset.py
```

Test cases:

- converts `{instruction,response}` to `messages`;
- rejects empty response;
- rejects known AI-tell phrase;
- rejects invented name in rewrite/direct missing-detail case;
- preserves placeholders;
- writes deterministic split with fixed seed;
- manifest includes source hashes.

---

## 12. Open risks

1. **Dataset may be too synthetic.** If synthetic rows dominate, the model will learn the generator's style rather than human style.
2. **Rewrite overrepresentation.** Too many rewrite rows will make the model good at editing but less reliable at direct drafting.
3. **Lexical scorer shortcuts.** Track A scorer is useful but cannot be the only reward/evaluator.
4. **Unsupported details.** Current pilot already shows invented names. This is the highest-priority data-quality issue.
5. **Gemma 4 template mismatch.** Training/eval/export must use the same Gemma 4 chat template.
6. **Overfitting small model style.** E2B can pick up dataset quirks quickly. Keep epochs low and evaluate every checkpoint.
7. **MLX parity.** Useful locally, but do not assume an MLX-trained adapter behaves identically to Unsloth/CUDA until proven.

---

## 13. My honest bottom line

The best path is:

1. **Use `gemma-4-E2B-it`, not base, for the first real run.**
2. **Freeze and audit the SFT dataset before touching training.**
3. **Train canonical SFT on Modal with Unsloth bf16 LoRA.**
4. **Use MLX only as a local smoke-test/prototyping tool.**
5. **Do not start RL yet.** Run SFT, measure it brutally, then decide whether preference tuning is needed.

If the final Track B dataset is clean and at least a few thousand rows, the first SFT run is worth doing now. If Track B has many rows like the pilot's invented-name PTO example, spend another day on data filtering before training; otherwise the fine-tune will faithfully learn those mistakes.
