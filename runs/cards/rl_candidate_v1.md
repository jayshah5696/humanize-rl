---
license: apache-2.0
language: en
library_name: peft
base_model: jayshah5696/gemma4-e2b-humanize-unsloth-merged
tags:
  - humanize
  - rl
  - grpo
  - lora
  - candidate
  - trl
  - gemma
---

# Gemma 4 E2B — Humanize RL Candidate v1

**Status:** `candidate-v1` — RL-tuned LoRA adapter, pending manual
review before promotion to `final`.

A LoRA adapter trained with GRPO (TRL + Modal, A100-40GB,
colocated vLLM) on top of
[`jayshah5696/gemma4-e2b-humanize-unsloth-merged`](https://huggingface.co/jayshah5696/gemma4-e2b-humanize-unsloth-merged),
which is itself an SFT-merged humanize model on Gemma 4 E2B IT.

The RL objective is a **softened scalar reward** combining a
ridge-rubric humanness scorer (`0.45 weight`), a deterministic
verifier (`0.35 weight`), and a clipped risk-compliance term
(`0.20 weight`, `penalty_cap=1.5`). The dataset is a filtered
mix of v01 (smoke), v02 (rewrite-heavy), and v03 (rich
instructions + 6 modes) totaling **964 tasks (771 train / 90
validation / 103 test)**, balanced ≈ 41% v02 / 58% v03 / <1% v01
(v01 collapsed via text dedup vs v02).

## Recipe

```yaml
# Frozen from Slices 5–8 (see logbook / plan).
num_generations: 16
gradient_accumulation_steps: 16
per_device_train_batch_size: 1   # effective rollout batch = 256
learning_rate: 1.0e-5
optim: adamw_8bit
loss_type: bnpo
epsilon: 0.2
epsilon_high: 0.28
delta: 1.5
mask_truncated_completions: true
temperature: 1.0
seed: 3407
max_steps: 600
warmup_ratio: 0.1
max_grad_norm: 0.5

# Reward.
reward_mode: scalar_softened
penalty_cap: 1.5
ridge_weight: 0.45
deterministic_weight: 0.35
risk_weight: 0.20

# Reward normalization.
scale_rewards: "group"   # TRL default

# Sampling stratification.
stratify_batches: true
stratify_by: reward_profile

# vLLM (colocated).
use_vllm: true
vllm_mode: colocate
vllm_gpu_memory_utilization: 0.55

# LoRA.
lora_rank: 16
lora_alpha: 32

# Lengths.
max_seq_length: 2048
max_prompt_length: 512
max_completion_length: 768
```

Training wall-clock: **117 minutes** (600 steps × ~11.7 s/step) on a
single A100-40GB.

## Training trajectory

| metric | first 20 steps | last 20 steps | Δ |
|---|---:|---:|---:|
| `reward` | 0.7757 | 0.8014 | +0.0256 |
| `diag/risk_compliance` | 0.6423 | 0.7329 | +0.0906 |

| metric | first 100 | last 100 | Δ |
|---|---:|---:|---:|
| `reward` | 0.7704 | 0.7898 | +0.0194 |

Final EMAs at step 600:

| metric | value |
|---|---:|
| `reward/ema_20` | 0.8080 |
| `reward/ema_50` | 0.7979 |
| `reward_std/ema_50` | 0.0336 |
| `grad_norm/ema_50` | 0.8348 |
| `entropy/ema_50` | 0.4731 |
| `diag/penalty_rate/ema_50` | 0.598 |
| `diag/risk_compliance/ema_50` | 0.717 |
| `diag/response_length_mean/ema_50` | 590.4 |
| `diag/response_length_p95/ema_50` | 706.3 |
| `completions/clipped_ratio/ema_50` | 0.000 |
| `frac_reward_zero_std/ema_50` | 0.000 |

The key delta vs the Slice 8 200-step pilot is `risk_compliance`:
0.20 → 0.72 (3.6×). This validates the
`penalty_cap=1.0 → 1.5` hypothesis from the full-run plan §3.2 —
the previous training-time risk penalty was saturating at the
softened-reward floor, killing the risk-side gradient.

## Eval results

All eval rewards are the **strict** reward (raw penalties applied,
no softening) — the user-facing metric, not the training-time
shaped scalar.

| eval | dataset | rows | baseline | post | Δ | gate | result |
|---|---|---:|---:|---:|---:|---|---|
| E1 mixed val | mix_v2 filtered | 80 | 0.2705 | 0.3241 | **+0.0536** | > +0.05 | ✅ |
| E2 v01 control | v01 smoke | 20 | 0.4815 | 0.5191 | **+0.0376** | ≥ 0 | ✅ |
| E3 v02 control | v02 raw | 50 | −0.0567 | −0.0106 | **+0.0461** | > +0.02 | ✅ |
| E4 v03 control | v03 filtered | 48 | 0.7005 | 0.7497 | **+0.0491** | > +0.03 | ✅ |
| E5 hard-penalty subset | mix_v2 val, mean penalty ≤ −0.7 | 41 | −0.0877 | −0.0332 | **+0.0545** reward (risk Δ +0.0134) | risk > +0.05 | ⚠️ reward-side gate passes; risk-side improves +0.013 (below +0.05 risk-side gate) |
| E6 long-form subset | mix_v2 val, mode ∈ {long_form_generate, expansion, multi_constraint_compose} | 17 | 0.6989 | 0.7397 | **+0.0408** | > +0.02 | ✅ |

Risk-penalty deltas (mean penalty improvement, larger is better):

| eval | baseline | post | Δ |
|---|---:|---:|---:|
| E1 mix_v2 | −0.5312 | −0.5181 | +0.0131 |
| E2 v01 | −0.3200 | −0.3050 | +0.0150 |
| E3 v02 | −0.8640 | −0.8540 | +0.0100 |
| E4 v03 | −0.0969 | −0.0833 | +0.0135 |
| E5 hard-penalty | −0.8976 | −0.8841 | +0.0134 |
| E6 long-form | −0.0735 | −0.0676 | +0.0059 |

## Qualitative inspection

Sampled 16 validation tasks × 4 completions each at `temperature=0.7`.
Manual side-by-side review of 4 random task pairs (compression and
rewrite tasks): the adapter outputs are noticeably warmer and more
conversational (e.g. opens with "Heads up", "hope you're having a
good week"), while keeping required facts intact. Mean response
length: 290 chars baseline → 272 chars adapter — slight tightening
on compression tasks where it's warranted, no length collapse on
generation tasks. No obvious style hacks ("always short to dodge
penalty" pattern not observed).

## Dataset mix

Built via `scripts/rl/build_rl_task_mix.py` + difficulty filter +
local rebucket under `scalar_softened_permissive` mode (see
`scripts/rl/rebucket_difficulty.py`).

| source | raw rows | in filtered mix |
|---|---:|---:|
| v01 (smoke) | 100 | 6 (collapsed via text dedup vs v02) |
| v02 (rewrite-heavy) | 512 | 491 |
| v03 (filtered) | 492 | 467 |
| **total** | **1,104** | **964** (771 train / 90 val / 103 test) |

Mode distribution in the train split:

| mode | count |
|---|---:|
| rewrite | 493 |
| rewrite_humanize | 104 |
| compression | 102 |
| tone_shift | 91 |
| expansion | 64 |
| long_form_generate | 56 |
| multi_constraint_compose | 51 |
| direct_generation | 1 |
| compress | 1 |
| repair | 1 |

## Known limitations

1. **Reward saturation only partially addressed.** Training-time
   `penalty_rate/ema_50` dropped from Slice 8's 0.998 to 0.598 thanks
   to `penalty_cap=1.5`, but the strict-eval mean risk penalty
   barely moved (E1: −0.531 → −0.518, +0.013). Most eval gain came
   from style/structure improvements (ridge + deterministic), not
   from genuinely avoiding penalties. A next-iteration follow-up
   should add synonym-tolerant fact matching in
   `_fact_is_present` to widen the achievable risk-compliance
   ceiling.
2. **v03 author imbalance.** The v03 task corpus is 57% GPT-5.4-mini
   / 43% Gemini Flash-Lite; the planned 40% Gemini Pro share is
   absent. The adapter therefore inherits any phrasing fingerprints
   from those two specific authors more than the v03 plan
   intended.
3. **No long-form ridge adapter.** The ridge scorer was trained on
   short samples; long-form v03 tasks (200–600 word completions)
   may systematically under-score relative to short tasks. The E6
   delta (+0.0408) is real but smaller than E1/E3/E4 deltas in
   relative terms.
4. **`scale_rewards="group"` (default), not `"batch"`.** Slice 7
   showed `"batch"` was a wash at 50 steps; we did not re-test at
   the 600-step horizon. The plan-default may be the right choice
   for a longer run; left as a follow-up ablation.
5. **`candidate-v1`, not final.** No external benchmark
   verification yet; humanness is measured only on our own ridge +
   deterministic reward. A blind A/B against the SFT-only base is
   the next gate before promoting to `final`.

## Reproducibility

| artifact | location |
|---|---|
| training config | `configs/rl/gemma4_e2b_rl_a100_full_v2.yaml` (in repo) |
| training script | `src/humanize_rl/training/rl_gemma4_trl_vllm_modal.py` |
| training data | `data/rl/humanize_tasks_rl_mix_v2_filtered_softened_midband.jsonl` |
| training summary JSON | `outputs/full_run/training_summary.json` |
| eval results | `outputs/full_run/{baseline,post}_eval_*.json` |
| plan doc | `docs/plans/gemma4_rl_modal_full_run.md` |
| logbook | `docs/logbook/gemma4_rl_stable_training_log.md` |
| Modal training app | `ap-dwT1lU0keVgDicpSgtov2d` |
| Modal eval apps | `ap-CBkKk3FzwZGR7SY9mEbm1r`, `ap-EZWCRuRwlgDw0YG94cNZzN`, `ap-TyqTZdKPa3koIuxkOYebYB`, plus E1–E6 (see `outputs/full_run/`) |
| W&B project | `humanize-rl` |

## Loading

```python
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

base = "jayshah5696/gemma4-e2b-humanize-unsloth-merged"
adapter = "jayshah5696/gemma4-e2b-humanize-rl-candidate-v1"

processor = AutoProcessor.from_pretrained(base)
model = AutoModelForImageTextToText.from_pretrained(base, torch_dtype="bfloat16")
model = PeftModel.from_pretrained(model, adapter)
model.eval()
```

(Note: requires the vLLM 0.20.x Gemma 4 kv-shared k_norm patch
when serving with vLLM — see
`patch_vllm_gemma4_kv_shared_k_norm` in the training script for
the 4-line inline fix until upstream PR #40117 lands.)
