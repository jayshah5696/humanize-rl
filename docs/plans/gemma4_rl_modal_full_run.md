# Gemma 4 E2B — Full RL Run Plan (v01+v02+v03)

**Date:** 2026-05-26
**Status:** Draft for execution — supersedes the "Slice 10"
placeholder in `gemma4_rl_modal_stable_training_continuation.md`.
**Scope:** Take everything we learned in Slices 0–8 of the
continuation plan and execute one clean, well-instrumented,
publication-grade RL run on the full v01+v02+v03 task pool.
**Constraint:** Keep TRL + Modal. Stay Google-only on
humanizer/judge/scorer; v03 multi-author exception applies to *task
authorship only* and is already enforced upstream.

---

## 1. Where we stand (recap, not redo)

Slices 0–8 are done. Numbers worth carrying into this plan:

- **Frozen recipe** (D8.D1.1): `G=16, acc=16, LR=1e-5,
  scalar_softened, scale_rewards="group",
  vllm_gpu_memory_utilization=0.55`.
- **Effective rollout batch**: 256 trajectories per optimizer step.
- **Throughput** at this recipe: ~11.2 s/step on A100-40GB colocated
  vLLM.
- **Cost ledger to date**: ~$7.13 of $20 cap → ~$12.87 runway.
- **Last gate result** (Slice 8, 200 steps, filtered v01+v02 mix,
  423 rows / 341 train):
  - mixed-val eval delta **+0.0273** (gate: > +0.02 ✅)
  - v01 control delta +0.005 (no regression ✅)
  - training reward delta first20→last20 **+0.0105** (positive ✅)
  - clipped_ratio 0, frac_0std 0, grad_norm 1.10 — all healthy.
- **Ceiling identified** (D8.D1.3): `penalty_rate/ema_50 ≈ 0.998`.
  Reward gain came from ridge/deterministic; risk side barely
  moved. The fix is to widen the softened-reward linear region —
  `penalty_cap=1.0 → 1.5` — committed up front (see §3.2).

This plan picks up from there. Three things change vs Slice 8:

1. **Dataset** — go from filtered v01+v02 (423 rows) to filtered
   v01+v02+v03 (target ~700–900 rows).
2. **Horizon** — go from 200 steps to 600 (resumable to 1,000).
3. **Reward calibration** — `penalty_cap` 1.0 → 1.5.

Everything else (model, LoRA rank, optimizer, KL, clipping, LR)
stays as-is. We do not re-explore the search space the ablations
already closed.

---

## 2. What v03 brings (and what it doesn't)

### 2.1 v03 status

From the v03 build plan (`docs/plans/v03-rl-tasks-dataset.md`) and
current `data/rl/`:

| artifact | rows | splits | notes |
|---|---:|---|---|
| `humanize_tasks_v03.jsonl` | 831 | 665/83/83 | raw assembled |
| `humanize_tasks_v03_judged_kept.jsonl` | 594 | 466/61/67 | passed arka judge gate |
| `humanize_tasks_v03_filtered.jsonl` | 492 | 390/48/54 | passed judge + ref-rollout filter |
| `humanize_tasks_rl_mix_v1_with_v03draft.jsonl` | 563 | 451/52/60 | **stale** — only 57 v03 rows (early draft) |

**The current `_with_v03draft` mix is stale.** We rebuild fresh
using `v03_filtered.jsonl` (492 rows) as the v03 source.

v03 mode distribution (in `_filtered`):

| mode | rows |
|---|---:|
| rewrite_humanize | 113 |
| compression | 104 |
| tone_shift | 97 |
| expansion | 65 |
| long_form_generate | 62 |
| multi_constraint_compose | 51 |

v03 author distribution (in `_filtered`):

| author | rows | share |
|---|---:|---:|
| openai/gpt-5.4-mini | 282 | 57% |
| google/gemini-3.1-flash-lite-preview | 210 | 43% |

**Author A (Gemini Pro) is missing from `_filtered`.** Per v03
plan §3.1 the target was 40% Pro / 35% mini / 25% Flash-Lite.
v03-build concern, not an RL-run blocker — we note it in the
candidate card as a known limitation.

### 2.2 What v03 buys the RL run

| dimension | v01+v02 only | with v03 |
|---|---|---|
| total filtered rows | 423 | ~700–900 (after re-filter) |
| modes covered | rewrite_repair / compression / tone_shift | + expansion, long_form_generate, multi_constraint_compose |
| mean instruction length | ~20 w | ~120 w |
| constraint density | ~1 active | ~4–6 active |
| ref-rollout sanity | none in v02 | already passed for v03 |

The big unlock is **mode and instruction-length diversity**. v01+v02
collapses to "rewrite this short thing"; v03 adds genuine long-form
generation and multi-constraint composition. GRPO needs reward
variation across heterogeneous prompts to keep `reward_std` healthy
without dither; mode diversity helps that directly.

### 2.3 What v03 does NOT buy

- **No new reward signal.** Env reward and ridge scorer are
  unchanged. v03 plan §6's eight new deterministic checks are
  *future work*; not built. v03's `forbidden_phrases` and
  `forbidden_facts` fields aren't consumed by `score_response()`.
- **No long-completion ridge adapter.** v03 plan §6.1 proposed a
  windowed ridge for >300-word completions; not built. Long-form
  v03 tasks may systematically under-score on ridge until that
  lands. E6 (long-form eval) will tell us how much it matters.

More data, more diversity, **same reward**. We're testing whether
the larger and more varied task pool by itself lifts training
reward and eval delta. Reward-side improvements are explicitly out
of scope for this run.

---

## 3. Committed configuration choices

These four decisions are made up front. No probe stages. If any
turns out wrong, the full run still produces a valid (possibly
weaker) candidate that we ship as `candidate-v1` with documented
limitations.

### 3.1 Dataset: fresh mix_v2 (v01+v02+v03 filtered)

Built once, then trained on. Two commands, no validation gate.

```bash
rtk uv run python scripts/rl/build_rl_task_mix.py \
  --v01 data/rl/humanize_tasks_v01_smoke.jsonl \
  --v02 data/rl/humanize_tasks_v02.jsonl \
  --v03 data/rl/humanize_tasks_v03_filtered.jsonl \
  --output data/rl/humanize_tasks_rl_mix_v2.jsonl \
  --summary outputs/rl_mix/humanize_tasks_rl_mix_v2_summary.json

rtk uvx modal run scripts/rl/score_task_difficulty_modal.py \
  --task-path data/rl/humanize_tasks_rl_mix_v2.jsonl \
  --output-dir outputs/rl_difficulty/mix_v2 \
  --filtered-output data/rl/humanize_tasks_rl_mix_v2_filtered_softened_midband.jsonl \
  --rollouts-per-task 6
```

Expected: ~1,100 raw rows → ~700–900 filtered. If the filtered
output has < 600 train rows we go back and loosen midband
thresholds; otherwise we proceed.

### 3.2 Reward: `penalty_cap=1.5` (was 1.0)

Slice 8 proved `penalty_cap=1.0` saturates risk_compliance at
0.20 with penalty_rate at 99.8%. The softened reward is

```
risk_compliance = clip(1 + total_penalty / penalty_cap, 0, 1)
```

so once `total_penalty < -1.0` the component bottoms out and the
gradient w.r.t. risk vanishes. Raising the cap to 1.5 widens the
linear region by 50%, letting the policy keep moving as it gets
closer to compliance.

If 1.5 doesn't lift `diag/risk_compliance/ema_50` above Slice 8's
0.20, the full run still produces a valid candidate — the
risk-side simply doesn't improve. That's a known-acceptable outcome
matching Slice 8's behavior; we ship and document. No probe stage
needed.

### 3.3 Horizon: 600 steps (resumable to 1,000)

Slice 8 EMA was still rising at step 200. Linear extrapolation says
~600 steps hits the plateau. Mix_v2 has ~2× the data so 600
optimizer steps × effective batch 256 ≈ 1.7–2.4 epochs over the
train split — the sweet spot before overfitting.

If `reward/ema_50` is clearly still rising at step 600 (slope >
+5e-5 over the last 100 steps), the same checkpoint resumes to 1,000
steps for an incremental ~$3. We make that decision when we see the
end-of-run curve, not now.

### 3.4 Everything else: frozen from Slices 5–8

```
num_generations: 16          # B1 winner
gradient_accumulation_steps: 16
learning_rate: 1.0e-5
reward_mode: scalar_softened
scale_rewards: "group"       # D7.D1.2
vllm_gpu_memory_utilization: 0.55  # D6.D1.2
stratify_batches: true
stratify_by: reward_profile
loss_type: bnpo
epsilon: 0.2 / epsilon_high: 0.28 / delta: 1.5
mask_truncated_completions: true
ridge_weight: 0.45 / deterministic_weight: 0.35 / risk_weight: 0.20
```

No tuning. No re-litigation. These all paid for themselves in
prior slices.

---

## 4. The full run

### 4.1 Config

`configs/rl/gemma4_e2b_rl_a100_full_v2.yaml`:

```yaml
model_name: "jayshah5696/gemma4-e2b-humanize-unsloth-merged"
experiment_name: "gemma4-rl-full-v2"
hf_lora_repo: "jayshah5696/gemma4-e2b-humanize-rl-candidate-v1"

task_path: "data/rl/humanize_tasks_rl_mix_v2_filtered_softened_midband.jsonl"
train_split: train
eval_split: validation

max_seq_length: 2048
max_prompt_length: 512
max_completion_length: 768

lora_rank: 16
lora_alpha: 32

# Frozen recipe (§3.4).
learning_rate: 1.0e-5
num_generations: 16
gradient_accumulation_steps: 16
per_device_train_batch_size: 1
max_steps: 600    # §3.3, resumable

warmup_ratio: 0.1
weight_decay: 0.001
max_grad_norm: 0.5
optim: adamw_8bit
loss_type: bnpo
epsilon: 0.2
epsilon_high: 0.28
delta: 1.5
mask_truncated_completions: true
temperature: 1.0
seed: 3407

stratify_batches: true
stratify_by: reward_profile

# Reward (D5.1 + §3.2 calibration).
reward_mode: scalar_softened
penalty_cap: 1.5
ridge_weight: 0.45
deterministic_weight: 0.35
risk_weight: 0.20

scale_rewards: "group"

use_vllm: true
vllm_mode: colocate
vllm_gpu_memory_utilization: 0.55

sample_before_after: 0
generation_batch_size: 8
artifact_generation_samples: 0

push_to_hub: false
report_to: wandb
wandb_project: "humanize-rl"
```

### 4.2 Training gates (runtime monitoring)

Kill the run if any of these trip after step 50:

| gate | threshold | action if violated |
|---|---|---|
| any NaN | yes | kill, investigate |
| `completions/clipped_ratio/ema_50` | > 0.10 | kill, lower LR or shorten max_completion_length |
| `frac_reward_zero_std/ema_50` | > 0.15 | kill, raise num_generations or re-filter dataset |
| `grad_norm/ema_50` | > 5.0 or < 0.1 | kill, investigate exploding/vanishing |
| `diag/response_length_mean/ema_50` | < 30 | kill, length-collapse hack |
| `reward/ema_50` slope (last 100 steps) | < -0.02 | kill, reward divergence |
| `train_samples_per_second` | < 0.045 | kill, infra problem |

The EMA callback already exposes all these to W&B; spot-check every
~50 steps.

### 4.3 Eval gates (post-train)

Six eval slices, all run in parallel after training completes.
Each runs against a different validation set. Baseline evals run
*in parallel with training* (they don't need the adapter), so we
have baseline numbers waiting by the time training finishes.

| eval | dataset | split | rows | acceptance Δ |
|---|---|---|---:|---|
| E1 mixed val | mix_v2 filtered | validation | 80 | **> +0.05** |
| E2 v01 control | v01 smoke | validation | 20 | **≥ 0** |
| E3 v02 control | v02 | validation | 50 | **> +0.02** |
| E4 v03 control | v03 filtered | validation | 48 | **> +0.03** |
| E5 hard-penalty subset | mix_v2 rows with `mean_risk_penalty ≤ -0.7` | validation | varies | **> +0.05** on mean_risk_penalty |
| E6 long-form subset | mix_v2 rows with mode ∈ {long_form_generate, expansion, multi_constraint_compose} | validation | varies | **> +0.02** |

E1 is the headline gate. E5 tests whether §3.2's `penalty_cap=1.5`
unlocked risk-side learning. E6 tests whether long-form (v03-only)
mode diversity helped despite no windowed ridge.

### 4.4 Qualitative inspection

```bash
rtk uvx modal run scripts/eval/sample_adapter_modal.py \
  --adapter-dir /checkpoints/gemma4-rl-full-v2/final_adapter \
  --task-path /workspace/data/rl/humanize_tasks_rl_mix_v2_filtered_softened_midband.jsonl \
  --split validation \
  --num-tasks 16 --completions-per-task 4 --temperature 0.7 \
  --output-path outputs/full_run/sampled_inspection.json
```

Read 8 random samples manually before pushing. Look for:

- No length collapse (completions still 100–500 words where warranted)
- No "always short to dodge penalty" hack
- No `must_include` regression (model isn't dropping required content
  to optimize style)
- No register collapse (warm tasks still sound warm, formal still formal)
- Subjective "does this read more human?" check

---

## 5. Pushing the candidate

Only if all training gates + E1, E2 + qualitative inspection pass.
E3–E6 contribute to the candidate card narrative but are not
gating (E1 is the headline; E2 is the no-regression floor).

```bash
rtk uvx modal volume get humanize-rl-checkpoints \
  /gemma4-rl-full-v2/final_adapter outputs/full_run/final_adapter/ --force

rtk uv run scripts/publish_to_hf.py model \
  --repo-id jayshah5696/gemma4-e2b-humanize-rl-candidate-v1 \
  --path outputs/full_run/final_adapter/ \
  --kind lora \
  --readme runs/cards/rl_candidate_v1.md
```

Model card must include:
- base model: `jayshah5696/gemma4-e2b-humanize-unsloth-merged`
- training data: `humanize_tasks_rl_mix_v2_filtered_softened_midband.jsonl`
  (with row count and v01/v02/v03 mix)
- recipe summary (the §4.1 yaml as a code block)
- E1–E6 eval table with baseline + post numbers
- W&B run link, Modal app ID
- known limitations:
  - reward saturation (98%+ penalty rate) — only partially
    addressed by penalty-cap recalibration
  - v03 author imbalance (no Pro author content)
  - no long-form ridge adapter
- explicitly tagged `candidate`, not final.

---

## 6. Total budget

| item | $ |
|---|---:|
| Build mix_v2 + difficulty score | $1.50 |
| Baseline evals (mix_v2 + v01 + v03, parallel w/ training) | $0.40 |
| Full run (600 steps) | $4.50 |
| Post-train evals E1–E6 | $1.50 |
| Qualitative inspection | $0.30 |
| Buffer | $0.50 |
| **plan total** | **~$8.70** |

Combined with ~$7.13 already spent: ~$15.83 of $20 cap.
~$4.17 buffer = one re-roll of training (~$4.50) or two re-rolls
of eval (~$1.50 each).

If we resume to 1,000 steps (§3.3): +~$3 → ~$18.83 cumulative.
Still inside the cap.

---

## 7. Risk register

| risk | mitigation | trigger / response |
|---|---|---|
| Mix_v2 has < 600 filtered train rows | loosen midband thresholds | re-filter with looser bounds |
| v03 ref-rollout filter too strict | use `_judged_kept` (594 rows) as v03 source | < 400 v03 train rows surviving the mix filter |
| `penalty_cap=1.5` doesn't lift risk_compliance | accept Slice 8's ceiling; ship; document | risk_compliance ema_50 stays < 0.25 |
| Mid-run grad-norm explosion on v03 long-form | reduce max_completion_length to 640 | grad_norm/ema_50 > 5.0 |
| Long-form completions score systematically lower (no windowed ridge) | accept; E6 quantifies the gap | E6 < +0.02 |
| E1 misses +0.05 (lands +0.03) | publish anyway as `candidate-v1` with weaker claim | document in card |
| Infra failure mid-run | TRL checkpoints every `save_steps`; resume from last save | use Trainer resume |
| Model regresses on v01 (E2 negative) | likely overfit to v03 long-form; reduce v03 share | regenerate mix with v03 share capped at 50% |

---

## 8. Runbook (literal command list, in order)

This is what to actually type on the day.

```bash
# === Pre-flight ===
rtk uv run pytest tests/reward/ tests/rl/ tests/training/test_wandb_ema_callback.py tests/scripts/test_build_rl_task_mix.py --ignore=tests/reward/test_offline_eval.py --ignore=tests/reward/test_prime_env.py -q
rtk uv run ruff check src/humanize_rl/training/rl_gemma4_trl_vllm_modal.py scripts/eval/run_vf_eval_modal.py scripts/rl/build_rl_task_mix.py

# === Build mix_v2 ===
rtk uv run python scripts/rl/build_rl_task_mix.py \
  --v01 data/rl/humanize_tasks_v01_smoke.jsonl \
  --v02 data/rl/humanize_tasks_v02.jsonl \
  --v03 data/rl/humanize_tasks_v03_filtered.jsonl \
  --output data/rl/humanize_tasks_rl_mix_v2.jsonl \
  --summary outputs/rl_mix/humanize_tasks_rl_mix_v2_summary.json

rtk uvx modal run scripts/rl/score_task_difficulty_modal.py \
  --task-path data/rl/humanize_tasks_rl_mix_v2.jsonl \
  --output-dir outputs/rl_difficulty/mix_v2 \
  --filtered-output data/rl/humanize_tasks_rl_mix_v2_filtered_softened_midband.jsonl \
  --rollouts-per-task 6

# (eyeball the summary; confirm ≥ 600 train rows)

# === Bake mix_v2 + full-run config into both Modal images ===
# (add_local_file for the new mix + config in
#  src/humanize_rl/training/rl_gemma4_trl_vllm_modal.py
#  AND scripts/eval/run_vf_eval_modal.py)

# === Kick off training + baseline evals in parallel ===
rtk context_tag full-run-kickoff

rtk uvx modal run --detach src/humanize_rl/training/rl_gemma4_trl_vllm_modal.py \
  --mode train \
  --config-path /workspace/configs/rl/gemma4_e2b_rl_a100_full_v2.yaml &

rtk uvx modal run scripts/eval/run_vf_eval_modal.py \
  --variant baseline \
  --task-path /workspace/data/rl/humanize_tasks_rl_mix_v2_filtered_softened_midband.jsonl \
  --split validation --max-examples 80 \
  --output-path outputs/full_run/baseline_eval_mix_v2.json &

rtk uvx modal run scripts/eval/run_vf_eval_modal.py \
  --variant baseline \
  --task-path /workspace/data/rl/humanize_tasks_v01_smoke.jsonl \
  --split validation --max-examples 20 \
  --output-path outputs/full_run/baseline_eval_v01.json &

rtk uvx modal run scripts/eval/run_vf_eval_modal.py \
  --variant baseline \
  --task-path /workspace/data/rl/humanize_tasks_v03_filtered.jsonl \
  --split validation --max-examples 48 \
  --output-path outputs/full_run/baseline_eval_v03.json &

wait

# (monitor W&B; kill if §4.2 gates trip)

# === After training stops (~2 hours later) ===
rtk uvx modal volume get humanize-rl-checkpoints \
  /gemma4-rl-full-v2/summary.json \
  outputs/full_run/training_summary.json --force

# === Eval gates E1–E6 in parallel ===
# (six modal-run lines using --variant adapter
#  --adapter-dir /checkpoints/gemma4-rl-full-v2/final_adapter;
#  outputs to outputs/full_run/eval_e{1..6}.json)

# === Qualitative inspection ===
rtk uvx modal run scripts/eval/sample_adapter_modal.py \
  --adapter-dir /checkpoints/gemma4-rl-full-v2/final_adapter \
  --task-path /workspace/data/rl/humanize_tasks_rl_mix_v2_filtered_softened_midband.jsonl \
  --split validation --num-tasks 16 --completions-per-task 4 \
  --temperature 0.7 \
  --output-path outputs/full_run/sampled_inspection.json

# === Decide: push or iterate ===
# If E1 > +0.05 AND E2 ≥ 0 AND qualitative passes: §5 push.
# If E1 in [+0.03, +0.05]: push as candidate-v1 with weaker
#   claim; log in docs/logbook/.
# Else: tag checkpoint as `attempt-1`, debrief, decide whether
#   to resume training or iterate on the mix.

rtk context_tag full-run-complete
```

---

## 9. What this plan does NOT do

Listed explicitly so we don't scope-creep mid-run:

- **No new reward components.** v03 plan §6's eight new
  deterministic checks are not built and are not built here.
- **No env version bump.** `humanize_rl_env` stays at its current
  version. v03 long-form / multi-constraint rows go through the
  existing `prime_dataset_row` path; if the env barfs on empty
  `input_text` we fix inline, not redesign.
- **No new ridge adapter.** Long-form windowed ridge (v03 plan
  §6.1) is deferred. E6 quantifies the gap.
- **No Author A (Gemini Pro) backfill in v03.** Deferred to
  v03-build followup.
- **No reward changes per dataset_version.** All rows go through
  the same reward — no v03-specific bonus.
- **No SFT re-merge.** Base model stays
  `gemma4-e2b-humanize-unsloth-merged`.
- **No final candidate, only `candidate-v1`.** Even with all
  gates green, this is a candidate — manual review before "final"
  promotion.
- **No probe stages.** Every committed choice in §3 stands or
  fails on the full run. If something turns out wrong, the run
  still ships a candidate; we just write up the limitation.

---

## 10. Acceptance summary

This plan succeeds if, when we're done, we can say:

1. `humanize_tasks_rl_mix_v2_filtered_softened_midband.jsonl`
   exists with ≥ 600 train rows, ref-rollout median in
   [0.40, 0.85].
2. The full run completed without any §4.2 training gate firing.
3. E1 (mixed eval) delta > +0.05.
4. E2 (v01 control) does not regress.
5. E3, E4, E5, E6 deltas at or above their plan thresholds (or
   the failures are documented and explained).
6. Qualitative inspection of 8 random samples reads as
   "noticeably more human" without obvious style hacks.
7. `jayshah5696/gemma4-e2b-humanize-rl-candidate-v1` is live on
   HF with a card that quotes every eval number and the W&B run
   link.
8. The logbook (`docs/logbook/gemma4_rl_stable_training_log.md`)
   has a "Full Run" entry mirroring the Slice 7 / Slice 8
   format: recipe, gates, findings, decisions, cost.
9. Total spend ≤ ~$17 of the $20 cap (i.e., still some buffer
   for one re-roll if the candidate doesn't pass review).
