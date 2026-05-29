# Gemma 4 E2B RL Continuation — Stable TRL + Modal Training Plan

**Date:** 2026-05-25  
**Status:** Draft for review  
**Scope:** Continue from `docs/plans/gemma4_rl_modal_20usd_budget_plan.md` after Slice 3 PASS.  
**Constraint:** Keep training on **TRL + Modal**. Do **not** switch to Prime RL / hosted training. Use Prime/TRL/DAPO guidance as design input only.  
**Goal:** Turn the successful but noisy $20 GRPO run into a stable, scalable RL recipe over all available task data: v01, v02, and v03-in-progress.

---

## 1. Executive summary

Slice 3 proved that Gemma 4 E2B can improve under our TRL + Modal GRPO setup:

| signal | result |
|---|---:|
| baseline eval mean reward | 0.4398 |
| post-RL eval mean reward | 0.6143 |
| eval delta | **+0.1745** |
| training reward first 100 steps | 0.4823 |
| training reward last 100 steps | 0.5863 |
| training reward delta | **+0.1040** |
| max clipped ratio | 0.000 |
| max zero-std group rate | 0.000 |
| actual training cost | $1.3169 |

But the W&B `train/reward` curve was jagged. The model improved, but the learning curve is not smooth enough to treat the current config as a final recipe.

Main conclusion:

> The reward is not conceptually too complicated. The problem is that our current training signal is too noisy for the sampling scale: small rollout groups, small effective batch, cliffy penalties, per-component reward functions, and tiny task data.

The next plan is to keep the rich reward diagnostics, but train on a **single calibrated scalar reward** over a **filtered, stratified, larger task mix** with **larger rollout groups** and **short parallel ablations** before the final run.

---

## 2. What happened in Slice 3

### 2.1 Training learned, but not cleanly

From `outputs/full_run/summary.json`:

| window | reward | ridge | deterministic | risk penalty |
|---|---:|---:|---:|---:|
| first 10 | 0.5886 | 0.3751 | 0.4829 | -0.2693 |
| last 10 | 0.6373 | 0.3653 | 0.4869 | -0.2149 |
| delta | +0.0487 | -0.0098 | +0.0040 | +0.0544 |
| first 100 | 0.4823 | 0.3440 | 0.4779 | -0.3395 |
| last 100 | 0.5863 | 0.3607 | 0.4831 | -0.2576 |
| delta | +0.1040 | +0.0167 | +0.0053 | +0.0820 |

Interpretation:

1. **The policy improved.** Both training and heldout eval moved up.
2. **Most of the training gain came from risk-penalty reduction.** The model learned to avoid obvious bad behaviors.
3. **Ridge/style improved only modestly.** That is useful, but not enough to say the model learned broad humanness.
4. **Raw per-step reward was noisy.** The jagged W&B graph is expected with tiny batches and heterogeneous task difficulty, but it is still a problem for final training.

### 2.2 Why the raw graph is not smooth

The current config uses:

```yaml
num_generations: 8
gradient_accumulation_steps: 8
per_device_train_batch_size: 1
max_steps: 200
train rows: 80
```

This is a small effective training signal. Every logged step is sensitive to:

- which prompt was sampled,
- whether that prompt carries hard risk penalties,
- whether the completions triggered a sparse penalty,
- whether ridge scores vary enough within the group,
- whether the task is too easy or too hard.

GRPO optimizes **relative advantage within a group**, not monotonic raw reward. A noisy raw reward graph does not prove failure, but it does reduce confidence and makes hyperparameter decisions harder.

---

## 3. Research guidance informing this plan

### 3.1 Prime Intellect Verifiers guidance

Prime Intellect’s Verifiers training docs recommend three things directly relevant to us:

1. **Evaluate baseline performance before training.** If reward is 0% after many attempts, the task is too hard; if it is already ~80%+, use harder examples.
2. **Ensure reward diversity inside generation groups.** GRPO needs within-group variation.
3. **For stability, increase rollouts and batch size.** Prime suggests larger `rollouts_per_example` such as 16–32 and larger batch sizes such as 512–1024 for stable training, plus online difficulty filtering.

Source: Prime Intellect Verifiers training docs, “RL Rules of Thumb” and “Performance Trade-offs”:  
<https://docs.primeintellect.ai/verifiers/training>

We are **not** using Prime RL, but we should port the principles into TRL + Modal:

- offline difficulty filtering,
- larger `num_generations`,
- larger gradient accumulation,
- stratified task sampling,
- richer eval and diagnostics.

### 3.2 Prime environment model

Prime’s environment model frames RL as:

> dataset + harness + rubric = training/eval environment.

That is exactly our setup: `humanize_rl_env` and `score_response(...)` are the same abstraction used for training reward and eval.

Source: Prime Intellect environment model docs:  
<https://docs.primeintellect.ai/hosted-training/environment-model>

Important implication: keep **one source of truth** for reward logic, but allow **different scalarizations** for training vs eval.

### 3.3 TRL GRPO reward normalization guidance

TRL’s GRPO docs explain that reward advantages are group-relative. They also note that scaling by within-group reward std can create question-level difficulty bias. TRL exposes alternatives:

- default group scaling,
- `scale_rewards="batch"`,
- `scale_rewards=False`.

Source: TRL GRPO Trainer docs:  
<https://huggingface.co/docs/trl/main/grpo_trainer>

For our shaped, multi-component reward, `scale_rewards="batch"` is an important ablation because it can reduce per-prompt difficulty amplification while still normalizing update scale.

### 3.4 DAPO / modern GRPO practice

DAPO-style GRPO improvements focus on:

- dynamic sampling,
- filtering zero-variance groups,
- clip-higher,
- token-level loss normalization,
- overlong response handling.

Source: NVIDIA NeMo DAPO guide:  
<https://docs.nvidia.com/nemo/rl/nightly/guides/dapo.html>

We already use some compatible ideas:

- `epsilon_high: 0.28` (clip-higher-ish),
- `mask_truncated_completions: true`,
- `loss_type: bnpo`,
- zero-std dither guard.

Missing:

- difficulty filtering,
- larger groups,
- larger batches,
- cleaner scalar reward,
- reward scaling ablation.

### 3.5 Zero-std groups and KL noise

TRL issue discussions in 2026 note that zero-std reward groups can produce useless or harmful gradients when KL terms are active. DAPO-style approaches prefer masking or filtering zero-variance groups rather than training on them.

Source: TRL issue on zero-std reward groups:  
<https://github.com/huggingface/trl/issues/5588>

Our run had `frac_reward_zero_std = 0`, so this did not bite Slice 3. But as we scale to v01+v02+v03, we should still track:

- zero-std rate,
- dither rate,
- reward_std distribution,
- difficulty buckets.

---

## 4. Available data and its role

We should use all available task data, but **not by naive concatenation**.

### 4.1 v01

File: `data/rl/humanize_tasks_v01_smoke.jsonl`

Strengths:

- balanced task families,
- known control set,
- good for smoke tests and regression.

Weaknesses:

- only 100 rows,
- 80 train rows,
- too small for final RL.

Role:

- keep as control/eval anchor,
- include a small fraction in training mix.

### 4.2 v02

File: `data/rl/humanize_tasks_v02.jsonl`

Strengths:

- 512 rows,
- more volume than v01.

Weaknesses noted in `docs/plans/v03-rl-tasks-dataset.md`:

- rewrite-heavy,
- templated instructions,
- limited family/domain coverage,
- noisy regex-derived required facts,
- weak constraint diversity.

Role:

- useful for volume,
- must be filtered and balanced,
- should not dominate final training.

### 4.3 v03

File: in progress per `docs/plans/v03-rl-tasks-dataset.md`.

Target:

- ~2,500 richer tasks,
- more modes,
- richer instructions,
- more machine-verifiable constraints,
- reference-rollout filtering.

Role:

- main future RL corpus,
- should become the dominant source once quality gates pass.

### 4.4 Proposed dataset mix

Early stable run:

```text
v01: 15%
v02: 35%
v03: 50%
```

Once v03 matures:

```text
v01: 10%
v02: 20%
v03: 70%
```

Every row should carry:

```text
dataset_version: v01 | v02 | v03
family
reward_profile
mode
length_bucket
difficulty_bucket
source_group
```

---

## 5. Is the reward too complicated?

Short answer: **not inherently**.

Human writing quality is multi-objective. We need style, faithfulness, length, format, placeholder discipline, and risk controls. The problem is not that the reward has multiple ideas. The problem is how those ideas are currently exposed to GRPO.

Current training uses separate reward functions:

```text
ridge_rubric_reward
deterministic_reward
risk_penalty_reward
```

This creates problems:

1. **Per-component noise enters the optimizer.** Each component has its own variance and dither behavior.
2. **Sparse penalties dominate.** A single risk penalty can swamp smoother style signals.
3. **Raw reward becomes task-difficulty dependent.** Harder task families look like worse policy steps.
4. **The optimizer sees components as additive gradients, not diagnostics.** We lose control over scalarization.

Better design:

> Use one calibrated scalar reward for training; log all components as diagnostics.

This preserves rich measurement while giving GRPO one cleaner signal.

---

## 6. Reward options

### Option A — Current reward, unchanged

Training reward:

```text
0.50 * ridge_rubric + 0.50 * deterministic + raw penalties
```

Implemented as multiple reward functions.

Pros:

- already works,
- passed Slice 3,
- simple continuity.

Cons:

- jagged reward graph,
- risk penalties dominate,
- dither can affect each component,
- harder to reason about scalar gradient.

Use only as baseline ablation.

### Option B — Single scalar current reward

Training reward stays mathematically similar:

```text
scalar = 0.50 * ridge_rubric + 0.50 * deterministic + raw penalties
```

But expose it as **one reward function** to TRL. Log components separately.

Pros:

- isolates whether the multi-function setup is causing noise,
- minimal semantic change,
- dither applies only to final scalar if needed.

Cons:

- raw penalties can still dominate.

This should be the first reward-side ablation.

### Option C — Softened scalar reward

Training reward:

```text
risk_compliance = clip(1 + total_penalty / penalty_cap, 0, 1)

scalar_train_reward =
  0.45 * ridge_rubric
+ 0.35 * deterministic
+ 0.20 * risk_compliance
```

Eval reward remains strict:

```text
strict_eval_reward = 0.50 * ridge_rubric + 0.50 * deterministic + raw penalties
```

Pros:

- smoother reward surface,
- avoids giant sparse negative cliffs,
- still teaches risk avoidance,
- reduces incentive to become overly terse just to avoid penalties.

Cons:

- train reward and eval reward diverge,
- needs careful monitoring to avoid reward hacking.

Expected best candidate for final run.

### Option D — Curriculum reward

Training phase 1:

```text
0.30 ridge + 0.50 deterministic + 0.20 risk_compliance
```

Training phase 2:

```text
0.45 ridge + 0.35 deterministic + 0.20 risk_compliance
```

Pros:

- first teaches constraints/risk,
- later emphasizes style.

Cons:

- more moving parts,
- harder to ablate,
- may be unnecessary.

Save for later unless Option C underperforms.

---

## 7. Hypotheses to validate in parallel

The next phase should be a small, parallel ablation suite. Each run should be cheap: 30–50 steps on A100, W&B logged, same eval script.

### Shared success metrics

Each ablation should produce:

1. `summary.json`
2. W&B run
3. baseline/post eval on the same heldout subset
4. component trends
5. response samples

Call an ablation successful if:

| metric | success threshold |
|---|---|
| eval mean reward delta | `> +0.02` |
| training reward EMA slope | positive |
| clipped ratio | `< 0.05` |
| zero-std group rate | `< 0.10` |
| dither rate | `< 0.10` once tracked |
| risk negative count | not increased |
| response length | no collapse vs baseline |
| qualitative sample | no obvious terse/evasive hack |

The winner is not necessarily the highest training reward. The winner is the best combination of:

- smoother EMA,
- positive eval delta,
- stable length,
- no risk regression,
- no component collapse.

---

## 8. Parallel ablation suite

### Ablation A — Reward scalarization

Purpose: test whether multi-function reward is causing noisy gradients.

| run | reward mode | expected result |
|---|---|---|
| A0 | current 3 reward funcs | noisy baseline |
| A1 | single scalar current reward | same direction, lower noise |
| A2 | softened scalar reward | smoothest and best eval stability |

What to look for:

- W&B `reward/ema_20`,
- reward_std stability,
- eval delta,
- risk penalty count,
- length distribution.

Expected winner: **A2 softened scalar**.

### Ablation B — Group size and batch size

Purpose: test Prime-style stability guidance with larger rollout groups.

| run | `num_generations` | `gradient_accumulation_steps` | expected result |
|---|---:|---:|---|
| B0 | 8 | 8 | current noise baseline |
| B1 | 16 | 16 | smoother reward, better std estimates |
| B2 | 16 | 8 | cheaper compromise if B1 too slow/OOM |

What to look for:

- sec/step,
- peak VRAM,
- reward EMA smoothness,
- reward_std,
- eval delta.

Expected winner: **B1 if memory/cost allow**, else B2.

### Ablation C — Reward normalization

Purpose: test TRL GRPO scaling strategies.

| run | `scale_rewards` | expected result |
|---|---|---|
| C0 | default / group | baseline |
| C1 | `"batch"` | more robust shaped reward scaling |
| C2 | `false` | maybe less difficulty bias, but unstable unless reward is calibrated |

Before running, verify TRL 1.1 accepts this field in `GRPOConfig`.

Expected winner: **C1 `scale_rewards="batch"`**.

### Ablation D — Learning rate

Purpose: test whether LR is too high for noisy reward.

| run | LR | expected result |
|---|---:|---|
| D0 | `2e-5` | faster, noisier |
| D1 | `1e-5` | slower, smoother, likely better final candidate |

Expected winner: **D1 for final**, unless eval delta is too small.

### Ablation E — Dataset mix

Purpose: test whether v02/v03 volume stabilizes learning.

| run | dataset | expected result |
|---|---|---|
| E0 | v01 only | high variance, overfits quickly |
| E1 | v01 + v02 raw | more volume, but noisy due v02 templates |
| E2 | v01 + v02 difficulty-filtered | smoother and stronger |
| E3 | v01 + v02 + v03 sample | best once v03 quality passes |

Expected winner before full v03: **E2**.  
Expected winner after v03 exists: **E3**.

### Ablation F — Difficulty filtering

Purpose: test whether training only on useful-difficulty prompts smooths GRPO.

Offline process:

1. For each prompt, generate `K=8` completions from the SFT model.
2. Score all completions.
3. Compute:
   - mean reward,
   - reward std,
   - penalty rate,
   - clipped rate,
   - length stats.
4. Keep prompts where:

```text
0.20 <= mean_reward <= 0.80
reward_std >= 0.03
clipped_rate < 0.10
not all completions share same penalty pattern
```

Bucket prompts:

```text
easy / normal / hard
low-var / useful-var / high-var
short / medium / long
```

Expected result:

- fewer dead groups,
- smoother reward EMA,
- lower need for dither,
- better eval transfer.

This is one of the highest-priority additions.

---

## 9. Proposed implementation order

### Step 1 — Instrumentation first

Add W&B metrics before running more ablations:

```text
reward/ema_20
reward/ema_50
ridge/ema_20
deterministic/ema_20
risk_penalty/ema_20
risk_compliance/ema_20
penalty_rate/ema_20
dither_rate
dataset_version/*
family/*
reward_profile/*
difficulty_bucket/*
response_length_mean
response_length_p95
eval/mean_reward
eval/risk_negative_count
eval/win_rate_vs_baseline
```

Raw `train/reward` should not be the main training graph.

### Step 2 — Build unified dataset mixer

Create:

```text
data/rl/humanize_tasks_rl_mix_v1.jsonl
```

Inputs:

- v01,
- v02,
- v03 draft when available.

Operations:

- schema normalization,
- dedupe,
- add `dataset_version`,
- add length buckets,
- add family/profile/source tags,
- optionally add difficulty buckets after scoring.

### Step 3 — Build difficulty scoring script

Create a Modal script that:

- loads SFT model,
- generates `K=8` completions per task,
- scores completions,
- writes:

```text
outputs/rl_difficulty/<dataset>/rollouts.jsonl
outputs/rl_difficulty/<dataset>/task_difficulty.jsonl
data/rl/humanize_tasks_rl_mix_v1_filtered.jsonl
```

### Step 4 — Run parallel 30–50 step ablations

Start with 4 runs:

| run | reward | G | grad accum | LR | data |
|---|---|---:|---:|---:|---|
| P0 | current 3 funcs | 8 | 8 | 2e-5 | v01 |
| P1 | single scalar current | 8 | 8 | 2e-5 | v01 |
| P2 | softened scalar | 8 | 8 | 2e-5 | v01 |
| P3 | softened scalar | 16 | 16 | 1e-5 | v01 |

Then repeat winner on:

- v01+v02 filtered,
- v01+v02+v03 filtered once v03 is available.

### Step 5 — Select final recipe

Choose based on:

- eval delta,
- EMA smoothness,
- length safety,
- risk count,
- per-family stability,
- qualitative samples.

---

## 10. Proposed final larger run recipe

Tentative target after ablations:

```yaml
model_name: "jayshah5696/gemma4-e2b-humanize-unsloth-merged"
task_path: "data/rl/humanize_tasks_rl_mix_v1_filtered.jsonl"

max_steps: 400-800
num_generations: 16
gradient_accumulation_steps: 16
per_device_train_batch_size: 1

learning_rate: 1.0e-5
warmup_ratio: 0.1
weight_decay: 0.001
max_grad_norm: 0.5

optim: adamw_8bit
loss_type: bnpo  # or dr_grpo if ablation wins
scale_rewards: batch  # if supported and ablation wins

epsilon: 0.2
epsilon_high: 0.28
delta: 1.5
mask_truncated_completions: true

max_completion_length: 768-1024
temperature: 1.0

reward_mode: scalar_softened
penalty_cap: 1.0

stratify_batches: true
stratify_by:
  - dataset_version
  - reward_profile
  - family
  - length_bucket
  - difficulty_bucket

use_vllm: true
vllm_mode: colocate
vllm_gpu_memory_utilization: 0.55-0.65

report_to: wandb
wandb_project: humanize-rl
push_to_hub: false
```

Expected final behavior:

- smoother `reward/ema_20`,
- positive eval deltas across v01/v02/v03 heldouts,
- no length collapse,
- lower risk penalty rate,
- modest ridge/style improvement,
- stable reward_std,
- clipped ratio near zero.

---

## 11. Final run success criteria

### Training gates

| gate | threshold |
|---|---:|
| training completes | yes |
| NaN | none |
| clipped ratio | `< 0.05` |
| zero-std group rate | `< 0.10` |
| dither rate | `< 0.10` |
| peak VRAM | `< 38 GB` on A100-40GB |
| actual cost | within approved cap |
| reward EMA slope | positive |
| response length | no collapse |

### Eval gates

Run eval on all heldout splits:

```text
v01 validation/test
v02 validation/test
v03 validation/test
hard penalty set
long-form set
```

Pass if:

| metric | threshold |
|---|---:|
| mean reward delta on v01 | `> +0.02` |
| mean reward delta on v02 | `> +0.02` |
| mean reward delta on v03 | `> +0.02` once available |
| risk negative count | not increased |
| mean risk penalty | not worse |
| per-family reward | no family drops > 0.05 without explanation |
| human side-by-side inspection | pass |

### Push gate

Only push LoRA if:

1. training gates pass,
2. eval gates pass,
3. qualitative inspection passes,
4. W&B artifacts and JSON summaries are saved.

Push as:

```text
candidate-v1
```

not final, until larger manual review.

---

## 12. What “smooth learning” should mean

Do not judge smoothness from raw `train/reward`.

Use:

1. `reward/ema_20` and `reward/ema_50`,
2. component EMAs,
3. per-family eval,
4. heldout eval deltas,
5. response length stability,
6. penalty-rate trend.

A good run can still have jagged raw reward. A bad run is one where:

- EMA is flat/down,
- eval does not improve,
- reward comes only from shorter outputs,
- risk improves but style drops,
- one task family collapses,
- dither/zero-std rates are high.

---

## 13. Recommended immediate next actions

1. Implement `reward_mode` with:
   - `current_components`,
   - `scalar_current`,
   - `scalar_softened`.
2. Move component reward functions to diagnostics for scalar modes.
3. Add W&B EMA/component/dataset/family logging.
4. Create v01+v02 dataset mixer.
5. Create offline difficulty scoring script.
6. Run four 30–50 step ablations in parallel on Modal.
7. Select final config.
8. Repeat on v01+v02 filtered.
9. Once v03 is ready, rerun difficulty filtering and final larger run.

---

## 14. Vertical implementation slices

These slices are ordered so each one produces a useful artifact and can stop the project early if the hypothesis fails. Run them as small, reviewable PRs. Do not jump straight to the larger final run.

### Slice 0 — Freeze current baseline artifacts

**Goal:** make the Slice 3 PASS reproducible before changing reward/dataset code.

**Ships:**

- `outputs/baseline_eval.json`
- `outputs/post_train_eval.json`
- `outputs/full_run/summary.json`
- `outputs/full_run/final_adapter/`
- W&B run link/name recorded in the plan
- a small `scripts/eval/compare_rl_eval.py` helper if not already present

**Run:**

```bash
rtk uv run python scripts/eval/check_rl_run_summary.py \
  outputs/full_run/summary.json \
  --phase full
```

```bash
rtk uv run python - <<'PY'
import json
pre = json.load(open("outputs/baseline_eval.json"))
post = json.load(open("outputs/post_train_eval.json"))
print(post["mean_reward"] - pre["mean_reward"])
PY
```

**Acceptance:**

- full checker passes,
- eval delta remains `> +0.02`,
- current adapter is not overwritten.

---

### Slice 1 — Reward-mode refactor without behavior change

**Goal:** support multiple reward modes while proving the default path is unchanged.

**Ships:**

- `src/humanize_rl/reward/grpo_rewards.py`
  - `reward_mode="current_components"`
  - `reward_mode="scalar_current"`
  - `reward_mode="scalar_softened"` stubbed or implemented behind config
- config fields:
  - `reward_mode`
  - `penalty_cap`
  - `risk_weight`
  - `ridge_weight`
  - `deterministic_weight`
- tests proving:
  - current component reward sum equals old scalar reward,
  - `scalar_current` equals current strict reward,
  - `scalar_softened` is bounded and monotonic with penalties,
  - dither applies only to final scalar in scalar modes.

**Run:**

```bash
rtk uv run pytest tests/reward/test_grpo_dither.py \
  tests/reward/test_grpo_adapters.py \
  tests/reward/test_reward_modes.py -q
```

**Acceptance:**

- local tests pass,
- current default behavior unchanged,
- no Modal spend.

---

### Slice 2 — W&B diagnostics and EMA logging

**Goal:** stop relying on raw `train/reward` as the only graph.

**Ships:**

- trainer callback or summary hook that logs:
  - `reward/ema_20`, `reward/ema_50`
  - `ridge/ema_20`
  - `deterministic/ema_20`
  - `risk_penalty/ema_20`
  - `risk_compliance/ema_20`
  - `penalty_rate/ema_20`
  - `response_length_mean`, `response_length_p95`
  - `dither_rate`
- summary JSON includes all final EMA values.

**Run:**

A 5-step smoke only:

```bash
rtk uvx modal run --detach \
  src/humanize_rl/training/rl_gemma4_trl_vllm_modal.py \
  --mode train \
  --config-path /workspace/configs/rl/gemma4_e2b_rl_a100_logging_smoke.yaml
```

**Acceptance:**

- W&B run has EMA charts,
- summary JSON records EMA fields,
- smoke cost remains tiny,
- no reward-code semantic change required.

---

### Slice 3 — Dataset mixer for v01 + v02 + v03-ready schema

**Goal:** create one normalized task mix format without waiting for all v03 tasks.

**Ships:**

- `scripts/rl/build_rl_task_mix.py`
- output:
  - `data/rl/humanize_tasks_rl_mix_v1.jsonl`
  - `data/rl/humanize_tasks_rl_mix_v1_summary.json`
- row fields added:
  - `dataset_version`
  - `length_bucket`
  - `difficulty_bucket` default `unknown`
  - `source_group`
  - `mix_weight`
- supports inputs:
  - v01
  - v02
  - optional v03 draft path

**Run:**

```bash
rtk uv run python scripts/rl/build_rl_task_mix.py \
  --v01 data/rl/humanize_tasks_v01_smoke.jsonl \
  --v02 data/rl/humanize_tasks_v02.jsonl \
  --output data/rl/humanize_tasks_rl_mix_v1.jsonl \
  --summary outputs/rl_mix/humanize_tasks_rl_mix_v1_summary.json
```

**Acceptance:**

- schema validates,
- no duplicate task IDs,
- v01/v02 counts recorded,
- family/profile/version distribution visible,
- v03 can be added later without changing downstream training code.

---

### Slice 4 — Offline difficulty scoring

**Goal:** implement Prime-style difficulty filtering inside our TRL + Modal workflow.

**Ships:**

- `scripts/rl/score_task_difficulty_modal.py`
- outputs:
  - `outputs/rl_difficulty/rollouts.jsonl`
  - `outputs/rl_difficulty/task_difficulty.jsonl`
  - `data/rl/humanize_tasks_rl_mix_v1_filtered.jsonl`
- per-task metrics:
  - reward mean/std,
  - ridge mean/std,
  - deterministic mean/std,
  - risk penalty mean,
  - penalty pattern diversity,
  - response length mean/p95,
  - clipped rate.

**Run:**

```bash
rtk uvx modal run scripts/rl/score_task_difficulty_modal.py \
  --task-path data/rl/humanize_tasks_rl_mix_v1.jsonl \
  --output-dir outputs/rl_difficulty/mix_v1 \
  --filtered-output data/rl/humanize_tasks_rl_mix_v1_filtered.jsonl \
  --rollouts-per-task 8
```

**Acceptance:**

- every kept task has `reward_std >= 0.03`,
- kept tasks satisfy `0.20 <= mean_reward <= 0.80` unless manually whitelisted,
- filtered dataset still preserves family/profile/version diversity,
- at least 50% of v01+v02 survives or thresholds are revisited.

---

### Slice 5 — Reward scalarization ablation pack

**Goal:** validate whether scalar reward reduces noise before changing batch size.

**Ships:**

- configs:
  - `configs/rl/ablations/reward_a0_current_components.yaml`
  - `configs/rl/ablations/reward_a1_scalar_current.yaml`
  - `configs/rl/ablations/reward_a2_scalar_softened.yaml`
- each runs 30–50 steps on the same small task set.

**Run in parallel:**

```bash
rtk uvx modal run --detach src/humanize_rl/training/rl_gemma4_trl_vllm_modal.py \
  --mode train --config-path /workspace/configs/rl/ablations/reward_a0_current_components.yaml
```

```bash
rtk uvx modal run --detach src/humanize_rl/training/rl_gemma4_trl_vllm_modal.py \
  --mode train --config-path /workspace/configs/rl/ablations/reward_a1_scalar_current.yaml
```

```bash
rtk uvx modal run --detach src/humanize_rl/training/rl_gemma4_trl_vllm_modal.py \
  --mode train --config-path /workspace/configs/rl/ablations/reward_a2_scalar_softened.yaml
```

**Acceptance / winner rule:**

Pick the mode with:

- positive eval delta,
- best `reward/ema_20` smoothness,
- no length collapse,
- no increase in risk-negative count,
- no component collapse.

Expected winner: `scalar_softened`.

---

### Slice 6 — Batch/group-size + LR ablation pack

**Goal:** test Prime-style stability recommendations on our Modal budget.

**Ships:**

- configs:
  - `configs/rl/ablations/stability_b0_g8_acc8_lr2e5.yaml`
  - `configs/rl/ablations/stability_b1_g16_acc16_lr1e5.yaml`
  - `configs/rl/ablations/stability_b2_g16_acc8_lr1e5.yaml`

**Run:** same detached Modal command with each config.

**Acceptance / winner rule:**

- no OOM,
- peak VRAM `< 38 GB`,
- clipped ratio `< 0.05`,
- reward EMA smoother than B0,
- eval delta positive,
- projected final cost acceptable.

Expected winner: `G=16, grad_accum=16, LR=1e-5` if it fits; otherwise `G=16, grad_accum=8`.

---

### Slice 7 — Reward scaling ablation

**Goal:** decide whether TRL reward scaling should use group, batch, or no scaling.

**Ships:**

- a preflight test that confirms installed TRL accepts `scale_rewards`,
- configs:
  - `scale_c0_group.yaml`
  - `scale_c1_batch.yaml`
  - `scale_c2_false.yaml`

**Run:** 30–50 steps each using the best reward mode from Slice 5 and best group/LR from Slice 6.

**Acceptance / winner rule:**

- positive eval delta,
- no unstable grad spikes,
- no reward_std collapse,
- smoother EMA than group scaling.

Expected winner: `scale_rewards: batch`.

---

### Slice 8 — Filtered v01+v02 pilot

**Goal:** verify that the chosen recipe works beyond v01 smoke.

**Ships:**

- `configs/rl/gemma4_e2b_rl_a100_mix_v1_pilot.yaml`
- run summary:
  - `outputs/mix_v1_pilot_summary.json`
- evals:
  - v01 control eval,
  - v02 control eval,
  - mixed validation eval.

**Run:**

```bash
rtk uvx modal run --detach \
  src/humanize_rl/training/rl_gemma4_trl_vllm_modal.py \
  --mode train \
  --config-path /workspace/configs/rl/gemma4_e2b_rl_a100_mix_v1_pilot.yaml
```

**Acceptance:**

- training completes,
- eval improves on mixed validation,
- v01 does not regress,
- v02 improves or stays neutral,
- per-family drops are investigated,
- cost projection for final run remains acceptable.

---

### Slice 9 — Add v03 and rerun difficulty filtering

**Goal:** incorporate the richer v03 task distribution once available.

**Ships:**

- `data/rl/humanize_tasks_rl_mix_v2.jsonl`
- `data/rl/humanize_tasks_rl_mix_v2_filtered.jsonl`
- distribution report:
  - v01/v02/v03 ratios,
  - family balance,
  - mode balance,
  - length buckets,
  - difficulty buckets.

**Acceptance:**

- v03 schema validates,
- v03 quality gates from `docs/plans/v03-rl-tasks-dataset.md` pass,
- filtered mix has enough useful-difficulty tasks,
- heldout splits exist for v01/v02/v03.

---

### Slice 10 — Final larger run candidate

**Goal:** run the selected stable recipe on the filtered v01+v02+v03 mix.

**Ships:**

- `configs/rl/gemma4_e2b_rl_a100_stable_full.yaml`
- `outputs/stable_full/summary.json`
- `outputs/stable_full/final_adapter/`
- eval outputs:
  - `outputs/stable_full/eval_v01.json`
  - `outputs/stable_full/eval_v02.json`
  - `outputs/stable_full/eval_v03.json`
  - `outputs/stable_full/eval_hard_penalty.json`
  - `outputs/stable_full/eval_longform.json`
- W&B run with full EMA/component dashboard.

**Run:**

```bash
rtk uvx modal run --detach \
  src/humanize_rl/training/rl_gemma4_trl_vllm_modal.py \
  --mode train \
  --config-path /workspace/configs/rl/gemma4_e2b_rl_a100_stable_full.yaml
```

**Acceptance:**

- all training gates in §11 pass,
- all eval gates in §11 pass,
- qualitative side-by-side inspection passes,
- LoRA can be pushed as `candidate-v1`.

---

### Slice 11 — Candidate push and model card

**Goal:** publish only after final gates pass.

**Ships:**

- HF LoRA repo, candidate tag only,
- model card with:
  - base model,
  - SFT checkpoint,
  - RL dataset mix,
  - reward mode,
  - eval tables,
  - known limitations,
  - W&B run link,
  - Modal app IDs.

**Acceptance:**

- repo loads via PEFT,
- adapter eval reproduces local/Modal score,
- card clearly says `candidate`, not final.

---

## 15. Summary recommendation

The current run should be treated as proof of feasibility, not the final recipe.

Best next direction:

> TRL + Modal, scalar softened reward, larger rollout groups, larger accumulated batch, difficulty-filtered v01+v02+v03 mix, and W&B EMA/component logging.

Most likely winning recipe:

```text
reward_mode: scalar_softened
num_generations: 16
gradient_accumulation_steps: 16
learning_rate: 1e-5
scale_rewards: batch
training data: difficulty-filtered v01+v02+v03
```

The purpose of the ablations is to prove or falsify that recipe cheaply before spending on a larger final run.
