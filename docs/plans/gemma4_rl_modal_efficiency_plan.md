# Gemma 4 E2B RL on Modal: GPU Efficiency Plan

## Goal

Run Gemma 4 E2B GRPO as fast as practical on Modal A100-40GB once the starting policy artifact is fixed.

This plan is intentionally aggressive about using the full GPU. The smoke run used only about 10GB of a 40GB A100. The next objective is to spend the unused 30GB on larger microbatches, more GRPO generations, and longer completions.

Constraints for this plan:

- Starting model will be fixed separately. Ignore the current bad merge for efficiency planning.
- Reward is fast and stays intact. Do not remove reward components to save time.
- Use 16-bit/bf16 LoRA, not QLoRA.
- Use Modal detached runs only.
- Use A100-40GB unless a benchmark proves another GPU is better per dollar.
- Avoid clipped completions for real runs; clipping corrupts the RL signal.
- Do not make many cautious intermediate runs. Do one max-capacity probe, one adjusted probe if needed, then full run.

## Current measured baseline

Smoke config:

```yaml
max_steps: 2
num_generations: 2
per_device_train_batch_size: 1
gradient_accumulation_steps: 2
max_prompt_length: 384
max_completion_length: 64
sample_before_after: 2
generation_batch_size: 2
```

Observed on Modal A100-40GB:

- model + LoRA memory after attach: about `9.7GB / 39.5GB`;
- model load: `23-40s` after cache warmup;
- 2 GRPO steps: about `50s` train runtime;
- full smoke wall: about `115s`;
- GPU cost: about `$0.07` at `$0.000583/sec`.

Important: the smoke was intentionally tiny and conservative. It proved wiring, not performance.

## Available RL taskset today

Current task file:

```text
data/rl/humanize_tasks_v01_smoke.jsonl
```

Inventory:

```text
rows: 100
splits: train=80, validation=10, test=10
families: 10 families x 10 rows each
reward profiles:
  rewrite_faithful_concise: 40
  direct_workplace_message: 30
  compression_update: 10
  technical_explain_natural: 10
  sensitive_comms: 10
```

Families:

```text
rewrite_repair
direct_email
slack_chat
compression
tone_shift
technical_explain
product_copy_cleanup
candidate_customer_comms
adversarial_ai_tell_removal
placeholder_discipline
```

This is a smoke/pilot taskset, not a 5k-10k full taskset. For now, the fastest useful path is:

1. use this 100-row taskset for max-capacity utilization probes;
2. generate or point to the larger RL taskset;
3. run the full taskset with the best config from the probes.

## Main bottleneck hypothesis

The bottleneck is model rollout + backward compute, not reward scoring.

The fastest path is to increase useful GPU work per step:

1. larger per-device batch;
2. more generations per prompt;
3. longer, non-clipped completions;
4. fixed-size padding/bucketing to reduce shape churn;
5. keep the GPU loaded while avoiding OOM and NaNs.

## Completion length policy

For real RL, do not cap completions at 64/128/192 just because the smoke did. The smoke cap existed only to prove plumbing cheaply.

A high cap is not the same thing as clipping. `max_completion_length` is the upper bound. If the model naturally stops at 80 tokens under a 2048-token cap, nothing is clipped. If it needs 500 tokens, a 192 cap would corrupt the reward signal.

Aggressive policy:

```yaml
max_completion_length: 2048
```

This is the cleanest way to avoid underestimating full-taskset runtime and avoid training on truncated answers.

Risks of `2048` are runtime and memory, not reward correctness. So the plan is to test `2048` directly on A100-40GB rather than assume it is too expensive.

A real run should fail its config gate if `completions/clipped_ratio` is high. High clipping means the model is hitting the cap and the cap is still too low for the task distribution.

## Max-capacity utilization probes

We are using only about 10GB after model + LoRA attach, so the next work is not a long conservative ladder. It is two aggressive probes:

### Probe 1: max-context A100 probe

```yaml
max_seq_length: 3072
max_prompt_length: 1024
max_completion_length: 2048
num_generations: 4
per_device_train_batch_size: 4
gradient_accumulation_steps: 1
max_steps: 2
```

Purpose:

- test real long-completion memory behavior;
- test whether Unsloth/TRL can actually batch GRPO generations for Gemma 4;
- measure wall time and VRAM under a realistic no-clipping cap.

### Probe 2: adjust only if Probe 1 fails or leaves too much VRAM idle

If Probe 1 OOMs:

```yaml
per_device_train_batch_size: 2
num_generations: 4
max_completion_length: 2048
max_seq_length: 3072
```

If Probe 1 succeeds and peak VRAM is still below ~30GB:

```yaml
per_device_train_batch_size: 6-8
num_generations: 4
max_completion_length: 2048
max_seq_length: 3072
```

If Probe 1 succeeds but is slow because generation is serial inside TRL/Unsloth, the bottleneck is framework behavior, not A100 memory. Document it and ask for Unsloth/TRL help.

Decision metric is not just memory. Pick the config with best:

```text
examples_per_second
completion_tokens_per_second
optimizer_steps_per_hour
cost_per_1k_train_tasks
stable KL / no NaN
clipped_ratio near 0
```

## Sequence length and padding

For the aggressive A100 run, start with:

```yaml
max_prompt_length: 1024
max_completion_length: 2048
max_seq_length: 3072
```

Why:

- it uses the A100 budget instead of protecting a tiny memory footprint;
- it avoids false confidence from short completions;
- it exposes the actual full-run runtime cost;
- it lets the model terminate naturally for almost all workplace-writing tasks.

If prompts exceed 1024 tokens in the full taskset, increase `max_seq_length` rather than silently truncating important input. If full taskset prompt p99 is short, the 3072 cap is still acceptable for Probe 1 because the point is capacity testing.

Use left padding for decoder-only generation. Consider bucketing prompts by rendered token length so each batch has less padding waste.

## Full taskset sizing

The repo currently has a 100-row RL smoke/pilot taskset, not a full 5k-10k RL taskset.

Current file:

```text
data/rl/humanize_tasks_v01_smoke.jsonl = 100 rows
```

Before estimating full cost beyond the existing 100 rows, create or point config to the real taskset, then compute:

- train row count;
- validation row count;
- prompt token p50/p90/p99;
- target completion token p50/p90;
- family mix.

Add a local report command or test that prints:

```text
rows_train
rows_validation
prompt_tokens_p50/p90/p99
recommended max_prompt_length
recommended max_completion_length
```

## Cost model

Modal A100-40GB current listed rate:

```text
$0.000583/sec ~= $2.10/hour
```

Approximate cost:

```text
cost = wall_clock_hours * 2.10
```

Runtime should be measured from canaries, not inferred from the 2-step smoke. Use:

```text
seconds_per_train_task = train_runtime_seconds / tasks_consumed
cost_per_1k_tasks = seconds_per_train_task * 1000 * 0.000583
```

Do not estimate full-run cost until the max-capacity probes run.

For the existing 100-row taskset, cost should be measured directly by running the full 80-train-row split after the 2-step probe. That run is small enough to be the practical benchmark.

## Expected full-run cost bands after utilization tuning

These are planning ranges, assuming a fixed model artifact and stable A100 run:

| Train rows | Good utilization target | Cost band |
|---:|---:|---:|
| 500 | under 30 min | under `$1.25` |
| 1,000 | 30-90 min | `$1-$3` |
| 3,000 | 1.5-4 hr | `$3-$9` |
| 10,000 | 5-12 hr | `$10-$25` |

If a 1k run costs much more than this on A100-40GB, utilization is poor or completions are much longer than expected.

## Instrumentation needed before full RL

Add to `rl_gemma4_modal.py` summary:

- peak allocated/reserved VRAM;
- train runtime;
- total runtime;
- steps/sec;
- examples/sec;
- generated completion tokens/sec if exposed by TRL logs;
- mean/min/max completion length;
- clipped ratio;
- KL mean/max;
- grad norm mean/max;
- NaN detector;
- Modal GPU type and rate assumption.

Write every run summary to:

```text
/checkpoints/<experiment>/summary.json
/checkpoints/<experiment>/trainer_state.json
/checkpoints/<experiment>/run_metrics.json
```

## Probe protocol: max 2 tests, then full run

Do not run a long conservative ladder.

### Test 1: aggressive A100 2-step probe

```yaml
experiment_name: gemma4-e2b-humanize-rl-a100-max-probe-01
max_steps: 2
max_seq_length: 3072
max_prompt_length: 1024
max_completion_length: 2048
num_generations: 4
per_device_train_batch_size: 4
gradient_accumulation_steps: 1
sample_before_after: 2
push_to_hub: false
report_to: none
```

### Test 2: one adjustment only

Run only one of these:

- if Test 1 OOMs: reduce batch to `2`, keep completion `2048`;
- if Test 1 uses under 30GB: increase batch to `6` or `8`, keep completion `2048`;
- if Test 1 is stable and uses enough VRAM: skip Test 2.

Detached command:

```bash
uvx modal run --detach src/humanize_rl/training/rl_gemma4_modal.py --mode train --config-path /workspace/configs/rl/<probe>.yaml
```

Acceptance:

- completes 2 steps;
- no OOM;
- no NaN grad/loss;
- clipped ratio near 0;
- KL not exploding from the fixed starting policy;
- peak VRAM approaches useful A100 utilization, ideally 25-36GB;
- runtime per generated token is acceptable.

## Full-run config target

Start from the max-capacity probe result. Expected aggressive shape:

```yaml
max_seq_length: 3072
max_prompt_length: 1024
max_completion_length: 2048
lora_rank: 16
lora_alpha: 32
learning_rate: 5.0e-6   # lower only if KL says so, not for conservatism
max_steps: null         # derive from epochs/taskset
num_generations: 4
per_device_train_batch_size: 4-8
gradient_accumulation_steps: 1
optim: adamw_8bit
loss_type: bnpo
mask_truncated_completions: true
temperature: 0.8-1.0
sample_before_after: 16-32
push_to_hub: false
```

If KL remains high, optimize stability before speed:

- lower LR;
- increase beta/KL control if available in current TRL config;
- reduce `num_generations` only if needed;
- use a cleaner reference policy path.

## What may require outside help

If A100 memory remains low while throughput does not improve after increasing batch/generations, the limitation is likely inside Unsloth/TRL Gemma 4 GRPO generation rather than Modal resources.

Ask for help specifically on:

- TRL/Unsloth GRPO batching behavior for Gemma 4;
- whether `per_device_train_batch_size` and `num_generations` are being used as expected;
- avoiding slow serial generation inside `UnslothGRPOTrainer`;
- Gemma 4 processor fields like `mm_token_type_ids` in text-only GRPO;
- safe use of compile after fixing shape churn.

## Execution order

1. Fix merged model artifact.
2. Create full RL taskset or identify the real task file.
3. Add task token-length report.
4. Add run metrics/VRAM/cost summary.
5. Run detached candidate A/B/C/D canaries.
6. Pick fastest stable config.
7. Run 100-200 step pilot.
8. Review KL, grad norm, clipped ratio, reward trend, generations.
9. Run full taskset.
10. Push RL LoRA only after eval passes.
