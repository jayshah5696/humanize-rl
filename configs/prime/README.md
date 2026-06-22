# Prime p50 sweep configs

Use these after refreshing Prime auth and pushing env `0.3.14`.
All configs set train/eval `max_tokens = 4096`; Qwen configs disable thinking
in both train and eval sampling.
Prime docs list `pre_batch_filters` / `post_batch_filters`, but CLI `0.6.14`
rejects those sections as extra inputs. Keep these TOMLs runnable under the
installed CLI; env `0.3.14` handles repetition/runaway, the observed
uplifting-romance recommendation leak, task-unrequested emoji/hashtag/all-caps
surface failures, and formal salutation/signature leaks inside deterministic
reward. It also filters prompt/email scaffold out of required facts and caps
p50's deterministic half to zero on semantic failures, including missing facts,
invented numbers/times, unsupported hostile phrasing, and low source overlap.
It also ignores title/discourse false entities such as `Compliance Review`,
`Firstly`, `Secondly`, and `Understanding`, and caps hosted `r8` failures such
as instruction leakage, polite closing leakage, exact phrase/structure misses,
and too-short creative outputs.

```bash
prime --plain train models --output json
prime --plain env push --path environments/humanize_rl_env
```

Local `prime eval run` caveat: do not pass `enable_thinking` via
`--sampling-args`; the local OpenAI-compatible client rejects that keyword. Use
plain `--max-tokens 4096 --temperature 0.0` for local eval smoke. Hosted
Training eval uses `temperature = 0.2`; `r9` showed Qwen greedy hosted eval at
`0.0` can run to the full `4096` cap. Keep `enable_thinking = false` in these
TOML configs for Hosted Training.

Run order:

```bash
prime --plain train configs/prime/qwen35_08b_docs_debug.toml
prime --plain train configs/prime/qwen35_08b_debug.toml
prime --plain train configs/prime/qwen35_2b_p5050.toml
```

`qwen35_08b_docs_debug.toml` is a one-step RL smoke, not a single-rollout
eval. It uses `rollouts_per_example = 4` because Prime applies post-batch
`zero_advantage` filtering by default; one rollout per prompt cannot produce
within-group advantage and can drop the whole train batch.

`qwen35_08b_debug.toml` is currently train-only for the 50-step smoke. Hosted
eval is intentionally absent because docs-debug `r10` still truncated hosted
Qwen eval completions at `4096/4096/4096` on every eval row, while train
generation had `0%` truncation and `0%` errors. After this run completes,
download rollouts and run `scripts/eval/audit_prime_rollouts.py`; evaluate the
checkpoint separately before scaling to 2B.

The first train-only attempt (`dlh5ecoj1niiznfu8qp6mbxn`, `r1`) was stopped
before step 0 because `128` simultaneous rollouts repeatedly hit Qwen renderer
`nan` request errors and stalled at `26/128` batch progress. The second
attempt (`e7pie3x7j6uywy5rfwy407ct`, `r2`) capped inflight rollouts at `16`
but still stalled before step 0 at `+26` buffered samples. The third attempt
(`ee1tk85aludc6606vqv5y2bs`, `r3`) copied the docs-debug `r10` train geometry
but still stalled before step 0 with `max_tokens = 4096`. The current `r4`
config keeps the simple geometry (`batch_size = 16`, `rollouts_per_example =
4`, `max_inflight_rollouts = 16`, `max_steps = 50`, no hosted eval) and lowers
training `max_tokens` to `1024`. `4096` is documented as unstable for this
Prime/Qwen train renderer path.

Do not start `qwen35_2b_p5050.toml` until a tiny 0.8B smoke passes. Run
`beixwu41osp530um7bfabn8k` was stopped at step 21 after p50 reward increased
while truncation and repetition exploded. Env `0.3.6` fixed that failure mode,
`0.3.7` fixed most unsuitable romance recommendations, `0.3.8` added
emoji/hashtag/all-caps diagnostics after user audit of the W&B samples, and
`0.3.9` adds formal salutation/signature caps after the `0.3.8` smoke still
rewarded letter-like outputs. The `0.3.9` 50-step smoke exposed high-reward
hallucinations that missed required facts; `0.3.13` added scaffold filtering,
semantic caps, unsupported-detail diagnostics, AI-tell surface caps, and
subject/discourse false-entity filtering. The `0.3.13` hosted `r8` one-step
then exposed instruction leakage, polite closing leakage, exact-constraint
misses, and too-short creative overreward; `0.3.14` patches those and must pass
the same one-step and 50-step smoke before scale-up.

Use Llama 3.2 3B only as fallback if Qwen eval remains reasoning-only or empty
after the env handles empty content without crashing.

`sprints/Llama-3.2-1B-Instruct` is free but rejects this custom env under
Prime's free-tier environment requirements.

Live model list checked with Prime CLI v0.6.14 after publishing `0.3.14`:

| model | train $/MTok | status |
|---|---:|---|
| `Qwen/Qwen3.5-0.8B` | 0.06 | available |
| `Qwen/Qwen3.5-2B` | 0.15 | available |
| `Qwen/Qwen3.5-4B` | 0.30 | available |
| `Qwen/Qwen3.5-9B` | 0.60 | available |
| `Qwen/Qwen3.5-35B-A3B` | 1.00 | available |
| `Qwen/Qwen3.5-397B-A17B` | 4.00 | available |
| `Qwen/Qwen3.6-35B-A3B` | 1.00 | available |
| `meta-llama/Llama-3.2-1B-Instruct` | 0.06 | available |
| `meta-llama/Llama-3.2-3B-Instruct` | 0.15 | available target |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | 0.60 | available |
| `openai/gpt-oss-20b` | 0.40 | available |
| `poolside/Laguna-XS.2` | 0.00 | free baseline |
| `sprints/Llama-3.2-1B-Instruct` | 0.00 | free tier rejects custom env |

Primary training env args:

```toml
args = { split = "train", task_set = "mix_v2_p5050", reward_mode = "p50_50_no_penalty" }
```

Eval gates in every config:

- `mix_v2_p5050` validation with `p50_50_no_penalty`
- `v02_smoke` validation with `strict`
- `v03` validation with `strict`
