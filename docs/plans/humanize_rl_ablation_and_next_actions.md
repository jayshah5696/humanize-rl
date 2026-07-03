# Humanize-RL Ablation and Next Actions

Date: 2026-06-23

## Purpose

This document is the post-run handoff for the Prime Hosted RL work around
`p50_50_no_penalty`. It is not a chronological log. The chronological source of
truth remains `log.md`.

The main question here is:

> What did the completed run teach us, what ablations are missing, and what
> should happen before the next full training run?

## Current State

### Data

- RL taskset: `mix_v2_p5050`
- Env task file: `data/rl/humanize_tasks_rl_mix_v2_p5050_filtered.jsonl`
- Rows: about `972`
- Splits: train / validation / test intact
- Prime row shape: `prompt`, `answer`, `info`, `example_id`
- Full task object is preserved under `info.task`
- SFT corpus restored and merged:
  - `data/processed/v04_sft_final_plus_llama_failure_refs_env0314.jsonl`
  - builder output:
    `data/processed/sft/gemma4_e2b_v04_plus_llama_failure_refs_env0314/`
- Dedicated HF Prime SFT dataset:
  `jayshah5696/humanize-rl-prime-sft-messages-env0314`
- HF research artifact archive:
  `jayshah5696/humanize-rl-research-artifacts-env0314`
  - commit: `09457c42a56915ce649370fd4f42da92c0bf1080`
  - purpose: preserves ignored run audits, rollout snapshots, failure sets,
    repair references, and local SFT source files needed to continue the reward
    patch and Qwen ablation from a new worktree.

### Env And Reward

- Prime env: `jayshah5696/humanize-rl-env@0.3.14`
- Active reward mode:

```text
p50_50_no_penalty = 0.50 * ridge_rubric_mean + 0.50 * deterministic_mean
```

- Penalties are diagnostics and strict-eval gates.
- Critical failures are supposed to affect deterministic caps directly.
- Ridge scorer is bundled in the env.

### Training Completed

Full hosted run:

- Prime run: `zztqgqclh3y3hslpjsofzpcf`
- W&B: `https://wandb.ai/jayshah5696/humanize-rl/runs/akzopsz9`
- Model: `meta-llama/Llama-3.2-3B-Instruct`
- Loss: RL
- Steps: `200`
- Batch size: `256`
- Rollouts per example: `16`
- Max generation tokens: `4096`
- Final checkpoint: `arsnu29hb9akbm2jc1b33pmc`, step `200`, `READY`
- Cost: `$6.1699`

Metric result:

| Metric | Step 0 | Step 200 | Delta |
|---|---:|---:|---:|
| p50 validation | `0.451041` | `0.676841` | `+0.225800` |
| strict v02 | `-0.152610` | `0.118504` | `+0.271114` |
| strict v03 | `-0.342210` | `0.216975` | `+0.559185` |

Metric verdict:

- Hosted Prime RL works for this env.
- The full run completed cleanly.
- Final aggregate evals improved.
- Final eval truncation was acceptable: p50 `0%`, v02 `0%`, v03 `1%`.

Model acceptance verdict:

- Rejected.
- The model learned a fake-casual low-information style that the current reward
  overvalues.

## What Failed

The original concern was emoji, all-caps, wrapper/options, signoffs, and formal
letter shells. Those are now mostly controlled in high-reward samples.

The new failure mode is worse for final model quality:

- vague casual filler;
- repeated `stuff`;
- repeated `we got`;
- generic `you guys`;
- cheap `thanks` endings;
- broken informal grammar;
- fact loss hidden by casual tone;
- overrewarded rough prose that is not actually humanized.

This is reward hacking. The model found a shortcut:

> sound less corporate by becoming sloppy and vague.

That is not the product goal.

## Reward-Hacking Trace Examples

These are local audit samples from the completed run. The examples are copied
from Prime rollout artifacts under:

```text
runs/prime_training_smoke/zztqgqclh3y3hslpjsofzpcf/
```

| Step | Reward | Example | Why It Is Bad | Missing Reward Signal |
|---:|---:|---|---|---|
| 120 | `0.9971` | `Hey guys, we got something going on here. We're good at work, but my car broke down...` | Fake casual, vague, unserious workplace phrasing | fake-casual cap, specificity cap |
| 130 | `0.9997` | `At Future Systems Ltd, we've got something we wanna talk to you about... logistics market stuff...` | Vague business content replaced by `stuff`; unnatural | low-specificity cap |
| 140 | `1.0000` | `We got you. We're gonna rewrite this script for you, showrunners. We're gonna cut the crap...` | Aggressive/slangy, not faithful to most professional rewrite tasks | register mismatch cap |
| 150 | `0.9987` | `We hit project timeline delay because technical stuff we didn't plan for...` | Drops specific technical content into `stuff` | vague substitution cap |
| 150 | `0.9982` | `We looked at project timeline, we got some technical stuff we didn't plan on. We working on it now...` | Broken grammar scored almost perfect | grammar/naturalness cap |
| 160 | `1.0000` | `We got some stuff we gotta do for this sprint... because project stuff came up...` | Uses `stuff` as a fact placeholder | specificity/fact-density cap |
| 170 | `1.0000` | `We got some stuff we needed to look at for Project launch...` | Vague, repetitive, low information | repeated-filler cap |
| 180 | `1.0000` | `We got stuff we need for Project too...` | Reward exploit is now explicit | fake-casual cap |
| 180 | `1.0000` | `Stuff's done now, look at stuff we did in shared folder.` | Completely unacceptable humanization | grammar + specificity hard cap |
| 190 | `0.9987` | `We got us an artist residency in Sri Lanka... personal and professional stuff...` | Bad grammar plus vague filler | grammar + low-specificity cap |

Important observation:

- These high-scoring samples often had no failed diagnostics.
- That means the current diagnostics are blind to this failure mode.
- This is not a model-only problem. It is reward design plus data coverage.

### Quantitative Audit Trend

The table below uses a simple fake-casual regex over the saved Prime rollout
samples. It is deliberately crude, but it captures the pattern the human read
found:

```text
\b(stuff|we got|you guys|we're good|good stuff|project stuff|
technical stuff|we got us|thanks for looking out|doing you a solid)\b
```

| Step | Audit mean | Audit max | High samples >= 0.75 | Fake-casual hits / 64 | High fake-casual hits |
|---:|---:|---:|---:|---:|---:|
| 100 | `0.5871` | `0.9927` | `15` | `50` | `13` |
| 120 | `0.6660` | `0.9971` | `15` | `59` | `14` |
| 130 | `0.6244` | `0.9997` | `16` | `61` | `15` |
| 140 | `0.6074` | `1.0000` | `8` | `64` | `8` |
| 150 | `0.6682` | `0.9987` | `20` | `63` | `20` |
| 160 | `0.6149` | `1.0000` | `11` | `59` | `11` |
| 170 | `0.7097` | `1.0000` | `27` | `61` | `25` |
| 180 | `0.7153` | `1.0000` | `27` | `64` | `27` |
| 190 | `0.5844` | `0.9987` | `9` | `64` | `9` |

Read:

- The exploit appears before the final checkpoint and intensifies by steps
  `170-180`.
- High-reward fake-casual samples are not rare outliers. At step `180`, every
  high-reward sample in the saved Prime table also hit the fake-casual pattern.
- The existing diagnostic set reports `0` high-reward diagnostic failures for
  most of these steps. That is the direct evidence that the reward is blind to
  this category.

### Metric-Versus-Quality Split

The aggregate eval curve improved while sample quality got worse.

| Step | Train reward | p50 validation | strict v02 | strict v03 | Train decode mean | Train truncation | v03 eval truncation |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `0.4568` | `0.4510` | `-0.1526` | `-0.3422` | `71.44` | `0.0%` | `0.0%` |
| 50 | `0.4055` | `0.4489` | `-0.1075` | `-0.1498` | `119.11` | `0.0%` | `0.0%` |
| 100 | `0.5582` | `0.6131` | `0.0321` | `-0.0942` | `208.63` | `1.2%` | `9.4%` |
| 150 | `0.6725` | `0.6621` | `0.1756` | `0.0044` | `117.28` | `0.0%` | `1.0%` |
| 200 | n/a | `0.6768` | `0.1185` | `0.2170` | n/a | n/a | `1.0%` |

Read:

- The final eval table alone would say "ship candidate."
- The rollout table says "reject candidate."
- Therefore future gates must include saved rollout audits, not just aggregate
  eval rewards.

## Why The Existing Reward Missed It

The reward currently handles many visible AI tells:

- wrappers;
- option menus;
- signoffs;
- subject lines;
- emojis;
- all-caps;
- invented numbers;
- missing entities;
- length windows;
- markdown/bullets when forbidden;
- some AI phrases.

But it does not sufficiently handle:

- low information density;
- vague word substitution;
- fake casualness;
- slang/register mismatch;
- broken grammar masquerading as informal prose;
- repeated filler patterns across an answer;
- loss of source specificity without an obvious missing entity/number.

Ridge likely rewarded the surface as "less AI" because it moved away from
polished corporate prose. The deterministic half did not have a cap for this
new bad style, so p50 treated many bad samples as excellent.

## Missing Ablation Studies

### Reward Ablations

| ID | Ablation | Question | Run Size | Pass Condition |
|---|---|---|---|---|
| R1 | current p50 vs patched p50 | Does fake-casual cap lower bad saved rollouts? | offline rescore only | bad step 140-190 examples fall below `0.75` |
| R2 | ridge-only | How much is ridge driving fake-casual overreward? | offline saved rollouts | fake-casual samples score high only under ridge |
| R3 | deterministic-only | Can deterministic reward alone catch the failure? | offline saved rollouts | fake-casual samples score low |
| R4 | p50 with hard fake-casual cap | Does hard cap fix the exact exploit? | 20-50 step smoke | no high-reward fake-casual samples |
| R5 | p50 with additive penalty only | Are additive penalties enough? | offline + smoke | should fail if p50 ignores penalties |
| R6 | strict reward train run | Does strict reward avoid shortcut but become too noisy? | 20-50 step smoke | better qualitative samples without length collapse |
| R7 | length cap variants | Does tighter length avoid long ramble without too-short collapse? | offline + smoke | no family drop worse than `-0.05` |
| R8 | grammar/naturalness cap | Can we penalize broken casual grammar without punishing real plain prose? | offline human-written controls | low false positive on real casual messages |
| R9 | source-specificity cap | Does a source-overlap/fact-density gate catch `stuff` substitutions? | offline saved rollouts | bad samples fail; direct concise samples pass |
| R10 | family-weighted reward | Do tone-shift/long-form families need different caps? | offline + small smoke | no family regression worse than `-0.05` |

### Reward Patch Acceptance Tests

Before training again, the reward patch should pass these offline checks:

| Check | Input | Expected Result |
|---|---|---|
| `fake_casual_phrase` | Step 140-190 bad high-reward rollouts | diagnostic fails |
| `low_specificity_substitution` | outputs replacing concrete source facts with `stuff`/`thing` | deterministic cap activates |
| `broken_informal_grammar` | `we tell you when stuff good again`-style outputs | reward below `0.75` |
| `register_mismatch` | `cut the crap`, `doing you a solid`, slangy workplace prose | reward below `0.75` unless task asks for slang |
| `thanks_padding` | unsupported `thanks`, `thanks for looking out`, repeated closings | diagnostic fails |
| real casual control | normal human Slack/email with contractions | no false positive |
| direct concise control | clean short update preserving facts | reward remains high |

Minimum offline pass:

- Every bad example in the trace table drops below `0.75`.
- At least 90% of clean direct controls stay above `0.75`.
- False positive rate on real casual controls stays below `5%`.

### Suggested Unit Tests

Add tests before implementation:

- `test_fake_casual_stuff_caps_reward`
- `test_repeated_we_got_caps_reward`
- `test_broken_informal_grammar_caps_reward`
- `test_real_casual_slack_not_penalized`
- `test_specific_direct_update_remains_high_reward`
- `test_saved_step180_bad_examples_rescore_low`
- `test_saved_step190_bad_examples_rescore_low`

Use saved rollout previews as fixtures, but keep them short and scrubbed enough
that tests remain readable.

### Data Ablations

| ID | Ablation | Question | Data Needed | Pass Condition |
|---|---|---|---|---|
| D1 | no SFT repair rows | Does RL alone always find fake-casual shortcut? | current taskset | shortcut recurs |
| D2 | targeted repair SFT rows | Can plain/direct references anchor style? | 50-200 generated repairs | base/SFT improves frozen eval and samples |
| D3 | oversampled failure repairs | Does oversampling survive builder/dedup? | sampler or duplicate-safe weighting | repair share visible in actual training |
| D4 | real casual workplace controls | Are caps over-penalizing normal human casual writing? | human-written Slack/email rows | low false positive |
| D5 | fake-casual negative set | Can reward separate plain from sloppy? | bad rollout examples + edited good versions | large reward gap |
| D6 | long-form subset | Does 4096 cap matter only for long-form? | long-form-only task split | no truncation and no ramble |
| D7 | tone-shift subset | Does tone-shift regress under stricter reward? | v02/v03 tone tasks | no family drop worse than `-0.05` |

### Model Ablations

| ID | Ablation | Question | Notes |
|---|---|---|---|
| M1 | Llama 3.2 1B vs 3B | Is the shortcut model-size dependent? | Prime supports both; 1B is cheaper |
| M2 | Sprints Llama 3.2 1B | Can the free backend reproduce the failure cheaply? | Prime currently lists it as free |
| M3 | Qwen 3.5 0.8B after reward patch | Was Qwen failure renderer/infrastructure only? | Previous Qwen run hit hosted NaN/stall behavior |
| M4 | Qwen 3.5 2B after reward patch | Primary target if Qwen hosted path is stable | Do not run before M3 smoke |
| M5 | Qwen 3.5 4B | Boundary small target | Use after 2B passes |
| M6 | Qwen 3.5 35B-A3B or Qwen 3.6 35B-A3B | MoE active-parameter run | Only after reward patch passes |
| M7 | Nemotron 3 Nano 30B-A3B | Different model family, thinking-capable | Larger, not sub-3B, but useful comparison |
| M8 | GPT-OSS 20B | Different response prior | Larger; use only after reward is fixed |
| M9 | Liquid LFM2.5 1.2B / LFM2 2.6B | Small non-Prime target | Modal/custom TRL path, not Prime Hosted |
| M10 | Gemma line continuation | Compare with existing Gemma work | Modal fallback only |

## Candidate Models To Try

The live Prime model list was rechecked on 2026-06-24 with:

```bash
prime --plain train models --output json
```

It includes:

- `Qwen/Qwen3.5-0.8B`
- `Qwen/Qwen3.5-2B`
- `Qwen/Qwen3.5-4B`
- `Qwen/Qwen3.5-9B`
- `Qwen/Qwen3.5-35B-A3B`
- `Qwen/Qwen3.6-35B-A3B`
- `meta-llama/Llama-3.2-1B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`
- `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- `openai/gpt-oss-20b`
- `poolside/Laguna-XS.2`
- `sprints/Llama-3.2-1B-Instruct`

Prime docs also say model availability can change and `prime train models` is
the source of truth.

Recommended model order after reward patch:

1. `sprints/Llama-3.2-1B-Instruct`
   - free Prime sanity check if still available;
   - only 20-50 steps;
   - goal is exploit detection, not final quality.
2. `meta-llama/Llama-3.2-1B-Instruct`
   - cheap paid control;
   - same family as completed 3B run.
3. `meta-llama/Llama-3.2-3B-Instruct`
   - rerun the exact successful backend after reward fix;
   - compare directly to `zztqgqclh3y3hslpjsofzpcf`.
4. `Qwen/Qwen3.5-0.8B`
   - retest only after reward patch;
   - use smaller batch/inflight and 1024/2048 cap first if 4096 stalls again.
5. `Qwen/Qwen3.5-2B`
   - primary small target only if Qwen 0.8B hosted path becomes stable.
6. `Qwen/Qwen3.5-4B`
   - boundary small/medium run after 2B passes.
7. `Qwen/Qwen3.5-35B-A3B` or `Qwen/Qwen3.6-35B-A3B`
   - stronger MoE comparison after the reward no longer overpays fake-casual.
8. Liquid LFM models on Modal/custom TRL:
   - `LiquidAI/LFM2.5-1.2B-Instruct`
   - `LiquidAI/LFM2-2.6B`
   - only if Prime cannot cover the needed path or we want a small-model
     cross-family comparison outside Prime.

### Current Prime Model Prices And Capacity

Live `prime train models --output json` on 2026-06-24 reported:

| Model | At capacity | Training $/M tok | Input $/M tok | Output $/M tok | Notes |
|---|---:|---:|---:|---:|---|
| `Qwen/Qwen3.5-0.8B` | no | `0.06` | `0.02` | `0.06` | cheap Qwen smoke, previously unstable |
| `Qwen/Qwen3.5-2B` | no | `0.15` | `0.05` | `0.15` | primary next target |
| `Qwen/Qwen3.5-4B` | yes | `0.30` | `0.10` | `0.30` | boundary target, not currently runnable |
| `Qwen/Qwen3.5-9B` | yes | `0.60` | `0.20` | `0.60` | second requested Qwen target, wait for capacity |
| `Qwen/Qwen3.5-35B-A3B` | no | `1.00` | `0.25` | `0.75` | MoE comparison |
| `Qwen/Qwen3.6-35B-A3B` | no | `1.00` | `0.25` | `0.75` | newer MoE comparison |
| `meta-llama/Llama-3.2-1B-Instruct` | no | `0.06` | `0.02` | `0.06` | cheap Llama control |
| `meta-llama/Llama-3.2-3B-Instruct` | no | `0.15` | `0.05` | `0.15` | completed full run |
| `sprints/Llama-3.2-1B-Instruct` | no | `0.00` | `0.00` | `0.00` | free exploit smoke if available |
| `poolside/Laguna-XS.2` | no | `0.00` | `0.00` | `0.00` | free baseline candidate |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | no | `0.60` | `0.15` | `0.45` | larger active-parameter comparison |
| `openai/gpt-oss-20b` | no | `0.40` | `0.10` | `0.30` | larger cross-family comparison |

Availability and prices can change. Re-run `prime train models --output json`
before launching.

## Next Two-Model Qwen Ablation Track

The next scientific target is not "try a bigger model." It is:

> Can a patched reward plus SFT warmup prevent fake-casual reward hacking on
> Qwen, then improve further with RL?

Use exactly two Qwen target sizes for the main comparison:

1. `Qwen/Qwen3.5-2B`
   - available on the 2026-06-24 live Prime model check;
   - cheap enough for several controlled ablations;
   - primary target for SFT and RL-after-SFT.
2. `Qwen/Qwen3.5-9B`
   - listed by Prime, but currently at capacity;
   - keep configs and eval plan ready;
   - launch only after capacity clears.

Do not silently replace the 9B target with 4B or MoE in the same study. If 9B
stays unavailable, record the blocker and run only the 2B branch.

### RL Env Update Ablations

These ablations happen before SFT or RL. They are cheap and should use saved
rollouts from `zztqgqclh3y3hslpjsofzpcf`.

| ID | Env Change | Test Set | Required Result |
|---|---|---|---|
| E0 | current `0.3.14` reward | saved step 140-190 rollouts | reproduces high reward for fake-casual samples |
| E1 | emoji and all-caps hard diagnostics | saved rollout plus synthetic controls | emojis and unsupported all-caps fail without hurting normal acronym text |
| E2 | fake-casual phrase cap | saved bad samples | `stuff`, repeated `we got`, `you guys`, filler `thanks` drop below `0.75` |
| E3 | low-specificity substitution cap | paired bad/good examples | source facts replaced by `stuff` or `thing` score low |
| E4 | broken casual grammar cap | saved bad samples plus real Slack controls | broken grammar scores low; normal contractions pass |
| E5 | register mismatch cap | workplace/technical/patient tasks | slang like `cut the crap` fails unless requested |
| E6 | full patched `p50_50_no_penalty` | frozen validation and saved rollouts | bad samples fall below `0.75`; clean direct controls stay above `0.75` |

Publish a new env only after E1 to E6 pass locally. The next version should be
`0.3.15` unless another env version has already been published.

### SFT Update Ablations

SFT is the stabilizer for style. RL should not be asked to discover from
scratch that `plain and direct` is better than `sloppy and casual`.

| ID | SFT Variant | Dataset | Model | Required Result |
|---|---|---|---|---|
| S0 | base, no SFT | frozen eval prompts only | Qwen 2B and 9B | establishes base wrapper/fake-casual rate |
| S1 | existing SFT corpus | `jayshah5696/humanize-rl-prime-sft-messages-env0314` | Qwen 2B first | improves directness without new reward repairs |
| S2 | SFT corpus plus repair rows | env0314 corpus plus fixed bad rollout references | Qwen 2B first | lowers fake-casual rate and improves p50/strict |
| S3 | repair-weighted SFT | same data with explicit weighting/curriculum if supported | Qwen 2B only | tests whether repairs need stronger exposure |
| S4 | 9B SFT repeat | best S1/S2/S3 variant | Qwen 9B | runs only after 2B proves the recipe and 9B capacity clears |

Current Prime status:

- Prime Hosted `loss = "sft"` was attempted and blocked by:
  `HTTP 403: loss='sft' is currently restricted to beta users`.
- Prime `prime-rl` dataset SFT configs exist for Qwen 0.8B and 2B under
  `configs/prime_rl/`, using renderer `qwen3.5` and sequence length `4096`.
- Use Prime first when the run surface is available. Use Modal/TRL only when
  Prime cannot run the needed dataset-SFT path or when continuing an already
  launched Modal job.

SFT gate before RL:

- SFT must beat base on the same frozen prompts.
- Compare p50 reward, strict diagnostics, wrapper/options rate, emoji/all-caps
  failures, fake-casual phrase rate, low-specificity substitutions, and a human
  read of 20 to 50 examples.
- If SFT does not beat base qualitatively, do not start RL.

### RL After SFT Ablations

Run RL only after the env patch and SFT gate pass.

| ID | RL Init | Model | Steps | Purpose | Launch Gate |
|---|---|---|---:|---|---|
| R0 | base model | Qwen 2B | 20 to 50 | checks whether patched env alone prevents hacking | E6 passes |
| R1 | best SFT checkpoint | Qwen 2B | 20 to 50 | checks whether SFT plus patched reward is stable | SFT beats base |
| R2 | best SFT checkpoint | Qwen 2B | 200 | first serious Qwen full run | R1 passes rollout audit |
| R3 | base model | Qwen 9B | 20 to 50 | capacity-gated larger baseline | 9B available and R2 clean |
| R4 | best SFT checkpoint | Qwen 9B | 200 | larger final comparison | R3 clean and budget approved |

If Prime cannot initialize RL from the SFT checkpoint, push the SFT adapter or
merged checkpoint to Hugging Face if supported, then reference that checkpoint
explicitly in the run notes. If that path is not supported, document it as a
platform blocker and do not pretend the run is RL-after-SFT.

### Two-Model Ablation Matrix

| Stage | Qwen 2B | Qwen 9B | Decision |
|---|---|---|---|
| base eval | required | required when capacity clears | establishes model prior |
| SFT full corpus | required | optional after 2B | checks corpus quality |
| SFT full plus repairs | required | required for final 9B comparison | checks repair usefulness |
| RL from base | 20 to 50 step diagnostic only | diagnostic only | isolates env effect |
| RL after SFT | 50 step smoke, then 200 step full | 200 step only after 2B succeeds | final comparison |

### Example Bad And Target Repairs

Use these as unit tests, repair-row seeds, and qualitative eval anchors.

| Bad high-reward output | Target repair | Reward lesson |
|---|---|---|
| `We got stuff we need for Project too...` | `The Project deliverables are ready in the shared folder. Please send feedback by Friday so we can close the next revision.` | preserve concrete objects and action |
| `Stuff's done now, look at stuff we did in shared folder.` | `The shared folder has the completed draft and supporting notes. Please review them and send any changes you want before tomorrow.` | reject vague noun replacement |
| `We hit project timeline delay because technical stuff we didn't plan for...` | `The project timeline slipped because we found technical issues during integration. We are fixing them now and will share an updated launch date tomorrow.` | keep cause, owner, and next step |
| `We got us an artist residency in Sri Lanka... personal and professional stuff...` | `The Sri Lanka residency became the turning point of the essay. It gave the narrator space to rethink their work, relationships, and next direction.` | casual does not mean ungrammatical |
| `Hey guys, we got good stuff going on here.` | `The update is ready. It covers the main changes, the open questions, and the next review date.` | no generic hype |

### Stop Conditions

Stop the next ablation immediately if any of these happens:

- patched reward still gives `>=0.75` to the saved bad examples;
- SFT increases wrapper/options, emoji, all-caps, or fake-casual rate;
- RL high-reward samples show `stuff`, repeated `we got`, `you guys`, vague
  substitutions, or broken casual grammar;
- any family drops worse than `-0.05`;
- length collapses or 4096-token truncation spikes;
- Qwen hosted runs re-enter NaN/stall behavior before meaningful training.

## Action Plan

### Phase 0: Freeze The Run

Done:

- Do not launch more training from the current reward.
- Keep final checkpoint but mark it rejected.
- Keep artifacts under:

```text
runs/prime_training_smoke/zztqgqclh3y3hslpjsofzpcf/
```

Still useful:

- final checkpoint as a negative example;
- rollout traces for reward patching;
- metrics as evidence that aggregate reward can lie.

### Phase 1: Patch Reward Offline

Add diagnostics first. Do not train.

Candidate diagnostics:

- `fake_casual_phrase`
  - repeated `we got`;
  - repeated `you guys`;
  - repeated `stuff`;
  - `good stuff`;
  - `project stuff`;
  - `technical stuff`;
  - `we're good`;
  - `we got us`.
- `low_specificity_substitution`
  - source has concrete noun/entity/date/detail, output replaces it with
    `stuff`, `thing`, `things`, `people`, `someone`, `something`.
- `broken_informal_grammar`
  - phrases like `we tell you when stuff good again`;
  - missing auxiliaries in professional rewrite outputs.
- `register_mismatch`
  - slang/aggressive phrases in workplace, technical, patient, or formal tasks.
- `thanks_padding`
  - repeated/unsupported `thanks`, `thanks for looking out`, `appreciate you`
    when not asked and when used as filler.

Offline rescore targets:

- `audit_step140.json`
- `audit_step150.json`
- `audit_step160.json`
- `audit_step170.json`
- `audit_step180.json`
- `audit_step190.json`

Exact saved rollout files:

```text
runs/prime_training_smoke/zztqgqclh3y3hslpjsofzpcf/rollouts_step140.json
runs/prime_training_smoke/zztqgqclh3y3hslpjsofzpcf/rollouts_step150.json
runs/prime_training_smoke/zztqgqclh3y3hslpjsofzpcf/rollouts_step160.json
runs/prime_training_smoke/zztqgqclh3y3hslpjsofzpcf/rollouts_step170.json
runs/prime_training_smoke/zztqgqclh3y3hslpjsofzpcf/rollouts_step180.json
runs/prime_training_smoke/zztqgqclh3y3hslpjsofzpcf/rollouts_step190.json
```

If these local ignored files are unavailable, download the archived copy:

```bash
hf download jayshah5696/humanize-rl-research-artifacts-env0314 \
  --type dataset \
  --local-dir artifacts/humanize-rl-research-artifacts-env0314
```

Current audit command pattern:

```bash
PYTHONPATH=src uv run --no-project \
  --with pydantic \
  --with numpy \
  --with scikit-learn==1.8.0 \
  --with click \
  python scripts/eval/audit_prime_rollouts.py \
  --rollouts runs/prime_training_smoke/zztqgqclh3y3hslpjsofzpcf/rollouts_step180.json \
  --output runs/prime_training_smoke/zztqgqclh3y3hslpjsofzpcf/audit_step180_patched.json
```

Use `scikit-learn==1.8.0` for local audit parity with the serialized ridge
state.

Patch gate:

- bad examples above must drop below `0.75`;
- clean direct samples should remain high;
- real casual control messages should not be over-penalized.

### Phase 2: Build Repair SFT Rows

Use the bad high-reward samples as paired counterexamples.

For each bad sample:

- prompt: original task prompt;
- bad completion: saved rollout;
- target response: plain/direct rewrite preserving facts;
- tags:
  - `fake_casual_repair`
  - `low_specificity_repair`
  - `grammar_repair`
  - original task family/mode.

Minimum useful repair set:

- 50 rows from steps 140-190;
- include direct rewrites, emails, Slack, long-form, technical explanations;
- include edited positive examples, not only bad negatives.

2026-06-29 status:

- Extracted `100` env0315 repair candidates from archived Prime audit reports:
  `data/processed/sft/prime_audit_failure_set_env0315.jsonl`.
- Generated `70` Google-model repair-reference candidates, then cleaned them to
  `50` rows:
  `data/processed/sft/reference_targets/prime_audit_failure_refs_env0315_clean50.jsonl`.
- Clean report:
  `data/processed/sft/reference_targets/prime_audit_failure_refs_env0315_clean50_report.json`.
  The clean set keeps `36` unique task IDs across compression, direct email,
  rewrite repair, and tone shift failures.
- Local merged S2 candidate:
  `data/processed/v04_sft_final_plus_prime_env0315_clean50.jsonl`.
  Builder output:
  `data/processed/sft/gemma4_e2b_v04_prime_env0315_clean50/`.
  Result: `4905` raw rows, `4843` accepted rows, `62` rejected rows, split
  counts `train=4358`, `valid=242`, `test=243`, and
  `prime_failure_reference_generation=70` accepted rows.
- Published the S2 clean50 corpus to a dedicated HF dataset:
  `jayshah5696/humanize-rl-prime-sft-messages-env0315-clean50`.
  HF commit: `e1e6b1e839c0dc0450313e5f558c90a6ac925557`.
- Added S2 Prime SFT config:
  `configs/prime_rl/qwen35_2b_sft_target_messages_env0315_clean50_gate_env0315.toml`.
- S2 live preflight report:
  `runs/prime_sft_preflight/qwen35_2b_env0315_clean50.json`.
  Result: config, local `HF_TOKEN`, HF Dataset Viewer counts, and Prime auth
  pass; live `prime train models` shows `Qwen/Qwen3.5-2B` available at
  `training_price_per_mtok=0.15`; `WANDB_API_KEY source` fails.
- Generated S2 launch kit:
  `runs/prime_sft_launch_kit/qwen35_2b_env0315_clean50/` and
  `runs/prime_sft_launch_kit/qwen35_2b_env0315_clean50.tar.gz`.
  The kit now embeds `sft_eval_manifest.json` so the sandbox artifact carries
  the exact post-SFT gate order and S2 promotion root.
- Generated S2 post-SFT eval manifest:
  `runs/prime_sft_promotion/TEMPLATE_QWEN35_2B_ENV0315_CLEAN50/sft_eval_manifest.json`.
  It writes future after-SFT RL outputs under
  `configs/prime/qwen35_2b_p5050_after_sft_env0315_clean50_<checkpoint_slug>.toml`
  and uses W&B run names ending in `env0315-clean50-<checkpoint_slug>`.
  It now also records `audit_specs`, base/SFT rollout placeholders, and concrete
  `--eval-label` audit commands for `mix_v2_p5050`, `v02_strict`, and
  `v03_strict` with `scikit-learn>=1.8,<1.9` pinned for ridge parity.
- SFT output verification now accepts both approved dataset-SFT corpora:
  `jayshah5696/humanize-rl-prime-sft-messages-env0314` and
  `jayshah5696/humanize-rl-prime-sft-messages-env0315-clean50`. Unknown
  datasets still fail the verifier.
- Decision: use S2 clean50 for the next quality-oriented Qwen 2B SFT run unless
  intentionally spending budget on the S1 no-new-repairs baseline. Do not launch
  either path until the W&B gate passes.
- Validation: expanded Prime/reward/SFT data suite `152 passed`; ruff and
  `git diff --check` pass.

Do not rely on duplicate oversampling through the current builder; it dedupes
exact pairs. If repairs need higher weight, implement sampler weighting or
split-level curriculum explicitly.

### Phase 3: Short Hosted RL Smoke

Only after Phase 1 offline rescore passes.

Run:

- Prime Hosted RL;
- 20-50 steps;
- small/free model first;
- eval every 25 or 50;
- save rollouts every 10;
- no full run.

Recommended first smoke:

```toml
model = "sprints/Llama-3.2-1B-Instruct"
max_steps = 50
batch_size = 128
rollouts_per_example = 8

[sampling]
max_tokens = 2048
temperature = 0.7
```

If free backend is unavailable, use:

```toml
model = "meta-llama/Llama-3.2-1B-Instruct"
```

Suggested smoke names:

- run name:
  `humanize-p5050-fakecasualfix-llama32-1b-smoke50-env0315`
- W&B name:
  `prime-fakecasualfix-llama32-1b-smoke50-env0315`

Use a new env version after patching, likely `0.3.15`.

Smoke pass criteria:

- p50 validation positive delta;
- strict v02/v03 non-negative delta;
- no family drop worse than `-0.05`;
- no truncation/repetition spike;
- no high-reward fake-casual samples;
- no high-reward wrapper/emoji/all-caps samples;
- qualitative samples directly answer the task.

### Phase 4: Qwen Two-Model Ablation

Only after the patched reward passes one short smoke.

Order for the next ablation:

1. Run base eval for `Qwen/Qwen3.5-2B`.
2. Run the approved SFT variant for `Qwen/Qwen3.5-2B`.
3. Run 20 to 50 step RL smoke from base and from SFT for `Qwen/Qwen3.5-2B`.
4. Run 200 step RL-after-SFT for `Qwen/Qwen3.5-2B` only if smoke passes.
5. Re-run the same base/SFT/RL-after-SFT sequence for `Qwen/Qwen3.5-9B`
   after Prime capacity clears.

Do not run MoE/larger models until the Qwen 2B/9B ablation shows no reward
hack. Do not substitute `Qwen/Qwen3.5-4B` for the 9B result without marking it
as a separate study.

### Phase 5: Full Run

Only one full run per approved target after the smoke:

- target model selected from sweep;
- reward patched;
- repair SFT data either trained or included;
- frozen eval prompts locked;
- family/mode gates defined before launch.

For the next study, the intended full runs are:

1. `Qwen/Qwen3.5-2B` RL-after-SFT, 200 steps.
2. `Qwen/Qwen3.5-9B` RL-after-SFT, 200 steps, capacity-gated.

Full-run acceptance requires:

- aggregate p50 improvement;
- aggregate strict v02/v03 improvement;
- no family drop worse than `-0.05`;
- no length drift collapse;
- no high-reward fake-casual samples;
- no wrapper/options/emoji/all-caps regression;
- blind human read prefers trained model over base and prior checkpoint.

## Concrete Next PR Scope

The next implementation PR should be narrow:

1. Add fake-casual diagnostics in reward checks.
2. Add unit tests with the saved bad phrases and clean casual controls.
3. Publish env `0.3.15`.
4. Offline-rescore saved rollout files from `zztqgqclh3y3hslpjsofzpcf`.
5. Update this document with before/after rescore deltas.
6. Build or refresh SFT repair rows from the failed high-reward samples.
7. Run Qwen 2B base eval and SFT eval on the frozen prompts.
8. Launch only a 20-50 step hosted RL smoke if offline rescoring and SFT gates
   pass.

Do not include the Qwen 9B full run in the same PR. Keep it as a capacity-gated
follow-up after Qwen 2B proves the recipe.

Use the archived artifacts if this worktree is gone:

- HF repo:
  `jayshah5696/humanize-rl-research-artifacts-env0314`
- commit:
  `09457c42a56915ce649370fd4f42da92c0bf1080`
- contains:
  - saved rollouts/audits for `zztqgqclh3y3hslpjsofzpcf`;
  - `llama32_3b_failure_set_env0314.jsonl`;
  - `llama32_3b_rl_vs_base_summary_env0314.json`;
  - taskset report;
  - repair-reference files;
  - local source files for the published SFT dataset.

## 2026-06-26 Progress Update

Completed locally:

- Added deterministic diagnostics/caps for:
  - `fake_casual_phrase`;
  - `low_specificity_substitution`;
  - `broken_informal_grammar`;
  - `register_mismatch`;
  - `thanks_padding`.
- Mirrored the patch into the Prime env package and bumped the local env
  package version to `0.3.15`.
- Published Prime env `jayshah5696/humanize-rl-env@0.3.15`.
- Verified the local wheel includes `ridge_state.pkl` and the bundled task
  files before publishing.
- Ran a local Prime eval smoke on `0.3.15`.
- Restored archived continuation artifacts from
  `jayshah5696/humanize-rl-research-artifacts-env0314`.
- Offline-rescored saved late-run rollouts with the patched reward.

Offline rescore result:

| Step | Rows | Mean | Max | High samples `>=0.75` |
|---:|---:|---:|---:|---:|
| 140 | `64` | `0.493312` | `0.500000` | `0` |
| 150 | `64` | `0.490982` | `0.500000` | `0` |
| 160 | `64` | `0.487403` | `0.500000` | `0` |
| 170 | `64` | `0.501132` | `0.696955` | `0` |
| 180 | `64` | `0.500554` | `0.657759` | `0` |
| 190 | `64` | `0.495079` | `0.500000` | `0` |

Clean-control check:

- Scored the `20` archived clean repair references.
- New diagnostics false-positive rows: `0/20`.
- Only `7/20` controls scored `>=0.75`, but those drops came from older
  diagnostics such as `subject_line`, `missing_entity`, and
  `invented_temporal_detail`, not the new fake-casual patch.

Prime publish and local smoke:

- Env: `jayshah5696/humanize-rl-env@0.3.15`
- Prime content hash: `51a8c3ea`
- Wheel SHA256:
  `56bffe065a8296052d7aac8153037b3e80c7738ed6f6b20a097289853d99d6a2`
- Local smoke output:
  `runs/prime_eval_smoke/qwen35_08b_p5050_env0315/evals/humanize-rl-env--Qwen--Qwen3.5-0.8B/b003a93c/results.jsonl`
- Model: `Qwen/Qwen3.5-0.8B`
- Examples / rollouts: `3 / 1`
- Reward avg/std: `0.557 / 0.221`
- Rewards: `[0.484, 0.329, 0.857]`
- Truncation: `0%`

Read:

- The smoke passed technically.
- Direct hotfix output stayed high.
- Formal/template-like and too-long outputs stayed low.
- This is not a hosted training gate by itself; it only validates that the
  published env loads and scores correctly.

Pangram-style detector decision:

- Treat Pangram or similar AI-detector behavior as an external evaluation rail,
  not as a Prime reward dependency.
- Do not add a live detector API call inside the env. It would make hosted
  reward scoring slower, costlier, and less reproducible.
- Built a frozen local detector-mimic eval set after `0.3.15` publication:
  `data/eval/detector_mimic_v01.jsonl`.
- Public Pangram docs expose document-level fractions and window-level segment
  labels; the local mimic mirrors that shape with deterministic markers.
- Added an offline Pangram-export alignment rail:
  `scripts/eval/compare_detector_mimic_to_pangram.py`.
  It accepts a saved Pangram JSON/JSONL export for the frozen mimic rows and
  compares coverage, human/nonhuman label agreement, and nonhuman-fraction
  deltas.
  This keeps external-detector calibration available without putting a live API
  call inside Prime reward scoring.
- 2026-07-01 update: Pangram v3 exposes `fraction_ai_assisted` alongside
  `fraction_ai`; the alignment rail now treats
  `fraction_ai + fraction_ai_assisted` as the external detector-risk fraction
  so AI-assisted text cannot look safe just because `fraction_ai` is low.
- Promotion integration: `scripts/eval/build_sft_promotion_gate.py` now accepts
  optional `--pangram-alignment-report`. Pangram remains optional for training,
  but if the external report is supplied for a promotion read, it must pass.
- 2026-06-29 update: expanded the frozen mimic set with harder human controls
  that use literal `seamless` / `unlock`, mixed wrapper segments, stacked
  corporate boilerplate, and a fake-casual plus corporate-gloss reward hack.

Historical gate at this point, later resolved by the hosted-run audit below:

1. Refresh Prime auth and audit hosted smoke `fj9oinokvx5zott096tgfwqw`.
2. Do not start a Qwen 2B full run until that hosted smoke is audited.

Direct clean-control refresh:

- Report:
  `runs/prime_eval_smoke/direct_clean_controls_env0315.json`
- Rows: `10`
- Mean/min/max: `0.852037` / `0.469274` / `0.975566`
- Rows `>=0.75`: `9/10`
- New diagnostic false-positive rows: `0/10`
- Read: clean direct controls pass the minimum `90%` gate. The one low row
  failed an older `missing_entity` check, not the fake-casual patch.

Hosted smoke launch:

- Date: 2026-06-27 PDT / 2026-06-28 UTC
- Config:
  `configs/prime/llama32_1b_fakecasualfix_env0315_smoke50.toml`
- Launch command:

```bash
prime --plain train configs/prime/llama32_1b_fakecasualfix_env0315_smoke50.toml \
  --yes --output json
```

- Run ID: `fj9oinokvx5zott096tgfwqw`
- Run name:
  `humanize-p5050-fakecasualfix-llama32-1b-smoke50-env0315-r1`
- Model: `meta-llama/Llama-3.2-1B-Instruct`
- Env: `jayshah5696/humanize-rl-env@0.3.15`
- Reason for not using `sprints/Llama-3.2-1B-Instruct`: existing Prime config
  notes say it rejects this custom env despite zero price.
- Launch status: `PENDING`, then `RUNNING` on first monitor poll.

Monitoring blocker:

- After launch, Prime authenticated endpoints returned `API key unauthorized`.
- Affected commands: `prime train get`, `prime train progress`,
  `prime train list`, `prime whoami`, and `prime wallet`.
- Resume after `prime login` or a token refresh:

```bash
prime --plain train get fj9oinokvx5zott096tgfwqw --output json
prime --plain train progress fj9oinokvx5zott096tgfwqw
```

Detector-mimic gate v01:

- Frozen set: `data/eval/detector_mimic_v01.jsonl`
- Runner: `scripts/eval/evaluate_detector_mimic.py`
- Pangram bulk export:
  `scripts/eval/export_detector_mimic_for_pangram.py`
- Optional offline Pangram comparison:
  `scripts/eval/compare_detector_mimic_to_pangram.py`
- Report:
  `runs/detector_mimic/detector_mimic_v01_report.json`
- Scored rows:
  `runs/detector_mimic/detector_mimic_v01_scored.jsonl`
- Command:

```bash
uv run scripts/eval/evaluate_detector_mimic.py \
  --input data/eval/detector_mimic_v01.jsonl \
  --output runs/detector_mimic/detector_mimic_v01_report.json \
  --scored-output runs/detector_mimic/detector_mimic_v01_scored.jsonl
```

- Result: `22` rows, `8` human controls, `14` nonhuman controls,
  `0` false positives, `0` false negatives, mean `fraction_ai` `0.452857`,
  gate `pass`.
- Use this as an external detector-style gate before promoting any SFT/RL
  candidate model.
- To collect a real Pangram export, first write SDK-ready bulk items:

```bash
uv run scripts/eval/export_detector_mimic_for_pangram.py \
  --input data/eval/detector_mimic_v01.jsonl \
  --output runs/detector_mimic/pangram_bulk_items.json
```

- Submit `payload["items"]` to `Pangram.submit_bulk(items=...)`, then save the
  result outside the reward env and compare:

```bash
uv run scripts/eval/run_pangram_bulk_detection.py \
  --input runs/detector_mimic/pangram_bulk_items.json \
  --output runs/detector_mimic/pangram_export.json \
  --submit-report runs/detector_mimic/pangram_bulk_submit.json \
  --timeout 3600 \
  --poll-interval 2

uv run scripts/eval/compare_detector_mimic_to_pangram.py \
  --input data/eval/detector_mimic_v01.jsonl \
  --pangram-output runs/detector_mimic/pangram_export.json \
  --output runs/detector_mimic/pangram_alignment_report.json
```

- The S2 eval manifest records `runs/detector_mimic/pangram_bulk_items.json`
  as an optional external-detector handoff artifact, pins its SHA256 when it
  exists, and records the export, bulk-run, and alignment commands. Launch
  readiness verifies that hash so the Pangram calibration payload cannot drift
  silently before SFT spend.

Validation status:

- Focused reward plus detector tests: `66 passed`.
- `ruff check` on touched reward, detector-mimic, script, and test files:
  passed.
- Hosted-smoke audit remains pending because Prime auth still returns
  `API key unauthorized`.

Qwen 2B env0315 SFT/RL ablation configs:

- Qwen 0.8B renderer smoke:
  `configs/prime/qwen35_08b_fakecasualfix_env0315_smoke50.toml`
- Qwen 2B base RL full template:
  `configs/prime/qwen35_2b_p5050_env0315_full200_template.toml`
- Qwen 2B SFT-to-RL full template:
  `configs/prime/qwen35_2b_p5050_after_sft_env0315_full200_template.toml`
- Qwen 2B dataset SFT target:
  `configs/prime_rl/qwen35_2b_sft_target_messages_env0314_gate_env0315.toml`

Post-auth launch order:

1. Audit hosted Llama smoke `fj9oinokvx5zott096tgfwqw`.
2. If it passes, launch the Qwen 0.8B env0315 smoke.
3. If that passes, run Qwen 2B dataset SFT.
4. Compare Qwen 2B base RL against Qwen 2B SFT-to-RL.
5. Run rollout audit plus detector-mimic gate before promoting a candidate.

Config guardrails:

- All new hosted RL configs use env `0.3.15`.
- Qwen hosted training/eval caps stay at `max_tokens = 1024`.
- Qwen 2B full templates use `batch_size = 64`, `rollouts_per_example = 8`,
  and `max_inflight_rollouts = 32`.
- Config tests: `3 passed`.

Prime auth and hosted-run audit update:

- Prime auth was refreshed successfully on 2026-06-28.
- Added bundle script:
  `scripts/eval/audit_prime_run_bundle.py`.
- Llama 1B smoke `fj9oinokvx5zott096tgfwqw` completed and bundle-gated:
  - rollout bundle: `pass`
  - detector-mimic gate: `pass`
  - step-50 eval delta: p50 `-0.021137`, v02 strict `+0.054831`,
    v03 strict `+0.067670`
  - read: stability/exploit pass, not a p50 quality win.
- Qwen 0.8B r1 `r8xlz0csp79fu0z0elp9h1dv` completed and bundle-gated:
  - rollout bundle: `pass`
  - step-50 eval delta: p50 `+0.069854`, v02 strict `-0.064143`,
    v03 strict `-0.051501`
  - read: p50 improved but strict-family guardrail failed.
- Qwen 0.8B r2 `pzcfi8aew2pnxnkhec2augts` completed and bundle-gated:
  - config: `qwen35_08b_fakecasualfix_env0315_smoke50_lr5e5.toml`
  - rollout bundle: `pass`
  - detector-mimic gate: `pass`
  - step-50 eval delta: p50 `+0.022284`, v02 strict `+0.111589`,
    v03 strict `+0.125909`
  - read: r2 clears the Qwen smoke gate.
- Qwen 2B base RL `o48ryskshkn06b3o1b1kauql` was launched from the full200
  template, then stopped at latest step `55` after step-50 eval failed:
  - rollout bundle: `pass`
  - detector-mimic gate: `pass`
  - step-50 eval delta: p50 `+0.056013`, v02 strict `-0.168230`,
    v03 strict `-0.128226`
  - read: base RL at `8e-5` over-optimizes p50 and hurts strict
    generalization. Do not continue this base-RL direction.
- 2026-06-29 matrix update:
  - script: `scripts/eval/build_prime_ablation_matrix.py`
  - report: `runs/prime_training_smoke/ablation_matrix_env0315.json`
  - all saved bundles were refreshed against the expanded `22` row
    detector-mimic gate before building the matrix.
  - criteria: completed status, rollout bundle pass, detector-mimic pass, p50
    delta `>= 0`, v02/v03 strict deltas `>= 0`.
  - result: `4` runs summarized, `1` selection pass.
  - selected smoke: `pzcfi8aew2pnxnkhec2augts`
    (`Qwen/Qwen3.5-0.8B`, lr `5e-5`), score `0.259782`.
  - rejected full-model base RL: `o48ryskshkn06b3o1b1kauql` because status is
    `STOPPED` and strict deltas are negative. This keeps the next full-model
    path as Qwen 2B dataset SFT, then SFT-to-RL.
  - validation: matrix helper focused tests `3 passed`; broad Prime/reward gate
    suite `109 passed`; ruff and `git diff --check` pass.

Updated next action:

1. Run Qwen 2B dataset SFT:
   `configs/prime_rl/qwen35_2b_sft_target_messages_env0314_gate_env0315.toml`.
2. Evaluate the SFT checkpoint against env `0.3.15`, rollout audit, and
   detector-mimic gate.
3. Only then fill
   `configs/prime/qwen35_2b_p5050_after_sft_env0315_full200_template.toml`
   with the READY SFT checkpoint ID and run SFT-to-RL.

Launch preflight from 2026-06-28:

- Prime CLI auth is restored: `prime --plain whoami` succeeds for
  `jayshah5696`.
- Hosted Training `prime train` is still the env/rollout schema. Do not submit
  `configs/prime_rl/*.toml` through `prime train`.
- Fresh Prime `prime-rl` source at commit `d700753` exposes
  `sft = prime_rl.entrypoints.sft:main`, so the dataset-SFT command remains the
  correct one on Linux/CUDA.
- Local shell has `HF_TOKEN`, but no `WANDB_API_KEY`; Prime secret store is also
  empty. Do not launch the tracked Qwen 2B target SFT until W&B is provided as a
  local env var or Prime secret.
- Prime sandbox is viable for the launch once secrets are present:
  create a GPU VM sandbox with a CUDA/PyTorch image, upload the current repo
  state, install open `prime-rl`, and run:
  `uv run sft @ /workspace/humanize-rl/configs/prime_rl/qwen35_2b_sft_target_messages_env0314_gate_env0315.toml`.
- Executable local preflight:
  `uv run scripts/train/prime_sft_preflight.py --check-hf-viewer`.
  Current live result: config, local `HF_TOKEN`, HF Dataset Viewer counts, and
  Prime auth pass; `WANDB_API_KEY` source fails.
- 2026-06-29 preflight report update:
  `scripts/train/prime_sft_preflight.py` now supports `--output` and
  `--no-fail-on-gate`. Current persisted report:
  `runs/prime_sft_preflight/qwen35_2b_env0315.json`.
  Gate result is `passed=false`; failed checks: `WANDB_API_KEY source`.
  Validation: preflight focused tests `6 passed`; broad Prime/reward gate suite
  `110 passed`; ruff and `git diff --check` pass.
- Current HF Dataset Viewer counts match the intended dataset:
  train `4313`, validation `239`, test `241`, total `4793`; the train split
  exposes a `messages` column.
- Prime `prime-rl` schema check against current source commit `d700753` passed
  for `configs/prime_rl/qwen35_2b_sft_target_messages_env0314_gate_env0315.toml`.
- Generated a secret-free Prime sandbox launch kit:
  `runs/prime_sft_launch_kit/qwen35_2b_env0315/` and
  `runs/prime_sft_launch_kit/qwen35_2b_env0315.tar.gz`.
  The kit copies the exact SFT config, writes a manifest with config SHA
  `f14efebb82630df65dbbab8441c87c60f687e9672ee0dff035cfa4a8c2bfc65c`,
  pins `prime-rl` to `d700753`, and includes `run_sft.sh` for the sandbox.
- Added SFT-to-RL config handoff helper:
  `scripts/train/prepare_prime_sft_to_rl_config.py`.
  After the SFT checkpoint is READY, use it to produce a concrete hosted RL
  config instead of manually editing the placeholder in
  `configs/prime/qwen35_2b_p5050_after_sft_env0315_full200_template.toml`.
- Added SFT promotion gate helper:
  `scripts/eval/build_sft_promotion_gate.py`.
  Before SFT-to-RL, compare base vs SFT rollout audits on the same labels,
  require detector-mimic pass, and require a human read JSON with `passed=true`
  and `20 <= sample_count <= 50`; it also requires a passing
  `sft_output_verification.json` from `scripts/train/verify_prime_sft_output.py`.
  It can also enforce an optional Pangram-alignment report when a real external
  detector export has been collected.
- Added SFT human-read packet helper:
  `scripts/eval/build_sft_human_read_packet.py`.
  After SFT candidate audits exist, use it to collect a bounded `20..50` sample
  packet from top audit samples before manually setting `passed=true` for the
  promotion gate.
- Added Prime `prime-rl` SFT output verifier:
  `scripts/train/verify_prime_sft_output.py`.
  After SFT finishes, verify `output_dir/weights/step_N` and adapter artifacts
  exist before treating the run as a usable SFT model for eval or handoff.
  The verifier accepts both approved S1/S2 datasets and still rejects untracked
  datasets.
- Added Prime warm-start checkpoint handoff verifier:
  `scripts/train/verify_prime_warm_start_checkpoint.py`.
  Before rendering the after-SFT hosted RL config, verify the checkpoint is
  present, `READY`, tied to the expected Prime run, and uses
  `Qwen/Qwen3.5-2B`; pass the resulting report to
  `scripts/train/prepare_prime_sft_to_rl_config.py` with
  `--checkpoint-handoff-report`. The renderer now requires this by default;
  `--allow-unverified-checkpoint` is only for dummy template validation.
- Tightened the after-SFT config renderer so real SFT-to-RL configs also
  require a passing `--promotion-gate-report` from
  `scripts/eval/build_sft_promotion_gate.py`. This enforces the documented rule
  that SFT must beat base and pass detector/human-read gates before RL.
- 2026-06-29 rechecked the documented preflight path. The lightweight train
  helpers now declare inline `uv` script metadata, so
  `uv run scripts/train/prime_sft_preflight.py --check-hf-viewer` reaches the
  preflight itself instead of trying to build the full local project deps. Live
  result: dataset SFT config, local `HF_TOKEN`, HF Dataset Viewer counts, and
  Prime auth pass; `WANDB_API_KEY` source still fails and Prime secret store is
  empty.
- Added SFT eval manifest helper:
  `scripts/eval/build_sft_eval_manifest.py`.
  Current template artifact:
  `runs/prime_sft_promotion/TEMPLATE_QWEN35_2B_ENV0315/sft_eval_manifest.json`.
  This records the ordered SFT-to-RL gate:
  `verify_sft_output`, collect base/SFT rollout audits for `mix_v2_p5050`,
  `v02_strict`, and `v03_strict`, build the human-read packet, build/pass the
  promotion gate, verify the Prime READY checkpoint handoff, then render the
  after-SFT RL config.
- Decision: do not render or launch the after-SFT RL config until the manifest
  gates pass: SFT output verification, bounded human-read approval, promotion
  gate, and checkpoint handoff.
- Validation: manifest focused tests `2 passed`; broad Prime/reward gate suite
  `112 passed`.
- Added Prime audit failure extractor:
  `scripts/data/build/extract_prime_audit_failures.py`.
  It converts saved Prime audit JSON top samples into the JSONL failure-set
  shape consumed by
  `scripts/data/build/generate_sft_references_from_failures.py`.
- Generated and cleaned the env0315 S2 repair-reference slice:
  `data/processed/sft/reference_targets/prime_audit_failure_refs_env0315_clean50.jsonl`.
  The local SFT candidate built from it accepts all `50` new clean rows, bringing
  total accepted `prime_failure_reference_generation` rows to `70`.
- Published the S2 clean50 HF dataset:
  `jayshah5696/humanize-rl-prime-sft-messages-env0315-clean50`
  at commit `e1e6b1e839c0dc0450313e5f558c90a6ac925557`.
- Added S2 config and launch kit:
  `configs/prime_rl/qwen35_2b_sft_target_messages_env0315_clean50_gate_env0315.toml`
  and `runs/prime_sft_launch_kit/qwen35_2b_env0315_clean50.tar.gz`.
  The launch kit now packages the S2 eval manifest as
  `prime_sft_launch_kit/sft_eval_manifest.json`.
- Added S2 eval/promotion manifest:
  `runs/prime_sft_promotion/TEMPLATE_QWEN35_2B_ENV0315_CLEAN50/sft_eval_manifest.json`.
  It keeps the future SFT-to-RL config and run name distinct from S1.
  It pins the after-SFT RL template:
  `configs/prime/qwen35_2b_p5050_after_sft_env0315_full200_template.toml`.
  It records that template's SHA256 and the SFT-to-RL renderer now requires the
  eval manifest so stale template or artifact-path drift fails before rendering
  a concrete RL config.
  The SFT output verifier also accepts the eval manifest and rejects config or
  verification-report path drift before promotion.
  The SFT human-read packet also accepts the eval manifest and rejects candidate
  audit or human-read output path drift before manual review.
  The SFT promotion gate also accepts the eval manifest and rejects base/SFT
  audit, detector, human-read, SFT-output, Pangram, or promotion-report path
  drift.
  It also stores base/SFT rollout placeholders and concrete audit commands for
  `mix_v2_p5050`, `v02_strict`, and `v03_strict`, using `--eval-label` defaults
  and pinned `scikit-learn>=1.8,<1.9`.
  Real checkpoint manifests now auto-replace `<checkpoint_slug>` in the
  after-SFT config path and hosted RL run name while template manifests keep the
  placeholder.
- Human-read and promotion-gate scripts now reject both template checkpoint
  strings, `READY_SFT_CHECKPOINT_ID` and `FILL_WITH_READY_SFT_CHECKPOINT_ID`, so
  the S2 template manifest cannot be accidentally promoted before a real
  checkpoint exists.
- Added S2 launch-readiness verification:
  `runs/prime_sft_preflight/qwen35_2b_env0315_clean50_launch_readiness.json`.
  It checks the S2 config, preflight report, launch-kit manifest, config hash,
  launch archive contents, eval manifest hash, and archived runner/readme files
  agree before launch. It also verifies the pinned after-SFT template hash in
  the eval manifest. Current result: artifact/archive/template consistency
  passes, but launch readiness fails on `WANDB_API_KEY source`.
- Added offline Pangram-export alignment for the frozen detector-mimic set so a
  real Pangram run can be compared against local mimic behavior before model
  promotion.
- The S2 eval manifest now records the SDK-ready Pangram bulk-items handoff and
  its hash as an optional external-detector artifact, plus the concrete Pangram
  bulk-run command that writes `runs/detector_mimic/pangram_export.json`.
- SFT promotion gate now accepts optional `--pangram-alignment-report`; the
  stored S2 manifest records this as optional and does not block launch on a
  missing Pangram export.
- S2 clean50 output verification no longer trips on the dataset guardrail; the
  verifier accepts S1 env0314 and S2 env0315-clean50, while rejecting unknown
  datasets.
- Live S2 preflight passes dataset config, local `HF_TOKEN`, HF Dataset Viewer
  counts `train=4358 validation=242 test=243`, Prime CLI version `0.6.14`,
  Prime auth, and Prime hosted training model availability for
  `Qwen/Qwen3.5-2B`; it still fails `WANDB_API_KEY source`.
- 2026-07-01 W&B source recheck: `/private/tmp/humanize_rl_prime_wandb.env` is
  missing, local `WANDB_API_KEY` is missing, and the Prime secret list is empty.
- Launch readiness now embeds the preflight report SHA256 and preflight check
  details so the Prime CLI/toolchain state is visible in the final pre-spend
  gate.
- 2026-07-02 sandbox-gate correction: `prime sandbox run --help` exposes
  `-e KEY=VALUE` env passthrough, but no automatic Prime global-secret
  injection. The S2 preflight now requires local `WANDB_API_KEY` by default for
  the sandbox launch path and only allows Prime-secret-only W&B with explicit
  `--allow-prime-wandb-secret` for a non-sandbox or custom secret-injected path.
  The S2 launch-kit README now records this caveat.
- 2026-07-02 policy-report update: the preflight JSON now records
  `launch_policy.runner=prime_sandbox` and
  `launch_policy.wandb_source.local_env_required=true`. Launch readiness
  rejects a preflight report that allowed Prime-only W&B secrets for this S2
  sandbox launch path.
- 2026-07-02 target-config guard: S2 preflight now rejects drift away from the
  intended target run shape, including `max_steps=200`, `seq_len=4096`,
  train/validation batches `128/64`, assistant-only loss masks, LoRA
  `rank=32 alpha=64`, `lr=2e-5`, and sharded safetensors checkpoints. The
  readiness report embeds the passing config detail.
- 2026-07-02 runtime-ref guard: S2 launch readiness now requires the launch-kit
  `prime_rl_ref` to match pinned runtime ref `d700753`, and records both the
  actual and expected refs in the readiness report.
- 2026-07-02 runner-ref guard: S2 launch readiness now inspects
  `run_sft.sh` and rejects a runner that does not fetch and check out
  `d700753`, even if the launch manifest still claims the correct
  `prime_rl_ref`.
- 2026-07-02 big-step env lock: env `0.3.15` is locked for the next SFT plus RL
  ablation. Stop adding reward/env guardrails unless the launched SFT/RL evals
  expose a concrete failure.
- Decision: next quality-oriented full-model SFT uses S2 clean50 unless the
  explicit goal is to spend budget on the S1 no-new-repairs baseline. After S2
  promotes, render the after-SFT RL config from the S2 eval manifest and launch
  Qwen 2B RL-after-SFT.
- Live launch state: Prime auth passes; S2 launch readiness fails only on local
  `WANDB_API_KEY source`; no Prime sandboxes exist; live Hosted Training reports
  `Qwen/Qwen3.5-2B` at capacity and `Qwen/Qwen3.5-9B` available; Hosted SFT
  still stops before launch when W&B is configured and `WANDB_API_KEY` is absent.
- 2026-07-02 live-doc correction: current Prime docs say GPU sandboxes are
  CPU-only/roadmap, so the tracked sandbox runner is not the live GPU path for
  open `prime-rl` SFT. Use Prime pods for GPU SFT/RL unless Prime GPU
  sandboxes become available.
- 2026-07-02 pod launch attempt: two Crusoe `A100_80GB x1` `prime_rl` pods were
  provisioned for S2 clean50 and then terminated after SSH public-key denial:
  `71b5034cc93941cd8c9ceeee4edc11d5` and
  `183c7c922f9842cd9a2bac97317dd8bd`. Prime showed zero active pods after
  cleanup.
- 2026-07-02 SSH decision: Prime account SSH keys were empty before the attempt;
  local key `codex-id-ed25519-20260702` was uploaded and became primary, but a
  new pod still rejected SSH. Do not create another GPU pod until key injection
  is fixed, likely by checking the dashboard key state or using a fresh RSA key
  upload/recreate path.
- 2026-07-02 secret-handling decision: do not pass HF/W&B secrets through
  `prime pods create --env`; the CLI echoed env values during pod creation.
- 2026-07-03 Prime pod continuation: a fresh RSA key was uploaded and the
  MassedCompute pod `62abc46cde1f4705b0ce65ab702005ae`
  (`humanize-s2-sft-a100-massed-r1`) is SSH-accessible at
  `ubuntu@154.54.100.38` with `A100_80GB x1`.
- The live S2 execution path is now Prime GPU pod, not local machine and not
  CPU-only sandbox. Local work is limited to launch-kit generation, transfer,
  docs, and monitoring.
- MassedCompute rejected the `prime_rl` image, Datacrunch `prime_rl` capacity
  returned no valid GPU configuration, and Crusoe `prime_rl` pods still failed
  SSH key auth. The accepted live path is MassedCompute Ubuntu CUDA plus
  bootstrap of the pinned `prime-rl` runtime on the pod.
- The refreshed launch kit is staged on the pod:
  `archive_sha256=9d0419131be9d84d4bb6ea29914479e8db6395afffef7bfe3ed6d355ca082e5c`,
  `runner_sha256=7cf0a338f99f617bb6448f4c570e7adff3cd0760a2d4520ee36295e776220807`,
  `config_sha256=b3775839dda7abac69be33eb28b1c83c74c2e9c486d9293f3e6f087cd2d5d18a`.
- Remote runtime bootstrap is in progress on the Prime pod: `prime-rl@d700753`
  is checked out, submodules are forced over HTTPS, CUDA 12.8 nvcc plus
  `g++-12` and `ninja` are installed, and `flash-attn==2.8.3.post1` is
  compiling with `FLASH_ATTN_CUDA_ARCHS=80` against the pod's `uv run`
  Python/Torch environment. Start SFT after `flash_attn_2_cuda` import passes.
- Online W&B is intentionally waived for this run because no local
  `WANDB_API_KEY` exists and pod-create env echo made secrets unsafe there.
  Remote `/workspace/sft.env` uses `WANDB_MODE=offline` and
  `WANDB_API_KEY=offline`; HF auth remains secret-file based on the pod.
- Validation: env/config/ablation gate subset `23 passed, 2 skipped`; focused
  launch-kit/readiness tests `16 passed`; launch readiness report fails only on
  local `WANDB_API_KEY source`, which is intentionally waived for the offline
  Prime pod launch; archive hashes match local and remote.
- 2026-07-03 Prime-compatible S2 dataset repair: the first pod SFT launch
  reached trainer startup but failed loading the old
  `jayshah5696/humanize-rl-prime-sft-messages-env0315-clean50` dataset because
  its nested `quality` column was exported by Dataset Viewer as `_type: Json`.
  Prime's pinned `datasets` stack rejected that feature before reading
  `messages`.
- Published the active replacement dataset
  `jayshah5696/humanize-rl-prime-sft-messages-env0315-clean50-primecompat` at
  HF commit `8f1d484cea21affed944479fdb3ef590de03a6ba`. It keeps the same
  `4358/242/243` train/validation/test rows and drops nested training columns
  from the Hub data files. Dataset Viewer now reports only `Value` features plus
  `messages` as `List`; no `Json` feature remains.
- S2 preflight now rejects any HF feature exposing `_type: Json`, so this Prime
  loader failure is caught before launch next time.
- The active S2 config now points to the `-primecompat` dataset. Rebuilt launch
  artifacts:
  `config_sha256=40071c8db47c0830a21d6dfb65c6a787971d0ab8aa20877663385ea68d12ade9`,
  guarded `archive_sha256=191342cf4bb07e5741e18dbe4c509037285b311a9cc17d129d7fa08ad6ea1836`,
  guarded `runner_sha256=02e117cc4934b77fc5ab5c65ff0bc2fcf9f04eb080ddab91b70e8ca8c0e9dbf7`.
- The first `-primecompat` SFT run passed dataset load and emitted step-0
  metrics, then failed in the Qwen3.5 gated-delta Conv1d path with
  `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`.
- The launch runner now writes a pod-side `sitecustomize.py`, disables
  `torch.backends.cudnn.enabled`, and exports `TORCH_CUDNN_V8_API_DISABLED=1`.
- The active Prime A100 pod run is live, not local:
  pod `62abc46cde1f4705b0ce65ab702005ae`, PID `22250`, outer log
  `/workspace/s2_sft_logs/run_sft_cudnn_guard_20260703T054716Z.log`, trainer log
  `/workspace/prime-rl/outputs/prime_sft/qwen35_2b_sft_target_messages_env0315_clean50_gate_env0315/logs/trainer.log`.
  The run loaded the `-primecompat` dataset and entered
  `Starting training loop (max_steps=200)`.
- Live SFT evidence: step 0 validation loss `2.0132`; step 0 train loss
  `2.1609`; step 1 train loss `2.1475`; step 1 grad norm `2.1875`; LR
  `2.00e-05`; step 1 throughput `2545 tokens/s`; step 2 train loss `2.0988`;
  step 2 grad norm `1.7734`; step 2 throughput `2544 tokens/s`; peak memory
  `20.2/79.2 GiB`; latest GPU poll showed `64%` utilization and
  `21653/81920 MiB` used.
- Next handoff: let S2 SFT reach a real checkpoint, then run the existing S2
  promotion gates. Only after promotion should the after-SFT RL config be
  rendered and launched.

## What Not To Do Next

- Do not deploy the step-200 checkpoint as a candidate model.
- Do not run another 200-step RL job with the current reward.
- Do not jump to Qwen 2B/9B before the fake-casual reward hole is closed.
- Do not treat aggregate p50/strict reward as enough.
- Do not call the Modal 5-step SFT smoke a useful SFT model.
- Do not spend budget on larger MoE models until the small smoke stops reward
  hacking.
- Do not start a duplicate S2 pod or duplicate flash-attn build while
  `62abc46cde1f4705b0ce65ab702005ae` is active.
- Do not point active Prime SFT runs at the old S2 clean50 Hub repo; use
  `jayshah5696/humanize-rl-prime-sft-messages-env0315-clean50-primecompat`.

## References

- Prime Hosted Training models and pricing:
  `https://docs.primeintellect.ai/hosted-training/models-and-pricing`
- Prime Hosted Training advanced configs:
  `https://docs.primeintellect.ai/hosted-training/advanced-configs`
- Prime sandboxes overview:
  `https://docs.primeintellect.ai/sandboxes/overview`
- Prime GPU pods / provision instance:
  `https://docs.primeintellect.ai/cli-reference/provision-gpu`
- Prime SSH key API:
  `https://docs.primeintellect.ai/api-reference/ssh-keys/get-ssh-keys`
- Pangram REST API quickstart:
  `https://docs.pangram.com/quickstart-rest`
- Pangram Python SDK:
  `https://docs.pangram.com/sdk/python`
- Liquid model library:
  `https://docs.liquid.ai/lfm/models/complete-library`
