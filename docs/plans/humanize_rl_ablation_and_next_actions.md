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

## What Not To Do Next

- Do not deploy the step-200 checkpoint as a candidate model.
- Do not run another 200-step RL job with the current reward.
- Do not jump to Qwen 2B/9B before the fake-casual reward hole is closed.
- Do not treat aggregate p50/strict reward as enough.
- Do not call the Modal 5-step SFT smoke a useful SFT model.
- Do not spend budget on larger MoE models until the small smoke stops reward
  hacking.

## References

- Prime Hosted Training models and pricing:
  `https://docs.primeintellect.ai/hosted-training/models-and-pricing`
- Prime Hosted Training advanced configs:
  `https://docs.primeintellect.ai/hosted-training/advanced-configs`
- Liquid model library:
  `https://docs.liquid.ai/lfm/models/complete-library`
