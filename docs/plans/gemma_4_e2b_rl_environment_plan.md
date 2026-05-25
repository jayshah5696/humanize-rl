# Gemma 4 E2B Humanize-RL Environment and Full RL Plan

**Date:** 2026-05-24  
**Status:** Proposed implementation plan  
**Starting policy:** `jayshah5696/gemma4-e2b-humanize-unsloth-merged`  
**Goal:** Build a real RL environment and diverse task suite for training Gemma 4 E2B-it to produce one faithful, concise, human-sounding answer without option menus, invented details, or AI-writing tells.

---

## 1. Core Decision

We should move toward full RL, but the first artifact is not a trainer. The first artifact is an **RL environment plus task dataset**.

SFT taught the model a better default style. RL should now stress the remaining behavioral failures:

- option menus instead of one answer;
- wrapper phrases like "Here's a version";
- invented names, dates, projects, tools, and teammates;
- placeholder overuse;
- unnecessary subject lines and signoffs;
- verbosity inflation;
- markdown/list overuse;
- too-polished corporate tone;
- loss of facts in rewrite/compression tasks;
- wrong register, e.g. Slack sounding like email.

The first RL target should not be vague global "humanness." It should be:

> single usable answer, faithful to the prompt, concise, correct register, no AI tells, no invented details.

This is concrete enough to reward.

---

## 2. Why the RL Dataset Must Differ from SFT

The SFT dataset teaches behavior by imitation. The RL task dataset should stress behavior through episodes.

Each row should define:

1. **Observation:** the user instruction plus optional source text.
2. **Action:** the model's generated response.
3. **Constraints:** explicit task rules that can be checked.
4. **Reward profile:** which reward components matter for this task.
5. **Trap tags:** known failure modes this task is designed to expose.
6. **Split:** train, validation, or frozen test.

Unlike SFT, RL rows do not always need gold responses. For rewrite tasks, the source text is the semantic reference. For direct-generation tasks, reward comes from instruction following, format checks, risk checks, and optional LLM judge scoring.

---

## 3. Task Families

The RL suite should cover the following families.

| Family | Purpose | Main failure modes tested |
|---|---|---|
| `rewrite_repair` | stiff draft to natural version | fact drift, over-polish, wrapper text |
| `direct_email` | write short workplace/client emails | fake names, subject/signoff overuse, corporate filler |
| `slack_chat` | quick peer/team messages | too formal, too long, needy tone, placeholders |
| `compression` | long note to 1-3 sentence update | dropped facts, length failure, vague summary |
| `tone_shift` | make text warmer/candid/less inflated | register mismatch, blandness, hype retention |
| `technical_explain` | natural explanation for junior engineer/user | over-markdown, fake technical claims, lecture tone |
| `product_copy_cleanup` | marketing bot to PM/user-facing voice | hype, buzzwords, vague benefits |
| `candidate_customer_comms` | sensitive hiring/support messages | over-apology, filler, wrong warmth |
| `adversarial_ai_tell_removal` | remove known AI phrases while preserving facts | shortcut phrase removal without meaning preservation |
| `placeholder_discipline` | decide when placeholders are allowed | placeholder overuse or invented specifics |

---

## 4. Diversity Axes

The task suite should be generated as a matrix, not as a flat pile of prompts.

Every task should vary across some combination of:

- **domain:** email, Slack, technical, product, leadership, support, hiring, creative/general;
- **mode:** direct generation, rewrite, compress, tone shift, repair;
- **register:** casual, neutral, warm-professional, candid, terse, technical, founder-like;
- **constraint type:** max words, exact sentence count, no subject line, no signoff, use placeholders, do not use placeholders, preserve numbers, preserve names, return only answer;
- **trap type:** missing names, fake deadline temptation, vague corporate source, overlong input, AI-tell-heavy input, emotionally sensitive context.

This prevents RL from overfitting to one prompt pattern.

---

## 5. Dataset Sizes

Use staged scale-up.

| Stage | Rows | Purpose |
|---|---:|---|
| Smoke | 100 | Validate schema, reward code, and rollout loop |
| Pilot | 500 | Short GRPO/DAPO run and manual inspection |
| Full RL v1 | 5,000-10,000 | First real RL run |
| Full RL v2 | 25,000+ | Only after reward is proven robust |

A frozen evaluation split should be created early and never used for RL updates.

Recommended first split for `v01`:

```text
train: 80%
validation: 10%
test: 10%
```

Hold out by template family, source group, and trap cluster where possible. Do not let near-identical task templates leak across splits.

---

## 6. RL Task Schema

Canonical JSONL row:

```json
{
  "id": "rl_v01_000001",
  "family": "slack_rewrite",
  "domain": "chat",
  "mode": "rewrite",
  "register": "casual_direct",
  "instruction": "Clean up this Slack update. Keep it casual and under 30 words. Return only the message.",
  "input_text": "Please be advised that the staging environment has been restored to full operational status. The root cause was identified as a missing environment variable, and we will continue monitoring performance parameters throughout the afternoon.",
  "constraints": {
    "max_words": 30,
    "exact_sentences": null,
    "return_only_answer": true,
    "no_subject_line": true,
    "no_signoff": true,
    "preserve_numbers": true,
    "preserve_entities": true,
    "allow_placeholders": false,
    "require_placeholders_for_missing_specifics": false
  },
  "reward_profile": "rewrite_faithful_concise",
  "trap_tags": ["over_polish", "option_menu", "verbosity"],
  "source": "synthetic_template_v01",
  "split": "train"
}
```

Optional fields for richer tasks:

```json
{
  "forbidden_phrases": ["here's a version", "certainly", "furthermore"],
  "required_facts": ["staging is restored", "missing environment variable", "monitoring this afternoon"],
  "forbidden_facts": ["database migration", "Q3 dashboard", "Sarah"],
  "reference_response": null,
  "source_group": "template_slack_status_001",
  "license": "project_synthetic",
  "release_eligible": true
}
```

---

## 7. Reward Design

The reward should combine soft style scoring with hard failure penalties.

Initial formula:

```text
reward =
  0.35 * style_reward
+ 0.30 * task_following_reward
+ 0.20 * faithfulness_reward
+ 0.10 * length_reward
+ 0.05 * format_reward
- risk_penalty
```

Important: Track A is not the reward by itself. It is a feature inside `style_reward`. The scorer is lexical and hackable, so hard penalties and task checks must dominate when a failure is obvious.

### 7.1 Reward Components

**Style reward**

- Layer 1 deterministic score.
- Track A AI-pattern probability inverted as `1 - p_ai`.
- Optional Track A rubric mean.
- Known AI-tell phrase audit.

Use caps so the model cannot maximize style while ignoring the task.

**Task-following reward**

Checks whether the output satisfies the instruction:

- one answer, not multiple options;
- answer only, no explanation of the rewrite;
- correct task type;
- obeys exact sentence or word constraints;
- correct register.

Some checks can be deterministic. Borderline cases can use a cheap LLM judge outside the inner loop or during periodic evaluation.

**Faithfulness reward**

For rewrite/compression tasks:

- preserve numbers;
- preserve named entities;
- preserve required facts;
- avoid forbidden facts;
- avoid unsupported additions.

For direct-generation tasks:

- use placeholders when missing details are required;
- avoid placeholders when concrete details are provided;
- do not invent specifics not in the prompt.

**Length reward**

- Full credit inside task-specific range.
- Smooth penalty for mild over/under length.
- Hard penalty for verbosity inflation.

**Format reward**

- No subject line when forbidden.
- No signoff when forbidden.
- Slack messages should not look like emails.
- Email tasks may allow subject/signoff only when requested or useful.
- Avoid markdown lists unless the prompt asks for structure.

**Risk penalty**

Hard penalties for dangerous or project-specific failures.

---

## 8. Hard Penalties

Initial penalty table:

| Failure | Penalty |
|---|---:|
| Option menu / multiple alternatives | -0.40 |
| Starts with wrapper phrase like "Here's" | -0.20 |
| Invented name/date/project/tool/company | -0.50 |
| Placeholder when disallowed | -0.35 |
| Missing required placeholder when details are absent | -0.30 |
| Missing required number/entity | -0.40 |
| Subject line when forbidden | -0.20 |
| Signoff when forbidden | -0.15 |
| Too long by more than 50% | -0.30 |
| Known AI-tell phrase | -0.25 |
| Refuses harmless writing task | -0.40 |
| Changes task type, e.g. summarizes instead of rewrites | -0.40 |

Reward should be clipped to a stable range, e.g. `[-1.0, 1.0]` or `[0.0, 1.0]` depending on trainer requirements.

---

## 9. Reward Profiles

Different task families need different reward weights.

### 9.1 `rewrite_faithful_concise`

```text
0.30 style
0.30 faithfulness
0.20 task following
0.10 length
0.10 format
- risk penalties
```

Use for stiff draft cleanup, AI-tell removal, and naturalization tasks.

### 9.2 `direct_workplace_message`

```text
0.35 task following
0.25 style
0.20 format
0.10 length
0.10 placeholder/specificity discipline
- risk penalties
```

Use for Slack, email, status updates, and quick requests.

### 9.3 `compression_update`

```text
0.30 fact preservation
0.25 length constraint
0.20 clarity
0.15 style
0.10 format
- risk penalties
```

Use for incident updates, leadership summaries, and long-to-short transformations.

### 9.4 `technical_explain_natural`

```text
0.30 correctness/adherence
0.25 clarity
0.20 naturalness
0.15 structure restraint
0.10 length
- fake claim / over-list penalties
```

Use for natural technical explanations. This may require an LLM judge for correctness during evaluation, not necessarily every training rollout.

### 9.5 `sensitive_comms`

```text
0.30 task following
0.25 tone appropriateness
0.20 concision
0.15 no corporate filler
0.10 format
- over-apology / invented-detail penalties
```

Use for candidate rejection, customer issue updates, and uncomfortable workplace messages.

---

## 10. Task Dataset Creation Strategy

Use four sources.

### 10.1 Failure Mining from Current SFT Outputs

Mine failures from base and current SFT evaluations.

Examples:

- model produced option menu -> create `option_menu` trap;
- model added `[Teammate's Name]` unnecessarily -> create `placeholder_disallowed` trap;
- model kept subject line/signoff -> create `email_overformat` trap;
- model said "Here's a more candid version" -> create `wrapper_phrase` trap;
- model turned Slack into polished email -> create `register_mismatch` trap.

This source is highest value because it targets known model weaknesses.

### 10.2 Template-Generated Task Matrix

Programmatically generate tasks from controlled templates:

```text
domain x mode x register x constraint x trap
```

Example templates:

- "Clean up this Slack update..."
- "Write a short email to..."
- "Turn this rough incident note into..."
- "Rewrite this product note so it sounds less inflated..."
- "Explain this concept to a junior engineer..."
- "Make this candidate rejection warmer but not apologetic..."

Templates should vary entities, constraints, and registers without creating fake over-specific benchmark tasks.

### 10.3 Real-Text Rewrite Contexts

Reuse approved Track B sources as source material, but convert them into RL tasks instead of SFT targets:

- business/email text;
- Enron-derived cleaned bodies;
- RAID human rows;
- WritingPrompts/general prose;
- technical snippets from approved sources.

For these rows, the reward can use source text as semantic reference.

### 10.4 Adversarial AI-Tell Tasks

Generate or collect texts that contain explicit AI-writing fingerprints:

- "furthermore";
- "moreover";
- "it is worth noting";
- "in conclusion";
- "I hope this email finds you well";
- "please do not hesitate";
- inflated phrases such as "robust", "seamless", "unlock", "empower", "mission-critical".

Task: rewrite naturally while preserving facts.

These tasks are useful for RL because the reward can directly verify removal, but they should not dominate the dataset or the model may over-optimize phrase deletion.

---

## 11. RL Environment Design

Start with a stateless single-turn environment.

```text
observation = rendered user prompt from task row
action = model response
reward = reward_fn(task_row, action)
done = true
info = component scores + triggered penalties
```

The environment should expose detailed reward diagnostics:

```json
{
  "reward": 0.72,
  "components": {
    "style": 0.81,
    "task_following": 0.75,
    "faithfulness": 0.90,
    "length": 0.65,
    "format": 1.00
  },
  "penalties": {
    "option_menu": 0.0,
    "invented_detail": 0.0,
    "wrapper_phrase": -0.2
  }
}
```

This is essential for debugging reward hacking.

---

## 12. Training Path

### 12.1 Offline Reward Validation

Before RL updates, run rollouts from:

1. base Gemma 4 E2B-it;
2. current SFT merged model;
3. optionally local MLX adapter.

The reward should rank current SFT above base on most task families. If it does not, fix the reward before training.

### 12.2 Smoke Run

- 100 tasks.
- 2-4 generations per prompt.
- tiny step count.
- inspect every checkpoint manually.

Goal: prove the environment works and reward is not obviously hackable.

### 12.3 Pilot Run

- 500 tasks.
- current SFT merged model as starting policy.
- frozen SFT model as reference if using KL.
- GRPO or DAPO.
- evaluate on frozen 100-200 task validation split.

Success criteria:

- fewer option menus/wrappers;
- no verbosity inflation;
- no increase in invented details;
- no placeholder overuse;
- win rate over SFT on manual review.

### 12.4 Full RL v1

- 5,000-10,000 tasks.
- DAPO preferred if available and stable.
- GRPO acceptable for first full implementation if DAPO integration is not ready.
- KL to SFT reference.
- overlong reward shaping.
- checkpoint every fixed number of updates.
- evaluate each checkpoint on frozen validation and test tasks.

### 12.5 Preference Tuning Bridge

Even though the goal is full RL, keep a preference-tuning bridge available.

From rollout failures, create pairs:

```text
chosen = concise faithful answer
rejected = option menu / invented detail / overformatted output
```

If full RL is unstable, run DPO/SimPO first and then resume RL.

---

## 13. Evaluation Plan

Evaluate each checkpoint on a frozen task suite.

Metrics:

- mean reward by task family;
- option-menu rate;
- wrapper-phrase rate;
- known AI-tell count;
- invented-detail rate;
- placeholder misuse rate;
- subject/signoff violation rate;
- word-count compliance;
- entity/number preservation;
- Track A AI probability;
- Layer 1 score;
- LLM judge naturalness/instruction-following on a sample;
- manual win rate vs SFT baseline.

The final decision should not use Track A alone.

---

## 14. Guardrails Against Reward Hacking

Likely reward hacks:

1. **Ultra-short outputs:** avoid AI tells by saying too little.
2. **Generic outputs:** sound natural but drop facts.
3. **Placeholder abuse:** avoid inventing details by using placeholders everywhere.
4. **Phrase deletion:** remove banned phrases but keep bad structure.
5. **Casual drift:** make everything Slack-like even when email/professional is requested.
6. **Reward-model overfitting:** optimize lexical scorer while quality degrades.

Countermeasures:

- length lower bounds;
- required fact extraction;
- placeholder allowed/disallowed constraints;
- format constraints by domain;
- frozen external judge eval;
- manual review sample;
- KL to SFT checkpoint;
- cap Track A contribution;
- diversify task families and trap tags.

---

## 15. Proposed Files

```text
data/rl/humanize_tasks_v01.jsonl
src/humanize_rl/reward/__init__.py
src/humanize_rl/reward/checks.py
src/humanize_rl/reward/profiles.py
src/humanize_rl/reward/reward.py
src/humanize_rl/reward/env.py
scripts/build_rl_tasks_v01.py
scripts/evaluate_reward_env.py
scripts/run_rl_rollouts.py
configs/rl/gemma4_e2b_rl_smoke.yaml
configs/rl/gemma4_e2b_rl_pilot.yaml
configs/rl/gemma4_e2b_rl_full_v1.yaml
```

---

## 16. Vertical Slices

### Slice 1 — RL Task Schema and 100-Row Smoke Dataset

**Goal:** Create a small but diverse task dataset.

Deliverables:

- `data/rl/humanize_tasks_v01_smoke.jsonl`
- schema validator;
- counts by family/domain/mode/trap;
- 100 manually inspectable rows.

Acceptance:

- at least 8 task families represented;
- no meta-AI framing unless explicitly adversarial;
- each row has constraints and reward profile;
- train/validation/test split assigned.

### Slice 2 — Deterministic Reward Checks

**Goal:** Implement hard checks and penalties.

Deliverables:

- `src/humanize_rl/reward/checks.py`
- tests for option menus, wrappers, invented names, placeholders, subject lines, signoffs, length, numbers, entities, AI-tell phrases.

Acceptance:

- known bad examples trigger expected penalties;
- known good examples pass;
- all checks return structured diagnostics.

### Slice 3 — Composite Reward Function

**Goal:** Combine style, task, faithfulness, length, format, and risk penalties.

Deliverables:

- `src/humanize_rl/reward/profiles.py`
- `src/humanize_rl/reward/reward.py`
- CLI to score `(task, response)` pairs.

Acceptance:

- reward returns scalar plus component breakdown;
- reward profiles change weights by task family;
- Track A scorer is optional and capped.

### Slice 4 — Offline Rollout Evaluation

**Goal:** Verify reward ranks better behavior higher before training.

Deliverables:

- `scripts/run_rl_rollouts.py`
- `scripts/evaluate_reward_env.py`
- base vs SFT comparison report.

Acceptance:

- SFT model beats base on mean reward;
- reward catches option-menu and wrapper failures;
- manual inspection agrees with reward on a sampled set.

### Slice 5 — RL Environment Wrapper

**Goal:** Expose the reward as a single-turn RL environment.

Deliverables:

- `src/humanize_rl/reward/env.py`
- environment `reset/step` or framework-compatible equivalent;
- JSONL logging of observations, responses, rewards, and diagnostics.

Acceptance:

- one episode per task;
- environment can run through smoke dataset;
- diagnostics are saved for every rollout.

### Slice 6 — GRPO/DAPO Smoke Run

**Goal:** Run the first tiny RL update loop.

Deliverables:

- `configs/rl/gemma4_e2b_rl_smoke.yaml`
- tiny RL run logs;
- before/after generations.

Acceptance:

- run completes without reward/runtime failure;
- no immediate verbosity explosion;
- no obvious collapse in manual sample.

### Slice 7 — 500-Task Pilot Run

**Goal:** Test whether RL improves known failure modes.

Deliverables:

- `data/rl/humanize_tasks_v01_pilot.jsonl`
- pilot training config;
- checkpoint eval report.

Acceptance:

- option-menu/wrapper rate decreases vs SFT;
- no increase in invented details;
- no increase in placeholder misuse;
- manual review prefers RL checkpoint over SFT on sampled tasks.

### Slice 8 — Full RL v1 Dataset and Run

**Goal:** Scale to first real RL run.

Deliverables:

- `data/rl/humanize_tasks_v01_train.jsonl`
- `data/rl/humanize_tasks_v01_valid.jsonl`
- `data/rl/humanize_tasks_v01_test.jsonl`
- full training config;
- checkpoint selection report.

Acceptance:

- 5,000-10,000 diverse training tasks;
- frozen validation/test untouched during training;
- best checkpoint improves over SFT on reward and manual review;
- no reward-hacking failure on held-out traps.
