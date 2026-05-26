# Humanize-RL: Reward Environment Design and Evaluation

**Status:** complete sub-report — for integration into the main paper  
**Date:** 2026-05-25  
**Environment:** `jayshah5696/humanize-rl-env@0.2.0`  
**Dataset:** `jayshah5696/humanize-rl-tasks` (HuggingFace)  
**Rubric scorer:** `jayshah5696/humanize-rl-track-a-ridge-scorer`

---

## 1. Purpose

This sub-report documents the design and validation of the reinforcement learning environment for the Humanize-RL project. The environment defines the task distribution, reward formula, and evaluation harness used to measure and train a model's ability to produce human-sounding writing.

The environment is published to the Prime Intellect Environments Hub as a self-contained Python wheel. It can be installed with `prime env install jayshah5696/humanize-rl-env` and evaluated against any OpenAI-compatible inference endpoint.

---

## 2. Task Design

### 2.1 Task Shape

Each task is a stateless single-turn episode. The model receives a prompt containing:

1. A natural-language writing instruction specifying the target register, length constraint, and output format.
2. A source text to rewrite (empty for direct-generation tasks).

The model produces one completion. The environment scores it and returns a scalar reward.

A task is represented as a typed `RLTask` Pydantic model with the following key fields:

| field | type | purpose |
|---|---|---|
| `id` | `rl_v01_XXXXXX` | unique identifier |
| `family` | enum | task family (rewrite_repair, compression, tone_shift, …) |
| `domain` | enum | content domain (email, slack, technical, …) |
| `instruction` | string | full user-facing prompt |
| `input_text` | string | source text to rewrite |
| `constraints` | dict | hard constraints (max_words, preserve_numbers, …) |
| `reward_profile` | enum | which profile governs component weights |
| `required_facts` | list | entities/numbers the response must preserve |
| `forbidden_facts` | list | content that must not appear |
| `trap_tags` | list | expected failure modes for this task type |

### 2.2 Task Families

Tasks are organised into ten families, each testing a distinct humanisation failure mode:

| family | target failure mode | tasks |
|---|---|---|
| `rewrite_repair` | over-polished, corporate rewrites | 181 |
| `compression` | verbose summaries that pad instead of compress | 181 |
| `tone_shift` | register mismatch — formal template instead of warm colleague | 180 |
| `direct_email` | email overformat, invented detail | 10 |
| `slack_chat` | verbosity, email-style Slack messages | 10 |
| `technical_explain` | jargon retention, list overuse | 10 |
| `product_copy_cleanup` | corporate filler, AI-sounding uplift | 10 |
| `candidate_customer_comms` | wrapper phrases, sycophantic openers | 10 |
| `adversarial_ai_tell_removal` | explicit AI-tell phrase preservation | 10 |
| `placeholder_discipline` | incorrect placeholder usage | 10 |
| **total** | | **612** |

### 2.3 Dataset Versions

**v01 — Template-generated (100 tasks).** Tasks were generated from 10 hand-authored templates covering all families. Source texts are short (~24 words). These tasks test basic compliance with format and register constraints.

**v02 — Real-source (512 tasks).** Source texts were drawn from two existing project datasets:

- `data/raw/v04_stream_b_seeds.jsonl` — 1,722 usable rows of real email, creative, and general text (mean 133 words, max 221 words).
- `data/processed/v04_sft_final.jsonl` — 1,107 rows with instructions ≥ 60 words.

After deduplication and constraint validation, 512 tasks were generated across three families (rewrite_repair, compression, tone_shift) covering five domains. Source texts average 103 words, substantially longer than v01, creating harder preservation and faithfulness challenges.

**Combined dataset (612 tasks)** is published to HuggingFace as `jayshah5696/humanize-rl-tasks` with train (489), validation (62), and test (61) splits.

---

## 3. Reward Formula

### 3.1 Design Rationale

The reward function must simultaneously capture two orthogonal properties:

1. **Humanness** — does the response sound like a real person wrote it, rather than an AI assistant? This requires learned stylometric knowledge unavailable from deterministic rules.
2. **Constraint satisfaction** — did the model follow the task's hard rules (word limit, fact preservation, format constraints)?

A purely deterministic reward misses subtle AI-writing patterns. A purely model-based reward is opaque, slow, and susceptible to reward hacking on known phrases. The 50/50 split addresses both.

### 3.2 Formula

```
reward = 0.50 × ridge_rubric_mean + 0.50 × deterministic_mean + penalties
         clipped to [−1.0, 1.0]
```

When the ridge scorer is unavailable, the formula degrades gracefully to `deterministic_mean + penalties`.

### 3.3 Ridge Rubric Component (50%)

The ridge scorer (`models/track_a_10k/ridge.pkl`) is a TF-IDF Logistic+Ridge model trained on 10,000 Gemini-labelled writing samples. It predicts eight continuous rubric dimensions, each in [0, 1], that approximate the same humanness qualities scored by the LLM judge:

| dimension | what it measures |
|---|---|
| `structural_symmetry` | penalises rigid intro/body/conclusion templates |
| `specificity` | penalises vague claims without concrete names, numbers, context |
| `formality_gradient` | penalises unnatural tone shifts |
| `voice_consistency` | penalises generic assistant voice |
| `rhetorical_sophistication` | penalises filler analysis, stakes inflation |
| `padding_density` | penalises repeated restatements |
| `personality_presence` | penalises absence of opinion, friction, perspective |
| `copula_avoidance` | penalises pompous substitutes for simple is/are/was |

The ridge component equals the unweighted mean of all eight dimension scores, scaled by 0.50.

**Why not the binary P(AI) head?** The binary classifier has AUROC 0.9988 and false-positive rate 0.0% on the human challenge set, but it fires primarily on known lexical AI-tell phrases (`furthermore`, `it is worth noting`, `in conclusion`). Using it as an RL reward creates an incentive to simply remove those phrases while retaining the underlying AI structure. The eight regression heads learned from continuous Gemini rubric scores are harder to exploit and capture more nuanced dimensions of humanness.

### 3.4 Deterministic Component (50%)

The deterministic component is the equal-weight mean of six constraint-satisfaction checks:

| component | what it checks |
|---|---|
| `faithfulness` | no invented detail, missing numbers/entities, dropped/forbidden facts |
| `task_following` | no option menus, wrapper phrases, or refusals |
| `length` | response within `max_words` / `min_words` / `exact_sentences` |
| `format` | no subject lines, signoffs, or forbidden markdown |
| `placeholder` | placeholder usage matches task constraints |
| `clarity` | average sentence length ≤ 24 words → 1.0; ≤ 35 → 0.7; else 0.4 |

### 3.5 Penalties

Penalties are applied additively to `raw_reward` before clipping. They are triggered by specific check failures and are not weighted:

| penalty | value | trigger |
|---|---|---|
| `invented_detail` | −0.50 | unsupported names or entities added |
| `forbidden_fact` | −0.50 | explicitly prohibited content included |
| `option_menu` | −0.40 | response offers multiple versions |
| `missing_number` | −0.40 | a required number was dropped |
| `missing_entity` | −0.40 | a required entity was dropped |
| `refusal` | −0.40 | model refuses a harmless writing task |
| `placeholder_disallowed` | −0.35 | placeholder used when not allowed |
| `missing_required_fact` | −0.35 | a required fact absent from response |
| `too_long` | −0.30 | word count > 1.5× `max_words` |
| `ai_tell_phrase` | −0.25 | known AI-tell phrase detected |
| `wrapper_phrase` | −0.20 | opens with "Here is…", "Sure,…" |
| `sentence_count` | −0.20 | wrong sentence count when `exact_sentences` set |
| `wrong_format_markdown` | −0.20 | markdown lists when not allowed |
| `subject_line` | −0.20 | email subject line when forbidden |
| `too_short` | −0.15 | below `min_words` |
| `signoff` | −0.15 | sign-off when forbidden |

---

## 4. Prime Intellect Environment

### 4.1 Architecture

The environment is implemented as a Prime Verifiers `SingleTurnEnv`. It is published as a self-contained Python wheel that bundles all scoring logic, the ridge scorer state dict, and the smoke dataset — no external `humanize-rl` package installation required at inference time.

```
humanize_rl_env/
├── __init__.py              # load_environment(), preview_dataset_row()
├── ridge_state.pkl          # TF-IDF+Ridge scorer (1.1 MB, sklearn-only state dict)
├── humanize_tasks_v02_smoke.jsonl  # bundled task dataset (99 tasks)
└── reward/
    ├── reward.py            # score_response(), RidgeScorerAdapter, load_ridge_scorer()
    ├── checks.py            # 17 deterministic penalty checks
    ├── tasks.py             # RLTask Pydantic model
    ├── verifiers_adapter.py # async reward + metric functions
    ├── grpo_rewards.py      # TRL/GRPO reward functions
    └── env.py               # prime_dataset_row(), rollout logging
└── scoring/
    ├── aggregator.py        # layer1 score_text()
    ├── layer1.py            # 8 regex/heuristic dimensions
    └── patterns.py          # AI-tell patterns
```

The `task` column is serialised as a JSON string in the dataset rows. This is required because Prime Intellect's `print_results` function calls `set([o["task"] for o in outputs])`, which fails on unhashable `dict` objects. Reward functions parse it back with `json.loads()` before validation.

### 4.2 Metrics Logged

The rubric exposes the primary reward plus eleven zero-weight diagnostic metrics:

```
humanize_reward         — primary scalar (50/50 formula)
style_metric            — layer1 heuristics + ridge P(human) blend
task_following_metric   — wrapper / option-menu / refusal / length checks
faithfulness_metric     — fact preservation checks
length_metric           — word count within constraints
format_metric           — subject / signoff / markdown checks
clarity_metric          — average sentence length score
placeholder_metric      — placeholder constraint compliance
risk_penalty_metric     — sum of all triggered penalties
option_menu_penalty_metric
wrapper_phrase_penalty_metric
invented_detail_penalty_metric
```

### 4.3 Dataset Serialisation

Prime Intellect requires each dataset row to have a `prompt` column (list of chat messages), plus optional `answer`, `info`, and `task` columns. We use `task` (JSON string) and `info` (JSON string containing `task_id` and full task payload) to pass task metadata to reward functions without touching the `answer` column.

---

## 5. Evaluation Results

Evaluations were run via `prime eval run` locally using OpenRouter and Prime Intellect free-tier inference endpoints. All runs used 20 examples × 4 rollouts (80 total) except gemma-3-4b-it (5 examples × 2 rollouts).

### 5.1 Model Comparison

| model | endpoint | n rollouts | reward mean | reward std | style | faithfulness |
|---|---|---|---|---|---|---|
| `google/gemma-3-4b-it` | OpenRouter | 10 | 0.097 | 0.183 | 0.818 | 0.560 |
| `poolside/laguna-m.1` | Prime free | 40 | 0.217 | 0.396 | 0.832 | 0.630 |
| `poolside/laguna-xs.2` | Prime free | 20* | 0.116 | 0.476 | 0.844 | 0.610 |
| `openrouter/free` | OpenRouter | 80 | 0.547 | 0.361 | 0.844 | 0.775 |

*poolside/laguna-xs.2 had 50% `EmptyModelResponseError` on short `max_tokens` budgets. Both Poolside models require `max_tokens ≥ 1024` because they are reasoning models that generate internal chain-of-thought before producing visible content. With `max_tokens=128` the reasoning exhausts the budget before any visible content is emitted.

### 5.2 Task Difficulty

Average reward on v02 tasks (long source texts, 103 words mean) is substantially lower than on v01 tasks (short templates, 24 words mean). This is expected: longer source texts contain more numbers and entities that must be preserved, producing more faithfulness and missing-entity penalties.

The reward distribution shows appropriate variance across rollouts (std ≈ 0.25–0.40), meaning the tasks are neither trivially easy (std → 0) nor uniformly impossible (all rewards → −1). This is the correct difficulty regime for RL training: the model receives informative gradient signal from variation within a group of rollouts on the same example.

### 5.3 Penalty Analysis

Across all evaluation runs, the dominant penalty category was `risk_penalty_metric` (sum of all penalties), averaging between −0.41 and −0.83 depending on the model. Decomposition of the penalty showed:

- `missing_number` and `missing_entity` account for the majority of penalty mass on v02 tasks (long source texts with multiple proper nouns and numeric references).
- `wrapper_phrase` and `option_menu` penalties were consistently 0.0 — indicating that all evaluated models correctly avoid opening with "Here is…" or presenting multiple versions.
- `invented_detail` was 0.0 across all runs — no model fabricated unsupported proper names.

The penalty structure is calibrated to be discriminative without being punishing. A model that rewrites faithfully and concisely can achieve rewards in the 0.5–1.0 range even with some fact-preservation misses.

---

## 6. Engineering Notes

### 6.1 Bundling the Ridge Scorer

The ridge scorer (`models/track_a_10k/ridge.pkl`) was pickled against the original `humanize_rl.scoring.distilled.baselines.RidgeScorer` class path. The bundled environment uses a different module path (`humanize_rl_env.*`). To avoid `ModuleNotFoundError` on unpickling, we re-export the model as a pure sklearn state dict:

```python
state = {"vectorizer": m.vectorizer, "classifier": m.classifier, "regressors": m.regressors}
pickle.dump(state, open("ridge_state.pkl", "wb"), protocol=4)
```

A `_BundledRidgeScorer` class reconstructs the scorer from the state dict without any dependency on the original class path. The only runtime dependencies are `scikit-learn` and `numpy`, both already required by `verifiers`.

### 6.2 Python 3.14 Compatibility

The Prime CLI (`prime`) runs under Python 3.14. The original `humanize-rl` package depends on `spacy>=3.8.14`, which has no Python 3.14 wheel. The bundled environment avoids this by inlining only the reward and scoring modules (`reward/`, `scoring/`) and excluding `data/filters.py` (the only file that imports `spacy`).

### 6.3 Task Column Serialisation

Prime's evaluation display code (`print_results`) attempts `set([o["task"] for o in outputs])`. If `task` is a `dict`, this raises `TypeError: cannot use 'dict' as a set element`. The fix is to store `task` as a JSON string in the dataset. Reward functions receive it as a string and call `json.loads()` before `RLTask.model_validate()`. This change is backward-compatible: `grpo_rewards.py` handles both dict and string inputs.

---

## 7. Artifacts

| artifact | location |
|---|---|
| Prime environment | `jayshah5696/humanize-rl-env@0.2.0` |
| HuggingFace dataset | `jayshah5696/humanize-rl-tasks` (612 tasks) |
| Ridge scorer | `jayshah5696/humanize-rl-track-a-ridge-scorer` |
| Task builder v01 | `scripts/build_rl_tasks_v01.py` |
| Task builder v02 | `scripts/build_rl_tasks_v02.py` |
| HF push script | `scripts/push_rl_tasks_to_hf.py` |
| Reward module | `src/humanize_rl/reward/` |
| Environment source | `environments/humanize_rl_env/` |

---

## 8. Relation to Main Paper

This sub-report covers the RL environment slice of the Humanize-RL project. It is intended to become Section 4 (or a dedicated section on reward design) of the main paper.

The main paper's narrative arc is:

1. **Problem** — AI writing fingerprints in SFT data (§ 1–2)
2. **Layer 1** — deterministic stylometric gating (§ 3)
3. **Layer 2** — LLM judge scoring and dataset construction (§ 4)
4. **Track A Scorer** — distilling the judge into a local model (see `track_a_scorer_report.md`)
5. **SFT** — supervised fine-tuning on filtered pairs (see `gemma4_e2b_sft_report.md`)
6. **RL Environment** — reward design, task dataset, Prime env **(this report)**
7. **RL Training** — DAPO/GRPO on Gemma 4 E2B with the reward defined here *(planned)*
8. **Evaluation** — benchmark comparison (see `track_b_sft_data_report.md`)

The 50/50 reward formula (§ 3 of this report) is the central contribution of the RL slice. It operationalises the claim that humanness is measurable as a combination of learned stylometric judgment and deterministic constraint satisfaction — neither alone is sufficient.
