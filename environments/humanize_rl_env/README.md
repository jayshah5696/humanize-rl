# humanize-rl-env

**Prime Intellect Verifiers** single-turn environment for Humanize-RL.

Published at: `jayshah5696/humanize-rl-env`

---

## What this does

Each episode is a single writing task — rewrite a corporate Slack update into
casual human prose, compress a meeting recap, or draft a direct workplace
message. The model receives the task prompt, produces one completion, and the
environment scores it.

---

## Reward Modes

Default training mode:

```
reward_mode="p50_50_no_penalty"
reward = 0.50 × ridge_rubric + 0.50 × deterministic
         clipped to [-1.0, 1.0]
```

Penalties are still computed and logged as diagnostics, but they do not affect
the optimizer scalar in this mode.

Strict eval mode:

```
reward_mode="strict"
reward = 0.50 × ridge_rubric + 0.50 × deterministic + penalties
         clipped to [-1.0, 1.0]
```

`reward_mode="scalar_softened"` is kept for legacy Modal/TRL comparison runs.

### Ridge Rubric (50%)

Mean of 8 regression heads from a TF-IDF Logistic+Ridge scorer trained on
10,000 Gemini-labelled writing samples (AUROC 0.9988, rubric MSE 0.041).
The scorer estimates the same humanness dimensions the LLM judge uses:

| dim | what it penalizes |
|---|---|
| `structural_symmetry` | rigid intro/body/conclusion templates, formulaic lists |
| `specificity` | vague claims without names, numbers, or concrete context |
| `formality_gradient` | unnatural tone shifts |
| `voice_consistency` | generic assistant voice instead of a specific writer |
| `rhetorical_sophistication` | filler analysis, stakes inflation, shallow reasoning |
| `padding_density` | repeated restatements, zero-information sentences |
| `personality_presence` | lack of opinion, perspective, friction |
| `copula_avoidance` | pompous substitutes for simple is/are/was |

The scorer pkl is **bundled inside the wheel** (`ridge_state.pkl`) — no
external download or API call required.

### Deterministic (50%)

Equal-weight mean of 6 constraint checks that fire on hard task rules:

| component | what it checks |
|---|---|
| `faithfulness` | no invented details, missing numbers/entities, dropped/added facts |
| `task_following` | no option menus, wrapper phrases, or refusals |
| `length` | within `max_words` / `min_words` / `exact_sentences` |
| `format` | no subject lines, signoffs, or forbidden markdown |
| `placeholder` | placeholder usage matches task constraints |
| `clarity` | avg sentence length (≤24 words → 1.0, ≤35 → 0.7, else 0.4) |

### Penalties (diagnostics; additive only in strict mode)

| penalty | value |
|---|---|
| `invented_detail` | −0.50 |
| `forbidden_fact` | −0.50 |
| `option_menu` | −0.40 |
| `missing_number` | −0.40 |
| `missing_entity` | −0.40 |
| `refusal` | −0.40 |
| `placeholder_disallowed` | −0.35 |
| `missing_required_fact` | −0.35 |
| `too_long` | −0.30 |
| `placeholder_required` | −0.30 |
| `ai_tell_phrase` | −0.25 |
| `wrapper_phrase` | −0.20 |
| `sentence_count` | −0.20 |
| `wrong_format_markdown` | −0.20 |
| `subject_line` | −0.20 |
| `too_short` | −0.15 |
| `signoff` | −0.15 |

---

## Metrics logged (zero-weight, diagnostics only)

```
style_metric            — layer1 heuristics + ridge P(human) blend
task_following_metric   — option_menu / wrapper / refusal / length checks
faithfulness_metric     — invented detail / missing facts / forbidden facts
length_metric           — word count within constraints
format_metric           — subject / signoff / markdown checks
clarity_metric          — avg sentence length
placeholder_metric      — placeholder constraint compliance
risk_penalty_metric     — sum of all triggered penalties
option_menu_penalty_metric
wrapper_phrase_penalty_metric
invented_detail_penalty_metric
```

---

## Dataset

Default task set: `mix_v2_p5050_filtered` — 972 tasks rebuilt from the saved
Modal `mix_v2` rollouts using `p50_50_no_penalty`.

Supported `task_set` values:

| task_set | file | rows |
|---|---:|---:|
| `v02_smoke` | `humanize_tasks_v02_smoke.jsonl` | 99 |
| `v03` | `humanize_tasks_v03_filtered.jsonl` | 492 |
| `mix_v2` | `humanize_tasks_rl_mix_v2.jsonl` | 998 |
| `mix_v2_p5050` | `humanize_tasks_rl_mix_v2_p5050_filtered.jsonl` | 972 |
| `mix_v2_p5050_filtered` | alias for `mix_v2_p5050` | 972 |

Pass `task_path=` to `load_environment()` only when testing a custom JSONL.

Task shape (key fields):

```json
{
  "id": "rl_v01_000001",
  "family": "rewrite_repair",
  "domain": "slack",
  "mode": "rewrite",
  "instruction": "Clean up this Slack update. Keep it casual and under 30 words.",
  "input_text": "Please be advised that staging has been restored...",
  "constraints": { "max_words": 30, "preserve_numbers": true, ... },
  "reward_profile": "rewrite_faithful_concise",
  "required_facts": ["staging restored", "missing STRIPE_WEBHOOK_SECRET"],
  "forbidden_facts": ["database migration"],
  "split": "train"
}
```

---

## Usage

```bash
# Install
prime env install jayshah5696/humanize-rl-env

# Run evaluation (3 examples, 2 rollouts each)
prime eval run jayshah5696/humanize-rl-env \
  --model google/gemma-3-4b-it \
  --api-base-url https://openrouter.ai/api/v1 \
  --api-key-var OPENROUTER_API_KEY \
  --num-examples 3 \
  --rollouts-per-example 2

# Python
from verifiers import load_environment
env = load_environment("humanize-rl-env")

# p50 default training set
env = load_environment(
    "humanize-rl-env",
    split="train",
    task_set="mix_v2_p5050",
    reward_mode="p50_50_no_penalty",
)

# strict eval gate
env = load_environment(
    "humanize-rl-env",
    split="validation",
    task_set="v03",
    reward_mode="strict",
)
```

---

## Package structure

```
humanize_rl_env/
├── __init__.py              # load_environment(), preview_dataset_row()
├── ridge_state.pkl          # bundled TF-IDF+Ridge scorer (1.1 MB)
├── humanize_tasks_v02_smoke.jsonl
├── humanize_tasks_v03_filtered.jsonl
├── humanize_tasks_rl_mix_v2.jsonl
├── humanize_tasks_rl_mix_v2_p5050_filtered.jsonl
├── reward/
│   ├── reward.py            # score_response(), RewardResult, load_ridge_scorer()
│   ├── checks.py            # deterministic penalty checks
│   ├── tasks.py             # RLTask model, load_tasks()
│   ├── profiles.py          # legacy reward profiles (reference)
│   ├── verifiers_adapter.py # Prime Verifiers async reward + metric functions
│   ├── grpo_rewards.py      # TRL/GRPO reward functions
│   └── env.py               # HumanizeRLEnv, prime_dataset_row()
└── scoring/
    ├── aggregator.py        # layer1 score_text()
    ├── layer1.py            # 8 regex/heuristic dimensions
    └── patterns.py          # AI-tell regex patterns
```

All scoring logic is **self-contained** — no external `humanize-rl` package
install required. The wheel includes the ridge pkl and smoke dataset.

---

## Versions

| version | change |
|---|---|
| 0.3.13 | Ignore subject-title and discourse false entities such as Compliance Review, Firstly, Secondly, and Understanding |
| 0.3.12 | Scaffold fact filtering plus invented-number/time, unsupported-negation, low-overlap, em-dash, inline-closing, and AI-tell surface caps |
| 0.3.10 | Core semantic-failure cap for p50 deterministic reward |
| 0.3.9 | Formal salutation and inline/multiline signature caps for direct-output tasks |
| 0.3.8 | Emoji, hashtag, all-caps, and expanded fake-social/recommendation diagnostics |
| 0.3.7 | Deterministic semantic suitability cap for uplifting romance recommendation failures |
| 0.3.6 | Target-word length and repetition diagnostics/caps for p50 training |
| 0.3.5 | Hosted-training example_id recovery for reshaped rollout inputs |
| 0.3.2 | Hosted-training prompt recovery; explicit question/answer/example_id fields |
| 0.3.1 | p50_50_no_penalty mode; default mix_v2_p5050 task set; v03/mix_v2 bundles |
| 0.1.7 | 50/50 ridge-rubric + deterministic formula; 8 rubric dims exposed |
| 0.1.6 | ridge scorer bundled as state dict pkl |
| 0.1.5 | self-contained bundle, passes prime eval run end-to-end |
| 0.1.0 | initial push |
