# Reward Module

Deterministic reward scoring for Humanize-RL training.

---

## Final Reward Formula

```
reward = 0.50 × ridge_rubric + 0.50 × deterministic + penalties
         clipped to [-1.0, 1.0]
```

When the ridge scorer is not loaded (e.g. no pkl found), falls back to:

```
reward = deterministic + penalties
```

---

## Two Halves

### Ridge Rubric (50%)

Mean of 8 regression heads from the TF-IDF Logistic+Ridge scorer
(`models/track_a_10k/ridge.pkl`, AUROC 0.9988, rubric MSE 0.041).

The scorer was distilled from 10,000 Gemini-labelled rows — it estimates
the same 8 humanness dimensions that the LLM judge uses:

| dim | what it measures |
|---|---|
| `structural_symmetry` | rigid intro/body/conclusion templates, formulaic lists |
| `specificity` | vague claims without names, numbers, or concrete context |
| `formality_gradient` | unnatural tone shifts between corporate, academic, casual |
| `voice_consistency` | generic assistant voice vs. a specific writer |
| `rhetorical_sophistication` | filler analysis, stakes inflation, shallow reasoning |
| `padding_density` | repeated restatements, zero-information sentences |
| `personality_presence` | lack of opinion, perspective, humor, friction |
| `copula_avoidance` | pompous substitutes for simple is/are/was |

Each dimension is in [0, 1]. Higher = more human-like.
The ridge mean is scaled by `RIDGE_WEIGHT = 0.50`.

### Deterministic (50%)

Equal-weight mean of 6 constraint-satisfaction checks.
These fire on hard task constraints the model must pass regardless of style:

| component | what it measures |
|---|---|
| `faithfulness` | no invented details, missing numbers/entities, dropped required facts, added forbidden facts |
| `task_following` | no option menus, wrapper phrases, refusals; correct sentence count and word length |
| `length` | within `max_words` / `min_words` / `exact_sentences` constraints |
| `format` | no subject lines, signoffs, or forbidden markdown when disallowed |
| `placeholder` | placeholder usage matches `allow_placeholders` / `require_placeholders` constraints |
| `clarity` | avg sentence length ≤ 24 words → 1.0, ≤ 35 → 0.7, else 0.4 |

Each component is in [0, 1]. The deterministic mean is scaled by `DETERMINISTIC_WEIGHT = 0.50`.

### Penalties (additive)

Applied on top of the 50/50 base. Each is triggered by a specific check
failure and deducted from `raw_reward` before clipping:

| penalty | value | trigger |
|---|---|---|
| `invented_detail` | −0.50 | response adds unsupported names/entities |
| `forbidden_fact` | −0.50 | response includes a fact explicitly forbidden |
| `option_menu` | −0.40 | response offers multiple versions/options |
| `missing_number` | −0.40 | a required number from source is dropped |
| `missing_entity` | −0.40 | a required entity from source is dropped |
| `refusal` | −0.40 | response refuses a harmless writing task |
| `placeholder_disallowed` | −0.35 | `[placeholder]` used when not allowed |
| `missing_required_fact` | −0.35 | a required fact is absent from response |
| `too_long` | −0.30 | word count > 1.5× `max_words` |
| `placeholder_required` | −0.30 | no placeholder when one was required |
| `wrapper_phrase` | −0.20 | response opens with "Here is…", "Sure, …" etc. |
| `sentence_count` | −0.20 | wrong number of sentences when `exact_sentences` set |
| `wrong_format_markdown` | −0.20 | markdown lists used when not allowed |
| `subject_line` | −0.20 | email subject line present when forbidden |
| `too_short` | −0.15 | word count < `min_words` |
| `signoff` | −0.15 | sign-off present when forbidden |
| `ai_tell_phrase` | −0.25 | known AI-tell phrase detected |

---

## RewardResult Fields

```python
@dataclass(frozen=True)
class RewardResult:
    reward: float               # clipped final scalar [-1, 1]
    raw_reward: float           # pre-clip sum
    profile: str                # "50_50_ridge_deterministic" | "deterministic_only"
    components: dict[str, float]  # all individual scores (style, faithfulness, etc.)
    weighted_components: dict[str, float]  # actual contribution to raw_reward:
                                #   ridge_rubric, deterministic,
                                #   ridge_structural_symmetry, ..., ridge_copula_avoidance
    penalties: dict[str, float]   # triggered penalties only (name → value)
    diagnostics: list[dict]       # per-check detail (name, passed, penalty, matches)
```

---

## Files

| file | purpose |
|---|---|
| `reward.py` | `score_response()`, `RewardResult`, `load_ridge_scorer()`, `RidgeScorerAdapter` |
| `checks.py` | all deterministic check functions, `PENALTIES` dict |
| `tasks.py` | `RLTask` Pydantic model, `load_tasks()` |
| `profiles.py` | legacy `RewardProfile` definitions (kept for reference; no longer drives weights) |
| `verifiers_adapter.py` | async reward + metric functions for Prime Verifiers |
| `grpo_rewards.py` | `ridge_rubric_reward`, `deterministic_reward`, `risk_penalty_reward` for TRL/GRPO |
| `grpo_dataset.py` | HF `Dataset` loader for GRPO training |
| `env.py` | `HumanizeRLEnv`, `prime_dataset_row()`, rollout logging |

---

## Fallback Behaviour

If `models/track_a_10k/ridge.pkl` is not found, `load_ridge_scorer()` returns
`None`. `score_response()` then uses `deterministic_only` mode:

```
reward = deterministic_mean + penalties
```

The `weighted_components` dict still exposes `deterministic` but `ridge_rubric`
is absent. All downstream code handles this via `.get("ridge_rubric", 0.0)`.
