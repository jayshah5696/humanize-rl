---
license: apache-2.0
language:
- en
task_categories:
- text-generation
tags:
- rl
- humanize
- writing
- reward-modeling
- ai-detection
- humanize-rl
pretty_name: humanize-rl RL Tasks v03
size_categories:
- n<1K
---

# humanize-rl RL Tasks v03

492 RL tasks designed to train and evaluate models that produce **human-sounding prose**
under specific constraints. Each row pairs an authored instruction with a constraint
spec consumable by the [`humanize-rl`](https://github.com/jayshah5696/humanize-rl) reward
environment (deterministic checks + Layer-1 heuristics + ridge regression style scorer).

Built per [`docs/plans/v03-rl-tasks-dataset.md`](https://github.com/jayshah5696/humanize-rl/blob/main/docs/plans/v03-rl-tasks-dataset.md).

## What's in a row

| field | meaning |
|---|---|
| `id` | stable task id (`rl_v03_NNNNNN`) |
| `family` | top-level family: `compression`, `rewrite_repair`, `direct_email`, `tone_shift` |
| `mode` | finer mode: `compression`, `expansion`, `long_form_generate`, `multi_constraint_compose`, `rewrite_humanize`, `tone_shift` |
| `domain` / `register_` | content domain + target register |
| `instruction` | authored prompt the model receives |
| `input_text` | optional source material (for rewrite / compression tasks) |
| `constraints` | structured spec: target word count, sentences, forbidden phrases, register, etc. |
| `reward_profile` | which composite profile the reward env applies |
| `quality_judge` | LLM-judge score on the *task itself* (instruction craftsmanship) |
| `task_author_model` | model that authored the task |
| `split` | `train` / `validation` / `test` |

## Distribution (n=492)

- **Splits**: train 390 / val 48 / test 54
- **Families**: compression 104, rewrite_repair 178, direct_email 113, tone_shift 97
- **Modes**: rewrite_humanize 113, compression 104, tone_shift 97, expansion 65, long_form_generate 62, multi_constraint_compose 51
- **Registers**: direct 211, formal 99, casual 66, warm 54, journalistic 29, literary 17, academic 16
- **Domains**: creative_general 436, slack 22, email 12, technical 10, product 6, leadership 4, support 2
- **Task authors**: `openai/gpt-5.4-mini` 57%, `google/gemini-3.1-flash-lite-preview` 43%
- **Reward profiles**: `rewrite_faithful_concise` 275, `direct_workplace_message` 113, `compression_update` 104

## How it was built (4-stage funnel)

1. **Author** instructions from writing seeds with `openai/gpt-5.4-mini` and
   `google/gemini-3.1-flash-lite-preview` (Gemini 3.1 Pro was dropped: OpenRouter
   ignores its reasoning-token caps).
2. **Judge** task quality with `google/gemini-3.1-flash-lite-preview`
   (reasoning_max_tokens=256); keep tasks with judge score ≥ 3.4. → 594 / 831 kept.
3. **Reference rollouts**: per task, sample 3 completions from
   `openai/gpt-5.4-mini` + 1 from `google/gemini-3.1-flash-lite-preview` (weak) +
   1 from `google/gemini-3-flash-preview` on a 200-task audit subset. Score every
   rollout with the humanize-rl reward env (deterministic checks + Layer 1
   heuristics + ridge style scorer).
4. **Filter**: drop **BROKEN** (mean reward < 0.4) and **TRIVIAL** (mean > 0.9
   AND std < 0.05). → 492 / 594 kept (19 broken, 83 trivial).

## Verification gates (slice 5)

5 PASS / 4 WARN / 5 FAIL on 14 gates (see `verification_report.md`).

| gate | status | value |
|---|---|---|
| A2_constraint_density | **PASS** | 8.44 mean populated dims per task |
| A5_author_balance | **PASS** | 2.32pp deviation from 55/45 target |
| B5_penalty_concentration | **PASS** | 0.378 (no single penalty dominates) |
| C1_rollout_std | **PASS** | 0.032 mean per-task reward std |
| C2_cross_model_rho | **PASS** | ρ=0.685 between gpt-5.4-mini and Flash on 200 paired tasks |
| A1_instruction_length | **FAIL** | p10=81 p50=120 p90=180 (target p90≥200) |
| A3_cell_coverage | **FAIL** | 8/40 domain×register cells filled at ≥10 tasks |
| A4_embedding_distance | **WARN** | 0.573 mean pairwise distance |
| B1_strong_reward_band | **WARN** | median 0.807 (slightly above soft 0.80 cap) |
| B2_strong_weak_gap | **FAIL** | gap 0.065 (gpt-5.4-mini 0.780 vs Flash-Lite 0.715) |
| B3_per_mode_variance | **FAIL** | min mode std 0.086 |
| B4_check_firing_rate | **FAIL** | min check rate 0.003 (most checks rarely trigger on strong models) |

## Known limitations

- **Bracket-template ceiling.** Instructions use `[ROLE]/[TASK]/[CONTENT]/[FORMAT]/[MUST]`
  scaffolding; this hits an `instruction_realism` ceiling at LLM-judge score ~2.
  Documented; not retuned in this slice.
- **Domain skew.** ~89% of tasks are `creative_general`. Future slices should
  redistribute toward `technical`, `email`, `slack`, `product`.
- **Author diversity.** Only two authors (no Gemini Pro). The plan called for
  three; we'll re-add a third when a stable third model is available.
- **B2/B3/B4 gates fail** because both strong and weak models already score high
  and check penalties rarely fire. This means the reward signal is dominated by
  ridge style scoring rather than constraint violations — fine for a starting
  RL curriculum, but watch for over-fitting to style during training.

## Repository

- Code: <https://github.com/jayshah5696/humanize-rl>
- Plan: `docs/plans/v03-rl-tasks-dataset.md`
- Reward env: `src/humanize_rl/reward/`
- Verification gates: `scripts/data/build/verify_rl_task_quality.py`

## Citation

```bibtex
@misc{humanize-rl-tasks-v03,
  title  = {humanize-rl RL Tasks v03},
  author = {Shah, Jay},
  year   = {2026},
  url    = {https://huggingface.co/datasets/jayshah5696/humanize-rl-tasks-v03}
}
```
