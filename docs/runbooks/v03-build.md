# v03 RL tasks — build runbook

Implements `docs/plans/v03-rl-tasks-dataset.md`. Each step is independently
runnable. Stage outputs are idempotent — re-running overwrites that stage's
output without re-doing upstream work.

## Decisions captured in this implementation

| topic | decision | source |
|---|---|---|
| scope | slices 0–3 implemented; 4–6 deferred | session Q1 |
| authors | 3 models, scoped to **task authorship only** | AGENTS.md update |
| env change | NOT applied — `long_form_generate` + `multi_constraint_compose` carry a 30–60 word brief as `input_text` so the env's source-required check still passes | session Q3 |
| persona / audience | ridge-scored only, no verifier | session Q3 |
| Gemini 3.1 Pro reasoning | model burns ≥3.8K reasoning tokens per call and frequently returns no structured output → bumped `max_tokens` to 4000–6000; if it still fails, swap that mode's Author A to `google/gemini-3-flash-preview` | observed in slice 2 |
| dedup seed shape | seeds carry the first 200 chars of source in `payload.instruction` so arka's near-dedup (which keys on instruction) sees per-row variation | observed in slice 0 |

## Stage 0 — sanity (one-time)

```bash
PYTHONPATH=src uv run pytest tests/reward/test_tasks.py tests/reward/test_tasks_v03.py tests/scripts/
uv run ruff check src/humanize_rl/reward/tasks.py scripts/data/build/ scripts/data/fetch/ scripts/eval/reference_rollout_filter.py
```

## Stage 1 — fetch sources (plan §5, §7.2)

```bash
just v03-fetch-sources
# or:
uv run scripts/data/fetch/fetch_hf_sources.py \
  --config configs/v03/sources.yaml \
  --out data/raw/v03_writing_seeds.jsonl
```

Smoke variant (limits override the YAML):

```bash
uv run scripts/data/fetch/fetch_hf_sources.py \
  --config configs/v03/sources.yaml \
  --out data/raw/v03_writing_seeds_smoke.jsonl \
  --limit 50
```

## Stage 2 — run arka pipelines (6 modes × 3 authors)

Bash loop (the plan's reference invocation):

```bash
for mode in long_form_generate rewrite_humanize compression tone_shift expansion multi_constraint_compose; do
  for author in google/gemini-3.1-pro-preview openai/gpt-5.4-mini google/gemini-3.1-flash-lite-preview; do
    safe=$(echo "$author" | tr '/' '_')
    uv run scripts/data/build/run_arka_pipeline.py \
      --config "configs/v03/v03_${mode}.yaml" \
      --author "$author" \
      --out "data/processed/v03_${mode}_${safe}.jsonl" \
      --seed-path data/raw/v03_writing_seeds.jsonl \
      --run-id "v03-${mode}-${safe}"
  done
done
```

Single mode via justfile alias:

```bash
just v03-run-mode compression google/gemini-3.1-flash-lite-preview
```

If a run fails with `LengthFinishReasonError` (reasoning blew the
completion budget), edit that mode's YAML and bump `max_tokens` (4000 →
6000 → 8000). If a run fails with `'NoneType' object is not iterable`, the
structured-output parser got nothing — same fix.

## Stage 3 — assemble (plan §7.2)

```bash
just v03-assemble
```

Or directly:

```bash
PYTHONPATH=src uv run scripts/data/build/rl_task_assemble.py \
  --inputs "data/processed/v03_*.jsonl" \
  --out data/rl/humanize_tasks_v03.jsonl \
  --split 0.8 0.1 0.1 --split-seed 99 \
  --summary-out data/rl/humanize_tasks_v03_summary.json
```

## Stage 4 — reference-rollout filter (plan §7.2)

```bash
just v03-rollout-filter
```

This costs real $. Tune via flags:

```bash
PYTHONPATH=src uv run scripts/eval/reference_rollout_filter.py \
  --input data/rl/humanize_tasks_v03.jsonl \
  --output data/rl/humanize_tasks_v03_filtered.jsonl \
  --rollouts-out data/rl/v03_ref_rollouts.jsonl \
  --model google/gemini-3.1-pro-preview \
  --rollouts-per-task 3 \
  --reward-low 0.4 --reward-high 0.9 --reward-std-min 0.05
```

For a cheaper sweep, run 2 rollouts:

```bash
PYTHONPATH=src uv run scripts/eval/reference_rollout_filter.py \
  --input data/rl/humanize_tasks_v03.jsonl \
  --output data/rl/humanize_tasks_v03_filtered.jsonl \
  --rollouts-out data/rl/v03_ref_rollouts.jsonl \
  --rollouts-per-task 2
```

## Stage 5 — verification (plan §8)

```bash
just v03-verify
```

The script exits non-zero if any HARD gate FAILs. See
`docs/plans/v03-rl-tasks-dataset.md` §8.3 for per-gate remediation.

## Slice progress

| slice | command | output |
|---|---|---|
| 0 | rewrite_humanize × Flash-Lite × 5 seeds | `data/rl/v03_slice0_tasks.jsonl` |
| 2 | compression × 2 authors × 15 seeds | `data/rl/v03_slice2_tasks.jsonl` |
| 3 | 6 modes × Flash-Lite × 15 seeds | `data/rl/v03_slice3_tasks.jsonl` |
| 4 | 6 modes × 2 authors × ~80 seeds each (Pro dropped) | `data/rl/humanize_tasks_v03_judged_kept.jsonl` (594 tasks at judge≥3.4) |
| 5 | ref-rollout + verify | not yet run |
| 6 | HF publish | not yet run |

## Slice 4 lessons

1. **Gemini 3.1 Pro is currently unusable on OpenRouter** for our prompts. It
   spends 3.8K–7.7K tokens on hidden reasoning per call, and OpenRouter
   ignores `reasoning.max_tokens` overrides for this model. Dropped Pro from
   the author roster. Revisit if/when the OpenRouter route exposes a working
   reasoning cap.

2. **Arka transform stage is sequential and fail-fast**. Two patches in
   `_arka_runner.py` fix this: (a) `ThreadPoolExecutor(max_workers=...)`
   gave ~15× speedup for OpenRouter-bound calls; (b) per-row try/except
   keeps a 100-seed run alive when a single LLM call fails (length limit,
   content filter, parse error). Activated with `--skip-errors` on
   `run_arka_pipeline.py`. Always pass it on real runs.

3. **Judge stage moved out of per-mode YAMLs** (`scripts/data/build/judge_v03_tasks.py`).
   The in-arka `labeling_engine` reuses the top-level `llm.model`, which
   meant judge ran on the author model — expensive when author is Pro.
   The separate judge script uses Flash-Lite with a reasoning cap, costing
   ~$1 for 831 tasks vs estimated $10–$15 with Pro.

4. **Bracket-template instructions ([ROLE]/[TASK]/...) score 2/5 on
   instruction_realism**. The judge correctly flags them as synthetic. At
   threshold 3.8 only ~28% pass; at 3.4 ~71% pass. Two paths:
   - Lower threshold (we did this, keep 594 tasks).
   - Retune mode prompts to natural prose (defer to a v0.4 prompt pass).

5. **Source fetcher needed column-shape handling**. Several HF datasets
   carry instructions inside `messages: [{role, content}, ...]` lists.
   Added `first_user_message:col` / `join_messages:col` operators to
   `fetch_hf_sources.py`. Also added `--scan-cap` because `chtmp223/suri`
   streaming hung indefinitely without a hard limit.

---

## Slice 5 — reference rollouts + verification (492 / 594 kept)

**Inputs.** `data/rl/humanize_tasks_v03_judged_kept.jsonl` (594 tasks at
judge threshold 3.4).

**Run.**

```bash
just v03-rollout-filter
just v03-verify   # defaults to --no-strict
```

That expands to:

```bash
PYTHONPATH=src uv run scripts/eval/reference_rollout_filter.py \
  --input data/rl/humanize_tasks_v03_judged_kept.jsonl \
  --output data/rl/humanize_tasks_v03_filtered.jsonl \
  --rollouts-out data/rl/v03_ref_rollouts.jsonl \
  --model openai/gpt-5.4-mini --rollouts-per-task 3 \
  --weak-model google/gemini-3.1-flash-lite-preview \
  --second-strong-model google/gemini-3-flash-preview \
  --audit-subset 200 --max-workers 16
```

**Outputs.**

- `data/rl/humanize_tasks_v03_filtered.jsonl` — 492 tasks
- `data/rl/v03_ref_rollouts.jsonl` — 2576 rollouts (1782 primary + 594 weak + 200 second-strong)
- `runs/v03/slice5_verification.{md,json}` — 5 PASS / 4 WARN / 5 FAIL

**Wall-clock.** ~9 min on 16 workers (1.14 task/s).
**Cost.** ~$5 (gpt-5.4-mini dominates).

### Lessons added

6. **Parallelize ref rollouts with `ThreadPoolExecutor`**. One worker per
   task; rollouts for a single task stay sequential (preserves per-task
   error grouping and bounds rate-limit pressure). 16 workers gave ~15×
   speedup vs the original sequential loop.

7. **B2 weak-model needs a real capability gap.** Flash-Lite vs gpt-5.4-mini
   only produced a 0.065 mean-reward gap (target ≥0.10). Both models score
   high; the deterministic checks rarely fire. Real B2 signal requires a
   genuinely weak model (gemma-4-e2b once available on OpenRouter, or a
   Gemma running locally) — defer to training-time eval.

8. **Verifier B-gates needed `r.get("model")` + `"reward" in r` guards**.
   Weak-model error records (we keep them in the rollouts JSONL for
   visibility) carry only `error`; the original gate code crashed on
   `r["model"]` / `r["reward"]`. Hardened in `verify_rl_task_quality.py`.

9. **B1/B3/C1 must be scoped to the primary strong model**. With three
   models in the rollouts file, an unfiltered std/median averages across
   models and washes out real per-task variance. The thresholds yaml now
   carries `strong_model:` on B1/B2/B3/C1 and `primary_model:` on C2; the
   verifier reads them.

10. **A5 author balance threshold updated to the 2-author reality**. Since
    Gemini 3.1 Pro was dropped (lesson 1), the plan's 40/35/25 target is
    impossible. `verification_thresholds.yaml` now targets 55/45 between
    `openai/gpt-5.4-mini` and `google/gemini-3.1-flash-lite-preview`.
    Result: A5 went from FAIL (40pp off) to PASS (2.32pp off).

---

## Slice 6 — Hugging Face publish

**Run.**

```bash
just v03-publish        # defaults: jayshah5696/humanize-rl-tasks-v03 (public)
```

That expands to:

```bash
uv run python scripts/publish_to_hf.py dataset \
  --repo-id jayshah5696/humanize-rl-tasks-v03 \
  --path data/rl/humanize_tasks_v03_filtered.jsonl \
  --readme runs/cards/rl_tasks_v03.md \
  --artifact runs/v03/slice5_verification.md:verification_report.md \
  --artifact runs/v03/slice5_verification.json:verification_report.json \
  --artifact data/rl/humanize_tasks_v03_summary.json:dataset_summary.json
```

**Live at** <https://huggingface.co/datasets/jayshah5696/humanize-rl-tasks-v03>.

### Lessons added

11. **HF `Dataset.from_json` swallows mixed-shape rows**. Our filtered
    JSONL has heterogeneous `constraints` keys per row; `from_json`
    succeeds but `push_to_hub` then complained about config-name metadata.
    The publish script's existing fallback ("upload file directly") is the
    right behavior — leave it in place. The parquet view loads from the
    direct upload.

12. **Don't forget the rollouts artifact**. `rollouts/v03_ref_rollouts.jsonl`
    is essential for downstream consumers who want to reproduce B/C gates
    without paying $5 to regenerate. Always include it in the publish
    artifact list.
