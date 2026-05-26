# V03 RL Task Dataset Plan

**Date:** 2026-05-25
**Status:** Proposal — iterate before execution
**Scope:** Build `humanize_tasks_v03` to replace v01 (100 templated) + v02 (512 rewrite-only) with a richer, larger, more discriminative RL task dataset.
**Target:** ~2,500 high-quality tasks, ~$40–$80 generation+judging spend, all built via arka YAML pipelines + 3–4 thin Click scripts.

---

## 1. Why v03

### What's wrong with v01/v02

Inspected `data/rl/humanize_tasks_v02.jsonl` (512 tasks):

| problem | evidence |
|---|---|
| Instructions are stamped templates | mean **20 words**, max 23. Only 3 wordings repeated 170× each. |
| `max_words` is the only real constraint | every task carries `max_words ∈ {25,30,40,79,80,86,87}`; no structure, format, persona, audience, or register constraints. |
| Only 3 families × 5 domains | rewrite_repair / compression / tone_shift × email/slack/product/technical/creative. The other 10 v01 families were dropped. |
| Zero long-form generation | every task is `mode="rewrite"` over a ~100-word source. No `write a 400-word…` style prompts exist. |
| `required_facts` is regex slop | first 2 numbers + first 2 entities via regex → noisy `missing_number/missing_entity` penalty mass (the dominant penalty in the env report §5.3). |
| Source pool narrow | ~5 datasets: business emails, RAID, WritingPrompts, FineWeb-edu, peS2o, Slack. 288 real-world sources, ~100 w mean. |

### What "v03 high-quality" means

1. **Rich instructions** — 80–250 words on average, slot-templated (role + task + audience + content + tone + format + must / must_not). Closer to WritingBench (1,500-token queries) and Suri (10-constraint instructions).
2. **Layered, machine-verifiable constraints** — 6–10 independent verifier dims per task instead of "did length fit".
3. **Six task modes** including pure `long_form_generate` (no source) and `multi_constraint_compose` (Suri-style).
4. **Wider domain + register coverage** — 8 domains × 5 registers, balanced quotas.
5. **Reference-rollout filtering** — drop broken tasks (ref reward < 0.4) and trivial tasks (ref ≥ 0.9 with no penalty variation).
6. **Multi-model authorship** — instructions written by 2–3 different frontier models to reduce single-model fingerprints in the task distribution itself.

---

## 2. Task design

### 2.1 Six task modes

| mode | model action | source needed | target response | tasks |
|---|---|---|---|---:|
| `long_form_generate` | write 200–600 w original prose from rich instruction | no | 200–600 w | 600 |
| `rewrite_humanize` | take AI-sounding source, rewrite as human | yes | source × 0.6 ±20% | 600 |
| `compression` | source → bounded summary preserving facts | yes | 30–80 w | 400 |
| `tone_shift` | source → target register | yes | source × 1.0 ±15% | 400 |
| `expansion` | bullets/notes → 200–450 w prose | yes (short) | 200–450 w | 250 |
| `multi_constraint_compose` | satisfy 5–10 explicit constraints (Suri / IFEval style) | optional brief | varies | 250 |
| **total** | | | | **~2,500** |

### 2.2 Instruction structure

Each instruction is a real prompt a human or product would write, not a stamped template. Built from explicit slots; mode dictates which slots are filled.

```
[ROLE]      Pretend you are a senior PM at a fintech startup.
[TASK]      Write the all-hands message announcing that the Q3 launch is slipping by six weeks.
[AUDIENCE]  ~120 engineers, mixed seniority; first all-hands since the layoff round.
[CONTENT]   Cover: revised timeline, what's NOT changing, two concrete things eng leads should do this week.
[TONE]      Direct, accountable, not cheerleading. No corporate filler. No "I'm excited to…".
[FORMAT]    180–260 words. 3 short sections, no bullets, no headings. Plain prose.
[MUST]      Mention "Q3 → mid-November", "feature freeze still 10/15", "Sarah's TLM huddle Friday".
[MUST_NOT]  Don't apologise more than once. No "moving forward". No em-dashes.
```

Instructions average 80–250 w and routinely fill 4–8 slots. v02's typical instruction fills 2.

### 2.3 Constraint vocabulary

Extend `TaskConstraints` (Pydantic) to expose machine-verifiable signals beyond `max_words`. Each becomes one deterministic verifier dim.

| group | fields |
|---|---|
| length | `min_words`, `max_words`, `target_words` + `target_tolerance` |
| structure | `min_sentences`, `max_sentences`, `exact_sentences`, `min_paragraphs`, `max_paragraphs`, `required_section_headings[str]`, `allow_bullets`, `allow_headings` |
| inclusion | `must_include_phrases[str]` (case-insensitive substring) — replaces regex-extracted `required_facts` |
| exclusion | `must_not_include_phrases[str]`, `forbidden_openers[str]`, `forbidden_phrases_global[str]` |
| format | `no_subject_line`, `no_signoff`, `no_markdown`, `must_not_use_em_dash` |
| register | `register_target` enum (casual/warm/direct/formal/academic/journalistic/literary), `max_passive_voice_pct`, `min_contraction_count` |
| persona | `persona_constraint` (free text, ridge-scored only), `audience_constraint` (free text, ridge-scored only) |
| facts | `required_facts[str]` (judge-only), `forbidden_facts[str]`, `must_contain_number_kind` enum |

Hard cap **3** `must_include_phrases` per task to avoid the "missing_number" exploit becoming "missing_phrase".

### 2.4 Length distribution targets

| mode | source_len mean/p90 | instruction_len mean/p90 | response target |
|---|---:|---:|---:|
| long_form_generate | n/a | 180 / 280 | 200–600 w |
| rewrite_humanize | 180 / 350 | 100 / 180 | source × 0.6 |
| compression | 250 / 450 | 90 / 160 | 30–80 w |
| tone_shift | 150 / 280 | 110 / 200 | source × 1.0 |
| expansion | 60 / 120 (bullets) | 130 / 220 | 200–450 w |
| multi_constraint_compose | 0 or short brief | 200 / 320 | varies, hard cap |

Result: a wide instruction-length distribution (peak ~120 w, tail to 320 w) and a wide response-length distribution (30 w → 600 w), versus v02's bimodal 25/80 word cap.

### 2.5 Domain × register matrix

8 domains × 5 registers. Quotas roughly proportional to real writing volume.

| domain | share | example registers |
|---|---:|---|
| email / business | 18% | warm, direct, formal |
| slack / chat | 12% | casual, warm |
| blog / opinion | 15% | direct, journalistic, literary |
| technical explainer | 12% | direct, formal |
| academic / abstract | 10% | academic, formal |
| product copy | 10% | warm, direct |
| creative narrative | 8% | literary, casual |
| candidate / customer comms | 8% | warm, formal |
| journalism / news | 7% | journalistic, direct |

---

## 3. Multi-model task authorship

Generating all instructions with a single model bakes that model's phrasing into the task distribution. Spread authorship across 3 frontier models with different houses.

### 3.1 Model roster (all via OpenRouter, no custom adapters)

| role | model | price (in/out per 1M) | share of generation |
|---|---|---|---:|
| Author A (long instructions, multi-constraint) | `google/gemini-3.1-pro-preview` | $2.00 / $12.00 | 40% |
| Author B (varied phrasing, rewrite_humanize, compression) | `openai/gpt-5.4-mini` | $1.50 / $6.00 | 35% |
| Author C (terse, slack, candidate comms, cheap bulk) | `google/gemini-3.1-flash-lite-preview` | $0.25 / $1.50 | 25% |
| Quality judge | `google/gemini-3.1-pro-preview` | $2.00 / $12.00 | judge stage |
| Reference completion | `google/gemini-3.1-pro-preview` | $2.00 / $12.00 | ref-rollout |

If `gpt-5.4-mini` is not yet stable on OpenRouter in May 2026, swap to `anthropic/claude-haiku-4.5` or `mistralai/medium-3.1` — the principle (3 different houses) holds regardless of exact picks. **`AGENTS.md` currently says "Google only".** Multi-author generation is a deliberate exception for the task corpus only, not for any humanizer/judge/scorer code path; rationale = avoid training a humanizer that just learns to imitate Gemini phrasing in tasks. Document this exception in `AGENTS.md` if approved.

### 3.2 Author assignment per mode

| mode | Author A (Gemini Pro) | Author B (GPT-5.4-mini) | Author C (Gemini Flash-Lite) |
|---|---:|---:|---:|
| long_form_generate | 50% | 40% | 10% |
| multi_constraint_compose | 60% | 35% | 5% |
| rewrite_humanize | 30% | 40% | 30% |
| compression | 25% | 35% | 40% |
| tone_shift | 35% | 35% | 30% |
| expansion | 40% | 35% | 25% |

Each task carries a `task_author_model` field for downstream analysis (e.g. is judge biased toward one author?).

### 3.3 Generation knobs per author

Each author uses a different temperature + system-prompt nudge to widen the lexical distribution:

- Author A: temp 0.7, system = "Write production-quality writing briefs for a creative ops team."
- Author B: temp 0.9, system = "You are drafting test prompts for an SAT-style writing exam."
- Author C: temp 0.6, system = "Generate short workplace messages briefs as if writing slack DMs."

---

## 4. arka utilisation

Confirmed by reading `/Users/jshah/Documents/GitHub/arka/src/arka/pipeline/` and `examples/`. Arka covers **every step** of v03 with stock stages — no custom stages, no fork.

### 4.1 Stages we will use

| arka stage | role in v03 |
|---|---|
| `seed_source` | load `data/raw/v03_writing_seeds.jsonl` (sources + instruction seeds) |
| `normalize_conversation` | normalise heterogeneous source rows |
| `transform_generator` | author rich instructions + constraints from each seed (per-mode prompt) |
| `prompt_based_generator` | for `long_form_generate` and `multi_constraint_compose` (no source needed) |
| `evol_instruct_generator` | optional: deepen / add constraints to a subset of instructions for harder split |
| `exact` + `near` dedup | drop duplicate / near-duplicate instructions and source texts |
| `length` filter | enforce instruction-length floors per mode |
| `language` filter | English-only |
| `complexity_elo_stage` | pairwise complexity ranking → keep mid+hard tasks, drop trivial |
| `double_critic_stage` | optional second-pass quality check |
| `labeling_engine` (rubric mode) | judge tasks against `rubrics/v03_task_quality.yaml` |
| `ifd_stage` | optional: instruction-following difficulty filter |

### 4.2 Pipeline layout (per mode)

One arka YAML per mode (6 configs). All share the dedup/length/language/judge tail. Per-mode head differs only in the generator stage + its prompt template.

```
configs/v03/
  v03_long_form_generate.yaml
  v03_rewrite_humanize.yaml
  v03_compression.yaml
  v03_tone_shift.yaml
  v03_expansion.yaml
  v03_multi_constraint_compose.yaml
  _shared/                 # YAML fragments included via anchors / merging
    llm.yaml               # 3 authors switchable per run
    dedup_filters.yaml
    judge_tail.yaml
```

Authors are switched by overriding `llm.model` per run; we invoke arka 3× per mode (once per author) and concatenate outputs.

### 4.3 Rubric (`rubrics/v03_task_quality.yaml`)

Judge dimensions (gemini-3.1-pro):

- `instruction_realism` (1–5) — is this how a real user would actually phrase the request?
- `constraint_richness` (1–5) — are constraints layered and non-trivial?
- `discriminative_difficulty` (1–5) — does following it well require skill (not just template-matching)?
- `humanness_relevance` (1–5) — does this task expose AI-vs-human writing failure modes?
- `source_usefulness` (1–5, skip if no source) — is the source text good material for the mode?

Keep tasks with overall ≥ 3.8 weighted score. Expect ~60–70% pass rate.

---

## 5. Source pool — what to fetch

Goal: ~3,000 license-clean source rows + instruction seeds, balanced across domains. We add to the existing `data/raw/v04_*` pool.

### 5.1 New sources

| HF dataset | use | license | rows targeted |
|---|---|---|---:|
| `Tongyi-Zhiwen/WritingBench` (1,000 queries, 6 domains, 100 subdomains, ~1500-token instructions) | instruction seeds for `long_form_generate`, `multi_constraint_compose` | check | all 1,000 |
| `chtmp223/suri` (20K multi-constraint long-form instructions) | instruction seeds for `multi_constraint_compose`, constraint-vocabulary mine | CC-BY | 1,500 sampled |
| `allenai/WildBench` writing slice | real-user prompts → seed for `long_form_generate` | ODC-BY | 400 |
| `allenai/tulu-3-sft-personas-instruction-following` | persona + IFEval-style verifiable constraint examples | ODC-BY | 600 |
| `google/IFEval` | lift verifier vocabulary (do not use prompts directly) | Apache-2 | full schema |
| `THUDM/LongWriter-6k` (output 2k–32k words) | sample down → seeds for `long_form_generate` long tail | Apache-2 | 200 |
| `HuggingFaceTB/smollm-corpus` blogs subset | source texts for `tone_shift`, `compression` | ODC-BY | 400 |
| `allenai/peS2o` abstracts | academic source texts | ODC-BY | 300 |
| `nampdn-ai/tiny-textbooks` | technical-explain register | ODC-BY | 200 |

### 5.2 Existing sources to keep

- `wardacoder/business-email-dataset` (50 already → expand to 300)
- `nikcane/slack` (38 → expand to 250 if licence allows; otherwise hold)
- `euclaise/writingprompts` (50 → 200)
- our existing `v04_stream_b_seeds.jsonl` (1.8 MB, ~1,915 rows)

### 5.3 Final raw pool target

~3,500 deduped rows after fetch, balanced 25% email/business · 20% blog/opinion · 20% academic/technical · 15% slack/chat · 10% creative · 10% misc/forum.

### 5.4 License handling

Two redistribution modes for the published `humanize_tasks_v03` dataset:

- **Mode A (recommended for v03 main):** include only license-clean source text; instructions are our own derivatives. Redistributable as one HF dataset.
- **Mode B (optional, separate split):** for sources whose terms forbid redistribution, store only `source_dataset` + `source_row_id` + our instruction + constraints. Users fetch source themselves.

Default to Mode A for v03; reserve Mode B for any later sources that need it.

---

## 6. Reward formula impact

50/50 ridge/deterministic split stays. The richer constraints turn the deterministic component from 2 informative signals (length, missing_number) into **8 informative signals**:

| new deterministic check | from which constraint |
|---|---|
| `slot_compliance_score` | `must_include_phrases` + `must_not_include_phrases` |
| `structure_score` | `min/max/exact_sentences`, `required_section_headings`, `allow_bullets/headings` |
| `register_score` | `max_passive_voice_pct`, `min_contraction_count`, contraction/hedge density per `register_target` |
| `em_dash_penalty` | `must_not_use_em_dash` |
| `opener_score` | `forbidden_openers` |
| `format_strictness_score` | `no_markdown`, `no_subject_line`, `no_signoff` |
| `length_target_score` | `target_words ± target_tolerance` (continuous, replaces binary `max_words`) |
| `number_kind_score` | `must_contain_number_kind` |

Add as new check functions in `src/humanize_rl/reward/checks.py`. Cap each at small penalty values (≤ 0.15) so no single check dominates. Existing checks stay backward-compatible (default to inactive when constraint not set).

### 6.1 Ridge scorer adaptation for long-form

Long completions (400–600 w) may score systematically lower under the current ridge scorer (trained on shorter samples). Mitigation: for any completion > 300 w, compute ridge on first/middle/last 200-w windows and average. Implement in `RidgeScorerAdapter` (env-side, backward compatible).

---

## 7. Generic scripts (aligned with `scripts-consolidation-and-folder-cleanup.md`)

The v01 + v02 build scripts are mode-specific (`build_rl_tasks_v01.py`, `build_rl_tasks_v02.py`). For v03 we generalise — no `build_rl_tasks_v03.py` as a monolith. Instead, **5 generic, reusable scripts** + 6 arka YAMLs.

### 7.1 New / generalised scripts

#### `scripts/data/fetch/fetch_hf_sources.py`

Generic HF dataset fetcher. Replaces `fetch_v04_sources.py`, `fetch_v04_stream_b.py`.

```bash
uv run scripts/data/fetch/fetch_hf_sources.py \
  --config configs/v03/sources.yaml \
  --out data/raw/v03_writing_seeds.jsonl \
  --dedupe-on instruction \
  --normalize-schema rl_seed
```

Where `configs/v03/sources.yaml` is:

```yaml
sources:
  - hf: Tongyi-Zhiwen/WritingBench
    split: train
    limit: 1000
    map:
      instruction: query
      domain: domain
    tags: [seed, long_form_generate]
  - hf: chtmp223/suri
    split: train
    limit: 1500
    sample_seed: 42
    map:
      instruction: instruction
    tags: [seed, multi_constraint_compose]
  - hf: HuggingFaceTB/smollm-corpus
    config: blogs
    split: train
    limit: 400
    map:
      input_text: text
    tags: [source, tone_shift, compression]
  # ... etc
```

#### `scripts/data/build/run_arka_pipeline.py`

Generic arka pipeline runner. Reads an arka YAML, optionally overrides `llm.model`, runs, copies output. Replaces all per-mode orchestrators.

```bash
uv run scripts/data/build/run_arka_pipeline.py \
  --config configs/v03/v03_long_form_generate.yaml \
  --author google/gemini-3.1-pro-preview \
  --out data/processed/v03_long_form_generate_authorA.jsonl
```

#### `scripts/data/build/rl_task_assemble.py`

Generic RL-task assembler. Replaces `build_rl_tasks_v01.py`, `build_rl_tasks_v02.py`. Reads arka outputs (per-mode jsonl) and emits validated `RLTask` rows.

```bash
uv run scripts/data/build/rl_task_assemble.py \
  --inputs data/processed/v03_*.jsonl \
  --out data/rl/humanize_tasks_v03.jsonl \
  --split-seed 99 \
  --split 0.8 0.1 0.1 \
  --summary-out data/rl/humanize_tasks_v03_summary.json
```

#### `scripts/eval/reference_rollout_filter.py`

Generate reference completions with `gemini-3.1-pro`, score with the env reward, drop broken (< 0.4) and trivial (> 0.9 with std < 0.05) tasks.

```bash
uv run scripts/eval/reference_rollout_filter.py \
  --input data/rl/humanize_tasks_v03.jsonl \
  --output data/rl/humanize_tasks_v03_filtered.jsonl \
  --model google/gemini-3.1-pro-preview \
  --rollouts-per-task 3 \
  --reward-low 0.4 \
  --reward-high 0.9 \
  --reward-std-min 0.05
```

#### `scripts/data/build/verify_rl_task_quality.py` (NEW — see §8)

Dataset-level verifier. Emits a pass/fail report on diversity, hardness, consistency. **Blocking gate** before publish.

```bash
uv run scripts/data/build/verify_rl_task_quality.py \
  --input data/rl/humanize_tasks_v03_filtered.jsonl \
  --rollouts data/rl/v03_ref_rollouts.jsonl \
  --report-out runs/v03/verification_report.md \
  --json-out runs/v03/verification_report.json \
  --thresholds configs/v03/verification_thresholds.yaml
```

Exits non-zero if any gate fails. Details in §8.

#### `scripts/publish_to_hf.py` (already on consolidation plan)

Use the planned generic uploader to push `humanize_tasks_v03` to HF.

```bash
uv run scripts/publish_to_hf.py dataset \
  --repo-id jayshah5696/humanize-rl-tasks \
  --path data/rl/humanize_tasks_v03_filtered.jsonl \
  --config-name v03 \
  --readme runs/cards/rl_tasks_v03.md
```

### 7.2 End-to-end pipeline (raw `uv run` commands, no justfile wrapper)

No `just v03-*` target. Each script is a standalone Click command. The orchestrator is a documented runbook (`docs/runbooks/v03-build.md`, written when slice 1 lands), not a wrapper task.

```bash
# ─── 1. Fetch + normalise raw sources (~3,500 rows) ──────────────────────────
uv run scripts/data/fetch/fetch_hf_sources.py \
  --config configs/v03/sources.yaml \
  --out data/raw/v03_writing_seeds.jsonl

# ─── 2. Run arka pipelines: 6 modes × 3 authors = 18 runs ────────────────────
# Each run is one command. Reproducible, restartable, no shell loops.
for mode in long_form_generate rewrite_humanize compression tone_shift expansion multi_constraint_compose; do
  for author in google/gemini-3.1-pro-preview openai/gpt-5.4-mini google/gemini-3.1-flash-lite-preview; do
    safe=$(echo "$author" | tr '/' '_')
    uv run scripts/data/build/run_arka_pipeline.py \
      --config "configs/v03/v03_${mode}.yaml" \
      --author "$author" \
      --out "data/processed/v03_${mode}_${safe}.jsonl"
  done
done

# ─── 3. Assemble validated RL tasks ──────────────────────────────────────────
uv run scripts/data/build/rl_task_assemble.py \
  --inputs "data/processed/v03_*.jsonl" \
  --out data/rl/humanize_tasks_v03.jsonl \
  --split-seed 99 \
  --split 0.8 0.1 0.1

# ─── 4. Reference-rollout filter (drops broken + trivial tasks) ──────────────
uv run scripts/eval/reference_rollout_filter.py \
  --input data/rl/humanize_tasks_v03.jsonl \
  --output data/rl/humanize_tasks_v03_filtered.jsonl \
  --rollouts-out data/rl/v03_ref_rollouts.jsonl \
  --model google/gemini-3.1-pro-preview \
  --rollouts-per-task 3

# ─── 5. Dataset verification (BLOCKING gate — must pass before publish) ──────
uv run scripts/data/build/verify_rl_task_quality.py \
  --input data/rl/humanize_tasks_v03_filtered.jsonl \
  --rollouts data/rl/v03_ref_rollouts.jsonl \
  --report-out runs/v03/verification_report.md \
  --json-out runs/v03/verification_report.json \
  --thresholds configs/v03/verification_thresholds.yaml

# ─── 6. Card + publish ───────────────────────────────────────────────────────
uv run scripts/write_hf_card.py rl-tasks \
  --input data/rl/humanize_tasks_v03_filtered.jsonl \
  --verification runs/v03/verification_report.json \
  --output runs/cards/rl_tasks_v03.md

uv run scripts/publish_to_hf.py dataset \
  --repo-id jayshah5696/humanize-rl-tasks \
  --path data/rl/humanize_tasks_v03_filtered.jsonl \
  --config-name v03 \
  --readme runs/cards/rl_tasks_v03.md
```

Each script is idempotent: re-running a stage overwrites its output but doesn't re-do upstream work. Stage 2 (arka) checkpoints inside each arka run dir (already arka behaviour), so a killed run resumes cleanly. Stages 1, 3, 4, 5 are pure functions of their inputs.

### 7.3 Archive after v03 lands

Move to `scripts/archive/`:

- `build_rl_tasks_v01.py`
- `build_rl_tasks_v02.py`
- `fetch_v04_sources.py`
- `fetch_v04_stream_b.py`

Their behaviour is preserved by `fetch_hf_sources.py` + `rl_task_assemble.py` + per-mode arka YAMLs.

---

## 8. Verification — diverse, hard, consistent

The arka judge gate (`labeling_engine` against `rubrics/v03_task_quality.yaml`) only filters individual tasks. It doesn't check **dataset-level** properties. v03 adds a second blocking gate that runs after assembly + reference rollouts and emits a markdown + JSON report. Publish is blocked unless all hard gates pass.

### 8.1 Three properties, ten gates

Three verification properties. Each has 3–4 measurable gates. Each gate has a hard threshold (blocks publish) and a soft threshold (warns). Thresholds live in `configs/v03/verification_thresholds.yaml` so they can be tuned without code changes.

#### A. Diversity — task distribution actually covers what we claim

| gate | metric | hard fail | soft warn |
|---|---|---|---|
| A1 instruction-length spread | p10 / p50 / p90 of instruction word count | p90 < 200 **or** p10 > 80 (bimodal collapse) | p50 outside [110, 170] |
| A2 constraint density | mean number of populated verifier-dim constraints per task | mean < 4 | mean < 5 |
| A3 domain × register coverage | count of (domain, register) cells with ≥ 10 tasks | < 25 of 40 cells | < 30 of 40 cells |
| A4 embedding diversity | mean pairwise cosine distance between instructions (sentence-transformers MiniLM, 1k sample) | < 0.55 | < 0.62 |
| A5 author balance per mode | max abs deviation from target author shares (§3.2) | > 10 pp on any mode | > 5 pp on any mode |

A4 catches the failure where an author keeps writing the same instruction shape; A2 catches the v02 problem (constraint count ≈ 1).

#### B. Hardness — strong models can't trivially ace it

Run reference rollouts (already in pipeline §7.2 stage 4) with `gemini-3.1-pro-preview` (strong) and one cheap baseline `google/gemma-4-e2b-it` (weak). 3 rollouts × task each.

| gate | metric | hard fail | soft warn |
|---|---|---|---|
| B1 strong-model reward band | median reward of strong model | < 0.40 or > 0.85 | < 0.50 or > 0.80 |
| B2 strong vs weak gap | `mean_reward(strong) − mean_reward(weak)` | < 0.10 | < 0.15 |
| B3 per-mode reward variance | min over modes of reward std | < 0.12 | < 0.18 |
| B4 deterministic-check firing rate | min over 8 det. checks of "fired ≥ once" rate | < 0.15 | < 0.25 |
| B5 penalty concentration | max share of total penalty mass from any single penalty | > 0.50 | > 0.40 |

B1 enforces the "informative gradient" regime cited in the env report §5.2. B2 confirms tasks discriminate. B5 prevents "missing_number is 80% of all penalty" regression that v02 suffered.

#### C. Consistency — same task gives same signal

| gate | metric | hard fail | soft warn |
|---|---|---|---|
| C1 same-model rollout stability | mean over tasks of reward std across 3 strong-model rollouts | > 0.25 | > 0.18 |
| C2 cross-model rank correlation | Spearman ρ between per-task mean reward for strong-A vs strong-B (gemini-pro vs gpt-5.4-mini reference rollouts on a 200-task audit subset) | < 0.50 | < 0.65 |
| C3 judge re-score agreement | Cohen's κ between original judge pass/fail and re-judged pass/fail on a 100-task audit subset | < 0.55 | < 0.70 |
| C4 deterministic-check determinism | bit-exact reproducibility of all 8 deterministic checks over 2 runs of the same input | any non-match | n/a |

C1 is the "low temperature gives stable score" check — if rollouts of the same model on the same task swing wildly, the env is broken. C2 is the "reward isn't single-author overfit" check. C3 audits the judge stage. C4 is a regression test that the deterministic verifier is actually deterministic.

### 8.2 What the verification script does

```
verify_rl_task_quality.py
├── load tasks + rollouts
├── compute 14 metrics across A/B/C
├── compare each to hard + soft thresholds
├── emit markdown report:
│     - top-line table (PASS / WARN / FAIL per gate)
│     - per-mode breakdown table
│     - histograms (saved as PNG side-cars) for length, reward, constraint count
│     - sample of 5 tasks per mode w/ rollouts inline
├── emit json report (machine-readable for HF card)
└── exit 0 if all hard gates pass, exit 1 otherwise
```

The markdown report is committed to `runs/v03/verification_report.md`; the HF dataset card pulls from `runs/v03/verification_report.json` so the published card always shows the latest gate results.

### 8.3 What we do when a gate fails

Each failure has a documented remediation:

| gate | remediation if failed |
|---|---|
| A1 (length spread) | tune per-mode `min/max` instruction length in arka YAMLs; regen affected mode |
| A2 (constraint density) | strengthen author prompt to fill more constraint slots; regen |
| A3 (cell coverage) | over-sample under-represented (domain, register) cells via `fetch_hf_sources.py --filter` |
| A4 (embedding) | add `near` dedup with smaller `lsh_bands`; raise judge `instruction_realism` threshold |
| A5 (author balance) | re-run under-represented author for affected modes |
| B1 (reward band) | if too high: add harder constraints (more `must_not_include`, tighter length); if too low: relax broken constraints (likely judge over-aggressive) |
| B2 (strong vs weak gap) | tasks are too easy — add `forbidden_openers`, `register_target` mismatch tasks |
| B3 (per-mode variance) | mode has collapsed difficulty — diversify source pool for that mode |
| B4 (check firing) | a verifier never fires — either remove it or generate tasks that require it |
| B5 (penalty concentration) | re-weight penalty values in `checks.py`; cap dominant penalty at smaller value |
| C1 (rollout stability) | check temperature settings; if env reward has nondeterminism, fix env |
| C2 (cross-model ρ) | reward formula may be over-fit to one model's phrasing — audit ridge scorer |
| C3 (judge κ) | judge prompt is unstable — add few-shot exemplars to `v03_task_quality.yaml` |
| C4 (det. check) | a check has hidden randomness — bug; fix immediately, blocks publish |

### 8.4 Verification thresholds config

```yaml
# configs/v03/verification_thresholds.yaml
diversity:
  A1_instruction_length:
    hard: { p90_min: 200, p10_max: 80 }
    soft: { p50_min: 110, p50_max: 170 }
  A2_constraint_density: { hard_min: 4, soft_min: 5 }
  A3_cell_coverage:
    cells_total: 40
    hard_min_filled: 25
    soft_min_filled: 30
  A4_embedding_distance: { hard_min: 0.55, soft_min: 0.62 }
  A5_author_balance:    { hard_max_pp: 10, soft_max_pp: 5 }

hardness:
  B1_strong_reward_band:
    hard: { median_min: 0.40, median_max: 0.85 }
    soft: { median_min: 0.50, median_max: 0.80 }
  B2_strong_weak_gap:    { hard_min: 0.10, soft_min: 0.15 }
  B3_per_mode_variance:  { hard_min: 0.12, soft_min: 0.18 }
  B4_check_firing_rate:  { hard_min: 0.15, soft_min: 0.25 }
  B5_penalty_concentration: { hard_max: 0.50, soft_max: 0.40 }

consistency:
  C1_rollout_std:        { hard_max: 0.25, soft_max: 0.18 }
  C2_cross_model_rho:    { hard_min: 0.50, soft_min: 0.65 }
  C3_judge_kappa:        { hard_min: 0.55, soft_min: 0.70 }
  C4_check_determinism:  { hard: bit_exact }
```

### 8.5 Cost of verification

The big-ticket item is B2 (weak-baseline rollouts) and C2 (second-author rollouts on a 200-task audit subset). Both reuse the reference-rollout machinery in stage 4 (§7.2). Marginal cost:

| extra stage | tokens | model | $ |
|---|---:|---|---:|
| B2: weak rollouts (2,500 × 3) | 6.0 in / 3.75 out | gemma-4-e2b-it (free tier) | $0 |
| C2: second-strong rollouts (200 × 3) | 0.48 in / 0.30 out | gpt-5.4-mini | $1.50 |
| C3: judge re-score (100 tasks) | 0.12 in / 0.012 out | gemini-3.1-pro | $0.40 |
| **verification total** | | | **~$2** |

Negligible. The verification budget is part of the §9 total.

---

## 9. Cost projection

Per-task token math (rough):

- arka transform_generate (one author): ~600 in / ~400 out per task
- judge stage: ~1,200 in / ~120 out per task
- reference rollout: ~800 in / ~500 out per task × 3 rollouts

For ~4,000 raw generations → ~2,500 kept tasks:

| stage | tokens (M) | model | $ |
|---|---:|---|---:|
| Author A 40% (1,600 tasks) | 0.96 in / 0.64 out | gemini-3.1-pro | $9.60 |
| Author B 35% (1,400 tasks) | 0.84 in / 0.56 out | gpt-5.4-mini | $4.62 |
| Author C 25% (1,000 tasks) | 0.60 in / 0.40 out | gemini-flash-lite | $0.75 |
| Judge (4,000 tasks) | 4.80 in / 0.48 out | gemini-3.1-pro | $15.36 |
| Reference rollouts (2,500 × 3) | 6.00 in / 3.75 out | gemini-3.1-pro | $57.00 |
| **total** | | | **~$87** |

The reference-rollout stage is the dominant cost. Trim by:

- 2 rollouts instead of 3 → ~$38 / ~$67 total
- Filter ridge-only (skip ref rollouts) for cheaper modes (`compression`, `tone_shift`) → ~$50 total

**Recommended budget: $80 ceiling** for the full v03 build. Tag with `context_tag` before kickoff so we can checkpoint per stage.

---

## 10. Risks & mitigations

| risk | mitigation |
|---|---|
| Multi-author rule conflicts with "Google only" in AGENTS.md | document exception scoped to task corpus generation; humanizer/judge/scorer stay Google-only |
| `must_include_phrases` becomes the new "missing_number" exploit | cap at 3 hard requirements per task; use forbidden lists for the rest |
| Ridge scorer drift on long-form (400–600 w) completions | windowed scoring (first/middle/last 200 w), average; backward-compatible |
| WritingBench / Suri licence ambiguity | verify before fetch; fall back to Mode B (instruction-only redistribution) if needed |
| Cost overrun on reference rollouts | budget cap $80; option to skip ref-rollout for cheap modes |
| Judge bias toward one author | `task_author_model` field on every task; post-hoc audit of judge scores by author |
| arka stages we haven't used | `transform_generator` + `labeling_engine` already used in our existing pipelines — no new arka territory |

---

## 11. Open questions (need user input)

1. **Budget cap?** $40 (skip ref-rollouts) / $80 (recommended) / $120 (3 rollouts everywhere)?
2. **Target size?** 2,500 (~$80) or push to 5,000 (~$160) for better coverage and bigger SFT/RL training set?
3. **Multi-author exception** — OK to widen "Google only" to include `gpt-5.4-mini` + `claude-haiku-4.5` (or equivalents) **for task authorship only**? Humanizer / judge / scorer remain Google-only.
4. **Redistribution mode** — Mode A only (license-clean sources only) or also support Mode B (instruction + source-row-id) for broader source pool?
5. **Long-form generation in env** — accept env-side change to `prime_dataset_row` so `input_text` can be empty for `long_form_generate` and `multi_constraint_compose`? (small change, but it's a v03 env bump too.)
6. **Persona / audience constraints** — ridge-scored only (recommended, low surface area) or attempt verifier (e.g., audience=engineers → check jargon density)?
7. **v03-fetch vs `fetch_hf_sources.py` generalisation timing** — build the generic version now (more upfront), or fork v04-style script for v03 first then generalise (faster slice)?

---

## 12. Vertical slices — end-to-end first, then expand

**Slice 0 is the tracer bullet: a tiny end-to-end run of every stage on a 30-task slice.** This is the change vs the earlier outline — we do not build all the breadth (3,500 sources, 6 modes, 3 authors, 2,500 tasks) before proving the pipeline works end-to-end including verification.

### Slice 0 — End-to-end tracer (30 tasks, 1 mode, 1 author)

Goal: prove every stage works on a tiny dataset, including all 14 verification gates.

- Sources: 30 rows from `wardacoder/business-email-dataset` (already in our pool)
- Mode: `rewrite_humanize` only (simplest; we have v02 baseline to compare)
- Author: `google/gemini-3.1-pro-preview` only
- Run every script in §7.2, end with `verify_rl_task_quality.py`
- Expected: most gates **fail** (too small to satisfy A3/A4, single author fails A5). That's fine — we want to see the report, confirm the **infrastructure** works, fix bugs before spending real money on stages 2 (generate) and 4 (rollouts).
- Gate to slice 1: every script runs end-to-end without error; verification report renders; at least 3 of the 14 gates can be meaningfully evaluated even at n=30.

### Slice 1 — Source fetch (real scale)

- Ship `scripts/data/fetch/fetch_hf_sources.py` + `configs/v03/sources.yaml` covering all 9 sources from §5.1
- Output: `data/raw/v03_writing_seeds.jsonl` (~3,500 rows)
- Gate: manual spot-check of 50 random rows; dedup ratio, domain mix vs §5.3 targets

### Slice 2 — One mode, all 3 authors

- Ship `configs/v03/v03_compression.yaml` (simplest mode)
- Run for all 3 authors → ~300 raw → ~200 kept tasks
- Gate: judge stage passes ≥ 60%; manual review of 10 tasks per author; A5 (author balance) passes on this mode at this scale

### Slice 3 — All 6 modes, 1 author

- Ship the other 5 mode YAMLs
- Run all 6 with Author A only → ~800 raw → ~500 kept
- Gate: judge pass rate ≥ 50% per mode; A3 (domain × register) shows ≥ 15 cells with ≥ 5 tasks

### Slice 4 — Full generation (6 modes × 3 authors)

- 18 arka runs total → ~4,000 raw → ~2,500 kept tasks
- Gate: arka logs complete; judge stage retains ≥ 50%; cost within $35 of projection

### Slice 5 — Reference-rollout filter + verification

- Strong + weak rollouts on all 2,500 tasks
- Run `verify_rl_task_quality.py` against full v03
- Gate: **all hard gates pass**; soft warns documented in the report
- If a hard gate fails: apply §8.3 remediation, regen affected slice, re-verify (this is the iteration loop, not a one-shot)

### Slice 6 — Publish

- Card + HF push
- Gate: dataset accessible via `load_dataset('jayshah5696/humanize-rl-tasks', 'v03')`; card renders verification table from `runs/v03/verification_report.json`

Each slice ends with a `context_tag`. Slices 0–3 are cheap (~$5 total). The expensive stages are 4 (generation) and 5 (rollouts). Failing slice 5 sends us back to slice 2/3/4 for the affected mode only, not the whole dataset.

---

## 13. Relation to other plans

- **scripts-consolidation-and-folder-cleanup.md** — v03 introduces the four generic scripts (`fetch_hf_sources.py`, `run_arka_pipeline.py`, `rl_task_assemble.py`, `reference_rollout_filter.py`) that consolidation plan expected; archives the v01/v02 builders.
- **v03-seed-benchmark-spec.md** — that plan covers human seeds for the benchmark / SFT side; v03-rl-tasks is the RL-side counterpart sharing the same source-fetching infrastructure.
- **gemma_4_e2b_rl_environment_plan.md** — v03 dataset feeds the RL env at `jayshah5696/humanize-rl-env@0.3.0`; reward formula stays 50/50, deterministic verifiers expand to 8 dims.
- **gemma4_rl_modal_20usd_budget_plan.md** — v03 tasks (~2,500) drop into existing GRPO training loop unchanged on the consumer side.
