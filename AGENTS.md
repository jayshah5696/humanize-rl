# humanize-rl Agent Rules

## Project Context

This is a **consumer** of [arka](https://github.com/jayshah5696/arka) — not a fork.
Arka provides: config-driven YAML pipelines, TransformGeneratorStage, LabelingEngine, rubric scoring, pair-delta filtering.
This project provides: humanness-specific scoring (Layer 1 heuristics), rubric YAMLs, prompt templates, pipeline orchestration, training scripts.

**Do NOT build custom stages in arka.** Everything is expressible as YAML config + prompt template + rubric YAML.

## Tooling

- Use `uv` for all Python dependency management. No pip, no venv, no requirements.txt.
- Use `just` for project tasks. No Makefile.
- `just` targets use `uv run`; one-off tools use `uvx`.
- Manage deps with `uv add` / `uv add --dev`.
- `src/` + `tests/` layout. Tests mirror src structure.
- `pytest` + `ruff` for quality. Run `just test` and `just lint` before committing.

## Development

- Use red/green TDD: write a failing test first, then implement.
- Prefer typed Pydantic models over raw dicts.
- Keep edits minimal, clear, and reversible.
- Ask before adding major frameworks or heavy dependencies.
- Use `ask_user` tool for clarification — do not assume.
- Avoid destructive commands or force operations unless asked.
- use click for CLI tools, not argparse or raw input.
- Be concise in the reposne. Use less words and less whitespace. Do not repeat yourself. Do not explain yourself. 

## Scripts

- Scripts are runnable as `uv run scripts/<script>.py --option value`; do not create a large app-style command router.
- Before adding a new script, check `scripts/README.md` and existing common scripts.
- Prefer common reusable scripts for repeated tasks, especially Hugging Face publishing, HF card generation, and JSONL utilities.
- Planned common scripts: `scripts/publish_to_hf.py`, `scripts/write_hf_card.py`, `scripts/jsonl_tool.py`.
- New or rewritten scripts must use Click for arguments.
- Move superseded scripts to `scripts/archive/` before deleting.
- Follow `docs/plans/scripts-consolidation-and-folder-cleanup.md` for script consolidation and folder cleanup.

## Models — Google Only (with one narrow exception)

All LLM calls go through OpenRouter. Only Google models.

**Exception (v03 RL tasks only):** the `humanize_tasks_v03` *task authorship*
stage may use 3 different houses to avoid baking one model's phrasing into the
task distribution. See `docs/plans/v03-rl-tasks-dataset.md` §3.1.
Approved authors: `google/gemini-3.1-pro-preview`, `openai/gpt-5.4-mini`,
`google/gemini-3.1-flash-lite-preview`. This exception is scoped to *task
generation only*. The humanizer, judge, scorer, and reference-rollout code
paths remain Google-only.

```
# Frontier reasoning (judge, humanize)
google/gemini-3.1-pro-preview          # $2.00/$12.00

# Cheap transforms (aiify, bulk)
google/gemini-3.1-flash-lite-preview   # $0.25/$1.50

# Mid-tier dev
google/gemini-3-flash-preview          # $0.50/$3.00

# Fine-tune target
google/gemma-4-e2b-it                 # Apache 2.0, 2B effective

# Free dev/testing
google/gemma-4-e2b-it:free
google/gemma-4-31b-it:free
```

## Architecture

- **Layer 1** (deterministic, free): 8 regex/heuristic dims in `src/humanize_rl/scoring/`
- **Layer 2** (LLM judge, paid): 8 dims via arka LabelingEngine + `rubrics/humanness_v01.yaml`
- **Pipeline**: Two arka runs (aiify → humanize) stitched by Python orchestrator in `src/humanize_rl/pipeline.py`
- **Training**: SFT first (Gemma 4 E2B, bf16 LoRA). RL only if SFT plateaus.

## What NOT to Do

- Do not add non-Google models to configs
- Do not build custom arka stages — use TransformGeneratorStage + YAML
- Do not use QLoRA for Gemma 4 — bf16 LoRA only
- Do not skip Layer 1 pre-filtering before Layer 2 (wastes API budget)
- Do not rewrite files unrelated to the current task

## Reference Docs (Obsidian Vault)

Design docs live outside this repo in the research vault. Read before making architecture decisions.

```
/Users/jshah/Documents/Obsidian Vault/Assitant/Research/humanize-rl/
├── 01-humanize-rl-design.md    # Master architecture — 4 modules, data flow
├── 02-gap-analysis.md          # Landscape — what exists, what's missing
├── 03-benchmark.md             # Benchmark design — 1300 samples, 3-class
├── 04-rubric.md                # Full rubric v0.1 — Layer 1 + Layer 2 dims
├── 05-rl-environment.md        # RL env — DAPO, verifiers, reward design
├── 06-finetuning.md            # SFT strategy — LoRA config, eval metrics
├── 07-data-pipeline-requirements.md  # Data sources, quality filters, output format
├── 08-two-layer-scoring.md     # Layer 1 implementation — regex, heuristics
├── 09-external-research.md     # External tools — claudiness, AutoRubric, papers
└── 10-build-plan.md            # THIS plan — vertical slices, model choices
```

Arka docs:
```
/Users/jshah/Documents/GitHub/arka/
├── docs/SPEC.md                # Arka architecture — config, stages, records
├── docs/rl-data-needed.md      # What arka provides vs what we need
├── rubrics/sft_quality.yaml    # Example rubric format
└── examples/04-evol-instruct.yaml  # Multi-round pipeline example
```

## RTK

**Always prefix commands with `rtk`**:
```bash
# ✅ Correct
rtk git add . && rtk git commit -m "msg"
# ❌ Wrong
git add . && git commit -m "msg"
```
