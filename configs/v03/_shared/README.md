# v03 shared fragments

These YAMLs live in `configs/v03/` and are intentionally **self-contained** —
arka does not support YAML anchors across files. Common stage blocks are
copy-pasted across the 6 mode configs (`v03_*.yaml`). When tuning shared
behaviour (dedup bands, length floors, judge threshold), grep across all 6
files and update consistently.

## Conventions

- `llm.model` is the **default author** (Author A). Override with
  `scripts/data/build/run_arka_pipeline.py --author <slug>` to run Author B / C.
- All configs read seeds from `data/raw/v03_writing_seeds.jsonl`.
- All configs write to `data/processed/v03_<mode>_<author-slug>.jsonl`; the
  `run_arka_pipeline.py` wrapper rewrites `output.path` from `--out`.
- All configs end with a `labeling_engine` stage that judges tasks against
  `rubrics/v03_task_quality.yaml` (threshold 3.8).

## Mode-prompt contract

Every mode generator prompt MUST instruct the model to return JSON with a single
key `text` containing a JSON-encoded `v03_task_payload`:

```json
{
  "instruction": "...",
  "input_text": "...",
  "constraints": { ... TaskConstraints fields ... },
  "domain": "email",
  "register": "direct",
  "trap_tags": ["wrapper_phrase"],
  "reward_profile": "rewrite_faithful_concise"
}
```

`scripts/data/build/rl_task_assemble.py` parses this from `payload.response`,
validates against `RLTask`, and assigns the final `rl_v03_NNNNNN` id +
train/val/test split.
