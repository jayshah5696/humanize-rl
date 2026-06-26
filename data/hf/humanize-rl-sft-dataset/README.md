---
configs:
  - config_name: v2
    data_files:
      - split: train
        path: data/v2/train-00000-of-00001.parquet
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
size_categories:
  - 1K<n<10K
---

# humanize-rl-sft-dataset (v2)

**4,835** high-quality SFT pairs for training a model to write natural, direct prose.

Part of the [humanize-rl](https://github.com/jayshah5696/humanize-rl) project — a two-layer scoring and alignment pipeline for training small models to generate natural, human-sounding text.

## What this trains

A model that can:
- Write natural Slack messages and emails from scratch.
- Rewrite stiff/formal/corporate text into direct, human-sounding prose.
- Fix grammar without making text formal.
- Shorten and compress without losing meaning.

## Domain breakdown

| Domain | Rows | % |
|--------|------|---|
| chat | 2288 | 47.3% |
| email | 1453 | 30.1% |
| general | 800 | 16.5% |
| creative | 255 | 5.3% |
| grammar | 26 | 0.5% |
| compress | 13 | 0.3% |

## Quality

- All rows passed a deterministic heuristic quality check.
- All rows passed a Flash Lite LLM judge (naturalness ≥ 4, fact preservation, no AI tells).
- Average naturalness score: **4.91 / 5.0**
- Bad-phrase rate (Certainly, Furthermore, etc.): < 0.1%

## Sources

| Source | Rows |
|--------|------|
| safe_expand_3000_raw | 1802 |
| stream_b | 1617 |
| safe_expand_raw | 667 |
| chat_expanded | 423 |
| curated | 326 |

- `safe_expand_*`: generated from curated seeds using a safe prompt-based expansion (Arka `prompt_based_generator`).
- `stream_b`: instruction-response pairs generated from real human-written text (wardacoder/business-email-dataset, corbt/enron-emails, liamdugan/raid human rows, euclaise/writingprompts).
- `chat_expanded`: Slack/chat focused pairs generated from hand-crafted direct seeds.
- `curated`: hand-verified base seeds.

## Schema

| Field | Type | Description |
|-------|------|-------------|
| id | str | Row ID |
| instruction | str | User task or rewrite request |
| response | str | Natural human-sounding response |
| messages | list | ShareGPT format: user/assistant turns |
| domain | str | chat / email / general / creative / grammar / compress |
| source | str | Data stream origin |
| mode | str | direct_generation or rewrite_humanize |
| naturalness_judge | int | Flash Lite judge score (1-5), null for curated rows |
| version | str | "v2" |

## Previous version

v1 (1,269 rows) is available as the default config. v2 is a full rebuild with broader domain coverage and stricter quality gates.

## Project

This dataset is built and maintained as part of the **humanize-rl** project.

- **GitHub:** [https://github.com/jayshah5696/humanize-rl](https://github.com/jayshah5696/humanize-rl)
- **Goal:** Train small open models (Gemma 4 E2B) to natively produce natural, human-sounding text — both from scratch and by rewriting stiff/formal drafts.
- **Architecture:** Two-layer scoring pipeline (Layer 1 deterministic heuristics + Layer 2 LLM judge), SFT on this dataset, optional RL post-training with DAPO.

## License

Apache-2.0. Source datasets have individual licenses — see [`data/source_manifest_v04.json`](https://github.com/jayshah5696/humanize-rl/blob/main/data/source_manifest_v04.json) in the training repo for attribution details.
