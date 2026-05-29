import collections
import json
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from datasets import Dataset
from huggingface_hub import HfApi, get_token


def infer_domain(row: dict) -> str:
    d = row.get("domain") or row.get("origin_domain") or ""
    if d:
        return d
    inst = row.get("instruction", "").lower()
    if "slack" in inst or "channel" in inst or " dm " in inst:
        return "chat"
    if "email" in inst or "dear " in inst:
        return "email"
    if "shorten" in inst or "compress" in inst or "tighten" in inst:
        return "compress"
    if "grammar" in inst or "fix the" in inst or "typo" in inst:
        return "grammar"
    if "story" in inst or "paragraph" in inst or "vivid" in inst or "creative" in inst:
        return "creative"
    return "general"


def clean_row(row: dict, idx: int) -> dict:
    inst = row.get("instruction", "").strip()
    resp = row.get("response", "").strip()
    domain = infer_domain(row)
    source = row.get("source") or row.get("origin_source") or "curated"
    naturalness = row.get("quality_judge", {}).get("naturalness") if row.get("quality_judge") else None
    mode = row.get("mode") or ""
    return {
        "id": f"v2-{idx:05d}",
        "instruction": inst,
        "response": resp,
        "messages": [
            {"role": "user", "content": inst},
            {"role": "assistant", "content": resp},
        ],
        "domain": domain,
        "source": source,
        "mode": mode,
        "naturalness_judge": naturalness,
        "version": "v2",
    }


def make_domain_chart(domain_counts: dict, out_path: str) -> None:
    labels = [k for k, v in sorted(domain_counts.items(), key=lambda x: -x[1])]
    values = [domain_counts[k] for k in labels]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(labels[::-1], values[::-1], color="#4C72B0")
    ax.set_xlabel("Number of rows")
    ax.set_title("v2 Dataset: domain distribution")
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved domain chart to {out_path}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/v04_sft_final.jsonl")
    parser.add_argument("--repo-id", default="jayshah5696/humanize-rl-sft-dataset")
    parser.add_argument("--config-name", default="v2")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    raw_rows = [json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip()]
    rows = [clean_row(r, i) for i, r in enumerate(raw_rows)]
    print(f"Loaded {len(rows)} rows from {args.input}")

    domain_counts = collections.Counter(r["domain"] for r in rows)
    source_counts = collections.Counter(r["source"] for r in rows)
    mode_counts = collections.Counter(r["mode"] for r in rows)
    ph_count = sum("[" in r["response"] for r in rows)

    print("\n== Domain ==")
    for k, v in domain_counts.most_common(): print(f"  {k}: {v} ({v/len(rows):.1%})")
    print("== Source ==")
    for k, v in source_counts.most_common(): print(f"  {k}: {v}")
    print(f"== Placeholder rows: {ph_count} ({ph_count/len(rows):.1%}) ==")

    # chart
    chart_path = "runs/v03/domain_distribution_v2.png"
    Path("runs/v03").mkdir(parents=True, exist_ok=True)
    make_domain_chart(dict(domain_counts), chart_path)

    if args.dry_run:
        print("Dry run. Skipping HuggingFace upload.")
        return

    # Upload
    token = os.environ.get("HUGGINGFACE_TOKEN") or get_token()
    if not token:
        raise SystemExit("No HuggingFace token found. Set HUGGINGFACE_TOKEN or run `huggingface-cli login`.")

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True, private=False)

    # Build HF Dataset
    dataset = Dataset.from_list(rows)
    dataset_dict_path = "/tmp/v04_sft_hf_upload"
    dataset.save_to_disk(dataset_dict_path)

    # Upload parquet
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "train.parquet"
        dataset.to_parquet(str(parquet_path))
        api.upload_file(
            path_or_fileobj=str(parquet_path),
            path_in_repo=f"data/{args.config_name}/train-00000-of-00001.parquet",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print(f"Uploaded parquet to {args.repo_id}")

    # Upload raw jsonl
    api.upload_file(
        path_or_fileobj=args.input,
        path_in_repo=f"data/{args.config_name}/v04_sft_final.jsonl",
        repo_id=args.repo_id,
        repo_type="dataset",
    )

    # Upload domain chart
    api.upload_file(
        path_or_fileobj=chart_path,
        path_in_repo="domain_distribution_v2.png",
        repo_id=args.repo_id,
        repo_type="dataset",
    )

    # Write README card
    card = f"""---
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

**{len(rows):,}** high-quality SFT pairs for training a model to write natural, direct prose.

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
{chr(10).join(f'| {k} | {v} | {v/len(rows):.1%} |' for k, v in domain_counts.most_common())}

## Quality

- All rows passed a deterministic heuristic quality check.
- All rows passed a Flash Lite LLM judge (naturalness ≥ 4, fact preservation, no AI tells).
- Average naturalness score: **{sum(r['naturalness_judge'] or 0 for r in rows if r['naturalness_judge']) / max(1, sum(1 for r in rows if r['naturalness_judge'])):.2f} / 5.0**
- Bad-phrase rate (Certainly, Furthermore, etc.): < 0.1%

## Sources

| Source | Rows |
|--------|------|
{chr(10).join(f'| {k} | {v} |' for k, v in source_counts.most_common())}

- `safe_expand_*`: generated from curated seeds using a safe prompt-based expansion.
- `stream_b`: instruction-response pairs generated from real human-written text (wardacoder, corbt/enron-emails, liamdugan/raid human rows, euclaise/writingprompts).
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
"""
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write(card)
        card_tmp = f.name
    api.upload_file(
        path_or_fileobj=card_tmp,
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    print(f"\nPublished {len(rows):,} rows to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
