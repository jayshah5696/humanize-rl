from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUT = Path("paper/figures/sft")
OUT.mkdir(parents=True, exist_ok=True)

COLOR_BASE = "#7a7a7a"
COLOR_FT = "#1b7837"
COLOR_ACCENT = "#2166ac"
COLOR_WARN = "#b35806"


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def theme() -> None:
    sns.set_theme(style="white", font="DejaVu Sans", font_scale=0.95)
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "grid.color": "#e6e6e6",
            "grid.linewidth": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_dataset_funnel() -> None:
    manifest = load_json("data/processed/sft/gemma4_e2b_v04/manifest.json")
    rows = pd.DataFrame(
        [
            {"stage": "Raw judged rows", "count": manifest["raw_rows"]},
            {"stage": "Accepted after gates", "count": manifest["accepted_rows"]},
            {"stage": "Train split", "count": manifest["split_counts"]["train"]},
            {"stage": "Validation split", "count": manifest["split_counts"]["valid"]},
            {"stage": "Test split", "count": manifest["split_counts"]["test"]},
        ]
    )
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    colors = [COLOR_ACCENT, COLOR_FT, "#5aae61", "#a6dba0", "#d9f0d3"]
    ax.barh(rows["stage"], rows["count"], color=colors, height=0.55)
    for idx, row in rows.iterrows():
        ax.text(row["count"] + 45, idx, f"{row['count']:,}", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("rows")
    ax.set_ylabel("")
    ax.set_title("SFT data funnel", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.4)
    sns.despine(left=True)
    plt.tight_layout()
    plt.savefig(OUT / "sft_data_funnel.png", dpi=260, bbox_inches="tight")
    plt.close()


def save_training_card() -> None:
    metrics = load_json("outputs_gemma4_humanize_mlx_full_r8/metrics.json")
    config = load_json("outputs_gemma4_humanize_mlx_full_r8/run_config.json")
    values = pd.DataFrame(
        [
            {"metric": "Rows", "value": metrics["rows"], "label": f"{metrics['rows']:,}"},
            {"metric": "Optimizer steps", "value": metrics["steps"], "label": f"{metrics['steps']:,}"},
            {"metric": "Minutes", "value": metrics["runtime_seconds"] / 60, "label": f"{metrics['runtime_seconds'] / 60:.1f}"},
            {"metric": "Sec / step", "value": metrics["seconds_per_step"], "label": f"{metrics['seconds_per_step']:.2f}"},
            {"metric": "Train loss", "value": metrics["train_loss"], "label": f"{metrics['train_loss']:.3f}"},
            {"metric": "LoRA rank", "value": config["lora_r"], "label": f"r={config['lora_r']}"},
        ]
    )
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.axis("off")
    x_positions = [0.03, 0.19, 0.36, 0.52, 0.68, 0.84]
    for x, (_, row) in zip(x_positions, values.iterrows(), strict=True):
        ax.text(x, 0.63, row["label"], fontsize=18, fontweight="bold", ha="center", color="#222222")
        ax.text(x, 0.42, row["metric"], fontsize=9, ha="center", color="#555555")
    ax.text(0.0, 0.95, "Local MLX-Tune full-epoch run", fontsize=13, fontweight="bold", ha="left")
    ax.text(
        0.0,
        0.13,
        f"Model: {config['model_name']} · max length {config['max_seq_length']} · effective batch {config['effective_batch_size']} · hardware {config['hardware'].get('cpu_brand', 'Apple Silicon')}, {config['hardware'].get('memsize_gb', '?')}GB unified memory",
        fontsize=8.5,
        color="#555555",
        ha="left",
    )
    plt.tight_layout()
    plt.savefig(OUT / "sft_training_card.png", dpi=260, bbox_inches="tight")
    plt.close()


def save_before_after_metrics() -> None:
    summary = load_json("outputs_gemma4_humanize_mlx_full_r8/eval_v01/summary.json")
    rows = []
    metric_map = {
        "distilled_rubric_mean": "Rubric mean ↑",
        "distilled_ai_probability_mean": "AI probability ↓",
        "layer1_mean": "Layer 1 ↑",
        "word_count_mean": "Words ↓",
        "option_style_count": "Option menus ↓",
        "ai_tell_total": "AI tells ↓",
    }
    for variant, values in summary.items():
        for key, label in metric_map.items():
            rows.append({"variant": "Base" if variant == "base" else "Fine-tuned", "metric": label, "value": values[key]})
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.8))
    for ax, metric in zip(axes.flatten(), metric_map.values(), strict=True):
        sub = df[df["metric"] == metric]
        sns.barplot(data=sub, x="variant", y="value", ax=ax, palette=[COLOR_BASE, COLOR_FT], hue="variant", legend=False)
        ax.set_title(metric, loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(axis="y", alpha=0.35)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
    plt.suptitle("Base vs fine-tuned outputs on 10 hand-written eval prompts", x=0.01, ha="left", fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(OUT / "sft_before_after_metrics.png", dpi=260, bbox_inches="tight")
    plt.close()


def save_prompt_level_delta() -> None:
    rows = load_jsonl("outputs_gemma4_humanize_mlx_full_r8/eval_v01/generations_scored.jsonl")
    df = pd.DataFrame(
        [
            {
                "id": row["id"].replace("eval_", ""),
                "variant": "Base" if row["model_variant"] == "base" else "Fine-tuned",
                "rubric": row["scores"]["distilled_rubric_mean"],
                "ai_probability": row["scores"]["distilled_ai_probability"],
                "words": row["scores"]["word_count"],
            }
            for row in rows
        ]
    )
    pivot = df.pivot(index="id", columns="variant", values="rubric").reset_index()
    pivot["delta"] = pivot["Fine-tuned"] - pivot["Base"]
    pivot = pivot.sort_values("delta")
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    colors = [COLOR_WARN if value < 0 else COLOR_FT for value in pivot["delta"]]
    ax.barh(pivot["id"], pivot["delta"], color=colors, height=0.55)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Fine-tuned minus base distilled rubric mean")
    ax.set_ylabel("")
    ax.set_title("Prompt-level rubric change", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.35)
    plt.tight_layout()
    plt.savefig(OUT / "sft_prompt_delta.png", dpi=260, bbox_inches="tight")
    plt.close()


def main() -> None:
    theme()
    save_dataset_funnel()
    save_training_card()
    save_before_after_metrics()
    save_prompt_level_delta()
    print(f"Wrote SFT figures to {OUT}")


if __name__ == "__main__":
    main()
