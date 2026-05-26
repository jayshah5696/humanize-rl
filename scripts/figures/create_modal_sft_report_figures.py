from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUT = Path("paper/figures/sft")
OUT.mkdir(parents=True, exist_ok=True)
BASE_COLOR = "#777777"
MLX_COLOR = "#2166ac"
MODAL_COLOR = "#1b7837"
WARN_COLOR = "#b35806"


def theme() -> None:
    sns.set_theme(style="white", font="DejaVu Sans", font_scale=0.95)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "grid.color": "#e6e6e6", "grid.linewidth": 0.6})


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def save_modal_loss_curve() -> None:
    hist = pd.read_csv("outputs_gemma4_humanize_modal_verified_eval/wandb_history.csv")
    train = hist.dropna(subset=["train/global_step", "train/loss"]).copy()
    evals = hist.dropna(subset=["train/global_step", "eval/loss"]).copy()
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    ax.plot(train["train/global_step"], train["train/loss"], color=MODAL_COLOR, linewidth=1.2, alpha=0.75, label="train loss")
    if not evals.empty:
        ax.scatter(evals["train/global_step"], evals["eval/loss"], color=WARN_COLOR, s=28, label="eval loss", zorder=3)
    ax.set_title("Modal H100 SFT loss curve", loc="left", fontweight="bold")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT / "modal_sft_loss_curve.png", dpi=260, bbox_inches="tight")
    plt.close()


def save_eval_comparison() -> None:
    mlx = load_json("outputs_gemma4_humanize_mlx_full_r8/eval_v01/summary.json")
    modal = load_json("outputs_gemma4_humanize_modal_verified_eval/summary.json")
    rows = []
    source_map = {
        "Base (MLX eval)": mlx["base"],
        "MLX LoRA": mlx["fine_tuned"],
        "Base (Modal eval)": modal["modal_base"],
        "Modal merged": modal["modal_merged"],
    }
    metrics = {
        "option_style_count": "Option menus ↓",
        "ai_tell_total": "AI tells ↓",
        "word_count_mean": "Mean words ↓",
        "layer1_mean": "Layer 1 ↑",
        "distilled_rubric_mean": "Track A rubric ↑",
        "distilled_ai_probability_mean": "AI probability ↓",
    }
    for variant, values in source_map.items():
        for key, label in metrics.items():
            rows.append({"variant": variant, "metric": label, "value": values[key]})
    df = pd.DataFrame(rows)
    palette = {"Base (MLX eval)": BASE_COLOR, "MLX LoRA": MLX_COLOR, "Base (Modal eval)": "#aaaaaa", "Modal merged": MODAL_COLOR}
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.2))
    for ax, metric in zip(axes.flatten(), metrics.values(), strict=True):
        sub = df[df["metric"] == metric]
        sns.barplot(data=sub, x="variant", y="value", hue="variant", palette=palette, legend=False, ax=ax)
        ax.set_title(metric, loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.35)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=7, padding=2)
    plt.suptitle("Base vs fine-tuned behavior across local and Modal runs", x=0.01, ha="left", fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(OUT / "modal_mlx_eval_comparison.png", dpi=260, bbox_inches="tight")
    plt.close()


def save_training_runtime_card() -> None:
    modal_summary = load_json("outputs_gemma4_humanize_modal_verified_eval/summary.json")
    hist = pd.read_csv("outputs_gemma4_humanize_modal_verified_eval/wandb_history.csv")
    last = hist.dropna(subset=["train/global_step"]).iloc[-1]
    values = [
        ("Rows", "4,585"),
        ("Steps", "1,147"),
        ("Runtime", "22.3 min"),
        ("Train loss", "0.580"),
        ("Eval loss", "2.490"),
        ("Option menus", f"5 → {modal_summary['modal_merged']['option_style_count']}"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.axis("off")
    for idx, (label, value) in enumerate(values):
        x = 0.04 + idx * 0.16
        ax.text(x, 0.62, value, ha="center", fontsize=17, fontweight="bold", color="#222222")
        ax.text(x, 0.42, label, ha="center", fontsize=9, color="#555555")
    ax.text(0.0, 0.95, "Modal H100 full SFT run", fontsize=13, fontweight="bold")
    ax.text(0.0, 0.14, f"Final logged step: {int(last['train/global_step'])}; merged artifact parity verified before upload.", fontsize=8.5, color="#555555")
    plt.tight_layout()
    plt.savefig(OUT / "modal_sft_training_card.png", dpi=260, bbox_inches="tight")
    plt.close()


def main() -> None:
    theme()
    save_modal_loss_curve()
    save_eval_comparison()
    save_training_runtime_card()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
