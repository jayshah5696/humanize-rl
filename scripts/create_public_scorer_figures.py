from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUT = Path("paper/figures/scorer")
OUT.mkdir(parents=True, exist_ok=True)

CASE_NAMES = {
    "l2_random": "Rubric labels\nrandom split",
    "l2_group_holdout": "Rubric labels\ngroup holdout",
    "l2_plus_hf_balanced_random": "Rubric + HF\nrandom split",
    "l2_plus_hf_balanced_group_holdout": "Rubric + HF\ngroup holdout",
}
MODEL_NAMES = {
    "ridge": "TF-IDF Ridge",
    "fasttext": "fastText",
    "dense_tiny": "MiniLM dense",
    "dense_luxical": "Luxical dense",
}
MODEL_COLORS = {
    "TF-IDF Ridge": "#1b7837",
    "fastText": "#5aae61",
    "MiniLM dense": "#8073ac",
    "Luxical dense": "#4393c3",
}


def load_metrics() -> pd.DataFrame:
    df = pd.read_csv("runs/track_a_capped/metrics.csv")
    df["Model"] = df["model"].map(MODEL_NAMES)
    df["Evaluation split"] = df["case"].map(CASE_NAMES)
    return df


def save_metric_bars(df: pd.DataFrame, metric: str, ylabel: str, title: str, filename: str, lower_better: bool = False) -> None:
    plt.figure(figsize=(11, 5.5))
    ax = sns.barplot(
        data=df,
        x="Evaluation split",
        y=metric,
        hue="Model",
        palette=MODEL_COLORS,
    )
    if metric in {"auroc", "average_precision"}:
        ax.set_ylim(0.75, 1.01)
        ax.axhline(0.5, color="#999999", linewidth=0.8, linestyle="--")
    if lower_better:
        ax.set_ylim(0, max(df[metric]) * 1.2)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, padding=2)
    sns.despine()
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=260, bbox_inches="tight")
    plt.close()


def decision_heatmap(df: pd.DataFrame) -> None:
    agg = df.groupby("Model").agg(
        AUROC=("auroc", "mean"),
        **{"Rubric error": ("rubric_mse", "mean"), "Latency": ("latency_ms_per_row", "mean")},
    )
    fp = pd.read_csv("runs/false_positive_eval/summary.csv")
    fp["Model"] = fp["model"].map(MODEL_NAMES)
    agg["False positives"] = agg.index.map(fp.set_index("Model")["false_positive_rate"])
    # Normalize to 0-1 where 1 is best.
    score = pd.DataFrame(index=agg.index)
    score["AI separation"] = (agg["AUROC"] - agg["AUROC"].min()) / (agg["AUROC"].max() - agg["AUROC"].min())
    for src, dst in [("Rubric error", "Rubric match"), ("Latency", "Speed"), ("False positives", "Human safety")]:
        vals = agg[src]
        score[dst] = 1 - ((vals - vals.min()) / (vals.max() - vals.min()))
    score = score.fillna(1.0)
    score["Overall"] = score.mean(axis=1)
    score = score.sort_values("Overall", ascending=False)

    plt.figure(figsize=(8.5, 3.6))
    ax = sns.heatmap(score, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1, linewidths=0.5, cbar_kws={"label": "0 = worst, 1 = best"})
    ax.set_title("Scorer selection matrix", loc="left", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(OUT / "scorer_selection_matrix.png", dpi=260, bbox_inches="tight")
    plt.close()


def main() -> None:
    sns.set_theme(style="white", font_scale=0.9)
    df = load_metrics()
    save_metric_bars(df, "auroc", "AUROC (higher is better)", "AI vs human separation", "ai_separation_auroc.png")
    save_metric_bars(df, "rubric_mse", "Rubric MSE (lower is better)", "Approximation error vs Gemini rubric labels", "rubric_error.png", lower_better=True)
    save_metric_bars(df, "latency_ms_per_row", "Milliseconds per row (lower is better)", "Scoring latency", "latency.png", lower_better=True)
    decision_heatmap(df)
    print(f"Wrote public figures to {OUT}")


if __name__ == "__main__":
    main()
