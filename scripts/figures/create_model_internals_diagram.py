from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path("paper/figures/scorer")
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "ridge": "#147A3D",
    "fasttext": "#4DAA57",
    "minilm": "#2C7FB8",
    "luxical": "#7B61A8",
    "light": "#F7F7F7",
    "mid": "#E9E9E9",
    "line": "#222222",
}


def box(ax, x, y, w, h, text, fc="#F7F7F7", ec="#222222", lw=1.1, fs=8.5):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": COLORS["line"]},
    )


def panel(ax, x, y, title, color, feature, binary, rubric, footer):
    box(ax, x, y, 5.6, 4.85, "", fc="#FFFFFF", ec="#CCCCCC", lw=1.0)
    ax.text(x + 0.25, y + 4.48, title, fontsize=14.5, fontweight="bold", color=color, ha="left")

    box(ax, x + 1.55, y + 3.78, 2.5, 0.38, "input text", fs=8)
    arrow(ax, x + 2.8, y + 3.78, x + 2.8, y + 3.43)
    box(ax, x + 1.0, y + 3.0, 3.6, 0.52, feature, fc="#FFFFFF", ec=color, lw=1.6, fs=8)
    arrow(ax, x + 2.8, y + 3.0, x + 2.8, y + 2.66)
    box(ax, x + 1.25, y + 2.32, 3.1, 0.42, "shared representation", fc=COLORS["mid"], fs=8)

    arrow(ax, x + 2.8, y + 2.32, x + 1.65, y + 1.86)
    arrow(ax, x + 2.8, y + 2.32, x + 3.95, y + 1.86)
    box(ax, x + 0.45, y + 1.42, 2.25, 0.58, binary, fc="#FFFFFF", fs=7.6)
    box(ax, x + 2.9, y + 1.42, 2.25, 0.58, rubric, fc="#FFFFFF", fs=7.6)
    arrow(ax, x + 1.58, y + 1.42, x + 1.58, y + 1.04)
    arrow(ax, x + 4.03, y + 1.42, x + 4.03, y + 1.04)
    box(ax, x + 0.45, y + 0.68, 2.25, 0.42, "AI probability\nP(label = AI)", fs=7.6)
    box(ax, x + 2.9, y + 0.68, 2.25, 0.42, "8 rubric scores\nŷ₁ … ŷ₈", fs=7.6)

    ax.text(x + 0.25, y + 0.25, footer, fontsize=7.8, ha="left", va="top", color="#333333")


def main() -> None:
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 10.5)
    ax.axis("off")

    ax.text(0.35, 10.05, "Scorer model internals", fontsize=24, fontweight="bold", ha="left")
    ax.text(
        0.35,
        9.68,
        "Each candidate maps one response to two outputs: an AI-pattern probability and eight normalized humanness-rubric scores.",
        fontsize=11,
        ha="left",
    )

    panel(
        ax,
        0.35,
        5.05,
        "TF-IDF Logistic+Ridge",
        COLORS["ridge"],
        "TF-IDF sparse vector\nword n-grams 1–4\nmax 10k features",
        "Logistic head\nlinear classifier\nσ(w·x + b)",
        "Ridge heads\n8 linear regressors\nargmin ||y-Xw||² + α||w||²",
        "Selected: small, inspectable, ~1 ms/row, best rubric MSE.",
    )
    panel(
        ax,
        6.45,
        5.05,
        "fastText + Ridge",
        COLORS["fasttext"],
        "fastText vector\nword + character n-grams\nsentence vector",
        "fastText head\nsupervised softmax\nP(AI | vector)",
        "Ridge heads\n8 linear regressors\non sentence vectors",
        "Strong binary baseline; large artifact and weaker rubric fit.",
    )
    panel(
        ax,
        0.35,
        0.05,
        "MiniLM dense head",
        COLORS["minilm"],
        "all-MiniLM-L6-v2\ndense sentence embedding\n384-d vector",
        "MLP binary head\nLinear → ReLU →\nLinear → Sigmoid",
        "MLP rubric head\nshared hidden layer →\n8 sigmoid outputs",
        "Semantic baseline; slower and weaker on current benchmark.",
    )
    panel(
        ax,
        6.45,
        0.05,
        "Luxical dense head",
        COLORS["luxical"],
        "DatologyAI/luxical-one\nquality-oriented embedding\ndense vector",
        "MLP binary head\nLinear → ReLU →\nLinear → Sigmoid",
        "MLP rubric head\nshared hidden layer →\n8 sigmoid outputs",
        "Fast dense candidate; higher false positives on human challenge set.",
    )

    plt.tight_layout()
    out = OUT / "model_candidate_internals.png"
    plt.savefig(out, dpi=260, bbox_inches="tight")
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
