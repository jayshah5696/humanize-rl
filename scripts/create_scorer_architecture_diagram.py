from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

OUT = Path("paper/figures/scorer")
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#0B78B6"
GREEN = "#1B7837"
PINK = "#D42A74"
GREY = "#D9D9D9"
DARK = "#555555"
LIGHT = "#F7F7F7"


def box(ax, x, y, w, h, text, fc=LIGHT, ec="black", lw=1.2, fs=9, color="black", ls="-"):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        linestyle=ls,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color)
    return p


def arrow(ax, x1, y1, x2, y2, color="black", lw=1.2, ls="-", head=True):
    ax.annotate(
        "",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>" if head else "-", lw=lw, color=color, linestyle=ls, shrinkA=0, shrinkB=0),
    )


def callout(ax, text, x, y, tx, ty, color="black"):
    ax.text(tx, ty, text, fontsize=9, fontweight="bold", color=color, ha="left", va="center")
    arrow(ax, tx, ty - 0.03, x, y, color=color, lw=1.5, ls=":")


def main() -> None:
    fig, ax = plt.subplots(figsize=(15, 8.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(1.0, 9.4, "Rubric-labeled data", fontsize=24, fontweight="bold", color=BLUE)
    ax.text(9.0, 9.4, "Selected local scorer", fontsize=24, fontweight="bold", color=GREEN)

    # Left big pipeline container
    box(ax, 0.9, 1.2, 5.2, 7.4, "", fc=GREY, ec="black", lw=1.1)
    box(ax, 1.7, 1.55, 3.6, 5.8, "", fc="#BDBDBD", ec="black", lw=1.0)

    box(ax, 2.35, 0.55, 2.3, 0.35, "Raw text pools", fs=8)
    arrow(ax, 3.5, 0.9, 3.5, 1.55)
    box(ax, 2.15, 1.65, 2.7, 0.45, "Human / AIified /\nhumanized rows", fs=8)
    arrow(ax, 3.5, 2.1, 3.5, 2.65)
    box(ax, 2.15, 2.65, 2.7, 0.45, "Layer 1 heuristics", fs=8)
    arrow(ax, 3.5, 3.1, 3.5, 3.65)
    box(ax, 2.15, 3.65, 2.7, 0.5, "Gemini rubric judge", fs=8, fc="#FFFFFF")
    arrow(ax, 3.5, 4.15, 3.5, 4.7)
    box(ax, 2.0, 4.7, 3.0, 0.6, "8-dimensional\nLayer-2 labels", fs=8, fc="#FFFFFF")
    arrow(ax, 3.5, 5.3, 3.5, 5.9)
    box(ax, 1.95, 5.9, 3.1, 0.7, "10,000-row scorer\ncalibration dataset", fs=9, fc="#FFFFFF")
    arrow(ax, 3.5, 6.6, 3.5, 7.15)
    box(ax, 2.25, 7.15, 2.5, 0.45, "dedupe + verification", fs=8)

    callout(ax, "4,946 AI\n3,308 human\n1,746 humanized", 5.05, 6.25, 5.65, 7.15, BLUE)
    callout(ax, "8 rubric dimensions\nnormalized to [0,1]", 4.9, 4.95, 5.65, 5.55, BLUE)
    callout(ax, "10k unique IDs\n0 duplicates", 4.85, 7.35, 5.65, 8.05, BLUE)

    # Right model container
    box(ax, 8.1, 1.2, 5.4, 7.4, "", fc=GREY, ec="black", lw=1.1)
    box(ax, 8.8, 2.05, 4.0, 5.3, "", fc="#B8E0C2", ec="black", lw=1.0)

    box(ax, 9.55, 0.55, 2.4, 0.35, "Input response", fs=8)
    arrow(ax, 10.75, 0.9, 10.75, 1.35)
    box(ax, 9.55, 1.35, 2.4, 0.45, "TF-IDF n-grams", fs=8)
    arrow(ax, 10.75, 1.8, 10.75, 2.3)
    box(ax, 9.35, 2.3, 2.8, 0.55, "Shared sparse\nfeature vector", fs=8, fc="#FFFFFF")

    # split branches
    arrow(ax, 10.75, 2.85, 9.65, 3.45)
    arrow(ax, 10.75, 2.85, 11.85, 3.45)
    box(ax, 8.85, 3.45, 1.6, 0.65, "Logistic\nbinary head", fs=8, fc="#FFFFFF")
    box(ax, 11.05, 3.45, 1.6, 0.65, "8 Ridge\nrubric heads", fs=8, fc="#FFFFFF")

    arrow(ax, 9.65, 4.1, 9.65, 4.9)
    arrow(ax, 11.85, 4.1, 11.85, 4.9)
    box(ax, 8.65, 4.9, 2.0, 0.6, "AI probability", fs=8, fc="#FFFFFF")
    box(ax, 10.85, 4.9, 2.0, 0.6, "Rubric vector", fs=8, fc="#FFFFFF")

    # combine
    arrow(ax, 9.65, 5.5, 10.75, 6.05)
    arrow(ax, 11.85, 5.5, 10.75, 6.05)
    circle = Circle((10.75, 6.15), 0.16, facecolor="white", edgecolor="black", linewidth=1.1)
    ax.add_patch(circle)
    ax.text(10.75, 6.15, "+", ha="center", va="center", fontsize=12)
    arrow(ax, 10.75, 6.31, 10.75, 6.85)
    box(ax, 9.55, 6.85, 2.4, 0.45, "Selected scorer\nridge.pkl", fs=8, fc="#FFFFFF")

    callout(ax, "AUROC ≈ 0.997-0.999", 9.65, 5.2, 13.2, 5.85, GREEN)
    callout(ax, "Rubric MSE ≈ 0.036-0.046", 11.85, 5.2, 13.2, 5.15, GREEN)
    callout(ax, "~1 ms / row", 10.75, 7.1, 13.2, 7.0, GREEN)
    callout(ax, "0.0% false positives\non 400 human hard negatives", 10.75, 6.95, 13.2, 6.35, GREEN)

    # Dense candidates mini panel
    box(ax, 6.6, 2.4, 1.1, 0.45, "fastText", fs=8, fc="#FFFFFF")
    box(ax, 6.6, 3.05, 1.1, 0.45, "MiniLM", fs=8, fc="#FFFFFF")
    box(ax, 6.6, 3.7, 1.1, 0.45, "Luxical", fs=8, fc="#FFFFFF")
    ax.text(6.45, 4.45, "Compared but\nnot selected", fontsize=9, fontweight="bold", ha="left")
    arrow(ax, 7.7, 3.05, 8.8, 3.9, ls=":")
    arrow(ax, 6.1, 5.95, 8.1, 5.95, lw=1.8)

    # Bottom notes
    ax.text(0.95, 0.15, "Figure: The local scorer distills Gemini rubric labels into a two-head TF-IDF Logistic+Ridge model. It is approved for filtering and diagnostics, not RL reward yet.", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUT / "scorer_architecture_llama_style.png", dpi=260, bbox_inches="tight")
    plt.close()
    print(OUT / "scorer_architecture_llama_style.png")


if __name__ == "__main__":
    main()
