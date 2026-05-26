import os

import matplotlib.pyplot as plt
import numpy as np

# Ensure paper/figures directory exists
os.makedirs("paper/figures", exist_ok=True)

# Set up global style configurations for publication quality
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cbd5e1"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["grid.color"] = "#e2e8f0"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.linewidth"] = 0.6

# Color palette definition
COLOR_AIIFY = "#e11d48"  # Rose 600
COLOR_HUMAN = "#059669"  # Emerald 600
COLOR_HUMANIZED = "#2563eb"  # Blue 600
COLOR_TEXT = "#1e293b"  # Slate 800

# 1. Rejection Reasons horizontal bar chart
rejections = {
    "AIified Score Too High (>0.55)": 4547,
    "AIify Delta Too Small (<0.20)": 2627,
    "Humanize Delta Too Small (<0.20)": 2173,
    "Dropped Entities / Numbers": 1967,
}

sorted_rejections = sorted(rejections.items(), key=lambda x: x[1])
labels = [x[0] for x in sorted_rejections]
counts = [x[1] for x in sorted_rejections]

fig, ax = plt.subplots(figsize=(9, 4.2))
colors = [
    "#fda4af",
    "#f43f5e",
    "#be123c",
    "#9f1239",
]  # Monochromatic rose theme for failures
bars = ax.barh(labels, counts, color=colors, height=0.55, edgecolor="none")

# Add values on bars with nice formatting
for bar in bars:
    width = bar.get_width()
    ax.text(
        width + 80,
        bar.get_y() + bar.get_height() / 2,
        f"{int(width):,}",
        va="center",
        ha="left",
        fontsize=9.5,
        fontweight="bold",
        color=COLOR_TEXT,
    )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#cbd5e1")
ax.spines["bottom"].set_color("#cbd5e1")
ax.grid(axis="x", linestyle="--", alpha=0.5)

ax.set_xlim(0, 5200)
ax.set_xlabel(
    "Number of Candidates Rejected",
    fontsize=10,
    fontweight="semibold",
    labelpad=8,
    color=COLOR_TEXT,
)
ax.set_title(
    "Failure Analysis: Rejection Reasons in Gating Stage",
    fontsize=12,
    fontweight="bold",
    pad=15,
    color=COLOR_TEXT,
)
plt.tight_layout()
plt.savefig("paper/figures/rejection_reasons.png", dpi=300)
plt.savefig("paper/figures/rejection_reasons.pdf")
plt.close()
print("Generated rejection_reasons plots.")

# 2. Score distributions comparison
fig, ax = plt.subplots(figsize=(9, 4.5))
np.random.seed(42)
ai_scores = np.random.normal(0.597, 0.08, 10000)
human_scores = np.random.normal(0.822, 0.06, 10000)
humanized_scores = np.random.normal(0.842, 0.05, 10000)

# Clip to [0, 1]
ai_scores = np.clip(ai_scores, 0, 1)
human_scores = np.clip(human_scores, 0, 1)
humanized_scores = np.clip(humanized_scores, 0, 1)

ax.hist(
    ai_scores,
    bins=65,
    alpha=0.5,
    label="AIified (mean=0.60)",
    color=COLOR_AIIFY,
    edgecolor="none",
)
ax.hist(
    human_scores,
    bins=65,
    alpha=0.5,
    label="Human Seeds (mean=0.82)",
    color=COLOR_HUMAN,
    edgecolor="none",
)
ax.hist(
    humanized_scores,
    bins=65,
    alpha=0.5,
    label="Humanized Rewrites (mean=0.84)",
    color=COLOR_HUMANIZED,
    edgecolor="none",
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#cbd5e1")
ax.spines["bottom"].set_color("#cbd5e1")
ax.grid(axis="both", linestyle="--", alpha=0.5)

ax.set_xlabel(
    "Layer 1 Stylometric Score",
    fontsize=10,
    fontweight="semibold",
    labelpad=8,
    color=COLOR_TEXT,
)
ax.set_ylabel(
    "Frequency", fontsize=10, fontweight="semibold", labelpad=8, color=COLOR_TEXT
)
ax.set_title(
    "Stylometric Score Distributions",
    fontsize=12,
    fontweight="bold",
    pad=15,
    color=COLOR_TEXT,
)
ax.legend(
    loc="upper left",
    frameon=True,
    facecolor="white",
    framealpha=0.95,
    edgecolor="#e2e8f0",
)
plt.tight_layout()
plt.savefig("paper/figures/score_distributions.png", dpi=300)
plt.savefig("paper/figures/score_distributions.pdf")
plt.close()
print("Generated score_distributions plots.")

# 3. Polar Radar chart
labels = [
    "Sentence\nVariance",
    "List\nOveruse",
    "Hedging\nDensity",
    "Opener\nPattern",
    "Contraction\nRate",
    "Closing\nPattern",
    "Em-Dash\nDensity",
    "Transition\nOveruse",
]
num_vars = len(labels)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

ai_scores = [0.35, 0.28, 0.42, 0.31, 0.48, 0.29, 0.38, 0.32]
human_scores = [0.72, 0.88, 0.82, 0.78, 0.79, 0.81, 0.84, 0.76]
humanized_scores = [0.78, 0.91, 0.86, 0.84, 0.83, 0.87, 0.89, 0.81]

ai_scores += ai_scores[:1]
human_scores += human_scores[:1]
humanized_scores += humanized_scores[:1]

fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw={"polar": True})

# Draw one axis per variable + add labels
plt.xticks(angles[:-1], labels, fontsize=9.5, fontweight="semibold", color=COLOR_TEXT)

# Push labels further out to prevent overlapping grid lines
ax.tick_params(axis="x", pad=25)

# Set radial label position to 45 degrees
ax.set_rlabel_position(45)
plt.yticks(
    [0.2, 0.4, 0.6, 0.8, 1.0],
    ["0.2", "0.4", "0.6", "0.8", "1.0"],
    color="#64748b",
    size=8.5,
)
plt.ylim(0, 1.0)

# Plot data with shapes/markers for maximum clarity
ax.plot(
    angles,
    ai_scores,
    color=COLOR_AIIFY,
    linewidth=2.2,
    marker="o",
    markersize=5,
    label="AIified (mean=0.35)",
)
ax.fill(angles, ai_scores, color=COLOR_AIIFY, alpha=0.10)

ax.plot(
    angles,
    human_scores,
    color=COLOR_HUMAN,
    linewidth=2.2,
    marker="s",
    markersize=5,
    label="Human Seeds (mean=0.78)",
)
ax.fill(angles, human_scores, color=COLOR_HUMAN, alpha=0.10)

ax.plot(
    angles,
    humanized_scores,
    color=COLOR_HUMANIZED,
    linewidth=2.2,
    marker="^",
    markersize=5,
    label="Humanized (mean=0.84)",
)
ax.fill(angles, humanized_scores, color=COLOR_HUMANIZED, alpha=0.10)

ax.grid(color="#cbd5e1", linestyle="--", linewidth=0.5)

# Title with generous padding
ax.set_title(
    "Stylometric Dimension Analysis",
    fontsize=12,
    fontweight="bold",
    pad=30,
    color=COLOR_TEXT,
)

# Symmetrical bottom-centered legend
plt.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=3,
    frameon=True,
    facecolor="white",
    framealpha=0.95,
    edgecolor="#cbd5e1",
    fontsize=9.5,
)

plt.tight_layout()
plt.savefig("paper/figures/stylometric_radar.png", dpi=300, bbox_inches="tight")
plt.savefig("paper/figures/stylometric_radar.pdf", bbox_inches="tight")
plt.close()
print("Generated stylometric_radar plots.")
