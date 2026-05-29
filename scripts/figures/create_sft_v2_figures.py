"""Generate figures for the Track B v2 SFT dataset preparation report.

Tufte principles applied:
- Grayscale + one accent colour.
- No outer borders or heavy grids.
- Labels on data, not in legends where possible.
- Small multiples for multi-panel comparisons.
- Bars over pies; horizontal bars for long category names.
"""
import collections
import json

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ACCENT = "#2171B5"
LIGHT   = "#BDD7E7"
RED     = "#CB181D"
GRAY    = "#636363"
LGRAY   = "#CCCCCC"
FIG_DIR = Path("paper/figures/sft_v2")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── load data ──────────────────────────────────────────────────────────────

def infer_domain(r):
    d = r.get("domain") or r.get("origin_domain") or ""
    if d: return d
    inst = r.get("instruction", "").lower()
    if "slack" in inst or "channel" in inst or " dm " in inst: return "chat"
    if "email" in inst or "dear " in inst: return "email"
    if "shorten" in inst or "compress" in inst or "tighten" in inst: return "compress"
    if "grammar" in inst or "fix the" in inst: return "grammar"
    if "story" in inst or "paragraph" in inst or "vivid" in inst: return "creative"
    return "general"

rows = [json.loads(l) for l in Path("data/processed/v04_sft_final.jsonl").read_text().splitlines() if l.strip()]
N = len(rows)
domain_counts = collections.Counter(infer_domain(r) for r in rows)
source_counts = collections.Counter(r.get("source") or "curated" for r in rows)
nat_vals = [r["quality_judge"]["naturalness"] for r in rows if r.get("quality_judge", {}).get("naturalness")]
resp_lens = [len(r["response"]) for r in rows]
inst_lens = [len(r["instruction"]) for r in rows]
placeholder_flag = ["[" in r["response"] for r in rows]

BAD_PHRASES = [
    ("certainly", "Certainly"),
    ("of course,", "Of course"),
    ("it is worth noting", "It is worth noting"),
    ("furthermore", "Furthermore"),
    ("moreover", "Moreover"),
    ("please don't hesitate", "Please don't hesitate"),
    ("i hope this email finds", "I hope this email finds"),
]
bad_counts = {label: sum(phrase in r["response"].lower() for r in rows) for phrase, label in BAD_PHRASES}

ITERATION_STEPS = [
    ("v1 seed run", 1269, "curated", "#CCCCCC"),
    ("v1 diagnosed fail", 0, "fail", RED),
    ("base150 generated", 145, "stream_a", LIGHT),
    ("synthetic500 expanded", 133, "stream_a", LIGHT),
    ("safe_expand_1000", 668, "stream_a", LIGHT),
    ("safe_expand_3000", 1806, "stream_a", ACCENT),
    ("stream_b (real HF)", 1619, "stream_b", "#6BAED6"),
    ("chat_base (48 seeds)", 48, "chat", "#FD8D3C"),
    ("chat_expanded (800→440→423)", 423, "chat", "#FD8D3C"),
]

# ── Figure 1: dataset pipeline funnel ────────────────────────────────────

def fig_pipeline():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.38)

    # Left: cumulative build-up timeline (horizontal bar)
    ax = axes[0]
    cumulative_kept = []
    running = 0
    for name, kept, kind, col in ITERATION_STEPS:
        if kind != "fail":
            running += kept
        cumulative_kept.append((name, running, kept, kind, col))

    names = [x[0] for x in cumulative_kept]
    cum   = [x[1] for x in cumulative_kept]
    delta = [x[2] for x in cumulative_kept]
    cols  = [x[4] for x in cumulative_kept]

    y = np.arange(len(names))
    ax.barh(y, delta, color=cols, edgecolor="none", height=0.65)
    for i, (d, c) in enumerate(zip(delta, cum)):
        if d > 0:
            ax.text(d + 15, i, f"+{d:,}", va="center", fontsize=7.5, color=GRAY)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Rows added per iteration", fontsize=9)
    ax.set_title("Dataset build-up by iteration", fontsize=10, loc="left")
    ax.axvline(0, color=LGRAY, lw=0.6)
    for sp in ["top", "right", "left"]: ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(LGRAY)
    ax.tick_params(axis="x", length=3, color=LGRAY)
    ax.tick_params(axis="y", length=0)

    # Right: final composition donut-style replaced by Tufte bar
    ax2 = axes[1]
    domain_order = sorted(domain_counts, key=lambda k: -domain_counts[k])
    d_vals = [domain_counts[k] for k in domain_order]
    d_pct  = [100 * v / N for v in d_vals]
    cols2  = [ACCENT, "#6BAED6", "#BDD7E7", "#FD8D3C", LGRAY, LGRAY][:len(domain_order)]
    yy = np.arange(len(domain_order))
    ax2.barh(yy, d_vals, color=cols2, edgecolor="none", height=0.65)
    for i, (v, p) in enumerate(zip(d_vals, d_pct)):
        ax2.text(v + 15, i, f"{v:,} ({p:.0f}%)", va="center", fontsize=8, color=GRAY)
    ax2.set_yticks(yy)
    ax2.set_yticklabels([k.replace("_", " ") for k in domain_order], fontsize=8.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("Rows", fontsize=9)
    ax2.set_title(f"Final domain distribution  (n={N:,})", fontsize=10, loc="left")
    for sp in ["top", "right", "left"]: ax2.spines[sp].set_visible(False)
    ax2.spines["bottom"].set_color(LGRAY)
    ax2.tick_params(axis="x", length=3, color=LGRAY)
    ax2.tick_params(axis="y", length=0)

    fig.savefig(FIG_DIR / "sft_v2_pipeline_funnel.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("fig1: sft_v2_pipeline_funnel.png")


# ── Figure 2: quality small multiples ────────────────────────────────────

def fig_quality():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.subplots_adjust(wspace=0.40)

    # Panel A: naturalness distribution
    ax = axes[0]
    nat_dist = collections.Counter(nat_vals)
    xs = sorted(nat_dist)
    ys = [nat_dist[x] for x in xs]
    ax.bar(xs, ys, color=ACCENT, edgecolor="none", width=0.6)
    for x, y in zip(xs, ys):
        ax.text(x, y + 12, f"{y:,}", ha="center", fontsize=8, color=GRAY)
    ax.set_xticks(xs)
    ax.set_xlabel("Naturalness score (1–5)", fontsize=9)
    ax.set_title(f"Judge naturalness\n(n={len(nat_vals):,}, avg={sum(nat_vals)/len(nat_vals):.2f})", fontsize=9.5, loc="left")
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(LGRAY)
    ax.spines["bottom"].set_color(LGRAY)
    ax.tick_params(length=3, color=LGRAY)

    # Panel B: response length histogram
    ax2 = axes[1]
    bins = [0, 50, 100, 200, 400, 600, 800, 1200]
    counts, edges = np.histogram(resp_lens, bins=bins)
    width = np.diff(edges)
    ax2.bar(edges[:-1], counts, width=width, align="edge", color=ACCENT, edgecolor="white", linewidth=0.4)
    ax2.set_xlabel("Response length (chars)", fontsize=9)
    ax2.set_title(f"Response length distribution\nmedian={int(np.median(resp_lens))}, avg={int(np.mean(resp_lens))}", fontsize=9.5, loc="left")
    ax2.set_xticks(bins)
    ax2.set_xticklabels([str(b) for b in bins], fontsize=7.5, rotation=30, ha="right")
    for sp in ["top", "right"]: ax2.spines[sp].set_visible(False)
    ax2.spines["left"].set_color(LGRAY)
    ax2.spines["bottom"].set_color(LGRAY)
    ax2.tick_params(length=3, color=LGRAY)

    # Panel C: bad-phrase audit
    ax3 = axes[2]
    bp_labels = list(bad_counts.keys())
    bp_vals   = list(bad_counts.values())
    bp_pct    = [100 * v / N for v in bp_vals]
    yy = np.arange(len(bp_labels))
    colors3 = [RED if v > 0 else LGRAY for v in bp_vals]
    ax3.barh(yy, bp_pct, color=colors3, edgecolor="none", height=0.6)
    for i, (v, p) in enumerate(zip(bp_vals, bp_pct)):
        label = f"{v} ({p:.2f}%)" if v else "0"
        ax3.text(max(p, 0) + 0.005, i, label, va="center", fontsize=7.5, color=GRAY)
    ax3.set_yticks(yy)
    ax3.set_yticklabels(bp_labels, fontsize=7.5)
    ax3.invert_yaxis()
    ax3.set_xlabel("% of rows", fontsize=9)
    ax3.set_title("Bad-phrase audit\n(response field)", fontsize=9.5, loc="left")
    for sp in ["top", "right", "left"]: ax3.spines[sp].set_visible(False)
    ax3.spines["bottom"].set_color(LGRAY)
    ax3.tick_params(axis="x", length=3, color=LGRAY)
    ax3.tick_params(axis="y", length=0)

    fig.savefig(FIG_DIR / "sft_v2_quality_panel.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("fig2: sft_v2_quality_panel.png")


# ── Figure 3: source provenance + mode composition ───────────────────────

def fig_sources():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.subplots_adjust(wspace=0.40)

    # Source stacked / side-by-side
    ax = axes[0]
    src_order = sorted(source_counts, key=lambda k: -source_counts[k])
    src_vals  = [source_counts[k] for k in src_order]
    src_pct   = [100 * v / N for v in src_vals]
    src_cols  = [ACCENT, "#6BAED6", LIGHT, "#FD8D3C", LGRAY][:len(src_order)]
    yy = np.arange(len(src_order))
    ax.barh(yy, src_vals, color=src_cols, edgecolor="none", height=0.65)
    for i, (v, p) in enumerate(zip(src_vals, src_pct)):
        ax.text(v + 15, i, f"{v:,}  {p:.0f}%", va="center", fontsize=8, color=GRAY)
    ax.set_yticks(yy)
    ax.set_yticklabels([k.replace("_", "\n") for k in src_order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Rows", fontsize=9)
    ax.set_title("Rows by data stream", fontsize=10, loc="left")
    for sp in ["top", "right", "left"]: ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(LGRAY)
    ax.tick_params(axis="x", length=3, color=LGRAY)
    ax.tick_params(axis="y", length=0)

    # Placeholder vs concrete split
    ax2 = axes[1]
    n_ph = sum(placeholder_flag)
    n_co = N - n_ph
    bars = ax2.barh([1, 0], [n_ph, n_co], color=[ACCENT, LIGHT], edgecolor="none", height=0.5)
    ax2.text(n_ph + 15, 1, f"{n_ph:,} ({100*n_ph/N:.0f}%)", va="center", fontsize=9, color=GRAY)
    ax2.text(n_co + 15, 0, f"{n_co:,} ({100*n_co/N:.0f}%)", va="center", fontsize=9, color=GRAY)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Concrete response\n(no placeholders)", "Uses placeholder\n[Name], [Date], …"], fontsize=9)
    ax2.set_xlabel("Rows", fontsize=9)
    ax2.set_title("Placeholder use in responses", fontsize=10, loc="left")
    ax2.set_xlim(0, max(n_ph, n_co) * 1.22)
    for sp in ["top", "right", "left"]: ax2.spines[sp].set_visible(False)
    ax2.spines["bottom"].set_color(LGRAY)
    ax2.tick_params(axis="x", length=3, color=LGRAY)
    ax2.tick_params(axis="y", length=0)

    fig.savefig(FIG_DIR / "sft_v2_sources.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("fig3: sft_v2_sources.png")


# ── Figure 4: instruction framing audit ──────────────────────────────────

def fig_instruction_framing():
    AI_FRAME = ["chatgpt", "ai-generated", "ai generated", "ai-written", "humanize this", "human wrote it"]
    ROBOTIC   = ["robotic", "too formal", "too stiff", "corporate", "sounds stiff"]
    PLAIN     = []

    explicit_ai = sum(any(p in r["instruction"].lower() for p in AI_FRAME) for r in rows)
    robotic     = sum(any(p in r["instruction"].lower() for p in ROBOTIC) for r in rows)
    plain       = N - explicit_ai - robotic

    labels = ["Plain task framing", "Robotic/stiff framing", "Explicit AI framing"]
    vals   = [plain, robotic, explicit_ai]
    cols   = [ACCENT, LIGHT, RED]

    fig, ax = plt.subplots(figsize=(8, 3))
    yy = np.arange(3)
    ax.barh(yy, vals, color=cols, edgecolor="none", height=0.55)
    for i, v in enumerate(vals):
        ax.text(v + 15, i, f"{v:,} ({100*v/N:.1f}%)", va="center", fontsize=9, color=GRAY)
    ax.set_yticks(yy)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Rows", fontsize=9)
    ax.set_title("Instruction framing audit — final dataset", fontsize=10, loc="left")
    ax.set_xlim(0, max(vals) * 1.22)
    for sp in ["top", "right", "left"]: ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(LGRAY)
    ax.tick_params(axis="x", length=3, color=LGRAY)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sft_v2_instruction_framing.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("fig4: sft_v2_instruction_framing.png")


if __name__ == "__main__":
    fig_pipeline()
    fig_quality()
    fig_sources()
    fig_instruction_framing()
    print(f"\nAll figures saved to {FIG_DIR}/")
