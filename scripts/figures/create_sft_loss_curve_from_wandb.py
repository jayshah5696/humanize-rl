from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUT = Path("paper/figures/sft")
OUT.mkdir(parents=True, exist_ok=True)
PANE = "gemma4-mlx-full:1.1"
SOCKET = Path("/var/folders/nv/30f4q12s3bz7r149xx20cvdr0000gn/T/claude-tmux-sockets/claude.sock")
LOSS_RE = re.compile(r"(\d+)/(\d+).*?loss=([0-9.]+), avg_loss=([0-9.]+)")


def capture_tmux() -> str:
    if not SOCKET.exists():
        return ""
    result = subprocess.run(
        ["tmux", "-S", str(SOCKET), "capture-pane", "-p", "-J", "-t", PANE, "-S", "-5000"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else ""


def load_loss_points() -> pd.DataFrame:
    text = capture_tmux()
    rows = []
    for match in LOSS_RE.finditer(text):
        rows.append(
            {
                "step": int(match.group(1)),
                "total_steps": int(match.group(2)),
                "loss": float(match.group(3)),
                "avg_loss": float(match.group(4)),
            }
        )
    if rows:
        deduped = {row["step"]: row for row in rows}
        return pd.DataFrame([deduped[key] for key in sorted(deduped)])

    metrics = json.loads(Path("outputs_gemma4_humanize_mlx_full_r8/metrics.json").read_text())
    return pd.DataFrame(
        [
            {
                "step": metrics["steps"],
                "total_steps": metrics["steps"],
                "loss": metrics["train_loss"],
                "avg_loss": metrics["train_loss"],
            }
        ]
    )


def main() -> None:
    sns.set_theme(style="white", font="DejaVu Sans", font_scale=0.95)
    df = load_loss_points()
    df.to_csv(OUT / "sft_loss_curve_points.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    if len(df) > 1:
        ax.plot(df["step"], df["avg_loss"], color="#1b7837", linewidth=2.0, label="running mean loss")
        ax.plot(df["step"], df["loss"], color="#7a7a7a", linewidth=1.0, alpha=0.65, label="reported step loss")
    else:
        ax.scatter(df["step"], df["avg_loss"], color="#1b7837", s=60)
        ax.text(df.iloc[0]["step"], df.iloc[0]["avg_loss"], f" final avg loss {df.iloc[0]['avg_loss']:.3f}", va="center")
    ax.set_title("Local MLX-Tune training loss", loc="left", fontweight="bold")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(frameon=False, loc="upper right")
    sns.despine()
    plt.tight_layout()
    plt.savefig(OUT / "sft_loss_curve.png", dpi=260, bbox_inches="tight")
    plt.close()
    print(f"Wrote {OUT / 'sft_loss_curve.png'} with {len(df)} points")


if __name__ == "__main__":
    main()
