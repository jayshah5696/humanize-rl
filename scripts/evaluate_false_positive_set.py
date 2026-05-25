from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from humanize_rl.data.scorer_dataset import load_scorer_data
from humanize_rl.scoring.distilled.baselines import FastTextScorer, RidgeScorer
from humanize_rl.scoring.distilled.dense import DenseScorer

OUT = Path("runs/false_positive_eval")
L2_PATH = "data/scorer/l2_labeled_scorer_v02_10k.jsonl"
MODELS = {
    "ridge": lambda: RidgeScorer(max_features=10000),
    "fasttext": lambda: FastTextScorer(model_dir=str(OUT / "tmp_models")),
    "dense_luxical": lambda: DenseScorer(model_name="DatologyAI/luxical-one", epochs=4, batch_size=64),
}


def load_fp(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {
        "texts": [row["text"] for row in rows],
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp", type=Path, default=Path("data/scorer/false_positive_human_v01.jsonl"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    train, _ = load_scorer_data(local_path=L2_PATH, hf_dataset_name=None, split_strategy="random")
    fp = load_fp(args.fp)
    rows = []
    scored_examples = []
    for name, factory in MODELS.items():
        print(f"Training {name}")
        model = factory()
        model.fit(train["texts"], train["labels"], train["rubrics"])
        preds = model.predict_binary(fp["texts"])
        false_positive_rate = float((preds >= args.threshold).mean())
        rows.append(
            {
                "model": name,
                "n": len(preds),
                "threshold": args.threshold,
                "false_positive_rate": false_positive_rate,
                "mean_ai_probability": float(preds.mean()),
                "p90_ai_probability": float(sorted(preds)[int(0.9 * len(preds))]),
                "p95_ai_probability": float(sorted(preds)[int(0.95 * len(preds))]),
            }
        )
        for row, pred in zip(fp["rows"], preds, strict=True):
            scored_examples.append({**row, "model": name, "ai_probability": float(pred)})

    cols = list(rows[0].keys())
    with (OUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "summary.md").open("w") as handle:
        handle.write("| " + " | ".join(cols) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in cols) + " |\n")
    with (OUT / "scored_examples.jsonl").open("w") as handle:
        for row in scored_examples:
            handle.write(json.dumps(row) + "\n")

    plt.figure(figsize=(9, 5))
    sns.barplot(data=pd.DataFrame(rows), x="model", y="false_positive_rate")
    plt.title("False positive rate on human-authored challenge set")
    plt.tight_layout()
    plt.savefig(OUT / "false_positive_rate.png", dpi=220)
    plt.close()

    report = "# False Positive Human Challenge Eval\n\n" + (OUT / "summary.md").read_text()
    report += "\n![False positive rate](false_positive_rate.png)\n"
    (OUT / "REPORT.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
