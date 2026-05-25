from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from humanize_rl.data.scorer_dataset import load_scorer_data, scorer_data_summary
from humanize_rl.scoring.distilled.baselines import RidgeScorer


def evaluate(train_data: dict, val_data: dict) -> dict[str, float | int | str]:
    model = RidgeScorer()
    start = perf_counter()
    model.fit(train_data["texts"], train_data["labels"], train_data["rubrics"])
    train_seconds = perf_counter() - start

    start = perf_counter()
    pred = model.predict_binary(val_data["texts"])
    latency_ms = ((perf_counter() - start) / max(len(val_data["texts"]), 1)) * 1000

    labels = val_data["labels"]
    if len(set(labels.tolist())) < 2:
        auroc = float("nan")
        ap = float("nan")
    else:
        auroc = float(roc_auc_score(labels, pred))
        ap = float(average_precision_score(labels, pred))

    train_summary = scorer_data_summary(train_data)
    val_summary = scorer_data_summary(val_data)
    return {
        "train_samples": train_summary["samples"],
        "val_samples": val_summary["samples"],
        "train_ai": train_summary["ai_samples"],
        "train_humanish": train_summary["humanish_samples"],
        "val_ai": val_summary["ai_samples"],
        "val_humanish": val_summary["humanish_samples"],
        "train_rubric": train_summary["rubric_samples"],
        "val_rubric": val_summary["rubric_samples"],
        "auroc": auroc,
        "average_precision": ap,
        "latency_ms_per_row": latency_ms,
        "train_seconds": train_seconds,
    }


def run_case(name: str, **kwargs) -> dict[str, float | int | str]:
    print(f"Running {name}...")
    train_data, val_data = load_scorer_data(**kwargs)
    row = evaluate(train_data, val_data)
    row["case"] = name
    row["split_strategy"] = kwargs.get("split_strategy", "random")
    row["hf_dataset"] = kwargs.get("hf_dataset_name") or "none"
    row["balance_format"] = bool(kwargs.get("balance_format", False))
    return row


def fmt(value: object) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    cases = [
        ("local_random", {"hf_dataset_name": None, "split_strategy": "random"}),
        ("local_group_holdout", {"hf_dataset_name": None, "split_strategy": "group_holdout"}),
        ("local_domain_holdout", {"hf_dataset_name": None, "split_strategy": "domain_holdout"}),
        ("hf_random_unbalanced", {"hf_dataset_name": "gsingh1-py/train", "split_strategy": "random"}),
        ("hf_random_format_balanced", {"hf_dataset_name": "gsingh1-py/train", "split_strategy": "random", "balance_format": True}),
        ("hf_group_holdout_balanced", {"hf_dataset_name": "gsingh1-py/train", "split_strategy": "group_holdout", "balance_format": True}),
        ("hf_domain_holdout_balanced", {"hf_dataset_name": "gsingh1-py/train", "split_strategy": "domain_holdout", "balance_format": True}),
        ("hf_source_holdout_balanced", {"hf_dataset_name": "gsingh1-py/train", "split_strategy": "source_holdout", "balance_format": True}),
    ]
    rows = []
    for name, kwargs in cases:
        rows.append(run_case(name, **kwargs))

    out_dir = Path("runs/track_a")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ridge_split_audit.csv"
    md_path = out_dir / "ridge_split_audit.md"
    columns = [
        "case",
        "split_strategy",
        "hf_dataset",
        "balance_format",
        "train_samples",
        "val_samples",
        "train_ai",
        "train_humanish",
        "val_ai",
        "val_humanish",
        "train_rubric",
        "val_rubric",
        "auroc",
        "average_precision",
        "latency_ms_per_row",
        "train_seconds",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w") as handle:
        handle.write("# Track A Ridge Split Audit\n\n")
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(fmt(row[col]) for col in columns) + " |\n")

    print(md_path.read_text())
    print(f"Wrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
