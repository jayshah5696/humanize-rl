from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from humanize_rl.data.scorer_dataset import load_scorer_data, scorer_data_summary


def evaluate(train_data: dict, val_data: dict) -> dict[str, float | int]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, min_df=2)
    classifier = LogisticRegression(max_iter=300, solver="liblinear")
    start = perf_counter()
    x_train = vectorizer.fit_transform(train_data["texts"])
    classifier.fit(x_train, train_data["labels"])
    train_seconds = perf_counter() - start

    start = perf_counter()
    x_val = vectorizer.transform(val_data["texts"])
    pred = classifier.predict_proba(x_val)[:, 1]
    latency_ms = ((perf_counter() - start) / max(len(val_data["texts"]), 1)) * 1000

    labels = val_data["labels"]
    auroc = float("nan")
    ap = float("nan")
    if len(set(labels.tolist())) > 1:
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
    ]
    rows = []
    for name, kwargs in cases:
        print(f"Running {name}...", flush=True)
        train_data, val_data = load_scorer_data(**kwargs)
        row = evaluate(train_data, val_data)
        row.update(
            {
                "case": name,
                "split_strategy": kwargs.get("split_strategy", "random"),
                "hf_dataset": kwargs.get("hf_dataset_name") or "none",
                "balance_format": bool(kwargs.get("balance_format", False)),
            }
        )
        rows.append(row)

    out_dir = Path("runs/track_a")
    out_dir.mkdir(parents=True, exist_ok=True)
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
    csv_path = out_dir / "ridge_fast_split_audit.csv"
    md_path = out_dir / "ridge_fast_split_audit.md"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with md_path.open("w") as handle:
        handle.write("# Track A Fast Ridge Split Audit\n\n")
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(fmt(row[col]) for col in columns) + " |\n")
    print(md_path.read_text())


if __name__ == "__main__":
    main()
