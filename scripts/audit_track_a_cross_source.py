from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from humanize_rl.data.scorer_dataset import load_scorer_data, scorer_data_summary


def fit_eval(train: dict, val: dict) -> dict[str, float | int]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, min_df=2)
    clf = LogisticRegression(max_iter=300, solver="liblinear")
    clf.fit(vectorizer.fit_transform(train["texts"]), train["labels"])
    pred = clf.predict_proba(vectorizer.transform(val["texts"]))[:, 1]
    labels = val["labels"]
    return {
        "train_samples": len(train["texts"]),
        "val_samples": len(val["texts"]),
        "train_rubric": scorer_data_summary(train)["rubric_samples"],
        "val_rubric": scorer_data_summary(val)["rubric_samples"],
        "val_ai": scorer_data_summary(val)["ai_samples"],
        "val_humanish": scorer_data_summary(val)["humanish_samples"],
        "auroc": float(roc_auc_score(labels, pred)),
        "average_precision": float(average_precision_score(labels, pred)),
        "mean_ai_pred_on_ai": float(np.mean(pred[labels == 1])),
        "mean_ai_pred_on_humanish": float(np.mean(pred[labels == 0])),
    }


def main() -> None:
    local_train, local_val = load_scorer_data(hf_dataset_name=None, split_strategy="random")
    hf_train, hf_val = load_scorer_data(
        local_paths=[], hf_dataset_name="gsingh1-py/train", split_strategy="random", balance_format=True
    )
    # combine train+val from each source family for cross-source tests
    local_all = {
        key: (local_train[key] + local_val[key] if isinstance(local_train[key], list) else np.concatenate([local_train[key], local_val[key]]))
        for key in local_train
    }
    hf_all = {
        key: (hf_train[key] + hf_val[key] if isinstance(hf_train[key], list) else np.concatenate([hf_train[key], hf_val[key]]))
        for key in hf_train
    }
    rows = [
        ("train_local_test_hf", fit_eval(local_all, hf_all)),
        ("train_hf_test_local", fit_eval(hf_all, local_all)),
    ]
    cols = ["case", *rows[0][1].keys()]
    out = Path("runs/track_a/cross_source_audit.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        handle.write("# Track A Cross Source Audit\n\n")
        handle.write("| " + " | ".join(cols) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for name, row in rows:
            vals = [name]
            for col in cols[1:]:
                val = row[col]
                vals.append(f"{val:.4f}" if isinstance(val, float) else str(val))
            handle.write("| " + " | ".join(vals) + " |\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
