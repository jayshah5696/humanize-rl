from __future__ import annotations

import json
import pickle
from pathlib import Path
from time import perf_counter

from humanize_rl.data.scorer_dataset import load_scorer_data, scorer_data_summary
from humanize_rl.scoring.distilled.baselines import FastTextScorer, RidgeScorer

OUT = Path("models/track_a_10k")
L2_PATH = "data/scorer/l2_labeled_scorer_v02_10k.jsonl"
MODELS = {
    "ridge": lambda: RidgeScorer(max_features=10000),
    "fasttext": lambda: FastTextScorer(model_dir=str(OUT / "fasttext_tmp")),
    # Luxical is evaluated in reports but not pickled here; its tokenizer wrapper
    # is not pickle-safe in sentence-transformers on this stack.
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train, val = load_scorer_data(
        local_path=L2_PATH,
        local_paths=[L2_PATH],
        hf_dataset_name=None,
        split_strategy="random",
    )
    # final fit on all 10k L2 rows from train+val
    all_data = {
        key: (train[key] + val[key] if isinstance(train[key], list) else __import__("numpy").concatenate([train[key], val[key]]))
        for key in train
    }
    summary = scorer_data_summary(all_data)
    metadata = {"data_path": L2_PATH, "summary": summary, "models": {}}
    for name, factory in MODELS.items():
        print(f"Training final {name}", flush=True)
        model = factory()
        start = perf_counter()
        model.fit(all_data["texts"], all_data["labels"], all_data["rubrics"])
        seconds = perf_counter() - start
        path = OUT / f"{name}.pkl"
        with path.open("wb") as handle:
            pickle.dump(model, handle)
        metadata["models"][name] = {"path": str(path), "train_seconds": seconds}
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
