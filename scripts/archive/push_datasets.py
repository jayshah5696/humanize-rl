import json
import os

from datasets import load_dataset
from huggingface_hub import HfApi


def push_to_hf():
    repo_id = "jayshah5696/humanize-rl-v03"
    api = HfApi()

    try:
        api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
        print(f"Verified dataset repo: {repo_id}")
    except Exception as e:
        print(f"Error creating/verifying repo: {e}")
        return

    print("Combining SFT datasets...")
    sft_files = [
        "data/processed/v03_corpus_sft.jsonl",
        "data/processed/expansion_sft.jsonl",
    ]
    combined_sft = []
    for f in sft_files:
        if os.path.exists(f):
            with open(f) as file:
                for line in file:
                    combined_sft.append(json.loads(line))
        else:
            print(f"Warning: {f} not found.")

    combined_sft_path = "data/processed/v03_combined_sft.jsonl"
    with open(combined_sft_path, "w") as f:
        for item in combined_sft:
            f.write(json.dumps(item) + "\n")
    print(
        f"Saved combined SFT dataset ({len(combined_sft)} pairs) to {combined_sft_path}"
    )

    print("Loading and pushing combined SFT dataset...")
    sft_dataset = load_dataset("json", data_files=combined_sft_path, split="train")
    sft_dataset.push_to_hub(repo_id, config_name="sft", split="train")
    print("Successfully pushed SFT dataset.")

    print("\nCombining and pushing Benchmark dataset...")
    bench_files = [
        "data/benchmark/v03_corpus_matched.jsonl",
        "data/benchmark/expansion_matched.jsonl",
    ]
    combined_bench = []
    for f in bench_files:
        if os.path.exists(f):
            with open(f) as file:
                for line in file:
                    combined_bench.append(json.loads(line))
        else:
            print(f"Warning: {f} not found.")

    combined_bench_path = "data/benchmark/v03_combined_matched.jsonl"
    with open(combined_bench_path, "w") as f:
        for item in combined_bench:
            f.write(json.dumps(item) + "\n")

    bench_dataset = load_dataset("json", data_files=combined_bench_path, split="train")
    bench_dataset.push_to_hub(repo_id, config_name="benchmark", split="train")
    print(f"Successfully pushed Benchmark dataset ({len(combined_bench)} rows).")

    print(
        "\nDone! Dataset available at: https://huggingface.co/datasets/jayshah5696/humanize-rl-v03"
    )


if __name__ == "__main__":
    push_to_hf()
