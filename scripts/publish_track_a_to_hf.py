from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, upload_folder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="jayshah5696/humanize-rl-track-a-scorer")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    readme = Path("models/track_a_10k/README.md")
    readme.write_text(
        "# Humanize-RL Track A Distilled Scorer\n\n"
        "Local surrogate scorers for humanness / AI-writing-pattern detection.\n\n"
        "## Artifacts\n\n"
        "- `ridge.pkl`: TF-IDF 1-4 gram + Logistic/Ridge heads. Recommended default.\n"
        "- `fasttext.pkl`: fastText supervised classifier + vector regression heads. Secondary comparator.\n"
        "- `metadata.json`: training summary.\n\n"
        "## Training data\n\n"
        "Trained on `data/scorer/l2_labeled_scorer_v02_10k.jsonl`: 10,000 L2-rubric-labeled rows.\n"
        "Labels include AI-generated, human-authored, and humanized-synthetic examples.\n\n"
        "## Evaluation summary\n\n"
        "Ridge achieved ~0.997-0.999 AUROC across capped random/group splits, with the best rubric MSE and ~1ms/row latency.\n"
        "False-positive challenge set showed 0.0% false positives at threshold 0.5 on 400 human-authored challenge rows.\n\n"
        "See the project repository for full reports and figures.\n"
    )
    upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path="models/track_a_10k",
        commit_message="Upload Track A 10k distilled scorer artifacts",
    )
    print(f"Uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
