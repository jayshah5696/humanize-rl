from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, upload_file

FILES = [
    ("models/track_a_10k/ridge.pkl", "ridge.pkl"),
    ("models/track_a_10k/metadata.json", "metadata.json"),
    ("runs/track_a_capped/REPORT.md", "reports/track_a_report.md"),
    ("runs/track_a_capped/metrics.csv", "reports/metrics.csv"),
    ("runs/track_a_capped/auroc.png", "reports/auroc.png"),
    ("runs/track_a_capped/rubric_mse.png", "reports/rubric_mse.png"),
    ("runs/track_a_capped/latency.png", "reports/latency.png"),
    ("runs/false_positive_eval/REPORT.md", "reports/false_positive_report.md"),
    ("runs/false_positive_eval/false_positive_rate.png", "reports/false_positive_rate.png"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="jayshah5696/humanize-rl-track-a-ridge-scorer")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    readme = Path("/tmp/track_a_ridge_readme.md")
    readme.write_text(
        "# Humanize-RL Track A Ridge Scorer\n\n"
        "Recommended local distilled humanness scorer trained on 10,000 Gemini Layer-2 rubric-labeled rows.\n\n"
        "## Why Ridge?\n\n"
        "Ridge/TF-IDF had the best practical tradeoff: ~0.997-0.999 AUROC, lowest rubric MSE among practical models, ~1ms/row latency, and 0.0% false positives on a 400-row human-authored challenge set.\n\n"
        "## Files\n\n"
        "- `ridge.pkl`: selected scorer artifact.\n"
        "- `metadata.json`: training summary.\n"
        "- `reports/`: evaluation figures and tables.\n"
    )
    upload_file(repo_id=args.repo_id, repo_type="model", path_or_fileobj=str(readme), path_in_repo="README.md")
    for local, remote in FILES:
        upload_file(repo_id=args.repo_id, repo_type="model", path_or_fileobj=local, path_in_repo=remote)
    print(f"Uploaded ridge scorer to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
