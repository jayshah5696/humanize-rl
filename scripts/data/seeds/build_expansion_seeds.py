import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from humanize_rl.data.loaders import stream_creative_tiny, stream_email_customer
from humanize_rl.data.seed import to_arka_seed_row


def main():
    seeds = []
    print("Extracting 500 customer emails...")
    seeds.extend(list(stream_email_customer(500)))

    print("Extracting 500 TinyStories...")
    seeds.extend(list(stream_creative_tiny(500)))

    out_path = Path("seeds/v03/expansion_seeds.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        for seed in seeds:
            f.write(json.dumps(to_arka_seed_row(seed)) + "\n")

    print(f"Success! Wrote {len(seeds)} expansion seeds to {out_path}")


if __name__ == "__main__":
    main()
