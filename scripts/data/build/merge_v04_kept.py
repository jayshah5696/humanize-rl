import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed/v04_kept_merged.jsonl")
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()
    seen = set()
    kept = []
    for inp in args.inputs:
        for line in Path(inp).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row.get("instruction", "").strip(), row.get("response", "").strip())
            if key in seen:
                continue
            seen.add(key)
            kept.append(row)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    print(f"merged={len(kept)} output={out}")


if __name__ == "__main__":
    main()
