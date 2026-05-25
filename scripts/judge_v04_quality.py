import argparse
import json
import os
import time
from pathlib import Path

from openai import OpenAI

SYSTEM = """You are a strict data-quality judge for an SFT writing dataset.
Return only valid JSON.

Judge whether the assistant response is good training data.
Be conservative. Fluent is not enough.
"""

PROMPT = """Evaluate this training row.

Instruction:
{instruction}

Response:
{response}

Checks:
1. naturalness: 1-5. 5 = a normal competent person wrote it. 3 = generic/stiff. 1 = unusable.
2. fact_preservation: pass/fail. For rewrite/edit tasks, response must preserve source facts and not add unsupported details.
3. instruction_following: pass/fail.
4. unsupported_details: pass/fail. fail if response invents names, tools, metrics, dates, causes, companies, locations, or project details not given. Placeholders like [Name] are okay.
5. ai_tells: pass/fail. fail if response has assistant boilerplate, corporate filler, excessive polish, or phrases like "Certainly", "it is worth noting", "furthermore", "please don't hesitate".
6. keep: true/false. keep only if naturalness >= 4 and all checks pass.
7. reason: one short sentence.

Return JSON exactly:
{{"naturalness": 4, "fact_preservation": "pass", "instruction_following": "pass", "unsupported_details": "pass", "ai_tells": "pass", "keep": true, "reason": "..."}}
"""


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def judge(client: OpenAI, model: str, row: dict) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT.format(instruction=row.get("instruction", ""), response=row.get("response", ""))},
        ],
        temperature=0,
        max_tokens=256,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    data = json.loads(content)
    data["usage"] = getattr(resp, "usage", None).model_dump() if getattr(resp, "usage", None) else None
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", default=None)
    parser.add_argument("--kept-output", default=None)
    parser.add_argument("--model", default="google/gemini-3.1-flash-lite-preview")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required")

    rows = load_rows(Path(args.input))
    if args.limit is not None:
        rows = rows[: args.limit]

    output = Path(args.output) if args.output else Path(args.input).with_suffix(".judged.jsonl")
    kept_output = Path(args.kept_output) if args.kept_output else Path(args.input).with_suffix(".kept.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    kept_output.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    kept = 0
    total = 0
    with output.open("w") as out, kept_output.open("w") as kept_out:
        for idx, row in enumerate(rows, start=1):
            try:
                verdict = judge(client, args.model, row)
            except Exception as exc:
                verdict = {"naturalness": 0, "fact_preservation": "fail", "instruction_following": "fail", "unsupported_details": "fail", "ai_tells": "fail", "keep": False, "reason": f"judge_error: {exc}"}
            judged = {**row, "quality_judge": verdict}
            out.write(json.dumps(judged, ensure_ascii=False) + "\n")
            if bool(verdict.get("keep")):
                kept += 1
                kept_out.write(json.dumps(judged, ensure_ascii=False) + "\n")
            total += 1
            print(f"{idx}/{len(rows)} keep={verdict.get('keep')} naturalness={verdict.get('naturalness')} reason={verdict.get('reason')}")
            if args.sleep:
                time.sleep(args.sleep)

    print(f"kept={kept} total={total} keep_rate={kept / total if total else 0:.3f}")
    print(f"judged={output}")
    print(f"kept_output={kept_output}")


if __name__ == "__main__":
    main()
