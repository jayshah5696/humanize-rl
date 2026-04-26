"""V-Slice 2 — best-of-N candidate selection.

Arka's `generation_multiplier: N` produces N candidates per seed in a flat
JSONL. This module groups by the original seed text (carried in
`system.transform_original.text`), Layer-1 scores each candidate, and
picks the best one per group according to a strategy.

Two strategies live here:

- `select_aiify_best`: pick the candidate whose Layer-1 score lands in the
  v03 target band (0.20-0.40, midpoint 0.30); ties broken by shortest
  candidate (helps the persistent length-blowup issue from V-Slice 0).

- `select_humanize_best`: pick the candidate that maximizes humanize delta
  (humanized - aiified) without overshooting `original_score + 0.05`.
  Falls back to the closest-to-original if no candidate satisfies the cap.

Output shape is byte-compatible with arka's transform output, so the
existing walking-skeleton runner consumes it without changes.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from humanize_rl.scoring.aggregator import score_text

_WORD_RE = re.compile(r"\b\w+\b")


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _original_text(record: dict) -> str:
    """Pull the seed text arka stored under system.transform_original."""
    system = record.get("system", "")
    if not system:
        return ""
    try:
        meta = json.loads(system)
    except (json.JSONDecodeError, TypeError):
        return ""
    return meta.get("transform_original", {}).get("text", "")


def load_candidates(path: Path) -> dict[str, list[dict]]:
    """Group arka transform output rows by their original seed text."""
    groups: dict[str, list[dict]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            original = _original_text(record)
            if not original or not record.get("response"):
                continue
            groups[original].append(record)
    return dict(groups)


# ---------------------------------------------------------------------------
# AIify selection
# ---------------------------------------------------------------------------

# Band: see spec §8.5. We aim for the midpoint and prefer shorter candidates
# on near-ties to combat the V-Slice 0 length-blowup pattern.
AIIFY_BAND_MIDPOINT = 0.30
AIIFY_BAND_LO = 0.20
AIIFY_BAND_HI = 0.40
AIIFY_TIE_EPSILON = 0.05


AIIFY_LENGTH_HARD_CAP = 1.25  # match pair gate


def _aiify_score(candidate: dict, original: str) -> tuple[int, float]:
    """Sort key. Lower is better.

    Two-tier ordering:
      Tier 0: length ratio ≤ hard cap (1.25×).
      Tier 1: length ratio above hard cap (only used as fallback).
    Within a tier, rank by distance from the band midpoint (0.30).

    Why two tiers: V-Slice 2 showed 4/7 rejects were pure length blowups,
    even when an in-band candidate existed. The selector was choosing the
    'best score' candidate that happened to be too long. This ordering
    forces the selector to prefer any in-cap candidate over an out-of-cap
    one, even if the out-of-cap one scores marginally better.
    """
    text = candidate["response"]
    s = score_text(text).overall
    distance = abs(s - AIIFY_BAND_MIDPOINT)
    orig_words = max(_word_count(original), 1)
    length_ratio = _word_count(text) / orig_words
    tier = 0 if length_ratio <= AIIFY_LENGTH_HARD_CAP else 1
    return (tier, distance)


def select_aiify_best(candidates: list[dict], original: str) -> dict:
    """Pick the AIify candidate closest to the target band midpoint."""
    return min(candidates, key=lambda c: _aiify_score(c, original))


# ---------------------------------------------------------------------------
# Humanize selection
# ---------------------------------------------------------------------------

HUMANIZE_OVERSHOOT_TOLERANCE = 0.05  # spec §10.4: flag if humanized > orig+0.05


def _humanize_score_key(
    candidate: dict, aiified_score: float, original_score: float
) -> tuple[int, float, float]:
    """Sort key. Lower tuple is better.

    Tier 0 (preferred): humanized in [aiified+0.30, original+0.05]
    Tier 1 (acceptable): humanized > aiified+0.30 but overshoots original
    Tier 2 (fallback):   anything else, ranked by closeness to original

    Within a tier, prefer larger humanize delta.
    """
    text = candidate["response"]
    h = score_text(text).overall
    delta = h - aiified_score

    if delta >= 0.30 and h <= original_score + HUMANIZE_OVERSHOOT_TOLERANCE:
        tier = 0
    elif delta >= 0.30:
        tier = 1
    else:
        tier = 2

    # Within tier, larger delta is better -> negate for "lower is better"
    return (tier, -delta, abs(h - original_score))


def select_humanize_best(
    candidates: list[dict], aiified_score: float, original_score: float
) -> dict:
    return min(
        candidates,
        key=lambda c: _humanize_score_key(c, aiified_score, original_score),
    )


# ---------------------------------------------------------------------------
# CLI: read multi-candidate arka output, write 1-best-per-original JSONL
# ---------------------------------------------------------------------------


def _emit(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Best-of-N candidate selector.")
    parser.add_argument(
        "--mode",
        choices=("aiify", "humanize"),
        required=True,
        help="aiify: pick by target band. humanize: pick by delta + overshoot cap.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    # humanize mode needs the original seeds to know `original_score`
    parser.add_argument(
        "--originals",
        type=Path,
        help="(humanize mode) the human seed JSONL (rows must have 'response').",
    )
    parser.add_argument(
        "--aiify-selected",
        type=Path,
        help=(
            "(humanize mode) the AIify-selector output JSONL. Used to bridge "
            "AIified-text -> human-original-text so the humanize selector "
            "can compute the overshoot ceiling against the *human* score."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON selection report.",
    )
    args = parser.parse_args()

    groups = load_candidates(args.input)
    if not groups:
        print(f"No candidates loaded from {args.input}")
        return

    # Build a lookup of aiified_text -> human_original_score (humanize mode).
    # Two hops:
    #   human_seeds:    response = human_text                -> score(human_text)
    #   aiify_selected: response = aiified_text, system.transform_original.text = human_text
    # Resulting map: aiified_text -> score(human_text).
    aiified_to_human_score: dict[str, float] = {}
    if args.mode == "humanize":
        if not args.originals or not args.originals.exists():
            raise SystemExit("humanize mode requires --originals (the human seed file)")
        if not args.aiify_selected or not args.aiify_selected.exists():
            raise SystemExit(
                "humanize mode requires --aiify-selected (the AIify selector output)"
            )
        human_score_by_text: dict[str, float] = {}
        with args.originals.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                seed = json.loads(line)
                txt = seed.get("response", "")
                if txt:
                    human_score_by_text[txt] = score_text(txt).overall
        with args.aiify_selected.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                aiified_text = rec.get("response", "")
                human_text = _original_text(rec)
                if aiified_text and human_text in human_score_by_text:
                    aiified_to_human_score[aiified_text] = human_score_by_text[
                        human_text
                    ]

    selected: list[dict] = []
    report: list[dict] = []

    for original, candidates in groups.items():
        if args.mode == "aiify":
            chosen = select_aiify_best(candidates, original)
            chosen_score = score_text(chosen["response"]).overall
            scored = [
                {
                    "score": round(score_text(c["response"]).overall, 4),
                    "words": _word_count(c["response"]),
                }
                for c in candidates
            ]
            report.append(
                {
                    "n_candidates": len(candidates),
                    "original_words": _word_count(original),
                    "candidates": scored,
                    "chosen_score": round(chosen_score, 4),
                    "chosen_words": _word_count(chosen["response"]),
                }
            )
        else:  # humanize
            # `original` here is what arka carried as transform_original, which
            # for the humanize stage is the AIified text we fed in.
            aiified_text = original
            aiified_score = score_text(aiified_text).overall
            original_score = aiified_to_human_score.get(aiified_text, 1.0)
            chosen = select_humanize_best(candidates, aiified_score, original_score)
            chosen_score = score_text(chosen["response"]).overall
            scored = [
                {
                    "score": round(score_text(c["response"]).overall, 4),
                    "delta": round(
                        score_text(c["response"]).overall - aiified_score, 4
                    ),
                    "words": _word_count(c["response"]),
                }
                for c in candidates
            ]
            report.append(
                {
                    "n_candidates": len(candidates),
                    "aiified_score": round(aiified_score, 4),
                    "original_score_known": original_score < 1.0,
                    "candidates": scored,
                    "chosen_score": round(chosen_score, 4),
                    "chosen_words": _word_count(chosen["response"]),
                }
            )
        selected.append(chosen)

    _emit(selected, args.output)
    print(
        f"Selected {len(selected)} from {sum(len(v) for v in groups.values())} candidates"
    )
    print(f"Wrote: {args.output}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps({"mode": args.mode, "groups": report}, indent=2)
        )
        print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
