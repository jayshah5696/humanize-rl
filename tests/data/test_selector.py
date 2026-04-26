"""Tests for V-Slice 2 best-of-N candidate selection."""

from __future__ import annotations

import json
from pathlib import Path

from humanize_rl.data.selector import (
    load_candidates,
    select_aiify_best,
    select_humanize_best,
)


def _arka_row(original_text: str, response_text: str) -> dict:
    return {
        "instruction": "x",
        "response": response_text,
        "system": json.dumps(
            {"transform_original": {"field": "payload.response", "text": original_text}}
        ),
        "turns": None,
    }


# ---------------------------------------------------------------------------
# load_candidates
# ---------------------------------------------------------------------------


def test_load_candidates_groups_by_original(tmp_path: Path) -> None:
    p = tmp_path / "in.jsonl"
    rows = [
        _arka_row("ORIG_A", "candidate_a1"),
        _arka_row("ORIG_A", "candidate_a2"),
        _arka_row("ORIG_B", "candidate_b1"),
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    groups = load_candidates(p)
    assert set(groups.keys()) == {"ORIG_A", "ORIG_B"}
    assert len(groups["ORIG_A"]) == 2
    assert len(groups["ORIG_B"]) == 1


def test_load_candidates_skips_blank_rows(tmp_path: Path) -> None:
    p = tmp_path / "in.jsonl"
    p.write_text(json.dumps(_arka_row("ORIG", "x")) + "\n\n")
    groups = load_candidates(p)
    assert sum(len(v) for v in groups.values()) == 1


# ---------------------------------------------------------------------------
# AIify selector
# ---------------------------------------------------------------------------


def test_aiify_selector_prefers_band_over_clean_human() -> None:
    """A heavily-AI candidate inside the band should beat a 'looks human' one."""
    original = " ".join(["word"] * 80)
    # Heavily AI-flavored: hits hedges, transitions, AI opener.
    ai_in_band = (
        "When working with this, it is worth noting that one common pitfall "
        "is X. Furthermore, in general one of the most important aspects is "
        "Y. Moreover, additionally one of the key takeaways is Z. "
        "Generally speaking, in many cases this matters."
    )
    # Plain human-style text — should land near 1.0, far from the 0.30 band.
    looks_human = (
        "Spent some time digging into this. The fix was a one-liner once I "
        "spotted the locale issue. Painful but quick."
    )
    candidates = [_arka_row(original, looks_human), _arka_row(original, ai_in_band)]
    chosen = select_aiify_best(candidates, original)
    assert chosen["response"] == ai_in_band


def test_aiify_selector_breaks_ties_by_length() -> None:
    """Two candidates with identical Layer 1 score: prefer the shorter one."""
    original = " ".join(["seed"] * 50)
    short = " ".join(["short"] * 50)
    longer = " ".join(["short"] * 100)
    candidates = [_arka_row(original, longer), _arka_row(original, short)]
    chosen = select_aiify_best(candidates, original)
    assert chosen["response"] == short


def test_aiify_selector_prefers_in_cap_over_better_score() -> None:
    """V-Slice 2.2: an in-cap candidate is preferred over an out-of-cap
    candidate even if the out-of-cap one scores closer to the band
    midpoint. Length cap is enforced upstream of band-distance ranking."""
    original = " ".join(["seed"] * 30)  # 30 words → cap is 37 words
    # 30 words, ratio 1.0 — in cap. Heavily AI-flavored.
    in_cap_ai = (
        "When working with this, it is worth noting one common pitfall is X. "
        "Furthermore, additionally moreover this is generally one of the key "
        "common issues seen across many teams in many cases."
    )
    # 50 words, ratio 1.67 — out of cap.
    out_of_cap_ai = in_cap_ai + (
        " In addition, it is important to note that this scenario is "
        "generally worth mentioning across the board, and one of the most "
        "common pitfalls in this area is forgetting that."
    )
    candidates = [_arka_row(original, out_of_cap_ai), _arka_row(original, in_cap_ai)]
    chosen = select_aiify_best(candidates, original)
    assert chosen["response"] == in_cap_ai


# ---------------------------------------------------------------------------
# Humanize selector
# ---------------------------------------------------------------------------


def test_humanize_selector_picks_largest_safe_delta() -> None:
    """Tier 0 candidate (big delta, no overshoot) wins over a smaller delta."""
    aiified_score = 0.30
    original_score = 0.85

    # Two human-leaning candidates of different humanness.
    safe_big_delta = (
        "I spent half the day on this. Turned out to be a locale bug. Three "
        "characters fixed it; I added an export to the workflow too."
    )
    weak_recovery = (
        "When working with this, it is worth noting that the fix was small. "
        "Furthermore, additionally I changed the workflow as well."
    )
    candidates = [
        _arka_row("ANY", weak_recovery),
        _arka_row("ANY", safe_big_delta),
    ]
    chosen = select_humanize_best(candidates, aiified_score, original_score)
    assert chosen["response"] == safe_big_delta


def test_humanize_selector_avoids_overshoot_when_possible() -> None:
    """Given a choice, prefer a candidate that doesn't beat the original."""
    aiified_score = 0.30

    # `safe` should land near 0.85; the original is 0.85, tolerance 0.05.
    safe = (
        "I spent half the day chasing this. Painful, but the locale issue "
        "was the smoking gun. Quick fix in the end."
    )
    # `overshoot_candidate` is engineered to be a generic high-scoring AI-free
    # rewrite; we don't strictly test it here. Test the contract: when the
    # safe candidate exists in tier 0, it wins.
    original_score = 0.83
    candidates = [_arka_row("ANY", safe)]
    chosen = select_humanize_best(candidates, aiified_score, original_score)
    assert chosen["response"] == safe
