"""Regression tests pinned to real V-Slice 0 AIify outputs.

These strings were emitted by the live AIify run (Gemini 3.1 Flash Lite)
on 2026-04-25 and exposed gaps where Layer 1 returned "looks human" on
text that was overtly AI-flavored. They lock the fixes in place.

If you change pattern lists or scoring math and these tests fail, the
walking-skeleton acceptance rate will silently drop. Re-run
`just v03-ws-score` and reason about the regression before relaxing
these expectations.
"""

from __future__ import annotations

from humanize_rl.scoring.layer1 import (
    score_hedging,
    score_opener,
    score_transitions,
)

# A representative AIified paragraph from output/v03/ws-aiify.jsonl.
# Contains: AI opener ("When working with X,"), hedging ("it is worth noting",
# "in general", "one common pitfall"), transitions ("Furthermore", "Moreover",
# "Additionally").
AIIFIED_TECH_NOTE = (
    "When working with pytest, it is worth noting that I spent half a day "
    "chasing a failure that only showed up in CI. Locally everything passed, "
    "furthermore in GitHub Actions one parametrize case blew up with a "
    "UnicodeDecodeError on a fixture file. Moreover, it turned out the "
    "runner image had LANG=C while my Mac defaults to en_US.UTF-8, "
    "additionally our YAML loader was implicitly using the locale encoding. "
    "In general, the fix was three characters: pass encoding='utf-8' to "
    "open(). Furthermore, I added an LC_ALL=C.UTF-8 export to the workflow "
    "as well so this stops biting other tests later."
)

HUMAN_TECH_NOTE = (
    "Spent half a day chasing a pytest failure that only showed up in CI. "
    "Locally everything passed, in GitHub Actions one parametrize case blew "
    "up with a UnicodeDecodeError on a fixture file. Turned out the runner "
    "image had LANG=C while my Mac defaults to en_US.UTF-8, and our YAML "
    "loader was implicitly using the locale encoding. The fix was three "
    "characters: pass encoding='utf-8' to open()."
)


# --- opener -----------------------------------------------------------------


def test_opener_catches_when_working_with() -> None:
    """The 10/10 opener the LLM produced for the walking skeleton."""
    assert score_opener(AIIFIED_TECH_NOTE) == 0.0


def test_opener_passes_human_anecdote() -> None:
    assert score_opener(HUMAN_TECH_NOTE) == 1.0


def test_opener_catches_uncontracted_id_be_happy() -> None:
    assert score_opener("I would be happy to walk you through this.") == 0.0


def test_opener_catches_in_modern_workflows() -> None:
    text = "In modern development workflows, CI pipelines often..."
    assert score_opener(text) == 0.0


# --- hedging ----------------------------------------------------------------


def test_hedging_catches_uncontracted_it_is_worth() -> None:
    """Bug: original regex required `it'?s` so `it is worth noting` slipped."""
    assert score_hedging("It is worth noting that this matters.") < 0.5


def test_hedging_counts_repeated_occurrences() -> None:
    """Bug: old code counted distinct-patterns-matched, not occurrences."""
    text = (
        "It is worth noting one thing. It is worth noting another. "
        "It is worth noting a third. It is worth noting a fourth."
    ) * 1
    # 4 occurrences of one pattern in ~30 words -> very high density
    assert score_hedging(text) <= 0.3


def test_hedging_catches_in_general_and_one_common_pitfall() -> None:
    text = (
        "In general, this works. Furthermore, one common pitfall is forgetting "
        "to handle the case where the value is None. In general, you should "
        "always check."
    )
    assert score_hedging(text) <= 0.5


def test_hedging_passes_clean_human_text() -> None:
    assert score_hedging(HUMAN_TECH_NOTE) >= 0.8


# --- transitions ------------------------------------------------------------


def test_transitions_flag_aiified_paragraph() -> None:
    """3 transitions in ~96 words -> density ~6 per 200 words -> AI-like."""
    assert score_transitions(AIIFIED_TECH_NOTE) <= 0.3
