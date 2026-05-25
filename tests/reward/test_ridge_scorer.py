"""Tests for the RidgeScorerAdapter and its integration into reward scoring."""
from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock

import pytest

RIDGE_PKL = Path("models/track_a_10k/ridge.pkl")
pytestmark = pytest.mark.skipif(
    not RIDGE_PKL.exists(), reason="ridge pkl not present"
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_ridge():
    with RIDGE_PKL.open("rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def adapter(raw_ridge):
    from humanize_rl.reward.reward import RidgeScorerAdapter
    return RidgeScorerAdapter(raw_ridge)


HUMAN_TEXT = "Staging is back. Root cause: missing STRIPE_WEBHOOK_SECRET. Monitoring till 3 pm."
AI_TEXT = (
    "It is important to note that, as a comprehensive outcome of our diligent "
    "investigation, the staging environment has been successfully restored to its "
    "fully operational state."
)


# ---------------------------------------------------------------------------
# 1. RidgeScorerAdapter protocol shape
# ---------------------------------------------------------------------------

def test_adapter_predict_proba_shape(adapter):
    result = adapter.predict_proba([HUMAN_TEXT, AI_TEXT])
    assert len(result) == 2
    for row in result:
        assert len(row) == 2
        assert abs(sum(row) - 1.0) < 1e-6, "rows must sum to 1"


def test_adapter_last_element_is_p_human(adapter):
    """Convention: last element = P(human). Human text must score higher than AI text."""
    result = adapter.predict_proba([HUMAN_TEXT, AI_TEXT])
    p_human_human = result[0][-1]
    p_human_ai    = result[1][-1]
    assert p_human_human > p_human_ai, (
        f"P(human) for human text ({p_human_human:.3f}) should exceed "
        f"P(human) for AI text ({p_human_ai:.3f})"
    )


def test_adapter_first_element_is_p_ai(adapter):
    """First element = P(AI). AI text must score higher than human text."""
    result = adapter.predict_proba([HUMAN_TEXT, AI_TEXT])
    assert result[1][0] > result[0][0]


def test_adapter_values_in_unit_interval(adapter):
    texts = [HUMAN_TEXT, AI_TEXT, "ok", "hi"]
    for row in adapter.predict_proba(texts):
        for v in row:
            assert 0.0 <= v <= 1.0, f"out of [0,1]: {v}"


# ---------------------------------------------------------------------------
# 2. load_ridge_scorer
# ---------------------------------------------------------------------------

def test_load_ridge_scorer_returns_adapter():
    from humanize_rl.reward.reward import RidgeScorerAdapter, load_ridge_scorer
    scorer = load_ridge_scorer()
    assert scorer is not None
    assert isinstance(scorer, RidgeScorerAdapter)


def test_load_ridge_scorer_explicit_path():
    from humanize_rl.reward.reward import load_ridge_scorer
    scorer = load_ridge_scorer(RIDGE_PKL)
    assert scorer is not None


def test_load_ridge_scorer_missing_path_returns_none():
    from humanize_rl.reward.reward import load_ridge_scorer
    scorer = load_ridge_scorer(Path("/nonexistent/path.pkl"))
    assert scorer is None


# ---------------------------------------------------------------------------
# 3. Style score is higher for human text when ridge is active
# ---------------------------------------------------------------------------

def test_style_score_higher_with_ridge_for_human_text():
    from humanize_rl.reward.reward import (
        RidgeScorerAdapter, _style_score, _track_a_human_probability,
    )
    from humanize_rl.scoring.aggregator import score_text

    with RIDGE_PKL.open("rb") as f:
        raw = pickle.load(f)
    adapter = RidgeScorerAdapter(raw)

    track_a_human = _track_a_human_probability(adapter, HUMAN_TEXT)
    track_a_ai    = _track_a_human_probability(adapter, AI_TEXT)

    assert track_a_human is not None
    assert track_a_ai is not None
    assert track_a_human > track_a_ai, (
        f"P(human) human={track_a_human:.3f} ai={track_a_ai:.3f}"
    )

    style_human = _style_score(HUMAN_TEXT, track_a_human)
    style_ai    = _style_score(AI_TEXT,    track_a_ai)
    assert style_human > style_ai, (
        f"style human={style_human:.3f} ai={style_ai:.3f}"
    )


def test_ridge_boosts_style_vs_layer1_alone():
    """Ridge should push human-text style above pure layer1 score."""
    from humanize_rl.reward.reward import (
        RidgeScorerAdapter, _style_score, _track_a_human_probability,
    )

    with RIDGE_PKL.open("rb") as f:
        raw = pickle.load(f)
    adapter = RidgeScorerAdapter(raw)

    track_a = _track_a_human_probability(adapter, HUMAN_TEXT)
    style_with    = _style_score(HUMAN_TEXT, track_a)
    style_without = _style_score(HUMAN_TEXT, None)

    # For a human-sounding text, ridge should lift the score (or at worst hold)
    assert style_with >= style_without - 0.01, (
        f"ridge degraded style: with={style_with:.3f} without={style_without:.3f}"
    )


# ---------------------------------------------------------------------------
# 4. score_response uses ridge when present
# ---------------------------------------------------------------------------

def test_score_response_with_ridge_ranks_correctly():
    from humanize_rl.reward.reward import load_ridge_scorer, score_response
    from humanize_rl.reward.tasks import RLTask

    task = RLTask.model_validate({
        "id": "rl_v01_000099",
        "family": "rewrite_repair",
        "domain": "slack",
        "mode": "rewrite",
        "register": "casual",
        "instruction": "Clean up this Slack update.",
        "input_text": "Please be advised that staging has been restored. The root cause was a missing STRIPE_WEBHOOK_SECRET value, and we will monitor it until 3 pm.",
        "constraints": {"max_words": 40, "preserve_numbers": True, "preserve_entities": True},
        "reward_profile": "rewrite_faithful_concise",
        "trap_tags": ["over_polish"],
        "split": "train",
        "required_facts": ["staging", "STRIPE_WEBHOOK_SECRET", "3 pm"],
    })

    scorer = load_ridge_scorer()
    r_human = score_response(task, HUMAN_TEXT, scorer)
    r_ai    = score_response(task, AI_TEXT, scorer)

    assert r_human.reward > r_ai.reward
    # Ridge must actually differentiate style
    assert r_human.components["style"] > r_ai.components["style"]


def test_score_response_ridge_style_capped_at_20pct():
    """Ridge contribution to style must be <= TRACK_A_STYLE_CAP (0.20)."""
    from humanize_rl.reward.reward import TRACK_A_STYLE_CAP, load_ridge_scorer, score_response
    from humanize_rl.reward.tasks import RLTask

    task = RLTask.model_validate({
        "id": "rl_v01_000098",
        "family": "rewrite_repair",
        "domain": "slack",
        "mode": "rewrite",
        "register": "casual",
        "instruction": "Rewrite this.",
        "input_text": "Staging is back after fix.",
        "constraints": {},
        "reward_profile": "rewrite_faithful_concise",
        "trap_tags": ["over_polish"],
        "split": "train",
        "required_facts": [],
    })

    scorer = load_ridge_scorer()
    r_with    = score_response(task, HUMAN_TEXT, scorer)
    r_without = score_response(task, HUMAN_TEXT, None)

    delta = abs(r_with.components["style"] - r_without.components["style"])
    assert delta <= TRACK_A_STYLE_CAP + 0.01, (
        f"Ridge contribution {delta:.3f} exceeds cap {TRACK_A_STYLE_CAP}"
    )


# ---------------------------------------------------------------------------
# 5. Module-level _RIDGE_SCORER wired in adapters
# ---------------------------------------------------------------------------

def test_verifiers_adapter_loads_ridge_scorer():
    import humanize_rl.reward.verifiers_adapter as va
    # If pkl exists, _RIDGE_SCORER must be non-None
    assert va._RIDGE_SCORER is not None, (
        "_RIDGE_SCORER is None in verifiers_adapter — ridge pkl not loaded"
    )


def test_grpo_rewards_loads_ridge_scorer():
    import humanize_rl.reward.grpo_rewards as gr
    assert gr._RIDGE_SCORER is not None, (
        "_RIDGE_SCORER is None in grpo_rewards — ridge pkl not loaded"
    )
