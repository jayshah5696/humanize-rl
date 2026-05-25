from __future__ import annotations

from humanize_rl.reward.reward import score_response
from humanize_rl.reward.tasks import RLTask


def _task(profile: str = "rewrite_faithful_concise", **overrides: object) -> RLTask:
    base: dict[str, object] = {
        "id": "rl_v01_000001",
        "family": "rewrite_repair",
        "domain": "slack",
        "mode": "rewrite",
        "register": "casual",
        "instruction": "Clean up this Slack update. Return only the message.",
        "input_text": "Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix.",
        "constraints": {
            "max_words": 20,
            "no_subject_line": True,
            "no_signoff": True,
            "preserve_numbers": True,
            "preserve_entities": True,
        },
        "reward_profile": profile,
        "trap_tags": ["wrapper_phrase"],
        "split": "train",
        "required_facts": ["Staging recovered", "3 pm", "STRIPE_WEBHOOK_SECRET"],
    }
    base.update(overrides)
    return RLTask.model_validate(base)


class FakeTrackAScorer:
    def predict_proba(self, rows: list[str]) -> list[list[float]]:
        return [[0.95, 0.05] for _ in rows]


def test_reward_prefers_clean_response_over_bad_response() -> None:
    task = _task()
    good = score_response(
        task,
        "Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix.",
    )
    bad = score_response(
        task,
        "Here's a version:\n\nOption 1: Staging recovered after Sarah fixed it.\n\nBest,",
    )

    assert good.reward > bad.reward
    assert good.components["faithfulness"] == 1.0
    assert "option_menu" in bad.penalties
    assert "wrapper_phrase" in bad.penalties


def test_reward_profile_changes_weighted_components() -> None:
    response = "Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix."
    rewrite = score_response(_task("rewrite_faithful_concise"), response)
    direct = score_response(
        _task(
            "direct_workplace_message",
            family="direct_email",
            domain="email",
            mode="direct_generation",
            input_text="",
            required_facts=[],
        ),
        response,
    )

    # New 50/50 schema: weighted_components has ridge_rubric, deterministic, ridge_* dims
    assert "ridge_rubric" in rewrite.weighted_components
    assert "deterministic" in rewrite.weighted_components
    assert rewrite.weighted_components["ridge_rubric"] + rewrite.weighted_components["deterministic"] > 0
    # Both tasks use the same 50/50 split — profile no longer changes weights
    assert "ridge_rubric" in direct.weighted_components
    assert "deterministic" in direct.weighted_components


def test_track_a_scorer_is_optional_and_capped() -> None:
    task = _task()
    without_track_a = score_response(
        task, "Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix."
    )
    with_track_a = score_response(
        task,
        "Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix.",
        FakeTrackAScorer(),
    )

    assert 0.0 <= without_track_a.components["style"] <= 1.0
    assert 0.0 <= with_track_a.components["style"] <= 1.0
    assert (
        abs(with_track_a.components["style"] - without_track_a.components["style"])
        <= 0.20
    )
