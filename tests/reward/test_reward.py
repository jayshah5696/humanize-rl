from __future__ import annotations

import math

import pytest

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

    def predict_rubric(self, rows: list[str]) -> list[list[float]]:
        return [[0.80] * 8 for _ in rows]


class FakePerfectRidgeScorer:
    def predict_proba(self, rows: list[str]) -> list[list[float]]:
        return [[0.0, 1.0] for _ in rows]

    def predict_rubric(self, rows: list[str]) -> list[list[float]]:
        return [[1.0] * 8 for _ in rows]


def _words(count: int) -> str:
    return " ".join(f"word{i}" for i in range(count))


def _target_task() -> RLTask:
    return _task(
        id="rl_v03_000001",
        family="compression",
        domain="creative_general",
        mode="compression",
        register="direct",
        input_text="Source text to compress.",
        constraints={
            "target_words": 100,
            "target_tolerance": 0.20,
            "no_subject_line": True,
            "no_signoff": True,
        },
        reward_profile="compression_update",
        trap_tags=["verbosity"],
        required_facts=[],
    )


def _romance_task() -> RLTask:
    return _task(
        id="rl_v03_000392",
        family="tone_shift",
        domain="creative_general",
        mode="tone_shift",
        register="warm",
        instruction=(
            "[ROLE] You are a social media copywriter. [TASK] Rewrite the "
            "source as a warm post. [AUDIENCE] Followers on social media who "
            "enjoy uplifting romance films. [MUST] Include a quotation from a "
            "famous romantic movie and add a postscript with a personal "
            "recommendation of one romance film with an uplifting theme."
        ),
        input_text=(
            "Please draft a social media post encouraging my followers to "
            "watch romance films with uplifting themes. Include a quotation "
            "from a famous romantic movie and make sure to add a postscript "
            "with a personal recommendation of one such film."
        ),
        constraints={
            "target_words": 29,
            "target_tolerance": 0.15,
            "no_subject_line": True,
            "no_signoff": True,
            "must_not_use_em_dash": True,
            "min_contraction_count": 2,
        },
        reward_profile="rewrite_faithful_concise",
        trap_tags=["register_mismatch", "over_polish", "verbosity"],
        required_facts=[],
    )


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
    assert (
        rewrite.weighted_components["ridge_rubric"]
        + rewrite.weighted_components["deterministic"]
        > 0
    )
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


def test_p50_50_no_penalty_blends_ridge_and_deterministic_only() -> None:
    task = _task()
    response = "Here's a version: Sarah fixed it."

    result = score_response(
        task,
        response,
        FakeTrackAScorer(),
        reward_mode="p50_50_no_penalty",
    )

    expected = (
        result.weighted_components["ridge_rubric"]
        + result.weighted_components["deterministic"]
    )
    assert result.penalties
    assert math.isclose(result.raw_reward, expected, abs_tol=1e-9)
    assert math.isclose(result.reward, expected, abs_tol=1e-9)
    assert result.profile == "p50_50_no_penalty"


def test_strict_reward_still_applies_penalties() -> None:
    task = _task()
    response = "Here's a version: Sarah fixed it."

    result = score_response(task, response, FakeTrackAScorer())
    base = (
        result.weighted_components["ridge_rubric"]
        + result.weighted_components["deterministic"]
    )
    assert result.penalties
    assert math.isclose(
        result.raw_reward,
        base + sum(result.penalties.values()),
        abs_tol=1e-9,
    )
    assert result.reward < base


def test_repeated_completion_scores_low_in_p50_and_records_diagnostic() -> None:
    task = _task(
        family="direct_email",
        domain="email",
        mode="direct_generation",
        input_text="",
        constraints={"max_words": 200, "no_subject_line": True, "no_signoff": True},
        reward_profile="direct_workplace_message",
        required_facts=[],
    )

    result = score_response(
        task,
        ("I want to talk " * 20).strip(),
        FakePerfectRidgeScorer(),
        reward_mode="p50_50_no_penalty",
    )
    diagnostics = {row["name"]: row for row in result.diagnostics}

    assert not diagnostics["repetition"]["passed"]
    assert result.components["repetition"] == 0.0
    assert "repetition" in result.penalties
    assert result.weighted_components["deterministic"] <= 0.10
    assert result.reward <= 0.60


def test_unsuitable_recommendation_caps_p50_even_with_perfect_ridge() -> None:
    result = score_response(
        _romance_task(),
        (
            "\"Love means never having to say you're sorry.\" Let's watch "
            "something hopeful, couldn't we? P.S. I recommend 12 Years a "
            "Slave."
        ),
        FakePerfectRidgeScorer(),
        reward_mode="p50_50_no_penalty",
    )
    diagnostics = {row["name"]: row for row in result.diagnostics}

    assert not diagnostics["unsuitable_recommendation"]["passed"]
    assert result.components["recommendation_suitability"] == 0.0
    assert "unsuitable_recommendation" in result.penalties
    assert result.weighted_components["deterministic"] == 0.0
    assert result.reward <= 0.50


def test_missing_required_fact_caps_p50_even_with_perfect_ridge() -> None:
    result = score_response(
        _task(required_facts=["Staging recovered", "3 pm", "STRIPE_WEBHOOK_SECRET"]),
        "This update sounds punchy, but it drops the actual incident details.",
        FakePerfectRidgeScorer(),
        reward_mode="p50_50_no_penalty",
    )

    assert result.components["semantic_faithfulness"] == 0.0
    assert "missing_required_fact" in result.penalties
    assert result.weighted_components["deterministic"] == 0.0
    assert result.reward <= 0.50


def test_surface_decoration_caps_p50_deterministic_half() -> None:
    star = chr(0x1F31F)
    result = score_response(
        _romance_task(),
        (
            f"GET DIRECT ROMANCE PICKS NOW. Hey friends! {star} "
            "Let's dive into movie night. P.S. Watch The Princess Bride. "
            "#RomanceFilms"
        ),
        FakePerfectRidgeScorer(),
        reward_mode="p50_50_no_penalty",
    )

    assert result.components["surface_naturalness"] < 1.0
    assert {"emoji", "all_caps", "hashtag"} <= set(result.penalties)
    assert result.weighted_components["deterministic"] <= 0.20
    assert result.reward <= 0.70


def test_ai_tell_phrase_caps_p50_surface_naturalness() -> None:
    result = score_response(
        _task(
            input_text=(
                "The server maintenance scheduled for [Date] has been delayed. "
                "Services remain unaffected."
            ),
            required_facts=[],
        ),
        "We are writing to inform you that server maintenance is delayed. Your services remain unaffected.",
        FakePerfectRidgeScorer(),
        reward_mode="p50_50_no_penalty",
    )

    assert result.components["surface_naturalness"] < 1.0
    assert "ai_tell_phrase" in result.penalties
    assert result.weighted_components["deterministic"] <= 0.20
    assert result.reward <= 0.70


def test_instruction_leak_caps_p50_deterministic_half() -> None:
    task = _task(
        id="rl_v03_000009",
        family="compression",
        domain="creative_general",
        mode="compression",
        register="formal",
        instruction="Compress this into a plain paragraph. Return only the message.",
        input_text="Prepare for the DataStream Analytics audit and meet next week.",
        constraints={
            "target_words": 50,
            "target_tolerance": 0.25,
            "no_subject_line": True,
            "no_signoff": True,
        },
        reward_profile="compression_update",
        required_facts=[],
    )
    result = score_response(
        task,
        (
            "DataStream Analytics should prepare for the upcoming audit and "
            "meeting next week. Please review the source message and ensure "
            "your draft is easy to reference."
        ),
        FakePerfectRidgeScorer(),
        reward_mode="p50_50_no_penalty",
    )

    assert "instruction_leak" in result.penalties
    assert result.components["hard_format"] < 1.0
    assert result.weighted_components["deterministic"] <= 0.20
    assert result.reward <= 0.70


def test_explicit_phrase_constraint_failure_zeros_p50_deterministic_half() -> None:
    task = _task(
        id="rl_v03_000010",
        family="compression",
        domain="creative_general",
        mode="compression",
        register="formal",
        instruction="Compress this into a plain paragraph. Return only the message.",
        input_text="Prepare for the DataStream Analytics audit and meet next week.",
        constraints={
            "target_words": 50,
            "target_tolerance": 0.25,
            "no_subject_line": True,
            "no_signoff": True,
            "must_include_phrases": [
                "DataStream Analytics",
                "upcoming audit",
                "meeting next week",
            ],
        },
        reward_profile="compression_update",
        required_facts=[],
    )
    result = score_response(
        task,
        (
            "Prepare for the upcoming DataStream Analytics review by gathering "
            "market data. Thank you for your feedback and collaboration."
        ),
        FakePerfectRidgeScorer(),
        reward_mode="p50_50_no_penalty",
    )

    assert "missing_must_include_phrase" in result.penalties
    assert "signoff" in result.penalties
    assert result.components["semantic_faithfulness"] == 0.0
    assert result.weighted_components["deterministic"] == 0.0
    assert result.reward <= 0.50


def test_hard_format_failure_caps_p50_deterministic_half() -> None:
    result = score_response(
        _task(required_facts=[]),
        (
            "Urgent: Quality Assurance Review and Action Items.\n\n"
            "Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix. "
            "Best regards, Patricia Adams."
        ),
        FakePerfectRidgeScorer(),
        reward_mode="p50_50_no_penalty",
    )

    assert result.components["hard_format"] < 1.0
    assert {"subject_line", "signoff"} <= set(result.penalties)
    assert result.weighted_components["deterministic"] <= 0.20
    assert result.reward <= 0.70


def test_formal_salutation_caps_p50_deterministic_half() -> None:
    result = score_response(
        _task(required_facts=[]),
        (
            "Dear William Brown, I am writing to request a meeting. "
            "Best regards, Patricia Adams, Engineer, Cloud Nine Systems."
        ),
        FakePerfectRidgeScorer(),
        reward_mode="p50_50_no_penalty",
    )

    assert result.components["hard_format"] < 1.0
    assert {"salutation", "signoff"} <= set(result.penalties)
    assert result.weighted_components["deterministic"] <= 0.20
    assert result.reward <= 0.70


def test_markdown_format_failure_caps_p50_deterministic_half() -> None:
    result = score_response(
        _task(required_facts=[]),
        (
            "**Metrics:**\n"
            "- Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix."
        ),
        FakePerfectRidgeScorer(),
        reward_mode="p50_50_no_penalty",
    )

    assert result.components["hard_format"] < 1.0
    assert "wrong_format_markdown" in result.penalties
    assert result.weighted_components["deterministic"] <= 0.20
    assert result.reward <= 0.70


def test_non_romance_title_still_caps_recommendation_reward() -> None:
    result = score_response(
        _romance_task(),
        (
            '"As you wish." We could all use a warm movie night. '
            "P.S. I highly recommend The Dark Knight."
        ),
        FakePerfectRidgeScorer(),
        reward_mode="p50_50_no_penalty",
    )

    assert result.components["recommendation_suitability"] == 0.0
    assert "unsuitable_recommendation" in result.penalties
    assert result.reward <= 0.50


def test_strict_reward_applies_repetition_penalty() -> None:
    task = _task(
        family="direct_email",
        domain="email",
        mode="direct_generation",
        input_text="",
        constraints={"max_words": 200, "no_subject_line": True, "no_signoff": True},
        reward_profile="direct_workplace_message",
        required_facts=[],
    )

    result = score_response(
        task,
        ("justice is really bad on jail today " * 12).strip(),
        FakePerfectRidgeScorer(),
    )
    base = (
        result.weighted_components["ridge_rubric"]
        + result.weighted_components["deterministic"]
    )

    assert "repetition" in result.penalties
    assert math.isclose(
        result.raw_reward,
        base + sum(result.penalties.values()),
        abs_tol=1e-9,
    )
    assert result.reward < base


def test_target_words_tolerance_affects_length_score() -> None:
    task = _target_task()

    on_target = score_response(task, _words(100)).components["length"]
    runaway = score_response(task, _words(200)).components["length"]
    short = score_response(task, _words(40)).components["length"]

    assert on_target == 1.0
    assert runaway == 0.0
    assert short == 0.0


def test_length_failure_caps_p50_deterministic_half() -> None:
    result = score_response(
        _target_task(),
        _words(200),
        FakePerfectRidgeScorer(),
        reward_mode="p50_50_no_penalty",
    )

    assert result.components["length"] == 0.0
    assert result.weighted_components["deterministic"] <= 0.10
    assert result.reward <= 0.60


def test_mild_length_window_failure_caps_p50_deterministic_half() -> None:
    result = score_response(
        _target_task(),
        _words(70),
        FakePerfectRidgeScorer(),
        reward_mode="p50_50_no_penalty",
    )

    assert result.components["length"] > 0.0
    assert "too_short" in result.penalties
    assert result.weighted_components["deterministic"] <= 0.10
    assert result.reward <= 0.60


def test_empty_completion_remains_safe() -> None:
    result = score_response(
        _target_task(),
        "",
        FakePerfectRidgeScorer(),
        reward_mode="p50_50_no_penalty",
    )

    assert math.isfinite(result.reward)
    assert 0.0 <= result.reward <= 1.0
    assert "repetition" not in result.penalties


def test_p50_50_no_penalty_requires_ridge_scorer() -> None:
    with pytest.raises(RuntimeError, match="ridge scorer"):
        score_response(
            _task(),
            "Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix.",
            reward_mode="p50_50_no_penalty",
        )
