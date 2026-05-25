from __future__ import annotations

from humanize_rl.reward.checks import run_deterministic_checks
from humanize_rl.reward.tasks import RLTask


def _task(**overrides: object) -> RLTask:
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
            "allow_placeholders": False,
        },
        "reward_profile": "rewrite_faithful_concise",
        "trap_tags": ["wrapper_phrase", "verbosity"],
        "split": "train",
        "required_facts": ["Staging recovered", "3 pm", "STRIPE_WEBHOOK_SECRET"],
        "forbidden_facts": ["database migration"],
    }
    base.update(overrides)
    return RLTask.model_validate(base)


def _by_name(response: str, task: RLTask | None = None):
    return run_deterministic_checks(task or _task(), response).by_name()


def test_known_good_response_passes_hard_checks() -> None:
    diagnostics = run_deterministic_checks(
        _task(), "Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix."
    )

    assert diagnostics.total_penalty == 0
    assert diagnostics.failed == []


def test_option_menu_and_wrapper_trigger_penalties() -> None:
    checks = _by_name(
        "Here's a cleaner version:\n\nOption 1: Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix."
    )

    assert not checks["option_menu"].passed
    assert checks["option_menu"].penalty == -0.40
    assert not checks["wrapper_phrase"].passed
    assert checks["wrapper_phrase"].penalty == -0.20


def test_invented_name_placeholder_subject_and_signoff_trigger() -> None:
    checks = _by_name(
        "Subject: Update\n\nSarah fixed staging at 3 pm after the STRIPE_WEBHOOK_SECRET fix. [Add details]\n\nBest,"
    )

    assert not checks["invented_detail"].passed
    assert "Sarah" in checks["invented_detail"].matches
    assert not checks["placeholder_disallowed"].passed
    assert not checks["subject_line"].passed
    assert not checks["signoff"].passed


def test_length_numbers_entities_and_ai_tells_trigger() -> None:
    response = (
        "It is worth noting that staging recovered after the fix, and furthermore "
        "we will keep carefully monitoring this throughout the afternoon with a detailed "
        "cross-functional operational excellence review for everyone involved."
    )
    checks = _by_name(response)

    assert not checks["too_long"].passed
    assert not checks["missing_number"].passed
    assert "3" in checks["missing_number"].matches
    assert not checks["missing_entity"].passed
    assert "STRIPE_WEBHOOK_SECRET" in checks["missing_entity"].matches
    assert not checks["ai_tell_phrase"].passed


def test_required_placeholder_when_missing_specifics() -> None:
    task = _task(
        id="rl_v01_000002",
        family="placeholder_discipline",
        domain="support",
        mode="direct_generation",
        input_text="",
        constraints={
            "max_words": 30,
            "allow_placeholders": True,
            "require_placeholders_for_missing_specifics": True,
        },
        reward_profile="direct_workplace_message",
        required_facts=[],
    )

    checks = _by_name("Send us the account email and we'll investigate.", task)

    assert not checks["placeholder_required"].passed


def test_forbidden_fact_and_markdown_trigger() -> None:
    checks = _by_name(
        "- Staging recovered at 3 pm after a database migration and the STRIPE_WEBHOOK_SECRET fix."
    )

    assert not checks["forbidden_fact"].passed
    assert not checks["wrong_format_markdown"].passed
