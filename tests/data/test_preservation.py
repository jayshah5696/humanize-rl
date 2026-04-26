"""Tests for V-Slice 3 preservation checks."""

from __future__ import annotations

from humanize_rl.data.preservation import (
    evaluate_preservation,
    extract_entities,
    extract_numbers,
    has_email_shape,
    is_question,
)

# ---------------------------------------------------------------------------
# Number extraction
# ---------------------------------------------------------------------------


def test_numbers_extracts_basic_integers_and_decimals() -> None:
    text = "We hit 80% CPU and saw a 2x speedup. Latency dropped to 180ms."
    nums = extract_numbers(text)
    assert "80%" in nums
    assert "2x" in nums
    assert "180ms" in nums


def test_numbers_extracts_dollar_amounts() -> None:
    nums = extract_numbers("Contractor cost is $185/hr, total $1,234.50.")
    assert "$185" in nums
    assert "$1234.50" in nums  # comma stripped in normalization


def test_numbers_extracts_dates_and_times() -> None:
    nums = extract_numbers("Outage was 11:30 to 14:00 on 2024-04-15.")
    assert "11:30" in nums
    assert "14:00" in nums
    assert "2024-04-15" in nums


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


def test_entities_catches_acronyms_anywhere() -> None:
    ents = extract_entities("The GDPR pipeline uses ETL with SSO.")
    assert "GDPR" in ents
    assert "ETL" in ents
    assert "SSO" in ents


def test_entities_catches_proper_names_mid_sentence() -> None:
    """Plain proper-noun-shaped tokens count when mentioned twice."""
    ents = extract_entities(
        "Sarah from infra opened a ticket with Datadog. "
        "Sarah said Datadog will reply Monday."
    )
    assert "Sarah" in ents
    assert "Datadog" in ents


def test_entities_skips_common_sentence_starters() -> None:
    """Capitalized stopwords (The/This/When/etc) are not treated as entities."""
    ents = extract_entities("The team shipped on Monday.")
    assert "The" not in ents


def test_entities_skips_singletons_to_avoid_false_positives() -> None:
    """V-Slice 3 finding: AIify legitimately rephrases sentence-starters
    like 'Spent half a day' → 'I spent half a day'. To stop this from
    firing as 'dropped entity Spent', single-mention plain proper-noun-
    shaped tokens are skipped. Strong-shape tokens still count."""
    text = "Spent half a day on this. Quick fix in the end. Reply if you can."
    ents = extract_entities(text)
    assert "Spent" not in ents
    assert "Quick" not in ents
    assert "Reply" not in ents


def test_entities_skips_common_pronouns_and_ai_transitions() -> None:
    ents = extract_entities("We did X. Furthermore, the team shipped it.")
    assert "We" not in ents
    assert "Furthermore" not in ents


# ---------------------------------------------------------------------------
# Role / shape detection
# ---------------------------------------------------------------------------


def test_is_question_detects_trailing_questionmark() -> None:
    assert is_question("Should we ship Tuesday?")
    assert not is_question("Should we ship Tuesday.")


def test_email_shape_detects_greeting() -> None:
    assert has_email_shape("Hi Mike — quick update.")
    assert has_email_shape("Re: vendor choice")
    assert not has_email_shape("Spent half the day on this.")


# ---------------------------------------------------------------------------
# Full preservation diff
# ---------------------------------------------------------------------------


def test_preservation_clean_when_facts_kept() -> None:
    original = "Sarah from Datadog reported 80% packet loss between 11:30 and 14:00."
    # Rewrite preserves all facts and capitalized tokens but uses different prose.
    rewrite = "Per Sarah at Datadog, packet loss hit 80% between 11:30 and 14:00."
    r = evaluate_preservation(original=original, rewrite=rewrite)
    assert r.numbers_dropped == ()
    assert r.entities_dropped == ()
    assert r.role_drift == ()
    assert not r.has_violations


def test_preservation_flags_dropped_number() -> None:
    original = "Latency rose to 180ms during the 14:00 incident."
    rewrite = "Latency rose during the incident."
    r = evaluate_preservation(original=original, rewrite=rewrite)
    assert "180ms" in r.numbers_dropped
    assert "14:00" in r.numbers_dropped
    assert r.has_violations


def test_preservation_flags_dropped_entity() -> None:
    """With double-mention to satisfy the new singleton skip."""
    original = (
        "Sarah filed the ticket with Datadog support. "
        "Sarah said Datadog should respond by Monday."
    )
    rewrite = "She filed the ticket with their support. They will respond Monday."
    r = evaluate_preservation(original=original, rewrite=rewrite)
    assert "Sarah" in r.entities_dropped
    assert "Datadog" in r.entities_dropped
    assert r.has_violations


def test_preservation_flags_dropped_acronym_on_single_mention() -> None:
    """Strong-shape tokens (acronyms) still trigger on single mention."""
    original = "The GDPR pipeline ran cleanly."
    rewrite = "The pipeline ran cleanly."
    r = evaluate_preservation(original=original, rewrite=rewrite)
    assert "GDPR" in r.entities_dropped


def test_preservation_flags_role_drift_question_to_statement() -> None:
    original = "Should we ship the SSO change Tuesday?"
    rewrite = "We should consider shipping the SSO change Tuesday."
    r = evaluate_preservation(original=original, rewrite=rewrite)
    assert "question_to_statement" in r.role_drift


def test_preservation_flags_email_losing_greeting() -> None:
    original = "Hi Mike — can you take a look at the auth-service today?"
    rewrite = "The auth-service needs review today."
    r = evaluate_preservation(original=original, rewrite=rewrite)
    assert "email_lost_greeting" in r.role_drift


def test_preservation_added_facts_are_not_violations() -> None:
    """Added entities/numbers are not violations — only dropped ones are."""
    original = "Latency rose to 180ms during the GDPR rollout."
    # Rewrite preserves all originals and adds extra Datadog mentions.
    rewrite = (
        "Latency rose to 180ms during the GDPR rollout (a 2x increase, "
        "per Datadog. Datadog also flagged it.)"
    )
    r = evaluate_preservation(original=original, rewrite=rewrite)
    assert r.numbers_dropped == ()
    assert r.entities_dropped == ()
    # Datadog is added (twice -> entity-shaped under singleton rule).
    assert "Datadog" in r.entities_added
    assert not r.has_violations
