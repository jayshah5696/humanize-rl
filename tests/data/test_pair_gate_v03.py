"""Tests for the v03 pair-acceptance gate."""

from __future__ import annotations

from humanize_rl.data.pair_gate_v03 import (
    GateThresholds,
    evaluate_triple,
)


def _good_triple_kwargs() -> dict[str, object]:
    text = " ".join(["word"] * 100)
    return {
        "original_text": text,
        "aiified_text": text,
        "humanized_text": text,
        "original_score": 0.85,
        "aiified_score": 0.30,
        "humanized_score": 0.80,
        "aiified_per_dim": {
            "opener_pattern": 0.4,
            "transition_overuse": 0.3,
            "hedging_density": 0.5,
        },
        "humanized_per_dim": {
            "opener_pattern": 0.9,
            "transition_overuse": 0.9,
            "hedging_density": 0.6,
        },
    }


def test_accepts_clean_triple() -> None:
    result = evaluate_triple(**_good_triple_kwargs())
    assert result.accepted, result.rejected_reasons
    assert result.suspicion_flags == ()
    assert result.metrics["n_dim_improvements"] >= 2


def test_rejects_humanized_too_low() -> None:
    kw = _good_triple_kwargs()
    kw["humanized_score"] = 0.50
    result = evaluate_triple(**kw)
    assert not result.accepted
    assert any("humanized_score" in r for r in result.rejected_reasons)


def test_rejects_aiified_too_high() -> None:
    kw = _good_triple_kwargs()
    kw["aiified_score"] = 0.60
    result = evaluate_triple(**kw)
    assert not result.accepted
    assert any("aiified_score" in r for r in result.rejected_reasons)


def test_rejects_only_one_dim_improvement() -> None:
    kw = _good_triple_kwargs()
    kw["humanized_per_dim"] = {  # type: ignore[index]
        "opener_pattern": 0.9,
        "transition_overuse": 0.3,
        "hedging_density": 0.5,
    }
    result = evaluate_triple(**kw)
    assert not result.accepted
    assert any("dim_improvements" in r for r in result.rejected_reasons)


def test_rejects_length_blowup() -> None:
    kw = _good_triple_kwargs()
    kw["humanized_text"] = " ".join(["word"] * 200)
    result = evaluate_triple(**kw)
    assert not result.accepted
    assert any("length_ratio" in r for r in result.rejected_reasons)


def test_flags_humanized_beats_original_without_rejecting() -> None:
    kw = _good_triple_kwargs()
    kw["original_score"] = 0.70
    kw["humanized_score"] = 0.85  # delta still ok, but exceeds original+0.05
    result = evaluate_triple(**kw)
    assert result.accepted
    assert "humanized_beats_original" in result.suspicion_flags


def test_thresholds_are_overridable() -> None:
    kw = _good_triple_kwargs()
    kw["aiified_score"] = 0.50  # would normally be rejected
    result = evaluate_triple(**kw, thresholds=GateThresholds(max_aiified_score=0.6))
    assert result.accepted


def test_gate_rejects_on_dropped_number() -> None:
    """V-Slice 3: preservation check fires when humanize loses a number."""
    kw = _good_triple_kwargs()
    kw["original_text"] = "Latency rose to 180ms during the 14:00 incident."
    kw["aiified_text"] = (
        "It is worth noting that latency rose to 180ms during the 14:00 incident."
    )
    kw["humanized_text"] = "Latency rose during the incident."  # dropped both
    result = evaluate_triple(**kw)
    assert not result.accepted
    assert any("humanize_dropped_numbers" in r for r in result.rejected_reasons)


def test_gate_rejects_on_dropped_entity() -> None:
    """V-Slice 3: preservation check fires when AIify loses a strong-shape entity.

    Uses GDPR (acronym) so the singleton-skip heuristic doesn't apply.
    """
    kw = _good_triple_kwargs()
    kw["original_text"] = (
        "The GDPR pipeline shipped Tuesday. Run details are in the dashboard."
    )
    kw["aiified_text"] = (
        "The compliance pipeline shipped Tuesday. Run details are in the dashboard."  # GDPR dropped
    )
    kw["humanized_text"] = (
        "The GDPR pipeline shipped Tuesday. Run details are in the dashboard."
    )
    result = evaluate_triple(**kw)
    assert not result.accepted
    assert any("aiify_dropped_entities" in r for r in result.rejected_reasons)


def test_gate_can_skip_preservation_check() -> None:
    """Preservation enforcement is opt-out for callers that want score-only gating."""
    kw = _good_triple_kwargs()
    base_text = (
        "The team shipped the rollout. Latency rose to 180ms during the "
        "window before recovery completed cleanly within the hour."
    )
    # humanized drops 180ms; would normally fail preservation, but length matches.
    drops_number = (
        "The team shipped the rollout. Latency rose during the window before "
        "recovery completed cleanly within the hour as expected."
    )
    kw["original_text"] = base_text
    kw["aiified_text"] = base_text
    kw["humanized_text"] = drops_number
    result = evaluate_triple(**kw, enforce_preservation=False)
    assert result.accepted, result.rejected_reasons
    assert result.aiify_preservation is None
    assert result.humanize_preservation is None

    # And with preservation on, the same triple is rejected.
    result_strict = evaluate_triple(**kw, enforce_preservation=True)
    assert not result_strict.accepted
    assert any("humanize_dropped_numbers" in r for r in result_strict.rejected_reasons)
