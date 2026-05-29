"""Tests for the EMA + diagnostics injection callback (Slice 2).

Plan: docs/plans/gemma4_rl_modal_stable_training_continuation.md Slice 2.

We test the pure ``compute_log_updates`` helper instead of the
``TrainerCallback`` wrapper to avoid pulling transformers/torch into the
unit test path.
"""

from __future__ import annotations

import math

import pytest

from humanize_rl.reward import diagnostics as diag
from humanize_rl.training.wandb_ema_callback import (
    EMA_WINDOWS,
    MIRROR_METRICS,
    EMATracker,
    _alpha,
    compute_log_updates,
)


@pytest.fixture(autouse=True)
def _reset_diag():
    diag.reset()
    diag.set_penalty_cap(1.0)
    yield
    diag.reset()


# ---------------------------------------------------------------------------
# EMA arithmetic
# ---------------------------------------------------------------------------


def test_alpha_matches_documented_formula() -> None:
    assert _alpha(20) == pytest.approx(2.0 / 21.0)
    assert _alpha(50) == pytest.approx(2.0 / 51.0)


def test_first_sample_initializes_ema_to_value() -> None:
    tracker = EMATracker(windows=(20,))
    out = tracker.update("reward", 0.5)
    assert out == {"reward/ema_20": pytest.approx(0.5)}


def test_ema_converges_to_constant_signal() -> None:
    tracker = EMATracker(windows=(20,))
    for _ in range(200):
        tracker.update("reward", 0.7)
    snap = tracker.snapshot()
    assert snap["reward/ema_20"] == pytest.approx(0.7, abs=1e-6)


def test_ema_lags_step_signal() -> None:
    """A step from 0 -> 1 must show EMA < 1 for several steps."""
    tracker = EMATracker(windows=(20,))
    tracker.update("reward", 0.0)
    for _ in range(5):
        out = tracker.update("reward", 1.0)
    assert 0.0 < out["reward/ema_20"] < 1.0


# ---------------------------------------------------------------------------
# compute_log_updates: TRL log injection
# ---------------------------------------------------------------------------


def test_compute_log_updates_emits_ema_for_known_metrics() -> None:
    tracker = EMATracker()
    logs = {
        "reward": 0.6,
        "loss": -0.2,
        "grad_norm": 1.5,
        "frac_reward_zero_std": 0.0,
        "completions/clipped_ratio": 0.0,
        "completions/mean_length": 42.0,
        "step": 5,                # not in MIRROR_METRICS \u2014 must be ignored
        "epoch": 0.5,             # likewise
    }
    out = compute_log_updates(logs, tracker)

    for window in EMA_WINDOWS:
        for key in ("reward", "loss", "grad_norm", "completions/mean_length"):
            assert f"{key}/ema_{window}" in out, f"missing {key}/ema_{window}"
    # Non-mirrored keys must not produce EMAs.
    for window in EMA_WINDOWS:
        assert f"step/ema_{window}" not in out
        assert f"epoch/ema_{window}" not in out


def test_compute_log_updates_handles_missing_metrics_gracefully() -> None:
    """If a metric is absent from this step's logs, no EMA key is emitted."""
    tracker = EMATracker()
    logs = {"reward": 0.5}
    out = compute_log_updates(logs, tracker)
    assert "reward/ema_20" in out
    assert "loss/ema_20" not in out


def test_compute_log_updates_ignores_nan_values() -> None:
    tracker = EMATracker()
    out = compute_log_updates({"reward": float("nan")}, tracker)
    assert all(
        not (isinstance(v, float) and math.isnan(v)) for v in out.values()
    )


# ---------------------------------------------------------------------------
# Diagnostic injection: dither, penalty, length
# ---------------------------------------------------------------------------


def test_diagnostics_inject_per_step_metrics() -> None:
    diag.record_call(response_length=40, penalty_total=0.0)
    diag.record_call(response_length=80, penalty_total=-0.3)
    diag.record_call(response_length=120, penalty_total=-0.8)
    diag.record_dither()  # one dithered group out of three calls

    tracker = EMATracker()
    out = compute_log_updates({"reward": 0.4}, tracker)

    assert out["diag/dither_rate"] == pytest.approx(1 / 3)
    assert out["diag/penalty_rate"] == pytest.approx(2 / 3)
    # risk_compliance is mean of [1.0, 0.7, 0.2] = 0.6333\u2026
    assert out["diag/risk_compliance"] == pytest.approx(
        (1.0 + 0.7 + 0.2) / 3, abs=1e-6
    )
    assert out["diag/response_length_mean"] == pytest.approx(80.0)
    # p95 of [40, 80, 120] (nearest-rank, 3-elt) lands on the top sample.
    assert out["diag/response_length_p95"] == 120.0
    # Diagnostics also get their own EMAs.
    assert "diag/dither_rate/ema_20" in out


def test_diagnostics_reset_after_snapshot() -> None:
    diag.record_call(response_length=50, penalty_total=-0.2)
    compute_log_updates({"reward": 0.5}, EMATracker())  # consumes + resets
    snap = diag.snapshot_and_reset()
    assert snap.n_calls == 0
    assert snap.response_lengths == []
    assert snap.penalty_totals == []


def test_no_diagnostics_emits_no_diag_keys() -> None:
    out = compute_log_updates({"reward": 0.4}, EMATracker())
    for k in (
        "diag/dither_rate",
        "diag/penalty_rate",
        "diag/risk_compliance",
        "diag/response_length_mean",
        "diag/response_length_p95",
    ):
        assert k not in out, f"unexpected {k} with no recorded calls"


# ---------------------------------------------------------------------------
# Sanity: required plan metrics are wired
# ---------------------------------------------------------------------------


def test_mirror_metrics_cover_plan_requirements() -> None:
    """Plan \u00a79 step 1 requires EMA on these base metrics."""
    required = {
        "reward",
        "rewards/ridge_rubric_reward/mean",
        "rewards/deterministic_reward/mean",
        "rewards/risk_penalty_reward/mean",
        "diag/dither_rate",
        "diag/risk_compliance",
        "diag/response_length_mean",
        "diag/response_length_p95",
    }
    assert required <= set(MIRROR_METRICS), (
        f"MIRROR_METRICS missing required entries: {required - set(MIRROR_METRICS)}"
    )
