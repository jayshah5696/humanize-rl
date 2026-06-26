"""W&B/log-history EMA + diagnostics injection for GRPO.

Slice 2 of ``docs/plans/gemma4_rl_modal_stable_training_continuation.md``:
stop relying on the raw ``train/reward`` curve. Wrap the trainer with a
callback that:

* Computes EMA(20) and EMA(50) of every metric in ``MIRROR_METRICS`` and
  injects them back into ``logs`` so they hit W&B and ``log_history``.
* Pulls diagnostic counters from :mod:`humanize_rl.reward.diagnostics`
  (dither hits, response lengths, penalty totals, risk_compliance) and
  emits per-step averages plus EMAs.

The callback is import-safe in pytest: it only depends on ``transformers``
at runtime inside the trainer process.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any

from humanize_rl.reward import diagnostics as diag

# Metrics from TRL's GRPOTrainer that we want EMA-smoothed copies of.
# Keys correspond to entries already present in ``logs`` passed to
# ``on_log``. New EMA keys are emitted as ``<base>/ema_<window>``.
MIRROR_METRICS: tuple[str, ...] = (
    "reward",
    "reward_std",
    "loss",
    "grad_norm",
    "entropy",
    "frac_reward_zero_std",
    "completions/clipped_ratio",
    "completions/mean_length",
    "rewards/ridge_rubric_reward/mean",
    "rewards/deterministic_reward/mean",
    "rewards/risk_penalty_reward/mean",
    "rewards/scalar_current_reward/mean",
    "rewards/scalar_softened_reward/mean",
    "rewards/p50_50_no_penalty_reward/mean",
    # Diagnostics emitted by this callback itself (see _emit_diag_metrics).
    "diag/dither_rate",
    "diag/penalty_rate",
    "diag/risk_compliance",
    "diag/response_length_mean",
    "diag/response_length_p95",
)

EMA_WINDOWS: tuple[int, ...] = (20, 50)


def _alpha(window: int) -> float:
    """Standard EMA smoothing factor: 2 / (N+1)."""
    return 2.0 / (window + 1.0)


@dataclass
class _EMA:
    """Single-window EMA tracker. Returns NaN until the first sample."""

    alpha: float
    value: float | None = None

    def update(self, x: float) -> float:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return self.value if self.value is not None else float("nan")
        if self.value is None:
            self.value = float(x)
        else:
            self.value = self.alpha * float(x) + (1.0 - self.alpha) * self.value
        return self.value


@dataclass
class EMATracker:
    """One-EMA-per-(metric, window) registry."""

    windows: tuple[int, ...] = EMA_WINDOWS
    state: dict[tuple[str, int], _EMA] = field(default_factory=dict)

    def update(self, metric: str, value: float) -> dict[str, float]:
        out: dict[str, float] = {}
        for window in self.windows:
            key = (metric, window)
            tracker = self.state.get(key)
            if tracker is None:
                tracker = _EMA(alpha=_alpha(window))
                self.state[key] = tracker
            ema = tracker.update(value)
            if ema is not None and not (isinstance(ema, float) and math.isnan(ema)):
                out[f"{metric}/ema_{window}"] = ema
        return out

    def snapshot(self) -> dict[str, float]:
        """Final EMA values (no update). Used for summary.json."""
        out: dict[str, float] = {}
        for (metric, window), tracker in self.state.items():
            if tracker.value is not None:
                out[f"{metric}/ema_{window}"] = tracker.value
        return out


def _emit_diag_metrics(window_size: int = 1) -> dict[str, float]:
    """Pull per-step diagnostics from the reward module and reset.

    Returns a dict with raw step-level values; the callback feeds these
    into the EMA tracker just like any other metric.
    """
    snap = diag.snapshot_and_reset()
    out: dict[str, float] = {}
    n_calls = snap.n_calls
    if n_calls > 0:
        out["diag/dither_rate"] = snap.n_dithered / n_calls
        out["diag/penalty_rate"] = snap.n_with_penalty / n_calls
    if snap.penalty_totals:
        out["diag/risk_compliance"] = statistics.fmean(
            max(0.0, min(1.0, 1.0 + p / max(1e-9, snap.penalty_cap)))
            for p in snap.penalty_totals
        )
    if snap.response_lengths:
        lengths = sorted(snap.response_lengths)
        out["diag/response_length_mean"] = statistics.fmean(lengths)
        # p95 — nearest-rank.
        rank = max(0, min(len(lengths) - 1, int(round(0.95 * (len(lengths) - 1)))))
        out["diag/response_length_p95"] = float(lengths[rank])
    return out


def compute_log_updates(logs: dict[str, Any], tracker: EMATracker) -> dict[str, float]:
    """Pure function: given current ``logs`` + tracker, return injected keys.

    Exposed for testing.
    """
    updates: dict[str, float] = {}
    # First inject diagnostics so they participate in EMA too.
    for k, v in _emit_diag_metrics().items():
        updates[k] = v
    # Now compute EMAs for every mirrored metric we have a sample for.
    combined = dict(logs)
    combined.update(updates)
    for metric in MIRROR_METRICS:
        if metric in combined:
            value = combined[metric]
            if isinstance(value, (int, float)) and not (
                isinstance(value, float) and math.isnan(value)
            ):
                updates.update(tracker.update(metric, float(value)))
    return updates


def build_callback(tracker: EMATracker | None = None) -> Any:
    """Construct the ``TrainerCallback``. Returns ``None`` if transformers is
    unavailable (e.g. pytest with no torch). The trainer call site should
    skip ``callbacks=[cb]`` in that case, but we don't expect to reach the
    trainer in that environment anyway.
    """
    try:
        from transformers import TrainerCallback
    except ImportError:  # pragma: no cover
        return None

    tracker = tracker or EMATracker()

    class WandBEMACallback(TrainerCallback):  # type: ignore[misc, valid-type]
        """Inject EMA + diagnostic metrics on every ``on_log``.

        Important wiring note: ``Trainer._log`` appends ``output`` to
        ``state.log_history`` BEFORE calling ``on_log``. Mutating ``logs``
        here therefore only reaches W&B (which the wandb integration
        callback fires off of ``logs`` after we run). To keep a single
        source of truth, we also patch the just-appended
        ``state.log_history[-1]`` so offline summary tooling sees the
        EMA + diag keys.
        """

        def __init__(self) -> None:
            self.tracker = tracker
            self.last_snapshot: dict[str, float] = {}

        def on_log(  # type: ignore[override]
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            if logs is None:
                return
            updates = compute_log_updates(logs, self.tracker)
            logs.update(updates)
            # Backfill into the freshly-appended log_history entry so
            # ``summary.json`` carries the diag/EMA series, not just
            # ``final_ema`` (Slice 2 bug discovered in Slice 5 review).
            history = getattr(state, "log_history", None)
            if history:
                history[-1].update(updates)
            self.last_snapshot = self.tracker.snapshot()

    return WandBEMACallback()
